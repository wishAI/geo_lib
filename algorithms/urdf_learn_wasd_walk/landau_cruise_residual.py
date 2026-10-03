"""Cruise-only bounded residual PPO around a frozen Landau controller.

Recorded-state resets are a training curriculum, never milestone evidence.
Composition inspired by https://arxiv.org/abs/1812.03201.
"""
import argparse
import copy
import importlib.util
import json
import math
from pathlib import Path

FAMILY = 'periodic_feedback_residual_ppo'
FEATURE_DIM = 69
BOUND_RAD = .02


def sample_active_rows(pools, ages, modes, early_fraction=0.):
    """Mix early-turn starts with the original broad, training-only row pools."""
    import torch
    result=torch.empty(len(modes),device=modes.device,dtype=torch.long)
    for mode,pool in enumerate(pools):
        selected=modes==mode;count=int(selected.sum())
        rows=pool[torch.randint(len(pool),(count,),device=modes.device)]
        if early_fraction:
            early=pool[(ages[pool]>=3.) & (ages[pool]<=7.)]
            if not len(early):raise ValueError('No early training states for protocol')
            use_early=torch.rand(count,device=modes.device)<early_fraction
            rows=torch.where(use_early,early[torch.randint(len(early),(count,),device=modes.device)],rows)
        result[selected]=rows
    return result


def features(observation, walking, memory):
    import torch
    # Derive phase exactly as the frozen gait does; never expose unbounded age.
    age=(1.-observation[:,62])*30.
    phase=2*torch.pi*(age-1.).clamp_min(0)/walking[0]
    heading=torch.atan2(observation[:,68],observation[:,69]+1.)
    error=heading-memory.reference
    blend=torch.as_tensor(memory.left_cruise_blend,device=observation.device,dtype=observation.dtype).expand(len(observation))
    feedback=torch.as_tensor(memory.left_feedback_blend,device=observation.device,dtype=observation.dtype).expand(len(observation))
    result=torch.cat((observation[:,:60],observation[:,63:66],
        torch.stack((phase.sin(),phase.cos(),error.sin(),error.cos()-1.,blend,feedback),dim=1)),dim=1)
    scales=result.new_tensor([.3]*3+[2.]*3+[1.]*3+[.3]*17+[.4]*17+[3.]*17+[.2,.2,.1]+[1.]*6)
    return result/scales


def compose(baseline, latent, memory, names, observation):
    """Joint target radian bounds apply before the existing actuator clipping."""
    import torch
    blend=torch.as_tensor(memory.left_cruise_blend,device=baseline.device,dtype=baseline.dtype).expand(len(baseline))
    blend=blend*(observation[:,63].abs()/.2).clamp(0,1)
    result=baseline.clone()
    # Explicit selection makes inactive rows bit-identical, even for bad latents.
    correction=torch.where(blend[:,None]>0.,BOUND_RAD*latent.tanh()*blend[:,None],0.)/.08
    for side,column in (('left',0),('right',1)):
        result[:,names.index(side+'_hip_roll_joint')]+=correction[:,column]
    return result


def make_actor(device='cpu'):
    import torch
    from tensordict import TensorDict
    from rsl_rl.models import MLPModel
    return MLPModel(TensorDict({'actor':torch.zeros(1,FEATURE_DIM,device=device)},batch_size=[1]),
        {'actor':['actor']},'actor',2,hidden_dims=[64,64],activation='elu',obs_normalization=False,
        distribution_cfg={'class_name':'GaussianDistribution','init_std':.1,'std_type':'scalar'}).to(device)


def apply(actor, baseline, observation, walking, memory, names):
    import torch
    from tensordict import TensorDict
    blend=torch.as_tensor(memory.left_cruise_blend,device=observation.device).expand(len(observation))
    if not bool((blend>0).any()):return baseline
    value=features(observation,walking,memory)
    latent=actor(TensorDict({'actor':value},batch_size=[len(value)]))
    return compose(baseline,latent,memory,names,observation)


def load_source(path, expected, module_name):
    import hashlib
    path=Path(path)
    if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:raise ValueError(f'Saved source changed: {path.name}')
    spec=importlib.util.spec_from_file_location(module_name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def build_bank(folder, parent, turn, output):
    """Replay controller memory only, then save pre-action physical states."""
    import numpy as np
    import torch
    from algorithms.urdf_learn_wasd_walk import mujoco_backend as backend
    folder=Path(folder).resolve()
    result=json.loads((folder/'dynamics.json').read_text())
    if result['config']['checkpoint_sha256']!=backend.digest(parent):raise ValueError('Warm-state parent mismatch')
    cfg=result['config'];parent_meta=json.loads((Path(parent).parent/'training.json').read_text())
    if cfg.get('direction')!='backward' or cfg['forward']!=.2 or cfg['assistance']!=0.:
        raise ValueError('Require unassisted0.2backward curriculum trace')
    if cfg['dt']!=.002 or cfg['contact_timeconst']!=.004 or cfg['gain_scale']!=1.:
        raise ValueError('Curriculum physics mismatch')
    for key in ('backend','mujoco_version','model_xml_sha256','urdf_sha256','mesh_tree_sha256'):
        if result['identity'][key]!=parent_meta[key]:raise ValueError('Curriculum model mismatch: '+key)
    for key in ('fall_count','done_count','reset_count'):
        if result['metrics'][key]!=0:raise ValueError('Require a nonfallen, unreset curriculum trace')
    if backend.digest(folder/'foot_trace.json')!=result['foot_trace_sha256']:raise ValueError('Foot curriculum trace changed')
    feet=json.loads((folder/'foot_trace.json').read_text())
    first_bad=min((row['time_s'] for row in feet if row['support_body_weight_ratio']>3. or row['max_joint_speed_rad_s']>4.+1e-6),default=float('inf'))
    upper=min(146.,first_bad-.2)
    if upper<145.:raise ValueError('Healthy curriculum does not reach the late impact region')
    for name,key in (('policy_trace.npz','policy_trace_sha256'),('trajectory.npz','trajectory_sha256')):
        if backend.digest(folder/name)!=result[key]:raise ValueError('Curriculum trace changed')
    trajectory=np.load(folder/'trajectory.npz');trace=np.load(folder/'policy_trace.npz')
    if not np.allclose(trajectory['time'][:-1],trace['time'],rtol=0,atol=1e-6):raise ValueError('Curriculum clocks differ')
    memory=turn.CommandMemory();selected=[];references=[];blends=[];feedback=[];starts=[]
    # Select healthy pre-impact states; six-second episodes cross the late failure region.
    # Cover several dozen phases, including envelope ramps and renewed steering.
    for i,(age,obs) in enumerate(zip(trace['time'],trace['observation'])):
        memory.observe(torch.from_numpy(obs)[None],torch.zeros(1,3),torch.eye(3)[None],float(age))
        if 90.<=age<upper and i%10==0:
            selected.append(i);references.append(memory.reference);blends.append(memory.left_cruise_blend)
            feedback.append(memory.left_feedback_blend);starts.append(memory.left_cruise_start if memory.left_cruise_start is not None else float('inf'))
    if not selected:raise ValueError('No cruise curriculum states')
    ix=np.array(selected)
    np.savez_compressed(output,qpos=trajectory['qpos'][ix],qvel=trajectory['qvel'][ix],
        previous=trace['observation'][ix,43:60],age=trace['time'][ix],reference=np.array(references),
        blend=np.array(blends),feedback=np.array(feedback),cruise_start=np.array(starts),
        previous_yaw=trace['observation'][ix,65])
    return {'source_directory':str(folder),'source_files_sha256':{name:backend.digest(folder/name)
        for name in ('trajectory.npz','policy_trace.npz','foot_trace.json','dynamics.json')},'count':len(ix),
        'age_range_s':[90.,upper],'first_unsafe_force_or_speed_s':first_bad if math.isfinite(first_bad) else None,'interval_s':.2,'training_indices':'even','screen_indices':'odd',
        'limitations':'Training resets only; solver warmstart absent and cleared. No nominal-start proof or exact continuation claim.'}


class CruiseMemory:
    """Per-world ages support asynchronous recorded-state training resets."""
    def __init__(self,n,device):
        import torch
        self._reference=torch.zeros(n,device=device,dtype=torch.float64)
        self._blend=torch.zeros(n,device=device,dtype=torch.float64)
        self._feedback=torch.zeros_like(self._blend)
        self.left_cruise_start=torch.full_like(self._reference,float('inf'))
        self.previous_yaw=torch.zeros_like(self._reference)
        self.turned=True;self.last_yaw_sign=1
        self.anchor_xy=None;self.anchor_time=None;self.restart_time=None

    @property
    def reference(self):return self._reference.float()
    @property
    def left_cruise_blend(self):return self._blend.float()
    @property
    def left_feedback_blend(self):return self._feedback.float()

    def advance(self,yaw,age,elapsed):
        import torch
        self._reference+=self.previous_yaw*elapsed
        cruising=yaw==0.
        self.left_cruise_start=torch.where(cruising,torch.minimum(self.left_cruise_start,age),float('inf'))
        target=(cruising & (age-self.left_cruise_start>=.2-1e-9)).double()
        self._blend+=(target-self._blend).clamp(-elapsed,elapsed)
        self._feedback+=((yaw>0.).double()-self._feedback).clamp(-elapsed,elapsed)
        self.previous_yaw=yaw.double()


def privileged_critic_features(actor_features,observation,mode,age,positions,rotations,memory,teacher,turn_duration):
    """Training-only reward state; the deployable actor retains its69 inputs.

    Asymmetric actor/critic observations: https://arxiv.org/abs/1710.06542.
    """
    import torch
    heading=torch.atan2(observation[:,68],observation[:,69]+1.)
    error=torch.atan2((heading-teacher).sin(),(heading-teacher).cos())/math.radians(10)
    valid=torch.isfinite(memory.stop_time);held=valid & ~memory.previous_moving
    target=torch.where(mode==1,math.pi/2,memory.hold_heading)
    hold_error=torch.where((mode==1)|held,torch.atan2((heading-target).sin(),(heading-target).cos())/math.radians(5),0.)
    delta=torch.zeros_like(positions);delta[:,:2]=positions[:,:2]-memory.stop_xy
    local=torch.bmm(rotations.transpose(1,2),delta[:,:,None])[:,:2,0]/.03
    local=torch.where(held[:,None],local,0.)
    fade=torch.where(mode==1,3.+turn_duration,18.)
    until=torch.stack(((fade-age)/10.,(fade+2.-age)/10.),1).clamp(-1,1)
    until=torch.where((mode!=0)[:,None],until,0.)
    settled=torch.where(held,((age-memory.stop_time)/2.).clamp(0,1),0.)
    extra=torch.cat((torch.nn.functional.one_hot(mode,3),error[:,None],hold_error[:,None],local,
        until,settled[:,None],valid[:,None]),1).to(actor_features.dtype)
    return torch.cat((actor_features,extra),1)


def extend_critic_state(original,target):
    """Keep the learned value mapping while initializing added inputs to zero."""
    import torch
    if original.keys()!=target.keys():raise ValueError('Critic architecture changed')
    result={}
    for key,value in original.items():
        if value.shape==target[key].shape:result[key]=value.clone();continue
        if key!='mlp.0.weight' or value.shape[0]!=target[key].shape[0] or value.shape[1]!=69 or target[key].shape[1]!=80:
            raise ValueError('Unsupported critic expansion')
        result[key]=torch.zeros_like(target[key]);result[key][:,:69]=value.to(result[key].device)
    return result


def active_episode_end(mode,age):
    """Complete both stops; bound the backward curriculum before cruise onset."""
    import torch
    return torch.where(mode==1,55.8,torch.where(mode==2,24.8,(age+24.).clamp_max(90.8)))


def train(args):
    import hashlib
    import time
    from dataclasses import asdict
    from datetime import datetime,timezone
    import numpy as np
    import torch
    import warp as wp
    from tensordict import TensorDict
    from mjlab.rl import MjlabOnPolicyRunner,RslRlModelCfg,RslRlOnPolicyRunnerCfg,RslRlPpoAlgorithmCfg
    from algorithms.urdf_learn_wasd_walk.landau_rsl_control import configure_model
    from algorithms.urdf_learn_wasd_walk.landau_forward_control import load_gait_source,make_actor as make_prior,action_scales
    from algorithms.urdf_learn_wasd_walk.landau_turn_control import install_physics_audit
    from algorithms.urdf_learn_wasd_walk import landau_direction_training as direction
    backend,batches,_=configure_model('balanced_hands_v1')
    parent=Path(args.checkpoint).resolve();parent.relative_to((backend.OUTPUT/'training').resolve())
    meta=json.loads((parent.parent/'training.json').read_text())
    if meta['checkpoints'][parent.name]!=backend.digest(parent):raise ValueError('Parent changed')
    if meta['policy_family']!='periodic_feedback_cem' or str(meta.get('status','')).startswith('invalidated_'):
        raise ValueError('Require valid frozen CEM parent')
    blob=torch.load(parent,map_location='cpu',weights_only=False)
    if len(blob['turn_parameters'])!=20:raise ValueError('Require20parameter incumbent, no rejected offsets')
    base,gait_source=load_gait_source(parent,meta)
    turn=load_source(parent.parent/'turn_source.py',meta['turn_source_sha256'],'frozen_cruise_turn')
    turn.configure_profile(meta['arguments']['turn_duration'])
    directory=backend.OUTPUT/'training'/args.name;directory.mkdir(exist_ok=False)
    active_mode=bool(args.active_curriculum)
    cruise_actor=None;cruise_blob=None;active_seed=None;seed_meta=None
    if active_mode:
        from algorithms.urdf_learn_wasd_walk import landau_active_curriculum as active_bank
        from algorithms.urdf_learn_wasd_walk import landau_active_residual as active_head
        cruise_path=Path(args.residual_checkpoint).resolve()
        cruise_meta=json.loads((cruise_path.parent/'training.json').read_text())
        if (cruise_meta['checkpoints'][cruise_path.name]!=backend.digest(cruise_path) or
            cruise_meta.get('policy_family')!=FAMILY or cruise_meta.get('residual_variant') is not None or
            cruise_meta['parent_checkpoint_sha256']!=backend.digest(parent) or
            cruise_meta['parent_training_sha256']!=backend.digest(parent.parent/'training.json') or
            str(cruise_meta.get('status','')).startswith('invalidated_')):
            raise ValueError('Invalid frozen cruise checkpoint')
        cruise_blob=torch.load(cruise_path,map_location='cpu',weights_only=False)
        if cruise_blob.get('residual_variant') is not None or 'active_residual_actor_state_dict' in cruise_blob:raise ValueError('Invalid cruise-only weights')
        for key in ('parameters','turn_parameters'):
            if not torch.equal(cruise_blob[key],blob[key]):raise ValueError('Cruise parent differs')
        left,right=cruise_blob['standing_actor_state_dict'],blob['standing_actor_state_dict']
        if left.keys()!=right.keys() or any(not torch.equal(left[k],right[k]) for k in left):raise ValueError('Cruise standing parent differs')
        old_source=load_source(cruise_path.parent/'residual_source.py',cruise_meta['residual_source_sha256'],'frozen_cruise_residual')
        cruise_actor=old_source.make_actor('cuda');cruise_actor.load_state_dict(cruise_blob['residual_actor_state_dict'])
        cruise_actor.eval();cruise_actor.requires_grad_(False)
        if args.active_seed:
            seed_path=Path(args.active_seed).resolve();seed_meta=json.loads((seed_path.parent/'training.json').read_text())
            if seed_meta.get('policy_family')!=FAMILY or seed_meta['checkpoints'][seed_path.name]!=backend.digest(seed_path) or str(seed_meta.get('status','')).startswith('invalidated_'):
                raise ValueError('Invalid active seed')
            active_seed=torch.load(seed_path,map_location='cpu',weights_only=False)
            if not active_head.check_contract(seed_meta,active_seed,backend.digest):raise ValueError('Seed is not active residual')
            if (seed_meta['frozen_cruise_checkpoint_sha256']!=backend.digest(cruise_path) or
                seed_meta['active_residual_source_sha256']!=backend.digest(active_head.__file__) or
                backend.digest(seed_path.parent/'active_residual_source.py')!=seed_meta['active_residual_source_sha256']):
                raise ValueError('Active seed composition differs')
        bank_meta=active_bank.build_bank(args.active_curriculum,parent,cruise_path,turn,directory/'warm_states.npz')
    else:bank_meta=build_bank(args.curriculum,parent,turn,directory/'warm_states.npz')
    torch.set_num_threads(4);torch.manual_seed(42)
    batch=batches.WarpBatch(args.num_envs,42,0.,'stand','load',.004,.2,.0225)
    if hashlib.sha256(batch.xml.encode()).hexdigest()!=meta['model_xml_sha256']:raise ValueError('Physical model changed')
    audit=install_physics_audit(batch,batches,args.num_envs)
    bank={k:torch.tensor(v,device='cuda') for k,v in np.load(directory/'warm_states.npz').items()}
    walking=blob['parameters'].to('cuda');turn_parameters=blob['turn_parameters'].to('cuda')
    prior=make_prior(63,'cuda');prior.load_state_dict(blob['standing_actor_state_dict']);prior.eval();prior.requires_grad_(False)
    names=batch.spec['action_joints'];limits=torch.tensor(action_scales(names,.4),device='cuda')/.08
    weight=float(batch.model.body_mass.sum())*9.81
    memory=active_bank.ActiveMemory(args.num_envs,'cuda') if active_mode else CruiseMemory(args.num_envs,'cuda')
    train_pools=[torch.where(bank['mode']==i)[0][::2] for i in range(3)] if active_mode else None
    stats={};began=time.perf_counter()

    class Env:
        num_envs=args.num_envs;num_actions=2;device='cuda:0';max_episode_length=2640 if args.active_long else round(args.seconds/.02)
        cfg={'task':'recorded_active_left_residual' if active_mode else 'recorded_cruise_residual','is_finite_horizon':False};common_step_counter=0
        def __init__(self):
            self.age=torch.zeros(self.num_envs,device=self.device,dtype=torch.float64)
            self.end_age=torch.zeros_like(self.age)
            if args.active_long:self.mode=torch.arange(self.num_envs,device=self.device)%3
            self.air=torch.zeros((self.num_envs,2),device=self.device);self.peak=torch.zeros_like(self.air)
            self.touched=torch.zeros_like(self.air,dtype=torch.bool)
            self.previous_residual=torch.zeros((self.num_envs,2),device=self.device)
            self.seen_stop=torch.zeros(self.num_envs,device=self.device,dtype=torch.bool)
            self.seen_settled=torch.zeros_like(self.seen_stop)
            self.coverage={key:torch.zeros(3,device=self.device,dtype=torch.long)
                for key in ('completed','failed','force_margin_failed','stopped','settled','successful_timeout')}
            self.completed=0;self.failed=0;self.returns=torch.zeros(self.num_envs,device=self.device)
            self.reset(torch.ones(self.num_envs,device=self.device,dtype=torch.bool))
        @property
        def unwrapped(self):return self
        @property
        def episode_length_buf(self):return batch.episode_steps
        def reset(self,mask):
            count=int(mask.sum())
            if not count:return
            if active_mode:
                choice=self.mode[mask] if args.active_long else torch.randint(3,(count,),device=self.device)
                ix=sample_active_rows(train_pools,bank['age'],choice,args.early_start_fraction)
                if not hasattr(self,'mode'):self.mode=torch.zeros(self.num_envs,device=self.device,dtype=torch.long)
                self.mode[mask]=bank['mode'][ix]
                for name in ('previous_moving','stop_xy','stop_time','hold_heading'):getattr(memory,name)[mask]=bank[name][ix].to(getattr(memory,name).dtype)
            else:ix=torch.randint((len(bank['age'])+1)//2,(count,),device=self.device)*2
            batch.q[mask]=bank['qpos'][ix].float();batch.v[mask]=bank['qvel'][ix].float()
            batch.previous[mask]=bank['previous'][ix].float();batch.ctrl[mask]=batch.targets
            wp.to_torch(batch.wd.qacc_warmstart)[mask]=0;wp.to_torch(batch.wd.qfrc_applied)[mask]=0
            batch.wrench[mask]=0;wp.to_torch(batch.wd.time)[mask]=bank['age'][ix].float()
            batch.episode_steps[mask]=0;self.age[mask]=bank['age'][ix]
            if args.active_long:self.end_age[mask]=active_episode_end(self.mode[mask],self.age[mask])
            memory._reference[mask]=bank['reference'][ix];memory._blend[mask]=bank['blend'][ix]
            memory._feedback[mask]=bank['feedback'][ix];memory.left_cruise_start[mask]=bank['cruise_start'][ix]
            memory.previous_yaw[mask]=bank['previous_yaw'][ix].double()
            self.air[mask]=0;self.peak[mask]=0;self.touched[mask]=False;self.previous_residual[mask]=0
            self.seen_stop[mask]=False;self.seen_settled[mask]=False
            self.returns[mask]=0
            for value in audit.values():value[mask]=0
            torch.cuda.synchronize();wp.capture_launch(batch.forward_graph);wp.synchronize()
        def get_observations(self):
            rotation=batch.rot[:,batch.base].double()
            heading=torch.atan2(rotation[:,1,0],rotation[:,0,0])
            command=direction.commands('backward',batch.pos[:,batch.pelvis,:2].double()-batch.reference_xy.double(),heading,100.)
            if active_mode:
                command=active_bank.commands(self.mode,self.age,batch.pos[:,batch.pelvis,:2].double()-batch.reference_xy.double(),heading,meta['arguments']['turn_duration'])
            extra=torch.zeros((self.num_envs,10),device=self.device)
            extra[:,2]=(1.-self.age/30.).float();extra[:,3:6]=command.float()
            phase=2*torch.pi*self.age/.6
            extra[:,6]=phase.sin().float();extra[:,7]=phase.cos().float()
            extra[:,8]=rotation[:,1,0].float();extra[:,9]=(rotation[:,0,0]-1).float()
            self.observation=torch.cat((batch.observations(),extra),dim=1)
            self.command=command
            self.prior_observation=memory.prior(self.observation,batch.pos[:,batch.pelvis],batch.rot[:,batch.base],self.age) if active_mode else self.observation[:,:63]
            value=features(self.observation,walking,memory)
            critic=value
            if args.privileged_critic:
                teacher=bank['teacher_heading'][self.mode,torch.round(self.age/.02).long()]
                critic=privileged_critic_features(value,self.observation,self.mode,self.age,batch.pos[:,batch.pelvis],
                    batch.rot[:,batch.base],memory,teacher,meta['arguments']['turn_duration'])
            return TensorDict({'actor':value,'critic':critic},batch_size=[self.num_envs])
        def step(self,latent):
            with torch.no_grad():
                baseline=turn.commanded_action(base,walking,prior,self.observation,self.prior_observation,
                    turn_parameters,names,memory.reference.float(),True,memory=memory)
                if active_mode:
                    baseline=old_source.apply(cruise_actor,baseline,self.observation,walking,memory,names)
                    action=active_head.compose(baseline,latent,memory,names,self.observation).clamp(-limits,limits)
                else:action=compose(baseline,latent,memory,names,self.observation).clamp(-limits,limits)
            before=batch.pos[:,batch.pelvis,:2].clone()
            residual=latent.tanh()*(active_head.envelope(memory,self.observation) if active_mode else memory.left_cruise_blend.float())[:,None]
            previous_command=self.command.clone()
            before_heading=torch.atan2(batch.rot[:,batch.base,1,0],batch.rot[:,batch.base,0,0])
            # Interval maxima drive reward; flight_current persists across controls.
            for key in ('speed','force','nonfoot','flight','foot_peak','foot_impulse'):audit[key].zero_()
            batch.ctrl[:]=batch.targets;batch.ctrl[:,batch.aids]+=.08*action;batch.wrench.zero_()
            torch.cuda.synchronize();wp.capture_launch(batch.graph);wp.synchronize()
            occupancy=batch.maximum_occupancy.numpy()
            if max(occupancy[:2])>=batch.wd.naconmax or occupancy[2]>=batch.wd.njmax:raise RuntimeError('Physics capacity exceeded')
            batch.previous[:]=action;batch.episode_steps+=1;self.age+=.02;self.common_step_counter+=1
            tilt=batch.rot[:,batch.base,2,2].clamp(-1,1).acos()
            force=audit['force']/weight
            failed=(tilt>math.pi/6)|(batch.pos[:,batch.pelvis,2]<batch.reference_height-.08)
            failed|=(audit['speed']>4.+1e-6)|(force>args.training_force_limit)|(audit['nonfoot']!=0)|(audit['flight']>60)
            failed|=~torch.isfinite(batch.q).all(1)|~torch.isfinite(batch.v).all(1)
            contacts,slip,clearance,_=batch.feet(with_forces=True)
            landed=contacts&self.touched&(self.air>=.06)&(self.peak>=.015)
            self.air=torch.where(contacts,0.,self.air+.02)
            self.peak=torch.where(contacts,0.,torch.maximum(self.peak,clearance));self.touched|=contacts
            linear,_=batch.local_velocity()
            progress=-(batch.pos[:,batch.pelvis,1]-before[:,1])
            desired=previous_command[:,0].float() if active_mode else .2
            reward=.02*(1.+2.*torch.exp(-(linear[:,1]-desired).square()/.0225)
                -args.force_cost*(force-2.5).clamp_min(0).square()-2.*tilt.square()-2.*slip
                -.1*residual.square().sum(1)-.1*(residual-self.previous_residual).square().sum(1))
            if active_mode:
                moving=previous_command.abs().amax(1)>0
                delta=batch.pos[:,batch.pelvis,:2]-before
                progress=-before_heading.sin()*delta[:,0]+before_heading.cos()*delta[:,1]
                post_heading=torch.atan2(batch.rot[:,batch.base,1,0],batch.rot[:,batch.base,0,0])
                teacher=bank['teacher_heading'][self.mode,torch.round(self.age/.02).long()]
                error=torch.atan2((post_heading-teacher).sin(),(post_heading-teacher).cos())
                target=torch.where(self.mode==1,math.pi/2,memory.hold_heading)
                hold_error=torch.atan2((post_heading-target).sin(),(post_heading-target).cos())
                heading_penalty=torch.where(moving,(error/math.radians(args.heading_scale_deg)).square(),(hold_error/math.radians(5)).square())
                drift=(batch.pos[:,batch.pelvis,:2]-memory.stop_xy).square().sum(1)
                if args.active_long:
                    held=~moving & torch.isfinite(memory.stop_time)
                    settled=held & (self.age-memory.stop_time>=2.-1e-8)
                    failed|=held & (drift>.03**2)
                    angular_world=torch.bmm(batch.rot[:,batch.base],batch.v[:,3:6,None]).squeeze(-1)
                    pelvis_velocity=batch.v[:,:3]+torch.cross(angular_world,batch.pos[:,batch.pelvis]-batch.q[:,:3],dim=1)
                    failed|=settled & ((hold_error.abs()>math.radians(5)) | (pelvis_velocity[:,:2].norm(dim=1)>.05) | (angular_world[:,2].abs()>.1))
                reward-=.02*(2.*heading_penalty+torch.where(moving,0.,2.*drift/.03**2)).float()
                progress=progress*moving;landed=landed & moving[:,None]
            reward+=2.*progress+.1*landed.float().sum(1)-args.failure_cost*failed.float()
            reward=torch.nan_to_num(reward,nan=-args.failure_cost,posinf=-args.failure_cost,neginf=-args.failure_cost)
            self.previous_residual=residual.detach().clone();self.returns+=reward
            timeout=self.age>=self.end_age-1e-8 if args.active_long else batch.episode_steps>=self.max_episode_length
            done=failed|timeout
            # Integrate the preceding command once, then update the live-pose steering dwell.
            self.get_observations()
            if active_mode:memory.advance_command(self.observation,batch.pos[:,batch.pelvis],self.age,.02)
            else:memory.advance(self.observation[:,65].double(),self.age,.02)
            if active_mode:
                stopped=torch.isfinite(memory.stop_time) & ~failed
                settled=stopped & (self.age-memory.stop_time>=2.-1e-8)
                masks={'completed':done,'failed':failed,'force_margin_failed':force>args.training_force_limit,
                    'stopped':stopped & ~self.seen_stop,'settled':settled & ~self.seen_settled,
                    'successful_timeout':timeout & ~failed}
                for key,mask in masks.items():self.coverage[key]+=torch.bincount(self.mode[mask],minlength=3)
                self.seen_stop|=stopped;self.seen_settled|=settled
                stats['coverage_by_protocol']={key:value.tolist() for key,value in self.coverage.items()}
            self.completed+=int(done.sum());self.failed+=int(failed.sum())
            stats.update(completed=self.completed,failed=self.failed,mean_interval_peak_bw=float(force.mean()),
                max_interval_peak_bw=float(force.max()),mean_forward_mps=float(linear[:,1].mean()),
                mean_abs_residual_rad=float(residual.abs().mean())*BOUND_RAD)
            extras={'time_outs':(timeout&~failed).float(),'log':{key:value for key,value in stats.items() if isinstance(value,(int,float))}}
            self.reset(done)
            return self.get_observations(),reward,done.float(),extras

    env=Env()
    cfg=RslRlOnPolicyRunnerCfg(actor=RslRlModelCfg(hidden_dims=(64,64),activation='elu',obs_normalization=False,
        distribution_cfg={'class_name':'GaussianDistribution','init_std':.1,'std_type':'scalar'}),
        critic=RslRlModelCfg(hidden_dims=(64,64),activation='elu',obs_normalization=False),
        algorithm=RslRlPpoAlgorithmCfg(value_loss_coef=1.,use_clipped_value_loss=True,clip_param=.2,entropy_coef=.001,
            num_learning_epochs=5,num_mini_batches=4,learning_rate=3e-4,schedule='adaptive',gamma=.998 if args.active_long else .995,lam=.95,desired_kl=.01,max_grad_norm=1.),
        experiment_name='landau_cruise_residual',save_interval=50,num_steps_per_env=24,max_iterations=args.iterations,
        logger='tensorboard',upload_model=False)
    config=asdict(cfg);runner=MjlabOnPolicyRunner(env,copy.deepcopy(config),str(directory),device='cuda:0')
    if active_seed is not None:
        runner.alg.actor.load_state_dict(active_seed['active_residual_actor_state_dict'],strict=True)
        runner.alg.critic.load_state_dict(extend_critic_state(active_seed['critic_state_dict'],runner.alg.critic.state_dict()),strict=True)
    else:
        with torch.no_grad():runner.alg.actor.mlp[-1].weight.zero_();runner.alg.actor.mlp[-1].bias.zero_()
    inherited=('backend','mujoco_version','observation_dim','period_s','observation_clock_period_s',
        'gait_period_source','packages','action_joints','action_scale','stand_action_scale','action_mapping',
        'nominal_q','nominal_ctrl','reference_xy','model_xml_sha256','urdf_sha256','mesh_tree_sha256',
        'mass_variant','standing_checkpoint','standing_checkpoint_sha256','source_sha256','command_extension',
        'turn_source_sha256','command_profile','memory_version','teleop_contract_sha256','assistance')
    training={key:copy.deepcopy(meta[key]) for key in inherited if key in meta}
    training.update(policy_family=FAMILY,status='running',created_at=datetime.now(timezone.utc).isoformat(),
        arguments={**vars(args),'turn_duration':meta['arguments']['turn_duration'],'hip_roll_range':.4},
        parent_checkpoint=str(parent),parent_checkpoint_sha256=backend.digest(parent),
        parent_training_sha256=backend.digest(parent.parent/'training.json'),
        residual_source_sha256=backend.digest(__file__),residual_bound_rad=BOUND_RAD,residual_feature_dim=FEATURE_DIM,
        residual_action_joints=['left_hip_roll_joint','right_hip_roll_joint'],
        residual_gate='left_cruise_blend * clamp(abs(forward)/0.2,0,1)',runner_config=config,
        curriculum=bank_meta,policy_composition='Frozen20parameter yaw controller and gait/standing network; cruise-only two-hip-roll bounded PPO residual',
        research_reference='https://arxiv.org/abs/1812.03201',checkpoints={},
        canonical_milestone_pass=False,
        training_reset_semantics='Recorded healthy state curriculum, qpos/qvel/previous-action/controller memory restored; solver warmstart zeroed; no evaluation resets',
        training_hard_guards={'support_force_bw':args.training_force_limit,'joint_speed_rad_s':4.,'max_flight_s':.12,'tilt_rad':math.pi/6,'root_drop_m':.08},
        reward='dt*(alive+2*forward_tracking-8*excess_peak_over2.5BW^2-2*tilt^2-2*slip-.1*residual^2-.1*residual_change^2)+2*world_backward_progress+.1*qualified_swings-5*failure',
        training_timeout='6s curriculum truncation with RSL timeout bootstrap; nominal-start evaluation unchanged')
    for filename in ('control_source.py','turn_source.py','teleop_contract.py'):
        (directory/filename).write_bytes((parent.parent/filename).read_bytes())
    if active_mode:
        training.update(residual_source_sha256=cruise_meta['residual_source_sha256'],residual_variant=active_head.VARIANT,
            active_residual_source_sha256=backend.digest(active_head.__file__),active_residual_bound_rad=.02,
            active_residual_feature_dim=69,active_residual_gate=active_head.GATE,
            active_residual_action_joints=['left_hip_roll_joint','right_hip_roll_joint'],
            frozen_cruise_checkpoint=str(cruise_path),frozen_cruise_checkpoint_sha256=backend.digest(cruise_path),
            frozen_cruise_training_sha256=backend.digest(cruise_path.parent/'training.json'),
            policy_composition='Frozen cruise PPO and CEM20; bounded active-left two-hip-roll PPO correction',
            research_reference=['https://arxiv.org/abs/1812.03201','https://arxiv.org/abs/1804.02717'],
            reward='Cruise force/stability costs plus command-speed tracking, body-forward progress and swings only while moving; 2*(teacher heading error/10deg)^2 moving or 2*(hold heading error/5deg)^2 holding and 2*(hold drift/.03m)^2; all rates multiplied by dt.')
        (directory/'residual_source.py').write_bytes((cruise_path.parent/'residual_source.py').read_bytes())
        (directory/'active_residual_source.py').write_bytes(Path(active_head.__file__).read_bytes())
        if args.privileged_critic:
            training.update(critic_feature_dim=80,critic_privileged_features=['protocol_onehot3','teacher_heading_error',
                'held_heading_error','local_stop_anchor_delta2','time_to_fade_and_stop2','settled_age','anchor_valid'],
                critic_research_reference='https://arxiv.org/abs/1710.06542',actor_feature_dim=69)
        if active_seed is not None:
            training.update(active_initialization_checkpoint=str(seed_path),active_initialization_checkpoint_sha256=backend.digest(seed_path),
                active_initialization_training_sha256=backend.digest(seed_path.parent/'training.json'),
                critic_initialization='Copy previous learned critic; zero added input columns; fresh optimizer.')
        if args.active_long:
            training.update(training_timeout='Per-world recorded-state truncations: backward min(start+24,90.8)s; M5 at55.8s; M6 at24.8s, with RSL timeout bootstrap.',
                protocol_sampling='Fixed roughly equal world counts by protocol; even-index warm states independently resampled within each protocol on reset.')
            training['training_hard_guards'].update(hold_drift_m=.03,settled_hold_heading_rad=math.radians(5),settled_hold_speed_mps=.05,settled_hold_yaw_speed_rad_s=.1)
            training.update(early_start_fraction=args.early_start_fraction,early_start_age_range_s=[3.,7.],
                coverage_protocol_order=['backward','M5','M6'],
                coverage_semantics='Per-episode first healthy stop/settled observation, including warm starts already in hold; successful timeout excludes all failures.',
                force_margin_scope='Training-only stricter termination; independent acceptance remains 3 BW.')
        training['reward']=f'Existing force penalty coefficient {args.force_cost}, failure cost {args.failure_cost}, moving teacher-heading scale {args.heading_scale_deg}deg; command-speed/body-progress and hold costs otherwise preserved.'
    else:(directory/'residual_source.py').write_bytes(Path(__file__).read_bytes())
    # Training dependencies are separate from the frozen deployment turn source.
    from algorithms.urdf_learn_wasd_walk import landau_turn_control as audit_implementation
    snapshots=directory/'training_sources';snapshots.mkdir()
    (snapshots/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    training['training_source_sha256']={Path(__file__).name:backend.digest(__file__)}
    for module in (audit_implementation,direction,batches,backend)+((active_bank,) if active_mode else ()):
        path=Path(module.__file__);(snapshots/path.name).write_bytes(path.read_bytes())
        training['training_source_sha256'][path.name]=backend.digest(path)
    training['audit_buffer_ownership']='Warp arrays allocated before graph capture retained by batch._physics_audit_buffers'
    (directory/'model.xml').write_text(batch.xml)
    training['curriculum']['bank_sha256']=backend.digest(directory/'warm_states.npz')
    original_save=runner.save
    def save(path,*a,**kw):
        original_save(path,*a,**kw)
        checkpoint=torch.load(path,map_location='cpu',weights_only=False)
        checkpoint.update(parameters=blob['parameters'],turn_parameters=blob['turn_parameters'],
            standing_actor_state_dict=blob['standing_actor_state_dict'],residual_actor_state_dict=checkpoint['actor_state_dict'])
        if active_mode:
            checkpoint['residual_variant']=active_head.VARIANT
            checkpoint['active_residual_actor_state_dict']=checkpoint['actor_state_dict']
            checkpoint['residual_actor_state_dict']=cruise_blob['residual_actor_state_dict']
        torch.save(checkpoint,path)
        training['checkpoints'][Path(path).name]=backend.digest(path)
        backend.write_json(directory/'training.json',training)
    runner.save=save
    original_update=runner.alg.update
    def update():
        result=original_update()
        with (directory/'iterations.jsonl').open('a') as stream:
            stream.write(json.dumps({'control_steps':env.common_step_counter,'wall_s':time.perf_counter()-began,**stats})+'\n')
        return result
    runner.alg.update=update
    runner.save(str(directory/'model_initial.pt'))
    runner.learn(num_learning_iterations=args.iterations,init_at_random_ep_len=False)
    training.update(status='complete',wall_s=time.perf_counter()-began,training_statistics=stats)
    backend.write_json(directory/'training.json',training)


def rebase(args):
    """Compose a calibrated active-turn parent with unchanged learned cruise weights."""
    import torch
    from algorithms.urdf_learn_wasd_walk import mujoco_backend as backend
    parent=Path(args.checkpoint).resolve();old=Path(args.residual_checkpoint).resolve()
    for path in (parent,old):path.relative_to((backend.OUTPUT/'training').resolve())
    new_meta=json.loads((parent.parent/'training.json').read_text())
    old_meta=json.loads((old.parent/'training.json').read_text())
    if new_meta['checkpoints'][parent.name]!=backend.digest(parent) or old_meta['checkpoints'][old.name]!=backend.digest(old):
        raise ValueError('Composition input checkpoint changed')
    if old_meta.get('residual_variant') is not None:raise ValueError('Rebase supports cruise-only residuals')
    if new_meta['policy_family']!='periodic_feedback_cem' or old_meta['policy_family']!=FAMILY:
        raise ValueError('Require calibrated CEM parent and learned cruise residual')
    if any(str(meta.get('status','')).startswith('invalidated_') for meta in (new_meta,old_meta)):
        raise ValueError('Invalidated composition input')
    original=torch.load(old,map_location='cpu',weights_only=False)
    fresh=torch.load(parent,map_location='cpu',weights_only=False)
    if fresh['turn_parameters'].shape!=(20,) or original['turn_parameters'].shape!=(20,):raise ValueError('Require20parameters')
    frozen=[i for i in range(20) if i not in (5,6,17)]
    if not torch.equal(fresh['parameters'],original['parameters']) or not torch.equal(fresh['turn_parameters'][frozen],original['turn_parameters'][frozen]):
        raise ValueError('Only active sway parameters5/6and left-yaw scale17may change')
    left=fresh['standing_actor_state_dict'];right=original['standing_actor_state_dict']
    if left.keys()!=right.keys() or any(not torch.equal(left[k],right[k]) for k in left):raise ValueError('Standing prior changed')
    for key in ('model_xml_sha256','urdf_sha256','mesh_tree_sha256','action_joints','action_scale','nominal_q','nominal_ctrl'):
        if new_meta[key]!=old_meta[key]:raise ValueError('Composition identity mismatch: '+key)
    if backend.digest(parent.parent/'control_source.py')!=backend.digest(old.parent/'control_source.py'):
        raise ValueError('Frozen gait source changed')
    if backend.digest(old.parent/'residual_source.py')!=old_meta['residual_source_sha256']:raise ValueError('Residual source changed')
    folder=backend.OUTPUT/'training'/args.name;folder.mkdir(exist_ok=False)
    meta=copy.deepcopy(old_meta)
    meta.update(status='composed_requires_nominal_validation',parent_checkpoint=str(parent),parent_checkpoint_sha256=backend.digest(parent),
        parent_training_sha256=backend.digest(parent.parent/'training.json'),turn_source_sha256=new_meta['turn_source_sha256'],
        source_sha256=new_meta['source_sha256'],canonical_milestone_pass=False,
        composition={'learned_residual_checkpoint':str(old),'learned_residual_checkpoint_sha256':backend.digest(old),
            'calibrated_parent_checkpoint':str(parent),'changed_turn_indices':[i for i in (5,6,17) if float(original['turn_parameters'][i])!=float(fresh['turn_parameters'][i])],
            'old_active_sway':original['turn_parameters'][5:7].tolist(),'new_active_sway':fresh['turn_parameters'][5:7].tolist(),
            'old_left_yaw_scale':float(original['turn_parameters'][17]),'new_left_yaw_scale':float(fresh['turn_parameters'][17]),
            'scope':'No new PPO training; all cruise weights and parameters outside active sway/left-yaw calibration unchanged. Full cumulative proof required.'})
    meta['arguments']={**old_meta['arguments'],'name':args.name,'turn_duration':new_meta['arguments']['turn_duration'],'hip_roll_range':.4}
    for name in ('control_source.py','turn_source.py','teleop_contract.py','model.xml'):
        (folder/name).write_bytes((parent.parent/name).read_bytes())
    for name in ('residual_source.py','warm_states.npz'):(folder/name).write_bytes((old.parent/name).read_bytes())
    blob={key:original[key] for key in ('residual_actor_state_dict','actor_state_dict')}
    blob.update({key:fresh[key] for key in ('parameters','turn_parameters','standing_actor_state_dict')})
    torch.save(blob,folder/'model_0.pt');meta['checkpoints']={'model_0.pt':backend.digest(folder/'model_0.pt')}
    backend.write_json(folder/'training.json',meta)
    print(str(folder/'model_0.pt'),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name',required=True);parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--mode',choices=('train','rebase'),default='train')
    parser.add_argument('--residual-checkpoint');parser.add_argument('--curriculum');parser.add_argument('--num-envs',type=int,default=256)
    parser.add_argument('--privileged-critic',action='store_true');parser.add_argument('--active-seed')
    parser.add_argument('--active-long',action='store_true')
    parser.add_argument('--force-cost',type=float,default=8.)
    parser.add_argument('--training-force-limit',type=float,default=3.)
    parser.add_argument('--early-start-fraction',type=float,default=0.)
    parser.add_argument('--failure-cost',type=float,default=5.)
    parser.add_argument('--heading-scale-deg',type=float,default=10.)
    parser.add_argument('--active-curriculum',nargs=3,metavar=('BACKWARD','M5','M6'))
    parser.add_argument('--iterations',type=int,default=200);parser.add_argument('--seconds',type=float,default=6.)
    args=parser.parse_args()
    if args.mode=='rebase':
        if not args.residual_checkpoint:raise ValueError('Rebase requires a learned residual checkpoint')
        rebase(args);return
    if not args.curriculum and not args.active_curriculum:raise ValueError('Training requires a recorded curriculum')
    if args.active_curriculum and not args.residual_checkpoint:raise ValueError('Active training requires frozen cruise checkpoint')
    if (args.privileged_critic or args.active_seed) and not args.active_long:raise ValueError('Privileged critic/active seed require long active training')
    if args.active_long and not args.active_curriculum:raise ValueError('Long mixed episodes require active curriculum')
    if not 2.85<=args.training_force_limit<=3. or not 0.<=args.early_start_fraction<=.75:raise ValueError('Invalid training margin or sampling mixture')
    if (args.training_force_limit!=3. or args.early_start_fraction) and not args.active_long:raise ValueError('Margin curriculum requires long active training')
    if not math.isfinite(args.force_cost) or not math.isfinite(args.failure_cost) or args.force_cost<=0 or args.failure_cost<=0 or not 1.<=args.heading_scale_deg<=10.:raise ValueError('Invalid bounded residual reward weights')
    if not 16<=args.num_envs<=512 or not 1<=args.iterations<=1000 or args.seconds!=6.:
        raise ValueError('Bounded initial curriculum requires16..512worlds,1..1000iterations and6secondepisodes')
    train(args)


if __name__=='__main__':main()
