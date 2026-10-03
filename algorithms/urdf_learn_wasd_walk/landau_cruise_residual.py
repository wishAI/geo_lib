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
    bank_meta=build_bank(args.curriculum,parent,turn,directory/'warm_states.npz')
    torch.set_num_threads(4);torch.manual_seed(42)
    batch=batches.WarpBatch(args.num_envs,42,0.,'stand','load',.004,.2,.0225)
    if hashlib.sha256(batch.xml.encode()).hexdigest()!=meta['model_xml_sha256']:raise ValueError('Physical model changed')
    audit=install_physics_audit(batch,batches,args.num_envs)
    bank={k:torch.tensor(v,device='cuda') for k,v in np.load(directory/'warm_states.npz').items()}
    walking=blob['parameters'].to('cuda');turn_parameters=blob['turn_parameters'].to('cuda')
    prior=make_prior(63,'cuda');prior.load_state_dict(blob['standing_actor_state_dict']);prior.eval();prior.requires_grad_(False)
    names=batch.spec['action_joints'];limits=torch.tensor(action_scales(names,.4),device='cuda')/.08
    weight=float(batch.model.body_mass.sum())*9.81
    memory=CruiseMemory(args.num_envs,'cuda')
    stats={};began=time.perf_counter()

    class Env:
        num_envs=args.num_envs;num_actions=2;device='cuda:0';max_episode_length=round(args.seconds/.02)
        cfg={'task':'recorded_cruise_residual','is_finite_horizon':False};common_step_counter=0
        def __init__(self):
            self.age=torch.zeros(self.num_envs,device=self.device,dtype=torch.float64)
            self.air=torch.zeros((self.num_envs,2),device=self.device);self.peak=torch.zeros_like(self.air)
            self.touched=torch.zeros_like(self.air,dtype=torch.bool)
            self.previous_residual=torch.zeros((self.num_envs,2),device=self.device)
            self.completed=0;self.failed=0;self.returns=torch.zeros(self.num_envs,device=self.device)
            self.reset(torch.ones(self.num_envs,device=self.device,dtype=torch.bool))
        @property
        def unwrapped(self):return self
        @property
        def episode_length_buf(self):return batch.episode_steps
        def reset(self,mask):
            count=int(mask.sum())
            if not count:return
            ix=torch.randint((len(bank['age'])+1)//2,(count,),device=self.device)*2
            batch.q[mask]=bank['qpos'][ix].float();batch.v[mask]=bank['qvel'][ix].float()
            batch.previous[mask]=bank['previous'][ix].float();batch.ctrl[mask]=batch.targets
            wp.to_torch(batch.wd.qacc_warmstart)[mask]=0;wp.to_torch(batch.wd.qfrc_applied)[mask]=0
            batch.wrench[mask]=0;wp.to_torch(batch.wd.time)[mask]=bank['age'][ix].float()
            batch.episode_steps[mask]=0;self.age[mask]=bank['age'][ix]
            memory._reference[mask]=bank['reference'][ix];memory._blend[mask]=bank['blend'][ix]
            memory._feedback[mask]=bank['feedback'][ix];memory.left_cruise_start[mask]=bank['cruise_start'][ix]
            memory.previous_yaw[mask]=bank['previous_yaw'][ix].double()
            self.air[mask]=0;self.peak[mask]=0;self.touched[mask]=False;self.previous_residual[mask]=0
            self.returns[mask]=0
            for value in audit.values():value[mask]=0
            torch.cuda.synchronize();wp.capture_launch(batch.forward_graph);wp.synchronize()
        def get_observations(self):
            rotation=batch.rot[:,batch.base].double()
            heading=torch.atan2(rotation[:,1,0],rotation[:,0,0])
            command=direction.commands('backward',batch.pos[:,batch.pelvis,:2].double()-batch.reference_xy.double(),heading,100.)
            extra=torch.zeros((self.num_envs,10),device=self.device)
            extra[:,2]=(1.-self.age/30.).float();extra[:,3:6]=command.float()
            phase=2*torch.pi*self.age/.6
            extra[:,6]=phase.sin().float();extra[:,7]=phase.cos().float()
            extra[:,8]=rotation[:,1,0].float();extra[:,9]=(rotation[:,0,0]-1).float()
            self.observation=torch.cat((batch.observations(),extra),dim=1)
            self.command=command
            value=features(self.observation,walking,memory)
            return TensorDict({'actor':value,'critic':value},batch_size=[self.num_envs])
        def step(self,latent):
            with torch.no_grad():
                baseline=turn.commanded_action(base,walking,prior,self.observation,self.observation[:,:63],
                    turn_parameters,names,memory.reference.float(),True,memory=memory)
                action=compose(baseline,latent,memory,names,self.observation).clamp(-limits,limits)
            before=batch.pos[:,batch.pelvis,:2].clone()
            residual=latent.tanh()*memory.left_cruise_blend.float()[:,None]
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
            failed|=(audit['speed']>4.+1e-6)|(force>3.)|(audit['nonfoot']!=0)|(audit['flight']>60)
            failed|=~torch.isfinite(batch.q).all(1)|~torch.isfinite(batch.v).all(1)
            contacts,slip,clearance,_=batch.feet(with_forces=True)
            landed=contacts&self.touched&(self.air>=.06)&(self.peak>=.015)
            self.air=torch.where(contacts,0.,self.air+.02)
            self.peak=torch.where(contacts,0.,torch.maximum(self.peak,clearance));self.touched|=contacts
            linear,_=batch.local_velocity()
            progress=-(batch.pos[:,batch.pelvis,1]-before[:,1])
            reward=.02*(1.+2.*torch.exp(-(linear[:,1]-.2).square()/.0225)
                -8.*(force-2.5).clamp_min(0).square()-2.*tilt.square()-2.*slip
                -.1*residual.square().sum(1)-.1*(residual-self.previous_residual).square().sum(1))
            reward+=2.*progress+.1*landed.float().sum(1)-5.*failed.float()
            reward=torch.nan_to_num(reward,nan=-5.,posinf=-5.,neginf=-5.)
            self.previous_residual=residual.detach().clone();self.returns+=reward
            timeout=batch.episode_steps>=self.max_episode_length;done=failed|timeout
            # Integrate the preceding command once, then update the live-pose steering dwell.
            self.get_observations()
            memory.advance(self.observation[:,65].double(),self.age,.02)
            self.completed+=int(done.sum());self.failed+=int(failed.sum())
            stats.update(completed=self.completed,failed=self.failed,mean_interval_peak_bw=float(force.mean()),
                max_interval_peak_bw=float(force.max()),mean_forward_mps=float(linear[:,1].mean()),
                mean_abs_residual_rad=float(residual.abs().mean())*BOUND_RAD)
            extras={'time_outs':(timeout&~failed).float(),'log':dict(stats)}
            self.reset(done)
            return self.get_observations(),reward,done.float(),extras

    env=Env()
    cfg=RslRlOnPolicyRunnerCfg(actor=RslRlModelCfg(hidden_dims=(64,64),activation='elu',obs_normalization=False,
        distribution_cfg={'class_name':'GaussianDistribution','init_std':.1,'std_type':'scalar'}),
        critic=RslRlModelCfg(hidden_dims=(64,64),activation='elu',obs_normalization=False),
        algorithm=RslRlPpoAlgorithmCfg(value_loss_coef=1.,use_clipped_value_loss=True,clip_param=.2,entropy_coef=.001,
            num_learning_epochs=5,num_mini_batches=4,learning_rate=3e-4,schedule='adaptive',gamma=.995,lam=.95,desired_kl=.01,max_grad_norm=1.),
        experiment_name='landau_cruise_residual',save_interval=50,num_steps_per_env=24,max_iterations=args.iterations,
        logger='tensorboard',upload_model=False)
    config=asdict(cfg);runner=MjlabOnPolicyRunner(env,copy.deepcopy(config),str(directory),device='cuda:0')
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
        training_hard_guards={'support_force_bw':3.,'joint_speed_rad_s':4.,'max_flight_s':.12,'tilt_rad':math.pi/6,'root_drop_m':.08},
        reward='dt*(alive+2*forward_tracking-8*excess_peak_over2.5BW^2-2*tilt^2-2*slip-.1*residual^2-.1*residual_change^2)+2*world_backward_progress+.1*qualified_swings-5*failure',
        training_timeout='6s curriculum truncation with RSL timeout bootstrap; nominal-start evaluation unchanged')
    for filename in ('control_source.py','turn_source.py','teleop_contract.py'):
        (directory/filename).write_bytes((parent.parent/filename).read_bytes())
    (directory/'residual_source.py').write_bytes(Path(__file__).read_bytes())
    # Training dependencies are separate from the frozen deployment turn source.
    from algorithms.urdf_learn_wasd_walk import landau_turn_control as audit_implementation
    snapshots=directory/'training_sources';snapshots.mkdir()
    training['training_source_sha256']={}
    for module in (audit_implementation,direction,batches,backend):
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


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name',required=True);parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--curriculum',required=True);parser.add_argument('--num-envs',type=int,default=256)
    parser.add_argument('--iterations',type=int,default=200);parser.add_argument('--seconds',type=float,default=6.)
    args=parser.parse_args()
    if not 16<=args.num_envs<=512 or not 1<=args.iterations<=1000 or args.seconds!=6.:
        raise ValueError('Bounded initial curriculum requires16..512worlds,1..1000iterations and6secondepisodes')
    train(args)


if __name__=='__main__':main()
