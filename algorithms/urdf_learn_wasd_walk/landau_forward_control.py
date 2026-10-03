"""Bounded RSL PPO walking continuation on the certified Landau MuJoCo model.

Semantic forward is body +Y. No auxiliary forces, root guidance or pose clamps.
Periodic contact shaping follows https://arxiv.org/abs/2011.01387; the physical
model stays fixed while joint target ranges reflect Landau's measured leg reach.
"""
import argparse
import copy
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

OBS_DIM = 70
PERIOD = .6


def action_scales(names, hip_roll=.2):
    values = []
    for name in names:
        values.append(.35 if 'hip_pitch' in name else .5 if 'knee' in name else
                      .4 if 'ankle_pitch' in name else hip_roll if 'hip_roll' in name else
                      .12 if any(k in name for k in ('hip_yaw', 'toe', 'shoulder')) else .08)
    return values


def expanded_state(state, target):
    """Append zero-connected features while retaining the trained normalizer."""
    result = {}
    for key, value in state.items():
        destination = target[key].clone()
        if destination.shape == value.shape:
            result[key] = value
        elif key == 'mlp.0.weight':
            destination.zero_(); destination[:, :value.shape[1]] = value
            result[key] = destination
        elif key.startswith('obs_normalizer.') and value.ndim == 2:
            destination.fill_(0. if key.endswith('_mean') else 1.)
            destination[:, :value.shape[1]] = value
            result[key] = destination
        else:
            raise ValueError(f'Unsupported observation expansion: {key}')
    return result


def make_actor(dimension, device='cpu'):
    import torch
    from tensordict import TensorDict
    from rsl_rl.models import MLPModel
    return MLPModel(TensorDict({'actor':torch.zeros(1,dimension,device=device)},batch_size=[1]),
        {'actor':['actor']},'actor',17,hidden_dims=[128,128],activation='elu',obs_normalization=True,
        distribution_cfg={'class_name':'GaussianDistribution','init_std':.25,'std_type':'scalar'}).to(device)


def install_moving_exploration_floor(actor, names):
    """Condition the actual PPO Gaussian on raw command, including minibatches."""
    import torch
    from torch.distributions import Normal
    floors=[.75 if 'hip_pitch' in n or 'ankle_pitch' in n else 1.25 if 'knee' in n
            else .375 if 'hip_roll' in n else 0. for n in names]
    floor=torch.tensor(floors,device=next(actor.parameters()).device)
    original=actor.forward
    def forward(obs,masks=None,hidden_state=None,stochastic_output=False):
        if not stochastic_output:
            return original(obs,masks,hidden_state,False)
        if masks is not None:raise ValueError('Exploration comparison requires unpadded MLP batches')
        mean=original(obs,None,hidden_state,False)
        actor.distribution.update(mean)
        std=actor.distribution.std
        moving=(obs['actor'][...,63]!=0)[...,None]
        std=torch.where(moving,torch.maximum(std,floor),std)
        actor.distribution._distribution=Normal(mean,std)
        return actor.distribution.sample()
    actor.forward=forward
    return floors


def initialize_fresh_moving(actor, critic, standing_state):
    """Fresh locomotion weights with physical input scales and a saved stand branch.

    Only zero-command inference selects the frozen, already trained branch.
    Moving control starts at nominal PD targets; no external assistance is added.
    """
    import torch
    device=next(actor.parameters()).device
    prior=make_actor(63,device)
    prior.load_state_dict(standing_state,strict=True)
    prior.requires_grad_(False);prior.eval();prior.obs_normalizer.until=0
    actor.add_module('standing_prior',prior)
    # Observe: linear(3), angular(3), gravity(3), q(17), 0.1*qdot(17),
    # previous applied actor units(17), drift(2), time(1), command/phase/heading(7).
    denominators=[.3]*3+[2.]*3+[1.]*3+[.3]*17+[.4]*17+[3.]*17+[1.]*10
    with torch.no_grad():
        for network in (actor,critic):
            normalizer=network.obs_normalizer
            normalizer._mean.zero_();normalizer._mean[0,8]=-1.
            normalizer._std.copy_(torch.tensor(denominators,device=device)[None]-.01)
            normalizer._var.copy_(normalizer._std.square());normalizer.until=0
        actor.mlp[-1].weight.zero_();actor.mlp[-1].bias.zero_()
    return denominators


def policy_mean(actor, observation):
    """Deterministic policy composition, including per-row command selection."""
    from tensordict import TensorDict
    result=actor(TensorDict({'actor':observation},batch_size=observation.shape[:-1]))
    if hasattr(actor,'standing_prior'):
        standing=observation[...,63]==0
        if standing.any():
            selected=observation[standing,:63]
            result[standing]=actor.standing_prior(TensorDict({'actor':selected},batch_size=selected.shape[:-1]))
    return result


def load_gait_source(checkpoint, metadata):
    """Load this run's hash-bound controller snapshot, independent of later edits."""
    import importlib.util
    source=Path(checkpoint).parent/'control_source.py'
    expected=[value for path,value in metadata['source_sha256'].items() if Path(path).name=='landau_gait_search.py']
    actual=hashlib.sha256(source.read_bytes()).hexdigest()
    if len(expected)!=1 or actual!=expected[0]:raise ValueError('Saved parametric controller source changed')
    spec=importlib.util.spec_from_file_location('landau_gait_'+actual[:16],source)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module,source


def train(args):
    import mujoco
    import torch
    from tensordict import TensorDict
    from mjlab.rl import MjlabOnPolicyRunner, RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
    from algorithms.urdf_learn_wasd_walk.landau_rsl_control import configure_model
    backend, batches, variant = configure_model('balanced_hands_v1')
    parent = Path(args.checkpoint).resolve()
    parent.relative_to((backend.OUTPUT/'training').resolve())
    parent_meta = json.loads((parent.parent/'training.json').read_text())
    if parent_meta['checkpoints'][parent.name] != backend.digest(parent):
        raise ValueError('Parent checkpoint hash changed')
    directory = (backend.OUTPUT/'training'/args.name).resolve()
    directory.relative_to((backend.OUTPUT/'training').resolve())
    directory.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4); torch.manual_seed(42)
    began = time.perf_counter()
    batch = batches.WarpBatch(args.num_envs,42,0.,'stand','load',.004,args.forward,.0225)
    if hashlib.sha256(batch.xml.encode()).hexdigest() != parent_meta['model_xml_sha256']:
        raise ValueError('Walking model differs from certified parent model')
    batch.commands[:] = args.forward
    if args.initialization!='fresh_moving':batch.commands[::4] = 0.
    moving = batch.commands != 0
    scales = torch.tensor(action_scales(batch.spec['action_joints'],args.hip_roll_range),device='cuda')
    scales = torch.where(moving[:,None],scales[None,:],.08)
    bounds = scales/.08 if args.action_mapping=='legacy_radians_v2' else torch.ones_like(scales)
    applied_scales = torch.full_like(scales,.08) if args.action_mapping=='legacy_radians_v2' else scales
    total_weight = float(batch.model.body_mass.sum())*9.81
    stats = {}
    def progress(state, iteration):
        path=backend.OUTPUT.parent/'backend_progress.json'
        record=json.loads(path.read_text()) if path.exists() else {}
        record.update(updated_at=datetime.now(timezone.utc).isoformat(),current_gate='gate_5m_no_reset',
            simulator='mujoco_warp_cuda',variant='balanced_hands_v1',iteration=iteration,
            active_process=args.name if state=='running' else None,
            next_step='Train current walking comparison; exact checkpoint validation and video follow.' if state=='running' else
                      'Evaluate the saved walking checkpoint and zero-command standing before any promotion.',
            artifact_paths=[str(directory)],assistance_coefficient=0.)
        backend.write_json(path,record)
    air_age=torch.zeros((args.num_envs,2),device='cuda')
    air_peak=torch.zeros_like(air_age)
    touched=torch.zeros_like(air_age,dtype=torch.bool)
    completed_swings=torch.zeros(2,device='cuda',dtype=torch.long)

    class Env:
        num_envs=args.num_envs
        num_actions=17
        max_episode_length=1500
        device='cuda:0'
        cfg={'task':'finite_30s_stand_or_forward','reward_units':'rate times 0.02 s'}
        common_step_counter=0

        @property
        def unwrapped(self): return self
        @property
        def episode_length_buf(self): return batch.episode_steps

        def get_observations(self):
            drift = torch.cat((batch.pos[:,batch.pelvis,:2]-batch.reference_xy,
                               torch.zeros((args.num_envs,1),device=self.device)),dim=1)
            local = torch.bmm(batch.rot[:,batch.base].transpose(1,2),drift[:,:,None]).squeeze(-1)
            local = torch.where(moving[:,None],0.,local[:,:2]/.03)
            remaining = 1.-batch.episode_steps.float()/1500
            phase = 2*torch.pi*batch.episode_steps.float()*.02/PERIOD
            extra = torch.stack((batch.commands,torch.zeros_like(phase),torch.zeros_like(phase),
                                 torch.where(moving,phase.sin(),0.),torch.where(moving,phase.cos(),0.),
                                 batch.rot[:,batch.base,1,0],batch.rot[:,batch.base,0,0]-1),dim=1)
            obs = torch.cat((batch.observations(),local,remaining[:,None],extra),dim=1)
            return TensorDict({'actor':obs,'critic':obs},batch_size=[self.num_envs])

        def step(self, actions):
            raw = actions
            actions = torch.maximum(torch.minimum(actions,bounds),-bounds)
            batch.ctrl[:] = batch.targets
            batch.ctrl[:,batch.aids] += applied_scales*actions
            batch.wrench.zero_()
            torch.cuda.synchronize(); batches.wp.capture_launch(batch.graph); batches.wp.synchronize()
            batch.episode_steps += 1
            occupancy = batch.maximum_occupancy.numpy()
            if max(occupancy[:2]) >= batch.wd.naconmax or occupancy[2] >= batch.wd.njmax:
                raise RuntimeError('Training contact capacity exceeded')
            tilt = batch.rot[:,batch.base,2,2].clamp(-1,1).acos()
            drift = (batch.pos[:,batch.pelvis,:2]-batch.reference_xy).norm(dim=1)
            fallen = (tilt>torch.pi/6) | (batch.pos[:,batch.pelvis,2]<batch.reference_height-.08)
            invalid = ~torch.isfinite(batch.q).all(dim=1) | ~torch.isfinite(batch.v).all(dim=1)
            timeout = batch.episode_steps>=1500
            done = fallen | invalid | timeout
            linear, angular = batch.local_velocity()
            contacts, slip, clearance, forces = batch.feet(with_forces=True)
            completed_swings.add_((contacts & touched & (air_age>=.06) & (air_peak>=.015) & moving[:,None]).sum(0))
            air_age.copy_(torch.where(contacts,0.,air_age+.02))
            air_peak.copy_(torch.where(contacts,0.,torch.maximum(air_peak,clearance)))
            touched.logical_or_(contacts)
            phase = batch.episode_steps.float()*.02/PERIOD
            sine = (2*torch.pi*phase).sin()
            left_load = .5+.5*(-2*sine).clamp(-1,1)
            target_load = torch.stack((left_load,1-left_load),dim=1)
            target_clear = .018*torch.stack((sine.clamp_min(0).square(),(-sine).clamp_min(0).square()),dim=1)
            load_score = (-((forces/total_weight-target_load).square().sum(-1))/.25).exp()
            clear_score = (-((clearance-target_clear).square().sum(-1))/(.008**2)).exp()
            gait = .75*load_score+.25*clear_score
            heading = torch.atan2(batch.rot[:,batch.base,1,0],batch.rot[:,batch.base,0,0])
            track = (-(linear[:,1]-batch.commands).square()/.0225-linear[:,0].square()/.0225-heading.square()/.25).exp()
            act_cost = .02*(actions/bounds).square().mean(-1)+.01*((actions-batch.previous)/bounds).square().mean(-1)
            speed_cost = .2*(batch.v[:,batch.jv].abs()-4.).clamp_min(0).square().mean(-1)
            stand = torch.exp(-drift.square()/.03**2)-4*tilt.square()-.02*batch.v.square().mean(-1)-act_cost
            walk = 1+2*track+.5*gait-args.moving_tilt_weight*tilt.square()-.25*angular[:,2].square()-.15*slip-act_cost-speed_cost
            reward = torch.where(moving,walk,stand)
            reward = torch.where(fallen|invalid,-5.,reward)*.02
            stats.update(forward_mps=float(linear[moving,1].mean()), gait=float(gait[moving].mean()),
                slip_mps=float(slip[moving].mean()), max_clearance_m=float(clearance[moving].max()),
                fall_fraction=float((fallen|invalid).float().mean()), saturation=float((raw.abs()>bounds).float().mean()),
                stand_drift_m=float(drift[~moving].mean()) if (~moving).any() else None, heading_abs_rad=float(heading[moving].abs().mean()))
            if args.moving_noise_floor:
                stats.update(clearance_p95_m=torch.quantile(clearance[moving],.95,dim=0).tolist(),
                    completed_swings=completed_swings.tolist(),
                    max_action_joint_target_error_rad=float((batch.ctrl[:,batch.aids]-batch.q[:,batch.jq]).abs().max()))
            batch.previous = actions.clone()
            if done.any():
                travel = batch.pos[:,batch.pelvis,1]-batch.reference_xy[1]
                success = timeout & ~fallen & ~invalid & torch.where(moving,travel>=5.,drift<.03)
                batch.completed.extend(success[done].tolist()); batch.total_resets += int(done.sum())
                batch.reset(done)
                air_age[done]=0.;air_peak[done]=0.;touched[done]=False
            self.common_step_counter += 1
            return self.get_observations(),reward,done.float(),{'log':{'Walking/'+k:v for k,v in stats.items() if v is not None}}

    cfg=RslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(hidden_dims=(128,128),activation='elu',obs_normalization=True,
            distribution_cfg={'class_name':'GaussianDistribution','init_std':.25,'std_type':'scalar'}),
        critic=RslRlModelCfg(hidden_dims=(128,128),activation='elu',obs_normalization=True),
        algorithm=RslRlPpoAlgorithmCfg(value_loss_coef=1.,use_clipped_value_loss=True,clip_param=.2,
            entropy_coef=.001,num_learning_epochs=5,num_mini_batches=4,learning_rate=1e-4,
            schedule='adaptive',gamma=.99,lam=.95,desired_kl=.01,max_grad_norm=1.),
        experiment_name='landau_forward',save_interval=100,num_steps_per_env=24,
        max_iterations=args.iterations,logger='tensorboard',upload_model=False)
    config=asdict(cfg); env=Env()
    runner=MjlabOnPolicyRunner(env,copy.deepcopy(config),str(directory),device='cuda:0')
    blob=torch.load(parent,map_location='cuda:0',weights_only=False)
    equivalence=None
    physical_denominators=None
    if parent_meta['observation_dim']==63 and args.initialization=='fresh_moving':
        physical_denominators=initialize_fresh_moving(runner.alg.actor,runner.alg.critic,blob['actor_state_dict'])
    elif parent_meta['observation_dim']==63:
        actor_state=expanded_state(blob['actor_state_dict'],runner.alg.actor.state_dict())
        critic_state=expanded_state(blob['critic_state_dict'],runner.alg.critic.state_dict())
        runner.alg.actor.load_state_dict(actor_state,strict=True);runner.alg.critic.load_state_dict(critic_state,strict=True)
        old=make_actor(63,'cuda:0');old.load_state_dict(blob['actor_state_dict']);old.eval();runner.alg.actor.eval()
        probe=old.obs_normalizer.mean+torch.randn(64,63,device='cuda')*(old.obs_normalizer.std+.01)
        expanded=torch.cat((probe,torch.zeros(64,7,device='cuda')),dim=1)
        with torch.no_grad():
            equivalence=float((old(TensorDict({'actor':probe},batch_size=[64]))-
                runner.alg.actor(TensorDict({'actor':expanded},batch_size=[64]))).abs().max())
        if equivalence>1e-5: raise ValueError('Standing actor changed during observation expansion')
        runner.alg.actor.train()
        with torch.no_grad():runner.alg.actor.distribution.std_param.fill_(.25)
    elif parent_meta['observation_dim']==OBS_DIM:
        if parent_meta['arguments'].get('initialization','transfer')!=args.initialization:
            raise ValueError('Resume policy composition changed')
        if args.initialization=='fresh_moving':
            prior=make_actor(63,'cuda:0');prior.requires_grad_(False);prior.obs_normalizer.until=0
            runner.alg.actor.add_module('standing_prior',prior)
            physical_denominators=parent_meta['normalization']['physical_denominators']
        if parent_meta.get('action_mapping','range_v1')!=args.action_mapping:
            raise ValueError('Resume action units changed')
        if parent_meta['period_s']!=PERIOD or parent_meta['action_scale']!=action_scales(batch.spec['action_joints'],args.hip_roll_range):
            raise ValueError('Resume control mapping changed')
        runner.load(str(parent),map_location='cuda:0')
    else: raise ValueError('Unsupported parent observation schema')
    # A changing normalizer changes the action distribution before PPO takes
    # any gradient. Keep inherited raw-input behavior fixed during transfer.
    runner.alg.actor.obs_normalizer.until = 0
    runner.alg.critic.obs_normalizer.until = 0
    exploration_floor = install_moving_exploration_floor(runner.alg.actor,batch.spec['action_joints']) if args.moving_noise_floor else None
    original_update=runner.alg.update
    def update():
        kls=[];get_kl=runner.alg.actor.get_kl_divergence
        def measured(*a,**kw):
            value=get_kl(*a,**kw);kls.append(value.mean().detach());return value
        runner.alg.actor.get_kl_divergence=measured
        try: outcome=original_update()
        finally: runner.alg.actor.get_kl_divergence=get_kl
        values=torch.stack(kls).cpu().tolist() if kls else []
        with (directory/'iterations.jsonl').open('a') as stream:
            stream.write(json.dumps({'control_steps':env.common_step_counter,'learning_rate':runner.alg.learning_rate,
                'first_pre_gradient_kl':values[0] if values else None,'mean_kl':sum(values)/len(values) if values else None,
                'wall_s':time.perf_counter()-began,**stats})+'\n')
        if env.common_step_counter%(24*20)==0:progress('running',env.common_step_counter//24)
        return outcome
    runner.alg.update=update
    metadata={'created_at':datetime.now(timezone.utc).isoformat(),'arguments':vars(args),
        'backend':'mujoco_warp_cuda','mujoco_version':mujoco.__version__,
        'model_xml_sha256':hashlib.sha256(batch.xml.encode()).hexdigest(),
        'urdf_sha256':batch.spec['source']['urdf_sha256'],'mesh_tree_sha256':batch.spec['source']['mesh_tree_sha256'],
        'action_joints':batch.spec['action_joints'],'nominal_q':batch.nominal_q.tolist(),
        'nominal_ctrl':batch.nominal_ctrl.tolist(),'action_scale':action_scales(batch.spec['action_joints'],args.hip_roll_range),
        'stand_action_scale':.08,'observation_dim':OBS_DIM,'reference_xy':batch.reference_xy.tolist(),
        'action_mapping':args.action_mapping,
        'applied_action_units_rad':.08 if args.action_mapping=='legacy_radians_v2' else 'per-joint range',
        'period_s':PERIOD,'mass_variant':variant,'runner_config':config,
        'parent_checkpoint':str(parent),'parent_checkpoint_sha256':backend.digest(parent),
        'standing_actor_expansion_max_error':equivalence,'optimizer_restarted':parent_meta['observation_dim']==63,
        'normalization':{'updates_frozen':True,'epsilon':.01,'new_feature_mean':0.,'new_feature_std':1.,
                         'physical_denominators':physical_denominators},
        'policy_composition':'fresh moving MLP; frozen certified standing MLP selected at zero command' if args.initialization=='fresh_moving' else 'single transferred MLP',
        'moving_command_fraction':1. if args.initialization=='fresh_moving' else .75,
        'moving_exploration_std_floor_legacy_units':exploration_floor,
        'observation_contract':'60 proprio + 2 stand-only local drift + remaining time + forward/strafe/yaw + sin/cos phase + sin/cos-minus-one heading',
        'source_sha256':{str(p):backend.digest(p) for p in (Path(__file__),Path(batches.__file__),Path(backend.__file__))},
        'packages':{p:importlib.metadata.version(p) for p in ('torch','mjlab','mujoco','mujoco-warp','rsl-rl-lib')},
        'assistance':0.,'finite_horizon':'30 s true terminal; remaining time observed; no timeout bootstrap'}
    backend.write_json(directory/'metadata.json',metadata)
    (directory/'model.xml').write_text(batch.xml);(directory/'control_source.py').write_text(Path(__file__).read_text())
    runner.save(str(directory/'model_initial.pt'))
    progress('running',0)
    runner.learn(num_learning_iterations=args.iterations,init_at_random_ep_len=False)
    metadata.update(wall_s=time.perf_counter()-began,total_resets=batch.total_resets,
        recent_30s_success_rate=sum(batch.completed)/max(1,len(batch.completed)),
        checkpoints={p.name:backend.digest(p) for p in directory.glob('model_*.pt')})
    backend.write_json(directory/'training.json',metadata)
    progress('completed',args.iterations)
    print(json.dumps({'wall_s':metadata['wall_s'],'total_resets':batch.total_resets}))


def evaluate(args):
    import numpy as np
    import torch
    from tensordict import TensorDict
    from types import SimpleNamespace
    from algorithms.urdf_learn_wasd_walk import mujoco_policy as old_policy
    from algorithms.urdf_learn_wasd_walk.landau_rsl_control import configure_model
    backend, _, _ = configure_model('balanced_hands_v1')
    checkpoint=Path(args.checkpoint).resolve()
    checkpoint.relative_to((backend.OUTPUT/'training').resolve())
    meta=json.loads((checkpoint.parent/'training.json').read_text())
    if backend.digest(checkpoint)!=meta['checkpoints'][checkpoint.name] or meta['observation_dim']!=OBS_DIM:
        raise ValueError('Checkpoint provenance or observation schema mismatch')
    if meta['period_s']!=PERIOD or meta['action_scale']!=action_scales(meta['action_joints'],meta['arguments'].get('hip_roll_range',.2)):
        raise ValueError('Checkpoint control mapping mismatch')
    blob=torch.load(checkpoint,map_location='cpu',weights_only=False)
    turn_mode=getattr(args,'turn',False)
    teleop_mode=getattr(args,'teleop',False)
    direction_origin=None
    direction=getattr(args,'direction',None)
    if sum((bool(turn_mode),bool(teleop_mode),bool(direction)))>1:raise ValueError('Choose one command test')
    if direction:
        from algorithms.urdf_learn_wasd_walk import landau_direction_contract as direction_contract
    if teleop_mode:
        from algorithms.urdf_learn_wasd_walk import landau_teleop_contract as teleop_contract
        teleop_profile=teleop_contract.command_profile
        if meta.get('memory_version')==2:
            import importlib.util
            protocol_source=checkpoint.parent/'teleop_contract.py'
            if backend.digest(protocol_source)!=meta.get('teleop_contract_sha256'):
                raise ValueError('Saved teleop protocol changed')
            protocol_spec=importlib.util.spec_from_file_location('saved_teleop_protocol',protocol_source)
            saved_protocol=importlib.util.module_from_spec(protocol_spec);protocol_spec.loader.exec_module(saved_protocol)
            teleop_profile=saved_protocol.command_profile
    has_turn=meta.get('command_extension')=='yaw_v1'
    turn_memory={}
    if (turn_mode or teleop_mode or direction) and not has_turn:raise ValueError('Checkpoint has no trained yaw extension')
    if has_turn:
        import importlib.util
        turn_source=checkpoint.parent/'turn_source.py'
        if backend.digest(turn_source)!=meta['turn_source_sha256']:raise ValueError('Turn source changed')
        module_spec=importlib.util.spec_from_file_location('saved_turn_control',turn_source)
        turn_implementation=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(turn_implementation)
        if hasattr(turn_implementation,'configure_profile'):
            turn_implementation.configure_profile(meta['arguments'].get('turn_duration',14.))
    command_memory=turn_implementation.CommandMemory() if has_turn and meta.get('memory_version')==2 else None
    if meta.get('policy_family')=='periodic_feedback_cem':
        gait_implementation,gait_source=load_gait_source(checkpoint,meta)
        prior=make_actor(63);prior.load_state_dict(blob['standing_actor_state_dict'],strict=True);prior.eval()
        actor=SimpleNamespace(parameters=blob['parameters'],standing_prior=prior)
    else:
        actor=make_actor(OBS_DIM)
        if meta['arguments'].get('initialization')=='fresh_moving':
            prior=make_actor(63);prior.requires_grad_(False);prior.obs_normalizer.until=0
            actor.add_module('standing_prior',prior)
        actor.load_state_dict(blob['actor_state_dict'],strict=True)
        actor.eval()
    original_observe=old_policy.observe
    def observe(model,data,nominal,joints,previous):
        nonlocal direction_origin
        base,pelvis=model.body('base_link').id,model.body('root_x').id
        rotation=data.xmat[base].reshape(3,3)
        displacement=np.r_[data.xpos[pelvis,:2]-np.array(meta['reference_xy']),0.]
        drift=(rotation.T@displacement)[:2]/.03 if not args.forward else np.zeros(2)
        phase=2*np.pi*data.time/PERIOD
        extra=[args.forward,0.,0.,np.sin(phase) if args.forward else 0.,np.cos(phase) if args.forward else 0.,
               rotation[1,0],rotation[0,0]-1.]
        result=np.r_[original_observe(model,data,nominal,joints,previous),drift,1.-data.time/30.,extra].astype(np.float32)
        if turn_mode:
            forward,yaw,_=turn_implementation.command_profile(round(float(data.time)/.02)*.02)
            result[60:62]=0.;result[63:66]=[forward,0.,yaw]
        if teleop_mode:
            result[60:62]=0.;result[63:66]=teleop_profile(round(float(data.time)/.02)*.02)
        if direction:
            if direction_origin is None:direction_origin=data.xpos[pelvis,:2].copy()
            delta=data.xpos[pelvis,:2]-direction_origin
            result[60:62]=0.
            result[63:66]=direction_contract.command(direction,delta,np.arctan2(rotation[1,0],rotation[0,0]),float(data.time),args.forward)
        if command_memory is not None:
            adjusted,prior_obs=command_memory.observe(torch.from_numpy(result)[None],
                torch.tensor(data.xpos[pelvis].copy(),dtype=torch.float32)[None],
                torch.tensor(rotation.copy(),dtype=torch.float32)[None],float(data.time))
            turn_memory['prior_obs']=prior_obs[0].numpy()
            return adjusted[0].numpy()
        if has_turn:
            # The command stream drives one controller in every cumulative test.
            elapsed=float(data.time)-turn_memory.get('previous_time',float(data.time))
            reference=turn_memory.get('reference',0.)+turn_memory.get('previous_yaw',0.)*elapsed
            turned=turn_memory.get('turned',False) or result[65]!=0.
            prior_obs=result[:63].copy()
            if turned and np.max(np.abs(result[63:66]))==0.:
                if 'anchor_xy' not in turn_memory:
                    turn_memory.update(anchor_xy=data.xpos[pelvis,:2].copy(),anchor_time=float(data.time))
                delta=np.r_[data.xpos[pelvis,:2]-turn_memory['anchor_xy'],0.]
                prior_obs[60:62]=(rotation.T@delta)[:2]/.03
                prior_obs[62]=1.-(data.time-turn_memory['anchor_time'])/30.
            turn_memory.update(prior_obs=prior_obs,reference=reference,
                turned=bool(turned),previous_time=float(data.time),previous_yaw=float(result[65]))
        return result
    def load_actor(path,model,spec):
        if meta['urdf_sha256']!=spec['source']['urdf_sha256'] or meta['mesh_tree_sha256']!=spec['source']['mesh_tree_sha256']:
            raise ValueError('Evaluation asset mismatch')
        def inference(obs):
            if has_turn:
                return turn_implementation.commanded_action(gait_implementation,actor.parameters,actor.standing_prior,
                    obs[None],torch.from_numpy(turn_memory['prior_obs'])[None],blob['turn_parameters'],
                    meta['action_joints'],command_memory.reference if command_memory else turn_memory['reference'],
                    command_memory.turned if command_memory else turn_memory['turned'],
                    **({'memory':command_memory} if command_memory else {}))[0]
            if meta.get('policy_family')=='periodic_feedback_cem':
                return gait_implementation.evaluate_mean(actor,obs[None],meta['action_joints'])[0]
            return policy_mean(actor,obs[None])[0]
        return SimpleNamespace(actor=inference),meta
    old_policy.observe,old_policy.load_actor=observe,load_actor
    from algorithms.urdf_learn_wasd_walk.landau_forward_evaluation import run
    run(SimpleNamespace(name=args.name,backend='mujoco_warp_cuda',seconds=args.seconds,dt=.002,seed=42,
        pose='geometric',gain_scale=1.,noslip_iterations=0,contact_timeconst=.004,assistance=0.,
        checkpoint=str(checkpoint),forward=args.forward,target_distance_m=getattr(args,'target_distance',5.),turn=turn_mode,
        turn_hold_start=turn_implementation.HOLD_START if turn_mode else None,teleop=teleop_mode,direction=direction,render=False))
    if teleop_mode:
        protocol_path=protocol_source if meta.get('memory_version')==2 else Path(teleop_contract.__file__)
        (backend.OUTPUT/args.name/'command_protocol_source.py').write_text(protocol_path.read_text())
    if has_turn:
        (backend.OUTPUT/args.name/'turn_source.py').write_text(turn_source.read_text())
        backend.write_json(backend.OUTPUT/args.name/'controller_memory.json',{
            'anchor_xy':turn_memory.get('anchor_xy',np.array([])).tolist(),
            'anchor_time':turn_memory.get('anchor_time'),'simulation_reset':False,
            'dispatch':'command_driven_yaw_extension','turned':turn_memory.get('turned',False),
            'integrated_reference_rad':turn_memory.get('reference',0.),
            'turn_source_sha256':backend.digest(turn_source),**(command_memory.record() if command_memory else {})})
    if meta.get('policy_family')=='periodic_feedback_cem':
        source=gait_source
        (backend.OUTPUT/args.name/'gait_source.py').write_text(source.read_text())
        result_path=backend.OUTPUT/args.name/'dynamics.json'
        result=json.loads(result_path.read_text());result['gait_source_sha256']=backend.digest(source)
        if has_turn:
            result.update(turn_source_sha256=backend.digest(turn_source),
                controller_memory_sha256=backend.digest(backend.OUTPUT/args.name/'controller_memory.json'))
        if teleop_mode:result['command_protocol_source_sha256']=backend.digest(protocol_path)
        backend.write_json(result_path,result)
    backend.write_json(backend.OUTPUT/args.name/'transfer.json',{
        'checkpoint':str(checkpoint),'checkpoint_sha256':backend.digest(checkpoint),
        'initial_untrained':checkpoint.name=='model_initial.pt','profile':'balanced_hands_v1',
        'policy_composition':meta.get('policy_composition','single transferred MLP'),
        'action_mapping':meta.get('action_mapping','range_v1'),'canonical_milestone_pass':False})
    # Every evaluated candidate, including failures, gets its exact replay.
    import subprocess
    import sys
    subprocess.run([sys.executable,'-m','algorithms.urdf_learn_wasd_walk.continuation',
        '--mode','render','--render-directory',str(backend.OUTPUT/args.name)],check=True,timeout=240)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--name',required=True);p.add_argument('--checkpoint',required=True)
    p.add_argument('--mode',choices=('train','evaluate'),default='train')
    p.add_argument('--num-envs',type=int,default=512);p.add_argument('--iterations',type=int,default=1000)
    p.add_argument('--forward',type=float,default=.2);p.add_argument('--seconds',type=float,default=30.)
    p.add_argument('--target-distance',type=float,choices=(5.,10.),default=5.)
    p.add_argument('--turn',action='store_true',help='Evaluate the saved yaw-command profile and settled hold')
    p.add_argument('--direction',choices=('forward','left','right','backward'),help='M7: turn and walk through a world-direction 10 m gate')
    p.add_argument('--teleop',action='store_true',help='Run the fixed60s joystick command replay')
    p.add_argument('--action-mapping',choices=('range_v1','legacy_radians_v2'),default='range_v1')
    p.add_argument('--moving-noise-floor',action='store_true')
    p.add_argument('--hip-roll-range',type=float,default=.2)
    p.add_argument('--moving-tilt-weight',type=float,default=2.)
    p.add_argument('--initialization',choices=('transfer','fresh_moving'),default='transfer')
    args=p.parse_args()
    if not 4<=args.num_envs<=1024 or not 1<=args.iterations<=2000 or not 0<=args.forward<=.4 or not 0<args.seconds<=(240 if args.direction else 120):
        raise ValueError('Experiment exceeds bounded limits')
    if args.hip_roll_range not in (.2,.4):raise ValueError('Use an audited hip-roll range')
    if args.moving_tilt_weight not in (.5,2.):raise ValueError('Use a bounded tilt-penalty comparison')
    if args.direction and (args.mode!='evaluate' or args.target_distance!=10. or args.forward<=0):
        raise ValueError('Direction mode requires moving evaluation and a 10 m target')
    if args.mode=='train':
        if args.turn or args.teleop:raise ValueError('Use landau_turn_control for yaw/teleop training')
        if args.forward<=0:raise ValueError('Training requires a moving-command subset')
        if args.moving_noise_floor and args.action_mapping!='legacy_radians_v2':raise ValueError('Exploration floors use legacy radian units')
        train(args)
    else:evaluate(args)


if __name__=='__main__':main()
