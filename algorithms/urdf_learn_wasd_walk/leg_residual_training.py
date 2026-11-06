"""Contact-aware residual PPO for the anatomical model's first walking gate.

Frozen, hash-bound gait and standing policies; only bounded motor targets change.
Residual composition: https://arxiv.org/abs/1812.03201.
Phase/contact rewards: https://arxiv.org/abs/2011.01387.
Training resets never count as no-reset validation.
"""
import argparse
import copy
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time

FAMILY = 'anatomical_feedback_residual_ppo'
FEATURE_DIM = 80
TIMING_CONTRACT = 'anatomical_schedule_v2'


def velocity_tracking_score(linear,target_speed,ramp):
    """Keep useful velocity error signal across the explicit speed curriculum.

    Preserve the original 0.05 m/s kernel through S1; widen only forward
    tracking for later speeds. Lateral drift retains its original penalty.
    This is a training objective, never a speed acceptance measurement.
    """
    import torch
    target_speed=torch.as_tensor(target_speed,device=linear.device,dtype=linear.dtype)
    width=(.5*target_speed).clamp_min(.05)
    return (-(linear[:,1]+target_speed*ramp).square()/width**2-linear[:,0].square()/.0025).exp()


def next_curriculum_level(level,fallen,swings,forward):
    """Advance only after a complete supported walking episode; never mid-step."""
    success=not fallen and min(swings)>=2 and forward>0
    return min(1.,level+.05) if success else max(0.,level-.025)


def touchdown_quality(airborne_vertical_velocity):
    """Reduce the existing landing bonus for fast downward impact; add no force."""
    excess=(-airborne_vertical_velocity-.05).clamp_min(0)/.05
    return 1/(1+excess.square())


def curriculum_parameters(start,target,levels):
    return start[None]+levels[:,None]*(target-start)[None]


def reference_targets(age, parameters):
    """Use the frozen anatomical controller's transfer and lift schedule.

    A positive lean unloads the left foot. The first transfer starts from
    double support; subsequent transfers start from the other support foot.
    """
    import torch
    if parameters.ndim not in (1,2) or parameters.shape[-1] not in (20,21,22,23):raise ValueError('Expected anatomical schedule parameters')
    p=parameters.expand(len(age),-1)
    def smooth(value):
        value=value.clamp(0,1)
        return value.square()*(3-2*value)
    clock=(age-2).clamp_min(0)
    half=torch.floor(clock/(p[:,0]*.5))
    fraction=clock/(p[:,0]*.5)-half
    sign=1-2*half.remainder(2)
    transfer=sign*(2*smooth(fraction/p[:,4])-1)
    startup_clock=(age-2+p[:,22]).clamp_min(0) if p.shape[1]>=23 else clock
    transfer=torch.where(half==0,smooth(startup_clock/p[:,15]),transfer)
    left=(.5-.5*transfer).clamp(0,1)
    lift=smooth((fraction-p[:,14])/p[:,19])*smooth((p[:,18]-fraction)/p[:,19])
    clearance=.035*torch.stack((torch.where(sign>0,lift,0.),torch.where(sign<0,lift,0.)),dim=1)
    return torch.stack((left,1-left),dim=1),clearance


def project_observation(observation, names, selected):
    """Map named q, qdot and previous-action blocks; retain the common tail."""
    import torch
    n=len(names); ids=[names.index(name) for name in selected]
    columns=list(range(9))+[9+block*n+i for block in range(3) for i in ids]
    columns+=list(range(9+3*n,observation.shape[-1]))
    return observation[:,columns]


def scatter_action(action, names, selected):
    result=action.new_zeros((len(action),len(names)))
    result[:,[names.index(name) for name in selected]]=action
    return result


def base_action(gait, parameters, observation, names, selected):
    projected=project_observation(observation,names,selected)
    raw=gait.gait_action(projected[:,:19+3*len(selected)],parameters,selected)
    result=scatter_action(raw,names,selected)
    if getattr(gait,'graft_parent',None) is not None:
        baseline=base_action(gait.graft_parent,gait.graft_parent_parameters,observation,names,gait.graft_parent_names)
        result=baseline+(result-baseline)*gait.graft_blend
    return result


def features(observation, period):
    import torch
    n=(observation.shape[-1]-24)//3; b=9+3*n
    if observation.shape[-1]!=24+3*n or n not in (19,33):raise ValueError('Unknown body observation layout')
    age=(1-observation[:,b+2])*30
    clock=(age-2).clamp_min(0)
    phase=2*torch.pi*clock/period
    values=torch.cat((observation[:,:b],observation[:,b+3:b+6],phase.sin()[:,None],
                      phase.cos()[:,None],observation[:,b+8:b+10],observation[:,b+10:],
                      (age/2).clamp(0,1)[:,None],(clock/period).clamp(0,1)[:,None]),dim=1)
    scale=values.new_tensor([.3]*3+[2.]*3+[1.]*3+[.5]*n+[.4]*n+[5.]*n+
                            [.2]*3+[1.]*4+[1.,1.,.03,.03,.03]+[1.,1.])
    return (values/scale).clamp(-10,10)


def residual_bounds(names):
    return [.25 if 'knee' in n else .12 if 'hip_pitch' in n or 'ankle_pitch' in n else
            .06 if 'hip_roll' in n else .1 if 'ankle_roll' in n else (.08 if len(names)==33 else .02) if 'waist' in n or 'shoulder' in n else .02 if 'hip_yaw' in n else (.04 if len(names)==33 else 0.) if 'toe' in n else .08 for n in names]


def action_scales(names):
    return [.8 if 'knee' in n else .5 if 'hip_pitch' in n else .55 if 'ankle_pitch' in n else
            .4 if 'hip_roll' in n else .3 if 'ankle_roll' in n else
            .12 if any(k in n for k in ('hip_yaw','toe','shoulder')) else .08 for n in names]


def compose(base, latent, observation, names, ramp_s):
    age=(1-observation[:,11+3*len(names)])*30
    ramp=((age-1)/ramp_s).clamp(0,1)
    ramp=ramp.square()*(3-2*ramp)
    bounds=latent.new_tensor(residual_bounds(names))
    correction=latent.tanh()*bounds
    # A single knee channel explores coordinated shortening of the whole leg.
    # Independent hip/ankle sampling almost never found the needed combination.
    for side in ('left','right'):
        knee=names.index(side+'_knee_joint')
        for joint in ('hip_pitch','ankle_pitch'):
            index=names.index(side+'_'+joint+'_joint')
            correction[:,index]=(.25*correction[:,index]-.5*correction[:,knee]).clamp(-bounds[index],bounds[index])
    correction=correction*ramp[:,None]/.08
    import torch
    return torch.where((observation[:,12+3*len(names)]!=0)[:,None],base+correction,base)


def conditioned_features(observation,period,level=None):
    import torch
    value=features(observation,period)
    if level is None:return value
    level=torch.as_tensor(level,device=value.device,dtype=value.dtype).expand(len(value))
    return torch.cat((value,level[:,None]),dim=1)


def load_transferred_weights(model,weights,expand_input=False):
    """Add one observable curriculum coordinate without altering old weights."""
    import torch
    target=model.state_dict();copied={};expanded=0
    if target.keys()!=weights.keys():raise ValueError('Weight donor architecture differs')
    for key,value in weights.items():
        if value.shape!=target[key].shape:
            if not (expand_input and value.ndim==2 and target[key].shape==(value.shape[0],value.shape[1]+1)):
                raise ValueError('Unsupported donor weight shape: '+key)
            value=torch.cat((value,value.new_zeros((len(value),1))),dim=1);expanded+=1
        copied[key]=value
    if expanded!=int(expand_input):raise ValueError('Expected exactly one input-layer expansion')
    model.load_state_dict(copied,strict=True)


def make_actor(device='cpu', num_actions=19, conditioned=False):
    import torch
    from tensordict import TensorDict
    from rsl_rl.models import MLPModel
    return MLPModel(TensorDict({'actor':torch.zeros(1,23+3*num_actions+int(conditioned),device=device)},batch_size=[1]),
        {'actor':['actor']},'actor',num_actions,hidden_dims=[128,128],activation='elu',obs_normalization=False,
        distribution_cfg={'class_name':'GaussianDistribution','init_std':.3,'std_type':'scalar'}).to(device)


def held_actor_mean(actor, observation, period):
    """One residual decision per hold interval; the base still updates at 50 Hz."""
    from tensordict import TensorDict
    import torch
    hold=getattr(actor,'residual_hold_steps',1)
    bucket=torch.round((1-observation[:,observation.shape[-1]-13])*1500).long()//hold
    previous=getattr(actor,'_residual_bucket',None)
    update=torch.ones_like(bucket,dtype=torch.bool) if previous is None else bucket!=previous
    if update.any():
        latent=actor(TensorDict({'actor':conditioned_features(observation,period,1. if getattr(actor,'gait_conditioned',False) else None)},batch_size=[len(observation)]))
        actor._residual_latent=latent if previous is None else torch.where(update[:,None],latent,actor._residual_latent)
        actor._residual_bucket=bucket.clone()
    return actor._residual_latent


def supported_swing_landing(contact, landed, history):
    """Reject hops even when both feet have regained contact at landing."""
    import torch
    rewarded=landed&history&contact.flip(1)
    return rewarded,torch.where(contact,True,history&contact.flip(1))


def recent_episode_metrics(terminals, window=256):
    """Completed training episodes, never an independent validation claim."""
    recent=terminals[-window:]
    if not recent:return {'recent_episode_count':0}
    return {'recent_episode_count':len(recent),
        'recent_episode_survival_fraction':sum(not e['fallen'] for e in recent)/len(recent),
        'recent_episode_both_feet_fraction':sum(min(e['swings'])>0 for e in recent)/len(recent),
        'recent_episode_mean_duration_s':sum(e['age_s'] for e in recent)/len(recent),
        'recent_episode_mean_forward_m':sum(e['forward_m'] for e in recent)/len(recent)}


def evaluate_mean(actor, prior, gait, parameters, observation, names, base_limits, base_names=None, standing_names=None):
    from tensordict import TensorDict
    base_names=base_names or names; standing_names=standing_names or names
    base=base_action(gait,parameters,observation,names,base_names).clamp(-base_limits,base_limits)
    latent=held_actor_mean(actor,observation,getattr(actor,'residual_clock_period',parameters[0]))
    action=compose(base,latent,observation,names,getattr(actor,'residual_ramp_s',parameters[15]))
    standing=observation[:,12+3*len(names)]==0
    if standing.any():
        value=project_observation(observation[standing],names,standing_names)[:,:12+3*len(standing_names)]
        action[standing]=scatter_action(prior(TensorDict({'actor':value},batch_size=[len(value)])),names,standing_names)
    return action


def train(args):
    import mujoco
    import torch
    import warp as wp
    from tensordict import TensorDict
    from mjlab.rl import MjlabOnPolicyRunner,RslRlModelCfg,RslRlOnPolicyRunnerCfg,RslRlPpoAlgorithmCfg
    from .landau_rsl_control import configure_model
    from .landau_forward_control import load_gait_source
    parent=Path(args.checkpoint).resolve();old=json.loads((parent.parent/'training.json').read_text())
    if getattr(args,'nominal_pose_profile',None) is None:
        args.nominal_pose_profile=old.get('nominal_pose_profile',old.get('arguments',{}).get('nominal_pose_profile','geometric'))
    backend,batches,variant=configure_model(old['model_profile'],'closed_fists_v1',whole_body=getattr(args,'whole_body',False),nominal_pose_profile=getattr(args,'nominal_pose_profile','geometric'))
    parent.relative_to((backend.OUTPUT/'training').resolve())
    if (old['policy_family']!='anatomical_periodic_feedback_cem' or old['checkpoints'][parent.name]!=backend.digest(parent)
        or old['support_profile']!='animation_approximate_v1'):raise ValueError('Require audited anatomical gait parent')
    gait,gait_path=load_gait_source(parent,old)
    blob=torch.load(parent,map_location='cuda:0',weights_only=False);parameters=blob['parameters'].detach()
    if (parameters.numel()>=21)!=(old.get('walking_position_observation')=='native_world_xy_m_v1'):
        raise ValueError('Gait position observation contract differs from parameter schema')
    if parameters.numel()>=21 and getattr(args,'graft_checkpoint',None):
        raise ValueError('Cross-track gait cannot be replaced by a legacy graft with a different observation contract')
    parent_gait=gait;parent_parameters=parameters.clone()
    graft=None; base_names=old['action_joints']; source_hashes=old['source_sha256']
    if getattr(args,'graft_checkpoint',None):
        if args.graft_blend==1:
            raise ValueError('Aligned anatomical rewards require the anatomical timing parent; full donor timing needs its own reward contract')
        donor=Path(args.graft_checkpoint).resolve(); donor_meta=json.loads((donor.parent/'training.json').read_text())
        donor.relative_to((backend.OUTPUT/'training').resolve())
        if backend.digest(donor)!=donor_meta['checkpoints'][donor.name]:raise ValueError('Gait donor changed')
        gait,gait_path=load_gait_source(donor,donor_meta)
        donor_blob=torch.load(donor,map_location='cuda:0',weights_only=False)
        parameters=donor_blob['parameters'].detach().clone()
        if len(parameters)!=16 or len(donor_meta['action_joints'])!=17:raise ValueError('Unsupported gait donor')
        original=parameters.tolist();parameters[2]*=-1
        base_names=donor_meta['action_joints'];source_hashes=donor_meta['source_sha256']
        graft=dict(checkpoint=str(donor),sha256=backend.digest(donor),source_sha256=backend.digest(gait_path),
            original_parameters=original,parameters=parameters.tolist(),
            transformation='Reverse fore_aft_hip_rad for native -Y; retain cadence, flexion and feedback as experimental seeds',
            omitted='Turn and cruise residual actors are inactive in initial straight walking; not grafted',
            inherited_milestone_passes=0,blend=args.graft_blend)
        gait.graft_parent=parent_gait;gait.graft_parent_parameters=parent_parameters
        gait.graft_parent_names=old['action_joints'];gait.graft_blend=args.graft_blend

    clock_period=float(parent_parameters[0] if graft and args.graft_blend<1 else parameters[0])
    ramp_s=float(parent_parameters[15] if graft and args.graft_blend<1 else parameters[15])
    curriculum=None;start_parameters=None
    if getattr(args,'curriculum_start_checkpoint',None):
        import inspect
        if graft or parameters.numel()!=23:raise ValueError('Progressive gait curriculum requires the explicit23-parameter anatomical controller')
        start=Path(args.curriculum_start_checkpoint).resolve();start.relative_to((backend.OUTPUT/'training').resolve())
        sm=json.loads((start.parent/'training.json').read_text())
        if (sm.get('policy_family')!='anatomical_periodic_feedback_cem' or sm['checkpoints'][start.name]!=backend.digest(start) or
            any(sm[k]!=old[k] for k in ('model_xml_sha256','action_joints','nominal_q','nominal_ctrl','support_profile','standing_checkpoint_sha256'))):
            raise ValueError('Curriculum start physical model or standing branch differs')
        start_gait,start_source=load_gait_source(start,sm)
        current=inspect.getsource(gait.gait_action)
        old_form=current.replace('    startup_clock=(age-2.+p[:,22]).clamp_min(0) if p.shape[1]>=23 else clock\n','').replace('smooth(startup_clock/','smooth(clock/')
        if inspect.getsource(start_gait.gait_action) not in (current,old_form):raise ValueError('Curriculum controller semantics differ')
        start_parameters=torch.load(start,map_location='cuda:0',weights_only=False)['parameters'].detach()
        if start_parameters.numel()==22:start_parameters=torch.cat((start_parameters,torch.zeros(1,device='cuda:0')))
        if start_parameters.shape!=parameters.shape:raise ValueError('Curriculum parameter schema differs')
        if not .002<=args.curriculum_start_speed<=args.target_speed:raise ValueError('Invalid curriculum starting speed')
        curriculum=dict(checkpoint=str(start),sha256=backend.digest(start),training_sha256=backend.digest(start.parent/'training.json'),
            source_sha256=backend.digest(start_source),start_parameters=start_parameters.tolist(),target_parameters=parameters.tolist(),
            start_speed_mps=args.curriculum_start_speed,target_speed_mps=args.target_speed,
            rule='Per-world levels start0; at episode reset only, +0.05 after full horizon with >=2 supported swings/foot and positive progress, otherwise -0.025; clamp[0,1]. Interpolate parameters and speed. Evaluation always uses fixed target gait.',
            inherited_milestone_passes=0)
    longest_period=max(clock_period,float(start_parameters[0])) if curriculum else clock_period
    donor_conditioning=json.loads((Path(args.residual_checkpoint).resolve().parent/'training.json').read_text()).get('gait_conditioning') if args.residual_checkpoint else None
    if donor_conditioning not in (None,'curriculum_level_v1'):raise ValueError('Unknown donor conditioning')
    conditioned=bool(curriculum) or donor_conditioning is not None
    if conditioned and not args.whole_body:
        raise ValueError('Gait conditioning requires all33 non-finger actions')
    if args.seconds is None:
        decision_dt=.02*args.residual_hold_steps
        args.seconds=math.ceil((2+2*longest_period)/decision_dt-1e-9)*decision_dt
    # The anatomical parent owns the support schedule even in a graft trial.
    reward_parameters=parent_parameters
    gamma=math.exp(-.02/20.)
    gae_lambda=math.exp(-.02/2.)
    if args.iterations>2 and not args.verify_hold_resets and args.seconds<longest_period+2:
        raise ValueError('Training episodes must include startup and a complete alternating gait cycle')
    directory=(backend.OUTPUT/'training'/args.name).resolve();directory.relative_to((backend.OUTPUT/'training').resolve())
    directory.mkdir(exist_ok=False)
    torch.set_num_threads(4);torch.manual_seed(43);began=time.perf_counter()
    batch=batches.WarpBatch(args.num_envs,43,0.,'stand','load',.004,.2,.0225)
    if hashlib.sha256(batch.xml.encode()).hexdigest()!=old['model_xml_sha256']:raise ValueError('Residual physical model changed')
    batch.commands[:]=-.2
    with wp.ScopedCapture() as capture:
        for _ in range(10):
            batches.mjwarp.step(batch.wm,batch.wd);batches.mjwarp.forward(batch.wm,batch.wd)
            wp.launch(batches.record_occupancy,dim=args.num_envs,
                inputs=[batch.wd.ncollision,batch.wd.nacon,batch.wd.nefc,batch.maximum_occupancy])
    batch.graph=capture.graph
    names=batch.spec['action_joints'];limits=torch.tensor(action_scales(names),device='cuda')/.08
    base_limits=torch.tensor(action_scales(names),device='cuda')/.08
    if not graft:base_limits[:len(old['action_joints'])]=torch.tensor(old['action_scale'],device='cuda')/.08
    weight=float(batch.model.body_mass.sum())*9.81
    air=torch.zeros((args.num_envs,2),device='cuda');peak=torch.zeros_like(air);touched=torch.zeros_like(air,dtype=torch.bool)
    completed=torch.zeros_like(air,dtype=torch.long);total_swings=torch.zeros(2,device='cuda',dtype=torch.long)
    previous_clearance=torch.zeros_like(air);previous_contact=torch.ones_like(touched)
    airborne_vertical_velocity=torch.zeros_like(air);total_landing_quality=torch.zeros(2,device='cuda')
    qualified_completed=torch.zeros_like(completed)
    supported=torch.ones_like(air,dtype=torch.bool);total_rewarded=torch.zeros_like(total_swings)
    previous_landing=torch.full_like(air,-1.);swing_events=[]
    previous_latent=torch.zeros((args.num_envs,len(names)),device='cuda');stats={};terminals=[];pending={}
    horizon=round(args.seconds/.02)
    levels=torch.zeros(args.num_envs,device='cuda')
    episode_parameters=curriculum_parameters(start_parameters,parameters,levels) if curriculum else None

    class Env:
        num_envs=args.num_envs
        num_actions=len(names)
        max_episode_length=horizon//args.residual_hold_steps
        device='cuda:0'
        common_step_counter=0
        cfg={'task':'anatomical_contact_residual','training_only_resets':True}
        @property
        def unwrapped(self):return self
        @property
        def episode_length_buf(self):return batch.episode_steps//args.residual_hold_steps
        def raw(self):
            age=batch.episode_steps.float()*.02
            extra=torch.zeros((self.num_envs,10),device=self.device)
            if parameters.numel()>=21:extra[:,:2]=batch.pos[:,batch.pelvis,:2]-batch.reference_xy
            extra[:,2]=1-age/30;extra[:,3]=-.2
            phase=2*torch.pi*(age-2).clamp_min(0)/(episode_parameters[:,0] if curriculum else clock_period)
            extra[:,6]=phase.sin();extra[:,7]=phase.cos()
            extra[:,8]=batch.rot[:,batch.base,1,0];extra[:,9]=batch.rot[:,batch.base,0,0]-1
            _,_,clearance,force=batch.feet(with_forces=True)
            sensors=torch.cat((force/weight,clearance,(batch.pos[:,batch.pelvis,2]-batch.reference_height)[:,None]),dim=1)
            return torch.cat((batch.observations(),extra,sensors),dim=1)
        def get_observations(self):
            value=conditioned_features(self.raw(),episode_parameters[:,0] if curriculum else clock_period,
                levels if curriculum else 1. if conditioned else None)
            return TensorDict({'actor':value,'critic':value},batch_size=[self.num_envs])
        def control_step(self,latent,finished):
            observation=self.raw()
            with torch.no_grad():base=base_action(gait,episode_parameters if curriculum else parameters,observation,names,base_names).clamp(-base_limits,base_limits)
            training_ramp=episode_parameters[:,15] if curriculum else ramp_s
            action=compose(base,latent,observation,names,training_ramp).clamp(-limits,limits)
            batch.ctrl[:]=batch.targets;batch.ctrl[:,batch.aids]+=.08*action;batch.wrench.zero_()
            torch.cuda.synchronize();wp.capture_launch(batch.graph);wp.synchronize();batch.episode_steps+=1
            occupancy=batch.maximum_occupancy.numpy()
            if max(occupancy[:2])>=batch.wd.naconmax or occupancy[2]>=batch.wd.njmax:raise RuntimeError('Contact capacity exceeded')
            tilt=batch.rot[:,batch.base,2,2].clamp(-1,1).acos()
            height=batch.pos[:,batch.pelvis,2]-batch.reference_height
            finite=torch.isfinite(batch.q).all(1)&torch.isfinite(batch.v).all(1)
            fallen=(tilt>torch.pi/6)|(height<-.08)|(~finite)
            timeout=batch.episode_steps>=horizon;done=fallen|timeout
            contact,slip,clearance,force=batch.feet(with_forces=True)
            landed=contact&touched&(air>=.06)&(peak>=.015)&(~fallen[:,None])&(~finished[:,None])
            rewarded,next_support=supported_swing_landing(contact,landed,supported)
            quality=touchdown_quality(airborne_vertical_velocity)
            weighted=rewarded*quality if getattr(args,'landing_quality',False) else rewarded.float()
            total_landing_quality.add_(weighted.sum(0))
            total_rewarded.add_(rewarded.sum(0))
            qualified_completed.add_(rewarded.long())
            age=batch.episode_steps.float()*.02
            for i,side in landed.nonzero().tolist():
                swing_events.append(dict(env_id=i,side=side,age_s=float(age[i]),airtime_s=float(air[i,side]),
                    peak_clearance_m=float(peak[i,side]),supported=bool(rewarded[i,side]),
                    preimpact_vertical_velocity_mps=float(airborne_vertical_velocity[i,side]),landing_bonus_multiplier=float(quality[i,side]) if getattr(args,'landing_quality',False) else 1.,
                    landing_interval_s=float(age[i]-previous_landing[i,side]) if previous_landing[i,side]>=0 else None))
                previous_landing[i,side]=age[i]
            airborne_vertical_velocity.copy_(torch.where(contact,0.,torch.where(~previous_contact,
                (clearance-previous_clearance)/.02,0.)))
            previous_clearance.copy_(clearance);previous_contact.copy_(contact)
            completed.add_(landed.long());total_swings.add_(landed.sum(0))
            # Check the opposite foot throughout the air segment, including
            # landing, before resetting support history for a new segment.
            supported.copy_(next_support)
            air.copy_(torch.where(contact,0.,air+.02));peak.copy_(torch.where(contact,0.,torch.maximum(peak,clearance)));touched.logical_or_(contact)
            age=batch.episode_steps.float()*.02
            ramp=((age-1)/training_ramp).clamp(0,1);ramp=ramp.square()*(3-2*ramp)
            target_load,target_clear=reference_targets(age,episode_parameters if curriculum else reward_parameters)
            load_score=(-((force/weight-target_load).square().sum(1))/.25).exp()
            # An exponential gave almost zero signal at the baseline's 2 mm
            # clearance during peak swing. Keep a usable improvement signal.
            clear_score=(1-((clearance-target_clear).square().sum(1))/(.022**2)).clamp_min(-2)
            linear,angular=batch.local_velocity()
            training_speed=args.curriculum_start_speed+levels*(args.target_speed-args.curriculum_start_speed) if curriculum else args.target_speed
            track=velocity_tracking_score(linear,training_speed,ramp)
            speed_penalty=(batch.v[:,6:].abs()-4).clamp_min(0).square().mean(1)
            heading=torch.atan2(batch.rot[:,batch.base,1,0],batch.rot[:,batch.base,0,0])
            reward=(.5*track+1.5*load_score+3*clear_score-3*tilt.square()-2*slip-2*speed_penalty
                    -.25*(height/.015).square()-.2*heading.square()-.02*latent.tanh().square().mean(1)
                    -.02*(latent.tanh()-previous_latent).square().mean(1))*.02
            reward+=args.swing_bonus*weighted.sum(1)
            reward=torch.where(fallen,torch.full_like(reward,-1.),reward)
            stats.update(forward_mps=float(-linear[:,1].mean()),height_error_m=float(height.mean()),
                tilt_rad=float(tilt.mean()),slip_mps=float(slip.mean()),clearance_p95_m=torch.quantile(clearance,.95,dim=0).tolist(),
                completed_swings=total_swings.tolist(),fall_fraction=float(fallen.float().mean()),
                supported_rewarded_swings=total_rewarded.tolist(),
                supported_landing_bonus_sum=total_landing_quality.tolist(),
                load_score=float(load_score.mean()),clear_score=float(clear_score.mean()))
            if curriculum:stats.update(curriculum_mean_level=float(levels.mean()),curriculum_max_level=float(levels.max()),
                curriculum_target_fraction=float((levels>=1.-1e-6).float().mean()),
                curriculum_period_mean_s=float(episode_parameters[:,0].mean()),
                curriculum_speed_target_mean_mps=float(training_speed.mean()))
            batch.previous=action;previous_latent.copy_(latent.tanh())
            travel=-(batch.pos[:,batch.pelvis,1]-batch.reference_xy[1])
            for i in (done&~finished).nonzero().flatten().tolist():
                record=dict(env_id=i,age_steps=int(batch.episode_steps[i]),age_s=float(age[i]),forward_m=float(travel[i]),fallen=bool(fallen[i]),swings=completed[i].tolist())
                if curriculum:record.update(curriculum_level=float(levels[i]),period_s=float(episode_parameters[i,0]),
                    stride_rad=float(episode_parameters[i,2]),supported_swings=qualified_completed[i].tolist())
                terminals.append(record);pending[i]=record
            return reward,done,timeout&~fallen
        def step(self,latent):
            # PPO samples/logs this latent once. Reset only after its hold ends,
            # so no old action or reward leaks into the next training episode.
            finished=torch.zeros(self.num_envs,device=self.device,dtype=torch.bool)
            timeouts=torch.zeros_like(finished)
            total=torch.zeros(self.num_envs,device=self.device)
            for index in range(args.residual_hold_steps):
                reward,done,timeout=self.control_step(latent,finished)
                timeouts|=timeout&~finished
                total+=torch.where(finished,0.,reward)*(gamma**index)
                finished|=done
            if finished.any():
                for i in finished.nonzero().flatten().tolist():
                    record=pending.pop(i);record['reset_age_steps']=int(batch.episode_steps[i])
                    if curriculum:
                        levels[i]=next_curriculum_level(float(levels[i]),record['fallen'],record['supported_swings'],record['forward_m'])
                        record['next_curriculum_level']=float(levels[i])
                if curriculum:episode_parameters[finished]=curriculum_parameters(start_parameters,parameters,levels[finished])
                batch.total_resets+=int(finished.sum());batch.reset(finished)
                air[finished]=0;peak[finished]=0;touched[finished]=False;completed[finished]=0;previous_latent[finished]=0
                supported[finished]=True;previous_landing[finished]=-1
                qualified_completed[finished]=0
                previous_clearance[finished]=0;previous_contact[finished]=True;airborne_vertical_velocity[finished]=0
            self.common_step_counter+=1
            # This is a continuing locomotion task; a training time limit is
            # truncation, not a physical failure. RSL-RL bootstraps these values.
            return self.get_observations(),total,finished.float(),{'log':{},'time_outs':timeouts}

    env=Env()
    if args.verify_hold_resets:
        for _ in range(2*env.max_episode_length):
            env.step(torch.zeros((args.num_envs,len(names)),device='cuda'))
        # Physics repetitions can fall before a timeout. Verify each world's
        # complete clock accounting, retaining every such fall in the report.
        consumed=[sum(e['reset_age_steps'] for e in terminals if e['env_id']==i) for i in range(args.num_envs)]
        expected_remaining=[2*horizon-v for v in consumed]
        valid=(batch.total_resets==len(terminals) and batch.episode_steps.tolist()==expected_remaining and
            all(e['reset_age_steps']==((e['age_steps']+args.residual_hold_steps-1)//args.residual_hold_steps)*args.residual_hold_steps and
                (e['fallen'] or e['age_steps']==horizon) for e in terminals))
        report={'diagnostic_only':True,'reset_accounting_passed':valid,'residual_hold_steps':args.residual_hold_steps,
            'gait_curriculum':curriculum,'curriculum_levels':levels.tolist() if curriculum else None,
            'actual_resets':batch.total_resets,'fall_episodes':sum(e['fallen'] for e in terminals),
            'expected_remaining_steps':expected_remaining,
            'episodes':terminals,'remaining_episode_steps':batch.episode_steps.tolist()}
        backend.write_json(directory/'hold_reset_diagnostic.json',report)
        if not valid:raise RuntimeError('Held residual timeout/reset integration failed')
        return
    cfg=RslRlOnPolicyRunnerCfg(actor=RslRlModelCfg(hidden_dims=(128,128),activation='elu',obs_normalization=False,
            distribution_cfg={'class_name':'GaussianDistribution','init_std':args.action_noise,'std_type':'scalar'}),
        critic=RslRlModelCfg(hidden_dims=(128,128),activation='elu',obs_normalization=False),
        algorithm=RslRlPpoAlgorithmCfg(value_loss_coef=1.,use_clipped_value_loss=True,clip_param=.2,entropy_coef=.002,
            num_learning_epochs=5,num_mini_batches=4,learning_rate=1e-4,schedule='adaptive',gamma=gamma**args.residual_hold_steps,lam=gae_lambda**args.residual_hold_steps,desired_kl=.01,max_grad_norm=1.),
        experiment_name='anatomical_residual',save_interval=25,num_steps_per_env=args.rollout_steps,max_iterations=args.iterations,logger='tensorboard',upload_model=False)
    runner=MjlabOnPolicyRunner(env,copy.deepcopy(asdict(cfg)),str(directory),device='cuda:0')
    last=[m for m in runner.alg.actor.mlp.modules() if isinstance(m,torch.nn.Linear)][-1]
    with torch.no_grad():last.weight.zero_();last.bias.zero_()
    with torch.no_grad():
        for side in ('left','right'):
            runner.alg.actor.distribution.std_param[names.index(side+'_knee_joint')]=args.lift_noise
    residual_parent=None
    if args.residual_checkpoint:
        import importlib.util
        import inspect
        donor=Path(args.residual_checkpoint).resolve();donor.relative_to((backend.OUTPUT/'training').resolve())
        prior_meta=json.loads((donor.parent/'training.json').read_text());source=donor.parent/'residual_source.py'
        if (prior_meta.get('timing_contract')!=TIMING_CONTRACT or prior_meta['checkpoints'][donor.name]!=backend.digest(donor) or
            prior_meta['residual_source_sha256']!=backend.digest(source) or prior_meta['model_xml_sha256']!=old['model_xml_sha256'] or
            prior_meta.get('residual_hold_steps',1)!=args.residual_hold_steps or
            prior_meta['action_joints']!=names or prior_meta['nominal_q']!=batch.nominal_q.tolist() or prior_meta['nominal_ctrl']!=batch.nominal_ctrl.tolist()):
            raise ValueError('Residual weight donor identity differs')
        ms=importlib.util.spec_from_file_location('residual_weight_donor',source)
        module=importlib.util.module_from_spec(ms);ms.loader.exec_module(module)
        for name in ('features','compose','residual_bounds','action_scales','make_actor'):
            current=inspect.getsource(globals()[name]);prior_source=inspect.getsource(getattr(module,name))
            legacy=current.replace(', conditioned=False','').replace('+int(conditioned)','') if name=='make_actor' else current
            if prior_source not in (current,legacy):
                raise ValueError('Residual donor observation/action semantics changed: '+name)
        donor_blob=torch.load(donor,map_location='cuda:0',weights_only=False)
        expanded=conditioned and not donor_conditioning
        load_transferred_weights(runner.alg.actor,donor_blob['actor_state_dict'],expanded)
        load_transferred_weights(runner.alg.critic,donor_blob['critic_state_dict'],expanded)
        residual_parent={'checkpoint':str(donor),'sha256':backend.digest(donor),'training_sha256':backend.digest(donor.parent/'training.json'),
            'source_sha256':backend.digest(source),'input_migration':'append_zero_curriculum_column_v1' if expanded else None,
            'scope':'Actor and critic weights; optional single zero input column for observed curriculum level; fresh optimizer and gait parent; no inherited certificates'}
    original_update=runner.alg.update
    def update():
        result=original_update()
        row=dict(iteration=env.common_step_counter//args.rollout_steps,wall_s=time.perf_counter()-began,
                 **stats,**recent_episode_metrics(terminals))
        with (directory/'iterations.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
        if row['iteration']%25==0:print(json.dumps(row),flush=True)
        return result
    runner.alg.update=update
    metadata={**{k:old[k] for k in ('backend','mujoco_version','model_profile','model_xml_sha256','urdf_sha256','mesh_tree_sha256',
        'action_joints','nominal_q','nominal_ctrl','hand_pose','support_profile','reference_xy','action_mapping','stand_action_scale')},
        'created_at':datetime.now(timezone.utc).isoformat(),'status':'running','arguments':vars(args),'policy_family':FAMILY,'observation_dim':24+3*len(names),'action_joints':names,'whole_body':getattr(args,'whole_body',False),
        'nominal_pose_profile':getattr(args,'nominal_pose_profile','geometric'),'nominal_q':batch.nominal_q.tolist(),'nominal_ctrl':batch.nominal_ctrl.tolist(),
        'gait_action_joints':base_names,'standing_action_joints':old['action_joints'],'graft':graft,'gait_parameters':parameters.tolist(),
        'residual_weight_parent':residual_parent,
        'gait_curriculum':curriculum,
        'gait_conditioning':'curriculum_level_v1' if conditioned else None,
        'walking_position_observation':old.get('walking_position_observation','zero_drift_v1'),
        'residual_coordinates':'coupled_leg_lift_v1','residual_hold_steps':args.residual_hold_steps,
        'residual_decision_dt_s':.02*args.residual_hold_steps,'lift_initial_std':None if residual_parent else args.lift_noise,
        'initial_action_std':runner.alg.actor.distribution.std_param.detach().cpu().tolist(),
        'supported_swing_bonus':args.swing_bonus,'swing_bonus_contract':'15 mm minimum foot mesh clearance, 60 ms air, opposite foot supported throughout at 20 ms training samples',
        'touchdown_quality_contract':{'enabled':getattr(args,'landing_quality',False),'downward_allowance_mps':.05,'scale_mps':.05,
            'velocity':'Last two airborne clearance samples at20ms, before contact; reset per episode',
            'formula':'1/(1+(max(downward_velocity-0.05,0)/0.05)^2); scales existing supported-swing bonus only'},
        'residual_feature_dim':23+3*len(names)+int(conditioned),'training_target_forward_mps':args.target_speed,'reference_height':float(batch.reference_height),
        'forward_tracking_width_mps':max(.05,.5*args.target_speed),'speed_reward_contract':'exp(-forward_error_squared/width_squared-lateral_velocity_squared/0.0025); width=max(0.05,0.5*target); independent measured-speed validation required',
        'timing_contract':TIMING_CONTRACT,'reward_schedule_parameters':reward_parameters.tolist(),
        'discount_horizon_s':20.,'gae_horizon_s':2.,
        'action_scale':action_scales(names),'base_action_scale':(base_limits*.08).tolist(),
        'residual_bounds_rad':residual_bounds(names),'period_s':clock_period,'residual_clock_period':clock_period,'residual_ramp_s':ramp_s,
        'parent_checkpoint':str(parent),'parent_checkpoint_sha256':backend.digest(parent),
        'parent_training_sha256':backend.digest(parent.parent/'training.json'),
        'standing_checkpoint':old['standing_checkpoint'],'standing_checkpoint_sha256':old['standing_checkpoint_sha256'],
        'semantic_forward_native':[0,-1,0],'assistance':0.,'canonical_milestone_pass':False,
        'policy_composition':'Frozen periodic gait plus named bounded contact-aware body residual outputs; exact frozen standing MLP at zero command',
        'observation_contract':'9 + 3*N named proprioceptive features, drift/time/command/phase/heading, then forces/clearances/height',
        'parent_source_sha256':source_hashes,
        'source_sha256':{str(Path(module.__file__)):backend.digest(module.__file__) for module in (backend,batches)},
        'trainer_source_sha256':backend.digest(__file__),
        'residual_source_sha256':backend.digest(__file__),'runner_config':asdict(cfg),
        'training_reset_semantics':'Independent training episodes; time limits bootstrap through RSL-RL time_outs, falls terminate; no validation claim'}
    (directory/'control_source.py').write_text(gait_path.read_text());(directory/'residual_source.py').write_text(Path(__file__).read_text())
    (directory/'model.xml').write_text(batch.xml);backend.write_json(directory/'metadata.json',metadata)
    runner.save(str(directory/'model_initial.pt'));runner.learn(num_learning_iterations=args.iterations,init_at_random_ep_len=False)
    metadata.update(status='completed',wall_s=time.perf_counter()-began,total_resets=batch.total_resets,
        checkpoints={p.name:backend.digest(p) for p in directory.glob('model_*.pt')})
    backend.write_json(directory/'training.json',metadata);backend.write_json(directory/'episodes.json',terminals)
    backend.write_json(directory/'swing_events.json',swing_events)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--name',required=True);p.add_argument('--checkpoint',required=True)
    p.add_argument('--num-envs',type=int,default=256);p.add_argument('--iterations',type=int,default=300)
    p.add_argument('--seconds',type=float,default=None,help='Default: startup plus two complete gait cycles, rounded to the decision clock')
    p.add_argument('--residual-hold-steps',type=int,choices=(1,5),default=5)
    p.add_argument('--lift-noise',type=float,default=.8)
    p.add_argument('--action-noise',type=float,default=.05)
    p.add_argument('--swing-bonus',type=float,default=1.)
    p.add_argument('--landing-quality',action='store_true',help='Reduce the existing supported-step bonus for hard downward touchdown')
    p.add_argument('--target-speed',type=float,default=.01)
    p.add_argument('--rollout-steps',type=int,choices=(24,64,96),default=64)
    p.add_argument('--nominal-pose-profile',choices=('geometric','upright_waist_v1'),default=None,help='Inherit parent posture unless explicitly testing a new nominal pose')
    p.add_argument('--whole-body',action='store_true')
    p.add_argument('--graft-blend',type=float,default=.1)
    p.add_argument('--graft-checkpoint',help='Hash-bound historical gait seed; reverse stride for anatomical -Y')
    p.add_argument('--residual-checkpoint',help='Warm-start compatible body actor/critic weights while retaining a fresh optimizer')
    p.add_argument('--curriculum-start-checkpoint',help='Hash-bound easier gait; progress period/stride/balance parameters only at training episode resets')
    p.add_argument('--curriculum-start-speed',type=float,default=.006)
    p.add_argument('--verify-hold-resets',action='store_true',help='Run two zero-residual episodes to verify hold/timeout boundaries; no training claim')
    args=p.parse_args()
    if not 32<=args.num_envs<=512 or not 1<=args.iterations<=1000 or (args.seconds is not None and not 6<=args.seconds<=60) or not .03<=args.lift_noise<=1.:raise ValueError('Training budget outside bounds')
    if not .03<=args.action_noise<=.3:raise ValueError('Action exploration outside bounds')
    if not .002<=args.target_speed<=1.2:raise ValueError('Target speed outside user-confirmed walking curriculum bounds')
    if not .05<=args.swing_bonus<=2.:raise ValueError('Swing bonus outside diagnostic bounds')
    if args.seconds is not None and abs(args.seconds/(.02*args.residual_hold_steps)-round(args.seconds/(.02*args.residual_hold_steps)))>1e-8:raise ValueError('Episode must align to residual decision clock')
    if not 0<=args.graft_blend<=1:raise ValueError('Graft blend outside [0,1]')
    train(args)

if __name__=='__main__':main()
