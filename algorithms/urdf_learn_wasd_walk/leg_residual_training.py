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
from pathlib import Path
import time

FAMILY = 'anatomical_feedback_residual_ppo'
FEATURE_DIM = 78


def features(observation, period):
    import torch
    if observation.shape[-1] != 81:raise ValueError('Residual requires 76 proprio/command + 5 contact/height features')
    phase=2*torch.pi*((1-observation[:,68])*30-1).clamp_min(0)/period
    # Exclude unbounded episode age; match the frozen gait's actual phase.
    values=torch.cat((observation[:,:66],observation[:,69:72],phase.sin()[:,None],
                      phase.cos()[:,None],observation[:,74:76],observation[:,76:]),dim=1)
    scale=values.new_tensor([.3]*3+[2.]*3+[1.]*3+[.5]*19+[.4]*19+[5.]*19+
                            [.2]*3+[1.]*4+[1.,1.,.03,.03,.03])
    return (values/scale).clamp(-10,10)


def residual_bounds(names):
    return [.25 if 'knee' in n else .12 if 'hip_pitch' in n or 'ankle_pitch' in n else
            .06 if 'hip_roll' in n else .1 if 'ankle_roll' in n else .02 if 'waist' in n or 'hip_yaw' in n or 'shoulder' in n else 0. for n in names]


def action_scales(names):
    return [.8 if 'knee' in n else .5 if 'hip_pitch' in n else .55 if 'ankle_pitch' in n else
            .4 if 'hip_roll' in n else .3 if 'ankle_roll' in n else
            .12 if any(k in n for k in ('hip_yaw','toe','shoulder')) else .08 for n in names]


def compose(base, latent, observation, names, ramp_s):
    age=(1-observation[:,68])*30
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
    return torch.where((observation[:,69]!=0)[:,None],base+correction,base)


def make_actor(device='cpu'):
    import torch
    from tensordict import TensorDict
    from rsl_rl.models import MLPModel
    return MLPModel(TensorDict({'actor':torch.zeros(1,FEATURE_DIM,device=device)},batch_size=[1]),
        {'actor':['actor']},'actor',19,hidden_dims=[128,128],activation='elu',obs_normalization=False,
        distribution_cfg={'class_name':'GaussianDistribution','init_std':.3,'std_type':'scalar'}).to(device)


def held_actor_mean(actor, observation, period):
    """One residual decision per hold interval; the base still updates at 50 Hz."""
    from tensordict import TensorDict
    import torch
    hold=getattr(actor,'residual_hold_steps',1)
    bucket=torch.round((1-observation[:,68])*1500).long()//hold
    previous=getattr(actor,'_residual_bucket',None)
    update=torch.ones_like(bucket,dtype=torch.bool) if previous is None else bucket!=previous
    if update.any():
        latent=actor(TensorDict({'actor':features(observation,period)},batch_size=[len(observation)]))
        actor._residual_latent=latent if previous is None else torch.where(update[:,None],latent,actor._residual_latent)
        actor._residual_bucket=bucket.clone()
    return actor._residual_latent


def supported_swing_landing(contact, landed, history):
    """Reject hops even when both feet have regained contact at landing."""
    import torch
    rewarded=landed&history&contact.flip(1)
    return rewarded,torch.where(contact,True,history&contact.flip(1))


def evaluate_mean(actor, prior, gait, parameters, observation, names, base_limits):
    from tensordict import TensorDict
    base=gait.gait_action(observation[:,:76],parameters,names).clamp(-base_limits,base_limits)
    latent=held_actor_mean(actor,observation,parameters[0])
    action=compose(base,latent,observation,names,parameters[15])
    standing=observation[:,69]==0
    if standing.any():
        value=observation[standing,:69]
        action[standing]=prior(TensorDict({'actor':value},batch_size=[len(value)]))
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
    backend,batches,variant=configure_model(old['model_profile'],'closed_fists_v1')
    parent.relative_to((backend.OUTPUT/'training').resolve())
    if (old['policy_family']!='anatomical_periodic_feedback_cem' or old['checkpoints'][parent.name]!=backend.digest(parent)
        or old['support_profile']!='animation_approximate_v1'):raise ValueError('Require audited anatomical gait parent')
    gait,gait_path=load_gait_source(parent,old)
    blob=torch.load(parent,map_location='cuda:0',weights_only=False);parameters=blob['parameters'].detach()
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
    base_limits=torch.tensor(old['action_scale'],device='cuda')/.08
    weight=float(batch.model.body_mass.sum())*9.81
    air=torch.zeros((args.num_envs,2),device='cuda');peak=torch.zeros_like(air);touched=torch.zeros_like(air,dtype=torch.bool)
    completed=torch.zeros_like(air,dtype=torch.long);total_swings=torch.zeros(2,device='cuda',dtype=torch.long)
    supported=torch.ones_like(air,dtype=torch.bool);total_rewarded=torch.zeros_like(total_swings)
    previous_landing=torch.full_like(air,-1.);swing_events=[]
    previous_latent=torch.zeros((args.num_envs,19),device='cuda');stats={};terminals=[];pending={}
    horizon=round(args.seconds/.02)

    class Env:
        num_envs=args.num_envs
        num_actions=19
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
            extra[:,2]=1-age/30;extra[:,3]=-.2
            phase=2*torch.pi*(age-1).clamp_min(0)/parameters[0]
            extra[:,6]=phase.sin();extra[:,7]=phase.cos()
            extra[:,8]=batch.rot[:,batch.base,1,0];extra[:,9]=batch.rot[:,batch.base,0,0]-1
            _,_,clearance,force=batch.feet(with_forces=True)
            sensors=torch.cat((force/weight,clearance,(batch.pos[:,batch.pelvis,2]-batch.reference_height)[:,None]),dim=1)
            return torch.cat((batch.observations(),extra,sensors),dim=1)
        def get_observations(self):
            value=features(self.raw(),parameters[0]);return TensorDict({'actor':value,'critic':value},batch_size=[self.num_envs])
        def control_step(self,latent,finished):
            observation=self.raw()
            with torch.no_grad():base=gait.gait_action(observation[:,:76],parameters,names).clamp(-base_limits,base_limits)
            action=compose(base,latent,observation,names,parameters[15]).clamp(-limits,limits)
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
            total_rewarded.add_(rewarded.sum(0))
            age=batch.episode_steps.float()*.02
            for i,side in landed.nonzero().tolist():
                swing_events.append(dict(env_id=i,side=side,age_s=float(age[i]),airtime_s=float(air[i,side]),
                    peak_clearance_m=float(peak[i,side]),supported=bool(rewarded[i,side]),
                    landing_interval_s=float(age[i]-previous_landing[i,side]) if previous_landing[i,side]>=0 else None))
                previous_landing[i,side]=age[i]
            completed.add_(landed.long());total_swings.add_(landed.sum(0))
            # Check the opposite foot throughout the air segment, including
            # landing, before resetting support history for a new segment.
            supported.copy_(next_support)
            air.copy_(torch.where(contact,0.,air+.02));peak.copy_(torch.where(contact,0.,torch.maximum(peak,clearance)));touched.logical_or_(contact)
            age=batch.episode_steps.float()*.02;phase=2*torch.pi*(age-1).clamp_min(0)/parameters[0]
            ramp=((age-1)/parameters[15]).clamp(0,1);ramp=ramp.square()*(3-2*ramp)
            sine=phase.sin()
            # Transfer load slightly before flexing the swing knee.
            unload=(phase+.5).sin()
            left=.5-.5*(2*unload).clamp(-1,1)*ramp
            target_load=torch.stack((left,1-left),dim=1)
            target_clear=.022*torch.stack((sine.clamp_min(0).square(),(-sine).clamp_min(0).square()),dim=1)*ramp[:,None]
            load_score=(-((force/weight-target_load).square().sum(1))/.25).exp()
            # An exponential gave almost zero signal at the baseline's 2 mm
            # clearance during peak swing. Keep a usable improvement signal.
            clear_score=(1-((clearance-target_clear).square().sum(1))/(.022**2)).clamp_min(-2)
            linear,angular=batch.local_velocity()
            track=(-(linear[:,1]+.08*ramp).square()/.04-linear[:,0].square()/.04).exp()
            speed_penalty=(batch.v[:,6:].abs()-4).clamp_min(0).square().mean(1)
            heading=torch.atan2(batch.rot[:,batch.base,1,0],batch.rot[:,batch.base,0,0])
            reward=(.5*track+1.5*load_score+3*clear_score-3*tilt.square()-2*slip-2*speed_penalty
                    -.25*(height/.015).square()-.2*heading.square()-.02*latent.tanh().square().mean(1)
                    -.02*(latent.tanh()-previous_latent).square().mean(1))*.02
            reward+=args.swing_bonus*rewarded.sum(1)
            reward=torch.where(fallen,torch.full_like(reward,-1.),reward)
            stats.update(forward_mps=float(-linear[:,1].mean()),height_error_m=float(height.mean()),
                tilt_rad=float(tilt.mean()),slip_mps=float(slip.mean()),clearance_p95_m=torch.quantile(clearance,.95,dim=0).tolist(),
                completed_swings=total_swings.tolist(),fall_fraction=float(fallen.float().mean()),
                supported_rewarded_swings=total_rewarded.tolist(),
                load_score=float(load_score.mean()),clear_score=float(clear_score.mean()))
            batch.previous=action;previous_latent.copy_(latent.tanh())
            travel=-(batch.pos[:,batch.pelvis,1]-batch.reference_xy[1])
            for i in (done&~finished).nonzero().flatten().tolist():
                record=dict(env_id=i,age_steps=int(batch.episode_steps[i]),age_s=float(age[i]),forward_m=float(travel[i]),fallen=bool(fallen[i]),swings=completed[i].tolist())
                terminals.append(record);pending[i]=record
            return reward,done
        def step(self,latent):
            # PPO samples/logs this latent once. Reset only after its hold ends,
            # so no old action or reward leaks into the next training episode.
            finished=torch.zeros(self.num_envs,device=self.device,dtype=torch.bool)
            total=torch.zeros(self.num_envs,device=self.device)
            for index in range(args.residual_hold_steps):
                reward,done=self.control_step(latent,finished)
                total+=torch.where(finished,0.,reward)*(.99**index)
                finished|=done
            if finished.any():
                for i in finished.nonzero().flatten().tolist():
                    pending.pop(i)['reset_age_steps']=int(batch.episode_steps[i])
                batch.total_resets+=int(finished.sum());batch.reset(finished)
                air[finished]=0;peak[finished]=0;touched[finished]=False;completed[finished]=0;previous_latent[finished]=0
                supported[finished]=True;previous_landing[finished]=-1
            self.common_step_counter+=1
            return self.get_observations(),total,finished.float(),{'log':{}}

    env=Env()
    if args.verify_hold_resets:
        for _ in range(2*env.max_episode_length):
            env.step(torch.zeros((args.num_envs,19),device='cuda'))
        # Physics repetitions can fall before a timeout. Verify each world's
        # complete clock accounting, retaining every such fall in the report.
        consumed=[sum(e['reset_age_steps'] for e in terminals if e['env_id']==i) for i in range(args.num_envs)]
        expected_remaining=[2*horizon-v for v in consumed]
        valid=(batch.total_resets==len(terminals) and batch.episode_steps.tolist()==expected_remaining and
            all(e['reset_age_steps']==((e['age_steps']+args.residual_hold_steps-1)//args.residual_hold_steps)*args.residual_hold_steps and
                (e['fallen'] or e['age_steps']==horizon) for e in terminals))
        report={'diagnostic_only':True,'reset_accounting_passed':valid,'residual_hold_steps':args.residual_hold_steps,
            'actual_resets':batch.total_resets,'fall_episodes':sum(e['fallen'] for e in terminals),
            'expected_remaining_steps':expected_remaining,
            'episodes':terminals,'remaining_episode_steps':batch.episode_steps.tolist()}
        backend.write_json(directory/'hold_reset_diagnostic.json',report)
        if not valid:raise RuntimeError('Held residual timeout/reset integration failed')
        return
    cfg=RslRlOnPolicyRunnerCfg(actor=RslRlModelCfg(hidden_dims=(128,128),activation='elu',obs_normalization=False,
            distribution_cfg={'class_name':'GaussianDistribution','init_std':.3,'std_type':'scalar'}),
        critic=RslRlModelCfg(hidden_dims=(128,128),activation='elu',obs_normalization=False),
        algorithm=RslRlPpoAlgorithmCfg(value_loss_coef=1.,use_clipped_value_loss=True,clip_param=.2,entropy_coef=.002,
            num_learning_epochs=5,num_mini_batches=4,learning_rate=1e-4,schedule='adaptive',gamma=.99**args.residual_hold_steps,lam=.95**args.residual_hold_steps,desired_kl=.01,max_grad_norm=1.),
        experiment_name='anatomical_residual',save_interval=50,num_steps_per_env=24,max_iterations=args.iterations,logger='tensorboard',upload_model=False)
    runner=MjlabOnPolicyRunner(env,copy.deepcopy(asdict(cfg)),str(directory),device='cuda:0')
    last=[m for m in runner.alg.actor.mlp.modules() if isinstance(m,torch.nn.Linear)][-1]
    with torch.no_grad():last.weight.zero_();last.bias.zero_()
    with torch.no_grad():
        for side in ('left','right'):
            runner.alg.actor.distribution.std_param[names.index(side+'_knee_joint')]=args.lift_noise
    original_update=runner.alg.update
    def update():
        result=original_update()
        row=dict(iteration=env.common_step_counter//24,wall_s=time.perf_counter()-began,**stats)
        with (directory/'iterations.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
        if row['iteration']%25==0:print(json.dumps(row),flush=True)
        return result
    runner.alg.update=update
    metadata={**{k:old[k] for k in ('backend','mujoco_version','model_profile','model_xml_sha256','urdf_sha256','mesh_tree_sha256',
        'action_joints','nominal_q','nominal_ctrl','hand_pose','support_profile','reference_xy','action_mapping','stand_action_scale')},
        'created_at':datetime.now(timezone.utc).isoformat(),'arguments':vars(args),'policy_family':FAMILY,'observation_dim':81,
        'residual_coordinates':'coupled_leg_lift_v1','residual_hold_steps':args.residual_hold_steps,
        'residual_decision_dt_s':.02*args.residual_hold_steps,'lift_initial_std':args.lift_noise,
        'supported_swing_bonus':args.swing_bonus,'swing_bonus_contract':'15 mm minimum foot mesh clearance, 60 ms air, opposite foot supported throughout at 20 ms training samples',
        'residual_feature_dim':FEATURE_DIM,'training_target_forward_mps':.08,'reference_height':float(batch.reference_height),
        'action_scale':action_scales(names),'base_action_scale':old['action_scale'],
        'residual_bounds_rad':residual_bounds(names),'period_s':float(parameters[0]),
        'parent_checkpoint':str(parent),'parent_checkpoint_sha256':backend.digest(parent),
        'parent_training_sha256':backend.digest(parent.parent/'training.json'),
        'standing_checkpoint':old['standing_checkpoint'],'standing_checkpoint_sha256':old['standing_checkpoint_sha256'],
        'semantic_forward_native':[0,-1,0],'assistance':0.,'canonical_milestone_pass':False,
        'policy_composition':'Frozen periodic gait plus 19 bounded contact-aware residual outputs; exact frozen standing MLP at zero command',
        'observation_contract':'76 base features + left/right force divided by BW + left/right mesh clearance + pelvis height error',
        'parent_source_sha256':old['source_sha256'],
        'source_sha256':{str(Path(module.__file__)):backend.digest(module.__file__) for module in (backend,batches)},
        'trainer_source_sha256':backend.digest(__file__),
        'residual_source_sha256':backend.digest(__file__),'runner_config':asdict(cfg),'training_reset_semantics':'Independent finite episodes; no validation claim'}
    (directory/'control_source.py').write_text(gait_path.read_text());(directory/'residual_source.py').write_text(Path(__file__).read_text())
    (directory/'model.xml').write_text(batch.xml);backend.write_json(directory/'metadata.json',metadata)
    runner.save(str(directory/'model_initial.pt'));runner.learn(num_learning_iterations=args.iterations,init_at_random_ep_len=False)
    metadata.update(wall_s=time.perf_counter()-began,total_resets=batch.total_resets,
        checkpoints={p.name:backend.digest(p) for p in directory.glob('model_*.pt')})
    backend.write_json(directory/'training.json',metadata);backend.write_json(directory/'episodes.json',terminals)
    backend.write_json(directory/'swing_events.json',swing_events)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--name',required=True);p.add_argument('--checkpoint',required=True)
    p.add_argument('--num-envs',type=int,default=256);p.add_argument('--iterations',type=int,default=300)
    p.add_argument('--seconds',type=float,default=6.)
    p.add_argument('--residual-hold-steps',type=int,choices=(1,5),default=5)
    p.add_argument('--lift-noise',type=float,default=.8)
    p.add_argument('--swing-bonus',type=float,default=1.)
    p.add_argument('--verify-hold-resets',action='store_true',help='Run two zero-residual episodes to verify hold/timeout boundaries; no training claim')
    args=p.parse_args()
    if not 32<=args.num_envs<=512 or not 1<=args.iterations<=1000 or not 6<=args.seconds<=30 or not .3<=args.lift_noise<=1.:raise ValueError('Training budget outside bounds')
    if not .05<=args.swing_bonus<=2.:raise ValueError('Swing bonus outside diagnostic bounds')
    if abs(args.seconds/(.02*args.residual_hold_steps)-round(args.seconds/(.02*args.residual_hold_steps)))>1e-8:raise ValueError('Episode must align to residual decision clock')
    train(args)

if __name__=='__main__':main()
