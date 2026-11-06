"""Independent no-reset deployment of anatomical gait and whole-body residual policies."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True,type=Path);p.add_argument('--name',required=True)
    p.add_argument('--seconds',type=float,default=45.);p.add_argument('--stand',action='store_true')
    p.add_argument('--target-distance',type=float,choices=(5.,10.),default=5.)
    p.add_argument('--stop-at-gate',action='store_true',help='End at the first distance crossing on a control boundary after30s; retain the declared search horizon')
    p.add_argument('--speed-target',type=float,choices=(.1,.4,.8,1.2),help='Fixed 10 s warmup plus 30 s sustained-speed proof')
    p.add_argument('--diagnostic-curriculum-level',type=float,help='Inspect an intermediate learned gait; explicitly ineligible for certification')
    p.add_argument('--diagnostic-phase-template',type=Path,help='Motor-only M7 retarget probe; cannot certify')
    p.add_argument('--diagnostic-period',type=float,default=1.8)
    p.add_argument('--diagnostic-ramp',type=float,default=3.)
    p.add_argument('--diagnostic-feedback',type=float,default=.5)
    p.add_argument('--diagnostic-phase-start',type=float,default=0.)
    p.add_argument('--diagnostic-startup-hold',action='store_true')
    p.add_argument('--diagnostic-velocity-feedforward',action='store_true')
    args=p.parse_args()
    if args.stand and args.stop_at_gate:raise ValueError('Distance stopping is only for walking')
    if args.diagnostic_phase_template and (args.stand or args.stop_at_gate or args.speed_target is not None or args.diagnostic_curriculum_level is not None):
        raise ValueError('Phase retarget requires a standalone walking diagnostic')
    if args.diagnostic_curriculum_level is not None and (
        not 0<=args.diagnostic_curriculum_level<=1 or args.stand or args.stop_at_gate or args.speed_target is not None):
        raise ValueError('Intermediate curriculum inspection requires a bounded walking diagnostic without a speed or distance gate')
    import torch
    from tensordict import TensorDict
    from rsl_rl.models import MLPModel
    from .landau_rsl_control import configure_model
    from .landau_forward_control import load_gait_source,PERIOD
    checkpoint=args.checkpoint.resolve();meta=json.loads((checkpoint.parent/'training.json').read_text())
    residual=meta.get('policy_family')=='anatomical_feedback_residual_ppo'
    backend,_,_=configure_model(meta['model_profile'],'closed_fists_v1',whole_body=meta.get('whole_body',False),nominal_pose_profile=meta.get('nominal_pose_profile','geometric'))
    checkpoint.relative_to((backend.OUTPUT/'training').resolve())
    if (meta.get('policy_family') not in ('anatomical_periodic_feedback_cem','anatomical_feedback_residual_ppo') or meta['observation_dim']!=(24+3*len(meta['action_joints']) if residual else 76)
        or len(meta['action_joints']) not in (19,33) or meta['checkpoints'][checkpoint.name]!=backend.digest(checkpoint)
        or meta.get('support_profile')!='animation_approximate_v1'):
        raise ValueError('Anatomical checkpoint identity mismatch')
    implementation,gait_source=load_gait_source(checkpoint,meta)
    blob=torch.load(checkpoint,map_location='cpu',weights_only=False)
    residual_implementation=residual_actor=residual_source=None
    if residual:
        import importlib.util
        parent=Path(meta['parent_checkpoint']);parent_meta=json.loads((parent.parent/'training.json').read_text())
        if (backend.digest(parent)!=meta['parent_checkpoint_sha256'] or
            backend.digest(parent.parent/'training.json')!=meta['parent_training_sha256'] or
            parent_meta['model_xml_sha256']!=meta['model_xml_sha256']):raise ValueError('Frozen gait parent changed')
        residual_source=checkpoint.parent/'residual_source.py'
        if backend.digest(residual_source)!=meta['residual_source_sha256']:raise ValueError('Residual source changed')
        ms=importlib.util.spec_from_file_location('anatomical_residual_'+meta['residual_source_sha256'][:16],residual_source)
        residual_implementation=importlib.util.module_from_spec(ms);ms.loader.exec_module(residual_implementation)
        conditioned=meta.get('gait_conditioning')=='curriculum_level_v1'
        residual_actor=(residual_implementation.make_actor(num_actions=len(meta['action_joints']),**({'conditioned':True} if conditioned else {})) if meta.get('whole_body') else residual_implementation.make_actor());residual_actor.load_state_dict(blob['actor_state_dict'],strict=True);residual_actor.eval()
        residual_actor.gait_conditioned=conditioned
        residual_actor.residual_hold_steps=meta.get('residual_hold_steps',1)
        if 'residual_clock_period' in meta:residual_actor.residual_clock_period=meta['residual_clock_period']
        if 'residual_ramp_s' in meta:residual_actor.residual_ramp_s=meta['residual_ramp_s']
        blob=torch.load(parent,map_location='cpu',weights_only=False)
        if meta.get('gait_parameters') is not None:blob['parameters']=torch.tensor(meta['gait_parameters'])
        if args.diagnostic_curriculum_level is not None:
            curriculum=meta.get('gait_curriculum')
            if not conditioned or not curriculum:raise ValueError('Checkpoint has no observed gait curriculum')
            level=args.diagnostic_curriculum_level
            start=torch.tensor(curriculum['start_parameters']);target=torch.tensor(curriculum['target_parameters'])
            blob['parameters']=start+level*(target-start)
            residual_actor.residual_clock_period=float(blob['parameters'][0]);residual_actor.residual_ramp_s=float(blob['parameters'][15])
            original_forward=residual_actor.forward
            def diagnostic_forward(observations,*forward_args,**forward_kwargs):
                observations=observations.clone();observations['actor'][...,-1]=level
                return original_forward(observations,*forward_args,**forward_kwargs)
            residual_actor.forward=diagnostic_forward
        graft=meta.get('graft')
        if graft and backend.digest(Path(graft['checkpoint']))!=graft['sha256']:raise ValueError('Graft donor changed')
        if graft and 'blend' in graft:
            implementation.graft_parent,_=load_gait_source(parent,parent_meta)
            implementation.graft_parent_parameters=torch.load(parent,map_location='cpu',weights_only=False)['parameters']
            implementation.graft_parent_names=meta['standing_action_joints'];implementation.graft_blend=graft['blend']
    elif args.diagnostic_curriculum_level is not None:raise ValueError('Intermediate curriculum inspection requires a residual policy')
    phase_probe=None
    if args.diagnostic_phase_template:
        if not residual or not meta.get('whole_body') or meta.get('graft'):
            raise ValueError('Phase retarget requires an ungrafted whole-body checkpoint')
        from .phase_retarget_probe import PhaseRetarget
        phase_probe=PhaseRetarget(args.diagnostic_phase_template,meta,args.diagnostic_period,args.diagnostic_ramp,args.diagnostic_feedback,
            args.diagnostic_phase_start,args.diagnostic_startup_hold,args.diagnostic_velocity_feedforward)
        implementation.gait_action=phase_probe.gait_action
        # The old residual was trained with another gait schedule. Disable its
        # correction for this motor-template feasibility probe, retaining all33
        # bounded joint targets and the exact physical model.
        original_forward=residual_actor.forward
        def zero_residual(observations,*forward_args,**forward_kwargs):
            return original_forward(observations,*forward_args,**forward_kwargs)*0
        residual_actor.forward=zero_residual
        residual_actor.residual_clock_period=args.diagnostic_period
    prior=MLPModel(TensorDict({'actor':torch.zeros(1,69)},batch_size=[1]),{'actor':['actor']},'actor',19,
                   hidden_dims=[128,128],activation='elu',obs_normalization=True,
                   distribution_cfg={'class_name':'GaussianDistribution','init_std':.1,'std_type':'scalar'})
    prior.load_state_dict(blob['standing_actor_state_dict'],strict=True);prior.eval()
    actor=SimpleNamespace(parameters=blob['parameters'],standing_prior=prior)
    from . import mujoco_policy as interface
    original_observe=interface.observe
    def observe(model,data,nominal,joints,previous):
        rotation=data.xmat[model.body('base_link').id].reshape(3,3)
        delta=np.r_[data.xpos[model.body('root_x').id,:2]-meta['reference_xy'],0.]
        drift=(rotation.T@delta)[:2]/.03 if args.stand else np.zeros(2)
        if not args.stand and meta.get('walking_position_observation')=='native_world_xy_m_v1':drift=delta[:2]
        phase=2*np.pi*data.time/PERIOD
        extra=[0. if args.stand else -.2,0.,0.,0. if args.stand else np.sin(phase),
               0. if args.stand else np.cos(phase),rotation[1,0],rotation[0,0]-1.]
        value=np.r_[original_observe(model,data,nominal,joints,previous),drift,1-data.time/30.,extra]
        if residual:
            from .mujoco_policy import foot_state
            feet=foot_state(model,data,clearance=True,diagnostics=True)
            weight=float(model.body_mass.sum())*9.81
            value=np.r_[value,feet['left_normal_force_N']/weight,feet['right_normal_force_N']/weight,
                feet['left_clearance_m'],feet['right_clearance_m'],data.xpos[model.body('root_x').id,2]-meta['reference_height']]
        return value.astype(np.float32)
    def load(path,model,spec):
        if (spec['source']['urdf_sha256']!=meta['urdf_sha256'] or
            spec['source']['mesh_tree_sha256']!=meta['mesh_tree_sha256'] or
            spec['action_joints']!=meta['action_joints']):raise ValueError('Deployment model changed')
        def inference(observation):
            if residual:
                return residual_implementation.evaluate_mean(residual_actor,prior,implementation,blob['parameters'],
                    observation[None],meta['action_joints'],torch.tensor(meta['base_action_scale'])/.08,
                    **({'base_names':meta['gait_action_joints'],'standing_names':meta['standing_action_joints']} if meta.get('whole_body') or meta.get('graft') else {}))[0]
            return implementation.evaluate_mean(actor,observation[None],meta['action_joints'])[0]
        return SimpleNamespace(actor=inference),meta
    interface.observe,interface.load_actor=observe,load
    # Import after model configuration so its builder is the exact new profile.
    from . import landau_forward_evaluation as evaluation
    cfg=SimpleNamespace(name=args.name,backend='mujoco_warp_cuda',seconds=30. if args.stand else args.seconds,
        dt=.002,seed=42,pose='geometric',gain_scale=1.,noslip_iterations=0,contact_timeconst=.004,
        assistance=0.,checkpoint=str(checkpoint),forward=0. if args.stand else .2,render=True,
        nominal_pose_profile=meta.get('nominal_pose_profile','geometric'),target_distance_m=args.target_distance,support_profile='animation_approximate_v1',model_profile=meta['model_profile'])
    if args.speed_target is not None:
        from .walking_speed_contract import DURATION_S, PROTOCOL
        if args.stand or args.stop_at_gate:raise ValueError('Speed proof cannot stand or stop at a distance gate')
        cfg.seconds=DURATION_S;cfg.speed_target_mps=args.speed_target;cfg.speed_protocol=PROTOCOL
    if args.stop_at_gate:cfg.stop_at_gate=True
    if args.diagnostic_curriculum_level is not None:cfg.diagnostic_curriculum_level=args.diagnostic_curriculum_level
    if phase_probe:cfg.diagnostic_phase_template=phase_probe.contract
    evaluation.run(cfg)
    folder=backend.OUTPUT/args.name
    result=json.loads((folder/'dynamics.json').read_text())
    (folder/'controller_source.py').write_text(Path(__file__).read_text())
    result['controller_source_sha256']=backend.digest(folder/'controller_source.py')
    (folder/'gait_source.py').write_text(gait_source.read_text())
    result['gait_source_sha256']=backend.digest(folder/'gait_source.py')
    if residual:
        (folder/'residual_source.py').write_text(residual_source.read_text())
        result['residual_source_sha256']=backend.digest(folder/'residual_source.py')
        result['residual_hold_steps']=residual_actor.residual_hold_steps
    result['policy_family']=meta['policy_family'];result['semantic_forward_native']=[0,-1,0]
    if args.diagnostic_curriculum_level is not None:
        result['diagnostic_curriculum_level']=args.diagnostic_curriculum_level
        result['failures'].append('Intermediate curriculum inspection is diagnostic only; fixed target gait validation required')
    if phase_probe:
        import shutil
        from . import phase_retarget_probe
        shutil.copy2(phase_probe.path,folder/'phase_template.json')
        shutil.copy2(phase_retarget_probe.__file__,folder/'phase_template_source.py')
        result['diagnostic_phase_template']=phase_probe.contract
        result['failures'].append('Phase retarget motor probe is diagnostic only; no trained policy certification')
    backend.write_json(folder/'dynamics.json',result)
    # The renderer's dynamics hash must bind this completed controller evidence.
    proof=json.loads((folder/'proof_metadata.json').read_text())
    proof.update(dynamics_sha256=backend.digest(folder/'dynamics.json'),model_xml_sha256=backend.digest(folder/'model.xml'))
    backend.write_json(folder/'proof_metadata.json',proof)
    from .motion_recording import export_motion
    export_motion(folder)

if __name__=='__main__':main()
