"""Independent no-reset deployment of the anatomical 19-action gait family."""
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
    args=p.parse_args()
    import torch
    from tensordict import TensorDict
    from rsl_rl.models import MLPModel
    from .landau_rsl_control import configure_model
    from .landau_forward_control import load_gait_source,PERIOD
    checkpoint=args.checkpoint.resolve();meta=json.loads((checkpoint.parent/'training.json').read_text())
    residual=meta.get('policy_family')=='anatomical_feedback_residual_ppo'
    backend,_,_=configure_model(meta['model_profile'],'closed_fists_v1')
    checkpoint.relative_to((backend.OUTPUT/'training').resolve())
    if (meta.get('policy_family') not in ('anatomical_periodic_feedback_cem','anatomical_feedback_residual_ppo') or meta['observation_dim']!=(81 if residual else 76)
        or len(meta['action_joints'])!=19 or meta['checkpoints'][checkpoint.name]!=backend.digest(checkpoint)
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
        residual_actor=residual_implementation.make_actor();residual_actor.load_state_dict(blob['actor_state_dict'],strict=True);residual_actor.eval()
        residual_actor.residual_hold_steps=meta.get('residual_hold_steps',1)
        blob=torch.load(parent,map_location='cpu',weights_only=False)
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
                    observation[None],meta['action_joints'],torch.tensor(meta['base_action_scale'])/.08)[0]
            return implementation.evaluate_mean(actor,observation[None],meta['action_joints'])[0]
        return SimpleNamespace(actor=inference),meta
    interface.observe,interface.load_actor=observe,load
    # Import after model configuration so its builder is the exact new profile.
    from . import landau_forward_evaluation as evaluation
    cfg=SimpleNamespace(name=args.name,backend='mujoco_warp_cuda',seconds=30. if args.stand else args.seconds,
        dt=.002,seed=42,pose='geometric',gain_scale=1.,noslip_iterations=0,contact_timeconst=.004,
        assistance=0.,checkpoint=str(checkpoint),forward=0. if args.stand else .2,render=True,
        target_distance_m=args.target_distance,support_profile='animation_approximate_v1',model_profile=meta['model_profile'])
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
    backend.write_json(folder/'dynamics.json',result)
    # The renderer's dynamics hash must bind this completed controller evidence.
    proof=json.loads((folder/'proof_metadata.json').read_text())
    proof.update(dynamics_sha256=backend.digest(folder/'dynamics.json'),model_xml_sha256=backend.digest(folder/'model.xml'))
    backend.write_json(folder/'proof_metadata.json',proof)
    from .motion_recording import export_motion
    export_motion(folder)

if __name__=='__main__':main()
