"""Bounded active-left correction composed with an unchanged cruise residual."""
VARIANT='cruise_plus_active_left_v1'
GATE='(1-left_cruise_blend) * left_feedback_blend * clamp(abs(forward)/0.2,0,1)'
DISPATCH='deterministic_cruise_plus_active_left_hip_roll_v1'


def envelope(memory,observation):
    import torch
    cruise=torch.as_tensor(memory.left_cruise_blend,device=observation.device,dtype=observation.dtype).expand(len(observation))
    feedback=torch.as_tensor(memory.left_feedback_blend,device=observation.device,dtype=observation.dtype).expand(len(observation))
    return (1.-cruise)*feedback*(observation[:,63].abs()/.2).clamp(0,1)


def compose(baseline,latent,memory,names,observation):
    import torch
    blend=envelope(memory,observation)
    correction=torch.where(blend[:,None]>0.,.02*latent.tanh()*blend[:,None],0.)/.08
    result=baseline.clone()
    for side,column in (('left',0),('right',1)):
        result[:,names.index(side+'_hip_roll_joint')]+=correction[:,column]
    return result


def apply(actor,baseline,observation,walking,memory,names,cruise):
    from tensordict import TensorDict
    if not bool((envelope(memory,observation)>0).any()):return baseline
    value=cruise.features(observation,walking,memory)
    return compose(baseline,actor(TensorDict({'actor':value},batch_size=[len(value)])),memory,names,observation)


def check_contract(metadata,blob,digest):
    """Deployment fail-closed validation; certification also checks independently."""
    import json
    from pathlib import Path
    import torch
    active_keys={k for k in metadata if k.startswith('active_residual_') or k.startswith('frozen_cruise_')}
    if blob.get('residual_variant')!=metadata.get('residual_variant'):raise ValueError('Active residual variant mismatch')
    enabled=metadata.get('residual_variant')==VARIANT
    if not enabled:
        if active_keys or 'active_residual_actor_state_dict' in blob or metadata.get('residual_variant') is not None:
            raise ValueError('Unexpected active residual state or metadata')
        return False
    if metadata.get('active_residual_gate')!=GATE or metadata.get('active_residual_bound_rad')!=.02 or metadata.get('active_residual_feature_dim')!=69 or metadata.get('active_residual_action_joints')!=['left_hip_roll_joint','right_hip_roll_joint']:
        raise ValueError('Active residual contract changed')
    if 'active_residual_actor_state_dict' not in blob or not metadata.get('active_residual_source_sha256'):
        raise ValueError('Missing active residual weights/source')
    frozen=Path(metadata['frozen_cruise_checkpoint'])
    if digest(frozen)!=metadata['frozen_cruise_checkpoint_sha256'] or digest(frozen.parent/'training.json')!=metadata['frozen_cruise_training_sha256']:
        raise ValueError('Frozen cruise input changed')
    old=json.loads((frozen.parent/'training.json').read_text())
    if old.get('policy_family')!='periodic_feedback_residual_ppo' or old.get('residual_variant') is not None or str(old.get('status','')).startswith('invalidated_') or old['checkpoints'][frozen.name]!=digest(frozen):
        raise ValueError('Invalid frozen cruise input')
    for key in ('parent_checkpoint_sha256','parent_training_sha256','residual_source_sha256','turn_source_sha256'):
        if old[key]!=metadata[key]:raise ValueError('Frozen cruise lineage changed: '+key)
    if digest(frozen.parent/'residual_source.py')!=metadata['residual_source_sha256']:raise ValueError('Frozen cruise source changed')
    original=torch.load(frozen,map_location='cpu',weights_only=False)
    if original.get('residual_variant') is not None or 'active_residual_actor_state_dict' in original:raise ValueError('Invalid cruise-only weights')
    for key in ('parameters','turn_parameters'):
        if not torch.equal(original[key],blob[key]):raise ValueError('Frozen cruise controller changed')
    for key in ('standing_actor_state_dict','residual_actor_state_dict'):
        a,b=original[key],blob[key]
        if a.keys()!=b.keys() or any(not torch.equal(a[k],b[k]) for k in a):raise ValueError('Frozen cruise weights changed')
    return True
