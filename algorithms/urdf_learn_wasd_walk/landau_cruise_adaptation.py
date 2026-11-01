"""Provenance for cruise adaptation with an unchanged active-turn network."""
import json
from pathlib import Path

VARIANT='adapted_cruise_frozen_active_v1'


def donor(checkpoint,digest):
    import torch
    from algorithms.urdf_learn_wasd_walk.landau_active_residual import check_contract as check_original
    checkpoint=Path(checkpoint);meta=json.loads((checkpoint.parent/'training.json').read_text())
    if (meta.get('policy_family')!='periodic_feedback_residual_ppo' or
        meta.get('residual_variant')!='cruise_plus_active_left_v1' or
        str(meta.get('status','')).startswith('invalidated_') or meta['checkpoints'][checkpoint.name]!=digest(checkpoint)):
        raise ValueError('Invalid frozen active donor')
    blob=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if not check_original(meta,blob,digest):raise ValueError('Donor lacks active correction')
    for source in ('residual_source','active_residual_source','turn_source'):
        if digest(checkpoint.parent/(source+'.py'))!=meta[source+'_sha256']:raise ValueError('Donor source changed')
    return meta,blob


def check_contract(meta,blob,digest):
    import torch
    if meta.get('residual_variant')!=VARIANT or blob.get('residual_variant')!=VARIANT:
        raise ValueError('Adapted cruise variant mismatch')
    if meta.get('policy_family')!='periodic_feedback_residual_ppo' or any(k.startswith('frozen_cruise_') for k in meta):
        raise ValueError('Adapted cruise cannot claim frozen cruise weights')
    path=Path(meta['frozen_active_checkpoint'])
    if digest(path)!=meta['frozen_active_checkpoint_sha256'] or digest(path.parent/'training.json')!=meta['frozen_active_training_sha256']:
        raise ValueError('Frozen active input changed')
    old,original=donor(path,digest)
    for key in ('parent_checkpoint_sha256','parent_training_sha256','residual_source_sha256','turn_source_sha256',
                'active_residual_source_sha256','active_residual_bound_rad','active_residual_feature_dim',
                'active_residual_action_joints','active_residual_gate','residual_bound_rad','residual_feature_dim',
                'residual_action_joints','residual_gate'):
        if meta.get(key)!=old[key]:raise ValueError('Adapted cruise contract changed: '+key)
    for key in ('parameters','turn_parameters'):
        if not torch.equal(blob[key],original[key]):raise ValueError('Frozen active controller changed')
    for key in ('standing_actor_state_dict','active_residual_actor_state_dict'):
        left,right=blob[key],original[key]
        if left.keys()!=right.keys() or any(not torch.equal(left[k],right[k]) for k in left):raise ValueError('Frozen active weights changed')
    init=Path(meta['cruise_initialization_checkpoint'])
    if (init.resolve()!=Path(old['frozen_cruise_checkpoint']).resolve() or
        digest(init)!=meta['cruise_initialization_checkpoint_sha256'] or
        digest(init.parent/'training.json')!=meta['cruise_initialization_training_sha256'] or
        meta['cruise_initialization_checkpoint_sha256']!=old['frozen_cruise_checkpoint_sha256'] or
        meta['cruise_initialization_training_sha256']!=old['frozen_cruise_training_sha256']):
        raise ValueError('Cruise initialization changed')
    if 'residual_actor_state_dict' not in blob:raise ValueError('Missing adapted cruise actor')
    return True


def prepare(args,parent,directory,turn,digest):
    import torch
    from algorithms.urdf_learn_wasd_walk.landau_residual_screen import build_late_diagnostic_bank
    path=Path(args.adapt_active_checkpoint).resolve();meta,blob=donor(path,digest)
    if meta['parent_checkpoint_sha256']!=digest(parent) or meta['parent_training_sha256']!=digest(parent.parent/'training.json'):
        raise ValueError('Adaptation CEM parent mismatch')
    init=Path(meta['frozen_cruise_checkpoint'])
    initial=torch.load(init,map_location='cpu',weights_only=False)
    bank=build_late_diagnostic_bank(args.curriculum,path,meta,turn,directory/'warm_states.npz',digest)
    bank['scope']='Healthy late prefix used for training resets; even rows train, odd rows reserved and correlated. No nominal or acceptance claim.'
    bank['count']=30
    return path,meta,blob,init,initial,bank


def metadata(training,path,old,init,digest):
    training.update(residual_variant=VARIANT,residual_source_sha256=old['residual_source_sha256'],
        frozen_active_checkpoint=str(path),frozen_active_checkpoint_sha256=digest(path),
        frozen_active_training_sha256=digest(path.parent/'training.json'),
        cruise_initialization_checkpoint=str(init),cruise_initialization_checkpoint_sha256=digest(init),
        cruise_initialization_training_sha256=digest(init.parent/'training.json'),
        policy_composition='Adapted cruise PPO with unchanged active-turn PPO, CEM20, gait and standing controller',
        training_timeout='16 s recorded-state episodes with timeout bootstrap; no padded heading teacher in reward')
    for key,value in old.items():
        if key.startswith('active_residual_'):training[key]=value
