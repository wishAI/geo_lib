"""Experimental half-cruise composition; all learned tensors stay frozen.

Attribution diagnostics motivate an experiment, never qualification. Every new
artifact requires its own complete nominal and cumulative milestone evidence.
"""
import argparse
import copy
import json
from datetime import datetime, timezone
from pathlib import Path

VARIANT = 'half_cruise_frozen_dual_v1'
SCALE = .5
DISPATCH = 'deterministic_half_cruise_plus_active_left_hip_roll_v1'
FILES = ('control_source.py','turn_source.py','teleop_contract.py','model.xml',
         'residual_source.py','active_residual_source.py','right_rebase_source.py',
         'yaw_hold_rebase_source.py','right_controller_rebase_source.py')
ADDED = {'cruise_output_scale','cruise_scale_donor_checkpoint',
         'cruise_scale_donor_checkpoint_sha256','cruise_scale_donor_training_sha256',
         'cruise_scale_source_sha256'}


def same_tree(a, b):
    import torch
    if torch.is_tensor(a):return torch.is_tensor(b) and a.dtype==b.dtype and torch.equal(a,b)
    if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(same_tree(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):return type(a) is type(b) and len(a)==len(b) and all(same_tree(x,y) for x,y in zip(a,b))
    return type(a) is type(b) and a==b


def apply(baseline, cruise_complete):
    """Scale the bounded cruise delta before applying the unchanged active head."""
    return baseline+SCALE*(cruise_complete-baseline)


def check_contract(meta, blob, digest):
    import torch
    from algorithms.urdf_learn_wasd_walk import landau_right_controller_rebase as right
    if meta.get('residual_variant')!=VARIANT or blob.get('residual_variant')!=VARIANT:
        raise ValueError('Half-cruise variant mismatch')
    for record in (meta,blob):
        if type(record.get('cruise_output_scale')) is not float or record['cruise_output_scale']!=SCALE:
            raise ValueError('Half-cruise requires exact scalar0.5')
    donor=Path(meta['cruise_scale_donor_checkpoint'])
    old=json.loads((donor.parent/'training.json').read_text())
    if (old.get('residual_variant')!=right.VARIANT or
        digest(donor)!=meta['cruise_scale_donor_checkpoint_sha256'] or
        digest(donor.parent/'training.json')!=meta['cruise_scale_donor_training_sha256'] or
        old['checkpoints'][donor.name]!=digest(donor) or str(old.get('status','')).startswith('invalidated_')):
        raise ValueError('Half-cruise donor changed or is not an unscaled right-controller donor')
    original=torch.load(donor,map_location='cpu',weights_only=False)
    right.check_contract(old,original,digest)
    if set(blob)!=set(original)|{'cruise_output_scale'}:
        raise ValueError('Half-cruise checkpoint keys changed')
    for key,value in original.items():
        if key!='residual_variant' and not same_tree(value,blob[key]):raise ValueError('Half-cruise frozen checkpoint changed: '+key)
    mutable={'residual_variant','arguments','checkpoints','created_at','status','canonical_milestone_pass','policy_composition'}
    if set(meta)!=set(old)|ADDED:raise ValueError('Half-cruise metadata keys changed')
    for key,value in old.items():
        if key not in mutable and meta[key]!=value:raise ValueError('Half-cruise donor metadata changed: '+key)
    if {k:v for k,v in meta['arguments'].items() if k!='name'}!={k:v for k,v in old['arguments'].items() if k!='name'}:
        raise ValueError('Half-cruise donor arguments changed')
    if meta['cruise_scale_source_sha256']!=digest(__file__):raise ValueError('Half-cruise helper changed')
    return True


def check_sources(folder, meta, digest):
    donor=Path(meta['cruise_scale_donor_checkpoint'])
    for filename in FILES:
        if digest(Path(folder)/filename)!=digest(donor.parent/filename):raise ValueError('Half-cruise source changed: '+filename)
    if digest(Path(folder)/'cruise_scale_source.py')!=meta['cruise_scale_source_sha256']:
        raise ValueError('Half-cruise composition snapshot changed')


def compose(previous, name):
    import torch
    from algorithms.urdf_learn_wasd_walk import mujoco_backend as backend
    previous=Path(previous).resolve();previous.relative_to((backend.OUTPUT/'training').resolve())
    old=json.loads((previous.parent/'training.json').read_text())
    blob=copy.deepcopy(torch.load(previous,map_location='cpu',weights_only=False))
    blob.update(residual_variant=VARIANT,cruise_output_scale=SCALE)
    meta=copy.deepcopy(old)
    meta.update(residual_variant=VARIANT,cruise_output_scale=SCALE,
        arguments={**old['arguments'],'name':name},created_at=datetime.now(timezone.utc).isoformat(),
        status='experimental_requires_nominal_validation',canonical_milestone_pass=False,
        policy_composition='Experimental half cruise output; every donor tensor and active correction unchanged; no inherited milestone credit',
        cruise_scale_donor_checkpoint=str(previous),cruise_scale_donor_checkpoint_sha256=backend.digest(previous),
        cruise_scale_donor_training_sha256=backend.digest(previous.parent/'training.json'),
        cruise_scale_source_sha256=backend.digest(__file__))
    check_contract(meta,blob,backend.digest)
    folder=(backend.OUTPUT/'training'/name).resolve();folder.relative_to((backend.OUTPUT/'training').resolve())
    folder.mkdir(exist_ok=False)
    for filename in FILES:(folder/filename).write_bytes((previous.parent/filename).read_bytes())
    (folder/'cruise_scale_source.py').write_bytes(Path(__file__).read_bytes())
    check_sources(folder,meta,backend.digest)
    torch.save(blob,folder/'model_0.pt');meta['checkpoints']={'model_0.pt':backend.digest(folder/'model_0.pt')}
    backend.write_json(folder/'training.json',meta)
    return folder/'model_0.pt'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True);parser.add_argument('--name',required=True)
    args=parser.parse_args();print(compose(args.checkpoint,args.name))


if __name__=='__main__':main()
