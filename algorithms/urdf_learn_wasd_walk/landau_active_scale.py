"""Experimental quarter active composition; all learned tensors stay frozen.

Attribution diagnostics motivate an experiment, never qualification. Every new
artifact requires its own complete nominal and cumulative milestone evidence.
"""
import argparse
import copy
import json
from datetime import datetime, timezone
from pathlib import Path

VARIANT = 'scaled_active_half_cruise_v1'
SCALE = .25
DISPATCH = 'deterministic_half_cruise_quarter_active_v1'
FILES = ('control_source.py','turn_source.py','teleop_contract.py','model.xml',
         'residual_source.py','active_residual_source.py','right_rebase_source.py',
         'yaw_hold_rebase_source.py','right_controller_rebase_source.py','cruise_scale_source.py')
ADDED = {'active_output_scale','active_scale_donor_checkpoint',
         'active_scale_donor_checkpoint_sha256','active_scale_donor_training_sha256',
         'active_scale_source_sha256'}


def same_tree(a, b):
    import torch
    if torch.is_tensor(a):return torch.is_tensor(b) and a.dtype==b.dtype and torch.equal(a,b)
    if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(same_tree(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):return type(a) is type(b) and len(a)==len(b) and all(same_tree(x,y) for x,y in zip(a,b))
    return type(a) is type(b) and a==b


def apply(baseline, active_complete):
    """Scale the bounded active delta after the unchanged half-cruise composition."""
    return baseline+SCALE*(active_complete-baseline)


def check_contract(meta, blob, digest):
    import torch
    from algorithms.urdf_learn_wasd_walk import landau_cruise_scale as right
    if meta.get('residual_variant')!=VARIANT or blob.get('residual_variant')!=VARIANT:
        raise ValueError('Quarter active variant mismatch')
    for record in (meta,blob):
        if type(record.get('active_output_scale')) is not float or record['active_output_scale']!=SCALE:
            raise ValueError('Quarter active requires exact scalar0.25')
    donor=Path(meta['active_scale_donor_checkpoint'])
    old=json.loads((donor.parent/'training.json').read_text())
    if (old.get('residual_variant')!=right.VARIANT or
        digest(donor)!=meta['active_scale_donor_checkpoint_sha256'] or
        digest(donor.parent/'training.json')!=meta['active_scale_donor_training_sha256'] or
        old['checkpoints'][donor.name]!=digest(donor) or str(old.get('status','')).startswith('invalidated_')):
        raise ValueError('Quarter active donor changed or is not a half-cruise donor')
    original=torch.load(donor,map_location='cpu',weights_only=False)
    right.check_contract(old,original,digest)
    right.check_sources(donor.parent,old,digest)
    if set(blob)!=set(original)|{'active_output_scale'}:
        raise ValueError('Quarter active checkpoint keys changed')
    for key,value in original.items():
        if key!='residual_variant' and not same_tree(value,blob[key]):raise ValueError('Quarter active frozen checkpoint changed: '+key)
    mutable={'residual_variant','arguments','checkpoints','created_at','status','canonical_milestone_pass','policy_composition'}
    if set(meta)!=set(old)|ADDED:raise ValueError('Quarter active metadata keys changed')
    for key,value in old.items():
        if key not in mutable and meta[key]!=value:raise ValueError('Quarter active donor metadata changed: '+key)
    if {k:v for k,v in meta['arguments'].items() if k!='name'}!={k:v for k,v in old['arguments'].items() if k!='name'}:
        raise ValueError('Quarter active donor arguments changed')
    if meta['active_scale_source_sha256']!=digest(__file__):raise ValueError('Quarter active helper changed')
    return True


def check_sources(folder, meta, digest):
    donor=Path(meta['active_scale_donor_checkpoint'])
    for filename in FILES:
        if digest(Path(folder)/filename)!=digest(donor.parent/filename):raise ValueError('Quarter active source changed: '+filename)
    if digest(Path(folder)/'active_scale_source.py')!=meta['active_scale_source_sha256']:
        raise ValueError('Quarter active composition snapshot changed')


def compose(previous, name):
    import torch
    from algorithms.urdf_learn_wasd_walk import mujoco_backend as backend
    previous=Path(previous).resolve();previous.relative_to((backend.OUTPUT/'training').resolve())
    old=json.loads((previous.parent/'training.json').read_text())
    blob=copy.deepcopy(torch.load(previous,map_location='cpu',weights_only=False))
    blob.update(residual_variant=VARIANT,active_output_scale=SCALE)
    meta=copy.deepcopy(old)
    meta.update(residual_variant=VARIANT,active_output_scale=SCALE,
        arguments={**old['arguments'],'name':name},created_at=datetime.now(timezone.utc).isoformat(),
        status='experimental_requires_nominal_validation',canonical_milestone_pass=False,
        policy_composition='Experimental quarter active output after half cruise; every donor tensor unchanged; no inherited milestone credit',
        active_scale_donor_checkpoint=str(previous),active_scale_donor_checkpoint_sha256=backend.digest(previous),
        active_scale_donor_training_sha256=backend.digest(previous.parent/'training.json'),
        active_scale_source_sha256=backend.digest(__file__))
    check_contract(meta,blob,backend.digest)
    folder=(backend.OUTPUT/'training'/name).resolve();folder.relative_to((backend.OUTPUT/'training').resolve())
    folder.mkdir(exist_ok=False)
    for filename in FILES:(folder/filename).write_bytes((previous.parent/filename).read_bytes())
    (folder/'active_scale_source.py').write_bytes(Path(__file__).read_bytes())
    check_sources(folder,meta,backend.digest)
    torch.save(blob,folder/'model_0.pt');meta['checkpoints']={'model_0.pt':backend.digest(folder/'model_0.pt')}
    backend.write_json(folder/'training.json',meta)
    return folder/'model_0.pt'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True);parser.add_argument('--name',required=True)
    args=parser.parse_args();print(compose(args.checkpoint,args.name))


if __name__=='__main__':main()
