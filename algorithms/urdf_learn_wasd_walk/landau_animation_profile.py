"""User-selected animation acceptance profile, with unchanged controller tensors."""
import argparse
import copy
import json
from datetime import datetime, timezone
from pathlib import Path

PROFILE='animation_approximate_v1'


def check_contract(meta,blob,folder,digest):
    import torch
    from algorithms.urdf_learn_wasd_walk.landau_zero_active import same_tree
    record=meta['animation_profile_transfer'];donor=Path(record['checkpoint'])
    old=json.loads((donor.parent/'training.json').read_text())
    if ('animation_profile_transfer' in old or meta.get('support_profile')!=PROFILE or
        record['profile']!=PROFILE or digest(donor)!=record['checkpoint_sha256'] or
        digest(donor.parent/'training.json')!=record['training_sha256'] or
        old['checkpoints'][donor.name]!=digest(donor)):
        raise ValueError('Animation profile donor changed')
    original=torch.load(donor,map_location='cpu',weights_only=False)
    if set(meta)!=set(old)|{'support_profile','animation_profile_transfer'} or set(blob)!=set(original)|{'animation_profile_transfer'}:
        raise ValueError('Animation profile schema changed')
    if blob['animation_profile_transfer']!=record:raise ValueError('Animation profile marker differs')
    for key,value in original.items():
        if not same_tree(value,blob[key]):raise ValueError('Animation profile changed controller: '+key)
    mutable={'arguments','created_at','status','checkpoints','canonical_milestone_pass','policy_composition','support_profile'}
    for key,value in old.items():
        if key not in mutable and meta[key]!=value:raise ValueError('Animation profile metadata changed: '+key)
    if {k:v for k,v in meta['arguments'].items() if k!='name'}!={k:v for k,v in old['arguments'].items() if k!='name'}:
        raise ValueError('Animation profile training arguments changed')
    for path in list(donor.parent.glob('*.py'))+[donor.parent/'model.xml']:
        if digest(Path(folder)/path.name)!=digest(path):raise ValueError('Animation profile deployment changed')
    if digest(Path(folder)/'animation_profile_source.py')!=record['source_sha256']:
        raise ValueError('Animation profile adapter changed')
    if meta['canonical_milestone_pass'] is not False:raise ValueError('Profile cannot inherit certification')
    return donor,old,original


def compose(previous,name):
    import torch
    from algorithms.urdf_learn_wasd_walk import mujoco_backend as backend
    previous=Path(previous).resolve();previous.relative_to((backend.OUTPUT/'training').resolve())
    old=json.loads((previous.parent/'training.json').read_text())
    record=dict(profile=PROFILE,checkpoint=str(previous),checkpoint_sha256=backend.digest(previous),
                training_sha256=backend.digest(previous.parent/'training.json'),source_sha256=backend.digest(__file__),
                authorization='User explicitly selected animation/game acceptance on 2026-11-04; peak landing force is diagnostic, other motion/physics guards remain.')
    blob=copy.deepcopy(torch.load(previous,map_location='cpu',weights_only=False));blob['animation_profile_transfer']=record
    meta=copy.deepcopy(old);meta.update(support_profile=PROFILE,animation_profile_transfer=record,
        arguments={**old['arguments'],'name':name},created_at=datetime.now(timezone.utc).isoformat(),
        status='animation_profile_requires_cumulative_validation',canonical_milestone_pass=False,
        policy_composition='Identical fist controller under the explicitly selected animation acceptance profile.')
    folder=(backend.OUTPUT/'training'/name).resolve();folder.relative_to((backend.OUTPUT/'training').resolve());folder.mkdir(exist_ok=False)
    for path in list(previous.parent.glob('*.py'))+[previous.parent/'model.xml']:(folder/path.name).write_bytes(path.read_bytes())
    (folder/'animation_profile_source.py').write_bytes(Path(__file__).read_bytes())
    check_contract(meta,blob,folder,backend.digest)
    torch.save(blob,folder/'model_0.pt');meta['checkpoints']={'model_0.pt':backend.digest(folder/'model_0.pt')}
    backend.write_json(folder/'training.json',meta);return folder/'model_0.pt'


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',required=True);p.add_argument('--name',required=True)
    args=p.parse_args();print(compose(args.checkpoint,args.name))


if __name__=='__main__':main()
