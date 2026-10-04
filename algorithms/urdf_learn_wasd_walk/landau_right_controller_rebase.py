"""Compose four right gains while preserving the calibrated learned controller."""
import argparse
import copy
import json
import math
from datetime import datetime, timezone
from pathlib import Path

VARIANT = 'right_controller_calibrated_dual_residual_v1'
INDICES = [7,8,12,13]
FIXED = [i for i in range(20) if i not in INDICES]
FILES = ('control_source.py','turn_source.py','teleop_contract.py','model.xml',
         'residual_source.py','active_residual_source.py','right_rebase_source.py','yaw_hold_rebase_source.py')


def check_contract(meta, blob, digest):
    import torch
    from algorithms.urdf_learn_wasd_walk import landau_yaw_hold_rebase as yaw
    from algorithms.urdf_learn_wasd_walk.landau_right_rebase import same_weights, deployment_ast
    if meta.get('residual_variant') != VARIANT or blob.get('residual_variant') != VARIANT:
        raise ValueError('Right controller calibration variant mismatch')
    donor = Path(meta['right_controller_donor_checkpoint'])
    old = json.loads((donor.parent/'training.json').read_text())
    if (digest(donor) != meta['right_controller_donor_checkpoint_sha256'] or
        digest(donor.parent/'training.json') != meta['right_controller_donor_training_sha256'] or
        old['checkpoints'][donor.name] != digest(donor) or str(old.get('status','')).startswith('invalidated_')):
        raise ValueError('Right controller donor changed')
    original = torch.load(donor,map_location='cpu',weights_only=False)
    yaw.check_contract(old,original,digest)
    mutable = {'residual_variant','arguments','checkpoints','created_at','status','canonical_milestone_pass','policy_composition'}
    for key,value in old.items():
        if key not in mutable and meta.get(key) != value:raise ValueError('Right donor metadata changed: '+key)
    if {k:v for k,v in meta['arguments'].items() if k!='name'} != {k:v for k,v in old['arguments'].items() if k!='name'}:
        raise ValueError('Right donor arguments changed')
    p = blob['turn_parameters']
    if tuple(p.shape)!=(20,) or not torch.isfinite(p).all() or not torch.equal(p[FIXED],original['turn_parameters'][FIXED]):
        raise ValueError('Only right controller parameters7/8/12/13 may change')
    for i,lo,hi in ((7,0.,3.),(8,-.12,.12),(12,-.06,.03),(13,-.25,.25)):
        if not lo<=float(p[i])<=hi:raise ValueError('Right controller gain outside bounds')
    if not torch.equal(blob['parameters'],original['parameters']):raise ValueError('Right donor gait changed')
    for key in ('standing_actor_state_dict','residual_actor_state_dict','active_residual_actor_state_dict'):
        if not same_weights(blob[key],original[key]):raise ValueError('Right donor actor changed')
    evidence = meta['right_controller_calibration']
    search = Path(evidence['search_checkpoint']);sm = json.loads((search.parent/'training.json').read_text())
    if (digest(search)!=evidence['search_checkpoint_sha256'] or
        digest(search.parent/'training.json')!=evidence['search_training_sha256'] or
        sm['checkpoints'][search.name]!=digest(search) or sm['status']!='completed' or
        sm['policy_family']!='periodic_feedback_cem'):
        raise ValueError('Right controller search changed')
    args=sm['arguments'];contract=sm['right_controller_search']
    if (not all(args[k] for k in ('right_controller_search','right_only','teleop','direction_train','common_starts')) or
        args['closed_loop_direction']!='right' or args['seconds']<100. or
        args['turn_duration']!=old['arguments']['turn_duration'] or
        contract['searched_indices']!=INDICES or evidence['changed_indices']!=INDICES or
        contract['training_force_target_bw']!=2.9 or contract['hard_force_limit_bw']!=3.):
        raise ValueError('Right search contract changed')
    for key in ('model_xml_sha256','urdf_sha256','mesh_tree_sha256','action_joints','action_scale','nominal_q','nominal_ctrl'):
        if sm[key]!=old[key]:raise ValueError('Right search model identity changed')
    if (digest(search.parent/'turn_source.py')!=sm['turn_source_sha256'] or
        deployment_ast(search.parent/'turn_source.py')!=deployment_ast(donor.parent/'turn_source.py') or
        digest(search.parent/'control_source.py')!=digest(donor.parent/'control_source.py')):
        raise ValueError('Right search deployment changed')
    for filename,key in (('direction_training_source.py','direction_training_source_sha256'),
                         ('direction_training_protocol.py','direction_training_protocol_sha256'),('teleop_contract.py','teleop_contract_sha256')):
        if digest(search.parent/filename)!=sm[key]:raise ValueError('Right search helper changed')
    seed=Path(evidence['seed_checkpoint']);seed_meta=json.loads((seed.parent/'training.json').read_text())
    if (seed!=(search.parent.parent/Path(args['seed_checkpoint']).parent.name/Path(args['seed_checkpoint']).name).resolve() or digest(seed)!=evidence['seed_checkpoint_sha256'] or
        digest(seed.parent/'training.json')!=evidence['seed_training_sha256'] or seed_meta['checkpoints'][seed.name]!=digest(seed)):
        raise ValueError('Right search seed changed')
    if (seed!=Path(old['parent_checkpoint']) or digest(seed)!=old['parent_checkpoint_sha256'] or
        digest(seed.parent/'training.json')!=old['parent_training_sha256']):
        raise ValueError('Right search seed is not the calibrated donor parent')
    seed_blob=torch.load(seed,map_location='cpu',weights_only=False)
    search_blob=torch.load(search,map_location='cpu',weights_only=False)
    for base_blob in (seed_blob,search_blob):
        if not torch.equal(base_blob['parameters'],original['parameters']) or not same_weights(base_blob['standing_actor_state_dict'],original['standing_actor_state_dict']):
            raise ValueError('Right search frozen gait or standing differs from donor')
    if tuple(seed_blob['turn_parameters'].shape)!=(20,):raise ValueError('Right search seed schema changed')
    records=evidence['records']
    if len(records)<2 or len({r['table'] for r in records})!=len(records):raise ValueError('Right search needs repeated distinct generations')
    for item in records:
        filename=item['table']
        if Path(filename).name!=filename or filename not in sm['generation_candidate_tables']:
            raise ValueError('Right search table not bound to generation')
        path=search.parent/filename
        if digest(path)!=item['sha256'] or item['sha256']!=sm['generation_candidate_tables'][filename]:
            raise ValueError('Right search candidate table changed')
        index=item['candidate'];row=json.loads(path.read_text())[index]
        values=torch.tensor(row['parameters'],dtype=p.dtype)
        if (row['candidate']!=index or not row['eligible'] or len(row['failure_bits'])!=4 or any(row['failure_bits']) or
            row['world_gates']['completed']!=[True]*4 or len(row['peak_support_bw'])!=4 or not all(math.isfinite(x) and 0<=x<=2.9 for x in row['peak_support_bw']) or
            tuple(values.shape)!=(20,) or not torch.equal(values[INDICES],p[INDICES].cpu()) or
            not torch.equal(values[FIXED],seed_blob['turn_parameters'][FIXED])):
            raise ValueError('Right search lacks repeated exact candidate qualification')
    return True


def compose(previous, search, selections, name):
    import torch
    from algorithms.urdf_learn_wasd_walk import mujoco_backend as backend
    previous=Path(previous).resolve();search=Path(search).resolve()
    for path in (previous,search):path.relative_to((backend.OUTPUT/'training').resolve())
    old=json.loads((previous.parent/'training.json').read_text());sm=json.loads((search.parent/'training.json').read_text())
    records=[]
    for selection in selections:
        generation,index=map(int,selection.split(':'))
        table=search.parent/f'candidates_{generation}.json'
        records.append({'table':table.name,'sha256':backend.digest(table),'candidate':index})
    row=json.loads((search.parent/records[0]['table']).read_text())[records[0]['candidate']]
    blob=copy.deepcopy(torch.load(previous,map_location='cpu',weights_only=False))
    blob['turn_parameters'][INDICES]=torch.tensor(row['parameters'],dtype=blob['turn_parameters'].dtype)[INDICES]
    blob['residual_variant']=VARIANT
    seed=(search.parent.parent/Path(sm['arguments']['seed_checkpoint']).parent.name/Path(sm['arguments']['seed_checkpoint']).name).resolve()
    meta=copy.deepcopy(old)
    meta.update(residual_variant=VARIANT,arguments={**old['arguments'],'name':name},created_at=datetime.now(timezone.utc).isoformat(),
        status='composed_requires_nominal_validation',canonical_milestone_pass=False,
        policy_composition='Right gains7/8/12/13 from repeated four-start search; calibrated donor actors and all other parameters unchanged',
        right_controller_donor_checkpoint=str(previous),right_controller_donor_checkpoint_sha256=backend.digest(previous),
        right_controller_donor_training_sha256=backend.digest(previous.parent/'training.json'),
        right_controller_calibration={'changed_indices':INDICES,'source_sha256':backend.digest(__file__),
            'search_checkpoint':str(search),'search_checkpoint_sha256':backend.digest(search),
            'search_training_sha256':backend.digest(search.parent/'training.json'),
            'seed_checkpoint':str(seed),'seed_checkpoint_sha256':backend.digest(seed),
            'seed_training_sha256':backend.digest(seed.parent/'training.json'),'records':records})
    check_contract(meta,blob,backend.digest)
    folder=(backend.OUTPUT/'training'/name).resolve();folder.relative_to((backend.OUTPUT/'training').resolve());folder.mkdir(exist_ok=False)
    for filename in FILES:(folder/filename).write_bytes((previous.parent/filename).read_bytes())
    (folder/'right_controller_rebase_source.py').write_bytes(Path(__file__).read_bytes())
    torch.save(blob,folder/'model_0.pt');meta['checkpoints']={'model_0.pt':backend.digest(folder/'model_0.pt')}
    backend.write_json(folder/'training.json',meta)
    return folder/'model_0.pt'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True);parser.add_argument('--search-checkpoint',required=True)
    parser.add_argument('--selection',action='append',required=True,help='generation:candidate, repeated at least twice')
    parser.add_argument('--name',required=True);args=parser.parse_args()
    print(compose(args.checkpoint,args.search_checkpoint,args.selection,args.name))


if __name__=='__main__':main()
