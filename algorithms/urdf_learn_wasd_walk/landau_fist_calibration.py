"""Audited yaw/hold calibration on an explicitly transferred fist controller."""
import argparse
import copy
import json
from datetime import datetime, timezone
from pathlib import Path

INDICES = [3, 17]
VERSIONS = {
    'fist_yaw_hold_v1': (INDICES, 'nominal_common_start_yaw_hold_calibration', 4),
    'fist_heading_hold_v2': ([2, 3, 17], 'nominal_common_start_heading_calibration', 8),
    'fist_heading_hold_v3': ([2, 3, 17], 'nominal_common_start_strong_heading_calibration', 8),
    'fist_hold_response_v4': ([3, 17], 'nominal_common_start_hold_response_calibration', 8),
}
MUTABLE = {'arguments', 'created_at', 'status', 'checkpoints',
           'canonical_milestone_pass', 'policy_composition'}


def check_contract(meta, blob, folder, digest):
    import torch
    from algorithms.urdf_learn_wasd_walk.landau_zero_active import same_tree
    from algorithms.urdf_learn_wasd_walk.landau_hand_transfer import check_contract as check_transfer
    record = meta['fist_calibration']
    if record['version'] not in VERSIONS:raise ValueError('Unknown fist calibration version')
    indices,screen_kind,starts = VERSIONS[record['version']]
    donor = Path(record['checkpoint'])
    old = json.loads((donor.parent/'training.json').read_text())
    if ('fist_calibration' in old or
            digest(donor) != record['checkpoint_sha256'] or
            digest(donor.parent/'training.json') != record['training_sha256'] or
            old['checkpoints'][donor.name] != digest(donor)):
        raise ValueError('Fist calibration donor changed')
    original = torch.load(donor, map_location='cpu', weights_only=False)
    check_transfer(old, original, donor.parent, digest)
    if set(meta) != set(old)|{'fist_calibration'} or set(blob) != set(original)|{'fist_calibration'}:
        raise ValueError('Fist calibration schema changed')
    if blob['fist_calibration'] != record:
        raise ValueError('Fist calibration checkpoint marker differs')
    for key, value in old.items():
        if key not in MUTABLE and meta[key] != value:
            raise ValueError('Fist calibration metadata changed: '+key)
    if ({k:v for k,v in meta['arguments'].items() if k!='name'} !=
            {k:v for k,v in old['arguments'].items() if k!='name'}):
        raise ValueError('Fist calibration training arguments changed')
    for key, value in original.items():
        if key != 'turn_parameters' and not same_tree(value, blob[key]):
            raise ValueError('Fist calibration frozen state changed: '+key)
    parameters = blob['turn_parameters']
    fixed = [i for i in range(20) if i not in indices]
    hold_bound=2. if record['version']=='fist_hold_response_v4' else .12
    if (tuple(parameters.shape) != (20,) or not torch.isfinite(parameters).all() or
            not torch.equal(parameters[fixed],original['turn_parameters'][fixed]) or
            not -hold_bound <= float(parameters[3]) <= hold_bound or not 0. <= float(parameters[17]) <= 3.):
        raise ValueError('Fist calibration changed undeclared gains')
    heading_min=-.120001 if record['version']=='fist_heading_hold_v3' else 0.
    if 2 in indices and not heading_min <= float(parameters[2]) <= .040001:
        raise ValueError('Moving heading gain exceeds the declared grid bounds')
    screen_path = Path(record['screen'])
    screen = json.loads(screen_path.read_text())
    if (digest(screen_path) != record['screen_sha256'] or
            screen['kind'] != screen_kind or
            screen['checkpoint_sha256'] != digest(donor) or
            screen['training_sha256'] != digest(donor.parent/'training.json') or
            screen['changed_indices'] != indices or screen['canonical_milestone_pass'] is not False):
        raise ValueError('Fist calibration screen identity changed')
    for filename,key in (('screen_source.py','source_sha256'),('audit_source.py','audit_source_sha256'),
                         ('teleop_source.py','teleop_source_sha256'),
                         ('direction_training_source.py','common_start_source_sha256')):
        if digest(screen_path.parent/filename) != screen[key]:
            raise ValueError('Fist calibration screen source changed')
    index = record['candidate']
    selected = torch.tensor(screen['candidate_parameters'][index],dtype=parameters.dtype)
    if not torch.equal(parameters.cpu(),selected):
        raise ValueError('Fist calibration differs from selected screen row')
    for mode in ('m5','m6'):
        row = screen['results'][mode]['candidate_records'][index]
        if row['index'] != index or row['eligible'] != [True]*starts or any(row['failure_bits']):
            raise ValueError('Fist calibration lacks all-start qualification')
    for filename in [p.name for p in donor.parent.glob('*.py')]+['model.xml']:
        if digest(Path(folder)/filename) != digest(donor.parent/filename):
            raise ValueError('Fist calibration deployment changed')
    if digest(Path(folder)/'fist_calibration_source.py') != record['source_sha256']:
        raise ValueError('Fist calibration adapter changed')
    if meta['canonical_milestone_pass'] is not False or meta['status']!='calibrated_requires_cumulative_validation':
        raise ValueError('Fist calibration cannot inherit certification')
    return donor, old, original


def compose(previous, screen_path, index, name):
    import torch
    from algorithms.urdf_learn_wasd_walk import mujoco_backend as backend
    previous,screen_path = Path(previous).resolve(),Path(screen_path).resolve()
    previous.relative_to((backend.OUTPUT/'training').resolve())
    screen_path.relative_to(backend.OUTPUT.resolve())
    old = json.loads((previous.parent/'training.json').read_text())
    screen = json.loads(screen_path.read_text())
    version=next((version for version,(_,kind,_) in VERSIONS.items() if kind==screen['kind']),None)
    if version is None:raise ValueError('Unsupported fist calibration screen')
    record = dict(version=version,checkpoint=str(previous),checkpoint_sha256=backend.digest(previous),
                  training_sha256=backend.digest(previous.parent/'training.json'),screen=str(screen_path),
                  screen_sha256=backend.digest(screen_path),candidate=index,source_sha256=backend.digest(__file__))
    blob = copy.deepcopy(torch.load(previous,map_location='cpu',weights_only=False))
    blob['turn_parameters'] = torch.tensor(screen['candidate_parameters'][index],dtype=blob['turn_parameters'].dtype)
    blob['fist_calibration'] = record
    meta = copy.deepcopy(old)
    meta.update(fist_calibration=record,arguments={**old['arguments'],'name':name},
                created_at=datetime.now(timezone.utc).isoformat(),status='calibrated_requires_cumulative_validation',
                canonical_milestone_pass=False,policy_composition='Fixed fists with screened gains '+str(VERSIONS[version][0])+'; all gait and actor tensors frozen.')
    folder = (backend.OUTPUT/'training'/name).resolve()
    folder.relative_to((backend.OUTPUT/'training').resolve())
    folder.mkdir(exist_ok=False)
    for path in list(previous.parent.glob('*.py'))+[previous.parent/'model.xml']:
        (folder/path.name).write_bytes(path.read_bytes())
    (folder/'fist_calibration_source.py').write_bytes(Path(__file__).read_bytes())
    check_contract(meta,blob,folder,backend.digest)
    torch.save(blob,folder/'model_0.pt')
    meta['checkpoints']={'model_0.pt':backend.digest(folder/'model_0.pt')}
    backend.write_json(folder/'training.json',meta)
    return folder/'model_0.pt'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--screen',required=True)
    parser.add_argument('--candidate',required=True,type=int)
    parser.add_argument('--name',required=True)
    args=parser.parse_args()
    print(compose(args.checkpoint,args.screen,args.candidate,args.name))


if __name__=='__main__':main()
