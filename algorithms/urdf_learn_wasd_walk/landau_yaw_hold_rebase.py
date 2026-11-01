"""Bind a bounded yaw/hold calibration to an unchanged learned dual controller."""
import argparse
import copy
import json
from datetime import datetime, timezone
from pathlib import Path

VARIANT = 'yaw_hold_calibrated_dual_residual_v1'
FIXED = [i for i in range(20) if i not in (3,17)]


def check_contract(meta, blob, digest):
    import torch
    from algorithms.urdf_learn_wasd_walk import landau_right_rebase as right
    if meta.get('residual_variant') != VARIANT or blob.get('residual_variant') != VARIANT:
        raise ValueError('Yaw/hold calibration variant mismatch')
    path = Path(meta['calibration_donor_checkpoint'])
    old = json.loads((path.parent/'training.json').read_text())
    if (digest(path) != meta['calibration_donor_checkpoint_sha256'] or
        digest(path.parent/'training.json') != meta['calibration_donor_training_sha256'] or
        old['checkpoints'][path.name] != digest(path) or str(old.get('status','')).startswith('invalidated_')):
        raise ValueError('Calibration donor changed')
    original = torch.load(path,map_location='cpu',weights_only=False)
    right.check_contract(old,original,digest)
    # Keep the whole underlying model/source/parent identity, not just actor weights.
    mutable = {'residual_variant','arguments','checkpoints','created_at','status','canonical_milestone_pass','policy_composition'}
    for key,value in old.items():
        if key not in mutable and meta.get(key) != value:raise ValueError('Calibration donor metadata changed: '+key)
    if {k:v for k,v in meta['arguments'].items() if k != 'name'} != {k:v for k,v in old['arguments'].items() if k != 'name'}:
        raise ValueError('Calibration command/training arguments changed')
    p = blob['turn_parameters']
    if tuple(p.shape) != (20,) or not torch.isfinite(p).all() or not torch.equal(p[FIXED],original['turn_parameters'][FIXED]):
        raise ValueError('Only hold/yaw parameters3/17 may change')
    if not -.12 <= float(p[3]) <= .12 or not .5 <= float(p[17]) <= 2.:
        raise ValueError('Yaw/hold calibration outside existing bounds')
    if not torch.equal(blob['parameters'],original['parameters']):raise ValueError('Calibration gait changed')
    for key in ('standing_actor_state_dict','residual_actor_state_dict','active_residual_actor_state_dict'):
        if not right.same_weights(blob[key],original[key]):raise ValueError('Calibration actor changed: '+key)
    for filename,key in (('turn_source.py','turn_source_sha256'),('residual_source.py','residual_source_sha256'),
                         ('active_residual_source.py','active_residual_source_sha256')):
        if digest(path.parent/filename) != meta[key]:raise ValueError('Calibration donor source changed')
    calibration = meta['yaw_hold_calibration']
    if calibration['changed_indices'] != [3,17] or len(calibration['screens']) < 2:
        raise ValueError('Calibration requires repeated bounded screens')
    if (len({str(Path(r['path']).resolve()) for r in calibration['screens']}) != len(calibration['screens']) or
        len({r['sha256'] for r in calibration['screens']}) != len(calibration['screens'])):
        raise ValueError('Calibration screens must be distinct')
    index = calibration['candidate_index']
    if not isinstance(index,int) or index < 0:raise ValueError('Invalid calibration candidate index')
    for record in calibration['screens']:
        screen_path = Path(record['path'])
        screen = json.loads(screen_path.read_text())
        if digest(screen_path) != record['sha256'] or screen['checkpoint_sha256'] != digest(path):
            raise ValueError('Calibration screen changed')
        if (screen['kind'] != 'nominal_common_start_yaw_hold_calibration' or screen['changed_indices'] != [3,17] or
            digest(screen_path.parent/'screen_source.py') != screen['source_sha256']):
            raise ValueError('Calibration screen provenance changed')
        if screen.get('training_sha256') != digest(path.parent/'training.json'):
            raise ValueError('Calibration screen donor metadata changed')
        for key in ('turn_source_sha256','residual_source_sha256','active_residual_source_sha256'):
            if screen.get(key) != old[key]:raise ValueError('Calibration screen controller source changed')
        for filename,key in (('audit_source.py','audit_source_sha256'),('teleop_source.py','teleop_source_sha256'),
                             ('direction_training_source.py','common_start_source_sha256')):
            if digest(screen_path.parent/filename) != screen.get(key):raise ValueError('Calibration screen helper changed')
        if not torch.equal(torch.tensor(screen['candidate_parameters'][index],dtype=p.dtype),p.cpu()):
            raise ValueError('Calibration parameters differ from selected screen row')
        for mode in ('m5','m6'):
            row = screen['results'][mode]['candidate_records'][index]
            if row['index'] != index or len(row['eligible']) != 4 or not all(row['eligible']):
                raise ValueError('Selected calibration lacks repeated four-start qualification')
    return True


def compose(previous, screens, index, name):
    import torch
    from algorithms.urdf_learn_wasd_walk import mujoco_backend as backend
    previous = Path(previous).resolve()
    previous.relative_to((backend.OUTPUT/'training').resolve())
    old = json.loads((previous.parent/'training.json').read_text())
    original = torch.load(previous,map_location='cpu',weights_only=False)
    screens = [Path(p).resolve() for p in screens]
    if len(set(screens)) != len(screens):raise ValueError('Require distinct repeated screens')
    screen = json.loads(screens[0].read_text())
    blob = copy.deepcopy(original)
    blob['turn_parameters'] = torch.tensor(screen['candidate_parameters'][index],dtype=original['turn_parameters'].dtype)
    blob['residual_variant'] = VARIANT
    meta = copy.deepcopy(old)
    meta.update(residual_variant=VARIANT,arguments={**old['arguments'],'name':name},
        status='calibrated_requires_nominal_validation',canonical_milestone_pass=False,
        created_at=datetime.now(timezone.utc).isoformat(),
        calibration_donor_checkpoint=str(previous),calibration_donor_checkpoint_sha256=backend.digest(previous),
        calibration_donor_training_sha256=backend.digest(previous.parent/'training.json'),
        policy_composition='Bounded learned left yaw/hold parameters3/17; all actors, gait, other18 turn parameters and saved deployment sources unchanged',
        yaw_hold_calibration={'changed_indices':[3,17],'candidate_index':index,'source_sha256':backend.digest(__file__),
            'screens':[{'path':str(p),'sha256':backend.digest(p)} for p in screens],'new_ppo_training':False})
    check_contract(meta,blob,backend.digest)
    folder = (backend.OUTPUT/'training'/name).resolve()
    folder.relative_to((backend.OUTPUT/'training').resolve());folder.mkdir(exist_ok=False)
    for filename in ('control_source.py','turn_source.py','teleop_contract.py','model.xml',
                     'residual_source.py','active_residual_source.py','right_rebase_source.py'):
        (folder/filename).write_bytes((previous.parent/filename).read_bytes())
    (folder/'yaw_hold_rebase_source.py').write_bytes(Path(__file__).read_bytes())
    torch.save(blob,folder/'model_0.pt')
    meta['checkpoints'] = {'model_0.pt':backend.digest(folder/'model_0.pt')}
    backend.write_json(folder/'training.json',meta)
    return folder/'model_0.pt'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True);parser.add_argument('--screen',action='append',required=True)
    parser.add_argument('--candidate',type=int,required=True);parser.add_argument('--name',required=True)
    args = parser.parse_args()
    print(compose(args.checkpoint,args.screen,args.candidate,args.name))


if __name__ == '__main__':main()
