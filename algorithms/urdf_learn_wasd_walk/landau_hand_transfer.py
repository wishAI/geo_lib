"""Explicit fixed-fist transfer of a frozen controller; no inherited gate credit."""
import argparse
import copy
import json
from datetime import datetime, timezone
from pathlib import Path

VERSION = 'fixed_fists_transfer_v1'
MUTABLE = {'hand_pose', 'nominal_q', 'nominal_ctrl', 'arguments', 'created_at',
           'status', 'checkpoints', 'canonical_milestone_pass', 'policy_composition'}
ADDED = {'hand_pose', 'hand_pose_transfer'}


def transferred_controls(old, model_path):
    import mujoco
    from algorithms.urdf_learn_wasd_walk.model_spec import fixed_hand_positions
    model = mujoco.MjModel.from_xml_path(str(model_path))
    q, ctrl = list(old['nominal_q']), list(old['nominal_ctrl'])
    for name, value in fixed_hand_positions().items():
        if name in old['action_joints']:
            raise ValueError('Finger entered policy action space')
        q[model.joint(name).qposadr[0]] = value
        ctrl[model.actuator(name).id] = value
    return q, ctrl


def check_contract(meta, blob, folder, digest):
    import torch
    from algorithms.urdf_learn_wasd_walk.landau_zero_active import same_tree
    record = meta['hand_pose_transfer']
    donor = Path(record['checkpoint'])
    old = json.loads((donor.parent/'training.json').read_text())
    if (record['version'] != VERSION or meta['hand_pose'] != 'closed_fists_v1' or
            old.get('hand_pose', 'open') != 'open' or 'hand_pose_transfer' in old or
            digest(donor) != record['checkpoint_sha256'] or
            digest(donor.parent/'training.json') != record['training_sha256'] or
            old['checkpoints'][donor.name] != digest(donor) or
            str(old.get('status', '')).startswith('invalidated_')):
        raise ValueError('Hand transfer donor/profile changed')
    original = torch.load(donor, map_location='cpu', weights_only=False)
    if set(blob) != set(original) | {'hand_pose_transfer'} or blob['hand_pose_transfer'] != record:
        raise ValueError('Hand transfer checkpoint schema changed')
    for key, value in original.items():
        if not same_tree(value, blob[key]):
            raise ValueError('Hand transfer frozen state changed: '+key)
    if set(meta) != set(old) | ADDED:
        raise ValueError('Hand transfer metadata schema changed')
    for key, value in old.items():
        if key not in MUTABLE and meta[key] != value:
            raise ValueError('Hand transfer donor metadata changed: '+key)
    if ({k:v for k,v in meta['arguments'].items() if k != 'name'} !=
            {k:v for k,v in old['arguments'].items() if k != 'name'}):
        raise ValueError('Hand transfer training arguments changed')
    q, ctrl = transferred_controls(old, donor.parent/'model.xml')
    if meta['nominal_q'] != q or meta['nominal_ctrl'] != ctrl:
        raise ValueError('Hand transfer changed nonfinger pose or fist targets')
    if meta['canonical_milestone_pass'] is not False or meta['status'] != 'transferred_requires_cumulative_validation':
        raise ValueError('Hand transfer must not inherit milestone credit')
    expected = sorted(p.name for p in donor.parent.glob('*.py')) + ['model.xml']
    if sorted(record['files']) != sorted(expected):
        raise ValueError('Hand transfer deployment file list changed')
    for filename, sha in record['files'].items():
        if digest(donor.parent/filename) != sha or digest(Path(folder)/filename) != sha:
            raise ValueError('Hand transfer deployment source changed: '+filename)
    if digest(Path(folder)/'hand_transfer_source.py') != record['source_sha256']:
        raise ValueError('Hand transfer adapter source changed')
    return donor, old, original


def compose(previous, name):
    import torch
    from algorithms.urdf_learn_wasd_walk import mujoco_backend as backend
    previous = Path(previous).resolve()
    previous.relative_to((backend.OUTPUT/'training').resolve())
    old = json.loads((previous.parent/'training.json').read_text())
    record = dict(version=VERSION, checkpoint=str(previous), checkpoint_sha256=backend.digest(previous),
                  training_sha256=backend.digest(previous.parent/'training.json'),
                  source_sha256=backend.digest(__file__),
                  files={p.name:backend.digest(p) for p in sorted(previous.parent.glob('*.py'))})
    record['files']['model.xml'] = backend.digest(previous.parent/'model.xml')
    blob = copy.deepcopy(torch.load(previous, map_location='cpu', weights_only=False))
    blob['hand_pose_transfer'] = record
    q, ctrl = transferred_controls(old, previous.parent/'model.xml')
    meta = copy.deepcopy(old)
    meta.update(hand_pose='closed_fists_v1', hand_pose_transfer=record, nominal_q=q, nominal_ctrl=ctrl,
                arguments={**old['arguments'], 'name':name}, created_at=datetime.now(timezone.utc).isoformat(),
                status='transferred_requires_cumulative_validation', canonical_milestone_pass=False,
                policy_composition='Frozen donor controller with constant fist targets; fresh cumulative proof required.')
    folder = (backend.OUTPUT/'training'/name).resolve()
    folder.relative_to((backend.OUTPUT/'training').resolve())
    folder.mkdir(exist_ok=False)
    for filename in record['files']:
        (folder/filename).write_bytes((previous.parent/filename).read_bytes())
    (folder/'hand_transfer_source.py').write_bytes(Path(__file__).read_bytes())
    check_contract(meta, blob, folder, backend.digest)
    torch.save(blob, folder/'model_0.pt')
    meta['checkpoints'] = {'model_0.pt':backend.digest(folder/'model_0.pt')}
    backend.write_json(folder/'training.json', meta)
    return folder/'model_0.pt'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--name', required=True)
    args = parser.parse_args()
    print(compose(args.checkpoint, args.name))


if __name__ == '__main__':
    main()
