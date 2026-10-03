"""Isaac-free JSON-lines retargeting worker. One process, bounded requests, no sockets."""
import json
import sys
from pathlib import Path
import numpy as np
from avp_tracking_schema import extract_tracking_frame
from asset_paths import landau_urdf_path, landau_skeleton_json_path
from landau_retarget import LandauUpperBodyRetargeter

ROOT = Path(__file__).resolve().parent


def validate_tracking(payload):
    if not isinstance(payload, dict):
        raise ValueError('Tracking must be an object')
    result = {}
    for key in ('head', 'left_wrist', 'right_wrist', 'left_arm', 'right_arm'):
        value = payload.get(key)
        if value is None:
            continue
        arr = np.asarray(value, dtype=float)
        shapes = ((25, 4, 4), (27, 4, 4)) if key.endswith('_arm') else ((4, 4),)
        if arr.shape not in shapes or not np.isfinite(arr).all():
            raise ValueError(f'Invalid finite transform array: {key}')
        if not np.allclose(arr[..., 3, :], [0, 0, 0, 1], atol=1e-4):
            raise ValueError(f'Invalid homogeneous transform: {key}')
        if np.any(np.abs(np.linalg.det(arr[..., :3, :3])) < 1e-6):
            raise ValueError(f'Singular tracking transform: {key}')
        result[key] = arr
    if not result:
        raise ValueError('No tracked head or hands')
    return result


def solve(retarget, payload):
    tracking = validate_tracking(payload)
    # Existing native capture has 27 joints. WebXR has exactly the same first 25;
    # forearm fields are not observed and are never fabricated.
    frame = extract_tracking_frame({k: v for k, v in tracking.items() if not k.endswith('_arm')})
    for side in ('left', 'right'):
        stack = tracking.get(f'{side}_arm')
        if stack is not None:
            frame[f'{side}_arm'] = stack
            frame[f'{side}_wrist'] = stack[0]
    # Independent request seeds make Snapshot deterministic and prevent clients
    # from sharing pose state. Missing hands remain at explicit zero pose.
    retarget.left_seed[:] = 0
    retarget.right_seed[:] = 0
    retarget.last_arm_pose = {}
    pose = retarget.retarget_frame(frame)
    return {'pose': pose, 'solver': 'joint-limited CCD', 'tracked': sorted(tracking)}


def main():
    retarget = LandauUpperBodyRetargeter(urdf_path=landau_urdf_path(),
        skeleton_json_path=landau_skeleton_json_path(), snapshot_path=ROOT / 'avp_snapshot.json',
        use_trac_ik=False, arm_ccd_iterations=80)
    for line in sys.stdin:
        try:
            response = solve(retarget, json.loads(line))
        except Exception as exc:
            response = {'error': str(exc)}
        print(json.dumps(response, allow_nan=False), flush=True)

if __name__ == '__main__':
    main()
