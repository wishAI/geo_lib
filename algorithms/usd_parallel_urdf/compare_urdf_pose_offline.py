from __future__ import annotations

import argparse
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from asset_paths import default_usd_path, resolve_asset_paths
from skeleton_common import (
    apply_pose_to_local_matrices,
    build_pose_preset,
    matrix_to_xyz_rpy,
    pose_preset_names,
    save_json,
    world_matrices_from_local,
)


def _parse_args() -> argparse.Namespace:
    folder = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description='Offline FK comparison between the generated URDF and USD skeleton records.')
    parser.add_argument('--usd-path', type=Path, default=default_usd_path())
    parser.add_argument('--skeleton-json', type=Path, default=None)
    parser.add_argument('--urdf-path', type=Path, default=None)
    parser.add_argument('--output-path', type=Path, default=None)
    parser.add_argument(
        '--pose-preset',
        choices=pose_preset_names(),
        default='demo',
        help='Named pose preset to evaluate.',
    )
    return parser.parse_args()


def _rotation_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    norm = np.linalg.norm(axis)
    if norm < 1e-12:
        return np.eye(3, dtype=float)
    axis = axis / norm
    x, y, z = axis
    c = math.cos(angle)
    s = math.sin(angle)
    C = 1.0 - c
    return np.array([
        [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
    ], dtype=float)


def _matrix_from_origin(xyz: list[float], rpy: list[float]) -> np.ndarray:
    roll, pitch, yaw = rpy
    sx, cx = math.sin(roll), math.cos(roll)
    sy, cy = math.sin(pitch), math.cos(pitch)
    sz, cz = math.sin(yaw), math.cos(yaw)
    rot_x = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=float)
    rot_y = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=float)
    rot_z = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=float)
    matrix = np.eye(4, dtype=float)
    matrix[:3, :3] = rot_z @ rot_y @ rot_x
    matrix[:3, 3] = np.asarray(xyz, dtype=float)
    return matrix


def load_records_from_json(skeleton_json_path: Path) -> tuple[dict, list[dict]]:
    import json

    payload = json.loads(skeleton_json_path.read_text())
    records = payload['records']
    for record in records:
        record['local_matrix'] = np.asarray(record['local_matrix'], dtype=float)
        record['world_matrix'] = np.asarray(record['world_matrix'], dtype=float)
        record['axis'] = np.asarray(record['axis'], dtype=float)
        record['limits'] = tuple(record['limits'])
    return payload, records


def compare_offline_pose(
    *,
    records: list[dict],
    urdf_path: Path,
    pose: dict[str, float],
    pose_preset: str,
    skeleton_json_path: Path | None = None,
) -> dict:
    skeleton_json_str = str(skeleton_json_path) if skeleton_json_path is not None else None
    record_by_name = {record['name']: record for record in records}
    usd_local = apply_pose_to_local_matrices(records, pose)
    usd_world = world_matrices_from_local(records, usd_local)
    usd_root_inv = np.linalg.inv(usd_world[0])
    usd_by_name = {record['name']: usd_root_inv @ usd_world[record['index']] for record in records}

    tree = ET.parse(urdf_path)
    root = tree.getroot()
    child_joint = {}
    parent_links = set()
    for joint in root.findall('joint'):
        name = joint.attrib['name']
        parent = joint.find('parent').attrib['link']
        child = joint.find('child').attrib['link']
        parent_links.add(parent)
        origin = joint.find('origin')
        xyz = [float(v) for v in origin.attrib.get('xyz', '0 0 0').split()]
        rpy = [float(v) for v in origin.attrib.get('rpy', '0 0 0').split()]
        axis_node = joint.find('axis')
        axis = [float(v) for v in axis_node.attrib.get('xyz', '0 0 1').split()] if axis_node is not None else [0.0, 0.0, 1.0]
        child_joint[child] = {
            'name': name,
            'parent': parent,
            'child': child,
            'origin': _matrix_from_origin(xyz, rpy),
            'axis': np.asarray(axis, dtype=float),
            'angle': float(pose.get(name, 0.0)),
        }

    root_links = parent_links.difference(child_joint)
    urdf_world = {link_name: np.eye(4, dtype=float) for link_name in root_links}
    unresolved = set(child_joint)
    while unresolved:
        progressed = False
        for child in list(unresolved):
            spec = child_joint[child]
            parent = spec['parent']
            if parent not in urdf_world:
                continue
            motion = np.eye(4, dtype=float)
            motion[:3, :3] = _rotation_matrix(spec['axis'], spec['angle'])
            urdf_world[child] = urdf_world[parent] @ spec['origin'] @ motion
            unresolved.remove(child)
            progressed = True
        if not progressed:
            raise RuntimeError(f'Unable to resolve URDF FK for links: {sorted(unresolved)}')

    urdf_root_name = records[0]['name']
    urdf_root_inv = np.linalg.inv(urdf_world[urdf_root_name])

    per_link = []
    for name, usd_matrix in usd_by_name.items():
        urdf_matrix = urdf_world.get(name)
        if urdf_matrix is None:
            continue
        urdf_matrix = urdf_root_inv @ urdf_matrix
        pos_error = float(np.linalg.norm(usd_matrix[:3, 3] - urdf_matrix[:3, 3]))
        usd_xyz, usd_rpy = matrix_to_xyz_rpy(usd_matrix)
        urdf_xyz, urdf_rpy = matrix_to_xyz_rpy(urdf_matrix)
        rot_error = float(np.linalg.norm(np.asarray(usd_rpy) - np.asarray(urdf_rpy)))
        per_link.append({
            'name': name,
            'position_error_m': pos_error,
            'rotation_rpy_delta_norm': rot_error,
            'usd_xyz': [float(v) for v in usd_xyz],
            'urdf_xyz': [float(v) for v in urdf_xyz],
        })
    per_link.sort(key=lambda item: item['position_error_m'], reverse=True)
    return {
        'skeleton_json': skeleton_json_str,
        'urdf_path': str(urdf_path),
        'pose_preset': pose_preset,
        'comparison_link_count': len(per_link),
        'max_position_error_m': max((item['position_error_m'] for item in per_link), default=0.0),
        'mean_position_error_m': float(np.mean([item['position_error_m'] for item in per_link])) if per_link else 0.0,
        'max_rotation_rpy_delta_norm': max((item['rotation_rpy_delta_norm'] for item in per_link), default=0.0),
        'worst_links': per_link[:15],
    }


def main() -> None:
    args = _parse_args()
    folder = Path(__file__).resolve().parent
    asset_paths = resolve_asset_paths(args.usd_path, folder / 'outputs')
    skeleton_json_path = args.skeleton_json or asset_paths.skeleton_json
    urdf_path = args.urdf_path or asset_paths.primitive_urdf
    output_path = args.output_path or (asset_paths.primitive_validation_dir / 'offline_transform_comparison.json')
    _, records = load_records_from_json(skeleton_json_path)
    pose = build_pose_preset(records, args.pose_preset)
    summary = compare_offline_pose(
        records=records,
        urdf_path=urdf_path,
        pose=pose,
        pose_preset=args.pose_preset,
        skeleton_json_path=skeleton_json_path,
    )
    save_json(output_path, summary)
    print(f"[OFFLINE] compared links: {summary['comparison_link_count']}")
    print(f"[OFFLINE] max position error: {summary['max_position_error_m']:.6f} m")
    print(f"[OFFLINE] wrote: {output_path}")


if __name__ == '__main__':
    main()
