from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Sequence

import numpy as np

from skeleton_common import (
    apply_pose_to_local_matrices,
    build_animation_clip,
    build_pose_preset,
    infer_lateral_axis_world,
    interpolate_pose_dict,
    remap_pose_to_urdf_joint_names,
    world_matrices_from_local,
)


ARM_LINK_PAIRS: tuple[tuple[str, str], ...] = (
    ('shoulder_l', 'shoulder_r'),
    ('arm_stretch_l', 'arm_stretch_r'),
    ('forearm_stretch_l', 'forearm_stretch_r'),
    ('hand_l', 'hand_r'),
)

ARM_PROGRESSIVE_STEPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ('rest', ()),
    ('shoulder_only', ('shoulder_l', 'shoulder_r')),
    ('shoulder_plus_upperarm', ('shoulder_l', 'shoulder_r', 'arm_stretch_l', 'arm_stretch_r')),
    (
        'shoulder_plus_upperarm_plus_forearm',
        (
            'shoulder_l',
            'shoulder_r',
            'arm_stretch_l',
            'arm_stretch_r',
            'forearm_stretch_l',
            'forearm_stretch_r',
        ),
    ),
    (
        'full_arm_chain',
        (
            'shoulder_l',
            'shoulder_r',
            'arm_stretch_l',
            'arm_stretch_r',
            'forearm_stretch_l',
            'forearm_stretch_r',
            'hand_l',
            'hand_r',
        ),
    ),
)


def mirror_matrix_from_records(records: Sequence[dict]) -> np.ndarray:
    lateral_axis = infer_lateral_axis_world(records)
    return np.eye(3, dtype=float) - 2.0 * np.outer(lateral_axis, lateral_axis)


def root_relative_world_map(records: Sequence[dict], pose: dict[str, float]) -> dict[str, np.ndarray]:
    local_matrices = apply_pose_to_local_matrices(records, pose)
    world_matrices = world_matrices_from_local(records, local_matrices)
    root_inverse = np.linalg.inv(world_matrices[0])
    return {
        record['name']: root_inverse @ world_matrices[record['index']]
        for record in records
    }


def _rotation_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    norm = np.linalg.norm(axis)
    if norm < 1e-12:
        return np.eye(3, dtype=float)
    axis = axis / norm
    x, y, z = axis
    c = math.cos(angle)
    s = math.sin(angle)
    one_minus_c = 1.0 - c
    return np.array(
        [
            [c + x * x * one_minus_c, x * y * one_minus_c - z * s, x * z * one_minus_c + y * s],
            [y * x * one_minus_c + z * s, c + y * y * one_minus_c, y * z * one_minus_c - x * s],
            [z * x * one_minus_c - y * s, z * y * one_minus_c + x * s, c + z * z * one_minus_c],
        ],
        dtype=float,
    )


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


def urdf_root_relative_world_map(records: Sequence[dict], urdf_path: Path, pose: dict[str, float]) -> dict[str, np.ndarray]:
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    urdf_pose = remap_pose_to_urdf_joint_names(
        pose,
        [joint.attrib['name'] for joint in root.findall('joint')],
    )
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
            'parent': parent,
            'origin': _matrix_from_origin(xyz, rpy),
            'axis': np.asarray(axis, dtype=float),
            'angle': float(urdf_pose.get(name, 0.0)),
        }

    root_links = parent_links.difference(child_joint)
    world_map = {link_name: np.eye(4, dtype=float) for link_name in root_links}
    unresolved = set(child_joint)
    while unresolved:
        progressed = False
        for child in list(unresolved):
            spec = child_joint[child]
            parent = spec['parent']
            if parent not in world_map:
                continue
            motion = np.eye(4, dtype=float)
            motion[:3, :3] = _rotation_matrix(spec['axis'], spec['angle'])
            world_map[child] = world_map[parent] @ spec['origin'] @ motion
            unresolved.remove(child)
            progressed = True
        if not progressed:
            raise RuntimeError(f'Unable to resolve URDF FK for links: {sorted(unresolved)}')

    root_name = records[0]['name']
    root_inverse = np.linalg.inv(world_map[root_name])
    return {name: root_inverse @ matrix for name, matrix in world_map.items()}


def mirrored_rotation_error_deg(left_matrix: np.ndarray, right_matrix: np.ndarray, mirror_matrix: np.ndarray) -> float:
    mirrored_right = mirror_matrix @ right_matrix[:3, :3] @ mirror_matrix
    delta = left_matrix[:3, :3].T @ mirrored_right
    trace = float(np.clip((np.trace(delta) - 1.0) * 0.5, -1.0, 1.0))
    return math.degrees(math.acos(trace))


def pair_mirror_metrics(world_map: dict[str, np.ndarray], mirror_matrix: np.ndarray, left_name: str, right_name: str) -> dict:
    left_matrix = world_map[left_name]
    right_matrix = world_map[right_name]
    mirrored_right_xyz = mirror_matrix @ right_matrix[:3, 3]
    return {
        'left': left_name,
        'right': right_name,
        'mirror_position_error_m': float(np.linalg.norm(left_matrix[:3, 3] - mirrored_right_xyz)),
        'mirror_rotation_error_deg': float(mirrored_rotation_error_deg(left_matrix, right_matrix, mirror_matrix)),
        'left_xyz': [float(v) for v in left_matrix[:3, 3]],
        'mirrored_right_xyz': [float(v) for v in mirrored_right_xyz],
    }


def progressive_arm_symmetry_scan(records: Sequence[dict], pose: dict[str, float]) -> list[dict]:
    mirror_matrix = mirror_matrix_from_records(records)
    scan = []
    for label, joint_names in ARM_PROGRESSIVE_STEPS:
        step_pose = {joint_name: float(pose[joint_name]) for joint_name in joint_names if joint_name in pose}
        world_map = root_relative_world_map(records, step_pose)
        scan.append(
            {
                'step': label,
                'joint_values': step_pose,
                'pair_metrics': [
                    pair_mirror_metrics(world_map, mirror_matrix, left_name, right_name)
                    for left_name, right_name in ARM_LINK_PAIRS
                ],
            }
        )
    return scan


def pose_for_clip_time(clip: list[tuple[str, dict[str, float], float]], time_s: float) -> tuple[str, str, dict[str, float]]:
    if len(clip) == 1:
        preset_name, pose, _ = clip[0]
        return preset_name, preset_name, dict(pose)
    total_duration = sum(duration for _, _, duration in clip)
    wrapped = time_s % total_duration
    elapsed = 0.0
    for index, (preset_name, pose, duration_s) in enumerate(clip):
        next_name, next_pose, _ = clip[(index + 1) % len(clip)]
        end = elapsed + duration_s
        if wrapped <= end or index == len(clip) - 1:
            alpha = 0.0 if duration_s <= 1e-8 else (wrapped - elapsed) / duration_s
            return preset_name, next_name, interpolate_pose_dict(pose, next_pose, alpha)
        elapsed = end
    preset_name, pose, _ = clip[-1]
    return preset_name, preset_name, dict(pose)


def animation_clip_balance_report(
    records: Sequence[dict],
    clip_name: str,
    *,
    link_pairs: Sequence[tuple[str, str]] = (('hand_l', 'hand_r'),),
    sample_count: int = 120,
) -> dict:
    clip = build_animation_clip(records, clip_name)
    total_duration = sum(duration for _, _, duration in clip)
    per_link_samples: dict[str, list[np.ndarray]] = {name: [] for pair in link_pairs for name in pair}

    for step in range(max(sample_count, 1) + 1):
        time_s = (step / max(sample_count, 1)) * total_duration
        _, _, pose = pose_for_clip_time(clip, time_s)
        world_map = root_relative_world_map(records, pose)
        for left_name, right_name in link_pairs:
            per_link_samples[left_name].append(world_map[left_name][:3, 3].copy())
            per_link_samples[right_name].append(world_map[right_name][:3, 3].copy())

    pair_metrics = []
    for left_name, right_name in link_pairs:
        left_samples = np.asarray(per_link_samples[left_name], dtype=float)
        right_samples = np.asarray(per_link_samples[right_name], dtype=float)
        left_path = float(np.sum(np.linalg.norm(np.diff(left_samples, axis=0), axis=1)))
        right_path = float(np.sum(np.linalg.norm(np.diff(right_samples, axis=0), axis=1)))
        smaller = min(left_path, right_path)
        ratio = float(max(left_path, right_path) / smaller) if smaller > 1e-8 else float('inf')
        pair_metrics.append(
            {
                'left': left_name,
                'right': right_name,
                'left_path_length_m': left_path,
                'right_path_length_m': right_path,
                'path_length_ratio': ratio,
                'left_range_xyz_m': [float(v) for v in (left_samples.max(axis=0) - left_samples.min(axis=0))],
                'right_range_xyz_m': [float(v) for v in (right_samples.max(axis=0) - right_samples.min(axis=0))],
            }
        )
    return {
        'clip_name': clip_name,
        'sample_count': int(max(sample_count, 1) + 1),
        'duration_s': float(total_duration),
        'pair_metrics': pair_metrics,
    }


def arm_pose_symmetry_report(
    records: Sequence[dict],
    pose_preset: str,
    *,
    urdf_path: Path | None = None,
) -> dict:
    pose = build_pose_preset(records, pose_preset)
    mirror_matrix = mirror_matrix_from_records(records)
    usd_world_map = root_relative_world_map(records, pose)
    report = {
        'pose_preset': pose_preset,
        'pose_joint_values': {joint_name: float(angle) for joint_name, angle in sorted(pose.items())},
        'mirror_matrix': mirror_matrix.tolist(),
        'usd_pair_metrics': [
            pair_mirror_metrics(usd_world_map, mirror_matrix, left_name, right_name)
            for left_name, right_name in ARM_LINK_PAIRS
        ],
        'progressive_scan': progressive_arm_symmetry_scan(records, pose),
    }

    if urdf_path is not None:
        urdf_world_map = urdf_root_relative_world_map(records, urdf_path, pose)
        report['urdf_path'] = str(urdf_path)
        report['urdf_pair_metrics'] = [
            pair_mirror_metrics(urdf_world_map, mirror_matrix, left_name, right_name)
            for left_name, right_name in ARM_LINK_PAIRS
        ]
        usd_vs_urdf = []
        for left_name, right_name in ARM_LINK_PAIRS:
            usd_left = usd_world_map[left_name]
            usd_right = usd_world_map[right_name]
            urdf_left = urdf_world_map[left_name]
            urdf_right = urdf_world_map[right_name]
            usd_vs_urdf.append(
                {
                    'left': left_name,
                    'right': right_name,
                    'left_position_error_m': float(np.linalg.norm(usd_left[:3, 3] - urdf_left[:3, 3])),
                    'right_position_error_m': float(np.linalg.norm(usd_right[:3, 3] - urdf_right[:3, 3])),
                    'left_rotation_error_deg': float(rotation_error_deg(usd_left[:3, :3], urdf_left[:3, :3])),
                    'right_rotation_error_deg': float(rotation_error_deg(usd_right[:3, :3], urdf_right[:3, :3])),
                }
            )
        report['usd_vs_urdf_pair_metrics'] = usd_vs_urdf

    return report


def rotation_error_deg(left_rotation: np.ndarray, right_rotation: np.ndarray) -> float:
    delta = left_rotation.T @ right_rotation
    trace = float(np.clip((np.trace(delta) - 1.0) * 0.5, -1.0, 1.0))
    return math.degrees(math.acos(trace))
