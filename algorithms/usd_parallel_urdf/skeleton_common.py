from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np


def _orthonormalize(rotation: np.ndarray) -> np.ndarray:
    u, _, vh = np.linalg.svd(rotation)
    fixed = u @ vh
    if np.linalg.det(fixed) < 0:
        u[:, -1] *= -1.0
        fixed = u @ vh
    return fixed


def matrix_to_xyz_rpy(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rotation = _orthonormalize(matrix[:3, :3])
    tx, ty, tz = matrix[:3, 3]
    sy = math.sqrt(rotation[0, 0] ** 2 + rotation[1, 0] ** 2)
    singular = sy < 1e-8
    if not singular:
        roll = math.atan2(rotation[2, 1], rotation[2, 2])
        pitch = math.atan2(-rotation[2, 0], sy)
        yaw = math.atan2(rotation[1, 0], rotation[0, 0])
    else:
        roll = math.atan2(-rotation[1, 2], rotation[1, 1])
        pitch = math.atan2(-rotation[2, 0], sy)
        yaw = 0.0
    return np.array([tx, ty, tz], dtype=float), np.array([roll, pitch, yaw], dtype=float)


def rpy_to_matrix(rpy: Sequence[float]) -> np.ndarray:
    roll, pitch, yaw = (float(value) for value in rpy)
    cr = math.cos(roll)
    sr = math.sin(roll)
    cp = math.cos(pitch)
    sp = math.sin(pitch)
    cy = math.cos(yaw)
    sy = math.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=float,
    )


def axis_angle_matrix(axis: Sequence[float], angle_rad: float) -> np.ndarray:
    axis_arr = np.asarray(axis, dtype=float)
    norm = np.linalg.norm(axis_arr)
    if norm < 1e-8 or abs(angle_rad) < 1e-10:
        return np.eye(3, dtype=float)
    x, y, z = axis_arr / norm
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    one_c = 1.0 - c
    return np.array(
        [
            [c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s],
            [y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s],
            [z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c],
        ],
        dtype=float,
    )


def align_x_to_vector(vector: Sequence[float]) -> np.ndarray:
    direction = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(direction)
    if norm < 1e-8:
        return np.eye(3, dtype=float)
    x_axis = direction / norm
    helper = np.array([0.0, 0.0, 1.0], dtype=float)
    if abs(np.dot(x_axis, helper)) > 0.92:
        helper = np.array([0.0, 1.0, 0.0], dtype=float)
    y_axis = np.cross(helper, x_axis)
    y_norm = np.linalg.norm(y_axis)
    if y_norm < 1e-8:
        return np.eye(3, dtype=float)
    y_axis /= y_norm
    z_axis = np.cross(x_axis, y_axis)
    return np.column_stack((x_axis, y_axis, z_axis))


def _keyword_base(name: str) -> str:
    return name.split('/')[-1]


def make_unique_link_names(paths: Sequence[str]) -> Dict[str, str]:
    counts: Dict[str, int] = {}
    result: Dict[str, str] = {}
    for path in paths:
        base = _keyword_base(path)
        count = counts.get(base, 0)
        counts[base] = count + 1
        result[path] = base if count == 0 else f'{base}_{count}'
    return result


def _normalize_axis(axis: Sequence[float], fallback: Sequence[float]) -> np.ndarray:
    axis_arr = np.asarray(axis, dtype=float)
    norm = np.linalg.norm(axis_arr)
    if norm < 1e-8:
        return np.asarray(fallback, dtype=float)
    return axis_arr / norm


def _nearest_cardinal_axis(vector: Sequence[float], fallback: Sequence[float]) -> np.ndarray:
    direction = _normalize_axis(vector, fallback)
    axis = np.zeros(3, dtype=float)
    dominant_index = int(np.argmax(np.abs(direction)))
    axis[dominant_index] = 1.0 if direction[dominant_index] >= 0.0 else -1.0
    return axis


def _orthogonal_local_axis(axis: Sequence[float], fallback: Sequence[float]) -> np.ndarray:
    direction = _normalize_axis(axis, fallback)
    candidates = [np.eye(3, dtype=float)[idx] for idx in range(3)]
    best_axis = candidates[0]
    best_norm = 0.0
    for candidate in candidates:
        projected = candidate - direction * float(np.dot(candidate, direction))
        norm = float(np.linalg.norm(projected))
        if norm > best_norm:
            best_axis = projected
            best_norm = norm
    return _normalize_axis(best_axis, fallback)


def infer_lateral_axis_world(records: Sequence[dict]) -> np.ndarray:
    by_name = {record['name']: record for record in records}
    preferred_pairs = (
        ('thigh_stretch_l', 'thigh_stretch_r'),
        ('shoulder_l', 'shoulder_r'),
        ('foot_l', 'foot_r'),
        ('arm_stretch_l', 'arm_stretch_r'),
    )
    for left_name, right_name in preferred_pairs:
        left = by_name.get(left_name)
        right = by_name.get(right_name)
        if left is None or right is None:
            continue
        lateral = np.asarray(left['world_xyz'], dtype=float) - np.asarray(right['world_xyz'], dtype=float)
        if np.linalg.norm(lateral) > 1e-8:
            return lateral / np.linalg.norm(lateral)
    return np.array([1.0, 0.0, 0.0], dtype=float)


@dataclass(frozen=True)
class BodyBasis:
    lateral_world: np.ndarray
    up_world: np.ndarray
    forward_world: np.ndarray


def infer_body_basis_world(records: Sequence[dict]) -> BodyBasis:
    by_name = {record['name']: record for record in records}
    lateral = infer_lateral_axis_world(records)

    up_candidates: list[np.ndarray] = []
    root = by_name.get('root_x')
    if root is not None:
        root_xyz = np.asarray(root['world_xyz'], dtype=float)
        for name in ('head_x', 'neck_x', 'spine_03_x', 'spine_02_x', 'spine_01_x'):
            record = by_name.get(name)
            if record is None:
                continue
            delta = np.asarray(record['world_xyz'], dtype=float) - root_xyz
            if np.linalg.norm(delta) > 1e-8:
                up_candidates.append(delta)
    if not up_candidates:
        for record in records:
            name = record['name']
            if 'head' in name or 'neck' in name or 'spine' in name:
                delta = np.asarray(record['world_xyz'], dtype=float)
                if np.linalg.norm(delta) > 1e-8:
                    up_candidates.append(delta)
    if up_candidates:
        up = np.mean(np.vstack(up_candidates), axis=0)
    else:
        up = np.array([0.0, 0.0, 1.0], dtype=float)
    up = up - lateral * float(np.dot(up, lateral))
    up = _normalize_axis(up, [0.0, 0.0, 1.0])
    if up[2] < 0.0:
        up = -up

    forward = np.cross(up, lateral)
    forward = _normalize_axis(forward, [0.0, 1.0, 0.0])

    toe_pairs = (
        ('toes_01_l', 'foot_l'),
        ('toes_01_r', 'foot_r'),
        ('foot_l', 'leg_stretch_l'),
        ('foot_r', 'leg_stretch_r'),
    )
    forward_votes: list[np.ndarray] = []
    for distal_name, proximal_name in toe_pairs:
        distal = by_name.get(distal_name)
        proximal = by_name.get(proximal_name)
        if distal is None or proximal is None:
            continue
        delta = np.asarray(distal['world_xyz'], dtype=float) - np.asarray(proximal['world_xyz'], dtype=float)
        delta = delta - lateral * float(np.dot(delta, lateral))
        if np.linalg.norm(delta) > 1e-8:
            forward_votes.append(delta)
    if forward_votes:
        forward_hint = np.mean(np.vstack(forward_votes), axis=0)
        if float(np.dot(forward, forward_hint)) < 0.0:
            forward = -forward
    up = _normalize_axis(np.cross(lateral, forward), [0.0, 0.0, 1.0])
    if up[2] < 0.0:
        up = -up
        forward = -forward
    return BodyBasis(lateral_world=lateral, up_world=up, forward_world=forward)


def infer_hand_width_axes_world(records: Sequence[dict], body_basis: BodyBasis) -> Dict[str, np.ndarray]:
    by_name = {record['name']: record for record in records}
    width_axes: Dict[str, np.ndarray] = {}
    for suffix, side_sign in (('l', 1.0), ('r', -1.0)):
        hand_name = f'hand_{suffix}'
        hand_record = by_name.get(hand_name)

        finger_dirs: list[np.ndarray] = []
        for proximal_name in (f'index1_{suffix}', f'middle1_{suffix}', f'ring1_{suffix}', f'pinky1_{suffix}'):
            proximal = by_name.get(proximal_name)
            if proximal is None:
                continue
            child_names = list(proximal.get('child_names', ()))
            distal = by_name.get(child_names[0]) if child_names else None
            if distal is not None:
                delta = np.asarray(distal['world_xyz'], dtype=float) - np.asarray(proximal['world_xyz'], dtype=float)
            else:
                delta = np.asarray(proximal['world_matrix'], dtype=float)[:3, :3] @ np.array([0.0, 1.0, 0.0], dtype=float)
            if np.linalg.norm(delta) > 1e-8:
                finger_dirs.append(delta)
        if finger_dirs:
            finger_dir_world = _normalize_axis(np.mean(np.vstack(finger_dirs), axis=0), body_basis.forward_world)
        elif hand_record is not None:
            finger_dir_world = _normalize_axis(
                np.asarray(hand_record['world_matrix'], dtype=float)[:3, :3][:, 1],
                body_basis.forward_world,
            )
        else:
            finger_dir_world = body_basis.forward_world

        width_candidates: list[np.ndarray] = []
        for radial_name, ulnar_name in (
            (f'index1_base_{suffix}', f'pinky1_base_{suffix}'),
            (f'index1_base_{suffix}', f'ring1_base_{suffix}'),
            (f'middle1_base_{suffix}', f'pinky1_base_{suffix}'),
        ):
            radial = by_name.get(radial_name)
            ulnar = by_name.get(ulnar_name)
            if radial is None or ulnar is None:
                continue
            delta = np.asarray(radial['world_xyz'], dtype=float) - np.asarray(ulnar['world_xyz'], dtype=float)
            delta = delta - finger_dir_world * float(np.dot(delta, finger_dir_world))
            if np.linalg.norm(delta) > 1e-8:
                width_candidates.append(delta)
        if width_candidates:
            width_world = np.mean(np.vstack(width_candidates), axis=0)
        elif hand_record is not None:
            width_world = np.asarray(hand_record['world_matrix'], dtype=float)[:3, :3][:, 0]
        else:
            width_world = body_basis.lateral_world * side_sign

        width_world = width_world - finger_dir_world * float(np.dot(width_world, finger_dir_world))
        if np.linalg.norm(width_world) < 1e-8 and hand_record is not None:
            width_world = np.asarray(hand_record['world_matrix'], dtype=float)[:3, :3][:, 0]
        width_world = _normalize_axis(width_world, body_basis.lateral_world * side_sign)

        thumb = by_name.get(f'thumb1_{suffix}')
        if hand_record is not None and thumb is not None:
            thumb_delta = np.asarray(thumb['world_xyz'], dtype=float) - np.asarray(hand_record['world_xyz'], dtype=float)
            thumb_delta = thumb_delta - finger_dir_world * float(np.dot(thumb_delta, finger_dir_world))
            if np.linalg.norm(thumb_delta) > 1e-8 and float(np.dot(width_world, thumb_delta)) < 0.0:
                width_world = -width_world
        elif float(np.dot(width_world, body_basis.lateral_world)) * side_sign < 0.0:
            width_world = -width_world

        width_axes[suffix] = width_world
    return width_axes


def _choose_axis_for_world_target(
    local_rotation: np.ndarray,
    target_world: Sequence[float],
    *,
    primary_local_axis: Sequence[float] | None = None,
    allow_primary: bool = False,
    fallback: Sequence[float] = (1.0, 0.0, 0.0),
    preferred_local_axis: Sequence[float] | None = None,
) -> np.ndarray:
    rotation = _orthonormalize(np.asarray(local_rotation, dtype=float))
    target = _normalize_axis(target_world, fallback)
    primary = None if primary_local_axis is None else _normalize_axis(primary_local_axis, [0.0, 1.0, 0.0])
    local_axis = rotation.T @ target
    if not allow_primary and primary is not None:
        local_axis = local_axis - primary * float(np.dot(local_axis, primary))
    if np.linalg.norm(local_axis) < 1e-8:
        fallback_world = _normalize_axis(fallback, [1.0, 0.0, 0.0])
        local_axis = rotation.T @ fallback_world
        if not allow_primary and primary is not None:
            local_axis = local_axis - primary * float(np.dot(local_axis, primary))
    if np.linalg.norm(local_axis) < 1e-8 and primary is not None:
        local_axis = _orthogonal_local_axis(primary, [1.0, 0.0, 0.0])
    local_axis = _normalize_axis(local_axis, [1.0, 0.0, 0.0])
    preferred = None if preferred_local_axis is None else _normalize_axis(preferred_local_axis, [1.0, 0.0, 0.0])
    if preferred is not None and float(np.dot(local_axis, preferred)) < 0.0:
        local_axis = -local_axis
    return local_axis


def infer_joint_axis(
    name: str,
    local_matrix: np.ndarray,
    primary_local_axis: Sequence[float],
    body_basis: BodyBasis,
    *,
    world_matrix: np.ndarray | None = None,
    hand_width_world: Sequence[float] | None = None,
    hinge_normal_world: Sequence[float] | None = None,
) -> np.ndarray:
    primary = _normalize_axis(primary_local_axis, [0.0, 1.0, 0.0])
    reference_matrix = np.asarray(world_matrix if world_matrix is not None else local_matrix, dtype=float)
    semantic_name = urdf_joint_name_for_link_name(name)
    if semantic_name.endswith('hip_yaw_joint') or semantic_name.endswith('waist_yaw_joint') or semantic_name.endswith('head_yaw_joint'):
        return _choose_axis_for_world_target(
            reference_matrix[:3, :3],
            body_basis.up_world,
            primary_local_axis=primary,
            allow_primary=True,
            fallback=[0.0, 0.0, 1.0],
        )
    if 'twist' in name:
        return _nearest_cardinal_axis(primary, [0.0, 1.0, 0.0])
    if name.startswith('thumb1_'):
        if hand_width_world is not None:
            thumb_direction_world = _normalize_axis(reference_matrix[:3, :3] @ primary, body_basis.forward_world)
            target_world = np.cross(thumb_direction_world, _normalize_axis(hand_width_world, body_basis.lateral_world))
            preferred_local_axis = np.array([1.0, 0.0, 0.0], dtype=float)
            if name.endswith('_r'):
                preferred_local_axis = -preferred_local_axis
            return _choose_axis_for_world_target(
                reference_matrix[:3, :3],
                target_world,
                primary_local_axis=primary,
                allow_primary=False,
                fallback=body_basis.up_world,
                preferred_local_axis=preferred_local_axis,
            )
        return np.array([0.0, 1.0, 0.0], dtype=float)
    if name.endswith('_base_l') or name.endswith('_base_r'):
        return np.array([1.0, 0.0, 0.0], dtype=float)
    if 'thumb' in name or 'index' in name or 'middle' in name or 'ring' in name or 'pinky' in name:
        if hand_width_world is not None:
            preferred_local_axis = np.array([0.0, 0.0, 1.0], dtype=float) if name.startswith('thumb') else np.array([1.0, 0.0, 0.0], dtype=float)
            return _choose_axis_for_world_target(
                reference_matrix[:3, :3],
                hand_width_world,
                primary_local_axis=primary,
                allow_primary=True,
                fallback=body_basis.lateral_world,
                preferred_local_axis=preferred_local_axis,
            )
        return np.array([0.0, 0.0, 1.0], dtype=float)
    if semantic_name.endswith('elbow_joint'):
        exact_local_axis = _choose_axis_for_world_target(
            reference_matrix[:3, :3],
            body_basis.forward_world,
            primary_local_axis=primary,
            allow_primary=False,
            fallback=body_basis.forward_world,
        )
        return _nearest_cardinal_axis(exact_local_axis, [1.0, 0.0, 0.0])
    if semantic_name.endswith('hip_roll_joint') or semantic_name.endswith('waist_roll_joint') or semantic_name.endswith('shoulder_lift_joint'):
        return _choose_axis_for_world_target(
            reference_matrix[:3, :3],
            body_basis.forward_world,
            primary_local_axis=primary,
            allow_primary=True,
            fallback=[0.0, 1.0, 0.0],
        )
    if (
        semantic_name.endswith('hip_pitch_joint')
        or semantic_name.endswith('knee_joint')
        or semantic_name.endswith('ankle_pitch_joint')
        or semantic_name.endswith('shoulder_pitch_joint')
        or semantic_name.endswith('elbow_joint')
        or semantic_name.endswith('wrist_pitch_joint')
        or semantic_name.endswith('waist_pitch_joint')
        or semantic_name.endswith('neck_pitch_joint')
        or semantic_name.endswith('toe_joint')
    ):
        return _choose_axis_for_world_target(
            reference_matrix[:3, :3],
            body_basis.lateral_world,
            primary_local_axis=primary,
            allow_primary=True,
            fallback=[1.0, 0.0, 0.0],
        )
    return _choose_axis_for_world_target(
        reference_matrix[:3, :3],
        body_basis.lateral_world,
        primary_local_axis=primary,
        allow_primary=True,
        fallback=[1.0, 0.0, 0.0],
    )


def infer_joint_limits(name: str) -> tuple[float, float]:
    if 'thumb1_' in name:
        return (-1.2, 1.2)
    if 'thumb' in name:
        return (-1.5, 1.0)
    if 'index' in name or 'middle' in name or 'ring' in name or 'pinky' in name:
        if name.endswith('_base_l') or name.endswith('_base_r'):
            return (-0.45, 0.45)
        return (-1.6, 0.0)
    if 'twist' in name:
        return (-1.2, 1.2)
    if 'spine' in name or 'neck' in name or 'head' in name:
        return (-0.7, 0.7)
    if 'foot' in name or 'toes' in name:
        return (-0.6, 0.8)
    return (-1.8, 1.8)


def infer_thickness(name: str, length: float, incoming_length: float) -> float:
    ref = max(length, incoming_length, 0.05)
    if 'thumb' in name or 'index' in name or 'middle' in name or 'ring' in name or 'pinky' in name:
        return float(np.clip(ref * 0.18, 0.006, 0.018))
    if name.startswith('hand_'):
        return float(np.clip(ref * 0.22, 0.012, 0.03))
    if 'forearm' in name or 'arm_' in name:
        return float(np.clip(ref * 0.22, 0.014, 0.04))
    if 'shoulder' in name or 'spine' in name or 'neck' in name or 'head' in name:
        return float(np.clip(ref * 0.25, 0.018, 0.05))
    if 'thigh' in name or 'leg' in name or 'foot' in name or 'toes' in name:
        return float(np.clip(ref * 0.24, 0.016, 0.055))
    return float(np.clip(ref * 0.20, 0.012, 0.035))


def extract_skeleton_records(skel) -> dict:
    joints = [str(item) for item in (skel.GetJointsAttr().Get() or [])]
    rest_transforms = skel.GetRestTransformsAttr().Get() or []
    if not joints or len(joints) != len(rest_transforms):
        raise RuntimeError('Skeleton joints/rest transform count mismatch.')

    link_name_by_path = make_unique_link_names(joints)
    path_to_index = {path: idx for idx, path in enumerate(joints)}
    local_mats = []
    for transform in rest_transforms:
        local_mats.append(np.array([[float(transform[r][c]) for c in range(4)] for r in range(4)], dtype=float).T)

    parents: List[int] = []
    children: List[List[int]] = [[] for _ in joints]
    for path in joints:
        parent_path = '/'.join(path.split('/')[:-1]) if '/' in path else ''
        parent_index = path_to_index.get(parent_path, -1)
        parents.append(parent_index)
    for index, parent_index in enumerate(parents):
        if parent_index >= 0:
            children[parent_index].append(index)

    world_mats = [np.eye(4, dtype=float) for _ in joints]
    for index, local_mat in enumerate(local_mats):
        parent_index = parents[index]
        world_mats[index] = local_mat if parent_index < 0 else world_mats[parent_index] @ local_mat

    records = []
    for index, path in enumerate(joints):
        local_xyz, local_rpy = matrix_to_xyz_rpy(local_mats[index])
        world_xyz, world_rpy = matrix_to_xyz_rpy(world_mats[index])
        incoming_length = 0.0 if parents[index] < 0 else float(np.linalg.norm(local_xyz))
        name = link_name_by_path[path]
        records.append(
            {
                'index': index,
                'path': path,
                'name': name,
                'parent_index': parents[index],
                'parent_path': joints[parents[index]] if parents[index] >= 0 else None,
                'parent_name': link_name_by_path[joints[parents[index]]] if parents[index] >= 0 else None,
                'children': [joints[child] for child in children[index]],
                'child_names': [link_name_by_path[joints[child]] for child in children[index]],
                'local_matrix': local_mats[index],
                'world_matrix': world_mats[index],
                'local_xyz': local_xyz,
                'local_rpy': local_rpy,
                'world_xyz': world_xyz,
                'world_rpy': world_rpy,
                'incoming_length': incoming_length,
            }
        )

    body_basis = infer_body_basis_world(records)
    hand_width_axes = infer_hand_width_axes_world(records, body_basis)
    for record in records:
        child_vectors = [
            np.asarray(records[child_index]['local_xyz'], dtype=float)
            for child_index in children[record['index']]
            if np.linalg.norm(records[child_index]['local_xyz']) > 1e-8
        ]
        if child_vectors:
            primary_local_axis = max(child_vectors, key=lambda vec: float(np.linalg.norm(vec)))
        elif np.linalg.norm(record['local_xyz']) > 1e-8:
            primary_local_axis = np.asarray(record['local_xyz'], dtype=float)
        else:
            primary_local_axis = np.array([0.0, 1.0, 0.0], dtype=float)
        hinge_normal_world = None
        if record['parent_index'] >= 0 and children[record['index']]:
            parent_world_xyz = np.asarray(records[record['parent_index']]['world_xyz'], dtype=float)
            joint_world_xyz = np.asarray(record['world_xyz'], dtype=float)
            dominant_child_index = max(
                children[record['index']],
                key=lambda child_index: float(np.linalg.norm(records[child_index]['local_xyz'])),
            )
            child_world_xyz = np.asarray(records[dominant_child_index]['world_xyz'], dtype=float)
            incoming_world = joint_world_xyz - parent_world_xyz
            outgoing_world = child_world_xyz - joint_world_xyz
            hinge_candidate = np.cross(incoming_world, outgoing_world)
            if np.linalg.norm(hinge_candidate) > 1e-8:
                hinge_normal_world = hinge_candidate
        record['axis'] = infer_joint_axis(
            record['name'],
            record['local_matrix'],
            primary_local_axis,
            body_basis,
            world_matrix=record['world_matrix'],
            hand_width_world=hand_width_axes.get(record['name'][-1]) if record['name'].endswith(('_l', '_r')) else None,
            hinge_normal_world=hinge_normal_world,
        )
        record['limits'] = infer_joint_limits(record['name'])

    return {
        'records': records,
        'joint_paths': joints,
        'link_name_by_path': link_name_by_path,
        'skeleton_path': str(skel.GetPrim().GetPath()),
    }


def _geom_box(origin_xyz: Sequence[float], origin_rpy: Sequence[float], size_xyz: Sequence[float]) -> dict:
    return {
        'kind': 'box',
        'origin_xyz': [float(v) for v in origin_xyz],
        'origin_rpy': [float(v) for v in origin_rpy],
        'size_xyz': [float(v) for v in size_xyz],
    }


def _geom_sphere(radius: float) -> dict:
    return {
        'kind': 'sphere',
        'origin_xyz': [0.0, 0.0, 0.0],
        'origin_rpy': [0.0, 0.0, 0.0],
        'radius': float(radius),
    }


def _geom_mesh(filename: str) -> dict:
    return {
        'kind': 'mesh',
        'origin_xyz': [0.0, 0.0, 0.0],
        'origin_rpy': [0.0, 0.0, 0.0],
        'filename': str(filename),
    }


def build_link_geometries(records: Sequence[dict]) -> Dict[str, List[dict]]:
    by_name: Dict[str, List[dict]] = {}
    children_by_parent: Dict[int, List[int]] = {record['index']: [] for record in records}
    for record in records:
        if record['parent_index'] >= 0:
            children_by_parent[record['parent_index']].append(record['index'])

    for record in records:
        geoms: List[dict] = []
        child_vectors = [records[child_index]['local_xyz'] for child_index in children_by_parent[record['index']]]
        max_length = max((float(np.linalg.norm(vec)) for vec in child_vectors), default=record['incoming_length'])
        thickness = infer_thickness(record['name'], max_length, record['incoming_length'])
        geoms.append(_geom_sphere(max(thickness * 0.55, 0.005)))
        for child_vec in child_vectors:
            length = float(np.linalg.norm(child_vec))
            if length < 1e-5:
                continue
            rotation = align_x_to_vector(child_vec)
            transform = np.eye(4, dtype=float)
            transform[:3, :3] = rotation
            _, rpy = matrix_to_xyz_rpy(transform)
            geoms.append(_geom_box(np.asarray(child_vec, dtype=float) * 0.5, rpy, [length, thickness, thickness]))
        if len(geoms) == 1 and record['incoming_length'] > 1e-5:
            length = max(record['incoming_length'] * 0.75, thickness)
            geoms.append(_geom_box([length * 0.4, 0.0, 0.0], [0.0, 0.0, 0.0], [length, thickness, thickness]))
        by_name[record['name']] = geoms
    return by_name


def _append_origin(parent, xyz: Sequence[float], rpy: Sequence[float]):
    import xml.etree.ElementTree as ET

    ET.SubElement(
        parent,
        'origin',
        xyz=' '.join(f'{float(v):.6f}' for v in xyz),
        rpy=' '.join(f'{float(v):.6f}' for v in rpy),
    )


def _inertial_from_geoms(geoms: Sequence[dict]) -> tuple[float, tuple[float, float, float]]:
    mass = 0.0
    ixx = iyy = izz = 0.0
    for geom in geoms:
        if geom['kind'] == 'sphere':
            radius = geom['radius']
            volume = 4.0 / 3.0 * math.pi * radius**3
            part_mass = max(volume * 250.0, 0.01)
            inertia = 0.4 * part_mass * radius**2
            mass += part_mass
            ixx += inertia
            iyy += inertia
            izz += inertia
        else:
            lx, ly, lz = geom['size_xyz']
            volume = lx * ly * lz
            part_mass = max(volume * 250.0, 0.01)
            mass += part_mass
            ixx += (part_mass / 12.0) * (ly**2 + lz**2)
            iyy += (part_mass / 12.0) * (lx**2 + lz**2)
            izz += (part_mass / 12.0) * (lx**2 + ly**2)
    mass = max(mass, 0.02)
    return mass, (max(ixx, 1e-5), max(iyy, 1e-5), max(izz, 1e-5))


@dataclass(frozen=True)
class UrdfLinkSpec:
    name: str
    world_matrix: np.ndarray
    render_source_names: tuple[str, ...] = ()
    inertial_source_names: tuple[str, ...] = ()


@dataclass(frozen=True)
class UrdfJointSpec:
    name: str
    parent_link: str
    child_link: str
    joint_type: str
    origin_xyz: np.ndarray
    origin_rpy: np.ndarray
    axis: np.ndarray | None = None
    limits: tuple[float, float] | None = None


def urdf_joint_name_for_link_name(link_name: str) -> str:
    if link_name.startswith('left_hip_roll_link'):
        return 'left_hip_roll_joint'
    if link_name.startswith('right_hip_roll_link'):
        return 'right_hip_roll_joint'
    if link_name == 'spine_01_x':
        return 'waist_yaw_joint'
    if link_name == 'spine_02_x':
        return 'waist_roll_joint'
    if link_name == 'spine_03_x':
        return 'waist_pitch_joint'
    if link_name == 'neck_x':
        return 'neck_pitch_joint'
    if link_name == 'head_x':
        return 'head_yaw_joint'

    side = None
    side_prefix = ''
    if link_name.endswith('_l'):
        side = 'left'
        side_prefix = 'left_'
    elif link_name.endswith('_r'):
        side = 'right'
        side_prefix = 'right_'

    base_name = link_name[:-2] if side is not None else link_name
    side_map = {
        'shoulder': 'shoulder_lift_joint',
        'arm_stretch': 'shoulder_pitch_joint',
        'arm_twist': 'upper_arm_roll_joint',
        'forearm_stretch': 'elbow_joint',
        'forearm_twist': 'forearm_roll_joint',
        'hand': 'wrist_pitch_joint',
        'thigh_stretch': 'hip_pitch_joint',
        'thigh_twist': 'hip_yaw_joint',
        'leg_stretch': 'knee_joint',
        'leg_twist': 'shin_roll_joint',
        'foot': 'ankle_pitch_joint',
        'toes_01': 'toe_joint',
        'thumb1': 'thumb_metacarpal_joint',
        'thumb2': 'thumb_proximal_joint',
        'thumb3': 'thumb_distal_joint',
        'index1_base': 'index_base_joint',
        'index1': 'index_proximal_joint',
        'index2': 'index_intermediate_joint',
        'index3': 'index_distal_joint',
        'middle1_base': 'middle_base_joint',
        'middle1': 'middle_proximal_joint',
        'middle2': 'middle_intermediate_joint',
        'middle3': 'middle_distal_joint',
        'ring1_base': 'ring_base_joint',
        'ring1': 'ring_proximal_joint',
        'ring2': 'ring_intermediate_joint',
        'ring3': 'ring_distal_joint',
        'pinky1_base': 'pinky_base_joint',
        'pinky1': 'pinky_proximal_joint',
        'pinky2': 'pinky_intermediate_joint',
        'pinky3': 'pinky_distal_joint',
    }
    if side is not None and base_name in side_map:
        return f'{side_prefix}{side_map[base_name]}'
    return link_name


def remap_pose_to_urdf_joint_names(
    pose: Dict[str, float],
    available_joint_names: Sequence[str],
) -> Dict[str, float]:
    available = set(available_joint_names)
    remapped: Dict[str, float] = {}
    for joint_name, angle in pose.items():
        target_name = joint_name if joint_name in available else urdf_joint_name_for_link_name(joint_name)
        if target_name not in available:
            continue
        remapped[target_name] = float(angle)
    return remapped


def _raw_link_world_matrix(record_by_name: Dict[str, dict], link_name: str) -> np.ndarray:
    return np.asarray(record_by_name[link_name]['world_matrix'], dtype=float)


def _build_virtual_hip_roll_axis(link_world_matrix: np.ndarray, body_basis: BodyBasis) -> np.ndarray:
    return _choose_axis_for_world_target(
        np.asarray(link_world_matrix, dtype=float)[:3, :3],
        body_basis.forward_world,
        allow_primary=True,
        fallback=[1.0, 0.0, 0.0],
    )


def build_urdf_model(records: Sequence[dict]) -> tuple[dict[str, UrdfLinkSpec], list[UrdfJointSpec]]:
    record_by_name = {record['name']: record for record in records}
    body_basis = infer_body_basis_world(records)
    root_records = [record for record in records if record['parent_index'] < 0]
    if len(root_records) != 1:
        raise RuntimeError(f'Expected exactly one root record, found {len(root_records)}.')

    links: dict[str, UrdfLinkSpec] = {
        record['name']: UrdfLinkSpec(
            name=record['name'],
            world_matrix=np.asarray(record['world_matrix'], dtype=float),
            render_source_names=(record['name'],),
            inertial_source_names=(record['name'],),
        )
        for record in records
    }
    parent_by_child = {
        record['name']: record['parent_name']
        for record in records
        if record['parent_name'] is not None
    }
    joint_name_by_child = {
        record['name']: urdf_joint_name_for_link_name(record['name'])
        for record in records
        if record['parent_name'] is not None
    }
    axis_by_child = {
        record['name']: np.asarray(record['axis'], dtype=float)
        for record in records
        if record['parent_name'] is not None
    }
    limits_by_child = {
        record['name']: tuple(record['limits'])
        for record in records
        if record['parent_name'] is not None
    }

    def _clear_link_geometry(link_name: str) -> None:
        if link_name not in links:
            return
        links[link_name] = replace(
            links[link_name],
            render_source_names=(),
            inertial_source_names=(),
        )

    def _set_link_geometry(link_name: str, *source_names: str) -> None:
        if link_name not in links:
            return
        deduped: list[str] = []
        for source_name in source_names:
            if source_name in record_by_name and source_name not in deduped:
                deduped.append(source_name)
        links[link_name] = replace(
            links[link_name],
            render_source_names=tuple(deduped),
            inertial_source_names=tuple(deduped),
        )

    for side_prefix in ('left', 'right'):
        suffix = 'l' if side_prefix == 'left' else 'r'
        arm_twist = f'arm_twist_{suffix}'
        arm_stretch = f'arm_stretch_{suffix}'
        forearm_stretch = f'forearm_stretch_{suffix}'
        forearm_twist = f'forearm_twist_{suffix}'
        hand = f'hand_{suffix}'
        if arm_stretch in links and arm_twist in links:
            _clear_link_geometry(arm_stretch)
            _set_link_geometry(arm_twist, arm_stretch, arm_twist)
        if arm_twist in links and forearm_stretch in links:
            parent_by_child[forearm_stretch] = arm_twist
        if forearm_stretch in links and forearm_twist in links:
            _set_link_geometry(forearm_stretch, forearm_stretch, forearm_twist)
            _clear_link_geometry(forearm_twist)
        if forearm_twist in links and hand in links:
            parent_by_child[hand] = forearm_twist

        thigh_stretch = f'thigh_stretch_{suffix}'
        thigh_twist = f'thigh_twist_{suffix}'
        leg_stretch = f'leg_stretch_{suffix}'
        leg_twist = f'leg_twist_{suffix}'
        foot = f'foot_{suffix}'
        if thigh_stretch in links and thigh_twist in links:
            _clear_link_geometry(thigh_stretch)
        if thigh_twist in links:
            _set_link_geometry(thigh_twist, thigh_stretch, thigh_twist)
        if thigh_stretch in links and thigh_twist in links and leg_stretch in links:
            virtual_link_name = f'{side_prefix}_hip_roll_link'
            links[virtual_link_name] = UrdfLinkSpec(
                name=virtual_link_name,
                world_matrix=_raw_link_world_matrix(record_by_name, thigh_twist),
                render_source_names=(),
                inertial_source_names=(),
            )
            parent_by_child[virtual_link_name] = thigh_stretch
            joint_name_by_child[virtual_link_name] = urdf_joint_name_for_link_name(virtual_link_name)
            axis_by_child[virtual_link_name] = _build_virtual_hip_roll_axis(
                np.asarray(links[virtual_link_name].world_matrix, dtype=float),
                body_basis,
            )
            limits_by_child[virtual_link_name] = (-0.7, 0.7)
            parent_by_child[thigh_twist] = virtual_link_name
            parent_by_child[leg_stretch] = thigh_twist
        if leg_stretch in links and leg_twist in links:
            _clear_link_geometry(leg_stretch)
        if leg_twist in links:
            _set_link_geometry(leg_twist, leg_stretch, leg_twist)
        if leg_twist in links and foot in links:
            parent_by_child[foot] = leg_twist

    spine_chain = [name for name in ('spine_01_x', 'spine_02_x', 'spine_03_x') if name in links]
    if len(spine_chain) >= 2:
        for link_name in spine_chain[:-1]:
            _clear_link_geometry(link_name)
        _set_link_geometry(spine_chain[-1], *spine_chain)

    joints: list[UrdfJointSpec] = []
    root_record = root_records[0]
    joints.append(
        UrdfJointSpec(
            name=f'{root_record["name"]}_base_fixed',
            parent_link='base_link',
            child_link=root_record['name'],
            joint_type='fixed',
            origin_xyz=np.asarray(root_record['local_xyz'], dtype=float),
            origin_rpy=np.asarray(root_record['local_rpy'], dtype=float),
        )
    )

    for child_name, parent_name in parent_by_child.items():
        parent_world = np.eye(4, dtype=float) if parent_name == 'base_link' else np.asarray(links[parent_name].world_matrix, dtype=float)
        child_world = np.asarray(links[child_name].world_matrix, dtype=float)
        local_matrix = np.linalg.inv(parent_world) @ child_world
        local_xyz, local_rpy = matrix_to_xyz_rpy(local_matrix)
        joints.append(
            UrdfJointSpec(
                name=joint_name_by_child[child_name],
                parent_link=parent_name,
                child_link=child_name,
                joint_type='revolute',
                origin_xyz=local_xyz,
                origin_rpy=local_rpy,
                axis=np.asarray(axis_by_child[child_name], dtype=float),
                limits=limits_by_child[child_name],
            )
        )
    return links, joints


def _matrix_from_origin(xyz: Sequence[float], rpy: Sequence[float]) -> np.ndarray:
    matrix = np.eye(4, dtype=float)
    matrix[:3, :3] = rpy_to_matrix(rpy)
    matrix[:3, 3] = np.asarray(xyz, dtype=float)
    return matrix


def _transform_geom_into_target_frame(geom: dict, source_world_matrix: np.ndarray, target_world_matrix: np.ndarray) -> dict:
    transformed = dict(geom)
    geom_matrix = _matrix_from_origin(geom['origin_xyz'], geom['origin_rpy'])
    relative = np.linalg.inv(np.asarray(target_world_matrix, dtype=float)) @ np.asarray(source_world_matrix, dtype=float) @ geom_matrix
    origin_xyz, origin_rpy = matrix_to_xyz_rpy(relative)
    transformed['origin_xyz'] = [float(v) for v in origin_xyz]
    transformed['origin_rpy'] = [float(v) for v in origin_rpy]
    return transformed


def _collect_link_geometries(
    geoms_by_name: Dict[str, List[dict]],
    record_by_name: Dict[str, dict],
    link_spec: UrdfLinkSpec,
    source_names: Sequence[str],
) -> List[dict]:
    combined: List[dict] = []
    for source_name in source_names:
        record = record_by_name.get(source_name)
        if record is None:
            continue
        source_world = np.asarray(record['world_matrix'], dtype=float)
        for geom in geoms_by_name.get(source_name, []):
            combined.append(
                _transform_geom_into_target_frame(
                    geom,
                    source_world_matrix=source_world,
                    target_world_matrix=link_spec.world_matrix,
                )
            )
    return combined


def generate_urdf_text(
    robot_name: str,
    records: Sequence[dict],
    geoms_by_name: Dict[str, List[dict]] | None = None,
    inertial_geoms_by_name: Dict[str, List[dict]] | None = None,
) -> str:
    import xml.etree.ElementTree as ET

    base_link_name = 'base_link'
    link_specs, joint_specs = build_urdf_model(records)
    record_by_name = {record['name']: record for record in records}

    robot = ET.Element('robot', name=robot_name)
    primitive_geoms_by_name = build_link_geometries(records)
    render_geoms_by_name = geoms_by_name or primitive_geoms_by_name
    inertial_geoms_by_name = inertial_geoms_by_name or primitive_geoms_by_name

    base_link = ET.SubElement(robot, 'link', name=base_link_name)
    inertial = ET.SubElement(base_link, 'inertial')
    _append_origin(inertial, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    ET.SubElement(inertial, 'mass', value='0.001000')
    ET.SubElement(
        inertial,
        'inertia',
        ixx='0.000001',
        ixy='0.0',
        ixz='0.0',
        iyy='0.000001',
        iyz='0.0',
        izz='0.000001',
    )

    for link_name in sorted(link_specs):
        link_spec = link_specs[link_name]
        link = ET.SubElement(robot, 'link', name=link_name)
        inertial_geoms = _collect_link_geometries(
            inertial_geoms_by_name,
            record_by_name,
            link_spec,
            link_spec.inertial_source_names,
        )
        mass, inertia = _inertial_from_geoms(inertial_geoms)
        inertial = ET.SubElement(link, 'inertial')
        _append_origin(inertial, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
        ET.SubElement(inertial, 'mass', value=f'{mass:.6f}')
        ET.SubElement(
            inertial,
            'inertia',
            ixx=f'{inertia[0]:.6f}',
            ixy='0.0',
            ixz='0.0',
            iyy=f'{inertia[1]:.6f}',
            iyz='0.0',
            izz=f'{inertia[2]:.6f}',
        )

        render_geoms = _collect_link_geometries(
            render_geoms_by_name,
            record_by_name,
            link_spec,
            link_spec.render_source_names,
        )
        for tag in ('visual', 'collision'):
            for geom in render_geoms:
                elem = ET.SubElement(link, tag)
                _append_origin(elem, geom['origin_xyz'], geom['origin_rpy'])
                geometry = ET.SubElement(elem, 'geometry')
                if geom['kind'] == 'sphere':
                    ET.SubElement(geometry, 'sphere', radius=f"{geom['radius']:.6f}")
                elif geom['kind'] == 'mesh':
                    ET.SubElement(geometry, 'mesh', filename=geom['filename'])
                else:
                    ET.SubElement(
                        geometry,
                        'box',
                        size=' '.join(f'{float(v):.6f}' for v in geom['size_xyz']),
                    )

    for joint_spec in joint_specs:
        joint = ET.SubElement(robot, 'joint', name=joint_spec.name, type=joint_spec.joint_type)
        ET.SubElement(joint, 'parent', link=joint_spec.parent_link)
        ET.SubElement(joint, 'child', link=joint_spec.child_link)
        _append_origin(joint, joint_spec.origin_xyz, joint_spec.origin_rpy)
        if joint_spec.axis is not None:
            ET.SubElement(joint, 'axis', xyz=' '.join(f'{float(v):.6f}' for v in joint_spec.axis))
        if joint_spec.limits is not None:
            lower, upper = joint_spec.limits
            ET.SubElement(
                joint,
                'limit',
                lower=f'{lower:.6f}',
                upper=f'{upper:.6f}',
                effort='50.0',
                velocity='4.0',
            )
            ET.SubElement(joint, 'dynamics', damping='1.0', friction='0.05')

    tree = ET.ElementTree(robot)
    try:
        ET.indent(tree, space='  ')
    except AttributeError:
        pass
    xml_bytes = ET.tostring(robot, encoding='utf-8', xml_declaration=True)
    text = xml_bytes.decode('utf-8')
    return text if text.endswith('\n') else text + '\n'


def pose_preset_names() -> List[str]:
    from pose_semantics import pose_preset_names as _pose_preset_names

    return _pose_preset_names()


def build_pose_preset(records: Sequence[dict], preset: str) -> Dict[str, float]:
    from pose_semantics import build_pose_preset as _build_pose_preset

    return _build_pose_preset(records, preset)


def animation_clip_names() -> List[str]:
    from pose_semantics import animation_clip_names as _animation_clip_names

    return _animation_clip_names()


def build_animation_clip(records: Sequence[dict], clip_name: str) -> List[tuple[str, Dict[str, float], float]]:
    from pose_semantics import build_animation_clip as _build_animation_clip

    return _build_animation_clip(records, clip_name)


def interpolate_pose_dict(start_pose: Dict[str, float], end_pose: Dict[str, float], alpha: float) -> Dict[str, float]:
    clamped = float(np.clip(alpha, 0.0, 1.0))
    result: Dict[str, float] = {}
    for joint_name in sorted(set(start_pose) | set(end_pose)):
        start = float(start_pose.get(joint_name, 0.0))
        end = float(end_pose.get(joint_name, 0.0))
        value = (1.0 - clamped) * start + clamped * end
        if abs(value) > 1e-10:
            result[joint_name] = value
    return result


def build_demo_pose(records: Sequence[dict]) -> Dict[str, float]:
    from pose_semantics import build_demo_pose as _build_demo_pose

    return _build_demo_pose(records)


def apply_pose_to_local_matrices(records: Sequence[dict], pose_by_name: Dict[str, float]) -> List[np.ndarray]:
    posed = [record['local_matrix'].copy() for record in records]
    for record in records:
        angle = pose_by_name.get(record['name'])
        if angle is None:
            continue
        rotation_delta = axis_angle_matrix(record['axis'], angle)
        updated = posed[record['index']].copy()
        updated[:3, :3] = _orthonormalize(updated[:3, :3] @ rotation_delta)
        posed[record['index']] = updated
    return posed


def world_matrices_from_local(records: Sequence[dict], local_matrices: Sequence[np.ndarray]) -> List[np.ndarray]:
    world = [np.eye(4, dtype=float) for _ in records]
    for record in records:
        parent_index = record['parent_index']
        world[record['index']] = (
            local_matrices[record['index']]
            if parent_index < 0
            else world[parent_index] @ local_matrices[record['index']]
        )
    return world


def records_to_jsonable(records: Sequence[dict]) -> List[dict]:
    payload = []
    for record in records:
        payload.append(
            {
                'index': record['index'],
                'path': record['path'],
                'name': record['name'],
                'parent_index': record['parent_index'],
                'parent_path': record['parent_path'],
                'parent_name': record['parent_name'],
                'children': list(record['children']),
                'child_names': list(record['child_names']),
                'local_xyz': [float(v) for v in record['local_xyz']],
                'local_rpy': [float(v) for v in record['local_rpy']],
                'world_xyz': [float(v) for v in record['world_xyz']],
                'world_rpy': [float(v) for v in record['world_rpy']],
                'local_matrix': np.asarray(record['local_matrix'], dtype=float).tolist(),
                'world_matrix': np.asarray(record['world_matrix'], dtype=float).tolist(),
                'incoming_length': float(record['incoming_length']),
                'axis': [float(v) for v in record['axis']],
                'limits': [float(record['limits'][0]), float(record['limits'][1])],
            }
        )
    return payload


def load_records_json(path: Path) -> List[dict]:
    payload = json.loads(path.read_text(encoding='utf-8'))
    raw_records = payload.get('records', ())
    if not raw_records:
        raise RuntimeError(f'Skeleton JSON has no records: {path}')

    records: List[dict] = []
    for raw in raw_records:
        record = dict(raw)
        record['index'] = int(record['index'])
        record['parent_index'] = int(record['parent_index'])
        record['children'] = list(record.get('children', ()))
        record['child_names'] = list(record.get('child_names', ()))
        record['incoming_length'] = float(record.get('incoming_length', 0.0))
        record['axis'] = np.asarray(record.get('axis', (0.0, 0.0, 1.0)), dtype=float)
        limits = record.get('limits', (-math.pi, math.pi))
        record['limits'] = (float(limits[0]), float(limits[1]))
        for key in ('local_xyz', 'local_rpy', 'world_xyz', 'world_rpy'):
            if key in record:
                record[key] = np.asarray(record[key], dtype=float)
        for key in ('local_matrix', 'world_matrix'):
            if key in record:
                record[key] = np.asarray(record[key], dtype=float)
        records.append(record)

    records.sort(key=lambda record: record['index'])
    return records


def write_records_json(path: Path, skeleton_path: str, usd_path: Path, records: Sequence[dict]) -> None:
    payload = {
        'usd_path': str(usd_path),
        'skeleton_path': skeleton_path,
        'joint_count': len(records),
        'records': records_to_jsonable(records),
        'demo_pose_radians': build_demo_pose(records),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding='utf-8')


def root_height_offset(records: Sequence[dict], clearance: float = 0.02) -> float:
    min_z = min(float(record['world_xyz'][2]) for record in records)
    return -min_z + clearance


def root_height_offset_from_world_matrices(world_matrices: Sequence[np.ndarray], clearance: float = 0.02) -> float:
    min_z = min(float(matrix[2, 3]) for matrix in world_matrices)
    return -min_z + clearance


def rotation_error_radians(a: np.ndarray, b: np.ndarray) -> float:
    rotation = _orthonormalize(a[:3, :3]).T @ _orthonormalize(b[:3, :3])
    trace = float(np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0))
    return math.acos(trace)


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding='utf-8')
