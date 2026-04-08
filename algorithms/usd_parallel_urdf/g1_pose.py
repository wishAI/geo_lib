from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from skeleton_common import axis_angle_matrix, rpy_to_matrix

TORSO_YAW_SCALE = 0.15
WRIST_PITCH_SCALE = 0.26
WRIST_YAW_TWIST_SCALE = 0.22
WRIST_YAW_HAND_SCALE = 0.10
DISPLAY_SHOULDER_PITCH_BIAS = -0.70
DISPLAY_SHOULDER_ROLL_BIAS = 0.35
DISPLAY_SHOULDER_YAW_BIAS = 0.18
DISPLAY_ELBOW_PITCH_BIAS = 0.50
MIN_DISPLAY_ELBOW_PITCH = 0.10
SHOULDER_PITCH_ARM_SCALE = -0.38
SHOULDER_PITCH_SHOULDER_SCALE = 0.0
SHOULDER_ROLL_SCALE = 0.20
SHOULDER_YAW_TWIST_SCALE = 0.0
ELBOW_PITCH_STRETCH_SCALE = -0.75
ELBOW_ROLL_TWIST_SCALE = 0.40
THUMB_YAW_BIAS = 0.22
THUMB_YAW_SCALE = 0.28
THUMB_PITCH_SCALE = 0.30
FINGER_FLEX_SCALE = 0.55


@dataclass(frozen=True)
class UrdfJointSpec:
    name: str
    joint_type: str
    parent_link: str
    child_link: str
    origin: np.ndarray
    axis: np.ndarray
    lower: float
    upper: float


def _origin_matrix(joint_el) -> np.ndarray:
    origin_el = joint_el.find('origin')
    xyz = (0.0, 0.0, 0.0)
    rpy = (0.0, 0.0, 0.0)
    if origin_el is not None:
        xyz_text = origin_el.attrib.get('xyz')
        rpy_text = origin_el.attrib.get('rpy')
        if xyz_text:
            xyz = tuple(float(value) for value in xyz_text.split())
        if rpy_text:
            rpy = tuple(float(value) for value in rpy_text.split())
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = rpy_to_matrix(rpy)
    transform[:3, 3] = np.asarray(xyz, dtype=float)
    return transform


def load_urdf_joint_specs(urdf_path: Path) -> dict[str, UrdfJointSpec]:
    root = ET.parse(urdf_path).getroot()
    specs: dict[str, UrdfJointSpec] = {}
    for joint_el in root.findall('joint'):
        joint_name = joint_el.attrib.get('name')
        joint_type = joint_el.attrib.get('type', 'fixed')
        if not joint_name:
            continue
        parent_el = joint_el.find('parent')
        child_el = joint_el.find('child')
        if parent_el is None or child_el is None:
            continue
        axis_el = joint_el.find('axis')
        axis = np.array([0.0, 0.0, 1.0], dtype=float)
        if axis_el is not None and axis_el.attrib.get('xyz'):
            axis = np.asarray([float(value) for value in axis_el.attrib['xyz'].split()], dtype=float)
        limit_el = joint_el.find('limit')
        lower = float(limit_el.attrib.get('lower', str(-math.pi))) if limit_el is not None else -math.pi
        upper = float(limit_el.attrib.get('upper', str(math.pi))) if limit_el is not None else math.pi
        specs[joint_name] = UrdfJointSpec(
            name=joint_name,
            joint_type=joint_type,
            parent_link=parent_el.attrib['link'],
            child_link=child_el.attrib['link'],
            origin=_origin_matrix(joint_el),
            axis=axis,
            lower=lower,
            upper=upper,
        )
    return specs


def _root_links(specs: Sequence[UrdfJointSpec]) -> tuple[str, ...]:
    parent_links = {spec.parent_link for spec in specs}
    child_links = {spec.child_link for spec in specs}
    roots = sorted(parent_links - child_links)
    return tuple(roots) if roots else ('base_link',)


def _joint_motion(spec: UrdfJointSpec, angle_rad: float) -> np.ndarray:
    motion = np.eye(4, dtype=float)
    if spec.joint_type in ('revolute', 'continuous'):
        motion[:3, :3] = axis_angle_matrix(spec.axis, angle_rad)
    elif spec.joint_type == 'prismatic':
        motion[:3, 3] = np.asarray(spec.axis, dtype=float) * float(angle_rad)
    return motion


def world_map_from_urdf_pose(urdf_path: Path, pose_by_name: Mapping[str, float] | None = None) -> dict[str, np.ndarray]:
    pose = pose_by_name or {}
    specs = tuple(load_urdf_joint_specs(urdf_path).values())
    roots = _root_links(specs)
    world = {link_name: np.eye(4, dtype=float) for link_name in roots}
    pending = {spec.name: spec for spec in specs}
    while pending:
        progressed = False
        for joint_name in list(pending):
            spec = pending[joint_name]
            if spec.parent_link not in world:
                continue
            world[spec.child_link] = world[spec.parent_link] @ spec.origin @ _joint_motion(spec, float(pose.get(joint_name, 0.0)))
            pending.pop(joint_name)
            progressed = True
        if not progressed:
            unresolved = ', '.join(sorted(pending))
            raise RuntimeError(f'Unable to resolve URDF joint world transforms for {urdf_path}: {unresolved}')
    return world


def estimate_urdf_root_height(urdf_path: Path, pose_by_name: Mapping[str, float] | None = None, clearance: float = 0.02) -> float:
    world = world_map_from_urdf_pose(urdf_path, pose_by_name)
    min_z = min(float(matrix[2, 3]) for matrix in world.values())
    return -min_z + clearance


def _pose_value(pose_by_name: Mapping[str, float], joint_name: str) -> float:
    return float(pose_by_name.get(joint_name, 0.0))


def _chain_average(pose_by_name: Mapping[str, float], joint_names: Sequence[str], weights: Sequence[float] | None = None) -> float:
    if not joint_names:
        return 0.0
    if weights is None:
        weights = tuple(1.0 for _ in joint_names)
    total_weight = max(sum(float(weight) for weight in weights), 1.0e-6)
    return sum(float(weight) * _pose_value(pose_by_name, joint_name) for joint_name, weight in zip(joint_names, weights, strict=False)) / total_weight


def _clamp_pose(pose_by_name: dict[str, float], joint_limits: Mapping[str, tuple[float, float]] | None) -> dict[str, float]:
    if joint_limits is None:
        return {joint_name: float(value) for joint_name, value in pose_by_name.items()}
    clipped = {}
    for joint_name, value in pose_by_name.items():
        limits = joint_limits.get(joint_name)
        if limits is None:
            clipped[joint_name] = float(value)
            continue
        lower, upper = limits
        clipped[joint_name] = float(np.clip(float(value), float(lower), float(upper)))
    return clipped


def load_joint_limits(urdf_path: Path) -> dict[str, tuple[float, float]]:
    return {
        spec.name: (spec.lower, spec.upper)
        for spec in load_urdf_joint_specs(urdf_path).values()
    }


def _lateral_sign(side_name: str) -> float:
    return 1.0 if side_name == 'left' else -1.0


def _robot_hand_prefix(side_name: str) -> str:
    return 'L' if side_name == 'left' else 'R'


def map_landau_pose_to_g1_pose(
    landau_pose: Mapping[str, float],
    *,
    joint_limits: Mapping[str, tuple[float, float]] | None = None,
) -> dict[str, float]:
    g1_pose: dict[str, float] = {}
    g1_pose['torso_joint'] = TORSO_YAW_SCALE * (
        _pose_value(landau_pose, 'shoulder_l') - _pose_value(landau_pose, 'shoulder_r')
    )

    for side_name, suffix in (('left', 'l'), ('right', 'r')):
        prefix = f'{side_name}_'
        hand_prefix = _robot_hand_prefix(side_name)
        lateral_sign = _lateral_sign(side_name)
        shoulder = _pose_value(landau_pose, f'shoulder_{suffix}')
        arm = _pose_value(landau_pose, f'arm_stretch_{suffix}')
        elbow = _pose_value(landau_pose, f'forearm_stretch_{suffix}')
        hand = _pose_value(landau_pose, f'hand_{suffix}')
        arm_twist = _pose_value(landau_pose, f'arm_twist_{suffix}')
        forearm_twist = _pose_value(landau_pose, f'forearm_twist_{suffix}')

        g1_pose[f'{prefix}shoulder_pitch_joint'] = (
            DISPLAY_SHOULDER_PITCH_BIAS
            + SHOULDER_PITCH_ARM_SCALE * arm
            + SHOULDER_PITCH_SHOULDER_SCALE * shoulder
        )
        g1_pose[f'{prefix}shoulder_roll_joint'] = lateral_sign * (
            DISPLAY_SHOULDER_ROLL_BIAS + SHOULDER_ROLL_SCALE * shoulder
        )
        g1_pose[f'{prefix}shoulder_yaw_joint'] = -lateral_sign * (
            DISPLAY_SHOULDER_YAW_BIAS + SHOULDER_YAW_TWIST_SCALE * arm_twist
        )
        g1_pose[f'{prefix}elbow_pitch_joint'] = max(
            MIN_DISPLAY_ELBOW_PITCH,
            DISPLAY_ELBOW_PITCH_BIAS + ELBOW_PITCH_STRETCH_SCALE * elbow,
        )
        g1_pose[f'{prefix}elbow_roll_joint'] = ELBOW_ROLL_TWIST_SCALE * forearm_twist
        g1_pose[f'{prefix}wrist_pitch_joint'] = -WRIST_PITCH_SCALE * hand
        g1_pose[f'{prefix}wrist_yaw_joint'] = WRIST_YAW_TWIST_SCALE * arm_twist + WRIST_YAW_HAND_SCALE * hand

        g1_pose[f'{hand_prefix}_thumb_proximal_yaw_joint'] = THUMB_YAW_BIAS + THUMB_YAW_SCALE * _pose_value(
            landau_pose,
            f'thumb1_{suffix}',
        )
        g1_pose[f'{hand_prefix}_thumb_proximal_pitch_joint'] = THUMB_PITCH_SCALE * _chain_average(
            landau_pose,
            (f'thumb2_{suffix}', f'thumb3_{suffix}'),
            weights=(0.45, 0.55),
        )
        g1_pose[f'{hand_prefix}_index_proximal_joint'] = FINGER_FLEX_SCALE * _chain_average(
            landau_pose,
            (f'index1_{suffix}', f'index2_{suffix}', f'index3_{suffix}'),
            weights=(0.45, 0.33, 0.22),
        )
        g1_pose[f'{hand_prefix}_middle_proximal_joint'] = FINGER_FLEX_SCALE * _chain_average(
            landau_pose,
            (f'middle1_{suffix}', f'middle2_{suffix}', f'middle3_{suffix}'),
            weights=(0.45, 0.33, 0.22),
        )
        g1_pose[f'{hand_prefix}_ring_proximal_joint'] = FINGER_FLEX_SCALE * _chain_average(
            landau_pose,
            (f'ring1_{suffix}', f'ring2_{suffix}', f'ring3_{suffix}'),
            weights=(0.45, 0.33, 0.22),
        )
        g1_pose[f'{hand_prefix}_pinky_proximal_joint'] = FINGER_FLEX_SCALE * _chain_average(
            landau_pose,
            (f'pinky1_{suffix}', f'pinky2_{suffix}', f'pinky3_{suffix}'),
            weights=(0.45, 0.33, 0.22),
        )

    return _clamp_pose(g1_pose, joint_limits)


def g1_hand_pose_summary(g1_pose: Mapping[str, float], side: str) -> dict[str, tuple[float, ...]]:
    prefix = _robot_hand_prefix(side)
    return {
        'thumb': (
            float(g1_pose.get(f'{prefix}_thumb_proximal_yaw_joint', 0.0)),
            float(g1_pose.get(f'{prefix}_thumb_proximal_pitch_joint', 0.0)),
        ),
        'index': (float(g1_pose.get(f'{prefix}_index_proximal_joint', 0.0)),),
        'middle': (float(g1_pose.get(f'{prefix}_middle_proximal_joint', 0.0)),),
        'ring': (float(g1_pose.get(f'{prefix}_ring_proximal_joint', 0.0)),),
        'pinky': (float(g1_pose.get(f'{prefix}_pinky_proximal_joint', 0.0)),),
    }
