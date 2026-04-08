from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from skeleton_common import apply_pose_to_local_matrices, load_records_json, world_matrices_from_local


HAND_JOINT_NAMES = (
    'wrist',
    'thumbKnuckle',
    'thumbIntermediateBase',
    'thumbIntermediateTip',
    'thumbTip',
    'indexMetacarpal',
    'indexKnuckle',
    'indexIntermediateBase',
    'indexIntermediateTip',
    'indexTip',
    'middleMetacarpal',
    'middleKnuckle',
    'middleIntermediateBase',
    'middleIntermediateTip',
    'middleTip',
    'ringMetacarpal',
    'ringKnuckle',
    'ringIntermediateBase',
    'ringIntermediateTip',
    'ringTip',
    'littleMetacarpal',
    'littleKnuckle',
    'littleIntermediateBase',
    'littleIntermediateTip',
    'littleTip',
    'forearmWrist',
    'forearmArm',
)

HAND_JOINT_INDEX = {name: index for index, name in enumerate(HAND_JOINT_NAMES)}

ARM_BASE_LINK = 'spine_03_x'
HEAD_LINK = 'head_x'
LEFT_ARM_TIP = 'hand_l'
RIGHT_ARM_TIP = 'hand_r'
LEFT_ARM_CHAIN = ('shoulder_l', 'arm_stretch_l', 'forearm_stretch_l', 'hand_l')
RIGHT_ARM_CHAIN = ('shoulder_r', 'arm_stretch_r', 'forearm_stretch_r', 'hand_r')

FINGER_LAYOUT = {
    'thumb': {
        'avp': ('thumbKnuckle', 'thumbIntermediateBase', 'thumbIntermediateTip', 'thumbTip'),
        'robot': ('thumb1', 'thumb2', 'thumb3'),
    },
    'index': {
        'avp': ('indexMetacarpal', 'indexKnuckle', 'indexIntermediateBase', 'indexIntermediateTip', 'indexTip'),
        'robot': ('index1_base', 'index1', 'index2', 'index3'),
    },
    'middle': {
        'avp': ('middleMetacarpal', 'middleKnuckle', 'middleIntermediateBase', 'middleIntermediateTip', 'middleTip'),
        'robot': ('middle1_base', 'middle1', 'middle2', 'middle3'),
    },
    'ring': {
        'avp': ('ringMetacarpal', 'ringKnuckle', 'ringIntermediateBase', 'ringIntermediateTip', 'ringTip'),
        'robot': ('ring1_base', 'ring1', 'ring2', 'ring3'),
    },
    'little': {
        'avp': ('littleMetacarpal', 'littleKnuckle', 'littleIntermediateBase', 'littleIntermediateTip', 'littleTip'),
        'robot': ('pinky1_base', 'pinky1', 'pinky2', 'pinky3'),
    },
}


@dataclass(frozen=True)
class TransformOptions:
    column_major: bool = True
    pretransform: np.ndarray | None = None
    posttransform: np.ndarray | None = None


@dataclass(frozen=True)
class UrdfJointSpec:
    name: str
    parent_link: str
    child_link: str
    lower: float
    upper: float


@dataclass(frozen=True)
class HandCalibration:
    scale: float
    rotation_offset: np.ndarray


def _rot_x_deg(deg: float) -> np.ndarray:
    rad = math.radians(float(deg))
    c = math.cos(rad)
    s = math.sin(rad)
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, c, -s],
            [0.0, s, c],
        ],
        dtype=float,
    )


def _rot_y_deg(deg: float) -> np.ndarray:
    rad = math.radians(float(deg))
    c = math.cos(rad)
    s = math.sin(rad)
    return np.array(
        [
            [c, 0.0, s],
            [0.0, 1.0, 0.0],
            [-s, 0.0, c],
        ],
        dtype=float,
    )


def _rot_z_deg(deg: float) -> np.ndarray:
    rad = math.radians(float(deg))
    c = math.cos(rad)
    s = math.sin(rad)
    return np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def build_xyz_transform(
    rotate_xyz_deg: Sequence[float],
    translate_m: Sequence[float],
    scale_xyz: Sequence[float] = (1.0, 1.0, 1.0),
) -> np.ndarray:
    x_deg, y_deg, z_deg = rotate_xyz_deg
    tx, ty, tz = translate_m
    sx, sy, sz = scale_xyz
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = _rot_x_deg(x_deg) @ _rot_y_deg(y_deg) @ _rot_z_deg(z_deg) @ np.diag([float(sx), float(sy), float(sz)])
    transform[3, :3] = [float(tx), float(ty), float(tz)]
    return transform


AVP_TO_SCENE_OPTIONS = TransformOptions(
    column_major=False,
    pretransform=build_xyz_transform(
        (0.0, 0.0, 180.0),
        (0.0, -0.13, 0.13),
        scale_xyz=(0.6, 0.6, 0.6),
    ).T,
    posttransform=None,
)


def to_usd_world(mat, options: TransformOptions) -> np.ndarray | None:
    if mat is None:
        return None
    matrix = np.asarray(mat, dtype=float)
    if matrix.shape != (4, 4):
        raise ValueError(f'Expected 4x4 matrix, got shape {matrix.shape}')
    if options.column_major:
        matrix = matrix.T
    if options.pretransform is not None:
        matrix = np.asarray(options.pretransform, dtype=float) @ matrix
    if options.posttransform is not None:
        matrix = matrix @ np.asarray(options.posttransform, dtype=float)
    return matrix


def _normalize(vec: np.ndarray) -> np.ndarray:
    arr = np.asarray(vec, dtype=float)
    norm = float(np.linalg.norm(arr))
    if norm < 1.0e-8:
        return np.zeros_like(arr)
    return arr / norm


def _orthonormalize(rotation: np.ndarray) -> np.ndarray:
    u, _, vh = np.linalg.svd(np.asarray(rotation, dtype=float))
    fixed = u @ vh
    if float(np.linalg.det(fixed)) < 0.0:
        u[:, -1] *= -1.0
        fixed = u @ vh
    return fixed


def _basis_from_vectors(lateral: np.ndarray, forward: np.ndarray) -> np.ndarray:
    x_axis = _normalize(lateral)
    if not np.any(x_axis):
        return np.eye(3, dtype=float)
    y_seed = np.asarray(forward, dtype=float) - x_axis * float(np.dot(x_axis, forward))
    y_axis = _normalize(y_seed)
    if not np.any(y_axis):
        return np.eye(3, dtype=float)
    z_axis = _normalize(np.cross(x_axis, y_axis))
    if not np.any(z_axis):
        return np.eye(3, dtype=float)
    y_axis = _normalize(np.cross(z_axis, x_axis))
    if not np.any(y_axis):
        return np.eye(3, dtype=float)
    return _orthonormalize(np.column_stack([x_axis, y_axis, z_axis]))


def _angle_between(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    a = _normalize(vec_a)
    b = _normalize(vec_b)
    if not np.any(a) or not np.any(b):
        return 0.0
    dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
    return float(np.arccos(dot))


def _side_suffix(side: str) -> str:
    return 'l' if side == 'left' else 'r'


def _get_value(data, key, default=None):
    if data is None:
        return default
    if hasattr(data, key):
        try:
            return getattr(data, key)
        except Exception:
            pass
    if isinstance(data, dict):
        return data.get(key, default)
    getter = getattr(data, 'get', None)
    if callable(getter):
        try:
            return getter(key, default)
        except Exception:
            try:
                return getter(key)
            except Exception:
                pass
    return default


def _as_mat4(value) -> np.ndarray | None:
    if value is None:
        return None
    try:
        mat = np.asarray(value, dtype=float)
    except Exception:
        return None
    if mat.shape == (1, 4, 4):
        mat = mat[0]
    if mat.shape != (4, 4):
        return None
    return mat


def _as_mat4_stack(value, expected_count: int | None = None) -> np.ndarray | None:
    if value is None:
        return None
    try:
        mats = np.asarray(value, dtype=float)
    except Exception:
        return None
    if mats.ndim == 4 and mats.shape[0] == 1:
        mats = mats[0]
    if mats.ndim != 3 or mats.shape[1:] != (4, 4):
        return None
    if expected_count is not None and mats.shape[0] != expected_count:
        return None
    return mats


def _as_float(value) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _extract_hand_stack_from_attributes(hand) -> np.ndarray | None:
    if hand is None:
        return None
    joints = []
    for index, joint_name in enumerate(HAND_JOINT_NAMES):
        joint = getattr(hand, joint_name, None)
        if joint is None:
            try:
                joint = hand[index]
            except Exception:
                return None
        mat = _as_mat4(joint)
        if mat is None:
            return None
        joints.append(mat)
    return np.stack(joints, axis=0)


def extract_hand_stack(data, side: str) -> np.ndarray | None:
    hand_obj = _get_value(data, side)
    mats = _extract_hand_stack_from_attributes(hand_obj)
    if mats is not None:
        return mats
    return _as_mat4_stack(_get_value(data, f'{side}_arm'), expected_count=len(HAND_JOINT_NAMES))


def extract_tracking_frame(data) -> dict[str, object]:
    left_arm = extract_hand_stack(data, 'left')
    right_arm = extract_hand_stack(data, 'right')
    left_hand = _get_value(data, 'left')
    right_hand = _get_value(data, 'right')
    frame = {
        'head': _as_mat4(_get_value(data, 'head')),
        'left_arm': left_arm,
        'right_arm': right_arm,
        'left_wrist': _as_mat4(_get_value(data, 'left_wrist')),
        'right_wrist': _as_mat4(_get_value(data, 'right_wrist')),
        'left_pinch_distance': _as_float(_get_value(data, 'left_pinch_distance', _as_float(_get_value(left_hand, 'pinch_distance')))),
        'right_pinch_distance': _as_float(_get_value(data, 'right_pinch_distance', _as_float(_get_value(right_hand, 'pinch_distance')))),
        'left_wrist_roll': _as_float(_get_value(data, 'left_wrist_roll', _as_float(_get_value(left_hand, 'wrist_roll')))),
        'right_wrist_roll': _as_float(_get_value(data, 'right_wrist_roll', _as_float(_get_value(right_hand, 'wrist_roll')))),
    }
    if frame['left_wrist'] is None and left_arm is not None:
        frame['left_wrist'] = left_arm[0]
    if frame['right_wrist'] is None and right_arm is not None:
        frame['right_wrist'] = right_arm[0]
    return frame


def load_tracking_frame(snapshot_path: Path) -> dict[str, object]:
    payload = json.loads(Path(snapshot_path).read_text(encoding='utf-8'))
    if not isinstance(payload, dict):
        raise RuntimeError(f'AVP snapshot must be a JSON object: {snapshot_path}')
    return extract_tracking_frame(payload)


def _tracking_matrix_world(mat) -> np.ndarray | None:
    if mat is None:
        return None
    return np.asarray(to_usd_world(mat, options=AVP_TO_SCENE_OPTIONS), dtype=float)


def _tracking_stack_world(stack) -> np.ndarray | None:
    if stack is None:
        return None
    return np.stack([_tracking_matrix_world(mat) for mat in stack], axis=0)


def load_urdf_joint_specs(urdf_path: Path) -> dict[str, UrdfJointSpec]:
    root = ET.parse(urdf_path).getroot()
    specs: dict[str, UrdfJointSpec] = {}
    for joint_el in root.findall('joint'):
        joint_name = joint_el.attrib.get('name')
        if not joint_name:
            continue
        parent_el = joint_el.find('parent')
        child_el = joint_el.find('child')
        limit_el = joint_el.find('limit')
        lower = float(limit_el.attrib.get('lower', str(-math.pi))) if limit_el is not None else -math.pi
        upper = float(limit_el.attrib.get('upper', str(math.pi))) if limit_el is not None else math.pi
        specs[joint_name] = UrdfJointSpec(
            name=joint_name,
            parent_link=parent_el.attrib['link'],
            child_link=child_el.attrib['link'],
            lower=lower,
            upper=upper,
        )
    return specs


def find_joint_chain(urdf_path: Path, base_link: str, tip_link: str) -> tuple[str, ...]:
    specs = load_urdf_joint_specs(urdf_path)
    by_child = {spec.child_link: spec for spec in specs.values()}
    current_link = tip_link
    chain: list[str] = []
    while current_link != base_link:
        spec = by_child.get(current_link)
        if spec is None:
            raise KeyError(f'Could not find joint chain from {base_link} to {tip_link}')
        chain.append(spec.name)
        current_link = spec.parent_link
    chain.reverse()
    return tuple(chain)


def landau_hand_pose_summary(pose_by_name: dict[str, float], side: str) -> dict[str, tuple[float, ...]]:
    suffix = _side_suffix(side)
    return {
        chain_key: tuple(float(pose_by_name.get(f'{joint_name}_{suffix}', 0.0)) for joint_name in chain['robot'])
        for chain_key, chain in FINGER_LAYOUT.items()
    }


class AvpLandauRetargeter:
    def __init__(
        self,
        *,
        urdf_path: Path,
        skeleton_json_path: Path,
        snapshot_path: Path,
    ) -> None:
        self.urdf_path = Path(urdf_path).resolve()
        self.skeleton_json_path = Path(skeleton_json_path).resolve()
        self.snapshot_path = Path(snapshot_path).expanduser().resolve()

        self.records = load_records_json(self.skeleton_json_path)
        self.records_by_name = {record['name']: record for record in self.records}
        self.rest_local = [np.asarray(record['local_matrix'], dtype=float) for record in self.records]
        self.rest_world = world_matrices_from_local(self.records, self.rest_local)
        self.rest_world_by_name = {
            record['name']: self.rest_world[record['index']]
            for record in self.records
        }
        self.base_world = self.rest_world_by_name[ARM_BASE_LINK]
        self.base_world_inv = np.linalg.inv(self.base_world)
        self.head_world = self.rest_world_by_name[HEAD_LINK]

        self.rest_hand_base_rotation = {
            side: self.base_world_inv @ self.rest_world_by_name[f'hand_{_side_suffix(side)}']
            for side in ('left', 'right')
        }
        self.rest_hand_base_radius = {
            side: float(np.linalg.norm(self.rest_hand_base_rotation[side][:3, 3]))
            for side in ('left', 'right')
        }
        self.rest_hand_local_positions = {
            side: self._build_hand_local_rest_positions(side)
            for side in ('left', 'right')
        }
        self.robot_hand_basis = {
            side: self._build_robot_hand_basis(side)
            for side in ('left', 'right')
        }

        self.joint_specs = load_urdf_joint_specs(self.urdf_path)
        self.left_chain = find_joint_chain(self.urdf_path, ARM_BASE_LINK, LEFT_ARM_TIP)
        self.right_chain = find_joint_chain(self.urdf_path, ARM_BASE_LINK, RIGHT_ARM_TIP)
        self.left_seed = np.zeros(len(self.left_chain), dtype=float)
        self.right_seed = np.zeros(len(self.right_chain), dtype=float)

        self.snapshot_frame = load_tracking_frame(self.snapshot_path)
        snapshot_head_world = _tracking_matrix_world(self.snapshot_frame.get('head'))
        if snapshot_head_world is None:
            head_frame_alignment = np.eye(3, dtype=float)
        else:
            head_frame_alignment = _orthonormalize(self.head_world[:3, :3]) @ _orthonormalize(snapshot_head_world[:3, :3]).T
        self.calibration = {
            'left': self._build_hand_calibration('left', self.snapshot_frame, head_frame_alignment),
            'right': self._build_hand_calibration('right', self.snapshot_frame, head_frame_alignment),
        }

    def _build_hand_calibration(self, side: str, frame, head_frame_alignment: np.ndarray) -> HandCalibration:
        head_world = _tracking_matrix_world(frame.get('head'))
        wrist_world = _tracking_matrix_world(frame.get(f'{side}_wrist'))
        if head_world is None or wrist_world is None:
            return HandCalibration(scale=1.0, rotation_offset=head_frame_alignment.copy())

        avp_hand_rel_head = np.linalg.inv(head_world) @ wrist_world
        robot_hand_rel_head = np.linalg.inv(self.head_world) @ self.rest_world_by_name[f'hand_{_side_suffix(side)}']
        avp_length = float(np.linalg.norm(avp_hand_rel_head[:3, 3]))
        robot_length = float(np.linalg.norm(robot_hand_rel_head[:3, 3]))
        scale = robot_length / max(avp_length, 1.0e-6)
        return HandCalibration(scale=scale, rotation_offset=head_frame_alignment.copy())

    def _build_hand_local_rest_positions(self, side: str) -> dict[str, np.ndarray]:
        suffix = _side_suffix(side)
        hand_name = f'hand_{suffix}'
        hand_world_inv = np.linalg.inv(self.rest_world_by_name[hand_name])
        local_positions = {hand_name: np.zeros(3, dtype=float)}
        for record in self.records:
            name = record['name']
            if not name.endswith(f'_{suffix}') or name == hand_name:
                continue
            local_positions[name] = (hand_world_inv @ self.rest_world_by_name[name])[:3, 3].copy()
        return local_positions

    def _build_robot_hand_basis(self, side: str) -> np.ndarray:
        suffix = _side_suffix(side)
        local_positions = self.rest_hand_local_positions[side]
        lateral = local_positions[f'index1_base_{suffix}'] - local_positions[f'pinky1_base_{suffix}']
        forward = (
            local_positions[f'index1_{suffix}']
            + local_positions[f'middle1_{suffix}']
            + local_positions[f'ring1_{suffix}']
        ) / 3.0
        return _basis_from_vectors(lateral, forward)

    def _base_relative_world_map(self, pose_by_name: dict[str, float]) -> dict[str, np.ndarray]:
        local_matrices = apply_pose_to_local_matrices(self.records, pose_by_name)
        world_matrices = world_matrices_from_local(self.records, local_matrices)
        return {
            record['name']: self.base_world_inv @ world_matrices[record['index']]
            for record in self.records
        }

    def _clamp_joint(self, joint_name: str, value: float) -> float:
        spec = self.joint_specs.get(joint_name)
        if spec is None:
            return float(value)
        return float(np.clip(float(value), spec.lower, spec.upper))

    def _retarget_head(self, frame, pose_by_name: dict[str, float]) -> None:
        head_world = _tracking_matrix_world(frame.get('head'))
        if head_world is None:
            return
        forward_axis = _normalize(head_world[:3, 1])
        forward_xy = math.sqrt(float(forward_axis[0] ** 2 + forward_axis[1] ** 2))
        pitch = math.atan2(float(forward_axis[2]), max(forward_xy, 1.0e-6))
        pose_by_name['neck_x'] = self._clamp_joint('neck_x', 0.55 * pitch)
        pose_by_name['head_x'] = self._clamp_joint('head_x', 0.45 * pitch)

    def _clamp_hand_base_position(self, side: str, target_pos: np.ndarray) -> np.ndarray:
        clipped = np.asarray(target_pos, dtype=float).copy()
        target_radius = float(np.linalg.norm(clipped))
        radius_cap = 1.15 * self.rest_hand_base_radius[side]
        if target_radius > radius_cap and target_radius > 1.0e-8:
            clipped *= radius_cap / target_radius
        return clipped

    def _target_hand_base_position(self, side: str, frame) -> np.ndarray | None:
        wrist_world = _tracking_matrix_world(frame.get(f'{side}_wrist'))
        if wrist_world is None:
            return None
        target_base = self.base_world_inv @ wrist_world
        return self._clamp_hand_base_position(side, target_base[:3, 3])

    def _solve_arm(self, side: str, frame, pose_by_name: dict[str, float]) -> None:
        target_pos = self._target_hand_base_position(side, frame)
        if target_pos is None:
            return

        seed = self.left_seed if side == 'left' else self.right_seed
        solution = self._solve_arm_ccd(side, target_pos, seed)
        if side == 'left':
            self.left_seed = np.asarray(solution, dtype=float)
        else:
            self.right_seed = np.asarray(solution, dtype=float)

        chain = self.left_chain if side == 'left' else self.right_chain
        for joint_name, joint_value in zip(chain, solution, strict=False):
            pose_by_name[joint_name] = self._clamp_joint(joint_name, float(joint_value))

        suffix = _side_suffix(side)
        wrist_roll = frame.get(f'{side}_wrist_roll')
        if wrist_roll is not None:
            pose_by_name[f'arm_twist_{suffix}'] = self._clamp_joint(f'arm_twist_{suffix}', 0.35 * float(wrist_roll))
            pose_by_name[f'forearm_twist_{suffix}'] = self._clamp_joint(
                f'forearm_twist_{suffix}',
                0.65 * float(wrist_roll),
            )

    def _solve_arm_ccd(self, side: str, target_pos: np.ndarray, seed: np.ndarray) -> np.ndarray:
        chain = self.left_chain if side == 'left' else self.right_chain
        hand_name = f'hand_{_side_suffix(side)}'
        pose = {
            joint_name: float(joint_value)
            for joint_name, joint_value in zip(chain, seed, strict=False)
        }

        for _ in range(80):
            base_relative_map = self._base_relative_world_map(pose)
            current_pos = base_relative_map[hand_name][:3, 3]
            if float(np.linalg.norm(target_pos - current_pos)) <= 1.0e-3:
                break

            for joint_name in reversed(chain):
                base_relative_map = self._base_relative_world_map(pose)
                current_pos = base_relative_map[hand_name][:3, 3]
                joint_transform = base_relative_map[joint_name]
                joint_pos = joint_transform[:3, 3]
                axis_world = _normalize(joint_transform[:3, :3] @ np.asarray(self.records_by_name[joint_name]['axis'], dtype=float))
                if not np.any(axis_world):
                    continue

                current_vec = current_pos - joint_pos
                target_vec = target_pos - joint_pos
                current_proj = current_vec - axis_world * float(np.dot(axis_world, current_vec))
                target_proj = target_vec - axis_world * float(np.dot(axis_world, target_vec))
                if float(np.linalg.norm(current_proj)) < 1.0e-8 or float(np.linalg.norm(target_proj)) < 1.0e-8:
                    continue

                current_proj = _normalize(current_proj)
                target_proj = _normalize(target_proj)
                signed_angle = math.atan2(
                    float(np.dot(axis_world, np.cross(current_proj, target_proj))),
                    float(np.clip(np.dot(current_proj, target_proj), -1.0, 1.0)),
                )
                pose[joint_name] = self._clamp_joint(joint_name, pose.get(joint_name, 0.0) + 0.85 * signed_angle)

        return np.asarray([pose.get(joint_name, 0.0) for joint_name in chain], dtype=float)

    def _wrist_local_positions(self, side: str, frame) -> dict[str, np.ndarray] | None:
        stack = _tracking_stack_world(frame.get(f'{side}_arm'))
        if stack is None or len(stack) != len(HAND_JOINT_NAMES):
            return None
        wrist_inv = np.linalg.inv(stack[HAND_JOINT_INDEX['wrist']])
        local_positions = {}
        for joint_name, joint_index in HAND_JOINT_INDEX.items():
            local_mat = wrist_inv @ stack[joint_index]
            local_positions[joint_name] = local_mat[:3, 3].copy()
        return local_positions

    def _aligned_hand_local_positions(self, side: str, frame) -> dict[str, np.ndarray] | None:
        local_positions = self._wrist_local_positions(side, frame)
        if local_positions is None:
            return None

        lateral = local_positions['littleMetacarpal'] - local_positions['indexMetacarpal']
        forward = (
            local_positions['indexKnuckle']
            + local_positions['middleKnuckle']
            + local_positions['ringKnuckle']
        ) / 3.0
        avp_basis = _basis_from_vectors(lateral, forward)
        alignment = self.robot_hand_basis[side] @ avp_basis.T
        return {
            joint_name: alignment @ np.asarray(position, dtype=float)
            for joint_name, position in local_positions.items()
        }

    def _apply_thumb(self, suffix: str, local_positions: dict[str, np.ndarray], pose_by_name: dict[str, float]) -> None:
        points = [local_positions[name] for name in FINGER_LAYOUT['thumb']['avp']]
        base_vec = points[0]
        seg_1 = points[1] - points[0]
        seg_2 = points[2] - points[1]
        seg_3 = points[3] - points[2]

        thumb1_name = f'thumb1_{suffix}'
        thumb2_name = f'thumb2_{suffix}'
        thumb3_name = f'thumb3_{suffix}'

        oppose = math.atan2(float(base_vec[0]), max(abs(float(base_vec[1])), 1.0e-6))
        flex_2 = 1.15 * _angle_between(seg_1, seg_2)
        flex_3 = 1.05 * _angle_between(seg_2, seg_3)

        pose_by_name[thumb1_name] = self._clamp_joint(thumb1_name, oppose)
        pose_by_name[thumb2_name] = self._clamp_joint(thumb2_name, flex_2)
        pose_by_name[thumb3_name] = self._clamp_joint(thumb3_name, flex_3)

    def _apply_finger_chain(
        self,
        *,
        suffix: str,
        chain_key: str,
        local_positions: dict[str, np.ndarray],
        pose_by_name: dict[str, float],
    ) -> None:
        avp_names = FINGER_LAYOUT[chain_key]['avp']
        robot_roots = FINGER_LAYOUT[chain_key]['robot']
        points = [local_positions[name] for name in avp_names]
        base_vec = points[1] - points[0]
        seg_1 = points[2] - points[1]
        seg_2 = points[3] - points[2]
        seg_3 = points[4] - points[3]

        spread_name = f'{robot_roots[0]}_{suffix}'
        flex_names = tuple(f'{robot_name}_{suffix}' for robot_name in robot_roots[1:])

        spread = 0.75 * math.atan2(float(base_vec[0]), max(abs(float(base_vec[1])), 1.0e-6))
        if chain_key == 'ring':
            spread *= 0.85
        elif chain_key == 'little':
            spread *= 0.75
        elif chain_key == 'middle':
            spread *= 0.35

        flex_1 = 1.10 * _angle_between(base_vec, seg_1)
        flex_2 = 1.15 * _angle_between(seg_1, seg_2)
        flex_3 = 1.05 * _angle_between(seg_2, seg_3)

        pose_by_name[spread_name] = self._clamp_joint(spread_name, spread)
        for joint_name, joint_value in zip(flex_names, (flex_1, flex_2, flex_3), strict=False):
            pose_by_name[joint_name] = self._clamp_joint(joint_name, joint_value)

    def _retarget_fingers(self, side: str, frame, pose_by_name: dict[str, float]) -> None:
        local_positions = self._aligned_hand_local_positions(side, frame)
        if local_positions is None:
            return
        suffix = _side_suffix(side)
        self._apply_thumb(suffix, local_positions, pose_by_name)
        for chain_key in ('index', 'middle', 'ring', 'little'):
            self._apply_finger_chain(
                suffix=suffix,
                chain_key=chain_key,
                local_positions=local_positions,
                pose_by_name=pose_by_name,
            )

    def retarget_frame(self, frame) -> dict[str, float]:
        if frame is None:
            return {}
        pose_by_name: dict[str, float] = {}
        self._retarget_head(frame, pose_by_name)
        self._solve_arm('left', frame, pose_by_name)
        self._solve_arm('right', frame, pose_by_name)
        self._retarget_fingers('left', frame, pose_by_name)
        self._retarget_fingers('right', frame, pose_by_name)
        return pose_by_name

    def retarget_snapshot_pose(self) -> dict[str, float]:
        return self.retarget_frame(self.snapshot_frame)
