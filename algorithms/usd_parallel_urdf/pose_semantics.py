from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np

from skeleton_common import (
    apply_pose_to_local_matrices,
    infer_lateral_axis_world,
    world_matrices_from_local,
)


DEFAULT_PAIR_MIRROR_SIGNS: dict[str, int] = {
    'shoulder': -1,
    'arm_stretch': 1,
    'forearm_stretch': 1,
    'hand': -1,
    'thumb1': -1,
    'thumb2': 1,
    'thumb3': 1,
    'index1_base': -1,
    'index1': 1,
    'index2': 1,
    'index3': 1,
    'middle1_base': -1,
    'middle1': 1,
    'middle2': 1,
    'middle3': 1,
    'ring1_base': -1,
    'ring1': 1,
    'ring2': 1,
    'ring3': 1,
    'pinky1_base': -1,
    'pinky1': 1,
    'pinky2': 1,
    'pinky3': 1,
    'thigh_stretch': 1,
    'leg_stretch': 1,
    'foot': 1,
    'toes_01': 1,
}


ANIMATION_CLIPS: Dict[str, List[tuple[str, float]]] = {
    'pose_cycle': [
        ('rest', 0.50),
        ('demo', 0.75),
        ('open_arms', 0.75),
        ('walk', 0.55),
        ('walk_right', 0.55),
    ],
    'walk_cycle': [
        ('walk', 0.45),
        ('walk_right', 0.45),
    ],
}


def _strip_side(name: str) -> str:
    if name.endswith('_l') or name.endswith('_r'):
        return name[:-2]
    return name


def _mirror_name(name: str) -> str | None:
    if name.endswith('_l'):
        return f'{name[:-2]}_r'
    if name.endswith('_r'):
        return f'{name[:-2]}_l'
    return None


def _rotation_error_deg(left_rotation: np.ndarray, right_rotation: np.ndarray) -> float:
    delta = left_rotation.T @ right_rotation
    trace = float(np.clip((np.trace(delta) - 1.0) * 0.5, -1.0, 1.0))
    return float(np.degrees(np.arccos(trace)))


def _root_relative_world_map(records: Sequence[dict], pose: dict[str, float]) -> dict[str, np.ndarray]:
    local_matrices = apply_pose_to_local_matrices(records, pose)
    world_matrices = world_matrices_from_local(records, local_matrices)
    root_inverse = np.linalg.inv(world_matrices[0])
    return {
        record['name']: root_inverse @ world_matrices[record['index']]
        for record in records
    }


@dataclass(frozen=True)
class PairChannel:
    left_name: str
    right_name: str
    symmetric: float = 0.0
    antisymmetric: float = 0.0


class SemanticRigAdapter:
    def __init__(self, records: Sequence[dict]):
        self.records = list(records)
        self.names = {record['name'] for record in self.records}
        self.record_by_name = {record['name']: record for record in self.records}
        self.children_by_name: dict[str, list[str]] = {record['name']: [] for record in self.records}
        for record in self.records:
            if 'child_names' in record:
                self.children_by_name[record['name']] = list(record['child_names'])
        for record in self.records:
            parent_name = record.get('parent_name')
            if parent_name is None:
                continue
            self.children_by_name.setdefault(parent_name, [])
            if record['name'] not in self.children_by_name[parent_name]:
                self.children_by_name[parent_name].append(record['name'])
        self._subtree_cache: dict[str, tuple[str, ...]] = {}
        self._mirror_sign_cache: dict[tuple[str, str], int] = {}
        self._mirror_matrix_cache: np.ndarray | None = None
        self._can_calibrate = all(
            'index' in record and 'local_matrix' in record and 'parent_index' in record
            for record in self.records
        )

    def merge(self, *chunks: dict[str, float]) -> dict[str, float]:
        pose: dict[str, float] = {}
        for chunk in chunks:
            for joint_name, angle in chunk.items():
                if joint_name not in self.names:
                    continue
                value = float(angle)
                if abs(value) <= 1e-10:
                    pose.pop(joint_name, None)
                else:
                    pose[joint_name] = value
        return pose

    def raw(self, mapping: dict[str, float]) -> dict[str, float]:
        return self.merge(mapping)

    def pair(self, left_name: str, right_name: str, *, symmetric: float = 0.0, antisymmetric: float = 0.0) -> dict[str, float]:
        if left_name not in self.names and right_name not in self.names:
            return {}
        mirror_sign = self.mirror_sign(left_name, right_name)
        left_value = float(symmetric + antisymmetric)
        right_value = float(mirror_sign * symmetric - mirror_sign * antisymmetric)
        return self.merge(
            {left_name: left_value} if left_name in self.names else {},
            {right_name: right_value} if right_name in self.names else {},
        )

    def pairs(self, *channels: PairChannel) -> dict[str, float]:
        return self.merge(*(self.pair(channel.left_name, channel.right_name, symmetric=channel.symmetric, antisymmetric=channel.antisymmetric) for channel in channels))

    def mirror_sign(self, left_name: str, right_name: str) -> int:
        cache_key = (left_name, right_name)
        if cache_key in self._mirror_sign_cache:
            return self._mirror_sign_cache[cache_key]

        default_sign = self._default_pair_sign(left_name)
        sign = default_sign
        if self._can_calibrate and left_name in self.names and right_name in self.names:
            calibrated = self._calibrate_pair_sign(left_name, right_name)
            if calibrated is not None:
                sign = calibrated
        self._mirror_sign_cache[cache_key] = sign
        return sign

    def _default_pair_sign(self, name: str) -> int:
        base_name = _strip_side(name)
        if base_name in DEFAULT_PAIR_MIRROR_SIGNS:
            return DEFAULT_PAIR_MIRROR_SIGNS[base_name]
        if base_name.endswith('_base') or 'shoulder' in base_name or base_name.startswith('hand'):
            return -1
        return 1

    def _mirror_matrix(self) -> np.ndarray:
        if self._mirror_matrix_cache is None:
            lateral_axis = infer_lateral_axis_world(self.records)
            self._mirror_matrix_cache = np.eye(3, dtype=float) - 2.0 * np.outer(lateral_axis, lateral_axis)
        return self._mirror_matrix_cache

    def _subtree_names(self, root_name: str) -> tuple[str, ...]:
        if root_name in self._subtree_cache:
            return self._subtree_cache[root_name]
        names: list[str] = []
        queue = [root_name]
        seen: set[str] = set()
        while queue:
            current = queue.pop(0)
            if current in seen or current not in self.names:
                continue
            seen.add(current)
            names.append(current)
            queue.extend(self.children_by_name.get(current, ()))
        result = tuple(names)
        self._subtree_cache[root_name] = result
        return result

    def _calibrate_pair_sign(self, left_name: str, right_name: str) -> int | None:
        angle = 0.35
        scores: dict[int, float] = {}
        for candidate in (1, -1):
            pose = {left_name: angle, right_name: candidate * angle}
            world_map = _root_relative_world_map(self.records, pose)
            scores[candidate] = self._subtree_mirror_score(world_map, left_name, right_name)
        best_candidate = min(scores, key=scores.get)
        best_score = scores[best_candidate]
        other_score = scores[-best_candidate]
        if other_score <= 1e-12:
            return None
        improvement_ratio = (other_score - best_score) / other_score
        if improvement_ratio < 0.05:
            return None
        return best_candidate

    def _subtree_mirror_score(self, world_map: dict[str, np.ndarray], left_name: str, right_name: str) -> float:
        mirror = self._mirror_matrix()
        right_subtree = set(self._subtree_names(right_name))
        pairs: list[tuple[str, str]] = []
        for left_desc in self._subtree_names(left_name):
            right_desc = _mirror_name(left_desc)
            if right_desc is None or right_desc not in right_subtree or right_desc not in world_map:
                continue
            pairs.append((left_desc, right_desc))
        if not pairs:
            pairs.append((left_name, right_name))

        score = 0.0
        for paired_left, paired_right in pairs:
            left_matrix = world_map[paired_left]
            right_matrix = world_map[paired_right]
            mirrored_right_xyz = mirror @ right_matrix[:3, 3]
            pos_error = float(np.linalg.norm(left_matrix[:3, 3] - mirrored_right_xyz))
            mirrored_right_rot = mirror @ right_matrix[:3, :3] @ mirror
            rot_error_deg = _rotation_error_deg(left_matrix[:3, :3], mirrored_right_rot)
            score += pos_error + 1e-3 * rot_error_deg
        return score / len(pairs)


def _build_demo_pose(adapter: SemanticRigAdapter) -> dict[str, float]:
    return adapter.merge(
        adapter.raw(
            {
                'spine_03_x': 0.10,
                'neck_x': -0.08,
                'arm_stretch_r': -0.40,
                'forearm_stretch_r': -0.85,
                'hand_r': 0.22,
                'thumb1_r': 0.35,
                'thumb2_r': 0.45,
                'thumb3_r': 0.35,
                'index1_base_r': 0.10,
                'index1_r': -0.65,
                'index2_r': -0.55,
                'index3_r': -0.35,
                'middle1_base_r': 0.06,
                'middle1_r': -0.70,
                'middle2_r': -0.58,
                'middle3_r': -0.38,
                'ring1_base_r': -0.04,
                'ring1_r': -0.66,
                'ring2_r': -0.55,
                'ring3_r': -0.35,
                'pinky1_base_r': -0.12,
                'pinky1_r': -0.60,
                'pinky2_r': -0.48,
                'pinky3_r': -0.30,
                'arm_stretch_l': 0.18,
                'forearm_stretch_l': -0.35,
                'hand_l': -0.10,
            }
        )
    )


def _build_open_arms_pose(adapter: SemanticRigAdapter) -> dict[str, float]:
    return adapter.merge(
        adapter.raw({'spine_03_x': 0.06, 'neck_x': -0.04}),
        adapter.pairs(
            PairChannel('shoulder_l', 'shoulder_r', symmetric=0.30),
            PairChannel('arm_stretch_l', 'arm_stretch_r', symmetric=1.45),
            PairChannel('forearm_stretch_l', 'forearm_stretch_r', symmetric=-0.08),
            PairChannel('hand_l', 'hand_r', symmetric=-0.04),
            PairChannel('thumb1_l', 'thumb1_r', symmetric=-0.10),
        ),
    )


def _build_walk_pose(adapter: SemanticRigAdapter, phase: float) -> dict[str, float]:
    return adapter.merge(
        adapter.raw({'spine_03_x': 0.08, 'neck_x': -0.05}),
        adapter.pairs(
            PairChannel('arm_stretch_l', 'arm_stretch_r', antisymmetric=0.41 * phase),
            PairChannel('forearm_stretch_l', 'forearm_stretch_r', symmetric=0.21, antisymmetric=-0.11 * phase),
            PairChannel('thigh_stretch_l', 'thigh_stretch_r', symmetric=0.20, antisymmetric=0.72 * phase),
            PairChannel('leg_stretch_l', 'leg_stretch_r', symmetric=0.54, antisymmetric=0.32 * phase),
            PairChannel('foot_l', 'foot_r', symmetric=-0.07, antisymmetric=-0.15 * phase),
            PairChannel('toes_01_l', 'toes_01_r', symmetric=0.05, antisymmetric=0.05 * phase),
        ),
    )


POSE_BUILDERS = {
    'demo': _build_demo_pose,
    'open_arms': _build_open_arms_pose,
    'walk': lambda adapter: _build_walk_pose(adapter, 1.0),
    'walk_right': lambda adapter: _build_walk_pose(adapter, -1.0),
}


def pose_preset_names() -> list[str]:
    return ['rest', *POSE_BUILDERS.keys()]


def build_pose_preset(records: Sequence[dict], preset: str) -> Dict[str, float]:
    if preset == 'rest':
        return {}
    if preset not in POSE_BUILDERS:
        raise KeyError(f'Unknown pose preset: {preset}')
    adapter = SemanticRigAdapter(records)
    return POSE_BUILDERS[preset](adapter)


def animation_clip_names() -> list[str]:
    return list(ANIMATION_CLIPS.keys())


def build_animation_clip(records: Sequence[dict], clip_name: str) -> List[tuple[str, Dict[str, float], float]]:
    if clip_name not in ANIMATION_CLIPS:
        raise KeyError(f'Unknown animation clip: {clip_name}')
    return [
        (preset_name, build_pose_preset(records, preset_name), float(duration_s))
        for preset_name, duration_s in ANIMATION_CLIPS[clip_name]
    ]


def build_demo_pose(records: Sequence[dict]) -> Dict[str, float]:
    return build_pose_preset(records, 'demo')
