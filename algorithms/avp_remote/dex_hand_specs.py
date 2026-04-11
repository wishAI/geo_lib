from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np

from avp_g1_pose import world_map_from_urdf_pose
from avp_tracking_schema import HAND_JOINT_NAMES
from urdf_kinematics import load_urdf_joint_specs


HAND_JOINT_INDEX = {name: index for index, name in enumerate(HAND_JOINT_NAMES)}


@dataclass(frozen=True)
class DexVectorTargetSpec:
    group: str
    side: str
    urdf_path: Path
    hand_link_name: str
    target_joint_names: tuple[str, ...]
    pose_joint_names: tuple[str, ...] | None
    target_origin_link_names: tuple[str, ...]
    target_task_link_names: tuple[str, ...]
    human_origin_indices: tuple[int, ...]
    human_task_indices: tuple[int, ...]
    basis_lateral_links: tuple[str, str]
    basis_forward_links: tuple[str, ...]

    @property
    def required_robot_link_names(self) -> tuple[str, ...]:
        names = [
            self.hand_link_name,
            *self.target_origin_link_names,
            *self.target_task_link_names,
            *self.basis_lateral_links,
            *self.basis_forward_links,
        ]
        return tuple(dict.fromkeys(names))

    @property
    def resolved_pose_joint_names(self) -> tuple[str, ...]:
        return self.pose_joint_names or self.target_joint_names

    def build_config(self, scaling_factor: float, low_pass_alpha: float = 0.0) -> dict[str, object]:
        return {
            "type": "vector",
            "urdf_path": str(self.urdf_path),
            "target_joint_names": list(self.target_joint_names),
            "target_origin_link_names": list(self.target_origin_link_names),
            "target_task_link_names": list(self.target_task_link_names),
            "target_link_human_indices": [
                list(self.human_origin_indices),
                list(self.human_task_indices),
            ],
            "scaling_factor": float(scaling_factor),
            "low_pass_alpha": float(low_pass_alpha),
        }


def _normalize(vec: np.ndarray) -> np.ndarray:
    arr = np.asarray(vec, dtype=float)
    norm = float(np.linalg.norm(arr))
    if norm < 1.0e-8:
        return np.zeros_like(arr)
    return arr / norm


def _orthonormalize(rotation: np.ndarray) -> np.ndarray:
    u, _, vh = np.linalg.svd(rotation)
    fixed = u @ vh
    if np.linalg.det(fixed) < 0.0:
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


def side_suffix(side: str) -> str:
    return "l" if side == "left" else "r"


def human_local_position_map(local_positions: np.ndarray) -> dict[str, np.ndarray]:
    arr = np.asarray(local_positions, dtype=float)
    if arr.shape != (len(HAND_JOINT_NAMES), 3):
        raise ValueError(
            f"Expected hand local positions with shape {(len(HAND_JOINT_NAMES), 3)}, got {arr.shape}",
        )
    return {
        joint_name: arr[index].copy()
        for joint_name, index in HAND_JOINT_INDEX.items()
    }


def avp_hand_basis(local_positions: Mapping[str, np.ndarray]) -> np.ndarray:
    lateral = np.asarray(local_positions["littleMetacarpal"], dtype=float) - np.asarray(
        local_positions["indexMetacarpal"],
        dtype=float,
    )
    forward = (
        np.asarray(local_positions["indexKnuckle"], dtype=float)
        + np.asarray(local_positions["middleKnuckle"], dtype=float)
        + np.asarray(local_positions["ringKnuckle"], dtype=float)
    ) / 3.0
    return _basis_from_vectors(lateral, forward)


def robot_hand_basis(spec: DexVectorTargetSpec, local_positions: Mapping[str, np.ndarray]) -> np.ndarray:
    lateral = np.asarray(local_positions[spec.basis_lateral_links[0]], dtype=float) - np.asarray(
        local_positions[spec.basis_lateral_links[1]],
        dtype=float,
    )
    forward = np.mean(
        [np.asarray(local_positions[name], dtype=float) for name in spec.basis_forward_links],
        axis=0,
    )
    return _basis_from_vectors(lateral, forward)


def urdf_local_positions(spec: DexVectorTargetSpec) -> dict[str, np.ndarray]:
    world_map = world_map_from_urdf_pose(spec.urdf_path)
    hand_world_inv = np.linalg.inv(world_map[spec.hand_link_name])
    return {
        link_name: (hand_world_inv @ world_map[link_name])[:3, 3].copy()
        for link_name in spec.required_robot_link_names
    }


def align_human_local_positions(
    spec: DexVectorTargetSpec,
    human_local_positions: Mapping[str, np.ndarray],
    robot_local_positions: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    alignment = robot_hand_basis(spec, robot_local_positions) @ avp_hand_basis(human_local_positions).T
    return {
        joint_name: alignment @ np.asarray(position, dtype=float)
        for joint_name, position in human_local_positions.items()
    }


def reference_vectors(
    spec: DexVectorTargetSpec,
    aligned_human_local_positions: Mapping[str, np.ndarray],
) -> np.ndarray:
    vectors = []
    for origin_index, task_index in zip(
        spec.human_origin_indices,
        spec.human_task_indices,
        strict=False,
    ):
        origin_name = HAND_JOINT_NAMES[origin_index]
        task_name = HAND_JOINT_NAMES[task_index]
        vectors.append(
            np.asarray(aligned_human_local_positions[task_name], dtype=float)
            - np.asarray(aligned_human_local_positions[origin_name], dtype=float),
        )
    return np.stack(vectors, axis=0)


def scaling_factor(
    spec: DexVectorTargetSpec,
    robot_local_positions: Mapping[str, np.ndarray],
    human_local_positions: Mapping[str, np.ndarray],
) -> float:
    ratios = []
    for robot_origin, robot_task, human_origin_index, human_task_index in zip(
        spec.target_origin_link_names,
        spec.target_task_link_names,
        spec.human_origin_indices,
        spec.human_task_indices,
        strict=False,
    ):
        robot_dist = float(
            np.linalg.norm(
                np.asarray(robot_local_positions[robot_task], dtype=float)
                - np.asarray(robot_local_positions[robot_origin], dtype=float),
            ),
        )
        human_dist = float(
            np.linalg.norm(
                np.asarray(human_local_positions[HAND_JOINT_NAMES[human_task_index]], dtype=float)
                - np.asarray(human_local_positions[HAND_JOINT_NAMES[human_origin_index]], dtype=float),
            ),
        )
        if human_dist > 1.0e-6:
            ratios.append(robot_dist / human_dist)
    if not ratios:
        return 1.0
    return float(np.mean(ratios))


def build_landau_target_specs(urdf_path: Path) -> dict[str, DexVectorTargetSpec]:
    urdf = Path(urdf_path).expanduser().resolve()
    child_link_specs = {spec.child_link: spec for spec in load_urdf_joint_specs(urdf).values()}
    specs = {}
    for side in ("left", "right"):
        suffix = side_suffix(side)
        pose_joint_names = (
            f"thumb1_{suffix}",
            f"thumb2_{suffix}",
            f"thumb3_{suffix}",
            f"index1_base_{suffix}",
            f"index1_{suffix}",
            f"index2_{suffix}",
            f"index3_{suffix}",
            f"middle1_base_{suffix}",
            f"middle1_{suffix}",
            f"middle2_{suffix}",
            f"middle3_{suffix}",
            f"ring1_base_{suffix}",
            f"ring1_{suffix}",
            f"ring2_{suffix}",
            f"ring3_{suffix}",
            f"pinky1_base_{suffix}",
            f"pinky1_{suffix}",
            f"pinky2_{suffix}",
            f"pinky3_{suffix}",
        )
        specs[side] = DexVectorTargetSpec(
            group="landau",
            side=side,
            urdf_path=urdf,
            hand_link_name=f"hand_{suffix}",
            target_joint_names=tuple(
                child_link_specs[joint_name].name
                for joint_name in pose_joint_names
            ),
            pose_joint_names=pose_joint_names,
            target_origin_link_names=(
                f"thumb1_{suffix}",
                f"index1_base_{suffix}",
                f"middle1_base_{suffix}",
                f"ring1_base_{suffix}",
                f"pinky1_base_{suffix}",
            ),
            target_task_link_names=(
                f"thumb3_{suffix}",
                f"index3_{suffix}",
                f"middle3_{suffix}",
                f"ring3_{suffix}",
                f"pinky3_{suffix}",
            ),
            human_origin_indices=(
                HAND_JOINT_INDEX["thumbKnuckle"],
                HAND_JOINT_INDEX["indexMetacarpal"],
                HAND_JOINT_INDEX["middleMetacarpal"],
                HAND_JOINT_INDEX["ringMetacarpal"],
                HAND_JOINT_INDEX["littleMetacarpal"],
            ),
            human_task_indices=(
                HAND_JOINT_INDEX["thumbTip"],
                HAND_JOINT_INDEX["indexTip"],
                HAND_JOINT_INDEX["middleTip"],
                HAND_JOINT_INDEX["ringTip"],
                HAND_JOINT_INDEX["littleTip"],
            ),
            basis_lateral_links=(f"index1_base_{suffix}", f"pinky1_base_{suffix}"),
            basis_forward_links=(f"index1_{suffix}", f"middle1_{suffix}", f"ring1_{suffix}"),
        )
    return specs


def build_h1_2_target_specs(urdf_path: Path) -> dict[str, DexVectorTargetSpec]:
    urdf = Path(urdf_path).expanduser().resolve()
    assets_root = urdf.parent.parent
    specs = {}
    for side in ("left", "right"):
        robot_prefix = "L" if side == "left" else "R"
        hand_urdf = assets_root / "inspire_hand" / f"inspire_hand_{side}.urdf"
        specs[side] = DexVectorTargetSpec(
            group="h1_2",
            side=side,
            urdf_path=hand_urdf,
            hand_link_name=f"{robot_prefix}_hand_base_link",
            target_joint_names=(
                f"{robot_prefix}_thumb_proximal_yaw_joint",
                f"{robot_prefix}_thumb_proximal_pitch_joint",
                f"{robot_prefix}_index_proximal_joint",
                f"{robot_prefix}_middle_proximal_joint",
                f"{robot_prefix}_ring_proximal_joint",
                f"{robot_prefix}_pinky_proximal_joint",
            ),
            pose_joint_names=None,
            target_origin_link_names=(
                f"{robot_prefix}_hand_base_link",
                f"{robot_prefix}_hand_base_link",
                f"{robot_prefix}_hand_base_link",
                f"{robot_prefix}_hand_base_link",
                f"{robot_prefix}_hand_base_link",
            ),
            target_task_link_names=(
                f"{robot_prefix}_thumb_tip",
                f"{robot_prefix}_index_tip",
                f"{robot_prefix}_middle_tip",
                f"{robot_prefix}_ring_tip",
                f"{robot_prefix}_pinky_tip",
            ),
            human_origin_indices=(
                HAND_JOINT_INDEX["thumbKnuckle"],
                HAND_JOINT_INDEX["indexMetacarpal"],
                HAND_JOINT_INDEX["middleMetacarpal"],
                HAND_JOINT_INDEX["ringMetacarpal"],
                HAND_JOINT_INDEX["littleMetacarpal"],
            ),
            human_task_indices=(
                HAND_JOINT_INDEX["thumbTip"],
                HAND_JOINT_INDEX["indexTip"],
                HAND_JOINT_INDEX["middleTip"],
                HAND_JOINT_INDEX["ringTip"],
                HAND_JOINT_INDEX["littleTip"],
            ),
            basis_lateral_links=(f"{robot_prefix}_index_proximal", f"{robot_prefix}_pinky_proximal"),
            basis_forward_links=(
                f"{robot_prefix}_index_tip",
                f"{robot_prefix}_middle_tip",
                f"{robot_prefix}_ring_tip",
            ),
        )
    return specs


def build_g1_target_specs(urdf_path: Path) -> dict[str, DexVectorTargetSpec]:
    # Backward-compatible alias while the session code migrates from the old G1 naming.
    return build_h1_2_target_specs(urdf_path)
