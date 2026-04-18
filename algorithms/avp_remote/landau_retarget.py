from __future__ import annotations

import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from asset_paths import module_root
from avp_snapshot_io import load_snapshot_payload
from avp_config import (
    AVP_HAND_RADIUS_CAP_SCALE,
    AVP_TRACKING_ROTATE_XYZ,
    AVP_TRACKING_SCALE_XYZ,
    AVP_TRACKING_TRANSLATE_XYZ,
)
from avp_tracking_schema import HAND_JOINT_NAMES, extract_tracking_frame
from avp_transform_utils import TransformOptions, build_xyz_transform, to_usd_world
from landau_mapping_config import (
    ARM_BASE_LINK,
    FINGER_LAYOUT,
    HEAD_LINK,
    LEFT_ARM_CHAIN,
    LEFT_ARM_TIP,
    RIGHT_ARM_CHAIN,
    RIGHT_ARM_TIP,
    UNTRACKED_POSE_DEFAULTS,
    apply_output_rule,
)
from urdf_kinematics import (
    find_joint_chain_from_specs,
    load_urdf_joint_specs,
    specs_by_child_link,
    world_map_from_joint_specs,
)


AVP_TO_SCENE_OPTIONS = TransformOptions(
    column_major=False,
    pretransform=build_xyz_transform(
        AVP_TRACKING_ROTATE_XYZ,
        AVP_TRACKING_TRANSLATE_XYZ,
        scale_xyz=AVP_TRACKING_SCALE_XYZ,
    ).T,
    posttransform=None,
)

HAND_JOINT_INDEX = {name: index for index, name in enumerate(HAND_JOINT_NAMES)}

@dataclass(frozen=True)
class HandCalibration:
    scale: float
    rotation_offset: np.ndarray


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


def _angle_between(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    a = _normalize(vec_a)
    b = _normalize(vec_b)
    if not np.any(a) or not np.any(b):
        return 0.0
    dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
    return float(np.arccos(dot))


def _side_suffix(side: str) -> str:
    return "l" if side == "left" else "r"


def _tracking_matrix_world(mat):
    if mat is None:
        return None
    return np.asarray(to_usd_world(mat, options=AVP_TO_SCENE_OPTIONS), dtype=float)


def _tracking_stack_world(stack):
    if stack is None:
        return None
    return np.stack([_tracking_matrix_world(mat) for mat in stack], axis=0)


def _load_trac_ik():
    helper_path = module_root().parents[1] / "helper_repos" / "pytracik"
    helper_str = str(helper_path)
    if helper_str not in sys.path:
        sys.path.insert(0, helper_str)
    try:
        from trac_ik import TracIK
    except Exception as exc:
        raise RuntimeError(
            "Failed to import local pytracik helper repo. "
            f"Expected at {helper_path}."
        ) from exc
    return TracIK


def find_joint_chain(urdf_path: Path, base_link: str, tip_link: str) -> tuple[str, ...]:
    return find_joint_chain_from_specs(load_urdf_joint_specs(urdf_path), base_link, tip_link, key="child_link")


class LandauUpperBodyRetargeter:
    def __init__(
        self,
        *,
        urdf_path: Path,
        skeleton_json_path: Path,
        snapshot_path: Path,
        hand_retargeting_client=None,
        hand_retarget_interval_sec: float = 0.0,
        arm_retarget_interval_sec: float = 0.0,
        arm_ik_error_tolerance: float | None = 3.0e-2,
        arm_ccd_iterations: int = 80,
        profile_enabled: bool = False,
    ) -> None:
        self.urdf_path = Path(urdf_path).resolve()
        self.skeleton_json_path = Path(skeleton_json_path).resolve()
        self.snapshot_path = Path(snapshot_path).expanduser().resolve()
        self.hand_retargeting_client = hand_retargeting_client
        self.hand_retarget_interval_sec = max(float(hand_retarget_interval_sec), 0.0)
        self.arm_retarget_interval_sec = max(float(arm_retarget_interval_sec), 0.0)
        self.arm_ik_error_tolerance = None if arm_ik_error_tolerance is None else float(arm_ik_error_tolerance)
        self.arm_ccd_iterations = max(int(arm_ccd_iterations), 1)
        self._last_hand_retarget_at = 0.0
        self._last_arm_retarget_at = 0.0
        self.profile_enabled = bool(profile_enabled)
        self.last_profile: dict[str, float] = {}
        self.last_hand_targets = {"landau": {}, "h1_2": {}}
        self.last_arm_pose: dict[str, float] = {}

        self.urdf_joint_specs = load_urdf_joint_specs(self.urdf_path)
        self.joint_specs = specs_by_child_link(self.urdf_joint_specs)
        self.rest_world_by_name = world_map_from_joint_specs(self.urdf_joint_specs)
        self.base_world = self.rest_world_by_name[ARM_BASE_LINK]
        self.base_world_inv = np.linalg.inv(self.base_world)
        self.head_world = self.rest_world_by_name[HEAD_LINK]
        self.rest_hand_base_rotation = {
            side: self.base_world_inv @ self.rest_world_by_name[f"hand_{_side_suffix(side)}"]
            for side in ("left", "right")
        }
        self.rest_hand_base_radius = {
            side: float(np.linalg.norm(self.rest_hand_base_rotation[side][:3, 3]))
            for side in ("left", "right")
        }
        self.rest_hand_local_positions = {
            side: self._build_hand_local_rest_positions(side)
            for side in ("left", "right")
        }
        self.robot_hand_basis = {
            side: self._build_robot_hand_basis(side)
            for side in ("left", "right")
        }

        self.left_chain = find_joint_chain_from_specs(self.urdf_joint_specs, ARM_BASE_LINK, LEFT_ARM_TIP, key="child_link")
        self.right_chain = find_joint_chain_from_specs(self.urdf_joint_specs, ARM_BASE_LINK, RIGHT_ARM_TIP, key="child_link")

        TracIK = _load_trac_ik()
        self.left_solver = TracIK(base_link_name=ARM_BASE_LINK, tip_link_name=LEFT_ARM_TIP, urdf_path=str(self.urdf_path))
        self.right_solver = TracIK(base_link_name=ARM_BASE_LINK, tip_link_name=RIGHT_ARM_TIP, urdf_path=str(self.urdf_path))
        self.left_seed = np.zeros(self.left_solver.dof, dtype=float)
        self.right_seed = np.zeros(self.right_solver.dof, dtype=float)

        snapshot_payload = load_snapshot_payload(self.snapshot_path)
        self.snapshot_frame = extract_tracking_frame(snapshot_payload)
        snapshot_head_world = _tracking_matrix_world(self.snapshot_frame.get("head"))
        if snapshot_head_world is None:
            self.head_frame_alignment = np.eye(3, dtype=float)
        else:
            self.head_frame_alignment = _orthonormalize(self.head_world[:3, :3]) @ _orthonormalize(snapshot_head_world[:3, :3]).T
        self.calibration = {
            "left": self._build_hand_calibration("left", self.snapshot_frame),
            "right": self._build_hand_calibration("right", self.snapshot_frame),
        }

    def _should_refresh_hand_targets(self) -> bool:
        if self.hand_retargeting_client is None:
            return False
        if self.hand_retarget_interval_sec <= 0.0:
            return True
        now = time.perf_counter()
        if not self.last_hand_targets.get("landau") and not self.last_hand_targets.get("h1_2"):
            self._last_hand_retarget_at = now
            return True
        if (now - self._last_hand_retarget_at) >= self.hand_retarget_interval_sec:
            self._last_hand_retarget_at = now
            return True
        return False

    def _should_refresh_arm_targets(self) -> bool:
        if self.arm_retarget_interval_sec <= 0.0:
            return True
        now = time.perf_counter()
        if not self.last_arm_pose:
            self._last_arm_retarget_at = now
            return True
        if (now - self._last_arm_retarget_at) >= self.arm_retarget_interval_sec:
            self._last_arm_retarget_at = now
            return True
        return False

    def _base_relative_world_map(self, pose_by_name: dict[str, float]) -> dict[str, np.ndarray]:
        world_map = world_map_from_joint_specs(self.urdf_joint_specs, pose_by_name, pose_key="child_link")
        return {
            link_name: self.base_world_inv @ link_world
            for link_name, link_world in world_map.items()
        }

    def _build_hand_calibration(self, side: str, frame) -> HandCalibration:
        head_world = _tracking_matrix_world(frame.get("head"))
        wrist_world = _tracking_matrix_world(frame.get(f"{side}_wrist"))
        if head_world is None or wrist_world is None:
            return HandCalibration(scale=1.0, rotation_offset=self.head_frame_alignment.copy())

        avp_hand_rel_head = np.linalg.inv(head_world) @ wrist_world
        robot_hand_rel_head = np.linalg.inv(self.head_world) @ self.rest_world_by_name[f"hand_{_side_suffix(side)}"]
        avp_length = float(np.linalg.norm(avp_hand_rel_head[:3, 3]))
        robot_length = float(np.linalg.norm(robot_hand_rel_head[:3, 3]))
        scale = robot_length / max(avp_length, 1.0e-6)
        return HandCalibration(scale=scale, rotation_offset=self.head_frame_alignment.copy())

    def _build_hand_local_rest_positions(self, side: str) -> dict[str, np.ndarray]:
        suffix = _side_suffix(side)
        hand_name = f"hand_{suffix}"
        hand_world_inv = np.linalg.inv(self.rest_world_by_name[hand_name])
        local_positions = {hand_name: np.zeros(3, dtype=float)}
        for name, world_transform in self.rest_world_by_name.items():
            if not name.endswith(f"_{suffix}") or name == hand_name:
                continue
            local_positions[name] = (hand_world_inv @ world_transform)[:3, 3].copy()
        return local_positions

    def _build_robot_hand_basis(self, side: str) -> np.ndarray:
        suffix = _side_suffix(side)
        local_positions = self.rest_hand_local_positions[side]
        lateral = local_positions[f"index1_base_{suffix}"] - local_positions[f"pinky1_base_{suffix}"]
        forward = (
            local_positions[f"index1_{suffix}"]
            + local_positions[f"middle1_{suffix}"]
            + local_positions[f"ring1_{suffix}"]
        ) / 3.0
        return _basis_from_vectors(lateral, forward)

    def _clamp_joint(self, joint_name: str, value: float) -> float:
        spec = self.joint_specs.get(joint_name)
        if spec is None:
            return float(value)
        return float(np.clip(float(value), spec.lower, spec.upper))

    def _mapped_joint_value(self, joint_name: str, raw_value: float) -> float:
        return self._clamp_joint(joint_name, apply_output_rule(joint_name, raw_value))

    def _merge_mapped_pose(self, pose_by_name: dict[str, float], updates: dict[str, float]) -> None:
        for joint_name, raw_value in updates.items():
            pose_by_name[joint_name] = self._mapped_joint_value(joint_name, raw_value)

    def _retarget_head(self, frame, pose_by_name: dict[str, float]) -> None:
        head_world = _tracking_matrix_world(frame.get("head"))
        if head_world is None:
            return

        forward_axis = _normalize(head_world[:3, 1])
        forward_xy = math.sqrt(float(forward_axis[0] ** 2 + forward_axis[1] ** 2))
        pitch = math.atan2(float(forward_axis[2]), max(forward_xy, 1.0e-6))
        pose_by_name["neck_x"] = self._mapped_joint_value("neck_x", pitch)
        pose_by_name["head_x"] = self._mapped_joint_value("head_x", pitch)

    def _clamp_hand_base_position(self, side: str, target_pos: np.ndarray) -> np.ndarray:
        clipped = np.asarray(target_pos, dtype=float).copy()
        target_radius = float(np.linalg.norm(clipped))
        radius_cap = AVP_HAND_RADIUS_CAP_SCALE * self.rest_hand_base_radius[side]
        if target_radius > radius_cap and target_radius > 1.0e-8:
            clipped *= radius_cap / target_radius
        return clipped

    def _finger_curl_amount(self, points: list[np.ndarray]) -> float:
        chain_length = sum(float(np.linalg.norm(curr - prev)) for prev, curr in zip(points[:-1], points[1:], strict=False))
        if chain_length <= 1.0e-8:
            return 0.0
        straight_length = float(np.linalg.norm(points[-1] - points[0]))
        return float(np.clip(1.0 - (straight_length / chain_length), 0.0, 1.0))

    def _target_hand_base_position(self, side: str, frame) -> np.ndarray | None:
        wrist_world = _tracking_matrix_world(frame.get(f"{side}_wrist"))
        if wrist_world is None:
            return None
        target_base = self.base_world_inv @ wrist_world
        return self._clamp_hand_base_position(side, target_base[:3, 3])

    def _arm_tip_base_position(self, side: str, joint_values: np.ndarray) -> np.ndarray:
        chain = self.left_chain if side == "left" else self.right_chain
        hand_name = f"hand_{_side_suffix(side)}"
        pose = {
            joint_name: float(joint_value)
            for joint_name, joint_value in zip(chain, joint_values, strict=False)
        }
        return self._base_relative_world_map(pose)[hand_name][:3, 3]

    def _solve_arm(self, side: str, frame, pose_by_name: dict[str, float]) -> None:
        target_pos = self._target_hand_base_position(side, frame)
        if target_pos is None:
            return

        solver = self.left_solver if side == "left" else self.right_solver
        chain = self.left_chain if side == "left" else self.right_chain
        seed = self.left_seed if side == "left" else self.right_seed
        solution = None

        # The mesh URDF now exposes a full 6-DOF torso-to-hand chain.
        # Prefer the solver when it can satisfy the wrist target, otherwise fall back to CCD.
        if solver.dof >= 6:
            ik_started_at = time.perf_counter() if self.profile_enabled else 0.0
            solution = solver.ik(
                target_pos,
                self.rest_hand_base_rotation[side][:3, :3],
                seed_jnt_values=seed,
            )
            if self.profile_enabled:
                self.last_profile[f"{side}_arm_ik"] = time.perf_counter() - ik_started_at
            if solution is not None:
                solved_pos = self._arm_tip_base_position(side, np.asarray(solution, dtype=float))
                error = float(np.linalg.norm(solved_pos - target_pos))
                if self.profile_enabled:
                    self.last_profile[f"{side}_arm_ik_error"] = error
                if self.arm_ik_error_tolerance is not None and error > self.arm_ik_error_tolerance:
                    solution = None
        if solution is None:
            ccd_started_at = time.perf_counter() if self.profile_enabled else 0.0
            solution = self._solve_arm_ccd(side, target_pos, seed)
            if self.profile_enabled:
                self.last_profile[f"{side}_arm_ccd"] = time.perf_counter() - ccd_started_at

        if side == "left":
            self.left_seed = np.asarray(solution, dtype=float)
        else:
            self.right_seed = np.asarray(solution, dtype=float)

        for joint_name, joint_value in zip(chain, solution, strict=False):
            pose_by_name[joint_name] = self._mapped_joint_value(joint_name, joint_value)

    def _solve_arm_ccd(self, side: str, target_pos: np.ndarray, seed: np.ndarray) -> np.ndarray:
        chain = self.left_chain if side == "left" else self.right_chain
        hand_name = f"hand_{_side_suffix(side)}"
        pose = {
            joint_name: float(joint_value)
            for joint_name, joint_value in zip(chain, seed, strict=False)
        }

        for _ in range(self.arm_ccd_iterations):
            base_relative_map = self._base_relative_world_map(pose)
            current_pos = base_relative_map[hand_name][:3, 3]
            if float(np.linalg.norm(target_pos - current_pos)) <= 1.0e-3:
                break

            for joint_name in reversed(chain):
                base_relative_map = self._base_relative_world_map(pose)
                current_pos = base_relative_map[hand_name][:3, 3]
                joint_transform = base_relative_map[joint_name]
                joint_pos = joint_transform[:3, 3]
                axis_world = _normalize(joint_transform[:3, :3] @ self.joint_specs[joint_name].axis)
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
                pose[joint_name] = self._clamp_joint(
                    joint_name,
                    pose.get(joint_name, 0.0) + 0.85 * signed_angle,
                )

        return np.asarray([pose.get(joint_name, 0.0) for joint_name in chain], dtype=float)

    def _wrist_local_positions(self, side: str, frame) -> dict[str, np.ndarray] | None:
        stack = _tracking_stack_world(frame.get(f"{side}_arm"))
        if stack is None or len(stack) != len(HAND_JOINT_NAMES):
            return None
        wrist_inv = np.linalg.inv(stack[HAND_JOINT_INDEX["wrist"]])
        local_positions = {}
        for joint_name, joint_index in HAND_JOINT_INDEX.items():
            local_mat = wrist_inv @ stack[joint_index]
            local_positions[joint_name] = local_mat[:3, 3].copy()
        return local_positions

    def _aligned_hand_local_positions(self, side: str, frame) -> dict[str, np.ndarray] | None:
        local_positions = self._wrist_local_positions(side, frame)
        if local_positions is None:
            return None

        lateral = local_positions["littleMetacarpal"] - local_positions["indexMetacarpal"]
        forward = (
            local_positions["indexKnuckle"]
            + local_positions["middleKnuckle"]
            + local_positions["ringKnuckle"]
        ) / 3.0
        avp_basis = _basis_from_vectors(lateral, forward)
        alignment = self.robot_hand_basis[side] @ avp_basis.T
        return {
            joint_name: alignment @ np.asarray(position, dtype=float)
            for joint_name, position in local_positions.items()
        }

    def _apply_thumb(self, suffix: str, local_positions: dict[str, np.ndarray], pose_by_name: dict[str, float]) -> None:
        points = [local_positions[name] for name in FINGER_LAYOUT["thumb"]["avp"]]
        base_vec = points[0]
        seg_1 = points[1] - points[0]
        seg_2 = points[2] - points[1]
        seg_3 = points[3] - points[2]

        thumb1_name = f"thumb1_{suffix}"
        thumb2_name = f"thumb2_{suffix}"
        thumb3_name = f"thumb3_{suffix}"

        oppose = math.atan2(float(base_vec[0]), max(abs(float(base_vec[1])), 1.0e-6))
        flex_2 = -1.15 * _angle_between(seg_1, seg_2)
        flex_3 = -1.05 * _angle_between(seg_2, seg_3)

        pose_by_name[thumb1_name] = self._mapped_joint_value(thumb1_name, oppose)
        pose_by_name[thumb2_name] = self._mapped_joint_value(thumb2_name, flex_2)
        pose_by_name[thumb3_name] = self._mapped_joint_value(thumb3_name, flex_3)

    def _apply_finger_chain(
        self,
        *,
        suffix: str,
        chain_key: str,
        local_positions: dict[str, np.ndarray],
        pose_by_name: dict[str, float],
    ) -> None:
        avp_names = FINGER_LAYOUT[chain_key]["avp"]
        robot_roots = FINGER_LAYOUT[chain_key]["robot"]
        points = [local_positions[name] for name in avp_names]
        base_vec = points[1] - points[0]
        seg_1 = points[2] - points[1]
        seg_2 = points[3] - points[2]
        seg_3 = points[4] - points[3]

        spread_name = f"{robot_roots[0]}_{suffix}"
        flex_names = tuple(f"{robot_name}_{suffix}" for robot_name in robot_roots[1:])

        spread = math.atan2(float(base_vec[0]), max(abs(float(base_vec[1])), 1.0e-6))
        curl = self._finger_curl_amount(points)
        curl_angle = 1.9 * curl
        flex_1 = -max(_angle_between(base_vec, seg_1), 0.55 * curl_angle)
        flex_2 = -max(_angle_between(seg_1, seg_2), 0.75 * curl_angle)
        flex_3 = -max(_angle_between(seg_2, seg_3), 0.65 * curl_angle)

        pose_by_name[spread_name] = self._mapped_joint_value(spread_name, spread)
        for joint_name, joint_value in zip(flex_names, (flex_1, flex_2, flex_3), strict=False):
            pose_by_name[joint_name] = self._mapped_joint_value(joint_name, joint_value)

    def _retarget_fingers(self, side: str, frame, pose_by_name: dict[str, float]) -> None:
        local_positions = self._aligned_hand_local_positions(side, frame)
        if local_positions is None:
            return

        suffix = _side_suffix(side)
        self._apply_thumb(suffix, local_positions, pose_by_name)
        for chain_key in ("index", "middle", "ring", "little"):
            self._apply_finger_chain(
                suffix=suffix,
                chain_key=chain_key,
                local_positions=local_positions,
                pose_by_name=pose_by_name,
            )

    def retarget_frame(self, frame) -> dict[str, float]:
        if frame is None:
            self._last_hand_retarget_at = 0.0
            self._last_arm_retarget_at = 0.0
            self.last_profile = {}
            self.last_hand_targets = {"landau": {}, "h1_2": {}}
            self.last_arm_pose = {}
            return {}

        self.last_profile = {}
        pose_by_name: dict[str, float] = dict(UNTRACKED_POSE_DEFAULTS)
        head_started_at = time.perf_counter() if self.profile_enabled else 0.0
        self._retarget_head(frame, pose_by_name)
        if self.profile_enabled:
            self.last_profile["head"] = time.perf_counter() - head_started_at
        if self._should_refresh_arm_targets():
            left_arm_started_at = time.perf_counter() if self.profile_enabled else 0.0
            self._solve_arm("left", frame, pose_by_name)
            if self.profile_enabled:
                self.last_profile["left_arm"] = time.perf_counter() - left_arm_started_at
            right_arm_started_at = time.perf_counter() if self.profile_enabled else 0.0
            self._solve_arm("right", frame, pose_by_name)
            if self.profile_enabled:
                self.last_profile["right_arm"] = time.perf_counter() - right_arm_started_at
            arm_joint_names = (*self.left_chain, *self.right_chain)
            self.last_arm_pose = {
                joint_name: float(pose_by_name[joint_name])
                for joint_name in arm_joint_names
                if joint_name in pose_by_name
            }
        else:
            self._merge_mapped_pose(pose_by_name, self.last_arm_pose)
        if self.hand_retargeting_client is None:
            self.last_hand_targets = {"landau": {}, "h1_2": {}}
            left_hand_started_at = time.perf_counter() if self.profile_enabled else 0.0
            self._retarget_fingers("left", frame, pose_by_name)
            if self.profile_enabled:
                self.last_profile["left_hand_heuristic"] = time.perf_counter() - left_hand_started_at
            right_hand_started_at = time.perf_counter() if self.profile_enabled else 0.0
            self._retarget_fingers("right", frame, pose_by_name)
            if self.profile_enabled:
                self.last_profile["right_hand_heuristic"] = time.perf_counter() - right_hand_started_at
            return pose_by_name

        if self._should_refresh_hand_targets():
            hand_helper_started_at = time.perf_counter() if self.profile_enabled else 0.0
            try:
                self.last_hand_targets = self.hand_retargeting_client.retarget_frame(frame)
            except Exception as exc:
                print(
                    f"[AVP] Dex hand retargeting failed, falling back to heuristic finger mapping: {exc}",
                    flush=True,
                )
                self.hand_retargeting_client = None
                self._last_hand_retarget_at = 0.0
                self.last_hand_targets = {"landau": {}, "h1_2": {}}
                left_hand_started_at = time.perf_counter() if self.profile_enabled else 0.0
                self._retarget_fingers("left", frame, pose_by_name)
                if self.profile_enabled:
                    self.last_profile["left_hand_heuristic"] = time.perf_counter() - left_hand_started_at
                right_hand_started_at = time.perf_counter() if self.profile_enabled else 0.0
                self._retarget_fingers("right", frame, pose_by_name)
                if self.profile_enabled:
                    self.last_profile["right_hand_heuristic"] = time.perf_counter() - right_hand_started_at
                return pose_by_name
            if self.profile_enabled:
                self.last_profile["hand_helper"] = time.perf_counter() - hand_helper_started_at

        self._merge_mapped_pose(pose_by_name, self.last_hand_targets.get("landau", {}))
        return pose_by_name

    def h1_2_hand_pose_overrides(self) -> dict[str, float]:
        return {
            joint_name: float(value)
            for joint_name, value in self.last_hand_targets.get("h1_2", {}).items()
        }

    def g1_hand_pose_overrides(self) -> dict[str, float]:
        return self.h1_2_hand_pose_overrides()
