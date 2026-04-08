from __future__ import annotations

import math
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from asset_paths import module_root
from avp_snapshot_io import load_snapshot_payload
from avp_tracking_schema import HAND_JOINT_NAMES, extract_tracking_frame
from avp_transform_utils import TransformOptions, build_xyz_transform, to_usd_world
from landau_pose import (
    SkeletonRecord,
    apply_joint_positions_to_local_matrices,
    load_skeleton_records,
    world_matrices_from_local,
)


AVP_TO_SCENE_OPTIONS = TransformOptions(
    column_major=False,
    pretransform=build_xyz_transform(
        (0.0, 0.0, 180.0),
        (0.0, -0.13, 0.13),
        scale_xyz=(0.6, 0.6, 0.6),
    ).T,
    posttransform=None,
)

HAND_JOINT_INDEX = {name: index for index, name in enumerate(HAND_JOINT_NAMES)}

ARM_BASE_LINK = "spine_03_x"
HEAD_LINK = "head_x"

LEFT_ARM_TIP = "hand_l"
RIGHT_ARM_TIP = "hand_r"

LEFT_ARM_CHAIN = ("shoulder_l", "arm_stretch_l", "forearm_stretch_l", "hand_l")
RIGHT_ARM_CHAIN = ("shoulder_r", "arm_stretch_r", "forearm_stretch_r", "hand_r")

FINGER_LAYOUT = {
    "thumb": {
        "avp": ("thumbKnuckle", "thumbIntermediateBase", "thumbIntermediateTip", "thumbTip"),
        "robot": ("thumb1", "thumb2", "thumb3"),
    },
    "index": {
        "avp": ("indexMetacarpal", "indexKnuckle", "indexIntermediateBase", "indexIntermediateTip", "indexTip"),
        "robot": ("index1_base", "index1", "index2", "index3"),
    },
    "middle": {
        "avp": ("middleMetacarpal", "middleKnuckle", "middleIntermediateBase", "middleIntermediateTip", "middleTip"),
        "robot": ("middle1_base", "middle1", "middle2", "middle3"),
    },
    "ring": {
        "avp": ("ringMetacarpal", "ringKnuckle", "ringIntermediateBase", "ringIntermediateTip", "ringTip"),
        "robot": ("ring1_base", "ring1", "ring2", "ring3"),
    },
    "little": {
        "avp": ("littleMetacarpal", "littleKnuckle", "littleIntermediateBase", "littleIntermediateTip", "littleTip"),
        "robot": ("pinky1_base", "pinky1", "pinky2", "pinky3"),
    },
}


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


def load_urdf_joint_specs(urdf_path: Path) -> dict[str, UrdfJointSpec]:
    root = ET.parse(urdf_path).getroot()
    specs: dict[str, UrdfJointSpec] = {}
    for joint_el in root.findall("joint"):
        joint_name = joint_el.attrib.get("name")
        if not joint_name:
            continue
        parent_el = joint_el.find("parent")
        child_el = joint_el.find("child")
        limit_el = joint_el.find("limit")
        lower = float(limit_el.attrib.get("lower", str(-math.pi))) if limit_el is not None else -math.pi
        upper = float(limit_el.attrib.get("upper", str(math.pi))) if limit_el is not None else math.pi
        specs[joint_name] = UrdfJointSpec(
            name=joint_name,
            parent_link=parent_el.attrib["link"],
            child_link=child_el.attrib["link"],
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
            raise KeyError(f"Could not find joint chain from {base_link} to {tip_link}")
        chain.append(spec.name)
        current_link = spec.parent_link
    chain.reverse()
    return tuple(chain)


class LandauUpperBodyRetargeter:
    def __init__(
        self,
        *,
        urdf_path: Path,
        skeleton_json_path: Path,
        snapshot_path: Path,
        hand_retargeting_client=None,
    ) -> None:
        self.urdf_path = Path(urdf_path).resolve()
        self.skeleton_json_path = Path(skeleton_json_path).resolve()
        self.snapshot_path = Path(snapshot_path).expanduser().resolve()
        self.hand_retargeting_client = hand_retargeting_client
        self.last_hand_targets = {"landau": {}, "h1_2": {}}

        self.records = load_skeleton_records(self.skeleton_json_path)
        self.records_by_name = {record.name: record for record in self.records}
        self.rest_local = [record.local_matrix.copy() for record in self.records]
        self.rest_world = world_matrices_from_local(self.records, self.rest_local)
        self.rest_world_by_name = {
            record.name: self.rest_world[record.index]
            for record in self.records
        }
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

        self.joint_specs = load_urdf_joint_specs(self.urdf_path)
        self.left_chain = find_joint_chain(self.urdf_path, ARM_BASE_LINK, LEFT_ARM_TIP)
        self.right_chain = find_joint_chain(self.urdf_path, ARM_BASE_LINK, RIGHT_ARM_TIP)

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

    def _base_relative_world_map(self, pose_by_name: dict[str, float]) -> dict[str, np.ndarray]:
        local_matrices = apply_joint_positions_to_local_matrices(self.records, pose_by_name)
        world_matrices = world_matrices_from_local(self.records, local_matrices)
        return {
            record.name: self.base_world_inv @ world_matrices[record.index]
            for record in self.records
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
        for record in self.records:
            name = record.name
            if not name.endswith(f"_{suffix}") or name == hand_name:
                continue
            local_positions[name] = (hand_world_inv @ self.rest_world_by_name[name])[:3, 3].copy()
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

    def _retarget_head(self, frame, pose_by_name: dict[str, float]) -> None:
        head_world = _tracking_matrix_world(frame.get("head"))
        if head_world is None:
            return

        forward_axis = _normalize(head_world[:3, 1])
        forward_xy = math.sqrt(float(forward_axis[0] ** 2 + forward_axis[1] ** 2))
        pitch = math.atan2(float(forward_axis[2]), max(forward_xy, 1.0e-6))
        pose_by_name["neck_x"] = self._clamp_joint("neck_x", 0.55 * pitch)
        pose_by_name["head_x"] = self._clamp_joint("head_x", 0.45 * pitch)

    def _clamp_hand_base_position(self, side: str, target_pos: np.ndarray) -> np.ndarray:
        clipped = np.asarray(target_pos, dtype=float).copy()
        target_radius = float(np.linalg.norm(clipped))
        radius_cap = 1.15 * self.rest_hand_base_radius[side]
        if target_radius > radius_cap and target_radius > 1.0e-8:
            clipped *= radius_cap / target_radius
        return clipped

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

        # The current Landau arm chain only exposes 4 DOF from torso to hand.
        # Full-pose IK is over-constrained there, so prefer position-only CCD.
        if solver.dof >= 6:
            solution = solver.ik(
                target_pos,
                self.rest_hand_base_rotation[side][:3, :3],
                seed_jnt_values=seed,
            )
            if solution is not None:
                solved_pos = self._arm_tip_base_position(side, np.asarray(solution, dtype=float))
                if float(np.linalg.norm(solved_pos - target_pos)) > 3.0e-2:
                    solution = None
        if solution is None:
            solution = self._solve_arm_ccd(side, target_pos, seed)

        if side == "left":
            self.left_seed = np.asarray(solution, dtype=float)
        else:
            self.right_seed = np.asarray(solution, dtype=float)

        for joint_name, joint_value in zip(chain, solution, strict=False):
            pose_by_name[joint_name] = self._clamp_joint(joint_name, float(joint_value))

        suffix = _side_suffix(side)
        wrist_roll = frame.get(f"{side}_wrist_roll")
        if wrist_roll is not None:
            pose_by_name[f"arm_twist_{suffix}"] = self._clamp_joint(f"arm_twist_{suffix}", 0.35 * float(wrist_roll))
            pose_by_name[f"forearm_twist_{suffix}"] = self._clamp_joint(
                f"forearm_twist_{suffix}",
                0.65 * float(wrist_roll),
            )

    def _solve_arm_ccd(self, side: str, target_pos: np.ndarray, seed: np.ndarray) -> np.ndarray:
        chain = self.left_chain if side == "left" else self.right_chain
        hand_name = f"hand_{_side_suffix(side)}"
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
                axis_world = _normalize(joint_transform[:3, :3] @ self.records_by_name[joint_name].axis)
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
        avp_names = FINGER_LAYOUT[chain_key]["avp"]
        robot_roots = FINGER_LAYOUT[chain_key]["robot"]
        points = [local_positions[name] for name in avp_names]
        base_vec = points[1] - points[0]
        seg_1 = points[2] - points[1]
        seg_2 = points[3] - points[2]
        seg_3 = points[4] - points[3]

        spread_name = f"{robot_roots[0]}_{suffix}"
        flex_names = tuple(f"{robot_name}_{suffix}" for robot_name in robot_roots[1:])

        spread = 0.75 * math.atan2(float(base_vec[0]), max(abs(float(base_vec[1])), 1.0e-6))
        if chain_key == "ring":
            spread *= 0.85
        elif chain_key == "little":
            spread *= 0.75
        elif chain_key == "middle":
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
        for chain_key in ("index", "middle", "ring", "little"):
            self._apply_finger_chain(
                suffix=suffix,
                chain_key=chain_key,
                local_positions=local_positions,
                pose_by_name=pose_by_name,
            )

    def retarget_frame(self, frame) -> dict[str, float]:
        if frame is None:
            self.last_hand_targets = {"landau": {}, "h1_2": {}}
            return {}

        pose_by_name: dict[str, float] = {}
        self._retarget_head(frame, pose_by_name)
        self._solve_arm("left", frame, pose_by_name)
        self._solve_arm("right", frame, pose_by_name)
        if self.hand_retargeting_client is None:
            self.last_hand_targets = {"landau": {}, "h1_2": {}}
            self._retarget_fingers("left", frame, pose_by_name)
            self._retarget_fingers("right", frame, pose_by_name)
            return pose_by_name

        try:
            self.last_hand_targets = self.hand_retargeting_client.retarget_frame(frame)
        except Exception as exc:
            print(
                f"[AVP] Dex hand retargeting failed, falling back to heuristic finger mapping: {exc}",
                flush=True,
            )
            self.hand_retargeting_client = None
            self.last_hand_targets = {"landau": {}, "h1_2": {}}
            self._retarget_fingers("left", frame, pose_by_name)
            self._retarget_fingers("right", frame, pose_by_name)
            return pose_by_name

        pose_by_name.update(self.last_hand_targets.get("landau", {}))
        return pose_by_name

    def h1_2_hand_pose_overrides(self) -> dict[str, float]:
        return {
            joint_name: float(value)
            for joint_name, value in self.last_hand_targets.get("h1_2", {}).items()
        }

    def g1_hand_pose_overrides(self) -> dict[str, float]:
        return self.h1_2_hand_pose_overrides()
