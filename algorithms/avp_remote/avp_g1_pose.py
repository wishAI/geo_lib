from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np

from urdf_kinematics import load_joint_limits, world_map_from_urdf_pose

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

def estimate_urdf_root_height(
    urdf_path: Path,
    pose_by_name: Mapping[str, float] | None = None,
    clearance: float = 0.02,
) -> float:
    world = world_map_from_urdf_pose(urdf_path, pose_by_name)
    min_z = min(float(matrix[2, 3]) for matrix in world.values())
    return -min_z + clearance


def _pose_value(pose_by_name: Mapping[str, float], joint_name: str) -> float:
    return float(pose_by_name.get(joint_name, 0.0))


def _chain_average(
    pose_by_name: Mapping[str, float],
    joint_names: Sequence[str],
    weights: Sequence[float] | None = None,
) -> float:
    if not joint_names:
        return 0.0
    if weights is None:
        weights = tuple(1.0 for _ in joint_names)
    total_weight = max(sum(float(weight) for weight in weights), 1.0e-6)
    return sum(
        float(weight) * _pose_value(pose_by_name, joint_name)
        for joint_name, weight in zip(joint_names, weights, strict=False)
    ) / total_weight


def _clamp_pose(
    pose_by_name: dict[str, float],
    joint_limits: Mapping[str, tuple[float, float]] | None,
) -> dict[str, float]:
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


def _lateral_sign(side_name: str) -> float:
    return 1.0 if side_name == "left" else -1.0


def _robot_hand_prefix(side_name: str) -> str:
    return "L" if side_name == "left" else "R"


def map_landau_pose_to_h1_2_pose(
    landau_pose: Mapping[str, float],
    *,
    hand_pose_override: Mapping[str, float] | None = None,
    joint_limits: Mapping[str, tuple[float, float]] | None = None,
) -> dict[str, float]:
    h1_pose: dict[str, float] = {}
    h1_pose["torso_joint"] = TORSO_YAW_SCALE * (
        _pose_value(landau_pose, "shoulder_l") - _pose_value(landau_pose, "shoulder_r")
    )

    for side_name, suffix in (("left", "l"), ("right", "r")):
        prefix = f"{side_name}_"
        hand_prefix = _robot_hand_prefix(side_name)
        lateral_sign = _lateral_sign(side_name)
        shoulder = _pose_value(landau_pose, f"shoulder_{suffix}")
        arm = _pose_value(landau_pose, f"arm_stretch_{suffix}")
        elbow = _pose_value(landau_pose, f"forearm_stretch_{suffix}")
        hand = _pose_value(landau_pose, f"hand_{suffix}")
        arm_twist = _pose_value(landau_pose, f"arm_twist_{suffix}")
        forearm_twist = _pose_value(landau_pose, f"forearm_twist_{suffix}")

        # Keep a stable presentation pose even when AVP shoulder values are sparse,
        # then blend the solved Landau arm motion on top.
        h1_pose[f"{prefix}shoulder_pitch_joint"] = (
            DISPLAY_SHOULDER_PITCH_BIAS
            + SHOULDER_PITCH_ARM_SCALE * arm
            + SHOULDER_PITCH_SHOULDER_SCALE * shoulder
        )
        h1_pose[f"{prefix}shoulder_roll_joint"] = lateral_sign * (
            DISPLAY_SHOULDER_ROLL_BIAS + SHOULDER_ROLL_SCALE * shoulder
        )
        h1_pose[f"{prefix}shoulder_yaw_joint"] = -lateral_sign * (
            DISPLAY_SHOULDER_YAW_BIAS + SHOULDER_YAW_TWIST_SCALE * arm_twist
        )
        h1_pose[f"{prefix}elbow_pitch_joint"] = max(
            MIN_DISPLAY_ELBOW_PITCH,
            DISPLAY_ELBOW_PITCH_BIAS + ELBOW_PITCH_STRETCH_SCALE * elbow,
        )
        h1_pose[f"{prefix}elbow_roll_joint"] = ELBOW_ROLL_TWIST_SCALE * forearm_twist
        h1_pose[f"{prefix}wrist_pitch_joint"] = -WRIST_PITCH_SCALE * hand
        h1_pose[f"{prefix}wrist_yaw_joint"] = WRIST_YAW_TWIST_SCALE * arm_twist + WRIST_YAW_HAND_SCALE * hand

        h1_pose[f"{hand_prefix}_thumb_proximal_yaw_joint"] = (
            THUMB_YAW_BIAS + THUMB_YAW_SCALE * _pose_value(landau_pose, f"thumb1_{suffix}")
        )
        h1_pose[f"{hand_prefix}_thumb_proximal_pitch_joint"] = THUMB_PITCH_SCALE * _chain_average(
            landau_pose,
            (f"thumb2_{suffix}", f"thumb3_{suffix}"),
            weights=(0.45, 0.55),
        )
        h1_pose[f"{hand_prefix}_index_proximal_joint"] = FINGER_FLEX_SCALE * _chain_average(
            landau_pose,
            (f"index1_{suffix}", f"index2_{suffix}", f"index3_{suffix}"),
            weights=(0.45, 0.33, 0.22),
        )
        h1_pose[f"{hand_prefix}_middle_proximal_joint"] = FINGER_FLEX_SCALE * _chain_average(
            landau_pose,
            (f"middle1_{suffix}", f"middle2_{suffix}", f"middle3_{suffix}"),
            weights=(0.45, 0.33, 0.22),
        )
        h1_pose[f"{hand_prefix}_ring_proximal_joint"] = FINGER_FLEX_SCALE * _chain_average(
            landau_pose,
            (f"ring1_{suffix}", f"ring2_{suffix}", f"ring3_{suffix}"),
            weights=(0.45, 0.33, 0.22),
        )
        h1_pose[f"{hand_prefix}_pinky_proximal_joint"] = FINGER_FLEX_SCALE * _chain_average(
            landau_pose,
            (f"pinky1_{suffix}", f"pinky2_{suffix}", f"pinky3_{suffix}"),
            weights=(0.45, 0.33, 0.22),
        )

    if hand_pose_override is not None:
        for joint_name, joint_value in hand_pose_override.items():
            h1_pose[joint_name] = float(joint_value)

    return _clamp_pose(h1_pose, joint_limits)


def h1_2_hand_pose_summary(h1_pose: Mapping[str, float], side: str) -> dict[str, tuple[float, ...]]:
    prefix = _robot_hand_prefix(side)
    return {
        "thumb": (
            float(h1_pose.get(f"{prefix}_thumb_proximal_yaw_joint", 0.0)),
            float(h1_pose.get(f"{prefix}_thumb_proximal_pitch_joint", 0.0)),
        ),
        "index": (float(h1_pose.get(f"{prefix}_index_proximal_joint", 0.0)),),
        "middle": (float(h1_pose.get(f"{prefix}_middle_proximal_joint", 0.0)),),
        "ring": (float(h1_pose.get(f"{prefix}_ring_proximal_joint", 0.0)),),
        "pinky": (float(h1_pose.get(f"{prefix}_pinky_proximal_joint", 0.0)),),
    }


def map_landau_pose_to_g1_pose(
    landau_pose: Mapping[str, float],
    *,
    hand_pose_override: Mapping[str, float] | None = None,
    joint_limits: Mapping[str, tuple[float, float]] | None = None,
) -> dict[str, float]:
    return map_landau_pose_to_h1_2_pose(
        landau_pose,
        hand_pose_override=hand_pose_override,
        joint_limits=joint_limits,
    )


def g1_hand_pose_summary(g1_pose: Mapping[str, float], side: str) -> dict[str, tuple[float, ...]]:
    return h1_2_hand_pose_summary(g1_pose, side)
