"""Canonical USD skeleton joint layout for the current model asset."""

from avp_tracking_schema import HAND_JOINT_NAMES


SKELETON_JOINT_NAMES = (
    "root_x",
    "root_x/spine_01_x",
    "root_x/spine_01_x/spine_02_x",
    "root_x/spine_01_x/spine_02_x/spine_03_x",
    "root_x/spine_01_x/spine_02_x/spine_03_x/neck_x",
    "root_x/spine_01_x/spine_02_x/spine_03_x/neck_x/head_x",
    "root_x/spine_01_x/spine_02_x/spine_03_x/shoulder_r",
    "root_x/spine_01_x/spine_02_x/spine_03_x/shoulder_r/arm_stretch_r",
    "root_x/spine_01_x/spine_02_x/spine_03_x/shoulder_r/arm_stretch_r/arm_twist_r",
    "root_x/spine_01_x/spine_02_x/spine_03_x/shoulder_r/arm_stretch_r/forearm_stretch_r",
    "root_x/spine_01_x/spine_02_x/spine_03_x/shoulder_r/arm_stretch_r/forearm_stretch_r/forearm_twist_r",
    "root_x/spine_01_x/spine_02_x/spine_03_x/shoulder_r/arm_stretch_r/forearm_stretch_r/hand_r",
    "root_x/spine_01_x/spine_02_x/spine_03_x/shoulder_r/arm_stretch_r/forearm_stretch_r/hand_r/thumb1_r",
    "root_x/spine_01_x/spine_02_x/spine_03_x/shoulder_r/arm_stretch_r/forearm_stretch_r/hand_r/thumb1_r/thumb2_r",
    "root_x/spine_01_x/spine_02_x/spine_03_x/shoulder_r/arm_stretch_r/forearm_stretch_r/hand_r/thumb1_r/thumb2_r/thumb3_r",
    "root_x/spine_01_x/spine_02_x/spine_03_x/shoulder_r/arm_stretch_r/forearm_stretch_r/hand_r/index1_base_r",
    "root_x/spine_01_x/spine_02_x/spine_03_x/shoulder_r/arm_stretch_r/forearm_stretch_r/hand_r/index1_base_r/index1_r",
    "root_x/spine_01_x/spine_02_x/spine_03_x/shoulder_r/arm_stretch_r/forearm_stretch_r/hand_r/index1_base_r/index1_r/index2_r",
    "root_x/spine_01_x/spine_02_x/spine_03_x/shoulder_r/arm_stretch_r/forearm_stretch_r/hand_r/index1_base_r/index1_r/index2_r/index3_r",
    "root_x/spine_01_x/spine_02_x/spine_03_x/shoulder_r/arm_stretch_r/forearm_stretch_r/hand_r/middle1_base_r",
    "root_x/spine_01_x/spine_02_x/spine_03_x/shoulder_r/arm_stretch_r/forearm_stretch_r/hand_r/middle1_base_r/middle1_r",
    "root_x/spine_01_x/spine_02_x/spine_03_x/shoulder_r/arm_stretch_r/forearm_stretch_r/hand_r/middle1_base_r/middle1_r/middle2_r",
    "root_x/spine_01_x/spine_02_x/spine_03_x/shoulder_r/arm_stretch_r/forearm_stretch_r/hand_r/middle1_base_r/middle1_r/middle2_r/middle3_r",
    "root_x/spine_01_x/spine_02_x/spine_03_x/shoulder_r/arm_stretch_r/forearm_stretch_r/hand_r/ring1_base_r",
    "root_x/spine_01_x/spine_02_x/spine_03_x/shoulder_r/arm_stretch_r/forearm_stretch_r/hand_r/ring1_base_r/ring1_r",
    "root_x/spine_01_x/spine_02_x/spine_03_x/shoulder_r/arm_stretch_r/forearm_stretch_r/hand_r/ring1_base_r/ring1_r/ring2_r",
    "root_x/spine_01_x/spine_02_x/spine_03_x/shoulder_r/arm_stretch_r/forearm_stretch_r/hand_r/ring1_base_r/ring1_r/ring2_r/ring3_r",
    "root_x/spine_01_x/spine_02_x/spine_03_x/shoulder_r/arm_stretch_r/forearm_stretch_r/hand_r/pinky1_base_r",
    "root_x/spine_01_x/spine_02_x/spine_03_x/shoulder_r/arm_stretch_r/forearm_stretch_r/hand_r/pinky1_base_r/pinky1_r",
    "root_x/spine_01_x/spine_02_x/spine_03_x/shoulder_r/arm_stretch_r/forearm_stretch_r/hand_r/pinky1_base_r/pinky1_r/pinky2_r",
    "root_x/spine_01_x/spine_02_x/spine_03_x/shoulder_r/arm_stretch_r/forearm_stretch_r/hand_r/pinky1_base_r/pinky1_r/pinky2_r/pinky3_r",
    "root_x/spine_01_x/spine_02_x/spine_03_x/shoulder_l",
    "root_x/spine_01_x/spine_02_x/spine_03_x/shoulder_l/arm_stretch_l",
    "root_x/spine_01_x/spine_02_x/spine_03_x/shoulder_l/arm_stretch_l/arm_twist_l",
    "root_x/spine_01_x/spine_02_x/spine_03_x/shoulder_l/arm_stretch_l/forearm_stretch_l",
    "root_x/spine_01_x/spine_02_x/spine_03_x/shoulder_l/arm_stretch_l/forearm_stretch_l/forearm_twist_l",
    "root_x/spine_01_x/spine_02_x/spine_03_x/shoulder_l/arm_stretch_l/forearm_stretch_l/hand_l",
    "root_x/spine_01_x/spine_02_x/spine_03_x/shoulder_l/arm_stretch_l/forearm_stretch_l/hand_l/thumb1_l",
    "root_x/spine_01_x/spine_02_x/spine_03_x/shoulder_l/arm_stretch_l/forearm_stretch_l/hand_l/thumb1_l/thumb2_l",
    "root_x/spine_01_x/spine_02_x/spine_03_x/shoulder_l/arm_stretch_l/forearm_stretch_l/hand_l/thumb1_l/thumb2_l/thumb3_l",
    "root_x/spine_01_x/spine_02_x/spine_03_x/shoulder_l/arm_stretch_l/forearm_stretch_l/hand_l/index1_base_l",
    "root_x/spine_01_x/spine_02_x/spine_03_x/shoulder_l/arm_stretch_l/forearm_stretch_l/hand_l/index1_base_l/index1_l",
    "root_x/spine_01_x/spine_02_x/spine_03_x/shoulder_l/arm_stretch_l/forearm_stretch_l/hand_l/index1_base_l/index1_l/index2_l",
    "root_x/spine_01_x/spine_02_x/spine_03_x/shoulder_l/arm_stretch_l/forearm_stretch_l/hand_l/index1_base_l/index1_l/index2_l/index3_l",
    "root_x/spine_01_x/spine_02_x/spine_03_x/shoulder_l/arm_stretch_l/forearm_stretch_l/hand_l/middle1_base_l",
    "root_x/spine_01_x/spine_02_x/spine_03_x/shoulder_l/arm_stretch_l/forearm_stretch_l/hand_l/middle1_base_l/middle1_l",
    "root_x/spine_01_x/spine_02_x/spine_03_x/shoulder_l/arm_stretch_l/forearm_stretch_l/hand_l/middle1_base_l/middle1_l/middle2_l",
    "root_x/spine_01_x/spine_02_x/spine_03_x/shoulder_l/arm_stretch_l/forearm_stretch_l/hand_l/middle1_base_l/middle1_l/middle2_l/middle3_l",
    "root_x/spine_01_x/spine_02_x/spine_03_x/shoulder_l/arm_stretch_l/forearm_stretch_l/hand_l/ring1_base_l",
    "root_x/spine_01_x/spine_02_x/spine_03_x/shoulder_l/arm_stretch_l/forearm_stretch_l/hand_l/ring1_base_l/ring1_l",
    "root_x/spine_01_x/spine_02_x/spine_03_x/shoulder_l/arm_stretch_l/forearm_stretch_l/hand_l/ring1_base_l/ring1_l/ring2_l",
    "root_x/spine_01_x/spine_02_x/spine_03_x/shoulder_l/arm_stretch_l/forearm_stretch_l/hand_l/ring1_base_l/ring1_l/ring2_l/ring3_l",
    "root_x/spine_01_x/spine_02_x/spine_03_x/shoulder_l/arm_stretch_l/forearm_stretch_l/hand_l/pinky1_base_l",
    "root_x/spine_01_x/spine_02_x/spine_03_x/shoulder_l/arm_stretch_l/forearm_stretch_l/hand_l/pinky1_base_l/pinky1_l",
    "root_x/spine_01_x/spine_02_x/spine_03_x/shoulder_l/arm_stretch_l/forearm_stretch_l/hand_l/pinky1_base_l/pinky1_l/pinky2_l",
    "root_x/spine_01_x/spine_02_x/spine_03_x/shoulder_l/arm_stretch_l/forearm_stretch_l/hand_l/pinky1_base_l/pinky1_l/pinky2_l/pinky3_l",
    "root_x/thigh_stretch_l",
    "root_x/thigh_stretch_l/thigh_twist_l",
    "root_x/thigh_stretch_l/leg_stretch_l",
    "root_x/thigh_stretch_l/leg_stretch_l/foot_l",
    "root_x/thigh_stretch_l/leg_stretch_l/foot_l/toes_01_l",
    "root_x/thigh_stretch_l/leg_stretch_l/leg_twist_l",
    "root_x/thigh_stretch_r",
    "root_x/thigh_stretch_r/thigh_twist_r",
    "root_x/thigh_stretch_r/leg_stretch_r",
    "root_x/thigh_stretch_r/leg_stretch_r/foot_r",
    "root_x/thigh_stretch_r/leg_stretch_r/foot_r/toes_01_r",
    "root_x/thigh_stretch_r/leg_stretch_r/leg_twist_r",
)

SKELETON_JOINT_SET = frozenset(SKELETON_JOINT_NAMES)

LEFT_HAND_ROOT_JOINT = (
    "root_x/spine_01_x/spine_02_x/spine_03_x/shoulder_l/arm_stretch_l/forearm_stretch_l/hand_l"
)
RIGHT_HAND_ROOT_JOINT = (
    "root_x/spine_01_x/spine_02_x/spine_03_x/shoulder_r/arm_stretch_r/forearm_stretch_r/hand_r"
)
LEFT_FOOT_JOINT = "root_x/thigh_stretch_l/leg_stretch_l/foot_l"
RIGHT_FOOT_JOINT = "root_x/thigh_stretch_r/leg_stretch_r/foot_r"

LEFT_HAND_JOINTS = tuple(
    name for name in SKELETON_JOINT_NAMES if name.startswith(LEFT_HAND_ROOT_JOINT)
)
RIGHT_HAND_JOINTS = tuple(
    name for name in SKELETON_JOINT_NAMES if name.startswith(RIGHT_HAND_ROOT_JOINT)
)


def get_missing_expected_joints(joint_names):
    joint_name_set = {str(name) for name in (joint_names or ())}
    return tuple(name for name in SKELETON_JOINT_NAMES if name not in joint_name_set)


def get_unexpected_joints(joint_names):
    joint_name_set = {str(name) for name in (joint_names or ())}
    return tuple(name for name in joint_name_set if name not in SKELETON_JOINT_SET)


def build_empty_future_hand_mapping():
    """Return placeholders for future AVP->USD hand mapping, intentionally unset."""
    return {
        "left": {name: None for name in HAND_JOINT_NAMES},
        "right": {name: None for name in HAND_JOINT_NAMES},
    }
