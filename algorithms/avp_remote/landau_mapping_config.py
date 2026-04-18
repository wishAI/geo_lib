from __future__ import annotations

from dataclasses import dataclass


# Pose keys in this file are Landau child-link names, not URDF joint names.
# `scale < 0.0` flips a joint sign without changing the rest of the retarget logic.

ARM_BASE_LINK = "spine_03_x"
HEAD_LINK = "head_x"

LEFT_ARM_TIP = "hand_l"
RIGHT_ARM_TIP = "hand_r"

LEFT_ARM_CHAIN = ("shoulder_l", "arm_stretch_l", "arm_twist_l", "forearm_stretch_l", "forearm_twist_l", "hand_l")
RIGHT_ARM_CHAIN = ("shoulder_r", "arm_stretch_r", "arm_twist_r", "forearm_stretch_r", "forearm_twist_r", "hand_r")

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

# AVP does not currently solve Landau legs. Keep these pose keys explicit so they
# are reset every frame instead of implicitly relying on missing-key behavior.
UNTRACKED_POSE_DEFAULTS = {
    "thigh_stretch_l": 0.0,
    "left_hip_roll_link": 0.0,
    "thigh_twist_l": 0.0,
    "leg_stretch_l": 0.0,
    "leg_twist_l": 0.0,
    "foot_l": 0.0,
    "toes_01_l": 0.0,
    "thigh_stretch_r": 0.0,
    "right_hip_roll_link": 0.0,
    "thigh_twist_r": 0.0,
    "leg_stretch_r": 0.0,
    "leg_twist_r": 0.0,
    "foot_r": 0.0,
    "toes_01_r": 0.0,
}


@dataclass(frozen=True)
class JointOutputRule:
    source: str
    formula: str
    scale: float = 1.0
    bias: float = 0.0


HEAD_OUTPUT_RULES = {
    "neck_x": JointOutputRule(source="head", formula="head pitch", scale=0.55),
    "head_x": JointOutputRule(source="head", formula="head pitch", scale=0.45),
}

ARM_OUTPUT_RULES = {
    joint_name: JointOutputRule(source=f"{side} arm IK", formula=f"{side} arm solver output")
    for side, chain in (("left", LEFT_ARM_CHAIN), ("right", RIGHT_ARM_CHAIN))
    for joint_name in chain
}

FINGER_OUTPUT_RULES = {}
for suffix in ("l", "r"):
    side_name = "left" if suffix == "l" else "right"
    FINGER_OUTPUT_RULES.update(
        {
            f"thumb1_{suffix}": JointOutputRule(
                source=f"{side_name} thumb",
                formula="atan2(base_vec.x, abs(base_vec.y))",
            ),
            f"thumb2_{suffix}": JointOutputRule(
                source=f"{side_name} thumb",
                formula="angle(seg_1, seg_2)",
                scale=1.15,
            ),
            f"thumb3_{suffix}": JointOutputRule(
                source=f"{side_name} thumb",
                formula="angle(seg_2, seg_3)",
                scale=1.05,
            ),
            f"index1_base_{suffix}": JointOutputRule(
                source=f"{side_name} index",
                formula="atan2(base_vec.x, abs(base_vec.y))",
                scale=0.75,
            ),
            f"index1_{suffix}": JointOutputRule(
                source=f"{side_name} index",
                formula="angle(base_vec, seg_1)",
                scale=1.10,
            ),
            f"index2_{suffix}": JointOutputRule(
                source=f"{side_name} index",
                formula="angle(seg_1, seg_2)",
                scale=1.15,
            ),
            f"index3_{suffix}": JointOutputRule(
                source=f"{side_name} index",
                formula="angle(seg_2, seg_3)",
                scale=1.05,
            ),
            f"middle1_base_{suffix}": JointOutputRule(
                source=f"{side_name} middle",
                formula="atan2(base_vec.x, abs(base_vec.y))",
                scale=0.75 * 0.35,
            ),
            f"middle1_{suffix}": JointOutputRule(
                source=f"{side_name} middle",
                formula="angle(base_vec, seg_1)",
                scale=1.10,
            ),
            f"middle2_{suffix}": JointOutputRule(
                source=f"{side_name} middle",
                formula="angle(seg_1, seg_2)",
                scale=1.15,
            ),
            f"middle3_{suffix}": JointOutputRule(
                source=f"{side_name} middle",
                formula="angle(seg_2, seg_3)",
                scale=1.05,
            ),
            f"ring1_base_{suffix}": JointOutputRule(
                source=f"{side_name} ring",
                formula="atan2(base_vec.x, abs(base_vec.y))",
                scale=0.75 * 0.85,
            ),
            f"ring1_{suffix}": JointOutputRule(
                source=f"{side_name} ring",
                formula="angle(base_vec, seg_1)",
                scale=1.10,
            ),
            f"ring2_{suffix}": JointOutputRule(
                source=f"{side_name} ring",
                formula="angle(seg_1, seg_2)",
                scale=1.15,
            ),
            f"ring3_{suffix}": JointOutputRule(
                source=f"{side_name} ring",
                formula="angle(seg_2, seg_3)",
                scale=1.05,
            ),
            f"pinky1_base_{suffix}": JointOutputRule(
                source=f"{side_name} little",
                formula="atan2(base_vec.x, abs(base_vec.y))",
                scale=0.75 * 0.75,
            ),
            f"pinky1_{suffix}": JointOutputRule(
                source=f"{side_name} little",
                formula="angle(base_vec, seg_1)",
                scale=1.10,
            ),
            f"pinky2_{suffix}": JointOutputRule(
                source=f"{side_name} little",
                formula="angle(seg_1, seg_2)",
                scale=1.15,
            ),
            f"pinky3_{suffix}": JointOutputRule(
                source=f"{side_name} little",
                formula="angle(seg_2, seg_3)",
                scale=1.05,
            ),
        }
    )


JOINT_OUTPUT_RULES = {
    **HEAD_OUTPUT_RULES,
    **ARM_OUTPUT_RULES,
    **FINGER_OUTPUT_RULES,
}


def output_rule_for_joint(joint_name: str) -> JointOutputRule | None:
    return JOINT_OUTPUT_RULES.get(joint_name)


def apply_output_rule(joint_name: str, raw_value: float) -> float:
    value = float(raw_value)
    rule = output_rule_for_joint(joint_name)
    if rule is None:
        return value
    return float(rule.scale * value + rule.bias)
