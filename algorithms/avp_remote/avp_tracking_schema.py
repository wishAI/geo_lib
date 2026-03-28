import numpy as np

HAND_JOINT_NAMES = (
    "wrist",
    "thumbKnuckle",
    "thumbIntermediateBase",
    "thumbIntermediateTip",
    "thumbTip",
    "indexMetacarpal",
    "indexKnuckle",
    "indexIntermediateBase",
    "indexIntermediateTip",
    "indexTip",
    "middleMetacarpal",
    "middleKnuckle",
    "middleIntermediateBase",
    "middleIntermediateTip",
    "middleTip",
    "ringMetacarpal",
    "ringKnuckle",
    "ringIntermediateBase",
    "ringIntermediateTip",
    "ringTip",
    "littleMetacarpal",
    "littleKnuckle",
    "littleIntermediateBase",
    "littleIntermediateTip",
    "littleTip",
    "forearmWrist",
    "forearmArm",
)


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
    getter = getattr(data, "get", None)
    if callable(getter):
        try:
            return getter(key, default)
        except Exception:
            try:
                return getter(key)
            except Exception:
                pass
    return default


def as_mat4(value):
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


def as_mat4_stack(value, expected_count=None):
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


def as_float(value):
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _extract_hand_stack_from_attributes(hand):
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
        mat = as_mat4(joint)
        if mat is None:
            return None
        joints.append(mat)
    return np.stack(joints, axis=0)


def extract_hand_stack(data, side):
    hand_obj = _get_value(data, side)
    mats = _extract_hand_stack_from_attributes(hand_obj)
    if mats is not None:
        return mats
    return as_mat4_stack(_get_value(data, f"{side}_arm"), expected_count=len(HAND_JOINT_NAMES))


def extract_tracking_frame(data):
    left_arm = extract_hand_stack(data, "left")
    right_arm = extract_hand_stack(data, "right")

    left_hand = _get_value(data, "left")
    right_hand = _get_value(data, "right")

    frame = {
        "head": as_mat4(_get_value(data, "head")),
        "left_arm": left_arm,
        "right_arm": right_arm,
        "left_wrist": as_mat4(_get_value(data, "left_wrist")),
        "right_wrist": as_mat4(_get_value(data, "right_wrist")),
        "left_pinch_distance": as_float(
            _get_value(data, "left_pinch_distance", as_float(_get_value(left_hand, "pinch_distance")))
        ),
        "right_pinch_distance": as_float(
            _get_value(data, "right_pinch_distance", as_float(_get_value(right_hand, "pinch_distance")))
        ),
        "left_wrist_roll": as_float(
            _get_value(data, "left_wrist_roll", as_float(_get_value(left_hand, "wrist_roll")))
        ),
        "right_wrist_roll": as_float(
            _get_value(data, "right_wrist_roll", as_float(_get_value(right_hand, "wrist_roll")))
        ),
    }

    if frame["left_wrist"] is None and left_arm is not None:
        frame["left_wrist"] = left_arm[0]
    if frame["right_wrist"] is None and right_arm is not None:
        frame["right_wrist"] = right_arm[0]

    return frame


def _mat_or_none_to_list(value):
    return np.asarray(value, dtype=float).tolist() if value is not None else None


def frame_to_payload(frame):
    return {
        "head": _mat_or_none_to_list(frame.get("head")),
        "left_arm": _mat_or_none_to_list(frame.get("left_arm")),
        "right_arm": _mat_or_none_to_list(frame.get("right_arm")),
        "left_wrist": _mat_or_none_to_list(frame.get("left_wrist")),
        "right_wrist": _mat_or_none_to_list(frame.get("right_wrist")),
        "left_pinch_distance": frame.get("left_pinch_distance"),
        "right_pinch_distance": frame.get("right_pinch_distance"),
        "left_wrist_roll": frame.get("left_wrist_roll"),
        "right_wrist_roll": frame.get("right_wrist_roll"),
    }
