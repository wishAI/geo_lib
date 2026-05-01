from __future__ import annotations

import math
from typing import Any


def default_walking_pose_thresholds(stage: str, *, target_distance_m: float) -> dict[str, float | int]:
    distance = max(float(target_distance_m), 0.0)
    contact_switch_floor = max(3, int(math.ceil(distance * 1.2)))
    if stage == "game":
        return {
            "min_control_root_height": 0.153,
            "min_single_support_ratio": 0.10,
            "max_double_support_ratio": 0.92,
            "max_flight_ratio": 0.20,
            "min_contact_switches_per_side": contact_switch_floor,
            "min_swing_clearance": 0.015,
            "min_touchdown_step_length_mean": 0.05,
            "min_touchdown_root_straddle_mean": 0.01,
            "max_non_support_contact_steps": max(2, int(math.ceil(distance * 0.2))),
        }
    if stage == "fwd_yaw":
        return {
            "min_control_root_height": 0.17,
            "min_single_support_ratio": 0.06,
            "max_double_support_ratio": 0.88,
            "max_flight_ratio": 0.12,
            "min_contact_switches_per_side": contact_switch_floor,
            "min_swing_clearance": 0.02,
            "min_touchdown_step_length_mean": 0.04,
            "min_touchdown_root_straddle_mean": 0.01,
            "max_non_support_contact_steps": max(40, int(math.ceil(distance * 18.0))),
        }
    if stage in {"fwd_only", "full"}:
        return {
            "min_control_root_height": 0.17,
            "min_single_support_ratio": 0.08,
            "max_double_support_ratio": 0.78,
            "max_flight_ratio": 0.08,
            "min_contact_switches_per_side": contact_switch_floor,
            "min_swing_clearance": 0.025,
            "min_touchdown_step_length_mean": 0.05,
            "min_touchdown_root_straddle_mean": 0.01,
            "max_non_support_contact_steps": max(20, int(math.ceil(distance * 12.0))),
            "min_forward_displacement": max(0.8, distance * 0.9),
        }
    return {
        "min_control_root_height": 0.17,
        "min_single_support_ratio": 0.08,
        "max_double_support_ratio": 0.80,
        "max_flight_ratio": 0.08,
        "min_contact_switches_per_side": contact_switch_floor,
        "min_swing_clearance": 0.025,
        "min_touchdown_step_length_mean": 0.05,
        "min_touchdown_root_straddle_mean": 0.01,
        "max_non_support_contact_steps": max(20, int(math.ceil(distance * 12.0))),
    }


def walking_pose_failures(metrics: dict[str, Any], thresholds: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    if "min_touchdown_step_length" in thresholds and "min_touchdown_step_length_mean" not in thresholds:
        thresholds = dict(thresholds)
        thresholds["min_touchdown_step_length_mean"] = thresholds["min_touchdown_step_length"]
    single_support_ratio = float(metrics.get("single_support_ratio", 0.0))
    raw_min_control_root_height = metrics.get("min_control_root_height", 0.0)
    min_control_root_height = float(raw_min_control_root_height or 0.0)
    if min_control_root_height < float(thresholds["min_control_root_height"]):
        failures.append(
            "min_control_root_height "
            f"{min_control_root_height:.4f} < {float(thresholds['min_control_root_height']):.4f}"
        )
    if single_support_ratio < float(thresholds["min_single_support_ratio"]):
        failures.append(
            "single_support_ratio "
            f"{single_support_ratio:.4f} < {float(thresholds['min_single_support_ratio']):.4f}"
        )

    double_support_ratio = float(metrics.get("double_support_ratio", 1.0))
    if double_support_ratio > float(thresholds["max_double_support_ratio"]):
        failures.append(
            "double_support_ratio "
            f"{double_support_ratio:.4f} > {float(thresholds['max_double_support_ratio']):.4f}"
        )

    flight_ratio = float(metrics.get("flight_ratio", 1.0))
    if flight_ratio > float(thresholds["max_flight_ratio"]):
        failures.append(f"flight_ratio {flight_ratio:.4f} > {float(thresholds['max_flight_ratio']):.4f}")

    min_contact_switches = int(thresholds["min_contact_switches_per_side"])
    for side_name, count in zip(("left", "right"), metrics.get("contact_switches", (0, 0)), strict=False):
        if int(count) < min_contact_switches:
            failures.append(f"{side_name}_contact_switches {int(count)} < {min_contact_switches}")

    min_swing_clearance = float(thresholds["min_swing_clearance"])
    for side_name, clearance in zip(("left", "right"), metrics.get("swing_clearance", (0.0, 0.0)), strict=False):
        if float(clearance) < min_swing_clearance:
            failures.append(f"{side_name}_swing_clearance {float(clearance):.4f} < {min_swing_clearance:.4f}")

    min_touchdown_step_length = float(thresholds["min_touchdown_step_length_mean"])
    for side_name, step_length in zip(
        ("left", "right"),
        metrics.get("touchdown_step_length_mean", (0.0, 0.0)),
        strict=False,
    ):
        if float(step_length) < min_touchdown_step_length:
            failures.append(
                f"{side_name}_touchdown_step_length_mean {float(step_length):.4f} < {min_touchdown_step_length:.4f}"
            )

    min_touchdown_root_straddle = float(thresholds["min_touchdown_root_straddle_mean"])
    for side_name, root_straddle in zip(
        ("left", "right"),
        metrics.get("touchdown_root_straddle_mean", (0.0, 0.0)),
        strict=False,
    ):
        if float(root_straddle) < min_touchdown_root_straddle:
            failures.append(
                f"{side_name}_touchdown_root_straddle_mean {float(root_straddle):.4f} < {min_touchdown_root_straddle:.4f}"
            )

    max_non_support_contact_steps = int(thresholds["max_non_support_contact_steps"])
    non_support_steps = int(metrics.get("non_support_contact_step_sum", 0))
    if non_support_steps > max_non_support_contact_steps:
        failures.append(f"non_support_contact_step_sum {non_support_steps} > {max_non_support_contact_steps}")

    if "min_forward_displacement" in thresholds:
        measured_forward_displacement = float(metrics.get("forward_displacement", 0.0))
        min_forward_displacement = float(thresholds["min_forward_displacement"])
        if measured_forward_displacement < min_forward_displacement:
            failures.append(
                f"forward_displacement {measured_forward_displacement:.4f} < {min_forward_displacement:.4f}"
            )

    return failures
