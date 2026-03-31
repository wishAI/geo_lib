from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class VelocityTrackingSample:
    commanded_vx: float
    commanded_vy: float
    commanded_yaw_rate: float
    actual_vx: float
    actual_vy: float
    actual_yaw_rate: float


def track_lin_vel_xy_yaw_frame_exp(sample: VelocityTrackingSample, std: float = 0.5) -> float:
    squared_error = (sample.commanded_vx - sample.actual_vx) ** 2 + (sample.commanded_vy - sample.actual_vy) ** 2
    return math.exp(-squared_error / (std * std))


def track_ang_vel_z_world_exp(sample: VelocityTrackingSample, std: float = 0.5) -> float:
    squared_error = (sample.commanded_yaw_rate - sample.actual_yaw_rate) ** 2
    return math.exp(-squared_error / (std * std))


def flat_orientation_l2(projected_gravity_xy_norm: float) -> float:
    return projected_gravity_xy_norm * projected_gravity_xy_norm


def action_rate_l2(current_action: tuple[float, ...], previous_action: tuple[float, ...]) -> float:
    if len(current_action) != len(previous_action):
        raise ValueError("Action vectors must have matching dimensions.")
    return sum((cur - prev) ** 2 for cur, prev in zip(current_action, previous_action))


def joint_deviation_l1(current_positions: dict[str, float], default_positions: dict[str, float], joint_names: tuple[str, ...]) -> float:
    return sum(abs(current_positions.get(name, 0.0) - default_positions.get(name, 0.0)) for name in joint_names)


@dataclass(frozen=True)
class RewardValidationReport:
    lin_tracking_perfect: float
    lin_tracking_bad: float
    yaw_tracking_perfect: float
    yaw_tracking_bad: float
    orientation_flat: float
    orientation_tilted: float
    action_rate_zero: float
    action_rate_large: float


def build_validation_report() -> RewardValidationReport:
    perfect = VelocityTrackingSample(0.8, -0.2, 0.4, 0.8, -0.2, 0.4)
    poor = VelocityTrackingSample(0.8, -0.2, 0.4, -0.1, 0.5, -0.6)
    return RewardValidationReport(
        lin_tracking_perfect=track_lin_vel_xy_yaw_frame_exp(perfect),
        lin_tracking_bad=track_lin_vel_xy_yaw_frame_exp(poor),
        yaw_tracking_perfect=track_ang_vel_z_world_exp(perfect),
        yaw_tracking_bad=track_ang_vel_z_world_exp(poor),
        orientation_flat=flat_orientation_l2(0.0),
        orientation_tilted=flat_orientation_l2(0.4),
        action_rate_zero=action_rate_l2((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        action_rate_large=action_rate_l2((0.6, -0.4, 0.2), (0.0, 0.0, 0.0)),
    )


def validate_reward_report(report: RewardValidationReport) -> None:
    if not report.lin_tracking_perfect > report.lin_tracking_bad:
        raise AssertionError("Linear velocity reward must prefer perfect tracking.")
    if not report.yaw_tracking_perfect > report.yaw_tracking_bad:
        raise AssertionError("Yaw velocity reward must prefer perfect tracking.")
    if report.orientation_flat != 0.0:
        raise AssertionError("Flat orientation penalty must be zero.")
    if not report.orientation_tilted > report.orientation_flat:
        raise AssertionError("Tilted orientation must have a larger penalty than flat orientation.")
    if report.action_rate_zero != 0.0:
        raise AssertionError("Zero action delta must have zero penalty.")
    if not report.action_rate_large > report.action_rate_zero:
        raise AssertionError("Large action deltas must have a larger penalty than zero deltas.")
