from __future__ import annotations

import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from .command_frame import semantic_forward_dir_xy


@dataclass(frozen=True)
class ManualGameCommandCfg:
    backward_scale: float = 0.35
    strafe_scale: float = 0.0
    turn_from_strafe_gain: float = 1.2


@dataclass(frozen=True)
class ObstacleBrakeCfg:
    resolution: float
    size: tuple[float, float]
    ordering: str = "xy"
    near_band: tuple[float, float] = (0.0, 0.2)
    lookahead_band: tuple[float, float] = (0.3, 0.8)
    lateral_limit: float = 0.2
    soft_rise: float = 0.08
    hard_rise: float = 0.18


@dataclass(frozen=True)
class PathFollowerCfg:
    max_forward_speed: float = 0.7
    max_strafe_speed: float = 0.0
    max_yaw_rate: float = 0.9
    yaw_gain: float = 1.6
    strafe_gain: float = 0.8
    arrival_radius: float = 0.35
    slow_radius: float = 1.0
    turn_in_place_angle: float = 0.7


GATE_DIRECTIONS = ("forward", "left", "right", "backward")
PATH_PRESETS = ("target", "gate", "triangle", "square")


def wrap_to_pi(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _normalize_xy(vector: tuple[float, float]) -> tuple[float, float]:
    norm = math.hypot(vector[0], vector[1])
    if norm <= 1.0e-8:
        return (1.0, 0.0)
    return (vector[0] / norm, vector[1] / norm)


def _left_of(forward_dir_xy: tuple[float, float]) -> tuple[float, float]:
    forward_x, forward_y = _normalize_xy(forward_dir_xy)
    return (-forward_y, forward_x)


def build_gate_waypoints(
    *,
    origin_xy: tuple[float, float],
    forward_dir_xy: tuple[float, float],
    distance_m: float,
    direction: str = "forward",
) -> list[tuple[float, float]]:
    normalized_direction = str(direction).strip().lower()
    if normalized_direction not in GATE_DIRECTIONS:
        raise ValueError(f"Unsupported gate direction '{direction}'. Expected one of: {', '.join(GATE_DIRECTIONS)}")
    forward_dir_xy = _normalize_xy(forward_dir_xy)
    left_dir_xy = _left_of(forward_dir_xy)
    heading_by_direction = {
        "forward": forward_dir_xy,
        "backward": (-forward_dir_xy[0], -forward_dir_xy[1]),
        "left": left_dir_xy,
        "right": (-left_dir_xy[0], -left_dir_xy[1]),
    }
    heading_x, heading_y = heading_by_direction[normalized_direction]
    distance = max(float(distance_m), 0.0)
    return [(origin_xy[0] + heading_x * distance, origin_xy[1] + heading_y * distance)]


def build_polygon_waypoints(
    *,
    origin_xy: tuple[float, float],
    forward_dir_xy: tuple[float, float],
    shape: str,
    edge_length_m: float,
    clockwise: bool = False,
    close_loop: bool = True,
) -> list[tuple[float, float]]:
    normalized_shape = str(shape).strip().lower()
    if normalized_shape not in {"triangle", "square"}:
        raise ValueError("Polygon preset must be 'triangle' or 'square'.")

    edge_length = max(float(edge_length_m), 0.0)
    sign = -1.0 if clockwise else 1.0
    forward_dir_xy = _normalize_xy(forward_dir_xy)
    left_dir_xy = _left_of(forward_dir_xy)

    def _compose(forward_scale: float, left_scale: float) -> tuple[float, float]:
        return (
            origin_xy[0] + edge_length * (forward_dir_xy[0] * forward_scale + left_dir_xy[0] * left_scale),
            origin_xy[1] + edge_length * (forward_dir_xy[1] * forward_scale + left_dir_xy[1] * left_scale),
        )

    if normalized_shape == "triangle":
        corners = [
            _compose(1.0, 0.0),
            _compose(0.5, sign * math.sqrt(3.0) * 0.5),
        ]
    else:
        corners = [
            _compose(1.0, 0.0),
            _compose(1.0, sign * 1.0),
            _compose(0.0, sign * 1.0),
        ]
    if close_loop:
        corners.append(tuple(origin_xy))
    return corners


def build_preset_waypoints(
    *,
    origin_xy: tuple[float, float],
    forward_dir_xy: tuple[float, float],
    preset: str,
    distance_m: float = 10.0,
    gate_direction: str = "forward",
    edge_length_m: float = 4.0,
    clockwise: bool = False,
    close_loop: bool = True,
) -> list[tuple[float, float]]:
    normalized_preset = str(preset).strip().lower()
    if normalized_preset == "gate":
        return build_gate_waypoints(
            origin_xy=origin_xy,
            forward_dir_xy=forward_dir_xy,
            distance_m=distance_m,
            direction=gate_direction,
        )
    if normalized_preset in {"triangle", "square"}:
        return build_polygon_waypoints(
            origin_xy=origin_xy,
            forward_dir_xy=forward_dir_xy,
            shape=normalized_preset,
            edge_length_m=edge_length_m,
            clockwise=clockwise,
            close_loop=close_loop,
        )
    if normalized_preset == "target":
        raise ValueError("Preset 'target' requires explicit waypoints and is not generated here.")
    raise ValueError(f"Unsupported path preset '{preset}'. Expected one of: {', '.join(PATH_PRESETS)}")


def manual_game_command(
    raw_command: tuple[float, float, float],
    cfg: ManualGameCommandCfg = ManualGameCommandCfg(),
) -> tuple[float, float, float]:
    forward, strafe, yaw = (float(value) for value in raw_command)
    if forward < 0.0:
        forward *= cfg.backward_scale
    return (
        forward,
        strafe * cfg.strafe_scale,
        yaw + strafe * cfg.turn_from_strafe_gain,
    )


def apply_forward_brake(command: tuple[float, float, float], brake_factor: float) -> tuple[float, float, float]:
    forward, strafe, yaw = (float(value) for value in command)
    clamped_brake = min(max(float(brake_factor), 0.0), 1.0)
    return (forward * clamped_brake, strafe, yaw)


@lru_cache(maxsize=32)
def _grid_xy_positions(
    resolution: float,
    size: tuple[float, float],
    ordering: str,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    if resolution <= 0.0:
        raise ValueError(f"Grid resolution must be positive, got {resolution}.")
    xs = []
    ys = []
    x_values = []
    y_values = []

    x = -size[0] / 2.0
    while x <= size[0] / 2.0 + 1.0e-9:
        x_values.append(round(x, 10))
        x += resolution
    y = -size[1] / 2.0
    while y <= size[1] / 2.0 + 1.0e-9:
        y_values.append(round(y, 10))
        y += resolution

    if ordering == "xy":
        for y_val in y_values:
            for x_val in x_values:
                xs.append(x_val)
                ys.append(y_val)
    elif ordering == "yx":
        for x_val in x_values:
            for y_val in y_values:
                xs.append(x_val)
                ys.append(y_val)
    else:
        raise ValueError(f"Unsupported ordering '{ordering}'. Expected 'xy' or 'yx'.")
    return tuple(xs), tuple(ys)


def front_terrain_rise(height_scan: list[float] | tuple[float, ...], cfg: ObstacleBrakeCfg) -> float:
    xs, ys = _grid_xy_positions(cfg.resolution, cfg.size, cfg.ordering)
    if len(height_scan) != len(xs):
        raise ValueError(
            f"Height scan size mismatch: expected {len(xs)} samples from the configured grid, got {len(height_scan)}."
        )
    near_samples = []
    lookahead_samples = []
    for sample, x_val, y_val in zip(height_scan, xs, ys, strict=True):
        if abs(y_val) > cfg.lateral_limit:
            continue
        if cfg.near_band[0] <= x_val <= cfg.near_band[1]:
            near_samples.append(float(sample))
        if cfg.lookahead_band[0] <= x_val <= cfg.lookahead_band[1]:
            lookahead_samples.append(float(sample))
    if not near_samples or not lookahead_samples:
        return 0.0
    baseline_height = sum(near_samples) / len(near_samples)
    forward_min_height = min(lookahead_samples)
    return max(0.0, baseline_height - forward_min_height)


def obstacle_brake_factor(height_scan: list[float] | tuple[float, ...], cfg: ObstacleBrakeCfg) -> float:
    rise = front_terrain_rise(height_scan, cfg)
    if rise <= cfg.soft_rise:
        return 1.0
    if rise >= cfg.hard_rise:
        return 0.0
    span = max(cfg.hard_rise - cfg.soft_rise, 1.0e-6)
    return 1.0 - (rise - cfg.soft_rise) / span


def load_waypoints(path: str | Path) -> list[tuple[float, float]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("waypoints", [])
    if not isinstance(payload, list):
        raise ValueError("Waypoint file must be a JSON list or a dict with a 'waypoints' list.")
    waypoints: list[tuple[float, float]] = []
    for entry in payload:
        if isinstance(entry, dict):
            x_val = entry.get("x")
            y_val = entry.get("y")
        else:
            x_val, y_val = entry
        waypoints.append((float(x_val), float(y_val)))
    if not waypoints:
        raise ValueError("Waypoint path is empty.")
    return waypoints


def joystick_path_command(
    *,
    robot_xy: tuple[float, float],
    robot_quat_wxyz: tuple[float, float, float, float],
    forward_body_axis: str,
    waypoints: list[tuple[float, float]],
    waypoint_index: int,
    cfg: PathFollowerCfg = PathFollowerCfg(),
) -> tuple[tuple[float, float, float], int, bool, dict[str, Any]]:
    if not waypoints:
        return (0.0, 0.0, 0.0), waypoint_index, True, {"done": True}

    index = min(max(int(waypoint_index), 0), len(waypoints) - 1)
    current_x, current_y = (float(robot_xy[0]), float(robot_xy[1]))
    while index < len(waypoints):
        target_x, target_y = waypoints[index]
        distance = math.hypot(target_x - current_x, target_y - current_y)
        if distance > cfg.arrival_radius or index == len(waypoints) - 1:
            break
        index += 1

    target_x, target_y = waypoints[index]
    delta_x = target_x - current_x
    delta_y = target_y - current_y
    distance = math.hypot(delta_x, delta_y)
    if distance <= cfg.arrival_radius and index == len(waypoints) - 1:
        return (0.0, 0.0, 0.0), index, True, {
            "done": True,
            "distance": distance,
            "target_waypoint_index": index,
        }

    forward_dir_x, forward_dir_y = semantic_forward_dir_xy(forward_body_axis, robot_quat_wxyz)
    left_dir_x, left_dir_y = _left_of((forward_dir_x, forward_dir_y))
    current_heading = math.atan2(forward_dir_y, forward_dir_x)
    target_heading = math.atan2(delta_y, delta_x)
    heading_error = wrap_to_pi(target_heading - current_heading)
    yaw_rate = max(-cfg.max_yaw_rate, min(cfg.max_yaw_rate, cfg.yaw_gain * heading_error))

    forward_error = delta_x * forward_dir_x + delta_y * forward_dir_y
    left_error = delta_x * left_dir_x + delta_y * left_dir_y
    forward_scale = min(distance / max(cfg.slow_radius, 1.0e-6), 1.0)
    heading_alignment = max(math.cos(heading_error), 0.0)
    forward_speed = cfg.max_forward_speed * forward_scale * heading_alignment
    strafe_speed = 0.0
    if cfg.max_strafe_speed > 0.0:
        strafe_speed = max(
            -cfg.max_strafe_speed,
            min(cfg.max_strafe_speed, cfg.strafe_gain * left_error),
        )
    if abs(heading_error) > cfg.turn_in_place_angle:
        forward_speed *= 0.15
        strafe_speed *= 0.35

    return (forward_speed, strafe_speed, yaw_rate), index, False, {
        "done": False,
        "distance": distance,
        "heading_error": heading_error,
        "forward_error": forward_error,
        "left_error": left_error,
        "target_waypoint_index": index,
        "target_waypoint": (target_x, target_y),
    }


def path_follow_command(
    *,
    robot_xy: tuple[float, float],
    robot_quat_wxyz: tuple[float, float, float, float],
    forward_body_axis: str,
    waypoints: list[tuple[float, float]],
    waypoint_index: int,
    cfg: PathFollowerCfg = PathFollowerCfg(),
) -> tuple[tuple[float, float, float], int, bool, dict[str, Any]]:
    return joystick_path_command(
        robot_xy=robot_xy,
        robot_quat_wxyz=robot_quat_wxyz,
        forward_body_axis=forward_body_axis,
        waypoints=waypoints,
        waypoint_index=waypoint_index,
        cfg=cfg,
    )
