from __future__ import annotations

import math

from algorithms.urdf_learn_wasd_walk.game_control import (
    GATE_DIRECTIONS,
    ManualGameCommandCfg,
    ObstacleBrakeCfg,
    PathFollowerCfg,
    apply_forward_brake,
    build_gate_waypoints,
    build_polygon_waypoints,
    manual_game_command,
    obstacle_brake_factor,
    path_follow_command,
)


def _flat_height_scan(length: int, base_height: float = -0.2) -> list[float]:
    return [base_height] * length


def test_manual_game_command_turns_strafe_into_yaw_and_softens_reverse() -> None:
    command = manual_game_command(
        (-1.0, 0.5, 0.1),
        ManualGameCommandCfg(backward_scale=0.25, strafe_scale=0.1, turn_from_strafe_gain=1.4),
    )
    assert command == (-0.25, 0.05, 0.7999999999999999)


def test_apply_forward_brake_only_limits_forward_component() -> None:
    assert apply_forward_brake((0.8, 0.0, 0.4), 0.25) == (0.2, 0.0, 0.4)


def test_obstacle_brake_factor_detects_rise_ahead() -> None:
    cfg = ObstacleBrakeCfg(resolution=0.1, size=(1.6, 1.0))
    scan = _flat_height_scan(187)
    # Inject a tall obstacle in the forward center band.
    for index in range(116, 126):
        scan[index] = -0.45
    brake = obstacle_brake_factor(scan, cfg)
    assert brake == 0.0


def test_build_gate_waypoints_uses_robot_forward_frame_for_four_directions() -> None:
    assert GATE_DIRECTIONS == ("forward", "left", "right", "backward")
    assert build_gate_waypoints(
        origin_xy=(1.0, 2.0),
        forward_dir_xy=(0.0, 1.0),
        distance_m=10.0,
        direction="forward",
    ) == [(1.0, 12.0)]
    assert build_gate_waypoints(
        origin_xy=(1.0, 2.0),
        forward_dir_xy=(0.0, 1.0),
        distance_m=10.0,
        direction="left",
    ) == [(-9.0, 2.0)]


def test_build_polygon_waypoints_emits_closed_square_and_triangle_paths() -> None:
    square = build_polygon_waypoints(
        origin_xy=(0.0, 0.0),
        forward_dir_xy=(0.0, 1.0),
        shape="square",
        edge_length_m=2.0,
        close_loop=True,
    )
    triangle = build_polygon_waypoints(
        origin_xy=(0.0, 0.0),
        forward_dir_xy=(0.0, 1.0),
        shape="triangle",
        edge_length_m=2.0,
        close_loop=True,
    )
    assert square[0] == (0.0, 2.0)
    assert square[-1] == (0.0, 0.0)
    assert len(square) == 4
    assert triangle[0] == (0.0, 2.0)
    assert triangle[-1] == (0.0, 0.0)
    assert len(triangle) == 3


def test_path_follow_command_turns_before_driving_forward() -> None:
    command, index, done, diagnostics = path_follow_command(
        robot_xy=(0.0, 0.0),
        robot_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
        forward_body_axis="x",
        waypoints=[(0.0, 1.0)],
        waypoint_index=0,
        cfg=PathFollowerCfg(max_forward_speed=0.8, max_yaw_rate=1.0, yaw_gain=2.0, turn_in_place_angle=0.5),
    )
    assert index == 0
    assert done is False
    assert command[0] < 0.2
    assert math.isclose(command[2], 1.0, rel_tol=1.0e-6)
    assert diagnostics["heading_error"] > 1.0


def test_path_follow_command_stops_at_final_waypoint() -> None:
    command, index, done, diagnostics = path_follow_command(
        robot_xy=(0.05, 0.02),
        robot_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
        forward_body_axis="x",
        waypoints=[(0.0, 0.0)],
        waypoint_index=0,
        cfg=PathFollowerCfg(arrival_radius=0.1),
    )
    assert command == (0.0, 0.0, 0.0)
    assert index == 0
    assert done is True
    assert diagnostics["done"] is True
