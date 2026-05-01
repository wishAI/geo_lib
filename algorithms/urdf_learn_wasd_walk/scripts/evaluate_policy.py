from __future__ import annotations

import argparse
import math

from isaaclab.app import AppLauncher

from algorithms.urdf_learn_wasd_walk.isaac_app_args import apply_project_kit_args
from algorithms.urdf_learn_wasd_walk.isaac_lock import acquire_isaac_lock
from algorithms.urdf_learn_wasd_walk.runtime import supported_robot_keys
from algorithms.urdf_learn_wasd_walk.task_registry import LANDAU_CURRICULUM_STAGES


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint on a joystick-style path scenario.")
    parser.add_argument("--robot", choices=supported_robot_keys(), required=True)
    parser.add_argument("--stage", choices=LANDAU_CURRICULUM_STAGES, default=None)
    parser.add_argument("--disable_fabric", action="store_true", default=False)
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--load_run", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--latest", action="store_true", default=False)
    parser.add_argument("--experiment_name", type=str, default=None)
    parser.add_argument("--workflow-id", type=str, default=None, help="Stable workflow identifier for history correlation.")
    parser.add_argument(
        "--playback-compat-mode",
        choices=("strict", "control_only", "off"),
        default="strict",
    )
    parser.add_argument("--path-file", type=str, default=None, help="Optional JSON path containing waypoint [x, y] pairs.")
    parser.add_argument("--path-preset", choices=("gate", "target", "triangle", "square"), default="gate")
    parser.add_argument("--gate-direction", choices=("forward", "left", "right", "backward"), default="forward")
    parser.add_argument("--path-distance", type=float, default=10.0)
    parser.add_argument("--target-x", type=float, default=0.0)
    parser.add_argument("--target-y", type=float, default=10.0)
    parser.add_argument("--path-edge-length", type=float, default=3.5)
    parser.add_argument("--path-clockwise", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--path-close-loop", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--stand-steps", type=int, default=120)
    parser.add_argument("--walk-steps-limit", type=int, default=2400)
    parser.add_argument("--hold-steps", type=int, default=240)
    parser.add_argument("--path-arrival-radius", type=float, default=0.35)
    parser.add_argument("--path-slow-radius", type=float, default=1.0)
    parser.add_argument("--path-max-forward", type=float, default=0.7)
    parser.add_argument("--path-max-yaw", type=float, default=0.9)
    parser.add_argument(
        "--terrain-mode",
        choices=("flat", "game"),
        default="flat",
        help="Playback scene for Landau game evaluation. 'flat' is the simple default; 'game' uses the small rough map.",
    )
    parser.add_argument(
        "--obstacles",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep obstacle sub-terrains when using --terrain-mode game.",
    )
    parser.add_argument("--hold-action-mode", choices=("policy", "default_pose"), default="policy")
    parser.add_argument(
        "--obstacle-brake",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="If a height scanner is available, scale down forward speed when a tall obstacle rises in front of the robot.",
    )
    parser.add_argument("--require-walking-pose", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-walk-single-support-ratio", type=float, default=None)
    parser.add_argument("--max-walk-double-support-ratio", type=float, default=None)
    parser.add_argument("--max-walk-flight-ratio", type=float, default=None)
    parser.add_argument("--min-walk-contact-switches", type=int, default=None)
    parser.add_argument("--min-walk-swing-clearance", type=float, default=None)
    parser.add_argument("--min-walk-step-length", type=float, default=None)
    parser.add_argument("--max-walk-non-support-contact-steps", type=int, default=None)
    AppLauncher.add_app_launcher_args(parser)
    return parser


parser = _build_parser()
args_cli = parser.parse_args()
apply_project_kit_args(args_cli)

isaac_lock_handle = acquire_isaac_lock("evaluate_policy", args_cli)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab_tasks.utils.wrappers.rsl_rl import RslRlVecEnvWrapper

from algorithms.urdf_learn_wasd_walk.command_frame import semantic_command_to_env_command, semantic_forward_dir_xy
from algorithms.urdf_learn_wasd_walk.game_control import (
    ObstacleBrakeCfg,
    PathFollowerCfg,
    apply_forward_brake,
    build_preset_waypoints,
    joystick_path_command,
    load_waypoints,
    obstacle_brake_factor,
)
from algorithms.urdf_learn_wasd_walk.isaac_workflow import (
    apply_checkpoint_playback_compat,
    build_init_pose_action,
    clamp_base_velocity_command,
    force_base_velocity_command,
    load_env_and_runner_cfg,
    load_runner_checkpoint,
    log_root_for_experiment,
    resolve_checkpoint_selection,
)
from algorithms.urdf_learn_wasd_walk.landau_env_cfg import configure_landau_game_playback_scene
from algorithms.urdf_learn_wasd_walk.rsl_rl_safety import install_safe_actor_critic_distribution_patch
from algorithms.urdf_learn_wasd_walk.run_history import write_evaluation_record
from algorithms.urdf_learn_wasd_walk.runtime import resolve_robot_task_spec
from algorithms.urdf_learn_wasd_walk.task_registry import register_gym_envs
from algorithms.urdf_learn_wasd_walk.walk_acceptance import default_walking_pose_thresholds, walking_pose_failures

install_safe_actor_critic_distribution_patch()


def _maybe_build_obstacle_brake_cfg(env_cfg) -> ObstacleBrakeCfg | None:
    sensor_cfg = getattr(getattr(env_cfg, "scene", None), "height_scanner", None)
    if sensor_cfg is None:
        return None
    pattern_cfg = getattr(sensor_cfg, "pattern_cfg", None)
    if pattern_cfg is None or not hasattr(pattern_cfg, "resolution") or not hasattr(pattern_cfg, "size"):
        return None
    return ObstacleBrakeCfg(
        resolution=float(pattern_cfg.resolution),
        size=(float(pattern_cfg.size[0]), float(pattern_cfg.size[1])),
        ordering=str(getattr(pattern_cfg, "ordering", "xy")),
    )


def _height_scan_values(height_sensor) -> list[float]:
    scan = height_sensor.data.pos_w[:, 2].unsqueeze(1) - height_sensor.data.ray_hits_w[..., 2] - 0.5
    return [float(value) for value in scan[0].detach().cpu().tolist()]


def _contact_mask(contact_sensor, body_ids: list[int], threshold: float) -> torch.Tensor:
    net_forces = contact_sensor.data.net_forces_w_history[:, 0, body_ids]
    return torch.linalg.norm(net_forces, dim=-1) > threshold


def _group_support_links(link_names: tuple[str, ...]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    left_links = tuple(name for name in link_names if name.endswith("_l"))
    right_links = tuple(name for name in link_names if name.endswith("_r"))
    return left_links, right_links


def _body_axes_xy(quat_wxyz: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    w, x, y, z = (float(value) for value in quat_wxyz)
    body_x = torch.tensor(
        (
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y + z * w),
        ),
        dtype=torch.float32,
    )
    body_y = torch.tensor(
        (
            2.0 * (x * y - z * w),
            1.0 - 2.0 * (x * x + z * z),
        ),
        dtype=torch.float32,
    )
    return body_x, body_y


def _path_distance_hint(
    waypoints: list[tuple[float, float]],
    *,
    start_xy: tuple[float, float] = (0.0, 0.0),
) -> float:
    total_distance = 0.0
    last_x = float(start_xy[0])
    last_y = float(start_xy[1])
    for waypoint_x, waypoint_y in waypoints:
        total_distance += math.hypot(waypoint_x - last_x, waypoint_y - last_y)
        last_x = waypoint_x
        last_y = waypoint_y
    return total_distance


def main() -> None:
    register_gym_envs()
    task_spec = resolve_robot_task_spec(args_cli.robot, stage=args_cli.stage)
    env_cfg, agent_cfg = load_env_and_runner_cfg(task_spec.play_task_id, args_cli)
    env_cfg.scene.num_envs = 1
    if args_cli.robot == "landau" and args_cli.stage == "game":
        configure_landau_game_playback_scene(
            env_cfg,
            terrain_mode=args_cli.terrain_mode,
            obstacles_enabled=args_cli.obstacles,
        )

    log_root_path = log_root_for_experiment(agent_cfg.experiment_name)
    checkpoint_selection = resolve_checkpoint_selection(log_root_path, agent_cfg, prefer_latest=args_cli.latest)
    resume_path = checkpoint_selection["path"]
    print(f"[EVAL] checkpoint={resume_path} source={checkpoint_selection['source']}", flush=True)
    if checkpoint_selection["source"] == "recommended":
        reason = checkpoint_selection["entry"].get("reason")
        if reason:
            print(f"[EVAL] recommended note: {reason}", flush=True)
    if apply_checkpoint_playback_compat(env_cfg, resume_path, mode=args_cli.playback_compat_mode):
        print(
            "[EVAL] applied playback compatibility overrides from checkpoint params/env.yaml "
            f"(mode={args_cli.playback_compat_mode})",
            flush=True,
        )

    env = gym.make(task_spec.play_task_id, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)
    robot = env.unwrapped.scene["robot"]
    contact_sensor = env.unwrapped.scene["contact_forces"]
    height_sensor = env.unwrapped.scene.sensors.get("height_scanner") if hasattr(env.unwrapped.scene, "sensors") else None
    obstacle_brake_cfg = _maybe_build_obstacle_brake_cfg(env_cfg)
    default_obstacle_brake = args_cli.stage == "game" and args_cli.terrain_mode == "game"
    obstacle_brake_enabled = (args_cli.obstacle_brake if args_cli.obstacle_brake is not None else default_obstacle_brake) and obstacle_brake_cfg is not None

    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    checkpoint_load = load_runner_checkpoint(ppo_runner, resume_path, load_optimizer=False)
    print(
        f"[EVAL] loaded inference checkpoint without optimizer state "
        f"(mode={checkpoint_load['mode']})",
        flush=True,
    )
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    idle_actions = build_init_pose_action(env.unwrapped, env_cfg)
    control_root_name = getattr(task_spec, "control_root_link", None) or getattr(task_spec, "root_link_name", None)
    support_names = tuple(getattr(task_spec, "support_link_names", ()))
    control_root_ids, _ = robot.find_bodies([control_root_name], preserve_order=True)
    support_ids, _ = robot.find_bodies(list(support_names), preserve_order=True)
    foot_names = tuple(getattr(task_spec, "primary_foot_links", ()))
    left_support_names, right_support_names = _group_support_links(support_names)
    left_support_robot_ids, _ = robot.find_bodies(list(left_support_names), preserve_order=True)
    right_support_robot_ids, _ = robot.find_bodies(list(right_support_names), preserve_order=True)
    left_support_sensor_ids, _ = contact_sensor.find_bodies(list(left_support_names), preserve_order=True)
    right_support_sensor_ids, _ = contact_sensor.find_bodies(list(right_support_names), preserve_order=True)
    support_name_set = set(support_names)
    non_support_names = tuple(getattr(task_spec, "gait_guard_link_names", ()) or ())
    if not non_support_names:
        non_support_names = tuple(name for name in robot.body_names if name not in support_name_set)
    non_support_sensor_ids: list[int] = []
    if non_support_names:
        non_support_sensor_ids, _ = contact_sensor.find_bodies(list(non_support_names), preserve_order=True)
    obs, _ = env.get_observations()

    initial_root_pos = robot.data.root_pos_w[0].detach().cpu()
    initial_root_quat = robot.data.root_quat_w[0].detach().cpu()
    initial_forward_dir_xy_tuple = semantic_forward_dir_xy(task_spec.forward_body_axis, initial_root_quat.tolist())
    if args_cli.path_file:
        waypoints = load_waypoints(args_cli.path_file)
    elif args_cli.path_preset == "target":
        waypoints = [(args_cli.target_x, args_cli.target_y)]
    else:
        waypoints = build_preset_waypoints(
            origin_xy=tuple(float(value) for value in initial_root_pos[:2].tolist()),
            forward_dir_xy=initial_forward_dir_xy_tuple,
            preset=args_cli.path_preset,
            distance_m=args_cli.path_distance,
            gate_direction=args_cli.gate_direction,
            edge_length_m=args_cli.path_edge_length,
            clockwise=args_cli.path_clockwise,
            close_loop=args_cli.path_close_loop,
        )
    target_distance_m = _path_distance_hint(
        waypoints,
        start_xy=tuple(float(value) for value in initial_root_pos[:2].tolist()),
    )
    walk_pose_thresholds = default_walking_pose_thresholds(args_cli.stage or "full", target_distance_m=target_distance_m)
    if args_cli.min_walk_single_support_ratio is not None:
        walk_pose_thresholds["min_single_support_ratio"] = args_cli.min_walk_single_support_ratio
    if args_cli.max_walk_double_support_ratio is not None:
        walk_pose_thresholds["max_double_support_ratio"] = args_cli.max_walk_double_support_ratio
    if args_cli.max_walk_flight_ratio is not None:
        walk_pose_thresholds["max_flight_ratio"] = args_cli.max_walk_flight_ratio
    if args_cli.min_walk_contact_switches is not None:
        walk_pose_thresholds["min_contact_switches_per_side"] = args_cli.min_walk_contact_switches
    if args_cli.min_walk_swing_clearance is not None:
        walk_pose_thresholds["min_swing_clearance"] = args_cli.min_walk_swing_clearance
    if args_cli.min_walk_step_length is not None:
        walk_pose_thresholds["min_touchdown_step_length_mean"] = args_cli.min_walk_step_length
    if args_cli.max_walk_non_support_contact_steps is not None:
        walk_pose_thresholds["max_non_support_contact_steps"] = args_cli.max_walk_non_support_contact_steps
    elif args_cli.stage == "game" and args_cli.terrain_mode == "flat":
        walk_pose_thresholds["max_non_support_contact_steps"] = min(
            int(walk_pose_thresholds["max_non_support_contact_steps"]),
            2,
        )
    path_cfg = PathFollowerCfg(
        max_forward_speed=args_cli.path_max_forward,
        max_yaw_rate=args_cli.path_max_yaw,
        arrival_radius=args_cli.path_arrival_radius,
        slow_radius=args_cli.path_slow_radius,
    )

    waypoint_index = 0
    phase = "stand"
    walk_phase_steps = 0
    hold_phase_steps = 0
    total_steps = 0
    done_count_total = 0
    done_count_by_phase = {"stand": 0, "walk": 0, "hold": 0}
    min_control_root_height = float("inf")
    walk_completed = False
    hold_completed = False
    last_diagnostics: dict[str, object] = {}
    walk_pose_started = False
    walk_pose_step_count = 0
    walk_single_support_steps = 0
    walk_double_support_steps = 0
    walk_flight_steps = 0
    walk_contact_switches = torch.zeros(2, dtype=torch.int64)
    walk_touchdown_step_lengths: tuple[list[float], list[float]] = ([], [])
    walk_touchdown_root_straddles: tuple[list[float], list[float]] = ([], [])
    walk_non_support_contact_counts = torch.zeros(len(non_support_names), dtype=torch.int64)
    walk_non_support_peak_forces = torch.zeros(len(non_support_names), dtype=torch.float32)
    walk_max_swing_side_height = torch.tensor([float("-inf"), float("-inf")], dtype=torch.float32)
    walk_min_contact_side_height = torch.tensor([0.0, 0.0], dtype=torch.float32)
    walk_last_contact = torch.zeros(2, dtype=torch.bool)
    walk_min_control_root_height = float("inf")
    walk_start_root_pos = None
    walk_start_root_quat = None
    walk_initial_forward_dir_xy = None
    walk_initial_body_x_xy = None
    walk_initial_body_y_xy = None

    while simulation_app.is_running():
        if phase == "stand":
            env_command = (0.0, 0.0, 0.0)
        elif phase == "walk":
            if not walk_pose_started:
                walk_pose_started = True
                walk_start_root_pos = robot.data.root_pos_w[0].detach().cpu()
                walk_start_root_quat = robot.data.root_quat_w[0].detach().cpu()
                walk_initial_body_x_xy, walk_initial_body_y_xy = _body_axes_xy(walk_start_root_quat)
                walk_initial_forward_dir_xy = torch.tensor(
                    semantic_forward_dir_xy(task_spec.forward_body_axis, walk_start_root_quat.tolist()),
                    dtype=torch.float32,
                )
                initial_left_support_pos = robot.data.body_pos_w[0, left_support_robot_ids].detach().cpu()
                initial_right_support_pos = robot.data.body_pos_w[0, right_support_robot_ids].detach().cpu()
                walk_min_contact_side_height = torch.tensor(
                    [
                        float(initial_left_support_pos[:, 2].mean()) if len(left_support_robot_ids) else 0.0,
                        float(initial_right_support_pos[:, 2].mean()) if len(right_support_robot_ids) else 0.0,
                    ],
                    dtype=torch.float32,
                )
                walk_last_contact = torch.stack(
                    (
                        _contact_mask(contact_sensor, left_support_sensor_ids, 1.0)[0].any(),
                        _contact_mask(contact_sensor, right_support_sensor_ids, 1.0)[0].any(),
                    )
                ).detach().cpu()
            robot_xy = tuple(float(value) for value in robot.data.root_pos_w[0, :2].detach().cpu().tolist())
            robot_quat_wxyz = tuple(float(value) for value in robot.data.root_quat_w[0].detach().cpu().tolist())
            semantic_command, waypoint_index, path_done, diagnostics = joystick_path_command(
                robot_xy=robot_xy,
                robot_quat_wxyz=robot_quat_wxyz,
                forward_body_axis=task_spec.forward_body_axis,
                waypoints=waypoints,
                waypoint_index=waypoint_index,
                cfg=path_cfg,
            )
            if obstacle_brake_enabled and height_sensor is not None:
                brake_factor = obstacle_brake_factor(_height_scan_values(height_sensor), obstacle_brake_cfg)
                semantic_command = apply_forward_brake(semantic_command, brake_factor)
                diagnostics["obstacle_brake_factor"] = brake_factor
            env_command = clamp_base_velocity_command(
                env_cfg, semantic_command_to_env_command(task_spec.forward_body_axis, semantic_command)
            )
            last_diagnostics = diagnostics
            if path_done:
                walk_completed = True
                phase = "hold"
                env_command = (0.0, 0.0, 0.0)
        else:
            env_command = (0.0, 0.0, 0.0)

        force_base_velocity_command(env.unwrapped, env_command)
        obs, _ = env.get_observations()
        with torch.inference_mode():
            if phase in {"stand", "hold"} and args_cli.hold_action_mode == "default_pose":
                actions = idle_actions
            else:
                actions = policy(obs)
            obs, _, dones, _ = env.step(actions)

        if control_root_ids and support_ids:
            control_height = float(robot.data.body_pos_w[0, control_root_ids[0], 2].detach().cpu().item())
            support_floor = float(torch.min(robot.data.body_pos_w[0, support_ids, 2]).detach().cpu().item())
            min_control_root_height = min(min_control_root_height, control_height - support_floor)

        done_step = int(torch.count_nonzero(dones).item())
        done_count_total += done_step
        done_count_by_phase[phase] += done_step

        if phase == "walk":
            walk_pose_step_count += 1
            root_pos = robot.data.root_pos_w[0].detach().cpu()
            left_support_pos = robot.data.body_pos_w[0, left_support_robot_ids].detach().cpu()
            right_support_pos = robot.data.body_pos_w[0, right_support_robot_ids].detach().cpu()
            left_support_center_xy = left_support_pos[:, :2].mean(dim=0)
            right_support_center_xy = right_support_pos[:, :2].mean(dim=0)
            if control_root_ids:
                control_root_xy = robot.data.body_pos_w[0, control_root_ids[0], :2].detach().cpu()
            else:
                control_root_xy = root_pos[:2]
            root_quat = robot.data.root_quat_w[0].detach().cpu()
            body_x_xy, body_y_xy = _body_axes_xy(root_quat)
            forward_axis_xy = body_y_xy if task_spec.forward_body_axis == "y" else body_x_xy
            current_contact = torch.stack(
                (
                    _contact_mask(contact_sensor, left_support_sensor_ids, 1.0)[0].any(),
                    _contact_mask(contact_sensor, right_support_sensor_ids, 1.0)[0].any(),
                )
            ).detach().cpu()
            if current_contact[0] and current_contact[1]:
                walk_double_support_steps += 1
            elif current_contact[0] or current_contact[1]:
                walk_single_support_steps += 1
            else:
                walk_flight_steps += 1

            left_mean_height = float(left_support_pos[:, 2].mean())
            right_mean_height = float(right_support_pos[:, 2].mean())
            if current_contact[0]:
                walk_min_contact_side_height[0] = min(float(walk_min_contact_side_height[0]), left_mean_height)
            elif current_contact[1]:
                walk_max_swing_side_height[0] = max(float(walk_max_swing_side_height[0]), left_mean_height)
            if current_contact[1]:
                walk_min_contact_side_height[1] = min(float(walk_min_contact_side_height[1]), right_mean_height)
            elif current_contact[0]:
                walk_max_swing_side_height[1] = max(float(walk_max_swing_side_height[1]), right_mean_height)

            if current_contact[0] and not walk_last_contact[0]:
                walk_touchdown_step_lengths[0].append(
                    float(torch.dot(left_support_center_xy - right_support_center_xy, forward_axis_xy))
                )
                walk_touchdown_root_straddles[0].append(
                    min(
                        float(torch.dot(left_support_center_xy - control_root_xy, forward_axis_xy)),
                        float(torch.dot(control_root_xy - right_support_center_xy, forward_axis_xy)),
                    )
                )
            if current_contact[1] and not walk_last_contact[1]:
                walk_touchdown_step_lengths[1].append(
                    float(torch.dot(right_support_center_xy - left_support_center_xy, forward_axis_xy))
                )
                walk_touchdown_root_straddles[1].append(
                    min(
                        float(torch.dot(right_support_center_xy - control_root_xy, forward_axis_xy)),
                        float(torch.dot(control_root_xy - left_support_center_xy, forward_axis_xy)),
                    )
                )
            if non_support_sensor_ids:
                non_support_forces = contact_sensor.data.net_forces_w_history[:, 0, non_support_sensor_ids].detach().cpu()[0]
                non_support_force_norm = torch.linalg.norm(non_support_forces, dim=-1)
                non_support_contact_mask = non_support_force_norm > 1.0
                walk_non_support_contact_counts += non_support_contact_mask.to(dtype=torch.int64)
                walk_non_support_peak_forces = torch.maximum(walk_non_support_peak_forces, non_support_force_norm)
            if control_root_ids and support_ids:
                control_height = float(robot.data.body_pos_w[0, control_root_ids[0], 2].detach().cpu().item())
                support_floor = float(torch.min(robot.data.body_pos_w[0, support_ids, 2]).detach().cpu().item())
                walk_min_control_root_height = min(walk_min_control_root_height, control_height - support_floor)
            walk_contact_switches += (current_contact != walk_last_contact).to(dtype=torch.int64)
            walk_last_contact = current_contact

        total_steps += 1
        if phase == "stand":
            if total_steps >= args_cli.stand_steps:
                phase = "walk"
        elif phase == "walk":
            walk_phase_steps += 1
            if walk_phase_steps >= args_cli.walk_steps_limit:
                phase = "hold"
        else:
            hold_phase_steps += 1
            if hold_phase_steps >= args_cli.hold_steps:
                hold_completed = True
                break

    robot_xy = tuple(float(value) for value in robot.data.root_pos_w[0, :2].detach().cpu().tolist())
    final_target = waypoints[min(max(waypoint_index, 0), len(waypoints) - 1)]
    final_position_error = ((robot_xy[0] - final_target[0]) ** 2 + (robot_xy[1] - final_target[1]) ** 2) ** 0.5
    walk_forward_displacement = 0.0
    if walk_pose_started and walk_start_root_pos is not None and walk_initial_forward_dir_xy is not None:
        walk_planar_delta = robot.data.root_pos_w[0].detach().cpu()[:2] - walk_start_root_pos[:2]
        walk_forward_displacement = float(torch.dot(walk_planar_delta, walk_initial_forward_dir_xy))
    walk_pose_metrics = {
        "step_count": walk_pose_step_count,
        "single_support_ratio": walk_single_support_steps / float(max(walk_pose_step_count, 1)),
        "double_support_ratio": walk_double_support_steps / float(max(walk_pose_step_count, 1)),
        "flight_ratio": walk_flight_steps / float(max(walk_pose_step_count, 1)),
        "min_control_root_height": None if walk_min_control_root_height == float("inf") else walk_min_control_root_height,
        "contact_switches": walk_contact_switches.tolist(),
        "swing_clearance": [
            max(0.0, float(walk_max_swing_side_height[0] - walk_min_contact_side_height[0]))
            if torch.isfinite(walk_max_swing_side_height[0])
            else 0.0,
            max(0.0, float(walk_max_swing_side_height[1] - walk_min_contact_side_height[1]))
            if torch.isfinite(walk_max_swing_side_height[1])
            else 0.0,
        ],
        "touchdown_step_length_max": [
            max(walk_touchdown_step_lengths[0]) if walk_touchdown_step_lengths[0] else 0.0,
            max(walk_touchdown_step_lengths[1]) if walk_touchdown_step_lengths[1] else 0.0,
        ],
        "touchdown_step_length_mean": [
            sum(walk_touchdown_step_lengths[0]) / len(walk_touchdown_step_lengths[0]) if walk_touchdown_step_lengths[0] else 0.0,
            sum(walk_touchdown_step_lengths[1]) / len(walk_touchdown_step_lengths[1]) if walk_touchdown_step_lengths[1] else 0.0,
        ],
        "touchdown_root_straddle_mean": [
            sum(walk_touchdown_root_straddles[0]) / len(walk_touchdown_root_straddles[0]) if walk_touchdown_root_straddles[0] else 0.0,
            sum(walk_touchdown_root_straddles[1]) / len(walk_touchdown_root_straddles[1]) if walk_touchdown_root_straddles[1] else 0.0,
        ],
        "non_support_contact_step_sum": int(walk_non_support_contact_counts.sum().item()) if non_support_names else 0,
        "forward_displacement": walk_forward_displacement,
        "top_non_support_contacts": [
            f"{body_name}:steps={int(step_count.item())},peak_force={float(peak_force.item()):.3f}"
            for body_name, step_count, peak_force in sorted(
                zip(non_support_names, walk_non_support_contact_counts, walk_non_support_peak_forces, strict=False),
                key=lambda item: (int(item[1].item()), float(item[2].item()), item[0]),
                reverse=True,
            )[:6]
        ],
    }
    walk_pose_gate_failures = walking_pose_failures(walk_pose_metrics, walk_pose_thresholds)
    metrics = {
        "done_count_total": done_count_total,
        "hard_reset_count": done_count_total,
        "done_count_by_phase": done_count_by_phase,
        "final_phase": phase,
        "walk_completed": walk_completed,
        "hold_completed": hold_completed,
        "walk_phase_steps": walk_phase_steps,
        "hold_phase_steps": hold_phase_steps,
        "stand_steps": args_cli.stand_steps,
        "total_steps": total_steps,
        "robot_xy": robot_xy,
        "final_target": final_target,
        "final_position_error": final_position_error,
        "path_preset": args_cli.path_preset,
        "gate_direction": args_cli.gate_direction,
        "waypoint_count": len(waypoints),
        "path_length_hint_m": target_distance_m,
        "path_arrival_radius": args_cli.path_arrival_radius,
        "min_control_root_height": None if min_control_root_height == float("inf") else min_control_root_height,
        "last_diagnostics": last_diagnostics,
        "hold_action_mode": args_cli.hold_action_mode,
        "walk_pose": walk_pose_metrics,
        "walk_pose_thresholds": walk_pose_thresholds,
        "walk_pose_failures": walk_pose_gate_failures,
        "walking_pose_pass": not walk_pose_gate_failures,
    }
    print(f"[EVAL] metrics={metrics}", flush=True)

    walking_pose_gate_failed = args_cli.require_walking_pose and bool(walk_pose_gate_failures)
    status = "completed" if walk_completed and hold_completed and done_count_total == 0 and not walking_pose_gate_failed else "failed"
    failure = None
    failure_code = None
    failed_checks: list[str] = []
    if not walk_completed:
        failure = "walk phase did not reach target waypoint"
        failure_code = "path_incomplete"
    elif not hold_completed:
        failure = "hold phase did not complete"
        failure_code = "hold_incomplete"
    elif done_count_total > 0:
        failure = f"policy fell or terminated during evaluation: done_count_total={done_count_total}"
        failure_code = "done_count"
        failed_checks.append(f"done_count_total={done_count_total}")
    elif walking_pose_gate_failed:
        failure = "walking pose gate failed: " + "; ".join(walk_pose_gate_failures)
        failure_code = "walking_pose_failed"
        failed_checks.extend(walk_pose_gate_failures)

    gate_result = {
        "scenario": "stand_walk_hold",
        "selection_source": checkpoint_selection["source"],
        "selected_checkpoint": str(resume_path),
        "failure_code": failure_code,
        "failed_checks": failed_checks,
    }

    write_evaluation_record(
        checkpoint_path=resume_path,
        task_spec=task_spec,
        args=args_cli,
        scenario="stand_walk_hold",
        status=status,
        metrics=metrics,
        experiment_name=agent_cfg.experiment_name,
        failure=failure,
        gate_result=gate_result,
    )
    if failure is not None:
        raise AssertionError(failure)

    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
        isaac_lock_handle.release()
