from __future__ import annotations

import argparse
import os

from isaaclab.app import AppLauncher

from algorithms.urdf_learn_wasd_walk.checkpoint_preflight import preflight_default_checkpoint_selection
from algorithms.urdf_learn_wasd_walk.isaac_app_args import apply_project_kit_args
from algorithms.urdf_learn_wasd_walk.isaac_lock import acquire_isaac_lock
from algorithms.urdf_learn_wasd_walk.runtime import supported_robot_keys
from algorithms.urdf_learn_wasd_walk.task_registry import LANDAU_CURRICULUM_STAGES


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Play a trained locomotion checkpoint.")
    parser.add_argument("--robot", choices=supported_robot_keys(), required=True)
    parser.add_argument("--stage", choices=LANDAU_CURRICULUM_STAGES, default=None,
                        help="Landau curriculum stage (sets experiment name for checkpoint lookup).")
    parser.add_argument("--visual-mode", choices=("auto", "urdf", "usd", "both"), default="auto")
    parser.add_argument("--disable_fabric", action="store_true", default=False)
    parser.add_argument("--num_envs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--load_run", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument(
        "--latest",
        action="store_true",
        default=False,
        help="Ignore the recommended checkpoint registry and load the latest checkpoint under the selected experiment.",
    )
    parser.add_argument("--experiment_name", type=str, default=None)
    parser.add_argument(
        "--playback-compat-mode",
        choices=("strict", "control_only", "off"),
        default="strict",
        help=(
            "How much checkpoint-era env config to restore. "
            "'strict' restores action mapping, checkpoint-era init pose, and actuator settings."
        ),
    )
    parser.add_argument("--idle-action-mode", choices=("default_pose", "policy"), default="policy")
    parser.add_argument("--command-vx", type=float, default=None)
    parser.add_argument("--command-vy", type=float, default=None)
    parser.add_argument("--command-yaw", type=float, default=None)
    parser.add_argument("--path-file", type=str, default=None, help="JSON path containing waypoint [x, y] pairs.")
    parser.add_argument("--path-arrival-radius", type=float, default=0.35)
    parser.add_argument("--path-slow-radius", type=float, default=1.0)
    parser.add_argument("--path-max-forward", type=float, default=0.7)
    parser.add_argument("--path-max-yaw", type=float, default=0.9)
    parser.add_argument(
        "--terrain-mode",
        choices=("flat", "game"),
        default="flat",
        help="Interactive scene for Landau game playback. 'flat' is the simple default; 'game' uses the small rough map.",
    )
    parser.add_argument(
        "--obstacles",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep obstacle sub-terrains when using --terrain-mode game.",
    )
    parser.add_argument(
        "--obstacle-brake",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="If a height scanner is available, scale down forward speed when a tall obstacle rises in front of the robot.",
    )
    parser.add_argument("--steps", type=int, default=None, help="Stop after this many environment steps.")
    AppLauncher.add_app_launcher_args(parser)
    return parser


parser = _build_parser()
args_cli = parser.parse_args()
preflight_default_checkpoint_selection(args_cli, mode="play")
apply_project_kit_args(args_cli)

isaac_lock_handle = acquire_isaac_lock("play", args_cli)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import omni.usd
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab_tasks.utils.wrappers.rsl_rl import RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx

from algorithms.urdf_learn_wasd_walk.command_frame import semantic_command_to_env_command
from algorithms.urdf_learn_wasd_walk.game_control import (
    ObstacleBrakeCfg,
    PathFollowerCfg,
    apply_forward_brake,
    load_waypoints,
    obstacle_brake_factor,
    path_follow_command,
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
from algorithms.urdf_learn_wasd_walk.runtime import resolve_robot_task_spec
from algorithms.urdf_learn_wasd_walk.task_registry import register_gym_envs
from algorithms.urdf_learn_wasd_walk.usd_visualizer import LandauUsdVisualizer

install_safe_actor_critic_distribution_patch()


def _resolve_visual_mode(robot_key: str, requested_mode: str) -> str:
    if requested_mode == "auto":
        return "usd" if robot_key == "landau" else "urdf"
    if requested_mode in {"usd", "both"} and robot_key != "landau":
        raise ValueError("USD visual mode is only supported for --robot landau.")
    return requested_mode


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


def main() -> None:
    register_gym_envs()
    task_spec = resolve_robot_task_spec(args_cli.robot, stage=args_cli.stage)
    env_cfg, agent_cfg = load_env_and_runner_cfg(task_spec.play_task_id, args_cli)
    if args_cli.robot == "landau" and args_cli.stage == "game":
        configure_landau_game_playback_scene(
            env_cfg,
            terrain_mode=args_cli.terrain_mode,
            obstacles_enabled=args_cli.obstacles,
        )
    visual_mode = _resolve_visual_mode(args_cli.robot, args_cli.visual_mode)

    log_root_path = log_root_for_experiment(agent_cfg.experiment_name)
    try:
        checkpoint_selection = resolve_checkpoint_selection(log_root_path, agent_cfg, prefer_latest=args_cli.latest)
        resume_path = checkpoint_selection["path"]
    except ValueError as exc:
        raise RuntimeError(
            f"Unable to resolve a usable checkpoint for experiment '{agent_cfg.experiment_name}' under '{log_root_path}'. "
            f"Details: {exc}"
        ) from exc
    selection_source = checkpoint_selection["source"]
    if selection_source == "recommended":
        print(f"[INFO] Loading recommended checkpoint: {resume_path}")
        reason = checkpoint_selection["entry"].get("reason")
        if reason:
            print(f"[INFO] Recommended checkpoint note: {reason}")
    elif selection_source == "latest":
        print(f"[INFO] Loading latest checkpoint: {resume_path}")
    else:
        print(f"[INFO] Loading explicit checkpoint: {resume_path}")
    if apply_checkpoint_playback_compat(env_cfg, resume_path, mode=args_cli.playback_compat_mode):
        print(
            "[INFO] Applied playback compatibility overrides from checkpoint params/env.yaml "
            f"(mode={args_cli.playback_compat_mode})"
        )

    if visual_mode != "urdf" and getattr(env_cfg.scene, "num_envs", 1) != 1:
        print("[INFO] Synced USD visual mode uses one displayed environment; overriding num_envs to 1.")
        env_cfg.scene.num_envs = 1

    env = gym.make(task_spec.play_task_id, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)
    robot = env.unwrapped.scene["robot"]
    requested_joint_names = list(getattr(env_cfg.actions.joint_pos, "joint_names", []))
    controlled_joint_names = [name for name in requested_joint_names if name in robot.joint_names]
    controlled_joint_ids = [robot.joint_names.index(name) for name in controlled_joint_names]
    initial_joint_pos = None
    previous_joint_pos = None
    if controlled_joint_ids:
        initial_joint_pos = robot.data.joint_pos[0, controlled_joint_ids].detach().clone()
        previous_joint_pos = initial_joint_pos.clone()
    height_sensor = env.unwrapped.scene.sensors.get("height_scanner") if hasattr(env.unwrapped.scene, "sensors") else None
    obstacle_brake_cfg = _maybe_build_obstacle_brake_cfg(env_cfg)
    default_obstacle_brake = args_cli.stage == "game" and args_cli.terrain_mode == "game"
    obstacle_brake_enabled = (args_cli.obstacle_brake if args_cli.obstacle_brake is not None else default_obstacle_brake) and obstacle_brake_cfg is not None

    visualizer = None
    if visual_mode != "urdf":
        stage = omni.usd.get_context().get_stage()
        visualizer = LandauUsdVisualizer(stage, env.unwrapped.scene.env_prim_paths[0])
        mapping = visualizer.mapping_summary()
        print(
            "[INFO] USD joint mapping "
            f"{mapping['mapped_joint_count']}/{mapping['expected_joint_count']} "
            f"({mapping['coverage']:.0%} coverage)",
            flush=True,
        )
        if visual_mode == "usd":
            if mapping["complete"]:
                visualizer.set_urdf_visibility(False)
            else:
                print(
                    "[WARN] USD joint mapping is incomplete; keeping the live URDF visible for diagnosis. "
                    "Use --visual-mode both to inspect both rigs explicitly.",
                    flush=True,
                )

    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    checkpoint_load = load_runner_checkpoint(ppo_runner, resume_path, load_optimizer=False)
    print(
        f"[INFO] Loaded inference checkpoint without optimizer state "
        f"(mode={checkpoint_load['mode']})",
        flush=True,
    )
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)
    idle_actions = build_init_pose_action(env.unwrapped, env_cfg)

    fixed_command = None
    waypoints = load_waypoints(args_cli.path_file) if args_cli.path_file else None
    path_cfg = PathFollowerCfg(
        max_forward_speed=args_cli.path_max_forward,
        max_yaw_rate=args_cli.path_max_yaw,
        arrival_radius=args_cli.path_arrival_radius,
        slow_radius=args_cli.path_slow_radius,
    )
    waypoint_index = 0
    last_path_index = None
    if any(value is not None for value in (args_cli.command_vx, args_cli.command_vy, args_cli.command_yaw)):
        semantic_command = (
            0.0 if args_cli.command_vx is None else args_cli.command_vx,
            0.0 if args_cli.command_vy is None else args_cli.command_vy,
            0.0 if args_cli.command_yaw is None else args_cli.command_yaw,
        )
        fixed_command = clamp_base_velocity_command(
            env_cfg, semantic_command_to_env_command(task_spec.forward_body_axis, semantic_command)
        )
        print(f"[INFO] Forcing base command: semantic={semantic_command} env={fixed_command}")
    elif waypoints is not None:
        print(f"[INFO] Path following enabled with {len(waypoints)} waypoint(s) from {args_cli.path_file}")
    else:
        fixed_command = (0.0, 0.0, 0.0)
        print(
            "[INFO] No command or path supplied; holding zero base command "
            f"with idle_action_mode={args_cli.idle_action_mode}.",
            flush=True,
        )
    if obstacle_brake_enabled:
        print("[INFO] Obstacle brake enabled from height scanner.")

    obs, _ = env.get_observations()
    if visualizer is not None:
        visualizer.sync_from_robot(robot)
    step_count = 0
    diag_interval = 50
    while simulation_app.is_running():
        commanded_env = fixed_command
        if waypoints is not None:
            robot_xy = tuple(float(value) for value in robot.data.root_pos_w[0, :2].detach().cpu().tolist())
            robot_quat_wxyz = tuple(float(value) for value in robot.data.root_quat_w[0].detach().cpu().tolist())
            semantic_command, waypoint_index, path_done, diagnostics = path_follow_command(
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
            commanded_env = clamp_base_velocity_command(
                env_cfg, semantic_command_to_env_command(task_spec.forward_body_axis, semantic_command)
            )
            if last_path_index != waypoint_index:
                print(f"[PLAY] waypoint_index={waypoint_index} diagnostics={diagnostics}", flush=True)
            last_path_index = waypoint_index
            if path_done:
                print("[PLAY] path complete", flush=True)
                force_base_velocity_command(env.unwrapped, (0.0, 0.0, 0.0))
                break
        if commanded_env is not None:
            force_base_velocity_command(env.unwrapped, commanded_env)
            obs, _ = env.get_observations()
        with torch.inference_mode():
            idle_command = commanded_env is not None and (
                (commanded_env[0] * commanded_env[0] + commanded_env[1] * commanded_env[1]) ** 0.5 <= 0.05
                and abs(commanded_env[2]) <= 0.05
            )
            if idle_command and args_cli.idle_action_mode == "default_pose":
                actions = idle_actions
            else:
                actions = policy(obs)
            obs, _, _, _ = env.step(actions)
        if visualizer is not None:
            visualizer.sync_from_robot(robot)
        step_count += 1
        if step_count % diag_interval == 0:
            lin_vel_b = robot.data.root_lin_vel_b[0].detach().cpu()
            mean_abs_action = float(torch.mean(torch.abs(actions[0])).detach().cpu().item())
            motion_summary = ""
            if controlled_joint_ids and previous_joint_pos is not None and initial_joint_pos is not None:
                joint_pos = robot.data.joint_pos[0, controlled_joint_ids].detach()
                mean_joint_step_delta = float(torch.mean(torch.abs(joint_pos - previous_joint_pos)).detach().cpu().item())
                max_joint_offset = float(torch.max(torch.abs(joint_pos - initial_joint_pos)).detach().cpu().item())
                previous_joint_pos = joint_pos.clone()
                motion_summary = (
                    f" mean_joint_step_delta={mean_joint_step_delta:.4f}"
                    f" max_joint_offset={max_joint_offset:.4f}"
                )
            print(
                f"[PLAY] vel_b=({lin_vel_b[0]:.3f}, {lin_vel_b[1]:.3f}, {lin_vel_b[2]:.3f}) "
                f"cmd={commanded_env} idle={idle_command} action_mode={args_cli.idle_action_mode} "
                f"mean_abs_action={mean_abs_action:.4f}{motion_summary}",
                flush=True,
            )
        if args_cli.steps is not None and step_count >= args_cli.steps:
            break

    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
        isaac_lock_handle.release()
