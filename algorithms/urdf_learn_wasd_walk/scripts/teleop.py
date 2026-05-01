from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

from algorithms.urdf_learn_wasd_walk.checkpoint_preflight import preflight_default_checkpoint_selection
from algorithms.urdf_learn_wasd_walk.isaac_app_args import apply_project_kit_args
from algorithms.urdf_learn_wasd_walk.isaac_lock import acquire_isaac_lock
from algorithms.urdf_learn_wasd_walk.runtime import supported_robot_keys
from algorithms.urdf_learn_wasd_walk.task_registry import LANDAU_CURRICULUM_STAGES


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Teleoperate a trained locomotion policy with keyboard or gamepad.")
    parser.add_argument("--robot", choices=supported_robot_keys(), required=True)
    parser.add_argument("--stage", choices=LANDAU_CURRICULUM_STAGES, default=None,
                        help="Landau curriculum stage (sets experiment name for checkpoint lookup).")
    parser.add_argument("--input-device", choices=("keyboard", "gamepad"), default="keyboard")
    parser.add_argument("--command-mode", choices=("auto", "se2", "game"), default="auto")
    parser.add_argument("--visual-mode", choices=("auto", "urdf", "usd", "both"), default="auto")
    parser.add_argument("--disable_fabric", action="store_true", default=False)
    parser.add_argument("--num_envs", type=int, default=1)
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
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument(
        "--terrain-mode",
        choices=("flat", "game"),
        default="flat",
        help="Interactive scene for Landau game teleop. 'flat' is the simple default; 'game' uses the small rough map.",
    )
    parser.add_argument(
        "--obstacles",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep obstacle sub-terrains when using --terrain-mode game.",
    )
    parser.add_argument(
        "--latch-command",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Persist keyboard commands after key release until changed or reset with L.",
    )
    parser.add_argument("--idle-action-mode", choices=("default_pose", "policy"), default="policy")
    parser.add_argument("--idle-command-threshold", type=float, default=0.02)
    parser.add_argument(
        "--obstacle-brake",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="If a height scanner is available, scale down forward speed when a tall obstacle rises in front of the robot.",
    )
    AppLauncher.add_app_launcher_args(parser)
    return parser


parser = _build_parser()
args_cli = parser.parse_args()
preflight_default_checkpoint_selection(args_cli, mode="teleop")
apply_project_kit_args(args_cli)

isaac_lock_handle = acquire_isaac_lock("teleop", args_cli)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import omni.ui
import omni.usd
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.devices import Se2Gamepad
from isaaclab_tasks.utils.wrappers.rsl_rl import RslRlVecEnvWrapper

from algorithms.urdf_learn_wasd_walk.command_frame import semantic_command_to_env_command
from algorithms.urdf_learn_wasd_walk.game_control import (
    ManualGameCommandCfg,
    ObstacleBrakeCfg,
    apply_forward_brake,
    manual_game_command,
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
from algorithms.urdf_learn_wasd_walk.runtime import resolve_robot_task_spec
from algorithms.urdf_learn_wasd_walk.task_registry import register_gym_envs
from algorithms.urdf_learn_wasd_walk.teleop_input import GameWasdKeyboard, WasdSe2Keyboard
from algorithms.urdf_learn_wasd_walk.usd_visualizer import LandauUsdVisualizer

install_safe_actor_critic_distribution_patch()


def _resolve_visual_mode(robot_key: str, requested_mode: str) -> str:
    if requested_mode == "auto":
        return "usd" if robot_key == "landau" else "urdf"
    if requested_mode in {"usd", "both"} and robot_key != "landau":
        raise ValueError("USD visual mode is only supported for --robot landau.")
    return requested_mode


def _resolve_command_mode(robot_key: str, stage: str | None, requested_mode: str) -> str:
    if requested_mode != "auto":
        return requested_mode
    if robot_key == "landau" and stage == "game":
        return "game"
    return "se2"


def _make_device(command_mode: str):
    if args_cli.input_device == "keyboard":
        if command_mode == "game":
            device = GameWasdKeyboard(hold_last_command=args_cli.latch_command)
        else:
            device = WasdSe2Keyboard(hold_last_command=args_cli.latch_command)
    else:
        device = Se2Gamepad()
    print(device)
    return device


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


def _focus_teleop_window() -> None:
    viewport_window = omni.ui.Workspace.get_window("Viewport")
    if viewport_window is not None:
        viewport_window.focus()
        print("[TELEOP] Focused Isaac 'Viewport' window for keyboard input.", flush=True)
    else:
        print("[TELEOP] Could not find the 'Viewport' window to focus automatically.", flush=True)


def main() -> None:
    if args_cli.headless:
        raise RuntimeError("Teleoperation requires GUI mode. Run without --headless.")

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
    command_mode = _resolve_command_mode(args_cli.robot, args_cli.stage, args_cli.command_mode)
    manual_game_cfg = ManualGameCommandCfg()
    obstacle_brake_cfg = _maybe_build_obstacle_brake_cfg(env_cfg)
    default_obstacle_brake = command_mode == "game" and args_cli.terrain_mode == "game"
    obstacle_brake_enabled = (args_cli.obstacle_brake if args_cli.obstacle_brake is not None else default_obstacle_brake) and obstacle_brake_cfg is not None

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
    command_ranges = env_cfg.commands.base_velocity.ranges
    height_sensor = env.unwrapped.scene.sensors.get("height_scanner") if hasattr(env.unwrapped.scene, "sensors") else None

    visualizer = None
    if visual_mode != "urdf":
        stage = omni.usd.get_context().get_stage()
        visualizer = LandauUsdVisualizer(stage, env.unwrapped.scene.env_prim_paths[0])
        mapping = visualizer.mapping_summary()
        print(
            "[TELEOP] USD joint mapping "
            f"{mapping['mapped_joint_count']}/{mapping['expected_joint_count']} "
            f"({mapping['coverage']:.0%} coverage)",
            flush=True,
        )
        if visual_mode == "usd":
            if mapping["complete"]:
                visualizer.set_urdf_visibility(False)
            else:
                print(
                    "[TELEOP] USD joint mapping is incomplete; keeping the live URDF visible for diagnosis. "
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
    teleop_device = _make_device(command_mode)
    idle_actions = build_init_pose_action(env.unwrapped, env_cfg)
    if args_cli.input_device == "keyboard":
        _focus_teleop_window()
        print(
            "[TELEOP] Click inside the Isaac GUI viewport before using the keyboard. "
            "The terminal does not receive teleop key input.",
            flush=True,
        )
        print(
            "[TELEOP] If W/A/S/D conflicts with viewport navigation, use Up/Down/Left/Right or Numpad 8/2/4/6. "
            "Extra trim axes are also available on Q/E, Z/X, or Numpad 7/9.",
            flush=True,
        )
        print(
            f"[TELEOP] Keyboard latch is {'on' if args_cli.latch_command else 'off'}. "
            "Press L to zero the command.",
            flush=True,
        )
        print(
            f"[TELEOP] Idle action mode is {args_cli.idle_action_mode}; "
            f"zero-command threshold={args_cli.idle_command_threshold:.3f}.",
            flush=True,
        )
        print("[TELEOP] Mapped key presses and releases will be printed below.", flush=True)
    print(f"[TELEOP] command_mode={command_mode}", flush=True)
    if obstacle_brake_enabled:
        print("[TELEOP] obstacle brake enabled from height scanner.", flush=True)
    print(
        "[TELEOP] base_velocity ranges "
        f"vx={tuple(float(value) for value in command_ranges.lin_vel_x)} "
        f"vy={tuple(float(value) for value in command_ranges.lin_vel_y)} "
        f"yaw={tuple(float(value) for value in command_ranges.ang_vel_z)}",
        flush=True,
    )

    obs, _ = env.get_observations()
    if visualizer is not None:
        visualizer.sync_from_robot(robot)
    step_count = 0
    last_env_command = None
    last_brake_factor = None
    diag_interval = 50  # print velocity diagnostics every N steps
    while simulation_app.is_running():
        raw_command = tuple(float(value) for value in teleop_device.advance())
        if command_mode == "game":
            if args_cli.input_device == "keyboard":
                semantic_command = raw_command
            else:
                semantic_command = manual_game_command(raw_command, manual_game_cfg)
        else:
            semantic_command = raw_command
        if obstacle_brake_enabled and height_sensor is not None:
            brake_factor = obstacle_brake_factor(_height_scan_values(height_sensor), obstacle_brake_cfg)
            semantic_command = apply_forward_brake(semantic_command, brake_factor)
            if last_brake_factor is None or abs(brake_factor - last_brake_factor) > 0.1:
                print(f"[TELEOP] obstacle_brake={brake_factor:.2f}", flush=True)
            last_brake_factor = brake_factor
        env_command = clamp_base_velocity_command(
            env_cfg, semantic_command_to_env_command(task_spec.forward_body_axis, semantic_command)
        )
        if env_command != last_env_command:
            print(f"[TELEOP] semantic_command -> {semantic_command} env_command -> {env_command}", flush=True)
        last_env_command = env_command
        force_base_velocity_command(env.unwrapped, env_command)
        obs, _ = env.get_observations()
        with torch.inference_mode():
            idle_command = (
                (env_command[0] * env_command[0] + env_command[1] * env_command[1]) ** 0.5
                <= args_cli.idle_command_threshold
                and abs(env_command[2]) <= args_cli.idle_command_threshold
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
            ang_vel_w = robot.data.root_ang_vel_w[0].detach().cpu()
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
                f"[TELEOP] vel_b=({lin_vel_b[0]:.3f}, {lin_vel_b[1]:.3f}, {lin_vel_b[2]:.3f}) "
                f"yaw_rate={ang_vel_w[2]:.3f} cmd={env_command} idle={idle_command} "
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
