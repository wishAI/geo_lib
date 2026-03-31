from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

from algorithms.urdf_learn_wasd_walk.runtime import supported_robot_keys


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Teleoperate a trained locomotion policy with keyboard or gamepad.")
    parser.add_argument("--robot", choices=supported_robot_keys(), required=True)
    parser.add_argument("--input-device", choices=("keyboard", "gamepad"), default="keyboard")
    parser.add_argument("--visual-mode", choices=("auto", "urdf", "usd", "both"), default="auto")
    parser.add_argument("--disable_fabric", action="store_true", default=False)
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--load_run", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--experiment_name", type=str, default=None)
    parser.add_argument("--steps", type=int, default=None)
    AppLauncher.add_app_launcher_args(parser)
    return parser


parser = _build_parser()
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import omni.usd
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.devices import Se2Gamepad
from isaaclab_tasks.utils.wrappers.rsl_rl import RslRlVecEnvWrapper

from algorithms.urdf_learn_wasd_walk.command_frame import semantic_command_to_env_command
from algorithms.urdf_learn_wasd_walk.isaac_workflow import (
    force_base_velocity_command,
    load_env_and_runner_cfg,
    log_root_for_experiment,
    resolve_checkpoint,
)
from algorithms.urdf_learn_wasd_walk.runtime import resolve_robot_task_spec
from algorithms.urdf_learn_wasd_walk.task_registry import register_gym_envs
from algorithms.urdf_learn_wasd_walk.teleop_input import WasdSe2Keyboard
from algorithms.urdf_learn_wasd_walk.usd_visualizer import LandauUsdVisualizer


def _resolve_visual_mode(robot_key: str, requested_mode: str) -> str:
    if requested_mode == "auto":
        return "usd" if robot_key == "landau" else "urdf"
    if requested_mode in {"usd", "both"} and robot_key != "landau":
        raise ValueError("USD visual mode is only supported for --robot landau.")
    return requested_mode


def _make_device():
    if args_cli.input_device == "keyboard":
        device = WasdSe2Keyboard()
    else:
        device = Se2Gamepad()
    print(device)
    return device


def main() -> None:
    if args_cli.headless:
        raise RuntimeError("Teleoperation requires GUI mode. Run without --headless.")

    register_gym_envs()
    task_spec = resolve_robot_task_spec(args_cli.robot)
    env_cfg, agent_cfg = load_env_and_runner_cfg(task_spec.play_task_id, args_cli)
    visual_mode = _resolve_visual_mode(args_cli.robot, args_cli.visual_mode)

    log_root_path = log_root_for_experiment(agent_cfg.experiment_name)
    try:
        resume_path = resolve_checkpoint(log_root_path, agent_cfg)
    except ValueError as exc:
        raise RuntimeError(
            f"Unable to resolve a usable checkpoint for experiment '{agent_cfg.experiment_name}' under '{log_root_path}'. "
            f"Details: {exc}"
        ) from exc
    print(f"[INFO] Loading checkpoint: {resume_path}")

    if visual_mode != "urdf" and getattr(env_cfg.scene, "num_envs", 1) != 1:
        print("[INFO] Synced USD visual mode uses one displayed environment; overriding num_envs to 1.")
        env_cfg.scene.num_envs = 1

    env = gym.make(task_spec.play_task_id, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)
    robot = env.unwrapped.scene["robot"]

    visualizer = None
    if visual_mode != "urdf":
        stage = omni.usd.get_context().get_stage()
        visualizer = LandauUsdVisualizer(stage, env.unwrapped.scene.env_prim_paths[0])
        if visual_mode == "usd":
            visualizer.set_urdf_visibility(False)

    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)
    teleop_device = _make_device()

    obs, _ = env.get_observations()
    if visualizer is not None:
        visualizer.sync_from_robot(robot)
    step_count = 0
    while simulation_app.is_running():
        semantic_command = tuple(float(value) for value in teleop_device.advance())
        env_command = semantic_command_to_env_command(task_spec.key, semantic_command)
        force_base_velocity_command(env.unwrapped, env_command)
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)
        if visualizer is not None:
            visualizer.sync_from_robot(robot)
        step_count += 1
        if args_cli.steps is not None and step_count >= args_cli.steps:
            break

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
