from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

from algorithms.urdf_learn_wasd_walk.runtime import supported_robot_keys
from algorithms.urdf_learn_wasd_walk.task_registry import LANDAU_CURRICULUM_STAGES


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Headless environment smoke test without loading a policy.")
    parser.add_argument("--robot", choices=supported_robot_keys(), required=True)
    parser.add_argument("--stage", choices=LANDAU_CURRICULUM_STAGES, default=None)
    parser.add_argument("--visual-mode", choices=("auto", "urdf", "usd", "both"), default="auto")
    parser.add_argument("--disable_fabric", action="store_true", default=False)
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--steps", type=int, default=32)
    AppLauncher.add_app_launcher_args(parser)
    return parser


parser = _build_parser()
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import omni.usd
import torch

from algorithms.urdf_learn_wasd_walk.isaac_workflow import load_env_and_runner_cfg
from algorithms.urdf_learn_wasd_walk.runtime import resolve_robot_task_spec
from algorithms.urdf_learn_wasd_walk.task_registry import register_gym_envs
from algorithms.urdf_learn_wasd_walk.usd_visualizer import LandauUsdVisualizer


def _resolve_visual_mode(robot_key: str, requested_mode: str) -> str:
    if requested_mode == "auto":
        return "urdf"
    if requested_mode in {"usd", "both"} and robot_key != "landau":
        raise ValueError("USD visual mode is only supported for --robot landau.")
    return requested_mode


def main() -> None:
    register_gym_envs()
    task_spec = resolve_robot_task_spec(args_cli.robot, stage=args_cli.stage)
    env_cfg, _ = load_env_and_runner_cfg(task_spec.play_task_id, args_cli)
    visual_mode = _resolve_visual_mode(args_cli.robot, args_cli.visual_mode)

    if visual_mode != "urdf" and getattr(env_cfg.scene, "num_envs", 1) != 1:
        print("[INFO] Synced USD visual mode uses one displayed environment; overriding num_envs to 1.")
        env_cfg.scene.num_envs = 1

    env = gym.make(task_spec.play_task_id, cfg=env_cfg)
    obs, _ = env.reset()
    robot = env.unwrapped.scene["robot"]

    visualizer = None
    if visual_mode != "urdf":
        stage = omni.usd.get_context().get_stage()
        visualizer = LandauUsdVisualizer(stage, env.unwrapped.scene.env_prim_paths[0])
        if visual_mode == "usd":
            visualizer.set_urdf_visibility(False)
        visualizer.sync_from_robot(robot)

    action_space = env.action_space
    for _ in range(args_cli.steps):
        if hasattr(action_space, "sample"):
            action_dim = int(env.unwrapped.action_manager.total_action_dim)
            action = torch.zeros((env.unwrapped.num_envs, action_dim), device=env.unwrapped.device)
        else:
            raise RuntimeError("Expected a Box action space.")
        obs, *_ = env.step(action)
        if visualizer is not None:
            visualizer.sync_from_robot(robot)
    print(f"[SMOKE] completed {args_cli.steps} steps for {task_spec.display_name}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
