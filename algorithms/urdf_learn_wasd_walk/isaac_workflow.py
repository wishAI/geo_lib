from __future__ import annotations

import os
import re
from typing import Any

import torch

from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry


def resolve_sim_device(use_cpu: bool) -> str:
    return "cpu" if use_cpu else "cuda:0"


def apply_runner_overrides(agent_cfg: Any, args: Any) -> Any:
    if getattr(args, "seed", None) is not None:
        agent_cfg.seed = args.seed
    if getattr(args, "resume", None) is not None:
        agent_cfg.resume = args.resume
    if getattr(args, "load_run", None) is not None:
        agent_cfg.load_run = args.load_run
    if getattr(args, "checkpoint", None) is not None:
        agent_cfg.load_checkpoint = args.checkpoint
    if getattr(args, "run_name", None) is not None:
        agent_cfg.run_name = args.run_name
    if getattr(args, "logger", None) is not None:
        agent_cfg.logger = args.logger
    if getattr(args, "log_project_name", None):
        if hasattr(agent_cfg, "wandb_project"):
            agent_cfg.wandb_project = args.log_project_name
        if hasattr(agent_cfg, "neptune_project"):
            agent_cfg.neptune_project = args.log_project_name
    if getattr(args, "experiment_name", None):
        agent_cfg.experiment_name = args.experiment_name
    if getattr(args, "max_iterations", None):
        agent_cfg.max_iterations = args.max_iterations
    return agent_cfg


def load_env_and_runner_cfg(task_id: str, args: Any):
    env_cfg = parse_env_cfg(
        task_id,
        device=resolve_sim_device(getattr(args, "cpu", False)),
        num_envs=getattr(args, "num_envs", None),
        use_fabric=not getattr(args, "disable_fabric", False),
    )
    agent_cfg = load_cfg_from_registry(task_id, "rsl_rl_cfg_entry_point")
    return env_cfg, apply_runner_overrides(agent_cfg, args)


def log_root_for_experiment(experiment_name: str) -> str:
    return os.path.abspath(os.path.join("logs", "rsl_rl", experiment_name))


def resolve_checkpoint(log_root: str, agent_cfg: Any) -> str:
    checkpoint_path = get_checkpoint_path(log_root, agent_cfg.load_run, agent_cfg.load_checkpoint)
    checkpoint_name = os.path.basename(checkpoint_path)
    if not re.fullmatch(r"model_.*\.pt", checkpoint_name):
        raise ValueError(
            f"Resolved checkpoint '{checkpoint_path}' is not an RSL-RL model checkpoint. "
            "Pass --checkpoint model_<iter>.pt or leave --checkpoint unset."
        )
    return checkpoint_path


def force_base_velocity_command(env, command: tuple[float, float, float]) -> None:
    command_term = env.command_manager.get_term("base_velocity")
    command_tensor = torch.tensor(command, device=env.device, dtype=torch.float32).repeat(env.num_envs, 1)
    command_term.vel_command_b[:] = command_tensor
    if hasattr(command_term, "is_heading_env"):
        command_term.is_heading_env[:] = False
    if hasattr(command_term, "is_standing_env"):
        command_term.is_standing_env[:] = False
    command_term.time_left[:] = 1.0e9


def clamp_base_velocity_command(env_cfg: Any, command: tuple[float, float, float]) -> tuple[float, float, float]:
    ranges = env_cfg.commands.base_velocity.ranges
    vx = min(max(float(command[0]), float(ranges.lin_vel_x[0])), float(ranges.lin_vel_x[1]))
    vy = min(max(float(command[1]), float(ranges.lin_vel_y[0])), float(ranges.lin_vel_y[1]))
    yaw = min(max(float(command[2]), float(ranges.ang_vel_z[0])), float(ranges.ang_vel_z[1]))
    return (vx, vy, yaw)
