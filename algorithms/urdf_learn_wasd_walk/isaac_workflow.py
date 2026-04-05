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
    algo_cfg = getattr(agent_cfg, "algorithm", None)
    if algo_cfg is not None:
        if getattr(args, "learning_rate", None) is not None:
            algo_cfg.learning_rate = args.learning_rate
        if getattr(args, "entropy_coef", None) is not None:
            algo_cfg.entropy_coef = args.entropy_coef
        if getattr(args, "desired_kl", None) is not None:
            algo_cfg.desired_kl = args.desired_kl
        if getattr(args, "num_learning_epochs", None) is not None:
            algo_cfg.num_learning_epochs = args.num_learning_epochs
        if getattr(args, "num_mini_batches", None) is not None:
            algo_cfg.num_mini_batches = args.num_mini_batches
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


def _load_checkpoint_env_params(checkpoint_path: str) -> dict[str, Any] | None:
    params_path = os.path.join(os.path.dirname(checkpoint_path), "params", "env.yaml")
    if not os.path.isfile(params_path):
        return None
    try:
        import yaml
    except ImportError:
        return None
    with open(params_path, "r", encoding="utf-8") as stream:
        loaded = yaml.unsafe_load(stream)
    return loaded if isinstance(loaded, dict) else None


def apply_checkpoint_playback_compat(env_cfg: Any, checkpoint_path: str) -> bool:
    """Restore checkpoint-era action scale and command ranges for inference tools.

    Older Landau checkpoints become misleadingly bad when replayed against later Stage A config
    changes. Replay/teleop/validation should prefer the saved training-time mapping if available.
    """

    env_params = _load_checkpoint_env_params(checkpoint_path)
    if env_params is None:
        return False

    applied = False
    action_scale = env_params.get("actions", {}).get("joint_pos", {}).get("scale")
    if isinstance(action_scale, dict) and hasattr(getattr(env_cfg, "actions", None), "joint_pos"):
        env_cfg.actions.joint_pos.scale = dict(action_scale)
        applied = True

    command_ranges = env_params.get("commands", {}).get("base_velocity", {}).get("ranges", {})
    target_ranges = getattr(getattr(getattr(env_cfg, "commands", None), "base_velocity", None), "ranges", None)
    if isinstance(command_ranges, dict) and target_ranges is not None:
        for attr_name in ("lin_vel_x", "lin_vel_y", "ang_vel_z"):
            range_value = command_ranges.get(attr_name)
            if range_value is None or not hasattr(target_ranges, attr_name):
                continue
            setattr(target_ranges, attr_name, tuple(float(component) for component in range_value))
            applied = True

    return applied


def force_base_velocity_command(env, command: tuple[float, float, float]) -> None:
    command_term = env.command_manager.get_term("base_velocity")
    command_tensor = torch.tensor(command, device=env.device, dtype=torch.float32).repeat(env.num_envs, 1)
    command_term.vel_command_b[:] = command_tensor
    if hasattr(command_term, "is_heading_env"):
        command_term.is_heading_env[:] = False
    if hasattr(command_term, "is_standing_env"):
        standing_mask = torch.linalg.norm(command_tensor[:, :2], dim=1) <= 0.05
        standing_mask &= torch.abs(command_tensor[:, 2]) <= 0.05
        command_term.is_standing_env[:] = standing_mask
    command_term.time_left[:] = 1.0e9


def _clamp_command_component(raw_value: float, lower: float, upper: float, preserve_zero_threshold: float) -> float:
    # In staged forward-only configs, snapping near-zero or backward user commands up to the
    # minimum forward velocity makes teleop look "automatic". Preserve an actual idle command.
    if lower > 0.0 and raw_value <= preserve_zero_threshold:
        return 0.0
    if upper < 0.0 and raw_value >= -preserve_zero_threshold:
        return 0.0
    return min(max(raw_value, lower), upper)


def clamp_base_velocity_command(
    env_cfg: Any,
    command: tuple[float, float, float],
    preserve_zero_threshold: float = 0.05,
) -> tuple[float, float, float]:
    ranges = env_cfg.commands.base_velocity.ranges
    vx = _clamp_command_component(
        float(command[0]),
        float(ranges.lin_vel_x[0]),
        float(ranges.lin_vel_x[1]),
        preserve_zero_threshold,
    )
    vy = _clamp_command_component(
        float(command[1]),
        float(ranges.lin_vel_y[0]),
        float(ranges.lin_vel_y[1]),
        preserve_zero_threshold,
    )
    yaw = _clamp_command_component(
        float(command[2]),
        float(ranges.ang_vel_z[0]),
        float(ranges.ang_vel_z[1]),
        preserve_zero_threshold,
    )
    return (vx, vy, yaw)
