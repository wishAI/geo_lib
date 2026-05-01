from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import torch

from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

from .run_history import resolve_recommended_checkpoint
from .training_lineage import resolve_landau_experiment_name


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
    elif getattr(args, "robot", None) == "landau":
        agent_cfg.experiment_name = resolve_landau_experiment_name(getattr(args, "stage", None) or "full")
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
    if getattr(args, "seed", None) is not None and hasattr(env_cfg, "seed"):
        env_cfg.seed = args.seed
    agent_cfg = load_cfg_from_registry(task_id, "rsl_rl_cfg_entry_point")
    return env_cfg, apply_runner_overrides(agent_cfg, args)


def log_root_for_experiment(experiment_name: str) -> str:
    return os.path.abspath(os.path.join("logs", "rsl_rl", experiment_name))


def _validate_checkpoint_path(checkpoint_path: str) -> str:
    checkpoint_path = os.path.abspath(checkpoint_path)
    checkpoint_name = os.path.basename(checkpoint_path)
    if not re.fullmatch(r"model_.*\.pt", checkpoint_name):
        raise ValueError(
            f"Resolved checkpoint '{checkpoint_path}' is not an RSL-RL model checkpoint. "
            "Pass --checkpoint model_<iter>.pt or leave --checkpoint unset."
        )
    return checkpoint_path


def resolve_resume_path(log_root: str, agent_cfg: Any, explicit_checkpoint: str | None = None) -> str:
    if explicit_checkpoint is not None:
        checkpoint_path = os.path.abspath(explicit_checkpoint)
        if not Path(checkpoint_path).is_file():
            raise FileNotFoundError(f"Resume checkpoint does not exist: {checkpoint_path}")
        return _validate_checkpoint_path(checkpoint_path)
    checkpoint_path = get_checkpoint_path(log_root, agent_cfg.load_run, agent_cfg.load_checkpoint)
    return _validate_checkpoint_path(checkpoint_path)


def _is_default_checkpoint_selector(load_run: Any, load_checkpoint: Any) -> bool:
    normalized_run = None if load_run is None else str(load_run).strip()
    normalized_checkpoint = None if load_checkpoint is None else str(load_checkpoint).strip()
    return normalized_run in {None, "", ".*"} and normalized_checkpoint in {None, "", "model_.*.pt"}


def resolve_checkpoint_selection(
    log_root: str,
    agent_cfg: Any,
    *,
    prefer_latest: bool = False,
) -> dict[str, Any]:
    experiment_name = getattr(agent_cfg, "experiment_name", None)
    load_run = getattr(agent_cfg, "load_run", None)
    load_checkpoint = getattr(agent_cfg, "load_checkpoint", None)
    explicit_request = not _is_default_checkpoint_selector(load_run, load_checkpoint)
    if not prefer_latest and not explicit_request:
        recommended_entry = resolve_recommended_checkpoint(experiment_name)
        if recommended_entry is not None:
            return {
                "path": _validate_checkpoint_path(recommended_entry["checkpoint_path"]),
                "source": "recommended",
                "entry": recommended_entry,
            }
        if isinstance(experiment_name, str) and experiment_name.startswith("geo_landau_"):
            raise ValueError(
                f"No recommended checkpoint is registered for '{experiment_name}'. "
                "Promote a checkpoint first or rerun with --latest / explicit --load_run --checkpoint."
            )

    checkpoint_path = resolve_resume_path(log_root, agent_cfg)
    return {
        "path": checkpoint_path,
        "source": "explicit" if explicit_request else "latest",
        "entry": None,
    }


def resolve_checkpoint(log_root: str, agent_cfg: Any, *, prefer_latest: bool = False) -> str:
    return resolve_checkpoint_selection(log_root, agent_cfg, prefer_latest=prefer_latest)["path"]


def _partially_copy_actor_critic_state(
    target_state: dict[str, torch.Tensor],
    source_state: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    merged_state = {key: value.clone() for key, value in target_state.items()}
    exact_keys: list[str] = []
    partial_keys: list[str] = []
    skipped_keys: list[str] = []
    zero_initialized_partial_keys: list[str] = []
    partial_copy_keys = {"actor.0.weight", "critic.0.weight"}

    for key, target_tensor in target_state.items():
        source_tensor = source_state.get(key)
        if source_tensor is None:
            skipped_keys.append(key)
            continue
        if target_tensor.shape == source_tensor.shape:
            merged_state[key] = source_tensor.clone()
            exact_keys.append(key)
            continue
        if (
            key in partial_copy_keys
            and target_tensor.ndim == 2
            and source_tensor.ndim == 2
            and target_tensor.shape[0] == source_tensor.shape[0]
        ):
            # Preserve the old Stage A/B behavior and let the new observation channels
            # learn in gradually. Leaving the extra inputs randomly initialized makes the
            # Stage B -> game handoff much noisier than necessary.
            merged_tensor = torch.zeros_like(target_tensor)
            shared_width = min(int(target_tensor.shape[1]), int(source_tensor.shape[1]))
            merged_tensor[:, :shared_width] = source_tensor[:, :shared_width]
            merged_state[key] = merged_tensor
            partial_keys.append(key)
            if target_tensor.shape[1] > shared_width:
                zero_initialized_partial_keys.append(key)
            continue
        skipped_keys.append(key)

    return merged_state, {
        "mode": "partial",
        "exact_key_count": len(exact_keys),
        "partial_key_count": len(partial_keys),
        "skipped_key_count": len(skipped_keys),
        "partial_keys": partial_keys,
        "skipped_keys": skipped_keys,
        "zero_initialized_partial_keys": zero_initialized_partial_keys,
    }


def load_runner_checkpoint(
    runner,
    checkpoint_path: str,
    *,
    load_optimizer: bool = True,
) -> dict[str, Any]:
    loaded_dict = torch.load(checkpoint_path, weights_only=False)
    model_state = loaded_dict["model_state_dict"]
    metadata: dict[str, Any] = {
        "checkpoint_path": os.path.abspath(checkpoint_path),
        "mode": "strict",
        "optimizer_loaded": False,
        "partial_key_count": 0,
        "skipped_key_count": 0,
        "partial_keys": [],
        "skipped_keys": [],
    }

    try:
        runner.alg.actor_critic.load_state_dict(model_state)
        strict_loaded = True
    except RuntimeError:
        strict_loaded = False

    if not strict_loaded:
        merged_state, partial_metadata = _partially_copy_actor_critic_state(
            runner.alg.actor_critic.state_dict(),
            model_state,
        )
        runner.alg.actor_critic.load_state_dict(merged_state)
        metadata.update(partial_metadata)

    if getattr(runner.alg, "rnd", None):
        runner.alg.rnd.load_state_dict(loaded_dict["rnd_state_dict"])

    if getattr(runner, "empirical_normalization", False):
        obs_norm_state = loaded_dict.get("obs_norm_state_dict")
        critic_obs_norm_state = loaded_dict.get("critic_obs_norm_state_dict")
        if obs_norm_state is not None:
            runner.obs_normalizer.load_state_dict(obs_norm_state)
        if critic_obs_norm_state is not None:
            runner.critic_obs_normalizer.load_state_dict(critic_obs_norm_state)

    if load_optimizer and metadata["mode"] == "strict":
        optimizer_state = loaded_dict.get("optimizer_state_dict")
        if optimizer_state is not None:
            runner.alg.optimizer.load_state_dict(optimizer_state)
            metadata["optimizer_loaded"] = True
        if getattr(runner.alg, "rnd", None) and "rnd_optimizer_state_dict" in loaded_dict:
            runner.alg.rnd_optimizer.load_state_dict(loaded_dict["rnd_optimizer_state_dict"])
    runner.current_learning_iteration = int(loaded_dict.get("iter", 0))
    return metadata


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


def _restore_checkpoint_action_cfg(env_cfg: Any, env_params: dict[str, Any]) -> bool:
    applied = False
    joint_pos_cfg = getattr(getattr(env_cfg, "actions", None), "joint_pos", None)
    if joint_pos_cfg is None:
        return False

    action_cfg = env_params.get("actions", {}).get("joint_pos", {})
    action_scale = action_cfg.get("scale")
    if isinstance(action_scale, dict):
        joint_pos_cfg.scale = dict(action_scale)
        applied = True

    joint_names = action_cfg.get("joint_names")
    if isinstance(joint_names, (list, tuple)):
        joint_pos_cfg.joint_names = [str(name) for name in joint_names]
        applied = True

    for attr_name in ("offset", "clip"):
        value = action_cfg.get(attr_name)
        if value is None or not hasattr(joint_pos_cfg, attr_name):
            continue
        setattr(joint_pos_cfg, attr_name, value)
        applied = True

    for attr_name in ("preserve_order", "use_default_offset"):
        value = action_cfg.get(attr_name)
        if value is None or not hasattr(joint_pos_cfg, attr_name):
            continue
        setattr(joint_pos_cfg, attr_name, bool(value))
        applied = True

    return applied


def _restore_checkpoint_command_ranges(env_cfg: Any, env_params: dict[str, Any]) -> bool:
    applied = False
    command_ranges = env_params.get("commands", {}).get("base_velocity", {}).get("ranges", {})
    target_ranges = getattr(getattr(getattr(env_cfg, "commands", None), "base_velocity", None), "ranges", None)
    if not isinstance(command_ranges, dict) or target_ranges is None:
        return False

    for attr_name in ("lin_vel_x", "lin_vel_y", "ang_vel_z"):
        range_value = command_ranges.get(attr_name)
        if range_value is None or not hasattr(target_ranges, attr_name):
            continue
        setattr(target_ranges, attr_name, tuple(float(component) for component in range_value))
        applied = True

    return applied


def _restore_checkpoint_init_state(env_cfg: Any, env_params: dict[str, Any]) -> bool:
    applied = False
    init_state_cfg = getattr(getattr(getattr(env_cfg, "scene", None), "robot", None), "init_state", None)
    if init_state_cfg is None:
        return False

    init_state = env_params.get("scene", {}).get("robot", {}).get("init_state", {})
    if not isinstance(init_state, dict):
        return False

    for attr_name in ("pos", "rot", "lin_vel", "ang_vel"):
        value = init_state.get(attr_name)
        if value is None or not hasattr(init_state_cfg, attr_name):
            continue
        setattr(init_state_cfg, attr_name, tuple(float(component) for component in value))
        applied = True

    joint_pos = init_state.get("joint_pos")
    if isinstance(joint_pos, dict) and hasattr(init_state_cfg, "joint_pos"):
        current_joint_pos = getattr(init_state_cfg, "joint_pos", None)
        restored_joint_pos = dict(current_joint_pos) if isinstance(current_joint_pos, dict) else {}
        restored_joint_pos.update({str(name): float(value) for name, value in joint_pos.items()})
        init_state_cfg.joint_pos = restored_joint_pos
        applied = True

    joint_vel = init_state.get("joint_vel")
    if joint_vel is not None and hasattr(init_state_cfg, "joint_vel"):
        if isinstance(joint_vel, dict):
            current_joint_vel = getattr(init_state_cfg, "joint_vel", None)
            restored_joint_vel = dict(current_joint_vel) if isinstance(current_joint_vel, dict) else {}
            restored_joint_vel.update({str(name): float(value) for name, value in joint_vel.items()})
            init_state_cfg.joint_vel = restored_joint_vel
            applied = True
        elif isinstance(joint_vel, (list, tuple)):
            init_state_cfg.joint_vel = tuple(float(component) for component in joint_vel)
            applied = True

    return applied


def _restore_checkpoint_actuators(env_cfg: Any, env_params: dict[str, Any]) -> bool:
    applied = False
    robot_cfg = getattr(getattr(env_cfg, "scene", None), "robot", None)
    current_actuators = getattr(robot_cfg, "actuators", None)
    saved_actuators = env_params.get("scene", {}).get("robot", {}).get("actuators", {})
    if not isinstance(current_actuators, dict) or not isinstance(saved_actuators, dict):
        return False

    field_names = (
        "effort_limit",
        "velocity_limit",
        "effort_limit_sim",
        "velocity_limit_sim",
        "stiffness",
        "damping",
        "armature",
        "friction",
    )

    actuator_aliases = {
        "ankles": ("ankles", "feet"),
        "toes": ("toes", "feet"),
    }

    for actuator_name, actuator_cfg in current_actuators.items():
        if actuator_cfg is None:
            continue
        candidate_names = actuator_aliases.get(actuator_name, (actuator_name,))
        saved_cfg = next(
            (
                saved_actuators[candidate_name]
                for candidate_name in candidate_names
                if isinstance(saved_actuators.get(candidate_name), dict)
            ),
            None,
        )
        if saved_cfg is None:
            continue
        for field_name in field_names:
            if field_name not in saved_cfg or not hasattr(actuator_cfg, field_name):
                continue
            setattr(actuator_cfg, field_name, saved_cfg[field_name])
            applied = True

    return applied


def apply_checkpoint_playback_compat(
    env_cfg: Any,
    checkpoint_path: str,
    *,
    mode: str = "strict",
) -> bool:
    """Restore checkpoint-era config for inference tools.

    Modes:
    - ``strict``: restore action mapping, command ranges, init pose, and actuators.
    - ``control_only``: restore only the action mapping and command ranges.
    - ``off``: disable checkpoint-era env restoration.
    """

    normalized_mode = str(mode).strip().lower()
    if normalized_mode == "off":
        return False
    if normalized_mode not in {"strict", "control_only"}:
        raise ValueError(f"Unsupported playback compat mode '{mode}'.")

    env_params = _load_checkpoint_env_params(checkpoint_path)
    if env_params is None:
        return False

    applied = False
    applied |= _restore_checkpoint_action_cfg(env_cfg, env_params)
    applied |= _restore_checkpoint_command_ranges(env_cfg, env_params)
    if normalized_mode == "strict":
        applied |= _restore_checkpoint_init_state(env_cfg, env_params)
        applied |= _restore_checkpoint_actuators(env_cfg, env_params)
    return applied


def _resolve_joint_action_scale(
    joint_names: list[str],
    scale_cfg: Any,
    *,
    device: str | torch.device,
    num_envs: int,
) -> torch.Tensor:
    if isinstance(scale_cfg, (float, int)):
        return torch.full((num_envs, len(joint_names)), float(scale_cfg), device=device, dtype=torch.float32)

    resolved = torch.ones((num_envs, len(joint_names)), device=device, dtype=torch.float32)
    if not isinstance(scale_cfg, dict):
        return resolved

    for joint_index, joint_name in enumerate(joint_names):
        for pattern, raw_value in scale_cfg.items():
            if re.fullmatch(str(pattern), joint_name):
                resolved[:, joint_index] = float(raw_value)
    return resolved


def build_init_pose_action(env, env_cfg: Any) -> torch.Tensor:
    """Build an action tensor that targets the configured robot init pose."""

    robot = env.scene["robot"]
    joint_pos_cfg = getattr(getattr(env_cfg, "actions", None), "joint_pos", None)
    init_state_cfg = getattr(getattr(getattr(env_cfg, "scene", None), "robot", None), "init_state", None)
    action_dim = int(env.action_manager.total_action_dim)
    actions = torch.zeros((env.num_envs, action_dim), device=env.device, dtype=torch.float32)
    if joint_pos_cfg is None or init_state_cfg is None:
        return actions

    try:
        joint_pos_term = env.action_manager.get_term("joint_pos")
    except Exception:
        joint_pos_term = None
    if joint_pos_term is None:
        return actions

    controlled_joint_names = list(getattr(joint_pos_term, "_joint_names", []))
    joint_ids = getattr(joint_pos_term, "_joint_ids", None)
    if not controlled_joint_names or joint_ids is None:
        return actions

    default_joint_pos = robot.data.default_joint_pos[:, joint_ids].to(dtype=torch.float32)
    target_joint_pos = default_joint_pos.clone()
    init_joint_pos = getattr(init_state_cfg, "joint_pos", None)
    if isinstance(init_joint_pos, dict):
        for joint_index, joint_name in enumerate(controlled_joint_names):
            if joint_name in init_joint_pos:
                target_joint_pos[:, joint_index] = float(init_joint_pos[joint_name])

    scale = _resolve_joint_action_scale(
        controlled_joint_names,
        getattr(joint_pos_cfg, "scale", 1.0),
        device=env.device,
        num_envs=env.num_envs,
    )
    valid_scale = torch.abs(scale) > 1.0e-6
    raw_action = torch.zeros_like(target_joint_pos)

    offset_cfg = getattr(joint_pos_cfg, "offset", 0.0)
    if bool(getattr(joint_pos_cfg, "use_default_offset", True)):
        offset = default_joint_pos.clone()
    elif isinstance(offset_cfg, (float, int)):
        offset = torch.full_like(target_joint_pos, float(offset_cfg))
    elif isinstance(offset_cfg, dict):
        offset = torch.zeros_like(target_joint_pos)
        for joint_index, joint_name in enumerate(controlled_joint_names):
            if joint_name in offset_cfg:
                offset[:, joint_index] = float(offset_cfg[joint_name])
    else:
        offset = torch.zeros_like(target_joint_pos)

    raw_action[valid_scale] = (target_joint_pos[valid_scale] - offset[valid_scale]) / scale[valid_scale]

    action_offset = 0
    for term_name, term_dim in zip(env.action_manager.active_terms, env.action_manager.action_term_dim, strict=False):
        if term_name == "joint_pos":
            actions[:, action_offset : action_offset + len(controlled_joint_names)] = raw_action
            break
        action_offset += int(term_dim)
    return actions


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
