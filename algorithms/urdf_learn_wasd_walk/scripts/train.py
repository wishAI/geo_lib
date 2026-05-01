from __future__ import annotations

import argparse
import faulthandler
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher

from algorithms.urdf_learn_wasd_walk.isaac_app_args import apply_project_kit_args
from algorithms.urdf_learn_wasd_walk.runtime import supported_robot_keys
from algorithms.urdf_learn_wasd_walk.task_registry import LANDAU_CURRICULUM_STAGES


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a flat-ground locomotion policy with RSL-RL.")
    parser.add_argument("--robot", choices=supported_robot_keys(), required=True)
    parser.add_argument(
        "--stage",
        choices=LANDAU_CURRICULUM_STAGES,
        default=None,
        help="Landau curriculum stage. Ignored for non-Landau robots.",
    )
    parser.add_argument("--video", action="store_true", default=False, help="Record training videos.")
    parser.add_argument("--video_length", type=int, default=200)
    parser.add_argument("--video_interval", type=int, default=2000)
    parser.add_argument("--disable_fabric", action="store_true", default=False)
    parser.add_argument("--num_envs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max_iterations", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--entropy_coef", type=float, default=None)
    parser.add_argument("--desired_kl", type=float, default=None)
    parser.add_argument("--num_learning_epochs", type=int, default=None)
    parser.add_argument("--num_mini_batches", type=int, default=None)
    parser.add_argument(
        "--action_noise_std",
        type=float,
        default=None,
        help="Override the actor action noise std after checkpoint load. Useful for rescue fine-tunes on rough terrain.",
    )
    parser.add_argument("--reset_optimizer", action="store_true", default=False)
    parser.add_argument("--experiment_name", type=str, default=None)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--workflow-id", type=str, default=None, help="Stable workflow identifier for history correlation.")
    parser.add_argument("--resume", type=bool, default=None)
    parser.add_argument("--load_run", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument(
        "--resume-checkpoint",
        type=str,
        default=None,
        help="Explicit model_*.pt path to load. Supports cross-stage handoff when stage experiments live under different log roots.",
    )
    parser.add_argument("--logger", type=str, default=None, choices={"wandb", "tensorboard", "neptune"})
    parser.add_argument("--log_project_name", type=str, default=None)
    AppLauncher.add_app_launcher_args(parser)
    return parser


parser = _build_parser()
args_cli = parser.parse_args()
apply_project_kit_args(args_cli)
if args_cli.video:
    args_cli.enable_cameras = True

faulthandler.enable(all_threads=True)

from algorithms.urdf_learn_wasd_walk.isaac_lock import acquire_isaac_lock

isaac_lock_handle = acquire_isaac_lock("train", args_cli)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml

from algorithms.urdf_learn_wasd_walk.isaac_workflow import (
    load_env_and_runner_cfg,
    load_runner_checkpoint,
    log_root_for_experiment,
    resolve_resume_path,
)
from algorithms.urdf_learn_wasd_walk.run_history import write_training_record
from algorithms.urdf_learn_wasd_walk.runtime import resolve_robot_task_spec
from algorithms.urdf_learn_wasd_walk.task_registry import register_gym_envs


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def _debug_trace_path() -> Path:
    return Path(__file__).resolve().parents[1] / "outputs" / "history" / "runtime_debug" / "train_bootstrap.jsonl"


def _emit_bootstrap_event(event: str, **payload) -> None:
    record = {
        "event": event,
        "pid": os.getpid(),
        "recorded_at": datetime.utcnow().isoformat() + "Z",
        "args": vars(args_cli),
        **payload,
    }
    line = json.dumps(record, sort_keys=True)
    print(f"[TRAIN_BOOTSTRAP] {line}", flush=True)
    trace_path = _debug_trace_path()
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    with trace_path.open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")


def _apply_runtime_algorithm_overrides(runner, args: argparse.Namespace) -> None:
    algorithm = getattr(runner, "alg", None)
    if algorithm is None:
        return
    if args.learning_rate is not None and hasattr(algorithm, "learning_rate"):
        algorithm.learning_rate = args.learning_rate
    if args.entropy_coef is not None and hasattr(algorithm, "entropy_coef"):
        algorithm.entropy_coef = args.entropy_coef
    if args.desired_kl is not None and hasattr(algorithm, "desired_kl"):
        algorithm.desired_kl = args.desired_kl
    if args.num_learning_epochs is not None and hasattr(algorithm, "num_learning_epochs"):
        algorithm.num_learning_epochs = args.num_learning_epochs
    if args.num_mini_batches is not None and hasattr(algorithm, "num_mini_batches"):
        algorithm.num_mini_batches = args.num_mini_batches
    optimizer = getattr(algorithm, "optimizer", None)
    if optimizer is not None and args.learning_rate is not None:
        for param_group in optimizer.param_groups:
            param_group["lr"] = args.learning_rate
    if optimizer is not None and args.reset_optimizer:
        optimizer.state.clear()


def _apply_runtime_policy_overrides(runner, args: argparse.Namespace) -> None:
    actor_critic = getattr(getattr(runner, "alg", None), "actor_critic", None)
    if actor_critic is None or args.action_noise_std is None:
        return
    target_std = max(float(args.action_noise_std), 1.0e-4)
    with torch.no_grad():
        if hasattr(actor_critic, "log_std"):
            actor_critic.log_std.fill_(math.log(target_std))
        elif hasattr(actor_critic, "std"):
            actor_critic.std.fill_(target_std)


def main() -> None:
    _emit_bootstrap_event("main_start")
    register_gym_envs()
    _emit_bootstrap_event("gym_registered")
    task_spec = resolve_robot_task_spec(args_cli.robot, stage=args_cli.stage)
    _emit_bootstrap_event("task_resolved", train_task_id=task_spec.train_task_id, play_task_id=task_spec.play_task_id)
    env_task_id = task_spec.train_task_id
    env_cfg, agent_cfg = load_env_and_runner_cfg(env_task_id, args_cli)
    _emit_bootstrap_event(
        "cfg_loaded",
        experiment_name=getattr(agent_cfg, "experiment_name", None),
        max_iterations=getattr(agent_cfg, "max_iterations", None),
        env_task_id=env_task_id,
        scene_num_envs=getattr(getattr(env_cfg, "scene", None), "num_envs", None),
        policy_corruption_enabled=getattr(getattr(env_cfg.observations, "policy", None), "enable_corruption", None),
    )

    log_root_path = log_root_for_experiment(agent_cfg.experiment_name)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    _emit_bootstrap_event("log_dir_prepared", log_dir=log_dir)

    if args_cli.video:
        env = gym.make(env_task_id, cfg=env_cfg, render_mode="rgb_array")
    else:
        env = gym.make(env_task_id, cfg=env_cfg)
    _emit_bootstrap_event("env_created", num_envs=getattr(env_cfg.scene, "num_envs", None))
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording training videos.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    from isaaclab_tasks.utils.wrappers.rsl_rl import RslRlVecEnvWrapper
    from rsl_rl.runners import OnPolicyRunner

    from algorithms.urdf_learn_wasd_walk.rsl_rl_safety import install_safe_actor_critic_distribution_patch

    install_safe_actor_critic_distribution_patch()
    env = RslRlVecEnvWrapper(env)
    _emit_bootstrap_event("env_wrapped")
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    _emit_bootstrap_event("runner_created", device=agent_cfg.device)
    runner.add_git_repo_to_log(__file__)
    _emit_bootstrap_event("git_logged")

    resume_path = None
    resume_metadata = None
    if args_cli.resume_checkpoint is not None:
        resume_path = resolve_resume_path(log_root_path, agent_cfg, explicit_checkpoint=args_cli.resume_checkpoint)
        print(f"[INFO] Loading explicit resume checkpoint: {resume_path}")
        resume_metadata = load_runner_checkpoint(runner, resume_path, load_optimizer=not args_cli.reset_optimizer)
        print(f"[INFO] Resume load summary: {resume_metadata}")
    elif agent_cfg.resume:
        resume_path = resolve_resume_path(log_root_path, agent_cfg)
        print(f"[INFO] Resuming from checkpoint: {resume_path}")
        resume_metadata = load_runner_checkpoint(runner, resume_path, load_optimizer=not args_cli.reset_optimizer)
        print(f"[INFO] Resume load summary: {resume_metadata}")
    _apply_runtime_algorithm_overrides(runner, args_cli)
    _apply_runtime_policy_overrides(runner, args_cli)
    _emit_bootstrap_event("runtime_overrides_applied", resume_path=resume_path)

    env.seed(agent_cfg.seed)
    _emit_bootstrap_event("env_seeded", seed=agent_cfg.seed)
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)
    _emit_bootstrap_event("params_dumped", log_dir=log_dir)
    write_training_record(
        log_dir=log_dir,
        task_spec=task_spec,
        args=args_cli,
        agent_cfg=agent_cfg,
        status="started",
        resume_path=resume_path,
        resume_metadata=resume_metadata,
    )
    _emit_bootstrap_event("training_record_started_written", log_dir=log_dir)
    try:
        _emit_bootstrap_event("learn_start", iterations=agent_cfg.max_iterations)
        runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    except Exception as exc:
        _emit_bootstrap_event("learn_failed", error=str(exc))
        write_training_record(
            log_dir=log_dir,
            task_spec=task_spec,
            args=args_cli,
            agent_cfg=agent_cfg,
            status="failed",
            resume_path=resume_path,
            resume_metadata=resume_metadata,
            error=str(exc),
        )
        raise
    _emit_bootstrap_event("learn_completed")
    write_training_record(
        log_dir=log_dir,
        task_spec=task_spec,
        args=args_cli,
        agent_cfg=agent_cfg,
        status="completed",
        resume_path=resume_path,
        resume_metadata=resume_metadata,
    )
    _emit_bootstrap_event("training_record_completed_written", log_dir=log_dir)
    env.close()
    _emit_bootstrap_event("env_closed")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
        isaac_lock_handle.release()
