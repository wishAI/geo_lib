from __future__ import annotations

import argparse
import os
from datetime import datetime

from isaaclab.app import AppLauncher

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
    parser.add_argument("--reset_optimizer", action="store_true", default=False)
    parser.add_argument("--experiment_name", type=str, default=None)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--resume", type=bool, default=None)
    parser.add_argument("--load_run", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--logger", type=str, default=None, choices={"wandb", "tensorboard", "neptune"})
    parser.add_argument("--log_project_name", type=str, default=None)
    AppLauncher.add_app_launcher_args(parser)
    return parser


parser = _build_parser()
args_cli = parser.parse_args()
if args_cli.video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml
from isaaclab_tasks.utils.wrappers.rsl_rl import RslRlVecEnvWrapper

from algorithms.urdf_learn_wasd_walk.isaac_workflow import load_env_and_runner_cfg, log_root_for_experiment, resolve_checkpoint
from algorithms.urdf_learn_wasd_walk.runtime import resolve_robot_task_spec
from algorithms.urdf_learn_wasd_walk.task_registry import register_gym_envs


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def _apply_runtime_algorithm_overrides(runner: OnPolicyRunner, args: argparse.Namespace) -> None:
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


def main() -> None:
    register_gym_envs()
    task_spec = resolve_robot_task_spec(args_cli.robot, stage=args_cli.stage)
    env_cfg, agent_cfg = load_env_and_runner_cfg(task_spec.train_task_id, args_cli)

    log_root_path = log_root_for_experiment(agent_cfg.experiment_name)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    env = gym.make(task_spec.train_task_id, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
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

    env = RslRlVecEnvWrapper(env)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    runner.add_git_repo_to_log(__file__)

    if agent_cfg.resume:
        resume_path = resolve_checkpoint(log_root_path, agent_cfg)
        print(f"[INFO] Resuming from checkpoint: {resume_path}")
        runner.load(resume_path)
    _apply_runtime_algorithm_overrides(runner, args_cli)

    env.seed(agent_cfg.seed)
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
