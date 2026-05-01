from __future__ import annotations

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher

from algorithms.urdf_learn_wasd_walk.isaac_app_args import apply_project_kit_args
from algorithms.urdf_learn_wasd_walk.isaac_lock import acquire_isaac_lock
from algorithms.urdf_learn_wasd_walk.runtime import supported_robot_keys
from algorithms.urdf_learn_wasd_walk.task_registry import LANDAU_CURRICULUM_STAGES


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Check reset / idle pose stability under zero actions or a zero-command policy."
    )
    parser.add_argument("--robot", choices=supported_robot_keys(), required=True)
    parser.add_argument("--stage", choices=LANDAU_CURRICULUM_STAGES, default=None)
    parser.add_argument(
        "--use-train-env",
        action="store_true",
        default=False,
        help="Run the diagnose scenario against the training env config instead of the playback env.",
    )
    parser.add_argument("--disable_fabric", action="store_true", default=False)
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--load_run", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--latest", action="store_true", default=False)
    parser.add_argument("--experiment_name", type=str, default=None)
    parser.add_argument("--workflow-id", type=str, default=None, help="Stable workflow identifier for history correlation.")
    parser.add_argument("--playback-compat-mode", choices=("strict", "control_only", "off"), default="strict")
    parser.add_argument("--steps", type=int, default=240)
    parser.add_argument("--command-vx", type=float, default=0.0)
    parser.add_argument("--command-vy", type=float, default=0.0)
    parser.add_argument("--command-yaw", type=float, default=0.0)
    parser.add_argument("--action-mode", choices=("zero", "init_pose", "policy"), default="zero")
    parser.add_argument("--pose-json", type=str, default=None, help="Optional JSON file with joint_pos overrides.")
    parser.add_argument("--sample-start", type=int, default=20)
    parser.add_argument("--sample-stop", type=int, default=120)
    parser.add_argument("--sample-until-first-done", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-done-count", type=int, default=None)
    parser.add_argument("--min-control-root-height", type=float, default=None)
    parser.add_argument(
        "--terrain-mode",
        choices=("flat", "game"),
        default="flat",
        help="Playback scene for Landau game diagnostics. 'flat' is the simple default; 'game' uses the small rough map.",
    )
    parser.add_argument(
        "--obstacles",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep obstacle sub-terrains when using --terrain-mode game.",
    )
    AppLauncher.add_app_launcher_args(parser)
    return parser


parser = _build_parser()
args_cli = parser.parse_args()
apply_project_kit_args(args_cli)

isaac_lock_handle = acquire_isaac_lock("check_pose_stability", args_cli)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab_tasks.utils.wrappers.rsl_rl import RslRlVecEnvWrapper

from algorithms.urdf_learn_wasd_walk.command_frame import semantic_command_to_env_command
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
from algorithms.urdf_learn_wasd_walk.run_history import write_diagnostic_record
from algorithms.urdf_learn_wasd_walk.runtime import resolve_robot_task_spec
from algorithms.urdf_learn_wasd_walk.task_registry import register_gym_envs

install_safe_actor_critic_distribution_patch()


def main() -> None:
    register_gym_envs()
    task_spec = resolve_robot_task_spec(args_cli.robot, stage=args_cli.stage)
    env_task_id = task_spec.train_task_id if args_cli.use_train_env else task_spec.play_task_id
    env_cfg, agent_cfg = load_env_and_runner_cfg(env_task_id, args_cli)
    env_cfg.scene.num_envs = 1
    if not args_cli.use_train_env and args_cli.robot == "landau" and args_cli.stage == "game":
        configure_landau_game_playback_scene(
            env_cfg,
            terrain_mode=args_cli.terrain_mode,
            obstacles_enabled=args_cli.obstacles,
        )
    if args_cli.pose_json:
        pose_overrides = json.loads(Path(args_cli.pose_json).read_text(encoding="utf-8"))
        if not isinstance(pose_overrides, dict):
            raise ValueError("--pose-json must contain a JSON object of joint_name -> value.")
        env_cfg.scene.robot.init_state.joint_pos.update(
            {str(joint_name): float(value) for joint_name, value in pose_overrides.items()}
        )

    resume_path: str | None = None
    checkpoint_selection_source = "none"
    if args_cli.action_mode == "policy":
        log_root_path = log_root_for_experiment(agent_cfg.experiment_name)
        checkpoint_selection = resolve_checkpoint_selection(log_root_path, agent_cfg, prefer_latest=args_cli.latest)
        checkpoint_selection_source = str(checkpoint_selection["source"])
        resume_path = checkpoint_selection["path"]
        print(f"[DIAG] checkpoint={resume_path} source={checkpoint_selection['source']}", flush=True)
        if checkpoint_selection["source"] == "recommended":
            reason = checkpoint_selection["entry"].get("reason")
            if reason:
                print(f"[DIAG] recommended note: {reason}", flush=True)
        if apply_checkpoint_playback_compat(env_cfg, resume_path, mode=args_cli.playback_compat_mode):
            print(
                "[DIAG] applied playback compatibility overrides from checkpoint params/env.yaml "
                f"(mode={args_cli.playback_compat_mode})",
                flush=True,
            )

    env = gym.make(env_task_id, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)
    robot = env.unwrapped.scene["robot"]
    termination_manager = getattr(env.unwrapped, "termination_manager", None)
    termination_terms = list(getattr(termination_manager, "active_terms", [])) if termination_manager is not None else []
    requested_joint_names = list(getattr(env_cfg.actions.joint_pos, "joint_names", []))
    controlled_joint_names = [name for name in requested_joint_names if name in robot.joint_names]
    controlled_joint_ids = [robot.joint_names.index(name) for name in controlled_joint_names]
    control_root_name = getattr(task_spec, "control_root_link", None) or getattr(task_spec, "root_link_name", None)
    support_names = tuple(getattr(task_spec, "support_link_names", ()))
    control_root_ids, _ = robot.find_bodies([control_root_name], preserve_order=True)
    support_ids, _ = robot.find_bodies(list(support_names), preserve_order=True)

    obs, _ = env.get_observations()
    zero_actions = torch.zeros(
        (env.unwrapped.num_envs, env.unwrapped.action_manager.total_action_dim),
        device=env.unwrapped.device,
        dtype=torch.float32,
    )
    init_pose_actions = build_init_pose_action(env.unwrapped, env_cfg)
    policy = None
    if resume_path is not None:
        ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        checkpoint_load = load_runner_checkpoint(ppo_runner, resume_path, load_optimizer=False)
        print(
            f"[DIAG] loaded inference checkpoint without optimizer state "
            f"(mode={checkpoint_load['mode']})",
            flush=True,
        )
        policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    semantic_command = (args_cli.command_vx, args_cli.command_vy, args_cli.command_yaw)
    env_command = clamp_base_velocity_command(
        env_cfg,
        semantic_command_to_env_command(task_spec.forward_body_axis, semantic_command),
    )
    print(
        f"[DIAG] action_mode={args_cli.action_mode} semantic_command={semantic_command} env_command={env_command} "
        f"sample_window=[{args_cli.sample_start}, {args_cli.sample_stop})",
        flush=True,
    )

    done_count_total = 0
    first_done_step: int | None = None
    first_termination_terms: list[str] = []
    min_control_root_height = float("inf")
    final_control_root_height = None
    sample_count = 0
    joint_sum = torch.zeros(len(controlled_joint_ids), dtype=torch.float64)
    joint_last = torch.zeros(len(controlled_joint_ids), dtype=torch.float64)
    action_abs_sum = 0.0
    termination_counts = {term_name: 0 for term_name in termination_terms}

    for step in range(args_cli.steps):
        if not simulation_app.is_running():
            break

        force_base_velocity_command(env.unwrapped, env_command)
        obs, _ = env.get_observations()
        with torch.inference_mode():
            if policy is None:
                actions = init_pose_actions if args_cli.action_mode == "init_pose" else zero_actions
            else:
                actions = policy(obs)
            obs, _, dones, _ = env.step(actions)

        if control_root_ids and support_ids:
            control_height = float(robot.data.body_pos_w[0, control_root_ids[0], 2].detach().cpu().item())
            support_floor = float(torch.min(robot.data.body_pos_w[0, support_ids, 2]).detach().cpu().item())
            relative_height = control_height - support_floor
            min_control_root_height = min(min_control_root_height, relative_height)
            final_control_root_height = relative_height

        done_step = int(torch.count_nonzero(dones).item())
        done_count_total += done_step
        active_termination_terms: list[str] = []
        if termination_manager is not None:
            for term_name in termination_terms:
                term_hits = int(torch.count_nonzero(termination_manager.get_term(term_name)).item())
                termination_counts[term_name] += term_hits
                if term_hits > 0:
                    active_termination_terms.append(term_name)
        if done_step > 0 and first_done_step is None:
            first_done_step = step
            first_termination_terms = list(active_termination_terms)

        in_sample_window = args_cli.sample_start <= step < args_cli.sample_stop
        if args_cli.sample_until_first_done and first_done_step is not None and step >= first_done_step:
            in_sample_window = False
        if in_sample_window and controlled_joint_ids:
            joint_snapshot = robot.data.joint_pos[0, controlled_joint_ids].detach().cpu().to(torch.float64)
            joint_sum += joint_snapshot
            joint_last = joint_snapshot
            sample_count += 1
            action_abs_sum += float(torch.mean(torch.abs(actions[0])).detach().cpu().item())

    robot_xy = tuple(float(value) for value in robot.data.root_pos_w[0, :2].detach().cpu().tolist())
    joint_mean = {}
    joint_last_map = {}
    mean_abs_action = None
    if sample_count > 0 and controlled_joint_ids:
        joint_mean_tensor = joint_sum / sample_count
        joint_mean = {
            joint_name: float(joint_mean_tensor[index].item())
            for index, joint_name in enumerate(controlled_joint_names)
        }
        joint_last_map = {
            joint_name: float(joint_last[index].item())
            for index, joint_name in enumerate(controlled_joint_names)
        }
        mean_abs_action = action_abs_sum / sample_count

    metrics = {
        "action_mode": args_cli.action_mode,
        "env_kind": "train" if args_cli.use_train_env else "play",
        "semantic_command": semantic_command,
        "env_command": env_command,
        "total_steps": args_cli.steps,
        "done_count_total": done_count_total,
        "first_done_step": -1 if first_done_step is None else first_done_step,
        "first_termination_terms": first_termination_terms,
        "termination_counts": termination_counts,
        "min_control_root_height": None if min_control_root_height == float("inf") else min_control_root_height,
        "final_control_root_height": final_control_root_height,
        "robot_xy": robot_xy,
        "sample_count": sample_count,
        "controlled_joint_pos_mean": joint_mean,
        "controlled_joint_pos_last": joint_last_map,
        "mean_abs_action": mean_abs_action,
    }
    print(f"[DIAG] metrics={metrics}", flush=True)

    failure: str | None = None
    failure_code = None
    failed_checks: list[str] = []
    if args_cli.max_done_count is not None and done_count_total > args_cli.max_done_count:
        failure = f"done count failed: value={done_count_total} threshold={args_cli.max_done_count}"
        failure_code = "done_count"
        failed_checks.append(f"done_count_total={done_count_total}")
    if failure is None and args_cli.min_control_root_height is not None:
        measured_height = 0.0 if min_control_root_height == float("inf") else min_control_root_height
        if measured_height < args_cli.min_control_root_height:
            failure = (
                f"control root height failed: value={measured_height:.4f} "
                f"threshold={args_cli.min_control_root_height:.4f}"
            )
            failure_code = "control_root_height"
            failed_checks.append(
                f"min_control_root_height={measured_height:.4f} < {args_cli.min_control_root_height:.4f}"
            )

    write_diagnostic_record(
        checkpoint_path=resume_path,
        task_spec=task_spec,
        args=args_cli,
        scenario="pose_stability",
        status="failed" if failure is not None else "completed",
        metrics=metrics,
        experiment_name=agent_cfg.experiment_name,
        failure=failure,
        gate_result={
            "selection_source": checkpoint_selection_source,
            "selected_checkpoint": None if resume_path is None else str(resume_path),
            "failure_code": failure_code,
            "failed_checks": failed_checks,
        },
    )

    env.close()
    if failure is not None:
        raise AssertionError(failure)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
        isaac_lock_handle.release()
