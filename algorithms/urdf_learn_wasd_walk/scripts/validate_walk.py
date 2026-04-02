from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

from algorithms.urdf_learn_wasd_walk.runtime import supported_robot_keys
from algorithms.urdf_learn_wasd_walk.task_registry import LANDAU_CURRICULUM_STAGES


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate that a trained checkpoint loads and produces biped locomotion.")
    parser.add_argument("--robot", choices=supported_robot_keys(), required=True)
    parser.add_argument(
        "--stage",
        choices=LANDAU_CURRICULUM_STAGES,
        default=None,
        help="Landau curriculum stage. Sets stage-appropriate default thresholds.",
    )
    parser.add_argument("--disable_fabric", action="store_true", default=False)
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--load_run", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--experiment_name", type=str, default=None)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--command-vx", type=float, default=0.5)
    parser.add_argument("--command-vy", type=float, default=0.0)
    parser.add_argument("--command-yaw", type=float, default=0.0)
    parser.add_argument("--contact-threshold", type=float, default=1.0)
    parser.add_argument("--min-planar-displacement", type=float, default=0.05)
    parser.add_argument("--min-forward-displacement", type=float, default=None)
    parser.add_argument("--min-lateral-separation", type=float, default=0.03)
    parser.add_argument("--min-foot-planar-travel", type=float, default=0.03)
    parser.add_argument("--min-foot-height-range", type=float, default=0.01)
    parser.add_argument("--min-contact-switches", type=int, default=1)
    parser.add_argument("--max-done-count", type=int, default=4)
    AppLauncher.add_app_launcher_args(parser)
    return parser


parser = _build_parser()
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab_tasks.utils.wrappers.rsl_rl import RslRlVecEnvWrapper

from algorithms.urdf_learn_wasd_walk.command_frame import semantic_command_to_env_command, semantic_forward_dir_xy
from algorithms.urdf_learn_wasd_walk.isaac_workflow import (
    clamp_base_velocity_command,
    force_base_velocity_command,
    load_env_and_runner_cfg,
    log_root_for_experiment,
    resolve_checkpoint,
)
from algorithms.urdf_learn_wasd_walk.runtime import resolve_robot_task_spec
from algorithms.urdf_learn_wasd_walk.task_registry import register_gym_envs


def _contact_mask(contact_sensor, body_ids: list[int], threshold: float) -> torch.Tensor:
    net_forces = contact_sensor.data.net_forces_w_history[:, 0, body_ids]
    return torch.linalg.norm(net_forces, dim=-1) > threshold


def _group_support_links(link_names: tuple[str, ...]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    left_links = tuple(name for name in link_names if name.endswith("_l"))
    right_links = tuple(name for name in link_names if name.endswith("_r"))
    return left_links, right_links


def _body_axes_xy(quat_wxyz: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    w, x, y, z = (float(value) for value in quat_wxyz)
    body_x = torch.tensor(
        (
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y + z * w),
        ),
        dtype=torch.float32,
    )
    body_y = torch.tensor(
        (
            2.0 * (x * y - z * w),
            1.0 - 2.0 * (x * x + z * z),
        ),
        dtype=torch.float32,
    )
    return body_x, body_y


def _validate_metric(condition: bool, label: str, value, threshold) -> None:
    status = "PASS" if condition else "FAIL"
    print(f"[VALIDATE] {status} {label}: value={value} threshold={threshold}", flush=True)
    if not condition:
        raise AssertionError(f"{label} failed: value={value} threshold={threshold}")


def _apply_stage_defaults(args) -> None:
    """Override CLI defaults with stage-specific acceptance thresholds."""
    stage = args.stage
    if stage is None or args.robot != "landau":
        return
    if stage == "fwd_only":
        # forward-only: must show clear translation, no yaw needed
        if args.command_vx == 0.5 and args.command_vy == 0.0:
            # default command — override to match stage semantics (forward = env Y for Landau)
            args.command_vx = 0.0
            args.command_vy = 0.5
        if args.min_planar_displacement == 0.05:
            args.min_planar_displacement = 0.3
        if args.steps == 256:
            args.steps = 500
    elif stage == "fwd_yaw":
        if args.command_vx == 0.5 and args.command_vy == 0.0:
            args.command_vx = 0.0
            args.command_vy = 0.4
            args.command_yaw = 0.3
        if args.min_planar_displacement == 0.05:
            args.min_planar_displacement = 0.2
        if args.steps == 256:
            args.steps = 500


def main() -> None:
    _apply_stage_defaults(args_cli)
    register_gym_envs()
    task_spec = resolve_robot_task_spec(args_cli.robot, stage=args_cli.stage)
    env_cfg, agent_cfg = load_env_and_runner_cfg(task_spec.play_task_id, args_cli)
    env_cfg.scene.num_envs = 1

    log_root_path = log_root_for_experiment(agent_cfg.experiment_name)
    resume_path = resolve_checkpoint(log_root_path, agent_cfg)
    print(f"[VALIDATE] checkpoint={resume_path}", flush=True)

    if agent_cfg.seed is not None and hasattr(env_cfg, "seed"):
        env_cfg.seed = agent_cfg.seed

    env = gym.make(task_spec.play_task_id, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)
    env.seed(agent_cfg.seed)

    robot = env.unwrapped.scene["robot"]
    contact_sensor = env.unwrapped.scene["contact_forces"]
    foot_names = tuple(task_spec.primary_foot_links) if hasattr(task_spec, "primary_foot_links") else ()
    if len(foot_names) != 2:
        raise AssertionError(f"Expected a biped with exactly two primary feet, found {foot_names}")
    support_names = tuple(getattr(task_spec, "support_link_names", foot_names))
    left_support_names, right_support_names = _group_support_links(support_names)
    if not left_support_names or not right_support_names:
        raise AssertionError(f"Expected left/right support links, found support={support_names}")

    robot_body_ids, robot_body_names = robot.find_bodies(list(foot_names), preserve_order=True)
    sensor_body_ids, sensor_body_names = contact_sensor.find_bodies(list(foot_names), preserve_order=True)
    left_support_robot_ids, _ = robot.find_bodies(list(left_support_names), preserve_order=True)
    right_support_robot_ids, _ = robot.find_bodies(list(right_support_names), preserve_order=True)
    left_support_sensor_ids, _ = contact_sensor.find_bodies(list(left_support_names), preserve_order=True)
    right_support_sensor_ids, _ = contact_sensor.find_bodies(list(right_support_names), preserve_order=True)
    if tuple(robot_body_names) != foot_names or tuple(sensor_body_names) != foot_names:
        raise AssertionError(
            f"Failed to resolve expected feet in scene. robot={robot_body_names} sensor={sensor_body_names} expected={foot_names}"
        )

    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    obs, _ = env.get_observations()
    semantic_command = (args_cli.command_vx, args_cli.command_vy, args_cli.command_yaw)
    env_command = clamp_base_velocity_command(env_cfg, semantic_command_to_env_command(task_spec.forward_body_axis, semantic_command))
    force_base_velocity_command(env.unwrapped, env_command)

    initial_root_pos = robot.data.root_pos_w[0].detach().cpu()
    initial_root_quat = robot.data.root_quat_w[0].detach().cpu()
    initial_body_x_xy, initial_body_y_xy = _body_axes_xy(initial_root_quat)
    initial_forward_dir_xy = torch.tensor(
        semantic_forward_dir_xy(task_spec.forward_body_axis, initial_root_quat.tolist()),
        dtype=torch.float32,
    )
    initial_left_support_pos = robot.data.body_pos_w[0, left_support_robot_ids].detach().cpu()
    initial_right_support_pos = robot.data.body_pos_w[0, right_support_robot_ids].detach().cpu()
    min_left_support_xy = initial_left_support_pos[:, :2].clone()
    max_left_support_xy = initial_left_support_pos[:, :2].clone()
    min_right_support_xy = initial_right_support_pos[:, :2].clone()
    max_right_support_xy = initial_right_support_pos[:, :2].clone()
    left_support_center_xy = initial_left_support_pos[:, :2].mean(dim=0)
    right_support_center_xy = initial_right_support_pos[:, :2].mean(dim=0)
    lateral_separation = float(torch.linalg.norm(left_support_center_xy - right_support_center_xy))

    max_forward_progress = 0.0
    max_planar_displacement = 0.0
    max_abs_body_axis_progress = torch.zeros(2, dtype=torch.float32)
    min_side_z = torch.tensor(
        [float(initial_left_support_pos[:, 2].min()), float(initial_right_support_pos[:, 2].min())], dtype=torch.float32
    )
    max_side_z = torch.tensor(
        [float(initial_left_support_pos[:, 2].max()), float(initial_right_support_pos[:, 2].max())], dtype=torch.float32
    )
    max_side_planar_travel = torch.zeros(2, dtype=torch.float32)
    sum_root_lin_vel_b = torch.zeros(3, dtype=torch.float32)
    sum_root_ang_vel_w = torch.zeros(3, dtype=torch.float32)
    last_contact = torch.stack(
        (
            _contact_mask(contact_sensor, left_support_sensor_ids, args_cli.contact_threshold)[0].any(),
            _contact_mask(contact_sensor, right_support_sensor_ids, args_cli.contact_threshold)[0].any(),
        )
    ).detach().cpu()
    contact_switches = torch.zeros(2, dtype=torch.int64)
    done_count = 0

    for _ in range(args_cli.steps):
        force_base_velocity_command(env.unwrapped, env_command)
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
        done_count += int(dones[0].item())

        root_pos = robot.data.root_pos_w[0].detach().cpu()
        root_lin_vel_b = robot.data.root_lin_vel_b[0].detach().cpu()
        root_ang_vel_w = robot.data.root_ang_vel_w[0].detach().cpu()
        left_support_pos = robot.data.body_pos_w[0, left_support_robot_ids].detach().cpu()
        right_support_pos = robot.data.body_pos_w[0, right_support_robot_ids].detach().cpu()
        current_contact = torch.stack(
            (
                _contact_mask(contact_sensor, left_support_sensor_ids, args_cli.contact_threshold)[0].any(),
                _contact_mask(contact_sensor, right_support_sensor_ids, args_cli.contact_threshold)[0].any(),
            )
        ).detach().cpu()

        planar_delta = root_pos[:2] - initial_root_pos[:2]
        forward_progress = float(torch.dot(planar_delta, initial_forward_dir_xy))
        max_forward_progress = max(max_forward_progress, forward_progress)
        max_planar_displacement = max(max_planar_displacement, float(torch.linalg.norm(planar_delta)))
        body_axis_progress = torch.tensor(
            (
                abs(float(torch.dot(planar_delta, initial_body_x_xy))),
                abs(float(torch.dot(planar_delta, initial_body_y_xy))),
            ),
            dtype=torch.float32,
        )
        max_abs_body_axis_progress = torch.maximum(max_abs_body_axis_progress, body_axis_progress)
        min_left_support_xy = torch.minimum(min_left_support_xy, left_support_pos[:, :2])
        max_left_support_xy = torch.maximum(max_left_support_xy, left_support_pos[:, :2])
        min_right_support_xy = torch.minimum(min_right_support_xy, right_support_pos[:, :2])
        max_right_support_xy = torch.maximum(max_right_support_xy, right_support_pos[:, :2])
        min_side_z = torch.minimum(
            min_side_z,
            torch.tensor([float(left_support_pos[:, 2].min()), float(right_support_pos[:, 2].min())], dtype=torch.float32),
        )
        max_side_z = torch.maximum(
            max_side_z,
            torch.tensor([float(left_support_pos[:, 2].max()), float(right_support_pos[:, 2].max())], dtype=torch.float32),
        )
        planar_travel = torch.tensor(
            [
                float(torch.linalg.norm(max_left_support_xy - min_left_support_xy, dim=1).max()),
                float(torch.linalg.norm(max_right_support_xy - min_right_support_xy, dim=1).max()),
            ],
            dtype=torch.float32,
        )
        max_side_planar_travel = torch.maximum(max_side_planar_travel, planar_travel)
        contact_switches += (current_contact != last_contact).to(dtype=torch.int64)
        last_contact = current_contact
        sum_root_lin_vel_b += root_lin_vel_b
        sum_root_ang_vel_w += root_ang_vel_w

    forward_displacement = max_forward_progress
    side_height_range = max_side_z - min_side_z
    final_planar_delta = robot.data.root_pos_w[0].detach().cpu()[:2] - initial_root_pos[:2]
    final_body_axis_displacement = torch.tensor(
        (
            float(torch.dot(final_planar_delta, initial_body_x_xy)),
            float(torch.dot(final_planar_delta, initial_body_y_xy)),
        ),
        dtype=torch.float32,
    )
    mean_root_lin_vel_b = sum_root_lin_vel_b / float(args_cli.steps)
    mean_root_ang_vel_w = sum_root_ang_vel_w / float(args_cli.steps)
    mean_command_error_b = torch.tensor(env_command, dtype=torch.float32) - torch.tensor(
        (
            float(mean_root_lin_vel_b[0]),
            float(mean_root_lin_vel_b[1]),
            float(mean_root_ang_vel_w[2]),
        ),
        dtype=torch.float32,
    )

    print(f"[VALIDATE] feet={foot_names}", flush=True)
    print(f"[VALIDATE] support_left={left_support_names}", flush=True)
    print(f"[VALIDATE] support_right={right_support_names}", flush=True)
    print(f"[VALIDATE] forward_body_axis={task_spec.forward_body_axis}", flush=True)
    print(f"[VALIDATE] semantic_command={semantic_command}", flush=True)
    print(f"[VALIDATE] env_command={env_command}", flush=True)
    print(f"[VALIDATE] forward_displacement={forward_displacement:.4f}", flush=True)
    print(f"[VALIDATE] planar_displacement={max_planar_displacement:.4f}", flush=True)
    print(f"[VALIDATE] raw_root_x_displacement={final_body_axis_displacement[0]:.4f}", flush=True)
    print(f"[VALIDATE] raw_root_y_displacement={final_body_axis_displacement[1]:.4f}", flush=True)
    print(f"[VALIDATE] body_axis_displacement_abs={max_abs_body_axis_progress.tolist()}", flush=True)
    print(f"[VALIDATE] final_body_axis_displacement={final_body_axis_displacement.tolist()}", flush=True)
    print(f"[VALIDATE] lateral_separation={lateral_separation:.4f}", flush=True)
    print(f"[VALIDATE] side_planar_travel={max_side_planar_travel.tolist()}", flush=True)
    print(f"[VALIDATE] side_height_range={side_height_range.tolist()}", flush=True)
    print(f"[VALIDATE] contact_switches={contact_switches.tolist()}", flush=True)
    print(f"[VALIDATE] mean_root_lin_vel_b={mean_root_lin_vel_b.tolist()}", flush=True)
    print(f"[VALIDATE] mean_root_ang_vel_w={mean_root_ang_vel_w.tolist()}", flush=True)
    print(f"[VALIDATE] mean_command_error={mean_command_error_b.tolist()}", flush=True)
    print(f"[VALIDATE] done_count={done_count}", flush=True)

    _validate_metric(lateral_separation >= args_cli.min_lateral_separation, "two-leg lateral separation", lateral_separation, args_cli.min_lateral_separation)
    _validate_metric(max_planar_displacement >= args_cli.min_planar_displacement, "planar displacement", max_planar_displacement, args_cli.min_planar_displacement)
    if args_cli.min_forward_displacement is not None:
        _validate_metric(forward_displacement >= args_cli.min_forward_displacement, "forward displacement", forward_displacement, args_cli.min_forward_displacement)
    _validate_metric(done_count <= args_cli.max_done_count, "done count", done_count, args_cli.max_done_count)
    for index, side_name in enumerate(("left_leg", "right_leg")):
        _validate_metric(
            float(max_side_planar_travel[index]) >= args_cli.min_foot_planar_travel,
            f"{side_name} planar travel",
            float(max_side_planar_travel[index]),
            args_cli.min_foot_planar_travel,
        )
        _validate_metric(
            float(side_height_range[index]) >= args_cli.min_foot_height_range,
            f"{side_name} height range",
            float(side_height_range[index]),
            args_cli.min_foot_height_range,
        )
        _validate_metric(
            int(contact_switches[index]) >= args_cli.min_contact_switches,
            f"{side_name} contact switches",
            int(contact_switches[index]),
            args_cli.min_contact_switches,
        )

    # Stage-specific acceptance gates
    if args_cli.stage == "fwd_only":
        # commanded axis should show at least 0.2 m/s mean speed
        fwd_axis_idx = 1 if task_spec.forward_body_axis == "y" else 0
        mean_fwd_speed = abs(float(mean_root_lin_vel_b[fwd_axis_idx]))
        _validate_metric(mean_fwd_speed >= 0.15, "fwd_only mean forward speed", f"{mean_fwd_speed:.3f}", 0.15)
        # orthogonal axis should stay small
        ortho_idx = 0 if fwd_axis_idx == 1 else 1
        mean_ortho_speed = abs(float(mean_root_lin_vel_b[ortho_idx]))
        _validate_metric(mean_ortho_speed < 0.1, "fwd_only orthogonal drift", f"{mean_ortho_speed:.3f}", "< 0.1")
    elif args_cli.stage == "fwd_yaw":
        fwd_axis_idx = 1 if task_spec.forward_body_axis == "y" else 0
        mean_fwd_speed = abs(float(mean_root_lin_vel_b[fwd_axis_idx]))
        _validate_metric(mean_fwd_speed >= 0.1, "fwd_yaw mean forward speed", f"{mean_fwd_speed:.3f}", 0.1)
        mean_yaw_rate = abs(float(mean_root_ang_vel_w[2]))
        if abs(args_cli.command_yaw) > 0.1:
            _validate_metric(mean_yaw_rate >= 0.1, "fwd_yaw mean yaw rate", f"{mean_yaw_rate:.3f}", 0.1)

    print("[VALIDATE] walk validation passed", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
