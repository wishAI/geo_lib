"""Train or evaluate the clean-lineage Landau policy-standing controller."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

REPO_BOOTSTRAP_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_BOOTSTRAP_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_BOOTSTRAP_ROOT))

from algorithms.urdf_learn_wasd_walk import model_spec, passive_stand
from algorithms.urdf_learn_wasd_walk import policy_stand_contract as contract


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _versions() -> dict:
    try:
        from isaacsim.core.version import get_version

        isaac_sim = ".".join(str(item) for item in get_version())
    except Exception as error:  # pragma: no cover - Isaac-only
        isaac_sim = f"unavailable:{type(error).__name__}"
    return {
        "isaac_sim": isaac_sim,
        "isaaclab": _package_version("isaaclab"),
        "rsl_rl": _package_version("rsl-rl-lib"),
        "torch": _package_version("torch"),
        "numpy": _package_version("numpy"),
        "python": platform.python_version(),
    }


def _source_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_BOOTSTRAP_ROOT, text=True,
        capture_output=True, check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "unavailable"


def _runner_cfg(seed: int, iterations: int):
    from isaaclab_rl.rsl_rl import (
        RslRlOnPolicyRunnerCfg,
        RslRlPpoActorCriticCfg,
        RslRlPpoAlgorithmCfg,
    )

    return RslRlOnPolicyRunnerCfg(
        seed=seed,
        device="cuda:0",
        num_steps_per_env=24,
        max_iterations=iterations,
        save_interval=50,
        experiment_name="landau_policy_stand",
        empirical_normalization=False,
        logger="tensorboard",
        policy=RslRlPpoActorCriticCfg(
            init_noise_std=0.2,
            actor_hidden_dims=[128, 128],
            critic_hidden_dims=[128, 128],
            activation="elu",
        ),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.005,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1.0e-3,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),
    )


def _write_failure(args, error: Exception, traceback_text: str) -> None:
    output_dir = contract.safe_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / contract.failure_artifact_name(args.mode, args.smoke)
    trace_path = output_dir / f"{args.mode}{'_smoke' if args.smoke else ''}_traceback.log"
    trace_path.write_text(traceback_text, encoding="utf-8")
    payload = {
        "schema_version": 1,
        "milestone": contract.MILESTONE_ID,
        "status": "failed_to_execute",
        "mode": args.mode,
        "runtime_stage": getattr(args, "runtime_stage", "unknown"),
        "run_identity": _timestamp(),
        "exception": {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback_text,
        },
        "traceback_path": str(trace_path.relative_to(REPO_BOOTSTRAP_ROOT)),
        "traceback_sha256": contract.sha256(trace_path),
        "input": {
            "urdf_sha256": model_spec.EXPECTED_URDF_SHA256,
            "robot_spec_sha256": contract.sha256(model_spec.ROBOT_SPEC_PATH),
        },
        "argv": list(sys.argv),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _stage(args, name: str) -> None:
    args.runtime_stage = name
    print(f"[landau-policy-stand] stage={name}", file=sys.stderr, flush=True)


def _train(args) -> dict:
    import torch
    from rsl_rl.runners import OnPolicyRunner
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from algorithms.urdf_learn_wasd_walk.policy_stand_env import build_env_cfg

    output_dir = contract.safe_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    training_path = output_dir / contract.TRAINING_EVIDENCE
    training_path.unlink(missing_ok=True)
    prior = contract.load_prior_gate()
    run_identity = _timestamp()
    run_dir = output_dir / "runs" / run_identity
    run_dir.mkdir(parents=True, exist_ok=False)
    requested = contract.training_contract(
        seed=args.seed, num_envs=args.num_envs, iterations=args.iterations
    )
    _stage(args, "manager_environment_construction")
    cfg = build_env_cfg(
        num_envs=args.num_envs,
        seed=args.seed,
        training=True,
        force_usd_conversion=not args.reuse_usd_cache,
    )
    cfg.sim.device = args.device
    env = ManagerBasedRLEnv(cfg=cfg, render_mode=None)
    wrapped = None
    try:
        _stage(args, "runtime_contract_audit")
        action_term = env.action_manager.get_term("joint_pos")
        runtime_action_names = list(action_term._joint_names)
        if runtime_action_names != list(model_spec.ACTION_JOINTS):
            raise RuntimeError(f"runtime action order differs from contract: {runtime_action_names}")
        if env.observation_manager.group_obs_dim["policy"] != (contract.ACTOR_OBSERVATION_DIM,):
            raise RuntimeError(
                f"runtime actor observation dimension differs: {env.observation_manager.group_obs_dim['policy']}"
            )
        if set(env.scene["robot"].joint_names) != set(model_spec.build_robot_spec()["nominal_pose"]["joint_positions_rad"]):
            raise RuntimeError("runtime articulation joint set differs from the audited URDF")
        wrapped = RslRlVecEnvWrapper(env)
        agent_cfg = _runner_cfg(args.seed, args.iterations)
        runner = OnPolicyRunner(
            wrapped, agent_cfg.to_dict(), log_dir=os.fspath(run_dir), device=args.device
        )
        _stage(args, "ppo_learning")
        runner.learn(num_learning_iterations=args.iterations, init_at_random_ep_len=True)
        checkpoints = list(run_dir.glob("model_*.pt"))
        if not checkpoints:
            raise RuntimeError("RSL-RL completed without a checkpoint")
        run_checkpoint = max(checkpoints, key=lambda path: int(path.stem.split("_")[-1]))
        checkpoint = output_dir / "checkpoint.pt"
        checkpoint.unlink(missing_ok=True)
        shutil.copy2(run_checkpoint, checkpoint)
        payload = {
            "schema_version": 1,
            "milestone": contract.MILESTONE_ID,
            "status": "completed_not_promoted",
            "lineage": contract.LINEAGE,
            "run_identity": run_identity,
            "completed_at": _timestamp(),
            "source_commit": _source_commit(),
            "versions": _versions(),
            "device": args.device,
            "single_process": True,
            "requested_contract": requested,
            "runtime_contract": {
                "environment_class": type(env).__name__,
                "action_order": runtime_action_names,
                "action_dimension": env.action_manager.total_action_dim,
                "actor_observation_dimension": env.observation_manager.group_obs_dim["policy"][0],
                "manager_reward_terms": list(env.reward_manager.active_terms),
                "manager_termination_terms": list(env.termination_manager.active_terms),
            },
            "input": model_spec.build_robot_spec()["source"],
            "robot_spec_sha256": contract.sha256(model_spec.ROBOT_SPEC_PATH),
            "checkpoint": {
                "kind": "rsl_rl_ppo",
                "path": str(checkpoint.relative_to(REPO_BOOTSTRAP_ROOT)),
                "sha256": contract.sha256(checkpoint),
                "size_bytes": checkpoint.stat().st_size,
                "learning_iteration": int(run_checkpoint.stem.split("_")[-1]),
                "run_checkpoint_path": str(run_checkpoint.relative_to(REPO_BOOTSTRAP_ROOT)),
            },
            "promotion": {
                "eligible": False,
                "reason": "training completion never passes a milestone; exact dynamics and proof are required",
            },
            "cumulative_gates": [
                prior,
                {"order": 2, "id": contract.MILESTONE_ID, "status": "awaiting_validation"},
            ],
        }
        training_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({
            "status": payload["status"], "training": os.fspath(training_path),
            "checkpoint": payload["checkpoint"],
        }, indent=2))
        return payload
    finally:
        if wrapped is not None:
            wrapped.close()
        else:
            env.close()


def _preflight_evaluation(args) -> tuple[dict, dict]:
    prior = contract.load_prior_gate()
    training = contract.load_training_evidence(args.output_dir)
    if args.mode == "proof":
        dynamics_path = args.output_dir / contract.component_artifact_name("dynamics", args.smoke)
        if not dynamics_path.is_file():
            raise ValueError("proof is blocked until matching camera-free policy dynamics passes")
        dynamics = json.loads(dynamics_path.read_text(encoding="utf-8"))
        if dynamics.get("status") != "passed" or dynamics.get("failures"):
            raise ValueError("proof is blocked because policy dynamics did not pass")
        if dynamics.get("checkpoint", {}).get("sha256") != training["checkpoint"]["sha256"]:
            raise ValueError("proof checkpoint differs from the passed policy dynamics checkpoint")
    return prior, training


def _inspect_policy_frame(path: Path, bbox: dict, index: int, time_s: float) -> dict:
    import numpy as np
    from PIL import Image, ImageDraw

    frame = Image.open(path).convert("RGB")
    pixels = np.asarray(frame)
    draw = ImageDraw.Draw(frame)
    draw.rectangle((7, 7, 202, 33), fill=(245, 245, 245), outline=(30, 30, 30))
    draw.text((13, 13), f"Landau policy · t={time_s:05.1f}s", fill=(10, 10, 10))
    frame.save(path)
    return {
        "frame_index": index,
        "time_s": round(time_s, 6),
        "nonblank": bool(float(pixels.std()) > 2.0),
        "rgb_stddev": round(float(pixels.std()), 5),
        "resolution": [frame.width, frame.height],
        **bbox,
    }


def _run_evaluation(args, prior: dict, training: dict) -> dict:
    import torch
    from rsl_rl.runners import OnPolicyRunner
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from algorithms.urdf_learn_wasd_walk.policy_stand_env import build_env_cfg

    output_dir = contract.safe_output_dir(args.output_dir)
    phase = args.mode
    evidence_path = output_dir / contract.component_artifact_name(phase, args.smoke)
    evidence_path.unlink(missing_ok=True)
    _stage(args, "manager_environment_construction")
    cfg = build_env_cfg(
        num_envs=1,
        seed=args.seed,
        training=False,
        force_usd_conversion=phase == "dynamics" and not args.reuse_usd_cache,
    )
    cfg.sim.device = args.device
    env = ManagerBasedRLEnv(cfg=cfg, render_mode=None)
    wrapped = None
    frame_context = None
    try:
        robot = env.scene["robot"]
        contacts = env.scene["contact_forces"]
        action_term = env.action_manager.get_term("joint_pos")
        runtime_action_names = list(action_term._joint_names)
        if runtime_action_names != list(model_spec.ACTION_JOINTS):
            raise RuntimeError(f"runtime action order differs from contract: {runtime_action_names}")
        if env.observation_manager.group_obs_dim["policy"] != (contract.ACTOR_OBSERVATION_DIM,):
            raise RuntimeError("runtime actor observation dimension differs from the deployable contract")
        spec = model_spec.build_robot_spec()
        if set(robot.joint_names) != set(spec["nominal_pose"]["joint_positions_rad"]):
            raise RuntimeError("runtime articulation joint set differs from the audited URDF")
        reference_ids, reference_names = robot.find_bodies("root_x", preserve_order=True)
        if reference_names != ["root_x"]:
            raise RuntimeError(f"expected one root_x reference, got {reference_names}")
        support_ids, support_names = contacts.find_bodies(
            ["foot_l", "foot_r", "toes_01_l", "toes_01_r"], preserve_order=True
        )
        if len(support_ids) != 4:
            raise RuntimeError(f"expected four support bodies, got {support_names}")

        imported_axes = None
        if phase == "dynamics":
            _stage(args, "importer_axis_audit")
            imported_axes = passive_stand._inspect_imported_joint_contract(
                env.sim.stage, "/World/envs/env_0/Robot", spec
            )
            if not imported_axes["passed"]:
                raise RuntimeError("imported joint axes differ from the current URDF contract")

        wrapped = RslRlVecEnvWrapper(env)
        agent_cfg = _runner_cfg(args.seed, training["requested_contract"]["iterations"])
        runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=args.device)
        checkpoint_path = Path(training["checkpoint"]["resolved_path"])
        _stage(args, "checkpoint_load")
        runner.load(os.fspath(checkpoint_path))
        policy = runner.get_inference_policy(device=env.device)
        obs, _ = wrapped.get_observations()

        viewport = None
        frame_dir = None
        visual_local_corners = None
        frames: list[dict] = []
        if phase == "proof":
            from omni.kit.viewport.utility import get_active_viewport

            viewport = get_active_viewport()
            if viewport is None:
                raise RuntimeError("proof phase has no active viewport")
            viewport.updates_enabled = True
            viewport.resolution = (args.video_width, args.video_height)
            visual_local_corners = model_spec.visual_local_aabb_corners()
            if not visual_local_corners:
                raise RuntimeError("exact URDF supplied no visual-mesh bounds")
            env.sim.set_camera_view(
                eye=contract.PROOF_CAMERA_EYE_OFFSET_M,
                target=(0.0, 0.0, contract.PROOF_CAMERA_TARGET_HEIGHT_M),
            )
            for _ in range(12):
                env.sim.render()
            if tuple(map(int, viewport.resolution)) != (args.video_width, args.video_height):
                raise RuntimeError(f"viewport resolution did not settle: {viewport.resolution}")
            frame_context = tempfile.TemporaryDirectory(
                prefix="landau_policy_frames_", dir=output_dir
            )
            frame_dir = Path(frame_context.name)

        control_steps = args.steps if args.steps is not None else math.ceil(args.duration / contract.CONTROL_DT_S)
        video_stride = max(1, round((1.0 / args.video_fps) / contract.CONTROL_DT_S))
        body_masses = robot.data.default_mass[0].to(device=robot.device, dtype=robot.data.joint_pos.dtype)
        total_mass = torch.sum(body_masses)
        robot_weight_n = float(total_mass.detach().cpu()) * 9.81
        support_hull = spec["nominal_pose"]["geometry"]["support_hull_xy_m"]
        initial_position = robot.data.body_pos_w[0, reference_ids[0]].detach().cpu().clone()
        initial_quaternion = robot.data.body_quat_w[0, reference_ids[0]].detach().cpu().clone()
        traces: list[dict] = []
        action_trace: list[dict] = []
        done_count = fall_count = reset_count = inference_steps = clipped_values = 0
        max_abs_action = max_abs_actor_output = max_tilt = max_height_drop = 0.0
        max_joint_target_error = 0.0
        minimum_support_margin = math.inf
        first_support_exit_time = None
        peak_support_force = support_force_sum = 0.0
        _stage(args, "policy_inference_loop")
        for step in range(control_steps):
            with torch.inference_mode():
                actor_output = policy(obs)
                max_abs_actor_output = max(
                    max_abs_actor_output, float(torch.max(torch.abs(actor_output)).detach().cpu())
                )
                clipped_values += int(torch.count_nonzero(torch.abs(actor_output) > contract.ACTION_CLIP).detach().cpu())
                actions = torch.clamp(actor_output, -contract.ACTION_CLIP, contract.ACTION_CLIP)
                inference_steps += 1
                obs, _, dones, _ = wrapped.step(actions)
            raw_action = actions[0].detach().cpu().tolist()
            max_abs_action = max(max_abs_action, max(abs(float(value)) for value in raw_action))
            terminated = int(torch.count_nonzero(env.reset_terminated).detach().cpu())
            done = int(torch.count_nonzero(dones).detach().cpu())
            fall_count += terminated
            done_count += done
            reset_count += done
            time_s = (step + 1) * contract.CONTROL_DT_S
            position = robot.data.body_pos_w[0, reference_ids[0]].detach().cpu()
            quaternion = robot.data.body_quat_w[0, reference_ids[0]].detach().cpu()
            tilt = passive_stand.quaternion_distance_wxyz(
                quaternion.tolist(), initial_quaternion.tolist()
            )
            height_drop = float(initial_position[2] - position[2])
            max_tilt = max(max_tilt, tilt)
            max_height_drop = max(max_height_drop, height_drop)
            system_com = (
                torch.sum(robot.data.body_com_pos_w[0] * body_masses.unsqueeze(-1), dim=0) / total_mass
            ).detach().cpu().tolist()
            support_margin = model_spec.support_polygon_margin(system_com[:2], support_hull)
            minimum_support_margin = min(minimum_support_margin, support_margin)
            if support_margin < 0.0 and first_support_exit_time is None:
                first_support_exit_time = time_s
            force_norms = torch.linalg.vector_norm(
                contacts.data.net_forces_w[0, support_ids].detach().cpu(), dim=-1
            ).tolist()
            support_force = sum(float(value) for value in force_norms)
            peak_support_force = max(peak_support_force, support_force)
            support_force_sum += support_force
            joint_error = torch.abs(robot.data.joint_pos_target[0] - robot.data.joint_pos[0])
            max_joint_target_error = max(
                max_joint_target_error, float(torch.max(joint_error).detach().cpu())
            )
            trace = {
                "time_s": round(time_s, 6),
                "root_x_position_m": [round(float(v), 8) for v in position.tolist()],
                "root_x_orientation_wxyz": [round(float(v), 9) for v in quaternion.tolist()],
                "root_x_linear_velocity_mps": [
                    round(float(v), 8)
                    for v in robot.data.body_lin_vel_w[0, reference_ids[0]].detach().cpu().tolist()
                ],
                "root_x_angular_velocity_radps": [
                    round(float(v), 8)
                    for v in robot.data.body_ang_vel_w[0, reference_ids[0]].detach().cpu().tolist()
                ],
                "reference_tilt_rad": round(tilt, 8),
                "system_center_of_mass_m": [round(float(v), 8) for v in system_com],
                "support_polygon_margin_m": round(support_margin, 8),
                "foot_contact_force_n": {
                    name: round(float(value), 6) for name, value in zip(support_names, force_norms)
                },
                "joint_position_rad": [
                    round(float(v), 8) for v in robot.data.joint_pos[0].detach().cpu().tolist()
                ],
                "joint_velocity_radps": [
                    round(float(v), 8) for v in robot.data.joint_vel[0].detach().cpu().tolist()
                ],
                "terminated": bool(terminated),
                "done": bool(done),
            }
            traces.append(trace)
            action_trace.append({
                "time_s": round(time_s, 6),
                "joint_order": runtime_action_names,
                "raw_policy_action": [round(float(v), 8) for v in raw_action],
            })
            if viewport is not None and (step + 1) % video_stride == 0:
                root_xy = position[:2].tolist()
                eye_offset = contract.PROOF_CAMERA_EYE_OFFSET_M
                env.sim.set_camera_view(
                    eye=(
                        eye_offset[0] + root_xy[0],
                        eye_offset[1] + root_xy[1],
                        eye_offset[2],
                    ),
                    target=(
                        root_xy[0], root_xy[1], contract.PROOF_CAMERA_TARGET_HEIGHT_M
                    ),
                )
                env.sim.render()
                index = len(frames)
                frame_path = frame_dir / f"frame_{index:04d}.png"  # type: ignore[operator]
                passive_stand._capture_viewport_frame(viewport, env.sim, frame_path)
                origin_bbox = passive_stand._projected_link_bbox(
                    viewport, robot.data.body_pos_w[0].detach().cpu().tolist(),
                    args.video_width, args.video_height,
                )
                visual_corners = passive_stand._visual_geometry_world_corners(
                    robot, visual_local_corners
                )
                bbox = {
                    key: value for key, value in origin_bbox.items() if key != "character_visible"
                }
                bbox["link_origin_scale_proxy_passed"] = origin_bbox["character_visible"]
                bbox.update(passive_stand._projected_visual_geometry_bbox(
                    viewport, visual_corners, args.video_width, args.video_height
                ))
                frames.append(_inspect_policy_frame(frame_path, bbox, index, time_s))

        final_position = robot.data.body_pos_w[0, reference_ids[0]].detach().cpu().tolist()
        metrics = {
            "duration_s": round(control_steps * contract.CONTROL_DT_S, 6),
            "physics_steps": control_steps * round(contract.CONTROL_DT_S / contract.PHYSICS_DT_S),
            "control_steps": control_steps,
            "policy_inference_steps": inference_steps,
            "reset_count": reset_count,
            "done_count": done_count,
            "fall_count": fall_count,
            "max_abs_command": 0.0,
            "max_abs_action": round(max_abs_action, 8),
            "max_abs_actor_output_before_clip": round(max_abs_actor_output, 8),
            "actor_output_clipped_value_count": clipped_values,
            "max_reference_tilt_rad": round(max_tilt, 8),
            "root_height_drop_m": round(max(0.0, max_height_drop), 8),
            "horizontal_drift_m": round(math.hypot(
                final_position[0] - float(initial_position[0]),
                final_position[1] - float(initial_position[1]),
            ), 8),
            "semantic_forward_displacement_m": round(
                final_position[1] - float(initial_position[1]), 8
            ),
            "semantic_strafe_displacement_m": round(
                final_position[0] - float(initial_position[0]), 8
            ),
            "max_joint_target_error_rad": round(max_joint_target_error, 8),
            "minimum_support_polygon_margin_m": round(minimum_support_margin, 8),
            "first_support_exit_time_s": (
                round(first_support_exit_time, 6) if first_support_exit_time is not None else None
            ),
            "peak_support_force_body_weight_ratio": round(peak_support_force / robot_weight_n, 8),
            "mean_support_force_body_weight_ratio": round(
                support_force_sum / control_steps / robot_weight_n, 8
            ),
        }
        required_duration = 0.0 if args.smoke else contract.MIN_GATE_DURATION_S
        failures = contract.evaluate_policy_gate(metrics, required_duration_s=required_duration)
        failures.extend(passive_stand.evaluate_free_root_support(metrics))

        video = None
        if phase == "proof":
            if not frames:
                failures.append("viewport produced no policy proof frames")
            else:
                from PIL import Image, ImageChops, ImageDraw, ImageStat

                video_path = output_dir / ("proof_smoke.mp4" if args.smoke else "proof.mp4")
                sheet_path = output_dir / (
                    "contact_sheet_smoke.png" if args.smoke else "contact_sheet.png"
                )
                passive_stand._encode_video(frame_dir, video_path, args.video_fps)  # type: ignore[arg-type]
                indices = sorted({0, len(frames) // 2, len(frames) - 1})
                images = [
                    Image.open(frame_dir / f"frame_{index:04d}.png").convert("RGB")  # type: ignore[operator]
                    for index in indices
                ]
                sheet = Image.new("RGB", (args.video_width * len(images), args.video_height + 34), "white")
                draw = ImageDraw.Draw(sheet)
                for column, (index, image) in enumerate(zip(indices, images)):
                    sheet.paste(image, (column * args.video_width, 34))
                    draw.text(
                        (column * args.video_width + 8, 9),
                        f"frame {index} · sim {frames[index]['time_s']:.1f}s",
                        fill="black",
                    )
                sheet.save(sheet_path)
                differences = [
                    round(sum(ImageStat.Stat(ImageChops.difference(a, b)).mean) / 3, 6)
                    for a, b in zip(images, images[1:])
                ]
                probe = passive_stand._probe_video(video_path)
                sampled = [frames[index] for index in indices]
                video = {
                    "capture_path": "active_viewport_LdrColor_AOV",
                    "synthetic_data_camera_graph_used": False,
                    "path": str(video_path.relative_to(REPO_BOOTSTRAP_ROOT)),
                    "sha256": contract.sha256(video_path),
                    "contact_sheet_path": str(sheet_path.relative_to(REPO_BOOTSTRAP_ROOT)),
                    "contact_sheet_sha256": contract.sha256(sheet_path),
                    "frame_count": len(frames),
                    "fps": args.video_fps,
                    "duration_s": float(probe["format"]["duration"]),
                    "expected_duration_s": len(frames) / args.video_fps,
                    "resolution": [args.video_width, args.video_height],
                    "representative_frames": sampled,
                    "representative_mean_absolute_differences": differences,
                    "nonblank_frames_passed": all(item["nonblank"] for item in sampled),
                    "character_visibility_passed": all(item["character_visible"] for item in sampled),
                    "visual_geometry_framing_passed": all(
                        item["visual_geometry_framing_passed"] for item in sampled
                    ),
                    "discernible_scale_passed": all(
                        item["discernible_scale_passed"] for item in sampled
                    ),
                    "temporal_progression_visible": (
                        len(sampled) >= 3 and all(value > 0.005 for value in differences)
                    ),
                }
                proof_passed, proof_failures = passive_stand.evaluate_proof(
                    video, required_duration_s=required_duration
                )
                if not proof_passed:
                    failures.extend(proof_failures)

        evidence = {
            "schema_version": 1,
            "milestone": contract.MILESTONE_ID,
            "component": phase,
            "scope": "component_only",
            "status": "passed" if not failures else "failed",
            "gate_eligible": not args.smoke,
            "milestone_status_changed": False,
            "lineage": contract.LINEAGE,
            "run_identity": _timestamp(),
            "seed": args.seed,
            "checkpoint": {
                key: value for key, value in training["checkpoint"].items() if key != "resolved_path"
            },
            "training_evidence": {
                "path": str((output_dir / contract.TRAINING_EVIDENCE).relative_to(REPO_BOOTSTRAP_ROOT)),
                "sha256": contract.sha256(output_dir / contract.TRAINING_EVIDENCE),
                "run_identity": training["run_identity"],
            },
            "source_commit": training["source_commit"],
            "versions": _versions(),
            "simulator": {
                "device": args.device,
                "physics_dt_s": contract.PHYSICS_DT_S,
                "control_dt_s": contract.CONTROL_DT_S,
                "single_process": True,
                "num_envs": 1,
                "rendering_enabled": phase == "proof",
                "camera_sensor_created": False,
                "tracking_camera": {
                    "eye_offset_m": list(contract.PROOF_CAMERA_EYE_OFFSET_M),
                    "target_height_m": contract.PROOF_CAMERA_TARGET_HEIGHT_M,
                    "visual_bounds": "runtime body transforms applied to exact URDF visual-mesh local AABBs",
                },
            },
            "input": spec["source"],
            "joint_contract": {
                "action_joints": spec["action_joints"],
                "locked_joints": spec["locked_joints"],
                "runtime_action_order": runtime_action_names,
                "nominal_pose": spec["nominal_pose"],
                "pd": spec["pd"],
                "importer_axis_contract": spec["importer_axis_contract"],
                "runtime_importer_axis_evidence": imported_axes,
            },
            "policy_contract": training["requested_contract"],
            "command_contract": spec["frames"],
            "metrics": metrics,
            "traces": traces,
            "action_traces": action_trace,
            "video_inspection": video,
            "failures": failures,
            "cumulative_gates": [
                prior,
                {"order": 2, "id": contract.MILESTONE_ID, "status": "pending_final_assembly"},
            ],
        }
        evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({
            "status": evidence["status"], "component": phase, "metrics": metrics,
            "evidence": os.fspath(evidence_path),
            "proof_video": video["path"] if video else None, "failures": failures,
        }, indent=2))
        return evidence
    finally:
        if frame_context is not None:
            frame_context.cleanup()
        if wrapped is not None:
            wrapped.close()
        else:
            env.close()


def _parser() -> argparse.ArgumentParser:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("train", "dynamics", "proof"), required=True)
    parser.add_argument("--output-dir", type=Path, default=contract.DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-envs", type=int, default=512)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--duration", type=float, default=contract.MIN_GATE_DURATION_S)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--reuse-usd-cache", action="store_true")
    parser.add_argument("--video-fps", type=int, default=5)
    parser.add_argument("--video-width", type=int, default=640)
    parser.add_argument("--video-height", type=int, default=480)
    AppLauncher.add_app_launcher_args(parser)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    from isaaclab.app import AppLauncher

    args = _parser().parse_args(argv)
    args.output_dir = contract.safe_output_dir(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.seed < 0 or args.num_envs <= 0 or args.iterations <= 0:
        raise SystemExit("seed must be non-negative; num-envs and iterations must be positive")
    if args.mode != "train":
        if args.duration <= 0 or (args.steps is not None and args.steps <= 0):
            raise SystemExit("duration and steps must be positive")
        evaluated_duration = (
            args.steps * contract.CONTROL_DT_S if args.steps is not None else args.duration
        )
        if not args.smoke and evaluated_duration + 1.0e-6 < contract.MIN_GATE_DURATION_S:
            raise SystemExit("short policy evaluation requires --smoke")
    if args.mode == "train" and args.smoke:
        raise SystemExit("training has no smoke identity; bound it with --iterations and --num-envs")
    if args.reuse_usd_cache and not (args.mode == "train" or args.smoke or args.mode == "proof"):
        raise SystemExit("exact camera-free dynamics must reconvert the current URDF")

    prior = training = None
    try:
        if args.mode != "train":
            prior, training = _preflight_evaluation(args)
    except ValueError as error:
        print(f"Policy evaluation preflight failed: {error}", file=sys.stderr)
        return 1

    os.environ["ENABLE_CAMERAS"] = "1" if args.mode == "proof" else "0"
    args.enable_cameras = args.mode == "proof"
    args.video = args.mode == "proof"
    launcher = None
    try:
        _stage(args, "app_launcher")
        launcher = AppLauncher(args)
        import torch

        torch.manual_seed(args.seed)
        if args.mode == "train":
            evidence = _train(args)
        else:
            evidence = _run_evaluation(args, prior, training)
        return 0 if evidence["status"] in {"completed_not_promoted", "passed"} else 1
    except Exception as error:
        traceback_text = traceback.format_exc()
        print(traceback_text, file=sys.stderr, flush=True)
        try:
            _write_failure(args, error, traceback_text)
        except Exception:
            traceback.print_exc(file=sys.stderr)
        return 1
    finally:
        if launcher is not None:
            launcher.app.close()


if __name__ == "__main__":
    raise SystemExit(main())
