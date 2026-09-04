"""Train and evaluate the clean-lineage Landau 5 m forward policy."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from algorithms.urdf_learn_wasd_walk import forward_walk_contract as contract
from algorithms.urdf_learn_wasd_walk import model_spec, passive_stand, policy_stand


def _stage(args, name: str) -> None:
    args.runtime_stage = name
    print(f"[landau-forward] stage={name}", file=sys.stderr, flush=True)


def _write_failure(args, error: Exception) -> None:
    output = contract.safe_output_dir(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    traceback_text = traceback.format_exc()
    trace_path = output / f"{args.mode}{'_smoke' if args.smoke else ''}_traceback.log"
    trace_path.write_text(traceback_text, encoding="utf-8")
    payload = {
        "schema_version": 1,
        "milestone": contract.MILESTONE_ID,
        "status": "failed_to_execute",
        "mode": args.mode,
        "runtime_stage": getattr(args, "runtime_stage", "unknown"),
        "exception": {"type": type(error).__name__, "message": str(error)},
        "traceback_path": str(trace_path.relative_to(REPO_ROOT)),
        "traceback_sha256": contract.sha256(trace_path),
        "argv": list(sys.argv),
    }
    (output / f"{args.mode}{'_smoke' if args.smoke else ''}_failure.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _transfer_stand_checkpoint(runner, checkpoint_path: Path) -> dict:
    """Preserve the 60-input stand network and zero-init three command columns."""

    import torch

    loaded = torch.load(checkpoint_path, map_location=runner.device, weights_only=False)
    source = loaded["model_state_dict"]
    target = {name: value.clone() for name, value in runner.alg.actor_critic.state_dict().items()}
    expanded = []
    exact = []
    for name, target_value in target.items():
        if name not in source:
            raise RuntimeError(f"stand checkpoint lacks policy tensor {name}")
        source_value = source[name]
        if source_value.shape == target_value.shape:
            target[name] = source_value
            exact.append(name)
        elif (
            name in {"actor.0.weight", "critic.0.weight"}
            and source_value.ndim == 2
            and source_value.shape[0] == target_value.shape[0]
            and source_value.shape[1] == 60
            and target_value.shape[1] == contract.ACTOR_OBSERVATION_DIM
        ):
            target_value.zero_()
            target_value[:, :60] = source_value
            target[name] = target_value
            expanded.append(name)
        else:
            raise RuntimeError(
                f"stand transfer shape mismatch for {name}: {tuple(source_value.shape)} -> "
                f"{tuple(target_value.shape)}"
            )
    runner.alg.actor_critic.load_state_dict(target, strict=True)
    if sorted(expanded) != ["actor.0.weight", "critic.0.weight"]:
        raise RuntimeError(f"unexpected expanded transfer tensors: {expanded}")
    return {
        "parent_checkpoint_path": str(checkpoint_path.relative_to(REPO_ROOT)),
        "parent_checkpoint_sha256": contract.sha256(checkpoint_path),
        "exact_tensor_count": len(exact),
        "expanded_tensors": sorted(expanded),
        "preserved_input_columns": [0, 59],
        "zero_initialized_command_columns": [60, 61, 62],
        "optimizer_state_loaded": False,
    }


def _train(args) -> dict:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from rsl_rl.runners import OnPolicyRunner

    from algorithms.urdf_learn_wasd_walk.forward_walk_env import build_env_cfg

    output = contract.safe_output_dir(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    training_path = output / contract.TRAINING_EVIDENCE
    training_path.unlink(missing_ok=True)
    prior, parent = contract.load_cumulative_prior()
    requested = contract.training_contract(
        seed=args.seed, num_envs=args.num_envs, iterations=args.iterations
    )
    run_identity = policy_stand._timestamp()
    run_dir = output / "runs" / run_identity
    run_dir.mkdir(parents=True, exist_ok=False)
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
        action_names = list(env.action_manager.get_term("joint_pos")._joint_names)
        if action_names != list(model_spec.ACTION_JOINTS):
            raise RuntimeError(f"runtime action order differs: {action_names}")
        observation_dim = env.observation_manager.group_obs_dim["policy"]
        if observation_dim != (contract.ACTOR_OBSERVATION_DIM,):
            raise RuntimeError(f"runtime actor observation dimension differs: {observation_dim}")
        command = env.command_manager.get_command("base_velocity")
        if command.shape != (args.num_envs, 3):
            raise RuntimeError(f"runtime velocity command shape differs: {tuple(command.shape)}")
        wrapped = RslRlVecEnvWrapper(env)
        runner_cfg = policy_stand._runner_cfg(args.seed, args.iterations)
        runner_cfg.experiment_name = "landau_forward_5m"
        runner = OnPolicyRunner(
            wrapped, runner_cfg.to_dict(), log_dir=os.fspath(run_dir), device=args.device
        )
        _stage(args, "stand_checkpoint_transfer")
        transfer = _transfer_stand_checkpoint(runner, Path(parent["resolved_path"]))
        _stage(args, "ppo_learning")
        runner.learn(num_learning_iterations=args.iterations, init_at_random_ep_len=True)
        checkpoints = list(run_dir.glob("model_*.pt"))
        if not checkpoints:
            raise RuntimeError("RSL-RL completed without a checkpoint")
        run_checkpoint = max(checkpoints, key=lambda path: int(path.stem.split("_")[-1]))
        checkpoint = output / "checkpoint.pt"
        checkpoint.unlink(missing_ok=True)
        shutil.copy2(run_checkpoint, checkpoint)
        payload = {
            "schema_version": 1,
            "milestone": contract.MILESTONE_ID,
            "status": "completed_not_promoted",
            "lineage": contract.LINEAGE,
            "run_identity": run_identity,
            "completed_at": policy_stand._timestamp(),
            "source_commit": policy_stand._source_commit(),
            "versions": policy_stand._versions(),
            "device": args.device,
            "single_process": True,
            "requested_contract": requested,
            "runtime_contract": {
                "environment_class": type(env).__name__,
                "action_order": action_names,
                "action_dimension": env.action_manager.total_action_dim,
                "actor_observation_dimension": observation_dim[0],
                "command_tensor_order": ["sim_linear_x_strafe", "sim_linear_y_forward", "yaw"],
                "policy_command_order": list(model_spec.SEMANTIC_COMMAND_ORDER),
                "reward_terms": list(env.reward_manager.active_terms),
                "termination_terms": list(env.termination_manager.active_terms),
            },
            "initialization": transfer,
            "input": model_spec.build_robot_spec()["source"],
            "robot_spec_sha256": contract.sha256(model_spec.ROBOT_SPEC_PATH),
            "checkpoint": {
                "kind": "rsl_rl_ppo",
                "path": str(checkpoint.relative_to(REPO_ROOT)),
                "sha256": contract.sha256(checkpoint),
                "size_bytes": checkpoint.stat().st_size,
                "learning_iteration": int(run_checkpoint.stem.split("_")[-1]),
                "run_checkpoint_path": str(run_checkpoint.relative_to(REPO_ROOT)),
            },
            "promotion": {
                "eligible": False,
                "reason": "training alone cannot pass the cumulative stand and 5 m validators",
            },
            "cumulative_gates": [
                *prior,
                {"order": 3, "id": contract.MILESTONE_ID, "status": "awaiting_validation"},
            ],
        }
        training_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({
            "status": payload["status"], "training": str(training_path),
            "checkpoint": payload["checkpoint"], "initialization": transfer,
        }, indent=2))
        return payload
    finally:
        if wrapped is not None:
            wrapped.close()
        else:
            env.close()


def _frame(path: Path, bbox: dict, index: int, time_s: float, displacement: float) -> dict:
    import numpy as np
    from PIL import Image, ImageDraw

    image = Image.open(path).convert("RGB")
    pixels = np.asarray(image)
    draw = ImageDraw.Draw(image)
    draw.rectangle((7, 7, 260, 33), fill=(245, 245, 245), outline=(30, 30, 30))
    draw.text((13, 13), f"Landau +Y forward · {displacement:04.2f} m · {time_s:04.1f}s", fill="black")
    image.save(path)
    return {
        "frame_index": index, "time_s": round(time_s, 6),
        "semantic_forward_displacement_m": round(displacement, 6),
        "nonblank": bool(float(pixels.std()) > 2.0),
        "rgb_stddev": round(float(pixels.std()), 5),
        "resolution": [image.width, image.height], **bbox,
    }


def _evaluate(args, training: dict, prior: list[dict]) -> dict:
    import torch
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from rsl_rl.runners import OnPolicyRunner

    from algorithms.urdf_learn_wasd_walk.forward_walk_env import build_env_cfg

    output = contract.safe_output_dir(args.output_dir)
    evidence_path = output / contract.component_artifact_name(args.mode, args.smoke)
    evidence_path.unlink(missing_ok=True)
    evaluation_command = "stand" if args.mode == "stand" else "forward"
    _stage(args, "manager_environment_construction")
    cfg = build_env_cfg(
        num_envs=1,
        seed=args.seed,
        training=False,
        evaluation_command=evaluation_command,
        force_usd_conversion=args.mode in {"stand", "forward"} and not args.reuse_usd_cache,
    )
    cfg.sim.device = args.device
    env = ManagerBasedRLEnv(cfg=cfg, render_mode=None)
    wrapped = None
    frame_context = None
    try:
        robot = env.scene["robot"]
        contacts = env.scene["contact_forces"]
        action_term = env.action_manager.get_term("joint_pos")
        action_names = list(action_term._joint_names)
        if action_names != list(model_spec.ACTION_JOINTS):
            raise RuntimeError(f"runtime action order differs: {action_names}")
        if env.observation_manager.group_obs_dim["policy"] != (contract.ACTOR_OBSERVATION_DIM,):
            raise RuntimeError("runtime actor observation dimension differs")
        spec = model_spec.build_robot_spec()
        reference_ids, names = robot.find_bodies("root_x", preserve_order=True)
        if names != ["root_x"]:
            raise RuntimeError(f"root reference differs: {names}")
        sensor_ids, support_names = contacts.find_bodies(
            ["foot_l", "toes_01_l", "foot_r", "toes_01_r"], preserve_order=True
        )
        body_ids, body_names = robot.find_bodies(support_names, preserve_order=True)
        if support_names != body_names or len(sensor_ids) != 4:
            raise RuntimeError("support-body contract differs")
        imported_axes = None
        if args.mode in {"stand", "forward"}:
            _stage(args, "importer_axis_audit")
            imported_axes = passive_stand._inspect_imported_joint_contract(
                env.sim.stage, "/World/envs/env_0/Robot", spec
            )
            if not imported_axes["passed"]:
                raise RuntimeError("imported axes differ from the URDF contract")
        wrapped = RslRlVecEnvWrapper(env)
        runner_cfg = policy_stand._runner_cfg(
            args.seed, training["requested_contract"]["iterations"]
        )
        runner_cfg.experiment_name = "landau_forward_5m"
        runner = OnPolicyRunner(wrapped, runner_cfg.to_dict(), log_dir=None, device=args.device)
        _stage(args, "checkpoint_load")
        runner.load(training["checkpoint"]["resolved_path"], load_optimizer=False)
        policy = runner.get_inference_policy(device=env.device)
        obs, _ = wrapped.get_observations()

        viewport = None
        frames = []
        visual_corners = None
        if args.mode == "proof":
            from omni.kit.viewport.utility import get_active_viewport

            viewport = get_active_viewport()
            if viewport is None:
                raise RuntimeError("proof phase has no active viewport")
            viewport.updates_enabled = True
            viewport.resolution = (args.video_width, args.video_height)
            visual_corners = model_spec.visual_local_aabb_corners()
            frame_context = tempfile.TemporaryDirectory(prefix="landau_forward_frames_", dir=output)
            frame_dir = Path(frame_context.name)

        max_steps = args.steps if args.steps is not None else round(
            (contract.STAND_DURATION_S if args.mode == "stand" else contract.MAX_GATE_DURATION_S)
            / contract.CONTROL_DT_S
        )
        video_stride = max(1, round((1.0 / args.video_fps) / contract.CONTROL_DT_S))
        initial_position = robot.data.body_pos_w[0, reference_ids[0]].detach().cpu().clone()
        initial_quaternion = robot.data.body_quat_w[0, reference_ids[0]].detach().cpu().clone()
        joint_indices = {name: index for index, name in enumerate(robot.joint_names)}
        gait_joints = (
            "left_hip_pitch_joint", "right_hip_pitch_joint", "left_knee_joint", "right_knee_joint"
        )
        minima = {name: math.inf for name in gait_joints}
        maxima = {name: -math.inf for name in gait_joints}
        previous_contact = {"left": None, "right": None}
        liftoffs = {"left": 0, "right": 0}
        air_steps = {"left": 0, "right": 0}
        simultaneous_air_steps = 0
        contact_slip_sum = 0.0
        contact_slip_samples = 0
        reset_count = done_count = fall_count = inference_steps = 0
        max_action = max_actor = max_tilt = max_height_drop = 0.0
        clipped = 0
        traces = []
        action_traces = []
        command_values = []
        body_masses = robot.data.default_mass[0].to(
            device=robot.device, dtype=robot.data.joint_pos.dtype
        )
        total_mass = torch.sum(body_masses)
        robot_weight_n = float(total_mass.detach().cpu()) * 9.81
        support_hull = spec["nominal_pose"]["geometry"]["support_hull_xy_m"]
        minimum_support_margin = math.inf
        first_support_exit_time = None
        peak_support_force = support_force_sum = 0.0
        _stage(args, "policy_inference_loop")
        for step in range(max_steps):
            with torch.inference_mode():
                actor = policy(obs)
                max_actor = max(max_actor, float(torch.max(torch.abs(actor)).detach().cpu()))
                clipped += int(torch.count_nonzero(torch.abs(actor) > contract.ACTION_CLIP).cpu())
                actions = torch.clamp(actor, -contract.ACTION_CLIP, contract.ACTION_CLIP)
                obs, _, dones, _ = wrapped.step(actions)
            inference_steps += 1
            raw_action = actions[0].detach().cpu().tolist()
            max_action = max(max_action, max(abs(float(value)) for value in raw_action))
            terminated = int(torch.count_nonzero(env.reset_terminated).detach().cpu())
            done = int(torch.count_nonzero(dones).detach().cpu())
            fall_count += terminated
            done_count += done
            reset_count += done
            time_s = (step + 1) * contract.CONTROL_DT_S
            position = robot.data.body_pos_w[0, reference_ids[0]].detach().cpu()
            quaternion = robot.data.body_quat_w[0, reference_ids[0]].detach().cpu()
            forward = float(position[1] - initial_position[1])
            strafe = float(position[0] - initial_position[0])
            tilt = passive_stand.quaternion_distance_wxyz(
                quaternion.tolist(), initial_quaternion.tolist()
            )
            max_tilt = max(max_tilt, tilt)
            max_height_drop = max(max_height_drop, float(initial_position[2] - position[2]))
            command = env.command_manager.get_command("base_velocity")[0].detach().cpu().tolist()
            semantic_command = [float(command[1]), float(command[0]), float(command[2])]
            command_values.append(semantic_command)
            forces = torch.linalg.vector_norm(
                contacts.data.net_forces_w[0, sensor_ids].detach().cpu(), dim=-1
            ).tolist()
            support_contact = [float(value) > 1.0 for value in forces]
            side_contact = {
                "left": support_contact[0] or support_contact[1],
                "right": support_contact[2] or support_contact[3],
            }
            for side in ("left", "right"):
                if previous_contact[side] is True and not side_contact[side]:
                    liftoffs[side] += 1
                if not side_contact[side]:
                    air_steps[side] += 1
                previous_contact[side] = side_contact[side]
            if not side_contact["left"] and not side_contact["right"]:
                simultaneous_air_steps += 1
            body_velocity = robot.data.body_lin_vel_w[0, body_ids].detach().cpu()
            for contact, velocity in zip(support_contact, body_velocity):
                if contact:
                    contact_slip_sum += math.hypot(float(velocity[0]), float(velocity[1]))
                    contact_slip_samples += 1
            current_joint = robot.data.joint_pos[0].detach().cpu()
            for name in gait_joints:
                value = float(current_joint[joint_indices[name]])
                minima[name] = min(minima[name], value)
                maxima[name] = max(maxima[name], value)
            system_com = (
                torch.sum(robot.data.body_com_pos_w[0] * body_masses.unsqueeze(-1), dim=0)
                / total_mass
            ).detach().cpu().tolist()
            support_margin = model_spec.support_polygon_margin(system_com[:2], support_hull)
            minimum_support_margin = min(minimum_support_margin, support_margin)
            if support_margin < 0.0 and first_support_exit_time is None:
                first_support_exit_time = time_s
            support_force = sum(float(value) for value in forces)
            peak_support_force = max(peak_support_force, support_force)
            support_force_sum += support_force
            traces.append({
                "time_s": round(time_s, 6),
                "root_x_position_m": [round(float(v), 8) for v in position.tolist()],
                "root_x_orientation_wxyz": [round(float(v), 9) for v in quaternion.tolist()],
                "semantic_command": [round(v, 6) for v in semantic_command],
                "semantic_forward_displacement_m": round(forward, 8),
                "semantic_strafe_displacement_m": round(strafe, 8),
                "reference_tilt_rad": round(tilt, 8),
                "system_center_of_mass_m": [round(float(v), 8) for v in system_com],
                "support_polygon_margin_m": round(support_margin, 8),
                "support_contact": dict(zip(support_names, support_contact)),
                "support_force_n": dict(zip(support_names, [round(float(v), 6) for v in forces])),
                "terminated": bool(terminated), "done": bool(done),
            })
            action_traces.append({
                "time_s": round(time_s, 6), "joint_order": action_names,
                "raw_policy_action": [round(float(v), 8) for v in raw_action],
            })
            if viewport is not None and (step + 1) % video_stride == 0:
                eye = policy_stand.contract.PROOF_CAMERA_EYE_OFFSET_M
                env.sim.set_camera_view(
                    eye=(eye[0] + float(position[0]), eye[1] + float(position[1]), eye[2]),
                    target=(float(position[0]), float(position[1]), policy_stand.contract.PROOF_CAMERA_TARGET_HEIGHT_M),
                )
                env.sim.render()
                frame_path = frame_dir / f"frame_{len(frames):04d}.png"
                passive_stand._capture_viewport_frame(viewport, env.sim, frame_path)
                origin_bbox = passive_stand._projected_link_bbox(
                    viewport, robot.data.body_pos_w[0].detach().cpu().tolist(),
                    args.video_width, args.video_height,
                )
                bbox = {key: value for key, value in origin_bbox.items() if key != "character_visible"}
                bbox["link_origin_scale_proxy_passed"] = origin_bbox["character_visible"]
                bbox.update(passive_stand._projected_visual_geometry_bbox(
                    viewport,
                    passive_stand._visual_geometry_world_corners(robot, visual_corners),
                    args.video_width,
                    args.video_height,
                ))
                frames.append(_frame(frame_path, bbox, len(frames), time_s, forward))
            if evaluation_command == "forward" and forward >= contract.TARGET_DISTANCE_M:
                break

        control_steps = len(traces)
        final = traces[-1]
        horizontal_drift = math.hypot(
            float(final["semantic_strafe_displacement_m"]),
            float(final["semantic_forward_displacement_m"]),
        )
        metrics = {
            "duration_s": round(control_steps * contract.CONTROL_DT_S, 6),
            "physics_steps": control_steps * round(contract.CONTROL_DT_S / contract.PHYSICS_DT_S),
            "control_steps": control_steps,
            "policy_inference_steps": inference_steps,
            "reset_count": reset_count, "done_count": done_count, "fall_count": fall_count,
            "max_abs_action": round(max_action, 8),
            "max_abs_actor_output_before_clip": round(max_actor, 8),
            "actor_output_clipped_value_count": clipped,
            "max_abs_command": round(max(abs(v) for values in command_values for v in values), 8),
            "max_reference_tilt_rad": round(max_tilt, 8),
            "root_height_drop_m": round(max(0.0, max_height_drop), 8),
            "horizontal_drift_m": round(horizontal_drift, 8),
            "semantic_forward_displacement_m": final["semantic_forward_displacement_m"],
            "semantic_strafe_displacement_m": final["semantic_strafe_displacement_m"],
            "left_foot_liftoff_count": liftoffs["left"],
            "right_foot_liftoff_count": liftoffs["right"],
            "left_foot_air_fraction": round(air_steps["left"] / control_steps, 8),
            "right_foot_air_fraction": round(air_steps["right"] / control_steps, 8),
            "simultaneous_air_fraction": round(simultaneous_air_steps / control_steps, 8),
            "leg_joint_excursion_rad": {
                name: round(maxima[name] - minima[name], 8) for name in gait_joints
            },
            "mean_contact_foot_slip_mps": round(
                contact_slip_sum / max(1, contact_slip_samples), 8
            ),
            "minimum_support_polygon_margin_m": round(minimum_support_margin, 8),
            "first_support_exit_time_s": (
                round(first_support_exit_time, 6) if first_support_exit_time is not None else None
            ),
            "peak_support_force_body_weight_ratio": round(
                peak_support_force / robot_weight_n, 8
            ),
            "mean_support_force_body_weight_ratio": round(
                support_force_sum / control_steps / robot_weight_n, 8
            ),
        }
        failures = (
            contract.evaluate_repassed_stand(
                metrics, required_duration_s=0.0 if args.smoke else contract.STAND_DURATION_S
            )
            if args.mode == "stand"
            else contract.evaluate_forward_gate(
                metrics,
                required_distance_m=0.0 if args.smoke else contract.TARGET_DISTANCE_M,
                require_gait=not args.smoke,
            )
        )
        video = None
        if args.mode == "proof":
            if not frames:
                failures.append("viewport produced no forward proof frames")
            else:
                from PIL import Image, ImageChops, ImageDraw, ImageStat

                video_path = output / ("proof_smoke.mp4" if args.smoke else "proof.mp4")
                sheet_path = output / ("contact_sheet_smoke.png" if args.smoke else "contact_sheet.png")
                passive_stand._encode_video(frame_dir, video_path, args.video_fps)
                indices = sorted({0, len(frames) // 2, len(frames) - 1})
                images = [
                    Image.open(frame_dir / f"frame_{index:04d}.png").convert("RGB") for index in indices
                ]
                sheet = Image.new("RGB", (args.video_width * len(images), args.video_height + 34), "white")
                draw = ImageDraw.Draw(sheet)
                for column, (index, image) in enumerate(zip(indices, images)):
                    sheet.paste(image, (column * args.video_width, 34))
                    draw.text((column * args.video_width + 8, 9), f"{frames[index]['semantic_forward_displacement_m']:.2f} m · {frames[index]['time_s']:.1f}s", fill="black")
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
                    "path": str(video_path.relative_to(REPO_ROOT)),
                    "sha256": contract.sha256(video_path),
                    "contact_sheet_path": str(sheet_path.relative_to(REPO_ROOT)),
                    "contact_sheet_sha256": contract.sha256(sheet_path),
                    "frame_count": len(frames), "fps": args.video_fps,
                    "duration_s": float(probe["format"]["duration"]),
                    "expected_duration_s": len(frames) / args.video_fps,
                    "resolution": [args.video_width, args.video_height],
                    "representative_frames": sampled,
                    "representative_mean_absolute_differences": differences,
                    "nonblank_frames_passed": all(item["nonblank"] for item in sampled),
                    "character_visibility_passed": all(item["character_visible"] for item in sampled),
                    "visual_geometry_framing_passed": all(item["visual_geometry_framing_passed"] for item in sampled),
                    "discernible_scale_passed": all(item["discernible_scale_passed"] for item in sampled),
                    "temporal_progression_visible": len(sampled) >= 3 and all(value > 0.005 for value in differences),
                    "motion_discernible": sampled[-1]["semantic_forward_displacement_m"] - sampled[0]["semantic_forward_displacement_m"] >= (0.1 if args.smoke else 4.5),
                }
                proof_passed, proof_failures = passive_stand.evaluate_proof(
                    video, required_duration_s=max(0.0, metrics["duration_s"] - 0.25)
                )
                if not proof_passed:
                    failures.extend(proof_failures)
                if not video["motion_discernible"]:
                    failures.append("proof frames do not visibly span the forward gate")

        evidence = {
            "schema_version": 1,
            "milestone": contract.MILESTONE_ID,
            "component": args.mode,
            "scope": "component_only",
            "status": "passed" if not failures else "failed",
            "gate_eligible": not args.smoke,
            "milestone_status_changed": False,
            "lineage": contract.LINEAGE,
            "run_identity": policy_stand._timestamp(),
            "seed": args.seed,
            "checkpoint": {key: value for key, value in training["checkpoint"].items() if key != "resolved_path"},
            "training_evidence": {
                "path": str((output / contract.TRAINING_EVIDENCE).relative_to(REPO_ROOT)),
                "sha256": contract.sha256(output / contract.TRAINING_EVIDENCE),
                "run_identity": training["run_identity"],
            },
            "source_commit": policy_stand._source_commit(),
            "versions": policy_stand._versions(),
            "simulator": {
                "device": args.device, "physics_dt_s": contract.PHYSICS_DT_S,
                "control_dt_s": contract.CONTROL_DT_S, "single_process": True,
                "num_envs": 1, "rendering_enabled": args.mode == "proof",
                "camera_sensor_created": False,
            },
            "input": spec["source"],
            "joint_contract": {
                "action_joints": spec["action_joints"], "locked_joints": spec["locked_joints"],
                "runtime_action_order": action_names, "nominal_pose": spec["nominal_pose"],
                "runtime_importer_axis_evidence": imported_axes,
            },
            "policy_contract": training["requested_contract"],
            "command_contract": {
                **spec["frames"],
                "evaluation_semantic_command": [0.0, 0.0, 0.0] if evaluation_command == "stand" else [contract.TARGET_FORWARD_SPEED_MPS, 0.0, 0.0],
                "evaluation_sim_command": list(contract.semantic_command_to_sim(
                    0.0 if evaluation_command == "stand" else contract.TARGET_FORWARD_SPEED_MPS
                )),
            },
            "metrics": metrics, "traces": traces, "action_traces": action_traces,
            "video_inspection": video, "failures": failures,
            "cumulative_gates": [
                *prior,
                {"order": 2, "id": "stand_30s_no_reset", "status": "candidate_repass" if args.mode == "stand" else "pending_component_assembly"},
                {"order": 3, "id": contract.MILESTONE_ID, "status": "pending_component_assembly"},
            ],
        }
        evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({
            "status": evidence["status"], "component": args.mode,
            "metrics": metrics, "evidence": str(evidence_path),
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
    parser.add_argument("--mode", choices=("train", "stand", "forward", "proof"), required=True)
    parser.add_argument("--output-dir", type=Path, default=contract.DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-envs", type=int, default=512)
    parser.add_argument("--iterations", type=int, default=600)
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
        raise SystemExit("seed must be non-negative and training sizes positive")
    if args.steps is not None and args.steps <= 0:
        raise SystemExit("steps must be positive")
    if args.mode == "train" and args.smoke:
        raise SystemExit("bound training with --iterations and --num-envs, not --smoke")
    prior = training = None
    try:
        if args.mode != "train":
            prior, _ = contract.load_cumulative_prior()
            training = contract.load_training_evidence(args.output_dir)
            if args.mode == "proof" and not args.smoke:
                for component_mode in ("stand", "forward"):
                    dynamics_path = args.output_dir / contract.component_artifact_name(component_mode)
                    dynamics = (
                        json.loads(dynamics_path.read_text(encoding="utf-8"))
                        if dynamics_path.is_file() else {}
                    )
                    if (
                        dynamics.get("status") != "passed"
                        or dynamics.get("checkpoint", {}).get("sha256")
                        != training["checkpoint"]["sha256"]
                    ):
                        raise ValueError(
                            "proof requires matching passed exact candidate-stand and forward dynamics"
                        )
    except ValueError as error:
        print(f"Forward preflight failed: {error}", file=sys.stderr)
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
        evidence = _train(args) if args.mode == "train" else _evaluate(args, training, prior)
        return 0 if evidence["status"] in {"completed_not_promoted", "passed"} else 1
    except Exception as error:
        traceback.print_exc(file=sys.stderr)
        try:
            _write_failure(args, error)
        except Exception:
            traceback.print_exc(file=sys.stderr)
        return 1
    finally:
        if launcher is not None:
            launcher.app.close()


if __name__ == "__main__":
    raise SystemExit(main())
