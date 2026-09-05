"""Run one camera-free, open-loop Landau phase-reference experiment."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import traceback
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from algorithms.urdf_learn_wasd_walk import forward_reference
from algorithms.urdf_learn_wasd_walk import forward_walk_contract as contract
from algorithms.urdf_learn_wasd_walk import model_spec, passive_stand, policy_stand
from algorithms.urdf_learn_wasd_walk import policy_stand_contract


DEFAULT_PARENT_OUTPUT = policy_stand_contract.DEFAULT_OUTPUT_DIR
DEFAULT_OUTPUT = contract.DEFAULT_OUTPUT_DIR / "reference_probe_v3" / "baseline"


def _stage(args, name: str) -> None:
    args.runtime_stage = name
    print(f"[landau-reference-probe] stage={name}", file=sys.stderr, flush=True)


def _write_failure(args, error: Exception) -> None:
    output = contract.safe_output_dir(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    traceback_path = output / "reference_probe_traceback.log"
    traceback_path.write_text(traceback.format_exc(), encoding="utf-8")
    (output / "reference_probe_failure.json").write_text(json.dumps({
        "schema_version": 1,
        "milestone": contract.MILESTONE_ID,
        "experiment": forward_reference.METHOD_ID,
        "status": "failed_to_execute",
        "runtime_stage": getattr(args, "runtime_stage", "unknown"),
        "exception": {"type": type(error).__name__, "message": str(error)},
        "traceback_path": str(traceback_path.relative_to(REPO_ROOT)),
        "traceback_sha256": contract.sha256(traceback_path),
        "argv": list(sys.argv),
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _longest_true_run(values: list[bool]) -> int:
    longest = current = 0
    for value in values:
        current = current + 1 if value else 0
        longest = max(longest, current)
    return longest


def probe_protocol(settle_steps: int, reference_steps: int) -> dict:
    """Describe the command-onset baseline without weakening positive progress."""

    if settle_steps < 0 or reference_steps <= 0:
        raise ValueError("settle steps must be non-negative and reference steps positive")
    return {
        "pre_command_zero_action_steps": settle_steps,
        "pre_command_zero_action_duration_s": round(settle_steps * contract.CONTROL_DT_S, 6),
        "reference_steps": reference_steps,
        "reference_duration_s": round(reference_steps * contract.CONTROL_DT_S, 6),
        "forward_displacement_baseline": "root +Y at command onset after zero-action settling",
        "total_displacement_also_recorded": True,
    }


def _run(args, training: dict, prior: list[dict]) -> dict:
    import torch
    from isaaclab.envs import ManagerBasedRLEnv

    from algorithms.urdf_learn_wasd_walk.forward_walk_env import build_env_cfg

    output = contract.safe_output_dir(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    evidence_path = output / "reference_probe.json"
    evidence_path.unlink(missing_ok=True)
    config = forward_reference.ReferenceConfig(
        period_s=args.period_s,
        amplitude_scale=args.amplitude_scale,
        startup_ramp_s=args.startup_ramp_s,
        hip_phase_offset_cycles=args.hip_phase_offset,
        knee_phase_offset_cycles=args.knee_phase_offset,
        ankle_phase_offset_cycles=args.ankle_phase_offset,
        toe_phase_offset_cycles=args.toe_phase_offset,
    )
    reference = forward_reference.reference_contract(config)
    _stage(args, "manager_environment_construction")
    cfg = build_env_cfg(
        num_envs=1,
        seed=args.seed,
        training=False,
        evaluation_command="forward",
        force_usd_conversion=not args.reuse_usd_cache,
    )
    cfg.sim.device = args.device
    env = ManagerBasedRLEnv(cfg=cfg, render_mode=None)
    try:
        robot = env.scene["robot"]
        contacts = env.scene["contact_forces"]
        action_term = env.action_manager.get_term("joint_pos")
        action_names = list(action_term._joint_names)
        if action_names != list(model_spec.ACTION_JOINTS):
            raise RuntimeError(f"runtime action order differs: {action_names}")
        spec = model_spec.build_robot_spec()
        root_ids, root_names = robot.find_bodies("root_x", preserve_order=True)
        if root_names != ["root_x"]:
            raise RuntimeError(f"root reference differs: {root_names}")
        sensor_ids, support_names = contacts.find_bodies(
            ["foot_l", "toes_01_l", "foot_r", "toes_01_r"], preserve_order=True
        )
        body_ids, body_names = robot.find_bodies(support_names, preserve_order=True)
        if support_names != body_names or len(sensor_ids) != 4:
            raise RuntimeError("support-body contract differs")
        _stage(args, "importer_axis_audit")
        imported_axes = passive_stand._inspect_imported_joint_contract(
            env.sim.stage, "/World/envs/env_0/Robot", spec
        )
        if not imported_axes["passed"]:
            raise RuntimeError("imported axes differ from the exact URDF contract")
        initial_root = robot.data.body_pos_w[0, root_ids[0]].detach().cpu().clone()
        initial_support_z = robot.data.body_pos_w[0, body_ids, 2].detach().cpu().clone()
        command_root = initial_root.clone()
        command_support_z = initial_support_z.clone()
        initial_quaternion = robot.data.body_quat_w[0, root_ids[0]].detach().cpu().clone()
        joint_indices = {name: index for index, name in enumerate(robot.joint_names)}
        air_samples = {"left": [], "right": []}
        max_height_gain = {"left": 0.0, "right": 0.0}
        max_target_errors = {name: 0.0 for name in action_names}
        reset_count = done_count = fall_count = 0
        max_tilt = max_height_drop = max_abs_action = 0.0
        traces = []
        _stage(args, "open_loop_reference")
        total_steps = args.settle_steps + args.steps
        for step in range(total_steps):
            reference_step = step - args.settle_steps
            in_reference = reference_step >= 0
            time_s = max(0, reference_step) * contract.CONTROL_DT_S
            action_values = (
                forward_reference.reference_action(
                    time_s, contract.TARGET_FORWARD_SPEED_MPS, config
                )
                if in_reference else [0.0] * len(action_names)
            )
            actions = torch.tensor([action_values], device=env.device, dtype=torch.float32)
            with torch.inference_mode():
                _, _, terminated, truncated, _ = env.step(actions)
            done = torch.logical_or(terminated, truncated)
            terminated_count = int(torch.count_nonzero(terminated).detach().cpu())
            done_count_step = int(torch.count_nonzero(done).detach().cpu())
            fall_count += terminated_count
            done_count += done_count_step
            reset_count += done_count_step
            max_abs_action = max(max_abs_action, max(map(abs, action_values), default=0.0))
            position = robot.data.body_pos_w[0, root_ids[0]].detach().cpu()
            quaternion = robot.data.body_quat_w[0, root_ids[0]].detach().cpu()
            tilt = passive_stand.quaternion_distance_wxyz(
                quaternion.tolist(), initial_quaternion.tolist()
            )
            max_tilt = max(max_tilt, tilt)
            max_height_drop = max(max_height_drop, float(initial_root[2] - position[2]))
            forces = torch.linalg.vector_norm(
                contacts.data.net_forces_w[0, sensor_ids].detach().cpu(), dim=-1
            ).tolist()
            support_contact = [float(value) > 1.0 for value in forces]
            side_contact = {
                "left": support_contact[0] or support_contact[1],
                "right": support_contact[2] or support_contact[3],
            }
            support_z = robot.data.body_pos_w[0, body_ids, 2].detach().cpu()
            if step == args.settle_steps - 1:
                command_root = position.clone()
                command_support_z = support_z.clone()
            for side, indices in (("left", (0, 1)), ("right", (2, 3))):
                if in_reference:
                    air_samples[side].append(not side_contact[side])
                    max_height_gain[side] = max(
                        max_height_gain[side],
                        max(float(support_z[index] - command_support_z[index]) for index in indices),
                    )
            current_joint = robot.data.joint_pos[0].detach().cpu()
            current_target = robot.data.joint_pos_target[0].detach().cpu()
            target_errors = []
            for name in action_names:
                index = joint_indices[name]
                error = abs(float(current_target[index] - current_joint[index]))
                target_errors.append(error)
                max_target_errors[name] = max(max_target_errors[name], error)
            traces.append({
                "time_s": round((step + 1) * contract.CONTROL_DT_S, 6),
                "phase": "reference" if in_reference else "pre_command_settle",
                "semantic_forward_displacement_m": round(float(position[1] - command_root[1]), 8),
                "total_semantic_forward_displacement_m": round(float(position[1] - initial_root[1]), 8),
                "semantic_strafe_displacement_m": round(float(position[0] - initial_root[0]), 8),
                "root_height_change_m": round(float(position[2] - initial_root[2]), 8),
                "reference_tilt_rad": round(tilt, 8),
                "normalized_reference_action": [round(float(value), 8) for value in action_values],
                "support_force_n": dict(zip(support_names, [round(float(value), 6) for value in forces])),
                "direct_side_contact": side_contact,
                "support_body_height_gain_m": dict(zip(
                    support_names,
                    [round(float(support_z[index] - initial_support_z[index]), 8) for index in range(4)],
                )),
                "max_action_joint_target_error_rad": round(max(target_errors), 8),
                "terminated": bool(terminated_count),
                "done": bool(done_count_step),
            })
        final = traces[-1]
        metrics = {
            "duration_s": round(total_steps * contract.CONTROL_DT_S, 6),
            "reference_duration_s": round(args.steps * contract.CONTROL_DT_S, 6),
            "physics_steps": total_steps * round(contract.CONTROL_DT_S / contract.PHYSICS_DT_S),
            "control_steps": total_steps,
            "reference_control_steps": args.steps,
            "policy_inference_steps": 0,
            "reset_count": reset_count,
            "done_count": done_count,
            "fall_count": fall_count,
            "semantic_forward_displacement_m": final["semantic_forward_displacement_m"],
            "total_semantic_forward_displacement_m": final["total_semantic_forward_displacement_m"],
            "semantic_strafe_displacement_m": final["semantic_strafe_displacement_m"],
            "max_reference_tilt_rad": round(max_tilt, 8),
            "root_height_drop_m": round(max(0.0, max_height_drop), 8),
            "max_abs_action": round(max_abs_action, 8),
            "max_joint_target_error_rad": round(max(max_target_errors.values()), 8),
            "max_joint_target_error_by_action_joint_rad": {
                name: round(value, 8) for name, value in max_target_errors.items()
            },
        }
        for side in ("left", "right"):
            metrics[f"{side}_direct_air_control_steps"] = sum(air_samples[side])
            metrics[f"{side}_direct_air_fraction"] = round(
                sum(air_samples[side]) / len(air_samples[side]), 8
            )
            metrics[f"{side}_max_consecutive_direct_air_steps"] = _longest_true_run(
                air_samples[side]
            )
            metrics[f"{side}_max_support_body_height_gain_m"] = round(
                max_height_gain[side], 8
            )
        failures = forward_reference.evaluate_probe(metrics, reference)
        evidence = {
            "schema_version": 1,
            "milestone": contract.MILESTONE_ID,
            "component": "open_loop_reference_probe",
            "experiment": forward_reference.METHOD_ID,
            "status": "passed" if not failures else "failed",
            "gate_eligible": False,
            "ppo_eligible": not failures,
            "milestone_status_changed": False,
            "lineage": contract.LINEAGE,
            "run_identity": policy_stand._timestamp(),
            "source_commit": policy_stand._source_commit(),
            "seed": args.seed,
            "versions": policy_stand._versions(),
            "simulator": {
                "device": args.device,
                "physics_dt_s": contract.PHYSICS_DT_S,
                "control_dt_s": contract.CONTROL_DT_S,
                "single_process": True,
                "num_envs": 1,
                "rendering_enabled": False,
                "camera_sensor_created": False,
            },
            "input": spec["source"],
            "parent_checkpoint": {
                key: value for key, value in training["checkpoint"].items() if key != "resolved_path"
            },
            "parent_training": {
                "path": str((args.parent_output_dir / contract.TRAINING_EVIDENCE).relative_to(REPO_ROOT)),
                "sha256": contract.sha256(args.parent_output_dir / contract.TRAINING_EVIDENCE),
                "run_identity": training["run_identity"],
                "training_method": "current_mesh_policy_stand_parent",
            },
            "reference_contract": reference,
            "probe_protocol": probe_protocol(args.settle_steps, args.steps),
            "joint_contract": {
                "action_joints": spec["action_joints"],
                "locked_joints": spec["locked_joints"],
                "runtime_action_order": action_names,
                "runtime_importer_axis_evidence": imported_axes,
                "nominal_pose": spec["nominal_pose"],
            },
            "metrics": metrics,
            "traces": traces,
            "failures": failures,
            "decision": (
                "reference probe passed; a bounded PPO smoke may use this reference plus residuals"
                if not failures else
                "do not start PPO; change exactly one bounded reference factor and probe again"
            ),
            "cumulative_gates": [
                *prior,
                {"order": 3, "id": contract.MILESTONE_ID, "status": "in_progress"},
            ],
        }
        temporary = evidence_path.with_suffix(".json.pending")
        temporary.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, evidence_path)
        print(json.dumps({
            "status": evidence["status"],
            "ppo_eligible": evidence["ppo_eligible"],
            "metrics": metrics,
            "failures": failures,
            "evidence": str(evidence_path),
        }, indent=2))
        return evidence
    finally:
        env.close()


def _parser() -> argparse.ArgumentParser:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--parent-output-dir", type=Path, default=DEFAULT_PARENT_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--settle-steps", type=int, default=50)
    parser.add_argument("--period-s", type=float, default=contract.GAIT_PERIOD_S)
    parser.add_argument("--amplitude-scale", type=float, default=1.0)
    parser.add_argument("--startup-ramp-s", type=float, default=0.4)
    parser.add_argument("--hip-phase-offset", type=float, default=0.0)
    parser.add_argument("--knee-phase-offset", type=float, default=0.0)
    parser.add_argument("--ankle-phase-offset", type=float, default=0.0)
    parser.add_argument("--toe-phase-offset", type=float, default=0.0)
    parser.add_argument("--reuse-usd-cache", action="store_true")
    AppLauncher.add_app_launcher_args(parser)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    from isaaclab.app import AppLauncher

    args = _parser().parse_args(argv)
    args.output_dir = contract.safe_output_dir(args.output_dir)
    args.parent_output_dir = contract.safe_output_dir(args.parent_output_dir)
    if args.seed < 0 or args.steps <= 0 or args.settle_steps < 0:
        raise SystemExit("seed/settle-steps must be non-negative and steps positive")
    try:
        prior, parent = contract.load_cumulative_prior()
        training = policy_stand_contract.load_training_evidence(args.parent_output_dir)
        if training.get("checkpoint", {}).get("sha256") != parent.get("sha256"):
            raise ValueError("reference probe parent is not the canonical current-mesh stand checkpoint")
        forward_reference.ReferenceConfig(
            period_s=args.period_s,
            amplitude_scale=args.amplitude_scale,
            startup_ramp_s=args.startup_ramp_s,
            hip_phase_offset_cycles=args.hip_phase_offset,
            knee_phase_offset_cycles=args.knee_phase_offset,
            ankle_phase_offset_cycles=args.ankle_phase_offset,
            toe_phase_offset_cycles=args.toe_phase_offset,
        ).validate()
    except ValueError as error:
        print(f"Reference probe preflight failed: {error}", file=sys.stderr)
        return 1
    os.environ["ENABLE_CAMERAS"] = "0"
    args.enable_cameras = False
    args.video = False
    launcher = None
    try:
        _stage(args, "app_launcher")
        launcher = AppLauncher(args)
        evidence = _run(args, training, prior)
        return 0 if evidence["status"] == "passed" else 1
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
