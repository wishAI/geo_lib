"""Build a compact, truthful evolution tree from Landau run artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path


ALGORITHM_ROOT = Path(__file__).resolve().parent
REPO_ROOT = ALGORITHM_ROOT.parents[1]
OUTPUT_ROOT = ALGORITHM_ROOT / "outputs"
MILESTONES_PATH = ALGORITHM_ROOT / "milestones.json"
DEFAULT_OUTPUT = OUTPUT_ROOT / "evolution.json"
VISIBLE_NODE_BUDGET = 40
METRIC_KEYS = (
    "semantic_forward_displacement_m",
    "mean_semantic_forward_velocity_mps",
    "reverse_motion_step_fraction",
    "left_foot_liftoff_count",
    "right_foot_liftoff_count",
    "left_max_consecutive_direct_air_steps",
    "right_max_consecutive_direct_air_steps",
    "left_max_support_body_height_gain_m",
    "right_max_support_body_height_gain_m",
    "max_joint_target_error_rad",
    "max_reference_tilt_rad",
    "reset_count",
    "done_count",
    "fall_count",
)


def _chronology(node: dict) -> str:
    stamp = str(node.get("startedAt") or "")
    return re.sub(r"\D", "", stamp)[:14].ljust(14, "0") if stamp.startswith("20") else "0" * 14


def _run_label(name: str) -> str:
    return {
        'rsl_transfer_20260922_teleop5_m5': '90° recheck · settling regression',
        'rsl_transfer_20260922_teleop_preserve_left': 'Joystick control · preserve certified turn',
        'rsl_transfer_20260922_teleop_refine': 'Joystick control · stop and turn refinement',
        'rsl_transfer_20260922_teleop_refine5_eval': 'Joystick control · independent60s check',
        'rsl_transfer_20260922_teleop_baseline': 'Joystick test · right-turn failure',
        'rsl_transfer_20260922_teleop_anchor_eval': 'Joystick test · restart failure',
        'rsl_transfer_20260922_teleop_smoke': 'Joystick control · first diagnostic',
        'rsl_transfer_20260922_teleop_continuous': 'Joystick control · continuous gait',
        'rsl_transfer_20260922_teleop_right_grid': 'Right-turn feedback · first grid',
        'rsl_transfer_20260922_teleop_anchor_only': 'Repeated stops · minimal memory fix',
        'rsl_transfer_20260922_teleop_restart_grid': 'Restart timing · ramp and phase search',
        'rsl_transfer_20260922_teleop_right_after_restart': 'Right turn · improved restart',
        'rsl_transfer_20260922_turn44_3_eval': '90° turn and hold · certified',
        'rsl_transfer_20260922_turn40_0_eval': 'Turn and hold · 81° reached',
        'rsl_transfer_20260922_turn_balance6_eval': 'Turn and hold · 15° reached',
        'rsl_transfer_20260922_turn_44_cem': '90° turn · local refinement',
        'rsl_transfer_20260922_turn_40_cem': '90° turn · duration calibration',
        'rsl_transfer_20260922_turn_slow70': 'Slow turn · sustained rotation test',
        'rsl_transfer_20260922_turn_balance_cem': 'Turn balance · learned weight shift',
        'rsl_transfer_20260922_turn_negative_eval': 'Differential stride · turn check',
        'rsl_transfer_20260922_turn_negative_full': 'Differential stride · full trials',
        'rsl_transfer_20260922_turn_smoke_eval': 'Turn attempt · support timing failed',
        'rsl_transfer_20260922_turn_stop_eval': 'Walk and stop · stable, turn angle failed',
        'rsl_transfer_20260922_turn_smoke': 'First yaw extension · failed',
        'rsl_transfer_20260922_turn_null_sweep': 'Yaw amplitude · eight-second diagnostic',
        'rsl_transfer_20260922_turn_heading_grid': 'Yaw feedback · bounded grid',
        'rsl_transfer_20260922_turn_heading_full': 'Turn and hold · full-duration grid',
        'rsl_transfer_20260922_turn_stride_grid': 'Differential step length · diagnostic',

        "resume_20260922_balanced_passive": "Adjusted mass · passive stand",
        "resume_20260922_student_stand": "Previous student · zero-command check",
        "resume_20260922_balanced_stand_100": "Adjusted mass · PPO 100 iterations",
        "resume_20260922_balanced_stand_lr1e4": "Adjusted mass · PPO lower learning rate",
        "resume_20260922_balanced_policy_eval": "PPO 100 · standing evaluation",
        "resume_20260922_balanced_policy_lr1e4_eval": "Lower-rate PPO · standing evaluation",
        "rsl_transfer_20260922_forward_initial_stand": "Transfer check · standing retained",
        "rsl_transfer_20260922_forward_initial_move": "Transfer check · amplified actions",
        "rsl_transfer_20260922_legacy_initial_move": "Transfer check · preserved radians",
        "rsl_transfer_20260922_forward1000": "Walking PPO · stopped at 424 updates",
        "rsl_transfer_20260922_legacy_units500": "Walking PPO · corrected action units",
        "rsl_transfer_20260922_legacy500_eval": "PPO 500 · no foot lift",
        "rsl_transfer_20260922_range400_eval": "PPO 400 · no foot lift",
        "rsl_transfer_20260922_exploration200": "Walking PPO · leg exploration",
        "rsl_transfer_20260922_exploration199_eval": "Leg exploration · walking check",
        "rsl_transfer_20260922_hiproll040_200": "Walking PPO · wider hip roll",
        "rsl_transfer_20260922_hiproll040_199_eval": "Wider hip roll · walking check",
        "rsl_transfer_20260922_tilt050_200": "Walking PPO · lower lean penalty",
        "rsl_transfer_20260922_tilt050_199_eval": "Lower lean penalty · walking check",
        "rsl_transfer_20260922_tilt050_199_stand": "Lower lean penalty · standing check",
        "rsl_transfer_20260922_freshmoving200": "Fresh walking PPO · protected stand",
        "rsl_transfer_20260922_freshmoving_199_eval": "Fresh walking branch · walking check",
        "rsl_transfer_20260922_freshmoving_199_stand": "Frozen standing branch · standing check",
        "rsl_transfer_20260922_feedback_cem_smoke": "Gait feedback search · smoke test",
        "rsl_transfer_20260922_feedback_cem20": "Gait and balance feedback · search",
        "rsl_transfer_20260922_feedback_cem19_eval": "Learned gait feedback · walking check",
        "rsl_transfer_20260922_feedback_cem0_eval": "Stepping reference · 30 s, no fall",
        "rsl_transfer_20260922_feedback_forward17_eval": "Forward progress · 2.31 m, limits exceeded",
        "rsl_transfer_20260922_speedlimit20": "Forward search · physics-step speed limits",
        "rsl_transfer_20260922_stride_sweep_best_eval": "Stepping reference · 0.940 m, only distance failed",
        "rsl_transfer_20260922_negative_coordination_grid": "Alternate leg coordination · phase search",
        "rsl_transfer_20260922_signed_stride_smoke": "Alternate leg coordination · smoke test",
        "rsl_transfer_20260922_stride_velocity_best_eval": "Stepping reference · 1.112 m / 5 m",
        "rsl_transfer_20260922_negative_cem20": "Alternate leg coordination · learning balance",
        "rsl_transfer_20260922_negative_best_eval": "Correct swing timing · balance failed at 7 s",
        "rsl_transfer_20260922_negative_robust20": "Sustained stepping · four-start balance training",
        "rsl_transfer_20260922_negative_robust_smoke": "Longer stepping · training smoke test",
        "rsl_transfer_20260922_negative_parity": "Training and validation · state comparison",
        "rsl_transfer_20260922_negative_parity_refresh": "Physics refresh timing · comparison",
        "rsl_transfer_20260922_negative_lateral12": "Lateral feedback · bounded comparison",
        "rsl_transfer_20260922_negative_robust17_eval": "Stepping reference · 3.080 m / 5 m",
        "rsl_transfer_20260922_negative_stride_period30": "Stride and cadence · full-duration search",
        "rsl_transfer_20260922_negative_stride_knee30": "Stride and knee lift · impact comparison",
        "rsl_transfer_20260922_negative_full30_refine": "Full-duration balance and distance refinement",
        "rsl_transfer_20260922_negative_full30_18_eval_retry": "Stepping reference · 4.046 m / 5 m",
        "rsl_transfer_20260922_negative_scale2048_smoke": "Training throughput · 2,048 worlds",
        "rsl_transfer_20260922_negative_scale512_smoke": "Training throughput · 512 worlds",
        "rsl_transfer_20260922_negative_scale1024_smoke": "Training throughput · 1,024 worlds",
        "rsl_transfer_20260922_negative_distance1024": "Distance refinement · 1,024 worlds",
        "rsl_transfer_20260922_matched_swing_smoke": "Swing counting and flight limits · verified",
        "rsl_transfer_20260922_matched_distance1024": "Distance refinement · matched walking checks",
        "rsl_transfer_20260922_matched9_eval": "Stepping reference · 4.264 m / 5 m",
        "rsl_transfer_20260922_matched9_gate10m80": "10 m endurance check · fell after 6.37 m",
        "rsl_transfer_20260922_endurance3_gate10m80": "10 m reached · lateral drift failed",
        "rsl_transfer_20260922_heading_fine80": "10 m straightness · refined heading control",
        "rsl_transfer_20260922_headingfine_gate10m80": "10 m walking · independent full-duration validation",
        "rsl_transfer_20260922_headingfine_gate5m45": "Current 10 m candidate · 5 m recheck",
        "rsl_transfer_20260922_headingfine_m1": "Current 10 m candidate · passive stand recheck",
        "rsl_transfer_20260922_headingfine_m2": "Current 10 m candidate · policy stand recheck",
        "rsl_transfer_20260922_heading_signed80": "10 m straightness · heading feedback comparison",
        "rsl_transfer_20260922_endurance80": "10 m endurance training · four starting states",
        "rsl_transfer_20260922_matched9_gate45": "5 m walking validation · 6.494 m in 45 s",
        "rsl_transfer_20260922_matched9_eval45": "Longer walking diagnostic · old time bound",
        "matched9_cumulative_m1": "Current walking checkpoint · passive stand recheck",
        "matched9_cumulative_m2": "Current walking checkpoint · policy stand recheck",
        "rsl_transfer_20260922_signed_phase_stride30": "Stride and touchdown phase · bounded comparison",
        "rsl_transfer_20260922_roll_stride30": "Stride and lateral sway · bounded comparison",
        "rsl_transfer_20260922_startup_stride30": "Stride and startup ramp · impact comparison",
        "rsl_transfer_20260922_velocity_gain_stride30": "Stride and velocity feedback · bounded comparison",
        "rsl_transfer_20260922_stride_velocity30": "Stride and speed-feedback grid",
        "rsl_transfer_20260922_stride_phase30": "Stride and touchdown timing grid",
        "rsl_transfer_20260922_stride_hipfeedback30": "Stride and hip balance grid",
        "rsl_transfer_20260922_stride_damping30": "Stride and ankle damping grid",
        "rsl_transfer_20260922_full30_forward20": "Full-duration search · foot-lift regression",
        "rsl_transfer_20260922_stride_sweep30": "Stride length sweep · fixed balance controller",
        "rsl_transfer_20260922_robustforce20": "Robust gait search · four starting poses",
        "rsl_transfer_20260922_robustforce_best_eval": "Stepping reference · 0.708 m, only distance failed",
        "rsl_transfer_20260922_feedback_forward20": "Forward gait · learned balance",
        "rsl_transfer_20260922_feedback_forward19_eval": "Forward gait · walking check",
        "resume_20260922_shortening_pulse_l": "Right leg shortening · diagnostic",
        "resume_20260922_shortening_pulse_r": "Left leg shortening · diagnostic",
        "resume_20260922_hiproll040_support_l": "Left support · target transition",
        "resume_20260922_hiproll040_support_r": "Right support · target transition",
        "resume_20260922_hiproll_preload_l": "Positive hip roll · weight shift",
        "resume_20260922_hiproll_preload_r": "Negative hip roll · weight shift",
    }.get(name, name.replace("_", " "))


def _read_json(path: Path) -> dict | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def _checkpoint_sha(payload: dict) -> str | None:
    value = payload.get("checkpoint")
    return value.get("sha256") if isinstance(value, dict) else None


def _parent_checkpoint_sha(payload: dict) -> str | None:
    predecessor = payload.get("predecessor_failed_gate")
    if isinstance(predecessor, dict):
        checkpoint = predecessor.get("checkpoint")
        if isinstance(checkpoint, dict) and checkpoint.get("sha256"):
            return str(checkpoint["sha256"])
    requested = payload.get("requested_contract", {})
    initialization = requested.get("initialization", {}) if isinstance(requested, dict) else {}
    for key in ("sha256", "parent_checkpoint_sha256", "source_checkpoint_sha256"):
        if isinstance(initialization, dict) and initialization.get(key):
            return str(initialization[key])
    return None


def _validation_for(training_path: Path) -> tuple[Path | None, dict | None]:
    candidates = (
        "validation.json",
        "forward_dynamics_validation.json",
        "forward_dynamics_smoke_validation.json",
        "dynamics_validation.json",
        "dynamics_smoke_validation.json",
    )
    for name in candidates:
        path = training_path.parent / name
        payload = _read_json(path)
        if payload is not None:
            return path, payload
    return None, None


def _metrics(validation: dict | None) -> dict[str, float]:
    raw = validation.get("metrics", {}) if validation else {}
    return {
        key: value
        for key in METRIC_KEYS
        if isinstance((value := raw.get(key)), (int, float)) and not isinstance(value, bool)
    }


def _run_status(training: dict, validation: dict | None, metrics: dict) -> tuple[str, str, bool]:
    if validation and validation.get("status") == "failed":
        failures = validation.get("failures") or ["validator rejected this checkpoint"]
        return "failed", "; ".join(map(str, failures[:3])), True
    if validation and validation.get("status") == "passed" and validation.get("gate_eligible") is True:
        return "completed", "candidate passed the recorded gate evaluation", True
    if validation and "semantic_forward_displacement_m" in metrics:
        displacement = float(metrics.get("semantic_forward_displacement_m", 0.0))
        left = int(metrics.get("left_foot_liftoff_count", 0))
        right = int(metrics.get("right_foot_liftoff_count", 0))
        if displacement < 0.1 or left < 1 or right < 1:
            return (
                "failed",
                f"diagnostic: {displacement:.3f} m, liftoff L/R {left}/{right}",
                True,
            )
        return "completed", f"diagnostic: {displacement:.3f} m with bilateral liftoff", True
    status = str(training.get("status", "completed_not_promoted"))
    return ("running", "training is still running", True) if status == "running" else (
        "completed",
        "training completed; exact gate validation is pending",
        False,
    )


def _artifact(path: Path, produced_by: str) -> dict:
    return {
        "id": f"artifact:{_relative(path)}",
        "kind": "text",
        "path": _relative(path),
        "mimeType": "application/json",
        "byteSize": path.stat().st_size,
        "producedBy": produced_by,
    }


def _digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _proof_artifacts(folder: Path, node_id: str, evidence: dict) -> list[dict]:
    """Attach only media whose saved provenance belongs to this exact result."""
    attached = []
    metadata = _read_json(folder / "proof_metadata.json")
    if metadata:
        video, trajectory = folder / "proof.mp4", folder / "trajectory.npz"
        evaluation = folder / "evaluation.json"
        for key, filename in (("dynamics_sha256", "dynamics.json"), ("model_xml_sha256", "model.xml")):
            if metadata.get(key) and (not (folder / filename).is_file() or metadata[key] != _digest(folder / filename)):
                return []
        if metadata.get("result_sha256") and (
                not (folder / "result.json").is_file() or metadata["result_sha256"] != _digest(folder / "result.json")):
            return []
        if metadata.get("evaluation_sha256") and (
                not evaluation.is_file() or metadata["evaluation_sha256"] != _digest(evaluation)):
            return []
        if (video.is_file() and trajectory.is_file()
                and metadata.get("video_sha256") == _digest(video)
                and metadata.get("trajectory_sha256") == _digest(trajectory)):
            attached.append({**_artifact(video, node_id), "kind": "video", "mimeType": "video/mp4",
                             "label": "Debug video · exact recorded trajectory"})
            attached.append(_artifact(folder / "proof_metadata.json", node_id))
    else:
        for name in ("proof_validation.json", "proof_smoke_validation.json"):
            proof = _read_json(folder / name)
            if not proof or proof.get("lineage") != evidence.get("lineage"):
                continue
            if _checkpoint_sha(evidence) != _checkpoint_sha(proof):
                continue
            if evidence.get("run_identity") and proof.get("run_identity") != evidence["run_identity"]:
                continue
            inspection = proof.get("video_inspection") or {}
            video = folder / ("proof_smoke.mp4" if "smoke" in name else "proof.mp4")
            if video.is_file() and inspection.get("sha256") == _digest(video):
                attached.append({**_artifact(video, node_id), "kind": "video", "mimeType": "video/mp4", "label": "Recorded proof / diagnostic"})
    return attached


def _backend_evolution(output_root: Path) -> tuple[list[dict], dict | None]:
    runtime = output_root / "tk2-backend-20260918/source/algorithms/urdf_learn_wasd_walk/outputs"
    progress = _read_json(runtime / "backend_progress.json")
    if not progress:
        return [], None
    root_id = "backend:ragdoll_walk_first_20260918"
    nodes = [{"id": root_id, "parentIds": [], "step": 0, "status": "completed", "kind": "experiment",
              "lineage": "ragdoll_walk_first_20260918", "label": "MuJoCo walking development",
              "approach": "Assisted teacher → learned student → zero assistance",
              "result": "Separate backend experiment; canonical Isaac milestones are unchanged.",
              "metrics": {}, "artifacts": [_artifact(runtime / "backend_progress.json", root_id)]}]
    base = runtime / "mujoco/ragdoll/ragdoll_walk_first_20260918"
    runs = []
    for path in sorted(base.glob("*/result.json")):
        record = _read_json(path)
        if record and record.get("lineage") == "ragdoll_walk_first_20260918":
            record.pop("samples", None)
            runs.append((path, record))
    # Only retain compact metadata in the GUI; dense per-step traces remain in result files.
    checkpoint_ids = {}
    for path in sorted(base.glob("*/training.json")):
        record = _read_json(path)
        if not record or not record.get("checkpoint_sha256"):
            continue
        node_id = "backend-training:" + path.parent.name
        checkpoint_ids[record["checkpoint_sha256"]] = node_id
        nodes.append({"id": node_id, "parentIds": [root_id], "kind": "checkpoint", "step": len(nodes),
                      "lineage": "ragdoll_walk_first_20260918", "status": "completed",
                      "label": _run_label(path.parent.name), "approach": "Supervised student training",
                      "result": "Training finished; physical evaluation determines usability.",
                      "checkpointSha256": record["checkpoint_sha256"], "metrics": {},
                      "startedAt": record.get("created_at", ""), "artifacts": [_artifact(path, node_id)]})
    for path, record in runs:
        metrics, cfg = record.get("metrics", {}), record.get("config", {})
        node_id = "backend-run:" + path.parent.name
        parent = checkpoint_ids.get(record.get("student_checkpoint_sha256"), root_id)
        fall = bool(metrics.get("fall"))
        duration, travel = float(metrics.get("duration_s", 0)), float(metrics.get("forward_m", 0))
        guidance = record.get("total_teacher_guidance_coefficient", record.get("teacher_blend_coefficient", cfg.get("coefficient")))
        stand = cfg.get("stride") == 0
        failed_stand = stand and abs(travel) > .03
        result = f"{duration:g} s · {travel:.3f} m · " + ("fell" if fall else "no fall")
        result += f" · teacher {guidance or 0:g}, support {record.get('balance_assistance_coefficient', 0):g}."
        if failed_stand:
            result += " Zero-command check failed: continued stepping / over 3 cm drift."
        result += " Development trial; no milestone promotion."
        media = _proof_artifacts(path.parent, node_id, record)
        nodes.append({"id": node_id, "parentIds": [parent], "step": len(nodes), "kind": "experiment",
                      "lineage": record["lineage"], "label": _run_label(path.parent.name),
                      "status": "failed" if fall or failed_stand else "completed", "approach": "Zero-assistance student" if guidance == 0 else "Teacher / blended student diagnostic",
                      "result": result, "startedAt": record.get("created_at", ""),
                      "checkpointSha256": record.get("student_checkpoint_sha256"),
                      "metrics": {"duration_s": duration, "semantic_forward_displacement_m": travel,
                                  "speed_mps": travel / duration if duration else 0,
                                  "left_foot_liftoff_count": metrics.get("liftoffs", {}).get("left", 0),
                                  "right_foot_liftoff_count": metrics.get("liftoffs", {}).get("right", 0),
                                  "reset_count": metrics.get("reset_count", 0), "fall_count": int(fall)},
                      "meshTreeSha256": record.get("audit", {}).get("source", {}).get("mesh_tree_sha256"),
                      "artifacts": media + [_artifact(path, node_id)],
                      "evidenceNote": "No video recorded for this attempt; the numeric trace is retained." if not media else None})
    for path in sorted(path for pattern in ("resume_*/dynamics.json", "rsl_transfer_*/dynamics.json", "matched9_cumulative*/dynamics.json")
                       for path in (runtime / "mujoco").glob(pattern)):
        record = _read_json(path)
        if not record:
            continue
        node_id = "backend-validation:" + path.parent.name
        metrics = record.get("metrics", {})
        walking = bool(record.get("config", {}).get("forward"))
        artifacts = _proof_artifacts(path.parent, node_id, record) + [_artifact(path, node_id)]
        variant = path.parent / "mass_variant.json"
        if variant.is_file():
            artifacts.append(_artifact(variant, node_id))
        transfer = _read_json(path.parent / "transfer.json")
        if transfer:
            artifacts.append(_artifact(path.parent / "transfer.json", node_id))
        review = _read_json(path.parent / "visual_review.json") or {}
        if review and (path.parent / "proof.mp4").is_file() and review.get("video_sha256") == _digest(path.parent / "proof.mp4"):
            artifacts.append(_artifact(path.parent / "visual_review.json", node_id))
        else:
            review = {}
        stepping_reference = (review.get("reference_decision") == "stepping_reference_only"
                              and review.get("dynamics_sha256") == _digest(path)
                              and (path.parent / "foot_trace.json").is_file()
                              and review.get("foot_trace_sha256") == _digest(path.parent / "foot_trace.json")
                              and review.get("checkpoint_sha256") == record.get("config", {}).get("checkpoint_sha256"))
        observed_summary=f"{metrics.get('duration_s',0):g} s · drift {metrics.get('horizontal_drift_m',0):.4f} m. "
        failure_summary='; '.join(record.get('failures',[]))
        if record.get('config',{}).get('turn'):
            observed_summary=(f"{metrics.get('duration_s',0):g} s · {metrics.get('final_heading_rad',0)*180/3.141592653589793:.1f}° turned · "
                              f"{metrics.get('hold_max_drift_m',0)*1000:.1f} mm drift while holding. ")
            failure_summary=failure_summary.replace('hold_max_heading_error_rad exceeded 0.0872665','Hold angle was outside the 90° ±5° target')
        if record.get('config',{}).get('teleop'):
            responses=metrics.get('teleop_response',{});blocks=responses.get('blocks',[])
            metrics={**metrics,'teleop_blocks_completed':sum(bool(b.get('complete')) for b in blocks),
                'teleop_left_turn_rad':next((b['heading_change_rad'] for b in blocks if b['label']=='left'),None),
                'teleop_right_turn_rad':next((b['heading_change_rad'] for b in blocks if b['label']=='right'),None)}
            observed_summary=f"{metrics.get('duration_s',0):.2f} / 60 s · {metrics['teleop_blocks_completed']} of 5 command blocks completed. "
            duration=metrics.get('duration_s',0)
            phase=next((label for end,label in [(6,'forward walking'),(18,'the left turn'),(20,'braking'),(25,'the first hold'),(32,'restart'),(44,'the right turn'),(48,'straight walking after turning'),(50,'the final braking transition'),(61,'the final hold')] if duration<end),'the command test')
            failure_summary=f'Fell during {phase}. The remaining command sequence was not completed.' if metrics.get('fall_count') else '; '.join(record.get('failures',[])[:3])
        nodes.append({"id": node_id, "parentIds": [root_id], "step": len(nodes), "kind": "experiment",
                      "lineage": "mass_distribution_20260922", "label": _run_label(path.parent.name),
                      "developmentReference": stepping_reference,
                      "status": "failed" if record.get("failures") or review.get("reference_decision") == "rejected_for_reference" else "completed", "approach": "Scripted actuator diagnostic · not a policy" if record.get("controller_kind") == "scripted_joint_targets" else "Learned periodic feedback · checkpoint validation" if record.get("gait_source_sha256") else "Transferred RSL PPO · walking validation" if walking else "Transferred RSL PPO · standing validation" if path.parent.name.startswith("rsl_transfer_") else "Mass redistribution · standing validation",
                      "result": ("Initialization checkpoint; see policy composition. " if (transfer or {}).get("initial_untrained") else "") + observed_summary +
                                (failure_summary if record.get("failures") else "Dynamics passed; certification and visual review are required.") + (" " + review["observations"] if review else ""),
                      "metrics": {k: v for k, v in metrics.items() if k in ("duration_s", "horizontal_drift_m", "fall_count", "reset_count", "minimum_support_polygon_margin_m", "semantic_forward_displacement_m", "left_completed_swings", "right_completed_swings", "mean_contact_foot_slip_mps", "final_heading_rad", "hold_max_heading_error_rad", "hold_max_drift_m", "hold_duration_s", "teleop_blocks_completed", "teleop_left_turn_rad", "teleop_right_turn_rad")},
                      "startedAt": record.get("created_at", ""), "artifacts": artifacts})
    for path in sorted(path for pattern in ("resume_*/training.json", "rsl_transfer_*/training.json")
                       for path in (runtime / "mujoco/training").glob(pattern)):
        record = _read_json(path)
        if not record:
            continue
        node_id = "backend-training:" + path.parent.name
        checkpoint_ids.update({sha: node_id for sha in record.get("checkpoints", {}).values()})
        nodes.append({"id": node_id, "parentIds": [root_id], "step": len(nodes), "kind": "checkpoint",
                      "lineage": "mass_distribution_20260922", "label": _run_label(path.parent.name),
                      "status": "failed" if (record.get("status") == "interrupted_after_checkpoint" or str(record.get("status", "")).startswith("invalidated_")) else "completed", "approach": "Bounded gait parameter grid · four-start trials" if record.get("arguments", {}).get("search_mode") == "stride_sweep" else "Learned gait and feedback parameters · CEM" if record.get("policy_family") == "periodic_feedback_cem" else "Fresh walking PPO · frozen standing branch" if record.get("arguments", {}).get("initialization") == "fresh_moving" else "Transferred RSL PPO · command-conditioned walking" if record.get("observation_dim") == 70 else "Transferred RSL PPO · normalized standing" if record.get("transfer") else "Mass redistribution · proprioceptive PPO standing",
                      "result": record.get("invalidation_reason", record.get("stop_reason", "Invalidated training evidence")) if str(record.get("status", "")).startswith("invalidated_") else f"Stopped after {record.get('completed_updates')} updates: {record.get('stop_reason')}" if record.get("status") == "interrupted_after_checkpoint" else "Bounded controller training completed; exact checkpoint validation is required.",
                      "startedAt": record.get("created_at") or datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
                      "checkpointSha256": record.get("checkpoint_sha256"), "checkpointPath": record.get("checkpoint"),
                      "parentCheckpointSha256": record.get("parent_checkpoint_sha256"),
                      "metrics": {k: v for k, v in record.get("metrics", {}).items() if k in ("iteration", "recent_30s_success_rate", "approximate_kl", "generation", "forward_m", "survival_s", "completed_swings", "peak_clearance_m", "eligible", "eligible_candidate_count", "max_joint_speed_rad_s", "peak_support_body_weight_ratio")},
                      "artifacts": [_artifact(path, node_id)]})
    for path in sorted((runtime / "mujoco/training").glob("rsl_transfer_*/metadata.json")):
        if (path.parent / "training.json").exists():
            continue
        record = _read_json(path)
        if not record:
            continue
        matches = [job for item in (runtime / "backend_gpu/results").glob("landau-20260922-*.json")
                   if (job := _read_json(item)) and path.parent.name in job.get("command", [])]
        job = max(matches, key=lambda item: item.get("started_at", ""), default={})
        node_id = "backend-training:" + path.parent.name
        status = "running" if job.get("state") == "running" else "failed"
        metrics = {}
        iterations = path.parent / "iterations.jsonl"
        if record.get("policy_family") == "periodic_feedback_cem":
            generations = path.parent / "generations.jsonl"
            if generations.is_file():
                try:
                    last = json.loads(generations.read_text().splitlines()[-1])
                    metrics = {k: last[k] for k in ("generation", "forward_m", "survival_s", "completed_swings", "peak_clearance_m", "wall_s") if k in last}
                except (IndexError, json.JSONDecodeError):
                    pass
        if iterations.is_file():
            try:
                last = json.loads(iterations.read_text().splitlines()[-1])
                metrics = {k: last[k] for k in ("forward_mps", "fall_fraction", "stand_drift_m", "slip_mps", "wall_s") if k in last}
                metrics["iteration"] = last["control_steps"] // record["runner_config"]["num_steps_per_env"]
            except (IndexError, KeyError, json.JSONDecodeError):
                pass
        nodes.append({"id": node_id, "parentIds": ["backend-validation:resume_20260922_balanced_passive"],
                      "step": len(nodes), "kind": "experiment", "lineage": "mass_distribution_20260922",
                      "label": _run_label(path.parent.name), "status": status,
                      "parentCheckpointSha256": record.get("parent_checkpoint_sha256"),
                      "approach": "Bounded gait parameter grid · four-start trials" if record.get("arguments", {}).get("search_mode") == "stride_sweep" else "Learned gait and feedback parameters · CEM" if record.get("policy_family") == "periodic_feedback_cem" else "Fresh walking PPO · frozen standing branch" if record.get("arguments", {}).get("initialization") == "fresh_moving" else "Transferred RSL PPO · command-conditioned walking" if record.get("observation_dim") == 70 else "Transferred RSL PPO · normalized standing",
                      "result": "Training in progress; validation pending." if status == "running" else "Execution stopped before completed training; see worker result.",
                      "startedAt": record.get("created_at", ""), "metrics": metrics,
                      "artifacts": [_artifact(path, node_id)]})
    control_root = runtime / "backend"
    references = _read_json(control_root / "g1_fresh_20260922_reference.json") or {}
    reference_results = {item.get("evaluation"): item.get("evaluation_sha256")
                         for item in references.get("references", [])}
    pretrained = control_root / "g1_official_seed42"
    prior = _read_json(pretrained / "result.json")
    review = _read_json(pretrained / "visual_review.json")
    if prior:
        node_id = "g1:pretrained"
        artifacts = [_artifact(pretrained / "result.json", node_id)]
        video = pretrained / "proof.mp4"
        if video.is_file() and review and review.get("video_sha256") == _digest(video):
            artifacts.insert(0, {**_artifact(video, node_id), "kind": "video", "mimeType": "video/mp4", "label": "Bundled pretrained G1 playback"})
        nodes.append({"id": node_id, "parentIds": [], "step": len(nodes), "kind": "experiment", "model": "Unitree G1",
                      "lineage": "unitree_g1_pretrained_control", "label": "G1 · pretrained playback",
                      "status": "completed", "approach": "Bundled official ONNX policy",
                      "result": "Pretrained positive control. This is not a checkpoint trained by us.",
                      "startedAt": "2026-09-18T00:00:00Z", "metrics": {"duration_s": prior.get("duration_s"), "semantic_forward_displacement_m": prior.get("displacement_m", [0])[0]},
                      "artifacts": artifacts})
    for folder in sorted(control_root.glob("g1_fresh_20260922*")):
        config = _read_json(folder / "benchmark_config.json")
        evaluation = _read_json(folder / "evaluation.json")
        if evaluation:
            node_id = "g1-evaluation:" + folder.name
            metric = evaluation.get("metrics", {})
            checkpoint = Path(evaluation["checkpoint"])
            # Checkpoint is below training-run/logs/rsl_rl/experiment/run/model.pt.
            parent_folder = next((parent for parent in checkpoint.parents if (parent / "benchmark_config.json").is_file()), None)
            nodes.append({"id": node_id, "parentIds": ["g1-training:" + parent_folder.name] if parent_folder else [], "step": len(nodes),
                          "kind": "experiment", "model": "Unitree G1", "lineage": "unitree_g1_fresh_20260922",
                          "label": f"G1 · {checkpoint.stem} · " + ("heading hold" if evaluation.get("evaluation_overrides", {}).get("heading_hold") else "fixed yaw"), "status": "completed" if evaluation.get("status") == "passed" else "failed",
                          "approach": "Our fresh checkpoint · 0.5 m/s · " + ("official heading feedback" if evaluation.get("evaluation_overrides", {}).get("heading_hold") else "fixed zero yaw") + " · stop on first done",
                          "result": f"{metric.get('duration_s', 0):g}s · {metric.get('forward_m', 0):.2f}m · resets {metric.get('reset_count', 0)}. " + evaluation.get("scope", ""),
                          "startedAt": evaluation.get("created_at"), "checkpointSha256": evaluation.get("checkpoint_sha256"),
                          "developmentReference": reference_results.get(str((folder / "evaluation.json").resolve())) == _digest(folder / "evaluation.json"),
                          "metrics": {**{k:v for k,v in metric.items() if isinstance(v,(int,float))}, "semantic_forward_displacement_m": metric.get("forward_m"),
                                      **dict(zip(("left_completed_swings", "right_completed_swings"), metric.get("completed_swings", [])))},
                          "artifacts": _proof_artifacts(folder, node_id, evaluation) + [_artifact(folder / "evaluation.json", node_id)]})
        elif (acceptance := _read_json(folder / "independent_acceptance.json")):
            record = _read_json(folder / "result.json") or {}
            node_id = "g1-native:" + folder.name
            checkpoint = Path(acceptance["checkpoint"])
            parent_folder = next((parent for parent in checkpoint.parents if (parent / "benchmark_config.json").is_file()), None)
            nodes.append({"id": node_id, "parentIds": ["g1-training:" + parent_folder.name] if parent_folder else [],
                          "step": len(nodes), "kind": "experiment", "model": "Unitree G1",
                          "lineage": "unitree_g1_fresh_20260922", "label": f"G1 · {checkpoint.stem} · native CPU",
                          "status": "completed" if acceptance["status"] == "passed" else "failed",
                          "approach": "Our fresh checkpoint · independent deployment physics · fixed zero yaw",
                          "result": f"{record.get('duration_s', 0):g}s · {record.get('displacement_m', [0])[0]:.2f}m forward. " +
                                    "Independent deployment check; no bundled pretrained weights loaded.",
                          "startedAt": datetime.fromtimestamp((folder / "independent_acceptance.json").stat().st_mtime, timezone.utc).isoformat(),
                          "checkpointSha256": acceptance["checkpoint_sha256"],
                          "metrics": {"duration_s": record.get("duration_s"),
                                      "semantic_forward_displacement_m": record.get("displacement_m", [0])[0],
                                      "lateral_m": record.get("displacement_m", [0, 0])[1],
                                      **dict(zip(("left_completed_swings", "right_completed_swings"), acceptance.get("completed_swings_with_15mm_clearance", [])))},
                          "artifacts": _proof_artifacts(folder, node_id, record) + [_artifact(folder / "independent_acceptance.json", node_id), _artifact(folder / "result.json", node_id)]})
        elif config:
            record = _read_json(folder / "result.json")
            node_id = "g1-training:" + folder.name
            metrics = {"iteration": (record or {}).get("completed_iterations", 0)}
            if record is None and (folder / "iterations.jsonl").is_file():
                lines = (folder / "iterations.jsonl").read_text().splitlines()
                try:
                    first, last = json.loads(lines[0]), json.loads(lines[-1])
                    metrics.update(iteration=last["iteration"] - first["iteration"] + 1,
                                   checkpoint_iteration=last["iteration"],
                                   control_transitions_per_s=last["control_transitions_per_s_total"])
                except (IndexError, KeyError, json.JSONDecodeError):
                    pass  # The worker may be appending its latest line.
            if record and record.get("control_transitions_per_s_total_steady"):
                metrics.update(training_wall_s=record["wall_s"], control_transitions_per_s=record["control_transitions_per_s_total_steady"])
            parent = config.get("parent") or {}
            parent_ids = ["g1-training:" + Path(parent["training_directory"]).name] if parent.get("training_directory") else []
            nodes.append({"id": node_id, "parentIds": parent_ids, "step": len(nodes), "kind": "checkpoint", "model": "Unitree G1",
                          "lineage": "unitree_g1_fresh_20260922", "label": "G1 · continued PPO training" if parent else "G1 · fresh PPO training",
                          "status": "running" if record is None else "completed" if record.get("returncode") == 0 else "failed",
                          "approach": ("Yaw precision comparison" if config.get("recipe") == "precise_yaw" else "Pinned official PPO recipe") + (" · resume our own checkpoint" if parent else " · fresh random weights"),
                          "result": f"{metrics['iteration']} / {config['iterations']} iterations. No pretrained policy loaded; checkpoint playback determines walking performance.",
                          "startedAt": datetime.fromtimestamp((folder / "benchmark_config.json").stat().st_mtime, timezone.utc).isoformat(),
                          "metrics": metrics, "artifacts": [_artifact(folder / "benchmark_config.json", node_id)] + ([_artifact(folder / "result.json", node_id)] if record else [])})
    checkpoint_ids.update({node["checkpointSha256"]: node["id"] for node in nodes if node.get("checkpointSha256") and node.get("kind") == "checkpoint"})
    for node in nodes:
        if node["id"].startswith("backend-validation:"):
            result_path = next((Path(REPO_ROOT / item["path"]) for item in node["artifacts"] if item["path"].endswith("dynamics.json")), None)
            record = _read_json(result_path) if result_path else None
            sha = (record or {}).get("config", {}).get("checkpoint_sha256")
            if sha in checkpoint_ids:
                node["parentIds"] = [checkpoint_ids[sha]]
                node["checkpointSha256"] = sha
        if node["id"].startswith(("backend-training:resume_", "backend-training:rsl_transfer_")):
            node["parentIds"] = [checkpoint_ids.get(node.get("parentCheckpointSha256"), "backend-validation:resume_20260922_balanced_passive")]
    stepping_references = [node for node in nodes if node["id"].startswith("backend-validation:") and node.get("developmentReference")]
    if stepping_references:
        latest_reference = max(stepping_references, key=lambda node: node.get("startedAt") or "")
        for node in stepping_references:
            node["developmentReference"] = node is latest_reference
    current = max(nodes[1:], key=lambda node: node.get("startedAt") or "", default=nodes[0])
    jobs = [_read_json(path) for pattern in ("resume-20260922-*.json", "g1-20260922-*.json", "landau-20260922-*.json")
            for path in (runtime / "backend_gpu/results").glob(pattern)]
    job = max((item for item in jobs if item), key=lambda item: item.get("started_at", ""), default={})
    return nodes, {"currentNodeId": current["id"], "next_step": progress.get("next_step"), "updatedAt": progress.get("updated_at"),
                   "sessionState": progress.get("session_state"),
                   "job": {key: job.get(key) for key in ("job_id", "state", "exit_code", "wall_s", "started_at")}}


MILESTONE_LABELS = {
    "stand_zero_signal_30s_no_reset": "Passive stand · 30 s",
    "stand_30s_no_reset": "Policy stand · 30 s",
    "gate_5m_no_reset": "Forward gate · 5 m",
    "gate_10m_no_reset": "Forward gate · 10 m",
    "yaw_turn_90deg_hold": "Turn 90° and hold",
    "teleop_60s_forward_turn": "Teleop · 60 s",
    "gate_10m_four_directions_no_reset": "Four directions · 10 m",
    "triangle_path_follow_no_reset": "Triangle path",
    "square_path_follow_no_reset": "Square path",
    "terrain_5m_no_reset": "Rough terrain · 5 m",
    "obstacle_stop_before_collision": "Obstacle braking",
    "game_10m_no_reset": "Mixed game gate · 10 m",
}

def _milestone_summaries(ledger: dict) -> list[dict]:
    summaries = []
    for record in ledger.get("milestones", []):
        checkpoint = record.get("checkpoint", {})
        checkpoint = checkpoint if isinstance(checkpoint, dict) else {}
        status = str(record.get("status", "not_started"))
        metrics = record.get("metrics", {}) if status == "passed" else {}
        if status == "passed":
            result = "Gate passed with recorded evidence"
        elif status == "in_progress":
            result = "Current hard gate"
        else:
            result = "Blocked by the preceding gate"
        summaries.append({
            "order": record.get("order"),
            "id": record.get("id"),
            "label": MILESTONE_LABELS.get(str(record.get("id")), str(record.get("id"))),
            "status": status,
            "result": result,
            "metrics": metrics,
            "checkpointPath": checkpoint.get("path") or checkpoint.get("identity"),
            "diskBytes": checkpoint.get("size_bytes"),
        })
    return summaries


def _backend_nodes(output_root: Path) -> list[dict]:
    """Expose simulator development evidence without promoting canonical gates."""
    progress_path = output_root / "backend_progress.json"
    progress = _read_json(progress_path)
    if not progress:
        return []
    latest = progress.get("latest_fully_unassisted") or {}
    lineage = "tk2_mujoco_development"
    root_id = "backend:tk2-mujoco"
    nodes = [{
        "id": root_id, "parentIds": [], "label": "TK2 MuJoCo · development",
        "kind": "root", "status": progress.get("status", "unknown"),
        "lineage": lineage, "step": 10000, "important": True,
        "startedAt": progress.get("updated_at"),
        "approach": progress.get("variant"),
        "changeSummary": "Synced TK2 ragdoll and student experiments; separate simulator evidence",
        "result": "Separate backend diagnostic; does not certify canonical gates",
        "metrics": {}, "artifacts": [_artifact(progress_path, root_id)],
        "experimentParameters": {"simulator": progress.get("simulator"),
                                 "backend status": progress.get("status")},
    }]
    for index, run in enumerate(latest.get("runs", [])):
        metrics = run.get("metrics") or {}
        lifts = metrics.get("liftoffs") or {}
        node_id = f"backend:unassisted-repeat-{index + 1}"
        nodes.append({
            "id": node_id, "parentIds": [root_id],
            "label": f"Unassisted repeat {index + 1} · development only",
            "kind": "experiment", "status": "completed", "lineage": lineage,
            "step": 10001 + index, "important": True,
            "startedAt": progress.get("updated_at"), "gateEligible": False,
            "approach": progress.get("variant"),
            "result": f"{metrics.get('forward_m', 0):.3f} m / {metrics.get('duration_s', 0):g} s; "
                      + str(latest.get("proof_status", "Visual review and gate validation pending")),
            "changeSummary": "All auxiliary assistance off; normal motor PD retained; not a gate pass",
            "metrics": {"semantic_forward_displacement_m": metrics.get("forward_m"),
                        "left_foot_liftoff_count": lifts.get("left"),
                        "right_foot_liftoff_count": lifts.get("right"),
                        "reset_count": metrics.get("reset_count"), "done_count": metrics.get("done_count"),
                        "fall_count": int(bool(metrics.get("fall")))},
            "trainingProgress": {"kind": "diagnostic", "durationSeconds": metrics.get("duration_s")},
            "experimentParameters": {"assistance coefficient": latest.get("assistance_coefficient"),
                                     "teacher blend": latest.get("teacher_blend"),
                                     "reference forcing": latest.get("reference_forcing")},
            "checkpointPath": progress.get("checkpoint"),
            "checkpointSha256": run.get("checkpoint_sha256"),
            "sourceResultPath": run.get("result"), "configSha256": run.get("config_sha256"),
            "artifacts": [_artifact(progress_path, node_id)],
        })
    # Synced summaries have no per-run proof hash; leave media unverified.
    return nodes


def _training_parameters(contract: dict) -> dict:
    environment = contract.get("environment", {}) if isinstance(contract, dict) else {}
    ppo = contract.get("ppo", {}) if isinstance(contract, dict) else {}
    initialization = contract.get("initialization", {}) if isinstance(contract, dict) else {}
    candidates = {
        "training method": contract.get("training_method"),
        "iterations": contract.get("iterations"),
        "environments": contract.get("num_envs"),
        "rollout steps / env": contract.get("num_steps_per_env"),
        "samples": contract.get("sample_count"),
        "action scale (rad)": environment.get("action_scale_rad"),
        "standing mix": environment.get("standing_environment_fraction"),
        "gait period (s)": environment.get("gait_phase_period_s"),
        "learning rate": ppo.get("learning_rate"),
        "initial noise std": ppo.get("initial_action_noise_std"),
        "initialization": initialization.get("kind"),
    }
    return {key: value for key, value in candidates.items() if value is not None}


def _training_progress(training: dict) -> dict:
    contract = training.get("requested_contract", {})
    checkpoint = training.get("checkpoint", {})
    learning_iteration = checkpoint.get("learning_iteration") if isinstance(checkpoint, dict) else None
    completed_iterations = learning_iteration + 1 if isinstance(learning_iteration, int) else None
    return {
        "kind": "training",
        "completedIterations": completed_iterations,
        "requestedIterations": contract.get("iterations"),
        "rolloutStepsPerEnv": contract.get("num_steps_per_env"),
        "numEnvs": contract.get("num_envs"),
        "sampleCount": contract.get("sample_count"),
    }


def _parameter_changes(nodes: list[dict]) -> None:
    by_id = {node["id"]: node for node in nodes}
    for node in nodes:
        parameters = node.get("experimentParameters", {})
        parent = by_id.get((node.get("parentIds") or [None])[0])
        parent_parameters = parent.get("experimentParameters", {}) if parent else {}
        changes = []
        for key, value in parameters.items():
            if key in parent_parameters and parent_parameters[key] == value:
                continue
            changes.append({"key": key, "from": parent_parameters.get(key), "to": value})
        node["parameterChanges"] = changes
        if changes and not node.get("changeSummary"):
            node["changeSummary"] = "; ".join(
                f"{item['key']}: {item['from'] if item['from'] is not None else '—'} → {item['to']}"
                for item in changes[:3]
            )


def build_evolution(
    output_root: Path = OUTPUT_ROOT,
    milestones_path: Path = MILESTONES_PATH,
) -> dict:
    ledger = _read_json(milestones_path)
    if ledger is None:
        raise ValueError(f"Milestone ledger is unavailable: {milestones_path}")
    lineage = str(ledger.get("lineage", "unknown"))
    invalidated_entries = [
        item for item in ledger.get("invalidatedLineages", [])
        if isinstance(item, dict) and item.get("lineage")
    ]
    legacy_invalidated = ledger.get("invalidatedLineage", {})
    if isinstance(legacy_invalidated, dict) and legacy_invalidated.get("lineage"):
        if not any(item.get("lineage") == legacy_invalidated.get("lineage") for item in invalidated_entries):
            invalidated_entries.append(legacy_invalidated)
    invalidated_by_lineage = {str(item["lineage"]): item for item in invalidated_entries}
    accepted_lineages = {lineage, *invalidated_by_lineage}
    milestone_records = {item["id"]: item for item in ledger.get("milestones", [])}
    passive = milestone_records.get("stand_zero_signal_30s_no_reset", {})
    nodes = []
    invalidated_root_ids = {}
    previous_invalidated_root_id = None
    for invalidated_index, invalidated in enumerate(invalidated_entries):
        invalidated_lineage = str(invalidated["lineage"])
        invalidated_root_id = f"invalidated:{invalidated_lineage}:stand_zero_signal_30s_no_reset"
        invalidated_root_ids[invalidated_lineage] = invalidated_root_id
        nodes.append({
            "id": invalidated_root_id,
            "parentIds": [previous_invalidated_root_id] if previous_invalidated_root_id else [],
            "label": str(invalidated.get("label") or f"Invalidated mesh · {invalidated_index + 1}"),
            "step": invalidated_index - len(invalidated_entries),
            "status": "failed",
            "kind": "root",
            "lineage": invalidated_lineage,
            "approach": "superseded visual/collision mesh package",
            "result": str(invalidated.get("reason", "asset lineage was invalidated")),
            "metrics": {},
            "important": True,
            "meshTreeSha256": invalidated.get("meshTreeSha256"),
        })
        previous_invalidated_root_id = invalidated_root_id
    passive_status = str(passive.get("status", "not_started"))
    passive_checkpoint = passive.get("checkpoint")
    passive_checkpoint_path = (
        passive_checkpoint.get("path") or passive_checkpoint.get("identity")
        if isinstance(passive_checkpoint, dict)
        else passive_checkpoint
    )
    nodes.append({
        "id": "milestone:stand_zero_signal_30s_no_reset",
        "parentIds": [previous_invalidated_root_id] if previous_invalidated_root_id else [],
        "label": "Rabbit-ear mesh · passive stand 30 s",
        "step": 0,
        "status": "completed" if passive_status == "passed" else "running" if passive_status == "in_progress" else "failed",
        "kind": "root",
        "milestoneId": "stand_zero_signal_30s_no_reset",
        "lineage": lineage,
        "approach": "URDF equilibrium pose + PD control",
        "result": "canonical zero-signal stand gate passed" if passive_status == "passed" else "latest visual/collision mesh awaits gate re-certification",
        "metrics": passive.get("metrics", {}),
        "important": True,
        "checkpointPath": passive_checkpoint_path if passive_status == "passed" else None,
        "meshTreeSha256": ledger.get("assetContract", {}).get("meshTreeSha256"),
    })
    policy_record = milestone_records.get("stand_30s_no_reset", {})
    if policy_record.get("status") == "in_progress":
        nodes.append({
            "id": "milestone:stand_30s_no_reset",
            "parentIds": ["milestone:stand_zero_signal_30s_no_reset"],
            "label": "Rabbit-ear mesh · policy stand 30 s",
            "step": 1,
            "status": "running",
            "kind": "milestone",
            "milestoneId": "stand_30s_no_reset",
            "lineage": lineage,
            "approach": "manager-based proprioceptive PPO",
            "result": "awaiting a fresh checkpoint on the corrected mesh package",
            "metrics": {},
            "important": True,
            "meshTreeSha256": ledger.get("assetContract", {}).get("meshTreeSha256"),
        })
    forward_record = milestone_records.get("gate_5m_no_reset", {})
    if forward_record.get("status") == "in_progress":
        nodes.append({
            "id": "milestone:gate_5m_no_reset",
            "parentIds": ["milestone:stand_30s_no_reset"],
            "label": "Rabbit-ear mesh · forward gate 5 m",
            "step": 2,
            "status": "running",
            "kind": "milestone",
            "milestoneId": "gate_5m_no_reset",
            "lineage": lineage,
            "approach": "flat +Y manager-based PPO",
            "result": "awaiting a fresh walking checkpoint on the corrected mesh package",
            "metrics": {},
            "important": True,
            "meshTreeSha256": ledger.get("assetContract", {}).get("meshTreeSha256"),
        })
    checkpoint_nodes: dict[str, str] = {}
    runs: list[tuple[Path, dict, Path | None, dict | None]] = []
    for training_path in sorted(output_root.rglob("training.json")) if output_root.exists() else []:
        training = _read_json(training_path)
        if training is None or training.get("lineage") not in accepted_lineages:
            continue
        validation_path, validation = _validation_for(training_path)
        runs.append((training_path, training, validation_path, validation))
        sha = _checkpoint_sha(training)
        if sha:
            checkpoint_nodes[sha] = f"run:{training.get('run_identity') or sha[:16]}"

    policy_checkpoint = policy_record.get("checkpoint", {})
    if isinstance(policy_checkpoint, dict) and policy_checkpoint.get("sha256"):
        checkpoint_nodes[str(policy_checkpoint["sha256"])] = "milestone:stand_30s_no_reset"

    runs.sort(key=lambda item: str(item[1].get("run_identity", item[0])))
    for step, (training_path, training, validation_path, validation) in enumerate(runs, start=1):
        checkpoint = training.get("checkpoint", {})
        run_lineage = str(training.get("lineage", "unknown"))
        is_invalidated = run_lineage in invalidated_by_lineage
        sha = _checkpoint_sha(training)
        node_id = checkpoint_nodes.get(sha or "", f"run:{training.get('run_identity', step)}")
        milestone = str(training.get("milestone", "unknown"))
        canonical = milestone_records.get(milestone, {})
        if milestone == "stand_30s_no_reset" and canonical.get("status") == "passed":
            node_id = "milestone:stand_30s_no_reset"
            if sha:
                checkpoint_nodes[sha] = node_id
        parent_sha = _parent_checkpoint_sha(training)
        parent_id = checkpoint_nodes.get(parent_sha or "")
        if not parent_id:
            if is_invalidated:
                parent_id = invalidated_root_ids[run_lineage]
            else:
                parent_id = "milestone:stand_zero_signal_30s_no_reset" if milestone == "stand_30s_no_reset" else "milestone:stand_30s_no_reset"
        metrics = _metrics(validation)
        status, result, important = _run_status(training, validation, metrics)
        if is_invalidated:
            status = "failed"
            result = f"invalidated asset lineage; {result}"
            important = True
        if canonical.get("status") == "passed" and isinstance(canonical.get("checkpoint"), dict) and canonical["checkpoint"].get("sha256") == sha:
            status, result, important = "completed", "canonical milestone passed", True
        contract = training.get("requested_contract", {})
        approach = contract.get("training_method") or contract.get("algorithm") or "training run"
        run_name = training_path.parent.name
        label = f"Policy stand · {status}" if milestone == "stand_30s_no_reset" else run_name.replace("_", " ")
        if is_invalidated:
            label = f"Old mesh · {label}"
        artifacts = [_artifact(training_path, node_id)]
        if validation_path is not None:
            artifacts.append(_artifact(validation_path, node_id))
        artifacts.extend(_proof_artifacts(training_path.parent, node_id, training))
        node = {
            "id": node_id,
            "parentIds": [parent_id],
            "label": label,
            "step": step,
            "status": status,
            "kind": "milestone" if canonical.get("status") == "passed" and canonical.get("checkpoint", {}).get("sha256") == sha else "checkpoint",
            "milestoneId": milestone,
            "lineage": run_lineage,
            "approach": approach,
            "result": result,
            "metrics": metrics,
            "checkpointPath": checkpoint.get("path"),
            "diskBytes": checkpoint.get("size_bytes"),
            "checkpointSha256": sha,
            "checkpointStorage": {
                "provider": "Nextcloud",
                "macHydration": "online-only",
                "localPreview": False,
            },
            "startedAt": training.get("run_identity"),
            "completedAt": training.get("completed_at"),
            "sourceRevision": training.get("source_commit"),
            "artifacts": artifacts,
            "important": important,
        }
        nodes = [item for item in nodes if item["id"] != node_id]
        nodes.append(node)

    # A constructor/settling crash precedes training.json. Preserve that failed
    # experiment without inventing a checkpoint, metrics, or a milestone pass.
    for path in sorted(output_root.rglob("*_failure.json")) if output_root.exists() else []:
        failure = _read_json(path)
        if (failure is None or failure.get("lineage") not in accepted_lineages
                or failure.get("milestone") != "stand_30s_no_reset"
                or failure.get("status") != "failed_to_execute"):
            continue
        failure_lineage = failure["lineage"]
        is_invalidated = failure_lineage in invalidated_by_lineage
        node_id = f"failure:{_relative(path)}"
        stage = str(failure.get("runtime_stage", "unknown"))
        error = failure.get("exception", {})
        result = f"{stage}: {error.get('type', 'error')}: {error.get('message', '')}"
        nodes.append({
            "id": node_id,
            "parentIds": [invalidated_root_ids[failure_lineage] if is_invalidated
                          else "milestone:stand_zero_signal_30s_no_reset"],
            "label": "Policy stand · execution failed",
            "step": len(nodes), "status": "failed", "kind": "experiment",
            "milestoneId": "stand_30s_no_reset", "lineage": failure_lineage,
            "approach": failure.get("initialization_protocol", {}).get("method", "policy stand"),
            "result": f"invalidated asset lineage; {result}" if is_invalidated else result,
            "metrics": {}, "startedAt": failure.get("run_identity"),
            "sourceRevision": failure.get("source_commit"),
            "artifacts": [_artifact(path, node_id)], "important": True,
        })

    probes = []
    for probe_path in sorted(output_root.rglob("reference_probe.json")) if output_root.exists() else []:
        probe = _read_json(probe_path)
        if (
            probe is None
            or probe.get("lineage") not in accepted_lineages
            or probe.get("component") != "open_loop_reference_probe"
        ):
            continue
        probes.append((probe_path, probe))
    probes.sort(key=lambda item: str(item[1].get("run_identity", item[0])))
    for step, (probe_path, probe) in enumerate(probes, start=len(runs) + 1):
        run_identity = str(probe.get("run_identity") or probe_path.parent.name)
        probe_lineage = str(probe.get("lineage", "unknown"))
        is_invalidated = probe_lineage in invalidated_by_lineage
        node_id = f"experiment:{run_identity}"
        parent_checkpoint = probe.get("parent_checkpoint", {})
        parent_sha = parent_checkpoint.get("sha256") if isinstance(parent_checkpoint, dict) else None
        parent_id = checkpoint_nodes.get(str(parent_sha or ""), "milestone:stand_30s_no_reset")
        metrics = _metrics(probe)
        passed = probe.get("status") == "passed" and probe.get("ppo_eligible") is True
        if passed:
            status = "completed"
            result = "open-loop reference passed all PPO-entry checks"
        else:
            status = "failed"
            failures = probe.get("failures") or ["reference probe did not pass"]
            result = "; ".join(map(str, failures[:3]))
        if is_invalidated:
            status = "failed"
            result = f"invalidated asset lineage; {result}"
        parameters = probe.get("reference_contract", {}).get("parameters", {})
        nodes.append({
            "id": node_id,
            "parentIds": [parent_id],
            "label": _run_label(probe_path.parent.name),
            "step": step,
            "status": status,
            "kind": "experiment",
            "milestoneId": "gate_5m_no_reset",
            "lineage": probe_lineage,
            "approach": str(probe.get("experiment", "open-loop reference probe")),
            "result": result,
            "metrics": metrics,
            "checkpointPath": parent_checkpoint.get("path") if isinstance(parent_checkpoint, dict) else None,
            "checkpointSha256": parent_sha,
            "experimentParameters": parameters,
            "startedAt": run_identity,
            "sourceRevision": probe.get("source_commit"),
            "artifacts": [_artifact(probe_path, node_id)] + _proof_artifacts(probe_path.parent, node_id, probe),
            "important": True,
        })

    passive_diagnostics = []
    passive_component_paths = (
        sorted({
            *output_root.rglob("dynamics_smoke_validation.json"),
            *output_root.rglob("dynamics_validation.json"),
        })
        if output_root.exists()
        else []
    )
    for diagnostic_path in passive_component_paths:
        diagnostic = _read_json(diagnostic_path)
        if (
            diagnostic is None
            or diagnostic.get("lineage") not in accepted_lineages
            or diagnostic.get("milestone") != "stand_zero_signal_30s_no_reset"
            or diagnostic.get("component") != "dynamics"
            or diagnostic.get("scope") not in {"diagnostic_experiment", "component_only"}
        ):
            continue
        passive_diagnostics.append((diagnostic_path, diagnostic))
    passive_diagnostics.sort(key=lambda item: str(item[1].get("run_identity", item[0])))
    diagnostic_step = len(runs) + len(probes) + 1
    for diagnostic_path, diagnostic in passive_diagnostics:
        run_identity = str(diagnostic.get("run_identity") or diagnostic_path.parent.name)
        diagnostic_lineage = str(diagnostic.get("lineage", "unknown"))
        is_invalidated = diagnostic_lineage in invalidated_by_lineage
        node_id = f"experiment:{run_identity}"
        raw_metrics = diagnostic.get("metrics", {})
        duration_s = float(raw_metrics.get("duration_s", 0.0))
        is_gate_attempt = bool(diagnostic.get("gate_eligible"))
        status = "completed" if diagnostic.get("status") == "passed" else "failed"
        failures = diagnostic.get("failures") or []
        if status == "completed" and is_gate_attempt:
            result = f"{duration_s:g} s dynamics passed; visual proof and final assembly remain separate"
        elif status == "completed":
            result = f"{duration_s:g} s static-pose diagnostic retained support without a fall or reset"
        else:
            result = "; ".join(map(str, failures[:3])) or "passive dynamics attempt failed"
        if is_invalidated:
            status = "failed"
            result = f"invalidated asset lineage; {result}"
        # Older component-only gate records used JSON null when no isolated
        # experiment was attached.  Treat that as the empty mapping so failed
        # attempts remain visible in the lineage instead of aborting the tree.
        experiment = diagnostic.get("experiment") or {}
        parent_id = (
            invalidated_root_ids[diagnostic_lineage]
            if is_invalidated
            else "milestone:stand_zero_signal_30s_no_reset"
        )
        nodes.append({
            "id": node_id,
            "parentIds": [parent_id],
            "label": (
                f"Passive dynamics gate · {duration_s:g} s"
                if is_gate_attempt
                else f"Static pose probe · {duration_s:g} s"
            ),
            "step": diagnostic_step,
            "status": status,
            "kind": "validation" if is_gate_attempt else "experiment",
            "milestoneId": "stand_zero_signal_30s_no_reset",
            "lineage": diagnostic_lineage,
            "approach": str(
                experiment.get("id")
                or diagnostic.get("checkpoint", {}).get("kind")
                or "static-pose diagnostic"
            ),
            "result": result,
            "metrics": _metrics(diagnostic),
            "experimentParameters": {
                "duration_s": duration_s,
                "physics_steps": raw_metrics.get("physics_steps"),
                "diagnostic_only": not is_gate_attempt,
                "gate_eligible": bool(diagnostic.get("gate_eligible")),
            },
            "trainingProgress": {
                "kind": "diagnostic",
                "physicsSteps": raw_metrics.get("physics_steps"),
                "durationSeconds": duration_s,
            },
            "checkpointPath": diagnostic.get("checkpoint", {}).get("identity"),
            "changeSummary": str(
                experiment.get("independent_variable")
                or diagnostic.get("checkpoint", {}).get("kind")
                or experiment.get("id", "static-pose diagnostic")
            ).replace("_", " "),
            "startedAt": run_identity,
            "artifacts": [_artifact(diagnostic_path, node_id)],
            "important": True,
        })
        diagnostic_step += 1

    # Cross-model controls are independent roots, never Landau gate evidence.
    comparison_root = output_root / "model_comparison"
    comparison_sources = list(comparison_root.glob("*/*/training.json"))
    comparison_sources += list(comparison_root.glob("*/*/checkpoint_import.json"))
    for path in sorted(comparison_sources):
        training = _read_json(path)
        if (training is None or training.get("model") not in {"unitree_g1", "landau_current"}
                or training.get("protocol") != "installed_manager_rsl_rl_comparison_v1"
                or training.get("landau_gate_eligible") is not False):
            continue
        model = training["model"]
        if path.parent.parent.name != model:
            continue
        node_id = f"comparison:{model}:{path.parent.name}"
        evaluation_path = path.parent / "evaluation.json"
        evaluation = _read_json(evaluation_path)
        valid_evaluation = (evaluation and evaluation.get("model") == model
                            and evaluation.get("lineage") == training.get("lineage")
                            and evaluation.get("checkpoint") == training.get("checkpoint")
                            and evaluation.get("asset") == training.get("asset"))
        result = evaluation if valid_evaluation else training
        assets = training.get("asset", {})
        nodes.append({
            "id": node_id, "parentIds": [], "label": f"{model} · M2 control diagnostic",
            "kind": "experiment", "step": len(nodes), "model": model,
            "lineage": training["lineage"], "status": "failed" if result.get("status") == "failed" else "completed",
            "approach": training["task"], "result": result.get("status", "unknown") + "; not Landau gate evidence",
            "metrics": result.get("metrics", {}), "important": True,
            "assetTreeSha256": assets.get("asset_tree_sha256") or assets.get("mesh_tree_sha256"),
            "checkpointSha256": training.get("checkpoint", {}).get("sha256"),
            "artifacts": [_artifact(path, node_id)] + ([_artifact(evaluation_path, node_id)] if valid_evaluation else []),
        })

    for path in sorted(comparison_root.glob("*/*/*_failure.json")):
        failure = _read_json(path)
        if (not failure or failure.get("model") != path.parent.parent.name
                or failure.get("model") not in {"unitree_g1", "landau_current"}
                or failure.get("protocol") != "installed_manager_rsl_rl_comparison_v1"
                or failure.get("landau_gate_eligible") is not False):
            continue
        node_id = f"comparison-failure:{failure['model']}:{path.parent.name}:{path.stem}"
        nodes.append({"id": node_id, "parentIds": [], "label": f"{failure['model']} · execution failure",
                      "kind": "experiment", "step": len(nodes), "model": failure["model"],
                      "lineage": failure["lineage"], "status": "failed", "important": True,
                      "result": str(failure.get("exception", "failed before evidence")),
                      "approach": failure.get("runtime_stage", "initialization"), "metrics": {},
                      "artifacts": [_artifact(path, node_id)]})

    backend_nodes, backend_progress = _backend_evolution(output_root)
    nodes.extend(backend_nodes)
    synced_backend_nodes = _backend_nodes(output_root) if not backend_nodes else []
    nodes.extend(synced_backend_nodes)
    # Active MuJoCo certificates have their own model identity and proof media;
    # do not present them through the historical Isaac milestone placeholders.
    if ledger.get("backend") == "mujoco_warp_cuda":
        nodes = [node for node in nodes if not node["id"].startswith("milestone:")]
        previous = None
        for record in ledger.get("milestones", []):
            if record.get("status") not in {"passed", "in_progress"}:
                continue
            node_id = "milestone:" + record["id"]
            artifacts = []
            for declaration in record.get("evidence", []):
                path = (REPO_ROOT / declaration["path"]).resolve()
                if path.suffix not in {".json", ".mp4", ".png"}:
                    continue
                if not path.is_file() or _digest(path) != declaration.get("sha256"):
                    continue
                artifact = _artifact(path, node_id)
                artifact["sha256"] = declaration["sha256"]
                if path.suffix == ".mp4":
                    artifact.update(kind="video", mimeType="video/mp4")
                elif path.suffix == ".png":
                    artifact.update(kind="image", mimeType="image/png")
                artifacts.append(artifact)
            passed = record["status"] == "passed"
            nodes.append({"id": node_id, "parentIds": [previous] if previous else [],
                          "label": {1:"M1 · passive stand 30 s",2:"M2 · policy stand 30 s",3:"M3 · walk 5 m",4:"M4 · walk 10 m",5:"M5 · turn90° and hold",6:"M6 · joystick commands60s"}.get(record['order'],f"M{record['order']} · {record['id'].replace('_', ' ')}"),
                          "step": record["order"], "status": "completed" if passed else "running",
                          "kind": "milestone", "milestoneId": record["id"], "lineage": lineage,
                          "startedAt": record.get("passedAt"), "important": True,
                          "developmentReference": passed, "artifacts": artifacts,
                          "approach": "MuJoCo Warp · balanced_hands_v1",
                          "result": "Certified with matching proof video" if passed else "Training; gate not yet passed",
                          "metrics": record.get("metrics", {}),
                          "checkpointPath": (record.get("checkpoint") or {}).get("path"),
                          "meshTreeSha256": ledger.get("assetContract", {}).get("meshTreeSha256")})
            previous = node_id
    child_count = {item["id"]: 0 for item in nodes}
    for node in nodes:
        for parent_id in node.get("parentIds", []):
            if parent_id in child_count:
                child_count[parent_id] += 1
    leaves = {node_id for node_id, count in child_count.items() if count == 0}
    mandatory = [
        node["id"] for node in nodes
        if node.get("important") or node["id"] in leaves or child_count.get(node["id"], 0) > 1
    ]
    visible = list(dict.fromkeys(mandatory))
    for node in reversed(nodes):
        if len(visible) >= VISIBLE_NODE_BUDGET:
            break
        if node["id"] not in visible:
            visible.append(node["id"])
    visible_set = set(visible[:VISIBLE_NODE_BUDGET])
    default_visible = [node["id"] for node in nodes if node["id"] in visible_set]
    _parameter_changes(nodes)
    active = next(
        (item.get("id") for item in ledger.get("milestones", []) if item.get("status") == "in_progress"),
        None,
    )
    current_candidates = [
        node for node in nodes
        if node.get("lineage") == lineage
        and (active is None or node.get("milestoneId") == active)
    ]
    if not current_candidates:
        current_candidates = [node for node in nodes if node.get("lineage") == lineage]
    # Runs and probes are assembled in separate passes, so their local `step`
    # values do not define a shared chronology.  Compact UTC run identities do.
    current = max(
        current_candidates,
        key=lambda item: (str(item.get("startedAt") or ""), item.get("step", 0)),
    )["id"]
    if synced_backend_nodes and ledger.get("backend") != "mujoco_warp_cuda":
        current = synced_backend_nodes[-1]["id"]
    if backend_progress:
        backend_current = next((node for node in nodes if node['id']==backend_progress['currentNodeId']), {})
        latest_pass = max((str(item.get('passedAt') or '') for item in ledger.get('milestones',[]) if item.get('status')=='passed'), default='')
        if ledger.get('backend')!='mujoco_warp_cuda' or str(backend_current.get('startedAt') or '')>latest_pass:
            current = backend_progress['currentNodeId']
        backend_progress['currentNodeId'] = current
    # Reserve the current experiment and its ancestors before older history.
    by_id = {node["id"]: node for node in nodes}
    priority = list(dict.fromkeys([current, *(node["id"] for node in nodes if node.get("developmentReference"))]))
    for node_id in priority:
        for parent in by_id.get(node_id, {}).get("parentIds", []):
            if parent in by_id and parent not in priority:
                priority.append(parent)
    recent = sorted(nodes, key=lambda node: (node.get("lineage") == by_id[current].get("lineage"), _chronology(node)), reverse=True)
    priority.extend(node["id"] for node in recent if node["id"] not in priority)
    visible_set = set(priority[:12])
    default_visible = [node["id"] for node in nodes if node["id"] in visible_set]
    return {
        "progress": backend_progress,
        "milestones": _milestone_summaries(ledger),
        "schemaVersion": 1,
        "type": "evolutionTree",
        "lineage": by_id[current].get("lineage", lineage),
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "primaryMetric": "duration_s" if active=="teleop_60s_forward_turn" else "final_heading_rad" if active=="yaw_turn_90deg_hold" else "semantic_forward_displacement_m",
        "targetMetricValue": 60. if active=="teleop_60s_forward_turn" else 1.5707963267948966 if active=="yaw_turn_90deg_hold" else 10.0 if active in ("gate_10m_no_reset", "gate_10m_four_directions_no_reset") else 5.0,
        "visibleNodeBudget": VISIBLE_NODE_BUDGET,
        "defaultVisibleNodeIds": default_visible,
        "currentNodeId": current,
        "nodes": nodes,
        "summary": {
            "passedMilestoneCount": sum(m.get("status") == "passed" for m in ledger.get("milestones", [])),
            "nodeCount": len(nodes),
            "failedCount": sum(node["status"] == "failed" for node in nodes),
            "milestoneCount": len(ledger.get("milestones", [])),
            "checkpointBytes": sum(int(node.get("diskBytes") or 0) for node in nodes),
            "checkpointStorage": "Nextcloud online-only; not hydrated on Mac",
        },
    }


def write_evolution(path: Path = DEFAULT_OUTPUT) -> dict:
    payload = build_evolution()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = write_evolution(args.output)
    print(json.dumps({"path": str(args.output), "nodes": len(payload["nodes"]), "current": payload["currentNodeId"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
