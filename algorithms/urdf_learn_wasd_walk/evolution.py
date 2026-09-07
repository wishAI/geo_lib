"""Build a compact, truthful evolution tree from Landau run artifacts."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from collections import Counter, defaultdict


ALGORITHM_ROOT = Path(__file__).resolve().parent
REPO_ROOT = ALGORITHM_ROOT.parents[1]
OUTPUT_ROOT = ALGORITHM_ROOT / "outputs"
MILESTONES_PATH = ALGORITHM_ROOT / "milestones.json"
DEFAULT_OUTPUT = OUTPUT_ROOT / "evolution.json"
VISIBLE_NODE_BUDGET = 40
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


def _overview_nodes(
    nodes: list[dict],
    *,
    current_id: str,
    current_lineage: str,
    invalidated_root_ids: dict[str, str],
) -> list[dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    retained = []
    for node in nodes:
        lineage = str(node.get("lineage", "unknown"))
        if (
            node.get("status") == "failed"
            and node.get("kind") != "root"
            and lineage != current_lineage
        ):
            groups[lineage].append(node)
        else:
            retained.append(node)

    summaries = []
    for lineage, members in groups.items():
        members.sort(key=lambda node: (str(node.get("startedAt") or ""), int(node.get("step") or 0)))
        failures = Counter(str(node.get("result") or "failed") for node in members)
        approaches = Counter(str(node.get("approach") or "unknown") for node in members)
        milestones = Counter(str(node.get("milestoneId") or "other") for node in members)
        latest_model = next((node for node in reversed(members) if node.get("checkpointPath")), None)
        run_steps = [int(node.get("step") or 0) for node in members]
        summaries.append({
            "id": f"group:{lineage}",
            "parentIds": [invalidated_root_ids[lineage]],
            "label": f"{len(members)} rejected runs · archived lineage",
            "step": min(run_steps),
            "status": "failed",
            "kind": "range",
            "milestoneId": None,
            "lineage": lineage,
            "approach": ", ".join(name for name, _ in approaches.most_common(3)),
            "result": f"Merged {len(members)} rejected runs; most common outcome: {failures.most_common(1)[0][0]}",
            "changeSummary": f"{len(approaches)} approaches merged; select Inspect runs for exact parameter changes",
            "metrics": {},
            "checkpointPath": latest_model.get("checkpointPath") if latest_model else None,
            "checkpointSha256": latest_model.get("checkpointSha256") if latest_model else None,
            "diskBytes": sum(int(node.get("diskBytes") or 0) for node in members),
            "checkpointStorage": latest_model.get("checkpointStorage") if latest_model else None,
            "trainingProgress": {
                "kind": "merged",
                "runCount": len(members),
                "fromSequence": min(run_steps),
                "toSequence": max(run_steps),
            },
            "collapsedCount": len(members),
            "memberNodeIds": [node["id"] for node in members],
            "range": {"fromStep": min(run_steps), "toStep": max(run_steps)},
            "failureBreakdown": [
                {"result": result, "count": count}
                for result, count in failures.most_common(5)
            ],
            "milestoneBreakdown": [
                {"milestone": MILESTONE_LABELS.get(milestone, milestone), "count": count}
                for milestone, count in milestones.most_common()
            ],
            "important": True,
        })

    summary_by_parent: dict[str, list[dict]] = defaultdict(list)
    for summary in summaries:
        summary_by_parent[summary["parentIds"][0]].append(summary)
    overview = []
    for node in retained:
        overview.append(node)
        overview.extend(sorted(summary_by_parent.get(node["id"], []), key=lambda item: item["step"]))
    if current_id not in {node["id"] for node in overview}:
        current = next(node for node in nodes if node["id"] == current_id)
        overview.append(current)
    return overview


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
            "label": f"Archived lineage · {invalidated_index + 1}",
            "step": invalidated_index - len(invalidated_entries),
            "status": "failed",
            "kind": "root",
            "lineage": invalidated_lineage,
            "approach": "superseded visual/collision mesh package",
            "result": str(invalidated.get("reason", "asset lineage was invalidated")),
            "changeSummary": "Archived lineage; its results cannot satisfy current milestones",
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
        "label": "Current · passive stand 30 s",
        "step": 0,
        "status": "completed" if passive_status == "passed" else "running" if passive_status == "in_progress" else "failed",
        "kind": "root",
        "milestoneId": "stand_zero_signal_30s_no_reset",
        "lineage": lineage,
        "approach": "URDF equilibrium pose + PD control",
        "result": "canonical zero-signal stand gate passed" if passive_status == "passed" else "30 s dynamics and visual proof are still required",
        "changeSummary": "Establish the zero-command passive stability baseline",
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
            "label": "Current · policy stand 30 s",
            "step": 1,
            "status": "running",
            "kind": "milestone",
            "milestoneId": "stand_30s_no_reset",
            "lineage": lineage,
            "approach": "manager-based proprioceptive PPO",
            "result": "awaiting a policy checkpoint after passive standing passes",
            "changeSummary": "Replace fixed PD output with a learned zero-command policy",
            "metrics": {},
            "important": True,
            "meshTreeSha256": ledger.get("assetContract", {}).get("meshTreeSha256"),
        })
    forward_record = milestone_records.get("gate_5m_no_reset", {})
    if forward_record.get("status") == "in_progress":
        nodes.append({
            "id": "milestone:gate_5m_no_reset",
            "parentIds": ["milestone:stand_30s_no_reset"],
            "label": "Current · forward gate 5 m",
            "step": 2,
            "status": "running",
            "kind": "milestone",
            "milestoneId": "gate_5m_no_reset",
            "lineage": lineage,
            "approach": "flat +Y manager-based PPO",
            "result": "awaiting a walking checkpoint after both standing gates pass",
            "changeSummary": "Add forward command tracking while preserving standing",
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
        parameters = _training_parameters(contract)
        run_name = training_path.parent.name
        label = "Policy stand · passed" if milestone == "stand_30s_no_reset" else run_name.replace("_", " ")
        if is_invalidated:
            label = f"Old mesh · {label}"
        artifacts = [_artifact(training_path, node_id)]
        if validation_path is not None:
            artifacts.append(_artifact(validation_path, node_id))
        for video_name in ("proof.mp4", "proof_smoke.mp4"):
            video_path = training_path.parent / video_name
            if video_path.is_file():
                artifacts.append({
                    "id": f"artifact:{_relative(video_path)}",
                    "kind": "video",
                    "path": _relative(video_path),
                    "mimeType": "video/mp4",
                    "byteSize": video_path.stat().st_size,
                    "producedBy": node_id,
                })
                break
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
            "trainingProgress": _training_progress(training),
            "experimentParameters": parameters,
            "changeSummary": str(approach).replace("_", " "),
            "startedAt": training.get("run_identity"),
            "completedAt": training.get("completed_at"),
            "sourceRevision": training.get("source_commit"),
            "artifacts": artifacts,
            "important": important,
        }
        nodes = [item for item in nodes if item["id"] != node_id]
        nodes.append(node)

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
        raw_metrics = probe.get("metrics", {})
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
            "label": probe_path.parent.name.replace("_", " "),
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
            "trainingProgress": {
                "kind": "diagnostic",
                "physicsSteps": raw_metrics.get("physics_steps"),
                "controlSteps": raw_metrics.get("control_steps"),
                "durationSeconds": raw_metrics.get("duration_s"),
            },
            "changeSummary": str(probe.get("experiment", "open-loop reference probe")).replace("_", " "),
            "startedAt": run_identity,
            "sourceRevision": probe.get("source_commit"),
            "artifacts": [_artifact(probe_path, node_id)],
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
    overview = _overview_nodes(
        nodes,
        current_id=current,
        current_lineage=lineage,
        invalidated_root_ids=invalidated_root_ids,
    )
    default_visible = [node["id"] for node in overview]
    milestones = _milestone_summaries(ledger)
    return {
        "schemaVersion": 1,
        "type": "evolutionTree",
        "lineage": lineage,
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "primaryMetric": "semantic_forward_displacement_m",
        "targetMetricValue": 5.0,
        "visibleNodeBudget": VISIBLE_NODE_BUDGET,
        "defaultVisibleNodeIds": default_visible,
        "currentNodeId": current,
        "nodes": nodes,
        "overviewNodes": overview,
        "milestones": milestones,
        "summary": {
            "nodeCount": len(nodes),
            "overviewNodeCount": len(overview),
            "failedCount": sum(node["status"] == "failed" for node in nodes),
            "failedGroupCount": sum(node.get("kind") == "range" for node in overview),
            "passedMilestoneCount": sum(item["status"] == "passed" for item in milestones),
            "milestoneCount": len(milestones),
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
