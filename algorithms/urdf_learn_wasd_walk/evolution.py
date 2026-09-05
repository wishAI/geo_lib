"""Build a compact, truthful evolution tree from Landau run artifacts."""

from __future__ import annotations

import argparse
import json
import os
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


def build_evolution(
    output_root: Path = OUTPUT_ROOT,
    milestones_path: Path = MILESTONES_PATH,
) -> dict:
    ledger = _read_json(milestones_path)
    if ledger is None:
        raise ValueError(f"Milestone ledger is unavailable: {milestones_path}")
    lineage = str(ledger.get("lineage", "unknown"))
    invalidated = ledger.get("invalidatedLineage", {})
    invalidated_lineage = (
        str(invalidated.get("lineage"))
        if isinstance(invalidated, dict) and invalidated.get("lineage")
        else None
    )
    accepted_lineages = {lineage, invalidated_lineage} - {None}
    milestone_records = {item["id"]: item for item in ledger.get("milestones", [])}
    passive = milestone_records.get("stand_zero_signal_30s_no_reset", {})
    nodes = []
    invalidated_root_id = None
    if invalidated_lineage:
        invalidated_root_id = f"invalidated:{invalidated_lineage}:stand_zero_signal_30s_no_reset"
        nodes.append({
            "id": invalidated_root_id,
            "parentIds": [],
            "label": "Previous mesh lineage · invalidated",
            "step": -1,
            "status": "failed",
            "kind": "root",
            "lineage": invalidated_lineage,
            "approach": "superseded visual/collision mesh package",
            "result": str(invalidated.get("reason", "asset lineage was invalidated")),
            "metrics": {},
            "important": True,
            "meshTreeSha256": invalidated.get("meshTreeSha256"),
        })
    passive_status = str(passive.get("status", "not_started"))
    nodes.append({
        "id": "milestone:stand_zero_signal_30s_no_reset",
        "parentIds": [invalidated_root_id] if invalidated_root_id else [],
        "label": "Latest mesh · passive stand 30 s",
        "step": 0,
        "status": "completed" if passive_status == "passed" else "running" if passive_status == "in_progress" else "failed",
        "kind": "root",
        "lineage": lineage,
        "approach": "URDF equilibrium pose + PD control",
        "result": "canonical zero-signal stand gate passed" if passive_status == "passed" else "latest visual/collision mesh awaits gate re-certification",
        "metrics": passive.get("metrics", {}),
        "important": True,
        "checkpointPath": passive.get("checkpoint"),
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

    policy_record = milestone_records.get("stand_30s_no_reset", {})
    policy_checkpoint = policy_record.get("checkpoint", {})
    if isinstance(policy_checkpoint, dict) and policy_checkpoint.get("sha256"):
        checkpoint_nodes[str(policy_checkpoint["sha256"])] = "milestone:stand_30s_no_reset"

    runs.sort(key=lambda item: str(item[1].get("run_identity", item[0])))
    for step, (training_path, training, validation_path, validation) in enumerate(runs, start=1):
        checkpoint = training.get("checkpoint", {})
        run_lineage = str(training.get("lineage", "unknown"))
        is_invalidated = bool(invalidated_lineage and run_lineage == invalidated_lineage)
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
                parent_id = invalidated_root_id
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
        is_invalidated = bool(invalidated_lineage and probe_lineage == invalidated_lineage)
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
            "label": probe_path.parent.name.replace("_", " "),
            "step": step,
            "status": status,
            "kind": "experiment",
            "lineage": probe_lineage,
            "approach": str(probe.get("experiment", "open-loop reference probe")),
            "result": result,
            "metrics": metrics,
            "checkpointPath": parent_checkpoint.get("path") if isinstance(parent_checkpoint, dict) else None,
            "checkpointSha256": parent_sha,
            "experimentParameters": parameters,
            "startedAt": run_identity,
            "sourceRevision": probe.get("source_commit"),
            "artifacts": [_artifact(probe_path, node_id)],
            "important": True,
        })

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
    current_candidates = [node for node in nodes if node.get("lineage") == lineage]
    current = max(current_candidates, key=lambda item: item.get("step", 0))["id"]
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
        "summary": {
            "nodeCount": len(nodes),
            "failedCount": sum(node["status"] == "failed" for node in nodes),
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
