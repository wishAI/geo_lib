from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from argparse import Namespace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .asset_paths import landau_input_dir, landau_mesh_root, landau_skeleton_json_path, landau_urdf_path, landau_usd_path, outputs_dir


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def history_dir() -> Path:
    path = outputs_dir() / "history"
    path.mkdir(parents=True, exist_ok=True)
    return path


def checkpoint_registry_path() -> Path:
    return history_dir() / "checkpoint_registry.json"


def history_refs_dir() -> Path:
    path = history_dir() / "refs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_checkpoint_registry(path: Path | None = None) -> dict[str, Any]:
    registry_path = path or checkpoint_registry_path()
    if not registry_path.is_file():
        return {"experiments": {}}
    try:
        payload = json.loads(registry_path.read_text(encoding="utf-8"))
    except Exception:
        return {"experiments": {}}
    if not isinstance(payload, dict):
        return {"experiments": {}}
    experiments = payload.get("experiments")
    if not isinstance(experiments, dict):
        payload["experiments"] = {}
    return payload


_LANDAU_STAGE_SUFFIX = {
    "stand": "stand",
    "full": "flat",
    "fwd_only": "fwd_only",
    "fwd_yaw": "fwd_yaw",
    "game": "game",
}


def _landau_stage_from_experiment_name(experiment_name: str) -> str | None:
    if not experiment_name.startswith("geo_landau_"):
        return None
    for stage, suffix in _LANDAU_STAGE_SUFFIX.items():
        if experiment_name.endswith(f"_{suffix}"):
            return stage
    return None


def _recommended_experiment_candidates(experiment_name: str) -> list[str]:
    # Keep recommendation lookup lineage-local. Falling back from an active-lineage
    # experiment (for example `geo_landau_restart_..._game`) to the legacy
    # `geo_landau_game` checkpoint silently mixed incompatible states and made the
    # simple play/teleop commands look healthier than the active lineage really was.
    return [experiment_name]


def _iter_checkpoint_registry_paths(primary_registry_path: Path | None = None) -> list[Path]:
    candidate_paths: list[Path] = []
    seen: set[Path] = set()

    primary_path = (primary_registry_path or checkpoint_registry_path()).resolve()
    if primary_path not in seen:
        candidate_paths.append(primary_path)
        seen.add(primary_path)

    archives_root = outputs_dir() / "archives"
    if archives_root.is_dir():
        for archive_path in sorted(archives_root.glob("*/history/checkpoint_registry.json"), reverse=True):
            resolved = archive_path.resolve()
            if resolved in seen:
                continue
            candidate_paths.append(resolved)
            seen.add(resolved)
    return candidate_paths


def _resolve_checkpoint_path_from_archive(
    checkpoint_path: str,
    *,
    registry_path: Path,
) -> Path | None:
    checkpoint = Path(checkpoint_path).expanduser()
    if checkpoint.is_file():
        return checkpoint.resolve()
    try:
        archive_root = registry_path.resolve().parents[1]
    except Exception:
        return None
    if archive_root.name == "history":
        archive_root = archive_root.parent
    checkpoint_text = str(checkpoint)
    normalized = checkpoint_text.replace("\\", "/")
    marker_index = normalized.find("/logs/")
    if marker_index == -1:
        return None
    logs_suffix = normalized[marker_index + len("/logs/") :]
    relocated = archive_root / "logs" / logs_suffix
    if relocated.is_file():
        return relocated.resolve()
    return None


def resolve_recommended_checkpoint(
    experiment_name: str | None,
    *,
    registry_path: Path | None = None,
) -> dict[str, Any] | None:
    if not experiment_name:
        return None
    for candidate_registry_path in _iter_checkpoint_registry_paths(registry_path):
        registry = load_checkpoint_registry(candidate_registry_path)
        experiments = registry.get("experiments", {})
        if not isinstance(experiments, dict):
            continue
        for candidate_experiment_name in _recommended_experiment_candidates(experiment_name):
            experiment_entry = experiments.get(candidate_experiment_name)
            if not isinstance(experiment_entry, dict):
                continue
            recommended_entry = experiment_entry.get("recommended")
            if not isinstance(recommended_entry, dict):
                continue
            checkpoint_path = recommended_entry.get("checkpoint_path")
            if not isinstance(checkpoint_path, str) or not checkpoint_path:
                continue
            checkpoint = _resolve_checkpoint_path_from_archive(
                checkpoint_path,
                registry_path=candidate_registry_path,
            )
            if checkpoint is None:
                continue
            resolved_entry = dict(recommended_entry)
            resolved_entry["checkpoint_path"] = str(checkpoint)
            resolved_entry["resolved_experiment_name"] = candidate_experiment_name
            resolved_entry["registry_path"] = str(candidate_registry_path)
            return resolved_entry
    return None


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_ready(value: Any) -> Any:
    if isinstance(value, Namespace):
        return {key: _json_ready(val) for key, val in vars(value).items()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _directory_fingerprint(root: Path, pattern: str) -> dict[str, Any]:
    digest = hashlib.sha256()
    files = []
    for path in sorted(root.rglob(pattern)):
        rel_path = path.relative_to(root).as_posix()
        file_hash = _sha256_file(path)
        digest.update(rel_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_hash.encode("utf-8"))
        digest.update(b"\0")
        files.append({"path": rel_path, "sha256": file_hash})
    return {
        "root": str(root),
        "file_count": len(files),
        "sha256": digest.hexdigest(),
        "files": files,
    }


def collect_landau_asset_snapshot() -> dict[str, Any]:
    return {
        "input_dir": str(landau_input_dir()),
        "urdf": {"path": str(landau_urdf_path()), "sha256": _sha256_file(landau_urdf_path())},
        "usd": {"path": str(landau_usd_path()), "sha256": _sha256_file(landau_usd_path())},
        "skeleton_json": {
            "path": str(landau_skeleton_json_path()),
            "sha256": _sha256_file(landau_skeleton_json_path()),
        },
        "mesh_tree": _directory_fingerprint(landau_mesh_root(), "*.stl"),
    }


def _run_git(args: list[str]) -> str | None:
    try:
        completed = subprocess.run(
            args,
            cwd=_repo_root(),
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return completed.stdout.strip()


def git_snapshot() -> dict[str, Any]:
    return {
        "head": _run_git(["git", "rev-parse", "HEAD"]),
        "branch": _run_git(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "status_short": (_run_git(["git", "status", "--short"]) or "").splitlines(),
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready(payload), indent=2, sort_keys=True), encoding="utf-8")


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(_json_ready(payload), sort_keys=True))
        stream.write("\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except Exception:
            continue
        if isinstance(payload, dict):
            records.append(payload)
    return records


def _record_timestamp(record: dict[str, Any]) -> str:
    value = record.get("recorded_at")
    return value if isinstance(value, str) else ""


def _record_matches_experiment(record: dict[str, Any], *, experiment_name: str) -> bool:
    task = record.get("task")
    return isinstance(task, dict) and task.get("experiment_name") == experiment_name


def _latest_record(records: list[dict[str, Any]], *, experiment_name: str) -> dict[str, Any] | None:
    matching = [record for record in records if _record_matches_experiment(record, experiment_name=experiment_name)]
    if not matching:
        return None
    matching.sort(key=_record_timestamp)
    return matching[-1]


def _latest_terminal_record(records: list[dict[str, Any]], *, experiment_name: str) -> dict[str, Any] | None:
    matching = [
        record
        for record in records
        if _record_matches_experiment(record, experiment_name=experiment_name)
        and record.get("status") in {"completed", "failed"}
    ]
    if not matching:
        return None
    matching.sort(key=_record_timestamp)
    return matching[-1]


def _resolve_experiment_name(
    *,
    task_spec,
    args: Namespace | None = None,
    agent_cfg: Any | None = None,
    experiment_name: str | None = None,
) -> str | None:
    for candidate in (
        experiment_name,
        getattr(agent_cfg, "experiment_name", None),
        getattr(args, "experiment_name", None) if args is not None else None,
        getattr(task_spec, "experiment_name", None),
    ):
        if isinstance(candidate, str) and candidate:
            return candidate
    return None


def _resolve_workflow_id(*, args: Namespace | None = None, workflow_id: str | None = None) -> str | None:
    if isinstance(workflow_id, str) and workflow_id:
        return workflow_id
    if args is None:
        return None
    candidate = getattr(args, "workflow_id", None)
    return candidate if isinstance(candidate, str) and candidate else None


def refresh_history_refs() -> dict[str, Any]:
    history_root = history_dir()
    refs_root = history_refs_dir()
    experiments_root = history_root / "experiments"
    registry = load_checkpoint_registry()
    try:
        from .training_lineage import load_active_lineage
    except Exception:
        load_active_lineage = lambda: None  # type: ignore[assignment]
    active_lineage = load_active_lineage()

    training_records = _read_jsonl(history_root / "training_runs.jsonl")
    validation_records = _read_jsonl(history_root / "validation_runs.jsonl")
    evaluation_records = _read_jsonl(history_root / "evaluation_runs.jsonl")
    diagnostic_records = _read_jsonl(history_root / "diagnostic_runs.jsonl")

    experiment_names: set[str] = set()
    registry_experiments = registry.get("experiments", {})
    if isinstance(registry_experiments, dict):
        experiment_names.update(str(name) for name in registry_experiments)
    if experiments_root.is_dir():
        experiment_names.update(path.stem for path in experiments_root.glob("*.json"))
    for record_group in (training_records, validation_records, evaluation_records, diagnostic_records):
        for record in record_group:
            task = record.get("task")
            if isinstance(task, dict) and isinstance(task.get("experiment_name"), str):
                experiment_names.add(task["experiment_name"])

    experiments_index: dict[str, Any] = {}
    for experiment_name in sorted(experiment_names):
        experiment_ref_path = experiments_root / f"{experiment_name}.json"
        latest_training_event = _latest_record(training_records, experiment_name=experiment_name)
        latest_training_terminal = _latest_terminal_record(training_records, experiment_name=experiment_name)
        experiment_summary = {
            "experiment_name": experiment_name,
            "recommended": registry_experiments.get(experiment_name, {}).get("recommended")
            if isinstance(registry_experiments, dict)
            else None,
            "latest_training": latest_training_terminal or latest_training_event,
            "latest_training_event": latest_training_event,
            "latest_training_terminal": latest_training_terminal,
            "latest_validation": _latest_record(validation_records, experiment_name=experiment_name),
            "latest_evaluation": _latest_record(evaluation_records, experiment_name=experiment_name),
            "latest_diagnostic": _latest_record(diagnostic_records, experiment_name=experiment_name),
            "experiment_ref_path": str(experiment_ref_path),
        }
        experiments_index[experiment_name] = experiment_summary
        _write_json(refs_root / "experiments" / f"{experiment_name}.json", experiment_summary)
        _write_json(experiments_root / f"{experiment_name}.json", experiment_summary)

    latest_training_event = training_records[-1] if training_records else None
    latest_training_terminal = None
    if training_records:
        terminal_records = [record for record in training_records if record.get("status") in {"completed", "failed"}]
        if terminal_records:
            latest_training_terminal = terminal_records[-1]
    latest_summary = {
        "training": latest_training_terminal or latest_training_event,
        "training_event": latest_training_event,
        "training_terminal": latest_training_terminal,
        "validation": validation_records[-1] if validation_records else None,
        "evaluation": evaluation_records[-1] if evaluation_records else None,
        "diagnostic": diagnostic_records[-1] if diagnostic_records else None,
    }
    _write_json(refs_root / "latest.json", latest_summary)

    defaults_payload = {
        "recommended_registry": registry.get("experiments", {}) if isinstance(registry.get("experiments"), dict) else {},
        "promoted_default": active_lineage.get("promoted_default") if isinstance(active_lineage, dict) else None,
    }
    current_payload = None
    if isinstance(active_lineage, dict):
        current_payload = {
            "lineage_name": active_lineage.get("lineage_name"),
            "stage": active_lineage.get("current_stage"),
            "run_name": active_lineage.get("current_run"),
            "checkpoint_path": active_lineage.get("current_checkpoint"),
            "workflow_id": active_lineage.get("current_workflow_id"),
        }

    index_payload = {
        "generated_at": _timestamp(),
        "history_root": str(history_root),
        "source_files": {
            "checkpoint_registry": str(checkpoint_registry_path()),
            "training_runs": str(history_root / "training_runs.jsonl"),
            "validation_runs": str(history_root / "validation_runs.jsonl"),
            "evaluation_runs": str(history_root / "evaluation_runs.jsonl"),
            "diagnostic_runs": str(history_root / "diagnostic_runs.jsonl"),
            "experiments_dir": str(experiments_root),
        },
        "experiments": experiments_index,
        "latest": latest_summary,
        "active_lineage": active_lineage,
        "current": current_payload,
        "defaults": defaults_payload,
        "training_rule": active_lineage.get("training_rule") if isinstance(active_lineage, dict) else None,
        "current_target_milestone": active_lineage.get("current_target_milestone") if isinstance(active_lineage, dict) else None,
        "milestone_registry": active_lineage.get("milestone_registry") if isinstance(active_lineage, dict) else None,
        "milestone_history": active_lineage.get("milestone_history") if isinstance(active_lineage, dict) else None,
        "stage_state": active_lineage.get("stage_state") if isinstance(active_lineage, dict) else None,
        "stage_history": active_lineage.get("stage_history") if isinstance(active_lineage, dict) else None,
    }
    _write_json(
        refs_root / "milestones.json",
        {
            "generated_at": _timestamp(),
            "lineage_name": active_lineage.get("lineage_name") if isinstance(active_lineage, dict) else None,
            "current_target_milestone": active_lineage.get("current_target_milestone") if isinstance(active_lineage, dict) else None,
            "milestone_registry": active_lineage.get("milestone_registry") if isinstance(active_lineage, dict) else None,
            "milestone_history": active_lineage.get("milestone_history") if isinstance(active_lineage, dict) else None,
        },
    )
    _write_json(
        refs_root / "stage_history.json",
        {
            "generated_at": _timestamp(),
            "lineage_name": active_lineage.get("lineage_name") if isinstance(active_lineage, dict) else None,
            "entries": active_lineage.get("stage_history") if isinstance(active_lineage, dict) else [],
        },
    )
    _write_json(refs_root / "index.json", index_payload)
    return index_payload


def _checkpoint_summary(run_dir: Path) -> dict[str, Any]:
    def _checkpoint_key(path: Path) -> tuple[int, str]:
        stem = path.stem
        try:
            return (int(stem.split("_")[-1]), stem)
        except Exception:
            return (-1, stem)

    checkpoints = sorted(run_dir.glob("model_*.pt"), key=_checkpoint_key)
    if not checkpoints:
        return {"count": 0, "latest": None}
    return {
        "count": len(checkpoints),
        "latest": str(checkpoints[-1]),
    }


def _sync_landau_lineage_training_state(
    *,
    task_spec,
    args: Namespace,
    payload: dict[str, Any],
    status: str,
    resume_path: str | None,
    error: str | None,
) -> None:
    if getattr(task_spec, "key", None) != "landau":
        return
    stage = getattr(args, "stage", None)
    if not isinstance(stage, str) or not stage:
        return
    status_map = {
        "started": "running",
        "completed": "trained",
        "failed": "failed",
    }
    lineage_status = status_map.get(status)
    if lineage_status is None:
        return

    try:
        from .training_lineage import update_lineage_stage_state
    except Exception:
        return

    latest_checkpoint = payload.get("checkpoints", {}).get("latest")
    produced_checkpoint = str(latest_checkpoint) if isinstance(latest_checkpoint, str) and latest_checkpoint else None
    failure_payload = None
    if status == "failed":
        failure_payload = {
            "step": "train",
            "reason": error or "training_failed",
        }

    update_lineage_stage_state(
        stage=stage,
        status=lineage_status,
        workflow_id=payload.get("workflow_id"),
        run_name=Path(str(payload["log_dir"])).name,
        selected_checkpoint=resume_path,
        produced_checkpoint=produced_checkpoint if status != "started" else None,
        last_training=payload,
        failure=failure_payload,
    )


def write_training_record(
    *,
    log_dir: str | Path,
    task_spec,
    args: Namespace,
    agent_cfg: Any,
    status: str,
    resume_path: str | None = None,
    resume_metadata: dict[str, Any] | None = None,
    error: str | None = None,
    workflow_id: str | None = None,
) -> dict[str, Any]:
    run_dir = Path(log_dir)
    resolved_experiment_name = _resolve_experiment_name(task_spec=task_spec, args=args, agent_cfg=agent_cfg)
    try:
        from .training_lineage import load_active_lineage
    except Exception:
        load_active_lineage = lambda: None  # type: ignore[assignment]
    active_lineage = load_active_lineage()
    payload = {
        "record_kind": "training",
        "recorded_at": _timestamp(),
        "status": status,
        "workflow_id": _resolve_workflow_id(args=args, workflow_id=workflow_id),
        "lineage_name": active_lineage.get("lineage_name") if isinstance(active_lineage, dict) else None,
        "cwd": str(Path.cwd()),
        "python_executable": sys.executable,
        "log_dir": str(run_dir),
        "task": {
            "robot": task_spec.key,
            "display_name": task_spec.display_name,
            "train_task_id": task_spec.train_task_id,
            "play_task_id": task_spec.play_task_id,
            "experiment_name": resolved_experiment_name,
        },
        "cli_args": _json_ready(args),
        "resume_path": resume_path,
        "resume_metadata": _json_ready(resume_metadata),
        "checkpoints": _checkpoint_summary(run_dir),
        "git": git_snapshot(),
        "landau_assets": collect_landau_asset_snapshot() if task_spec.key == "landau" else None,
        "error": error,
    }
    _sync_landau_lineage_training_state(
        task_spec=task_spec,
        args=args,
        payload=payload,
        status=status,
        resume_path=resume_path,
        error=error,
    )
    _write_json(run_dir / "repro" / "training_record.json", payload)
    _append_jsonl(history_dir() / "training_runs.jsonl", payload)
    refresh_history_refs()
    return payload


def write_validation_record(
    *,
    checkpoint_path: str | Path,
    task_spec,
    args: Namespace,
    status: str,
    metrics: dict[str, Any],
    experiment_name: str | None = None,
    failure: str | None = None,
    gate_result: dict[str, Any] | None = None,
    workflow_id: str | None = None,
) -> dict[str, Any]:
    checkpoint = Path(checkpoint_path)
    resolved_experiment_name = _resolve_experiment_name(task_spec=task_spec, args=args, experiment_name=experiment_name)
    try:
        from .training_lineage import load_active_lineage
    except Exception:
        load_active_lineage = lambda: None  # type: ignore[assignment]
    active_lineage = load_active_lineage()
    payload = {
        "record_kind": "validation",
        "recorded_at": _timestamp(),
        "status": status,
        "workflow_id": _resolve_workflow_id(args=args, workflow_id=workflow_id),
        "lineage_name": active_lineage.get("lineage_name") if isinstance(active_lineage, dict) else None,
        "checkpoint_path": str(checkpoint),
        "task": {
            "robot": task_spec.key,
            "display_name": task_spec.display_name,
            "train_task_id": task_spec.train_task_id,
            "play_task_id": task_spec.play_task_id,
            "experiment_name": resolved_experiment_name,
        },
        "cli_args": _json_ready(args),
        "metrics": _json_ready(metrics),
        "failure": failure,
        "gate_result": _json_ready(gate_result),
        "git": git_snapshot(),
        "landau_assets": collect_landau_asset_snapshot() if task_spec.key == "landau" else None,
    }
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    _write_json(checkpoint.parent / "validation" / f"{timestamp}.json", payload)
    _append_jsonl(history_dir() / "validation_runs.jsonl", payload)
    refresh_history_refs()
    return payload


def write_evaluation_record(
    *,
    checkpoint_path: str | Path,
    task_spec,
    args: Namespace,
    scenario: str,
    status: str,
    metrics: dict[str, Any],
    experiment_name: str | None = None,
    failure: str | None = None,
    gate_result: dict[str, Any] | None = None,
    workflow_id: str | None = None,
) -> dict[str, Any]:
    checkpoint = Path(checkpoint_path)
    resolved_experiment_name = _resolve_experiment_name(task_spec=task_spec, args=args, experiment_name=experiment_name)
    try:
        from .training_lineage import load_active_lineage
    except Exception:
        load_active_lineage = lambda: None  # type: ignore[assignment]
    active_lineage = load_active_lineage()
    payload = {
        "record_kind": "evaluation",
        "recorded_at": _timestamp(),
        "status": status,
        "workflow_id": _resolve_workflow_id(args=args, workflow_id=workflow_id),
        "lineage_name": active_lineage.get("lineage_name") if isinstance(active_lineage, dict) else None,
        "scenario": scenario,
        "checkpoint_path": str(checkpoint),
        "task": {
            "robot": task_spec.key,
            "display_name": task_spec.display_name,
            "train_task_id": task_spec.train_task_id,
            "play_task_id": task_spec.play_task_id,
            "experiment_name": resolved_experiment_name,
        },
        "cli_args": _json_ready(args),
        "metrics": _json_ready(metrics),
        "failure": failure,
        "gate_result": _json_ready(gate_result),
        "git": git_snapshot(),
        "landau_assets": collect_landau_asset_snapshot() if task_spec.key == "landau" else None,
    }
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    _write_json(checkpoint.parent / "evaluation" / f"{timestamp}_{scenario}.json", payload)
    _append_jsonl(history_dir() / "evaluation_runs.jsonl", payload)
    refresh_history_refs()
    return payload


def write_diagnostic_record(
    *,
    task_spec,
    args: Namespace,
    scenario: str,
    status: str,
    metrics: dict[str, Any],
    checkpoint_path: str | Path | None = None,
    experiment_name: str | None = None,
    failure: str | None = None,
    gate_result: dict[str, Any] | None = None,
    workflow_id: str | None = None,
) -> dict[str, Any]:
    checkpoint = Path(checkpoint_path).resolve() if checkpoint_path is not None else None
    resolved_experiment_name = _resolve_experiment_name(task_spec=task_spec, args=args, experiment_name=experiment_name)
    try:
        from .training_lineage import load_active_lineage
    except Exception:
        load_active_lineage = lambda: None  # type: ignore[assignment]
    active_lineage = load_active_lineage()
    payload = {
        "record_kind": "diagnostic",
        "recorded_at": _timestamp(),
        "status": status,
        "workflow_id": _resolve_workflow_id(args=args, workflow_id=workflow_id),
        "lineage_name": active_lineage.get("lineage_name") if isinstance(active_lineage, dict) else None,
        "scenario": scenario,
        "checkpoint_path": None if checkpoint is None else str(checkpoint),
        "task": {
            "robot": task_spec.key,
            "display_name": task_spec.display_name,
            "train_task_id": task_spec.train_task_id,
            "play_task_id": task_spec.play_task_id,
            "experiment_name": resolved_experiment_name,
        },
        "cli_args": _json_ready(args),
        "metrics": _json_ready(metrics),
        "failure": failure,
        "gate_result": _json_ready(gate_result),
        "git": git_snapshot(),
        "landau_assets": collect_landau_asset_snapshot() if task_spec.key == "landau" else None,
    }
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_root = history_dir() / "diagnostics"
    _write_json(output_root / f"{timestamp}_{scenario}.json", payload)
    _append_jsonl(history_dir() / "diagnostic_runs.jsonl", payload)
    refresh_history_refs()
    return payload
