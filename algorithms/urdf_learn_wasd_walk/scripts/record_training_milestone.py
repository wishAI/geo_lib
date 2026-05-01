from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from algorithms.urdf_learn_wasd_walk.run_history import history_dir
from algorithms.urdf_learn_wasd_walk.training_lineage import (
    goal_ladder_ids,
    load_active_lineage,
    record_lineage_milestone,
    resolve_landau_experiment_name,
)


_LEDGER_BY_RECORD_KIND = {
    "diagnostic": "diagnostic_runs.jsonl",
    "evaluation": "evaluation_runs.jsonl",
    "validation": "validation_runs.jsonl",
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _is_none_marker(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"none", "passive", "null"}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        records.append(json.loads(line))
    return records


def _resolve_checkpoint_path(
    *,
    stage: str,
    load_run: str | None,
    checkpoint: str | None,
    checkpoint_path: str | None,
) -> Path | None:
    if _is_none_marker(checkpoint_path) or _is_none_marker(checkpoint):
        return None
    if checkpoint_path:
        path = Path(checkpoint_path).expanduser().resolve()
    else:
        if not load_run or not checkpoint:
            raise ValueError("Pass either --checkpoint-path or both --load_run and --checkpoint.")
        experiment_name = resolve_landau_experiment_name(stage)
        path = (_repo_root() / "logs" / "rsl_rl" / experiment_name / load_run / checkpoint).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return path


def _compact_record(record: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(record, dict):
        return None
    payload = {
        "record_kind": record.get("record_kind"),
        "recorded_at": record.get("recorded_at"),
        "status": record.get("status"),
        "workflow_id": record.get("workflow_id"),
        "checkpoint_path": record.get("checkpoint_path") or record.get("checkpoints", {}).get("latest"),
        "failure": record.get("failure"),
    }
    if "scenario" in record:
        payload["scenario"] = record.get("scenario")
    metrics = record.get("metrics")
    if isinstance(metrics, dict):
        payload["metrics"] = metrics
    summary = record.get("summary")
    if isinstance(summary, dict):
        payload["summary"] = summary
    return payload


def _latest_evidence_record(*, record_kind: str, checkpoint_path: str, stage: str) -> dict[str, Any] | None:
    ledger_name = _LEDGER_BY_RECORD_KIND.get(record_kind)
    if ledger_name is None:
        return None
    experiment_name = resolve_landau_experiment_name(stage)
    target = str(Path(checkpoint_path).expanduser().resolve())
    for record in reversed(_read_jsonl(history_dir() / ledger_name)):
        if record.get("record_kind") != record_kind:
            continue
        task = record.get("task")
        if isinstance(task, dict) and task.get("experiment_name") != experiment_name:
            continue
        candidate_path = record.get("checkpoint_path")
        if not isinstance(candidate_path, str):
            checkpoints = record.get("checkpoints")
            if isinstance(checkpoints, dict):
                candidate_path = checkpoints.get("latest")
        if not isinstance(candidate_path, str):
            continue
        if str(Path(candidate_path).expanduser().resolve()) == target:
            return record
    return None


def _latest_stage_evidence_record(*, record_kind: str, stage: str) -> dict[str, Any] | None:
    ledger_name = _LEDGER_BY_RECORD_KIND.get(record_kind)
    if ledger_name is None:
        return None
    experiment_name = resolve_landau_experiment_name(stage)
    for record in reversed(_read_jsonl(history_dir() / ledger_name)):
        if record.get("record_kind") != record_kind:
            continue
        task = record.get("task")
        if isinstance(task, dict) and task.get("experiment_name") == experiment_name:
            return record
    return None


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Record a staged training milestone for the active Landau lineage.")
    parser.add_argument("--milestone-id", choices=goal_ladder_ids(), required=True)
    parser.add_argument("--stage", choices=("stand", "fwd_only", "fwd_yaw", "game", "full"), default=None)
    parser.add_argument("--load_run", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--checkpoint-path", type=str, default=None)
    parser.add_argument("--status", choices=("passed", "failed", "blocked"), default="passed")
    parser.add_argument(
        "--evidence-record-kind",
        choices=("auto", "diagnostic", "evaluation", "validation", "manual"),
        default="auto",
    )
    parser.add_argument("--note", type=str, default=None)
    parser.add_argument(
        "--manual-review",
        type=str,
        default=None,
        help="Short note for GUI review, for example: 'arms swing visible; no frozen upper body'.",
    )
    parser.add_argument("--recorded-by", type=str, default="codex")
    parser.add_argument("--print-json", action=argparse.BooleanOptionalAction, default=False)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    active_lineage = load_active_lineage()
    if not isinstance(active_lineage, dict):
        raise RuntimeError("No active training lineage is initialized.")

    registry = active_lineage.get("milestone_registry", {})
    milestone_entry = registry.get(args.milestone_id) if isinstance(registry, dict) else None
    inferred_stage = None
    if isinstance(milestone_entry, dict) and isinstance(milestone_entry.get("stage"), str):
        inferred_stage = str(milestone_entry["stage"])
    stage = args.stage or inferred_stage
    if not isinstance(stage, str) or not stage:
        raise ValueError(f"Could not resolve stage for milestone {args.milestone_id!r}. Pass --stage explicitly.")

    checkpoint_path = _resolve_checkpoint_path(
        stage=stage,
        load_run=args.load_run,
        checkpoint=args.checkpoint,
        checkpoint_path=args.checkpoint_path,
    )

    record_kind = args.evidence_record_kind
    if record_kind == "auto":
        if isinstance(milestone_entry, dict) and isinstance(milestone_entry.get("evidence_record_kind"), str):
            record_kind = str(milestone_entry["evidence_record_kind"])
        else:
            record_kind = "manual"

    evidence = None
    if record_kind != "manual":
        if checkpoint_path is None:
            evidence = _compact_record(_latest_stage_evidence_record(record_kind=record_kind, stage=stage))
        else:
            evidence = _compact_record(
                _latest_evidence_record(
                    record_kind=record_kind,
                    checkpoint_path=str(checkpoint_path),
                    stage=stage,
                )
            )

    payload = record_lineage_milestone(
        milestone_id=args.milestone_id,
        checkpoint_path=None if checkpoint_path is None else str(checkpoint_path),
        run_name=args.load_run,
        stage=stage,
        status=args.status,
        evidence=evidence,
        manual_review=args.manual_review,
        notes=args.note,
        recorded_by=args.recorded_by,
    )

    result = {
        "lineage_name": payload.get("lineage_name"),
        "current_target_milestone": payload.get("current_target_milestone"),
        "milestone": payload.get("milestone_registry", {}).get(args.milestone_id),
    }
    if args.print_json:
        print(json.dumps(result, indent=2, sort_keys=True), flush=True)
        return
    print(
        f"[MILESTONE] {args.milestone_id} -> {args.status} "
        f"checkpoint={checkpoint_path or 'none'} next={payload.get('current_target_milestone')}",
        flush=True,
    )
    if evidence is not None:
        print(f"[MILESTONE] evidence={record_kind} recorded_at={evidence.get('recorded_at')}", flush=True)
    elif record_kind != "manual":
        print(f"[MILESTONE] no {record_kind} ledger entry matched the checkpoint; recorded without attached evidence", flush=True)


if __name__ == "__main__":
    main()
