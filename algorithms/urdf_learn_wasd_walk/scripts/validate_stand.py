from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from algorithms.urdf_learn_wasd_walk.run_history import history_dir


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the passive or policy stand validation scenario and return the matching diagnostic record."
    )
    parser.add_argument("--robot", default="landau")
    parser.add_argument("--action-mode", choices=("zero", "policy"), default="zero")
    parser.add_argument("--steps", type=int, default=900)
    parser.add_argument("--max-done-count", type=int, default=0)
    parser.add_argument("--min-control-root-height", type=float, default=0.17)
    parser.add_argument("--load_run", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--latest", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--playback-compat-mode", choices=("strict", "control_only", "off"), default="strict")
    parser.add_argument("--workflow-id", type=str, default=None)
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    return parser


def _resolve_workflow_id(explicit_id: str | None) -> str:
    if explicit_id:
        return explicit_id
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"stand_validate_{timestamp}"


def _latest_matching_record(workflow_id: str) -> dict[str, Any] | None:
    records = _read_jsonl(history_dir() / "diagnostic_runs.jsonl")
    for record in reversed(records):
        if record.get("record_kind") != "diagnostic":
            continue
        if record.get("workflow_id") == workflow_id:
            return record
    return None


def main() -> None:
    args = _build_parser().parse_args()
    workflow_id = _resolve_workflow_id(args.workflow_id)
    command = [
        str(_repo_root() / "geo"),
        "walk",
        "diagnose",
        "--robot",
        args.robot,
        "--stage",
        "stand",
        "--workflow-id",
        workflow_id,
        "--action-mode",
        args.action_mode,
        "--steps",
        str(args.steps),
        "--max-done-count",
        str(args.max_done_count),
        "--min-control-root-height",
        str(args.min_control_root_height),
        "--playback-compat-mode",
        args.playback_compat_mode,
    ]
    if args.headless:
        command.append("--headless")
    if args.load_run:
        command.extend(["--load_run", args.load_run])
    if args.checkpoint:
        command.extend(["--checkpoint", args.checkpoint])
    if args.latest:
        command.append("--latest")

    completed = subprocess.run(command, check=False)
    record = _latest_matching_record(workflow_id)
    if record is None:
        raise RuntimeError(f"No diagnostic record found for workflow_id={workflow_id!r}.")

    print(json.dumps(record, indent=2, sort_keys=True), flush=True)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
