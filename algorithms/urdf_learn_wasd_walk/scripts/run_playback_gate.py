from __future__ import annotations

import argparse
import json
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from algorithms.urdf_learn_wasd_walk.asset_paths import outputs_dir
from algorithms.urdf_learn_wasd_walk.training_lineage import load_active_lineage, resolve_landau_experiment_name


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _timestamp_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _geo_path() -> Path:
    return _repo_root() / "geo"


def _workflow_dir() -> Path:
    path = outputs_dir() / "history" / "workflows"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _history_dir() -> Path:
    path = outputs_dir() / "history"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _write_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(_json_ready(payload), indent=2, sort_keys=True), encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except Exception:
            continue
        if isinstance(payload, dict):
            records.append(payload)
    return records


def _latest_record_after(
    path: Path,
    *,
    recorded_after: str,
    record_kind: str,
    experiment_name: str,
    workflow_id: str,
) -> dict[str, Any] | None:
    matching = []
    for record in _read_jsonl(path):
        if record.get("record_kind") != record_kind:
            continue
        if record.get("recorded_at", "") < recorded_after:
            continue
        task = record.get("task", {})
        if not isinstance(task, dict) or task.get("experiment_name") != experiment_name:
            continue
        if record.get("workflow_id") != workflow_id:
            continue
        matching.append(record)
    if not matching:
        return None
    matching.sort(key=lambda record: str(record.get("recorded_at", "")))
    return matching[-1]


def _run_step(
    name: str,
    command: list[str],
    *,
    experiment_name: str,
    ledger_name: str,
    record_kind: str,
    workflow_id: str,
) -> dict[str, Any]:
    started_at = _timestamp()
    completed = subprocess.run(command, cwd=_repo_root(), check=False)
    record = _latest_record_after(
        _history_dir() / ledger_name,
        recorded_after=started_at,
        record_kind=record_kind,
        experiment_name=experiment_name,
        workflow_id=workflow_id,
    )
    return {
        "name": name,
        "command": command,
        "cwd": str(_repo_root()),
        "started_at": started_at,
        "finished_at": _timestamp(),
        "exit_code": completed.returncode,
        "record": record,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the long-horizon Landau playback gate and write a workflow manifest."
    )
    parser.add_argument("--robot", choices=("landau",), default="landau")
    parser.add_argument("--stage", choices=("stand", "fwd_only", "fwd_yaw", "game", "full"), default="game")
    parser.add_argument("--load_run", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--latest", action="store_true", default=False)
    parser.add_argument("--experiment_name", type=str, default=None)
    parser.add_argument("--terrain-mode", choices=("flat", "game"), default="flat")
    parser.add_argument("--obstacles", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--policy-steps", type=int, default=120)
    parser.add_argument("--policy-max-done-count", type=int, default=0)
    parser.add_argument("--policy-min-control-root-height", type=float, default=0.15)
    parser.add_argument("--stand-steps", type=int, default=120)
    parser.add_argument("--walk-steps-limit", type=int, default=2400)
    parser.add_argument("--hold-steps", type=int, default=240)
    parser.add_argument("--hold-action-mode", choices=("policy", "default_pose"), default="policy")
    parser.add_argument("--path-file", type=str, default=None)
    parser.add_argument("--path-preset", choices=("gate", "target", "triangle", "square"), default="gate")
    parser.add_argument("--gate-direction", choices=("forward", "left", "right", "backward"), default="forward")
    parser.add_argument("--path-distance", type=float, default=10.0)
    parser.add_argument("--target-x", type=float, default=0.0)
    parser.add_argument("--target-y", type=float, default=10.0)
    parser.add_argument("--path-arrival-radius", type=float, default=0.35)
    parser.add_argument("--path-slow-radius", type=float, default=1.0)
    parser.add_argument("--path-max-forward", type=float, default=0.7)
    parser.add_argument("--path-max-yaw", type=float, default=0.9)
    parser.add_argument("--path-edge-length", type=float, default=3.5)
    parser.add_argument("--path-clockwise", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--path-close-loop", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--run-tag", type=str, default="playback_gate")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    geo_path = _geo_path()
    if not geo_path.is_file():
        raise FileNotFoundError(f"geo launcher not found: {geo_path}")

    workflow_path = _workflow_dir() / f"{_timestamp_slug()}_{args.run_tag}.json"
    experiment_name = resolve_landau_experiment_name(args.stage, args.experiment_name)
    workflow_id = uuid.uuid4().hex
    active_lineage = load_active_lineage()
    acceptance = active_lineage.get("acceptance", {}) if isinstance(active_lineage, dict) else {}
    if args.path_distance == 10.0 and isinstance(acceptance.get("path_follow_distance_m"), (int, float)):
        args.path_distance = float(acceptance["path_follow_distance_m"])
    if args.target_y == 10.0 and isinstance(acceptance.get("path_follow_distance_m"), (int, float)):
        args.target_y = float(acceptance["path_follow_distance_m"])
    if args.policy_max_done_count == 0 and acceptance.get("require_zero_falls") is False:
        args.policy_max_done_count = 999999

    manifest: dict[str, Any] = {
        "record_kind": "workflow",
        "workflow_name": "landau_playback_gate",
        "workflow_id": workflow_id,
        "recorded_at": _timestamp(),
        "repo_root": str(_repo_root()),
        "geo_path": str(geo_path),
        "cli_args": vars(args),
        "experiment_name": experiment_name,
        "active_lineage": active_lineage,
        "status": "running",
        "steps": [],
    }
    _write_manifest(workflow_path, manifest)

    shared_args = [
        "--robot",
        args.robot,
        "--stage",
        args.stage,
        "--headless",
        "--terrain-mode",
        args.terrain_mode,
        f"--{'obstacles' if args.obstacles else 'no-obstacles'}",
        "--workflow-id",
        workflow_id,
    ]
    if args.load_run:
        shared_args.extend(["--load_run", args.load_run])
    if args.checkpoint:
        shared_args.extend(["--checkpoint", args.checkpoint])
    if args.latest:
        shared_args.append("--latest")
    if args.experiment_name:
        shared_args.extend(["--experiment_name", args.experiment_name])

    diag_command = [
        str(geo_path),
        "walk",
        "diagnose",
        *shared_args,
        "--action-mode",
        "policy",
        "--steps",
        str(args.policy_steps),
        "--max-done-count",
        str(args.policy_max_done_count),
        "--min-control-root-height",
        str(args.policy_min_control_root_height),
    ]
    diag_step = _run_step(
        "policy_idle_gate",
        diag_command,
        experiment_name=experiment_name,
        ledger_name="diagnostic_runs.jsonl",
        record_kind="diagnostic",
        workflow_id=workflow_id,
    )
    manifest["steps"].append(diag_step)
    _write_manifest(workflow_path, manifest)

    eval_command = [
        str(geo_path),
        "walk",
        "eval",
        *shared_args,
        "--stand-steps",
        str(args.stand_steps),
        "--walk-steps-limit",
        str(args.walk_steps_limit),
        "--hold-steps",
        str(args.hold_steps),
        "--hold-action-mode",
        args.hold_action_mode,
        "--path-preset",
        args.path_preset,
        "--gate-direction",
        args.gate_direction,
        "--path-distance",
        str(args.path_distance),
        "--target-x",
        str(args.target_x),
        "--target-y",
        str(args.target_y),
        "--path-arrival-radius",
        str(args.path_arrival_radius),
        "--path-slow-radius",
        str(args.path_slow_radius),
        "--path-max-forward",
        str(args.path_max_forward),
        "--path-max-yaw",
        str(args.path_max_yaw),
        "--path-edge-length",
        str(args.path_edge_length),
        f"--{'path-clockwise' if args.path_clockwise else 'no-path-clockwise'}",
        f"--{'path-close-loop' if args.path_close_loop else 'no-path-close-loop'}",
    ]
    if args.path_file:
        eval_command.extend(["--path-file", args.path_file])
    eval_step = _run_step(
        "stand_walk_hold_gate",
        eval_command,
        experiment_name=experiment_name,
        ledger_name="evaluation_runs.jsonl",
        record_kind="evaluation",
        workflow_id=workflow_id,
    )
    manifest["steps"].append(eval_step)

    policy_record = diag_step.get("record")
    evaluation_record = eval_step.get("record")
    manifest["summary"] = {
        "selected_checkpoint_path": (
            (policy_record or {}).get("checkpoint_path")
            or (evaluation_record or {}).get("checkpoint_path")
        ),
        "policy_idle_gate": policy_record,
        "stand_walk_hold_gate": evaluation_record,
    }

    manifest["status"] = "completed"
    for step in manifest["steps"]:
        record = step.get("record") or {}
        if step.get("exit_code") != 0 or record.get("status") == "failed":
            manifest["status"] = "failed"
            break
    _write_manifest(workflow_path, manifest)
    print(f"[WORKFLOW] wrote manifest to {workflow_path}", flush=True)
    if manifest["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
