from __future__ import annotations

import argparse
import json
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from algorithms.urdf_learn_wasd_walk.asset_paths import outputs_dir
from algorithms.urdf_learn_wasd_walk.training_lineage import (
    load_active_lineage,
    reset_landau_training_state,
    resolve_landau_experiment_name,
    update_lineage_stage_state,
)


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _timestamp_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _default_isaaclab_sh() -> Path:
    return _repo_root().parent / "IsaacLab" / "isaaclab.sh"


def _workflow_dir() -> Path:
    path = outputs_dir() / "history" / "workflows"
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


def _latest_training_record(run_name: str, stage: str, workflow_id: str) -> dict[str, Any] | None:
    history_path = outputs_dir() / "history" / "training_runs.jsonl"
    for record in reversed(_read_jsonl(history_path)):
        cli_args = record.get("cli_args", {})
        if record.get("record_kind") != "training":
            continue
        if record.get("status") not in {"completed", "failed"}:
            continue
        if cli_args.get("robot") != "landau":
            continue
        if cli_args.get("stage") != stage:
            continue
        if cli_args.get("run_name") != run_name:
            continue
        if record.get("workflow_id") != workflow_id:
            continue
        return record
    return None


def _latest_validation_record(checkpoint_path: str, workflow_id: str) -> dict[str, Any] | None:
    history_path = outputs_dir() / "history" / "validation_runs.jsonl"
    target = str(Path(checkpoint_path).resolve())
    for record in reversed(_read_jsonl(history_path)):
        if record.get("record_kind") != "validation":
            continue
        if record.get("workflow_id") != workflow_id:
            continue
        if str(record.get("checkpoint_path")) == target:
            return record
    return None


def _latest_evaluation_record(checkpoint_path: str, workflow_id: str) -> dict[str, Any] | None:
    history_path = outputs_dir() / "history" / "evaluation_runs.jsonl"
    target = str(Path(checkpoint_path).resolve())
    for record in reversed(_read_jsonl(history_path)):
        if record.get("record_kind") != "evaluation":
            continue
        if record.get("workflow_id") != workflow_id:
            continue
        if str(record.get("checkpoint_path")) == target:
            return record
    return None


def _latest_diagnostic_record(checkpoint_path: str | None, scenario: str, workflow_id: str) -> dict[str, Any] | None:
    history_path = outputs_dir() / "history" / "diagnostic_runs.jsonl"
    target = None if checkpoint_path is None else str(Path(checkpoint_path).resolve())
    for record in reversed(_read_jsonl(history_path)):
        if record.get("record_kind") != "diagnostic":
            continue
        if record.get("scenario") != scenario:
            continue
        if record.get("workflow_id") != workflow_id:
            continue
        if target is None:
            if record.get("checkpoint_path") is None:
                return record
            continue
        if str(record.get("checkpoint_path")) == target:
            return record
    return None


def _write_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(_json_ready(payload), indent=2, sort_keys=True), encoding="utf-8")


def _run_step(name: str, command: list[str]) -> dict[str, Any]:
    started_at = _timestamp()
    completed = subprocess.run(command, cwd=_repo_root(), check=False)
    return {
        "name": name,
        "command": command,
        "cwd": str(_repo_root()),
        "started_at": started_at,
        "finished_at": _timestamp(),
        "exit_code": completed.returncode,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the Landau staged curriculum serially and write a workflow manifest for reproducibility."
    )
    parser.add_argument("--isaaclab-sh", type=str, default=str(_default_isaaclab_sh()))
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-tag", type=str, default="curriculum")
    parser.add_argument("--from-start", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--lineage-name", type=str, default=None)
    parser.add_argument("--archive-tag", type=str, default=None)
    parser.add_argument("--note", type=str, default=None)
    parser.add_argument("--smoke-steps", type=int, default=16)
    parser.add_argument("--stand-iters", type=int, default=250)
    parser.add_argument("--fwd-only-iters", type=int, default=200)
    parser.add_argument("--fwd-yaw-iters", type=int, default=150)
    parser.add_argument("--game-iters", type=int, default=200)
    parser.add_argument("--accept-stage", choices=("stand", "fwd_only", "fwd_yaw", "game"), default=None)
    parser.add_argument("--accept-distance", type=float, default=10.0)
    parser.add_argument(
        "--accept-terrain-mode",
        "--game-eval-terrain-mode",
        dest="accept_terrain_mode",
        choices=("flat", "game"),
        default=None,
    )
    parser.add_argument("--skip-smoke", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--skip-validation", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--skip-diagnostics", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--skip-evaluation", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--stop-on-validation-failure", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--stop-on-diagnostic-failure", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--stop-on-evaluation-failure", action=argparse.BooleanOptionalAction, default=False)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    isaaclab_sh = Path(args.isaaclab_sh).expanduser().resolve()
    if not isaaclab_sh.is_file():
        raise FileNotFoundError(f"isaaclab.sh not found: {isaaclab_sh}")

    reset_summary = None
    if args.from_start:
        reset_summary = reset_landau_training_state(
            lineage_name=args.lineage_name,
            archive_tag=args.archive_tag or args.run_tag,
            note=args.note,
        )
    active_lineage = load_active_lineage()
    acceptance = active_lineage.get("acceptance", {}) if isinstance(active_lineage, dict) else {}
    if args.accept_stage is None and isinstance(acceptance.get("accept_stage"), str):
        args.accept_stage = str(acceptance["accept_stage"])
    if args.accept_stage is None:
        args.accept_stage = "fwd_only"
    if args.accept_terrain_mode is None and isinstance(acceptance.get("accept_terrain_mode"), str):
        args.accept_terrain_mode = str(acceptance["accept_terrain_mode"])
    if args.accept_terrain_mode is None:
        args.accept_terrain_mode = "flat"
    if args.accept_distance == 10.0 and isinstance(acceptance.get("path_follow_distance_m"), (int, float)):
        args.accept_distance = float(acceptance["path_follow_distance_m"])
    if args.accept_stage != "stand" and args.skip_evaluation:
        raise ValueError("--skip-evaluation is incompatible with --accept-stage other than 'stand'.")
    if args.accept_stage == "stand" and args.skip_diagnostics:
        raise ValueError("--skip-diagnostics is incompatible with --accept-stage stand.")
    workflow_id = uuid.uuid4().hex

    workflow_path = _workflow_dir() / f"{_timestamp_slug()}_{args.run_tag}.json"
    manifest: dict[str, Any] = {
        "record_kind": "workflow",
        "workflow_name": "landau_curriculum",
        "workflow_id": workflow_id,
        "recorded_at": _timestamp(),
        "repo_root": str(_repo_root()),
        "isaaclab_sh": str(isaaclab_sh),
        "cli_args": vars(args),
        "reset_summary": reset_summary,
        "active_lineage": active_lineage,
        "stages": [],
        "status": "running",
    }
    _write_manifest(workflow_path, manifest)

    stage_iterations = {
        "stand": int(args.stand_iters),
        "fwd_only": int(args.fwd_only_iters),
        "fwd_yaw": int(args.fwd_yaw_iters),
        "game": int(args.game_iters),
    }
    if stage_iterations.get(args.accept_stage, 0) <= 0:
        raise ValueError(f"--accept-stage {args.accept_stage!r} requires a positive iteration count.")
    previous_checkpoint: str | None = None
    acceptance_completed = False

    for stage_name in ("stand", "fwd_only", "fwd_yaw", "game"):
        iterations = stage_iterations[stage_name]
        if iterations <= 0:
            continue

        run_name = f"{args.run_tag}_{stage_name}"
        stage_record: dict[str, Any] = {
            "stage": stage_name,
            "experiment_name": resolve_landau_experiment_name(stage_name),
            "run_name": run_name,
            "iterations": iterations,
            "resume_checkpoint": previous_checkpoint,
        }
        manifest["stages"].append(stage_record)
        update_lineage_stage_state(
            stage=stage_name,
            status="running",
            workflow_id=workflow_id,
            run_name=run_name,
            selected_checkpoint=previous_checkpoint,
        )
        _write_manifest(workflow_path, manifest)

        if not args.skip_smoke:
            smoke_command = [
                str(isaaclab_sh),
                "-p",
                "-m",
                "algorithms.urdf_learn_wasd_walk.scripts.smoke_test",
                "--robot",
                "landau",
                "--stage",
                stage_name,
                "--headless",
                "--steps",
                str(args.smoke_steps),
            ]
            stage_record["smoke"] = _run_step(f"{stage_name}_smoke", smoke_command)
            _write_manifest(workflow_path, manifest)
            if stage_record["smoke"]["exit_code"] != 0:
                update_lineage_stage_state(
                    stage=stage_name,
                    status="failed",
                    workflow_id=workflow_id,
                    run_name=run_name,
                    selected_checkpoint=previous_checkpoint,
                    failure={"step": "smoke", "exit_code": stage_record["smoke"]["exit_code"]},
                )
                manifest["status"] = "failed"
                manifest["failed_stage"] = stage_name
                _write_manifest(workflow_path, manifest)
                raise SystemExit(stage_record["smoke"]["exit_code"])

        train_command = [
            str(isaaclab_sh),
            "-p",
            "-m",
            "algorithms.urdf_learn_wasd_walk.scripts.train",
            "--robot",
            "landau",
            "--stage",
            stage_name,
            "--headless",
            "--num_envs",
            str(args.num_envs),
            "--max_iterations",
            str(iterations),
            "--seed",
            str(args.seed),
            "--run_name",
            run_name,
            "--workflow-id",
            workflow_id,
        ]
        if previous_checkpoint is not None:
            train_command.extend(["--resume-checkpoint", previous_checkpoint, "--reset_optimizer"])

        stage_record["train"] = _run_step(f"{stage_name}_train", train_command)
        stage_record["train_record"] = _latest_training_record(run_name, stage_name, workflow_id)
        _write_manifest(workflow_path, manifest)

        train_record = stage_record["train_record"]
        if stage_record["train"]["exit_code"] != 0 or not isinstance(train_record, dict):
            update_lineage_stage_state(
                stage=stage_name,
                status="failed",
                workflow_id=workflow_id,
                run_name=run_name,
                selected_checkpoint=previous_checkpoint,
                failure={"step": "train", "exit_code": stage_record["train"]["exit_code"] or 1},
            )
            manifest["status"] = "failed"
            manifest["failed_stage"] = stage_name
            _write_manifest(workflow_path, manifest)
            raise SystemExit(stage_record["train"]["exit_code"] or 1)

        latest_checkpoint = train_record.get("checkpoints", {}).get("latest")
        if not latest_checkpoint:
            update_lineage_stage_state(
                stage=stage_name,
                status="failed",
                workflow_id=workflow_id,
                run_name=run_name,
                failure={"step": "train", "reason": "no_checkpoint_recorded"},
            )
            manifest["status"] = "failed"
            manifest["failed_stage"] = stage_name
            manifest["failure"] = f"No checkpoint was recorded for stage '{stage_name}'."
            _write_manifest(workflow_path, manifest)
            raise SystemExit(1)

        previous_checkpoint = str(latest_checkpoint)
        stage_record["selected_checkpoint"] = previous_checkpoint
        update_lineage_stage_state(
            stage=stage_name,
            status="trained",
            workflow_id=workflow_id,
            run_name=run_name,
            selected_checkpoint=previous_checkpoint,
            produced_checkpoint=previous_checkpoint,
            last_training=train_record,
        )

        if stage_name == "stand":
            if args.skip_diagnostics:
                if stage_name == args.accept_stage:
                    manifest["status"] = "failed"
                    manifest["failed_stage"] = stage_name
                    manifest["failure"] = "acceptance stage skipped diagnostics"
                    _write_manifest(workflow_path, manifest)
                    raise SystemExit(1)
                continue

            diagnose_policy_command = [
                str(isaaclab_sh),
                "-p",
                "-m",
                "algorithms.urdf_learn_wasd_walk.scripts.check_pose_stability",
                "--robot",
                "landau",
                "--stage",
                stage_name,
                "--headless",
                "--workflow-id",
                workflow_id,
                "--load_run",
                Path(train_record["log_dir"]).name,
                "--checkpoint",
                Path(previous_checkpoint).name,
                "--action-mode",
                "policy",
                "--steps",
                "600",
                "--max-done-count",
                "0",
                "--min-control-root-height",
                "0.17",
            ]
            stage_record["diagnose_policy"] = _run_step(f"{stage_name}_diagnose_policy", diagnose_policy_command)
            stage_record["diagnose_record"] = _latest_diagnostic_record(previous_checkpoint, "pose_stability", workflow_id)
            update_lineage_stage_state(
                stage=stage_name,
                status="completed" if stage_record["diagnose_policy"]["exit_code"] == 0 else "failed",
                workflow_id=workflow_id,
                run_name=run_name,
                selected_checkpoint=previous_checkpoint,
                produced_checkpoint=previous_checkpoint,
                last_training=train_record,
                last_diagnostic=stage_record["diagnose_record"],
                achieved_capability={
                    "diagnostic_exit_code": stage_record["diagnose_policy"]["exit_code"],
                    "diagnostic_status": None
                    if not isinstance(stage_record["diagnose_record"], dict)
                    else stage_record["diagnose_record"].get("status"),
                },
                failure=None
                if stage_record["diagnose_policy"]["exit_code"] == 0
                else {"step": "diagnostic", "exit_code": stage_record["diagnose_policy"]["exit_code"]},
            )
            _write_manifest(workflow_path, manifest)
            if stage_record["diagnose_policy"]["exit_code"] != 0 and (
                args.stop_on_diagnostic_failure or stage_name == args.accept_stage
            ):
                manifest["status"] = "failed"
                manifest["failed_stage"] = stage_name
                _write_manifest(workflow_path, manifest)
                raise SystemExit(stage_record["diagnose_policy"]["exit_code"])
            if stage_name == args.accept_stage:
                acceptance_completed = True
                break
            continue

        if args.skip_validation:
            if stage_name == args.accept_stage:
                manifest["status"] = "failed"
                manifest["failed_stage"] = stage_name
                manifest["failure"] = "acceptance stage skipped validation"
                _write_manifest(workflow_path, manifest)
                raise SystemExit(1)
            continue

        run_dir_name = Path(train_record["log_dir"]).name
        validate_command = [
            str(isaaclab_sh),
            "-p",
            "-m",
            "algorithms.urdf_learn_wasd_walk.scripts.validate_walk",
            "--robot",
            "landau",
            "--stage",
            stage_name,
            "--headless",
            "--workflow-id",
            workflow_id,
            "--load_run",
            run_dir_name,
            "--checkpoint",
            Path(previous_checkpoint).name,
        ]
        stage_record["validate"] = _run_step(f"{stage_name}_validate", validate_command)
        stage_record["validation_record"] = _latest_validation_record(previous_checkpoint, workflow_id)
        update_lineage_stage_state(
            stage=stage_name,
            status="validated" if stage_record["validate"]["exit_code"] == 0 else "failed",
            workflow_id=workflow_id,
            run_name=run_name,
            selected_checkpoint=previous_checkpoint,
            produced_checkpoint=previous_checkpoint,
            last_training=train_record,
            last_validation=stage_record["validation_record"],
            achieved_capability={
                "validation_exit_code": stage_record["validate"]["exit_code"],
                "validation_status": None
                if not isinstance(stage_record["validation_record"], dict)
                else stage_record["validation_record"].get("status"),
            },
            failure=None
            if stage_record["validate"]["exit_code"] == 0
            else {"step": "validation", "exit_code": stage_record["validate"]["exit_code"]},
        )
        _write_manifest(workflow_path, manifest)

        if stage_record["validate"]["exit_code"] != 0 and (
            args.stop_on_validation_failure or stage_name == args.accept_stage
        ):
            manifest["status"] = "failed"
            manifest["failed_stage"] = stage_name
            _write_manifest(workflow_path, manifest)
            raise SystemExit(stage_record["validate"]["exit_code"])

        if stage_name == "game" and not args.skip_diagnostics:
            diagnose_command = [
                str(isaaclab_sh),
                "-p",
                "-m",
                "algorithms.urdf_learn_wasd_walk.scripts.check_pose_stability",
                "--robot",
                "landau",
                "--stage",
                stage_name,
                "--headless",
                "--workflow-id",
                workflow_id,
                "--action-mode",
                "zero",
                "--steps",
                "120",
                "--max-done-count",
                "0",
                "--min-control-root-height",
                "0.15",
            ]
            stage_record["diagnose_reset"] = _run_step(f"{stage_name}_diagnose_reset", diagnose_command)
            stage_record["diagnose_record"] = _latest_diagnostic_record(None, "pose_stability", workflow_id)
            update_lineage_stage_state(
                stage=stage_name,
                status="validated" if stage_record["diagnose_reset"]["exit_code"] == 0 else "failed",
                workflow_id=workflow_id,
                run_name=run_name,
                selected_checkpoint=previous_checkpoint,
                produced_checkpoint=previous_checkpoint,
                last_training=train_record,
                last_validation=stage_record.get("validation_record"),
                last_diagnostic=stage_record["diagnose_record"],
                failure=None
                if stage_record["diagnose_reset"]["exit_code"] == 0
                else {"step": "diagnostic", "exit_code": stage_record["diagnose_reset"]["exit_code"]},
            )
            _write_manifest(workflow_path, manifest)
            if stage_record["diagnose_reset"]["exit_code"] != 0 and (
                args.stop_on_diagnostic_failure or stage_name == args.accept_stage
            ):
                manifest["status"] = "failed"
                manifest["failed_stage"] = stage_name
                _write_manifest(workflow_path, manifest)
                raise SystemExit(stage_record["diagnose_reset"]["exit_code"])

        if stage_name != args.accept_stage:
            continue

        if args.skip_evaluation:
            manifest["status"] = "failed"
            manifest["failed_stage"] = stage_name
            manifest["failure"] = "acceptance stage skipped evaluation"
            _write_manifest(workflow_path, manifest)
            raise SystemExit(1)

        evaluate_command = [
            str(isaaclab_sh),
            "-p",
            "-m",
            "algorithms.urdf_learn_wasd_walk.scripts.evaluate_policy",
            "--robot",
            "landau",
            "--stage",
            stage_name,
            "--headless",
            "--workflow-id",
            workflow_id,
            "--load_run",
            run_dir_name,
            "--checkpoint",
            Path(previous_checkpoint).name,
            "--stand-steps",
            "120",
            "--walk-steps-limit",
            "2400",
            "--hold-steps",
            "120",
            "--target-y",
            str(args.accept_distance),
        ]
        if stage_name == "game":
            evaluate_command.extend(["--terrain-mode", args.accept_terrain_mode])
        stage_record["evaluate"] = _run_step(f"{stage_name}_evaluate", evaluate_command)
        stage_record["evaluation_record"] = _latest_evaluation_record(previous_checkpoint, workflow_id)
        update_lineage_stage_state(
            stage=stage_name,
            status="completed" if stage_record["evaluate"]["exit_code"] == 0 else "failed",
            workflow_id=workflow_id,
            run_name=run_name,
            selected_checkpoint=previous_checkpoint,
            produced_checkpoint=previous_checkpoint,
            last_training=train_record,
            last_validation=stage_record.get("validation_record"),
            last_diagnostic=stage_record.get("diagnose_record"),
            last_evaluation=stage_record["evaluation_record"],
            achieved_capability=None
            if not isinstance(stage_record["evaluation_record"], dict)
            else stage_record["evaluation_record"].get("metrics"),
            failure=None
            if stage_record["evaluate"]["exit_code"] == 0
            else {"step": "evaluation", "exit_code": stage_record["evaluate"]["exit_code"]},
        )
        _write_manifest(workflow_path, manifest)
        if stage_record["evaluate"]["exit_code"] != 0 and (
            args.stop_on_evaluation_failure or stage_name == args.accept_stage
        ):
            manifest["status"] = "failed"
            manifest["failed_stage"] = stage_name
            _write_manifest(workflow_path, manifest)
            raise SystemExit(stage_record["evaluate"]["exit_code"])
        acceptance_completed = True
        break

    if not acceptance_completed:
        manifest["status"] = "failed"
        manifest["failure"] = f"acceptance stage '{args.accept_stage}' did not complete"
        _write_manifest(workflow_path, manifest)
        raise SystemExit(1)
    manifest["status"] = "completed"
    _write_manifest(workflow_path, manifest)
    print(f"[WORKFLOW] wrote manifest to {workflow_path}", flush=True)


if __name__ == "__main__":
    main()
