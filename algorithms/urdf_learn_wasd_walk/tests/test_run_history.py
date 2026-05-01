from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import algorithms.urdf_learn_wasd_walk.run_history as run_history
import algorithms.urdf_learn_wasd_walk.training_lineage as training_lineage


def _write_jsonl(path: Path, payloads: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(payload, sort_keys=True) + "\n" for payload in payloads),
        encoding="utf-8",
    )


def test_refresh_history_refs_builds_experiment_index() -> None:
    with TemporaryDirectory() as temp_dir:
        temp_root = Path(temp_dir)
        original_outputs_dir = run_history.outputs_dir
        original_lineage_outputs_dir = training_lineage.outputs_dir
        try:
            run_history.outputs_dir = lambda: temp_root
            training_lineage.outputs_dir = lambda: temp_root
            history_root = run_history.history_dir()
            (history_root / "active_lineage.json").write_text(
                json.dumps(
                    {
                        "lineage_name": "restart_20260418_rebuild",
                        "current_stage": "stand",
                        "current_run": "rebuild_stand_tuned_smoke_40",
                        "current_checkpoint": "/tmp/model_39.pt",
                        "current_workflow_id": "wf_curriculum",
                        "promoted_default": None,
                        "stage_state": {
                            "stand": {
                                "status": "trained",
                                "selected_checkpoint": "/tmp/model_39.pt",
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            (history_root / "checkpoint_registry.json").write_text(
                json.dumps(
                    {
                        "experiments": {
                            "geo_landau_game": {
                                "recommended": {
                                    "checkpoint_path": "/tmp/model_150.pt",
                                    "load_run": "2026-04-18_game_v11",
                                }
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            _write_jsonl(
                history_root / "training_runs.jsonl",
                [
                    {
                        "recorded_at": "2026-04-18T05:00:00+00:00",
                        "task": {"experiment_name": "geo_landau_game"},
                        "status": "completed",
                        "log_dir": "/tmp/train_a",
                    },
                    {
                        "recorded_at": "2026-04-18T05:05:00+00:00",
                        "task": {"experiment_name": "geo_landau_game"},
                        "status": "started",
                        "log_dir": "/tmp/train_b",
                    }
                ],
            )
            _write_jsonl(
                history_root / "validation_runs.jsonl",
                [
                    {
                        "recorded_at": "2026-04-18T05:10:00+00:00",
                        "task": {"experiment_name": "geo_landau_game"},
                        "status": "failed",
                        "checkpoint_path": "/tmp/model_150.pt",
                    }
                ],
            )
            _write_jsonl(
                history_root / "evaluation_runs.jsonl",
                [
                    {
                        "recorded_at": "2026-04-18T05:20:00+00:00",
                        "task": {"experiment_name": "geo_landau_game"},
                        "status": "failed",
                        "checkpoint_path": "/tmp/model_150.pt",
                    }
                ],
            )
            _write_jsonl(
                history_root / "diagnostic_runs.jsonl",
                [
                    {
                        "recorded_at": "2026-04-18T05:30:00+00:00",
                        "task": {"experiment_name": "geo_landau_game"},
                        "status": "failed",
                    }
                ],
            )

            payload = run_history.refresh_history_refs()
        finally:
            run_history.outputs_dir = original_outputs_dir
            training_lineage.outputs_dir = original_lineage_outputs_dir

    experiment_payload = payload["experiments"]["geo_landau_game"]
    assert experiment_payload["recommended"]["load_run"] == "2026-04-18_game_v11"
    assert experiment_payload["latest_training"]["log_dir"] == "/tmp/train_a"
    assert experiment_payload["latest_training_event"]["log_dir"] == "/tmp/train_b"
    assert experiment_payload["latest_training_terminal"]["log_dir"] == "/tmp/train_a"
    assert experiment_payload["latest_validation"]["checkpoint_path"] == "/tmp/model_150.pt"
    assert experiment_payload["latest_evaluation"]["status"] == "failed"
    assert experiment_payload["latest_diagnostic"]["status"] == "failed"
    assert experiment_payload["experiment_ref_path"].endswith("geo_landau_game.json")
    assert payload["current"]["stage"] == "stand"
    assert payload["current"]["checkpoint_path"] == "/tmp/model_39.pt"
    assert payload["active_lineage"]["lineage_name"] == "restart_20260418_rebuild"
    assert payload["training_rule"]["goal_order"][0] == "stand_zero_signal_30s_no_reset"
    assert payload["current_target_milestone"] == "stand_zero_signal_30s_no_reset"
    assert payload["milestone_registry"]["gate_10m_no_reset"]["status"] == "pending"
    assert payload["stage_state"]["stand"]["status"] == "trained"


def test_resolve_recommended_checkpoint_does_not_cross_fallback_to_legacy_stage() -> None:
    with TemporaryDirectory() as temp_dir:
        temp_root = Path(temp_dir)
        archived_checkpoint_path = (
            temp_root
            / "archives"
            / "20260418T075338Z_restart_20260418"
            / "logs"
            / "rsl_rl"
            / "geo_landau_game"
            / "run_a"
            / "model_150.pt"
        )
        archived_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        archived_checkpoint_path.write_text("", encoding="utf-8")
        stale_live_checkpoint_path = temp_root / "logs" / "rsl_rl" / "geo_landau_game" / "run_a" / "model_150.pt"

        original_outputs_dir = run_history.outputs_dir
        try:
            run_history.outputs_dir = lambda: temp_root
            archived_registry = temp_root / "archives" / "20260418T075338Z_restart_20260418" / "history" / "checkpoint_registry.json"
            archived_registry.parent.mkdir(parents=True, exist_ok=True)
            archived_registry.write_text(
                json.dumps(
                    {
                        "experiments": {
                            "geo_landau_game": {
                                "recommended": {
                                    "checkpoint_path": str(stale_live_checkpoint_path),
                                    "load_run": "2026-04-18_10-41-48_game_v11_zero_init_200",
                                    "reason": "archived promoted fallback",
                                }
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )

            resolved = run_history.resolve_recommended_checkpoint("geo_landau_restart_20260418_rebuild_armswing_game")
        finally:
            run_history.outputs_dir = original_outputs_dir

    assert resolved is None


def test_resolve_recommended_checkpoint_uses_legacy_stage_when_requested_explicitly() -> None:
    with TemporaryDirectory() as temp_dir:
        temp_root = Path(temp_dir)
        archived_checkpoint_path = (
            temp_root
            / "archives"
            / "20260418T075338Z_restart_20260418"
            / "logs"
            / "rsl_rl"
            / "geo_landau_game"
            / "run_a"
            / "model_150.pt"
        )
        archived_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        archived_checkpoint_path.write_text("", encoding="utf-8")
        stale_live_checkpoint_path = temp_root / "logs" / "rsl_rl" / "geo_landau_game" / "run_a" / "model_150.pt"

        original_outputs_dir = run_history.outputs_dir
        try:
            run_history.outputs_dir = lambda: temp_root
            archived_registry = temp_root / "archives" / "20260418T075338Z_restart_20260418" / "history" / "checkpoint_registry.json"
            archived_registry.parent.mkdir(parents=True, exist_ok=True)
            archived_registry.write_text(
                json.dumps(
                    {
                        "experiments": {
                            "geo_landau_game": {
                                "recommended": {
                                    "checkpoint_path": str(stale_live_checkpoint_path),
                                    "load_run": "2026-04-18_10-41-48_game_v11_zero_init_200",
                                    "reason": "archived promoted fallback",
                                }
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )

            resolved = run_history.resolve_recommended_checkpoint("geo_landau_game")
        finally:
            run_history.outputs_dir = original_outputs_dir

    assert resolved is not None
    assert resolved["checkpoint_path"] == str(archived_checkpoint_path.resolve())
    assert resolved["resolved_experiment_name"] == "geo_landau_game"
    assert resolved["reason"] == "archived promoted fallback"


def test_write_diagnostic_record_prefers_explicit_experiment_name() -> None:
    with TemporaryDirectory() as temp_dir:
        temp_root = Path(temp_dir)
        original_outputs_dir = run_history.outputs_dir
        try:
            run_history.outputs_dir = lambda: temp_root
            payload = run_history.write_diagnostic_record(
                task_spec=SimpleNamespace(
                    key="g1",
                    display_name="Unitree G1",
                    train_task_id="Geo-Velocity-Flat-G1-v0",
                    play_task_id="Geo-Velocity-Flat-G1-Play-v0",
                    experiment_name="legacy_experiment",
                ),
                args=Namespace(experiment_name=None),
                scenario="pose_stability",
                status="completed",
                metrics={"done_count_total": 0},
                experiment_name="fresh_start_game",
                gate_result={"failure_code": None, "failed_checks": []},
                workflow_id="wf_diag",
            )
        finally:
            run_history.outputs_dir = original_outputs_dir

    assert payload["task"]["experiment_name"] == "fresh_start_game"
    assert payload["workflow_id"] == "wf_diag"
    assert payload["gate_result"]["failed_checks"] == []
    assert "lineage_name" in payload


def test_checkpoint_summary_sorts_numeric_suffix() -> None:
    with TemporaryDirectory() as temp_dir:
        run_dir = Path(temp_dir)
        for checkpoint_name in ("model_0.pt", "model_50.pt", "model_119.pt"):
            (run_dir / checkpoint_name).write_text("", encoding="utf-8")

        summary = run_history._checkpoint_summary(run_dir)

    assert summary["count"] == 3
    assert summary["latest"].endswith("model_119.pt")


def test_write_training_record_updates_landau_lineage_state() -> None:
    with TemporaryDirectory() as temp_dir:
        temp_root = Path(temp_dir)
        run_dir = temp_root / "logs" / "rsl_rl" / "geo_landau_restart_fwd_only" / "2026-04-18_17-18-28_stagea_primaryfoot_long_500"
        (run_dir / "model_49.pt").parent.mkdir(parents=True, exist_ok=True)
        (run_dir / "model_49.pt").write_text("checkpoint", encoding="utf-8")

        original_outputs_dir = run_history.outputs_dir
        original_lineage_outputs_dir = training_lineage.outputs_dir
        try:
            run_history.outputs_dir = lambda: temp_root
            training_lineage.outputs_dir = lambda: temp_root
            training_lineage.initialize_lineage(lineage_name="restart_case")

            run_history.write_training_record(
                log_dir=run_dir,
                task_spec=SimpleNamespace(
                    key="landau",
                    display_name="Landau v10 URDF",
                    train_task_id="Geo-Velocity-Flat-Landau-FwdOnly-v0",
                    play_task_id="Geo-Velocity-Flat-Landau-FwdOnly-Play-v0",
                    experiment_name="geo_landau_restart_case_fwd_only",
                ),
                args=Namespace(
                    robot="landau",
                    stage="fwd_only",
                    workflow_id="wf_train",
                    run_name="stagea_primaryfoot_long_500",
                    experiment_name=None,
                ),
                agent_cfg=SimpleNamespace(experiment_name="geo_landau_restart_case_fwd_only"),
                status="completed",
            )
            active_lineage = json.loads((temp_root / "history" / "active_lineage.json").read_text(encoding="utf-8"))
        finally:
            run_history.outputs_dir = original_outputs_dir
            training_lineage.outputs_dir = original_lineage_outputs_dir

    assert active_lineage["current_stage"] == "fwd_only"
    assert active_lineage["current_run"] == "2026-04-18_17-18-28_stagea_primaryfoot_long_500"
    assert active_lineage["current_checkpoint"].endswith("model_49.pt")
    assert active_lineage["stage_state"]["fwd_only"]["status"] == "trained"
    assert active_lineage["stage_state"]["fwd_only"]["last_training"]["status"] == "completed"
