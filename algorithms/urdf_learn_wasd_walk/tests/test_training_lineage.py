from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import algorithms.urdf_learn_wasd_walk.run_history as run_history
import algorithms.urdf_learn_wasd_walk.training_lineage as training_lineage


def test_resolve_landau_experiment_name_uses_legacy_without_active_lineage() -> None:
    with TemporaryDirectory() as temp_dir:
        temp_root = Path(temp_dir)
        original_outputs_dir = training_lineage.outputs_dir
        try:
            training_lineage.outputs_dir = lambda: temp_root
            assert training_lineage.resolve_landau_experiment_name("game") == "geo_landau_game"
        finally:
            training_lineage.outputs_dir = original_outputs_dir


def test_reset_landau_training_state_archives_old_history_and_logs() -> None:
    with TemporaryDirectory() as temp_dir:
        temp_root = Path(temp_dir)
        temp_repo_root = temp_root / "repo"
        (temp_root / "history" / "refs").mkdir(parents=True, exist_ok=True)
        (temp_root / "history" / "training_runs.jsonl").write_text('{"record_kind":"training"}\n', encoding="utf-8")
        logs_root = temp_repo_root / "logs" / "rsl_rl" / "geo_landau_game"
        logs_root.mkdir(parents=True, exist_ok=True)
        (logs_root / "model_1.pt").write_text("checkpoint", encoding="utf-8")

        original_lineage_outputs_dir = training_lineage.outputs_dir
        original_history_outputs_dir = run_history.outputs_dir
        original_repo_root = training_lineage._repo_root
        try:
            training_lineage.outputs_dir = lambda: temp_root
            run_history.outputs_dir = lambda: temp_root
            training_lineage._repo_root = lambda: temp_repo_root
            payload = training_lineage.reset_landau_training_state(
                lineage_name="from_start",
                archive_tag="reset",
                note="fresh start",
            )
        finally:
            training_lineage.outputs_dir = original_lineage_outputs_dir
            run_history.outputs_dir = original_history_outputs_dir
            training_lineage._repo_root = original_repo_root

        archive_manifest_path = Path(payload["archive_manifest_path"])
        active_lineage = payload["active_lineage"]
        manifest = json.loads(archive_manifest_path.read_text(encoding="utf-8"))
        assert active_lineage["lineage_name"] == "from_start"
        assert active_lineage["schema_version"] == 5
        assert active_lineage["experiments"]["game"] == "geo_landau_from_start_game"
        assert active_lineage["current_stage"] == "stand"
        assert active_lineage["current_checkpoint"] is None
        assert active_lineage["current_target_milestone"] == "stand_zero_signal_30s_no_reset"
        assert active_lineage["acceptance"]["accept_stage"] == "fwd_only"
        assert active_lineage["acceptance"]["accept_terrain_mode"] == "flat"
        assert active_lineage["training_rule"]["goal_order"][0] == "stand_zero_signal_30s_no_reset"
        assert active_lineage["training_rule"]["version"] == 3
        assert active_lineage["milestone_registry"]["gate_5m_no_reset"]["status"] == "pending"
        assert active_lineage["milestone_registry"]["gate_10m_four_directions_no_reset"]["status"] == "pending"
        assert active_lineage["stage_state"]["game"]["status"] == "pending"
        assert active_lineage["stage_state"]["game"]["objective"] is not None
        assert active_lineage["stage_state"]["game"]["required_milestones"][-1] == "game_10m_no_reset"
        assert active_lineage["stage_history"] == []
        assert any(entry["kind"] == "history" for entry in manifest["moved_paths"])
        assert any(entry["kind"] == "log_dir" for entry in manifest["moved_paths"])
        assert (temp_root / "history" / "active_lineage.json").is_file()
        assert (temp_root / "history" / "training_runs.jsonl").is_file()
        assert (temp_root / "history" / "README.md").is_file()
        assert (temp_root / "history" / "refs" / "milestones.json").is_file()
        assert (temp_root / "history" / "refs" / "stage_history.json").is_file()
        assert not (temp_repo_root / "logs" / "rsl_rl" / "geo_landau_game").exists()


def test_record_lineage_milestone_updates_registry_and_current_target() -> None:
    with TemporaryDirectory() as temp_dir:
        temp_root = Path(temp_dir)
        original_lineage_outputs_dir = training_lineage.outputs_dir
        original_history_outputs_dir = run_history.outputs_dir
        try:
            training_lineage.outputs_dir = lambda: temp_root
            run_history.outputs_dir = lambda: temp_root
            training_lineage.initialize_lineage(lineage_name="from_start")
            checkpoint_path = temp_root / "logs" / "rsl_rl" / "geo_landau_from_start_stand" / "run_a" / "model_10.pt"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            checkpoint_path.write_text("checkpoint", encoding="utf-8")

            payload = training_lineage.record_lineage_milestone(
                milestone_id="stand_zero_signal_30s_no_reset",
                checkpoint_path=str(checkpoint_path),
                run_name="run_a",
                status="passed",
                manual_review="upright for 30 s",
                notes="first stable stand checkpoint",
                recorded_by="test",
            )
        finally:
            training_lineage.outputs_dir = original_lineage_outputs_dir
            run_history.outputs_dir = original_history_outputs_dir

    milestone_entry = payload["milestone_registry"]["stand_zero_signal_30s_no_reset"]
    assert milestone_entry["status"] == "passed"
    assert milestone_entry["run_name"] == "run_a"
    assert milestone_entry["checkpoint_path"].endswith("model_10.pt")
    assert payload["current_target_milestone"] == "stand_30s_no_reset"
    assert payload["stage_state"]["stand"]["achieved_capability"]["milestone_id"] == "stand_zero_signal_30s_no_reset"
    assert payload["milestone_history"][-1]["milestone_id"] == "stand_zero_signal_30s_no_reset"


def test_record_lineage_milestone_accepts_passive_checkpoint() -> None:
    with TemporaryDirectory() as temp_dir:
        temp_root = Path(temp_dir)
        original_lineage_outputs_dir = training_lineage.outputs_dir
        original_history_outputs_dir = run_history.outputs_dir
        try:
            training_lineage.outputs_dir = lambda: temp_root
            run_history.outputs_dir = lambda: temp_root
            training_lineage.initialize_lineage(lineage_name="from_start")

            payload = training_lineage.record_lineage_milestone(
                milestone_id="stand_zero_signal_30s_no_reset",
                checkpoint_path=None,
                run_name="passive",
                status="passed",
                manual_review="zero-action stand passed",
                notes="passive stability gate",
                recorded_by="test",
            )
        finally:
            training_lineage.outputs_dir = original_lineage_outputs_dir
            run_history.outputs_dir = original_history_outputs_dir

    milestone_entry = payload["milestone_registry"]["stand_zero_signal_30s_no_reset"]
    assert milestone_entry["checkpoint_path"] is None
    assert milestone_entry["run_name"] == "passive"
    assert payload["current_target_milestone"] == "stand_30s_no_reset"
    assert payload["stage_state"]["stand"]["achieved_capability"]["checkpoint_path"] is None
