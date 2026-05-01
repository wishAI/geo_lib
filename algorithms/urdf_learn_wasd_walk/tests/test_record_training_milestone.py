from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import algorithms.urdf_learn_wasd_walk.run_history as run_history
from algorithms.urdf_learn_wasd_walk.scripts import record_training_milestone
import algorithms.urdf_learn_wasd_walk.training_lineage as training_lineage


def test_resolve_checkpoint_path_accepts_none_marker() -> None:
    assert (
        record_training_milestone._resolve_checkpoint_path(
            stage="stand",
            load_run="passive",
            checkpoint="none",
            checkpoint_path=None,
        )
        is None
    )


def test_latest_stage_evidence_record_returns_latest_matching_stage() -> None:
    with TemporaryDirectory() as temp_dir:
        temp_root = Path(temp_dir)
        original_outputs_dir = run_history.outputs_dir
        original_lineage_outputs_dir = training_lineage.outputs_dir
        try:
            run_history.outputs_dir = lambda: temp_root
            training_lineage.outputs_dir = lambda: temp_root
            history_root = run_history.history_dir()
            history_root.mkdir(parents=True, exist_ok=True)
            (history_root / "diagnostic_runs.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "record_kind": "diagnostic",
                                "recorded_at": "2026-04-19T00:00:00+00:00",
                                "task": {"experiment_name": "geo_landau_from_start_game"},
                                "status": "failed",
                            }
                        ),
                        json.dumps(
                            {
                                "record_kind": "diagnostic",
                                "recorded_at": "2026-04-19T00:01:00+00:00",
                                "task": {"experiment_name": "geo_landau_from_start_stand"},
                                "status": "failed",
                            }
                        ),
                        json.dumps(
                            {
                                "record_kind": "diagnostic",
                                "recorded_at": "2026-04-19T00:02:00+00:00",
                                "task": {"experiment_name": "geo_landau_from_start_stand"},
                                "status": "completed",
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            training_lineage.initialize_lineage(lineage_name="from_start")

            record = record_training_milestone._latest_stage_evidence_record(
                record_kind="diagnostic",
                stage="stand",
            )
        finally:
            run_history.outputs_dir = original_outputs_dir
            training_lineage.outputs_dir = original_lineage_outputs_dir

    assert record is not None
    assert record["status"] == "completed"
    assert record["task"]["experiment_name"] == "geo_landau_from_start_stand"
