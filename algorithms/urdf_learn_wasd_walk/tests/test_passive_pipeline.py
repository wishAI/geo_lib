from __future__ import annotations

import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from algorithms.urdf_learn_wasd_walk import model_spec, passive_pipeline, passive_stand


def _metrics() -> dict:
    return {
        "duration_s": 30.0,
        "physics_steps": 15000,
        "reset_count": 0,
        "done_count": 0,
        "fall_count": 0,
        "max_abs_action": 0.0,
        "max_abs_command": 0.0,
        "max_reference_tilt_rad": 0.01,
        "root_height_drop_m": 0.005,
        "horizontal_drift_m": 0.002,
        "first_support_exit_time_s": None,
        "minimum_support_polygon_margin_m": 0.04,
        "peak_support_force_body_weight_ratio": 1.2,
        "mean_support_force_body_weight_ratio": 1.0,
    }


class PassivePipelineTests(unittest.TestCase):
    def test_finalizer_requires_both_matching_components_and_hashed_visuals(self) -> None:
        passive_stand.OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=passive_stand.OUTPUT_ROOT) as temporary:
            output_dir = Path(temporary)
            video_path = output_dir / "proof.mp4"
            sheet_path = output_dir / "contact_sheet.png"
            video_path.write_bytes(b"test video artifact")
            sheet_path.write_bytes(b"test contact sheet artifact")
            shared = {
                "schema_version": 3,
                "milestone": passive_stand.MILESTONE_ID,
                "scope": "component_only",
                "status": "passed",
                "gate_eligible": True,
                "failures": [],
                "lineage": "clean_restart_2026_08_22",
                "seed": 42,
                "checkpoint": {"identity": "robot-spec", "policy_checkpoint": None},
                "input": {"urdf_sha256": model_spec.EXPECTED_URDF_SHA256},
                "command_contract": {"body_forward_axis": "+Y"},
                "joint_contract": {
                    "action_joints": ["action"],
                    "locked_joints": ["locked"],
                    "runtime_importer_axis_evidence": {"passed": True, "joint_count": 69},
                },
                "versions": {"isaac_sim": "4.5.0"},
                "metrics": _metrics(),
                "traces": [],
                "run_identity": "run",
            }
            dynamics = {**copy.deepcopy(shared), "component": "dynamics", "video_inspection": None}
            video = {
                "capture_path": "active_viewport_LdrColor_AOV",
                "path": str(video_path.relative_to(passive_pipeline.REPO_ROOT)),
                "sha256": passive_stand._sha256(video_path),
                "contact_sheet_path": str(sheet_path.relative_to(passive_pipeline.REPO_ROOT)),
                "contact_sheet_sha256": passive_stand._sha256(sheet_path),
                "duration_s": 30.0,
                "nonblank_frames_passed": True,
                "character_visibility_passed": True,
                "temporal_progression_visible": True,
            }
            proof = {**copy.deepcopy(shared), "component": "proof", "video_inspection": video}
            (output_dir / "dynamics_validation.json").write_text(json.dumps(dynamics), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "missing proof"):
                passive_pipeline.finalize(output_dir)
            (output_dir / "proof_validation.json").write_text(json.dumps(proof), encoding="utf-8")
            final = passive_pipeline.finalize(output_dir)
            self.assertEqual(final["status"], "passed")
            self.assertTrue((output_dir / "validation.json").is_file())
            self.assertFalse(final["simulator"]["dynamics_rendering_enabled"])

    def test_direct_script_entry_point_bootstraps_repository_package(self) -> None:
        script = model_spec.ALGORITHM_ROOT / "passive_pipeline.py"
        completed = subprocess.run(
            [sys.executable, str(script), "--help"],
            cwd="/tmp",
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--finalize-only", completed.stdout)


if __name__ == "__main__":
    unittest.main()
