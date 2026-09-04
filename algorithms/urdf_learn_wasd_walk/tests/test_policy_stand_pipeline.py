from __future__ import annotations

import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from algorithms.urdf_learn_wasd_walk import model_spec, passive_stand
from algorithms.urdf_learn_wasd_walk import policy_stand_contract as contract
from algorithms.urdf_learn_wasd_walk import policy_stand_pipeline as pipeline


class PolicyStandPipelineTests(unittest.TestCase):
    def test_direct_pipeline_entry_point_bootstraps_repository_package(self) -> None:
        script = model_spec.ALGORITHM_ROOT / "policy_stand_pipeline.py"
        completed = subprocess.run(
            [sys.executable, str(script), "--help"], cwd="/tmp", text=True,
            capture_output=True, check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--finalize-only", completed.stdout)

    def test_finalizer_requires_hashed_checkpoint_components_and_visuals(self) -> None:
        contract.OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=contract.OUTPUT_ROOT) as temporary:
            output = Path(temporary)
            run = output / "runs" / "test"
            run.mkdir(parents=True)
            checkpoint = run / "model_1.pt"
            checkpoint.write_bytes(b"policy checkpoint")
            requested = contract.training_contract(seed=42, num_envs=8, iterations=1)
            training = {
                "schema_version": 1,
                "milestone": contract.MILESTONE_ID,
                "status": "completed_not_promoted",
                "lineage": contract.LINEAGE,
                "run_identity": "training-run",
                "source_commit": "test",
                "requested_contract": requested,
                "input": model_spec.build_robot_spec()["source"],
                "robot_spec_sha256": contract.sha256(model_spec.ROBOT_SPEC_PATH),
                "checkpoint": {
                    "kind": "rsl_rl_ppo",
                    "path": str(checkpoint.relative_to(pipeline.REPO_BOOTSTRAP_ROOT)),
                    "sha256": contract.sha256(checkpoint),
                    "size_bytes": checkpoint.stat().st_size,
                    "learning_iteration": 1,
                },
            }
            training_path = output / contract.TRAINING_EVIDENCE
            training_path.write_text(json.dumps(training), encoding="utf-8")
            video_path = output / "proof.mp4"
            sheet_path = output / "contact_sheet.png"
            video_path.write_bytes(b"video")
            sheet_path.write_bytes(b"sheet")
            metrics = {
                "duration_s": 30.0, "physics_steps": 15000, "control_steps": 1500,
                "policy_inference_steps": 1500, "reset_count": 0, "done_count": 0,
                "fall_count": 0, "max_abs_command": 0.0, "max_abs_action": 0.1,
                "max_reference_tilt_rad": 0.02, "root_height_drop_m": 0.002,
                "horizontal_drift_m": 0.002, "first_support_exit_time_s": None,
                "minimum_support_polygon_margin_m": 0.04,
                "peak_support_force_body_weight_ratio": 1.2,
                "mean_support_force_body_weight_ratio": 1.0,
            }
            prior = contract.load_prior_gate()
            checkpoint_record = training["checkpoint"]
            shared = {
                "schema_version": 1, "milestone": contract.MILESTONE_ID,
                "scope": "component_only", "status": "passed", "gate_eligible": True,
                "failures": [], "lineage": contract.LINEAGE, "seed": 42,
                "checkpoint": checkpoint_record,
                "training_evidence": {
                    "path": str(training_path.relative_to(pipeline.REPO_BOOTSTRAP_ROOT)),
                    "sha256": contract.sha256(training_path), "run_identity": "training-run",
                },
                "source_commit": "test", "versions": {},
                "input": model_spec.build_robot_spec()["source"],
                "joint_contract": {
                    "runtime_action_order": list(model_spec.ACTION_JOINTS),
                    "runtime_importer_axis_evidence": {"passed": True, "joint_count": 69},
                },
                "policy_contract": requested,
                "command_contract": model_spec.build_robot_spec()["frames"],
                "metrics": metrics, "traces": [], "action_traces": [],
                "cumulative_gates": [prior],
            }
            dynamics = {**copy.deepcopy(shared), "component": "dynamics", "run_identity": "dynamics"}
            proof_video = {
                "capture_path": "active_viewport_LdrColor_AOV",
                "path": str(video_path.relative_to(pipeline.REPO_BOOTSTRAP_ROOT)),
                "sha256": contract.sha256(video_path),
                "contact_sheet_path": str(sheet_path.relative_to(pipeline.REPO_BOOTSTRAP_ROOT)),
                "contact_sheet_sha256": contract.sha256(sheet_path),
                "duration_s": 30.0, "nonblank_frames_passed": True,
                "character_visibility_passed": True, "temporal_progression_visible": True,
            }
            proof = {
                **copy.deepcopy(shared), "component": "proof", "run_identity": "proof",
                "video_inspection": proof_video,
            }
            (output / contract.component_artifact_name("dynamics")).write_text(
                json.dumps(dynamics), encoding="utf-8"
            )
            (output / contract.component_artifact_name("proof")).write_text(
                json.dumps(proof), encoding="utf-8"
            )
            final = pipeline.finalize(output)
            self.assertEqual(final["status"], "passed")
            self.assertEqual([item["status"] for item in final["cumulative_gates"]], ["passed", "passed"])
            self.assertEqual(final["checkpoint"]["sha256"], contract.sha256(checkpoint))


if __name__ == "__main__":
    unittest.main()
