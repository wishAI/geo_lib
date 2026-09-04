from __future__ import annotations

import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from algorithms.urdf_learn_wasd_walk import forward_walk_contract as contract
from algorithms.urdf_learn_wasd_walk import forward_walk_pipeline as pipeline
from algorithms.urdf_learn_wasd_walk import model_spec
from algorithms.urdf_learn_wasd_walk.tests.test_forward_walk_contract import passing_forward


class ForwardWalkPipelineTests(unittest.TestCase):
    def test_direct_entry_point_bootstraps_repository_package(self) -> None:
        completed = subprocess.run(
            [sys.executable, str(model_spec.ALGORITHM_ROOT / "forward_walk_pipeline.py"), "--help"],
            cwd="/tmp", text=True, capture_output=True, check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--finalize-only", completed.stdout)

    def test_finalizer_requires_candidate_stand_forward_and_visual_proof(self) -> None:
        with tempfile.TemporaryDirectory(dir=contract.OUTPUT_ROOT) as temporary:
            output = Path(temporary)
            checkpoint = output / "checkpoint.pt"
            checkpoint.write_bytes(b"forward checkpoint")
            requested = contract.training_contract(seed=42, num_envs=8, iterations=1)
            prior, _ = contract.load_cumulative_prior()
            training = {
                "schema_version": 1, "milestone": contract.MILESTONE_ID,
                "status": "completed_not_promoted", "lineage": contract.LINEAGE,
                "run_identity": "training", "input": model_spec.build_robot_spec()["source"],
                "robot_spec_sha256": contract.sha256(model_spec.ROBOT_SPEC_PATH),
                "requested_contract": requested,
                "checkpoint": {
                    "kind": "rsl_rl_ppo", "path": str(checkpoint.relative_to(pipeline.REPO_ROOT)),
                    "sha256": contract.sha256(checkpoint), "size_bytes": checkpoint.stat().st_size,
                    "learning_iteration": 1,
                },
            }
            training_path = output / contract.TRAINING_EVIDENCE
            training_path.write_text(json.dumps(training), encoding="utf-8")
            video_path = output / "proof.mp4"
            sheet_path = output / "contact_sheet.png"
            video_path.write_bytes(b"video")
            sheet_path.write_bytes(b"sheet")
            checkpoint_record = training["checkpoint"]
            training_record = {
                "path": str(training_path.relative_to(pipeline.REPO_ROOT)),
                "sha256": contract.sha256(training_path), "run_identity": "training",
            }
            shared = {
                "schema_version": 1, "milestone": contract.MILESTONE_ID,
                "scope": "component_only", "status": "passed", "gate_eligible": True,
                "failures": [], "lineage": contract.LINEAGE, "seed": 42,
                "checkpoint": checkpoint_record, "training_evidence": training_record,
                "source_commit": "test", "versions": {},
                "input": model_spec.build_robot_spec()["source"],
                "joint_contract": {
                    "runtime_action_order": list(model_spec.ACTION_JOINTS),
                    "runtime_importer_axis_evidence": {"passed": True, "joint_count": 69},
                },
                "policy_contract": requested, "command_contract": {},
                "traces": [], "action_traces": [], "cumulative_gates": prior,
            }
            stand_metrics = {
                "duration_s": 30.0, "control_steps": 1500, "policy_inference_steps": 1500,
                "reset_count": 0, "done_count": 0, "fall_count": 0,
                "max_abs_command": 0.0, "max_abs_action": 0.5,
                "max_reference_tilt_rad": 0.1, "root_height_drop_m": 0.01,
                "horizontal_drift_m": 0.02,
                "minimum_support_polygon_margin_m": 0.01,
                "first_support_exit_time_s": None,
                "peak_support_force_body_weight_ratio": 2.0,
                "mean_support_force_body_weight_ratio": 1.0,
            }
            forward_metrics = passing_forward()
            stand = {**copy.deepcopy(shared), "component": "stand", "run_identity": "stand", "metrics": stand_metrics}
            forward = {**copy.deepcopy(shared), "component": "forward", "run_identity": "forward", "metrics": forward_metrics}
            video = {
                "capture_path": "active_viewport_LdrColor_AOV",
                "path": str(video_path.relative_to(pipeline.REPO_ROOT)),
                "sha256": contract.sha256(video_path),
                "contact_sheet_path": str(sheet_path.relative_to(pipeline.REPO_ROOT)),
                "contact_sheet_sha256": contract.sha256(sheet_path),
                "duration_s": 14.0, "nonblank_frames_passed": True,
                "character_visibility_passed": True, "temporal_progression_visible": True,
                "motion_discernible": True,
            }
            proof = {
                **copy.deepcopy(shared), "component": "proof", "run_identity": "proof",
                "metrics": forward_metrics, "video_inspection": video,
            }
            paths = {
                "stand": output / contract.component_artifact_name("stand"),
                "forward": output / contract.component_artifact_name("forward"),
                "proof": output / contract.component_artifact_name("proof"),
            }
            paths["stand"].write_text(json.dumps(stand), encoding="utf-8")
            paths["forward"].write_text(json.dumps(forward), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "missing proof"):
                pipeline.finalize(output)
            paths["proof"].write_text(json.dumps(proof), encoding="utf-8")
            final = pipeline.finalize(output)
            self.assertEqual(final["status"], "passed")
            self.assertTrue(final["cumulative_gates"][1]["candidate_checkpoint_repassed"])
            self.assertEqual(final["metrics"]["semantic_forward_displacement_m"], 5.01)

    def test_canonical_recorder_advances_only_after_hashing_all_artifacts(self) -> None:
        with tempfile.TemporaryDirectory(dir=contract.OUTPUT_ROOT) as temporary:
            root = Path(temporary)
            milestones_path = root / "milestones.json"
            manifest_path = root / "manifest.json"
            milestones_path.write_text(
                (model_spec.ALGORITHM_ROOT / "milestones.json").read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            manifest_path.write_text(
                (model_spec.ALGORITHM_ROOT / "gui" / "manifest.json").read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            for name in (
                "validation.json", "training.json", "stand_dynamics_validation.json",
                "forward_dynamics_validation.json", "proof_validation.json", "checkpoint.pt",
                "proof.mp4", "contact_sheet.png",
            ):
                (root / name).write_bytes(name.encode())
            forward_metrics = passing_forward()
            final = {
                "assembled_at": "20260904T100000.000000Z",
                "seed": 42,
                "checkpoint": {
                    "path": str((root / "checkpoint.pt").relative_to(pipeline.REPO_ROOT)),
                    "sha256": contract.sha256(root / "checkpoint.pt"),
                },
                "input": {"urdf_sha256": model_spec.EXPECTED_URDF_SHA256},
                "metrics": forward_metrics,
                "cumulative_gate_metrics": {"stand_30s_no_reset": {"duration_s": 30.0}},
            }
            pipeline._record_canonical_pass(
                final,
                root / "validation.json",
                root,
                root / "proof.mp4",
                root / "contact_sheet.png",
                milestones_path=milestones_path,
                manifest_path=manifest_path,
            )
            ledger = json.loads(milestones_path.read_text(encoding="utf-8"))
            by_id = {item["id"]: item for item in ledger["milestones"]}
            self.assertEqual(by_id[contract.MILESTONE_ID]["status"], "passed")
            self.assertEqual(by_id["gate_10m_no_reset"]["status"], "in_progress")
            self.assertEqual(len(by_id[contract.MILESTONE_ID]["evidence"]), 8)
            self.assertTrue(all(len(item["sha256"]) == 64 for item in by_id[contract.MILESTONE_ID]["evidence"]))


if __name__ == "__main__":
    unittest.main()
