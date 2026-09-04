from __future__ import annotations

import inspect
import json
import tempfile
import unittest
from pathlib import Path

from algorithms.urdf_learn_wasd_walk import model_spec, passive_stand


class PassiveStandContractTests(unittest.TestCase):
    def test_output_path_is_allowlisted_to_algorithm_outputs(self) -> None:
        child = passive_stand.OUTPUT_ROOT / "test" / "run"
        self.assertEqual(passive_stand.safe_output_dir(child), child.resolve())
        with self.assertRaises(ValueError):
            passive_stand.safe_output_dir(Path("/tmp/outside-walk-output"))

    def test_quaternion_distance_is_sign_invariant(self) -> None:
        identity = (1.0, 0.0, 0.0, 0.0)
        self.assertEqual(passive_stand.quaternion_distance_wxyz(identity, identity), 0.0)
        self.assertEqual(passive_stand.quaternion_distance_wxyz(identity, (-1.0, 0.0, 0.0, 0.0)), 0.0)

    def test_gate_rejects_events_motion_and_short_duration(self) -> None:
        metrics = {
            "duration_s": 29.0,
            "reset_count": 1,
            "done_count": 1,
            "fall_count": 1,
            "max_reference_tilt_rad": 0.0,
            "root_height_drop_m": 0.0,
            "horizontal_drift_m": 0.0,
            "max_abs_action": 0.1,
            "max_abs_command": 0.0,
        }
        passed, failures = passive_stand.evaluate_gate(metrics)
        self.assertFalse(passed)
        self.assertGreaterEqual(len(failures), 5)

    def test_gate_accepts_exact_zero_signal_stand(self) -> None:
        metrics = {
            "duration_s": 30.0,
            "reset_count": 0,
            "done_count": 0,
            "fall_count": 0,
            "max_reference_tilt_rad": 0.01,
            "root_height_drop_m": 0.005,
            "horizontal_drift_m": 0.002,
            "max_abs_action": 0.0,
            "max_abs_command": 0.0,
        }
        self.assertEqual(passive_stand.evaluate_gate(metrics), (True, []))

    def test_physics_source_has_no_isaac_camera_sensor(self) -> None:
        source = inspect.getsource(passive_stand)
        self.assertNotIn("from isaaclab.sensors import Camera", source)
        self.assertNotIn("CameraCfg(", source)
        self.assertIn("active_viewport_LdrColor_AOV", source)
        self.assertEqual(passive_stand.component_artifact_name("dynamics", False), "dynamics_validation.json")
        self.assertEqual(passive_stand.component_artifact_name("proof", False), "proof_validation.json")

    def test_proof_gate_checks_duration_frames_visibility_and_progression(self) -> None:
        good = {
            "duration_s": 30.0,
            "nonblank_frames_passed": True,
            "character_visibility_passed": True,
            "temporal_progression_visible": True,
        }
        self.assertEqual(passive_stand.evaluate_proof(good), (True, []))
        for key in ("nonblank_frames_passed", "character_visibility_passed", "temporal_progression_visible"):
            bad = dict(good)
            bad[key] = False
            self.assertFalse(passive_stand.evaluate_proof(bad)[0])

    def test_proof_preflight_requires_current_passing_dynamics(self) -> None:
        passive_stand.OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=passive_stand.OUTPUT_ROOT) as temporary:
            output_dir = Path(temporary)
            path = output_dir / "dynamics_validation.json"
            evidence = {
                "component": "dynamics",
                "status": "passed",
                "seed": 42,
                "input": {"urdf_sha256": model_spec.EXPECTED_URDF_SHA256},
                "checkpoint": {
                    "identity": f"robot_spec_sha256:{passive_stand._sha256(model_spec.ROBOT_SPEC_PATH)}"
                },
                "joint_contract": {"runtime_importer_axis_evidence": {"passed": True}},
            }
            path.write_text(json.dumps(evidence), encoding="utf-8")
            self.assertEqual(passive_stand.preflight_proof(output_dir, smoke=False, seed=42), evidence)
            evidence["status"] = "failed"
            path.write_text(json.dumps(evidence), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "passing dynamics"):
                passive_stand.preflight_proof(output_dir, smoke=False, seed=42)


if __name__ == "__main__":
    unittest.main()
