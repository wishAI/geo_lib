from __future__ import annotations

import unittest
from pathlib import Path

from algorithms.urdf_learn_wasd_walk import passive_stand


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


if __name__ == "__main__":
    unittest.main()
