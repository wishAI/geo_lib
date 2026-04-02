from __future__ import annotations

import sys
import unittest
from pathlib import Path


MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT.parent.parent) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT.parent.parent))

from algorithms.urdf_learn_wasd_walk.runtime import resolve_robot_task_spec


class RuntimeTests(unittest.TestCase):
    def test_g1_ignores_landau_stage_argument(self) -> None:
        spec = resolve_robot_task_spec("g1", stage="fwd_only")

        self.assertEqual(spec.key, "g1")
        self.assertEqual(spec.train_task_id, "Geo-Velocity-Flat-G1-v0")
        self.assertEqual(spec.play_task_id, "Geo-Velocity-Flat-G1-Play-v0")

    def test_landau_fwd_only_stage_exposes_stage_specific_ids(self) -> None:
        spec = resolve_robot_task_spec("landau", stage="fwd_only")

        self.assertEqual(spec.train_task_id, "Geo-Velocity-Flat-Landau-FwdOnly-v0")
        self.assertEqual(spec.play_task_id, "Geo-Velocity-Flat-Landau-FwdOnly-Play-v0")
        self.assertEqual(spec.experiment_name, "geo_landau_fwd_only")
        self.assertEqual(spec.forward_body_axis, "y")
        self.assertEqual(spec.control_root_link, "root_x")

    def test_landau_unknown_stage_raises(self) -> None:
        with self.assertRaises(ValueError):
            resolve_robot_task_spec("landau", stage="unknown")


if __name__ == "__main__":
    unittest.main()
