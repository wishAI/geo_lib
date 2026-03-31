from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT.parent.parent) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT.parent.parent))

from algorithms.urdf_learn_wasd_walk.asset_setup import prepare_landau_inputs
from algorithms.urdf_learn_wasd_walk.usd_visualizer import (
    apply_joint_positions_to_local_matrices,
    axis_angle_matrix,
    inverse_rigid_transform,
    load_skeleton_records,
    rigid_transform,
)


class UsdVisualizerMathTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        prepared = prepare_landau_inputs(refresh=False)
        cls.records = load_skeleton_records(prepared.skeleton_json_path)
        cls.records_by_name = {record.name: record for record in cls.records}

    def test_load_skeleton_records_finds_single_root(self) -> None:
        roots = [record for record in self.records if record.parent_index < 0]

        self.assertEqual(len(roots), 1)
        self.assertEqual(roots[0].name, "root_x")

    def test_apply_joint_positions_rotates_but_does_not_translate_joint(self) -> None:
        record = self.records_by_name["spine_02_x"]

        posed = apply_joint_positions_to_local_matrices(self.records, {"spine_02_x": 0.35})

        np.testing.assert_allclose(posed[record.index][:3, 3], record.local_matrix[:3, 3], atol=1.0e-9)
        self.assertGreater(np.linalg.norm(posed[record.index][:3, :3] - record.local_matrix[:3, :3]), 1.0e-6)

    def test_inverse_rigid_transform_round_trips(self) -> None:
        transform = rigid_transform(axis_angle_matrix((0.0, 0.0, 1.0), 0.4), (1.2, -0.5, 0.3))
        transform_inv = inverse_rigid_transform(transform)

        np.testing.assert_allclose(transform @ transform_inv, np.eye(4, dtype=float), atol=1.0e-9)
        np.testing.assert_allclose(transform_inv @ transform, np.eye(4, dtype=float), atol=1.0e-9)


if __name__ == "__main__":
    unittest.main()
