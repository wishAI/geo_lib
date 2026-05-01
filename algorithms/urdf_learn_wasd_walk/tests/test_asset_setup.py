from __future__ import annotations

import sys
import unittest
from pathlib import Path


MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT.parent.parent) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT.parent.parent))

from algorithms.urdf_learn_wasd_walk.asset_setup import prepare_landau_inputs
from algorithms.urdf_learn_wasd_walk.robot_specs import load_landau_robot_spec
from algorithms.urdf_learn_wasd_walk.urdf_utils import load_urdf_model, total_mass


class AssetSetupTests(unittest.TestCase):
    def test_prepare_landau_inputs_resolves_existing_handoff(self) -> None:
        prepared = prepare_landau_inputs(refresh=False)

        self.assertTrue(prepared.urdf_path.exists())
        self.assertTrue(prepared.urdf_path.name.endswith(".fixed.urdf"))
        self.assertTrue(prepared.mesh_root.exists())
        self.assertTrue(prepared.usd_path.exists())
        self.assertTrue(prepared.skeleton_json_path.exists())
        self.assertTrue(prepared.texture_dir.exists())

    def test_landau_spec_has_no_missing_meshes(self) -> None:
        spec = load_landau_robot_spec()

        self.assertEqual(spec.missing_meshes, ())
        self.assertGreaterEqual(len(spec.joint_groups.leg_joints), 6)
        self.assertGreaterEqual(len(spec.joint_groups.foot_joints), 2)
        self.assertGreaterEqual(len(spec.joint_groups.torso_joints), 3)

    def test_landau_lower_body_runtime_joints_match_refreshed_urdf(self) -> None:
        spec = load_landau_robot_spec()
        model = load_urdf_model(spec.urdf_path)

        self.assertEqual(
            set(spec.joint_groups.foot_joints),
            {
                "left_ankle_pitch_joint",
                "right_ankle_pitch_joint",
                "left_toe_joint",
                "right_toe_joint",
            },
        )
        self.assertTrue(set(spec.default_joint_positions).issubset(model.joints))
        self.assertEqual(spec.primary_foot_links, ("foot_l", "foot_r"))
        self.assertAlmostEqual(total_mass(model), 36.67, places=2)


if __name__ == "__main__":
    unittest.main()
