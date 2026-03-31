from __future__ import annotations

import sys
import unittest
from pathlib import Path


MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT.parent.parent) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT.parent.parent))

from algorithms.urdf_learn_wasd_walk.asset_setup import prepare_landau_inputs
from algorithms.urdf_learn_wasd_walk.robot_specs import load_landau_robot_spec


class AssetSetupTests(unittest.TestCase):
    def test_prepare_landau_inputs_resolves_existing_handoff(self) -> None:
        prepared = prepare_landau_inputs(refresh=False)

        self.assertTrue(prepared.urdf_path.exists())
        self.assertTrue(prepared.mesh_root.exists())
        self.assertTrue(prepared.usd_path.exists())
        self.assertTrue(prepared.skeleton_json_path.exists())
        self.assertTrue(prepared.texture_dir.exists())

    def test_landau_spec_has_no_missing_meshes(self) -> None:
        spec = load_landau_robot_spec()

        self.assertEqual(spec.missing_meshes, ())


if __name__ == "__main__":
    unittest.main()
