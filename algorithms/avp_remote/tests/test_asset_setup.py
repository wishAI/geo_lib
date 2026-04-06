import sys
import unittest
from pathlib import Path


MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from asset_paths import (
    landau_mesh_root,
    landau_skeleton_json_path,
    landau_urdf_path,
    landau_usd_path,
)
from asset_setup import prepare_landau_inputs


class TestAssetSetup(unittest.TestCase):
    def test_prepare_landau_inputs_populates_expected_files(self):
        prepared = prepare_landau_inputs(refresh=False)

        self.assertTrue(prepared.input_dir.exists())
        self.assertEqual(prepared.urdf_path, landau_urdf_path())
        self.assertTrue(landau_urdf_path().exists())
        self.assertTrue(landau_mesh_root().exists())
        self.assertTrue(landau_usd_path().exists())
        self.assertTrue(landau_skeleton_json_path().exists())
        self.assertGreater(len(list(landau_mesh_root().rglob("*.stl"))), 10)


if __name__ == "__main__":
    unittest.main()
