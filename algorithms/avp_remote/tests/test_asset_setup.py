import sys
import unittest
from filecmp import cmp
from pathlib import Path


MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from asset_paths import (
    default_landau_source_mesh_root,
    default_landau_source_skeleton_json,
    default_landau_source_urdf,
    landau_mesh_root,
    landau_skeleton_json_path,
    landau_urdf_path,
    landau_usd_path,
)
from asset_setup import prepare_landau_inputs
from landau_pose import load_urdf_visual_mesh_records


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
        self.assertTrue(cmp(prepared.urdf_path, default_landau_source_urdf(), shallow=False))
        self.assertTrue(cmp(prepared.skeleton_json_path, default_landau_source_skeleton_json(), shallow=False))

        source_stls = sorted(path.relative_to(default_landau_source_mesh_root()) for path in default_landau_source_mesh_root().rglob("*.stl"))
        copied_stls = sorted(path.relative_to(landau_mesh_root()) for path in landau_mesh_root().rglob("*.stl"))
        self.assertEqual(copied_stls, source_stls)
        for rel_path in source_stls[:5]:
            self.assertTrue(cmp(default_landau_source_mesh_root() / rel_path, landau_mesh_root() / rel_path, shallow=False))

    def test_mesh_urdf_loader_keeps_multi_visual_links(self):
        mesh_records = load_urdf_visual_mesh_records(landau_urdf_path())

        self.assertEqual(len(mesh_records["spine_03_x"]), 3)
        self.assertEqual(len(mesh_records["forearm_stretch_l"]), 2)
        self.assertEqual(len(mesh_records["arm_twist_r"]), 2)


if __name__ == "__main__":
    unittest.main()
