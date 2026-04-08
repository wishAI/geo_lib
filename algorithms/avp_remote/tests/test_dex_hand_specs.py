import sys
import unittest
from pathlib import Path


MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from dex_hand_specs import build_h1_2_target_specs, build_landau_target_specs


class TestDexHandSpecs(unittest.TestCase):
    def test_build_landau_target_specs_cover_all_finger_joints(self):
        specs = build_landau_target_specs(
            MODULE_ROOT / "inputs" / "landau_v10" / "landau_v10_parallel_mesh.urdf",
        )

        self.assertEqual(set(specs), {"left", "right"})
        self.assertEqual(len(specs["left"].target_joint_names), 19)
        self.assertEqual(len(specs["left"].target_origin_link_names), 5)
        self.assertEqual(len(specs["left"].target_task_link_names), 5)
        self.assertEqual(specs["left"].hand_link_name, "hand_l")
        self.assertEqual(specs["right"].hand_link_name, "hand_r")

    def test_build_h1_2_target_specs_cover_all_five_fingers(self):
        specs = build_h1_2_target_specs(
            MODULE_ROOT.parents[2] / "helper_repos" / "xr_teleoperate_shallow" / "assets" / "h1_2" / "h1_2.urdf",
        )

        self.assertEqual(set(specs), {"left", "right"})
        self.assertEqual(len(specs["left"].target_joint_names), 6)
        self.assertEqual(len(specs["left"].target_origin_link_names), 5)
        self.assertEqual(len(specs["left"].target_task_link_names), 5)
        self.assertEqual(specs["left"].hand_link_name, "L_hand_base_link")
        self.assertEqual(specs["right"].hand_link_name, "R_hand_base_link")


if __name__ == "__main__":
    unittest.main()
