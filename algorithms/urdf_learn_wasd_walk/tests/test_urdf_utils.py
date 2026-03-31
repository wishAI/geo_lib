from __future__ import annotations

import sys
import unittest
from pathlib import Path


MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT.parent.parent) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT.parent.parent))

from algorithms.urdf_learn_wasd_walk.asset_paths import landau_urdf_path
from algorithms.urdf_learn_wasd_walk.urdf_utils import (
    classify_joint_groups,
    detect_primary_foot_links,
    detect_support_links,
    detect_termination_links,
    estimate_root_height,
    load_urdf_model,
)


class UrdfParsingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model = load_urdf_model(landau_urdf_path())

    def test_model_contains_expected_root_and_feet(self) -> None:
        self.assertIn("base_link", self.model.root_links)
        self.assertIn("foot_l", self.model.links)
        self.assertIn("foot_r", self.model.links)

    def test_joint_classification_finds_leg_and_finger_groups(self) -> None:
        groups = classify_joint_groups(self.model)

        self.assertIn("thigh_stretch_l", groups.leg_joints)
        self.assertIn("toes_01_l", groups.foot_joints)
        self.assertIn("thumb1_l", groups.finger_joints)
        self.assertIn("spine_01_x", groups.torso_joints)

    def test_support_and_termination_detection_are_stable(self) -> None:
        self.assertEqual(detect_primary_foot_links(self.model), ("foot_l", "foot_r"))
        self.assertEqual(
            detect_support_links(self.model),
            ("foot_l", "foot_r", "toes_01_l", "toes_01_r"),
        )
        self.assertEqual(
            detect_termination_links(self.model),
            ("base_link", "root_x", "spine_01_x", "spine_02_x", "spine_03_x", "neck_x", "head_x"),
        )

    def test_estimated_root_height_is_positive(self) -> None:
        pose = {
            "thigh_stretch_l": 0.20,
            "thigh_stretch_r": 0.20,
            "leg_stretch_l": 0.54,
            "leg_stretch_r": 0.54,
            "foot_l": -0.07,
            "foot_r": -0.07,
            "toes_01_l": 0.05,
            "toes_01_r": 0.05,
        }
        height = estimate_root_height(
            model=self.model,
            root_link_name="base_link",
            support_link_names=("foot_l", "foot_r", "toes_01_l", "toes_01_r"),
            joint_positions=pose,
            clearance=0.01,
        )

        self.assertGreaterEqual(height, 0.0)
        self.assertLess(height, 0.1)


if __name__ == "__main__":
    unittest.main()
