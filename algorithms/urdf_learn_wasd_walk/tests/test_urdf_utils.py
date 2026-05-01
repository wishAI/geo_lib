from __future__ import annotations

import sys
import unittest
from pathlib import Path


MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT.parent.parent) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT.parent.parent))

from algorithms.urdf_learn_wasd_walk.asset_paths import landau_urdf_path
from algorithms.urdf_learn_wasd_walk.robot_specs import load_landau_robot_spec
from algorithms.urdf_learn_wasd_walk.urdf_utils import (
    classify_joint_groups,
    compute_link_world_transforms,
    detect_primary_foot_links,
    detect_support_links,
    detect_termination_links,
    estimate_root_height,
    mass_bearing_links,
    load_urdf_model,
    support_surface_world_z,
    total_mass,
    transform_point,
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

        self.assertIn("left_hip_pitch_joint", groups.leg_joints)
        self.assertIn("left_toe_joint", groups.foot_joints)
        self.assertIn("left_thumb_metacarpal_joint", groups.finger_joints)
        self.assertIn("waist_yaw_joint", groups.torso_joints)

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

    def test_mass_rescale_produces_realistic_total_mass(self) -> None:
        self.assertEqual(len(mass_bearing_links(self.model)), 71)
        self.assertAlmostEqual(total_mass(self.model), 36.67, places=2)
        self.assertGreater(self.model.links["root_x"].mass + self.model.links["spine_01_x"].mass, 9.0)

    def test_estimated_root_height_preserves_support_clearance_for_mesh_surfaces(self) -> None:
        pose = {
            "left_hip_pitch_joint": 0.20,
            "right_hip_pitch_joint": 0.20,
            "left_knee_joint": 0.54,
            "right_knee_joint": 0.54,
            "left_ankle_pitch_joint": -0.07,
            "right_ankle_pitch_joint": -0.07,
            "left_toe_joint": 0.05,
            "right_toe_joint": 0.05,
        }
        height = estimate_root_height(
            model=self.model,
            root_link_name="base_link",
            support_link_names=("foot_l", "foot_r", "toes_01_l", "toes_01_r"),
            joint_positions=pose,
            clearance=0.01,
        )
        world = compute_link_world_transforms(self.model, joint_positions=pose, root_link="base_link")
        min_support_z = min(
            support_surface_world_z(self.model, world, link_name)
            for link_name in ("foot_l", "foot_r", "toes_01_l", "toes_01_r")
        )

        self.assertGreater(height, 0.0)
        self.assertAlmostEqual(min_support_z + height, 0.01, places=4)

    def test_support_surface_is_not_the_same_as_link_origin_height(self) -> None:
        spec = load_landau_robot_spec(stage="stand")
        world = compute_link_world_transforms(self.model, joint_positions=spec.default_joint_positions, root_link="base_link")

        left_foot_z = support_surface_world_z(self.model, world, "foot_l")
        left_toe_z = support_surface_world_z(self.model, world, "toes_01_l")
        left_foot_origin_z = transform_point(world["foot_l"], (0.0, 0.0, 0.0))[2]
        left_toe_origin_z = transform_point(world["toes_01_l"], (0.0, 0.0, 0.0))[2]

        self.assertGreater(left_foot_origin_z - left_toe_origin_z, 0.02)
        self.assertLess(abs(left_foot_z - left_toe_z), 0.01)


if __name__ == "__main__":
    unittest.main()
