from __future__ import annotations

import json
import math
import unittest

from algorithms.urdf_learn_wasd_walk import model_spec


class ModelSpecTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.spec = model_spec.build_robot_spec()

    def test_exact_current_urdf_and_mesh_package(self) -> None:
        source = self.spec["source"]
        self.assertEqual(source["urdf_sha256"], model_spec.EXPECTED_URDF_SHA256)
        self.assertFalse(source["derived_urdf_used"])
        self.assertEqual(source["mesh_reference_count"], 136)
        self.assertEqual(source["unique_mesh_count"], 68)
        self.assertEqual(source["missing_meshes"], [])

    def test_mass_inertia_collision_and_root_transform_audit(self) -> None:
        structure = self.spec["structure"]
        self.assertEqual(structure["link_count"], 71)
        self.assertEqual(structure["joint_count"], 70)
        self.assertEqual(structure["movable_joint_count"], 69)
        self.assertAlmostEqual(structure["total_mass_kg"], 1.829753, places=6)
        self.assertEqual(structure["invalid_inertia_links"], [])
        self.assertEqual(structure["root_link"], "base_link")
        self.assertEqual(structure["skeleton_root_link"], "root_x")
        self.assertAlmostEqual(structure["skeleton_root_origin_rpy"][0], math.pi / 2.0, places=5)

        bounds = self.spec["nominal_pose"]["zero_pose_collision_bounds"]
        for foot_link in ("foot_l", "foot_r", "toes_01_l", "toes_01_r"):
            self.assertAlmostEqual(bounds[foot_link]["minimum"][2], 0.0, delta=3.0e-5)

    def test_action_and_locked_joint_sets_are_explicit_partition(self) -> None:
        action = self.spec["action_joints"]
        locked = self.spec["locked_joints"]
        all_names = [record["name"] for record in self.spec["joints"]]
        self.assertEqual(len(action), 17)
        self.assertEqual(len(action) + len(locked), 69)
        self.assertEqual(set(action) | set(locked), set(all_names))
        self.assertFalse(set(action) & set(locked))
        self.assertIn("left_shin_roll_joint", locked)
        self.assertIn("right_shin_roll_joint", locked)
        self.assertTrue(all(name not in action for name in locked))

    def test_axes_were_interpreted_in_world_zero_pose(self) -> None:
        joints = {record["name"]: record for record in self.spec["joints"]}
        self.assertEqual(joints["left_hip_pitch_joint"]["dominant_world_axis"], "+X")
        self.assertEqual(joints["left_hip_roll_joint"]["dominant_world_axis"], "+Y")
        self.assertEqual(joints["left_hip_yaw_joint"]["dominant_world_axis"], "+Z")
        self.assertEqual(joints["left_shin_roll_joint"]["dominant_world_axis"], "-Z")
        self.assertGreater(abs(joints["left_shin_roll_joint"]["axis_world_zero_pose"][2]), 0.98)
        self.assertEqual(joints["left_shoulder_pitch_joint"]["dominant_world_axis"], "+X")

    def test_semantic_forward_mapping_is_body_plus_y(self) -> None:
        self.assertEqual(model_spec.SEMANTIC_COMMAND_ORDER, ("forward", "strafe", "yaw"))
        self.assertEqual(model_spec.semantic_to_sim_command(1.25, -0.4, 0.3), (-0.4, 1.25, 0.3))
        self.assertEqual(
            self.spec["frames"]["mapping"],
            {"linear_x": "strafe", "linear_y": "forward", "angular_z": "yaw"},
        )

    def test_checked_in_robot_spec_is_current(self) -> None:
        checked_in = json.loads(model_spec.ROBOT_SPEC_PATH.read_text(encoding="utf-8"))
        self.assertEqual(checked_in, self.spec)


if __name__ == "__main__":
    unittest.main()
