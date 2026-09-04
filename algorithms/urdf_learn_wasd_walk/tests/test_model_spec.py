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
        self.assertAlmostEqual(structure["zero_pose_center_of_mass_m"][1], -0.00601558, places=7)
        self.assertEqual(structure["invalid_inertia_links"], [])
        self.assertEqual(structure["root_link"], "base_link")
        self.assertEqual(structure["skeleton_root_link"], "root_x")
        self.assertAlmostEqual(structure["skeleton_root_origin_rpy"][0], math.pi / 2.0, places=5)

        bounds = self.spec["nominal_pose"]["zero_pose_collision_bounds"]
        for foot_link in ("foot_l", "foot_r", "toes_01_l", "toes_01_r"):
            self.assertAlmostEqual(bounds[foot_link]["minimum"][2], 0.0, delta=3.0e-5)
        support = self.spec["nominal_pose"]["zero_pose_ground_support"]
        self.assertAlmostEqual(support["ground_z_m"], 0.0, delta=3.0e-5)
        minimum = support["support_aabb_xy_m"]["minimum"]
        maximum = support["support_aabb_xy_m"]["maximum"]
        com = structure["zero_pose_center_of_mass_m"]
        self.assertLess(minimum[0], com[0])
        self.assertLess(com[0], maximum[0])
        self.assertLess(minimum[1], com[1])
        self.assertLess(com[1], maximum[1])

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

    def test_first_stability_hypothesis_changes_only_ankle_pitch_damping(self) -> None:
        groups = self.spec["pd"]["groups"]
        ankle = groups["ankle_pitch_contact"]
        self.assertEqual(
            ankle["joints"], ["left_ankle_pitch_joint", "right_ankle_pitch_joint"]
        )
        self.assertEqual((ankle["stiffness"], ankle["damping"]), (20.0, 4.0))
        self.assertEqual(groups["leg_sagittal"]["damping"], 1.0)
        controlled = [joint for group in groups.values() for joint in group["joints"]]
        self.assertEqual(len(controlled), len(set(controlled)))
        self.assertEqual(set(controlled), set(self.spec["action_joints"]))
        hypothesis = self.spec["pd"]["stability_hypothesis"]
        self.assertEqual(hypothesis["id"], "ankle_pitch_contact_damping_v1")
        self.assertEqual(hypothesis["change"]["damping_before"], 1.0)
        self.assertEqual(hypothesis["change"]["damping_after"], 4.0)

    def test_axes_were_interpreted_in_world_zero_pose(self) -> None:
        joints = {record["name"]: record for record in self.spec["joints"]}
        self.assertEqual(joints["left_hip_pitch_joint"]["dominant_world_axis"], "+X")
        self.assertEqual(joints["left_hip_roll_joint"]["dominant_world_axis"], "+Y")
        self.assertEqual(joints["left_hip_yaw_joint"]["dominant_world_axis"], "+Z")
        self.assertEqual(joints["left_shin_roll_joint"]["dominant_world_axis"], "-Z")
        self.assertGreater(abs(joints["left_shin_roll_joint"]["axis_world_zero_pose"][2]), 0.98)
        self.assertEqual(joints["left_shoulder_pitch_joint"]["dominant_world_axis"], "+X")

    def test_importer_reorientation_is_explicitly_a_runtime_contract(self) -> None:
        joints = self.spec["joints"]
        self.assertEqual(sum(not joint["source_axis_is_primary"] for joint in joints), 51)
        self.assertTrue(all(joint["runtime_importer_axis_verification_required"] for joint in joints))
        importer = self.spec["importer_axis_contract"]
        self.assertEqual(importer["minimum_directed_axis_cosine"], 0.995)
        self.assertIn("PhysX X primary axis", importer["observed_importer_behavior"])

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
