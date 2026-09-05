from __future__ import annotations

import json
import math
import unittest
import xml.etree.ElementTree as ET

from algorithms.urdf_learn_wasd_walk import model_spec


class ModelSpecTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.spec = model_spec.build_robot_spec()

    def test_exact_current_urdf_and_mesh_package(self) -> None:
        source = self.spec["source"]
        self.assertEqual(source["urdf_sha256"], model_spec.EXPECTED_URDF_SHA256)
        self.assertEqual(source["mesh_tree_sha256"], model_spec.EXPECTED_MESH_TREE_SHA256)
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
            self.assertAlmostEqual(bounds[foot_link]["minimum"][2], 0.0, delta=1.1e-4)
        support = self.spec["nominal_pose"]["zero_pose_ground_support"]
        self.assertAlmostEqual(support["ground_z_m"], 0.0, delta=3.0e-5)
        minimum = support["support_aabb_xy_m"]["minimum"]
        maximum = support["support_aabb_xy_m"]["maximum"]
        com = structure["zero_pose_center_of_mass_m"]
        self.assertLess(minimum[0], com[0])
        self.assertLess(com[0], maximum[0])
        self.assertLess(minimum[1], com[1])
        self.assertLess(com[1], maximum[1])
        self.assertGreater(len(support["candidate_contact_hull_xy_m"]), 3)
        self.assertEqual(support["flat_ground_contact_normal_w"], [0.0, 0.0, 1.0])
        self.assertEqual(support["gravity_direction_w"], [0.0, 0.0, -1.0])
        self.assertEqual(set(support["links"]), {"foot_l", "foot_r", "toes_01_l", "toes_01_r"})
        self.assertTrue(
            all(value["candidate_contact_hull_xy_m"] for value in support["links"].values())
        )

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

    def test_canonical_ground_alignment_is_retained_and_authority_probe_is_isolated(self) -> None:
        groups = self.spec["pd"]["groups"]
        self.assertAlmostEqual(self.spec["nominal_pose"]["base_position_m"][2], -0.004884927)
        self.assertGreater(self.spec["nominal_pose"]["geometry"]["support_margin_m"], 0.0)
        self.assertEqual(groups["leg_sagittal"]["damping"], 1.0)
        controlled = [joint for group in groups.values() for joint in group["joints"]]
        self.assertEqual(len(controlled), len(set(controlled)))
        self.assertEqual(set(controlled), set(self.spec["action_joints"]))
        experiments = self.spec["pd"]["stability_experiments"]
        self.assertEqual(experiments[0]["id"], "ankle_pitch_contact_damping_v1")
        self.assertEqual(experiments[0]["status"], "rejected")
        self.assertEqual(experiments[1]["id"], "ground_aligned_spawn_v1")
        self.assertEqual(experiments[1]["status"], "insufficient_supported")
        self.assertEqual(experiments[1]["change"], {"base_z_before_m": 0.002, "base_z_after_m": 0.0})
        self.assertEqual(experiments[2]["id"], "zero_pose_upper_body_authority_probe_v1")
        self.assertEqual(experiments[2]["status"], "rejected")
        self.assertEqual(experiments[3]["id"], "gravity_static_pose_release_v1")
        self.assertEqual(experiments[3]["status"], "free_root_supported_non_gate")
        self.assertFalse(experiments[3]["fixed_root_load_contract"]["contact_force_is_gating"])
        self.assertEqual(experiments[4]["id"], "canonical_settled_pose_10s_v1")
        self.assertEqual(experiments[4]["status"], "active_bounded_test")
        limit_audit = experiments[3]["limit_audit"]
        self.assertEqual(limit_audit["tolerance_rad"], 0.002)
        self.assertEqual(limit_audit["maximum_tip_displacement_at_40mm_m"], 0.00008)
        self.assertEqual(set(limit_audit["finger_joints"]), set(model_spec.FINGER_JOINTS))
        self.assertTrue(set(model_spec.FINGER_JOINTS) <= set(self.spec["locked_joints"]))
        probe = self.spec["pd"]["authority_probe_actuator_groups"]
        flattened = [joint for group in probe.values() for joint in group["joints"]]
        self.assertEqual(set(flattened), {joint["name"] for joint in self.spec["joints"]})
        self.assertEqual(len(flattened), len(set(flattened)))
        self.assertEqual(probe["leg_sagittal"]["stiffness"], 20.0)
        self.assertEqual(probe["leg_balance"]["damping"], 0.8)
        self.assertEqual(probe["waist_authority"]["stiffness"], 40.0)
        self.assertEqual(probe["upper_body_authority"]["damping"], 10.0)
        self.assertIn("right_shoulder_lift_joint", probe["upper_body_authority"]["joints"])

    def test_canonical_pose_has_exact_archived_provenance_and_zero_fingers(self) -> None:
        nominal = self.spec["nominal_pose"]
        provenance = nominal["provenance"]
        self.assertEqual(provenance["source_run_identity"], "20260904T071956.210897Z")
        self.assertEqual(
            provenance["source_evidence_sha256"],
            "6bbd26111a61026ae33ecdc7ee7b296e2faefbb7b4500530c320635b98519765",
        )
        self.assertEqual(provenance["free_root_result"]["fall_count"], 0)
        self.assertAlmostEqual(nominal["joint_positions_rad"]["waist_pitch_joint"], -0.2342214286327362)
        self.assertAlmostEqual(nominal["joint_positions_rad"]["left_knee_joint"], 0.21072299778461456)
        self.assertAlmostEqual(
            nominal["joint_position_targets_rad"]["right_shoulder_lift_joint"],
            0.213185280561,
        )
        self.assertTrue(all(nominal["joint_positions_rad"][name] == 0.0 for name in model_spec.FINGER_JOINTS))
        self.assertTrue(all(nominal["joint_position_targets_rad"][name] == 0.0 for name in model_spec.FINGER_JOINTS))
        for joint in self.spec["joints"]:
            lower, upper = joint["limits_rad"]
            self.assertLessEqual(lower, joint["nominal_position_rad"])
            self.assertLessEqual(joint["nominal_position_rad"], upper)
            self.assertLessEqual(lower, joint["nominal_target_rad"])
            self.assertLessEqual(joint["nominal_target_rad"], upper)
            self.assertEqual(
                joint["nominal_position_rad"], nominal["joint_positions_rad"][joint["name"]]
            )

    def test_static_pose_is_collision_supported_and_improves_com_margin(self) -> None:
        experiments = self.spec["pd"]["stability_experiments"]
        candidate = experiments[3]["geometry_candidate"]
        zero = model_spec.analyze_pose_geometry({})
        self.assertEqual(candidate["joint_positions_rad"]["left_hip_pitch_joint"], -0.1)
        self.assertEqual(candidate["joint_positions_rad"]["left_knee_joint"], 0.21)
        self.assertEqual(candidate["joint_positions_rad"]["left_ankle_pitch_joint"], -0.115)
        self.assertAlmostEqual(candidate["joint_positions_rad"]["waist_pitch_joint"], -0.206)
        self.assertGreater(candidate["support_margin_m"], zero["support_margin_m"] + 0.01)
        self.assertLess(candidate["maximum_fixed_root_gravity_torque_limit_fraction"], 0.05)
        contacts = candidate["near_ground_contact_vertex_count_by_link"]
        self.assertGreater(contacts["foot_l"] + contacts["toes_01_l"], 0)
        self.assertGreater(contacts["foot_r"] + contacts["toes_01_r"], 0)
        self.assertEqual(candidate["derivation"]["waist_search_rad"]["increment"], 0.001)

    def test_fixed_root_gravity_audit_explains_measured_shoulder_compliance(self) -> None:
        audit = self.spec["pd"]["zero_pose_static_authority_audit"]
        self.assertAlmostEqual(audit["left_shoulder_lift_gravity_torque_nm"], 0.893606, places=5)
        self.assertAlmostEqual(audit["right_shoulder_lift_gravity_torque_nm"], -0.893606, places=5)
        self.assertAlmostEqual(audit["current_linear_pd_shoulder_lift_error_rad"], 0.223402, places=5)
        self.assertAlmostEqual(
            audit["authority_probe_linear_pd_shoulder_lift_error_rad"], 0.0223402, places=5
        )
        joints = {joint["name"]: joint for joint in self.spec["joints"]}
        self.assertAlmostEqual(
            joints["left_shoulder_lift_joint"]["zero_pose_linear_pd_gravity_error_rad"],
            0.223402,
            places=5,
        )

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

    def test_visual_mesh_bounds_cover_every_visual_link(self) -> None:
        corners = model_spec.visual_local_aabb_corners()
        self.assertEqual(len(corners), 58)
        root = ET.parse(model_spec.URDF_PATH).getroot()
        expected = {
            link.get("name") for link in root.findall("link")
            if link.find("visual/geometry/mesh") is not None
        }
        self.assertEqual(set(corners), expected)
        for link_corners in corners.values():
            self.assertEqual(len(link_corners), 8)
            self.assertTrue(all(len(corner) == 3 for corner in link_corners))
            for axis in range(3):
                self.assertEqual(len({corner[axis] for corner in link_corners}), 2)

    def test_checked_in_robot_spec_is_current(self) -> None:
        checked_in = json.loads(model_spec.ROBOT_SPEC_PATH.read_text(encoding="utf-8"))
        self.assertEqual(checked_in, self.spec)


if __name__ == "__main__":
    unittest.main()
