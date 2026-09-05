from __future__ import annotations

import unittest

from algorithms.urdf_learn_wasd_walk import (
    forward_reference,
    forward_reference_probe,
    model_spec,
    policy_stand_contract,
)


class ForwardReferenceTests(unittest.TestCase):
    def test_probe_defaults_to_current_policy_stand_parent(self) -> None:
        self.assertEqual(
            forward_reference_probe.DEFAULT_PARENT_OUTPUT,
            policy_stand_contract.DEFAULT_OUTPUT_DIR,
        )

    def test_probe_protocol_uses_command_onset_without_hiding_total_motion(self) -> None:
        protocol = forward_reference_probe.probe_protocol(50, 250)
        self.assertEqual(protocol["pre_command_zero_action_duration_s"], 1.0)
        self.assertEqual(protocol["reference_duration_s"], 5.0)
        self.assertTrue(protocol["total_displacement_also_recorded"])
        with self.assertRaises(ValueError):
            forward_reference_probe.probe_protocol(-1, 250)

    def test_zero_command_is_exact_stand_reference(self) -> None:
        for time_s in (0.0, 0.2, 1.7):
            self.assertEqual(
                forward_reference.reference_action(time_s, 0.0),
                [0.0] * len(model_spec.ACTION_JOINTS),
            )

    def test_bilateral_phase_difference_and_soft_landing(self) -> None:
        config = forward_reference.ReferenceConfig(startup_ramp_s=0.0)
        start = forward_reference.reference_action(0.0, 0.4, config)
        quarter = forward_reference.reference_action(0.2, 0.4, config)
        half = forward_reference.reference_action(0.4, 0.4, config)
        three_quarter = forward_reference.reference_action(0.6, 0.4, config)
        index = {name: model_spec.ACTION_JOINTS.index(name) for name in model_spec.ACTION_JOINTS}
        self.assertAlmostEqual(start[index["left_knee_joint"]], 0.0)
        self.assertAlmostEqual(start[index["right_knee_joint"]], 0.0)
        self.assertAlmostEqual(half[index["left_knee_joint"]], 0.0)
        self.assertAlmostEqual(half[index["right_knee_joint"]], 0.0)
        self.assertAlmostEqual(quarter[index["left_knee_joint"]], 1.0)
        self.assertAlmostEqual(quarter[index["right_knee_joint"]], 0.0)
        self.assertAlmostEqual(three_quarter[index["left_knee_joint"]], 0.0)
        self.assertAlmostEqual(three_quarter[index["right_knee_joint"]], 1.0)

    def test_reference_is_bounded_and_only_uses_verified_sagittal_joints(self) -> None:
        used = set()
        for step in range(80):
            actions = forward_reference.reference_action(step * 0.01, 0.4)
            self.assertLessEqual(max(map(abs, actions)), 1.0)
            used.update(
                name for name, value in zip(model_spec.ACTION_JOINTS, actions) if abs(value) > 1.0e-9
            )
        self.assertEqual(used, {
            f"{side}_{joint}_joint"
            for side in ("left", "right")
            for joint in ("hip_pitch", "knee", "ankle_pitch", "toe")
        })

    def test_toe_amplitude_is_an_independent_recorded_probe_factor(self) -> None:
        config = forward_reference.ReferenceConfig(startup_ramp_s=0.0, toe_amplitude=0.0)
        action = forward_reference.reference_action(0.2, 0.4, config)
        toe_indices = [
            model_spec.ACTION_JOINTS.index(f"{side}_toe_joint")
            for side in ("left", "right")
        ]
        self.assertEqual([action[index] for index in toe_indices], [0.0, 0.0])
        reference = forward_reference.reference_contract(config, action_scale_rad=0.17)
        self.assertEqual(reference["parameters"]["toe_amplitude"], 0.0)
        self.assertEqual(reference["action_scale_rad"], 0.17)

    def test_lateral_weight_transfer_is_phase_opposed_and_command_gated(self) -> None:
        config = forward_reference.ReferenceConfig(
            startup_ramp_s=0.0, hip_roll_amplitude=0.5
        )
        action = forward_reference.reference_action(0.2, 0.4, config)
        left = model_spec.ACTION_JOINTS.index("left_hip_roll_joint")
        right = model_spec.ACTION_JOINTS.index("right_hip_roll_joint")
        self.assertAlmostEqual(action[left], -action[right])
        self.assertAlmostEqual(action[left], 0.5)
        self.assertEqual(
            forward_reference.reference_action(0.2, 0.0, config),
            [0.0] * len(model_spec.ACTION_JOINTS),
        )

    def test_waist_weight_transfer_is_bounded_and_command_gated(self) -> None:
        config = forward_reference.ReferenceConfig(
            startup_ramp_s=0.0, waist_roll_amplitude=0.5
        )
        waist = model_spec.ACTION_JOINTS.index("waist_roll_joint")
        self.assertAlmostEqual(forward_reference.reference_action(0.2, 0.4, config)[waist], 0.5)
        self.assertEqual(
            forward_reference.reference_action(0.2, 0.0, config),
            [0.0] * len(model_spec.ACTION_JOINTS),
        )

    def test_probe_acceptance_uses_direct_air_runs_and_clearance(self) -> None:
        reference = forward_reference.reference_contract(forward_reference.ReferenceConfig())
        audit = reference["offline_kinematic_audit"]["full_amplitude_mid_swing"]
        self.assertGreater(audit["left"]["minimum_collision_height_gain_m"], 0.002)
        self.assertGreater(audit["right"]["minimum_collision_height_gain_m"], 0.002)
        metrics = {
            "left_max_consecutive_direct_air_steps": 2,
            "right_max_consecutive_direct_air_steps": 3,
            "left_max_support_body_height_gain_m": 0.002,
            "right_max_support_body_height_gain_m": 0.003,
            "semantic_forward_displacement_m": 0.001,
            "reset_count": 0,
            "done_count": 0,
            "fall_count": 0,
            "max_joint_target_error_rad": 0.2,
        }
        self.assertEqual(forward_reference.evaluate_probe(metrics, reference), [])
        metrics["left_max_consecutive_direct_air_steps"] = 1
        self.assertIn("robust direct-observation", " ".join(
            forward_reference.evaluate_probe(metrics, reference)
        ))

    def test_invalid_single_factor_bounds_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "amplitude scale"):
            forward_reference.ReferenceConfig(amplitude_scale=1.01).validate()
        with self.assertRaisesRegex(ValueError, "action scale"):
            forward_reference.reference_contract(
                forward_reference.ReferenceConfig(), action_scale_rad=0.31
            )

    def test_probe_action_scale_is_recorded_in_fk_audit(self) -> None:
        reference = forward_reference.reference_contract(
            forward_reference.ReferenceConfig(), action_scale_rad=0.24
        )
        self.assertEqual(reference["action_scale_rad"], 0.24)
        self.assertEqual(reference["offline_kinematic_audit"]["action_scale_rad"], 0.24)


if __name__ == "__main__":
    unittest.main()
