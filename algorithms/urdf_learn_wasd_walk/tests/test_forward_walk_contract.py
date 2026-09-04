from __future__ import annotations

import unittest

from algorithms.urdf_learn_wasd_walk import forward_walk_contract as contract, model_spec


def passing_forward() -> dict:
    return {
        "duration_s": 14.0,
        "control_steps": 700,
        "policy_inference_steps": 700,
        "reset_count": 0,
        "done_count": 0,
        "fall_count": 0,
        "semantic_forward_displacement_m": 5.01,
        "semantic_strafe_displacement_m": 0.1,
        "max_reference_tilt_rad": 0.2,
        "root_height_drop_m": 0.02,
        "max_abs_command": 0.4,
        "max_abs_action": 0.8,
        "left_foot_liftoff_count": 4,
        "right_foot_liftoff_count": 4,
        "leg_joint_excursion_rad": {
            "left_hip_pitch_joint": 0.1,
            "right_hip_pitch_joint": 0.1,
            "left_knee_joint": 0.1,
            "right_knee_joint": 0.1,
        },
        "mean_contact_foot_slip_mps": 0.1,
        "simultaneous_air_fraction": 0.0,
    }


class ForwardWalkContractTests(unittest.TestCase):
    def test_command_mapping_keeps_landau_body_plus_y_forward(self) -> None:
        self.assertEqual(contract.semantic_command_to_sim(0.4, -0.1, 0.2), (-0.1, 0.4, 0.2))
        self.assertEqual(model_spec.SEMANTIC_COMMAND_ORDER, ("forward", "strafe", "yaw"))

    def test_observation_appends_command_for_exact_stand_transfer(self) -> None:
        self.assertEqual(contract.ACTOR_OBSERVATION_DIM, 63)
        self.assertEqual(contract.ACTOR_OBSERVATION_TERMS[-1], ("semantic_velocity_command", 3))
        self.assertEqual(sum(width for _, width in contract.ACTOR_OBSERVATION_TERMS[:-1]), 60)

    def test_current_two_gate_lineage_and_parent_checkpoint_are_hash_checked(self) -> None:
        prior, parent = contract.load_cumulative_prior()
        self.assertEqual([item["status"] for item in prior], ["passed", "passed"])
        self.assertEqual(parent["sha256"], contract.PARENT_CHECKPOINT_SHA256)

    def test_forward_gate_requires_distance_gait_and_no_events(self) -> None:
        self.assertEqual(contract.evaluate_forward_gate(passing_forward()), [])
        for field in ("reset_count", "done_count", "fall_count"):
            metrics = passing_forward()
            metrics[field] = 1
            self.assertTrue(contract.evaluate_forward_gate(metrics))
        metrics = passing_forward()
        metrics["left_foot_liftoff_count"] = 0
        self.assertIn("swing", " ".join(contract.evaluate_forward_gate(metrics)))
        metrics = passing_forward()
        metrics["leg_joint_excursion_rad"]["right_knee_joint"] = 0.01
        self.assertIn("right_knee_joint", " ".join(contract.evaluate_forward_gate(metrics)))
        metrics = passing_forward()
        metrics["simultaneous_air_fraction"] = 0.2
        self.assertIn("hopping", " ".join(contract.evaluate_forward_gate(metrics)))

    def test_training_contract_has_no_mid_gate_curriculum_or_rough_terrain(self) -> None:
        requested = contract.training_contract(seed=42, num_envs=512, iterations=600)
        self.assertEqual(requested["initialization"]["preserved_observation_prefix_width"], 60)
        self.assertEqual(requested["environment"]["sim_command_mapping"]["forward"], "linear_y")
        self.assertIsNone(requested["environment"]["curriculum"])
        self.assertFalse(requested["environment"]["rough_terrain"])

    def test_forward_velocity_diagnostics_distinguish_progress_and_reverse_motion(self) -> None:
        summary = contract.summarize_forward_velocity_samples(
            [-0.2, 0.0, 0.2, 0.4], [0.4, 0.4, 0.4, 0.4], control_dt_s=0.02
        )
        self.assertEqual(summary["mean_semantic_forward_velocity_mps"], 0.1)
        self.assertEqual(summary["mean_abs_semantic_forward_velocity_tracking_error_mps"], 0.3)
        self.assertEqual(summary["forward_progress_step_fraction"], 0.5)
        self.assertEqual(summary["reverse_motion_step_fraction"], 0.25)
        self.assertEqual(summary["first_positive_forward_velocity_time_s"], 0.06)

    def test_forward_velocity_diagnostics_reject_misaligned_samples(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-empty and aligned"):
            contract.summarize_forward_velocity_samples([0.1], [], control_dt_s=0.02)


if __name__ == "__main__":
    unittest.main()
