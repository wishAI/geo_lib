from __future__ import annotations

import json
import unittest

from algorithms.urdf_learn_wasd_walk import forward_walk_contract as contract, model_spec


def _forward_prerequisites_are_current() -> bool:
    ledger = json.loads((model_spec.ALGORITHM_ROOT / "milestones.json").read_text())
    status = {item["id"]: item["status"] for item in ledger["milestones"]}
    return all(status.get(item) == "passed" for item in ("stand_zero_signal_30s_no_reset", contract.PARENT_MILESTONE_ID))


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
    def test_gait_phase_clock_tolerates_manager_shape_probe_before_rl_buffer_exists(self) -> None:
        class ConstructionPhaseEnv:
            num_envs = 64

        env = ConstructionPhaseEnv()
        self.assertIsNone(contract.episode_phase_step_buffer(env))
        env.episode_length_buf = [0, 7]
        self.assertEqual(contract.episode_phase_step_buffer(env), [0, 7])

    def test_command_mapping_keeps_landau_body_plus_y_forward(self) -> None:
        self.assertEqual(contract.semantic_command_to_sim(0.4, -0.1, 0.2), (-0.1, 0.4, 0.2))
        self.assertEqual(model_spec.SEMANTIC_COMMAND_ORDER, ("forward", "strafe", "yaw"))

    def test_observation_appends_command_for_exact_stand_transfer(self) -> None:
        self.assertEqual(contract.ACTOR_OBSERVATION_DIM, 65)
        self.assertEqual(contract.ACTOR_OBSERVATION_TERMS[-2], ("semantic_velocity_command", 3))
        self.assertEqual(contract.ACTOR_OBSERVATION_TERMS[-1], ("gait_phase", 2))
        self.assertEqual(sum(width for _, width in contract.ACTOR_OBSERVATION_TERMS[:-2]), 60)

    def test_current_two_gate_lineage_and_parent_checkpoint_are_hash_checked(self) -> None:
        if not _forward_prerequisites_are_current():
            self.skipTest("latest-mesh stand gates are awaiting re-certification")
        prior, parent = contract.load_cumulative_prior()
        self.assertEqual([item["status"] for item in prior], ["passed", "passed"])
        ledger = json.loads((model_spec.ALGORITHM_ROOT / "milestones.json").read_text())
        recorded = next(item for item in ledger["milestones"] if item["id"] == contract.PARENT_MILESTONE_ID)
        self.assertEqual(parent["sha256"], recorded["checkpoint"]["sha256"])

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
        if not _forward_prerequisites_are_current():
            self.skipTest("latest-mesh stand gates are awaiting re-certification")
        requested = contract.training_contract(seed=42, num_envs=512, iterations=600)
        self.assertEqual(requested["initialization"]["preserved_observation_prefix_width"], 60)
        self.assertEqual(requested["environment"]["sim_command_mapping"]["forward"], "linear_y")
        self.assertIsNone(requested["environment"]["curriculum"])
        self.assertFalse(requested["environment"]["rough_terrain"])
        self.assertEqual(requested["training_method"], contract.TRAINING_METHOD_ID)
        self.assertEqual(requested["environment"]["gait_phase_period_s"], 0.8)
        reward_names = [item["name"] for item in requested["environment"]["reward_terms"]]
        self.assertNotIn("track_linear_velocity", reward_names)
        self.assertIn("alternating_single_support", reward_names)

    def test_failed_evidence_classifies_period_two_no_liftoff(self) -> None:
        actions = []
        for step in range(4):
            values = [0.0] * len(model_spec.ACTION_JOINTS)
            for name in ("left_hip_pitch_joint", "right_knee_joint"):
                values[model_spec.ACTION_JOINTS.index(name)] = 0.5 if step % 2 == 0 else -0.5
            actions.append({"joint_order": list(model_spec.ACTION_JOINTS), "raw_policy_action": values})
        evidence = {
            "status": "failed", "milestone": contract.MILESTONE_ID, "component": "forward",
            "metrics": {
                "left_foot_liftoff_count": 0, "right_foot_liftoff_count": 0,
                "semantic_forward_displacement_m": 0.3,
                "mean_semantic_forward_velocity_mps": 0.0,
                "reverse_motion_step_fraction": 0.5,
            },
            "action_traces": actions,
        }
        diagnosis = contract.analyze_failed_forward_evidence(evidence)
        self.assertEqual(diagnosis["classification"], "period_two_no_liftoff")
        self.assertEqual(diagnosis["next_method"], contract.TRAINING_METHOD_ID)

    def test_failed_gate_resume_preserves_command_inputs_and_adds_only_phase(self) -> None:
        if not _forward_prerequisites_are_current():
            self.skipTest("latest-mesh stand gates are awaiting re-certification")
        requested = contract.training_contract(
            seed=42,
            num_envs=64,
            iterations=2,
            initialization_source={
                "kind": "failed_gate_method_change_transfer",
                "path": "outputs/checkpoint.pt",
                "sha256": "a" * 64,
                "actor_observation_dim": 63,
                "source_milestone": contract.MILESTONE_ID,
            },
        )
        initialization = requested["initialization"]
        self.assertEqual(initialization["preserved_observation_prefix_width"], 63)
        self.assertEqual(initialization["new_input_columns_zero_initialized"], 2)
        self.assertFalse(initialization["optimizer_state_loaded"])

    def test_training_smoke_decision_requires_progress_and_bilateral_liftoff(self) -> None:
        passing = {
            "semantic_forward_displacement_m": 0.2,
            "mean_semantic_forward_velocity_mps": 0.02,
            "left_foot_liftoff_count": 1,
            "right_foot_liftoff_count": 1,
            "reset_count": 0,
            "done_count": 0,
            "fall_count": 0,
        }
        self.assertEqual(contract.evaluate_training_smoke_diagnostic(passing), [])
        stalled = {**passing, "semantic_forward_displacement_m": 0.004}
        stalled["mean_semantic_forward_velocity_mps"] = -0.003
        stalled["left_foot_liftoff_count"] = 0
        stalled["right_foot_liftoff_count"] = 0
        failures = contract.evaluate_training_smoke_diagnostic(stalled)
        self.assertEqual(len(failures), 4)
        self.assertEqual(
            contract.NEXT_TRAINING_HYPOTHESIS["id"],
            "command_gated_phase_reference_residual_v3",
        )

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
