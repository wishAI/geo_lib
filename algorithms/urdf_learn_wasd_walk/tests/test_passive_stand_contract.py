from __future__ import annotations

import inspect
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from algorithms.urdf_learn_wasd_walk import model_spec, passive_stand


class PassiveStandContractTests(unittest.TestCase):
    def test_output_path_is_allowlisted_to_algorithm_outputs(self) -> None:
        child = passive_stand.OUTPUT_ROOT / "test" / "run"
        self.assertEqual(passive_stand.safe_output_dir(child), child.resolve())
        with self.assertRaises(ValueError):
            passive_stand.safe_output_dir(Path("/tmp/outside-walk-output"))

    def test_quaternion_distance_is_sign_invariant(self) -> None:
        identity = (1.0, 0.0, 0.0, 0.0)
        self.assertEqual(passive_stand.quaternion_distance_wxyz(identity, identity), 0.0)
        self.assertEqual(passive_stand.quaternion_distance_wxyz(identity, (-1.0, 0.0, 0.0, 0.0)), 0.0)

    def test_semantic_projected_gravity_removes_imported_root_rotation(self) -> None:
        root_x_upright = (2**-0.5, 2**-0.5, 0.0, 0.0)
        gravity = passive_stand.semantic_projected_gravity_wxyz(root_x_upright, root_x_upright)
        self.assertAlmostEqual(gravity[0], 0.0)
        self.assertAlmostEqual(gravity[1], 0.0)
        self.assertAlmostEqual(gravity[2], -1.0)
        # A semantic -30 degree X pitch composed with the imported +90 degree
        # root rotation produces a +60 degree X root quaternion.
        pitched_root = (3**0.5 / 2.0, 0.5, 0.0, 0.0)
        pitched_gravity = passive_stand.semantic_projected_gravity_wxyz(
            pitched_root, root_x_upright
        )
        self.assertAlmostEqual(pitched_gravity[1], 0.5)
        self.assertAlmostEqual(pitched_gravity[2], -(3**0.5 / 2.0))

    def test_joint_tracking_summary_is_complete_and_worst_first(self) -> None:
        summary = passive_stand.summarize_joint_tracking(
            ["ankle", "knee"],
            [0.1, 0.4],
            [0.5, 1.2],
            [1.0, 2.0],
            [3.0, 6.0],
            [3.0, 5.0],
            [0, 25],
            100,
        )
        self.assertEqual([item["name"] for item in summary], ["knee", "ankle"])
        self.assertEqual(summary[0]["max_target_error_time_s"], 1.2)
        self.assertEqual(summary[0]["torque_saturation_step_fraction"], 0.25)
        with self.assertRaises(ValueError):
            passive_stand.summarize_joint_tracking([], [], [], [], [], [], [], 0)

    def test_runtime_failure_evidence_records_stage_and_traceback(self) -> None:
        passive_stand.OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=passive_stand.OUTPUT_ROOT) as temporary:
            args = SimpleNamespace(
                output_dir=Path(temporary), phase="dynamics", smoke=True,
                runtime_stage="first_system_com_sample", device="cuda:0",
                steps=1, duration=30.0, headless=True,
            )
            error = RuntimeError("device mismatch")
            evidence_path, traceback_path = passive_stand.write_failure_evidence(
                args, error, "Traceback: device mismatch\n"
            )
            evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
            self.assertEqual(evidence["status"], "failed_to_execute")
            self.assertEqual(evidence["runtime_stage"], "first_system_com_sample")
            self.assertEqual(evidence["exception"]["type"], "RuntimeError")
            self.assertEqual(traceback_path.read_text(encoding="utf-8"), "Traceback: device mismatch\n")
            self.assertEqual(
                evidence_path.name, passive_stand.failure_artifact_name("dynamics", True)
            )

    def test_mass_diagnostic_explicitly_moves_physx_masses_to_asset_device(self) -> None:
        source = inspect.getsource(passive_stand._run)
        self.assertIn("default_mass[0].to(device=robot.device, dtype=targets.dtype)", source)

    def test_static_pose_probe_pins_then_releases_root_with_baseline_gains(self) -> None:
        source = inspect.getsource(passive_stand._run)
        self.assertIn('"fixed_root_gravity_settling"', source)
        self.assertIn("robot.write_root_state_to_sim(initial_root_state)", source)
        self.assertIn("release_targets = settled_positions + required_torques / stiffness_tensor", source)
        self.assertIn('"hands_and_fingers_use_baseline_locked_pd": True', source)
        self.assertIn('"high_authority_profile_used": False', source)

    def test_finger_limit_tolerance_at_within_and_beyond_boundary(self) -> None:
        tolerance = model_spec.DERIVED_POSE_FINGER_LIMIT_TOLERANCE_RAD
        finger = "left_index_proximal_joint"
        for excess in (tolerance, tolerance - 1.0e-9):
            desired, target, audit = passive_stand.audit_and_clamp_derived_limit(
                finger, 1.0e-12, excess, (-1.6, 0.0)
            )
            self.assertTrue(audit["passed"])
            self.assertEqual(desired, 0.0)
            self.assertEqual(target, 0.0)
            self.assertTrue(audit["desired_was_clamped"])
            self.assertTrue(audit["target_was_clamped"])

        desired, target, audit = passive_stand.audit_and_clamp_derived_limit(
            finger, 0.0, tolerance + 1.0e-9, (-1.6, 0.0)
        )
        self.assertFalse(audit["passed"])
        self.assertEqual(desired, 0.0)
        self.assertGreater(target, 0.0)
        self.assertFalse(audit["target_was_clamped"])

    def test_limit_tolerance_never_applies_to_locomotion_joint(self) -> None:
        _, target, audit = passive_stand.audit_and_clamp_derived_limit(
            "left_ankle_pitch_joint", 0.0, 1.0e-12, (-0.6, 0.0)
        )
        self.assertFalse(audit["passed"])
        self.assertEqual(target, 1.0e-12)
        self.assertEqual(audit["tolerance_rad"], 0.0)

    def test_unmeasured_fixed_root_reaction_makes_contact_load_non_gating(self) -> None:
        derivation = {
            "settled_geometry": {"support_margin_m": 0.03},
            "maximum_required_torque_limit_fraction": 0.1,
            "maximum_mean_abs_settling_velocity_radps": 0.01,
            "fixed_root_load_balance": {
                "contact_force_body_weight_ratio": 0.0,
                "root_constraint_reaction_included": False,
                "observed_total_upward_reaction_body_weight_ratio": None,
            },
        }
        self.assertEqual(passive_stand.evaluate_static_derivation(derivation), [])
        derivation["fixed_root_load_balance"].update(
            root_constraint_reaction_included=True,
            observed_total_upward_reaction_body_weight_ratio=0.0,
        )
        self.assertIn(
            "measured fixed-root contact plus constraint reaction is outside 0.8-1.2 body weights",
            passive_stand.evaluate_static_derivation(derivation),
        )

    def test_free_root_support_checks_are_fail_closed(self) -> None:
        metrics = {
            "first_support_exit_time_s": None,
            "minimum_support_polygon_margin_m": 0.03,
            "peak_support_force_body_weight_ratio": 1.7,
            "mean_support_force_body_weight_ratio": 1.0,
        }
        self.assertEqual(passive_stand.evaluate_free_root_support(metrics), [])
        metrics.update(
            first_support_exit_time_s=0.8,
            minimum_support_polygon_margin_m=-0.001,
            peak_support_force_body_weight_ratio=3.1,
            mean_support_force_body_weight_ratio=0.2,
        )
        self.assertEqual(len(passive_stand.evaluate_free_root_support(metrics)), 4)

    def test_gate_rejects_events_motion_and_short_duration(self) -> None:
        metrics = {
            "duration_s": 29.0,
            "reset_count": 1,
            "done_count": 1,
            "fall_count": 1,
            "max_reference_tilt_rad": 0.0,
            "root_height_drop_m": 0.0,
            "horizontal_drift_m": 0.0,
            "max_abs_action": 0.1,
            "max_abs_command": 0.0,
        }
        passed, failures = passive_stand.evaluate_gate(metrics)
        self.assertFalse(passed)
        self.assertGreaterEqual(len(failures), 5)

    def test_gate_accepts_exact_zero_signal_stand(self) -> None:
        metrics = {
            "duration_s": 30.0,
            "reset_count": 0,
            "done_count": 0,
            "fall_count": 0,
            "max_reference_tilt_rad": 0.01,
            "root_height_drop_m": 0.005,
            "horizontal_drift_m": 0.002,
            "max_abs_action": 0.0,
            "max_abs_command": 0.0,
        }
        self.assertEqual(passive_stand.evaluate_gate(metrics), (True, []))

    def test_physics_source_has_no_isaac_camera_sensor(self) -> None:
        source = inspect.getsource(passive_stand)
        self.assertNotIn("from isaaclab.sensors import Camera", source)
        self.assertNotIn("CameraCfg(", source)
        self.assertIn("active_viewport_LdrColor_AOV", source)
        self.assertEqual(passive_stand.component_artifact_name("dynamics", False), "dynamics_validation.json")
        self.assertEqual(passive_stand.component_artifact_name("proof", False), "proof_validation.json")

    def test_proof_gate_checks_duration_frames_visibility_and_progression(self) -> None:
        good = {
            "duration_s": 30.0,
            "nonblank_frames_passed": True,
            "character_visibility_passed": True,
            "temporal_progression_visible": True,
        }
        self.assertEqual(passive_stand.evaluate_proof(good), (True, []))
        for key in ("nonblank_frames_passed", "character_visibility_passed", "temporal_progression_visible"):
            bad = dict(good)
            bad[key] = False
            self.assertFalse(passive_stand.evaluate_proof(bad)[0])

    def test_visual_geometry_projection_separates_framing_from_scale(self) -> None:
        def ndc(x: float, y: float, depth: float = 0.5) -> tuple[float, float, float]:
            return (x / 320.0 - 1.0, 1.0 - y / 240.0, depth)

        origin_proxy = passive_stand.projected_visual_bbox_metrics(
            [ndc(297.0, 226.0), ndc(338.0, 328.0)], 640, 480
        )
        self.assertTrue(origin_proxy["visual_geometry_framing_passed"])
        self.assertFalse(origin_proxy["discernible_scale_passed"])
        self.assertFalse(origin_proxy["character_visible"])

        visible_mesh = passive_stand.projected_visual_bbox_metrics(
            [ndc(270.0, 125.0), ndc(370.0, 355.0)], 640, 480
        )
        self.assertTrue(visible_mesh["visual_geometry_framing_passed"])
        self.assertTrue(visible_mesh["discernible_scale_passed"])
        self.assertTrue(visible_mesh["character_visible"])
        self.assertEqual(visible_mesh["required_visual_geometry_margin_px"], 19.2)

    def test_visual_geometry_projection_rejects_frustum_outlier(self) -> None:
        projected = [(-1.02, 0.0, 0.5), (0.4, -0.5, 0.5), (0.4, 0.5, 0.5)]
        result = passive_stand.projected_visual_bbox_metrics(projected, 640, 480)
        self.assertFalse(result["visual_geometry_framing_passed"])
        self.assertTrue(result["discernible_scale_passed"])
        self.assertFalse(result["character_visible"])

    def test_proof_gate_explains_visual_framing_and_scale_separately(self) -> None:
        common = {
            "duration_s": 30.0,
            "nonblank_frames_passed": True,
            "character_visibility_passed": False,
            "temporal_progression_visible": True,
        }
        framing = dict(common, visual_geometry_framing_passed=False, discernible_scale_passed=True)
        self.assertIn("frustum", " ".join(passive_stand.evaluate_proof(framing)[1]))
        scale = dict(common, visual_geometry_framing_passed=True, discernible_scale_passed=False)
        self.assertIn("too small", " ".join(passive_stand.evaluate_proof(scale)[1]))

    def test_proof_preflight_requires_current_passing_dynamics(self) -> None:
        passive_stand.OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=passive_stand.OUTPUT_ROOT) as temporary:
            output_dir = Path(temporary)
            path = output_dir / "dynamics_validation.json"
            evidence = {
                "component": "dynamics",
                "status": "passed",
                "seed": 42,
                "input": {"urdf_sha256": model_spec.EXPECTED_URDF_SHA256, "mesh_tree_sha256": model_spec.EXPECTED_MESH_TREE_SHA256},
                "checkpoint": {
                    "identity": f"robot_spec_sha256:{passive_stand._sha256(model_spec.ROBOT_SPEC_PATH)}"
                },
                "joint_contract": {"runtime_importer_axis_evidence": {"passed": True}},
            }
            path.write_text(json.dumps(evidence), encoding="utf-8")
            self.assertEqual(passive_stand.preflight_proof(output_dir, smoke=False, seed=42), evidence)
            evidence["status"] = "failed"
            path.write_text(json.dumps(evidence), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "passing dynamics"):
                passive_stand.preflight_proof(output_dir, smoke=False, seed=42)


if __name__ == "__main__":
    unittest.main()
