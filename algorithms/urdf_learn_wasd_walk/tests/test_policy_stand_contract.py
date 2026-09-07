from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from algorithms.urdf_learn_wasd_walk import model_spec, policy_stand_contract as contract


def _passive_gate_is_current() -> bool:
    import json

    ledger = json.loads((model_spec.ALGORITHM_ROOT / "milestones.json").read_text())
    return next(item for item in ledger["milestones"] if item["id"] == contract.PRIOR_MILESTONE_ID)["status"] == "passed"


def passing_metrics() -> dict:
    return {
        "duration_s": 30.0,
        "control_steps": 1500,
        "policy_inference_steps": 1500,
        "reset_count": 0,
        "done_count": 0,
        "fall_count": 0,
        "max_reference_tilt_rad": 0.05,
        "root_height_drop_m": 0.005,
        "horizontal_drift_m": 0.01,
        "max_abs_command": 0.0,
        "max_abs_action": 0.2,
    }


class PolicyStandContractTests(unittest.TestCase):
    def test_actor_observation_and_action_contract_is_deployable(self) -> None:
        self.assertEqual(contract.ACTOR_OBSERVATION_DIM, 60)
        self.assertEqual(contract.ACTOR_OBSERVATION_TERMS[-1], ("previous_action", 17))
        training = contract.training_contract(seed=42, num_envs=512, iterations=200)
        self.assertEqual(training["environment"]["action_joints"], list(model_spec.ACTION_JOINTS))
        self.assertFalse(training["environment"]["privileged_critic_observations"])
        self.assertFalse(training["environment"]["symmetry_augmentation"])
        self.assertEqual(training["sample_count"], 512 * 200 * 24)

    def test_policy_gate_fails_closed(self) -> None:
        self.assertEqual(contract.evaluate_policy_gate(passing_metrics()), [])
        for field in ("reset_count", "done_count", "fall_count"):
            metrics = passing_metrics()
            metrics[field] = 1
            self.assertTrue(contract.evaluate_policy_gate(metrics), field)
        metrics = passing_metrics()
        metrics["policy_inference_steps"] -= 1
        self.assertIn("checkpoint actor", " ".join(contract.evaluate_policy_gate(metrics)))
        metrics = passing_metrics()
        metrics["max_abs_action"] = 1.01
        self.assertIn("clip contract", " ".join(contract.evaluate_policy_gate(metrics)))

    def test_prior_passive_gate_is_hash_checked(self) -> None:
        checkpoint = {
            "identity": "robot_spec_sha256:test",
            "kind": "derived_static_pose_passive_pd_configuration",
            "policy_checkpoint": None,
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            evidence_path = root / "passive_validation.json"
            evidence_path.write_text(json.dumps({
                "status": "passed",
                "milestone": contract.PRIOR_MILESTONE_ID,
                "input": {
                    "urdf_sha256": model_spec.EXPECTED_URDF_SHA256,
                    "mesh_tree_sha256": model_spec.EXPECTED_MESH_TREE_SHA256,
                },
                "checkpoint": checkpoint,
            }), encoding="utf-8")
            milestones_path = root / "milestones.json"
            milestones_path.write_text(json.dumps({
                "milestones": [{
                    "id": contract.PRIOR_MILESTONE_ID,
                    "status": "passed",
                    "checkpoint": checkpoint,
                    "evidence": [{
                        "kind": "validation",
                        "path": evidence_path.name,
                        "sha256": contract.sha256(evidence_path),
                    }],
                }],
            }), encoding="utf-8")

            prior = contract.load_prior_gate(
                milestones_path=milestones_path, repo_root=root,
            )
            self.assertEqual(prior["status"], "passed")
            self.assertEqual(prior["urdf_sha256"], model_spec.EXPECTED_URDF_SHA256)

            evidence_path.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "declared hash"):
                contract.load_prior_gate(
                    milestones_path=milestones_path, repo_root=root,
                )


if __name__ == "__main__":
    unittest.main()
