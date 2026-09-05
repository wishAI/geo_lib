import json
import tempfile
import unittest
from pathlib import Path

from algorithms.urdf_learn_wasd_walk import evolution


class EvolutionTests(unittest.TestCase):
    def test_real_artifacts_define_parentage_and_models_are_metadata_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "outputs"
            ledger = root / "milestones.json"
            ledger.write_text(json.dumps({
                "lineage": "clean_restart_2026_08_22",
                "milestones": [
                    {"id": "stand_zero_signal_30s_no_reset", "status": "passed", "checkpoint": "robot"},
                    {"id": "stand_30s_no_reset", "status": "passed", "checkpoint": {"sha256": "stand-sha"}},
                    {"id": "gate_5m_no_reset", "status": "in_progress"},
                ],
            }))
            stand = output / "stand_30s_no_reset"
            stand.mkdir(parents=True)
            stand.joinpath("training.json").write_text(json.dumps({
                "lineage": "clean_restart_2026_08_22", "milestone": "stand_30s_no_reset",
                "run_identity": "stand-run", "status": "completed_not_promoted",
                "requested_contract": {"algorithm": "PPO"},
                "checkpoint": {"sha256": "stand-sha", "path": "secret/model.pt", "size_bytes": 10},
            }))
            branch = output / "gate_5m_no_reset" / "phase_v2"
            branch.mkdir(parents=True)
            branch.joinpath("training.json").write_text(json.dumps({
                "lineage": "clean_restart_2026_08_22", "milestone": "gate_5m_no_reset",
                "run_identity": "phase-run", "status": "completed_not_promoted",
                "requested_contract": {"training_method": "phase", "initialization": {"sha256": "stand-sha"}},
                "checkpoint": {"sha256": "phase-sha", "path": "secret/phase.pt", "size_bytes": 20},
            }))
            branch.joinpath("forward_dynamics_validation.json").write_text(json.dumps({
                "status": "passed", "gate_eligible": False,
                "metrics": {"semantic_forward_displacement_m": 0.004, "left_foot_liftoff_count": 0, "right_foot_liftoff_count": 0},
            }))
            probe_dir = output / "gate_5m_no_reset" / "reference_probe_v3" / "baseline"
            probe_dir.mkdir(parents=True)
            probe_dir.joinpath("reference_probe.json").write_text(json.dumps({
                "lineage": "clean_restart_2026_08_22",
                "component": "open_loop_reference_probe",
                "experiment": "command_gated_phase_reference_residual_v3",
                "run_identity": "probe-run",
                "status": "failed",
                "ppo_eligible": False,
                "parent_checkpoint": {"sha256": "phase-sha", "path": "secret/phase.pt"},
                "reference_contract": {"parameters": {"amplitude_scale": 1.0}},
                "metrics": {
                    "semantic_forward_displacement_m": 0.01,
                    "left_foot_liftoff_count": 0,
                    "right_foot_liftoff_count": 0,
                },
                "failures": ["left side stayed planted"],
            }))
            payload = evolution.build_evolution(output, ledger)
            nodes = {item["id"]: item for item in payload["nodes"]}
            self.assertEqual(nodes["run:phase-run"]["parentIds"], ["milestone:stand_30s_no_reset"])
            self.assertEqual(nodes["run:phase-run"]["status"], "failed")
            self.assertEqual(nodes["run:phase-run"]["checkpointStorage"]["macHydration"], "online-only")
            self.assertEqual(nodes["experiment:probe-run"]["parentIds"], ["run:phase-run"])
            self.assertEqual(nodes["experiment:probe-run"]["status"], "failed")
            self.assertEqual(nodes["experiment:probe-run"]["experimentParameters"]["amplitude_scale"], 1.0)
            self.assertEqual(payload["currentNodeId"], "experiment:probe-run")
            self.assertTrue(all(not artifact["path"].endswith(".pt") for node in nodes.values() for artifact in node.get("artifacts", [])))
            self.assertLessEqual(len(payload["defaultVisibleNodeIds"]), 40)

    def test_invalidated_asset_lineage_remains_visible_but_is_not_current(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "outputs"
            ledger = root / "milestones.json"
            ledger.write_text(json.dumps({
                "lineage": "latest-mesh",
                "assetContract": {"meshTreeSha256": "new-mesh"},
                "invalidatedLineage": {
                    "lineage": "old-mesh",
                    "meshTreeSha256": "old-mesh-sha",
                    "reason": "stale collision meshes",
                },
                "milestones": [
                    {"id": "stand_zero_signal_30s_no_reset", "status": "in_progress"},
                    {"id": "stand_30s_no_reset", "status": "not_started"},
                ],
            }))
            old_run = output / "stand_30s_no_reset"
            old_run.mkdir(parents=True)
            old_run.joinpath("training.json").write_text(json.dumps({
                "lineage": "old-mesh",
                "milestone": "stand_30s_no_reset",
                "run_identity": "old-stand",
                "status": "completed_not_promoted",
                "requested_contract": {"algorithm": "PPO"},
                "checkpoint": {"sha256": "old-checkpoint", "path": "old.pt"},
            }))

            payload = evolution.build_evolution(output, ledger)
            nodes = {item["id"]: item for item in payload["nodes"]}
            self.assertEqual(payload["currentNodeId"], "milestone:stand_zero_signal_30s_no_reset")
            self.assertEqual(nodes["milestone:stand_zero_signal_30s_no_reset"]["status"], "running")
            self.assertEqual(nodes["run:old-stand"]["status"], "failed")
            self.assertIn("invalidated asset lineage", nodes["run:old-stand"]["result"])
            self.assertEqual(nodes["run:old-stand"]["parentIds"], ["invalidated:old-mesh:stand_zero_signal_30s_no_reset"])

            current_ledger = json.loads(ledger.read_text())
            current_ledger["milestones"][0]["status"] = "passed"
            current_ledger["milestones"][1] = {
                "id": "stand_30s_no_reset", "status": "in_progress"
            }
            ledger.write_text(json.dumps(current_ledger))
            payload = evolution.build_evolution(output, ledger)
            nodes = {item["id"]: item for item in payload["nodes"]}
            self.assertEqual(payload["currentNodeId"], "milestone:stand_30s_no_reset")
            self.assertEqual(nodes["milestone:stand_30s_no_reset"]["status"], "running")
            self.assertEqual(nodes["run:old-stand"]["status"], "failed")

            current_ledger["milestones"][1]["status"] = "passed"
            current_ledger["milestones"].append({
                "id": "gate_5m_no_reset", "status": "in_progress"
            })
            ledger.write_text(json.dumps(current_ledger))
            payload = evolution.build_evolution(output, ledger)
            nodes = {item["id"]: item for item in payload["nodes"]}
            self.assertEqual(payload["currentNodeId"], "milestone:gate_5m_no_reset")
            self.assertEqual(nodes["milestone:gate_5m_no_reset"]["status"], "running")


if __name__ == "__main__":
    unittest.main()
