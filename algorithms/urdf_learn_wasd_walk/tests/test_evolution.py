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
            payload = evolution.build_evolution(output, ledger)
            nodes = {item["id"]: item for item in payload["nodes"]}
            self.assertEqual(nodes["run:phase-run"]["parentIds"], ["milestone:stand_30s_no_reset"])
            self.assertEqual(nodes["run:phase-run"]["status"], "failed")
            self.assertEqual(nodes["run:phase-run"]["checkpointStorage"]["macHydration"], "online-only")
            self.assertTrue(all(not artifact["path"].endswith(".pt") for node in nodes.values() for artifact in node.get("artifacts", [])))
            self.assertLessEqual(len(payload["defaultVisibleNodeIds"]), 40)


if __name__ == "__main__":
    unittest.main()
