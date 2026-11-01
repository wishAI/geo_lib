import json
import tempfile
import unittest
from pathlib import Path

from algorithms.urdf_learn_wasd_walk import evolution


class EvolutionTests(unittest.TestCase):
    def test_initialization_crash_is_a_failed_experiment_not_a_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ledger = root / "milestones.json"
            ledger.write_text(json.dumps({
                "lineage": "current", "invalidatedLineages": [{"lineage": "stale"}],
                "milestones": [
                    {"id": "stand_zero_signal_30s_no_reset", "status": "passed"},
                    {"id": "stand_30s_no_reset", "status": "in_progress"},
                ],
            }))
            output = root / "outputs"
            for lineage in ("current", "stale", "unknown"):
                folder = output / lineage
                folder.mkdir(parents=True)
                (folder / "train_failure.json").write_text(json.dumps({
                    "lineage": lineage, "milestone": "stand_30s_no_reset",
                    "status": "failed_to_execute", "run_identity": "20260906T090000Z",
                    "runtime_stage": "fixed_root_gravity_settling",
                    "exception": {"type": "ValueError", "message": "unsettled joint"},
                }))
            tree = evolution.build_evolution(output, ledger)
            failures = [n for n in tree["nodes"] if n["id"].startswith("failure:")]
            self.assertEqual(len(failures), 2)
            for node in failures:
                self.assertEqual(node["status"], "failed")
                self.assertEqual(node["kind"], "experiment")
                self.assertEqual(node["metrics"], {})
                self.assertNotIn("checkpointSha256", node)
            current = next(n for n in failures if n["lineage"] == "current")
            stale = next(n for n in failures if n["lineage"] == "stale")
            self.assertEqual(tree["currentNodeId"], current["id"])
            self.assertTrue(stale["parentIds"][0].startswith("invalidated:"))
            self.assertIn("invalidated asset lineage", stale["result"])
    def test_backend_development_does_not_promote_canonical_gates(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "outputs"
            output.mkdir()
            ledger = root / "milestones.json"
            ledger.write_text(json.dumps({"lineage": "canonical", "milestones": [
                {"id": "stand_zero_signal_30s_no_reset", "status": "passed"},
                {"id": "stand_30s_no_reset", "status": "in_progress"}]}))
            (output / "backend_progress.json").write_text(json.dumps({
                "status": "paused_by_user", "updated_at": "2026-09-18T13:23:14Z",
                "latest_fully_unassisted": {"assistance_coefficient": 0, "teacher_blend": 0,
                    "proof_status": "visual review pending", "runs": [{"metrics": {
                        "duration_s": 120, "forward_m": 0.223, "fall": False,
                        "liftoffs": {"left": 5, "right": 4}, "reset_count": 0, "done_count": 0}}]}}))
            result = evolution.build_evolution(output, ledger)
            node = next(n for n in result["nodes"] if n["id"] == result["currentNodeId"])
            self.assertEqual(node["lineage"], "tk2_mujoco_development")
            self.assertFalse(node["gateEligible"])
            self.assertEqual(node["metrics"]["left_foot_liftoff_count"], 5)
            self.assertIn("visual review pending", node["result"])
            self.assertEqual(result["summary"]["passedMilestoneCount"], 1)
            self.assertEqual(result["milestones"][1]["status"], "in_progress")

    def test_real_artifacts_define_parentage_and_models_are_metadata_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "outputs"
            ledger = root / "milestones.json"
            ledger.write_text(json.dumps({
                "lineage": "clean_restart_2026_08_22",
                "milestones": [
                    {"id": "stand_zero_signal_30s_no_reset", "status": "passed", "checkpoint": {"identity": "robot-spec"}},
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
            payload = evolution.build_evolution(root / "outputs", ledger)
            nodes = {item["id"]: item for item in payload["nodes"]}
            self.assertEqual(nodes["milestone:stand_zero_signal_30s_no_reset"]["checkpointPath"], "robot-spec")
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

    def test_multiple_invalidated_meshes_form_auditable_ancestry(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ledger = root / "milestones.json"
            ledger.write_text(json.dumps({
                "lineage": "rabbit-ear-mesh",
                "assetContract": {"meshTreeSha256": "rabbit-ear-sha"},
                "invalidatedLineages": [
                    {"lineage": "old-mesh", "meshTreeSha256": "old-sha", "reason": "stale"},
                    {"lineage": "false-latest", "meshTreeSha256": "false-sha", "reason": "ears absent"},
                ],
                "milestones": [
                    {"id": "stand_zero_signal_30s_no_reset", "status": "in_progress"},
                    {"id": "stand_30s_no_reset", "status": "not_started"},
                ],
            }))
            payload = evolution.build_evolution(root / "outputs", ledger)
            nodes = {item["id"]: item for item in payload["nodes"]}
            first = "invalidated:old-mesh:stand_zero_signal_30s_no_reset"
            second = "invalidated:false-latest:stand_zero_signal_30s_no_reset"
            self.assertEqual(nodes[first]["parentIds"], [])
            self.assertEqual(nodes[second]["parentIds"], [first])
            self.assertEqual(nodes["milestone:stand_zero_signal_30s_no_reset"]["parentIds"], [second])
            self.assertEqual(payload["currentNodeId"], "milestone:stand_zero_signal_30s_no_reset")

    def test_current_node_uses_shared_run_chronology_not_artifact_kind(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "outputs"
            ledger = root / "milestones.json"
            ledger.write_text(json.dumps({
                "lineage": "latest-mesh",
                "milestones": [{"id": "gate_5m_no_reset", "status": "in_progress"}],
            }))
            probe_dir = output / "gate_5m_no_reset" / "reference_probe" / "passed"
            probe_dir.mkdir(parents=True)
            probe_dir.joinpath("reference_probe.json").write_text(json.dumps({
                "lineage": "latest-mesh", "milestone": "gate_5m_no_reset",
                "component": "open_loop_reference_probe", "run_identity": "20260905T010000Z",
                "status": "passed", "ppo_eligible": True,
            }))
            stage = output / "gate_5m_no_reset" / "stage40"
            stage.mkdir(parents=True)
            stage.joinpath("training.json").write_text(json.dumps({
                "lineage": "latest-mesh", "milestone": "gate_5m_no_reset",
                "run_identity": "20260905T020000Z", "status": "completed_not_promoted",
                "requested_contract": {"training_method": "v3"},
                "checkpoint": {"sha256": "stage-sha", "path": "ignored.pt"},
            }))
            payload = evolution.build_evolution(output, ledger)
            self.assertEqual(payload["currentNodeId"], "run:20260905T020000Z")

    def test_current_passive_diagnostic_is_visible_without_promoting_the_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "outputs"
            ledger = root / "milestones.json"
            ledger.write_text(json.dumps({
                "lineage": "rabbit-ear-mesh",
                "milestones": [
                    {"id": "stand_zero_signal_30s_no_reset", "status": "in_progress"},
                    {"id": "stand_30s_no_reset", "status": "not_started"},
                ],
            }))
            diagnostic_dir = output / "stand_zero_signal_30s_no_reset"
            diagnostic_dir.mkdir(parents=True)
            diagnostic_dir.joinpath("dynamics_smoke_validation.json").write_text(json.dumps({
                "lineage": "rabbit-ear-mesh",
                "milestone": "stand_zero_signal_30s_no_reset",
                "component": "dynamics",
                "scope": "diagnostic_experiment",
                "run_identity": "20260905T091152Z",
                "status": "passed",
                "gate_eligible": False,
                "experiment": {"id": "gravity_static_pose_release_v1", "diagnostic_only": True},
                "metrics": {
                    "duration_s": 5.0,
                    "physics_steps": 2500,
                    "reset_count": 0,
                    "fall_count": 0,
                    "max_reference_tilt_rad": 0.1,
                },
                "failures": [],
            }))

            payload = evolution.build_evolution(output, ledger)
            node = next(item for item in payload["nodes"] if item["id"] == "experiment:20260905T091152Z")
            self.assertEqual(node["parentIds"], ["milestone:stand_zero_signal_30s_no_reset"])
            self.assertEqual(node["status"], "completed")
            self.assertEqual(node["experimentParameters"]["duration_s"], 5.0)
            self.assertFalse(node["experimentParameters"]["gate_eligible"])
            self.assertEqual(payload["currentNodeId"], node["id"])

class EvidenceIdentityTests(unittest.TestCase):
    def test_stale_or_corrupt_video_is_not_attached(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            video = folder / 'proof.mp4'
            video.write_bytes(b'video')
            proof = {'lineage': 'old', 'run_identity': 'run', 'video_inspection': {'sha256': evolution._digest(video)}}
            (folder / 'proof_validation.json').write_text(json.dumps(proof))
            self.assertEqual(evolution._proof_artifacts(folder, 'node', {'lineage': 'new', 'run_identity': 'run'}), [])
            proof['lineage'] = 'new'
            (folder / 'proof_validation.json').write_text(json.dumps(proof))
            self.assertEqual(len(evolution._proof_artifacts(folder, 'node', {'lineage': 'new', 'run_identity': 'run'})), 1)
            video.write_bytes(b'changed')
            self.assertEqual(evolution._proof_artifacts(folder, 'node', {'lineage': 'new', 'run_identity': 'run'}), [])

    def test_iso_dates_sort_after_older_compact_run_dates(self):
        self.assertGreater(evolution._chronology({'startedAt': '2026-09-22T01:00:00Z'}), evolution._chronology({'startedAt': '20260905T120000Z'}))

    def test_same_rollout_video_requires_unchanged_evidence(self):
        for key, filename in (("evaluation_sha256", "evaluation.json"),
                              ("result_sha256", "result.json"),
                              ("dynamics_sha256", "dynamics.json"),
                              ("model_xml_sha256", "model.xml")):
            with self.subTest(filename=filename), tempfile.TemporaryDirectory() as temporary:
                folder = Path(temporary)
                for name in ('proof.mp4', 'trajectory.npz', filename):
                    (folder / name).write_bytes(b'original')
                metadata = {field: evolution._digest(folder / name) for field, name in (
                    ('video_sha256', 'proof.mp4'), ('trajectory_sha256', 'trajectory.npz'),
                    (key, filename))}
                (folder / 'proof_metadata.json').write_text(json.dumps(metadata))
                self.assertEqual(len(evolution._proof_artifacts(folder, 'node', {})), 2)
                (folder / filename).write_bytes(b'changed')
                self.assertEqual(evolution._proof_artifacts(folder, 'node', {}), [])


if __name__ == "__main__":
    unittest.main()
