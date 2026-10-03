import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from algorithms.urdf_learn_wasd_walk import model_comparison as comparison
from algorithms.urdf_learn_wasd_walk import evolution
from algorithms.urdf_learn_wasd_walk.tests.test_geo_launcher import _load_geo_module


class ModelComparisonTests(unittest.TestCase):
    def test_checkpoint_only_control_requires_identical_config_and_reference_identity(self):
        training = {**comparison.identity("unitree_g1"), "asset": {"hash": "asset"},
                    "checkpoint": {"sha256": "local-smoke"}}
        evaluation = {**training, "status": "failed"}
        config = {"environment": {"seed": 42, "command": .5}, "ppo": {"hidden": [256, 128, 128]}}
        comparison.require_same_evaluation(config, config, evaluation, training)
        with self.assertRaisesRegex(ValueError, "configuration"):
            comparison.require_same_evaluation({**config, "seed": 7}, config, evaluation, training)
        for key in ("model", "protocol", "lineage", "checkpoint", "asset", "landau_gate_eligible"):
            with self.assertRaises(ValueError):
                comparison.require_same_evaluation(config, config, {**evaluation, key: "different"}, training)

    def test_builtin_mdl_resolution_is_exact_hashed_and_not_a_geometry_exemption(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            module = root / "mdl/core/Base/OmniPBR.mdl"
            module.parent.mkdir(parents=True)
            module.write_bytes(b"mdl 1.7; // test only")
            records = comparison.resolve_builtin_mdl(["OmniPBR.mdl"] * 3, root)
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["sha256"], comparison.digest(module))
            self.assertEqual(records[0]["authored_path"], "OmniPBR.mdl")
            self.assertEqual(records[0]["kind"], "installed_kit_builtin_mdl")
            for missing in ("foot.stl", "robot.usd", "Unknown.mdl", "./OmniPBR.mdl", "../OmniPBR.mdl"):
                with self.assertRaisesRegex(ValueError, "unresolved dependency"):
                    comparison.resolve_builtin_mdl(["OmniPBR.mdl", missing], root)
            module.unlink()
            with self.assertRaisesRegex(ValueError, "missing installed Kit"):
                comparison.resolve_builtin_mdl(["OmniPBR.mdl"], root)
            self.assertEqual(comparison.resolve_builtin_mdl([], root), [])

    def test_contact_noise_or_sliding_cannot_pass_g1_locomotion_diagnostic(self):
        metrics = {"done_count": 0, "reset_count": 0, "fall_count": 0,
                   "max_root_tilt_rad": .1, "root_height_drop_m": .01,
                   "forward_axis_world_displacement_m": .4,
                   "left_liftoffs": 3, "right_liftoffs": 3,
                   "max_air_steps": [5, 4], "max_foot_height_gain_m": [.03, .02]}
        self.assertEqual(comparison.evaluation_failures(metrics, "unitree_g1"), [])
        for changed in ({"max_air_steps": [1, 1]}, {"max_foot_height_gain_m": [0., 0.]},
                        {"max_root_tilt_rad": 1.}, {"done_count": 1},
                        {"forward_axis_world_displacement_m": -.2}):
            self.assertTrue(comparison.evaluation_failures({**metrics, **changed}, "unitree_g1"))

    def test_symlink_cannot_alias_another_models_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "landau_current").mkdir()
            (root / "unitree_g1").symlink_to(root / "landau_current", target_is_directory=True)
            with patch.object(comparison, "OUTPUT", root), self.assertRaisesRegex(ValueError, "another model"):
                comparison.output_dir("unitree_g1", "smoke")

    def test_selection_is_allowlisted_and_output_roots_do_not_overlap(self):
        g1 = comparison.output_dir("unitree_g1", "smoke")
        landau = comparison.output_dir("landau_current", "smoke")
        self.assertNotEqual(g1, landau)
        for model, tag in (("unknown", "smoke"), ("unitree_g1", "../escape"), ("landau_current", "/tmp/x")):
            with self.assertRaises(ValueError):
                comparison.output_dir(model, tag)
        self.assertEqual(comparison.identity("landau_current")["semantic_forward_axis"], "+Y")
        self.assertFalse(comparison.identity("unitree_g1")["landau_gate_eligible"])

    def test_wrong_model_and_changed_checkpoint_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "model.pt"
            checkpoint.write_bytes(b"fake unit test checkpoint")
            evidence = {**comparison.identity("unitree_g1"), "status": "completed_not_promoted",
                        "checkpoint": {"path": str(checkpoint), "sha256": comparison.digest(checkpoint)}}
            comparison.validate_checkpoint(evidence, "unitree_g1", root)
            with self.assertRaisesRegex(ValueError, "lineage"):
                comparison.validate_checkpoint(evidence, "landau_current", root)
            checkpoint.write_bytes(b"different")
            with self.assertRaisesRegex(ValueError, "hash"):
                comparison.validate_checkpoint(evidence, "unitree_g1", root)

    def test_direct_contacts_do_not_require_float_first_contact_timers(self):
        previous, air, liftoffs = [True, True], [0, 0], [0, 0]
        for current in ([False, True], [False, False], [True, False], [True, True]):
            previous = comparison.observed_contact(previous, current, air, liftoffs)
        self.assertEqual(liftoffs, [1, 1])
        self.assertEqual(air, [0, 0])

    def test_geo_model_selection_routes_separate_artifacts(self):
        geo = _load_geo_module()
        specs = []
        for model in comparison.MODELS:
            args, extra = geo._build_parser().parse_known_args([
                "walk", "compare-model", "--model", model, "--mode", "train", "--headless"])
            specs.append(geo._build_spec(args, extra))
        self.assertTrue(all(s.runner == "isaac" for s in specs))
        self.assertNotEqual(specs[0].success_artifact, specs[1].success_artifact)

    def test_comparison_branch_cannot_replace_canonical_landau_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ledger = root / "milestones.json"
            ledger.write_text(json.dumps({"lineage": "current-landau", "milestones": [
                {"id": "stand_zero_signal_30s_no_reset", "status": "passed"},
                {"id": "stand_30s_no_reset", "status": "in_progress"}]}))
            output = root / "outputs"
            folder = output / "model_comparison/unitree_g1/smoke"
            folder.mkdir(parents=True)
            training = {**comparison.identity("unitree_g1"), "status": "completed_not_promoted",
                        "asset": {"asset_tree_sha256": "test-hash"}, "checkpoint": {"sha256": "fake"}}
            (folder / "training.json").write_text(json.dumps(training))
            tree = evolution.build_evolution(output, ledger)
            node = next(n for n in tree["nodes"] if n.get("model") == "unitree_g1")
            self.assertEqual(node["parentIds"], [])
            self.assertEqual(node["assetTreeSha256"], "test-hash")
            self.assertEqual(tree["currentNodeId"], "milestone:stand_30s_no_reset")

            (folder / "training.json").rename(folder / "checkpoint_import.json")
            training["status"] = "imported_not_trained_not_promoted"
            (folder / "checkpoint_import.json").write_text(json.dumps(training))
            tree = evolution.build_evolution(output, ledger)
            node = next(n for n in tree["nodes"] if n.get("model") == "unitree_g1")
            self.assertIn("imported_not_trained", node["result"])
            self.assertEqual(node["parentIds"], [])
            self.assertEqual(tree["currentNodeId"], "milestone:stand_30s_no_reset")


if __name__ == "__main__":
    unittest.main()
