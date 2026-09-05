from __future__ import annotations

import json
import unittest
from pathlib import Path

from webgui import server, storage


class ManifestTests(unittest.TestCase):
    def test_every_algorithm_has_a_unique_gui_manifest(self) -> None:
        manifests = server.discover_manifests()
        icon_names = {"headset", "point-cloud", "arm", "route", "map", "vector", "walk", "robot", "nest"}
        algorithm_names = {
            path.name
            for path in (server.REPO_ROOT / "algorithms").iterdir()
            if path.is_dir() and not path.name.startswith(".") and path.name != "__pycache__"
        }
        self.assertEqual({item["id"] for item in manifests}, algorithm_names)
        self.assertEqual(len(manifests), len({item["id"] for item in manifests}))
        for manifest in manifests:
            self.assertTrue(manifest["name"])
            self.assertTrue(manifest["summary"])
            self.assertTrue(manifest["accent"].startswith("#"))
            self.assertIn(manifest["icon"], icon_names)
        self.assertEqual(len(manifests), len({item["icon"] for item in manifests}))

    def test_examples_are_allowlisted_argument_arrays(self) -> None:
        for manifest in server.discover_manifests():
            for example in manifest.get("examples", []):
                self.assertIsInstance(example["command"], list)
                self.assertTrue(example["command"])
                self.assertLessEqual(set(example.get("targets", [])), {"local", "tk2"})
                commands = [example["command"], *example.get("commands", {}).values()]
                for command in commands:
                    self.assertIsInstance(command, list)
                    self.assertTrue(command)
                    self.assertNotIn("sh", command[:1])
                    self.assertNotIn("bash", command[:1])
                    for token in command:
                        self.assertNotIn(";", token)
                        self.assertNotIn("&&", token)

    def test_numeric_parameters_are_validated_before_command_build(self) -> None:
        manifest = server.manifest_map()["svg_scene_builder"]
        example = manifest["examples"][0]
        command = server.build_example_command(manifest, example, {"resolution": 0.04})
        self.assertEqual(command[-1], "0.04")
        with self.assertRaisesRegex(ValueError, "below the minimum"):
            server.build_example_command(manifest, example, {"resolution": 0.0001})
        with self.assertRaisesRegex(ValueError, "Unknown parameters"):
            server.build_example_command(manifest, example, {"resolution": 0.02, "command": "oops"})

    def test_walk_sandbox_preserves_clean_lineage_and_exposes_current_gate(self) -> None:
        root = server.REPO_ROOT / "algorithms" / "urdf_learn_wasd_walk"
        payload = json.loads((root / "milestones.json").read_text(encoding="utf-8"))
        self.assertEqual(len(payload["milestones"]), 12)
        self.assertEqual(payload["milestones"][0]["status"], "passed")
        self.assertEqual(payload["milestones"][1]["status"], "passed")
        self.assertEqual(payload["milestones"][2]["status"], "in_progress")
        self.assertEqual({item["status"] for item in payload["milestones"][3:]}, {"not_started"})
        self.assertFalse(payload["historyCarriedForward"])
        manifest = server.manifest_map()["urdf_learn_wasd_walk"]
        self.assertEqual(
            [example["id"] for example in manifest["examples"]],
            [
                "validate_passive_stand", "train_policy_stand", "validate_policy_stand",
                "train_forward_walk", "validate_forward_walk",
            ],
        )
        example = manifest["examples"][0]
        self.assertEqual(example["command"][:3], ["./geo", "walk", "validate-passive"])
        self.assertEqual({artifact["kind"] for artifact in example["artifacts"]}, {"json", "video", "image"})
        policy_validation = manifest["examples"][2]
        self.assertEqual(policy_validation["command"][:3], ["./geo", "walk", "validate-policy-stand"])
        self.assertIn("video", {artifact["kind"] for artifact in policy_validation["artifacts"]})
        forward_validation = manifest["examples"][4]
        self.assertEqual(
            forward_validation["command"][:3], ["./geo", "walk", "validate-forward-walk"]
        )
        self.assertIn("video", {artifact["kind"] for artifact in forward_validation["artifacts"]})
        self.assertEqual(manifest["inspector"]["type"], "evolutionTree")
        self.assertIn(manifest["inspector"]["path"], server.declared_artifact_paths())
        self.assertFalse(any(
            artifact["path"].endswith((".pt", ".pth", ".ckpt", ".onnx", ".engine", ".safetensors"))
            for example in manifest["examples"] for artifact in example.get("artifacts", [])
        ))


class StorageAndRobotTests(unittest.TestCase):
    def test_large_file_manifest_is_deduplicated_and_complete(self) -> None:
        manifest = storage.load_manifest()
        self.assertEqual(manifest["thresholdBytes"], 5 * 1024 * 1024)
        self.assertEqual(len(manifest["files"]), 12)
        self.assertEqual(len({item["cloudPath"] for item in manifest["files"]}), 3)
        self.assertTrue(all(len(item["sha256"]) == 64 for item in manifest["files"]))

    def test_repo_has_no_tracked_file_over_threshold(self) -> None:
        self.assertEqual(storage.audit_tracked_files()["oversizedTrackedFiles"], [])

    def test_path_guards_reject_escape(self) -> None:
        with self.assertRaises(ValueError):
            storage._safe_repo_path("../outside")
        with self.assertRaises(ValueError):
            server._safe_under(server.REPO_ROOT, "../outside")

    def test_declared_robot_urdfs_are_parseable_and_have_controls(self) -> None:
        candidates = [item for item in server.robot_candidates() if item["exists"]]
        self.assertGreaterEqual(len(candidates), 4)
        for candidate in candidates:
            resolved = server.resolve_artifact(candidate["path"])
            self.assertIsNotNone(resolved)
            joints = server._robot_joint_info(resolved[1].read_text(encoding="utf-8"))
            self.assertTrue(joints, candidate["path"])
            self.assertTrue(all(item["lower"] <= 0 <= item["upper"] for item in joints))

    def test_humanoid_joint_controls_are_grouped_semantically(self) -> None:
        self.assertEqual(server._joint_group("left_shoulder_pitch_joint"), "left_arm")
        self.assertEqual(server._joint_group("right_index_distal_joint"), "right_arm")
        self.assertEqual(server._joint_group("left_knee_joint"), "left_leg")
        self.assertEqual(server._joint_group("right_ankle_pitch_joint"), "right_leg")
        self.assertEqual(server._joint_group("waist_yaw_joint"), "body")

    def test_mesh_workbench_catalog_and_apply_command_are_allowlisted(self) -> None:
        catalog = server._mesh_catalog("usd_parallel_urdf")
        self.assertTrue(catalog["parts"])
        self.assertEqual(catalog["applyExample"], "apply_mesh_settings")
        self.assertEqual(catalog["target"], "local")
        self.assertEqual(catalog["targets"], ["local", "tk2"])
        manifest = server.manifest_map()["usd_parallel_urdf"]
        example = next(item for item in manifest["examples"] if item["id"] == catalog["applyExample"])
        parameters = {
            "method": "convex_hull",
            "target_face_ratio": 0.2,
            "max_faces": 800,
            "max_hull_faces": 64,
            "target_hull_points": 32,
            "min_thickness": 0.003,
        }
        command = server.build_example_command(manifest, example, parameters, "local")
        remote_command = server.build_example_command(manifest, example, parameters, "tk2")
        self.assertEqual(command[0], "blender")
        self.assertEqual(remote_command[:3], ["./geo", "usd", "build-mesh"])
        self.assertIn("convex_hull", command)
        self.assertIn("0.003", command)
        preview = server._mesh_part_urdf("usd_parallel_urdf", catalog["parts"][0]["name"], "stl")
        self.assertIn("<mesh", preview["urdf"])
        self.assertEqual(preview["joints"], [])


if __name__ == "__main__":
    unittest.main()
