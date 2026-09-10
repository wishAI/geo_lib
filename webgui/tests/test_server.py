from __future__ import annotations

import json
import math
import shutil
import struct
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from webgui import server, storage


class ManifestTests(unittest.TestCase):
    def test_every_algorithm_has_a_unique_gui_manifest(self) -> None:
        manifests = server.discover_manifests()
        icon_names = {"headset", "point-cloud", "arm", "route", "map", "vector", "walk", "robot", "nest", "ship", "sliders", "pawn"}
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

    def test_stellaris_designer_uses_a_valid_editable_input_contract(self) -> None:
        manifest = server.manifest_map()["stellaris_ship_designer"]
        self.assertEqual(manifest["designer"]["type"], "stellarisShipDesigner")
        self.assertEqual([item["id"] for item in manifest["designer"]["designs"]], [
            "mammalian_battleship", "biogenesis_mauler_stage_1",
        ])
        design = server.load_ship_design("stellaris_ship_designer", "mammalian_battleship")
        self.assertEqual([section["name"] for section in design["sections"]], [
            "Spinal Mount Bow", "Artillery Core", "Artillery Stern",
        ])
        self.assertEqual(
            [slot["size"] for section in design["sections"] for slot in section["slots"] if slot["type"] == "weapon"],
            ["X", "L", "L", "L", "L"],
        )
        self.assertTrue(all(len(locator["position"]) == 3 for locator in design["locators"]))
        mauler = server.load_ship_design("stellaris_ship_designer", "biogenesis_mauler_stage_1")
        self.assertEqual(len(mauler["model"]["bones"]), 16)
        self.assertEqual([clip["id"] for clip in mauler["animation"]["clips"]], [
            "idle", "combat_moving", "attack_source_disabled",
        ])
        self.assertFalse(mauler["animation"]["clips"][-1]["officialBinding"])
        self.assertEqual(mauler["officialRules"]["maxHitpoints"], 300)
        self.assertEqual(mauler["officialRules"]["maxSpeed"], 160)
        self.assertEqual(mauler["officialRules"]["fleetSlotSize"], 1)
        mauler_weapon_slots = [
            slot for slot in mauler["sections"][0]["slots"]
            if slot["type"] in {"weapon", "guided"}
        ]
        self.assertEqual({slot["sourceTemplateLocator"] for slot in mauler_weapon_slots}, {"root"})
        self.assertEqual({slot["locatorId"] for slot in mauler_weapon_slots}, {"loc_official_fire_root"})
        locator_map = {locator["id"]: locator for locator in mauler["locators"]}
        self.assertEqual(locator_map["loc_official_fire_root"]["usage"], "official_slot_binding")
        self.assertEqual(locator_map["loc_weapon_01"]["usage"], "embedded_unbound")
        self.assertEqual(locator_map["loc_weapon_02"]["usage"], "embedded_unbound")
        self.assertNotIn("fireAxis", locator_map["loc_official_fire_root"])
        self.assertEqual(design["officialRules"]["fleetSlotSize"], 4)
        self.assertEqual(design["officialRules"]["rotationSpeed"], 0.15)
        self.assertEqual(design["sections"][0]["key"], "BATTLESHIP_BOW_M2S4")
        self.assertEqual(design["ship"]["forwardAxis"], "+Z")
        self.assertEqual(design["model"]["cameraDirection"], [1.05, 0.62, 0.18])
        weapon_locators = [
            locator for locator in design["locators"]
            if locator.get("usage") == "official_slot_binding"
        ]
        self.assertEqual({locator["fireAxis"] for locator in weapon_locators}, {"+Y"})
        self.assertTrue(all("ship +Z/bow" in locator["fireAxisEvidence"] for locator in weapon_locators))
        visible_battle_locators = [locator for locator in design["locators"] if locator["visible"]]
        self.assertEqual(len(visible_battle_locators), 5)
        self.assertTrue(all(locator.get("usage") == "official_slot_binding" for locator in visible_battle_locators))
        model_bytes = server._designer_model("stellaris_ship_designer", "mammalian_battleship").read_bytes()
        json_length, = struct.unpack_from("<I", model_bytes, 12)
        glb = json.loads(model_bytes[20:20 + json_length].decode("utf-8").rstrip(" \x00"))
        glb_nodes = {node.get("name"): node for node in glb["nodes"]}
        for locator in design["locators"]:
            source_node = glb_nodes.get(locator.get("sourceNode"))
            if not source_node or "translation" not in source_node:
                continue
            for configured, embedded in zip(locator["position"], source_node["translation"]):
                self.assertAlmostEqual(configured, embedded, places=4, msg=locator["id"])
            x, y, z = [math.radians(value) / 2 for value in locator["rotation"]]
            sx, cx = math.sin(x), math.cos(x)
            sy, cy = math.sin(y), math.cos(y)
            sz, cz = math.sin(z), math.cos(z)
            configured_quaternion = (
                sx * cy * cz + cx * sy * sz,
                cx * sy * cz - sx * cy * sz,
                cx * cy * sz + sx * sy * cz,
                cx * cy * cz - sx * sy * sz,
            )
            embedded_quaternion = source_node.get("rotation", [0, 0, 0, 1])
            quaternion_dot = sum(a * b for a, b in zip(configured_quaternion, embedded_quaternion))
            self.assertAlmostEqual(abs(quaternion_dot), 1, places=4, msg=locator["id"])
        self.assertEqual(server._designer_model("stellaris_ship_designer", "mammalian_battleship").stat().st_size, 9218152)
        self.assertEqual(server._designer_model("stellaris_ship_designer", "biogenesis_mauler_stage_1").stat().st_size, 10079592)
        with self.assertRaisesRegex(ValueError, "Unknown ship design"):
            server.load_ship_design("stellaris_ship_designer", "not_declared")
        with self.assertRaisesRegex(ValueError, "duplicate id"):
            invalid = json.loads(json.dumps(design))
            invalid["locators"] = [
                {"id": "duplicate", "position": [0, 0, 0], "rotation": [0, 0, 0]},
                {"id": "duplicate", "position": [1, 0, 0], "rotation": [0, 0, 0]},
            ]
            server.validate_ship_design(invalid)

    def test_rimworld_prepare_contract_and_asset_guard(self) -> None:
        manifest = server.manifest_map()["rimworld_prepare"]
        self.assertEqual(manifest["designer"]["type"], "rimworldPrepare")
        payload = server.rimworld_prepare_workspace("rimworld_prepare")
        self.assertEqual(payload["catalog"]["source"]["workshopId"], "2844129100")
        self.assertGreaterEqual(len(payload["catalog"]["assets"]), 200)
        self.assertGreaterEqual(len(payload["catalog"]["apparel"]), 30)
        preview = payload["catalog"]["sheets"]["whole"]
        self.assertTrue(server.rimworld_prepare_asset("rimworld_prepare", preview).is_file())
        metadata_path = server.REPO_ROOT / "algorithms" / "rimworld_prepare" / payload["catalog"]["sheets"]["skinMetadata"]
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.assertEqual(metadata["version"], 3)
        self.assertEqual(metadata["canvasSize"], [2048, 736])
        self.assertEqual({item["direction"] for item in metadata["assets"]}, {"south", "east", "north"})
        self.assertEqual(len(metadata["assets"]), 3)
        self.assertEqual(
            {(layer["kind"], item["direction"]) for item in metadata["assets"] for layer in item["layers"]},
            {(kind, direction) for kind in ("body", "face") for direction in ("south", "east", "north")},
        )
        for relative in ("../builder.py", "inputs/workspace.json", "outputs/catalog.json"):
            with self.assertRaises(ValueError):
                server.rimworld_prepare_asset("rimworld_prepare", relative)

    def test_rimworld_skin_roundtrip_preserves_masks_and_has_zero_baseline_diff(self) -> None:
        from PIL import Image, ImageChops
        from algorithms.rimworld_prepare import builder

        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            (output / "previews").mkdir()
            for name in ("yuran_skin_img2img.json", "yuran_skin_img2img.png"):
                shutil.copy2(builder.OUTPUTS / "previews" / name, output / "previews" / name)
            with patch.object(builder, "OUTPUTS", output):
                result = builder.split_skin(builder.ROOT / "outputs" / "previews" / "yuran_skin_img2img.png")
            self.assertEqual(result["summary"]["assetCount"], 6)
            self.assertEqual(result["summary"]["changedPercent"], 0)
            self.assertEqual([stage["id"] for stage in result["stages"]], ["upload", "restore", "mask", "compare"])
            for item in result["files"]:
                original = Image.open(builder.ROOT / item["original"]).convert("RGBA")
                generated = Image.open(item["generated"]).convert("RGBA")
                alpha_diff = ImageChops.difference(original.getchannel("A"), generated.getchannel("A"))
                self.assertIsNone(alpha_diff.getbbox())

    def test_rimworld_skin_upload_rejects_outdated_sheet_dimensions(self) -> None:
        import base64
        import io
        from PIL import Image

        encoded = io.BytesIO()
        Image.new("RGBA", (1152, 326)).save(encoded, format="PNG")
        with self.assertRaisesRegex(ValueError, "2048x736"):
            server.import_rimworld_skin(
                "rimworld_prepare", "old-template.png", base64.b64encode(encoded.getvalue()).decode("ascii")
            )

    def test_walk_sandbox_exposes_current_latest_mesh_gate(self) -> None:
        root = server.REPO_ROOT / "algorithms" / "urdf_learn_wasd_walk"
        payload = json.loads((root / "milestones.json").read_text(encoding="utf-8"))
        self.assertEqual(len(payload["milestones"]), 12)
        self.assertEqual(payload["milestones"][0]["status"], "passed")
        self.assertEqual(payload["milestones"][1]["status"], "in_progress")
        self.assertEqual({item["status"] for item in payload["milestones"][2:]}, {"not_started"})
        self.assertEqual(payload["assetContract"]["meshTreeSha256"], "a34be1b4f2732de526c23fd1bc53e945b9e647110432fe466521fb7e73676f73")
        self.assertEqual(payload["invalidatedLineage"]["meshTreeSha256"], "b69eb237022c9f390ff5ebcf8014ecdc13e21d2b9ba9ca0ba234a46dcb2f1435")
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
        paths = [item["repoPath"] for item in manifest["files"]]
        self.assertEqual(len(paths), len(set(paths)))
        by_cloud = {}
        for item in manifest["files"]:
            signature = (item["size"], item["sha256"])
            self.assertEqual(by_cloud.setdefault(item["cloudPath"], signature), signature)
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
