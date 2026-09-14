from __future__ import annotations

import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from algorithms.minecraft_building_mapper.catalog import (
    Catalog,
    build_legacy_catalog,
    build_modern_catalog,
    cover_visible_legacy_catalog,
    write_catalog,
)
from algorithms.minecraft_building_mapper.evidence import resolution_entry
from algorithms.minecraft_building_mapper.legacy_visuals import vanilla_visual
from algorithms.minecraft_building_mapper.overview import (
    _dominant_connected_chunks,
    _valid_chunk_position,
    inventory_world,
    inventory_world_to_output,
    merge_inventory_reports,
    render_overview_to_output,
)
from algorithms.minecraft_building_mapper.render import TextureRenderer, TileRequest, _shade, render_tile_to_output
from algorithms.minecraft_building_mapper.server import TileApplication, WorldSpec
from algorithms.minecraft_building_mapper.tests.helpers import legacy_chunk, legacy_layered_chunk, modern_chunk, write_world
from algorithms.minecraft_building_mapper.world import AnvilWorld, Block


def png(path: Path, color: tuple[int, int, int], size: int = 16) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (size, size), color).save(path)


class CatalogAndRenderTests(unittest.TestCase):
    def test_exact_legacy_biome_colorizer_tints_grayscale_grass(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bc3 = root / "bc3"
            bc3.mkdir()
            (bc3 / "optionsof.txt").write_text(
                "ofCustomColors:true\nofSmoothBiomes:true\nofSwampColors:true\n"
            )
            pack = root / "pack"
            png(pack / "textures/blocks/grass_top.png", (255, 255, 255))
            png(pack / "textures/blocks/dirt.png", (90, 50, 20))
            png(pack / "misc/grasscolor.png", (100, 200, 50), 256)
            catalog_data = build_legacy_catalog(bc3, [pack])
            colorizer = catalog_data["colorizers"]["grass"]
            self.assertEqual(colorizer["member"], "misc/grasscolor.png")
            self.assertEqual(len(colorizer["memberSha256"]), 64)
            self.assertTrue(catalog_data["legacyRendering"]["options"]["smoothBiomes"])
            catalog_path = root / "catalog.json"
            write_catalog(catalog_data, catalog_path)
            world = write_world(root / "world", legacy_layered_chunk([(14, 2, 0), (13, 3, 0)], biome_id=1))
            output = root / "output"
            result = render_tile_to_output(
                world_path=world, dimension="minecraft:overworld", catalog_path=catalog_path,
                request=TileRequest(0, 0), output=output,
            )
            self.assertEqual(Image.open(output / "tile_0_0.png").getpixel((8, 8)), (100, 200, 50))
            self.assertEqual(result["biomeTint"]["applied"]["grass"], 256)
            self.assertEqual(result["unknown"]["visibleBlocks"], 0)

    def test_exact_forgotten_nature_biome_id_climate_has_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bc3 = root / "bc3"
            config = bc3 / "config/ForgottenNature.cfg"
            config.parent.mkdir(parents=True)
            config.write_text('general {\n I:"Biomes: Neo Tropical Forest ID"=70\n}\n')
            archive = root / "ForgottenNature for 1.5.2.zip"
            member = "ForgottenNature/Biomes/BiomeGenTropicalForest.class"
            with zipfile.ZipFile(archive, "w") as output:
                output.writestr(member, b"fixture climate F=.9 G=.9")
            from algorithms.minecraft_building_mapper.catalog import sha256_file

            with patch("algorithms.minecraft_building_mapper.catalog.FORGOTTEN_NATURE_152_SHA256", sha256_file(archive)):
                catalog = build_legacy_catalog(bc3, [archive])
            biome = catalog["legacyBiomes"]["70"]
            self.assertEqual((biome["temperature"], biome["rainfall"]), (0.9, 0.9))
            self.assertEqual(biome["resolution"], "exact")
            self.assertEqual({item["kind"] for item in biome["provenance"]}, {"exact-config", "exact-mod-bytecode"})
            self.assertEqual(len(biome["provenance"][1]["memberSha256"]), 64)

    def test_alpha_block_composites_exact_texture_over_substrate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bc3 = root / "bc3"
            bc3.mkdir()
            pack = root / "pack"
            png(pack / "textures/blocks/dirt.png", (200, 0, 0))
            glass = pack / "textures/blocks/glass.png"
            glass.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGBA", (16, 16), (0, 0, 200, 128)).save(glass)
            catalog_path = root / "catalog.json"
            write_catalog(build_legacy_catalog(bc3, [pack]), catalog_path)
            world = write_world(root / "world", legacy_layered_chunk([(15, 20, 0), (14, 3, 0)]))
            output = root / "output"
            result = render_tile_to_output(
                world_path=world, dimension="minecraft:overworld", catalog_path=catalog_path,
                request=TileRequest(0, 0), output=output,
            )
            red, green, blue = Image.open(output / "tile_0_0.png").getpixel((8, 8))
            self.assertTrue(95 <= red <= 105 and green == 0 and 95 <= blue <= 105)
            self.assertEqual(result["geometry"]["counts"], {"alpha_cube": 256, "cube": 256})

    def test_legacy_geometry_categories_cover_directional_and_partial_blocks(self) -> None:
        expected = {
            9: "fluid", 18: "alpha_cube", 50: "point", 53: "stair", 55: "line",
            64: "plane", 78: "cover", 85: "connected", 126: "slab", 141: "cross",
        }
        self.assertEqual({block_id: vanilla_visual(block_id, 0)["geometry"] for block_id in expected}, expected)

    def test_height_shading_is_bounded_and_creates_edge_contrast(self) -> None:
        image = Image.new("RGBA", (3, 3), (100, 100, 100, 255))
        report = _shade(image, [[10.0, 10.0, 4.0], [10.0, 10.0, 4.0], [4.0, 4.0, 4.0]], 1)
        values = {pixel[0] for pixel in image.getdata()}
        self.assertGreater(len(values), 1)
        self.assertGreaterEqual(report["minimumFactor"], 0.68)
        self.assertLessEqual(report["maximumFactor"], 1.22)

    def test_animated_legacy_texture_uses_one_square_frame(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bc3 = root / "bc3"
            bc3.mkdir()
            pack = root / "pack"
            water = pack / "textures/blocks/water.png"
            water.parent.mkdir(parents=True)
            strip = Image.new("RGBA", (2, 4), (0, 0, 255, 255))
            for y in range(2):
                for x in range(2):
                    strip.putpixel((x, y), (255, 0, 0, 255))
            strip.save(water)
            catalog_path = root / "catalog.json"
            write_catalog(build_legacy_catalog(bc3, [pack]), catalog_path)
            texture = TextureRenderer(Catalog(catalog_path), 2)._texture(Block("legacy:9:0", 10, 9, 0), None)
            self.assertEqual(set(texture.getdata()), {(255, 0, 0, 255)})

    def test_whole_world_inventory_and_overview_use_allocated_chunk_bounds(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bc3 = root / "bc3"
            bc3.mkdir()
            pack = root / "pack"
            png(pack / "textures/blocks/stone.png", (100, 110, 120))
            catalog_path = root / "catalog.json"
            write_catalog(build_legacy_catalog(bc3, [pack]), catalog_path)
            world = write_world(root / "world", legacy_chunk(1, 0, biome_id=1))
            inventory = inventory_world_to_output(
                world_path=world, dimension="minecraft:overworld", catalog_path=catalog_path,
                output=root / "inventory",
            )
            self.assertEqual(inventory["chunks"]["allocated"], 1)
            self.assertEqual(inventory["bounds"]["maxXExclusive"], 16)
            self.assertEqual(inventory["visuallyUnhandled"]["surfaceContributions"], 0)
            overview = render_overview_to_output(
                world_path=world, dimension="minecraft:overworld", catalog_path=catalog_path,
                output=root / "overview", max_size=256,
            )
            self.assertEqual(Image.open(overview["overview"]).size, (16, 16))
            self.assertEqual(overview["chunks"]["rendered"], 1)

    def test_whole_world_skips_truncated_regions_and_unplayable_coordinate_outliers(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bc3 = root / "bc3"
            bc3.mkdir()
            pack = root / "pack"
            png(pack / "textures/blocks/stone.png", (100, 110, 120))
            catalog_path = root / "catalog.json"
            write_catalog(build_legacy_catalog(bc3, [pack]), catalog_path)
            world = write_world(root / "world", legacy_chunk(1, 0, biome_id=1))
            (world / "region/r.1.0.mca").write_bytes(b"")
            inventory = inventory_world_to_output(
                world_path=world, dimension="minecraft:overworld", catalog_path=catalog_path,
                output=root / "inventory",
            )
            self.assertEqual(len(inventory["chunks"]["regionErrors"]), 1)
            self.assertFalse(_valid_chunk_position((134_217_727, 0)))
            self.assertTrue(_valid_chunk_position((-6405, 0)))

    def test_overview_quarantines_disconnected_coordinate_jump_chunks(self) -> None:
        dominant, quarantined = _dominant_connected_chunks([(0, 0), (1, 0), (1, 1), (1000, 1000)])
        self.assertEqual(set(dominant), {(0, 0), (1, 0), (1, 1)})
        self.assertEqual(quarantined[0]["chunks"], 1)
        self.assertEqual(quarantined[0]["bounds"]["minChunkX"], 1000)

    def test_inventory_shards_merge_to_complete_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bc3 = root / "bc3"
            bc3.mkdir()
            pack = root / "pack"
            png(pack / "textures/blocks/stone.png", (100, 110, 120))
            catalog_path = root / "catalog.json"
            write_catalog(build_legacy_catalog(bc3, [pack]), catalog_path)
            world_path = write_world(root / "world", legacy_chunk(1, 0, biome_id=1))
            with AnvilWorld(world_path) as world:
                first = inventory_world(world, Catalog(catalog_path), shard_count=2, shard_index=0)
            with AnvilWorld(world_path) as world:
                second = inventory_world(world, Catalog(catalog_path), shard_count=2, shard_index=1)
            merged = merge_inventory_reports([second, first])
            self.assertEqual(merged["chunks"]["allocated"], 1)
            self.assertEqual(merged["columns"], 256)
            self.assertEqual(merged["coverage"]["resolved"]["surfaceContributions"], 256)
            self.assertEqual(merged["shardsMerged"]["indices"], [0, 1])

    def test_inference_requires_explicit_confidence_reason_and_provenance(self) -> None:
        entry = resolution_entry(
            "inferred",
            name="Private decorative block",
            confidence=0.65,
            reason="Nearest private atlas label; exact code is unavailable.",
            provenance=[{"kind": "texture-name", "member": "private/decorative.png"}],
            texture={"source": 0, "member": "private/decorative.png"},
        )
        self.assertEqual(entry["resolution"], "inferred")
        self.assertEqual(entry["confidence"], 0.65)
        with self.assertRaisesRegex(ValueError, "confidence"):
            resolution_entry(
                "inferred",
                name="Unlabeled guess",
                reason="A guess.",
                provenance=[{"kind": "texture-name", "member": "guess.png"}],
            )
        with self.assertRaisesRegex(ValueError, "provenance"):
            resolution_entry("unknown", name="Missing evidence", reason="Not resolved.", provenance=[])

    def test_visible_coverage_inference_is_inventory_scoped_and_auditable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bc3 = root / "bc3"
            config = bc3 / "config/railcraft/railcraft.cfg"
            config.parent.mkdir(parents=True)
            config.write_text("block {\n I:block.track=454\n}\n")
            assets = root / "assets"
            png(assets / "textures/blocks/rail.png", (80, 70, 60))
            png(assets / "textures/blocks/stonebricksmooth.png", (100, 100, 100))
            member = assets / "mods/railcraft/textures/blocks/tracks/track.reinforced.png"
            png(member, (120, 100, 70))
            base = build_legacy_catalog(bc3, [assets])
            inventory = {
                "schema": "geo.minecraft-surface-inventory/v1",
                "surfaceStates": {
                    "legacy:454:0": {"surfaceContributions": 12, "topColumns": 10},
                },
            }
            covered = cover_visible_legacy_catalog(base, inventory)
            entry = covered["entries"]["legacy:454:0"]
            self.assertEqual(entry["resolution"], "inferred")
            self.assertLess(entry["confidence"], 1.0)
            self.assertIn("not presented as an exact", entry["reason"])
            self.assertEqual(entry["texture"]["member"], "mods/railcraft/textures/blocks/tracks/track.reinforced.png")
            self.assertEqual(
                {item["kind"] for item in entry["provenance"]},
                {"forge-config", "surface-inventory", "exact-representative-texture"},
            )
            self.assertNotIn("legacy:454:1", covered["entries"])
            family = covered["entries"]["legacy:454:*"]
            self.assertEqual(family["resolution"], "inferred")
            self.assertIn("not presented as exact", family["reason"])
            self.assertIn("metadata-family-fallback", {item["kind"] for item in family["provenance"]})

    def test_modern_visible_fallback_uses_exact_sprite_and_inference_label(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            assets = Path(temporary) / "assets"
            state = assets / "assets/minecraft/blockstates/redstone_wire.json"
            state.parent.mkdir(parents=True)
            state.write_text(json.dumps({"multipart": []}))
            png(assets / "assets/minecraft/textures/block/redstone_dust_dot.png", (160, 20, 20))
            base = build_modern_catalog([assets])
            key = "minecraft:redstone_wire[east=none,north=none,power=0,south=none,west=none]"
            covered = cover_visible_legacy_catalog(base, {
                "schema": "geo.minecraft-surface-inventory/v1",
                "surfaceStates": {key: {"surfaceContributions": 9, "topColumns": 9}},
            })
            entry = covered["entries"][key]
            self.assertEqual(entry["resolution"], "inferred")
            self.assertEqual(entry["texture"]["member"], "assets/minecraft/textures/block/redstone_dust_dot.png")
            self.assertIn("surface-inventory", {item["kind"] for item in entry["provenance"]})

    def test_modern_blockstate_model_texture_resolution(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            assets = Path(temporary) / "assets"
            state = assets / "assets/test/blockstates/cube.json"
            state.parent.mkdir(parents=True)
            state.write_text(json.dumps({"variants": {"": {"model": "test:block/cube"}}}))
            model = assets / "assets/test/models/block/cube.json"
            model.parent.mkdir(parents=True)
            model.write_text(json.dumps({
                "textures": {"top": {"sprite": "test:block/cube", "force_translucent": True}},
                "elements": [{"faces": {"up": {"texture": "#top"}}}],
            }))
            png(assets / "assets/test/textures/block/cube.png", (10, 120, 30))
            catalog = build_modern_catalog([assets])
            self.assertEqual(catalog["entries"]["test:cube"]["status"], "resolved")
            self.assertEqual(catalog["entries"]["test:cube"]["texture"]["member"], "assets/test/textures/block/cube.png")

    def test_modern_biome_palette_tints_exact_grass_texture(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            assets = root / "assets"
            state = assets / "assets/minecraft/blockstates/grass_block.json"
            state.parent.mkdir(parents=True)
            state.write_text(json.dumps({"variants": {"snowy=false": {"model": "minecraft:block/grass"}}}))
            model = assets / "assets/minecraft/models/block/grass.json"
            model.parent.mkdir(parents=True)
            model.write_text(json.dumps({"textures": {"top": "minecraft:block/grass_block_top"}}))
            png(assets / "assets/minecraft/textures/block/grass_block_top.png", (200, 200, 200))
            png(assets / "assets/minecraft/textures/colormap/grass.png", (80, 160, 40), 256)
            biome = assets / "data/minecraft/worldgen/biome/plains.json"
            biome.parent.mkdir(parents=True)
            biome.write_text(json.dumps({
                "temperature": 0.8, "downfall": 0.4,
                "effects": {"water_color": "#3f76e4"},
            }))
            catalog_path = root / "catalog.json"
            catalog_data = build_modern_catalog([assets])
            self.assertEqual(catalog_data["modernBiomes"]["minecraft:plains"]["waterMultiplier"], 0x3F76E4)
            write_catalog(catalog_data, catalog_path)
            world = write_world(
                root / "world",
                modern_chunk("minecraft:grass_block", {"snowy": "false"}, "minecraft:plains"),
            )
            output = root / "output"
            result = render_tile_to_output(
                world_path=world, dimension="minecraft:overworld", catalog_path=catalog_path,
                request=TileRequest(0, 0), output=output,
            )
            self.assertEqual(Image.open(output / "tile_0_0.png").getpixel((8, 8)), (62, 125, 31))
            self.assertEqual(result["biomeTint"]["applied"]["grass"], 256)

    def test_legacy_catalog_keeps_unproven_mod_texture_unknown(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bc3 = root / "bc3"
            config = bc3 / "config/example.cfg"
            config.parent.mkdir(parents=True)
            config.write_text("block {\n I:machineBlock=200\n I:glassBlocks=3924\n}\n")
            custom = bc3 / "config/CustomStuff/mods/BilicraftMOD"
            definition = custom / "blocks/glass.js"
            definition.parent.mkdir(parents=True)
            definition.write_text('name="ExactGlass";\nvar id=config.getBlockId("glassBlocks");\ndisplayName[3]="Blue glass";\ntextureFileYP[3]="Blue.png";\n')
            png(custom / "textures/blocks/Blue.png", (0, 20, 200))
            pack = root / "BiliCraft-1.5.1-32x"
            png(pack / "terrain.png", (90, 70, 50), 512)
            catalog = build_legacy_catalog(bc3, [pack, custom])
            self.assertEqual(catalog["entries"]["legacy:1:0"]["status"], "resolved")
            self.assertEqual(catalog["entries"]["legacy:200:*"]["status"], "unknown")
            self.assertIn("Forge config proves", catalog["entries"]["legacy:200:*"]["reason"])
            self.assertEqual(catalog["entries"]["legacy:3924:3"]["status"], "resolved")
            self.assertEqual(catalog["entries"]["legacy:3924:3"]["texture"]["member"], "textures/blocks/Blue.png")

    def test_fingerprint_gated_forgotten_nature_resolver_records_exact_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bc3 = root / "bc3"
            config = bc3 / "config/ForgottenNature.cfg"
            config.parent.mkdir(parents=True)
            config.write_text("block {\n I:leafIDindex=4084\n I:logIDindex=4079\n I:FlowerID=4092\n}\n")
            (bc3 / "options.txt").write_text("fancyGraphics:true\n")
            archive = root / "ForgottenNature for 1.5.2.zip"
            class_names = ["BlockNewFlowers"]
            class_names += [f"BlockNewLeaves{suffix}" for suffix in ("", "2", "3", "4", "5", "6")]
            class_names += [f"BlockNewLogs{suffix}" for suffix in ("", "2", "3", "4")]
            with zipfile.ZipFile(archive, "w") as output:
                output.writestr("ForgottenNature/ForgottenNature.class", b"exact registration fixture")
                output.writestr("ForgottenNature/Proxy/FNClientProxy.class", b"exact language fixture")
                for class_name in class_names:
                    output.writestr(f"ForgottenNature/Blocks/{class_name}.class", class_name.encode())
            pack = root / "pack"
            for stem in ("FigLeaves", "AcaciaLeaves", "PoplarLeaves", "Hydrangea", "LogCrossSection"):
                png(pack / f"mods/ForgottenNature/textures/blocks/{stem}.png", (30, 120, 50))
            from algorithms.minecraft_building_mapper.catalog import sha256_file

            with patch("algorithms.minecraft_building_mapper.legacy_resolver.FORGOTTEN_NATURE_152_SHA256", sha256_file(archive)):
                catalog = build_legacy_catalog(bc3, [pack, archive])
            for key in ("legacy:4084:4", "legacy:4084:12", "legacy:4085:6", "legacy:4086:7", "legacy:4092:6", "legacy:4079:8"):
                self.assertEqual(catalog["entries"][key]["resolution"], "exact")
                self.assertEqual(catalog["entries"][key]["confidence"], 1.0)
                self.assertIn("bundled-render-bytecode", {item["kind"] for item in catalog["entries"][key]["provenance"]})

    def test_tile_render_writes_image_metadata_and_unknown_report(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            world = write_world(root / "world", modern_chunk("test:cube"))
            assets = root / "assets"
            state = assets / "assets/test/blockstates/cube.json"
            state.parent.mkdir(parents=True)
            state.write_text(json.dumps({"variants": {"": {"model": "test:block/cube"}}}))
            model = assets / "assets/test/models/block/cube.json"
            model.parent.mkdir(parents=True)
            model.write_text(json.dumps({"textures": {"all": "test:block/cube"}, "elements": [{"faces": {"up": {"texture": "#all"}}}]}))
            png(assets / "assets/test/textures/block/cube.png", (20, 180, 80))
            catalog_path = root / "catalog.json"
            write_catalog(build_modern_catalog([assets]), catalog_path)
            output = root / "outputs/sample"
            result = render_tile_to_output(world_path=world, dimension="minecraft:overworld", catalog_path=catalog_path, request=TileRequest(0, 0), output=output)
            self.assertEqual(result["chunks"], {"loaded": 1, "missing": 255, "errors": []})
            self.assertEqual(result["visibleBlocks"], 256)
            self.assertEqual(result["unknown"]["visibleBlocks"], 0)
            self.assertEqual(result["resolution"]["exact"]["visibleBlocks"], 256)
            self.assertEqual(result["resolution"]["inferred"]["visibleBlocks"], 0)
            self.assertEqual(Image.open(output / "tile_0_0.png").size, (256, 256))
            metadata = json.loads((output / "metadata.json").read_text())
            self.assertTrue(metadata["readOnly"])
            self.assertEqual(metadata["extensions"], {})
            Catalog(catalog_path)

    def test_tile_application_exposes_world_and_dimension_selectors(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            world = write_world(root / "world", modern_chunk("test:cube"), "Selector fixture")
            assets = root / "assets"
            state = assets / "assets/test/blockstates/cube.json"
            state.parent.mkdir(parents=True)
            state.write_text(json.dumps({"variants": {"": {"model": "test:block/cube"}}}))
            model = assets / "assets/test/models/block/cube.json"
            model.parent.mkdir(parents=True)
            model.write_text(json.dumps({"textures": {"all": "test:block/cube"}, "elements": [{"faces": {"up": {"texture": "#all"}}}]}))
            png(assets / "assets/test/textures/block/cube.png", (30, 80, 170))
            catalog_path = root / "catalog.json"
            write_catalog(build_modern_catalog([assets]), catalog_path)
            app = TileApplication([WorldSpec("fixture", world, catalog_path)], root / "cache")
            self.assertEqual(app.inventory()[0]["dimensions"], ["minecraft:overworld"])
            image_path, report_path = app.tile("fixture", "minecraft:overworld", TileRequest(0, 0))
            self.assertTrue(image_path.is_file())
            self.assertEqual(json.loads(report_path.read_text())["chunks"]["loaded"], 1)


if __name__ == "__main__":
    unittest.main()
