from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from algorithms.minecraft_building_mapper.catalog import Catalog, build_legacy_catalog, build_modern_catalog, write_catalog
from algorithms.minecraft_building_mapper.render import TileRequest, render_tile_to_output
from algorithms.minecraft_building_mapper.server import TileApplication, WorldSpec
from algorithms.minecraft_building_mapper.tests.helpers import legacy_chunk, modern_chunk, write_world


def png(path: Path, color: tuple[int, int, int], size: int = 16) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (size, size), color).save(path)


class CatalogAndRenderTests(unittest.TestCase):
    def test_modern_blockstate_model_texture_resolution(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            assets = Path(temporary) / "assets"
            state = assets / "assets/test/blockstates/cube.json"
            state.parent.mkdir(parents=True)
            state.write_text(json.dumps({"variants": {"": {"model": "test:block/cube"}}}))
            model = assets / "assets/test/models/block/cube.json"
            model.parent.mkdir(parents=True)
            model.write_text(json.dumps({"textures": {"top": "test:block/cube"}, "elements": [{"faces": {"up": {"texture": "#top"}}}]}))
            png(assets / "assets/test/textures/block/cube.png", (10, 120, 30))
            catalog = build_modern_catalog([assets])
            self.assertEqual(catalog["entries"]["test:cube"]["status"], "resolved")
            self.assertEqual(catalog["entries"]["test:cube"]["texture"]["member"], "assets/test/textures/block/cube.png")

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
