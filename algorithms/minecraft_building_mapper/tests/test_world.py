from __future__ import annotations

import tempfile
import unittest
import zipfile
from pathlib import Path

from algorithms.minecraft_building_mapper.nbt import loads
from algorithms.minecraft_building_mapper.tests.helpers import legacy_chunk, modern_chunk, root, write_world
from algorithms.minecraft_building_mapper.world import AnvilWorld, ChunkView, discover_dimensions, open_world, read_level_metadata


class WorldTests(unittest.TestCase):
    def test_nbt_root_compound(self) -> None:
        self.assertEqual(loads(root([])), {})

    def test_legacy_id_and_metadata_are_preserved(self) -> None:
        chunk = ChunkView(loads(legacy_chunk(300, 7)))
        block = chunk.top_block(3, 4)
        self.assertEqual((block.key, block.legacy_id, block.metadata, block.y), ("legacy:300:7", 300, 7, 15))
        self.assertIsNone(chunk.top_block(3, 4, skip=lambda candidate: candidate.legacy_id == 300))

    def test_modern_namespaced_palette_is_preserved(self) -> None:
        chunk = ChunkView(loads(modern_chunk("lunamatrix:moon_tile")))
        self.assertEqual(chunk.top_block(0, 0).key, "lunamatrix:moon_tile")
        self.assertEqual(chunk.top_block(0, 0, max_y=-1), None)

    def test_directory_and_zip_worlds_are_read_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root_path = write_world(Path(temporary) / "world", legacy_chunk())
            with AnvilWorld(root_path) as world:
                self.assertEqual(world.chunk(0, 0).top_block(0, 0).key, "legacy:1:0")
                self.assertIsNone(world.chunk(1, 0))
            archive = Path(temporary) / "world.zip"
            with zipfile.ZipFile(archive, "w") as output:
                for path in root_path.rglob("*"):
                    if path.is_file():
                        output.write(path, f"save/{path.relative_to(root_path).as_posix()}")
            fs = open_world(archive)
            self.assertEqual(read_level_metadata(fs)["levelName"], "Fixture")
            self.assertEqual([item.id for item in discover_dimensions(fs)], ["minecraft:overworld"])

    def test_26_1_dimension_layout_is_discovered(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root_path = write_world(Path(temporary) / "world", modern_chunk())
            modern_region = root_path / "dimensions/minecraft/overworld/region"
            modern_region.mkdir(parents=True)
            (root_path / "region/r.0.0.mca").replace(modern_region / "r.0.0.mca")
            self.assertEqual(
                [(item.id, item.region_dir) for item in discover_dimensions(open_world(root_path))],
                [("minecraft:overworld", "dimensions/minecraft/overworld/region")],
            )


if __name__ == "__main__":
    unittest.main()
