from __future__ import annotations

import gzip
import tempfile
import unittest
import zipfile
from pathlib import Path

from algorithms.minecraft_building_mapper.nbt import loads
from algorithms.minecraft_building_mapper.tests.helpers import compound, flattened_palette_chunk, integer, legacy_chunk, long, modern_chunk, root, string, write_world
from algorithms.minecraft_building_mapper.world import (
    AnvilWorld,
    ChunkView,
    _modern_surface_heights,
    discover_dimensions,
    open_world,
    read_level_metadata,
)


class WorldTests(unittest.TestCase):
    def test_nbt_root_compound(self) -> None:
        self.assertEqual(loads(root([])), {})

    def test_legacy_id_and_metadata_are_preserved(self) -> None:
        chunk = ChunkView(loads(legacy_chunk(300, 7, 200)))
        block = chunk.top_block(3, 4)
        self.assertEqual((block.key, block.legacy_id, block.metadata, block.y), ("legacy:300:7", 300, 7, 15))
        self.assertEqual(chunk.biome_at(3, 4), 200)
        self.assertIsNone(chunk.top_block(3, 4, skip=lambda candidate: candidate.legacy_id == 300))

    def test_modern_namespaced_palette_is_preserved(self) -> None:
        chunk = ChunkView(loads(modern_chunk("lunamatrix:moon_tile")))
        self.assertEqual(chunk.top_block(0, 0).key, "lunamatrix:moon_tile")
        self.assertEqual(chunk.top_block(0, 0, max_y=-1), None)

    def test_flattened_1_15_palette_is_not_treated_as_empty(self) -> None:
        chunk = ChunkView(loads(flattened_palette_chunk("minecraft:grass_block", biome_id=132)))
        self.assertEqual(chunk.top_block(8, 8).key, "minecraft:grass_block")
        self.assertEqual(chunk.biome_at(8, 8, 64), "minecraft:flower_forest")

    def test_modern_world_surface_heightmap_decodes_negative_min_y(self) -> None:
        bits = 9
        per_long = 64 // bits
        packed = [0] * ((256 + per_long - 1) // per_long)
        for index in range(256):
            packed[index // per_long] |= 132 << ((index % per_long) * bits)
        heights = _modern_surface_heights({"Heightmaps": {"WORLD_SURFACE": packed}}, list(range(-4, 20)))
        self.assertEqual(set(heights), {67})

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

    def test_nested_custom_dimension_path_is_discovered(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root_path = write_world(Path(temporary) / "world", modern_chunk())
            nested_region = root_path / "dimensions/minecraft/custom/resource/region"
            nested_region.mkdir(parents=True)
            (root_path / "region/r.0.0.mca").replace(nested_region / "r.0.0.mca")
            self.assertEqual(
                [(item.id, item.region_dir) for item in discover_dimensions(open_world(root_path))],
                [("minecraft:custom/resource", "dimensions/minecraft/custom/resource/region")],
            )

    def test_26_1_compound_spawn_is_reported_without_rewriting_level(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            world = Path(temporary) / "world"
            world.mkdir()
            data = compound([
                (8, "LevelName", string("Current")),
                (3, "DataVersion", integer(4790)),
                (4, "LastPlayed", long(1780032536144)),
                (10, "spawn", compound([
                    (9, "pos", bytes([3]) + integer(3) + integer(744) + integer(68) + integer(-622)),
                    (8, "dimension", string("minecraft:overworld")),
                ])),
            ])
            before = gzip.compress(root([(10, "Data", data)]))
            (world / "level.dat").write_bytes(before)
            metadata = read_level_metadata(open_world(world))
            self.assertEqual(metadata["spawn"], {
                "x": 744, "y": 68, "z": -622, "dimension": "minecraft:overworld",
            })
            self.assertEqual((world / "level.dat").read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
