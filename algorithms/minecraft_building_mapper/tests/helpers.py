from __future__ import annotations

import gzip
import struct
import zlib
from pathlib import Path


def name(value: str) -> bytes:
    encoded = value.encode()
    return struct.pack(">H", len(encoded)) + encoded


def byte(value: int) -> bytes:
    return struct.pack(">b", value)


def integer(value: int) -> bytes:
    return struct.pack(">i", value)


def long(value: int) -> bytes:
    return struct.pack(">q", value)


def string(value: str) -> bytes:
    return name(value)


def byte_array(value: bytes) -> bytes:
    return integer(len(value)) + value


def long_array(values: list[int]) -> bytes:
    return integer(len(values)) + b"".join(long(value) for value in values)


def compound(entries: list[tuple[int, str, bytes]]) -> bytes:
    return b"".join(bytes([tag]) + name(key) + payload for tag, key, payload in entries) + b"\x00"


def list_of(tag: int, payloads: list[bytes]) -> bytes:
    return bytes([tag]) + integer(len(payloads)) + b"".join(payloads)


def root(entries: list[tuple[int, str, bytes]]) -> bytes:
    return b"\x0a\x00\x00" + compound(entries)


def legacy_chunk(block_id: int = 1, metadata: int = 0, biome_id: int = 1) -> bytes:
    blocks = bytes([block_id & 0xFF]) * 4096
    nibble = metadata | (metadata << 4)
    entries = [(1, "Y", byte(0)), (7, "Blocks", byte_array(blocks)), (7, "Data", byte_array(bytes([nibble]) * 2048))]
    high = block_id >> 8
    if high:
        entries.append((7, "Add", byte_array(bytes([high | (high << 4)]) * 2048)))
    section = compound(entries)
    level = compound([(9, "Sections", list_of(10, [section])), (7, "Biomes", byte_array(bytes([biome_id]) * 256))])
    return root([(10, "Level", level)])


def legacy_layered_chunk(layers: list[tuple[int, int, int]], biome_id: int = 1) -> bytes:
    """Build one section with the same requested stack in all 256 columns."""

    blocks = bytearray(4096)
    data = bytearray(2048)
    add = bytearray(2048)
    has_add = False
    for y, block_id, metadata in layers:
        for z in range(16):
            for x in range(16):
                index = (y << 8) | (z << 4) | x
                blocks[index] = block_id & 0xFF
                shift = 4 * (index & 1)
                data[index // 2] |= (metadata & 15) << shift
                high = block_id >> 8
                if high:
                    add[index // 2] |= (high & 15) << shift
                    has_add = True
    entries = [(1, "Y", byte(0)), (7, "Blocks", byte_array(bytes(blocks))), (7, "Data", byte_array(bytes(data)))]
    if has_add:
        entries.append((7, "Add", byte_array(bytes(add))))
    section = compound(entries)
    level = compound([(9, "Sections", list_of(10, [section])), (7, "Biomes", byte_array(bytes([biome_id]) * 256))])
    return root([(10, "Level", level)])


def modern_chunk(
    block_name: str = "test:cube", properties: dict[str, str] | None = None,
    biome_name: str | None = None,
) -> bytes:
    palette_fields = [(8, "Name", string(block_name))]
    if properties:
        palette_fields.append((10, "Properties", compound([(8, key, string(value)) for key, value in properties.items()])))
    palette_entry = compound(palette_fields)
    states = compound([(9, "palette", list_of(10, [palette_entry]))])
    section_fields = [(1, "Y", byte(0)), (10, "block_states", states)]
    if biome_name:
        section_fields.append((10, "biomes", compound([(9, "palette", list_of(8, [string(biome_name)]))])))
    section = compound(section_fields)
    return root([(9, "sections", list_of(10, [section]))])


def write_world(root_path: Path, chunk: bytes, level_name: str = "Fixture") -> Path:
    root_path.mkdir(parents=True)
    level = root([(10, "Data", compound([(8, "LevelName", string(level_name)), (3, "DataVersion", integer(1343))]))])
    (root_path / "level.dat").write_bytes(gzip.compress(level))
    region_dir = root_path / "region"
    region_dir.mkdir()
    payload = zlib.compress(chunk)
    record = struct.pack(">I", len(payload) + 1) + b"\x02" + payload
    sectors = (len(record) + 4095) // 4096
    header = bytearray(8192)
    header[:4] = (2).to_bytes(3, "big") + bytes([sectors])
    region = bytes(header) + record + b"\x00" * (sectors * 4096 - len(record))
    (region_dir / "r.0.0.mca").write_bytes(region)
    return root_path
