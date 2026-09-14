"""Read-only Anvil region and chunk decoding for legacy and modern worlds."""

from __future__ import annotations

import gzip
import io
import math
import re
import struct
import zipfile
import zlib
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import BinaryIO, Callable, Iterator, Protocol

from .nbt import NBTError, loads


AIR_NAMES = {"minecraft:air", "minecraft:cave_air", "minecraft:void_air"}
REGION_RE = re.compile(r"r\.(-?\d+)\.(-?\d+)\.mca$")


class WorldReadError(RuntimeError):
    pass


class WorldFS(Protocol):
    label: str

    def exists(self, relative: str) -> bool: ...
    def open(self, relative: str) -> BinaryIO: ...
    def names(self) -> Iterator[str]: ...


class DirectoryWorldFS:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.label = str(self.root)

    def _path(self, relative: str) -> Path:
        path = (self.root / relative).resolve()
        if self.root != path and self.root not in path.parents:
            raise WorldReadError(f"Path escapes world root: {relative}")
        return path

    def exists(self, relative: str) -> bool:
        return self._path(relative).is_file()

    def open(self, relative: str) -> BinaryIO:
        return self._path(relative).open("rb")

    def names(self) -> Iterator[str]:
        level = self.root / "level.dat"
        if level.is_file():
            yield "level.dat"
        patterns = (
            "region/r.*.*.mca",
            "DIM-1/region/r.*.*.mca",
            "DIM1/region/r.*.*.mca",
            "dimensions/**/region/r.*.*.mca",
        )
        for pattern in patterns:
            for path in self.root.glob(pattern):
                if path.is_file():
                    yield path.relative_to(self.root).as_posix()


class ZipWorldFS:
    """Random-access view of one world ZIP; archive members are never extracted."""

    def __init__(self, archive: Path):
        self.archive = archive.resolve()
        self.zip = zipfile.ZipFile(self.archive)
        members = [item.filename for item in self.zip.infolist() if not item.is_dir()]
        roots = {name[: -len("level.dat")].rstrip("/") for name in members if name.endswith("level.dat")}
        if len(roots) != 1:
            raise WorldReadError(
                f"Expected exactly one level.dat in {archive}; found {len(roots)}. "
                "Point --world at the inner world ZIP, not a backup bundle."
            )
        self.prefix = (next(iter(roots)) + "/") if next(iter(roots)) else ""
        self._members = set(members)
        self.label = str(self.archive)

    def _member(self, relative: str) -> str:
        normalized = PurePosixPath(relative).as_posix().lstrip("/")
        if ".." in PurePosixPath(normalized).parts:
            raise WorldReadError(f"Path escapes ZIP world root: {relative}")
        return self.prefix + normalized

    def exists(self, relative: str) -> bool:
        return self._member(relative) in self._members

    def open(self, relative: str) -> BinaryIO:
        # BytesIO makes region seeks deterministic while leaving the archive unchanged.
        return io.BytesIO(self.zip.read(self._member(relative)))

    def names(self) -> Iterator[str]:
        for name in sorted(self._members):
            if name.startswith(self.prefix):
                yield name[len(self.prefix) :]


def open_world(path: str | Path) -> WorldFS:
    source = Path(path).expanduser()
    if source.is_dir():
        return DirectoryWorldFS(source)
    if source.is_file() and source.suffix.lower() == ".zip":
        return ZipWorldFS(source)
    raise WorldReadError(f"World source must be a directory or a world ZIP: {source}")


@dataclass(frozen=True)
class Dimension:
    id: str
    region_dir: str


@dataclass(frozen=True)
class Block:
    key: str
    y: int
    legacy_id: int | None = None
    metadata: int | None = None


def discover_dimensions(fs: WorldFS) -> list[Dimension]:
    names = set(fs.names())
    dimensions: dict[str, Dimension] = {}
    if any(name.startswith("region/") and REGION_RE.search(name) for name in names):
        dimensions["minecraft:overworld"] = Dimension("minecraft:overworld", "region")
    if any(name.startswith("DIM-1/region/") and REGION_RE.search(name) for name in names):
        dimensions["minecraft:the_nether"] = Dimension("minecraft:the_nether", "DIM-1/region")
    if any(name.startswith("DIM1/region/") and REGION_RE.search(name) for name in names):
        dimensions["minecraft:the_end"] = Dimension("minecraft:the_end", "DIM1/region")
    for name in names:
        parts = PurePosixPath(name).parts
        if (
            len(parts) >= 5
            and parts[0] == "dimensions"
            and parts[-2] == "region"
            and REGION_RE.match(parts[-1])
        ):
            dimension_path = "/".join(parts[2:-2])
            if dimension_path:
                dim_id = f"{parts[1]}:{dimension_path}"
                dimensions[dim_id] = Dimension(dim_id, "/".join(parts[:-1]))
    return sorted(dimensions.values(), key=lambda item: item.id)


def read_level_metadata(fs: WorldFS) -> dict[str, object]:
    if not fs.exists("level.dat"):
        raise WorldReadError(f"No level.dat in world {fs.label}")
    with fs.open("level.dat") as stream:
        root = loads(stream.read(), compressed=True)
    data = root.get("Data", root)
    spawn = data.get("spawn")
    spawn_pos = spawn.get("pos") if isinstance(spawn, dict) else None
    result = {
        "levelName": data.get("LevelName"),
        "dataVersion": data.get("DataVersion"),
        "version": data.get("Version"),
        "lastPlayed": data.get("LastPlayed"),
        "spawn": {
            "x": spawn_pos[0] if isinstance(spawn_pos, list) and len(spawn_pos) == 3 else data.get("SpawnX"),
            "y": spawn_pos[1] if isinstance(spawn_pos, list) and len(spawn_pos) == 3 else data.get("SpawnY"),
            "z": spawn_pos[2] if isinstance(spawn_pos, list) and len(spawn_pos) == 3 else data.get("SpawnZ"),
        },
    }
    if isinstance(spawn, dict) and isinstance(spawn.get("dimension"), str):
        result["spawn"]["dimension"] = spawn["dimension"]
    player = data.get("Player")
    if isinstance(player, dict) and isinstance(player.get("Pos"), list) and len(player["Pos"]) == 3:
        result["player"] = {"x": player["Pos"][0], "y": player["Pos"][1], "z": player["Pos"][2]}
    players = []
    for name in sorted(fs.names()):
        if not (name.startswith(("playerdata/", "players/data/")) and name.endswith(".dat")):
            continue
        try:
            with fs.open(name) as stream:
                saved = loads(stream.read(), compressed=True)
        except (OSError, NBTError):
            continue
        pos = saved.get("Pos")
        if not isinstance(pos, list) or len(pos) != 3:
            continue
        record = {
            "id": PurePosixPath(name).stem,
            "path": name,
            "x": pos[0], "y": pos[1], "z": pos[2],
        }
        if isinstance(saved.get("Dimension"), str):
            record["dimension"] = saved["Dimension"]
        players.append(record)
    if players:
        result["players"] = players
    return result


class RegionReader:
    def __init__(self, stream: BinaryIO, *, fs: WorldFS, region_x: int, region_z: int, dimension_dir: str = ""):
        self.stream = stream
        self.fs = fs
        self.region_x = region_x
        self.region_z = region_z
        self.dimension_dir = dimension_dir
        header = stream.read(8192)
        if len(header) < 8192:
            raise WorldReadError("Truncated Anvil region header")
        self.locations = header[:4096]

    def chunk_positions(self) -> Iterator[tuple[int, int]]:
        """Yield allocated chunk coordinates without decoding or changing chunks."""

        for index in range(1024):
            location = self.locations[index * 4 : index * 4 + 4]
            if int.from_bytes(location[:3], "big") and location[3]:
                local_x, local_z = index % 32, index // 32
                yield self.region_x * 32 + local_x, self.region_z * 32 + local_z

    def read_chunk(self, chunk_x: int, chunk_z: int) -> dict[str, object] | None:
        local_x, local_z = chunk_x % 32, chunk_z % 32
        index = 4 * (local_x + local_z * 32)
        location = self.locations[index : index + 4]
        offset = int.from_bytes(location[:3], "big")
        sectors = location[3]
        if offset == 0 or sectors == 0:
            return None
        self.stream.seek(offset * 4096)
        length_data = self.stream.read(4)
        if len(length_data) != 4:
            raise WorldReadError(f"Truncated chunk length for {chunk_x},{chunk_z}")
        length = struct.unpack(">I", length_data)[0]
        if length < 1 or length > sectors * 4096 - 4:
            raise WorldReadError(f"Invalid chunk length {length} for {chunk_x},{chunk_z}")
        compression = self.stream.read(1)[0]
        external = bool(compression & 0x80)
        compression &= 0x7F
        if external:
            member = f"{self.dimension_dir}/c.{chunk_x}.{chunk_z}.mcc" if self.dimension_dir else f"c.{chunk_x}.{chunk_z}.mcc"
            if not self.fs.exists(member):
                raise WorldReadError(f"Missing external chunk stream {member}")
            with self.fs.open(member) as external_stream:
                payload = external_stream.read()
        else:
            payload = self.stream.read(length - 1)
        try:
            if compression == 1:
                raw = gzip.decompress(payload)
            elif compression == 2:
                raw = zlib.decompress(payload)
            elif compression == 3:
                raw = payload
            else:
                raise WorldReadError(f"Unsupported Anvil compression type {compression}")
            return loads(raw)
        except (OSError, zlib.error, NBTError) as exc:
            raise WorldReadError(f"Cannot decode chunk {chunk_x},{chunk_z}: {exc}") from exc


def _nibble(data: bytes, index: int) -> int:
    value = data[index // 2]
    return (value >> (4 * (index & 1))) & 0x0F


def _legacy_sections(chunk: dict[str, object]) -> dict[int, tuple[bytes, bytes, bytes | None]]:
    body = chunk.get("Level", chunk)
    sections = body.get("Sections", []) if isinstance(body, dict) else []
    result = {}
    for section in sections:
        if not isinstance(section, dict) or "Blocks" not in section:
            continue
        y = int(section.get("Y", 0))
        blocks = bytes(section["Blocks"])
        data = bytes(section.get("Data", b"\x00" * 2048))
        add = bytes(section["Add"]) if "Add" in section else None
        if len(blocks) == 4096 and len(data) == 2048:
            result[y] = (blocks, data, add)
    return result


def _modern_sections(chunk: dict[str, object]) -> dict[int, tuple[list[object], list[int]]]:
    body = chunk.get("Level", chunk)
    if not isinstance(body, dict):
        return {}
    sections = body.get("sections", body.get("Sections", []))
    result = {}
    for section in sections:
        if not isinstance(section, dict):
            continue
        states = section.get("block_states", section.get("BlockStates"))
        if not isinstance(states, dict):
            continue
        palette = states.get("palette", states.get("Palette", []))
        packed = states.get("data", states.get("BlockStates", []))
        if isinstance(palette, list) and palette:
            result[int(section.get("Y", 0))] = (palette, list(packed))
    return result


def _packed_palette_index(
    packed: list[int], index: int, palette_size: int, entries: int, minimum_bits: int,
) -> int:
    if palette_size <= 1 or not packed:
        return 0
    bits = max(minimum_bits, (palette_size - 1).bit_length())
    per_long = 64 // bits
    padded_length = math.ceil(entries / per_long)
    mask = (1 << bits) - 1
    unsigned = [value & ((1 << 64) - 1) for value in packed]
    if len(unsigned) == padded_length:
        return (unsigned[index // per_long] >> ((index % per_long) * bits)) & mask
    bit = index * bits
    word, shift = divmod(bit, 64)
    value = unsigned[word] >> shift
    if shift + bits > 64 and word + 1 < len(unsigned):
        value |= unsigned[word + 1] << (64 - shift)
    return value & mask


def _palette_index(packed: list[int], index: int, palette_size: int) -> int:
    return _packed_palette_index(packed, index, palette_size, 4096, 4)


def _modern_biomes(chunk: dict[str, object]) -> dict[int, tuple[list[object], list[int]]]:
    body = chunk.get("Level", chunk)
    if not isinstance(body, dict):
        return {}
    result = {}
    for section in body.get("sections", body.get("Sections", [])):
        if not isinstance(section, dict):
            continue
        biomes = section.get("biomes")
        if not isinstance(biomes, dict):
            continue
        palette = biomes.get("palette", [])
        packed = biomes.get("data", [])
        if isinstance(palette, list) and palette:
            result[int(section.get("Y", 0))] = (palette, list(packed))
    return result


def _modern_surface_heights(
    chunk: dict[str, object], section_ids: list[int]
) -> list[int] | None:
    body = chunk.get("Level", chunk)
    if not isinstance(body, dict) or not section_ids:
        return None
    heightmaps = body.get("Heightmaps", body.get("heightmaps"))
    if not isinstance(heightmaps, dict):
        return None
    packed = heightmaps.get("WORLD_SURFACE", heightmaps.get("world_surface"))
    if not isinstance(packed, list) or not packed:
        return None
    minimum_y = min(section_ids) * 16
    world_height = (max(section_ids) - min(section_ids) + 1) * 16
    bits = max(1, world_height.bit_length())
    palette_size = 1 << bits
    return [
        _packed_palette_index(packed, index, palette_size, 256, bits) + minimum_y - 1
        for index in range(256)
    ]


def _modern_key(entry: object) -> str:
    if not isinstance(entry, dict):
        return "unknown:invalid_palette_entry"
    name = str(entry.get("Name", entry.get("name", "unknown:unnamed")))
    properties = entry.get("Properties", entry.get("properties"))
    if not isinstance(properties, dict) or not properties:
        return name
    encoded = ",".join(f"{key}={properties[key]}" for key in sorted(properties))
    return f"{name}[{encoded}]"


class ChunkView:
    def __init__(self, chunk: dict[str, object]):
        self.legacy = _legacy_sections(chunk)
        self.modern = _modern_sections(chunk)
        self.modern_biomes = _modern_biomes(chunk)
        self.modern_surface_heights = _modern_surface_heights(chunk, sorted(self.modern))
        body = chunk.get("Level", chunk)
        raw_biomes = body.get("Biomes", b"") if isinstance(body, dict) else b""
        self.legacy_biomes = bytes(raw_biomes) if isinstance(raw_biomes, (bytes, bytearray, list)) else b""

    def biome_at(self, x: int, z: int, y: int | None = None) -> int | str | None:
        """Return the stored legacy column biome ID.

        Legacy Anvil stores one unsigned byte per X/Z column. Modern Anvil
        stores a 4×4×4 namespaced biome palette in each section.
        """

        if len(self.legacy_biomes) == 256:
            return self.legacy_biomes[(z & 15) * 16 + (x & 15)]
        if self.modern_biomes:
            section_y = y // 16 if y is not None else max(self.modern_biomes)
            palette, packed = self.modern_biomes.get(section_y, self.modern_biomes[max(self.modern_biomes)])
            local_y = ((y or (section_y * 16 + 15)) & 15) >> 2
            index = (local_y << 4) | ((z & 15) >> 2 << 2) | ((x & 15) >> 2)
            palette_index = _packed_palette_index(packed, index, len(palette), 64, 1)
            return str(palette[palette_index]) if palette_index < len(palette) else None
        return None

    def block_at(self, x: int, y: int, z: int) -> Block | None:
        section_y = y // 16
        local_y = y - section_y * 16
        index = (local_y << 8) | (z << 4) | x
        if section_y in self.modern:
            palette, packed = self.modern[section_y]
            palette_index = _palette_index(packed, index, len(palette))
            if palette_index >= len(palette):
                return Block("unknown:palette_index_out_of_range", y)
            key = _modern_key(palette[palette_index])
            return None if key in AIR_NAMES else Block(key, y)
        if section_y in self.legacy:
            blocks, metadata, add = self.legacy[section_y]
            block_id = blocks[index]
            if add is not None:
                block_id |= _nibble(add, index) << 8
            if block_id == 0:
                return None
            meta = _nibble(metadata, index)
            return Block(f"legacy:{block_id}:{meta}", y, block_id, meta)
        return None

    def top_block(
        self,
        x: int,
        z: int,
        max_y: int | None = None,
        skip: Callable[[Block], bool] | None = None,
    ) -> Block | None:
        if self.modern_surface_heights is not None and self.modern:
            top = self.modern_surface_heights[(z & 15) * 16 + (x & 15)]
            if max_y is not None:
                top = min(top, max_y)
            bottom = min(self.modern) * 16
            for y in range(top, bottom - 1, -1):
                block = self.block_at(x, y, z)
                if block is not None and not (skip and skip(block)):
                    return block
            return None
        section_ids = sorted(set(self.legacy) | set(self.modern), reverse=True)
        for section_y in section_ids:
            top = section_y * 16 + 15
            bottom = section_y * 16
            if max_y is not None:
                if bottom > max_y:
                    continue
                top = min(top, max_y)
            for y in range(top, bottom - 1, -1):
                block = self.block_at(x, y, z)
                if block is not None and not (skip and skip(block)):
                    return block
        return None

    def blocks_top_down(self, x: int, z: int, max_y: int | None = None) -> Iterator[Block]:
        """Stream non-air blocks in one column from highest to lowest."""

        if self.modern_surface_heights is not None and self.modern:
            top = self.modern_surface_heights[(z & 15) * 16 + (x & 15)]
            if max_y is not None:
                top = min(top, max_y)
            bottom = min(self.modern) * 16
            for y in range(top, bottom - 1, -1):
                block = self.block_at(x, y, z)
                if block is not None:
                    yield block
            return
        section_ids = sorted(set(self.legacy) | set(self.modern), reverse=True)
        for section_y in section_ids:
            top = section_y * 16 + 15
            bottom = section_y * 16
            if max_y is not None:
                if bottom > max_y:
                    continue
                top = min(top, max_y)
            for y in range(top, bottom - 1, -1):
                block = self.block_at(x, y, z)
                if block is not None:
                    yield block


class AnvilWorld:
    def __init__(self, source: str | Path, dimension: str = "minecraft:overworld"):
        self.fs = open_world(source)
        dimensions = {item.id: item for item in discover_dimensions(self.fs)}
        if dimension not in dimensions:
            raise WorldReadError(f"Dimension {dimension!r} not found; available: {sorted(dimensions)}")
        self.dimension = dimensions[dimension]
        self._regions: dict[tuple[int, int], tuple[BinaryIO, RegionReader]] = {}
        self._chunks: OrderedDict[tuple[int, int], ChunkView | None] = OrderedDict()
        self.chunk_cache_size = 128
        self.region_errors: list[dict[str, object]] = []

    def _region(self, region_x: int, region_z: int) -> RegionReader | None:
        key = (region_x, region_z)
        if key in self._regions:
            return self._regions[key][1]
        relative = f"{self.dimension.region_dir}/r.{region_x}.{region_z}.mca"
        if not self.fs.exists(relative):
            return None
        stream = self.fs.open(relative)
        reader = RegionReader(stream, fs=self.fs, region_x=region_x, region_z=region_z, dimension_dir=self.dimension.region_dir)
        self._regions[key] = (stream, reader)
        return reader

    def chunk(self, chunk_x: int, chunk_z: int) -> ChunkView | None:
        key = (chunk_x, chunk_z)
        if key in self._chunks:
            self._chunks.move_to_end(key)
            return self._chunks[key]
        region = self._region(chunk_x // 32, chunk_z // 32)
        if region is None:
            result = None
        else:
            chunk = region.read_chunk(chunk_x, chunk_z)
            result = ChunkView(chunk) if chunk is not None else None
        self._chunks[key] = result
        if len(self._chunks) > self.chunk_cache_size:
            self._chunks.popitem(last=False)
        return result

    def chunk_positions(self) -> Iterator[tuple[int, int]]:
        """Yield all allocated chunks in the selected dimension."""

        regions = []
        prefix = self.dimension.region_dir.rstrip("/") + "/"
        for name in self.fs.names():
            if not name.startswith(prefix):
                continue
            match = REGION_RE.search(name)
            if match:
                regions.append((int(match.group(1)), int(match.group(2))))
        for region_x, region_z in sorted(set(regions), key=lambda item: (item[1], item[0])):
            try:
                region = self._region(region_x, region_z)
            except WorldReadError as exc:
                self.region_errors.append({"regionX": region_x, "regionZ": region_z, "error": str(exc)})
                continue
            if region is not None:
                yield from region.chunk_positions()

    def block_at(self, x: int, y: int, z: int) -> Block | None:
        chunk = self.chunk(x // 16, z // 16)
        return None if chunk is None else chunk.block_at(x & 15, y, z & 15)

    def biome_at(self, x: int, z: int, y: int | None = None) -> int | str | None:
        chunk = self.chunk(x // 16, z // 16)
        return None if chunk is None else chunk.biome_at(x & 15, z & 15, y)

    def close(self) -> None:
        self._chunks.clear()
        for stream, _ in self._regions.values():
            stream.close()
        self._regions.clear()

    def __enter__(self) -> "AnvilWorld":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
