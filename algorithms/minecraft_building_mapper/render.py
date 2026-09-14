"""Render fixed world tiles while recording complete resolution evidence."""

from __future__ import annotations

import hashlib
import io
import json
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .catalog import Catalog
from .evidence import RESOLUTION_KINDS
from .world import AnvilWorld, Block, WorldReadError, read_level_metadata


TILE_BLOCKS = 256
METADATA_SCHEMA = "geo.minecraft-top-map/v1"
REPORT_SCHEMA = "geo.minecraft-top-map-report/v1"


def _pillow():
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required for rendering; install the algorithm's render extra") from exc
    return Image


@dataclass(frozen=True)
class TileRequest:
    tile_x: int
    tile_z: int
    layer: int | None = None
    pixels_per_block: int = 1

    def __post_init__(self) -> None:
        if not 1 <= self.pixels_per_block <= 16:
            raise ValueError("pixels_per_block must be between 1 and 16")

    @property
    def layer_label(self) -> str:
        return "surface" if self.layer is None else str(self.layer)

    @property
    def bounds(self) -> dict[str, int]:
        min_x, min_z = self.tile_x * TILE_BLOCKS, self.tile_z * TILE_BLOCKS
        return {"minX": min_x, "minZ": min_z, "maxXExclusive": min_x + TILE_BLOCKS, "maxZExclusive": min_z + TILE_BLOCKS}


class TextureRenderer:
    def __init__(self, catalog: Catalog, pixels_per_block: int):
        self.catalog = catalog
        self.ppb = pixels_per_block
        self.Image = _pillow()
        self._cache: dict[str, Any] = {}
        self.failures: dict[str, str] = {}

    def _unknown(self, key: str):
        digest = hashlib.sha256(key.encode()).digest()
        image = self.Image.new("RGBA", (self.ppb, self.ppb), (238, 0, 238, 255))
        pixels = image.load()
        for z in range(self.ppb):
            for x in range(self.ppb):
                if ((x + z + digest[0]) & 1) == 0:
                    pixels[x, z] = (20, 20, 20, 255)
        return image

    def block_image(self, block: Block):
        if block.key in self._cache:
            return self._cache[block.key]
        entry = self.catalog.entry(block.key)
        if not entry or entry.get("status") != "resolved" or "texture" not in entry:
            image = self._unknown(block.key)
        else:
            try:
                image = self.Image.open(io.BytesIO(self.catalog.texture_bytes(entry))).convert("RGBA")
                atlas = entry["texture"].get("atlas")
                if atlas:
                    columns = int(atlas["columns"])
                    cell_w = image.width // columns
                    rows = image.height // cell_w
                    index = int(atlas["index"])
                    if index >= columns * rows:
                        raise ValueError("Atlas index lies outside texture")
                    left = (index % columns) * cell_w
                    top = (index // columns) * cell_w
                    image = image.crop((left, top, left + cell_w, top + cell_w))
                image = image.resize((self.ppb, self.ppb), self.Image.Resampling.LANCZOS)
            except Exception as exc:
                self.failures[block.key] = f"texture read failed: {exc}"
                image = self._unknown(block.key)
        self._cache[block.key] = image
        return image


def render_tile(world: AnvilWorld, catalog: Catalog, request: TileRequest) -> tuple[Any, dict[str, object]]:
    Image = _pillow()
    size = TILE_BLOCKS * request.pixels_per_block
    canvas = Image.new("RGBA", (size, size), (14, 18, 24, 255))
    textures = TextureRenderer(catalog, request.pixels_per_block)
    block_counts: Counter[str] = Counter()
    resolution_counts: dict[str, Counter[str]] = {kind: Counter() for kind in RESOLUTION_KINDS}
    skipped_as_air: Counter[str] = Counter()
    y_values: list[int] = []
    missing_chunks = 0
    loaded_chunks = 0
    chunk_errors: list[dict[str, object]] = []
    bounds = request.bounds

    def should_skip(candidate: Block) -> bool:
        if catalog.renders_as_air(candidate.key):
            skipped_as_air[candidate.key] += 1
            return True
        return False

    for chunk_z in range(bounds["minZ"] // 16, bounds["maxZExclusive"] // 16):
        for chunk_x in range(bounds["minX"] // 16, bounds["maxXExclusive"] // 16):
            try:
                chunk = world.chunk(chunk_x, chunk_z)
            except WorldReadError as exc:
                chunk_errors.append({"chunkX": chunk_x, "chunkZ": chunk_z, "error": str(exc)})
                continue
            if chunk is None:
                missing_chunks += 1
                continue
            loaded_chunks += 1
            for local_z in range(16):
                for local_x in range(16):
                    block = chunk.top_block(local_x, local_z, request.layer, should_skip)
                    if block is None:
                        continue
                    block_counts[block.key] += 1
                    entry = catalog.entry(block.key)
                    y_values.append(block.y)
                    pixel_x = (chunk_x * 16 + local_x - bounds["minX"]) * request.pixels_per_block
                    pixel_z = (chunk_z * 16 + local_z - bounds["minZ"]) * request.pixels_per_block
                    block_image = textures.block_image(block)
                    if not entry or entry.get("status") != "resolved" or block.key in textures.failures:
                        resolution = "unknown"
                    else:
                        resolution = str(entry.get("resolution", "resolved"))
                        if resolution not in resolution_counts:
                            resolution = "unknown"
                    resolution_counts[resolution][block.key] += 1
                    canvas.alpha_composite(block_image, (pixel_x, pixel_z))
    unknown_details = {}
    for key, count in sorted(resolution_counts["unknown"].items()):
        entry = catalog.entry(key) or {}
        unknown_details[key] = {
            "count": count,
            "catalogName": entry.get("name"),
            "reason": textures.failures.get(key) or entry.get("reason") or "no exact catalog entry",
        }
    resolution_report = {
        kind: {
            "visibleBlocks": sum(counts.values()),
            "distinctBlockStates": len(counts),
            "blockCounts": dict(sorted(counts.items())),
        }
        for kind, counts in resolution_counts.items()
    }
    resolution_report["unknown"]["details"] = unknown_details
    report = {
        "schema": REPORT_SCHEMA,
        "status": "ok" if not chunk_errors else "partial",
        "tile": {"x": request.tile_x, "z": request.tile_z, "blocks": TILE_BLOCKS, "pixelsPerBlock": request.pixels_per_block},
        "layer": request.layer_label,
        "bounds": bounds,
        "chunks": {"loaded": loaded_chunks, "missing": missing_chunks, "errors": chunk_errors},
        "visibleBlocks": sum(block_counts.values()),
        "height": {"min": min(y_values) if y_values else None, "max": max(y_values) if y_values else None},
        "blockCounts": dict(sorted(block_counts.items())),
        "skippedAsAir": {"blocks": sum(skipped_as_air.values()), "blockCounts": dict(sorted(skipped_as_air.items()))},
        "resolution": resolution_report,
        "unknown": resolution_report["unknown"],
    }
    return canvas.convert("RGB"), report


def render_tile_to_output(
    *, world_path: str | Path, dimension: str, catalog_path: str | Path, request: TileRequest, output: str | Path
) -> dict[str, object]:
    output_path = Path(output).expanduser()
    output_path.mkdir(parents=True, exist_ok=True)
    catalog = Catalog(catalog_path)
    with AnvilWorld(world_path, dimension) as world:
        level = read_level_metadata(world.fs)
        image, report = render_tile(world, catalog, request)
    tile_name = f"tile_{request.tile_x}_{request.tile_z}.png"
    image.save(output_path / tile_name, format="PNG", optimize=True)
    metadata = {
        "schema": METADATA_SCHEMA,
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "readOnly": True,
        "world": {"source": str(Path(world_path).expanduser().resolve()), **level},
        "selector": {"dimension": dimension, "layer": request.layer_label},
        "tile": {"x": request.tile_x, "z": request.tile_z, "blocks": TILE_BLOCKS, "image": tile_name},
        "catalog": {"path": str(Path(catalog_path).expanduser().resolve()), "schema": catalog.data["schema"], "sources": catalog.data.get("sources", [])},
        "extensions": {},
    }
    (output_path / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    (output_path / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    return {"image": str(output_path / tile_name), "metadata": str(output_path / "metadata.json"), "report": str(output_path / "report.json"), **report}
