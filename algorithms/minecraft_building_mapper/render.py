"""Render read-only world tiles with legacy color, geometry, and provenance."""

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
from .legacy_visuals import VANILLA_152_BIOMES, biome_colorizer_xy
from .world import AnvilWorld, Block, WorldReadError, read_level_metadata


TILE_BLOCKS = 256
METADATA_SCHEMA = "geo.minecraft-top-map/v1"
REPORT_SCHEMA = "geo.minecraft-top-map-report/v2"
TRANSPARENT_GEOMETRIES = {
    "alpha_cube", "fluid", "cross", "point", "line", "plane",
    "connected", "partial", "stair", "slab", "nonstandard",
}


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


@dataclass(frozen=True)
class SurfaceLayer:
    block: Block
    fluid_depth: int = 0


def _resolution(catalog: Catalog, key: str, failures: dict[str, str]) -> str:
    entry = catalog.entry(key)
    if not entry or entry.get("status") != "resolved" or key in failures:
        return "unknown"
    value = str(entry.get("resolution", "resolved"))
    return value if value in RESOLUTION_KINDS else "unknown"


class LegacyTintResolver:
    """Apply the exact bundled 1.5.2/OptiFine colormap lookup semantics."""

    def __init__(self, catalog: Catalog, world: AnvilWorld):
        self.catalog = catalog
        self.world = world
        self.Image = _pillow()
        self.maps: dict[str, Any] = {}
        for name in catalog.data.get("colorizers", {}):
            try:
                image = self.Image.open(io.BytesIO(catalog.colorizer_bytes(name))).convert("RGB")
                if image.size == (256, 256):
                    self.maps[name] = image
            except Exception:
                continue
        options = catalog.data.get("legacyRendering", {}).get("options", {})
        self.smooth = bool(options.get("smoothBiomes"))
        self.swamp = bool(options.get("swampColors"))
        self.fixed = catalog.data.get("fixedColors", {})
        self.biome_definitions = catalog.data.get("legacyBiomes") or catalog.data.get("modernBiomes") or {
            str(biome_id): {
                "name": name, "temperature": temperature, "rainfall": rainfall,
                "waterMultiplier": water,
            }
            for biome_id, (name, temperature, rainfall, water) in VANILLA_152_BIOMES.items()
        }
        self.applied: Counter[str] = Counter()
        self.failures: Counter[str] = Counter()
        self.biomes: Counter[str] = Counter()

    @staticmethod
    def _rgb(value: int) -> tuple[int, int, int]:
        return value >> 16 & 255, value >> 8 & 255, value & 255

    def _biome(self, x: int, z: int, y: int) -> tuple[int | str, tuple[str, float, float, int]] | None:
        biome = self.world.biome_at(x, z, y)
        definition = self.biome_definitions.get(str(biome)) if isinstance(biome, (int, str)) else None
        if not isinstance(definition, dict):
            self.failures[f"unknown-biome:{biome}"] += 1
            return None
        self.biomes[str(biome)] += 1
        return biome, (
            str(definition["name"]), float(definition["temperature"]),
            float(definition["rainfall"]), int(definition.get("waterMultiplier", 0xFFFFFF)),
        )

    def _map_color(self, map_name: str, x: int, z: int, y: int) -> tuple[int, int, int] | None:
        biome = self._biome(x, z, y)
        image = self.maps.get(map_name)
        if biome is None or image is None:
            if image is None:
                self.failures[f"missing-colorizer:{map_name}"] += 1
            return None
        definition = self.biome_definitions[str(biome[0])]
        override = definition.get("grassColor" if map_name == "grass" else "foliageColor")
        if isinstance(override, int):
            return self._rgb(override)
        px, py = biome_colorizer_xy(biome[1][1], biome[1][2])
        color = image.getpixel((px, py))
        modifier = definition.get("grassColorModifier") if map_name == "grass" else None
        if modifier == "dark_forest":
            value = ((color[0] << 16 | color[1] << 8 | color[2]) & 0xFEFEFE) + 0x28340A >> 1
            return self._rgb(value)
        if modifier == "swamp":
            self.failures["modern-swamp-noise-fallback:climate-color"] += 1
        return color

    def _sample(self, kind: str, x: int, z: int, y: int) -> tuple[int, int, int] | None:
        map_name = kind
        biome = self.world.biome_at(x, z, y)
        if self.swamp and biome == 6 and kind in {"grass", "foliage"} and f"swamp_{kind}" in self.maps:
            map_name = f"swamp_{kind}"
        return self._map_color(map_name, x, z, y)

    def color(self, block: Block, x: int, z: int) -> tuple[int, int, int] | None:
        entry = self.catalog.entry(block.key) or {}
        visual = entry.get("visual", {})
        kind = self.catalog.block_colorizer(block.key) or visual.get("tint")
        if not kind:
            return None
        if kind == "lily_pad":
            value = self.fixed.get("lily_pad")
            if isinstance(value, int):
                self.applied[kind] += 1
                return self._rgb(value)
            self.failures["missing-fixed-color:lily_pad"] += 1
            return None
        if kind == "stem":
            meta = block.metadata or 0
            self.applied[kind] += 1
            return min(meta * 32, 255), max(255 - meta * 8, 0), min(meta * 4, 255)
        if kind == "water":
            colors = []
            offsets = (-1, 0, 1) if self.smooth else (0,)
            for dz in offsets:
                for dx in offsets:
                    biome = self._biome(x + dx, z + dz, block.y)
                    if biome is not None:
                        colors.append(self._rgb(biome[1][3]))
        else:
            offsets = (-1, 0, 1) if self.smooth else (0,)
            colors = [
                color for dz in offsets for dx in offsets
                if (color := self._sample(str(kind), x + dx, z + dz, block.y))
            ]
        if not colors:
            return None
        self.applied[str(kind)] += 1
        return tuple(sum(item[channel] for item in colors) // len(colors) for channel in range(3))


class TextureRenderer:
    def __init__(self, catalog: Catalog, pixels_per_block: int):
        self.catalog = catalog
        self.ppb = pixels_per_block
        self.Image = _pillow()
        self._cache: dict[tuple[str, tuple[int, int, int] | None, int], Any] = {}
        self._overview_cache: dict[tuple[str, tuple[int, int, int] | None, str, int], tuple[int, int, int, int]] = {}
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

    def _texture(self, block: Block, tint: tuple[int, int, int] | None):
        cache_key = (block.key, tint, self.ppb)
        if cache_key in self._cache:
            return self._cache[cache_key].copy()
        entry = self.catalog.entry(block.key)
        if not entry or entry.get("status") != "resolved" or "texture" not in entry:
            image = self._unknown(block.key)
        else:
            try:
                image = self.Image.open(io.BytesIO(self.catalog.texture_bytes(entry))).convert("RGBA")
                texture = entry["texture"]
                atlas = texture.get("atlas")
                if atlas:
                    columns = int(atlas["columns"])
                    cell_w = image.width // columns
                    rows = image.height // cell_w
                    index = int(atlas["index"])
                    if index >= columns * rows:
                        raise ValueError("Atlas index lies outside texture")
                    left, top = (index % columns) * cell_w, (index // columns) * cell_w
                    image = image.crop((left, top, left + cell_w, top + cell_w))
                elif "crop" in texture:
                    left, top, width, height = (float(value) for value in texture["crop"])
                    image = image.crop((round(left * image.width), round(top * image.height),
                                        round((left + width) * image.width), round((top + height) * image.height)))
                elif image.height > image.width and image.height % image.width == 0:
                    image = image.crop((0, 0, image.width, image.width))
                if tint is not None:
                    pixels = image.load()
                    for py in range(image.height):
                        for px in range(image.width):
                            red, green, blue, alpha = pixels[px, py]
                            pixels[px, py] = (
                                red * tint[0] // 255, green * tint[1] // 255,
                                blue * tint[2] // 255, alpha,
                            )
                image = image.resize((self.ppb, self.ppb), self.Image.Resampling.LANCZOS)
            except Exception as exc:
                self.failures[block.key] = f"texture read failed: {exc}"
                image = self._unknown(block.key)
        self._cache[cache_key] = image.copy()
        return image

    @staticmethod
    def _average(image: Any) -> tuple[int, int, int, int]:
        opaque = [pixel for pixel in image.getdata() if pixel[3] > 0]
        if not opaque:
            return 255, 0, 255, 220
        return tuple(sum(pixel[index] for pixel in opaque) // len(opaque) for index in range(4))

    def _line_art(
        self, image: Any, lines: list[tuple[tuple[float, float], tuple[float, float]]], width: int | None = None,
    ):
        from PIL import ImageDraw

        result = self.Image.new("RGBA", (self.ppb, self.ppb), (0, 0, 0, 0))
        color = self._average(image)
        draw = ImageDraw.Draw(result)
        stroke = width or max(1, self.ppb // 4)
        maximum = self.ppb - 1
        for start, end in lines:
            draw.line((round(start[0] * maximum), round(start[1] * maximum),
                       round(end[0] * maximum), round(end[1] * maximum)), fill=color, width=stroke)
        return result

    def geometry_image(
        self, block: Block, tint: tuple[int, int, int] | None, visual: dict[str, Any],
        connections: set[str], world: AnvilWorld, x: int, z: int, fluid_depth: int = 0,
    ):
        image = self._texture(block, tint)
        geometry = visual.get("geometry", "cube")
        if self.ppb == 1 and geometry in {"cross", "point", "line", "plane", "connected"}:
            red, green, blue, alpha = self._average(image)
            return self.Image.new("RGBA", (1, 1), (red, green, blue, max(150, min(alpha, 220))))
        if geometry == "cross":
            return self._line_art(image, [((0.0, 0.0), (1.0, 1.0)), ((1.0, 0.0), (0.0, 1.0))])
        if geometry == "point":
            return self._line_art(image, [((0.5, 0.5), (0.5, 0.5))], max(1, self.ppb // 3))
        if geometry == "line":
            meta = block.metadata or 0
            lines = []
            if block.legacy_id in {27, 28, 66, 157}:
                rail = meta & 7
                if rail in {0, 4, 5}:
                    lines = [((0.5, 0.0), (0.5, 1.0))]
                elif rail in {1, 2, 3}:
                    lines = [((0.0, 0.5), (1.0, 0.5))]
                else:
                    corners = {6: ("E", "S"), 7: ("W", "S"), 8: ("W", "N"), 9: ("E", "N")}
                    points = {"N": (0.5, 0.0), "S": (0.5, 1.0), "W": (0.0, 0.5), "E": (1.0, 0.5)}
                    lines = [((0.5, 0.5), points[direction]) for direction in corners.get(meta & 15, ("N", "S"))]
            else:
                points = {"N": (0.5, 0.0), "S": (0.5, 1.0), "W": (0.0, 0.5), "E": (1.0, 0.5)}
                lines = [((0.5, 0.5), points[direction]) for direction in connections]
                if not lines:
                    lines = [((0.0, 0.5), (1.0, 0.5))]
            return self._line_art(image, lines)
        if geometry == "plane":
            meta = block.metadata or 0
            if block.legacy_id == 96 and not meta & 4:
                return image
            if block.legacy_id == 106:
                lines = []
                if meta & 1: lines.append(((0.0, 0.0), (1.0, 0.0)))
                if meta & 2: lines.append(((1.0, 0.0), (1.0, 1.0)))
                if meta & 4: lines.append(((0.0, 1.0), (1.0, 1.0)))
                if meta & 8: lines.append(((0.0, 0.0), (0.0, 1.0)))
                return self._line_art(image, lines or [((0.0, 0.0), (1.0, 1.0))])
            orientation = meta & 3
            if block.legacy_id in {64, 71} and meta & 8:
                below = world.block_at(x, block.y - 1, z)
                if below and below.legacy_id == block.legacy_id:
                    orientation = (below.metadata or 0) & 3
            if block.legacy_id in {64, 71} and meta & 4:
                orientation = (orientation + (1 if meta & 1 else -1)) & 3
            lines = [
                ((0.0, 0.1), (1.0, 0.1)), ((0.9, 0.0), (0.9, 1.0)),
                ((0.0, 0.9), (1.0, 0.9)), ((0.1, 0.0), (0.1, 1.0)),
            ]
            return self._line_art(image, [lines[orientation]])
        if geometry == "connected":
            points = {"N": (0.5, 0.0), "S": (0.5, 1.0), "W": (0.0, 0.5), "E": (1.0, 0.5)}
            lines = [((0.5, 0.5), points[direction]) for direction in connections]
            lines.append(((0.5, 0.5), (0.5, 0.5)))
            return self._line_art(image, lines, max(1, self.ppb // 3))
        if geometry == "fluid":
            pixels = image.load()
            target_alpha = min(220, 92 + max(fluid_depth, 1) * 14)
            for py in range(image.height):
                for px in range(image.width):
                    red, green, blue, alpha = pixels[px, py]
                    depth_dark = min(max(fluid_depth - 1, 0) * 5, 45)
                    pixels[px, py] = (
                        max(0, red - depth_dark), max(0, green - depth_dark),
                        max(0, blue - depth_dark), max(alpha, target_alpha),
                    )
            return image
        if geometry == "partial" and self.ppb >= 4:
            from PIL import ImageDraw

            inset = max(1, self.ppb // 8)
            mask = self.Image.new("L", image.size, 0)
            ImageDraw.Draw(mask).rectangle((inset, inset, self.ppb - 1 - inset, self.ppb - 1 - inset), fill=255)
            image.putalpha(mask)
        if geometry == "stair" and self.ppb >= 4:
            from PIL import ImageDraw

            orientation = (block.metadata or 0) & 3
            divider = [
                ((self.ppb // 2, 0), (self.ppb // 2, self.ppb - 1)),
                ((self.ppb // 2, 0), (self.ppb // 2, self.ppb - 1)),
                ((0, self.ppb // 2), (self.ppb - 1, self.ppb // 2)),
                ((0, self.ppb // 2), (self.ppb - 1, self.ppb // 2)),
            ][orientation]
            ImageDraw.Draw(image).line((*divider[0], *divider[1]), fill=(0, 0, 0, 70), width=1)
        return image

    def overview_rgba(
        self, block: Block, tint: tuple[int, int, int] | None,
        visual: dict[str, Any], fluid_depth: int = 0,
    ) -> tuple[int, int, int, int]:
        """Return the cached one-pixel satellite representative for a layer."""

        geometry = str(visual.get("geometry", "cube"))
        cache_key = (block.key, tint, geometry, fluid_depth if geometry == "fluid" else 0)
        if cache_key in self._overview_cache:
            return self._overview_cache[cache_key]
        red, green, blue, alpha = self._average(self._texture(block, tint))
        if geometry in {"cross", "point", "line", "plane", "connected"}:
            alpha = max(150, min(alpha, 220))
        elif geometry == "fluid":
            depth_dark = min(max(fluid_depth - 1, 0) * 5, 45)
            red, green, blue = max(0, red - depth_dark), max(0, green - depth_dark), max(0, blue - depth_dark)
            alpha = max(alpha, min(220, 92 + max(fluid_depth, 1) * 14))
        value = red, green, blue, alpha
        self._overview_cache[cache_key] = value
        return value


def _visual(catalog: Catalog, block: Block) -> dict[str, Any]:
    entry = catalog.entry(block.key) or {}
    value = entry.get("visual")
    return value if isinstance(value, dict) else {"geometry": "cube", "height": 1.0}


def _connects(world: AnvilWorld, catalog: Catalog, block: Block, x: int, z: int) -> set[str]:
    result = set()
    for direction, dx, dz in (("N", 0, -1), ("S", 0, 1), ("W", -1, 0), ("E", 1, 0)):
        neighbor = world.block_at(x + dx, block.y, z + dz)
        if neighbor is None:
            continue
        neighbor_geometry = _visual(catalog, neighbor).get("geometry", "cube")
        if neighbor.legacy_id == block.legacy_id or neighbor_geometry in {"cube", "connected", "alpha_cube"}:
            result.add(direction)
    return result


def _column_layers(
    chunk: Any, local_x: int, local_z: int, catalog: Catalog, layer: int | None, skipped: Counter[str],
) -> list[SurfaceLayer]:
    layers: list[SurfaceLayer] = []
    fluid_id = None
    fluid_index = None
    fluid_depth = 0
    for block in chunk.blocks_top_down(local_x, local_z, layer):
        if catalog.renders_as_air(block.key):
            skipped[block.key] += 1
            continue
        visual = _visual(catalog, block)
        geometry = visual.get("geometry", "cube")
        if geometry == "fluid":
            if fluid_id is None:
                fluid_id = block.legacy_id
                fluid_index = len(layers)
                layers.append(SurfaceLayer(block, 1))
            if block.legacy_id == fluid_id:
                fluid_depth += 1
                assert fluid_index is not None
                layers[fluid_index] = SurfaceLayer(layers[fluid_index].block, fluid_depth)
                continue
        layers.append(SurfaceLayer(block))
        if geometry not in TRANSPARENT_GEOMETRIES or geometry == "cover":
            break
        if len(layers) >= 32:
            break
    return layers


def _shade(canvas: Any, heights: list[list[float | None]], ppb: int) -> dict[str, float]:
    pixels = canvas.load()
    rows, columns = len(heights), len(heights[0]) if heights else 0
    minimum, maximum = 1.0, 1.0
    for z in range(rows):
        for x in range(columns):
            height = heights[z][x]
            if height is None:
                continue
            north = heights[z - 1][x] if z else height
            south = heights[z + 1][x] if z + 1 < rows else height
            west = heights[z][x - 1] if x else height
            east = heights[z][x + 1] if x + 1 < columns else height
            neighbors = [value if value is not None else height for value in (north, south, west, east)]
            light = (neighbors[0] - neighbors[1] + neighbors[2] - neighbors[3]) * 0.028
            edge = max(0.0, height - max(neighbors[1], neighbors[3])) * 0.018
            factor = max(0.68, min(1.22, 1.0 + light - edge))
            minimum, maximum = min(minimum, factor), max(maximum, factor)
            for py in range(z * ppb, (z + 1) * ppb):
                for px in range(x * ppb, (x + 1) * ppb):
                    red, green, blue, alpha = pixels[px, py]
                    pixels[px, py] = (
                        min(255, round(red * factor)), min(255, round(green * factor)),
                        min(255, round(blue * factor)), alpha,
                    )
    return {"minimumFactor": round(minimum, 4), "maximumFactor": round(maximum, 4)}


def render_tile(world: AnvilWorld, catalog: Catalog, request: TileRequest) -> tuple[Any, dict[str, object]]:
    Image = _pillow()
    size = TILE_BLOCKS * request.pixels_per_block
    canvas = Image.new("RGBA", (size, size), (14, 18, 24, 255))
    textures = TextureRenderer(catalog, request.pixels_per_block)
    tints = LegacyTintResolver(catalog, world)
    block_counts: Counter[str] = Counter()
    geometry_counts: Counter[str] = Counter()
    coverage_counts: dict[str, Counter[str]] = {kind: Counter() for kind in RESOLUTION_KINDS}
    skipped_as_air: Counter[str] = Counter()
    water_depths: Counter[str] = Counter()
    heights: list[list[float | None]] = [[None] * TILE_BLOCKS for _ in range(TILE_BLOCKS)]
    missing_chunks = 0
    loaded_chunks = 0
    chunk_errors: list[dict[str, object]] = []
    bounds = request.bounds

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
                    world_x, world_z = chunk_x * 16 + local_x, chunk_z * 16 + local_z
                    layers = _column_layers(chunk, local_x, local_z, catalog, request.layer, skipped_as_air)
                    if not layers:
                        continue
                    top = layers[0].block
                    block_counts[top.key] += 1
                    top_visual = _visual(catalog, top)
                    grid_x, grid_z = world_x - bounds["minX"], world_z - bounds["minZ"]
                    heights[grid_z][grid_x] = top.y + float(top_visual.get("height", 1.0))
                    pixel_x, pixel_z = grid_x * request.pixels_per_block, grid_z * request.pixels_per_block
                    for surface in reversed(layers):
                        block = surface.block
                        visual = _visual(catalog, block)
                        geometry = str(visual.get("geometry", "cube"))
                        geometry_counts[geometry] += 1
                        tint = tints.color(block, world_x, world_z)
                        connections = _connects(world, catalog, block, world_x, world_z) if geometry in {"line", "connected"} else set()
                        image = textures.geometry_image(
                            block, tint, visual, connections, world, world_x, world_z, surface.fluid_depth,
                        )
                        coverage_counts[_resolution(catalog, block.key, textures.failures)][block.key] += 1
                        if surface.fluid_depth:
                            water_depths[str(surface.fluid_depth)] += 1
                        canvas.alpha_composite(image, (pixel_x, pixel_z))
    shade_report = _shade(canvas, heights, request.pixels_per_block)
    resolution_report = {
        kind: {
            "surfaceContributions": sum(counts.values()),
            "visibleBlocks": sum(counts.values()),
            "distinctBlockStates": len(counts),
            "blockCounts": dict(sorted(counts.items())),
        }
        for kind, counts in coverage_counts.items()
    }
    unknown_details = {}
    for key, count in sorted(coverage_counts["unknown"].items()):
        entry = catalog.entry(key) or {}
        unknown_details[key] = {
            "count": count, "catalogName": entry.get("name"),
            "reason": textures.failures.get(key) or entry.get("reason") or "no catalog entry",
        }
    resolution_report["unknown"]["details"] = unknown_details
    y_values = [value for row in heights for value in row if value is not None]
    report = {
        "schema": REPORT_SCHEMA,
        "status": "ok" if not chunk_errors else "partial",
        "tile": {"x": request.tile_x, "z": request.tile_z, "blocks": TILE_BLOCKS, "pixelsPerBlock": request.pixels_per_block},
        "layer": request.layer_label,
        "bounds": bounds,
        "chunks": {"loaded": loaded_chunks, "missing": missing_chunks, "errors": chunk_errors},
        "visibleBlocks": sum(block_counts.values()),
        "height": {"min": min(y_values) if y_values else None, "max": max(y_values) if y_values else None,
                   "hillshade": shade_report},
        "blockCounts": dict(sorted(block_counts.items())),
        "geometry": {"surfaceContributions": sum(geometry_counts.values()), "counts": dict(sorted(geometry_counts.items()))},
        "fluidDepth": dict(sorted(water_depths.items(), key=lambda item: int(item[0]))),
        "biomeTint": {
            "algorithm": catalog.data.get("legacyRendering", {}),
            "applied": dict(sorted(tints.applied.items())),
            "sampledBiomeIds": dict(sorted(
                tints.biomes.items(), key=lambda item: (0, int(item[0])) if item[0].lstrip("-").isdigit() else (1, item[0]),
            )),
            "failures": dict(sorted(tints.failures.items())),
        },
        "skippedAsAir": {"blocks": sum(skipped_as_air.values()), "blockCounts": dict(sorted(skipped_as_air.items()))},
        "resolution": resolution_report,
        "unknown": resolution_report["unknown"],
    }
    return canvas.convert("RGB"), report


def render_tile_to_output(
    *, world_path: str | Path, dimension: str, catalog_path: str | Path, request: TileRequest, output: str | Path,
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
        "catalog": {
            "path": str(Path(catalog_path).expanduser().resolve()), "schema": catalog.data["schema"],
            "sources": catalog.data.get("sources", []), "legacyRendering": catalog.data.get("legacyRendering"),
        },
        "extensions": {},
    }
    (output_path / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    (output_path / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    return {
        "image": str(output_path / tile_name), "metadata": str(output_path / "metadata.json"),
        "report": str(output_path / "report.json"), **report,
    }
