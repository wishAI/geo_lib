"""Whole explored-world inventory and overview rendering for offline QA."""

from __future__ import annotations

import json
import math
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .catalog import Catalog
from .evidence import RESOLUTION_KINDS
from .render import (
    LegacyTintResolver,
    TextureRenderer,
    _column_layers,
    _resolution,
    _visual,
)
from .world import AnvilWorld, WorldReadError, read_level_metadata


INVENTORY_SCHEMA = "geo.minecraft-surface-inventory/v1"
OVERVIEW_SCHEMA = "geo.minecraft-world-overview/v1"
MAX_PLAYABLE_BLOCK_COORDINATE = 30_000_000

# An intentionally narrow QA signal, not a building recognizer. It only ranks
# chunks for human visual inspection of known constructed vanilla materials.
BUILT_QA_IDS = {
    4, 5, 20, 22, 23, 25, 26, 27, 28, 29, 33, 35, 41, 42, 43, 44, 45, 46, 47,
    50, 53, 54, 55, 57, 58, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72,
    76, 84, 85, 91, 92, 93, 94, 96, 101, 102, 107, 108, 109, 112, 113, 114,
    116, 117, 118, 120, 123, 124, 125, 126, 128, 130, 131, 132, 133, 134, 135,
    136, 137, 138, 139, 140, 143, 145, 146, 147, 148, 149, 150, 151, 152, 154,
    155, 156, 157, 158,
}


def _bounds(positions: list[tuple[int, int]]) -> dict[str, int] | None:
    if not positions:
        return None
    min_cx = min(x for x, _ in positions)
    max_cx = max(x for x, _ in positions)
    min_cz = min(z for _, z in positions)
    max_cz = max(z for _, z in positions)
    return {
        "minChunkX": min_cx, "maxChunkX": max_cx, "minChunkZ": min_cz, "maxChunkZ": max_cz,
        "minX": min_cx * 16, "maxXExclusive": (max_cx + 1) * 16,
        "minZ": min_cz * 16, "maxZExclusive": (max_cz + 1) * 16,
    }


def _valid_chunk_position(position: tuple[int, int]) -> bool:
    """Minecraft 1.5.2 cannot expose chunks outside the ±30M world border."""

    chunk_x, chunk_z = position
    return abs(chunk_x * 16) <= MAX_PLAYABLE_BLOCK_COORDINATE and abs(chunk_z * 16) <= MAX_PLAYABLE_BLOCK_COORDINATE


def _dominant_connected_chunks(
    positions: list[tuple[int, int]],
) -> tuple[list[tuple[int, int]], list[dict[str, object]]]:
    """Keep the dominant 4-neighbour explored component for a useful overview.

    Old servers can contain isolated coordinate-jump chunks that are technically
    within the world border but make the inhabited world unreadably tiny.  They
    remain counted and are fully reported; only the overview canvas quarantines
    disconnected components.  Surface inventory always covers every chunk.
    """

    remaining = set(positions)
    components: list[list[tuple[int, int]]] = []
    while remaining:
        seed = remaining.pop()
        component = [seed]
        pending = [seed]
        while pending:
            x, z = pending.pop()
            for neighbor in ((x - 1, z), (x + 1, z), (x, z - 1), (x, z + 1)):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    pending.append(neighbor)
                    component.append(neighbor)
        components.append(component)
    components.sort(key=lambda item: (-len(item), min(item)))
    dominant = components[0] if components else []
    quarantined = [
        {"chunks": len(component), "bounds": _bounds(component)}
        for component in components[1:]
    ]
    return dominant, quarantined


def inventory_world(
    world: AnvilWorld, catalog: Catalog, progress: Callable[[int, int], None] | None = None,
    shard_count: int = 1, shard_index: int = 0,
) -> dict[str, object]:
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError("inventory shard index must be within the positive shard count")
    all_positions = list(world.chunk_positions())
    positions = all_positions[shard_index::shard_count]
    invalid_positions = [position for position in positions if not _valid_chunk_position(position)]
    valid_positions = [position for position in positions if _valid_chunk_position(position)]
    top_counts: Counter[str] = Counter()
    layer_counts: Counter[str] = Counter()
    geometry_counts: Counter[str] = Counter()
    biome_counts: Counter[str] = Counter()
    resolution_counts: dict[str, Counter[str]] = {kind: Counter() for kind in RESOLUTION_KINDS}
    skipped: Counter[str] = Counter()
    built_by_chunk: Counter[tuple[int, int]] = Counter()
    loaded = 0
    errors = []
    min_y = None
    max_y = None
    for index, (chunk_x, chunk_z) in enumerate(positions, 1):
        try:
            chunk = world.chunk(chunk_x, chunk_z)
        except WorldReadError as exc:
            errors.append({"chunkX": chunk_x, "chunkZ": chunk_z, "error": str(exc)})
            continue
        if chunk is None:
            continue
        loaded += 1
        for local_z in range(16):
            for local_x in range(16):
                layers = _column_layers(chunk, local_x, local_z, catalog, None, skipped)
                if not layers:
                    continue
                top = layers[0].block
                biome = chunk.biome_at(local_x, local_z, top.y)
                biome_counts[str(biome)] += 1
                top_counts[top.key] += 1
                min_y = top.y if min_y is None else min(min_y, top.y)
                max_y = top.y if max_y is None else max(max_y, top.y)
                if top.legacy_id in BUILT_QA_IDS:
                    built_by_chunk[(chunk_x, chunk_z)] += 1
                for surface in layers:
                    block = surface.block
                    layer_counts[block.key] += 1
                    geometry_counts[str(_visual(catalog, block).get("geometry", "cube"))] += 1
                    resolution_counts[_resolution(catalog, block.key, {})][block.key] += 1
        if progress and (index == 1 or index % 1000 == 0 or index == len(positions)):
            progress(index, len(positions))
    coverage = {
        kind: {
            "surfaceContributions": sum(counts.values()),
            "distinctBlockStates": len(counts),
            "blockCounts": dict(sorted(counts.items())),
        }
        for kind, counts in resolution_counts.items()
    }
    states = {}
    for key, count in sorted(layer_counts.items()):
        entry = catalog.entry(key) or {}
        states[key] = {
            "surfaceContributions": count,
            "topColumns": top_counts.get(key, 0),
            "name": entry.get("name"),
            "resolution": _resolution(catalog, key, {}),
            "confidence": entry.get("confidence", 0.0),
            "geometry": (entry.get("visual") or {}).get("geometry", "cube"),
            "reason": entry.get("reason", "no catalog entry"),
            "provenance": entry.get("provenance", []),
        }
    ranked = [
        {"chunkX": x, "chunkZ": z, "constructedSurfaceColumns": score}
        for (x, z), score in sorted(built_by_chunk.items(), key=lambda item: (-item[1], item[0][1], item[0][0]))[:100]
    ]
    return {
        "schema": INVENTORY_SCHEMA,
        "readOnly": True,
        "dimension": world.dimension.id,
        "bounds": _bounds(valid_positions),
        "chunks": {
            "allocated": len(positions), "loaded": loaded, "errors": errors,
            "worldAllocated": len(all_positions),
            "regionErrors": list(world.region_errors),
            "outsidePlayableWorldBorder": [
                {"chunkX": x, "chunkZ": z} for x, z in invalid_positions
            ],
        },
        "columns": sum(top_counts.values()),
        "height": {"min": min_y, "max": max_y},
        "biomeIds": dict(sorted(biome_counts.items())),
        "topSurfaceBlockCounts": dict(sorted(top_counts.items())),
        "surfaceStates": states,
        "geometry": dict(sorted(geometry_counts.items())),
        "coverage": coverage,
        "visuallyUnhandled": coverage["unknown"],
        "skippedAsAir": {"blocks": sum(skipped.values()), "blockCounts": dict(sorted(skipped.items()))},
        "qaConstructedDensity": {
            "purpose": "human crop selection only; not exposed as building recognition",
            "vanillaIds": sorted(BUILT_QA_IDS),
            "rankedChunks": ranked,
        },
        "shard": {"count": shard_count, "index": shard_index},
    }


def inventory_world_to_output(
    *, world_path: str | Path, dimension: str, catalog_path: str | Path, output: str | Path,
    progress: Callable[[int, int], None] | None = None,
    shard_count: int = 1, shard_index: int = 0,
) -> dict[str, object]:
    output_path = Path(output).expanduser()
    output_path.mkdir(parents=True, exist_ok=True)
    catalog = Catalog(catalog_path)
    with AnvilWorld(world_path, dimension) as world:
        report = inventory_world(world, catalog, progress, shard_count, shard_index)
        report["world"] = {"source": str(Path(world_path).expanduser().resolve()), **read_level_metadata(world.fs)}
    path = output_path / "surface_inventory.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return {"inventory": str(path), **report}


def merge_inventory_reports(reports: list[dict[str, object]]) -> dict[str, object]:
    """Merge a complete deterministic inventory shard set without rescanning."""

    if not reports:
        raise ValueError("at least one inventory report is required")
    shard_count = int(reports[0].get("shard", {}).get("count", 1))  # type: ignore[union-attr]
    indices = {int(report.get("shard", {}).get("index", -1)) for report in reports}  # type: ignore[union-attr]
    if len(reports) != shard_count or indices != set(range(shard_count)):
        raise ValueError(f"expected complete shard indices 0..{shard_count - 1}, got {sorted(indices)}")
    if len({str(report.get("dimension")) for report in reports}) != 1:
        raise ValueError("inventory shard dimensions differ")

    def merged_counter(path: tuple[str, ...]) -> dict[str, int]:
        total: Counter[str] = Counter()
        for report in reports:
            value: Any = report
            for key in path:
                value = value.get(key, {}) if isinstance(value, dict) else {}
            total.update({str(key): int(count) for key, count in value.items()})
        return dict(sorted(total.items()))

    valid_bounds = [report.get("bounds") for report in reports if isinstance(report.get("bounds"), dict)]
    bounds = {
        "minChunkX": min(item["minChunkX"] for item in valid_bounds),
        "maxChunkX": max(item["maxChunkX"] for item in valid_bounds),
        "minChunkZ": min(item["minChunkZ"] for item in valid_bounds),
        "maxChunkZ": max(item["maxChunkZ"] for item in valid_bounds),
        "minX": min(item["minX"] for item in valid_bounds),
        "maxXExclusive": max(item["maxXExclusive"] for item in valid_bounds),
        "minZ": min(item["minZ"] for item in valid_bounds),
        "maxZExclusive": max(item["maxZExclusive"] for item in valid_bounds),
    } if valid_bounds else None
    surface_states: dict[str, dict[str, object]] = {}
    for report in reports:
        for key, state in report.get("surfaceStates", {}).items():  # type: ignore[union-attr]
            if key not in surface_states:
                surface_states[key] = deepcopy(state)
                continue
            surface_states[key]["surfaceContributions"] = int(surface_states[key]["surfaceContributions"]) + int(state["surfaceContributions"])
            surface_states[key]["topColumns"] = int(surface_states[key]["topColumns"]) + int(state["topColumns"])
    coverage = {}
    for kind in RESOLUTION_KINDS:
        block_counts = merged_counter(("coverage", kind, "blockCounts"))
        coverage[kind] = {
            "surfaceContributions": sum(block_counts.values()),
            "distinctBlockStates": len(block_counts),
            "blockCounts": block_counts,
        }
    ranked = [
        item for report in reports
        for item in report.get("qaConstructedDensity", {}).get("rankedChunks", [])  # type: ignore[union-attr]
    ]
    ranked.sort(key=lambda item: (-int(item["constructedSurfaceColumns"]), int(item["chunkZ"]), int(item["chunkX"])))
    def unique_report_items(name: str) -> list[object]:
        unique: dict[str, object] = {}
        for report in reports:
            for item in report.get("chunks", {}).get(name, []):  # type: ignore[union-attr]
                unique.setdefault(json.dumps(item, ensure_ascii=False, sort_keys=True), item)
        return [unique[key] for key in sorted(unique)]

    errors = unique_report_items("errors")
    region_errors = unique_report_items("regionErrors")
    outside = unique_report_items("outsidePlayableWorldBorder")
    heights = [report.get("height", {}) for report in reports]
    minimums = [item.get("min") for item in heights if item.get("min") is not None]
    maximums = [item.get("max") for item in heights if item.get("max") is not None]
    skipped_counts = merged_counter(("skippedAsAir", "blockCounts"))
    return {
        "schema": INVENTORY_SCHEMA, "readOnly": True, "dimension": reports[0]["dimension"],
        "world": reports[0].get("world"), "bounds": bounds,
        "chunks": {
            "allocated": sum(int(report.get("chunks", {}).get("allocated", 0)) for report in reports),  # type: ignore[union-attr]
            "loaded": sum(int(report.get("chunks", {}).get("loaded", 0)) for report in reports),  # type: ignore[union-attr]
            "worldAllocated": reports[0].get("chunks", {}).get("worldAllocated"),  # type: ignore[union-attr]
            "errors": errors, "regionErrors": region_errors,
            "outsidePlayableWorldBorder": outside,
        },
        "columns": sum(int(report.get("columns", 0)) for report in reports),
        "height": {"min": min(minimums) if minimums else None, "max": max(maximums) if maximums else None},
        "biomeIds": merged_counter(("biomeIds",)),
        "topSurfaceBlockCounts": merged_counter(("topSurfaceBlockCounts",)),
        "surfaceStates": dict(sorted(surface_states.items())),
        "geometry": merged_counter(("geometry",)), "coverage": coverage,
        "visuallyUnhandled": coverage["unknown"],
        "skippedAsAir": {"blocks": sum(skipped_counts.values()), "blockCounts": skipped_counts},
        "qaConstructedDensity": {
            "purpose": "human crop selection only; not exposed as building recognition",
            "vanillaIds": reports[0].get("qaConstructedDensity", {}).get("vanillaIds", []),  # type: ignore[union-attr]
            "rankedChunks": ranked[:100],
        },
        "shardsMerged": {"count": shard_count, "indices": sorted(indices)},
    }


def _overview_scale(bounds: dict[str, int], max_size: int) -> tuple[int, int, int]:
    width = bounds["maxXExclusive"] - bounds["minX"]
    height = bounds["maxZExclusive"] - bounds["minZ"]
    blocks_per_pixel = 1
    while math.ceil(max(width, height) / blocks_per_pixel) > max_size:
        blocks_per_pixel *= 2
    return blocks_per_pixel, math.ceil(width / blocks_per_pixel), math.ceil(height / blocks_per_pixel)


def render_overview_to_output(
    *, world_path: str | Path, dimension: str, catalog_path: str | Path, output: str | Path, max_size: int = 4096,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, object]:
    if not 256 <= max_size <= 8192:
        raise ValueError("max_size must be between 256 and 8192 pixels")
    Image = __import__("PIL.Image", fromlist=["Image"])
    output_path = Path(output).expanduser()
    output_path.mkdir(parents=True, exist_ok=True)
    catalog = Catalog(catalog_path)
    with AnvilWorld(world_path, dimension) as world:
        all_positions = list(world.chunk_positions())
        invalid_positions = [position for position in all_positions if not _valid_chunk_position(position)]
        playable_positions = [position for position in all_positions if _valid_chunk_position(position)]
        positions, disconnected_components = _dominant_connected_chunks(playable_positions)
        bounds = _bounds(positions)
        if bounds is None:
            raise ValueError("World dimension has no allocated chunks")
        blocks_per_pixel, width, height = _overview_scale(bounds, max_size)
        canvas = Image.new("RGBA", (width, height), (14, 18, 24, 255))
        height_map = Image.new("F", (width, height), -10000.0)
        coarse: dict[tuple[int, int], list[float]] = {}
        textures = TextureRenderer(catalog, 1)
        tints = LegacyTintResolver(catalog, world)
        errors = []
        rendered = 0
        for chunk_x, chunk_z in positions:
            try:
                chunk = world.chunk(chunk_x, chunk_z)
            except WorldReadError as exc:
                errors.append({"chunkX": chunk_x, "chunkZ": chunk_z, "error": str(exc)})
                continue
            if chunk is None:
                continue
            block_x = chunk_x * 16 - bounds["minX"]
            block_z = chunk_z * 16 - bounds["minZ"]
            left, top = block_x // blocks_per_pixel, block_z // blocks_per_pixel
            right = math.ceil((block_x + 16) / blocks_per_pixel)
            bottom = math.ceil((block_z + 16) / blocks_per_pixel)

            def column_value(local_x: int, local_z: int) -> tuple[tuple[int, int, int], float] | None:
                world_x, world_z = chunk_x * 16 + local_x, chunk_z * 16 + local_z
                layers = _column_layers(chunk, local_x, local_z, catalog, None, Counter())
                if not layers:
                    return None
                surface_top = layers[0].block
                block_height = surface_top.y + float(_visual(catalog, surface_top).get("height", 1.0))
                dest = (14, 18, 24)
                for surface in reversed(layers):
                    block = surface.block
                    visual = _visual(catalog, block)
                    tint = tints.color(block, world_x, world_z)
                    red, green, blue, alpha = textures.overview_rgba(block, tint, visual, surface.fluid_depth)
                    inverse = 255 - alpha
                    dest = (
                        (red * alpha + dest[0] * inverse + 127) // 255,
                        (green * alpha + dest[1] * inverse + 127) // 255,
                        (blue * alpha + dest[2] * inverse + 127) // 255,
                    )
                return dest, block_height

            if blocks_per_pixel <= 16:
                step = blocks_per_pixel
                for start_z in range(0, 16, step):
                    for start_x in range(0, 16, step):
                        colors = []
                        column_heights = []
                        for local_z in range(start_z, min(start_z + step, 16)):
                            for local_x in range(start_x, min(start_x + step, 16)):
                                value = column_value(local_x, local_z)
                                if value is not None:
                                    colors.append(value[0])
                                    column_heights.append(value[1])
                        if not colors:
                            continue
                        pixel_x = left + start_x // step
                        pixel_z = top + start_z // step
                        canvas.putpixel((pixel_x, pixel_z), (
                            sum(item[0] for item in colors) // len(colors),
                            sum(item[1] for item in colors) // len(colors),
                            sum(item[2] for item in colors) // len(colors), 255,
                        ))
                        height_map.putpixel((pixel_x, pixel_z), sum(column_heights) / len(column_heights))
            else:
                for local_z in range(16):
                    for local_x in range(16):
                        column = column_value(local_x, local_z)
                        if column is None:
                            continue
                        world_x, world_z = chunk_x * 16 + local_x, chunk_z * 16 + local_z
                        pixel_x = (world_x - bounds["minX"]) // blocks_per_pixel
                        pixel_z = (world_z - bounds["minZ"]) // blocks_per_pixel
                        (red, green, blue), block_height = column
                        value = coarse.setdefault((pixel_x, pixel_z), [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                        value[0] += red
                        value[1] += green
                        value[2] += blue
                        value[3] += 1
                        value[4] += block_height
                        value[5] += 1
            rendered += 1
            if progress and (rendered == 1 or rendered % 1000 == 0 or rendered == len(positions)):
                progress(rendered, len(positions))
        if coarse:
            for (pixel_x, pixel_z), value in coarse.items():
                samples = value[3]
                canvas.putpixel((pixel_x, pixel_z), (
                    round(value[0] / samples), round(value[1] / samples), round(value[2] / samples), 255,
                ))
                if value[5]:
                    height_map.putpixel((pixel_x, pixel_z), value[4] / value[5])
        pixels = canvas.load()
        heights = height_map.load()
        min_factor, max_factor = 1.0, 1.0
        for z in range(height):
            for x in range(width):
                current = heights[x, z]
                if current < -1000:
                    continue
                north = heights[x, max(0, z - 1)]
                south = heights[x, min(height - 1, z + 1)]
                west = heights[max(0, x - 1), z]
                east = heights[min(width - 1, x + 1), z]
                values = [current if value < -1000 else value for value in (north, south, west, east)]
                factor = max(0.72, min(1.20, 1.0 + (values[0] - values[1] + values[2] - values[3]) * 0.025))
                min_factor, max_factor = min(min_factor, factor), max(max_factor, factor)
                red, green, blue, alpha = pixels[x, z]
                pixels[x, z] = (min(255, round(red * factor)), min(255, round(green * factor)),
                                min(255, round(blue * factor)), alpha)
        image_name = "world_overview.png"
        canvas.convert("RGB").save(output_path / image_name, "PNG", optimize=True)
        report = {
            "schema": OVERVIEW_SCHEMA,
            "generatedAt": datetime.now(timezone.utc).isoformat(),
            "readOnly": True,
            "world": {"source": str(Path(world_path).expanduser().resolve()), **read_level_metadata(world.fs)},
            "dimension": dimension,
            "bounds": bounds,
            "chunks": {
                "allocated": len(all_positions), "rendered": rendered, "errors": errors,
                "regionErrors": list(world.region_errors),
                "outsidePlayableWorldBorder": [
                    {"chunkX": x, "chunkZ": z} for x, z in invalid_positions
                ],
                "disconnectedComponentsQuarantinedFromCanvas": disconnected_components,
                "disconnectedChunksQuarantinedFromCanvas": len(playable_positions) - len(positions),
            },
            "image": {"path": image_name, "width": width, "height": height, "blocksPerPixel": blocks_per_pixel},
            "heightShade": {"minimumFactor": round(min_factor, 4), "maximumFactor": round(max_factor, 4)},
            "biomeTint": {"applied": dict(sorted(tints.applied.items())), "failures": dict(sorted(tints.failures.items()))},
            "textureFailures": textures.failures,
        }
    report_path = output_path / "overview_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return {"overview": str(output_path / image_name), "report": str(report_path), **report}
