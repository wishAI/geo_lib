"""Command line entry point for inventory, cataloging, rendering, and serving."""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

from .catalog import build_legacy_catalog, build_modern_catalog, cover_visible_legacy_catalog, write_catalog
from .overview import inventory_world_to_output, merge_inventory_reports, render_overview_to_output
from .render import TileRequest, render_tile_to_output
from .server import TileApplication, WorldSpec, serve
from .world import discover_dimensions, open_world, read_level_metadata


def _layer(value: str) -> int | None:
    if value == "surface":
        return None
    try:
        return int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("layer must be 'surface' or an integer Y coordinate") from exc


def _labeled(values: list[str], label: str) -> dict[str, Path]:
    result = {}
    for value in values:
        name, separator, path = value.partition("=")
        if not separator or not name or not path:
            raise ValueError(f"{label} must use LABEL=PATH: {value}")
        if not name.replace("_", "").replace("-", "").replace(".", "").isalnum():
            raise ValueError(f"Unsafe {label} label: {name}")
        if name in result:
            raise ValueError(f"Duplicate {label} label: {name}")
        result[name] = Path(path).expanduser()
    return result


def _archive_inventory(path: Path) -> dict[str, object]:
    with zipfile.ZipFile(path) as archive:
        members = [
            {"name": item.filename, "size": item.file_size, "compressedSize": item.compress_size, "crc32": f"{item.CRC:08x}"}
            for item in archive.infolist()
            if not item.is_dir()
        ]
    return {"archive": str(path.resolve()), "readOnly": True, "members": members}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Read-only Minecraft Anvil top-map renderer")
    sub = parser.add_subparsers(dest="command", required=True)

    archive = sub.add_parser("inspect-archive", help="List a backup ZIP without extracting it")
    archive.add_argument("--archive", required=True)
    archive.add_argument("--output")

    inspect = sub.add_parser("inspect-world", help="Report level metadata and dimensions without changing the world")
    inspect.add_argument("--world", required=True)
    inspect.add_argument("--output")

    modern = sub.add_parser("catalog-modern", help="Resolve modern blockstate/model top textures from exact assets")
    modern.add_argument("--asset", action="append", required=True, help="Highest-precedence resource pack first; repeat for JARs")
    modern.add_argument("--output", required=True)

    legacy = sub.add_parser("catalog-legacy", help="Resolve 1.5.2 IDs from exact bc3 configs and texture assets")
    legacy.add_argument("--bc3-root", required=True, help="Read-only extracted bc3 directory on UGREEN or TK2")
    legacy.add_argument("--asset", action="append", required=True, help="32x pack first, then CustomStuff directory and exact mod JARs")
    legacy.add_argument("--output", required=True)

    cover = sub.add_parser(
        "catalog-cover-visible",
        help="Explicitly infer only unresolved legacy states proven visible by a completed inventory",
    )
    cover.add_argument("--catalog", required=True)
    cover.add_argument("--inventory", required=True)
    cover.add_argument("--output", required=True)

    render = sub.add_parser("render-tile", help="Render one fixed 256x256-block tile for validation")
    render.add_argument("--world", required=True)
    render.add_argument("--catalog", required=True)
    render.add_argument("--dimension", default="minecraft:overworld")
    render.add_argument("--layer", type=_layer, default=None)
    render.add_argument("--tile-x", type=int, default=0)
    render.add_argument("--tile-z", type=int, default=0)
    render.add_argument("--pixels-per-block", type=int, default=1)
    render.add_argument("--output", required=True)

    inventory = sub.add_parser("inventory-world", help="Inventory every explored top-surface state for offline coverage QA")
    inventory.add_argument("--world", required=True)
    inventory.add_argument("--catalog", required=True)
    inventory.add_argument("--dimension", default="minecraft:overworld")
    inventory.add_argument("--output", required=True)
    inventory.add_argument("--shards", type=int, default=1)
    inventory.add_argument("--shard-index", type=int, default=0)

    merge = sub.add_parser("merge-inventories", help="Merge a complete deterministic set of inventory shards")
    merge.add_argument("--input", action="append", required=True)
    merge.add_argument("--output", required=True)

    overview = sub.add_parser("render-overview", help="Render the complete allocated chunk bounds at a bounded image size")
    overview.add_argument("--world", required=True)
    overview.add_argument("--catalog", required=True)
    overview.add_argument("--dimension", default="minecraft:overworld")
    overview.add_argument("--max-size", type=int, default=4096)
    overview.add_argument("--output", required=True)

    server = sub.add_parser("serve", help="Open the world/dimension/layer pan-and-zoom tile viewer")
    server.add_argument("--world", action="append", required=True, metavar="LABEL=PATH")
    server.add_argument("--catalog", action="append", required=True, metavar="LABEL=PATH")
    server.add_argument("--cache", required=True)
    server.add_argument("--host", default="127.0.0.1")
    server.add_argument("--port", type=int, default=8782)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "inspect-archive":
        payload = _archive_inventory(Path(args.archive).expanduser())
        encoded = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        if args.output:
            output = Path(args.output).expanduser()
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(encoded)
        else:
            print(encoded, end="")
        return 0
    if args.command == "inspect-world":
        fs = open_world(args.world)
        payload = {
            "world": {"source": str(Path(args.world).expanduser().resolve()), **read_level_metadata(fs)},
            "dimensions": [item.__dict__ for item in discover_dimensions(fs)], "readOnly": True,
        }
        encoded = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        if args.output:
            output = Path(args.output).expanduser()
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(encoded)
        else:
            print(encoded, end="")
        return 0
    if args.command == "catalog-modern":
        write_catalog(build_modern_catalog(args.asset), args.output)
        return 0
    if args.command == "catalog-legacy":
        write_catalog(build_legacy_catalog(args.bc3_root, args.asset), args.output)
        return 0
    if args.command == "catalog-cover-visible":
        catalog = json.loads(Path(args.catalog).expanduser().read_text())
        inventory = json.loads(Path(args.inventory).expanduser().read_text())
        write_catalog(cover_visible_legacy_catalog(catalog, inventory), args.output)
        return 0
    if args.command == "render-tile":
        request = TileRequest(args.tile_x, args.tile_z, args.layer, args.pixels_per_block)
        print(json.dumps(render_tile_to_output(world_path=args.world, dimension=args.dimension, catalog_path=args.catalog, request=request, output=args.output), ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.command == "inventory-world":
        def progress(done: int, total: int) -> None:
            print(f"inventory chunks: {done}/{total}", file=sys.stderr, flush=True)

        print(json.dumps(inventory_world_to_output(
            world_path=args.world, dimension=args.dimension, catalog_path=args.catalog, output=args.output,
            progress=progress, shard_count=args.shards, shard_index=args.shard_index,
        ), ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.command == "merge-inventories":
        reports = [json.loads(Path(path).expanduser().read_text()) for path in args.input]
        merged = merge_inventory_reports(reports)
        output = Path(args.output).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(merged, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"inventory": str(output), **merged}, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.command == "render-overview":
        def overview_progress(done: int, total: int) -> None:
            print(f"overview chunks: {done}/{total}", file=sys.stderr, flush=True)

        print(json.dumps(render_overview_to_output(
            world_path=args.world, dimension=args.dimension, catalog_path=args.catalog,
            output=args.output, max_size=args.max_size, progress=overview_progress,
        ), ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.command == "serve":
        worlds = _labeled(args.world, "world")
        catalogs = _labeled(args.catalog, "catalog")
        if set(worlds) != set(catalogs):
            raise ValueError("--world and --catalog labels must match exactly")
        specs = [WorldSpec(name, worlds[name], catalogs[name]) for name in worlds]
        serve(TileApplication(specs, Path(args.cache).expanduser()), args.host, args.port)
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
