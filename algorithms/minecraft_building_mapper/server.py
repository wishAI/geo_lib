"""Local read-only HTTP tile server and pan/zoom viewer."""

from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .catalog import Catalog
from .render import TileRequest, render_tile
from .world import AnvilWorld, discover_dimensions, open_world, read_level_metadata


WEB_ROOT = Path(__file__).with_name("web")


@dataclass(frozen=True)
class WorldSpec:
    id: str
    path: Path
    catalog_path: Path


class TileApplication:
    def __init__(self, worlds: list[WorldSpec], cache_dir: Path):
        if not worlds:
            raise ValueError("At least one world is required")
        self.worlds = {item.id: item for item in worlds}
        self.catalogs = {item.id: Catalog(item.catalog_path) for item in worlds}
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._locks: dict[Path, threading.Lock] = {}
        self._locks_guard = threading.Lock()

    def inventory(self) -> list[dict[str, object]]:
        result = []
        for item in self.worlds.values():
            try:
                fs = open_world(item.path)
                result.append({
                    "id": item.id,
                    "metadata": read_level_metadata(fs),
                    "dimensions": [dimension.id for dimension in discover_dimensions(fs)],
                })
            except Exception as exc:
                result.append({"id": item.id, "error": str(exc), "dimensions": []})
        return result

    def _paths(self, world_id: str, dimension: str, request: TileRequest) -> tuple[Path, Path]:
        safe_dimension = re.sub(r"[^A-Za-z0-9_.-]+", "_", dimension)
        base = self.cache_dir / world_id / safe_dimension / request.layer_label / f"ppb_{request.pixels_per_block}"
        stem = f"tile_{request.tile_x}_{request.tile_z}"
        return base / f"{stem}.png", base / f"{stem}.report.json"

    def tile(self, world_id: str, dimension: str, request: TileRequest) -> tuple[Path, Path]:
        if world_id not in self.worlds:
            raise ValueError(f"Unknown world selector: {world_id}")
        image_path, report_path = self._paths(world_id, dimension, request)
        with self._locks_guard:
            lock = self._locks.setdefault(image_path, threading.Lock())
        with lock:
            if image_path.is_file() and report_path.is_file():
                return image_path, report_path
            image_path.parent.mkdir(parents=True, exist_ok=True)
            spec = self.worlds[world_id]
            with AnvilWorld(spec.path, dimension) as world:
                image, report = render_tile(world, self.catalogs[world_id], request)
            image.save(image_path, "PNG", optimize=True)
            report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        return image_path, report_path


def _query_request(query: dict[str, list[str]]) -> tuple[str, str, TileRequest]:
    def one(name: str, default: str | None = None) -> str:
        values = query.get(name)
        if not values:
            if default is None:
                raise ValueError(f"Missing query parameter: {name}")
            return default
        return values[0]

    layer_value = one("layer", "surface")
    layer = None if layer_value == "surface" else int(layer_value)
    request = TileRequest(int(one("x")), int(one("z")), layer, int(one("ppb", "1")))
    return one("world"), one("dimension", "minecraft:overworld"), request


def handler_for(app: TileApplication):
    class Handler(BaseHTTPRequestHandler):
        def _bytes(self, payload: bytes, content_type: str, status: int = 200) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store" if content_type == "application/json" else "public, max-age=31536000, immutable")
            self.end_headers()
            self.wfile.write(payload)

        def _json(self, value: object, status: int = 200) -> None:
            self._bytes(json.dumps(value, ensure_ascii=False).encode(), "application/json", status)

        def do_GET(self) -> None:  # noqa: N802 - stdlib callback name
            parsed = urlparse(self.path)
            try:
                if parsed.path == "/":
                    self._bytes((WEB_ROOT / "index.html").read_bytes(), "text/html; charset=utf-8")
                    return
                if parsed.path == "/api/worlds":
                    self._json({"worlds": app.inventory(), "tileBlocks": 256})
                    return
                if parsed.path in {"/api/tile.png", "/api/report"}:
                    world_id, dimension, request = _query_request(parse_qs(parsed.query))
                    image_path, report_path = app.tile(world_id, dimension, request)
                    if parsed.path == "/api/report":
                        self._bytes(report_path.read_bytes(), "application/json")
                    else:
                        self._bytes(image_path.read_bytes(), "image/png")
                    return
                self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
            except (ValueError, OSError, RuntimeError) as exc:
                self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)

        def log_message(self, format: str, *args: object) -> None:
            return

    return Handler


def serve(app: TileApplication, host: str, port: int) -> None:
    server = ThreadingHTTPServer((host, port), handler_for(app))
    print(f"Minecraft map viewer: http://{host}:{server.server_port}", flush=True)
    server.serve_forever()
