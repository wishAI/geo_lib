"""Build and consume texture catalogs with source provenance.

Catalog generation never copies texture assets. It records hashes and logical members
from the exact resource packs, mod JARs, Forge configs, and CustomStuff definitions.
"""

from __future__ import annotations

import hashlib
import json
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable, Iterator


CATALOG_SCHEMA = "geo.minecraft-block-catalog/v1"
CONFIG_ID_RE = re.compile(r"^\s*I:([^=]+)=(-?\d+)\s*$")
ASSIGNMENT_RE = re.compile(r"^\s*([A-Za-z0-9_.-]+)\s*[=:]\s*['\"]?([^'\"#;]+)")
CUSTOM_ID_RE = re.compile(r"config\.getBlockId\(\s*['\"]([^'\"]+)['\"]\s*\)")
CUSTOM_NAME_RE = re.compile(r"^\s*name\s*=\s*['\"]([^'\"]+)['\"]", re.MULTILINE)
CUSTOM_DISPLAY_RE = re.compile(r"displayName\[(\d+)]\s*=\s*['\"]([^'\"]*)['\"]")
CUSTOM_TOP_RE = re.compile(r"textureFileYP\[(\d+)]\s*=\s*['\"]([^'\"]+)['\"]")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for piece in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(piece)
    return digest.hexdigest()


def sha256_tree(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        relative = item.relative_to(path).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(item)))
    return digest.hexdigest()


@dataclass(frozen=True)
class AssetSource:
    path: Path
    kind: str
    sha256: str

    @classmethod
    def create(cls, value: str | Path) -> "AssetSource":
        path = Path(value).expanduser().resolve()
        if path.is_dir():
            return cls(path, "directory", sha256_tree(path))
        if path.is_file() and path.suffix.lower() in {".jar", ".zip"}:
            return cls(path, "zip", sha256_file(path))
        raise ValueError(f"Asset source must be a directory, JAR, or ZIP: {path}")

    def names(self) -> Iterator[str]:
        if self.kind == "directory":
            for path in self.path.rglob("*"):
                if path.is_file():
                    yield path.relative_to(self.path).as_posix()
        else:
            with zipfile.ZipFile(self.path) as archive:
                yield from (item.filename for item in archive.infolist() if not item.is_dir())

    def read(self, member: str) -> bytes:
        member = PurePosixPath(member).as_posix().lstrip("/")
        if ".." in PurePosixPath(member).parts:
            raise ValueError(f"Unsafe asset member: {member}")
        if self.kind == "directory":
            return (self.path / member).read_bytes()
        with zipfile.ZipFile(self.path) as archive:
            return archive.read(member)

    def has(self, member: str) -> bool:
        if self.kind == "directory":
            return (self.path / member).is_file()
        with zipfile.ZipFile(self.path) as archive:
            try:
                archive.getinfo(member)
                return True
            except KeyError:
                return False

    def record(self) -> dict[str, object]:
        return {"path": str(self.path), "kind": self.kind, "sha256": self.sha256}


class SourceSet:
    """Resources in highest-to-lowest precedence, matching pack semantics."""

    def __init__(self, sources: Iterable[AssetSource]):
        self.sources = list(sources)

    def locate(self, member: str) -> tuple[int, bytes] | None:
        for index, source in enumerate(self.sources):
            if source.has(member):
                return index, source.read(member)
        return None

    def effective_names(self, suffix: str) -> dict[str, int]:
        result: dict[str, int] = {}
        for index in range(len(self.sources) - 1, -1, -1):
            source = self.sources[index]
            for name in source.names():
                if name.endswith(suffix):
                    result[name] = index
        return result


def _canonical_state(name: str, properties: str) -> str:
    if not properties:
        return name
    pairs = []
    for pair in properties.split(","):
        key, separator, value = pair.partition("=")
        if not separator:
            return f"{name}[{properties}]"
        pairs.append((key, value))
    return f"{name}[{','.join(f'{key}={value}' for key, value in sorted(pairs))}]"


def _model_path(reference: str, default_namespace: str) -> str:
    namespace, separator, name = reference.partition(":")
    if not separator:
        namespace, name = default_namespace, namespace
    return f"assets/{namespace}/models/{name}.json"


def _texture_path(reference: str, default_namespace: str) -> str:
    namespace, separator, name = reference.partition(":")
    if not separator:
        namespace, name = default_namespace, namespace
    return f"assets/{namespace}/textures/{name}.png"


def _merged_model(resources: SourceSet, reference: str, namespace: str, seen: set[str] | None = None) -> tuple[dict, list[str]]:
    path = _model_path(reference, namespace)
    seen = set() if seen is None else seen
    if path in seen:
        raise ValueError(f"Cyclic model parent: {path}")
    seen.add(path)
    found = resources.locate(path)
    if found is None:
        raise ValueError(f"Missing model {path}")
    _, raw = found
    model = json.loads(raw)
    provenance = [path]
    parent = model.get("parent")
    if parent:
        base, parent_provenance = _merged_model(resources, parent, namespace, seen)
        merged = dict(base)
        merged_textures = dict(base.get("textures", {}))
        merged_textures.update(model.get("textures", {}))
        merged.update(model)
        merged["textures"] = merged_textures
        model = merged
        provenance = parent_provenance + provenance
    return model, provenance


def _top_texture(resources: SourceSet, model_ref: str, namespace: str) -> tuple[dict[str, object], list[str]]:
    model, provenance = _merged_model(resources, model_ref, namespace)
    textures = model.get("textures", {})
    candidates: set[str] = set()
    for element in model.get("elements", []):
        face = element.get("faces", {}).get("up")
        if isinstance(face, dict) and isinstance(face.get("texture"), str):
            candidates.add(face["texture"])
    if not candidates:
        for key in ("top", "up", "all", "end", "particle"):
            if key in textures:
                candidates.add("#" + key)
                break
    resolved: set[str] = set()
    for candidate in candidates:
        for _ in range(32):
            if not candidate.startswith("#"):
                break
            candidate = textures.get(candidate[1:], "")
        if candidate:
            resolved.add(candidate)
    if len(resolved) != 1:
        raise ValueError(f"Model has {len(resolved)} unambiguous top textures")
    texture = _texture_path(next(iter(resolved)), namespace)
    located = resources.locate(texture)
    if located is None:
        raise ValueError(f"Missing texture {texture}")
    source_index, _ = located
    return {"source": source_index, "member": texture}, provenance + [texture]


def build_modern_catalog(asset_paths: Iterable[str | Path]) -> dict[str, object]:
    sources = [AssetSource.create(path) for path in asset_paths]
    resources = SourceSet(sources)
    entries: dict[str, dict[str, object]] = {}
    for blockstate_path, source_index in sorted(resources.effective_names(".json").items()):
        parts = PurePosixPath(blockstate_path).parts
        if len(parts) < 4 or parts[0] != "assets" or parts[2] != "blockstates":
            continue
        namespace = parts[1]
        name = PurePosixPath(parts[-1]).stem
        block_name = f"{namespace}:{name}"
        try:
            raw = resources.sources[source_index].read(blockstate_path)
            state = json.loads(raw)
        except (ValueError, KeyError, json.JSONDecodeError) as exc:
            entries[block_name] = {"status": "unknown", "reason": f"invalid blockstate: {exc}", "provenance": [blockstate_path]}
            continue
        variants = state.get("variants")
        if not isinstance(variants, dict):
            entries[block_name] = {"status": "unknown", "reason": "multipart blockstate requires geometry-aware rendering", "provenance": [blockstate_path]}
            continue
        for properties, variant in variants.items():
            key = _canonical_state(block_name, properties)
            choices = variant if isinstance(variant, list) else [variant]
            try:
                resolved = [_top_texture(resources, item["model"], namespace) for item in choices if isinstance(item, dict) and "model" in item]
                textures = {json.dumps(item[0], sort_keys=True) for item in resolved}
                if not resolved or len(textures) != 1:
                    raise ValueError("weighted variants do not share one top texture")
                texture, model_provenance = resolved[0]
                entries[key] = {
                    "status": "resolved",
                    "name": block_name,
                    "texture": texture,
                    "provenance": [blockstate_path] + model_provenance,
                }
            except (ValueError, KeyError, json.JSONDecodeError) as exc:
                entries[key] = {"status": "unknown", "name": block_name, "reason": str(exc), "provenance": [blockstate_path]}
    return {"schema": CATALOG_SCHEMA, "edition": "modern", "sources": [item.record() for item in sources], "entries": entries}


# Canonical terrain.png coordinates for stable, non-animated top faces in Java 1.5.2.
# Pack pixels remain external; these entries only identify the exact atlas cells.
VANILLA_152_TOP: dict[tuple[int, int], tuple[str, int]] = {
    (1, 0): ("minecraft:stone", 1),
    (2, 0): ("minecraft:grass", 0),
    (3, 0): ("minecraft:dirt", 2),
    (4, 0): ("minecraft:cobblestone", 16),
    (5, 0): ("minecraft:oak_planks", 4),
    (7, 0): ("minecraft:bedrock", 17),
    (12, 0): ("minecraft:sand", 18),
    (13, 0): ("minecraft:gravel", 19),
    (14, 0): ("minecraft:gold_ore", 32),
    (15, 0): ("minecraft:iron_ore", 33),
    (16, 0): ("minecraft:coal_ore", 34),
    (17, 0): ("minecraft:oak_log", 21),
    (20, 0): ("minecraft:glass", 49),
    (22, 0): ("minecraft:lapis_block", 144),
    (24, 0): ("minecraft:sandstone", 176),
    (41, 0): ("minecraft:gold_block", 23),
    (42, 0): ("minecraft:iron_block", 22),
    (45, 0): ("minecraft:bricks", 7),
    (48, 0): ("minecraft:mossy_cobblestone", 36),
    (49, 0): ("minecraft:obsidian", 37),
    (56, 0): ("minecraft:diamond_ore", 50),
    (57, 0): ("minecraft:diamond_block", 24),
    (73, 0): ("minecraft:redstone_ore", 51),
    (80, 0): ("minecraft:snow_block", 66),
    (82, 0): ("minecraft:clay", 72),
    (87, 0): ("minecraft:netherrack", 103),
    (88, 0): ("minecraft:soul_sand", 104),
    (89, 0): ("minecraft:glowstone", 105),
    (112, 0): ("minecraft:nether_brick", 224),
    (121, 0): ("minecraft:end_stone", 175),
    (129, 0): ("minecraft:emerald_ore", 171),
    (133, 0): ("minecraft:emerald_block", 25),
}

VANILLA_152_FILES: dict[tuple[int, int], tuple[str, str]] = {
    (1, 0): ("minecraft:stone", "textures/blocks/stone.png"),
    (2, 0): ("minecraft:grass", "textures/blocks/grass_top.png"),
    (3, 0): ("minecraft:dirt", "textures/blocks/dirt.png"),
    (4, 0): ("minecraft:cobblestone", "textures/blocks/stonebrick.png"),
    (5, 0): ("minecraft:oak_planks", "textures/blocks/wood.png"),
    (7, 0): ("minecraft:bedrock", "textures/blocks/bedrock.png"),
    (12, 0): ("minecraft:sand", "textures/blocks/sand.png"),
    (13, 0): ("minecraft:gravel", "textures/blocks/gravel.png"),
    (14, 0): ("minecraft:gold_ore", "textures/blocks/oreGold.png"),
    (15, 0): ("minecraft:iron_ore", "textures/blocks/oreIron.png"),
    (16, 0): ("minecraft:coal_ore", "textures/blocks/oreCoal.png"),
    (17, 0): ("minecraft:oak_log", "textures/blocks/tree_top.png"),
    (20, 0): ("minecraft:glass", "textures/blocks/glass.png"),
    (22, 0): ("minecraft:lapis_block", "textures/blocks/blockLapis.png"),
    (24, 0): ("minecraft:sandstone", "textures/blocks/sandstone_top.png"),
    (41, 0): ("minecraft:gold_block", "textures/blocks/blockGold.png"),
    (42, 0): ("minecraft:iron_block", "textures/blocks/blockIron.png"),
    (45, 0): ("minecraft:bricks", "textures/blocks/brick.png"),
    (48, 0): ("minecraft:mossy_cobblestone", "textures/blocks/stoneMoss.png"),
    (49, 0): ("minecraft:obsidian", "textures/blocks/obsidian.png"),
    (56, 0): ("minecraft:diamond_ore", "textures/blocks/oreDiamond.png"),
    (57, 0): ("minecraft:diamond_block", "textures/blocks/blockDiamond.png"),
    (73, 0): ("minecraft:redstone_ore", "textures/blocks/oreRedstone.png"),
    (80, 0): ("minecraft:snow_block", "textures/blocks/snow.png"),
    (82, 0): ("minecraft:clay", "textures/blocks/clay.png"),
    (87, 0): ("minecraft:netherrack", "textures/blocks/hellrock.png"),
    (88, 0): ("minecraft:soul_sand", "textures/blocks/hellsand.png"),
    (89, 0): ("minecraft:glowstone", "textures/blocks/lightgem.png"),
    (112, 0): ("minecraft:nether_brick", "textures/blocks/netherBrick.png"),
    (121, 0): ("minecraft:end_stone", "textures/blocks/whiteStone.png"),
    (129, 0): ("minecraft:emerald_ore", "textures/blocks/oreEmerald.png"),
    (133, 0): ("minecraft:emerald_block", "textures/blocks/blockEmerald.png"),
}


def _find_terrain_source(sources: list[AssetSource]) -> tuple[int, str] | None:
    candidates = ("terrain.png", "textures/terrain.png")
    for index, source in enumerate(sources):
        names = set(source.names())
        for candidate in candidates:
            if candidate in names:
                return index, candidate
        suffix_matches = [name for name in names if name.lower().endswith("/terrain.png")]
        if len(suffix_matches) == 1:
            return index, suffix_matches[0]
    return None


def _config_assignments(root: Path) -> list[dict[str, object]]:
    records = []
    for path in sorted(root.rglob("*.cfg")):
        category = ""
        for line_number, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
            stripped = line.strip()
            if stripped.endswith("{"):
                category = stripped[:-1].strip()
            elif stripped == "}":
                category = ""
            match = CONFIG_ID_RE.match(line)
            if match and 0 < int(match.group(2)) <= 4095 and (not category or category.lower() == "block"):
                records.append({
                    "id": int(match.group(2)),
                    "key": match.group(1).strip(),
                    "category": category,
                    "file": str(path.resolve()),
                    "line": line_number,
                    "sha256": sha256_file(path),
                })
    return records


def _decoded_text(path: Path) -> str:
    raw = path.read_bytes()
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("gb18030", errors="replace")


def _customstuff_definitions(root: Path, assignments: list[dict[str, object]]) -> list[dict[str, object]]:
    records = []
    config_ids: dict[str, set[int]] = {}
    for item in assignments:
        config_ids.setdefault(str(item["key"]), set()).add(int(item["id"]))
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".block", ".properties", ".cfg", ".js"}:
            continue
        if "customstuff" not in path.as_posix().lower() and path.suffix.lower() != ".block":
            continue
        text = _decoded_text(path)
        property_match = CUSTOM_ID_RE.search(text)
        if property_match:
            ids = config_ids.get(property_match.group(1), set())
            if len(ids) != 1:
                continue
            block_id = next(iter(ids))
            internal_name = (CUSTOM_NAME_RE.search(text).group(1) if CUSTOM_NAME_RE.search(text) else path.stem)
            display_names = {int(meta): value for meta, value in CUSTOM_DISPLAY_RE.findall(text)}
            top_textures = {int(meta): value for meta, value in CUSTOM_TOP_RE.findall(text)}
            for metadata in sorted(set(display_names) | set(top_textures)):
                records.append({
                    "id": block_id,
                    "metadata": metadata,
                    "name": display_names.get(metadata) or f"{internal_name}:{metadata}",
                    "texture": top_textures.get(metadata),
                    "configProperty": property_match.group(1),
                    "file": str(path.resolve()),
                    "sha256": sha256_file(path),
                })
            continue
        values: dict[str, str] = {}
        for line in text.splitlines():
            match = ASSIGNMENT_RE.match(line)
            if match:
                values[match.group(1).lower()] = match.group(2).strip()
        raw_id = next((values[key] for key in ("id", "blockid", "block.id") if key in values), None)
        if raw_id is None or not raw_id.lstrip("-").isdigit():
            continue
        raw_meta = next((values[key] for key in ("metadata", "meta", "damage") if key in values), "*")
        texture = next((values[key] for key in ("texture.top", "toptexture", "texture") if key in values), None)
        records.append({
            "id": int(raw_id),
            "metadata": int(raw_meta) if raw_meta.isdigit() else "*",
            "name": values.get("name", path.stem),
            "texture": texture,
            "file": str(path.resolve()),
            "sha256": sha256_file(path),
        })
    return records


def _locate_loose_texture(sources: list[AssetSource], value: str) -> tuple[int, str] | None:
    normalized = value.replace("\\", "/").lstrip("/")
    stems = {PurePosixPath(normalized).stem.lower(), normalized.lower()}
    for index, source in enumerate(sources):
        matches: list[tuple[int, str]] = []
        for name in source.names():
            if name.lower().endswith(".png") and (name.lower() in stems or PurePosixPath(name).stem.lower() in stems):
                matches.append((index, name))
        if matches:
            return matches[0] if len(matches) == 1 else None
    return None


def build_legacy_catalog(bc3_root: str | Path, asset_paths: Iterable[str | Path]) -> dict[str, object]:
    root = Path(bc3_root).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"bc3 root is not a directory: {root}")
    sources = [AssetSource.create(path) for path in asset_paths]
    terrain = _find_terrain_source(sources)
    entries: dict[str, dict[str, object]] = {}
    for (block_id, metadata), (name, member) in VANILLA_152_FILES.items():
        located = next(((index, member) for index, source in enumerate(sources) if source.has(member)), None)
        if located:
            source_index, resolved_member = located
            entries[f"legacy:{block_id}:{metadata}"] = {
                "status": "resolved",
                "name": name,
                "texture": {"source": source_index, "member": resolved_member},
                "provenance": ["Minecraft Java 1.5.2 canonical numeric ID", resolved_member],
            }
    if terrain:
        source_index, member = terrain
        for (block_id, metadata), (name, atlas_index) in VANILLA_152_TOP.items():
            if f"legacy:{block_id}:{metadata}" in entries:
                continue
            entries[f"legacy:{block_id}:{metadata}"] = {
                "status": "resolved",
                "name": name,
                "texture": {"source": source_index, "member": member, "atlas": {"columns": 16, "index": atlas_index}},
                "provenance": ["Minecraft Java 1.5.2 canonical numeric ID", member],
            }
    assignments = _config_assignments(root)
    by_id: dict[int, list[dict[str, object]]] = {}
    for record in assignments:
        by_id.setdefault(int(record["id"]), []).append(record)
    for block_id, records in sorted(by_id.items()):
        key = f"legacy:{block_id}:*"
        entries[key] = {
            "status": "unknown",
            "name": records[0]["key"] if len(records) == 1 else f"configured_id_{block_id}",
            "reason": "Forge config proves the numeric ID but not a metadata-specific top texture",
            "provenance": records,
        }
    for definition in _customstuff_definitions(root, assignments):
        metadata = definition["metadata"]
        key = f"legacy:{definition['id']}:{metadata}"
        located = _locate_loose_texture(sources, str(definition["texture"])) if definition.get("texture") else None
        if located:
            source_index, member = located
            entries[key] = {
                "status": "resolved",
                "name": definition["name"],
                "texture": {"source": source_index, "member": member},
                "provenance": [definition],
            }
        else:
            entries[key] = {
                "status": "unknown",
                "name": definition["name"],
                "reason": "CustomStuff definition has no uniquely resolvable top texture",
                "provenance": [definition],
            }
    return {
        "schema": CATALOG_SCHEMA,
        "edition": "legacy-1.5.2-forge",
        "sources": [item.record() for item in sources],
        "bc3": {"root": str(root), "treeSha256": sha256_tree(root)},
        "entries": entries,
    }


def write_catalog(catalog: dict[str, object], output: str | Path) -> None:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(catalog, indent=2, ensure_ascii=False, sort_keys=True) + "\n")


class Catalog:
    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser().resolve()
        self.data = json.loads(self.path.read_text())
        if self.data.get("schema") != CATALOG_SCHEMA:
            raise ValueError(f"Unsupported catalog schema in {self.path}")
        self.sources = [AssetSource.create(item["path"]) for item in self.data.get("sources", [])]
        for expected, actual in zip(self.data.get("sources", []), self.sources):
            if expected.get("sha256") != actual.sha256:
                raise ValueError(f"Asset changed since catalog creation: {actual.path}")
        self.entries = self.data.get("entries", {})

    def entry(self, key: str) -> dict[str, object] | None:
        if key in self.entries:
            return self.entries[key]
        if key.startswith("legacy:"):
            block_id = key.split(":")[1]
            return self.entries.get(f"legacy:{block_id}:*")
        base = key.partition("[")[0]
        direct = self.entries.get(base)
        if direct is not None:
            return direct
        if "[" not in key:
            return None
        actual = dict(pair.split("=", 1) for pair in key.removesuffix("]").partition("[")[2].split(","))
        matches = []
        for candidate, entry in self.entries.items():
            if not candidate.startswith(base + "["):
                continue
            required = dict(pair.split("=", 1) for pair in candidate.removesuffix("]").partition("[")[2].split(","))
            if all(actual.get(name) == value for name, value in required.items()):
                matches.append(entry)
        fingerprints = {json.dumps(item.get("texture"), sort_keys=True) for item in matches if item.get("status") == "resolved"}
        return matches[0] if len(matches) == 1 or (matches and all(item.get("status") == "resolved" for item in matches) and len(fingerprints) == 1) else None

    def texture_bytes(self, entry: dict[str, object]) -> bytes:
        texture = entry["texture"]
        return self.sources[int(texture["source"])].read(str(texture["member"]))
