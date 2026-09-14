"""Build and consume texture catalogs with source provenance.

Catalog generation never copies texture assets. It records hashes and logical members
from the exact resource packs, mod JARs, Forge configs, and CustomStuff definitions.
"""

from __future__ import annotations

import hashlib
import json
import re
import zipfile
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Iterator

from .evidence import resolution_entry, validate_resolution_entry
from .legacy_resolver import apply_exact_legacy_resolvers
from .legacy_visuals import (
    EXACT_BC3_CLIENT_SHA256,
    FORGOTTEN_NATURE_152_SHA256,
    VANILLA_152_BIOMES,
    VANILLA_NAMES,
    vanilla_texture,
    vanilla_visual,
)


CATALOG_SCHEMA = "geo.minecraft-block-catalog/v1"
CONFIG_ID_RE = re.compile(r"^\s*I:([^=]+)=(-?\d+)\s*$")
ASSIGNMENT_RE = re.compile(r"^\s*([A-Za-z0-9_.-]+)\s*[=:]\s*['\"]?([^'\"#;]+)")
CUSTOM_ID_RE = re.compile(r"config\.getBlockId\(\s*['\"]([^'\"]+)['\"]\s*\)")
CUSTOM_NAME_RE = re.compile(r"^\s*name\s*=\s*['\"]([^'\"]+)['\"]", re.MULTILINE)
CUSTOM_DISPLAY_RE = re.compile(r"displayName\[(\d+)]\s*=\s*['\"]([^'\"]*)['\"]")
CUSTOM_TOP_RE = re.compile(r"textureFileYP\[(\d+)]\s*=\s*['\"]([^'\"]+)['\"]")
PALETTE_RE = re.compile(r"^\s*palette\.block\.([^=]+)\s*=\s*(.+?)\s*$")
FN_BIOME_ID_RE = re.compile(r'^\s*I:"Biomes: ([^"]+) ID"=(\d+)\s*$')


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
        # ZIP membership is immutable after fingerprinting. Index it once;
        # reopening a large client/mod archive for every model parent made
        # complete modern catalogs needlessly quadratic in archive members.
        self._names = [set(source.names()) for source in self.sources]
        self._archives = [zipfile.ZipFile(source.path) if source.kind == "zip" else None for source in self.sources]

    def read(self, index: int, member: str) -> bytes:
        archive = self._archives[index]
        if archive is not None:
            return archive.read(member)
        return self.sources[index].read(member)

    def locate(self, member: str) -> tuple[int, bytes] | None:
        for index, source in enumerate(self.sources):
            if member in self._names[index]:
                return index, self.read(index, member)
        return None

    def effective_names(self, suffix: str) -> dict[str, int]:
        result: dict[str, int] = {}
        for index in range(len(self.sources) - 1, -1, -1):
            for name in self._names[index]:
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
            # 26.1 introduced texture descriptors so a model can force a
            # translucent render layer while still naming one exact sprite.
            if isinstance(candidate, dict):
                candidate = candidate.get("sprite", "")
            if not isinstance(candidate, str):
                candidate = ""
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


def _modern_visual(block_name: str) -> dict[str, object]:
    path = block_name.partition(":")[2]
    geometry = "cube"
    tint = None
    if path in {"water", "lava"}:
        geometry = "fluid"
    elif path.endswith("_leaves") or path in {"glass", "ice", "tinted_glass", "spawner"}:
        geometry = "alpha_cube"
    elif path.endswith(("_sapling", "_flower", "_mushroom", "_roots", "_tulip")) or path in {
        "short_grass", "tall_grass", "fern", "large_fern", "dead_bush", "sugar_cane", "wheat",
        "carrots", "potatoes", "beetroots", "nether_wart", "cocoa", "torch", "redstone_torch",
    }:
        geometry = "cross"
    elif path.endswith("_stairs"):
        geometry = "stair"
    elif path.endswith("_slab"):
        geometry = "slab"
    elif path.endswith(("_fence", "_wall", "_pane")) or path in {"iron_bars", "fence_gate"}:
        geometry = "connected"
    elif path.endswith(("_door", "_trapdoor", "_sign", "_hanging_sign")) or path in {"ladder", "vine"}:
        geometry = "plane"
    elif path.endswith("_rail") or path in {"rail", "redstone_wire", "tripwire"}:
        geometry = "line"
    elif path == "snow":
        geometry = "cover"
    elif path in {"chest", "trapped_chest", "ender_chest", "bed", "cake", "farmland", "daylight_detector"}:
        geometry = "partial"
    if path == "grass_block" or path in {"short_grass", "tall_grass", "fern", "large_fern"}:
        tint = "grass"
    elif path.endswith("_leaves") or path == "vine":
        tint = "foliage"
    elif path == "water":
        tint = "water"
    result: dict[str, object] = {"geometry": geometry, "height": 1.0}
    if geometry == "slab":
        result["height"] = 0.5
    if geometry == "cover":
        result["height"] = 0.125
    if tint:
        result["tint"] = tint
    return result


def _modern_environment(resources: SourceSet) -> tuple[dict[str, dict[str, object]], dict[str, dict[str, object]]]:
    def color_value(value: object, default: int | None = None) -> int | None:
        if value is None:
            return default
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.startswith("#"):
            return int(value[1:], 16)
        return int(value)

    colorizers: dict[str, dict[str, object]] = {}
    for kind in ("grass", "foliage"):
        member = f"assets/minecraft/textures/colormap/{kind}.png"
        located = resources.locate(member)
        if located:
            source, raw = located
            colorizers[kind] = {
                "source": source, "member": member, "memberSha256": hashlib.sha256(raw).hexdigest(),
            }
    biomes: dict[str, dict[str, object]] = {}
    for member, source_index in resources.effective_names(".json").items():
        parts = PurePosixPath(member).parts
        if len(parts) != 5 or parts[0] != "data" or tuple(parts[2:4]) != ("worldgen", "biome"):
            continue
        try:
            raw = resources.read(source_index, member)
            data = json.loads(raw)
            effects = data.get("effects", {})
            name = f"{parts[1]}:{PurePosixPath(parts[4]).stem}"
            biomes[name] = {
                "name": name,
                "temperature": float(data["temperature"]),
                "rainfall": float(data["downfall"]),
                "waterMultiplier": color_value(effects.get("water_color"), 0x3F76E4),
                "grassColor": color_value(effects.get("grass_color")),
                "foliageColor": color_value(effects.get("foliage_color")),
                "grassColorModifier": effects.get("grass_color_modifier"),
                "resolution": "exact",
                "provenance": [{
                    "kind": "exact-biome-json", "source": source_index, "member": member,
                    "memberSha256": hashlib.sha256(raw).hexdigest(),
                }],
            }
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
    return colorizers, biomes


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
            raw = resources.read(source_index, blockstate_path)
            state = json.loads(raw)
        except (ValueError, KeyError, json.JSONDecodeError) as exc:
            entries[block_name] = resolution_entry(
                "unknown", name=block_name, reason=f"invalid blockstate: {exc}",
                provenance=[{"kind": "asset-member", "source": source_index, "member": blockstate_path}],
            )
            continue
        variants = state.get("variants")
        if not isinstance(variants, dict):
            multipart = state.get("multipart")
            applications = []
            if isinstance(multipart, list):
                for part in multipart:
                    applied = part.get("apply") if isinstance(part, dict) else None
                    applications.extend(applied if isinstance(applied, list) else [applied])
            try:
                resolved = [
                    _top_texture(resources, item["model"], namespace)
                    for item in applications if isinstance(item, dict) and "model" in item
                ]
                textures = {json.dumps(item[0], sort_keys=True) for item in resolved}
                if not resolved:
                    raise ValueError("multipart blockstate has no resolvable supplied top texture")
                texture, model_provenance = resolved[0]
                all_provenance = sorted({path for _, paths in resolved for path in paths})
                if len(textures) == 1:
                    entries[block_name] = resolution_entry(
                        "exact", name=block_name, texture=texture, visual=_modern_visual(block_name),
                        reason="Every supplied multipart model resolves to the same exact top-face texture.",
                        provenance=[{"kind": "asset-member", "member": path} for path in [blockstate_path] + all_provenance],
                    )
                else:
                    entries[block_name] = resolution_entry(
                        "inferred", name=block_name, texture=texture, visual=_modern_visual(block_name),
                        confidence=0.82,
                        reason="The exact multipart models use multiple sprites; this supplied top-cap sprite is an explicitly inferred single-pixel overhead representative, not an exact composite.",
                        provenance=[{"kind": "asset-member", "member": path} for path in [blockstate_path] + all_provenance],
                    )
            except (ValueError, KeyError, json.JSONDecodeError) as exc:
                entries[block_name] = resolution_entry(
                    "unknown", name=block_name, reason=str(exc),
                    provenance=[{"kind": "asset-member", "source": source_index, "member": blockstate_path}],
                )
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
                entries[key] = resolution_entry(
                    "exact", name=block_name, texture=texture, visual=_modern_visual(block_name),
                    reason="Supplied blockstate and model chain select one supplied top-face texture.",
                    provenance=[{"kind": "asset-member", "member": path} for path in [blockstate_path] + model_provenance],
                )
            except (ValueError, KeyError, json.JSONDecodeError) as exc:
                entries[key] = resolution_entry(
                    "unknown", name=block_name, reason=str(exc),
                    provenance=[{"kind": "asset-member", "source": source_index, "member": blockstate_path}],
                )
    # Fluids have no normal blockstate/model, but their canonical saved names
    # and exact client textures are stable and required for overhead maps.
    for block_name, texture_name in (("minecraft:water", "water_still"), ("minecraft:lava", "lava_still")):
        member = f"assets/minecraft/textures/block/{texture_name}.png"
        located = resources.locate(member)
        if located:
            source_index, raw = located
            entries[block_name] = resolution_entry(
                "resolved", name=block_name, texture={"source": source_index, "member": member},
                visual=_modern_visual(block_name),
                reason="Canonical saved fluid name selects the exact matching client still-fluid texture.",
                provenance=[
                    {"kind": "canonical-modern-fluid", "name": block_name},
                    {"kind": "exact-texture", "source": source_index, "member": member,
                     "memberSha256": hashlib.sha256(raw).hexdigest()},
                ],
            )
    # Untouched pre-rename chunks can retain this palette identifier even when
    # level.dat has the current DataVersion. Never datafix the save; resolve the
    # stable resource rename against the exact current client instead.
    short_grass = entries.get("minecraft:short_grass")
    if short_grass and short_grass.get("status") == "resolved":
        entries["minecraft:grass"] = resolution_entry(
            "resolved", name="minecraft:grass", texture=deepcopy(short_grass["texture"]),
            visual=_modern_visual("minecraft:short_grass"),
            reason="An untouched older chunk palette retains minecraft:grass; the canonical short_grass resource rename selects the exact supplied current-client sprite without rewriting the chunk.",
            provenance=[
                {"kind": "canonical-palette-alias", "from": "minecraft:grass", "to": "minecraft:short_grass"},
                *deepcopy(short_grass["provenance"]),
            ],
        )
    colorizers, biomes = _modern_environment(resources)
    return {
        "schema": CATALOG_SCHEMA, "edition": "modern", "sources": [item.record() for item in sources],
        "colorizers": colorizers, "modernBiomes": biomes,
        "legacyRendering": {
            "edition": "Minecraft Java modern",
            "algorithm": "exact biome JSON climate/effects and exact client colormaps; no unrecorded pack substitution",
            "options": {"smoothBiomes": False, "swampColors": False},
        },
        "entries": entries,
    }


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
    (17, 1): ("minecraft:spruce_log", "textures/blocks/tree_top.png"),
    (17, 2): ("minecraft:birch_log", "textures/blocks/tree_top.png"),
    (17, 3): ("minecraft:jungle_log", "textures/blocks/tree_top.png"),
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
    (98, 0): ("minecraft:stone_bricks", "textures/blocks/stonebricksmooth.png"),
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


def _legacy_colorizers(
    resources: SourceSet,
) -> tuple[dict[str, dict[str, object]], dict[str, str], dict[str, int], list[dict[str, object]]]:
    """Record exact OptiFine/vanilla color assets and block-palette selectors."""

    colorizers: dict[str, dict[str, object]] = {}
    members = {
        "grass": "misc/grasscolor.png",
        "foliage": "misc/foliagecolor.png",
        "pine": "misc/pinecolor.png",
        "birch": "misc/birchcolor.png",
        "swamp_grass": "misc/swampgrasscolor.png",
        "swamp_foliage": "misc/swampfoliagecolor.png",
    }
    for name, member in members.items():
        located = resources.locate(member)
        if located:
            source, raw = located
            colorizers[name] = {"source": source, "member": member, "memberSha256": hashlib.sha256(raw).hexdigest()}

    selectors: dict[str, str] = {}
    fixed_colors: dict[str, int] = {}
    properties_evidence: list[dict[str, object]] = []
    located_properties = resources.locate("color.properties")
    if located_properties:
        source, raw = located_properties
        properties_evidence.append({
            "kind": "exact-color-properties", "source": source, "member": "color.properties",
            "memberSha256": hashlib.sha256(raw).hexdigest(),
        })
        for line in raw.decode("iso-8859-1", errors="replace").splitlines():
            key, separator, value = line.partition("=")
            if separator and key.strip() == "lilypad":
                try:
                    fixed_colors["lily_pad"] = int(value.strip(), 16)
                except ValueError:
                    pass
            match = PALETTE_RE.match(line)
            if not match:
                continue
            member = match.group(1).lstrip("/")
            located = resources.locate(member)
            if not located:
                continue
            palette_source, pixels = located
            palette_name = f"block_palette_{len(colorizers)}"
            colorizers[palette_name] = {
                "source": palette_source, "member": member, "memberSha256": hashlib.sha256(pixels).hexdigest(),
            }
            for token in match.group(2).split():
                block_id, separator, metadata = token.partition(":")
                if block_id.isdigit() and (not separator or metadata.isdigit()):
                    selectors[f"legacy:{int(block_id)}:{int(metadata) if separator else '*'}"] = palette_name
    return colorizers, selectors, fixed_colors, properties_evidence


def _legacy_runtime_evidence(root: Path, sources: list[AssetSource]) -> dict[str, object]:
    client_index = next((index for index, source in enumerate(sources) if source.sha256 == EXACT_BC3_CLIENT_SHA256), None)
    provenance: list[dict[str, object]] = []
    if client_index is not None:
        source = sources[client_index]
        members = [name for name in ("aav.class", "aaa.class", "zx.class", "CustomColorizer.class") if source.has(name)]
        provenance.extend({
            "kind": "exact-bundled-client-bytecode", "source": client_index,
            "archiveSha256": source.sha256, "member": member,
            "memberSha256": hashlib.sha256(source.read(member)).hexdigest(),
        } for member in members)
    options: dict[str, bool] = {"customColors": False, "smoothBiomes": False, "swampColors": False}
    candidates = list(root.rglob("optionsof.txt"))
    if len(candidates) == 1:
        path = candidates[0]
        parsed = dict(line.split(":", 1) for line in path.read_text(errors="replace").splitlines() if ":" in line)
        options = {
            "customColors": parsed.get("ofCustomColors") == "true",
            "smoothBiomes": parsed.get("ofSmoothBiomes") == "true",
            "swampColors": parsed.get("ofSwampColors") == "true",
        }
        provenance.append({
            "kind": "exact-client-options", "file": str(path.resolve()), "sha256": sha256_file(path), **options,
        })
    return {
        "edition": "Minecraft Java 1.5.2 Forge/OptiFine",
        "algorithm": "clamped temperature/rainfall lookup; rainfall *= temperature; optional exact 3x3 smooth-biome average",
        "options": options,
        "provenance": provenance or [{"kind": "canonical-edition", "edition": "Minecraft Java 1.5.2"}],
    }


def _legacy_biomes(root: Path, sources: list[AssetSource]) -> dict[str, dict[str, object]]:
    """Record exact 1.5.2 biome climates used by the supplied colorizers."""

    client = next((source for source in sources if source.sha256 == EXACT_BC3_CLIENT_SHA256), None)
    vanilla_resolution = "exact" if client is not None and client.has("aav.class") else "resolved"
    vanilla_provenance = ({
        "kind": "exact-bundled-client-bytecode",
        "archiveSha256": EXACT_BC3_CLIENT_SHA256,
        "member": "aav.class",
        "memberSha256": hashlib.sha256(client.read("aav.class")).hexdigest(),
    } if vanilla_resolution == "exact" else {
        "kind": "canonical-edition", "edition": "Minecraft Java 1.5.2",
        "subject": "biome climate table",
    })
    biomes: dict[str, dict[str, object]] = {
        str(biome_id): {
            "name": name,
            "temperature": temperature,
            "rainfall": rainfall,
            "waterMultiplier": water,
            "resolution": vanilla_resolution,
            "provenance": [vanilla_provenance],
        }
        for biome_id, (name, temperature, rainfall, water) in VANILLA_152_BIOMES.items()
    }
    mod_index = next((index for index, source in enumerate(sources)
                      if source.sha256 == FORGOTTEN_NATURE_152_SHA256), None)
    configs = list(root.rglob("ForgottenNature.cfg"))
    if mod_index is None or len(configs) != 1:
        return biomes
    config = configs[0]
    config_sha = sha256_file(config)
    ids = {match.group(1): int(match.group(2)) for line in config.read_text(errors="replace").splitlines()
           if (match := FN_BIOME_ID_RE.match(line))}
    # F/G assignments in the fingerprinted classes are temperature/rainfall.
    definitions = {
        "Neo Tropical Forest": ("BiomeGenTropicalForest.class", "Neo Tropical Forest", 0.9, 0.9, "0x6ECB01"),
        "Neo Redwood Forest": ("BiomeGenNeoRedwoodForest.class", "Redwood Forest", 0.7, 0.7, "0x33AA22"),
        "Neo Tropical Forest Hills": ("BiomeGenNeoTropicalForestHills.class", "Neo Tropical Forest Hills", 0.9, 0.9, "0x6ECB01"),
        "Neo Redwood Forest Hills": ("BiomeGenNeoRedwoodForestHills.class", "Redwood Forest Hills", 0.5, 0.7, "0x33AA22"),
        "Neo Redwood Forest Snow Hills": ("BiomeGenNeoRedwoodForestSnowHills.class", "Redwood Forest Snow Hills", 0.0, 0.5, "0x33AA22"),
        "Neo Redwood Forest Snow": ("BiomeGenNeoRedwoodForestSnow.class", "Snowy Redwood Forest", 0.0, 0.5, "0x33AA22"),
        "Crystal Forest": ("BiomeGenCrystalForest.class", "Crystal Forest", 0.8, 0.2, "0x51EAEC"),
    }
    source = sources[mod_index]
    for config_name, (class_name, display_name, temperature, rainfall, native_mask) in definitions.items():
        biome_id = ids.get(config_name)
        member = f"ForgottenNature/Biomes/{class_name}"
        if biome_id is None or not source.has(member):
            continue
        biomes[str(biome_id)] = {
            "name": display_name,
            "temperature": temperature,
            "rainfall": rainfall,
            "waterMultiplier": 0xFFFFFF,
            "nativeGrassFoliageMask": native_mask,
            "resolution": "exact",
            "provenance": [
                {"kind": "exact-config", "file": str(config.resolve()), "sha256": config_sha,
                 "key": f"Biomes: {config_name} ID", "value": biome_id},
                {"kind": "exact-mod-bytecode", "source": mod_index, "archiveSha256": source.sha256,
                 "member": member, "memberSha256": hashlib.sha256(source.read(member)).hexdigest()},
            ],
        }
    return biomes


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
            # A number of 1.5.2 mods ship an inventory icon and a world-block
            # texture with the same stem.  The block directory is decisive for
            # this top-surface catalog; ambiguity only remains among multiple
            # block textures with the same stem.
            block_matches = [
                item for item in matches
                if "/textures/blocks/" in f"/{item[1].lower()}"
                or item[1].lower().startswith("textures/blocks/")
            ]
            if len(block_matches) == 1:
                return block_matches[0]
            return matches[0] if len(matches) == 1 else None
    return None


def build_legacy_catalog(bc3_root: str | Path, asset_paths: Iterable[str | Path]) -> dict[str, object]:
    root = Path(bc3_root).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"bc3 root is not a directory: {root}")
    sources = [AssetSource.create(path) for path in asset_paths]
    resources = SourceSet(sources)
    terrain = _find_terrain_source(sources)
    entries: dict[str, dict[str, object]] = {}
    for (block_id, metadata), (name, member) in VANILLA_152_FILES.items():
        located = next(((index, member) for index, source in enumerate(sources) if source.has(member)), None)
        if located:
            source_index, resolved_member = located
            entries[f"legacy:{block_id}:{metadata}"] = resolution_entry(
                "resolved", name=name, texture={"source": source_index, "member": resolved_member},
                reason="Canonical Minecraft Java 1.5.2 numeric ID and metadata select an exact supplied pack texture.",
                provenance=[
                    {"kind": "canonical-id", "edition": "Minecraft Java 1.5.2", "id": block_id, "metadata": metadata},
                    {"kind": "exact-texture", "source": source_index, "member": resolved_member},
                ],
            )
    if terrain:
        source_index, member = terrain
        for (block_id, metadata), (name, atlas_index) in VANILLA_152_TOP.items():
            if f"legacy:{block_id}:{metadata}" in entries:
                continue
            entries[f"legacy:{block_id}:{metadata}"] = resolution_entry(
                "resolved", name=name,
                texture={"source": source_index, "member": member, "atlas": {"columns": 16, "index": atlas_index}},
                reason="Canonical Minecraft Java 1.5.2 numeric ID selects an exact supplied terrain-atlas cell.",
                provenance=[
                    {"kind": "canonical-id", "edition": "Minecraft Java 1.5.2", "id": block_id, "metadata": metadata},
                    {"kind": "exact-texture", "source": source_index, "member": member, "atlasIndex": atlas_index},
                ],
            )
    # Resolve every canonical vanilla metadata value that has an exact supplied
    # upward-view texture. This replaces the earlier small terrain-only table
    # and carries geometry/tint semantics separately from pixel provenance.
    for block_id, vanilla_name in VANILLA_NAMES.items():
        for metadata in range(16):
            spec = vanilla_texture(block_id, metadata)
            if spec is None:
                continue
            located = resources.locate(str(spec["member"]))
            if located is None:
                continue
            source_index, raw = located
            texture = {"source": source_index, **spec}
            entries[f"legacy:{block_id}:{metadata}"] = resolution_entry(
                "resolved", name=f"minecraft:{vanilla_name}", texture=texture,
                visual=vanilla_visual(block_id, metadata),
                reason="Canonical Minecraft Java 1.5.2 ID/metadata and bundled client top-face rules select this exact supplied texture.",
                provenance=[
                    {"kind": "canonical-id", "edition": "Minecraft Java 1.5.2", "id": block_id, "metadata": metadata},
                    {"kind": "exact-texture", "source": source_index, "member": spec["member"],
                     "memberSha256": hashlib.sha256(raw).hexdigest()},
                ],
            )
    entries["legacy:36:*"] = resolution_entry(
        "resolved", name="minecraft:moving_piston", renderAsAir=True, visual=vanilla_visual(36, 0),
        reason="Minecraft 1.5.2 moving-piston is a transient tile-entity helper with no independently rendered saved top surface.",
        provenance=[{"kind": "canonical-id", "edition": "Minecraft Java 1.5.2", "id": 36}],
    )
    assignments = _config_assignments(root)
    by_id: dict[int, list[dict[str, object]]] = {}
    for record in assignments:
        by_id.setdefault(int(record["id"]), []).append(record)
    for block_id, records in sorted(by_id.items()):
        key = f"legacy:{block_id}:*"
        entries[key] = resolution_entry(
            "unknown", name=records[0]["key"] if len(records) == 1 else f"configured_id_{block_id}",
            reason="Forge config proves the numeric ID but not a metadata-specific top texture",
            provenance=[{"kind": "forge-config", **record} for record in records],
        )
    for definition in _customstuff_definitions(root, assignments):
        metadata = definition["metadata"]
        key = f"legacy:{definition['id']}:{metadata}"
        located = _locate_loose_texture(sources, str(definition["texture"])) if definition.get("texture") else None
        if located:
            source_index, member = located
            exact_label = f"{definition.get('name', '')} {definition.get('file', '')}".lower()
            geometry = "alpha_cube" if "glass" in exact_label else "cube"
            entries[key] = resolution_entry(
                "exact", name=str(definition["name"]), texture={"source": source_index, "member": member},
                visual={"geometry": geometry, "height": 1.0},
                reason="Exact CustomStuff definition links this configured ID and metadata to a uniquely supplied top texture.",
                provenance=[{"kind": "customstuff-definition", **definition}, {"kind": "exact-texture", "source": source_index, "member": member}],
            )
        else:
            entries[key] = resolution_entry(
                "unknown", name=str(definition["name"]), reason="CustomStuff definition has no uniquely resolvable top texture",
                provenance=[{"kind": "customstuff-definition", **definition}],
            )
    resolver_reports = apply_exact_legacy_resolvers(
        root=root,
        sources=sources,
        assignments=assignments,
        entries=entries,
        locate_texture=lambda value: _locate_loose_texture(sources, value),
    )
    colorizers, block_colorizers, fixed_colors, color_properties = _legacy_colorizers(resources)
    runtime = _legacy_runtime_evidence(root, sources)
    runtime["provenance"] = list(runtime["provenance"]) + color_properties
    biomes = _legacy_biomes(root, sources)
    return {
        "schema": CATALOG_SCHEMA,
        "edition": "legacy-1.5.2-forge",
        "sources": [item.record() for item in sources],
        "bc3": {"root": str(root), "treeSha256": sha256_tree(root)},
        "resolvers": resolver_reports,
        "colorizers": colorizers,
        "blockColorizers": block_colorizers,
        "fixedColors": fixed_colors,
        "legacyRendering": runtime,
        "legacyBiomes": biomes,
        "entries": entries,
    }


def _cover_visible_modern_catalog(
    catalog: dict[str, object], inventory: dict[str, object]
) -> dict[str, object]:
    result = deepcopy(catalog)
    entries: dict[str, dict[str, Any]] = result["entries"]  # type: ignore[assignment]
    sources = [AssetSource.create(item["path"]) for item in result.get("sources", [])]  # type: ignore[index]
    resources = SourceSet(sources)

    def current(key: str) -> dict[str, Any] | None:
        """Resolve an observed full palette state like Catalog.entry does."""
        direct = entries.get(key)
        if direct is not None:
            return direct
        base = key.partition("[")[0]
        direct = entries.get(base)
        if direct is not None or "[" not in key:
            return direct
        actual = dict(
            pair.split("=", 1)
            for pair in key.removesuffix("]").partition("[")[2].split(",")
            if "=" in pair
        )
        matches: list[dict[str, Any]] = []
        for candidate, entry in entries.items():
            if not candidate.startswith(base + "["):
                continue
            required = dict(
                pair.split("=", 1)
                for pair in candidate.removesuffix("]").partition("[")[2].split(",")
                if "=" in pair
            )
            if all(actual.get(name) == value for name, value in required.items()):
                matches.append(entry)
        fingerprints = {
            json.dumps(item.get("texture"), sort_keys=True)
            for item in matches if item.get("status") == "resolved"
        }
        if len(matches) == 1:
            return matches[0]
        if matches and all(item.get("status") == "resolved" for item in matches) and len(fingerprints) == 1:
            return matches[0]
        return None

    def locate(candidates: Iterable[str]) -> tuple[int, str, str] | None:
        for member in candidates:
            located = resources.locate(member)
            if located:
                index, raw = located
                return index, member, hashlib.sha256(raw).hexdigest()
        return None

    observed_states = inventory.get("surfaceStates")
    if not isinstance(observed_states, dict):
        unknown = inventory.get("unknown") or (inventory.get("resolution") or {}).get("unknown", {})  # type: ignore[union-attr]
        block_counts = unknown.get("blockCounts", {}) if isinstance(unknown, dict) else {}
        observed_states = {
            key: {"surfaceContributions": count, "topColumns": 0}
            for key, count in block_counts.items()
        }

    inferred = 0
    contributions = 0
    for key, observed in sorted(observed_states.items()):
        existing = current(key)
        if existing and existing.get("status") == "resolved":
            continue
        base = key.partition("[")[0]
        namespace, _, path = base.partition(":")
        namespace = namespace or "minecraft"
        stripped = path
        for suffix in (
            "_stairs", "_slab", "_fence_gate", "_fence", "_wall", "_button",
            "_pressure_plate", "_trapdoor", "_door", "_hanging_sign", "_wall_sign", "_sign",
        ):
            if stripped.endswith(suffix):
                stripped = stripped.removesuffix(suffix)
                break
        special = {
            "redstone_wire": "redstone_dust_dot", "tripwire": "tripwire",
            "iron_bars": "iron_bars", "grass": "short_grass",
        }.get(path)
        candidates = []
        if special:
            candidates.append(f"assets/{namespace}/textures/block/{special}.png")
        if path.endswith("_pane"):
            candidates.append(f"assets/{namespace}/textures/block/{path}_top.png")
        if path.endswith(("_sign", "_hanging_sign")) or "_wall_sign" in path:
            candidates.append(f"assets/{namespace}/textures/block/{stripped}_planks.png")
        candidates.extend([
            f"assets/{namespace}/textures/block/{path}.png",
            f"assets/{namespace}/textures/block/{stripped}.png",
            f"assets/{namespace}/textures/block/{stripped}_planks.png",
            "assets/minecraft/textures/block/stone.png",
        ])
        located = locate(candidates)
        if located is None:
            continue
        source_index, member, member_hash = located
        exact_named = member != "assets/minecraft/textures/block/stone.png"
        provenance = list((existing or {}).get("provenance", []))
        provenance.extend([
            {
                "kind": "surface-inventory", "schema": inventory.get("schema"), "state": key,
                "surfaceContributions": int(observed.get("surfaceContributions", 0)),
                "topColumns": int(observed.get("topColumns", 0)),
            },
            {
                "kind": "exact-representative-texture", "source": source_index,
                "member": member, "memberSha256": member_hash,
            },
        ])
        entries[key] = resolution_entry(
            "inferred", name=base, texture={"source": source_index, "member": member},
            visual=_modern_visual(base), confidence=0.72 if exact_named else 0.25,
            reason=(
                "The exact current-client asset name matches this unresolved multipart/state family, but one flat top sprite cannot reproduce the complete model; it is explicitly inferred."
                if exact_named else
                "No unique current-client top sprite can be proven for this visible state; exact current-client stone is an explicit neutral fallback, not an exact mapping."
            ),
            provenance=provenance,
        )
        inferred += 1
        contributions += int(observed.get("surfaceContributions", 0))
    result["visibleCoverageAugmentation"] = {
        "inventorySchema": inventory.get("schema"), "statesInferred": inferred,
        "surfaceContributionsInferred": contributions,
        "scope": "only unresolved modern states present in the supplied read-only inventory",
    }
    if isinstance(inventory.get("surfaceStates"), dict):
        coverage = {
            kind: {"distinctBlockStates": 0, "surfaceContributions": 0, "topColumns": 0}
            for kind in ("exact", "resolved", "inferred", "unknown")
        }
        for state_key, observation in inventory["surfaceStates"].items():  # type: ignore[index]
            entry = current(state_key)
            resolution = str(entry.get("resolution")) if entry else "unknown"
            if resolution not in coverage:
                resolution = "unknown"
            coverage[resolution]["distinctBlockStates"] += 1
            coverage[resolution]["surfaceContributions"] += int(observation.get("surfaceContributions", 0))
            coverage[resolution]["topColumns"] += int(observation.get("topColumns", 0))
        result["visibleCoverageFromInventory"] = coverage
    return result


def cover_visible_legacy_catalog(
    catalog: dict[str, object], inventory: dict[str, object]
) -> dict[str, object]:
    """Add explicit, auditable inferences only for inventoried visible states.

    This is deliberately inventory-scoped: it does not claim that an arbitrary
    legacy ID is known.  Exact/resolved entries always win.  The fallback keeps
    the exact archived pixels and records where legacy metadata or tile-entity
    model data is insufficient to prove the actual top face.
    """

    if catalog.get("edition") == "modern":
        return _cover_visible_modern_catalog(catalog, inventory)
    result = deepcopy(catalog)
    entries: dict[str, dict[str, Any]] = result["entries"]  # type: ignore[assignment]
    sources = [AssetSource.create(item["path"]) for item in result.get("sources", [])]  # type: ignore[index]

    railcraft_bricks = {
        466: "abyssal", 467: "infernal", 468: "bloodstained", 469: "sandy",
        470: "bleachedbone", 471: "quarried", 472: "frostbound",
    }
    railcraft_cubes = (
        "cube.coke", "concrete", "cube.steel", "cube.brick.infernal",
        "cube.crushed.obsidian", "cube.brick.sandy", "cube.stone.abyssal", "cube.stone.quarried",
    )
    furniture = {
        500: "itemovenoverhead", 501: "itemoven", 502: "itembedsidecabinet",
        503: "itemcoffeetablestone", 504: "itemtablestone", 505: "itemchairstone",
        506: "itemlamp", 507: "itemlamp", 508: "itemcoffeetablewood", 509: "itemtablewood",
        510: "itemchairwood", 511: "itemfridge", 512: "itemfridge", 513: "itemcabinet",
        514: "itemcouchwhite", 515: "itemcouchgreen", 516: "itemcouchbrown",
        517: "itemcouchred", 518: "itemcouchblack", 519: "itemcarpetwhite",
        520: "itemblinds", 521: "itemblinds", 522: "itemcurtains", 523: "itemcurtains",
        524: "itemhedge", 525: "itembirdbath", 526: "itemstonepath", 527: "itemwhitefence",
        528: "itemtap", 529: "itemmailbox",
    }
    bibliocraft = {
        2248: "writingdesk1", 2249: "armorstand", 2250: "BookcaseTexture1",
        2251: "woodlabel0", 2252: "potionshelf1", 2253: "toolrack1",
        2254: "genericshelf1", 2255: "weaponcase1",
    }

    def current(key: str) -> dict[str, Any] | None:
        value = entries.get(key)
        if value is not None:
            return value
        block_id = key.split(":")[1]
        return entries.get(f"legacy:{block_id}:*")

    def locate(member: str) -> tuple[int, str, str] | None:
        for index, source in enumerate(sources):
            if source.has(member):
                raw = source.read(member)
                return index, member, hashlib.sha256(raw).hexdigest()
        return None

    def choice(block_id: int, metadata: int, name: str) -> tuple[str | None, str, float, str]:
        """Return exact member, geometry, confidence, and semantic rationale."""
        if block_id == 454:
            return "mods/railcraft/textures/blocks/tracks/track.reinforced.png", "line", 0.72, "Railcraft stores track subtype in its tile entity; the archived reinforced straight-track pixels are representative."
        if block_id == 457:
            return f"mods/railcraft/textures/blocks/{railcraft_cubes[metadata & 7]}.png", "cube", 0.92, "The fingerprinted Railcraft EnumCube constant order identifies this metadata family."
        if block_id in railcraft_bricks:
            family = railcraft_bricks[block_id]
            return f"mods/railcraft/textures/blocks/brick.{family}.png", "cube", 0.9, "The exact Railcraft config identifies the brick family; decorative variant details are not independently encoded in the surface state."
        if block_id == 459:
            return "mods/railcraft/textures/blocks/post.wood.png", "connected", 0.86, "The exact config names this as the Railcraft wood-post block."
        if block_id == 460:
            member = "post.metal.painted.png" if metadata else "post.metal.png"
            return f"mods/railcraft/textures/blocks/{member}", "connected", 0.84, "The exact config and metadata distinguish unpainted from painted metal posts; paint tint is unavailable from the top state alone."
        if block_id in {461, 463}:
            return "mods/railcraft/textures/blocks/concrete.png", "connected", 0.62, "Railcraft wall material selection is not recoverable from this block state alone; exact archived concrete is a neutral wall representative."
        if block_id == 464:
            return "mods/railcraft/textures/blocks/concrete.png", "stair", 0.58, "Railcraft stores stair material outside orientation metadata; exact archived concrete is a neutral representative."
        if block_id == 465:
            return "mods/railcraft/textures/blocks/concrete.png", "slab", 0.58, "Railcraft stores slab material outside ordinary block metadata; exact archived concrete is a neutral representative."
        if block_id in {450, 451, 452, 453, 455, 456, 458}:
            members = {
                450: "detector.any.png", 451: "machine.alpha.png", 452: "tank.iron.wall.png",
                453: "loader.item.png", 455: "tracks/track.elevator.png",
                456: "signal.box.png", 458: "ore.sulfur.png",
            }
            return f"mods/railcraft/textures/blocks/{members[block_id]}", "line" if block_id == 455 else "partial", 0.48, "The exact config proves the Railcraft family, but metadata-specific model/state pixels are not fully recoverable without live mod rendering."
        if 500 <= block_id <= 529 and block_id in furniture:
            geometry = "cover" if block_id in {519, 526} else "connected" if block_id in {524, 527} else "partial"
            return f"textures/items/{furniture[block_id]}.png", geometry, 0.7, "The bundled Furniture Mod has a custom model and no standalone top face; its exact matching inventory icon is used as an explicitly inferred overhead symbol."
        if block_id == 1225:
            return "mods/ComputerCraft/textures/blocks/computerTop.png", "cube", 0.78, "The exact ComputerCraft config and bundled top texture identify the computer family; orientation/subtype details are approximate."
        if block_id == 1226:
            return "mods/ComputerCraft/textures/blocks/printerTop.png", "cube", 0.55, "The exact ComputerCraft peripheral family is known, but its subtype needs tile-entity context; printer-top pixels are representative."
        if block_id == 1229:
            return "mods/ComputerCraft/textures/blocks/cableSide.png", "connected", 0.72, "The exact ComputerCraft config identifies the cable/modem family and supplies these cable pixels."
        if 2248 <= block_id <= 2255:
            return f"mods/BiblioCraft/textures/models/{bibliocraft[block_id]}.png", "partial", 0.68, "BiblioCraft uses a custom 3-D model; the exact model skin is used as an explicitly inferred overhead representation."
        if block_id in {3500, 3501}:
            return None, "partial", 0.35, "MinePainter stores canvas/sculpture appearance in tile-entity data and dynamic textures, so a neutral exact-pack construction material is used."
        if 1525 <= block_id <= 1527:
            return None, "partial", 0.42, "CustomNPCs registers this contiguous block range, but the saved metadata does not prove a unique custom-model top texture."
        if 188 <= block_id <= 191:
            return None, "point", 0.64, "The exact ForgottenNature config identifies a flower-pot bank; plant contents are not fully recoverable from the ordinary state."
        if block_id == 181:
            return None, "cross", 0.75, "The exact ForgottenNature mushroom family is known; this out-of-range metadata is represented by its exact base mushroom texture."
        tokens = name.lower()
        if any(word in tokens for word in ("leaf", "hedge", "foliage")):
            return None, "alpha_cube", 0.35, "The configured name indicates foliage, but no exact metadata-to-texture linkage remains."
        if any(word in tokens for word in ("rail", "track", "wire", "cable")):
            return None, "line", 0.3, "The configured name indicates a line-like transport block, but no exact top-face linkage remains."
        if any(word in tokens for word in ("flower", "crop", "fruit", "mushroom", "sap")):
            return None, "cross", 0.3, "The configured name indicates a plant-like block, but no exact metadata-to-texture linkage remains."
        return None, "cube", 0.25, "No exact metadata-specific top face survives in the archived evidence; a neutral exact-pack construction texture avoids a destructive checker."

    def vanilla_reference(block_id: int, metadata: int, geometry: str) -> tuple[str, dict[str, Any]]:
        if geometry == "line":
            key = "legacy:66:0"
        elif geometry == "alpha_cube":
            key = "legacy:18:0"
        elif geometry == "cross":
            key = "legacy:37:0"
        elif geometry == "point":
            key = "legacy:140:0"
        elif geometry == "cover" and block_id == 519:
            key = f"legacy:35:{metadata & 15}"
        elif geometry == "cover":
            key = "legacy:4:0"
        elif "wood" in str((current(f"legacy:{block_id}:{metadata}") or {}).get("name", "")).lower():
            key = "legacy:5:0"
        else:
            key = "legacy:98:0"
        reference = entries[key]
        return key, reference

    observed_states = inventory.get("surfaceStates")
    if not isinstance(observed_states, dict):
        unknown = inventory.get("unknown")
        if not isinstance(unknown, dict):
            unknown = (inventory.get("resolution") or {}).get("unknown", {})  # type: ignore[union-attr]
        block_counts = unknown.get("blockCounts", {}) if isinstance(unknown, dict) else {}
        observed_states = {
            key: {"surfaceContributions": count, "topColumns": 0}
            for key, count in block_counts.items()
        }

    augmented_states = 0
    augmented_contributions = 0
    for key, observed in sorted(observed_states.items()):
        prior = current(key)
        if prior and prior.get("status") == "resolved":
            continue
        _, raw_id, raw_meta = key.split(":")
        block_id, metadata = int(raw_id), int(raw_meta)
        name = str((prior or {}).get("name") or f"unregistered legacy block {block_id}:{metadata}")
        member, geometry, confidence, semantic_reason = choice(block_id, metadata, name)
        located = locate(member) if member else None
        provenance: list[dict[str, Any]] = []
        if prior:
            provenance.extend(prior.get("provenance", []))
        provenance.append({
            "kind": "surface-inventory", "schema": inventory.get("schema"), "state": key,
            "surfaceContributions": int(observed.get("surfaceContributions", 0)),
            "topColumns": int(observed.get("topColumns", 0)),
        })
        if located:
            source_index, exact_member, member_hash = located
            texture = {"source": source_index, "member": exact_member}
            provenance.append({
                "kind": "exact-representative-texture", "source": source_index,
                "member": exact_member, "memberSha256": member_hash,
            })
        else:
            reference_key, reference = vanilla_reference(block_id, metadata, geometry)
            texture = deepcopy(reference["texture"])
            provenance.append({
                "kind": "exact-pack-representative", "fromState": reference_key,
                "texture": deepcopy(texture),
            })
        entries[key] = resolution_entry(
            "inferred", name=name, texture=texture,
            visual={"geometry": geometry, "height": 0.0625 if geometry in {"line", "cover"} else 0.5 if geometry in {"slab", "stair"} else 1.0},
            confidence=confidence,
            reason=semantic_reason + " This is not presented as an exact state mapping.",
            provenance=provenance,
        )
        wildcard_key = f"legacy:{block_id}:*"
        wildcard = entries.get(wildcard_key)
        if wildcard and wildcard.get("status") != "resolved":
            family_provenance = deepcopy(provenance)
            family_provenance.append({
                "kind": "metadata-family-fallback", "triggerState": key,
                "scope": "unresolved metadata values for this exactly configured numeric ID",
            })
            entries[wildcard_key] = resolution_entry(
                "inferred", name=name, texture=deepcopy(texture),
                visual=deepcopy(entries[key]["visual"]),
                confidence=max(0.1, confidence - 0.08),
                reason=semantic_reason + " Unobserved metadata values in this exactly configured ID family use the same explicitly inferred fallback and are not presented as exact.",
                provenance=family_provenance,
            )
        augmented_states += 1
        augmented_contributions += int(observed.get("surfaceContributions", 0))
    previous = result.get("visibleCoverageAugmentation", {})
    result["visibleCoverageAugmentation"] = {
        "inventorySchema": inventory.get("schema"),
        "statesInferred": int(previous.get("statesInferred", 0)) + augmented_states,
        "surfaceContributionsInferred": int(previous.get("surfaceContributionsInferred", 0)) + augmented_contributions,
        "passes": int(previous.get("passes", 0)) + 1,
        "scope": "only visible states present in the supplied read-only inventory",
    }
    if isinstance(inventory.get("surfaceStates"), dict):
        coverage = {
            kind: {"distinctBlockStates": 0, "surfaceContributions": 0, "topColumns": 0}
            for kind in ("exact", "resolved", "inferred", "unknown")
        }
        for state_key, observation in inventory["surfaceStates"].items():  # type: ignore[index]
            entry = current(state_key)
            resolution = str(entry.get("resolution")) if entry else "unknown"
            if resolution not in coverage:
                resolution = "unknown"
            coverage[resolution]["distinctBlockStates"] += 1
            coverage[resolution]["surfaceContributions"] += int(observation.get("surfaceContributions", 0))
            coverage[resolution]["topColumns"] += int(observation.get("topColumns", 0))
        result["visibleCoverageFromInventory"] = coverage
    return result


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
        for key, entry in self.entries.items():
            # v1 catalogs created before certainty labels remain readable. New
            # catalogs always include and validate the richer evidence fields.
            if "resolution" not in entry:
                continue
            try:
                validate_resolution_entry(entry)
            except ValueError as exc:
                raise ValueError(f"Invalid catalog entry {key}: {exc}") from exc

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

    def colorizer_bytes(self, name: str) -> bytes:
        colorizer = self.data.get("colorizers", {}).get(name)
        if not colorizer:
            raise KeyError(name)
        return self.sources[int(colorizer["source"])].read(str(colorizer["member"]))

    def block_colorizer(self, key: str) -> str | None:
        selectors = self.data.get("blockColorizers", {})
        if key in selectors:
            return str(selectors[key])
        if key.startswith("legacy:"):
            block_id = key.split(":")[1]
            wildcard = selectors.get(f"legacy:{block_id}:*")
            return str(wildcard) if wildcard else None
        return None

    def renders_as_air(self, key: str) -> bool:
        entry = self.entry(key)
        return bool(entry and entry.get("status") == "resolved" and entry.get("renderAsAir") is True)
