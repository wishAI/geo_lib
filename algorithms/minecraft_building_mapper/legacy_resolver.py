"""Fingerprint-gated resolvers for exact legacy mod binaries.

The tables here are executable summaries of methods in the named bundled class
files. They are applied only when the whole mod archive hash matches the audited
bc3 copy, and only when every selected texture exists in the supplied assets.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Callable

from .evidence import resolution_entry


FORGOTTEN_NATURE_152_SHA256 = "063cc073a6fb18990073ea632c151526ce8fb24bb14bdcc13c15702e46176ee3"
RAILCRAFT_7230_SHA256 = "42b6b736a544303eafc420504e6097c1a60ac1e3e9eeb16d9217a05b209438c2"

LEAF_VARIANTS: tuple[tuple[str, tuple[tuple[str, str] | None, ...]], ...] = (
    (
        "BlockNewLeaves",
        (
            ("Red Maple Leaves", "RedMapleLeaves"),
            ("Angel Oak Leaves", "AngelLeaves"),
            ("Yellow Maple Leaves", "YellowMapleLeaves"),
            ("Jacaranda Leaves", "JacarandaLeaves"),
            ("Fig Leaves", "FigLeaves"),
            ("Cypress Leaves", "CypressLeaves"),
            ("Desert Ironwood Leaves", "DesertIronwoodLeaves"),
            ("Eucalyptus Leaves", "EucalyptusLeaves"),
        ),
    ),
    (
        "BlockNewLeaves2",
        (
            ("Sequoia Leaves", "SequoiaLeaves"),
            ("Pink Cherry Leaves", "CherryPinkLeaves"),
            ("White Cherry Leaves", "CherryWhiteLeaves"),
            ("Apple Leaves", "AppleLeaves"),
            ("Apple Bud Leaves", "AppleBudLeaves"),
            ("Apple Fruit Leaves", "AppleFruitLeaves"),
            ("Acacia Leaves", "AcaciaLeaves"),
            ("Joshua Leaves", "JoshuaLeaves"),
        ),
    ),
    (
        "BlockNewLeaves3",
        (
            ("Swamp Willow Leaves", "SwampWillowLeaves"),
            ("Deciduous Bush Leaves", "DeciduousBushLeaves"),
            ("Evergreen Bush Leaves", "EvergreenBushLeaves"),
            ("Palm Leaves", "PalmLeaves"),
            ("Desert Willow Leaves", "DesertWillowLeaves"),
            ("Cedar Leaves", "CedarLeaves"),
            ("Ginkgo Leaves", "GinkgoLeaves"),
            ("Poplar Leaves", "PoplarLeaves"),
        ),
    ),
    (
        "BlockNewLeaves4",
        (
            ("Beech Leaves", "BeechLeaves"),
            ("Walnut Leaves", "WalnutLeaves"),
            None,
            ("WideTop Eucalyptus Leaves", "wTEucalyptusLeaves"),
            ("Bukkit Leaves", "BukkitLeaves"),
            ("Banana Leaves", "BananaLeaves"),
            ("Orange Leaves", "OrangeLeaves"),
            ("Peach Leaves", "PeachLeaves"),
        ),
    ),
    (
        "BlockNewLeaves5",
        (
            ("Lemon Leaves", "LemonLeaves"),
            ("Blueberry Leaves", "DeciduousBushLeaves"),
            ("Blueberry Fruit Leaves", "BlueberryFruitLeaves"),
            ("Raspberry Leaves", "DeciduousBushLeaves"),
            ("Raspberry Fruit Leaves", "RaspberryFruitLeaves"),
            ("Blackberry Leaves", "DeciduousBushLeaves"),
            ("Blackberry Fruit Leaves", "BlackberryFruitLeaves"),
            ("Cherry Fruit Leaves", "CherryFruitLeaves"),
        ),
    ),
    (
        "BlockNewLeaves6",
        (
            ("Huckleberry Leaves", "DeciduousBushLeaves"),
            ("Huckleberry Fruit Leaves", "HuckleberryFruitLeaves"),
            None,
            None,
            None,
            None,
            None,
            None,
        ),
    ),
)

LOG_TEXTURES: tuple[tuple[str, tuple[str | None, ...]], ...] = (
    (
        "BlockNewLogs",
        ("LogCherry", "LogIronwood", "LogEucalyptus", "LogSequoia", "LogAcacia", "LogJoshua", "LogSwampWillow", "LogPalm"),
    ),
    (
        "BlockNewLogs2",
        ("LogDesertWillow", "LogCedar", "LogGinkgo", "LogBeech", "LogWalnut", "LogCocoa", "LogCocoaFruit", "LogWTEucalyptus"),
    ),
    ("BlockNewLogs3", ("LogBukkit", "LogBanana", "LogOrange", "LogPeach", "LogLemon")),
    (
        "BlockNewLogs4",
        (
            "LogCherry", "LogDesertWillow", "LogIronwood", "LogCedar", "LogEucalyptus", "LogGinkgo", "LogSequoia", "LogBeech",
            "LogAcacia", "LogWalnut", "LogJoshua", None, "LogSwampWillow", None, "LogPalm", "LogWTEucalyptus",
        ),
    ),
)

FLOWERS = (
    ("Allium Drumstick", "AlliumDrumstick"),
    ("Bachelor's Button", "BachelorsButton"),
    ("Billy Buttons", "BillyButtons"),
    ("Delphinium Belladonna", "DelphiniumBelladonna"),
    ("Fernflower Yarrow", "FernflowerYarrow"),
    ("Gerbera Daisy", "GerberaDaisy"),
    ("Hydrangea", "Hydrangea"),
    ("Red Rover", "RedRover"),
    ("Snapdragon Magenta", "SnapdragonMagenta"),
    ("Star of Bethlehem", "StarOfBethlehem"),
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _source_index(sources: list[Any], sha256: str) -> int | None:
    return next((index for index, source in enumerate(sources) if source.sha256 == sha256), None)


def _assignment(assignments: list[dict[str, Any]], key: str) -> dict[str, Any] | None:
    matches = [item for item in assignments if item["key"] == key]
    if len(matches) != 1:
        return None
    return {"kind": "forge-config", **matches[0]}


def _member_evidence(source: Any, source_index: int, member: str, role: str) -> dict[str, Any]:
    raw = source.read(member)
    return {
        "kind": role,
        "source": source_index,
        "archiveSha256": source.sha256,
        "member": member,
        "memberSha256": _sha256(raw),
    }


def _texture_evidence(sources: list[Any], located: tuple[int, str]) -> dict[str, Any]:
    source_index, member = located
    return {
        "kind": "exact-texture",
        "source": source_index,
        "archiveSha256": sources[source_index].sha256,
        "member": member,
        "memberSha256": _sha256(sources[source_index].read(member)),
    }


def _fancy_graphics(root: Path) -> dict[str, Any] | None:
    matches = []
    for path in root.rglob("options.txt"):
        text = path.read_text(errors="replace")
        if "fancyGraphics:true" in text.splitlines():
            matches.append(path)
    if len(matches) != 1:
        return None
    path = matches[0]
    return {
        "kind": "client-option",
        "file": str(path.resolve()),
        "sha256": _sha256(path.read_bytes()),
        "setting": "fancyGraphics",
        "value": True,
    }


def _forgotten_nature(
    *,
    root: Path,
    sources: list[Any],
    assignments: list[dict[str, Any]],
    entries: dict[str, dict[str, Any]],
    locate_texture: Callable[[str], tuple[int, str] | None],
) -> dict[str, Any] | None:
    source_index = _source_index(sources, FORGOTTEN_NATURE_152_SHA256)
    leaf_config = _assignment(assignments, "leafIDindex")
    log_config = _assignment(assignments, "logIDindex")
    flower_config = _assignment(assignments, "FlowerID")
    graphics = _fancy_graphics(root)
    if source_index is None or leaf_config is None or log_config is None or flower_config is None or graphics is None:
        return None
    source = sources[source_index]
    main_member = "ForgottenNature/ForgottenNature.class"
    labels_member = "ForgottenNature/Proxy/FNClientProxy.class"
    required = [main_member, labels_member, "ForgottenNature/Blocks/BlockNewFlowers.class"]
    required += [f"ForgottenNature/Blocks/{name}.class" for name, _ in LEAF_VARIANTS + LOG_TEXTURES]
    if not all(source.has(member) for member in required):
        return None
    main = _member_evidence(source, source_index, main_member, "bundled-registration-bytecode")
    labels = _member_evidence(source, source_index, labels_member, "bundled-language-bytecode")
    applied = 0

    leaf_base = int(leaf_config["id"])
    for offset, (class_name, variants) in enumerate(LEAF_VARIANTS):
        class_evidence = _member_evidence(source, source_index, f"ForgottenNature/Blocks/{class_name}.class", "bundled-render-bytecode")
        class_evidence["methods"] = ["getIcon(int,int)", "registerIcons(IconRegister)"]
        class_evidence["mapping"] = "stored metadata above 7 subtracts 8; fancy graphics selects the non-Solid icon array"
        for variant_meta, variant in enumerate(variants):
            if variant is None:
                continue
            name, texture_stem = variant
            located = locate_texture(texture_stem)
            if located is None:
                continue
            for stored_meta in (variant_meta, variant_meta + 8):
                entries[f"legacy:{leaf_base + offset}:{stored_meta}"] = resolution_entry(
                    "exact",
                    name=name,
                    texture={"source": located[0], "member": located[1]},
                    reason="Exact bc3 config, bundled 1.5.2 registration/render bytecode, client graphics option, and supplied texture agree.",
                    provenance=[leaf_config, main, class_evidence, labels, graphics, _texture_evidence(sources, located)],
                )
                applied += 1

    log_base = int(log_config["id"])
    for offset, (class_name, bark_textures) in enumerate(LOG_TEXTURES):
        class_evidence = _member_evidence(source, source_index, f"ForgottenNature/Blocks/{class_name}.class", "bundled-render-bytecode")
        class_evidence["methods"] = ["getIcon(int,int)", "registerIcons(IconRegister)"]
        for metadata in range(16):
            if offset < 2:
                texture_stem = "LogCrossSection" if metadata % 2 == 0 else bark_textures[metadata // 2]
            elif offset == 2:
                texture_stem = "LogCrossSection" if metadata < 5 else bark_textures[metadata % 5]
            else:
                texture_stem = bark_textures[metadata]
            if texture_stem is None:
                continue
            located = locate_texture(texture_stem)
            if located is None:
                continue
            entries[f"legacy:{log_base + offset}:{metadata}"] = resolution_entry(
                "exact",
                name=f"Forgotten Nature {texture_stem}",
                texture={"source": located[0], "member": located[1]},
                reason="Exact bc3 config and bundled 1.5.2 top-face bytecode select this supplied texture for the stored metadata.",
                provenance=[log_config, main, class_evidence, _texture_evidence(sources, located)],
            )
            applied += 1

    flower_member = "ForgottenNature/Blocks/BlockNewFlowers.class"
    flower_evidence = _member_evidence(source, source_index, flower_member, "bundled-render-bytecode")
    flower_evidence["methods"] = ["getIcon(int,int)", "registerIcons(IconRegister)"]
    for metadata, (name, texture_stem) in enumerate(FLOWERS):
        located = locate_texture(texture_stem)
        if located is None:
            continue
        entries[f"legacy:{int(flower_config['id'])}:{metadata}"] = resolution_entry(
            "exact",
            name=name,
            texture={"source": located[0], "member": located[1]},
            reason="Exact bc3 config and bundled 1.5.2 metadata-indexed icon array select this supplied texture.",
            provenance=[flower_config, main, flower_evidence, labels, _texture_evidence(sources, located)],
        )
        applied += 1
    return {"name": "ForgottenNature 1.5.2 exact bc3", "archiveSha256": source.sha256, "entriesApplied": applied}


def _railcraft_hidden(
    *,
    sources: list[Any],
    assignments: list[dict[str, Any]],
    entries: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    source_index = _source_index(sources, RAILCRAFT_7230_SHA256)
    config = _assignment(assignments, "block.hidden")
    member = "mods/railcraft/common/blocks/hidden/BlockHidden.class"
    if source_index is None or config is None or not sources[source_index].has(member):
        return None
    evidence = _member_evidence(sources[source_index], source_index, member, "bundled-render-bytecode")
    evidence["methods"] = ["getRenderType()=-1", "isAirBlock(...)=true"]
    entries[f"legacy:{int(config['id'])}:*"] = resolution_entry(
        "exact",
        name="Railcraft Residual Heat",
        renderAsAir=True,
        reason="The exact bundled Railcraft class has render type -1 and reports itself as air; no appearance is inferred.",
        provenance=[config, evidence],
    )
    return {"name": "Railcraft 7.2.3.0 hidden block", "archiveSha256": sources[source_index].sha256, "entriesApplied": 1}


def apply_exact_legacy_resolvers(
    *,
    root: Path,
    sources: list[Any],
    assignments: list[dict[str, Any]],
    entries: dict[str, dict[str, Any]],
    locate_texture: Callable[[str], tuple[int, str] | None],
) -> list[dict[str, Any]]:
    """Apply only resolvers whose complete exact-pack fingerprint is recognized."""

    reports = []
    forgotten = _forgotten_nature(
        root=root,
        sources=sources,
        assignments=assignments,
        entries=entries,
        locate_texture=locate_texture,
    )
    if forgotten:
        reports.append(forgotten)
    railcraft = _railcraft_hidden(sources=sources, assignments=assignments, entries=entries)
    if railcraft:
        reports.append(railcraft)
    return reports
