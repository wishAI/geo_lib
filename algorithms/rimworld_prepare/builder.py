#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shlex
import shutil
import subprocess
import uuid
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[1]
OUTPUTS = ROOT / "outputs"
SOURCE = OUTPUTS / "source" / "yuran"
TEXTURES = SOURCE / "cont" / "Textures" / "Yuran" / "Yuranlike"
CONFIG = ROOT / "inputs" / "workspace.json"
CATALOG = OUTPUTS / "catalog.json"
MOD_OUTPUT = OUTPUTS / "DragonYuran"
CHARACTER_OUTPUT = OUTPUTS / "PrepareCarefully"
REMOTE_SOURCE = "/home/wishai/.local/share/Steam/steamapps/workshop/downloads/294100/2844129100/"
REMOTE_MOD = "/home/wishai/.local/share/Steam/steamapps/common/RimWorld/Mods/DragonYuran/"
REMOTE_SAVE = "/home/wishai/.config/unity3d/Ludeon Studios/RimWorld by Ludeon Studios/PrepareCarefully/"
REMOTE_HARMONY_DLL = "/home/wishai/.local/share/Steam/steamapps/common/RimWorld/Mods/Harmony/Current/Assemblies/0Harmony.dll"
REMOTE_HAR_DLL = "/home/wishai/.local/share/Steam/steamapps/common/RimWorld/Mods/HumanoidAlienRaces/1.6/Assemblies/AlienRace.dll"
REMOTE_PREPARE_CAREFULLY_DLL = "/home/wishai/.local/share/Steam/steamapps/common/RimWorld/Mods/EdBPrepareCarefully/1.6/Assemblies/EdBPrepareCarefully.dll"
DIRECTIONS = ("south", "east", "north", "west")
IMG2IMG_DIRECTIONS = ("south", "east", "north")
IMG2IMG_LAYERS = (
    ("body", "Body", "Bodies/Naked_Thin", "Bodies/Naked_Thin"),
    ("face", "Face · no hair", "Heads/Female_YR_head", "Heads/Female_YR_head"),
)
IMG2IMG_CELL = 640
IMG2IMG_GAP = 32
IMG2IMG_LEFT = 32
IMG2IMG_TOP = 64
IMG2IMG_SIZE = (
    IMG2IMG_LEFT * 2 + IMG2IMG_CELL * len(IMG2IMG_DIRECTIONS) + IMG2IMG_GAP * (len(IMG2IMG_DIRECTIONS) - 1),
    IMG2IMG_TOP + IMG2IMG_CELL + 32,
)
SKILLS = ("Animals", "Artistic", "Construction", "Cooking", "Crafting", "Intellectual", "Medical", "Melee", "Mining", "Plants", "Shooting", "Social")
TRAITS = {
    "Abrasive", "Ascetic", "Bloodlust", "Brawler", "Cannibal", "CarefulShooter", "FastLearner",
    "GreatMemory", "Greedy", "HardWorker", "Industrious", "IronWilled", "Jogger", "Kind",
    "Masochist", "Nimble", "Optimist", "PsychicallyDeaf", "PsychicallyHypersensitive", "Psychopath",
    "Sanguine", "TooSmart", "Tough", "TriggerHappy", "Undergrounder", "Wimp",
}
SAFE_ID = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,79}$")
SAFE_COLOR = re.compile(r"^#[0-9a-fA-F]{6}$")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _report_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _run(command: list[str], *, timeout: int = 180) -> subprocess.CompletedProcess[str]:
    print("$", " ".join(command), flush=True)
    completed = subprocess.run(command, text=True, capture_output=True, timeout=timeout, check=False)
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.returncode:
        raise RuntimeError(completed.stderr.strip() or f"command failed with exit {completed.returncode}")
    return completed


def sync_source() -> dict:
    SOURCE.mkdir(parents=True, exist_ok=True)
    _run([
        "rsync", "-a", "--partial", "-e", "ssh -o BatchMode=yes -o ConnectTimeout=10",
        f"tk2:{REMOTE_SOURCE}", str(SOURCE) + "/",
    ], timeout=360)
    catalog = build_catalog()
    return {"status": "success", "source": REMOTE_SOURCE, "fileCount": sum(1 for path in SOURCE.rglob("*") if path.is_file()), "catalog": catalog}


def _category(relative: Path) -> str:
    first = relative.parts[0] if relative.parts else "other"
    return {
        "Bodies": "body", "Heads": "head", "Ear": "ear", "tail": "tail",
        "Hairs": "hair", "AddonHair": "hair", "Apparel": "apparel",
        "Stump": "body", "Wounds": "body",
    }.get(first, "other")


def _directional_group(relative: Path) -> tuple[str, str | None]:
    match = re.match(r"^(.*)_(south|north|east|west)(m?)$", relative.stem, re.IGNORECASE)
    if not match:
        return relative.with_suffix("").as_posix(), None
    base, direction, mask = match.groups()
    key = (relative.parent / (base + ("_mask" if mask else ""))).as_posix()
    return key, direction.lower()


def _asset_catalog() -> list[dict]:
    groups: dict[str, dict] = {}
    if not TEXTURES.is_dir():
        return []
    for path in sorted(TEXTURES.rglob("*.png")):
        relative = path.relative_to(TEXTURES)
        category = _category(relative)
        if category == "other" or any(part in {"BlackSnake", "Miko", "NotUse", "OLD", "Tra"} for part in relative.parts):
            continue
        key, direction = _directional_group(relative)
        item = groups.setdefault(key, {"id": key, "category": category, "label": Path(key).name, "files": {}, "byteSize": 0})
        item["byteSize"] += path.stat().st_size
        if direction:
            item["files"][direction] = str(path.relative_to(ROOT))
        else:
            item["files"]["icon"] = str(path.relative_to(ROOT))
    assets = []
    for item in groups.values():
        files = item["files"]
        preview = files.get("south") or files.get("east") or files.get("north") or files.get("icon")
        if not preview:
            continue
        item["preview"] = preview
        item["directions"] = sorted(key for key in files if key != "icon")
        assets.append(item)
    return sorted(assets, key=lambda item: (item["category"], item["id"]))


def _all_apparel_defs() -> tuple[dict[str, ET.Element], dict[str, ET.Element]]:
    concrete: dict[str, ET.Element] = {}
    named: dict[str, ET.Element] = {}
    defs_root = SOURCE / "1.6" / "Defs" / "ThingDefs" / "Apparel"
    for xml_path in sorted(defs_root.glob("*.xml")):
        try:
            root = ET.parse(xml_path).getroot()
        except ET.ParseError:
            continue
        for node in root.findall("ThingDef"):
            if node.get("Name"):
                named[node.get("Name", "")] = node
            def_name = node.findtext("defName")
            if def_name:
                concrete[def_name] = node
    return concrete, named


def _inherited_text(node: ET.Element, named: dict[str, ET.Element], path: str) -> str | None:
    value = node.findtext(path)
    if value:
        return value.strip()
    parent = named.get(node.get("ParentName", ""))
    return _inherited_text(parent, named, path) if parent is not None else None


def _inherited_node(node: ET.Element, named: dict[str, ET.Element], path: str) -> ET.Element | None:
    value = node.find(path)
    if value is not None:
        return value
    parent = named.get(node.get("ParentName", ""))
    return _inherited_node(parent, named, path) if parent is not None else None


def _inherited_list(node: ET.Element, named: dict[str, ET.Element], path: str) -> list[str]:
    value = node.find(path)
    if value is not None and list(value):
        return [str(item.text or "").strip() for item in value.findall("li") if str(item.text or "").strip()]
    parent = named.get(node.get("ParentName", ""))
    return _inherited_list(parent, named, path) if parent is not None else []


def _apparel_catalog() -> list[dict]:
    concrete, named = _all_apparel_defs()
    catalog = []
    for def_name, node in concrete.items():
        if node.find("apparel") is None and _inherited_node(node, named, "apparel") is None:
            continue
        worn = _inherited_text(node, named, "apparel/wornGraphicPath")
        if not worn or not worn.startswith("Yuran/Yuranlike/Apparel/") or worn.endswith("/Non"):
            continue
        relative_base = worn.removeprefix("Yuran/Yuranlike/")
        candidates = sorted(TEXTURES.glob(relative_base + "_*.png"))
        preview = next((path for path in candidates if path.stem.endswith("_south")), candidates[0] if candidates else None)
        if preview is None:
            continue
        catalog.append({
            "defName": def_name,
            "label": (node.findtext("label") or def_name).strip(),
            "description": (node.findtext("description") or "").strip(),
            "wornGraphicPath": worn,
            "layers": _inherited_list(node, named, "apparel/layers") or ["OnSkin"],
            "bodyPartGroups": _inherited_list(node, named, "apparel/bodyPartGroups") or ["Torso"],
            "preview": str(preview.relative_to(ROOT)),
            "source": "APHidden.xml" if "Addon_" in def_name or "Demigod" in def_name else "AP.xml",
        })
    return sorted(catalog, key=lambda item: (item["source"], item["label"].lower()))


def _direction_path(base: str, direction: str) -> Path | None:
    candidate = TEXTURES / f"{base}_{direction}.png"
    if candidate.is_file():
        return candidate
    if direction == "west":
        east = TEXTURES / f"{base}_east.png"
        if east.is_file():
            return east
    return None


def _load_layer(base: str, direction: str, *, offset: tuple[int, int] = (0, 0), mirror_west: bool = True) -> Image.Image:
    path = _direction_path(base, direction)
    layer = Image.new("RGBA", (256, 256))
    if path is None:
        return layer
    source = Image.open(path).convert("RGBA")
    if direction == "west" and mirror_west and not path.stem.endswith("_west"):
        source = source.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    layer.alpha_composite(source, offset)
    return layer


def compose_yuran(direction: str, *, dragon_skin: bool = False) -> Image.Image:
    if direction not in DIRECTIONS:
        raise ValueError("invalid direction")
    canvas = Image.new("RGBA", (256, 256))
    body_base = "Bodies/Naked_Thin"
    head_base = "Heads/Female_YR_head"
    generated = MOD_OUTPUT / "Textures" / "DragonYuran" / "DragonSkin"
    if dragon_skin and (generated / f"Bodies/Naked_Thin_{direction}.png").is_file():
        body = Image.open(generated / f"Bodies/Naked_Thin_{direction}.png").convert("RGBA")
    else:
        body = _load_layer(body_base, direction)
    # Mirror RimWorld 1.6 + HAR's actual transform chain.  Humanlike pawn
    # textures span a 1.5-world-unit mesh, hence 256/1.5 px per world unit:
    # Thin.headOffset=(0.09,0.34), Yuran head directional=(0,-0.075), then
    # each addon's named default offset plus its directional offset.  East
    # addon X is inverted by AlienPawnRenderNodeWorker_BodyAddon.
    tail_offsets = {
        "south": (72, 38), "north": (0, 2),
        "east": (5, -10), "west": (-5, -10),
    }
    canvas.alpha_composite(_load_layer("tail/YR_tail", direction, offset=tail_offsets[direction]))
    canvas.alpha_composite(body)
    head_shift = {
        "south": (0, -45), "north": (0, -45),
        "east": (15, -45), "west": (-15, -45),
    }[direction]
    canvas.alpha_composite(_load_layer(head_base, direction, offset=head_shift))
    offsets = {
        "south": ((0, -57), (2, -57)),
        "north": ((0, -57), (0, -57)),
        "east": ((2, -54), (2, -54)),
        "west": ((-2, -54), (-2, -54)),
    }
    left_offset, right_offset = offsets[direction]
    canvas.alpha_composite(_load_layer("Ear/L/YR_earL", direction, offset=left_offset))
    canvas.alpha_composite(_load_layer("Ear/R/YR_earR", direction, offset=right_offset))
    canvas.alpha_composite(_load_layer("AddonHair/YR_hair", direction, offset=head_shift))
    return canvas


def _make_sheets() -> dict:
    preview_dir = OUTPUTS / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)
    background = (19, 34, 31, 255)
    sheet = Image.new("RGBA", (4 * 288, 326), (244, 238, 222, 255))
    draw = ImageDraw.Draw(sheet)
    for index, direction in enumerate(DIRECTIONS):
        x = 16 + index * 288
        panel = Image.new("RGBA", (256, 256), background)
        panel.alpha_composite(compose_yuran(direction))
        sheet.alpha_composite(panel, (x, 42))
        draw.text((x + 8, 14), direction.upper(), fill=(24, 55, 47, 255))
    whole = preview_dir / "yuran_whole_four_views.png"
    sheet.save(whole)

    skin = Image.new("RGBA", IMG2IMG_SIZE, (239, 234, 223, 255))
    skin_draw = ImageDraw.Draw(skin)
    assets = []
    skin_draw.text((IMG2IMG_LEFT, IMG2IMG_TOP - 28), "JOINED BODY + HAIRLESS FACE · NO HAIR OR EARS", fill=(48, 62, 57, 255))
    for column, direction in enumerate(IMG2IMG_DIRECTIONS):
        x = IMG2IMG_LEFT + column * (IMG2IMG_CELL + IMG2IMG_GAP)
        y = IMG2IMG_TOP
        head_shift = {
            "south": (0, -45), "east": (15, -45), "north": (0, -45),
        }[direction]
        composite = Image.new("RGBA", (256, 256))
        layers = []
        for kind, label, source_base, output_base in IMG2IMG_LAYERS:
            source_path = _direction_path(source_base, direction)
            if source_path is None:
                raise RuntimeError(f"missing img2img source: {source_base}_{direction}.png")
            original = Image.open(source_path).convert("RGBA")
            source_bounds = original.getchannel("A").getbbox()
            if source_bounds is None:
                raise RuntimeError(f"empty img2img source: {source_path}")
            layer_offset = head_shift if kind == "face" else (0, 0)
            composite.alpha_composite(original, layer_offset)
            layers.append({
                "id": f"{kind}-{direction}", "kind": kind, "label": f"{label} · {direction.title()}",
                "source": str(source_path.relative_to(ROOT)), "sourceBounds": list(source_bounds),
                "compositeOffset": list(layer_offset),
                "gameOutput": f"Textures/DragonYuran/DragonSkin/{output_base}_{direction}.png",
            })
        composite_bounds = composite.getchannel("A").getbbox()
        if composite_bounds is None:
            raise RuntimeError(f"empty img2img composite: {direction}")
        cropped = composite.crop(composite_bounds)
        available = IMG2IMG_CELL - 48
        scale = min(available / cropped.width, available / cropped.height)
        render_size = (max(1, round(cropped.width * scale)), max(1, round(cropped.height * scale)))
        enlarged = cropped.resize(render_size, Image.Resampling.LANCZOS)
        offset = ((IMG2IMG_CELL - render_size[0]) // 2, (IMG2IMG_CELL - render_size[1]) // 2)
        panel = Image.new("RGBA", (IMG2IMG_CELL, IMG2IMG_CELL), (250, 248, 243, 255))
        panel.alpha_composite(enlarged, offset)
        skin.alpha_composite(panel, (x, y))
        skin_draw.rectangle((x, y, x + IMG2IMG_CELL - 1, y + IMG2IMG_CELL - 1), outline=(184, 171, 152, 255), width=2)
        skin_draw.text((x + 14, y + 14), direction.upper() + (" · MIRRORS WEST" if direction == "east" else ""), fill=(49, 68, 62, 255))
        assets.append({
            "id": f"joined-{direction}", "label": f"Joined body + face · {direction.title()}",
            "direction": direction, "compositeBounds": list(composite_bounds),
            "sheetBox": [x, y, IMG2IMG_CELL, IMG2IMG_CELL],
            "contentBox": [x + offset[0], y + offset[1], *render_size], "layers": layers,
        })
    skin_path = preview_dir / "yuran_skin_img2img.png"
    skin.save(skin_path)
    metadata = {
        "version": 3,
        "description": "Tightly framed, engine-aligned Yuran body plus hairless face composites for continuous img2img editing. The importer projects each joined result back into separate body and head game textures.",
        "canvasSize": list(IMG2IMG_SIZE),
        "directions": list(IMG2IMG_DIRECTIONS),
        "westRule": "RimWorld mirrors the east texture for west-facing pawns.",
        "assets": assets,
    }
    (preview_dir / "yuran_skin_img2img.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return {
        "whole": str(whole.relative_to(ROOT)), "skin": str(skin_path.relative_to(ROOT)),
        "skinMetadata": str((preview_dir / "yuran_skin_img2img.json").relative_to(ROOT)),
    }


def build_catalog() -> dict:
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    sheets = _make_sheets() if TEXTURES.is_dir() else {}
    payload = {
        "version": 1,
        "source": {"workshopId": "2844129100", "version": "1.6", "path": str(SOURCE.relative_to(ROOT)), "observedAt": now()},
        "rendering": {
            "kind": "directional-layer-composite",
            "directions": list(DIRECTIONS),
            "westFallback": "mirror east when no west texture exists",
            "coordinateModel": "RimWorld 1.6 Thin.headOffset + Yuran 1.6 headOffsetDirectional + HAR named/directional addon offsets on the 1.5-world-unit humanlike mesh",
            "headPixelOffsets": {"south": [0, -45], "east": [15, -45], "north": [0, -45], "west": [-15, -45]},
            "earPixelOffsets": {"south": [[0, -57], [2, -57]], "east": [[2, -54], [2, -54]], "north": [[0, -57], [0, -57]], "west": [[-2, -54], [-2, -54]]},
            "fact": "Prepare Carefully calls RimWorld PortraitsCache.Get with pawn, direction, portrait size, apparel and hat flags.",
            "limitation": "The WebGUI is a texture-layer preview, not a captured Unity PortraitsCache frame. Launching RimWorld was intentionally not used for validation."
        },
        "assets": _asset_catalog(),
        "apparel": _apparel_catalog(),
        "sheets": sheets,
    }
    CATALOG.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {"assetCount": len(payload["assets"]), "apparelCount": len(payload["apparel"]), **sheets}


def _validate_name(value: object, label: str, *, allow_empty: bool = False) -> str:
    text = str(value or "").strip()
    if not allow_empty and not text:
        raise ValueError(f"{label} is required")
    if len(text) > 40 or any(character in text for character in "<>/&"):
        raise ValueError(f"{label} is invalid")
    return text


def validate_config(payload: object, *, catalog: dict | None = None) -> dict:
    if not isinstance(payload, dict) or payload.get("version") != 1:
        raise ValueError("workspace must be a version 1 object")
    mod = payload.get("mod")
    characters = payload.get("characters")
    if not isinstance(mod, dict) or not isinstance(characters, list) or not 1 <= len(characters) <= 24:
        raise ValueError("workspace requires mod and 1-24 characters")
    package_id = str(mod.get("packageId", ""))
    if not SAFE_ID.fullmatch(package_id):
        raise ValueError("mod.packageId is invalid")
    _validate_name(mod.get("name"), "mod.name")
    selected = mod.get("selectedApparel", [])
    if not isinstance(selected, list) or len(selected) > 48 or any(not SAFE_ID.fullmatch(str(item)) for item in selected):
        raise ValueError("mod.selectedApparel is invalid")
    allowed_apparel = {item["defName"] for item in (catalog or {}).get("apparel", [])}
    if allowed_apparel and not set(selected).issubset(allowed_apparel):
        raise ValueError("mod.selectedApparel contains an unknown Yuran apparel def")
    ids: set[str] = set()
    for index, character in enumerate(characters):
        if not isinstance(character, dict):
            raise ValueError(f"characters[{index}] must be an object")
        identifier = str(character.get("id", ""))
        if not SAFE_ID.fullmatch(identifier) or identifier in ids:
            raise ValueError(f"characters[{index}].id is invalid or duplicated")
        ids.add(identifier)
        name = character.get("name", {})
        if not isinstance(name, dict):
            raise ValueError(f"characters[{index}].name is invalid")
        _validate_name(name.get("first"), f"characters[{index}].name.first")
        _validate_name(name.get("nick"), f"characters[{index}].name.nick", allow_empty=True)
        _validate_name(name.get("last"), f"characters[{index}].name.last")
        if character.get("gender") != "Female":
            raise ValueError("Dragon Yuran currently supports female pawns only")
        for age_key in ("biologicalAge", "chronologicalAge"):
            age = character.get(age_key)
            if not isinstance(age, int) or isinstance(age, bool) or not 13 <= age <= 1000:
                raise ValueError(f"characters[{index}].{age_key} must be 13-1000")
        if character.get("direction", "south") not in DIRECTIONS:
            raise ValueError(f"characters[{index}].direction is invalid")
        for color_key in ("skinColor", "hairColor"):
            if not SAFE_COLOR.fullmatch(str(character.get(color_key, ""))):
                raise ValueError(f"characters[{index}].{color_key} is invalid")
        apparel = character.get("apparel", [])
        if not isinstance(apparel, list) or not set(apparel).issubset(set(selected)):
            raise ValueError(f"characters[{index}].apparel must use included apparel")
        traits = character.get("traits", [])
        if not isinstance(traits, list) or len(traits) > 8:
            raise ValueError(f"characters[{index}].traits is invalid")
        for trait in traits:
            if not isinstance(trait, dict) or trait.get("def") not in TRAITS or not isinstance(trait.get("degree", 0), int):
                raise ValueError(f"characters[{index}] contains an unsupported trait")
        skills = character.get("skills", {})
        if not isinstance(skills, dict) or set(skills) != set(SKILLS):
            raise ValueError(f"characters[{index}] must define the 12 RimWorld skills")
        for name, value in skills.items():
            if not isinstance(value, dict) or not isinstance(value.get("level"), int) or not 0 <= value["level"] <= 20:
                raise ValueError(f"characters[{index}].skills.{name}.level is invalid")
            if value.get("passion") not in {"None", "Minor", "Major"}:
                raise ValueError(f"characters[{index}].skills.{name}.passion is invalid")
    if payload.get("selectedCharacterId") not in ids:
        raise ValueError("selectedCharacterId does not identify a character")
    return payload


def load_config(path: Path = CONFIG) -> dict:
    catalog = json.loads(CATALOG.read_text(encoding="utf-8")) if CATALOG.is_file() else None
    return validate_config(json.loads(path.read_text(encoding="utf-8")), catalog=catalog)


def _clone_xml(element: ET.Element) -> ET.Element:
    return ET.fromstring(ET.tostring(element, encoding="utf-8"))


def _xml_text(element: ET.Element) -> str:
    ET.indent(element, space="  ")
    return '<?xml version="1.0" encoding="utf-8"?>\n' + ET.tostring(element, encoding="unicode") + "\n"


def _write_about(config: dict) -> None:
    about = MOD_OUTPUT / "About"
    about.mkdir(parents=True, exist_ok=True)
    root = ET.Element("ModMetaData")
    ET.SubElement(root, "name").text = config["mod"]["name"]
    ET.SubElement(root, "author").text = config["mod"].get("author", "Hao Wang")
    versions = ET.SubElement(root, "supportedVersions")
    ET.SubElement(versions, "li").text = "1.6"
    ET.SubElement(root, "packageId").text = config["mod"]["packageId"]
    ET.SubElement(root, "description").text = "A clean, race-only Yuran derivative with selectable apparel and an installable dragon-skin texture swap. Buildings, factions, fiction, weapons, research, Miko and Shikigami content are intentionally excluded."
    dependencies = ET.SubElement(root, "modDependencies")
    dependency = ET.SubElement(dependencies, "li")
    ET.SubElement(dependency, "packageId").text = "erdelf.HumanoidAlienRaces"
    ET.SubElement(dependency, "displayName").text = "Humanoid Alien Races"
    ET.SubElement(dependency, "steamWorkshopUrl").text = "https://steamcommunity.com/sharedfiles/filedetails/?id=839005762"
    (about / "About.xml").write_text(_xml_text(root), encoding="utf-8")


def _write_race_defs() -> None:
    defs = ET.Element("Defs")
    head_base = ET.SubElement(defs, "HeadTypeDef", {"ParentName": "AverageBase", "Name": "DragonYuran_HeadBase", "Abstract": "True"})
    ET.SubElement(head_base, "gender").text = "Female"
    head = ET.SubElement(defs, "HeadTypeDef", {"ParentName": "DragonYuran_HeadBase"})
    ET.SubElement(head, "defName").text = "DragonYuran_Female_AverageNormal"
    ET.SubElement(head, "graphicPath").text = "DragonYuran/Heads/Female_YR_head"

    race = ET.SubElement(defs, "AlienRace.ThingDef_AlienRace", {"ParentName": "BasePawn"})
    ET.SubElement(race, "defName").text = "DragonYuran_Race"
    ET.SubElement(race, "label").text = "dragon yuran"
    ET.SubElement(race, "description").text = "A Yuran with a modular cybernetic dragon-skin system."
    stats = ET.SubElement(race, "statBases")
    for key, value in {"MarketValue": "1200", "MoveSpeed": "4.7", "Mass": "60", "CarryingCapacity": "75", "PainShockThreshold": "0.75", "SocialImpact": "1.25"}.items():
        ET.SubElement(stats, key).text = value
    race_props = ET.SubElement(race, "race")
    for key, value in {
        "thinkTreeMain": "Humanlike", "thinkTreeConstant": "HumanlikeConstant", "intelligence": "Humanlike",
        "renderTree": "Humanlike", "lifeExpectancy": "100", "nameCategory": "HumanStandard", "body": "Human",
        "baseBodySize": "0.9", "baseHealthScale": "0.95", "foodType": "OmnivoreHuman",
    }.items():
        ET.SubElement(race_props, key).text = value
    life_stages = ET.SubElement(race_props, "lifeStageAges")
    for stage, age in (("HumanlikeBaby", "0"), ("HumanlikeChild", "3"), ("HumanlikeAdult", "13")):
        attributes = {"Class": "AlienRace.LifeStageAgeAlien"} if stage == "HumanlikeAdult" else {}
        item = ET.SubElement(life_stages, "li", attributes)
        ET.SubElement(item, "def").text = stage
        ET.SubElement(item, "minAge").text = age
    alien = ET.SubElement(race, "alienRace")
    general = ET.SubElement(alien, "generalSettings")
    ET.SubElement(general, "maleGenderProbability").text = "0.0000001"
    ET.SubElement(general, "minAgeForAdulthood").text = "13"
    ET.SubElement(general, "humanRecipeImport").text = "true"
    generator = ET.SubElement(general, "alienPartGenerator")
    ET.SubElement(generator, "atlasScale").text = "4"
    heads = ET.SubElement(generator, "headTypes")
    ET.SubElement(heads, "li").text = "DragonYuran_Female_AverageNormal"
    bodies = ET.SubElement(generator, "bodyTypes")
    ET.SubElement(bodies, "li").text = "Thin"
    ET.SubElement(generator, "customDrawSize").text = "(1,1)"
    ET.SubElement(generator, "customHeadDrawSize").text = "(1,1)"
    ET.SubElement(generator, "customPortraitDrawSize").text = "(1,1)"
    head_offsets = ET.SubElement(generator, "headOffsetDirectional")
    for direction in DIRECTIONS:
        directional = ET.SubElement(head_offsets, direction)
        ET.SubElement(directional, "offset").text = "(0,-0.075)"
    channels = ET.SubElement(generator, "colorChannels", {"Inherit": "False"})
    for channel, color in (("skin", "(250,250,250)"), ("hair", "(255,255,255)")):
        item = ET.SubElement(channels, "li")
        ET.SubElement(item, "name").text = channel
        first = ET.SubElement(item, "first", {"Class": "ColorGenerator_Single"})
        ET.SubElement(first, "color").text = color
        second = ET.SubElement(item, "second", {"Class": "ColorGenerator_Single"})
        ET.SubElement(second, "color").text = color
    addons = ET.SubElement(generator, "bodyAddons")
    addon_specs = [
        ("DragonYuran/AddonHair/YR_hair", True, True, {
            "south": ("(0,0)", "-0.274"), "north": ("(0,0)", "-0.326"),
            "east": ("(0,0)", "-0.274"), "west": ("(0,0)", "-0.274"),
        }),
        ("DragonYuran/Ear/L/YR_earL", True, True, {
            "south": ("(-0.42,0.29)", "-0.27"), "north": ("(0,0.62)", "-0.33"),
            "east": ("(-0.34,0.27)", "-0.3"), "west": ("(-0.34,0.27)", "-0.269"),
        }),
        ("DragonYuran/Ear/R/YR_earR", True, True, {
            "south": ("(-0.41,0.29)", "-0.27"), "north": ("(0,0.62)", "-0.33"),
            "east": ("(-0.34,0.27)", "-0.269"), "west": ("(-0.34,0.27)", "-0.3"),
        }),
        ("DragonYuran/tail/YR_tail", True, False, {
            "south": ("(0,0)", "-0.319"), "north": ("(0,0.54)", "-0.5"),
            "east": ("(-0.45,0.28)", "-0.319"), "west": ("(-0.45,0.28)", "-0.319"),
        }),
    ]
    for path, front, head_aligned, directional_offsets in addon_specs:
        item = ET.SubElement(addons, "li")
        ET.SubElement(item, "path").text = path
        ET.SubElement(item, "inFrontOfBody").text = str(front).lower()
        ET.SubElement(item, "alignWithHead").text = str(head_aligned).lower()
        ET.SubElement(item, "colorChannel").text = "base"
        ET.SubElement(item, "drawSize").text = "1"
        ET.SubElement(item, "shaderType").text = "Transparent"
        offsets = ET.SubElement(item, "offsets")
        for direction, (offset, layer_offset) in directional_offsets.items():
            directional = ET.SubElement(offsets, direction)
            ET.SubElement(directional, "offset").text = offset
            ET.SubElement(directional, "layerOffset").text = layer_offset
    graphics = ET.SubElement(alien, "graphicPaths")
    graphic = ET.SubElement(graphics, "li")
    ET.SubElement(graphic, "skinColor").text = "(1,1,1,1)"
    ET.SubElement(graphic, "skinShader").text = "Cutout"
    head_graphic = ET.SubElement(graphic, "head")
    ET.SubElement(head_graphic, "path").text = "DragonYuran/Heads/Female_YR_head"
    head_extended = ET.SubElement(head_graphic, "extendedGraphics")
    ET.SubElement(head_extended, "Hediff", {"For": "DragonYuran_DragonSkin"}).text = "DragonYuran/DragonSkin/Heads/Female_YR_head"
    body_graphic = ET.SubElement(graphic, "body")
    ET.SubElement(body_graphic, "path").text = "DragonYuran/Bodies/Naked_Thin"
    body_extended = ET.SubElement(body_graphic, "extendedGraphics")
    ET.SubElement(body_extended, "Hediff", {"For": "DragonYuran_DragonSkin"}).text = "DragonYuran/DragonSkin/Bodies/Naked_Thin"

    kind = ET.SubElement(defs, "PawnKindDef")
    ET.SubElement(kind, "defName").text = "DragonYuran_Colonist"
    ET.SubElement(kind, "label").text = "dragon yuran colonist"
    ET.SubElement(kind, "race").text = "DragonYuran_Race"
    ET.SubElement(kind, "combatPower").text = "45"
    ET.SubElement(kind, "apparelMoney").text = "0~0"
    ET.SubElement(kind, "weaponMoney").text = "0~0"

    settings = ET.SubElement(defs, "AlienRace.RaceSettings")
    ET.SubElement(settings, "defName").text = "DragonYuran_RaceSettings"
    pawn_settings = ET.SubElement(settings, "pawnKindSettings")
    starting = ET.SubElement(pawn_settings, "startingColonists")
    start = ET.SubElement(starting, "li")
    entries = ET.SubElement(start, "pawnKindEntries")
    entry = ET.SubElement(entries, "li")
    kinds = ET.SubElement(entry, "kindDefs")
    ET.SubElement(kinds, "li").text = "DragonYuran_Colonist"
    ET.SubElement(entry, "chance").text = "100"
    factions = ET.SubElement(start, "factionDefs")
    ET.SubElement(factions, "li").text = "PlayerColony"
    ET.SubElement(factions, "li").text = "PlayerTribe"
    defs_path = MOD_OUTPUT / "1.6" / "Defs"
    defs_path.mkdir(parents=True, exist_ok=True)
    (defs_path / "DragonYuran_Race.xml").write_text(_xml_text(defs), encoding="utf-8")


def _write_dragon_skin_defs() -> None:
    defs = ET.Element("Defs")
    item = ET.SubElement(defs, "ThingDef", {"ParentName": "BodyPartBase"})
    ET.SubElement(item, "defName").text = "DragonYuran_DragonSkinItem"
    ET.SubElement(item, "label").text = "cybernetic dragon skin"
    ET.SubElement(item, "description").text = "A full-body scale lattice that activates the Dragon Yuran alternate skin graphics."
    graphic = ET.SubElement(item, "graphicData")
    ET.SubElement(graphic, "texPath").text = "Things/Item/Health/HealthItem"
    ET.SubElement(graphic, "graphicClass").text = "Graphic_Single"
    stats = ET.SubElement(item, "statBases")
    ET.SubElement(stats, "MarketValue").text = "2400"
    ET.SubElement(stats, "Mass").text = "4"
    hediff = ET.SubElement(defs, "HediffDef", {"ParentName": "AddedBodyPartBase"})
    ET.SubElement(hediff, "defName").text = "DragonYuran_DragonSkin"
    ET.SubElement(hediff, "label").text = "dragon skin"
    ET.SubElement(hediff, "description").text = "A cybernetic scale skin installed across the torso."
    ET.SubElement(hediff, "spawnThingOnRemoved").text = "DragonYuran_DragonSkinItem"
    props = ET.SubElement(hediff, "addedPartProps")
    ET.SubElement(props, "partEfficiency").text = "1"
    stages = ET.SubElement(hediff, "stages")
    stage = ET.SubElement(stages, "li")
    offsets = ET.SubElement(stage, "statOffsets")
    ET.SubElement(offsets, "ArmorRating_Sharp").text = "0.18"
    ET.SubElement(offsets, "ArmorRating_Heat").text = "0.10"
    recipe = ET.SubElement(defs, "RecipeDef")
    ET.SubElement(recipe, "defName").text = "InstallDragonYuranDragonSkin"
    ET.SubElement(recipe, "label").text = "install dragon skin"
    ET.SubElement(recipe, "description").text = "Install the cybernetic dragon-skin lattice."
    ET.SubElement(recipe, "jobString").text = "Installing dragon skin."
    ingredients = ET.SubElement(recipe, "ingredients")
    ingredient = ET.SubElement(ingredients, "li")
    ET.SubElement(ingredient, "count").text = "1"
    filt = ET.SubElement(ingredient, "filter")
    thing_defs = ET.SubElement(filt, "thingDefs")
    ET.SubElement(thing_defs, "li").text = "DragonYuran_DragonSkinItem"
    fixed = ET.SubElement(recipe, "appliedOnFixedBodyParts")
    ET.SubElement(fixed, "li").text = "Torso"
    ET.SubElement(recipe, "addsHediff").text = "DragonYuran_DragonSkin"
    users = ET.SubElement(recipe, "recipeUsers")
    ET.SubElement(users, "li").text = "DragonYuran_Race"
    ET.SubElement(recipe, "workAmount").text = "3200"
    ET.SubElement(recipe, "surgerySuccessChanceFactor").text = "0.85"
    defs_path = MOD_OUTPUT / "1.6" / "Defs"
    (defs_path / "DragonYuran_DragonSkin.xml").write_text(_xml_text(defs), encoding="utf-8")


def _write_apparel_defs(config: dict, catalog: dict) -> list[dict]:
    by_def = {item["defName"]: item for item in catalog["apparel"]}
    root = ET.Element("Defs")
    built = []
    for original in config["mod"]["selectedApparel"]:
        source = by_def[original]
        def_name = "DragonYuran_" + original.removeprefix("YR_")
        thing = ET.SubElement(root, "ThingDef", {"ParentName": "ApparelMakeableBase"})
        ET.SubElement(thing, "defName").text = def_name
        ET.SubElement(thing, "label").text = source["label"]
        ET.SubElement(thing, "description").text = source["description"] or f"Dragon Yuran adaptation of {source['label']}."
        graphic = ET.SubElement(thing, "graphicData")
        ET.SubElement(graphic, "texPath").text = source["wornGraphicPath"].replace("Yuran/Yuranlike/", "DragonYuran/")
        ET.SubElement(graphic, "graphicClass").text = "Graphic_Single"
        stats = ET.SubElement(thing, "statBases")
        ET.SubElement(stats, "MaxHitPoints").text = "120"
        ET.SubElement(stats, "MarketValue").text = "160"
        ET.SubElement(stats, "Mass").text = "1.2"
        ET.SubElement(stats, "Insulation_Cold").text = "12"
        ET.SubElement(stats, "Insulation_Heat").text = "6"
        ET.SubElement(thing, "costStuffCount").text = "60"
        apparel = ET.SubElement(thing, "apparel")
        ET.SubElement(apparel, "wornGraphicPath").text = source["wornGraphicPath"].replace("Yuran/Yuranlike/", "DragonYuran/")
        body_groups = ET.SubElement(apparel, "bodyPartGroups")
        for group in source["bodyPartGroups"]:
            ET.SubElement(body_groups, "li").text = group
        layers = ET.SubElement(apparel, "layers")
        for layer in source["layers"]:
            ET.SubElement(layers, "li").text = layer
        tags = ET.SubElement(apparel, "tags")
        ET.SubElement(tags, "li").text = "DragonYuran"
        recipe = ET.SubElement(thing, "recipeMaker")
        ET.SubElement(recipe, "workSpeedStat").text = "TailoringSpeed"
        ET.SubElement(recipe, "workSkill").text = "Crafting"
        ET.SubElement(recipe, "effectWorking").text = "Tailor"
        ET.SubElement(recipe, "soundWorking").text = "Recipe_Tailor"
        recipe_users = ET.SubElement(recipe, "recipeUsers")
        ET.SubElement(recipe_users, "li").text = "HandTailoringBench"
        ET.SubElement(recipe_users, "li").text = "ElectricTailoringBench"
        built.append({"sourceDef": original, "defName": def_name, "label": source["label"]})
    defs_path = MOD_OUTPUT / "1.6" / "Defs"
    (defs_path / "DragonYuran_Apparel.xml").write_text(_xml_text(root), encoding="utf-8")
    return built


def _copy_direction_set(source_base: Path, target_base: Path) -> list[str]:
    copied = []
    target_base.parent.mkdir(parents=True, exist_ok=True)
    for direction in DIRECTIONS:
        source = source_base.with_name(source_base.name + f"_{direction}.png")
        mirror = False
        if not source.is_file() and direction == "west":
            source = source_base.with_name(source_base.name + "_east.png")
            mirror = source.is_file()
        if not source.is_file():
            continue
        target = target_base.with_name(target_base.name + f"_{direction}.png")
        if mirror:
            Image.open(source).convert("RGBA").transpose(Image.Transpose.FLIP_LEFT_RIGHT).save(target)
        else:
            shutil.copy2(source, target)
        copied.append(str(target.relative_to(MOD_OUTPUT)))
    return copied


def _copy_textures(config: dict, catalog: dict) -> list[str]:
    target = MOD_OUTPUT / "Textures" / "DragonYuran"
    copied = []
    for base in ("Bodies/Naked_Thin", "Heads/Female_YR_head", "AddonHair/YR_hair", "Ear/L/YR_earL", "Ear/R/YR_earR", "tail/YR_tail"):
        copied.extend(_copy_direction_set(TEXTURES / base, target / base))
    by_def = {item["defName"]: item for item in catalog["apparel"]}
    for def_name in config["mod"]["selectedApparel"]:
        source = by_def[def_name]
        relative = source["wornGraphicPath"].removeprefix("Yuran/Yuranlike/")
        worn_relative = relative if "Overhead" in source["layers"] else relative + "_Thin"
        copied.extend(_copy_direction_set(TEXTURES / worn_relative, target / worn_relative))
        icon = TEXTURES / (relative + ".png")
        if icon.is_file():
            output = target / (relative + ".png")
            output.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(icon, output)
            copied.append(str(output.relative_to(MOD_OUTPUT)))
    dragon_root = target / "DragonSkin"
    for base in ("Bodies/Naked_Thin", "Heads/Female_YR_head"):
        copied.extend(_copy_direction_set(TEXTURES / base, dragon_root / base))
    imported = OUTPUTS / "imports" / "DragonSkin"
    for relative in ("Bodies/Naked_Thin", "Heads/Female_YR_head"):
        parent = imported / Path(relative).parent
        if not parent.is_dir():
            continue
        for path in parent.glob(Path(relative).name + "_*.png"):
            output = dragon_root / Path(relative).parent / path.name
            output.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, output)
            copied.append(str(output.relative_to(MOD_OUTPUT)))
    return sorted(set(copied))


def _color_tuple(value: str) -> str:
    rgb = tuple(int(value[index:index + 2], 16) / 255 for index in (1, 3, 5))
    return "(" + ", ".join(f"{channel:.6f}" for channel in (*rgb, 1.0)) + ")"


def _write_pcc(character: dict, apparel_map: dict[str, str]) -> Path:
    root = ET.Element("character")
    ET.SubElement(root, "version").text = "5"
    ET.SubElement(root, "mods").text = "Harmony, Humanoid Alien Races, Dragon Yuran, EdB Prepare Carefully"
    pawn = ET.SubElement(root, "pawn")
    values = {
        "id": str(uuid.uuid5(uuid.NAMESPACE_URL, "dragon-yuran:" + character["id"])),
        "type": "Colony", "pawnKindDef": "DragonYuran_Colonist", "originalFactionDef": "PlayerColony",
        "thingDef": "DragonYuran_Race", "gender": "Female", "skinColor": _color_tuple(character["skinColor"]),
        "hairColor": _color_tuple(character["hairColor"]), "hairDef": "Shaved", "bodyType": "Thin",
        "headType": "DragonYuran_Female_AverageNormal", "nameType": "Triple",
        "firstName": character["name"]["first"], "nickName": character["name"]["nick"], "lastName": character["name"]["last"],
        "biologicalAgeInTicks": str(character["biologicalAge"] * 3600000),
        "chronologicalAgeInTicks": str(character["chronologicalAge"] * 3600000),
        "randomInjuries": "false", "randomRelations": "false",
    }
    for key, value in values.items():
        ET.SubElement(pawn, key).text = value
    traits = ET.SubElement(pawn, "traits")
    for trait in character["traits"]:
        item = ET.SubElement(traits, "li")
        ET.SubElement(item, "def").text = trait["def"]
        ET.SubElement(item, "degree").text = str(trait.get("degree", 0))
    skills = ET.SubElement(pawn, "skills")
    for name in SKILLS:
        item = ET.SubElement(skills, "li")
        ET.SubElement(item, "name").text = name
        ET.SubElement(item, "value").text = str(character["skills"][name]["level"])
        ET.SubElement(item, "passion").text = character["skills"][name]["passion"]
    apparel = ET.SubElement(pawn, "apparel")
    for source_def in character["apparel"]:
        item = ET.SubElement(apparel, "li")
        ET.SubElement(item, "apparel").text = apparel_map[source_def]
        ET.SubElement(item, "stuff").text = "Cloth"
        ET.SubElement(item, "quality").text = "Normal"
        ET.SubElement(item, "hitPoints").text = "100"
        ET.SubElement(item, "color").text = "(1, 1, 1, 1)"
    if character.get("dragonSkinInstalled"):
        implants = ET.SubElement(pawn, "implants")
        implant = ET.SubElement(implants, "li")
        ET.SubElement(implant, "recipe").text = "InstallDragonYuranDragonSkin"
        ET.SubElement(implant, "hediff").text = "DragonYuran_DragonSkin"
        ET.SubElement(implant, "bodyPart").text = "Torso"
        ET.SubElement(implant, "severity").text = "1"
    CHARACTER_OUTPUT.mkdir(parents=True, exist_ok=True)
    filename = re.sub(r"[^A-Za-z0-9._-]+", "_", character["id"]) + ".pcc"
    output = CHARACTER_OUTPUT / filename
    output.write_text(_xml_text(root), encoding="utf-8")
    return output


def build() -> dict:
    if not TEXTURES.is_dir():
        raise RuntimeError("Yuran source cache is missing; run `builder.py sync` first")
    if not CATALOG.is_file():
        build_catalog()
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    config = load_config()
    if MOD_OUTPUT.exists():
        shutil.rmtree(MOD_OUTPUT)
    if CHARACTER_OUTPUT.exists():
        shutil.rmtree(CHARACTER_OUTPUT)
    _write_about(config)
    _write_race_defs()
    _write_dragon_skin_defs()
    apparel = _write_apparel_defs(config, catalog)
    textures = _copy_textures(config, catalog)
    apparel_map = {item["sourceDef"]: item["defName"] for item in apparel}
    characters = [_write_pcc(character, apparel_map) for character in config["characters"]]
    report = {
        "status": "success", "builtAt": now(), "sourceWorkshopId": "2844129100", "rimworldVersion": "1.6.4871",
        "mod": str(MOD_OUTPUT.relative_to(REPO_ROOT)), "packageId": config["mod"]["packageId"],
        "included": {"raceDefs": ["DragonYuran_Race", "DragonYuran_Colonist"], "apparel": apparel, "textureFiles": len(textures), "characters": [path.name for path in characters]},
        "excluded": ["Yuran fiction and backstories", "factions", "buildings", "weapons", "research", "Miko variants", "Black Snake variants", "Shikigami", "custom Yuran assemblies"],
        "dragonSkin": {"hediff": "DragonYuran_DragonSkin", "recipe": "InstallDragonYuranDragonSkin", "fallback": "original Yuran body/head textures until an imported img2img sheet replaces the body and hairless-face canvases"},
        "gameLaunched": False,
    }
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    (OUTPUTS / "build_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def split_skin(sheet_path: Path) -> dict:
    metadata_path = OUTPUTS / "previews" / "yuran_skin_img2img.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("version") != 3:
        raise ValueError("dragon skin metadata must be version 3")
    expected_size = tuple(metadata["canvasSize"])
    run_id = _sha256(sheet_path)[:12]
    if sheet_path.parent.parent.name == "runs" and re.fullmatch(r"[0-9a-f]{12}", sheet_path.parent.name):
        run_id = sheet_path.parent.name
    run_root = OUTPUTS / "imports" / "runs" / run_id
    run_game = run_root / "game"
    comparison_root = run_root / "comparisons"
    active_root = OUTPUTS / "imports" / "DragonSkin"
    run_game.mkdir(parents=True, exist_ok=True)
    comparison_root.mkdir(parents=True, exist_ok=True)
    baseline_sheet = Image.open(OUTPUTS / "previews" / "yuran_skin_img2img.png").convert("RGBA")
    with Image.open(sheet_path) as source:
        source = source.convert("RGBA")
        if source.size != expected_size:
            raise ValueError(f"dragon skin sheet must remain exactly {expected_size[0]}x{expected_size[1]}")
        files = []
        for item in metadata["assets"]:
            x, y, width, height = item["contentBox"]
            generated_crop = source.crop((x, y, x + width, y + height))
            baseline_crop = baseline_sheet.crop((x, y, x + width, y + height))
            left, top, right, bottom = item["compositeBounds"]
            restored = generated_crop.resize((right - left, bottom - top), Image.Resampling.LANCZOS)
            baseline_restored = baseline_crop.resize((right - left, bottom - top), Image.Resampling.LANCZOS)
            generated_composite = Image.new("RGBA", (256, 256))
            baseline_composite = Image.new("RGBA", (256, 256))
            original_composite = Image.new("RGBA", (256, 256))
            generated_composite.alpha_composite(restored, (left, top))
            baseline_composite.alpha_composite(baseline_restored, (left, top))
            for layer in item["layers"]:
                original = Image.open(ROOT / layer["source"]).convert("RGBA")
                original_composite.alpha_composite(original, tuple(layer["compositeOffset"]))
            composite_alpha = original_composite.getchannel("A")
            generated_composite.putalpha(composite_alpha)
            baseline_composite.putalpha(composite_alpha)

            for layer in item["layers"]:
                original = Image.open(ROOT / layer["source"]).convert("RGBA")
                offset_x, offset_y = layer["compositeOffset"]
                canvas = Image.new("RGBA", original.size)
                baseline_canvas = Image.new("RGBA", original.size)
                canvas.alpha_composite(generated_composite, (-offset_x, -offset_y))
                baseline_canvas.alpha_composite(baseline_composite, (-offset_x, -offset_y))
                original_alpha = original.getchannel("A")
                canvas.putalpha(original_alpha)
                baseline_canvas.putalpha(original_alpha)
                output_relative = Path(layer["gameOutput"]).relative_to("Textures/DragonYuran/DragonSkin")
                run_path = run_game / output_relative
                active_path = active_root / output_relative
                run_path.parent.mkdir(parents=True, exist_ok=True)
                active_path.parent.mkdir(parents=True, exist_ok=True)
                canvas.save(run_path)
                shutil.copy2(run_path, active_path)

                difference = ImageChops.difference(baseline_canvas, canvas).convert("RGB")
                gray = difference.convert("L")
                alpha_values = list(original_alpha.get_flattened_data())
                gray_values = list(gray.get_flattened_data())
                masked_pixels = sum(1 for alpha in alpha_values if alpha > 0)
                changed_pixels = sum(1 for delta, alpha in zip(gray_values, alpha_values) if alpha > 0 and delta > 3)
                mean_delta = sum(delta for delta, alpha in zip(gray_values, alpha_values) if alpha > 0) / max(1, masked_pixels)
                enhanced = difference.point(lambda value: min(255, value * 4))
                enhanced.putalpha(original_alpha)
                diff_path = comparison_root / f"{layer['id']}_diff.png"
                enhanced.save(diff_path)
                files.append({
                    "id": layer["id"], "kind": layer["kind"], "label": layer["label"], "direction": item["direction"],
                    "original": layer["source"], "generated": _report_path(run_path),
                    "active": _report_path(active_path), "diff": _report_path(diff_path),
                    "gameOutput": layer["gameOutput"], "sha256": _sha256(run_path),
                    "metrics": {"changedPixels": changed_pixels, "maskedPixels": masked_pixels,
                                "changedPercent": round(changed_pixels * 100 / max(1, masked_pixels), 2),
                                "meanDelta": round(mean_delta, 2)},
                })
    changed_total = sum(item["metrics"]["changedPixels"] for item in files)
    masked_total = sum(item["metrics"]["maskedPixels"] for item in files)
    result = {
        "version": 3, "status": "success", "run": {"id": run_id, "label": f"Dragon skin {run_id}",
        "caseId": "dragon-yuran-skin", "startedAt": now(), "status": "success"},
        "source": _report_path(sheet_path), "files": files, "splitAt": now(),
        "comparisonBaseline": "Joined original Yuran body/head after the same enlarge/restore/project normalization pass",
        "summary": {"assetCount": len(files), "changedPixels": changed_total, "maskedPixels": masked_total,
                    "changedPercent": round(changed_total * 100 / max(1, masked_total), 2)},
        "stages": [
            {"id": "upload", "order": 1, "label": "Uploaded sheet", "status": "success"},
            {"id": "restore", "order": 2, "label": "Restored joined 256×256 views", "status": "success"},
            {"id": "mask", "order": 3, "label": "Projected body/head masks", "status": "success"},
            {"id": "compare", "order": 4, "label": "Compared with Yuran originals", "status": "success"},
        ],
    }
    (OUTPUTS / "imports" / "dragon_skin_import.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def deploy() -> dict:
    report = build()
    remote_mod = shlex.quote(REMOTE_MOD)
    remote_save = shlex.quote(REMOTE_SAVE)
    _run(["ssh", "-o", "BatchMode=yes", "tk2", f"mkdir -p {remote_mod} {remote_save}"], timeout=30)
    _run(["rsync", "-a", str(MOD_OUTPUT) + "/", f"tk2:{remote_mod}"], timeout=180)
    _run(["rsync", "-a", str(CHARACTER_OUTPUT) + "/", f"tk2:{remote_save}"], timeout=60)
    verify = _run([
        "ssh", "-o", "BatchMode=yes", "tk2",
        (
            f"test -f '{REMOTE_MOD}About/About.xml' "
            f"&& test -f '{REMOTE_HARMONY_DLL}' "
            f"&& test -f '{REMOTE_HAR_DLL}' "
            f"&& test -f '{REMOTE_PREPARE_CAREFULLY_DLL}' "
            f"&& find '{REMOTE_MOD}' -type f | wc -l "
            f"&& find '{REMOTE_SAVE}' -maxdepth 1 -name '*.pcc' -type f | wc -l "
            "&& echo dependencies-ok"
        ),
    ], timeout=30)
    lines = [line.strip() for line in verify.stdout.splitlines() if line.strip()]
    remote_files = int(lines[0]) if lines else 0
    remote_characters = int(lines[1]) if len(lines) > 1 else 0
    if len(lines) < 3 or lines[2] != "dependencies-ok":
        raise RuntimeError("TK2 dependency verification failed")
    if remote_characters != len(report["included"]["characters"]):
        raise RuntimeError(f"TK2 character verification failed: expected {len(report['included']['characters'])}, found {remote_characters}")
    deployed = {
        "status": "success", "deployedAt": now(), "remoteMod": REMOTE_MOD, "remotePrepareCarefully": REMOTE_SAVE,
        "remoteFileCount": remote_files, "remoteCharacterCount": remote_characters,
        "dependencies": {"harmony": True, "humanoidAlienRaces": True, "prepareCarefully": True},
        "packageId": report["packageId"], "gameLaunched": False, "activeModListChanged": False,
    }
    (OUTPUTS / "deploy_report.json").write_text(json.dumps(deployed, indent=2) + "\n", encoding="utf-8")
    return deployed


def _print(payload: object) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare and deploy the Dragon Yuran RimWorld mod")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("sync")
    sub.add_parser("inventory")
    sub.add_parser("build")
    sub.add_parser("deploy")
    validate = sub.add_parser("validate-config")
    validate.add_argument("--config", type=Path, default=CONFIG)
    split = sub.add_parser("split-skin")
    split.add_argument("sheet", type=Path)
    args = parser.parse_args(argv)
    if args.command == "sync":
        _print(sync_source())
    elif args.command == "inventory":
        _print(build_catalog())
    elif args.command == "build":
        _print(build())
    elif args.command == "deploy":
        _print(deploy())
    elif args.command == "validate-config":
        catalog = json.loads(CATALOG.read_text(encoding="utf-8")) if CATALOG.is_file() else None
        _print({"status": "valid", "characters": len(validate_config(json.loads(args.config.read_text(encoding="utf-8")), catalog=catalog)["characters"])})
    elif args.command == "split-skin":
        _print(split_skin(args.sheet))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
