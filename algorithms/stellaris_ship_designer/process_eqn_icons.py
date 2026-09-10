"""Prepare transparent, native-size EQN-GO icons from Dreamina results.

Dreamina currently returns a rendered checkerboard rather than real alpha.  This
tool isolates the centered emblem, removes corner marks/background pixels, fits
the art tightly, and writes the uncompressed RGBA DDS sizes used by Stellaris.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter


COMPONENT_JOBS = {
    "horn_m": "9fb5718d-8485-446b-abfd-af468ecc88ec",
    "horn_l": "88d6a265-1e9c-4fde-84b0-bb7f81c02683",
    "horn_x": "2a97b451-3ccc-40b4-80bd-3bfb1b29c996",
    "horn_t": "4036b154-ff05-46cb-84dd-da1b8cbca8b2",
    "horn_p": "d0067022-a25d-4f5a-99cb-a272d5f11820",
    "horn_g": "460124d4-e7fc-4ada-9d30-ff980f0ad683",
    "horn_w": "60f67480-0c7d-47c4-a3e8-68cbd33c78f7",
    "crystal_core": "f0d4d837-7a5c-4526-8a27-b095bd61baa8",
    "crystal_wings": "9d9401f2-33a6-4bf4-a91f-960d891b3de0",
}

TECHNOLOGY_JOBS = {
    "tech_guardian": "10e8661e-45df-4bdf-8122-e2c263cb0383",
    "tech_crown": "ea7ae32b-8bdb-4f8c-b9ff-0e7945e2d379",
    "tech_sovereign": "9299f90e-4995-4de7-a53f-226a3be94dd4",
    "tech_ascendant": "ede57d27-cb08-4c49-8f98-dfbfeb8cd559",
    "tech_delta": "d0c92c8c-fa4b-499a-8e15-692fd7435057",
    "tech_alpha": "0cd65012-4984-4401-9573-7d31113b9a80",
    "tech_sigma": "97e314ca-0af7-4485-9c8c-e0114e8816cb",
    "tech_phi": "022d08f9-6ca2-49b5-829d-7b41f20a350b",
    "tech_omega": "cc87b2b5-31f3-47b5-87b2-fbb74da2af1a",
    "tech_eternal": "3889c7f3-cadd-4633-afd9-0cb504d883da",
    "tech_body_2": "4a1292ea-714b-4b78-bb0b-e766f4cdaf71",
}

# The first sheet's watermarked top-left cell is deliberately never used.
SHEET_FALLBACKS = {
    "horn_s": (3, 0),
    "royal_mind": (2, 1),
    "tech_body_3": (3, 3),
}


def connected_component(mask: Image.Image) -> Image.Image:
    width, height = mask.size
    source = bytearray(mask.tobytes())
    seen = bytearray(width * height)
    best: list[int] = []
    for start, value in enumerate(source):
        if not value or seen[start]:
            continue
        seen[start] = 1
        stack = [start]
        component = []
        while stack:
            index = stack.pop()
            component.append(index)
            x, y = index % width, index // width
            for other in (index - 1 if x else -1, index + 1 if x + 1 < width else -1,
                          index - width if y else -1, index + width if y + 1 < height else -1):
                if other >= 0 and source[other] and not seen[other]:
                    seen[other] = 1
                    stack.append(other)
        if len(component) > len(best):
            best = component
    if not best:
        raise ValueError("No centered icon foreground found")
    result = bytearray(width * height)
    for index in best:
        result[index] = 255
    return Image.frombytes("L", (width, height), bytes(result))


def edge_background(mask: Image.Image) -> Image.Image:
    """Keep background-colored regions connected to an edge or a large hole."""
    width, height = mask.size
    source = bytearray(mask.tobytes())
    seen = bytearray(width * height)
    result = bytearray(width * height)
    large_hole = width * height * 0.04
    for start, value in enumerate(source):
        if not value or seen[start]:
            continue
        seen[start] = 1
        stack = [start]
        component = []
        touches_edge = False
        while stack:
            index = stack.pop()
            component.append(index)
            x, y = index % width, index // width
            touches_edge = touches_edge or x == 0 or y == 0 or x + 1 == width or y + 1 == height
            for other in (index - 1 if x else -1, index + 1 if x + 1 < width else -1,
                          index - width if y else -1, index + width if y + 1 < height else -1):
                if other >= 0 and source[other] and not seen[other]:
                    seen[other] = 1
                    stack.append(other)
        if touches_edge or len(component) >= large_hole:
            for index in component:
                result[index] = 255
    return Image.frombytes("L", (width, height), bytes(result))


def isolate(image: Image.Image, dark_background: bool = False) -> Image.Image:
    image = image.convert("RGB")
    image.thumbnail((512, 512), Image.Resampling.LANCZOS)
    background_candidates = []
    for red, green, blue in image.getdata():
        if dark_background:
            background = max(red, green, blue) <= 92
        else:
            background = max(red, green, blue) - min(red, green, blue) <= 14 and min(red, green, blue) >= 222
        background_candidates.append(255 if background else 0)
    background = edge_background(Image.frombytes("L", image.size, bytes(background_candidates)))
    mask = connected_component(background.point(lambda value: 0 if value else 255))
    mask = mask.filter(ImageFilter.MaxFilter(5)).filter(ImageFilter.GaussianBlur(1.0))
    bounds = mask.getbbox()
    if not bounds:
        raise ValueError("Icon foreground has no bounds")
    left, top, right, bottom = bounds
    size = max(right - left, bottom - top)
    padding = max(3, round(size * 0.065))
    center_x, center_y = (left + right) / 2, (top + bottom) / 2
    size += padding * 2
    square = (round(center_x - size / 2), round(center_y - size / 2),
              round(center_x + size / 2), round(center_y + size / 2))
    rgba = image.convert("RGBA")
    rgba.putalpha(mask)
    cropped = rgba.crop(square).resize((256, 256), Image.Resampling.LANCZOS)
    return cropped


def find_job(raw: Path, name: str, submit_id: str) -> Path:
    matches = list((raw / name).glob(submit_id + "*.png"))
    if len(matches) != 1:
        raise ValueError(f"Expected one Dreamina result for {name} ({submit_id}), found {len(matches)}")
    return matches[0]


def write_icon(master: Image.Image, output: Path, name: str, native_size: int) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    png = output / (name + ".png")
    dds = output / (name + ".dds")
    master.save(png, optimize=True)
    master.resize((native_size, native_size), Image.Resampling.LANCZOS).save(dds)
    alpha = master.getchannel("A")
    visible = sum(value > 8 for value in alpha.getdata()) / (256 * 256)
    corners = [alpha.getpixel(point) for point in ((0, 0), (255, 0), (0, 255), (255, 255))]
    # Long lances naturally occupy much less area than hearts or shields while
    # still touching most of the tile's width/height.
    if any(corners) or not 0.08 <= visible <= 0.9:
        raise ValueError(f"Bad alpha coverage for {name}: visible={visible:.3f}, corners={corners}")
    return {
        "id": name,
        "preview": str(png),
        "native": str(dds),
        "size": [native_size, native_size],
        "visibleFraction": round(visible, 4),
        "sha256": hashlib.sha256(dds.read_bytes()).hexdigest(),
    }


def fit_existing(image: Image.Image) -> Image.Image:
    image = image.convert("RGBA")
    bounds = image.getchannel("A").getbbox()
    if not bounds:
        raise ValueError("Reference image has no visible pixels")
    foreground = image.crop(bounds)
    side = round(max(foreground.size) * 1.12)
    result = Image.new("RGBA", (side, side))
    result.alpha_composite(foreground, ((side - foreground.width) // 2, (side - foreground.height) // 2))
    return result.resize((256, 256), Image.Resampling.LANCZOS)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--sheet", type=Path, required=True)
    parser.add_argument("--abstract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    masters = {}
    providers = {}
    for name, submit_id in {**COMPONENT_JOBS, **TECHNOLOGY_JOBS}.items():
        masters[name] = isolate(Image.open(find_job(args.raw, name, submit_id)))
        providers[name] = {"provider": "Dreamina", "model": "4.7", "submitId": submit_id, "source": "individual image2image"}

    # Dreamina's dedicated ascendant attempt produced a malformed pony with a
    # missing head.  Keep the approved individual wing generation, shrink it
    # slightly, and add a simple central ascension star instead.
    ascendant = Image.new("RGBA", (256, 256))
    wings = masters["crystal_wings"].resize((222, 222), Image.Resampling.LANCZOS)
    ascendant.alpha_composite(wings, (17, 17))
    draw = ImageDraw.Draw(ascendant)
    outer = [(128, 70), (139, 116), (176, 128), (139, 140), (128, 188), (117, 140), (80, 128), (117, 116)]
    inner = [(128, 87), (135, 120), (160, 128), (135, 136), (128, 169), (121, 136), (96, 128), (121, 120)]
    draw.polygon(outer, fill=(225, 174, 70, 255))
    draw.polygon(inner, fill=(88, 235, 245, 255))
    masters["tech_ascendant"] = ascendant
    providers["tech_ascendant"] = {
        "provider": "local correction from individual Dreamina wings",
        "submitId": COMPONENT_JOBS["crystal_wings"],
        "source": "malformed pony rejected; gold-and-cyan ascension star composited over the isolated wings",
    }

    sheet = Image.open(args.sheet).convert("RGB")
    cell_width, cell_height = sheet.width // 4, sheet.height // 4
    for name, (row, column) in SHEET_FALLBACKS.items():
        margin = round(min(cell_width, cell_height) * 0.035)
        cell = sheet.crop((column * cell_width + margin, row * cell_height + margin,
                           (column + 1) * cell_width - margin, (row + 1) * cell_height - margin))
        masters[name] = isolate(cell, dark_background=True)
        providers[name] = {"provider": "Dreamina", "model": "4.7", "submitId": "2b8091cb-8fca-4a2d-a971-83bcf5d1a097", "source": f"non-watermarked sheet cell {row + 1},{column + 1}"}

    records = {"components": [], "technologies": [], "providers": providers}
    for name in ("horn_s", "horn_m", "horn_l", "horn_x", "horn_t", "horn_p", "horn_g", "horn_w", "crystal_core", "royal_mind", "crystal_wings"):
        records["components"].append(write_icon(masters[name], args.output / "components", name, 58))
    for name in ("tech_guardian", "tech_crown", "tech_sovereign", "tech_ascendant", "tech_delta", "tech_alpha", "tech_sigma", "tech_phi", "tech_omega", "tech_eternal", "tech_body_2", "tech_body_3"):
        records["technologies"].append(write_icon(masters[name], args.output / "technologies", name, 52))

    ground = fit_existing(Image.open(args.abstract))
    records["ground"] = write_icon(ground, args.output, "crystal_pony", 34)
    for usage, size in (("component", 58), ("trait", 29), ("origin", 40), ("technology", 52)):
        path = args.output / f"crystal_pony_{usage}.dds"
        ground.resize((size, size), Image.Resampling.LANCZOS).save(path)
        records.setdefault("groundDerivatives", {})[usage] = {"path": str(path), "size": [size, size], "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    (args.output / "generated_icons.json").write_text(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    main()
