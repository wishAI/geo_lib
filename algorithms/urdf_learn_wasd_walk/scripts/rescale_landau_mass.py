from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path


MASS_RESCALE_VERSION = "2026-04-19-v1"


def _target_mass_for_link(link_name: str) -> float:
    if link_name == "base_link":
        return 0.2
    if link_name == "root_x":
        return 6.5
    if link_name == "spine_01_x":
        return 3.2
    if link_name == "spine_02_x":
        return 2.9
    if link_name == "spine_03_x":
        return 2.7
    if link_name == "neck_x":
        return 0.6
    if link_name == "head_x":
        return 1.3
    if link_name in {"left_hip_roll_link", "right_hip_roll_link"}:
        return 0.5
    if link_name.startswith("thigh_stretch"):
        return 1.8
    if link_name.startswith("thigh_twist"):
        return 1.5
    if link_name.startswith("leg_stretch"):
        return 1.4
    if link_name.startswith("leg_twist"):
        return 1.1
    if link_name.startswith("foot_"):
        return 0.7
    if link_name.startswith("toes_"):
        return 0.2
    if link_name.startswith("shoulder_"):
        return 0.35
    if link_name.startswith("arm_stretch"):
        return 0.55
    if link_name.startswith("arm_twist"):
        return 0.45
    if link_name.startswith("forearm_stretch"):
        return 0.35
    if link_name.startswith("forearm_twist"):
        return 0.2
    if link_name.startswith("hand_"):
        return 0.25
    if link_name.startswith(("thumb", "index", "middle", "ring", "pinky")):
        return 0.015
    raise KeyError(f"Unclassified Landau link '{link_name}'.")


def _scale_inertia_element(inertia_el: ET.Element, scale: float) -> None:
    for attribute in ("ixx", "ixy", "ixz", "iyy", "iyz", "izz"):
        raw_value = inertia_el.attrib.get(attribute)
        if raw_value is None:
            continue
        inertia_el.attrib[attribute] = f"{float(raw_value) * scale:.9f}"


def rescale_landau_urdf(source_path: Path, output_path: Path) -> dict[str, float]:
    tree = ET.parse(source_path)
    root = tree.getroot()

    total_mass = 0.0
    torso_pelvis_mass = 0.0
    mass_bearing_links = 0

    for link_el in root.findall("link"):
        link_name = link_el.attrib["name"]
        inertial_el = link_el.find("inertial")
        if inertial_el is None:
            continue
        mass_el = inertial_el.find("mass")
        inertia_el = inertial_el.find("inertia")
        if mass_el is None:
            continue

        old_mass = float(mass_el.attrib.get("value", "0.0"))
        new_mass = _target_mass_for_link(link_name)
        scale = 1.0 if old_mass <= 0.0 else new_mass / old_mass
        mass_el.attrib["value"] = f"{new_mass:.6f}"
        if inertia_el is not None:
            _scale_inertia_element(inertia_el, scale)

        total_mass += new_mass
        if link_name in {"root_x", "spine_01_x", "spine_02_x", "spine_03_x"}:
            torso_pelvis_mass += new_mass
        if new_mass > 0.0:
            mass_bearing_links += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(tree, space="  ")
    tree.write(output_path, encoding="utf-8", xml_declaration=True)

    return {
        "mass_bearing_links": float(mass_bearing_links),
        "total_mass": total_mass,
        "torso_pelvis_mass": torso_pelvis_mass,
        "torso_pelvis_share": torso_pelvis_mass / total_mass if total_mass > 0.0 else 0.0,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rescale the Landau URDF masses and inertias.")
    parser.add_argument("--source", type=Path, required=True, help="Path to the source URDF.")
    parser.add_argument("--output", type=Path, required=True, help="Path to the fixed URDF to write.")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    summary = rescale_landau_urdf(args.source, args.output)
    print(
        {
            "version": MASS_RESCALE_VERSION,
            "source": str(args.source.resolve()),
            "output": str(args.output.resolve()),
            **summary,
        }
    )


if __name__ == "__main__":
    main()
