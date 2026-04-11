from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class UrdfJointSpec:
    name: str
    joint_type: str
    parent_link: str
    child_link: str
    origin: np.ndarray
    axis: np.ndarray
    lower: float
    upper: float


def axis_angle_matrix(axis: Sequence[float], angle_rad: float) -> np.ndarray:
    axis_arr = np.asarray(axis, dtype=float)
    norm = float(np.linalg.norm(axis_arr))
    if norm < 1.0e-8 or abs(angle_rad) < 1.0e-10:
        return np.eye(3, dtype=float)
    x, y, z = axis_arr / norm
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    one_c = 1.0 - c
    return np.array(
        [
            [c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s],
            [y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s],
            [z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c],
        ],
        dtype=float,
    )


def rpy_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    return (
        axis_angle_matrix((1.0, 0.0, 0.0), roll)
        @ axis_angle_matrix((0.0, 1.0, 0.0), pitch)
        @ axis_angle_matrix((0.0, 0.0, 1.0), yaw)
    )


def _origin_matrix(joint_el) -> np.ndarray:
    origin_el = joint_el.find("origin")
    xyz = (0.0, 0.0, 0.0)
    rpy = (0.0, 0.0, 0.0)
    if origin_el is not None:
        xyz_text = origin_el.attrib.get("xyz")
        rpy_text = origin_el.attrib.get("rpy")
        if xyz_text:
            xyz = tuple(float(value) for value in xyz_text.split())
        if rpy_text:
            rpy = tuple(float(value) for value in rpy_text.split())
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = rpy_matrix(*rpy)
    transform[:3, 3] = np.asarray(xyz, dtype=float)
    return transform


def _joint_motion(spec: UrdfJointSpec, position: float) -> np.ndarray:
    motion = np.eye(4, dtype=float)
    if spec.joint_type in ("revolute", "continuous"):
        motion[:3, :3] = axis_angle_matrix(spec.axis, position)
    elif spec.joint_type == "prismatic":
        motion[:3, 3] = np.asarray(spec.axis, dtype=float) * float(position)
    return motion


def joint_local_transform(spec: UrdfJointSpec, position: float) -> np.ndarray:
    return spec.origin @ _joint_motion(spec, position)


def load_urdf_joint_specs(urdf_path: Path) -> dict[str, UrdfJointSpec]:
    root = ET.parse(urdf_path).getroot()
    specs: dict[str, UrdfJointSpec] = {}
    for joint_el in root.findall("joint"):
        joint_name = joint_el.attrib.get("name")
        joint_type = joint_el.attrib.get("type", "fixed")
        if not joint_name:
            continue
        parent_el = joint_el.find("parent")
        child_el = joint_el.find("child")
        if parent_el is None or child_el is None:
            continue
        axis_el = joint_el.find("axis")
        axis = np.array([0.0, 0.0, 1.0], dtype=float)
        if axis_el is not None and axis_el.attrib.get("xyz"):
            axis = np.asarray([float(value) for value in axis_el.attrib["xyz"].split()], dtype=float)
        limit_el = joint_el.find("limit")
        lower = float(limit_el.attrib.get("lower", str(-math.pi))) if limit_el is not None else -math.pi
        upper = float(limit_el.attrib.get("upper", str(math.pi))) if limit_el is not None else math.pi
        specs[joint_name] = UrdfJointSpec(
            name=joint_name,
            joint_type=joint_type,
            parent_link=parent_el.attrib["link"],
            child_link=child_el.attrib["link"],
            origin=_origin_matrix(joint_el),
            axis=axis,
            lower=lower,
            upper=upper,
        )
    return specs


def specs_by_child_link(specs: Sequence[UrdfJointSpec] | Mapping[str, UrdfJointSpec]) -> dict[str, UrdfJointSpec]:
    values = specs.values() if isinstance(specs, Mapping) else specs
    return {spec.child_link: spec for spec in values}


def load_joint_limits(urdf_path: Path) -> dict[str, tuple[float, float]]:
    limits: dict[str, tuple[float, float]] = {}
    for spec in load_urdf_joint_specs(urdf_path).values():
        limits[spec.name] = (spec.lower, spec.upper)
        limits.setdefault(spec.child_link, (spec.lower, spec.upper))
    return limits


def _root_links(specs: Sequence[UrdfJointSpec]) -> tuple[str, ...]:
    parent_links = {spec.parent_link for spec in specs}
    child_links = {spec.child_link for spec in specs}
    roots = sorted(parent_links - child_links)
    return tuple(roots) if roots else ("base_link",)


def world_map_from_joint_specs(
    specs: Sequence[UrdfJointSpec] | Mapping[str, UrdfJointSpec],
    pose_by_name: Mapping[str, float] | None = None,
    *,
    pose_key: str = "joint_name",
) -> dict[str, np.ndarray]:
    pose = pose_by_name or {}
    ordered_specs = tuple(specs.values()) if isinstance(specs, Mapping) else tuple(specs)
    world = {link_name: np.eye(4, dtype=float) for link_name in _root_links(ordered_specs)}
    pending = {spec.name: spec for spec in ordered_specs}
    while pending:
        progressed = False
        for joint_name in list(pending):
            spec = pending[joint_name]
            if spec.parent_link not in world:
                continue
            pose_name = spec.child_link if pose_key == "child_link" else spec.name
            world[spec.child_link] = world[spec.parent_link] @ joint_local_transform(spec, float(pose.get(pose_name, 0.0)))
            pending.pop(joint_name)
            progressed = True
        if not progressed:
            unresolved = ", ".join(sorted(pending))
            raise RuntimeError(f"Unable to resolve URDF joint world transforms: {unresolved}")
    return world


def world_map_from_urdf_pose(
    urdf_path: Path,
    pose_by_name: Mapping[str, float] | None = None,
    *,
    pose_key: str = "joint_name",
) -> dict[str, np.ndarray]:
    return world_map_from_joint_specs(load_urdf_joint_specs(urdf_path), pose_by_name, pose_key=pose_key)


def find_joint_chain_from_specs(
    specs: Sequence[UrdfJointSpec] | Mapping[str, UrdfJointSpec],
    base_link: str,
    tip_link: str,
    *,
    key: str = "joint_name",
) -> tuple[str, ...]:
    by_child = specs_by_child_link(specs)
    current_link = tip_link
    chain: list[str] = []
    while current_link != base_link:
        spec = by_child.get(current_link)
        if spec is None:
            raise KeyError(f"Could not find joint chain from {base_link} to {tip_link}")
        chain.append(spec.child_link if key == "child_link" else spec.name)
        current_link = spec.parent_link
    chain.reverse()
    return tuple(chain)


def find_joint_chain(
    urdf_path: Path,
    base_link: str,
    tip_link: str,
    *,
    key: str = "joint_name",
) -> tuple[str, ...]:
    return find_joint_chain_from_specs(load_urdf_joint_specs(urdf_path), base_link, tip_link, key=key)
