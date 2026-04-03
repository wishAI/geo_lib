from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True, slots=True)
class JointLimit:
    lower: float
    upper: float
    continuous: bool = False

    def contains(self, value: float, atol: float = 1e-8) -> bool:
        return (self.lower - atol) <= value <= (self.upper + atol)


@dataclass(frozen=True, slots=True)
class UrdfJoint:
    name: str
    joint_type: str
    parent: str
    child: str
    axis: np.ndarray
    origin_xyz: np.ndarray
    origin_rpy: np.ndarray
    limit: JointLimit | None

    @property
    def is_active(self) -> bool:
        return self.joint_type != "fixed"


@dataclass(frozen=True, slots=True)
class UrdfChain:
    base_link: str
    tip_link: str
    joints: tuple[UrdfJoint, ...]

    @property
    def active_joints(self) -> tuple[UrdfJoint, ...]:
        return tuple(joint for joint in self.joints if joint.is_active)

    @property
    def active_joint_names(self) -> tuple[str, ...]:
        return tuple(joint.name for joint in self.active_joints)

    @property
    def active_joint_limits(self) -> tuple[JointLimit, ...]:
        limits: list[JointLimit] = []
        for joint in self.active_joints:
            if joint.limit is None:
                raise ValueError(f"Active joint {joint.name!r} does not define a usable limit.")
            limits.append(joint.limit)
        return tuple(limits)


def resolve_chain(
    urdf_path: str | Path,
    base_link: str,
    tip_link: str,
    *,
    continuous_joint_limits: tuple[float, float] = (-2.0 * math.pi, 2.0 * math.pi),
) -> UrdfChain:
    _, joints_by_name, child_to_joint = parse_urdf(urdf_path, continuous_joint_limits=continuous_joint_limits)
    chain: list[UrdfJoint] = []
    current_link = tip_link
    while current_link != base_link:
        try:
            joint_name = child_to_joint[current_link]
        except KeyError as exc:
            raise ValueError(f"Unable to resolve URDF chain from {base_link!r} to {tip_link!r}; missing parent for link {current_link!r}.") from exc
        joint = joints_by_name[joint_name]
        chain.append(joint)
        current_link = joint.parent
    chain.reverse()
    return UrdfChain(base_link=base_link, tip_link=tip_link, joints=tuple(chain))


def parse_urdf(
    urdf_path: str | Path,
    *,
    continuous_joint_limits: tuple[float, float] = (-2.0 * math.pi, 2.0 * math.pi),
) -> tuple[ET.ElementTree, dict[str, UrdfJoint], dict[str, str]]:
    tree = ET.parse(Path(urdf_path))
    root = tree.getroot()
    joints_by_name: dict[str, UrdfJoint] = {}
    child_to_joint: dict[str, str] = {}
    for joint_elem in root.findall("joint"):
        name = joint_elem.attrib["name"]
        joint_type = joint_elem.attrib.get("type", "fixed")
        parent_elem = joint_elem.find("parent")
        if parent_elem is None or "link" not in parent_elem.attrib:
            raise ValueError(f"Joint {name!r} is missing a parent link.")
        parent = parent_elem.attrib["link"]
        child_elem = joint_elem.find("child")
        if child_elem is None or "link" not in child_elem.attrib:
            raise ValueError(f"Joint {name!r} is missing a child link.")
        child = child_elem.attrib["link"]
        axis = _parse_vector(joint_elem.find("axis"), default=(0.0, 0.0, 1.0))
        origin_xyz, origin_rpy = _parse_origin(joint_elem.find("origin"))
        limit = _parse_limit(joint_elem, joint_type, continuous_joint_limits)
        joint = UrdfJoint(
            name=name,
            joint_type=joint_type,
            parent=parent,
            child=child,
            axis=axis,
            origin_xyz=origin_xyz,
            origin_rpy=origin_rpy,
            limit=limit,
        )
        joints_by_name[name] = joint
        if child in child_to_joint:
            raise ValueError(f"URDF link {child!r} has multiple parent joints, which is not supported.")
        child_to_joint[child] = name
    return tree, joints_by_name, child_to_joint


def sanitize_urdf_for_mujoco(src_path: str | Path, dst_path: str | Path) -> Path:
    tree = ET.parse(Path(src_path))
    root = tree.getroot()
    child_links = {
        child_elem.attrib["link"]
        for joint_elem in root.findall("joint")
        for child_elem in [joint_elem.find("child")]
        if child_elem is not None and "link" in child_elem.attrib
    }
    for tag in ("transmission", "gazebo", "ros2_control", "material"):
        for elem in list(root.findall(tag)):
            root.remove(elem)
    for link in root.findall("link"):
        for tag in ("visual", "collision"):
            for elem in list(link.findall(tag)):
                link.remove(elem)
        link_name = link.attrib.get("name", "")
        if link.find("inertial") is None and link_name in child_links:
            inertial = ET.SubElement(link, "inertial")
            ET.SubElement(inertial, "origin", xyz="0 0 0", rpy="0 0 0")
            ET.SubElement(inertial, "mass", value="0.001")
            ET.SubElement(
                inertial,
                "inertia",
                ixx="1e-6",
                ixy="0",
                ixz="0",
                iyy="1e-6",
                iyz="0",
                izz="1e-6",
            )
    dst = Path(dst_path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tree.write(dst, encoding="utf-8", xml_declaration=True)
    return dst


def _parse_origin(origin_elem: ET.Element | None) -> tuple[np.ndarray, np.ndarray]:
    if origin_elem is None:
        return np.zeros(3, dtype=np.float64), np.zeros(3, dtype=np.float64)
    xyz = _parse_float_list(origin_elem.attrib.get("xyz", "0 0 0"), count=3)
    rpy = _parse_float_list(origin_elem.attrib.get("rpy", "0 0 0"), count=3)
    return np.asarray(xyz, dtype=np.float64), np.asarray(rpy, dtype=np.float64)


def _parse_vector(axis_elem: ET.Element | None, *, default: tuple[float, float, float]) -> np.ndarray:
    if axis_elem is None:
        values = np.asarray(default, dtype=np.float64)
    else:
        values = np.asarray(_parse_float_list(axis_elem.attrib.get("xyz", " ".join(str(v) for v in default)), count=3), dtype=np.float64)
    norm = np.linalg.norm(values)
    if norm < 1e-12:
        return np.asarray(default, dtype=np.float64)
    return values / norm


def _parse_limit(
    joint_elem: ET.Element,
    joint_type: str,
    continuous_joint_limits: tuple[float, float],
) -> JointLimit | None:
    if joint_type == "fixed":
        return None
    if joint_type == "continuous":
        lower, upper = continuous_joint_limits
        return JointLimit(lower=float(lower), upper=float(upper), continuous=True)
    limit_elem = joint_elem.find("limit")
    if limit_elem is None:
        return None
    lower = float(limit_elem.attrib.get("lower", "0.0"))
    upper = float(limit_elem.attrib.get("upper", "0.0"))
    return JointLimit(lower=lower, upper=upper, continuous=False)


def _parse_float_list(text: str, *, count: int) -> list[float]:
    values = [float(chunk) for chunk in text.split()]
    if len(values) != count:
        raise ValueError(f"Expected {count} floats but received {values!r}.")
    return values
