from __future__ import annotations

import math
from functools import lru_cache
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
import struct


def _parse_floats(raw_value: str | None, expected_count: int, default: float = 0.0) -> tuple[float, ...]:
    if not raw_value:
        return tuple([default] * expected_count)
    values = [float(item) for item in raw_value.split()]
    if len(values) != expected_count:
        raise ValueError(f"Expected {expected_count} values, received {len(values)} from '{raw_value}'.")
    return tuple(values)


def rpy_to_matrix(rpy: tuple[float, float, float]) -> tuple[tuple[float, float, float], ...]:
    roll, pitch, yaw = rpy
    cr = math.cos(roll)
    sr = math.sin(roll)
    cp = math.cos(pitch)
    sp = math.sin(pitch)
    cy = math.cos(yaw)
    sy = math.sin(yaw)
    return (
        (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
        (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
        (-sp, cp * sr, cp * cr),
    )


def axis_angle_matrix(axis: tuple[float, float, float], angle: float) -> tuple[tuple[float, float, float], ...]:
    x, y, z = axis
    norm = math.sqrt(x * x + y * y + z * z)
    if norm < 1e-8 or abs(angle) < 1e-10:
        return (
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        )
    x /= norm
    y /= norm
    z /= norm
    c = math.cos(angle)
    s = math.sin(angle)
    one_c = 1.0 - c
    return (
        (c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s),
        (y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s),
        (z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c),
    )


def compose_transform(
    xyz: tuple[float, float, float],
    rotation: tuple[tuple[float, float, float], ...],
) -> tuple[tuple[float, float, float, float], ...]:
    return (
        (rotation[0][0], rotation[0][1], rotation[0][2], xyz[0]),
        (rotation[1][0], rotation[1][1], rotation[1][2], xyz[1]),
        (rotation[2][0], rotation[2][1], rotation[2][2], xyz[2]),
        (0.0, 0.0, 0.0, 1.0),
    )


def matrix_multiply(
    lhs: tuple[tuple[float, float, float, float], ...],
    rhs: tuple[tuple[float, float, float, float], ...],
) -> tuple[tuple[float, float, float, float], ...]:
    rows = []
    for row in range(4):
        rows.append(
            tuple(sum(lhs[row][inner] * rhs[inner][col] for inner in range(4)) for col in range(4))
        )
    return tuple(rows)


def transform_point(
    transform: tuple[tuple[float, float, float, float], ...], point: tuple[float, float, float]
) -> tuple[float, float, float]:
    return (
        transform[0][0] * point[0] + transform[0][1] * point[1] + transform[0][2] * point[2] + transform[0][3],
        transform[1][0] * point[0] + transform[1][1] * point[1] + transform[1][2] * point[2] + transform[1][3],
        transform[2][0] * point[0] + transform[2][1] * point[1] + transform[2][2] * point[2] + transform[2][3],
    )


IDENTITY_4 = (
    (1.0, 0.0, 0.0, 0.0),
    (0.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, 0.0),
    (0.0, 0.0, 0.0, 1.0),
)


@dataclass(frozen=True)
class UrdfLimit:
    lower: float | None
    upper: float | None
    effort: float | None
    velocity: float | None


@dataclass(frozen=True)
class UrdfLink:
    name: str
    visual_meshes: tuple[Path, ...]
    collision_meshes: tuple[Path, ...]
    mass: float
    inertia: tuple[float, float, float, float, float, float] | None


@dataclass(frozen=True)
class UrdfJoint:
    name: str
    joint_type: str
    parent_link: str
    child_link: str
    origin_xyz: tuple[float, float, float]
    origin_rpy: tuple[float, float, float]
    axis_xyz: tuple[float, float, float]
    limit: UrdfLimit


@dataclass(frozen=True)
class UrdfModel:
    path: Path
    robot_name: str
    links: dict[str, UrdfLink]
    joints: dict[str, UrdfJoint]
    children_by_link: dict[str, tuple[str, ...]]
    child_joint_by_link: dict[str, str]
    root_links: tuple[str, ...]


@dataclass(frozen=True)
class JointGroups:
    leg_joints: tuple[str, ...]
    foot_joints: tuple[str, ...]
    arm_joints: tuple[str, ...]
    hand_joints: tuple[str, ...]
    finger_joints: tuple[str, ...]
    torso_joints: tuple[str, ...]


def load_urdf_model(urdf_path: Path) -> UrdfModel:
    urdf_path = Path(urdf_path).resolve()
    root = ET.parse(urdf_path).getroot()
    links: dict[str, UrdfLink] = {}
    joints: dict[str, UrdfJoint] = {}

    for link_el in root.findall("link"):
        link_name = link_el.attrib["name"]
        visual_meshes = []
        collision_meshes = []
        inertial_el = link_el.find("inertial")
        mass = 0.0
        inertia = None
        if inertial_el is not None:
            mass_el = inertial_el.find("mass")
            inertia_el = inertial_el.find("inertia")
            if mass_el is not None:
                mass = float(mass_el.attrib.get("value", "0.0"))
            if inertia_el is not None:
                inertia = tuple(
                    float(inertia_el.attrib.get(attribute, "0.0"))
                    for attribute in ("ixx", "ixy", "ixz", "iyy", "iyz", "izz")
                )
        for visual_el in link_el.findall("visual"):
            mesh_el = visual_el.find("./geometry/mesh")
            if mesh_el is not None and mesh_el.attrib.get("filename"):
                visual_meshes.append((urdf_path.parent / mesh_el.attrib["filename"]).resolve())
        for collision_el in link_el.findall("collision"):
            mesh_el = collision_el.find("./geometry/mesh")
            if mesh_el is not None and mesh_el.attrib.get("filename"):
                collision_meshes.append((urdf_path.parent / mesh_el.attrib["filename"]).resolve())
        links[link_name] = UrdfLink(
            name=link_name,
            visual_meshes=tuple(visual_meshes),
            collision_meshes=tuple(collision_meshes),
            mass=mass,
            inertia=inertia,
        )

    children_by_link: dict[str, list[str]] = {name: [] for name in links}
    child_joint_by_link: dict[str, str] = {}
    child_links: set[str] = set()

    for joint_el in root.findall("joint"):
        joint_name = joint_el.attrib["name"]
        origin_el = joint_el.find("origin")
        axis_el = joint_el.find("axis")
        limit_el = joint_el.find("limit")
        parent_link = joint_el.find("parent").attrib["link"]
        child_link = joint_el.find("child").attrib["link"]
        joint = UrdfJoint(
            name=joint_name,
            joint_type=joint_el.attrib["type"],
            parent_link=parent_link,
            child_link=child_link,
            origin_xyz=_parse_floats(origin_el.attrib.get("xyz") if origin_el is not None else None, 3),
            origin_rpy=_parse_floats(origin_el.attrib.get("rpy") if origin_el is not None else None, 3),
            axis_xyz=_parse_floats(axis_el.attrib.get("xyz") if axis_el is not None else None, 3, default=0.0),
            limit=UrdfLimit(
                lower=float(limit_el.attrib["lower"]) if limit_el is not None and "lower" in limit_el.attrib else None,
                upper=float(limit_el.attrib["upper"]) if limit_el is not None and "upper" in limit_el.attrib else None,
                effort=float(limit_el.attrib["effort"]) if limit_el is not None and "effort" in limit_el.attrib else None,
                velocity=float(limit_el.attrib["velocity"]) if limit_el is not None and "velocity" in limit_el.attrib else None,
            ),
        )
        joints[joint_name] = joint
        children_by_link.setdefault(parent_link, []).append(child_link)
        child_joint_by_link[child_link] = joint_name
        child_links.add(child_link)

    root_links = tuple(sorted(link_name for link_name in links if link_name not in child_links))
    return UrdfModel(
        path=urdf_path,
        robot_name=root.attrib.get("name", urdf_path.stem),
        links=links,
        joints=joints,
        children_by_link={name: tuple(children) for name, children in children_by_link.items()},
        child_joint_by_link=child_joint_by_link,
        root_links=root_links,
    )


def find_missing_meshes(model: UrdfModel) -> tuple[Path, ...]:
    missing = []
    for link in model.links.values():
        for mesh_path in (*link.visual_meshes, *link.collision_meshes):
            if not mesh_path.exists():
                missing.append(mesh_path)
    return tuple(sorted(set(missing)))


@lru_cache(maxsize=512)
def _load_mesh_vertices(mesh_path: str) -> tuple[tuple[float, float, float], ...]:
    path = Path(mesh_path)
    data = path.read_bytes()
    preview = data[:512].lower()
    if preview.startswith(b"solid") and b"facet" in preview:
        vertices: list[tuple[float, float, float]] = []
        for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = raw_line.strip()
            if not line.startswith("vertex "):
                continue
            _, xs, ys, zs = line.split()
            vertices.append((float(xs), float(ys), float(zs)))
        return tuple(vertices)

    triangle_count = struct.unpack_from("<I", data, 80)[0]
    offset = 84
    vertices = []
    for _ in range(triangle_count):
        offset += 12  # normal
        for _ in range(3):
            vertices.append(struct.unpack_from("<fff", data, offset))
            offset += 12
        offset += 2  # attribute byte count
    return tuple(vertices)


def _link_collision_vertices(link: UrdfLink) -> tuple[tuple[float, float, float], ...]:
    vertices: list[tuple[float, float, float]] = []
    mesh_paths = link.collision_meshes or link.visual_meshes
    for mesh_path in mesh_paths:
        if not mesh_path.exists():
            continue
        vertices.extend(_load_mesh_vertices(str(mesh_path)))
    return tuple(vertices)


def support_surface_world_z(
    model: UrdfModel,
    world: dict[str, tuple[tuple[float, float, float, float], ...]],
    link_name: str,
) -> float:
    if link_name not in world:
        raise KeyError(f"Unknown link '{link_name}'.")
    link = model.links[link_name]
    vertices = _link_collision_vertices(link)
    if not vertices:
        return transform_point(world[link_name], (0.0, 0.0, 0.0))[2]
    return min(transform_point(world[link_name], vertex)[2] for vertex in vertices)


def classify_joint_groups(model: UrdfModel) -> JointGroups:
    leg_joints = []
    foot_joints = []
    arm_joints = []
    hand_joints = []
    finger_joints = []
    torso_joints = []

    for joint_name, joint in model.joints.items():
        if joint_name.endswith("_base_fixed"):
            continue
        joint_name_lower = joint_name.lower()
        child_link_lower = joint.child_link.lower()
        if joint_name_lower.startswith(("thumb", "index", "middle", "ring", "pinky")) or child_link_lower.startswith(
            ("thumb", "index", "middle", "ring", "pinky")
        ):
            finger_joints.append(joint_name)
            continue
        if joint_name_lower.startswith(("hand_",)) or child_link_lower.startswith(("hand_",)) or "wrist" in joint_name_lower:
            hand_joints.append(joint_name)
            continue
        if (
            joint_name_lower.startswith(("shoulder_", "arm_", "forearm_"))
            or child_link_lower.startswith(("shoulder_", "arm_", "forearm_"))
            or any(token in joint_name_lower for token in ("shoulder", "upper_arm", "elbow", "forearm"))
        ):
            arm_joints.append(joint_name)
            continue
        if (
            joint_name_lower.startswith(("thigh_", "leg_"))
            or child_link_lower.startswith(("thigh_", "leg_"))
            or any(token in joint_name_lower for token in ("hip", "knee", "shin"))
        ):
            leg_joints.append(joint_name)
            continue
        if (
            joint_name_lower.startswith(("foot_", "toes_"))
            or child_link_lower.startswith(("foot_", "toes_"))
            or any(token in joint_name_lower for token in ("ankle", "toe"))
        ):
            foot_joints.append(joint_name)
            continue
        if (
            joint_name_lower.startswith(("spine_", "neck_", "head_"))
            or child_link_lower.startswith(("spine_", "neck_", "head_"))
            or any(token in joint_name_lower for token in ("waist", "torso", "neck", "head"))
        ):
            torso_joints.append(joint_name)

    return JointGroups(
        leg_joints=tuple(sorted(leg_joints)),
        foot_joints=tuple(sorted(foot_joints)),
        arm_joints=tuple(sorted(arm_joints)),
        hand_joints=tuple(sorted(hand_joints)),
        finger_joints=tuple(sorted(finger_joints)),
        torso_joints=tuple(sorted(torso_joints)),
    )


def detect_primary_foot_links(model: UrdfModel) -> tuple[str, ...]:
    preferred = [name for name in model.links if name.startswith("foot_")]
    if len(preferred) >= 2:
        return tuple(sorted(preferred))
    fallback = [name for name in model.links if name.startswith("toes_")]
    if len(fallback) >= 2:
        return tuple(sorted(fallback))
    keyword_matches = [name for name in model.links if "foot" in name or "toe" in name]
    return tuple(sorted(keyword_matches))


def detect_support_links(model: UrdfModel) -> tuple[str, ...]:
    support = [name for name in model.links if name.startswith(("foot_", "toes_"))]
    if support:
        return tuple(sorted(support))
    return detect_primary_foot_links(model)


def detect_termination_links(model: UrdfModel) -> tuple[str, ...]:
    preferred_order = ("base_link", "root_x", "spine_01_x", "spine_02_x", "spine_03_x", "neck_x", "head_x")
    preferred = [name for name in preferred_order if name in model.links]
    if preferred:
        return tuple(preferred)
    torso_like = [name for name in model.links if name.startswith(("root", "spine", "neck", "head", "torso", "pelvis"))]
    return tuple(sorted(torso_like))


def total_mass(model: UrdfModel) -> float:
    return sum(link.mass for link in model.links.values() if link.mass > 0.0)


def mass_bearing_links(model: UrdfModel) -> tuple[str, ...]:
    return tuple(sorted(link.name for link in model.links.values() if link.mass > 0.0))


def _joint_motion_transform(joint: UrdfJoint, angle_or_offset: float) -> tuple[tuple[float, float, float, float], ...]:
    if joint.joint_type in {"fixed"}:
        return IDENTITY_4
    if joint.joint_type in {"revolute", "continuous"}:
        return compose_transform((0.0, 0.0, 0.0), axis_angle_matrix(joint.axis_xyz, angle_or_offset))
    if joint.joint_type == "prismatic":
        x, y, z = joint.axis_xyz
        return compose_transform((x * angle_or_offset, y * angle_or_offset, z * angle_or_offset), rpy_to_matrix((0.0, 0.0, 0.0)))
    return IDENTITY_4


def compute_link_world_transforms(
    model: UrdfModel,
    joint_positions: dict[str, float] | None = None,
    root_link: str | None = None,
) -> dict[str, tuple[tuple[float, float, float, float], ...]]:
    joint_positions = joint_positions or {}
    root_link = root_link or (model.root_links[0] if model.root_links else None)
    if root_link is None:
        raise ValueError("URDF has no root link.")

    world: dict[str, tuple[tuple[float, float, float, float], ...]] = {root_link: IDENTITY_4}
    stack = [root_link]
    while stack:
        parent_link = stack.pop()
        for child_link in model.children_by_link.get(parent_link, ()):
            joint = model.joints[model.child_joint_by_link[child_link]]
            joint_origin = compose_transform(joint.origin_xyz, rpy_to_matrix(joint.origin_rpy))
            joint_motion = _joint_motion_transform(joint, joint_positions.get(joint.name, 0.0))
            world[child_link] = matrix_multiply(world[parent_link], matrix_multiply(joint_origin, joint_motion))
            stack.append(child_link)
    return world


def estimate_root_height(
    model: UrdfModel,
    root_link_name: str,
    support_link_names: tuple[str, ...],
    joint_positions: dict[str, float] | None = None,
    clearance: float = 0.03,
) -> float:
    world = compute_link_world_transforms(model, joint_positions=joint_positions, root_link=model.root_links[0])
    if root_link_name not in world:
        raise KeyError(f"Unknown root link '{root_link_name}'.")
    support_points = [support_surface_world_z(model, world, link_name) for link_name in support_link_names if link_name in world]
    if not support_points:
        raise ValueError("Unable to estimate root height without support links.")
    root_z = transform_point(world[root_link_name], (0.0, 0.0, 0.0))[2]
    return root_z - min(support_points) + clearance
