"""Deterministic Landau URDF audit and locomotion joint contract.

This module is deliberately free of Isaac Lab imports so the robot contract can be
checked with ordinary Python before launching the simulator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterable, Sequence


ALGORITHM_ROOT = Path(__file__).resolve().parent
URDF_PATH = ALGORITHM_ROOT / "inputs" / "landau_v10" / "landau_v10_parallel_mesh.urdf"
ROBOT_SPEC_PATH = ALGORITHM_ROOT / "robot_spec.json"
EXPECTED_URDF_SHA256 = "859d3c29930822f77750f6dcc0940e1c7e84393817cdefdcdc36c0025ddb46ca"
EXPECTED_MESH_TREE_SHA256 = "a34be1b4f2732de526c23fd1bc53e945b9e647110432fe466521fb7e73676f73"
LINEAGE = "rabbit_ear_parallel_mesh_recertification_2026_09_05"

SEMANTIC_COMMAND_ORDER = ("forward", "strafe", "yaw")
SIM_COMMAND_ORDER = ("linear_x", "linear_y", "angular_z")

# These joints were selected from their measured zero-pose axes, not their names.
# The shin-roll axes are almost vertical and are therefore locked as distal twists.
ACTION_JOINT_ROLES = {
    "left_hip_pitch_joint": "left_hip_pitch",
    "left_hip_yaw_joint": "left_hip_yaw",
    "left_hip_roll_joint": "left_hip_roll",
    "left_knee_joint": "left_knee_pitch",
    "left_ankle_pitch_joint": "left_ankle_pitch",
    "left_toe_joint": "left_toe_pitch",
    "right_hip_pitch_joint": "right_hip_pitch",
    "right_hip_yaw_joint": "right_hip_yaw",
    "right_hip_roll_joint": "right_hip_roll",
    "right_knee_joint": "right_knee_pitch",
    "right_ankle_pitch_joint": "right_ankle_pitch",
    "right_toe_joint": "right_toe_pitch",
    "waist_yaw_joint": "waist_yaw",
    "waist_roll_joint": "waist_roll",
    "waist_pitch_joint": "waist_pitch",
    "left_shoulder_pitch_joint": "left_shoulder_counter_swing",
    "right_shoulder_pitch_joint": "right_shoulder_counter_swing",
}
ACTION_JOINTS = tuple(ACTION_JOINT_ROLES)

PD_GROUPS = {
    "leg_sagittal": {
        "joints": tuple(
            f"{side}_{joint}_joint"
            for side in ("left", "right")
            for joint in ("hip_pitch", "knee", "ankle_pitch", "toe")
        ),
        "stiffness": 20.0,
        "damping": 1.0,
    },
    "leg_balance": {
        "joints": tuple(
            f"{side}_{joint}_joint"
            for side in ("left", "right")
            for joint in ("hip_yaw", "hip_roll")
        ),
        "stiffness": 12.0,
        "damping": 0.8,
    },
    "waist": {
        "joints": ("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"),
        "stiffness": 10.0,
        "damping": 0.7,
    },
    "shoulder_counter_swing": {
        "joints": ("left_shoulder_pitch_joint", "right_shoulder_pitch_joint"),
        "stiffness": 5.0,
        "damping": 0.4,
    },
}
LOCKED_PD_STIFFNESS = 4.0
LOCKED_PD_DAMPING = 0.35
DERIVED_POSE_FINGER_LIMIT_TOLERANCE_RAD = 0.002
AUTHORITY_PROBE_WAIST_STIFFNESS = 40.0
AUTHORITY_PROBE_WAIST_DAMPING = 5.0
AUTHORITY_PROBE_UPPER_STIFFNESS = 40.0
AUTHORITY_PROBE_UPPER_DAMPING = 10.0

FINGER_JOINTS = tuple(
    [
        f"{side}_thumb_{segment}_joint"
        for side in ("left", "right")
        for segment in ("metacarpal", "proximal", "distal")
    ]
    + [
        f"{side}_{finger}_{segment}_joint"
        for side in ("left", "right")
        for finger in ("index", "middle", "ring", "pinky")
        for segment in ("base", "proximal", "intermediate", "distal")
    ]
)

# Transferred only as the first bounded seed from the clamped settled state of
# gravity_static_pose_release_v1. That run belongs to an invalidated mesh lineage;
# the pose is not proven on the rabbit-ear package until milestone 1 passes again.
# Finger states are intentionally zeroed; every other imported joint preserves
# the diagnostic's recorded float value and corresponding released PD target.
CANONICAL_NOMINAL_NONFINGER_POSITIONS_RAD = {
    "head_yaw_joint": -3.6964324934274373e-09,
    "left_ankle_pitch_joint": -0.11557767540216446,
    "left_elbow_joint": -0.0893860012292862,
    "left_forearm_roll_joint": 0.0018204370280727744,
    "left_hip_pitch_joint": -0.1012502983212471,
    "left_hip_roll_joint": -5.807069828733802e-05,
    "left_hip_yaw_joint": -1.3841487998433877e-05,
    "left_knee_joint": 0.21072299778461456,
    "left_shin_roll_joint": -3.935918357456103e-05,
    "left_shoulder_lift_joint": -0.21318794786930084,
    "left_shoulder_pitch_joint": -0.04680832847952843,
    "left_toe_joint": 2.073991152429233e-11,
    "left_upper_arm_roll_joint": -0.021588018164038658,
    "left_wrist_pitch_joint": -0.013015508651733398,
    "neck_pitch_joint": -4.6467419451801106e-05,
    "right_ankle_pitch_joint": -0.115577831864357,
    "right_elbow_joint": 0.08938424289226532,
    "right_forearm_roll_joint": -0.0018204532098025084,
    "right_hip_pitch_joint": -0.10125018656253815,
    "right_hip_roll_joint": 5.807937486679293e-05,
    "right_hip_yaw_joint": 1.3842480257153511e-05,
    "right_knee_joint": 0.21072296798229218,
    "right_shin_roll_joint": 3.9359278162010014e-05,
    "right_shoulder_lift_joint": 0.21318525075912476,
    "right_shoulder_pitch_joint": -0.046809207648038864,
    "right_toe_joint": -4.708099188288628e-11,
    "right_upper_arm_roll_joint": 0.06532390415668488,
    "right_wrist_pitch_joint": -0.013015508651733398,
    "waist_pitch_joint": -0.2342214286327362,
    "waist_roll_joint": -4.540688891552236e-08,
    "waist_yaw_joint": -2.6809352515755336e-09,
}

CANONICAL_NOMINAL_NONFINGER_TARGETS_RAD = {
    "left_hip_pitch_joint": -0.101250290871,
    "right_hip_pitch_joint": -0.101250179112,
    "waist_yaw_joint": -4.476e-09,
    "left_hip_roll_joint": -5.8070698e-05,
    "right_hip_roll_joint": 5.8079069e-05,
    "waist_roll_joint": -4.2567e-08,
    "left_hip_yaw_joint": -1.3841432e-05,
    "right_hip_yaw_joint": 1.3842453e-05,
    "waist_pitch_joint": -0.234221488237,
    "left_knee_joint": 0.210722982883,
    "right_knee_joint": 0.210722953081,
    "left_shoulder_lift_joint": -0.213187932968,
    "neck_pitch_joint": -4.6429348e-05,
    "right_shoulder_lift_joint": 0.213185280561,
    "left_shin_roll_joint": -3.9359344e-05,
    "right_shin_roll_joint": 3.9359995e-05,
    "left_shoulder_pitch_joint": -0.04680833593,
    "head_yaw_joint": -1.588e-09,
    "right_shoulder_pitch_joint": -0.046809121966,
    "left_ankle_pitch_joint": -0.115577682853,
    "right_ankle_pitch_joint": -0.115577816963,
    "left_upper_arm_roll_joint": -0.021587988362,
    "right_upper_arm_roll_joint": 0.065324053168,
    "left_toe_joint": 0.0,
    "right_toe_joint": 2e-12,
    "left_elbow_joint": -0.089385993779,
    "right_elbow_joint": 0.089384287596,
    "left_forearm_roll_joint": 0.001820431091,
    "right_forearm_roll_joint": -0.001820385922,
    "left_wrist_pitch_joint": -0.013015515171,
    "right_wrist_pitch_joint": -0.013015516102,
}

Matrix3 = tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]
Vector3 = tuple[float, float, float]


def _vector(value: str | None, default: str = "0 0 0") -> Vector3:
    parts = tuple(float(item) for item in (value or default).split())
    if len(parts) != 3:
        raise ValueError(f"Expected three components, got {value!r}")
    return parts  # type: ignore[return-value]


def _matrix_multiply(left: Matrix3, right: Matrix3) -> Matrix3:
    return tuple(
        tuple(sum(left[row][k] * right[k][col] for k in range(3)) for col in range(3))
        for row in range(3)
    )  # type: ignore[return-value]


def _matrix_vector(matrix: Matrix3, vector: Vector3) -> Vector3:
    return tuple(sum(matrix[row][col] * vector[col] for col in range(3)) for row in range(3))  # type: ignore[return-value]


def _vector_add(left: Vector3, right: Vector3) -> Vector3:
    return tuple(left[index] + right[index] for index in range(3))  # type: ignore[return-value]


def _vector_subtract(left: Vector3, right: Vector3) -> Vector3:
    return tuple(left[index] - right[index] for index in range(3))  # type: ignore[return-value]


def _dot(left: Vector3, right: Vector3) -> float:
    return sum(left[index] * right[index] for index in range(3))


def _cross(left: Vector3, right: Vector3) -> Vector3:
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def _normalized(vector: Vector3) -> Vector3:
    norm = math.sqrt(sum(component * component for component in vector))
    if norm <= 1.0e-12:
        raise ValueError("Joint axis must be non-zero")
    return tuple(component / norm for component in vector)  # type: ignore[return-value]


def _is_primary_axis(vector: Vector3, *, tolerance: float = 1.0e-6) -> bool:
    normalized = _normalized(vector)
    return max(abs(component) for component in normalized) >= 1.0 - tolerance


def _rpy_matrix(rpy: Vector3) -> Matrix3:
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rotate_x: Matrix3 = ((1.0, 0.0, 0.0), (0.0, cr, -sr), (0.0, sr, cr))
    rotate_y: Matrix3 = ((cp, 0.0, sp), (0.0, 1.0, 0.0), (-sp, 0.0, cp))
    rotate_z: Matrix3 = ((cy, -sy, 0.0), (sy, cy, 0.0), (0.0, 0.0, 1.0))
    return _matrix_multiply(_matrix_multiply(rotate_z, rotate_y), rotate_x)


def _axis_angle_matrix(axis: Vector3, angle: float) -> Matrix3:
    x, y, z = _normalized(axis)
    cosine, sine = math.cos(angle), math.sin(angle)
    one_minus_cosine = 1.0 - cosine
    return (
        (
            cosine + x * x * one_minus_cosine,
            x * y * one_minus_cosine - z * sine,
            x * z * one_minus_cosine + y * sine,
        ),
        (
            y * x * one_minus_cosine + z * sine,
            cosine + y * y * one_minus_cosine,
            y * z * one_minus_cosine - x * sine,
        ),
        (
            z * x * one_minus_cosine - y * sine,
            z * y * one_minus_cosine + x * sine,
            cosine + z * z * one_minus_cosine,
        ),
    )


def _compose(
    parent: tuple[Matrix3, Vector3], child: tuple[Matrix3, Vector3]
) -> tuple[Matrix3, Vector3]:
    parent_rotation, parent_position = parent
    child_rotation, child_position = child
    return (
        _matrix_multiply(parent_rotation, child_rotation),
        _vector_add(parent_position, _matrix_vector(parent_rotation, child_position)),
    )


def _origin_transform(element: ET.Element | None) -> tuple[Matrix3, Vector3]:
    if element is None:
        return _rpy_matrix((0.0, 0.0, 0.0)), (0.0, 0.0, 0.0)
    return _rpy_matrix(_vector(element.get("rpy"))), _vector(element.get("xyz"))


def semantic_to_sim_command(forward: float, strafe: float, yaw: float) -> tuple[float, float, float]:
    """Map Landau body +Y forward into Isaac's conventional body XY velocity order."""

    return strafe, forward, yaw


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _inertia_positive(inertia: dict[str, float]) -> bool:
    ixx, iyy, izz = inertia["ixx"], inertia["iyy"], inertia["izz"]
    ixy, ixz, iyz = inertia["ixy"], inertia["ixz"], inertia["iyz"]
    second_minor = ixx * iyy - ixy * ixy
    determinant = (
        ixx * (iyy * izz - iyz * iyz)
        - ixy * (ixy * izz - iyz * ixz)
        + ixz * (ixy * iyz - iyy * ixz)
    )
    return ixx > 0.0 and second_minor > 0.0 and determinant > 0.0


def _joint_world_transforms(
    root: ET.Element, joint_positions: dict[str, float] | None = None
) -> tuple[dict[str, tuple[Matrix3, Vector3]], dict[str, ET.Element]]:
    joint_positions = joint_positions or {}
    joints = root.findall("joint")
    child_joints = {joint.find("child").get("link"): joint for joint in joints}  # type: ignore[union-attr]
    link_transforms: dict[str, tuple[Matrix3, Vector3]] = {
        "base_link": (_rpy_matrix((0.0, 0.0, 0.0)), (0.0, 0.0, 0.0))
    }
    pending = dict(child_joints)
    while pending:
        progressed = False
        for child, joint in list(pending.items()):
            parent = joint.find("parent").get("link")  # type: ignore[union-attr]
            if parent in link_transforms:
                transform = _compose(link_transforms[parent], _origin_transform(joint.find("origin")))
                if joint.get("type") != "fixed":
                    axis = _normalized(_vector(joint.find("axis").get("xyz")))  # type: ignore[union-attr]
                    transform = _compose(
                        transform,
                        (_axis_angle_matrix(axis, float(joint_positions.get(joint.get("name"), 0.0))), (0.0, 0.0, 0.0)),
                    )
                link_transforms[child] = transform
                del pending[child]
                progressed = True
        if not progressed:
            raise ValueError(f"Disconnected or cyclic URDF links: {sorted(pending)}")
    return link_transforms, child_joints


def _stl_vertices(path: Path) -> Iterable[Vector3]:
    with path.open("rb") as stream:
        header = stream.read(84)
        if len(header) != 84:
            raise ValueError(f"Invalid binary STL header: {path}")
        triangle_count = struct.unpack_from("<I", header, 80)[0]
        for _ in range(triangle_count):
            record = stream.read(50)
            if len(record) != 50:
                raise ValueError(f"Truncated binary STL: {path}")
            values = struct.unpack_from("<12fH", record)
            for offset in (3, 6, 9):
                yield values[offset], values[offset + 1], values[offset + 2]


def _collision_world_points(
    root: ET.Element,
    urdf_path: Path,
    transforms: dict[str, tuple[Matrix3, Vector3]],
) -> dict[str, list[Vector3]]:
    result: dict[str, list[Vector3]] = {}
    for link in root.findall("link"):
        link_name = link.get("name")
        points: list[Vector3] = []
        for collision in link.findall("collision"):
            mesh = collision.find("geometry/mesh")
            if mesh is None:
                continue
            mesh_path = (urdf_path.parent / mesh.get("filename")).resolve()  # type: ignore[arg-type]
            collision_transform = _compose(transforms[link_name], _origin_transform(collision.find("origin")))
            rotation, position = collision_transform
            points.extend(_vector_add(position, _matrix_vector(rotation, vertex)) for vertex in _stl_vertices(mesh_path))
        if points:
            result[link_name] = points
    return result


def visual_local_aabb_corners(urdf_path: Path = URDF_PATH) -> dict[str, list[list[float]]]:
    """Return conservative local-frame corners for every visual-mesh link.

    The proof camera uses these corners with Isaac's runtime body transforms.
    Unlike a link-origin proxy, this catches an imported visual mesh that is
    partly behind the camera or outside the viewport.  Corners bound all STL
    vertices after applying each URDF visual origin.
    """

    root = ET.parse(urdf_path).getroot()
    result: dict[str, list[list[float]]] = {}
    for link in root.findall("link"):
        link_name = link.get("name")
        points: list[Vector3] = []
        for visual in link.findall("visual"):
            mesh = visual.find("geometry/mesh")
            if mesh is None:
                continue
            scale = _vector(mesh.get("scale"), "1 1 1")
            mesh_path = (urdf_path.parent / mesh.get("filename")).resolve()  # type: ignore[arg-type]
            rotation, position = _origin_transform(visual.find("origin"))
            for vertex in _stl_vertices(mesh_path):
                scaled = tuple(vertex[index] * scale[index] for index in range(3))
                points.append(_vector_add(position, _matrix_vector(rotation, scaled)))
        if not points:
            continue
        minimum = [min(point[axis] for point in points) for axis in range(3)]
        maximum = [max(point[axis] for point in points) for axis in range(3)]
        result[link_name] = [
            [x, y, z]
            for x in (minimum[0], maximum[0])
            for y in (minimum[1], maximum[1])
            for z in (minimum[2], maximum[2])
        ]
    return result


def _convex_hull_xy(points: Sequence[Vector3]) -> list[list[float]]:
    """Return the deterministic counter-clockwise hull of XY projected points."""

    unique = sorted({(round(point[0], 9), round(point[1], 9)) for point in points})
    if len(unique) <= 1:
        return [list(point) for point in unique]

    def cross(origin, left, right) -> float:
        return (left[0] - origin[0]) * (right[1] - origin[1]) - (
            left[1] - origin[1]
        ) * (right[0] - origin[0])

    lower = []
    for point in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0.0:
            lower.pop()
        lower.append(point)
    upper = []
    for point in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0.0:
            upper.pop()
        upper.append(point)
    return [list(point) for point in lower[:-1] + upper[:-1]]


def support_polygon_margin(point: Sequence[float], polygon: Sequence[Sequence[float]]) -> float:
    margins = []
    for index, start in enumerate(polygon):
        end = polygon[(index + 1) % len(polygon)]
        edge_x, edge_y = end[0] - start[0], end[1] - start[1]
        length = math.hypot(edge_x, edge_y)
        margins.append(
            (edge_x * (point[1] - start[1]) - edge_y * (point[0] - start[0])) / length
        )
    return min(margins)


def _pose_center_of_mass(root: ET.Element, transforms: dict[str, tuple[Matrix3, Vector3]]) -> Vector3:
    total_mass = 0.0
    weighted = [0.0, 0.0, 0.0]
    for link in root.findall("link"):
        inertial = link.find("inertial")
        if inertial is None:
            continue
        mass = float(inertial.find("mass").get("value"))  # type: ignore[union-attr]
        center = _compose(
            transforms[link.get("name")], _origin_transform(inertial.find("origin"))
        )[1]
        total_mass += mass
        for axis in range(3):
            weighted[axis] += mass * center[axis]
    return tuple(value / total_mass for value in weighted)  # type: ignore[return-value]


def collision_world_bounds(urdf_path: Path = URDF_PATH) -> dict[str, dict[str, list[float]]]:
    """Return zero-pose world bounds for every link with collision geometry."""

    root = ET.parse(urdf_path).getroot()
    transforms, _ = _joint_world_transforms(root)
    result: dict[str, dict[str, list[float]]] = {}
    for link_name, points in _collision_world_points(root, urdf_path, transforms).items():
        result[link_name] = {
            "minimum": [min(point[axis] for point in points) for axis in range(3)],
            "maximum": [max(point[axis] for point in points) for axis in range(3)],
        }
    return result


def zero_pose_ground_support(urdf_path: Path = URDF_PATH, tolerance_m: float = 0.0005) -> dict:
    """Measure the actual near-ground collision vertices of both feet and toes."""

    root = ET.parse(urdf_path).getroot()
    transforms, _ = _joint_world_transforms(root)
    support_links = ("foot_l", "foot_r", "toes_01_l", "toes_01_r")
    points_by_link = _collision_world_points(root, urdf_path, transforms)
    all_points = [point for name in support_links for point in points_by_link[name]]
    ground_z = min(point[2] for point in all_points)
    contacts = [point for point in all_points if point[2] <= ground_z + tolerance_m]
    links = {}
    for name in support_links:
        link_contacts = [point for point in points_by_link[name] if point[2] <= ground_z + tolerance_m]
        links[name] = {
            "near_ground_vertex_count": len(link_contacts),
            "minimum_z_m": round(min(point[2] for point in points_by_link[name]), 9),
            "candidate_contact_hull_xy_m": _convex_hull_xy(link_contacts),
        }
    return {
        "method": "collision mesh vertices within tolerance of the zero-pose minimum Z",
        "support_links": list(support_links),
        "links": links,
        "ground_z_m": round(ground_z, 9),
        "vertex_tolerance_m": tolerance_m,
        "near_ground_vertex_count": len(contacts),
        "candidate_contact_hull_xy_m": _convex_hull_xy(contacts),
        "flat_ground_contact_normal_w": [0.0, 0.0, 1.0],
        "gravity_direction_w": [0.0, 0.0, -1.0],
        "gravity_normal_dot": -1.0,
        "support_aabb_xy_m": {
            "minimum": [round(min(point[axis] for point in contacts), 9) for axis in (0, 1)],
            "maximum": [round(max(point[axis] for point in contacts), 9) for axis in (0, 1)],
        },
    }


def analyze_pose_geometry(
    joint_positions: dict[str, float],
    urdf_path: Path = URDF_PATH,
    tolerance_m: float = 0.0005,
) -> dict:
    """Measure COM and collision support for a candidate pose from exact URDF FK."""

    root = ET.parse(urdf_path).getroot()
    transforms, _ = _joint_world_transforms(root, joint_positions)
    total_mass = 0.0
    weighted = [0.0, 0.0, 0.0]
    link_mass_properties = {}
    for link in root.findall("link"):
        inertial = link.find("inertial")
        if inertial is None:
            continue
        mass = float(inertial.find("mass").get("value"))  # type: ignore[union-attr]
        center = _compose(
            transforms[link.get("name")], _origin_transform(inertial.find("origin"))
        )[1]
        link_mass_properties[link.get("name")] = (mass, center)
        total_mass += mass
        for axis in range(3):
            weighted[axis] += mass * center[axis]
    com = [value / total_mass for value in weighted]
    support_links = ("foot_l", "foot_r", "toes_01_l", "toes_01_r")
    points_by_link = _collision_world_points(root, urdf_path, transforms)
    support_points = [point for link in support_links for point in points_by_link[link]]
    ground_z = min(point[2] for point in support_points)
    contacts = [point for point in support_points if point[2] <= ground_z + tolerance_m]
    hull = _convex_hull_xy(contacts)

    contact_counts = {
        link: sum(point[2] <= ground_z + tolerance_m for point in points_by_link[link])
        for link in support_links
    }
    gravity_torques = _fixed_root_gravity_torques(root, transforms, link_mass_properties)
    effort_limits = {
        joint.get("name"): float(joint.find("limit").get("effort"))  # type: ignore[union-attr]
        for joint in root.findall("joint")
        if joint.get("type") != "fixed"
    }
    return {
        "method": "exact current-URDF FK, inertial origins, and collision vertices",
        "joint_positions_rad": dict(sorted(joint_positions.items())),
        "ground_aligned_base_z_m": round(-ground_z, 9),
        "center_of_mass_m": [round(value, 9) for value in com],
        "center_of_mass_height_above_ground_m": round(com[2] - ground_z, 9),
        "support_hull_xy_m": hull,
        "support_aabb_xy_m": {
            "minimum": [round(min(point[axis] for point in contacts), 9) for axis in (0, 1)],
            "maximum": [round(max(point[axis] for point in contacts), 9) for axis in (0, 1)],
        },
        "support_margin_m": round(support_polygon_margin(com[:2], hull), 9),
        "near_ground_contact_vertex_count_by_link": contact_counts,
        "fixed_root_gravity_torque_nm": {
            name: round(value, 9) for name, value in sorted(gravity_torques.items())
        },
        "maximum_fixed_root_gravity_torque_limit_fraction": round(
            max(abs(value) / effort_limits[name] for name, value in gravity_torques.items()), 9
        ),
    }


def derive_static_pose(urdf_path: Path = URDF_PATH) -> dict:
    """Derive a small crouch and center its COM using only exact current assets."""

    # Half of the installed G1 leg flexion retains Landau's heel/toe support;
    # the full G1 angles do not. Search waist pitch at 1 mrad resolution for
    # maximum collision-hull margin without moving hands, fingers, or head.
    leg_pose = {
        f"{side}_hip_pitch_joint": -0.10
        for side in ("left", "right")
    }
    leg_pose.update({f"{side}_knee_joint": 0.21 for side in ("left", "right")})
    leg_pose.update({f"{side}_ankle_pitch_joint": -0.115 for side in ("left", "right")})
    root = ET.parse(urdf_path).getroot()
    seed_geometry = analyze_pose_geometry(leg_pose, urdf_path)
    support_hull = seed_geometry["support_hull_xy_m"]
    contacts = seed_geometry["near_ground_contact_vertex_count_by_link"]
    bilateral = (
        contacts["foot_l"] + contacts["toes_01_l"] > 0
        and contacts["foot_r"] + contacts["toes_01_r"] > 0
    )
    if not bilateral:
        raise ValueError("static-pose leg seed lacks bilateral ground support")
    candidates = []
    for step in range(351):
        pose = {**leg_pose, "waist_pitch_joint": -step / 1000.0}
        transforms, _ = _joint_world_transforms(root, pose)
        com = _pose_center_of_mass(root, transforms)
        candidates.append((support_polygon_margin(com[:2], support_hull), pose))
    _, selected_pose = max(
        candidates,
        key=lambda item: (item[0], -abs(item[1]["waist_pitch_joint"])),
    )
    selected = analyze_pose_geometry(selected_pose, urdf_path)
    selected["derivation"] = {
        "leg_seed": "0.5 * installed Isaac Lab G1 hip/knee/ankle flexion",
        "leg_seed_rad": leg_pose,
        "waist_search_rad": {"minimum": -0.35, "maximum": 0.0, "increment": 0.001},
        "selection": "maximum signed COM distance inside the bilateral near-ground collision hull",
        "upper_body_offsets_during_geometry_search": "zero; fixed-root settling derives only necessary offsets",
    }
    return selected


def passive_actuator_groups(movable_names: Sequence[str], *, authority_probe: bool = False) -> dict:
    """Return a non-overlapping actuator contract for normal or diagnostic holding.

    The authority probe deliberately leaves all lower-body gains and the zero pose
    unchanged.  Only waist and upper-body holding authority changes, so a bounded
    run can separate compliant upper-body motion from a support-pose failure.
    """

    names = set(movable_names)
    controlled = {joint for group in PD_GROUPS.values() for joint in group["joints"]}
    locked = names - controlled
    if not authority_probe:
        groups = {
            name: {**group, "joints": list(group["joints"])}
            for name, group in PD_GROUPS.items()
        }
        groups["locked"] = {
            "joints": sorted(locked),
            "stiffness": LOCKED_PD_STIFFNESS,
            "damping": LOCKED_PD_DAMPING,
        }
    else:
        waist = set(PD_GROUPS["waist"]["joints"])
        lower = set(PD_GROUPS["leg_sagittal"]["joints"]) | set(PD_GROUPS["leg_balance"]["joints"])
        distal_twist = {"left_shin_roll_joint", "right_shin_roll_joint"}
        upper = names - lower - distal_twist - waist
        groups = {
            "leg_sagittal": {**PD_GROUPS["leg_sagittal"], "joints": list(PD_GROUPS["leg_sagittal"]["joints"])},
            "leg_balance": {**PD_GROUPS["leg_balance"], "joints": list(PD_GROUPS["leg_balance"]["joints"])},
            "distal_twist_hold": {
                "joints": sorted(distal_twist),
                "stiffness": LOCKED_PD_STIFFNESS,
                "damping": LOCKED_PD_DAMPING,
            },
            "waist_authority": {
                "joints": sorted(waist),
                "stiffness": AUTHORITY_PROBE_WAIST_STIFFNESS,
                "damping": AUTHORITY_PROBE_WAIST_DAMPING,
            },
            "upper_body_authority": {
                "joints": sorted(upper),
                "stiffness": AUTHORITY_PROBE_UPPER_STIFFNESS,
                "damping": AUTHORITY_PROBE_UPPER_DAMPING,
            },
        }
    flattened = [joint for group in groups.values() for joint in group["joints"]]
    if len(flattened) != len(set(flattened)) or set(flattened) != names:
        raise ValueError("actuator groups must form an exact, non-overlapping movable-joint partition")
    return groups


def _fixed_root_gravity_torques(
    root: ET.Element,
    transforms: dict[str, tuple[Matrix3, Vector3]],
    link_mass_properties: dict[str, tuple[float, Vector3]],
) -> dict[str, float]:
    """Compute fixed-root generalized gravity torques from the current URDF."""

    children: dict[str, list[str]] = {}
    for joint in root.findall("joint"):
        parent = joint.find("parent").get("link")  # type: ignore[union-attr]
        child = joint.find("child").get("link")  # type: ignore[union-attr]
        children.setdefault(parent, []).append(child)

    def subtree(start: str) -> set[str]:
        result, pending = set(), [start]
        while pending:
            link = pending.pop()
            if link in result:
                continue
            result.add(link)
            pending.extend(children.get(link, []))
        return result

    torques = {}
    gravity: Vector3 = (0.0, 0.0, -9.81)
    for joint in root.findall("joint"):
        if joint.get("type") == "fixed":
            continue
        name = joint.get("name")
        parent = joint.find("parent").get("link")  # type: ignore[union-attr]
        child = joint.find("child").get("link")  # type: ignore[union-attr]
        joint_transform = _compose(transforms[parent], _origin_transform(joint.find("origin")))
        axis = _normalized(
            _matrix_vector(
                joint_transform[0],
                _normalized(_vector(joint.find("axis").get("xyz"))),  # type: ignore[union-attr]
            )
        )
        moment = (0.0, 0.0, 0.0)
        for link in subtree(child):
            if link not in link_mass_properties:
                continue
            mass, center = link_mass_properties[link]
            force = tuple(mass * value for value in gravity)  # type: ignore[assignment]
            moment = _vector_add(moment, _cross(_vector_subtract(center, joint_transform[1]), force))
        torques[name] = _dot(axis, moment)
    return torques


def build_robot_spec(urdf_path: Path = URDF_PATH) -> dict:
    root = ET.parse(urdf_path).getroot()
    urdf_sha256 = _sha256(urdf_path)
    links = root.findall("link")
    joints = root.findall("joint")
    transforms, _ = _joint_world_transforms(root)
    movable_joints = [joint for joint in joints if joint.get("type") != "fixed"]
    movable_names = [joint.get("name") for joint in movable_joints]
    unknown_actions = set(ACTION_JOINTS) - set(movable_names)
    if unknown_actions:
        raise ValueError(f"Action joints missing from URDF: {sorted(unknown_actions)}")
    locked_names = [name for name in movable_names if name not in ACTION_JOINTS]
    canonical_positions = {
        name: CANONICAL_NOMINAL_NONFINGER_POSITIONS_RAD.get(name, 0.0)
        for name in movable_names
    }
    canonical_targets = {
        name: CANONICAL_NOMINAL_NONFINGER_TARGETS_RAD.get(name, 0.0)
        for name in movable_names
    }
    if set(CANONICAL_NOMINAL_NONFINGER_POSITIONS_RAD) - set(movable_names):
        raise ValueError("canonical pose contains joints absent from the current URDF")
    if set(CANONICAL_NOMINAL_NONFINGER_TARGETS_RAD) - set(movable_names):
        raise ValueError("canonical targets contain joints absent from the current URDF")

    joint_records = []
    for joint in movable_joints:
        name = joint.get("name")
        parent_name = joint.find("parent").get("link")  # type: ignore[union-attr]
        joint_transform = _compose(transforms[parent_name], _origin_transform(joint.find("origin")))
        local_axis = _normalized(_vector(joint.find("axis").get("xyz")))  # type: ignore[union-attr]
        world_axis = _normalized(_matrix_vector(joint_transform[0], local_axis))
        dominant_index = max(range(3), key=lambda index: abs(world_axis[index]))
        limit = joint.find("limit")
        joint_records.append(
            {
                "name": name,
                "control": "action" if name in ACTION_JOINTS else "locked_pd",
                "role": ACTION_JOINT_ROLES.get(name, "non_locomotion_locked"),
                "parent_link": parent_name,
                "child_link": joint.find("child").get("link"),  # type: ignore[union-attr]
                "axis_joint_frame": list(local_axis),
                "axis_world_zero_pose": [round(value, 9) for value in world_axis],
                "axis_evidence": "source_urdf_forward_kinematics",
                "source_axis_is_primary": _is_primary_axis(local_axis),
                "runtime_importer_axis_verification_required": True,
                "dominant_world_axis": f"{'+' if world_axis[dominant_index] >= 0 else '-'}{'XYZ'[dominant_index]}",
                "limits_rad": [float(limit.get("lower")), float(limit.get("upper"))],  # type: ignore[union-attr]
                "effort_limit": float(limit.get("effort")),  # type: ignore[union-attr]
                "velocity_limit": float(limit.get("velocity")),  # type: ignore[union-attr]
                "nominal_position_rad": canonical_positions[name],
                "nominal_target_rad": canonical_targets[name],
            }
        )

    masses = []
    link_mass_properties: dict[str, tuple[float, Vector3]] = {}
    mass_weighted_positions = [0.0, 0.0, 0.0]
    invalid_inertias = []
    for link in links:
        inertial = link.find("inertial")
        if inertial is None:
            continue
        mass = float(inertial.find("mass").get("value"))  # type: ignore[union-attr]
        inertia = {key: float(inertial.find("inertia").get(key, "0")) for key in ("ixx", "ixy", "ixz", "iyy", "iyz", "izz")}  # type: ignore[union-attr]
        masses.append(mass)
        inertial_position = _compose(
            transforms[link.get("name")], _origin_transform(inertial.find("origin"))
        )[1]
        link_mass_properties[link.get("name")] = (mass, inertial_position)
        for axis in range(3):
            mass_weighted_positions[axis] += mass * inertial_position[axis]
        if not _inertia_positive(inertia):
            invalid_inertias.append(link.get("name"))

    mesh_paths = [
        (urdf_path.parent / mesh.get("filename")).resolve()  # type: ignore[arg-type]
        for mesh in root.findall(".//mesh")
    ]
    missing_meshes = sorted({str(path) for path in mesh_paths if not path.is_file()})
    unique_mesh_paths = sorted(set(mesh_paths))
    mesh_tree_digest = hashlib.sha256()
    for path in unique_mesh_paths:
        mesh_tree_digest.update(path.name.encode("utf-8"))
        if path.is_file():
            mesh_tree_digest.update(bytes.fromhex(_sha256(path)))
    mesh_tree_sha256 = mesh_tree_digest.hexdigest()
    if missing_meshes:
        raise ValueError(f"Landau mesh package is incomplete: {missing_meshes}")
    if urdf_path == URDF_PATH and mesh_tree_sha256 != EXPECTED_MESH_TREE_SHA256:
        raise ValueError(
            "Landau mesh package differs from the latest usd_parallel_urdf contract: "
            f"{mesh_tree_sha256} != {EXPECTED_MESH_TREE_SHA256}"
        )

    fixed_root = next(joint for joint in joints if joint.find("parent").get("link") == "base_link")  # type: ignore[union-attr]
    fixed_origin = fixed_root.find("origin")
    bounds = collision_world_bounds(urdf_path) if not missing_meshes else {}
    total_mass = sum(masses)
    gravity_torques = _fixed_root_gravity_torques(root, transforms, link_mass_properties)
    default_groups = passive_actuator_groups(movable_names)
    default_gains = {
        joint: (group["stiffness"], group["damping"])
        for group in default_groups.values()
        for joint in group["joints"]
    }
    for record in joint_records:
        torque = gravity_torques[record["name"]]
        stiffness, damping = default_gains[record["name"]]
        record["zero_pose_fixed_root_gravity_torque_nm"] = round(torque, 9)
        record["zero_pose_linear_pd_gravity_error_rad"] = round(torque / stiffness, 9)
        record["nominal_stiffness_nm_per_rad"] = stiffness
        record["nominal_damping_nm_s_per_rad"] = damping
    zero_pose_com = [round(value / total_mass, 9) for value in mass_weighted_positions]
    ground_support = zero_pose_ground_support(urdf_path) if not missing_meshes else {}
    ground_support["zero_pose_com_projection_xy_m"] = zero_pose_com[:2]
    ground_support["zero_pose_com_support_aabb_margin_m"] = {
        "negative_x": round(zero_pose_com[0] - ground_support["support_aabb_xy_m"]["minimum"][0], 9),
        "positive_x": round(ground_support["support_aabb_xy_m"]["maximum"][0] - zero_pose_com[0], 9),
        "negative_y": round(zero_pose_com[1] - ground_support["support_aabb_xy_m"]["minimum"][1], 9),
        "positive_y": round(ground_support["support_aabb_xy_m"]["maximum"][1] - zero_pose_com[1], 9),
    }
    static_pose_candidate = derive_static_pose(urdf_path)
    canonical_pose_geometry = analyze_pose_geometry(canonical_positions, urdf_path)
    return {
        "version": 8,
        "lineage": LINEAGE,
        "source": {
            "urdf_path": str(urdf_path.relative_to(ALGORITHM_ROOT.parent.parent)),
            "urdf_sha256": urdf_sha256,
            "expected_urdf_sha256": EXPECTED_URDF_SHA256,
            "derived_urdf_used": False,
            "mesh_reference_count": len(mesh_paths),
            "unique_mesh_count": len(unique_mesh_paths),
            "missing_meshes": missing_meshes,
            "mesh_tree_sha256": mesh_tree_sha256,
            "expected_mesh_tree_sha256": EXPECTED_MESH_TREE_SHA256,
        },
        "structure": {
            "link_count": len(links),
            "joint_count": len(joints),
            "movable_joint_count": len(movable_joints),
            "total_mass_kg": round(total_mass, 9),
            "zero_pose_center_of_mass_m": zero_pose_com,
            "invalid_inertia_links": invalid_inertias,
            "root_link": "base_link",
            "skeleton_root_link": fixed_root.find("child").get("link"),  # type: ignore[union-attr]
            "skeleton_root_origin_xyz": list(_vector(fixed_origin.get("xyz"))),  # type: ignore[union-attr]
            "skeleton_root_origin_rpy": list(_vector(fixed_origin.get("rpy"))),  # type: ignore[union-attr]
        },
        "frames": {
            "body_forward_axis": "+Y",
            "body_strafe_axis": "+X",
            "body_up_axis": "+Z",
            "semantic_command_order": list(SEMANTIC_COMMAND_ORDER),
            "sim_command_order": list(SIM_COMMAND_ORDER),
            "mapping": {"linear_x": "strafe", "linear_y": "forward", "angular_z": "yaw"},
        },
        "importer_axis_contract": {
            "authority": "installed Isaac Sim 4.5 URDF importer and PhysX USD schemas",
            "source_interpretation": (
                "axis_world_zero_pose is computed from the current URDF kinematic chain; it is not inferred from "
                "the joint name and is not assumed to remain the authored USD primary axis"
            ),
            "observed_importer_behavior": (
                "Isaac Sim warns that non-primary URDF axes are aligned to a PhysX X primary axis by rotating "
                "the imported body and joint local frames; in the generated USD, an unauthored physics:axis "
                "therefore means the PhysX X fallback"
            ),
            "runtime_requirement": (
                "Each Isaac dynamics validation must record the imported USD primary axis and local frame, "
                "reconstruct its zero-pose world axis, and verify directed alignment with this source contract"
            ),
            "minimum_directed_axis_cosine": 0.995,
            "runtime_evidence_artifact": (
                "algorithms/urdf_learn_wasd_walk/outputs/stand_zero_signal_30s_no_reset/"
                "dynamics_validation.json"
            ),
        },
        "nominal_pose": {
            "base_position_m": [0.0, 0.0, canonical_pose_geometry["ground_aligned_base_z_m"]],
            "base_orientation_wxyz": [1.0, 0.0, 0.0, 0.0],
            "joint_positions_rad": canonical_positions,
            "joint_position_targets_rad": canonical_targets,
            "geometry": canonical_pose_geometry,
            "provenance": {
                "kind": "transferred_pose_seed_from_invalidated_mesh_lineage",
                "current_asset_status": "awaiting_rabbit_ear_mesh_passive_recertification",
                "source_experiment": "gravity_static_pose_release_v1",
                "source_run_identity": "20260904T071956.210897Z",
                "source_evidence": (
                    "algorithms/urdf_learn_wasd_walk/outputs/stand_zero_signal_30s_no_reset/"
                    "experiments/gravity_static_pose_release_v1_20260904T071956Z.json"
                ),
                "source_evidence_sha256": "6bbd26111a61026ae33ecdc7ee7b296e2faefbb7b4500530c320635b98519765",
                "source_probe_status": "failed_due_to_invalid_fixed_root_contact_criterion",
                "free_root_result": {
                    "duration_s": 3.0,
                    "reset_count": 0,
                    "done_count": 0,
                    "fall_count": 0,
                    "max_reference_tilt_rad": 0.0412149,
                    "root_height_drop_m": 0.00143432,
                    "horizontal_drift_m": 0.00062895,
                    "minimum_positive_y_support_margin_m": 0.04963283,
                    "peak_support_force_body_weight_ratio": 1.69168651,
                },
                "canonicalization": {
                    "nonfinger_positions": "exact clamped settled_geometry values from source evidence",
                    "nonfinger_targets": "raw released targets from source evidence",
                    "finger_positions_and_targets": "zeroed to prevent fixed-root numerical noise from selecting pose",
                    "zeroed_finger_joints": list(FINGER_JOINTS),
                },
            },
            "zero_pose_collision_bounds": bounds,
            "zero_pose_ground_support": ground_support,
        },
        "action_joints": list(ACTION_JOINTS),
        "locked_joints": locked_names,
        "pd": {
            "groups": {
                name: {**group, "joints": list(group["joints"])}
                for name, group in PD_GROUPS.items()
            },
            "locked": {"stiffness": LOCKED_PD_STIFFNESS, "damping": LOCKED_PD_DAMPING},
            "stability_experiment_history": {
                "status": "invalidated_asset_history",
                "mesh_tree_sha256": "e912ac2e7fcc16a52d726ef410c2b0eb860727d033e07b1728d63a3f906d4da0",
                "use": "hypothesis history only; no recorded result promotes the current rabbit-ear lineage",
            },
            "stability_experiments": [
                {
                    "id": "ankle_pitch_contact_damping_v1",
                    "status": "rejected",
                    "change": {"stiffness": 20.0, "damping_before": 1.0, "damping_tested": 4.0},
                    "result": {
                        "duration_s": 3.0, "first_fall_time_s": 0.958,
                        "max_reference_tilt_rad": 1.3513303,
                        "semantic_forward_displacement_m": 0.40405083,
                        "ankle_torque_saturated": False,
                    },
                    "evidence": (
                        "algorithms/urdf_learn_wasd_walk/outputs/stand_zero_signal_30s_no_reset/"
                        "experiments/ankle_pitch_contact_damping_v1_20260904T043639Z.json"
                    ),
                    "evidence_sha256": "a45c224c1d95edc9b02dea9ced0de9ebecd9d6fa064f1db0029143ccf75b77b3",
                },
                {
                    "id": "ground_aligned_spawn_v1",
                    "status": "insufficient_supported",
                    "scope": "base spawn Z only; restore baseline ankle damping",
                    "change": {"base_z_before_m": 0.002, "base_z_after_m": 0.0},
                    "reason": {
                        "zero_pose_collision_ground_z_m": ground_support.get("ground_z_m"),
                        "previous_first_contact_time_s": 0.016,
                        "previous_sum_of_individual_body_peak_forces_n": 99.814347,
                        "robot_weight_n": round(total_mass * 9.81, 6),
                        "interpretation": "remove the artificial free-fall and asymmetric impact before changing pose or gains",
                    },
                    "result": {
                        "duration_s": 3.0,
                        "first_fall_time_s": 1.212,
                        "first_zero_pose_support_exit_time_s": 0.696,
                        "max_reference_tilt_rad": 1.34737886,
                        "peak_support_force_body_weight_ratio": 2.73713852,
                        "interpretation": "initial impact improved materially, but zero-pose long-term stability did not pass",
                    },
                    "evidence": (
                        "algorithms/urdf_learn_wasd_walk/outputs/stand_zero_signal_30s_no_reset/"
                        "experiments/ground_aligned_spawn_v1_20260904T054229Z.json"
                    ),
                    "evidence_sha256": "3732537deaf5ac6e6cb909bd5387f8cd99b8c5c9c53fb3d8e7ca9b8127875eb8",
                },
                {
                    "id": "zero_pose_upper_body_authority_probe_v1",
                    "status": "rejected",
                    "scope": "zero pose, ground alignment, lower-body gains, effort limits, and contact geometry unchanged",
                    "hypothesis": (
                        "If the zero pose survives with explicit upper-body holding, compliant upper-body motion is causal; "
                        "if support exit remains, the next experiment must change nominal posture/contact equilibrium"
                    ),
                    "change": {
                        "waist_stiffness_nm_per_rad": AUTHORITY_PROBE_WAIST_STIFFNESS,
                        "waist_damping_nm_s_per_rad": AUTHORITY_PROBE_WAIST_DAMPING,
                        "upper_body_stiffness_nm_per_rad": AUTHORITY_PROBE_UPPER_STIFFNESS,
                        "upper_body_damping_nm_s_per_rad": AUTHORITY_PROBE_UPPER_DAMPING,
                    },
                    "source_comparison": {
                        "installed_file": "IsaacLab/source/isaaclab_assets/isaaclab_assets/robots/unitree.py",
                        "G1_and_H1_arm_stiffness_nm_per_rad": 40.0,
                        "G1_and_H1_arm_damping_nm_s_per_rad": 10.0,
                        "H1_torso_damping_nm_s_per_rad": 5.0,
                        "adaptation": "retain Landau effort/velocity limits and all lower-body gains",
                    },
                    "result": {
                        "duration_s": 3.0,
                        "first_fall_time_s": 1.156,
                        "first_zero_pose_support_exit_time_s": 0.662,
                        "max_reference_tilt_rad": 1.35250473,
                        "horizontal_drift_m": 0.43666864,
                        "peak_support_force_body_weight_ratio": 3.36800819,
                        "right_shoulder_pitch_saturation_fraction": 0.20866667,
                        "interpretation": "broad upper-body authority injected energy and worsened stability",
                    },
                    "evidence": (
                        "algorithms/urdf_learn_wasd_walk/outputs/stand_zero_signal_30s_no_reset/"
                        "experiments/zero_pose_upper_body_authority_probe_v1_20260904T061528Z.json"
                    ),
                    "evidence_sha256": "0876def2384e1e14963c8935ac03ed78e8c68ee0be52e0d2bac703bbce3c89a2",
                },
                {
                    "id": "gravity_static_pose_release_v1",
                    "status": "invalidated_asset_history",
                    "scope": "derive with pinned root and baseline low-authority PD, then release in the same Isaac process",
                    "hypothesis": (
                        "A small collision-compatible crouch with centered COM and measured PD preload will avoid "
                        "the zero-pose forward support exit without applying high authority to hands or fingers"
                    ),
                    "geometry_candidate": static_pose_candidate,
                    "runtime_derivation": {
                        "fixed_root_method": "rewrite audited root pose and zero velocity before every settling step",
                        "settled_pose": "mean joint positions over the final settling window",
                        "required_torque": "mean applied actuator torque over the final settling window",
                        "released_pd_target": "settled position + required torque / baseline stiffness",
                        "release": "restore free root at collision-derived ground-aligned height with zero velocity",
                    },
                    "limit_audit": {
                        "finger_joints": list(FINGER_JOINTS),
                        "tolerance_rad": DERIVED_POSE_FINGER_LIMIT_TOLERANCE_RAD,
                        "tolerance_deg": round(
                            math.degrees(DERIVED_POSE_FINGER_LIMIT_TOLERANCE_RAD), 9
                        ),
                        "maximum_tip_displacement_at_40mm_m": round(
                            0.04 * DERIVED_POSE_FINGER_LIMIT_TOLERANCE_RAD, 9
                        ),
                        "policy": (
                            "Only non-locomotion finger desired positions and PD targets within tolerance are "
                            "clamped to the imported hard limit; raw and clamped values are recorded. Any action-"
                            "joint violation or finger excess beyond tolerance remains a hard failure."
                        ),
                    },
                    "attempts": [
                        {
                            "run_identity": "20260904T064751.568514Z",
                            "status": "failed_to_execute_before_release",
                            "runtime_stage": "fixed_root_gravity_settling",
                            "reason": "strict floating-point comparison rejected finger targets at most 0.00114611 rad above a zero upper limit",
                            "physical_hypothesis_updated": False,
                            "evidence": (
                                "algorithms/urdf_learn_wasd_walk/outputs/stand_zero_signal_30s_no_reset/"
                                "experiments/gravity_static_pose_release_v1_limit_audit_failure_20260904T064751Z.json"
                            ),
                            "evidence_sha256": "c0ba691d8863ddf9b0495d503b21f73f9bdfa964137408b454b714ae905d6222",
                            "traceback_sha256": "3c8544875512a3c08b6bc442a1475b9b78e0f4e9c05d407036981a1e03605ce2",
                        },
                        {
                            "run_identity": "20260904T071956.210897Z",
                            "status": "free_root_supported_probe_status_not_promoted",
                            "free_root_duration_s": 3.0,
                            "free_root_fall_count": 0,
                            "free_root_max_reference_tilt_rad": 0.0412149,
                            "free_root_horizontal_drift_m": 0.00062895,
                            "fixed_root_contact_force_body_weight_ratio": 0.0,
                            "criterion_correction": (
                                "contact force alone cannot close fixed-root load balance because the root "
                                "constraint reaction was not measured; fixed-root contact load is diagnostic"
                            ),
                            "evidence": (
                                "algorithms/urdf_learn_wasd_walk/outputs/stand_zero_signal_30s_no_reset/"
                                "experiments/gravity_static_pose_release_v1_20260904T071956Z.json"
                            ),
                            "evidence_sha256": "6bbd26111a61026ae33ecdc7ee7b296e2faefbb7b4500530c320635b98519765",
                        }
                    ],
                    "fixed_root_load_contract": {
                        "contact_force_is_gating": False,
                        "root_constraint_reaction_included": False,
                        "reason": (
                            "the unmeasured kinematic root constraint may carry any residual weight reaction; "
                            "contact force becomes gating only when summed with a measured constraint reaction"
                        ),
                    },
                    "comparison": {
                        "zero_pose_support_margin_m": analyze_pose_geometry({}, urdf_path)["support_margin_m"],
                        "candidate_support_margin_m": static_pose_candidate["support_margin_m"],
                        "ground_alignment_retained": True,
                        "authority_probe_rejected": True,
                    },
                    "result": {
                        "physical_result": "free_root_supported_for_3_seconds",
                        "probe_status": "failed_only_by_invalid_fixed_root_contact_criterion",
                        "pose_promoted": False,
                        "current_asset_status": "not_evaluated_on_rabbit_ear_mesh",
                        "milestone_status_changed": False,
                    },
                },
                {
                    "id": "rabbit_ear_passive_recertification_v1",
                    "status": "active_bounded_test",
                    "scope": "transferred pose seed, exact rabbit-ear mesh, free root, zero action/command",
                    "predecessor": "gravity_static_pose_release_v1",
                    "promotion_requirement": "pass the exact 30-second passive gate and separate viewport proof with the 8,864-triangle asset",
                },
            ],
            "zero_pose_static_authority_audit": {
                "method": "current URDF subtree gravity moments with the floating base held fixed",
                "limitation": "diagnoses joint holding compliance; it is not a floating-base contact inverse-dynamics solution",
                "left_shoulder_lift_gravity_torque_nm": round(
                    gravity_torques["left_shoulder_lift_joint"], 9
                ),
                "right_shoulder_lift_gravity_torque_nm": round(
                    gravity_torques["right_shoulder_lift_joint"], 9
                ),
                "current_linear_pd_shoulder_lift_error_rad": round(
                    abs(gravity_torques["left_shoulder_lift_joint"]) / LOCKED_PD_STIFFNESS, 9
                ),
                "authority_probe_linear_pd_shoulder_lift_error_rad": round(
                    abs(gravity_torques["left_shoulder_lift_joint"])
                    / AUTHORITY_PROBE_UPPER_STIFFNESS,
                    9,
                ),
                "measured_ground_aligned_shoulder_lift_rad_at_0_3_s": {
                    "left": 0.221,
                    "right": -0.242,
                },
            },
            "authority_probe_actuator_groups": passive_actuator_groups(
                movable_names, authority_probe=True
            ),
            "nominal_pose_selection": {
                "selected": "unproven transferred seed from invalidated gravity-static run, realigned to current collision geometry",
                "installed_humanoid_comparison": {
                    "G1_CFG": {"hip_pitch_rad": -0.20, "knee_rad": 0.42, "ankle_pitch_rad": -0.23},
                    "H1_CFG": {"hip_pitch_rad": -0.28, "knee_rad": 0.79, "ankle_pitch_rad": -0.52},
                },
                "selection_reason": (
                    "the canonical pose completed a 3-second free-root release with zero falls and 0.63 mm "
                    "drift, while all zero-pose variants fell near 1.2 seconds"
                ),
            },
        },
        "joints": joint_records,
    }


def write_robot_spec(path: Path = ROBOT_SPEC_PATH) -> dict:
    spec = build_robot_spec()
    path.write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return spec


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", type=Path, help="Write the complete machine-readable robot contract.")
    args = parser.parse_args(argv)
    spec = write_robot_spec(args.write) if args.write else build_robot_spec()
    print(json.dumps(spec, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
