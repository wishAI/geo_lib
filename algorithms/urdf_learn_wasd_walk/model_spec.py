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


def _rpy_matrix(rpy: Vector3) -> Matrix3:
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rotate_x: Matrix3 = ((1.0, 0.0, 0.0), (0.0, cr, -sr), (0.0, sr, cr))
    rotate_y: Matrix3 = ((cp, 0.0, sp), (0.0, 1.0, 0.0), (-sp, 0.0, cp))
    rotate_z: Matrix3 = ((cy, -sy, 0.0), (sy, cy, 0.0), (0.0, 0.0, 1.0))
    return _matrix_multiply(_matrix_multiply(rotate_z, rotate_y), rotate_x)


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


def _joint_world_transforms(root: ET.Element) -> tuple[dict[str, tuple[Matrix3, Vector3]], dict[str, ET.Element]]:
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
                link_transforms[child] = _compose(link_transforms[parent], _origin_transform(joint.find("origin")))
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


def collision_world_bounds(urdf_path: Path = URDF_PATH) -> dict[str, dict[str, list[float]]]:
    """Return zero-pose world bounds for every link with collision geometry."""

    root = ET.parse(urdf_path).getroot()
    transforms, _ = _joint_world_transforms(root)
    result: dict[str, dict[str, list[float]]] = {}
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
            result[link_name] = {
                "minimum": [min(point[axis] for point in points) for axis in range(3)],
                "maximum": [max(point[axis] for point in points) for axis in range(3)],
            }
    return result


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

    joint_records = []
    for joint in movable_joints:
        name = joint.get("name")
        parent_name = joint.find("parent").get("link")  # type: ignore[union-attr]
        joint_transform = _compose(transforms[parent_name], _origin_transform(joint.find("origin")))
        local_axis = _vector(joint.find("axis").get("xyz"))  # type: ignore[union-attr]
        world_axis = _matrix_vector(joint_transform[0], local_axis)
        norm = math.sqrt(sum(component * component for component in world_axis))
        world_axis = tuple(component / norm for component in world_axis)  # type: ignore[assignment]
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
                "dominant_world_axis": f"{'+' if world_axis[dominant_index] >= 0 else '-'}{'XYZ'[dominant_index]}",
                "limits_rad": [float(limit.get("lower")), float(limit.get("upper"))],  # type: ignore[union-attr]
                "effort_limit": float(limit.get("effort")),  # type: ignore[union-attr]
                "velocity_limit": float(limit.get("velocity")),  # type: ignore[union-attr]
                "nominal_position_rad": 0.0,
            }
        )

    masses = []
    invalid_inertias = []
    for link in links:
        inertial = link.find("inertial")
        if inertial is None:
            continue
        mass = float(inertial.find("mass").get("value"))  # type: ignore[union-attr]
        inertia = {key: float(inertial.find("inertia").get(key, "0")) for key in ("ixx", "ixy", "ixz", "iyy", "iyz", "izz")}  # type: ignore[union-attr]
        masses.append(mass)
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

    fixed_root = next(joint for joint in joints if joint.find("parent").get("link") == "base_link")  # type: ignore[union-attr]
    fixed_origin = fixed_root.find("origin")
    bounds = collision_world_bounds(urdf_path) if not missing_meshes else {}
    return {
        "version": 1,
        "lineage": "clean_restart_2026_08_22",
        "source": {
            "urdf_path": str(urdf_path.relative_to(ALGORITHM_ROOT.parent.parent)),
            "urdf_sha256": urdf_sha256,
            "expected_urdf_sha256": EXPECTED_URDF_SHA256,
            "derived_urdf_used": False,
            "mesh_reference_count": len(mesh_paths),
            "unique_mesh_count": len(unique_mesh_paths),
            "missing_meshes": missing_meshes,
            "mesh_tree_sha256": mesh_tree_digest.hexdigest(),
        },
        "structure": {
            "link_count": len(links),
            "joint_count": len(joints),
            "movable_joint_count": len(movable_joints),
            "total_mass_kg": round(sum(masses), 9),
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
        "nominal_pose": {
            "base_position_m": [0.0, 0.0, 0.002],
            "base_orientation_wxyz": [1.0, 0.0, 0.0, 0.0],
            "joint_positions_rad": {name: 0.0 for name in movable_names},
            "zero_pose_collision_bounds": bounds,
        },
        "action_joints": list(ACTION_JOINTS),
        "locked_joints": locked_names,
        "pd": {
            "groups": {
                name: {**group, "joints": list(group["joints"])}
                for name, group in PD_GROUPS.items()
            },
            "locked": {"stiffness": LOCKED_PD_STIFFNESS, "damping": LOCKED_PD_DAMPING},
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
