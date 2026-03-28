from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np


def _orthonormalize(rotation: np.ndarray) -> np.ndarray:
    u, _, vh = np.linalg.svd(rotation)
    fixed = u @ vh
    if np.linalg.det(fixed) < 0:
        u[:, -1] *= -1.0
        fixed = u @ vh
    return fixed


def matrix_to_xyz_rpy(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rotation = _orthonormalize(matrix[:3, :3])
    tx, ty, tz = matrix[:3, 3]
    sy = math.sqrt(rotation[0, 0] ** 2 + rotation[1, 0] ** 2)
    singular = sy < 1e-8
    if not singular:
        roll = math.atan2(rotation[2, 1], rotation[2, 2])
        pitch = math.atan2(-rotation[2, 0], sy)
        yaw = math.atan2(rotation[1, 0], rotation[0, 0])
    else:
        roll = math.atan2(-rotation[1, 2], rotation[1, 1])
        pitch = math.atan2(-rotation[2, 0], sy)
        yaw = 0.0
    return np.array([tx, ty, tz], dtype=float), np.array([roll, pitch, yaw], dtype=float)


def rpy_to_matrix(rpy: Sequence[float]) -> np.ndarray:
    roll, pitch, yaw = (float(value) for value in rpy)
    cr = math.cos(roll)
    sr = math.sin(roll)
    cp = math.cos(pitch)
    sp = math.sin(pitch)
    cy = math.cos(yaw)
    sy = math.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=float,
    )


def axis_angle_matrix(axis: Sequence[float], angle_rad: float) -> np.ndarray:
    axis_arr = np.asarray(axis, dtype=float)
    norm = np.linalg.norm(axis_arr)
    if norm < 1e-8 or abs(angle_rad) < 1e-10:
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


def align_x_to_vector(vector: Sequence[float]) -> np.ndarray:
    direction = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(direction)
    if norm < 1e-8:
        return np.eye(3, dtype=float)
    x_axis = direction / norm
    helper = np.array([0.0, 0.0, 1.0], dtype=float)
    if abs(np.dot(x_axis, helper)) > 0.92:
        helper = np.array([0.0, 1.0, 0.0], dtype=float)
    y_axis = np.cross(helper, x_axis)
    y_norm = np.linalg.norm(y_axis)
    if y_norm < 1e-8:
        return np.eye(3, dtype=float)
    y_axis /= y_norm
    z_axis = np.cross(x_axis, y_axis)
    return np.column_stack((x_axis, y_axis, z_axis))


def _keyword_base(name: str) -> str:
    return name.split('/')[-1]


def make_unique_link_names(paths: Sequence[str]) -> Dict[str, str]:
    counts: Dict[str, int] = {}
    result: Dict[str, str] = {}
    for path in paths:
        base = _keyword_base(path)
        count = counts.get(base, 0)
        counts[base] = count + 1
        result[path] = base if count == 0 else f'{base}_{count}'
    return result


def infer_joint_axis(name: str) -> np.ndarray:
    if 'twist' in name:
        return np.array([1.0, 0.0, 0.0], dtype=float)
    if name.startswith('thumb1_'):
        return np.array([0.0, 1.0, 0.0], dtype=float)
    if 'thumb' in name or 'index' in name or 'middle' in name or 'ring' in name or 'pinky' in name:
        return np.array([0.0, 0.0, 1.0], dtype=float)
    if 'shoulder' in name or name.startswith('hand_'):
        return np.array([0.0, 0.0, 1.0], dtype=float)
    return np.array([0.0, 1.0, 0.0], dtype=float)


def infer_joint_limits(name: str) -> tuple[float, float]:
    if 'thumb1_' in name:
        return (-1.2, 1.2)
    if 'thumb' in name:
        return (-1.5, 1.0)
    if 'index' in name or 'middle' in name or 'ring' in name or 'pinky' in name:
        if name.endswith('_base_l') or name.endswith('_base_r'):
            return (-0.45, 0.45)
        return (0.0, 1.6)
    if 'twist' in name:
        return (-1.2, 1.2)
    if 'spine' in name or 'neck' in name or 'head' in name:
        return (-0.7, 0.7)
    if 'foot' in name or 'toes' in name:
        return (-0.6, 0.8)
    return (-1.8, 1.8)


def infer_thickness(name: str, length: float, incoming_length: float) -> float:
    ref = max(length, incoming_length, 0.05)
    if 'thumb' in name or 'index' in name or 'middle' in name or 'ring' in name or 'pinky' in name:
        return float(np.clip(ref * 0.18, 0.006, 0.018))
    if name.startswith('hand_'):
        return float(np.clip(ref * 0.22, 0.012, 0.03))
    if 'forearm' in name or 'arm_' in name:
        return float(np.clip(ref * 0.22, 0.014, 0.04))
    if 'shoulder' in name or 'spine' in name or 'neck' in name or 'head' in name:
        return float(np.clip(ref * 0.25, 0.018, 0.05))
    if 'thigh' in name or 'leg' in name or 'foot' in name or 'toes' in name:
        return float(np.clip(ref * 0.24, 0.016, 0.055))
    return float(np.clip(ref * 0.20, 0.012, 0.035))


def extract_skeleton_records(skel) -> dict:
    joints = [str(item) for item in (skel.GetJointsAttr().Get() or [])]
    rest_transforms = skel.GetRestTransformsAttr().Get() or []
    if not joints or len(joints) != len(rest_transforms):
        raise RuntimeError('Skeleton joints/rest transform count mismatch.')

    link_name_by_path = make_unique_link_names(joints)
    path_to_index = {path: idx for idx, path in enumerate(joints)}
    local_mats = []
    for transform in rest_transforms:
        local_mats.append(np.array([[float(transform[r][c]) for c in range(4)] for r in range(4)], dtype=float).T)

    parents: List[int] = []
    children: List[List[int]] = [[] for _ in joints]
    for path in joints:
        parent_path = '/'.join(path.split('/')[:-1]) if '/' in path else ''
        parent_index = path_to_index.get(parent_path, -1)
        parents.append(parent_index)
    for index, parent_index in enumerate(parents):
        if parent_index >= 0:
            children[parent_index].append(index)

    world_mats = [np.eye(4, dtype=float) for _ in joints]
    for index, local_mat in enumerate(local_mats):
        parent_index = parents[index]
        world_mats[index] = local_mat if parent_index < 0 else world_mats[parent_index] @ local_mat

    records = []
    for index, path in enumerate(joints):
        local_xyz, local_rpy = matrix_to_xyz_rpy(local_mats[index])
        world_xyz, world_rpy = matrix_to_xyz_rpy(world_mats[index])
        incoming_length = 0.0 if parents[index] < 0 else float(np.linalg.norm(local_xyz))
        name = link_name_by_path[path]
        axis = infer_joint_axis(name)
        lower, upper = infer_joint_limits(name)
        records.append(
            {
                'index': index,
                'path': path,
                'name': name,
                'parent_index': parents[index],
                'parent_path': joints[parents[index]] if parents[index] >= 0 else None,
                'parent_name': link_name_by_path[joints[parents[index]]] if parents[index] >= 0 else None,
                'children': [joints[child] for child in children[index]],
                'child_names': [link_name_by_path[joints[child]] for child in children[index]],
                'local_matrix': local_mats[index],
                'world_matrix': world_mats[index],
                'local_xyz': local_xyz,
                'local_rpy': local_rpy,
                'world_xyz': world_xyz,
                'world_rpy': world_rpy,
                'incoming_length': incoming_length,
                'axis': axis,
                'limits': (lower, upper),
            }
        )

    return {
        'records': records,
        'joint_paths': joints,
        'link_name_by_path': link_name_by_path,
        'skeleton_path': str(skel.GetPrim().GetPath()),
    }


def _geom_box(origin_xyz: Sequence[float], origin_rpy: Sequence[float], size_xyz: Sequence[float]) -> dict:
    return {
        'kind': 'box',
        'origin_xyz': [float(v) for v in origin_xyz],
        'origin_rpy': [float(v) for v in origin_rpy],
        'size_xyz': [float(v) for v in size_xyz],
    }


def _geom_sphere(radius: float) -> dict:
    return {
        'kind': 'sphere',
        'origin_xyz': [0.0, 0.0, 0.0],
        'origin_rpy': [0.0, 0.0, 0.0],
        'radius': float(radius),
    }


def _geom_mesh(filename: str) -> dict:
    return {
        'kind': 'mesh',
        'origin_xyz': [0.0, 0.0, 0.0],
        'origin_rpy': [0.0, 0.0, 0.0],
        'filename': str(filename),
    }


def build_link_geometries(records: Sequence[dict]) -> Dict[str, List[dict]]:
    by_name: Dict[str, List[dict]] = {}
    children_by_parent: Dict[int, List[int]] = {record['index']: [] for record in records}
    for record in records:
        if record['parent_index'] >= 0:
            children_by_parent[record['parent_index']].append(record['index'])

    for record in records:
        geoms: List[dict] = []
        child_vectors = [records[child_index]['local_xyz'] for child_index in children_by_parent[record['index']]]
        max_length = max((float(np.linalg.norm(vec)) for vec in child_vectors), default=record['incoming_length'])
        thickness = infer_thickness(record['name'], max_length, record['incoming_length'])
        geoms.append(_geom_sphere(max(thickness * 0.55, 0.005)))
        for child_vec in child_vectors:
            length = float(np.linalg.norm(child_vec))
            if length < 1e-5:
                continue
            rotation = align_x_to_vector(child_vec)
            transform = np.eye(4, dtype=float)
            transform[:3, :3] = rotation
            _, rpy = matrix_to_xyz_rpy(transform)
            geoms.append(_geom_box(np.asarray(child_vec, dtype=float) * 0.5, rpy, [length, thickness, thickness]))
        if len(geoms) == 1 and record['incoming_length'] > 1e-5:
            length = max(record['incoming_length'] * 0.75, thickness)
            geoms.append(_geom_box([length * 0.4, 0.0, 0.0], [0.0, 0.0, 0.0], [length, thickness, thickness]))
        by_name[record['name']] = geoms
    return by_name


def _append_origin(parent, xyz: Sequence[float], rpy: Sequence[float]):
    import xml.etree.ElementTree as ET

    ET.SubElement(
        parent,
        'origin',
        xyz=' '.join(f'{float(v):.6f}' for v in xyz),
        rpy=' '.join(f'{float(v):.6f}' for v in rpy),
    )


def _inertial_from_geoms(geoms: Sequence[dict]) -> tuple[float, tuple[float, float, float]]:
    mass = 0.0
    ixx = iyy = izz = 0.0
    for geom in geoms:
        if geom['kind'] == 'sphere':
            radius = geom['radius']
            volume = 4.0 / 3.0 * math.pi * radius**3
            part_mass = max(volume * 250.0, 0.01)
            inertia = 0.4 * part_mass * radius**2
            mass += part_mass
            ixx += inertia
            iyy += inertia
            izz += inertia
        else:
            lx, ly, lz = geom['size_xyz']
            volume = lx * ly * lz
            part_mass = max(volume * 250.0, 0.01)
            mass += part_mass
            ixx += (part_mass / 12.0) * (ly**2 + lz**2)
            iyy += (part_mass / 12.0) * (lx**2 + lz**2)
            izz += (part_mass / 12.0) * (lx**2 + ly**2)
    mass = max(mass, 0.02)
    return mass, (max(ixx, 1e-5), max(iyy, 1e-5), max(izz, 1e-5))


def generate_urdf_text(
    robot_name: str,
    records: Sequence[dict],
    geoms_by_name: Dict[str, List[dict]] | None = None,
    inertial_geoms_by_name: Dict[str, List[dict]] | None = None,
) -> str:
    import xml.etree.ElementTree as ET

    root_records = [record for record in records if record['parent_index'] < 0]
    if len(root_records) != 1:
        raise RuntimeError(f'Expected exactly one root record, found {len(root_records)}.')
    root_record = root_records[0]
    base_link_name = 'base_link'

    robot = ET.Element('robot', name=robot_name)
    primitive_geoms_by_name = build_link_geometries(records)
    render_geoms_by_name = geoms_by_name or primitive_geoms_by_name
    inertial_geoms_by_name = inertial_geoms_by_name or primitive_geoms_by_name

    base_link = ET.SubElement(robot, 'link', name=base_link_name)
    inertial = ET.SubElement(base_link, 'inertial')
    _append_origin(inertial, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    ET.SubElement(inertial, 'mass', value='0.001000')
    ET.SubElement(
        inertial,
        'inertia',
        ixx='0.000001',
        ixy='0.0',
        ixz='0.0',
        iyy='0.000001',
        iyz='0.0',
        izz='0.000001',
    )

    for record in records:
        link = ET.SubElement(robot, 'link', name=record['name'])
        inertial_geoms = inertial_geoms_by_name[record['name']]
        mass, inertia = _inertial_from_geoms(inertial_geoms)
        inertial = ET.SubElement(link, 'inertial')
        _append_origin(inertial, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
        ET.SubElement(inertial, 'mass', value=f'{mass:.6f}')
        ET.SubElement(
            inertial,
            'inertia',
            ixx=f'{inertia[0]:.6f}',
            ixy='0.0',
            ixz='0.0',
            iyy=f'{inertia[1]:.6f}',
            iyz='0.0',
            izz=f'{inertia[2]:.6f}',
        )

        for tag in ('visual', 'collision'):
            for geom in render_geoms_by_name[record['name']]:
                elem = ET.SubElement(link, tag)
                _append_origin(elem, geom['origin_xyz'], geom['origin_rpy'])
                geometry = ET.SubElement(elem, 'geometry')
                if geom['kind'] == 'sphere':
                    ET.SubElement(geometry, 'sphere', radius=f"{geom['radius']:.6f}")
                elif geom['kind'] == 'mesh':
                    ET.SubElement(geometry, 'mesh', filename=geom['filename'])
                else:
                    ET.SubElement(
                        geometry,
                        'box',
                        size=' '.join(f'{float(v):.6f}' for v in geom['size_xyz']),
                    )

    fixed_root_joint = ET.SubElement(robot, 'joint', name=f'{root_record["name"]}_base_fixed', type='fixed')
    ET.SubElement(fixed_root_joint, 'parent', link=base_link_name)
    ET.SubElement(fixed_root_joint, 'child', link=root_record['name'])
    _append_origin(fixed_root_joint, root_record['local_xyz'], root_record['local_rpy'])

    for record in records:
        if record['parent_index'] < 0:
            continue
        joint = ET.SubElement(robot, 'joint', name=record['name'], type='revolute')
        ET.SubElement(joint, 'parent', link=record['parent_name'])
        ET.SubElement(joint, 'child', link=record['name'])
        _append_origin(joint, record['local_xyz'], record['local_rpy'])
        axis = record['axis']
        ET.SubElement(joint, 'axis', xyz=' '.join(f'{float(v):.1f}' for v in axis))
        lower, upper = record['limits']
        ET.SubElement(
            joint,
            'limit',
            lower=f'{lower:.6f}',
            upper=f'{upper:.6f}',
            effort='50.0',
            velocity='4.0',
        )
        ET.SubElement(joint, 'dynamics', damping='1.0', friction='0.05')

    xml_bytes = ET.tostring(robot, encoding='utf-8')
    return "<?xml version=\'1.0\'?>\n" + xml_bytes.decode('utf-8')


def build_demo_pose(records: Sequence[dict]) -> Dict[str, float]:
    names = {record['name'] for record in records}
    pose = {
        'spine_02_x': 0.10,
        'neck_x': -0.08,
        'arm_stretch_r': -0.40,
        'forearm_stretch_r': -0.85,
        'hand_r': 0.22,
        'thumb1_r': 0.35,
        'thumb2_r': 0.45,
        'thumb3_r': 0.35,
        'index1_base_r': 0.10,
        'index1_r': 0.65,
        'index2_r': 0.55,
        'index3_r': 0.35,
        'middle1_base_r': 0.06,
        'middle1_r': 0.70,
        'middle2_r': 0.58,
        'middle3_r': 0.38,
        'ring1_base_r': -0.04,
        'ring1_r': 0.66,
        'ring2_r': 0.55,
        'ring3_r': 0.35,
        'pinky1_base_r': -0.12,
        'pinky1_r': 0.60,
        'pinky2_r': 0.48,
        'pinky3_r': 0.30,
        'arm_stretch_l': 0.18,
        'forearm_stretch_l': -0.35,
        'hand_l': -0.10,
    }
    return {name: angle for name, angle in pose.items() if name in names}


def apply_pose_to_local_matrices(records: Sequence[dict], pose_by_name: Dict[str, float]) -> List[np.ndarray]:
    posed = [record['local_matrix'].copy() for record in records]
    for record in records:
        angle = pose_by_name.get(record['name'])
        if angle is None:
            continue
        rotation_delta = axis_angle_matrix(record['axis'], angle)
        updated = posed[record['index']].copy()
        updated[:3, :3] = _orthonormalize(updated[:3, :3] @ rotation_delta)
        posed[record['index']] = updated
    return posed


def world_matrices_from_local(records: Sequence[dict], local_matrices: Sequence[np.ndarray]) -> List[np.ndarray]:
    world = [np.eye(4, dtype=float) for _ in records]
    for record in records:
        parent_index = record['parent_index']
        world[record['index']] = (
            local_matrices[record['index']]
            if parent_index < 0
            else world[parent_index] @ local_matrices[record['index']]
        )
    return world


def records_to_jsonable(records: Sequence[dict]) -> List[dict]:
    payload = []
    for record in records:
        payload.append(
            {
                'index': record['index'],
                'path': record['path'],
                'name': record['name'],
                'parent_index': record['parent_index'],
                'parent_path': record['parent_path'],
                'parent_name': record['parent_name'],
                'children': list(record['children']),
                'child_names': list(record['child_names']),
                'local_xyz': [float(v) for v in record['local_xyz']],
                'local_rpy': [float(v) for v in record['local_rpy']],
                'world_xyz': [float(v) for v in record['world_xyz']],
                'world_rpy': [float(v) for v in record['world_rpy']],
                'local_matrix': np.asarray(record['local_matrix'], dtype=float).tolist(),
                'world_matrix': np.asarray(record['world_matrix'], dtype=float).tolist(),
                'incoming_length': float(record['incoming_length']),
                'axis': [float(v) for v in record['axis']],
                'limits': [float(record['limits'][0]), float(record['limits'][1])],
            }
        )
    return payload


def write_records_json(path: Path, skeleton_path: str, usd_path: Path, records: Sequence[dict]) -> None:
    payload = {
        'usd_path': str(usd_path),
        'skeleton_path': skeleton_path,
        'joint_count': len(records),
        'records': records_to_jsonable(records),
        'demo_pose_radians': build_demo_pose(records),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding='utf-8')


def root_height_offset(records: Sequence[dict], clearance: float = 0.02) -> float:
    min_z = min(float(record['world_xyz'][2]) for record in records)
    return -min_z + clearance


def root_height_offset_from_world_matrices(world_matrices: Sequence[np.ndarray], clearance: float = 0.02) -> float:
    min_z = min(float(matrix[2, 3]) for matrix in world_matrices)
    return -min_z + clearance


def rotation_error_radians(a: np.ndarray, b: np.ndarray) -> float:
    rotation = _orthonormalize(a[:3, :3]).T @ _orthonormalize(b[:3, :3])
    trace = float(np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0))
    return math.acos(trace)


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding='utf-8')
