from __future__ import annotations

import os
import struct
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np

from config import DEFAULT_MESH_BUILD_CONFIG, LowpolyMeshConfig, MeshBuildConfig, resolve_lowpoly_link_config
from skeleton_common import build_link_geometries, rpy_to_matrix


def _find_skel_root_prim(skeleton_prim):
    from pxr import UsdSkel

    current = skeleton_prim
    while current and current.IsValid():
        if current.IsA(UsdSkel.Root):
            return current
        current = current.GetParent()
    raise RuntimeError(f'Could not find a UsdSkel.Root ancestor for {skeleton_prim.GetPath()}.')


def _triangulated_faces(face_vertex_counts: Sequence[int], face_vertex_indices: Sequence[int]) -> Iterable[tuple[int, int, int]]:
    cursor = 0
    for count in face_vertex_counts:
        count = int(count)
        polygon = [int(index) for index in face_vertex_indices[cursor : cursor + count]]
        cursor += count
        if count < 3:
            continue
        base = polygon[0]
        for offset in range(1, count - 1):
            yield base, polygon[offset], polygon[offset + 1]


def _transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    if points.size == 0:
        return np.zeros((0, 3), dtype=float)
    hom = np.concatenate((points, np.ones((points.shape[0], 1), dtype=float)), axis=1)
    return (hom @ transform.T)[:, :3]


def _transform_triangles(triangles: np.ndarray, transform: np.ndarray) -> np.ndarray:
    if triangles.size == 0:
        return np.zeros((0, 3, 3), dtype=float)
    reshaped = triangles.reshape(-1, 3)
    transformed = _transform_points(reshaped, transform)
    return transformed.reshape(-1, 3, 3)


def _unique_points(points: np.ndarray, tolerance: float = 1e-5) -> np.ndarray:
    if len(points) <= 1:
        return points
    quantized = np.round(points / tolerance).astype(np.int64)
    _, indices = np.unique(quantized, axis=0, return_index=True)
    return points[np.sort(indices)]


def _select_extreme_points(points: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return points
    directions = np.array(
        [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
            [1.0, 1.0, 0.0],
            [1.0, -1.0, 0.0],
            [1.0, 0.0, 1.0],
            [1.0, 0.0, -1.0],
            [0.0, 1.0, 1.0],
            [0.0, 1.0, -1.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, -1.0],
        ],
        dtype=float,
    )
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    keep_indices: set[int] = set()
    projections = points @ directions.T
    for column in range(projections.shape[1]):
        keep_indices.add(int(np.argmax(projections[:, column])))
        keep_indices.add(int(np.argmin(projections[:, column])))
    return points[sorted(keep_indices)]


def _downsample_points(points: np.ndarray, max_points: int) -> np.ndarray:
    if len(points) <= max_points:
        return points
    bounds_min = points.min(axis=0)
    diag = float(np.linalg.norm(points.max(axis=0) - bounds_min))
    if diag < 1e-8:
        return points[:1]

    preserved = _select_extreme_points(points)
    candidate = preserved
    for divisor in (42.0, 32.0, 24.0, 18.0, 14.0, 10.0, 8.0, 6.0):
        voxel = max(diag / divisor, 1e-4)
        quantized = np.floor((points - bounds_min) / voxel).astype(np.int64)
        _, indices = np.unique(quantized, axis=0, return_index=True)
        merged = np.vstack((preserved, points[np.sort(indices)]))
        candidate = _unique_points(merged)
        if len(candidate) <= max_points:
            return candidate

    selection = np.linspace(0, len(candidate) - 1, max_points, dtype=int)
    return candidate[selection]


def _box_mesh(center: np.ndarray, size: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int, int]]]:
    hx, hy, hz = (float(value) * 0.5 for value in size)
    cx, cy, cz = (float(value) for value in center)
    vertices = np.array(
        [
            [cx - hx, cy - hy, cz - hz],
            [cx + hx, cy - hy, cz - hz],
            [cx + hx, cy + hy, cz - hz],
            [cx - hx, cy + hy, cz - hz],
            [cx - hx, cy - hy, cz + hz],
            [cx + hx, cy - hy, cz + hz],
            [cx + hx, cy + hy, cz + hz],
            [cx - hx, cy + hy, cz + hz],
        ],
        dtype=float,
    )
    faces = [
        (0, 2, 1),
        (0, 3, 2),
        (4, 5, 6),
        (4, 6, 7),
        (0, 1, 5),
        (0, 5, 4),
        (1, 2, 6),
        (1, 6, 5),
        (2, 3, 7),
        (2, 7, 6),
        (3, 0, 4),
        (3, 4, 7),
    ]
    return vertices, faces


def _box_from_points(points: np.ndarray, min_thickness: float) -> tuple[np.ndarray, list[tuple[int, int, int]], dict]:
    bounds_min = points.min(axis=0)
    bounds_max = points.max(axis=0)
    center = (bounds_min + bounds_max) * 0.5
    size = np.maximum(bounds_max - bounds_min, min_thickness)
    vertices, faces = _box_mesh(center, size)
    return vertices, faces, {'center': center.tolist(), 'size': size.tolist()}


def _oriented_box_from_points(points: np.ndarray, min_thickness: float) -> tuple[np.ndarray, list[tuple[int, int, int]], dict]:
    center = points.mean(axis=0)
    centered = points - center
    if len(points) < 3 or np.allclose(centered, 0.0):
        vertices, faces, details = _box_from_points(points, min_thickness)
        return vertices, faces, {'principal_axes': np.eye(3).tolist(), **details}

    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    basis = vt.T
    if np.linalg.det(basis) < 0.0:
        basis[:, -1] *= -1.0

    local_points = centered @ basis
    bounds_min = local_points.min(axis=0)
    bounds_max = local_points.max(axis=0)
    local_center = (bounds_min + bounds_max) * 0.5
    size = np.maximum(bounds_max - bounds_min, min_thickness)

    local_vertices, faces = _box_mesh(local_center, size)
    vertices = local_vertices @ basis.T + center
    return vertices, faces, {
        'center': center.tolist(),
        'size': size.tolist(),
        'principal_axes': basis.tolist(),
        'point_rank': int(np.count_nonzero(singular_values > 1e-8)),
    }


def _box_from_geoms(geoms: Sequence[dict], min_thickness: float) -> tuple[np.ndarray, list[tuple[int, int, int]], dict]:
    mins = []
    maxs = []
    for geom in geoms:
        origin = np.asarray(geom['origin_xyz'], dtype=float)
        if geom['kind'] == 'sphere':
            radius = float(geom['radius'])
            mins.append(origin - radius)
            maxs.append(origin + radius)
            continue
        if geom['kind'] != 'box':
            continue
        rotation = np.abs(rpy_to_matrix(geom['origin_rpy']))
        half_extents = np.asarray(geom['size_xyz'], dtype=float) * 0.5
        world_half_extents = rotation @ half_extents
        mins.append(origin - world_half_extents)
        maxs.append(origin + world_half_extents)

    if not mins:
        size = np.array([min_thickness, min_thickness, min_thickness], dtype=float)
        vertices, faces = _box_mesh(np.zeros(3, dtype=float), size)
        return vertices, faces, {'center': [0.0, 0.0, 0.0], 'size': size.tolist()}

    bounds_min = np.min(np.vstack(mins), axis=0)
    bounds_max = np.max(np.vstack(maxs), axis=0)
    center = (bounds_min + bounds_max) * 0.5
    size = np.maximum(bounds_max - bounds_min, min_thickness)
    vertices, faces = _box_mesh(center, size)
    return vertices, faces, {'center': center.tolist(), 'size': size.tolist()}


def _convex_hull_mesh(points: np.ndarray, max_hull_faces: int) -> tuple[np.ndarray, list[tuple[int, int, int]], dict] | None:
    try:
        from scipy.spatial import ConvexHull, QhullError
    except ImportError:
        return None

    try:
        hull = ConvexHull(points, qhull_options='QJ')
    except QhullError:
        return None

    if len(hull.simplices) > max_hull_faces:
        return None

    used = np.unique(hull.simplices.reshape(-1))
    vertex_map = {int(old_index): new_index for new_index, old_index in enumerate(used.tolist())}
    vertices = points[used]
    centroid = vertices.mean(axis=0)
    faces: list[tuple[int, int, int]] = []
    for simplex in hull.simplices:
        tri = [vertex_map[int(simplex[0])], vertex_map[int(simplex[1])], vertex_map[int(simplex[2])]]
        a, b, c = vertices[tri]
        normal = np.cross(b - a, c - a)
        face_center = (a + b + c) / 3.0
        if float(np.dot(normal, face_center - centroid)) < 0.0:
            tri[1], tri[2] = tri[2], tri[1]
        faces.append((tri[0], tri[1], tri[2]))

    return vertices, faces, {'vertex_count': len(vertices), 'face_count': len(faces), 'volume': float(hull.volume)}


def _cluster_mesh_vertices(
    vertices: np.ndarray,
    faces: np.ndarray,
    cell_size: float,
) -> tuple[np.ndarray, np.ndarray]:
    if len(vertices) == 0 or len(faces) == 0:
        return vertices, faces

    quantized = np.round(vertices / max(cell_size, 1e-6)).astype(np.int64)
    unique_keys, inverse = np.unique(quantized, axis=0, return_inverse=True)
    remapped_faces = inverse[np.asarray(faces, dtype=np.int64)]
    valid = np.logical_and.reduce(
        (
            remapped_faces[:, 0] != remapped_faces[:, 1],
            remapped_faces[:, 1] != remapped_faces[:, 2],
            remapped_faces[:, 0] != remapped_faces[:, 2],
        )
    )
    remapped_faces = remapped_faces[valid]
    if len(remapped_faces) == 0:
        return vertices[:0], remapped_faces

    unique_faces = np.unique(np.sort(remapped_faces, axis=1), axis=0)
    clustered_vertices = np.zeros((len(unique_keys), 3), dtype=float)
    counts = np.zeros(len(unique_keys), dtype=np.int64)
    for old_index, new_index in enumerate(inverse):
        clustered_vertices[new_index] += vertices[old_index]
        counts[new_index] += 1
    clustered_vertices /= np.maximum(counts[:, None], 1)
    return clustered_vertices, unique_faces


def _clean_mesh(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    if len(vertices) == 0 or len(faces) == 0:
        return vertices[:0], faces[:0], {'watertight': False, 'volume': 0.0}

    try:
        import trimesh

        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
        mesh.remove_unreferenced_vertices()
        return (
            np.asarray(mesh.vertices, dtype=float),
            np.asarray(mesh.faces, dtype=np.int64),
            {'watertight': bool(mesh.is_watertight), 'volume': float(abs(mesh.volume))},
        )
    except Exception:
        return vertices, faces, {'watertight': False, 'volume': 0.0}


def _surface_sample_points(
    triangles: np.ndarray,
    max_points: int = 45000,
    tolerance: float = 5e-4,
) -> np.ndarray:
    if len(triangles) == 0:
        return np.zeros((0, 3), dtype=float)

    a = triangles[:, 0]
    b = triangles[:, 1]
    c = triangles[:, 2]
    centroids = (a + b + c) / 3.0
    mid_ab = (a + b) * 0.5
    mid_bc = (b + c) * 0.5
    mid_ca = (c + a) * 0.5
    points = np.vstack((triangles.reshape(-1, 3), centroids, mid_ab, mid_bc, mid_ca))
    points = _unique_points(points, tolerance=tolerance)
    if len(points) > max_points:
        points = _downsample_points(points, max_points)
    return points


def _reconstruction_pitch(points: np.ndarray, min_thickness: float, target_cells: int, max_cells: int = 30) -> float:
    extent = points.max(axis=0) - points.min(axis=0)
    longest = float(max(extent.max(), min_thickness * 6.0))
    pitch = longest / max(target_cells, 1)
    min_pitch = max(min_thickness * 0.5, 0.0025)
    max_pitch = max(min_thickness * 4.0, 0.03)
    pitch = float(np.clip(pitch, min_pitch, max_pitch))
    if longest / pitch > max_cells:
        pitch = longest / max_cells
    return pitch


def _marching_surface_mesh(
    points: np.ndarray,
    pitch: float,
    link_config: LowpolyMeshConfig,
) -> tuple[np.ndarray, np.ndarray, dict] | None:
    from scipy import ndimage
    from skimage import measure

    if len(points) < 4:
        return None

    padding = max(int(link_config.padding_cells), 1)
    bounds_min = points.min(axis=0) - pitch * padding
    bounds_max = points.max(axis=0) + pitch * padding
    shape = np.ceil((bounds_max - bounds_min) / pitch).astype(np.int32) + 1
    if np.any(shape < 4):
        shape = np.maximum(shape, 4)
    if np.any(shape > int(link_config.max_grid_cells)):
        return None

    occupancy = np.zeros(shape.tolist(), dtype=bool)
    indices = np.round((points - bounds_min) / pitch).astype(np.int32)
    indices = np.clip(indices, 0, shape - 1)
    occupancy[indices[:, 0], indices[:, 1], indices[:, 2]] = True

    structure = ndimage.generate_binary_structure(3, 2)
    if link_config.dilation_iterations > 0:
        occupancy = ndimage.binary_dilation(occupancy, structure=structure, iterations=int(link_config.dilation_iterations))
    if link_config.closing_iterations > 0:
        occupancy = ndimage.binary_closing(
            occupancy,
            structure=np.ones((3, 3, 3), dtype=bool),
            iterations=int(link_config.closing_iterations),
        )
    occupancy = ndimage.binary_fill_holes(occupancy)

    labels, label_count = ndimage.label(occupancy)
    if label_count > 1:
        counts = np.bincount(labels.ravel())
        keep = counts >= max(6, int(counts.max() * 0.04))
        keep[0] = False
        occupancy = keep[labels]

    field = ndimage.gaussian_filter(occupancy.astype(np.float32), sigma=float(link_config.smooth_sigma))
    if float(field.max()) <= 0.05:
        return None

    vertices, faces, normals, _ = measure.marching_cubes(field, level=min(0.45, float(field.max()) * 0.55), spacing=(pitch, pitch, pitch))
    vertices += bounds_min

    cleaned_vertices, cleaned_faces, mesh_details = _clean_mesh(vertices, np.asarray(faces, dtype=np.int64))
    if len(cleaned_faces) == 0:
        return None

    return cleaned_vertices, cleaned_faces, {
        'pitch': float(pitch),
        'grid_shape': shape.tolist(),
        **mesh_details,
    }


def _fit_vertices_to_reference_bounds(
    vertices: np.ndarray,
    reference_points: np.ndarray,
    link_config: LowpolyMeshConfig,
) -> tuple[np.ndarray, dict]:
    if len(vertices) == 0 or len(reference_points) == 0:
        return vertices, {
            'fit_applied': False,
            'source_extent': [0.0, 0.0, 0.0],
            'mesh_extent_before_fit': [0.0, 0.0, 0.0],
            'mesh_extent_after_fit': [0.0, 0.0, 0.0],
            'fit_scale_xyz': [1.0, 1.0, 1.0],
        }

    ref_min = reference_points.min(axis=0)
    ref_max = reference_points.max(axis=0)
    ref_extent = np.maximum(ref_max - ref_min, 1e-8)
    ref_center = (ref_min + ref_max) * 0.5

    mesh_min = vertices.min(axis=0)
    mesh_max = vertices.max(axis=0)
    mesh_extent = np.maximum(mesh_max - mesh_min, 1e-8)

    fit_margin = np.maximum(ref_extent * float(link_config.fit_margin_ratio), float(link_config.fit_margin_min))
    max_extent = ref_extent * np.asarray(link_config.max_extent_ratio_xyz, dtype=float)
    target_extent = np.maximum(ref_extent, np.minimum(ref_extent + fit_margin * 2.0, max_extent))
    scale = np.minimum(1.0, target_extent / mesh_extent)

    fitted = (vertices - ref_center) * scale + ref_center
    fitted_min = fitted.min(axis=0)
    fitted_max = fitted.max(axis=0)
    target_min = ref_center - target_extent * 0.5
    target_max = ref_center + target_extent * 0.5

    shift = np.zeros(3, dtype=float)
    for axis in range(3):
        if fitted_min[axis] < target_min[axis]:
            shift[axis] += target_min[axis] - fitted_min[axis]
        if fitted_max[axis] > target_max[axis]:
            shift[axis] += target_max[axis] - fitted_max[axis]
    fitted = fitted + shift
    fitted_extent = fitted.max(axis=0) - fitted.min(axis=0)

    return fitted, {
        'fit_applied': bool(np.any(scale < 0.9999) or np.linalg.norm(shift) > 1e-9),
        'source_extent': ref_extent.tolist(),
        'mesh_extent_before_fit': mesh_extent.tolist(),
        'mesh_extent_after_fit': fitted_extent.tolist(),
        'fit_scale_xyz': scale.tolist(),
        'fit_shift_xyz': shift.tolist(),
    }


def _lowpoly_surface_mesh(
    triangles: np.ndarray,
    min_thickness: float,
    link_config: LowpolyMeshConfig,
) -> tuple[np.ndarray, list[tuple[int, int, int]], dict] | None:
    points = _surface_sample_points(
        triangles,
        max_points=int(link_config.max_sample_points),
        tolerance=float(link_config.sample_tolerance),
    )
    if len(points) < 4:
        return None

    best_candidate: tuple[np.ndarray, list[tuple[int, int, int]], dict] | None = None
    for target_cells in link_config.target_cells:
        pitch = _reconstruction_pitch(
            points,
            min_thickness=min_thickness,
            target_cells=int(target_cells),
            max_cells=int(link_config.max_grid_cells),
        )
        reconstructed = _marching_surface_mesh(points, pitch=pitch, link_config=link_config)
        if reconstructed is None:
            continue
        vertices, faces, details = reconstructed
        for cluster_scale in link_config.cluster_scales:
            clustered_vertices, clustered_faces = _cluster_mesh_vertices(vertices, faces, max(pitch * cluster_scale, 1e-4))
            clustered_vertices, clustered_faces, mesh_details = _clean_mesh(clustered_vertices, clustered_faces)
            if len(clustered_faces) == 0:
                continue
            clustered_vertices, fit_details = _fit_vertices_to_reference_bounds(clustered_vertices, points, link_config)
            candidate = (
                clustered_vertices,
                [tuple(int(v) for v in face) for face in clustered_faces.tolist()],
                {
                    'method': 'skinned_lowpoly_surface',
                    'vertex_count': int(len(clustered_vertices)),
                    'face_count': int(len(clustered_faces)),
                    'target_cells': int(target_cells),
                    'cluster_scale': float(cluster_scale),
                    **details,
                    **mesh_details,
                    **fit_details,
                },
            )
            if best_candidate is None or candidate[2]['face_count'] < best_candidate[2]['face_count']:
                best_candidate = candidate
            if len(clustered_faces) <= link_config.max_faces:
                return candidate
    return best_candidate


def _write_binary_stl(path: Path, vertices: np.ndarray, faces: Sequence[tuple[int, int, int]], solid_name: str) -> None:
    header = solid_name.encode('ascii', errors='ignore')[:80].ljust(80, b'\0')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('wb') as handle:
        handle.write(header)
        handle.write(struct.pack('<I', len(faces)))
        for face in faces:
            a = vertices[face[0]]
            b = vertices[face[1]]
            c = vertices[face[2]]
            normal = np.cross(b - a, c - a)
            norm = float(np.linalg.norm(normal))
            if norm > 1e-12:
                normal = normal / norm
            else:
                normal = np.zeros(3, dtype=np.float32)
            packed = struct.pack(
                '<12fH',
                float(normal[0]),
                float(normal[1]),
                float(normal[2]),
                float(a[0]),
                float(a[1]),
                float(a[2]),
                float(b[0]),
                float(b[1]),
                float(b[2]),
                float(c[0]),
                float(c[1]),
                float(c[2]),
                0,
            )
            handle.write(packed)


def _seed_surface_clouds(stage, skel, records: Sequence[dict]) -> tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, dict]]:
    from pxr import Usd, UsdGeom, UsdSkel

    records_by_path = {record['path']: record for record in records}
    seed_points: Dict[str, List[np.ndarray]] = {record['name']: [] for record in records}
    seed_triangles: Dict[str, List[np.ndarray]] = {record['name']: [] for record in records}
    seed_info: Dict[str, dict] = {
        record['name']: {'seed_point_count': 0, 'seed_triangle_count': 0, 'source_meshes': []} for record in records
    }

    skel_root_prim = _find_skel_root_prim(skel.GetPrim())
    cache = UsdSkel.Cache()
    cache.Populate(UsdSkel.Root(skel_root_prim), Usd.PrimDefaultPredicate)
    skel_query = cache.GetSkelQuery(skel)
    skeleton_joint_order = [str(token) for token in skel_query.GetJointOrder()]
    skinning_transforms = skel_query.ComputeSkinningTransforms(Usd.TimeCode.Default())

    for prim in Usd.PrimRange(skel_root_prim):
        mesh = UsdGeom.Mesh(prim)
        if not mesh or not mesh.GetPrim().IsValid():
            continue
        query = cache.GetSkinningQuery(prim)
        if not query or not query.GetPrim().IsValid() or not query.HasJointInfluences():
            continue

        points = mesh.GetPointsAttr().Get()
        if not points:
            continue
        skinned_points = points
        if not query.ComputeSkinnedPoints(skinning_transforms, skinned_points, Usd.TimeCode.Default()):
            continue
        skinned_points_np = np.asarray([[float(v[0]), float(v[1]), float(v[2])] for v in skinned_points], dtype=float)

        joint_order = query.GetJointOrder()
        ordered_joints = [str(token) for token in joint_order] if joint_order else skeleton_joint_order
        varying_indices, varying_weights = query.ComputeVaryingJointInfluences(
            len(skinned_points),
            Usd.TimeCode.Default(),
        )
        influences_per_point = int(query.GetNumInfluencesPerComponent())
        if influences_per_point <= 0:
            continue

        index_matrix = np.asarray(varying_indices, dtype=np.int32).reshape(-1, influences_per_point)
        weight_matrix = np.asarray(varying_weights, dtype=np.float64).reshape(-1, influences_per_point)
        face_counts = mesh.GetFaceVertexCountsAttr().Get() or []
        face_indices = mesh.GetFaceVertexIndicesAttr().Get() or []

        touched_links: set[str] = set()
        for tri in _triangulated_faces(face_counts, face_indices):
            joint_scores: Dict[int, float] = {}
            for vertex_index in tri:
                for influence_slot in range(influences_per_point):
                    weight = float(weight_matrix[vertex_index, influence_slot])
                    if weight <= 1e-6:
                        continue
                    joint_index = int(index_matrix[vertex_index, influence_slot])
                    joint_scores[joint_index] = joint_scores.get(joint_index, 0.0) + weight
            if not joint_scores:
                continue
            best_joint_index = max(joint_scores.items(), key=lambda item: item[1])[0]
            if best_joint_index < 0 or best_joint_index >= len(ordered_joints):
                continue
            record = records_by_path.get(ordered_joints[best_joint_index])
            if record is None:
                continue
            link_name = record['name']
            triangle_points = skinned_points_np[list(tri)]
            seed_points[link_name].append(triangle_points)
            seed_triangles[link_name].append(triangle_points)
            seed_info[link_name]['seed_triangle_count'] += 1
            touched_links.add(link_name)

        mesh_path = str(prim.GetPath())
        for link_name in touched_links:
            seed_info[link_name]['source_meshes'].append(mesh_path)

    compact_points: Dict[str, np.ndarray] = {}
    compact_triangles: Dict[str, np.ndarray] = {}
    for record in records:
        link_name = record['name']
        if seed_points[link_name]:
            stacked = np.vstack(seed_points[link_name])
            compact_points[link_name] = stacked
            seed_info[link_name]['seed_point_count'] = int(stacked.shape[0])
        else:
            compact_points[link_name] = np.zeros((0, 3), dtype=float)
        if seed_triangles[link_name]:
            compact_triangles[link_name] = np.stack(seed_triangles[link_name], axis=0)
        else:
            compact_triangles[link_name] = np.zeros((0, 3, 3), dtype=float)

    return compact_points, compact_triangles, seed_info


def _build_link_mesh(
    link_name: str,
    local_points: np.ndarray,
    local_triangles: np.ndarray,
    fallback_geoms: Sequence[dict],
    max_hull_faces: int,
    target_hull_points: int,
    min_thickness: float,
    strategy: str,
    build_config: MeshBuildConfig,
) -> tuple[np.ndarray, list[tuple[int, int, int]], dict]:
    if len(local_points) >= 4:
        unique_points = _unique_points(local_points)
        if len(unique_points) >= 4:
            if strategy == 'lowpoly_surface':
                surface = _lowpoly_surface_mesh(
                    local_triangles,
                    min_thickness=min_thickness,
                    link_config=resolve_lowpoly_link_config(build_config, link_name),
                )
                if surface is not None:
                    return surface
            if strategy == 'convex_hull':
                for limit in (max(target_hull_points * 2, target_hull_points), target_hull_points, 40, 24):
                    sampled = _downsample_points(unique_points, limit)
                    if len(sampled) < 4:
                        continue
                    hull = _convex_hull_mesh(sampled, max_hull_faces)
                    if hull is not None:
                        vertices, faces, details = hull
                        return vertices, faces, {'method': 'skinned_convex_hull', **details}
                vertices, faces, details = _oriented_box_from_points(unique_points, min_thickness)
                return vertices, faces, {'method': 'skinned_oriented_box_fallback', **details}

            vertices, faces, details = _oriented_box_from_points(unique_points, min_thickness)
            return vertices, faces, {'method': 'skinned_oriented_box', **details}

    vertices, faces, details = _box_from_geoms(fallback_geoms, min_thickness)
    return vertices, faces, {'method': 'primitive_box_fallback', **details}


def build_mesh_collision_assets(
    stage,
    skel,
    records: Sequence[dict],
    urdf_dir: Path,
    mesh_dir: Path,
    strategy: str = 'obb',
    max_hull_faces: int = 128,
    target_hull_points: int = 96,
    build_config: MeshBuildConfig | None = None,
) -> dict:
    build_config = build_config or DEFAULT_MESH_BUILD_CONFIG
    mesh_dir.mkdir(parents=True, exist_ok=True)
    primitive_geoms_by_name = build_link_geometries(records)
    seed_points_by_name, seed_triangles_by_name, seed_info_by_name = _seed_surface_clouds(stage, skel, records)

    geoms_by_name: Dict[str, List[dict]] = {}
    summary: Dict[str, dict] = {}

    for record in records:
        link_name = record['name']
        inverse_world = np.linalg.inv(np.asarray(record['world_matrix'], dtype=float))
        local_points = _transform_points(seed_points_by_name[link_name], inverse_world)
        local_triangles = _transform_triangles(seed_triangles_by_name[link_name], inverse_world)
        fallback_geoms = primitive_geoms_by_name[link_name]
        min_thickness = float(build_config.min_thickness)
        vertices, faces, details = _build_link_mesh(
            link_name,
            local_points,
            local_triangles,
            fallback_geoms,
            max_hull_faces=max_hull_faces,
            target_hull_points=target_hull_points,
            min_thickness=min_thickness,
            strategy=strategy,
            build_config=build_config,
        )

        mesh_path = mesh_dir / f'{link_name}.stl'
        _write_binary_stl(mesh_path, vertices, faces, link_name)
        relative_filename = os.path.relpath(mesh_path, start=urdf_dir).replace(os.sep, '/')
        geoms_by_name[link_name] = [
            {
                'kind': 'mesh',
                'origin_xyz': [0.0, 0.0, 0.0],
                'origin_rpy': [0.0, 0.0, 0.0],
                'filename': relative_filename,
            }
        ]
        summary[link_name] = {
            'stl_path': str(mesh_path),
            'seed_point_count': int(seed_info_by_name[link_name]['seed_point_count']),
            'seed_triangle_count': int(seed_info_by_name[link_name]['seed_triangle_count']),
            'source_meshes': seed_info_by_name[link_name]['source_meshes'],
            'mesh_vertex_count': int(vertices.shape[0]),
            'mesh_triangle_count': int(len(faces)),
            'bounds_min': vertices.min(axis=0).tolist(),
            'bounds_max': vertices.max(axis=0).tolist(),
            'resolved_lowpoly_config': resolve_lowpoly_link_config(build_config, link_name).__dict__,
            **details,
        }

    return {'geoms_by_name': geoms_by_name, 'summary': summary}
