from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class RepairConfig:
    target_face_ratio: float = 0.25
    max_faces: int = 1200
    max_hole_edges: int = 500
    component_keep_area_ratio: float = 0.05
    planar_deviation_ratio: float = 0.05
    force_fill_max_edges: int = 10
    merge_tolerance: float = 1e-6
    fill_projection_offset_ratio: float = 1e-3
    smoothing_iterations: int = 0
    smoothing_lambda: float = 0.35
    max_surface_snap_distance_ratio: float = 0.04


@dataclass
class OriginalMeshContext:
    vertices: np.ndarray
    faces: np.ndarray
    triangles: np.ndarray
    kdtree: object
    bbox_diag: float


@dataclass
class RepairStats:
    original_face_count: int
    final_face_count: int
    holes_found: int
    holes_filled: int
    watertight: bool
    euler_number: int | None
    warnings: list[str]
    method: str


def _import_trimesh():
    import trimesh

    return trimesh


def _import_ckdtree():
    from scipy.spatial import cKDTree

    return cKDTree


def _clean_faces(vertices: np.ndarray, faces: np.ndarray, area_eps: float = 1e-12) -> tuple[np.ndarray, np.ndarray]:
    if len(vertices) == 0 or len(faces) == 0:
        return vertices[:0], faces[:0]

    faces = np.asarray(faces, dtype=np.int64)
    valid = np.logical_and.reduce(
        (
            faces[:, 0] != faces[:, 1],
            faces[:, 1] != faces[:, 2],
            faces[:, 0] != faces[:, 2],
        )
    )
    faces = faces[valid]
    if len(faces) == 0:
        return vertices[:0], faces

    tri = vertices[faces]
    area = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1) * 0.5
    faces = faces[area > area_eps]
    if len(faces) == 0:
        return vertices[:0], faces

    used = np.unique(faces.reshape(-1))
    remap = np.full(len(vertices), -1, dtype=np.int64)
    remap[used] = np.arange(len(used), dtype=np.int64)
    return vertices[used], remap[faces]


def triangles_to_mesh_arrays(triangles: np.ndarray, tolerance: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    if len(triangles) == 0:
        return np.zeros((0, 3), dtype=float), np.zeros((0, 3), dtype=np.int64)

    points = np.asarray(triangles, dtype=float).reshape(-1, 3)
    quantized = np.round(points / max(tolerance, 1e-9)).astype(np.int64)
    _, unique_indices, inverse = np.unique(quantized, axis=0, return_index=True, return_inverse=True)
    vertices = points[unique_indices]
    faces = inverse.reshape(-1, 3)
    return _clean_faces(vertices, faces)


def mesh_arrays_from_vertices_faces(vertices: np.ndarray, faces: np.ndarray, tolerance: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    if len(vertices) == 0 or len(faces) == 0:
        return np.zeros((0, 3), dtype=float), np.zeros((0, 3), dtype=np.int64)

    points = np.asarray(vertices, dtype=float)
    if tolerance <= 0.0:
        return _clean_faces(points, np.asarray(faces, dtype=np.int64))
    quantized = np.round(points / max(tolerance, 1e-9)).astype(np.int64)
    _, unique_indices, inverse = np.unique(quantized, axis=0, return_index=True, return_inverse=True)
    dedup_vertices = points[unique_indices]
    remapped_faces = inverse[np.asarray(faces, dtype=np.int64)]
    return _clean_faces(dedup_vertices, remapped_faces)


def make_trimesh(vertices: np.ndarray, faces: np.ndarray):
    trimesh = _import_trimesh()

    vertices, faces = _clean_faces(np.asarray(vertices, dtype=float), np.asarray(faces, dtype=np.int64))
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    if len(mesh.faces) == 0:
        return mesh
    try:
        mesh.process(validate=True)
    except Exception:
        pass
    mesh.remove_duplicate_faces()
    mesh.remove_degenerate_faces()
    mesh.remove_unreferenced_vertices()
    return mesh


def solid_angle_winding_number(points: np.ndarray, triangles: np.ndarray, chunk_size: int = 2048) -> np.ndarray:
    if len(points) == 0 or len(triangles) == 0:
        return np.zeros(len(points), dtype=float)

    points = np.asarray(points, dtype=float)
    triangles = np.asarray(triangles, dtype=float)
    winding = np.zeros(len(points), dtype=float)
    tri_a = triangles[:, 0]
    tri_b = triangles[:, 1]
    tri_c = triangles[:, 2]

    for start in range(0, len(points), chunk_size):
        batch = points[start : start + chunk_size]
        a = tri_a[None, :, :] - batch[:, None, :]
        b = tri_b[None, :, :] - batch[:, None, :]
        c = tri_c[None, :, :] - batch[:, None, :]

        la = np.linalg.norm(a, axis=2)
        lb = np.linalg.norm(b, axis=2)
        lc = np.linalg.norm(c, axis=2)
        cross_bc = np.cross(b, c)
        numerator = np.einsum('bfi,bfi->bf', a, cross_bc)
        denominator = (
            la * lb * lc
            + np.einsum('bfi,bfi->bf', a, b) * lc
            + np.einsum('bfi,bfi->bf', b, c) * la
            + np.einsum('bfi,bfi->bf', c, a) * lb
        )
        winding[start : start + len(batch)] = np.sum(2.0 * np.arctan2(numerator, denominator + 1e-12), axis=1) / (
            4.0 * math.pi
        )
    return winding


def prepare_original_mesh_context(vertices: np.ndarray, faces: np.ndarray) -> OriginalMeshContext:
    cKDTree = _import_ckdtree()

    mesh = make_trimesh(vertices, faces)
    bbox_diag = float(np.linalg.norm(mesh.bounds[1] - mesh.bounds[0])) if len(mesh.vertices) else 1.0
    return OriginalMeshContext(
        vertices=np.asarray(mesh.vertices, dtype=float),
        faces=np.asarray(mesh.faces, dtype=np.int64),
        triangles=np.asarray(mesh.triangles, dtype=float),
        kdtree=cKDTree(np.asarray(mesh.vertices, dtype=float)) if len(mesh.vertices) else None,
        bbox_diag=max(bbox_diag, 1e-6),
    )


def prepare_original_mesh_context_from_triangles(triangles: np.ndarray, tolerance: float = 1e-6) -> OriginalMeshContext:
    vertices, faces = triangles_to_mesh_arrays(triangles, tolerance=tolerance)
    return prepare_original_mesh_context(vertices, faces)


def is_inside(points: np.ndarray, context: OriginalMeshContext) -> np.ndarray:
    return solid_angle_winding_number(points, context.triangles) >= 0.5


def closest_on_surface(points: np.ndarray, context: OriginalMeshContext) -> tuple[np.ndarray, np.ndarray]:
    if context.kdtree is None or len(points) == 0:
        return np.zeros((0, 3), dtype=float), np.zeros(0, dtype=float)
    distances, indices = context.kdtree.query(np.asarray(points, dtype=float))
    return context.vertices[np.asarray(indices, dtype=np.int64)], np.asarray(distances, dtype=float)


def _remove_small_components(mesh, keep_ratio: float):
    if len(mesh.faces) == 0:
        return mesh

    parts = mesh.split(only_watertight=False)
    if len(parts) <= 1:
        return mesh

    largest_area = max(float(part.area) for part in parts)
    keep = [part for part in parts if float(part.area) >= largest_area * keep_ratio]
    if not keep:
        keep = [max(parts, key=lambda part: float(part.area))]
    trimesh = _import_trimesh()
    return trimesh.util.concatenate(keep)


def extract_boundary_loops(mesh) -> list[list[int]]:
    if len(mesh.faces) == 0:
        return []

    edges = np.sort(np.asarray(mesh.edges_sorted, dtype=np.int64), axis=1)
    unique_edges, counts = np.unique(edges, axis=0, return_counts=True)
    boundary_edges = unique_edges[counts == 1]
    if len(boundary_edges) == 0:
        return []

    adjacency: dict[int, list[int]] = {}
    for a, b in boundary_edges.tolist():
        adjacency.setdefault(int(a), []).append(int(b))
        adjacency.setdefault(int(b), []).append(int(a))

    visited: set[tuple[int, int]] = set()
    loops: list[list[int]] = []
    for a_raw, b_raw in boundary_edges.tolist():
        a = int(a_raw)
        b = int(b_raw)
        key = tuple(sorted((a, b)))
        if key in visited:
            continue
        loop = [a, b]
        visited.add(key)
        prev = a
        current = b
        while True:
            neighbors = adjacency.get(current, [])
            next_vertex = None
            for candidate in neighbors:
                edge_key = tuple(sorted((current, candidate)))
                if candidate == prev or edge_key in visited:
                    continue
                next_vertex = candidate
                break
            if next_vertex is None:
                break
            if next_vertex == loop[0]:
                visited.add(tuple(sorted((current, next_vertex))))
                loops.append(loop.copy())
                break
            visited.add(tuple(sorted((current, next_vertex))))
            loop.append(next_vertex)
            prev, current = current, next_vertex
    return loops


def _newell_normal(loop_vertices: np.ndarray) -> np.ndarray:
    normal = np.zeros(3, dtype=float)
    rolled = np.roll(loop_vertices, -1, axis=0)
    normal[0] = np.sum((loop_vertices[:, 1] - rolled[:, 1]) * (loop_vertices[:, 2] + rolled[:, 2]))
    normal[1] = np.sum((loop_vertices[:, 2] - rolled[:, 2]) * (loop_vertices[:, 0] + rolled[:, 0]))
    normal[2] = np.sum((loop_vertices[:, 0] - rolled[:, 0]) * (loop_vertices[:, 1] + rolled[:, 1]))
    norm = float(np.linalg.norm(normal))
    if norm < 1e-9:
        return np.array([0.0, 0.0, 1.0], dtype=float)
    return normal / norm


def _inward_direction(loop_vertices: np.ndarray, context: OriginalMeshContext, offset_ratio: float) -> np.ndarray:
    center = loop_vertices.mean(axis=0)
    normal = _newell_normal(loop_vertices)
    offset = context.bbox_diag * float(offset_ratio)
    plus = solid_angle_winding_number((center + normal * offset)[None, :], context.triangles)[0]
    minus = solid_angle_winding_number((center - normal * offset)[None, :], context.triangles)[0]
    return normal if plus >= minus else -normal


def _fit_loop_plane(loop_vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    center = loop_vertices.mean(axis=0)
    centered = loop_vertices - center
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    axis_u = vh[0]
    axis_v = vh[1]
    normal = vh[2]
    projected = np.column_stack((centered @ axis_u, centered @ axis_v))
    diag = float(np.linalg.norm(loop_vertices.max(axis=0) - loop_vertices.min(axis=0)))
    deviation = float(np.max(np.abs(centered @ normal))) if len(loop_vertices) else 0.0
    return center, axis_u, axis_v, diag, deviation


def _polygon_area_2d(points_2d: np.ndarray) -> float:
    x = points_2d[:, 0]
    y = points_2d[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - y * np.roll(x, -1)))


def _point_in_triangle_2d(point: np.ndarray, triangle: np.ndarray, eps: float = 1e-9) -> bool:
    a, b, c = triangle
    v0 = c - a
    v1 = b - a
    v2 = point - a
    dot00 = float(np.dot(v0, v0))
    dot01 = float(np.dot(v0, v1))
    dot02 = float(np.dot(v0, v2))
    dot11 = float(np.dot(v1, v1))
    dot12 = float(np.dot(v1, v2))
    denom = dot00 * dot11 - dot01 * dot01
    if abs(denom) < eps:
        return False
    inv = 1.0 / denom
    u = (dot11 * dot02 - dot01 * dot12) * inv
    v = (dot00 * dot12 - dot01 * dot02) * inv
    return u >= -eps and v >= -eps and (u + v) <= 1.0 + eps


def _ear_clip_polygon(loop_indices: Sequence[int], loop_points_2d: np.ndarray) -> list[tuple[int, int, int]] | None:
    if len(loop_indices) < 3:
        return None
    if len(loop_indices) == 3:
        return [(int(loop_indices[0]), int(loop_indices[1]), int(loop_indices[2]))]

    winding_positive = _polygon_area_2d(loop_points_2d) > 0.0
    remaining = list(range(len(loop_indices)))
    triangles: list[tuple[int, int, int]] = []

    while len(remaining) > 3:
        best_ear = None
        best_score = -1.0
        for offset, current_local in enumerate(remaining):
            prev_local = remaining[offset - 1]
            next_local = remaining[(offset + 1) % len(remaining)]
            a2 = loop_points_2d[prev_local]
            b2 = loop_points_2d[current_local]
            c2 = loop_points_2d[next_local]
            cross = float(np.cross(b2 - a2, c2 - b2))
            is_convex = cross > 1e-10 if winding_positive else cross < -1e-10
            if not is_convex:
                continue

            tri2 = np.stack((a2, b2, c2), axis=0)
            contains_other = False
            for candidate_local in remaining:
                if candidate_local in (prev_local, current_local, next_local):
                    continue
                if _point_in_triangle_2d(loop_points_2d[candidate_local], tri2):
                    contains_other = True
                    break
            if contains_other:
                continue

            edge_lengths = (
                np.linalg.norm(b2 - a2),
                np.linalg.norm(c2 - b2),
                np.linalg.norm(a2 - c2),
            )
            if min(edge_lengths) < 1e-9:
                continue
            area = abs(float(np.cross(b2 - a2, c2 - a2))) * 0.5
            quality = area / max(sum(edge_lengths), 1e-9)
            if quality > best_score:
                best_score = quality
                best_ear = (offset, prev_local, current_local, next_local)

        if best_ear is None:
            return None
        offset, prev_local, current_local, next_local = best_ear
        triangles.append(
            (
                int(loop_indices[prev_local]),
                int(loop_indices[current_local]),
                int(loop_indices[next_local]),
            )
        )
        del remaining[offset]

    triangles.append(
        (
            int(loop_indices[remaining[0]]),
            int(loop_indices[remaining[1]]),
            int(loop_indices[remaining[2]]),
        )
    )
    return triangles


def _orient_face(face: tuple[int, int, int], vertices: np.ndarray, inward_direction: np.ndarray) -> tuple[int, int, int]:
    a, b, c = vertices[list(face)]
    normal = np.cross(b - a, c - a)
    if float(np.dot(normal, inward_direction)) < 0.0:
        return (face[0], face[2], face[1])
    return face


def _interior_fill_point(
    loop_vertices: np.ndarray,
    vertices: np.ndarray,
    context: OriginalMeshContext,
    inward_direction: np.ndarray,
) -> np.ndarray:
    center = loop_vertices.mean(axis=0)
    loop_extent = float(np.linalg.norm(loop_vertices.max(axis=0) - loop_vertices.min(axis=0)))
    loop_radius = float(np.max(np.linalg.norm(loop_vertices - center, axis=1))) if len(loop_vertices) else 0.0
    step = max(loop_extent * 0.16, loop_radius * 0.2, context.bbox_diag * 0.005, 2e-4)
    min_separation = max(loop_extent * 0.05, context.bbox_diag * 0.001, 5e-5)

    for factor in (1.0, 1.5, 2.0, 3.0, 4.0):
        candidate = center + inward_direction * step * factor
        if not bool(is_inside(candidate[None, :], context)[0]):
            continue
        if len(vertices) > 0:
            distances = np.linalg.norm(vertices - candidate[None, :], axis=1)
            if float(np.min(distances)) < min_separation:
                continue
        return candidate

    if bool(is_inside(center[None, :], context)[0]):
        return center

    closest_points, _ = closest_on_surface(center[None, :], context)
    if len(closest_points):
        fallback = closest_points[0] + inward_direction * step
        if bool(is_inside(fallback[None, :], context)[0]):
            return fallback
        return closest_points[0]
    return center


def _fan_fill(loop: Sequence[int], vertices: np.ndarray, context: OriginalMeshContext, inward_direction: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int, int]], list[int]]:
    loop_vertices = vertices[np.asarray(loop, dtype=np.int64)]
    centroid = _interior_fill_point(loop_vertices, vertices, context, inward_direction)
    new_vertices = np.vstack((vertices, centroid))
    center_index = len(new_vertices) - 1
    faces: list[tuple[int, int, int]] = []
    for idx, current in enumerate(loop):
        nxt = loop[(idx + 1) % len(loop)]
        faces.append(_orient_face((int(current), int(nxt), center_index), new_vertices, inward_direction))
    return new_vertices, faces, [center_index]


def _force_fill_small_loops(mesh, context: OriginalMeshContext, max_edges: int):
    loops = extract_boundary_loops(mesh)
    if not loops:
        return mesh, 0

    vertices = np.asarray(mesh.vertices, dtype=float)
    faces_list = [tuple(int(v) for v in face) for face in np.asarray(mesh.faces, dtype=np.int64).tolist()]
    filled = 0
    for loop in loops:
        if len(loop) >= int(max_edges):
            continue
        inward = _inward_direction(vertices[np.asarray(loop, dtype=np.int64)], context, 1e-3)
        vertices, new_faces, _ = _fan_fill(loop, vertices, context, inward)
        faces_list.extend(new_faces)
        filled += 1
    if filled == 0:
        return mesh, 0
    return make_trimesh(vertices, np.asarray(faces_list, dtype=np.int64)), filled


def fill_boundary_loop(
    vertices: np.ndarray,
    loop: Sequence[int],
    context: OriginalMeshContext,
    config: RepairConfig,
) -> tuple[np.ndarray, list[tuple[int, int, int]], list[int]] | None:
    if len(loop) < 3 or len(loop) > int(config.max_hole_edges):
        return None

    loop_vertices = vertices[np.asarray(loop, dtype=np.int64)]
    inward = _inward_direction(loop_vertices, context, float(config.fill_projection_offset_ratio))
    # Collision meshes care more about watertightness than perfect cap tessellation.
    # Using only boundary vertices can create diagonals that collide with existing
    # surface edges and turn the result non-manifold, so prefer a projected centroid fan.
    return _fan_fill(loop, vertices, context, inward)


def _fix_normals(mesh, context: OriginalMeshContext):
    trimesh = _import_trimesh()

    trimesh.repair.fix_normals(mesh, multibody=True)
    if len(mesh.faces) == 0:
        return mesh
    centroid = mesh.triangles_center[0]
    normal = mesh.face_normals[0]
    probe = centroid + normal * max(context.bbox_diag * 1e-4, 1e-5)
    if bool(is_inside(probe[None, :], context)[0]):
        mesh.invert()
    return mesh


def _cluster_vertices(vertices: np.ndarray, faces: np.ndarray, cell_size: float) -> tuple[np.ndarray, np.ndarray]:
    if len(vertices) == 0 or len(faces) == 0:
        return vertices[:0], faces[:0]

    quantized = np.round(vertices / max(cell_size, 1e-9)).astype(np.int64)
    unique_keys, inverse = np.unique(quantized, axis=0, return_inverse=True)
    remapped = inverse[np.asarray(faces, dtype=np.int64)]
    valid = np.logical_and.reduce(
        (
            remapped[:, 0] != remapped[:, 1],
            remapped[:, 1] != remapped[:, 2],
            remapped[:, 0] != remapped[:, 2],
        )
    )
    remapped = remapped[valid]
    if len(remapped) == 0:
        return vertices[:0], remapped

    unique_faces = np.unique(np.sort(remapped, axis=1), axis=0)
    vertex_count = len(unique_keys)
    clustered = np.zeros((vertex_count, 3), dtype=float)
    counts = np.zeros(vertex_count, dtype=np.int64)
    for old_index, new_index in enumerate(inverse):
        clustered[new_index] += vertices[old_index]
        counts[new_index] += 1
    clustered /= np.maximum(counts[:, None], 1)
    return _clean_faces(clustered, unique_faces)


def _simplify_mesh(mesh, target_faces: int):
    if len(mesh.faces) <= target_faces:
        return mesh, 'no_simplification'

    try:
        import igl  # type: ignore

        success, vertices, faces, _, _ = igl.decimate(
            np.asarray(mesh.vertices, dtype=np.float64),
            np.asarray(mesh.faces, dtype=np.int64),
            int(target_faces),
        )
        if success and len(faces) > 0:
            return make_trimesh(vertices, faces), 'igl_decimate'
    except Exception:
        pass

    if hasattr(mesh, 'simplify_quadric_decimation'):
        try:
            return mesh.simplify_quadric_decimation(int(target_faces)), 'trimesh_quadric'
        except Exception:
            pass

    diag = float(np.linalg.norm(mesh.bounds[1] - mesh.bounds[0])) if len(mesh.vertices) else 1.0
    best_mesh = mesh
    best_delta = abs(len(mesh.faces) - target_faces)
    for divisor in (160, 120, 96, 80, 64, 52, 44, 36, 30, 24, 20, 16, 12, 10, 8, 6):
        cell = max(diag / divisor, 1e-5)
        vertices, faces = _cluster_vertices(np.asarray(mesh.vertices), np.asarray(mesh.faces), cell)
        if len(faces) == 0:
            continue
        candidate = make_trimesh(vertices, faces)
        delta = abs(len(candidate.faces) - target_faces)
        if delta < best_delta or (delta == best_delta and len(candidate.faces) <= target_faces):
            best_mesh = candidate
            best_delta = delta
        if len(candidate.faces) <= target_faces:
            return candidate, 'vertex_cluster'
    return best_mesh, 'vertex_cluster'


def _laplacian_smooth_fill_vertices(mesh, fill_vertex_indices: Sequence[int], context: OriginalMeshContext, config: RepairConfig):
    if int(config.smoothing_iterations) <= 0 or not fill_vertex_indices:
        return mesh

    vertices = np.asarray(mesh.vertices, dtype=float).copy()
    neighbors: dict[int, set[int]] = {}
    for face in np.asarray(mesh.faces, dtype=np.int64):
        for a, b in ((0, 1), (1, 2), (2, 0)):
            neighbors.setdefault(int(face[a]), set()).add(int(face[b]))
            neighbors.setdefault(int(face[b]), set()).add(int(face[a]))

    allowed_snap = context.bbox_diag * float(config.max_surface_snap_distance_ratio)
    fill_set = {int(index) for index in fill_vertex_indices if 0 <= int(index) < len(vertices)}
    for _ in range(int(config.smoothing_iterations)):
        updated = vertices.copy()
        for index in fill_set:
            neighbor_ids = sorted(neighbors.get(index, ()))
            if not neighbor_ids:
                continue
            target = vertices[neighbor_ids].mean(axis=0)
            blended = vertices[index] * (1.0 - float(config.smoothing_lambda)) + target * float(config.smoothing_lambda)
            closest_points, distances = closest_on_surface(blended[None, :], context)
            if len(closest_points) and float(distances[0]) <= allowed_snap:
                blended = closest_points[0]
            updated[index] = blended
        vertices = updated
    return make_trimesh(vertices, np.asarray(mesh.faces, dtype=np.int64))


def _repair_mesh_arrays(fragment_vertices: np.ndarray, fragment_faces: np.ndarray, context: OriginalMeshContext, config: RepairConfig) -> tuple[np.ndarray, np.ndarray, RepairStats] | None:
    warnings: list[str] = []
    vertices, faces = _clean_faces(np.asarray(fragment_vertices, dtype=float), np.asarray(fragment_faces, dtype=np.int64))
    mesh = make_trimesh(vertices, faces)
    if len(mesh.faces) == 0:
        return None

    mesh = _remove_small_components(mesh, keep_ratio=float(config.component_keep_area_ratio))
    mesh = make_trimesh(np.asarray(mesh.vertices), np.asarray(mesh.faces))
    original_face_count = int(len(mesh.faces))

    loops = extract_boundary_loops(mesh)
    holes_found = len(loops)
    holes_filled = 0
    fill_vertex_indices: list[int] = []

    vertices = np.asarray(mesh.vertices, dtype=float)
    faces_list = [tuple(int(v) for v in face) for face in np.asarray(mesh.faces, dtype=np.int64).tolist()]
    for loop in sorted(loops, key=len, reverse=True):
        filled = fill_boundary_loop(vertices, loop, context, config)
        if filled is None:
            warnings.append(f'skipped_loop_edges={len(loop)}')
            continue
        vertices, new_faces, new_vertices = filled
        faces_list.extend(new_faces)
        fill_vertex_indices.extend(new_vertices)
        holes_filled += 1

    mesh = make_trimesh(vertices, np.asarray(faces_list, dtype=np.int64))
    remaining_loops = extract_boundary_loops(mesh)
    for loop in remaining_loops:
        if len(loop) >= int(config.force_fill_max_edges):
            continue
        vertices = np.asarray(mesh.vertices, dtype=float)
        faces_list = [tuple(int(v) for v in face) for face in np.asarray(mesh.faces, dtype=np.int64).tolist()]
        inward = _inward_direction(vertices[np.asarray(loop, dtype=np.int64)], context, float(config.fill_projection_offset_ratio))
        vertices, new_faces, new_vertices = _fan_fill(loop, vertices, context, inward)
        faces_list.extend(new_faces)
        fill_vertex_indices.extend(new_vertices)
        mesh = make_trimesh(vertices, np.asarray(faces_list, dtype=np.int64))

    try:
        trimesh = _import_trimesh()

        trimesh.repair.fill_holes(mesh)
    except Exception:
        pass
    mesh = _fix_normals(mesh, context)
    pre_simplify_mesh = mesh.copy()
    target_faces = max(4, min(int(math.ceil(original_face_count * float(config.target_face_ratio))), int(config.max_faces)))
    mesh, simplify_method = _simplify_mesh(mesh, target_faces)
    mesh, forced_loop_count = _force_fill_small_loops(mesh, context, int(config.force_fill_max_edges))
    if forced_loop_count > 0:
        warnings.append(f'post_simplify_force_fill={forced_loop_count}')
    try:
        trimesh = _import_trimesh()

        trimesh.repair.fill_holes(mesh)
    except Exception:
        pass
    mesh = _fix_normals(mesh, context)
    mesh = _laplacian_smooth_fill_vertices(mesh, fill_vertex_indices, context, config)
    mesh = _fix_normals(mesh, context)

    watertight = bool(mesh.is_watertight)
    if not watertight and bool(pre_simplify_mesh.is_watertight):
        mesh = pre_simplify_mesh
        mesh = _fix_normals(mesh, context)
        watertight = bool(mesh.is_watertight)
        simplify_method = f'{simplify_method}_reverted'
    euler_number = int(mesh.euler_number) if len(mesh.faces) else None
    if not watertight:
        warnings.append('mesh_not_watertight')

    return (
        np.asarray(mesh.vertices, dtype=float),
        np.asarray(mesh.faces, dtype=np.int64),
        RepairStats(
            original_face_count=original_face_count,
            final_face_count=int(len(mesh.faces)),
            holes_found=holes_found,
            holes_filled=holes_filled,
            watertight=watertight,
            euler_number=euler_number,
            warnings=warnings,
            method=simplify_method,
        ),
    )


def repair_fragment_mesh(fragment_triangles: np.ndarray, context: OriginalMeshContext, config: RepairConfig) -> tuple[np.ndarray, np.ndarray, RepairStats] | None:
    vertices, faces = triangles_to_mesh_arrays(fragment_triangles, tolerance=config.merge_tolerance)
    return _repair_mesh_arrays(vertices, faces, context, config)


def repair_fragment_arrays(
    fragment_vertices: np.ndarray,
    fragment_faces: np.ndarray,
    context: OriginalMeshContext,
    config: RepairConfig,
) -> tuple[np.ndarray, np.ndarray, RepairStats] | None:
    vertices, faces = mesh_arrays_from_vertices_faces(fragment_vertices, fragment_faces, tolerance=0.0)
    return _repair_mesh_arrays(vertices, faces, context, config)


def _load_stl_mesh(path: Path):
    trimesh = _import_trimesh()

    loaded = trimesh.load_mesh(path, force='mesh')
    if isinstance(loaded, trimesh.Scene):
        loaded = trimesh.util.concatenate(tuple(loaded.geometry.values()))
    return make_trimesh(np.asarray(loaded.vertices), np.asarray(loaded.faces))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Repair and simplify open mesh fragments against an original closed mesh.')
    parser.add_argument('--original', type=Path, required=True, help='Original watertight STL mesh.')
    parser.add_argument('--parts-dir', type=Path, required=True, help='Directory of open fragment STL meshes.')
    parser.add_argument('--output-dir', type=Path, required=True, help='Directory for repaired STL outputs.')
    parser.add_argument('--target-ratio', type=float, default=0.25, help='Target face ratio after simplification.')
    parser.add_argument('--max-faces', type=int, default=1200, help='Absolute face cap per repaired output.')
    parser.add_argument('--max-hole-edges', type=int, default=500, help='Maximum boundary loop size to attempt filling.')
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    original = _load_stl_mesh(args.original)
    context = prepare_original_mesh_context(np.asarray(original.vertices), np.asarray(original.faces))
    config = RepairConfig(
        target_face_ratio=float(args.target_ratio),
        max_faces=int(args.max_faces),
        max_hole_edges=int(args.max_hole_edges),
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[RepairStats] = []
    for path in sorted(args.parts_dir.glob('*.stl')):
        part = _load_stl_mesh(path)
        result = repair_fragment_mesh(np.asarray(part.triangles, dtype=float), context, config)
        if result is None:
            print(f'[REPAIR] skip empty {path.name}', flush=True)
            continue
        vertices, faces, stats = result
        mesh = make_trimesh(vertices, faces)
        mesh.export(args.output_dir / path.name)
        rows.append(stats)
        print(
            f'[REPAIR] {path.name}: faces {stats.original_face_count}->{stats.final_face_count} '
            f'holes {stats.holes_filled}/{stats.holes_found} watertight={stats.watertight}',
            flush=True,
        )

    print('[REPAIR] summary', flush=True)
    for stats in rows:
        print(
            f'faces={stats.original_face_count}->{stats.final_face_count} '
            f'holes={stats.holes_filled}/{stats.holes_found} watertight={stats.watertight} '
            f'euler={stats.euler_number} method={stats.method} warnings={";".join(stats.warnings) or "-"}',
            flush=True,
        )


if __name__ == '__main__':
    main()
