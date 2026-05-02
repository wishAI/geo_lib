from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from asset_paths import default_usd_path, resolve_asset_paths
from config import DEFAULT_MESH_BUILD_CONFIG, resolve_lowpoly_link_config
from mesh_collision_builder import (
    _aligned_points,
    _cluster_mesh_vertices,
    _fit_vertices_to_reference_bounds,
    _is_arm_or_finger_link,
    _link_root_axis_local,
    _rotation_basis_from_x_axis,
    _unaligned_points,
    _write_binary_stl,
)
from skeleton_common import load_records_json


def _load_trimesh(path: Path):
    import trimesh

    loaded = trimesh.load_mesh(path, force='mesh')
    if isinstance(loaded, trimesh.Scene):
        loaded = trimesh.util.concatenate(tuple(loaded.geometry.values()))
    return trimesh.Trimesh(vertices=np.asarray(loaded.vertices), faces=np.asarray(loaded.faces), process=True)


def _clean_mesh(vertices: np.ndarray, faces: np.ndarray):
    import trimesh

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
    mesh.remove_unreferenced_vertices()
    return mesh


def _pitch_for_points(points: np.ndarray, target_cells: int, min_pitch: float, max_grid_cells: int) -> float:
    extent = points.max(axis=0) - points.min(axis=0)
    longest = float(max(extent.max(), 0.024))
    pitch = max(longest / max(int(target_cells), 1), float(min_pitch))
    if longest / pitch > int(max_grid_cells):
        pitch = longest / int(max_grid_cells)
    return float(max(pitch, 1e-5))


def _candidate_from_mesh(mesh, reference_points: np.ndarray, link_cfg, target_cells: int, cluster_scale: float):
    source_points = np.asarray(mesh.vertices, dtype=float)
    pitch = _pitch_for_points(
        source_points,
        target_cells=int(target_cells),
        min_pitch=float(link_cfg.min_pitch),
        max_grid_cells=int(link_cfg.max_grid_cells),
    )
    try:
        voxel = mesh.voxelized(pitch=pitch).fill()
        remeshed = voxel.marching_cubes
    except Exception:
        return None

    cleaned = _clean_mesh(np.asarray(remeshed.vertices, dtype=float), np.asarray(remeshed.faces, dtype=np.int64))
    if len(cleaned.faces) == 0:
        return None

    if cluster_scale > 0.0:
        vertices, faces = _cluster_mesh_vertices(
            np.asarray(cleaned.vertices, dtype=float),
            np.asarray(cleaned.faces, dtype=np.int64),
            max(pitch * float(cluster_scale), 1e-5),
        )
        cleaned = _clean_mesh(vertices, faces)
        if len(cleaned.faces) == 0:
            return None

    fitted_vertices, fit_details = _fit_vertices_to_reference_bounds(
        np.asarray(cleaned.vertices, dtype=float),
        reference_points,
        link_cfg,
    )
    fitted = _clean_mesh(fitted_vertices, np.asarray(cleaned.faces, dtype=np.int64))
    return {
        'vertices': np.asarray(fitted.vertices, dtype=float),
        'faces': np.asarray(fitted.faces, dtype=np.int64),
        'pitch': float(pitch),
        'target_cells': int(target_cells),
        'cluster_scale': float(cluster_scale),
        'watertight': bool(fitted.is_watertight),
        'volume': float(abs(fitted.volume)),
        'fit': fit_details,
    }


def _remesh_link(link_name: str, mesh, link_cfg, basis: np.ndarray | None):
    source_vertices = np.asarray(mesh.vertices, dtype=float)
    source_faces = np.asarray(mesh.faces, dtype=np.int64)
    aligned_vertices = _aligned_points(source_vertices, basis)
    aligned_mesh = _clean_mesh(aligned_vertices, source_faces)
    reference_points = np.asarray(aligned_mesh.vertices, dtype=float)

    best = None
    cluster_scales = tuple(float(value) for value in link_cfg.cluster_scales)
    for target_cells in link_cfg.target_cells:
        for cluster_scale in cluster_scales:
            candidate = _candidate_from_mesh(aligned_mesh, reference_points, link_cfg, int(target_cells), cluster_scale)
            if candidate is None or not candidate['watertight']:
                continue
            if best is None or candidate['faces'].shape[0] < best['faces'].shape[0]:
                best = candidate
            if candidate['faces'].shape[0] <= int(link_cfg.max_faces):
                best = candidate
                break
        if best is not None and best['faces'].shape[0] <= int(link_cfg.max_faces):
            break

    if best is None:
        for target_cells in reversed(link_cfg.target_cells):
            candidate = _candidate_from_mesh(aligned_mesh, reference_points, link_cfg, int(target_cells), 0.0)
            if candidate is not None and candidate['watertight']:
                best = candidate
                break

    if best is None:
        raise RuntimeError(f'Unable to produce watertight marching-cubes mesh for {link_name}')

    output_vertices = _unaligned_points(best['vertices'], basis)
    output_mesh = _clean_mesh(output_vertices, best['faces'])
    source_bounds = np.asarray(mesh.bounds, dtype=float)
    output_bounds = np.asarray(output_mesh.bounds, dtype=float)
    source_extent = np.maximum(source_bounds[1] - source_bounds[0], 1e-9)
    output_extent = np.maximum(output_bounds[1] - output_bounds[0], 1e-9)
    return output_mesh, {
        'method': 'postprocess_voxel_marching_cubes',
        'axis_aligned': basis is not None,
        'source_vertex_count': int(len(mesh.vertices)),
        'source_face_count': int(len(mesh.faces)),
        'vertex_count': int(len(output_mesh.vertices)),
        'face_count': int(len(output_mesh.faces)),
        'watertight': bool(output_mesh.is_watertight),
        'volume': float(abs(output_mesh.volume)),
        'pitch': float(best['pitch']),
        'target_cells': int(best['target_cells']),
        'cluster_scale': float(best['cluster_scale']),
        'bounds_min': output_bounds[0].tolist(),
        'bounds_max': output_bounds[1].tolist(),
        'source_extent': source_extent.tolist(),
        'output_extent': output_extent.tolist(),
        'extent_ratio_xyz': (output_extent / source_extent).tolist(),
    }


def _parse_args() -> argparse.Namespace:
    folder = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description='Create a self-contained marching-cubes STL URDF package.')
    parser.add_argument('--usd-path', type=Path, default=default_usd_path())
    parser.add_argument('--output-dir', type=Path, default=folder / 'outputs')
    parser.add_argument('--source-urdf', type=Path, default=None)
    parser.add_argument('--source-mesh-dir', type=Path, default=None)
    parser.add_argument('--package-dir', type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    paths = resolve_asset_paths(args.usd_path, args.output_dir, mesh_package_dir=args.package_dir)
    source_urdf = args.source_urdf or paths.mesh_urdf
    source_mesh_dir = args.source_mesh_dir or paths.mesh_output_dir
    package_dir = paths.mesh_package_dir
    package_mesh_dir = paths.mesh_package_output_dir
    package_urdf = paths.mesh_package_urdf

    records = load_records_json(paths.skeleton_json)
    record_by_name = {record['name']: record for record in records}

    package_dir.mkdir(parents=True, exist_ok=True)
    if package_mesh_dir.exists():
        shutil.rmtree(package_mesh_dir)
    package_mesh_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_urdf, package_urdf)

    summary = {
        'source_urdf': str(source_urdf),
        'source_mesh_dir': str(source_mesh_dir),
        'package_dir': str(package_dir),
        'package_urdf': str(package_urdf),
        'package_mesh_dir': str(package_mesh_dir),
        'links': {},
    }

    stl_paths = sorted(source_mesh_dir.glob('*.stl'))
    if not stl_paths:
        raise RuntimeError(f'No STL files found in {source_mesh_dir}')

    for stl_path in stl_paths:
        link_name = stl_path.stem
        mesh = _load_trimesh(stl_path)
        link_cfg = resolve_lowpoly_link_config(DEFAULT_MESH_BUILD_CONFIG, link_name)
        basis = None
        axis_local = None
        if _is_arm_or_finger_link(link_name) and link_name in record_by_name:
            axis_local = _link_root_axis_local(record_by_name[link_name], record_by_name)
            basis = _rotation_basis_from_x_axis(axis_local) if axis_local is not None else None
        remeshed, stats = _remesh_link(link_name, mesh, link_cfg, basis)
        output_path = package_mesh_dir / stl_path.name
        _write_binary_stl(
            output_path,
            np.asarray(remeshed.vertices, dtype=float),
            [tuple(int(v) for v in face) for face in np.asarray(remeshed.faces, dtype=np.int64).tolist()],
            link_name,
        )
        if axis_local is not None and basis is not None:
            axis = axis_local / max(float(np.linalg.norm(axis_local)), 1e-9)
            stats['marching_axis_local'] = axis.tolist()
        else:
            stats['marching_axis_local'] = None
        stats['stl_path'] = str(output_path)
        summary['links'][link_name] = stats
        print(
            f"[PKG] {link_name}: faces {stats['source_face_count']}->{stats['face_count']} "
            f"watertight={stats['watertight']} axis_aligned={stats['axis_aligned']}",
            flush=True,
        )

    summary_path = package_dir / f'{paths.asset_tag}_marching_cube_summary.json'
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding='utf-8')
    print(f'[PKG] wrote package: {package_dir}', flush=True)
    print(f'[PKG] wrote summary: {summary_path}', flush=True)


if __name__ == '__main__':
    main()
