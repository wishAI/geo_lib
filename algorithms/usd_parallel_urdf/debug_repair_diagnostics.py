from __future__ import annotations

import math
import json
from pathlib import Path

import numpy as np

from asset_paths import default_usd_path
from mesh_collision_builder import _seed_surface_clouds
from mesh_repair_pipeline import (
    RepairConfig,
    _fix_normals,
    _remove_small_components,
    _simplify_mesh,
    extract_boundary_loops,
    fill_boundary_loop,
    make_trimesh,
    prepare_original_mesh_context_from_triangles,
)
from skeleton_common import extract_skeleton_records


def _edge_manifold_stats(mesh) -> dict[str, int]:
    if len(mesh.faces) == 0:
        return {
            'boundary_edges': 0,
            'nonmanifold_edges': 0,
            'edges_total': 0,
        }

    edges = np.sort(np.asarray(mesh.edges_sorted, dtype=np.int64), axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    return {
        'boundary_edges': int(np.count_nonzero(counts == 1)),
        'nonmanifold_edges': int(np.count_nonzero(counts > 2)),
        'edges_total': int(len(counts)),
    }


def main() -> None:
    folder = Path(__file__).resolve().parent
    portable_root = folder / '.kit_portable' / 'debug_repair_diagnostics'
    home_root = portable_root / 'home'
    (home_root / 'Documents').mkdir(parents=True, exist_ok=True)

    import os

    os.environ['HOME'] = str(home_root)

    from isaacsim import AppFramework

    isaac_path = os.environ['ISAAC_PATH']
    app = AppFramework(
        'debug_repair_diagnostics',
        [
            '--no-window',
            '--empty',
            '--ext-folder',
            f'{isaac_path}/exts',
            '--/app/asyncRendering=False',
            '--/app/fastShutdown=True',
            '--portable-root',
            str(portable_root),
            '--enable',
            'omni.usd',
        ],
    )
    app.update()

    from pxr import Usd, UsdSkel

    usd_path = default_usd_path()
    try:
        stage = Usd.Stage.Open(str(usd_path))
        if not stage:
            raise RuntimeError(f'Unable to open USD: {usd_path}')

        skel = None
        for prim in stage.Traverse():
            candidate = UsdSkel.Skeleton(prim)
            if candidate and candidate.GetPrim().IsValid():
                skel = candidate
                break
        if skel is None:
            raise RuntimeError('No skeleton found.')

        records = extract_skeleton_records(skel)['records']
        _, _, _, original_triangles, fragment_vertices_by_name, fragment_faces_by_name = _seed_surface_clouds(
            stage,
            skel,
            records,
        )
        context = prepare_original_mesh_context_from_triangles(original_triangles)
        config = RepairConfig()
        rows: dict[str, dict] = {}

        for link_name in ('head_x', 'root_x', 'spine_02_x', 'hand_l', 'neck_x'):
            fragment_vertices = fragment_vertices_by_name[link_name]
            fragment_faces = fragment_faces_by_name[link_name]
            initial_mesh = make_trimesh(fragment_vertices, fragment_faces)
            mesh = initial_mesh
            print(f'\nLINK {link_name} triangles={len(fragment_faces)}', flush=True)
            initial_stats = _edge_manifold_stats(mesh)
            print(
                f' initial faces={len(mesh.faces)} loops={len(extract_boundary_loops(mesh))} '
                f'watertight={mesh.is_watertight} euler={mesh.euler_number} '
                f'boundary_edges={initial_stats["boundary_edges"]} nonmanifold_edges={initial_stats["nonmanifold_edges"]}',
                flush=True,
            )
            mesh = _remove_small_components(mesh, keep_ratio=float(config.component_keep_area_ratio))
            mesh = make_trimesh(np.asarray(mesh.vertices), np.asarray(mesh.faces))
            cleaned_mesh = mesh.copy()
            cleaned_stats = _edge_manifold_stats(mesh)
            print(
                f' cleaned faces={len(mesh.faces)} loops={len(extract_boundary_loops(mesh))} '
                f'watertight={mesh.is_watertight} euler={mesh.euler_number} '
                f'boundary_edges={cleaned_stats["boundary_edges"]} nonmanifold_edges={cleaned_stats["nonmanifold_edges"]}',
                flush=True,
            )

            loops = extract_boundary_loops(mesh)
            vertices = np.asarray(mesh.vertices, dtype=float)
            faces_list = [tuple(int(v) for v in face) for face in np.asarray(mesh.faces, dtype=np.int64).tolist()]
            holes_filled = 0
            for loop in sorted(loops, key=len, reverse=True):
                filled = fill_boundary_loop(vertices, loop, context, config)
                if filled is None:
                    print(f'  skipped loop edges={len(loop)}', flush=True)
                    continue
                vertices, new_faces, _ = filled
                faces_list.extend(new_faces)
                holes_filled += 1
            mesh = make_trimesh(vertices, np.asarray(faces_list, dtype=np.int64))
            filled_mesh = mesh.copy()
            fill_stats = _edge_manifold_stats(mesh)
            print(
                f' after_fill faces={len(mesh.faces)} holes_filled={holes_filled} loops={len(extract_boundary_loops(mesh))} '
                f'watertight={mesh.is_watertight} euler={mesh.euler_number} '
                f'boundary_edges={fill_stats["boundary_edges"]} nonmanifold_edges={fill_stats["nonmanifold_edges"]}',
                flush=True,
            )

            import trimesh

            trimesh.repair.fill_holes(mesh)
            mesh = _fix_normals(mesh, context)
            pre_simplify_mesh = mesh.copy()
            pre_stats = _edge_manifold_stats(mesh)
            print(
                f' pre_simplify faces={len(mesh.faces)} loops={len(extract_boundary_loops(mesh))} '
                f'watertight={mesh.is_watertight} euler={mesh.euler_number} '
                f'boundary_edges={pre_stats["boundary_edges"]} nonmanifold_edges={pre_stats["nonmanifold_edges"]}',
                flush=True,
            )
            target_faces = max(
                4,
                min(int(math.ceil(len(mesh.faces) * float(config.target_face_ratio))), int(config.max_faces)),
            )
            simplified, method = _simplify_mesh(mesh, target_faces)
            simplified_stats = _edge_manifold_stats(simplified)
            print(
                f' simplified faces={len(simplified.faces)} method={method} loops={len(extract_boundary_loops(simplified))} '
                f'watertight={simplified.is_watertight} euler={simplified.euler_number} '
                f'boundary_edges={simplified_stats["boundary_edges"]} nonmanifold_edges={simplified_stats["nonmanifold_edges"]}',
                flush=True,
            )
            rows[link_name] = {
                'initial': {
                    'faces': int(len(fragment_faces)),
                    'loops': int(len(extract_boundary_loops(initial_mesh))),
                    'watertight': bool(initial_mesh.is_watertight),
                    'euler': int(initial_mesh.euler_number),
                    **initial_stats,
                },
                'cleaned': {
                    'faces': int(len(cleaned_mesh.faces)),
                    'loops': int(len(extract_boundary_loops(cleaned_mesh))),
                    'watertight': bool(cleaned_mesh.is_watertight),
                    'euler': int(cleaned_mesh.euler_number),
                    **cleaned_stats,
                },
                'after_fill': {
                    'faces': int(len(filled_mesh.faces)),
                    'holes_filled': int(holes_filled),
                    'loops': int(len(extract_boundary_loops(filled_mesh))),
                    'watertight': bool(filled_mesh.is_watertight),
                    'euler': int(filled_mesh.euler_number),
                    **fill_stats,
                },
                'pre_simplify': {
                    'faces': int(len(pre_simplify_mesh.faces)),
                    'loops': int(len(extract_boundary_loops(pre_simplify_mesh))),
                    'watertight': bool(pre_simplify_mesh.is_watertight),
                    'euler': int(pre_simplify_mesh.euler_number),
                    **pre_stats,
                },
                'simplified': {
                    'faces': int(len(simplified.faces)),
                    'method': method,
                    'loops': int(len(extract_boundary_loops(simplified))),
                    'watertight': bool(simplified.is_watertight),
                    'euler': int(simplified.euler_number),
                    **simplified_stats,
                },
            }

        output_path = folder / 'outputs' / 'debug_repair_diagnostics.json'
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(rows, indent=2), encoding='utf-8')
    finally:
        app.close()


if __name__ == '__main__':
    main()
