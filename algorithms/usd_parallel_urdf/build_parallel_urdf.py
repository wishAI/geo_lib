from __future__ import annotations

import argparse
import os
from pathlib import Path

from mesh_collision_builder import build_mesh_collision_assets
from skeleton_common import build_link_geometries, extract_skeleton_records, generate_urdf_text, save_json, write_records_json


def _parse_args() -> argparse.Namespace:
    folder = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description='Generate a simplified collision URDF from the source USD skeleton.')
    parser.add_argument(
        '--usd-path',
        type=Path,
        default=folder.parents[1] / 'algorithms' / 'avp_remote' / 'landau_v10.usdc',
        help='Path to the articulated USD/USDC file.',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=folder / 'outputs',
        help='Directory for the generated JSON and URDF outputs.',
    )
    parser.add_argument(
        '--geometry-mode',
        choices=('primitives', 'mesh', 'both'),
        default='both',
        help='Which URDF geometry variants to write.',
    )
    parser.add_argument('--robot-name', default='usd_landau_parallel', help='Name of the generated URDF robot.')
    parser.add_argument(
        '--mesh-robot-name',
        default=None,
        help='Optional robot name override for the mesh-backed URDF. Defaults to "<robot-name>_mesh".',
    )
    parser.add_argument(
        '--mesh-output-dir',
        type=Path,
        default=folder / 'outputs' / 'mesh_collision_stl',
        help='Directory where the per-link STL collision meshes will be written.',
    )
    parser.add_argument(
        '--mesh-simplify-mode',
        choices=('lowpoly_surface', 'obb', 'convex_hull'),
        default='lowpoly_surface',
        help='How to close and simplify the extracted per-link surface data into STL collision meshes.',
    )
    parser.add_argument(
        '--max-hull-faces',
        type=int,
        default=48,
        help='Upper bound for the convex-hull triangle count before falling back to a box mesh.',
    )
    parser.add_argument(
        '--target-hull-points',
        type=int,
        default=24,
        help='Point budget used when downsampling link-local point clouds before hull generation.',
    )
    parser.add_argument(
        '--portable-root',
        type=Path,
        default=folder / '.kit_portable' / 'build_parallel_urdf',
        help='Kit portable root used to keep Isaac runtime data inside the repo.',
    )
    parser.add_argument('--headless', action='store_true', default=True, help='Run Isaac Sim headless.')
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    portable_root = args.portable_root.resolve()
    home_root = portable_root / 'home'
    (home_root / 'Documents').mkdir(parents=True, exist_ok=True)
    os.environ['HOME'] = str(home_root)

    from isaacsim import AppFramework

    isaac_path = os.environ['ISAAC_PATH']
    app_args = [
        '--empty',
        '--ext-folder',
        f'{isaac_path}/exts',
        '--/app/asyncRendering=False',
        '--/app/fastShutdown=True',
        '--portable-root',
        str(portable_root),
        '--enable',
        'omni.usd',
        '--enable',
        'omni.kit.uiapp',
    ]
    if args.headless:
        app_args.insert(3, '--no-window')
    app = AppFramework('build_parallel_urdf', app_args)

    try:
        app.update()
        print('[GEN] app framework ready', flush=True)
        from pxr import Usd, UsdSkel

        stage = Usd.Stage.Open(str(args.usd_path))
        if not stage:
            raise RuntimeError(f'Unable to open USD: {args.usd_path}')
        print(f'[GEN] opened USD: {args.usd_path}', flush=True)

        skel = None
        for prim in stage.Traverse():
            candidate = UsdSkel.Skeleton(prim)
            if candidate and candidate.GetPrim().IsValid():
                skel = candidate
                break
        if skel is None:
            raise RuntimeError('No skeleton found in the selected USD.')

        extracted = extract_skeleton_records(skel)
        records = extracted['records']
        args.output_dir.mkdir(parents=True, exist_ok=True)
        primitive_geoms_by_name = build_link_geometries(records)
        print(f'[GEN] extracted skeleton: {len(records)} joints', flush=True)

        skeleton_json_path = args.output_dir / 'landau_v10_skeleton.json'
        write_records_json(skeleton_json_path, extracted['skeleton_path'], args.usd_path, records)

        print(f'[GEN] source USD: {args.usd_path}')
        print(f"[GEN] skeleton path: {extracted['skeleton_path']}")
        print(f'[GEN] joint count: {len(records)}')
        print(f'[GEN] wrote: {skeleton_json_path}')

        if args.geometry_mode in ('primitives', 'both'):
            primitive_urdf_path = args.output_dir / f'{args.robot_name}.urdf'
            primitive_urdf_path.write_text(
                generate_urdf_text(args.robot_name, records, geoms_by_name=primitive_geoms_by_name),
                encoding='utf-8',
            )
            print(f'[GEN] wrote primitive URDF: {primitive_urdf_path}')

        if args.geometry_mode in ('mesh', 'both'):
            mesh_robot_name = args.mesh_robot_name or f'{args.robot_name}_mesh'
            mesh_urdf_path = args.output_dir / f'{mesh_robot_name}.urdf'
            print('[GEN] building mesh collision assets...', flush=True)
            mesh_assets = build_mesh_collision_assets(
                stage=stage,
                skel=skel,
                records=records,
                urdf_dir=args.output_dir,
                mesh_dir=args.mesh_output_dir,
                strategy=args.mesh_simplify_mode,
                max_hull_faces=args.max_hull_faces,
                target_hull_points=args.target_hull_points,
            )
            print('[GEN] mesh collision assets ready', flush=True)
            mesh_summary_path = args.output_dir / 'mesh_collision_summary.json'
            mesh_urdf_path.write_text(
                generate_urdf_text(
                    mesh_robot_name,
                    records,
                    geoms_by_name=mesh_assets['geoms_by_name'],
                    inertial_geoms_by_name=primitive_geoms_by_name,
                ),
                encoding='utf-8',
            )
            save_json(
                mesh_summary_path,
                {
                    'usd_path': str(args.usd_path),
                    'skeleton_path': extracted['skeleton_path'],
                    'mesh_output_dir': str(args.mesh_output_dir),
                    'mesh_simplify_mode': args.mesh_simplify_mode,
                    'max_hull_faces': int(args.max_hull_faces),
                    'target_hull_points': int(args.target_hull_points),
                    'links': mesh_assets['summary'],
                },
            )
            print(f'[GEN] wrote mesh URDF: {mesh_urdf_path}')
            print(f'[GEN] wrote mesh summary: {mesh_summary_path}')
            print(f'[GEN] wrote mesh STL directory: {args.mesh_output_dir}')
    finally:
        app.close()


if __name__ == '__main__':
    main()
