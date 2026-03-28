from __future__ import annotations

import argparse
from pathlib import Path

from isaacsim import SimulationApp

from skeleton_common import extract_skeleton_records, write_records_json


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Print the articulated skeleton structure for the source USD.')
    parser.add_argument(
        '--usd-path',
        type=Path,
        default=Path(__file__).resolve().parents[1] / 'avp_remote' / 'landau_v10.usdc',
        help='Path to the source USD/USDC character asset.',
    )
    parser.add_argument(
        '--output-json',
        type=Path,
        default=Path(__file__).resolve().parent / 'outputs' / 'landau_v10_skeleton.json',
        help='Optional JSON export path.',
    )
    parser.add_argument('--headless', action='store_true', default=False, help='Run Isaac Sim headless.')
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    app = SimulationApp({'headless': args.headless})

    try:
        from pxr import Usd, UsdGeom, UsdSkel

        stage = Usd.Stage.Open(str(args.usd_path))
        if not stage:
            raise RuntimeError(f'Unable to open USD: {args.usd_path}')

        default_prim = stage.GetDefaultPrim()
        start = default_prim if default_prim and default_prim.IsValid() else stage.GetPseudoRoot()
        skel = None
        skel_root_path = None
        mesh_paths = []
        for prim in Usd.PrimRange(start):
            if prim.IsA(UsdGeom.Mesh) and len(mesh_paths) < 24:
                mesh_paths.append(str(prim.GetPath()))
            candidate = UsdSkel.Skeleton(prim)
            if skel is None and candidate and candidate.GetPrim().IsValid():
                skel = candidate
                skel_root_path = str(prim.GetPath())
        if skel is None:
            raise RuntimeError(f'No UsdSkel.Skeleton found under {start.GetPath()}')

        extracted = extract_skeleton_records(skel)
        records = extracted['records']

        print(f'[USD] file: {args.usd_path}')
        print(f'[USD] default prim: {default_prim.GetPath() if default_prim else "None"}')
        print(f'[USD] skeleton prim: {skel_root_path}')
        print(f'[USD] joint count: {len(records)}')
        print(f'[USD] sampled mesh prims: {len(mesh_paths)}')
        for mesh_path in mesh_paths:
            print(f'[USD]   mesh: {mesh_path}')
        for record in records:
            print(
                '[USD] joint='
                f"{record['name']} parent={record['parent_name'] or 'None'} "
                f"children={len(record['children'])} "
                f"local_xyz={[round(float(v), 4) for v in record['local_xyz']]} "
                f"world_xyz={[round(float(v), 4) for v in record['world_xyz']]}"
            )

        write_records_json(args.output_json, extracted['skeleton_path'], args.usd_path, records)
        print(f'[USD] wrote JSON summary to {args.output_json}')
    finally:
        app.close()


if __name__ == '__main__':
    main()
