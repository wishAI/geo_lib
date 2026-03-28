from __future__ import annotations

import argparse
from pathlib import Path

from isaacsim import SimulationApp

from skeleton_common import extract_skeleton_records, generate_urdf_text, write_records_json


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Generate a simplified collision URDF from the source USD skeleton.')
    parser.add_argument(
        '--usd-path',
        type=Path,
        default=Path(__file__).resolve().parents[1] / 'avp_remote' / 'landau_v10.usdc',
        help='Path to the articulated USD/USDC file.',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path(__file__).resolve().parent / 'outputs',
        help='Directory for the generated JSON and URDF outputs.',
    )
    parser.add_argument('--robot-name', default='usd_landau_parallel', help='Name of the generated URDF robot.')
    parser.add_argument('--headless', action='store_true', default=True, help='Run Isaac Sim headless.')
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    app = SimulationApp({'headless': args.headless})

    try:
        from pxr import Usd, UsdSkel

        stage = Usd.Stage.Open(str(args.usd_path))
        if not stage:
            raise RuntimeError(f'Unable to open USD: {args.usd_path}')

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

        skeleton_json_path = args.output_dir / 'landau_v10_skeleton.json'
        urdf_path = args.output_dir / f'{args.robot_name}.urdf'
        write_records_json(skeleton_json_path, extracted['skeleton_path'], args.usd_path, records)
        urdf_path.write_text(generate_urdf_text(args.robot_name, records), encoding='utf-8')

        print(f'[GEN] source USD: {args.usd_path}')
        print(f"[GEN] skeleton path: {extracted['skeleton_path']}")
        print(f'[GEN] joint count: {len(records)}')
        print(f'[GEN] wrote: {skeleton_json_path}')
        print(f'[GEN] wrote: {urdf_path}')
    finally:
        app.close()


if __name__ == '__main__':
    main()
