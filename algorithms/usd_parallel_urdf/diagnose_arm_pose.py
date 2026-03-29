from __future__ import annotations

import argparse
from pathlib import Path

from asset_paths import default_usd_path, resolve_asset_paths
from compare_urdf_pose_offline import load_records_from_json
from pose_diagnostics import arm_pose_symmetry_report
from skeleton_common import pose_preset_names, save_json


def _parse_args() -> argparse.Namespace:
    folder = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description='Diagnose left/right arm-pose symmetry for a generated USD-parallel URDF.')
    parser.add_argument('--usd-path', type=Path, default=default_usd_path())
    parser.add_argument('--skeleton-json', type=Path, default=None)
    parser.add_argument('--urdf-path', type=Path, default=None)
    parser.add_argument('--output-path', type=Path, default=None)
    parser.add_argument(
        '--pose-preset',
        choices=pose_preset_names(),
        default='open_arms',
        help='Named pose preset to diagnose.',
    )
    parser.add_argument(
        '--skip-urdf',
        action='store_true',
        help='Only compute USD-side symmetry metrics.',
    )
    return parser.parse_args()


def _print_pair_metrics(prefix: str, pair_metrics: list[dict]) -> None:
    print(prefix)
    for metric in pair_metrics:
        print(
            '  '
            f"{metric['left']} vs {metric['right']}: "
            f"pos={metric['mirror_position_error_m']:.6e} m, "
            f"rot={metric['mirror_rotation_error_deg']:.3f} deg"
        )


def main() -> None:
    args = _parse_args()
    folder = Path(__file__).resolve().parent
    asset_paths = resolve_asset_paths(args.usd_path, folder / 'outputs')
    skeleton_json_path = args.skeleton_json or asset_paths.skeleton_json
    urdf_path = None if args.skip_urdf else (args.urdf_path or asset_paths.mesh_urdf)
    output_path = args.output_path or (asset_paths.output_dir / f'{asset_paths.asset_tag}_{args.pose_preset}_arm_pose_diagnostics.json')

    _, records = load_records_from_json(skeleton_json_path)
    report = arm_pose_symmetry_report(records, args.pose_preset, urdf_path=urdf_path)
    report['usd_path'] = str(args.usd_path)
    report['skeleton_json'] = str(skeleton_json_path)
    save_json(output_path, report)

    print(f'[DIAG] pose preset: {args.pose_preset}')
    print(f'[DIAG] skeleton json: {skeleton_json_path}')
    if urdf_path is not None:
        print(f'[DIAG] urdf path: {urdf_path}')
    print('[DIAG] joint values:')
    for joint_name, angle in report['pose_joint_values'].items():
        if '_l' in joint_name or '_r' in joint_name:
            print(f'  {joint_name}={angle:.6f}')
    _print_pair_metrics('[DIAG] USD mirror metrics:', report['usd_pair_metrics'])
    if 'urdf_pair_metrics' in report:
        _print_pair_metrics('[DIAG] URDF mirror metrics:', report['urdf_pair_metrics'])
        print('[DIAG] USD vs URDF arm-link metrics:')
        for metric in report['usd_vs_urdf_pair_metrics']:
            print(
                '  '
                f"{metric['left']} / {metric['right']}: "
                f"left_pos={metric['left_position_error_m']:.6e} m, "
                f"right_pos={metric['right_position_error_m']:.6e} m, "
                f"left_rot={metric['left_rotation_error_deg']:.3f} deg, "
                f"right_rot={metric['right_rotation_error_deg']:.3f} deg"
            )
    print('[DIAG] progressive USD scan:')
    for step in report['progressive_scan']:
        max_pos = max((metric['mirror_position_error_m'] for metric in step['pair_metrics']), default=0.0)
        max_rot = max((metric['mirror_rotation_error_deg'] for metric in step['pair_metrics']), default=0.0)
        print(f"  {step['step']}: max_pos={max_pos:.6e} m, max_rot={max_rot:.3f} deg, joints={step['joint_values']}")
    print(f'[DIAG] wrote: {output_path}')


if __name__ == '__main__':
    main()
