from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

from skeleton_common import (
    apply_pose_to_local_matrices,
    build_demo_pose,
    extract_skeleton_records,
)
from validate_parallel_scene import (
    _apply_pose_to_usd_skeleton,
    _capture_rgba,
    _experience_path,
    _find_first_skeleton,
    _sanitize_dome_lights,
    _set_camera_view,
    _upsert_physics_scene,
    _wait_for_app_ready,
    _wait_for_prim,
)


def _parse_args() -> argparse.Namespace:
    folder = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description='Render the source USD with optional authored skeleton animation.')
    parser.add_argument('--usd-path', type=Path, default=folder.parents[1] / 'algorithms' / 'avp_remote' / 'landau_v10.usdc')
    parser.add_argument('--output-path', type=Path, required=True)
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--mode', choices=['none', 'rest', 'posed'], default='none')
    parser.add_argument(
        '--portable-root',
        type=Path,
        default=folder / '.kit_portable' / 'debug_usd_animation',
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    portable_root = args.portable_root.resolve()
    home_root = portable_root / 'home'
    (home_root / 'Documents').mkdir(parents=True, exist_ok=True)
    screenshot_dir = portable_root / 'documents' / 'Kit' / 'shared' / 'screenshots'
    screenshot_dir.mkdir(parents=True, exist_ok=True)
    os.environ['HOME'] = str(home_root)

    from isaacsim import SimulationApp

    app = SimulationApp(
        {
            'headless': args.headless,
            'renderer': 'RayTracedLighting',
            'extra_args': [
                '--portable-root',
                str(portable_root),
                f'--/app/captureFrame/path={screenshot_dir}',
                f'--/persistent/app/captureFrame/path={screenshot_dir}',
            ],
        },
        experience=str(_experience_path(args.headless)),
    )

    try:
        import omni.timeline
        import omni.usd
        from isaacsim.core.utils.stage import add_reference_to_stage
        from isaacsim.sensors.camera import Camera
        from pxr import Sdf, UsdLux

        _wait_for_app_ready(app)
        stage = omni.usd.get_context().get_stage()
        _upsert_physics_scene(stage)

        light = UsdLux.DistantLight.Define(stage, Sdf.Path('/World/DistantLight'))
        light.CreateIntensityAttr(3500)

        usd_root = '/World/UsdCharacter'
        add_reference_to_stage(usd_path=str(args.usd_path), prim_path=usd_root)
        _wait_for_prim(app, stage, usd_root)
        _sanitize_dome_lights(stage, usd_root)
        usd_skel = _find_first_skeleton(stage, usd_root)
        if usd_skel is None:
            raise RuntimeError('Referenced USD did not expose a skeleton under /World/UsdCharacter')

        records = extract_skeleton_records(usd_skel)['records']
        local_mats = [record['local_matrix'] for record in records]
        if args.mode == 'posed':
            local_mats = apply_pose_to_local_matrices(records, build_demo_pose(records))
        if args.mode in {'rest', 'posed'}:
            _apply_pose_to_usd_skeleton(stage, usd_skel, local_mats)

        timeline = omni.timeline.get_timeline_interface()
        timeline.play()
        for _ in range(24):
            app.update()

        eye = np.array([2.8, -2.8, 1.6], dtype=float)
        target = np.array([0.0, 0.0, 0.7], dtype=float)
        camera = Camera(prim_path='/World/RenderCamera', position=eye, resolution=(1280, 720), frequency=1)
        camera.initialize()
        _set_camera_view(camera, eye, target)
        _capture_rgba(camera, app, args.output_path)
        print(f'[DEBUG] wrote: {args.output_path}', flush=True)
    finally:
        try:
            import omni.timeline

            omni.timeline.get_timeline_interface().stop()
        except Exception:
            pass
        app.close()


if __name__ == '__main__':
    main()
