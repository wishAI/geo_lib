from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

from skeleton_common import (
    apply_pose_to_local_matrices,
    build_demo_pose,
    extract_skeleton_records,
    root_height_offset,
    root_height_offset_from_world_matrices,
    world_matrices_from_local,
)
from validate_parallel_scene import (
    _add_ground_plane,
    _apply_pose_to_usd_skeleton,
    _capture_rgba,
    _configure_urdf_pose,
    _experience_path,
    _find_first_skeleton,
    _log,
    _sanitize_dome_lights,
    _set_camera_view,
    _set_translate,
    _upsert_physics_scene,
    _wait_for_app_ready,
    _wait_for_prim,
)


def _parse_args() -> argparse.Namespace:
    folder = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description='Render one source-USD plus generated-URDF scene view.')
    parser.add_argument('--usd-path', type=Path, default=folder.parents[1] / 'algorithms' / 'avp_remote' / 'landau_v10.usdc')
    parser.add_argument('--urdf-path', type=Path, default=folder / 'outputs' / 'usd_landau_parallel.urdf')
    parser.add_argument('--output-path', type=Path, required=True)
    parser.add_argument('--view', choices=['overview', 'hands'], default='overview')
    parser.add_argument('--posed', action='store_true', default=False)
    parser.add_argument(
        '--usd-animation',
        choices=['auto', 'none', 'rest', 'posed'],
        default='auto',
        help='How to drive the source USD skeleton. "auto" uses posed mode only when --posed is set.',
    )
    parser.add_argument('--headless', action='store_true')
    parser.add_argument(
        '--portable-root',
        type=Path,
        default=folder / '.kit_portable' / 'render_parallel_scene',
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
        import omni.kit.commands
        import omni.timeline
        import omni.usd
        from isaacsim.core.utils.stage import add_reference_to_stage
        from isaacsim.sensors.camera import Camera
        from pxr import Sdf, UsdLux

        _wait_for_app_ready(app)
        stage = omni.usd.get_context().get_stage()
        _upsert_physics_scene(stage)
        _add_ground_plane(stage)
        light = UsdLux.DistantLight.Define(stage, Sdf.Path('/World/DistantLight'))
        light.CreateIntensityAttr(3500)

        usd_root = '/World/UsdCharacter'
        add_reference_to_stage(usd_path=str(args.usd_path), prim_path=usd_root)
        _wait_for_prim(app, stage, usd_root)
        _sanitize_dome_lights(stage, usd_root)
        usd_skel = _find_first_skeleton(stage, usd_root)
        if usd_skel is None:
            raise RuntimeError('Referenced USD did not expose a skeleton under /World/UsdCharacter')
        extracted = extract_skeleton_records(usd_skel)
        records = extracted['records']
        pose = build_demo_pose(records)
        local_rest = [record['local_matrix'].copy() for record in records]
        local_posed = apply_pose_to_local_matrices(records, pose)
        posed_world_local = world_matrices_from_local(records, local_posed)
        base_z = root_height_offset(records)
        posed_base_z = max(base_z, root_height_offset_from_world_matrices(posed_world_local))
        usd_animation_mode = args.usd_animation
        if usd_animation_mode == 'auto':
            usd_animation_mode = 'posed' if args.posed else 'none'

        status, import_config = omni.kit.commands.execute('URDFCreateImportConfig')
        if not status:
            raise RuntimeError('Unable to create Isaac URDF import config.')
        import_config.merge_fixed_joints = False
        import_config.convex_decomp = False
        import_config.import_inertia_tensor = True
        import_config.fix_base = True
        import_config.distance_scale = 1.0
        if hasattr(import_config, 'set_self_collision'):
            import_config.set_self_collision(True)
        status, urdf_root = omni.kit.commands.execute('URDFParseAndImportFile', urdf_path=str(args.urdf_path), import_config=import_config, get_articulation_root=True)
        if not status:
            raise RuntimeError('URDF import failed.')
        _wait_for_prim(app, stage, urdf_root)

        timeline = omni.timeline.get_timeline_interface()
        timeline.play()
        for _ in range(8):
            app.update()
        scene_base_z = posed_base_z if (args.posed or usd_animation_mode == 'posed') else base_z
        side_offset = 0.35 if args.view == 'hands' else 0.55
        _set_translate(stage, usd_root, np.array([-side_offset, 0.0, scene_base_z], dtype=float))
        _configure_urdf_pose(stage, str(urdf_root), np.array([side_offset, 0.0, scene_base_z], dtype=float), pose if args.posed else {})

        if usd_animation_mode == 'rest':
            _apply_pose_to_usd_skeleton(stage, usd_skel, local_rest)
        elif usd_animation_mode == 'posed':
            _apply_pose_to_usd_skeleton(stage, usd_skel, local_posed)

        for _ in range(20):
            app.update()

        if args.view == 'overview':
            eye = np.array([4.0, -4.0, 2.2], dtype=float)
            target = np.array([0.0, 0.0, 0.5], dtype=float)
        else:
            eye = np.array([0.0, -3.0, 1.0], dtype=float)
            target = np.array([0.0, 0.0, 0.68], dtype=float)
        camera = Camera(prim_path='/World/RenderCamera', position=eye, resolution=(1280, 720), frequency=1)
        camera.initialize()
        _set_camera_view(camera, eye, target)
        _capture_rgba(camera, app, args.output_path)
        _log(f'[RENDER] wrote: {args.output_path}')
    finally:
        try:
            import omni.timeline

            omni.timeline.get_timeline_interface().stop()
        except Exception:
            pass
        app.close()


if __name__ == '__main__':
    main()
