from __future__ import annotations

import argparse
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from asset_paths import default_usd_path, resolve_asset_paths
from skeleton_common import (
    animation_clip_names,
    apply_pose_to_local_matrices,
    build_animation_clip,
    extract_skeleton_records,
    interpolate_pose_dict,
    root_height_offset,
    root_height_offset_from_world_matrices,
    world_matrices_from_local,
)
from validate_parallel_scene import (
    _add_ground_plane,
    _apply_pose_to_usd_skeleton,
    _camera_eye_target,
    _capture_rgba,
    _configure_urdf_pose,
    _ensure_gui_environment,
    _experience_path,
    _find_first_skeleton,
    _sanitize_dome_lights,
    _set_camera_view,
    _scene_offsets,
    _set_translate,
    _upsert_physics_scene,
    _wait_for_app_ready,
    _wait_for_prim,
)


def _parse_args() -> argparse.Namespace:
    folder = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description='Play a synchronized USD + URDF pose animation in Isaac Sim.')
    parser.add_argument('--usd-path', type=Path, default=default_usd_path())
    parser.add_argument('--urdf-path', type=Path, default=None, help='Defaults to the mesh-backed URDF for the selected input asset.')
    parser.add_argument('--headless', action='store_true')
    parser.add_argument(
        '--portable-root',
        type=Path,
        default=folder / '.kit_portable' / 'play_parallel_animation',
    )
    parser.add_argument(
        '--animation-clip',
        choices=animation_clip_names(),
        default='walk_cycle',
        help='Named animation clip to play.',
    )
    parser.add_argument(
        '--cycle-count',
        type=int,
        default=0,
        help='How many clip cycles to play. Use 0 to loop until you close Isaac Sim in GUI mode.',
    )
    parser.add_argument(
        '--step-dt',
        type=float,
        default=1.0 / 30.0,
        help='Animation sampling step in seconds.',
    )
    parser.add_argument(
        '--camera-view',
        choices=('overview', 'front', 'walk_side', 'hands'),
        default='walk_side',
        help='Camera view used for optional captures and initial GUI framing.',
    )
    parser.add_argument(
        '--capture-dir',
        type=Path,
        default=None,
        help='Optional directory for one PNG per named keyframe.',
    )
    parser.add_argument(
        '--self-collision',
        action='store_true',
        help='Enable URDF self-collision during Isaac import.',
    )
    parser.add_argument(
        '--post-import-warmup-steps',
        type=int,
        default=30,
        help='Number of Kit updates to run after URDF import before animation starts.',
    )
    parser.add_argument(
        '--pose-lift',
        type=float,
        default=0.12,
        help='Extra vertical lift applied to both characters during animation playback to keep walking poses clear of the ground plane.',
    )
    parser.add_argument(
        '--preserve-usd-dome-lights',
        action='store_true',
        help='Keep authored dome-light textures on the source USD instead of sanitizing them for headless robustness.',
    )
    return parser.parse_args()


def _count_revolute_joints(urdf_path: Path) -> int:
    robot = ET.parse(urdf_path).getroot()
    return sum(1 for joint in robot.findall('joint') if joint.attrib.get('type') == 'revolute')


def _pose_for_time(clip: list[tuple[str, dict[str, float], float]], time_s: float) -> tuple[str, str, dict[str, float]]:
    if len(clip) == 1:
        preset_name, pose, _ = clip[0]
        return preset_name, preset_name, dict(pose)
    total_duration = sum(duration for _, _, duration in clip)
    wrapped = time_s % total_duration
    elapsed = 0.0
    for index, (preset_name, pose, duration_s) in enumerate(clip):
        next_name, next_pose, _ = clip[(index + 1) % len(clip)]
        end = elapsed + duration_s
        if wrapped <= end or index == len(clip) - 1:
            alpha = 0.0 if duration_s <= 1e-8 else (wrapped - elapsed) / duration_s
            return preset_name, next_name, interpolate_pose_dict(pose, next_pose, alpha)
        elapsed = end
    preset_name, pose, _ = clip[-1]
    return preset_name, preset_name, dict(pose)


def main() -> None:
    args = _parse_args()
    folder = Path(__file__).resolve().parent
    asset_paths = resolve_asset_paths(args.usd_path, folder / 'outputs')
    urdf_path = args.urdf_path or asset_paths.mesh_urdf
    if not urdf_path.exists():
        raise RuntimeError(f'URDF does not exist yet: {urdf_path}')
    _ensure_gui_environment(args.headless)

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
            'extra_args': [
                '--portable-root',
                str(portable_root),
                f'--/app/captureFrame/path={screenshot_dir}',
                f'--/persistent/app/captureFrame/path={screenshot_dir}',
                *(['--no-window', '--/app/window/hideUi=1'] if args.headless else []),
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
        if not args.preserve_usd_dome_lights:
            _sanitize_dome_lights(stage, usd_root)
        usd_skel = _find_first_skeleton(stage, usd_root)
        if usd_skel is None:
            raise RuntimeError('Referenced USD did not expose a skeleton under /World/UsdCharacter')

        extracted = extract_skeleton_records(usd_skel)
        records = extracted['records']
        clip = build_animation_clip(records, args.animation_clip)
        total_clip_duration = sum(duration for _, _, duration in clip)
        base_z = root_height_offset(records)
        for _, pose, _ in clip:
            local_matrices = apply_pose_to_local_matrices(records, pose)
            posed_world = world_matrices_from_local(records, local_matrices)
            base_z = max(base_z, root_height_offset_from_world_matrices(posed_world))
        base_z += max(args.pose_lift, 0.0)

        status, import_config = omni.kit.commands.execute('URDFCreateImportConfig')
        if not status:
            raise RuntimeError('Unable to create Isaac URDF import config.')
        import_config.merge_fixed_joints = False
        import_config.convex_decomp = False
        import_config.import_inertia_tensor = True
        import_config.fix_base = True
        import_config.distance_scale = 1.0
        if hasattr(import_config, 'set_self_collision'):
            import_config.set_self_collision(args.self_collision)

        status, urdf_root = omni.kit.commands.execute(
            'URDFParseAndImportFile',
            urdf_path=str(urdf_path),
            import_config=import_config,
            get_articulation_root=True,
        )
        if not status:
            raise RuntimeError('URDF import failed.')
        _wait_for_prim(app, stage, urdf_root)
        for _ in range(max(args.post_import_warmup_steps, 0)):
            app.update()

        timeline = omni.timeline.get_timeline_interface()
        timeline.play()
        for _ in range(8):
            app.update()

        usd_offset, urdf_offset = _scene_offsets(args.camera_view, base_z)
        _set_translate(stage, usd_root, usd_offset)

        eye, target = _camera_eye_target(args.camera_view)
        camera = Camera(prim_path='/World/AnimationCamera', position=eye, resolution=(1280, 720), frequency=1)
        camera.initialize()
        _set_camera_view(camera, eye, target)

        if args.capture_dir is not None:
            capture_dir = args.capture_dir.resolve()
            capture_dir.mkdir(parents=True, exist_ok=True)
            seen = set()
            for preset_name, pose, _ in clip:
                if preset_name in seen:
                    continue
                seen.add(preset_name)
                local_matrices = apply_pose_to_local_matrices(records, pose)
                _apply_pose_to_usd_skeleton(stage, usd_skel, local_matrices)
                _configure_urdf_pose(stage, str(urdf_root), urdf_offset, pose, reset_missing=True)
                for _ in range(16):
                    app.update()
                _capture_rgba(camera, app, capture_dir / f'{preset_name}.png')

        loop_limit = args.cycle_count if args.cycle_count > 0 else (1 if args.headless else 0)
        elapsed_s = 0.0
        revolute_joint_count = _count_revolute_joints(urdf_path)
        print(f'[ANIM] usd path: {args.usd_path}', flush=True)
        print(f'[ANIM] urdf path: {urdf_path}', flush=True)
        print(f'[ANIM] revolute joint count: {revolute_joint_count}', flush=True)
        print(f'[ANIM] clip: {args.animation_clip} ({len(clip)} keyframes, {total_clip_duration:.3f}s)', flush=True)

        while app.is_running():
            start_name, end_name, pose = _pose_for_time(clip, elapsed_s)
            local_matrices = apply_pose_to_local_matrices(records, pose)
            _apply_pose_to_usd_skeleton(stage, usd_skel, local_matrices)
            _configure_urdf_pose(stage, str(urdf_root), urdf_offset, pose, reset_missing=True)
            app.update()
            elapsed_s += float(args.step_dt)
            if loop_limit > 0 and elapsed_s >= total_clip_duration * loop_limit:
                break

        print(f'[ANIM] playback finished after {elapsed_s:.3f}s', flush=True)
    finally:
        try:
            import omni.timeline

            omni.timeline.get_timeline_interface().stop()
        except Exception:
            pass
        app.close()


if __name__ == '__main__':
    main()
