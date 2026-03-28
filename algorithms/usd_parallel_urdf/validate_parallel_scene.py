from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np

from skeleton_common import (
    apply_pose_to_local_matrices,
    build_demo_pose,
    extract_skeleton_records,
    root_height_offset,
    root_height_offset_from_world_matrices,
    rotation_error_radians,
    save_json,
    world_matrices_from_local,
)


def _parse_args() -> argparse.Namespace:
    folder = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description='Load the source USD and generated URDF together in Isaac Sim and compare poses.')
    parser.add_argument(
        '--usd-path',
        type=Path,
        default=folder.parents[1] / 'algorithms' / 'avp_remote' / 'landau_v10.usdc',
    )
    parser.add_argument(
        '--urdf-path',
        type=Path,
        default=folder / 'outputs' / 'usd_landau_parallel.urdf',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=folder / 'outputs' / 'validation',
    )
    parser.add_argument('--headless', action='store_true')
    parser.add_argument(
        '--portable-root',
        type=Path,
        default=folder / '.kit_portable' / 'validate_parallel_scene',
    )
    parser.add_argument(
        '--live-prim-comparison',
        action='store_true',
        help='Compare live Isaac prim transforms instead of using the offline URDF-vs-USD kinematic comparison.',
    )
    parser.add_argument(
        '--self-collision',
        action='store_true',
        help='Enable URDF self-collision during Isaac import. Disabled by default for stability with approximate colliders.',
    )
    parser.add_argument(
        '--post-import-warmup-steps',
        type=int,
        default=30,
        help='Number of Kit updates to run after URDF import before applying poses.',
    )
    parser.add_argument(
        '--stay-open',
        action='store_true',
        help='Keep the Isaac Sim GUI open after the scene is prepared. Close the window yourself to exit.',
    )
    return parser.parse_args()


def _ensure_gui_environment(headless: bool) -> None:
    if headless:
        return
    if os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY'):
        return
    session_type = os.environ.get('XDG_SESSION_TYPE', '')
    raise RuntimeError(
        'GUI mode requires a graphical desktop session, but this shell has no DISPLAY/WAYLAND_DISPLAY '
        f'(XDG_SESSION_TYPE={session_type or "unset"}). Run the script from a local graphical terminal or '
        'export the appropriate display environment first.'
    )


def _experience_path(headless: bool) -> Path:
    del headless
    isaac_root = Path('/home/wishai/vscode/IsaacLab/_isaac_sim')
    return isaac_root / 'apps' / 'isaacsim.exp.base.kit'


def _upsert_physics_scene(stage):
    from pxr import Gf, PhysxSchema, Sdf, UsdPhysics

    scene = UsdPhysics.Scene.Define(stage, Sdf.Path('/physicsScene'))
    scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
    scene.CreateGravityMagnitudeAttr().Set(9.81)
    physx_api = PhysxSchema.PhysxSceneAPI.Apply(stage.GetPrimAtPath('/physicsScene'))
    physx_api.CreateEnableCCDAttr(True)
    physx_api.CreateEnableStabilizationAttr(True)
    physx_api.CreateEnableGPUDynamicsAttr(False)
    physx_api.CreateBroadphaseTypeAttr('MBP')
    physx_api.CreateSolverTypeAttr('TGS')


def _add_ground_plane(stage) -> None:
    import omni.kit.commands
    from pxr import Gf

    omni.kit.commands.execute(
        'AddGroundPlaneCommand',
        stage=stage,
        planePath='/World/groundPlane',
        axis='Z',
        size=1500.0,
        position=Gf.Vec3f(0.0, 0.0, 0.0),
        color=Gf.Vec3f(0.35),
    )


def _set_translate(stage, prim_path: str, xyz: np.ndarray) -> None:
    from pxr import Gf, UsdGeom

    prim = stage.GetPrimAtPath(prim_path)
    xformable = UsdGeom.Xformable(prim)
    local_transform = xformable.GetLocalTransformation()
    if isinstance(local_transform, tuple):
        local_transform = local_transform[0]
    matrix = Gf.Matrix4d(local_transform)
    matrix.SetTranslateOnly(Gf.Vec3d(float(xyz[0]), float(xyz[1]), float(xyz[2])))
    xformable.ClearXformOpOrder()
    transform_op = xformable.AddTransformOp(opSuffix='parallelPose', precision=UsdGeom.XformOp.PrecisionDouble)
    transform_op.Set(matrix)


def _find_first_skeleton(stage, root_prim_path: str):
    from pxr import Usd, UsdSkel

    root_prim = stage.GetPrimAtPath(root_prim_path)
    for prim in Usd.PrimRange(root_prim):
        candidate = UsdSkel.Skeleton(prim)
        if candidate and candidate.GetPrim().IsValid():
            return candidate
    return None


def _trs_arrays_from_local_matrices(matrices):
    from pxr import Gf

    translations = []
    rotations = []
    scales = []
    for matrix in matrices:
        rotation = np.asarray(matrix[:3, :3], dtype=float)
        scale = np.linalg.norm(rotation, axis=0)
        safe_scale = np.where(scale < 1e-8, 1.0, scale)
        pure_rotation = rotation / safe_scale
        quat_wxyz = _quat_wxyz_from_matrix(pure_rotation)
        translations.append(Gf.Vec3f(float(matrix[0, 3]), float(matrix[1, 3]), float(matrix[2, 3])))
        rotations.append(Gf.Quatf(float(quat_wxyz[0]), Gf.Vec3f(float(quat_wxyz[1]), float(quat_wxyz[2]), float(quat_wxyz[3]))))
        scales.append(Gf.Vec3h(float(scale[0]), float(scale[1]), float(scale[2])))
    return translations, rotations, scales


def _sanitize_dome_lights(stage, root_prim_path: str) -> int:
    from pxr import Gf, Sdf, Usd, UsdLux

    fixed_count = 0
    root_prim = stage.GetPrimAtPath(root_prim_path)
    for prim in Usd.PrimRange(root_prim):
        if not prim.IsA(UsdLux.DomeLight):
            continue
        dome = UsdLux.DomeLight(prim)
        texture_attr = dome.GetTextureFileAttr()
        if texture_attr:
            try:
                texture_attr.Set(Sdf.AssetPath(''))
            except Exception:
                texture_attr.Set('')
        format_attr = prim.GetAttribute('inputs:texture:format')
        if format_attr and format_attr.IsValid():
            format_attr.Set('latlong')
        dome.GetColorAttr().Set(Gf.Vec3f(0.08, 0.08, 0.08))
        if not dome.GetIntensityAttr().HasAuthoredValue():
            dome.GetIntensityAttr().Set(200.0)
        fixed_count += 1
    return fixed_count


def _wait_for_app_ready(sim_app, max_updates: int = 7200) -> None:
    import omni.kit.app
    import omni.usd

    kit_app = omni.kit.app.get_app()
    for _ in range(max_updates):
        sim_app.update()
        stage = omni.usd.get_context().get_stage()
        if kit_app.is_app_ready() and stage is not None:
            return
    raise RuntimeError(f'Timed out waiting for Isaac Sim app readiness after {max_updates} updates.')


def _wait_for_prim(sim_app, stage, prim_path: str, max_updates: int = 600):
    for _ in range(max_updates):
        prim = stage.GetPrimAtPath(prim_path)
        if prim and prim.IsValid():
            return prim
        sim_app.update()
    raise RuntimeError(f'Timed out waiting for prim to appear: {prim_path}')


def _log(message: str) -> None:
    print(message, flush=True)


def _write_checkpoint(output_dir: Path, stage_name: str) -> None:
    save_json(output_dir / '_checkpoint.json', {'stage': stage_name})


def _robot_root_path(articulation_root_path: str) -> str:
    from pxr import Sdf

    return str(Sdf.Path(articulation_root_path).GetParentPath())


def _set_drive_parameters(drive, target_deg: float, stiffness: float, damping: float, max_force: float | None = None) -> None:
    if not drive.GetTargetPositionAttr():
        drive.CreateTargetPositionAttr(target_deg)
    else:
        drive.GetTargetPositionAttr().Set(target_deg)
    if not drive.GetStiffnessAttr():
        drive.CreateStiffnessAttr(stiffness)
    else:
        drive.GetStiffnessAttr().Set(stiffness)
    if not drive.GetDampingAttr():
        drive.CreateDampingAttr(damping)
    else:
        drive.GetDampingAttr().Set(damping)
    if max_force is not None:
        if not drive.GetMaxForceAttr():
            drive.CreateMaxForceAttr(max_force)
        else:
            drive.GetMaxForceAttr().Set(max_force)


def _configure_urdf_pose(stage, urdf_root: str, root_xyz: np.ndarray, pose: dict[str, float]) -> int:
    from pxr import PhysxSchema, Usd, UsdPhysics

    robot_root = _robot_root_path(urdf_root)
    _set_translate(stage, robot_root, root_xyz)
    articulation_api = PhysxSchema.PhysxArticulationAPI.Get(stage, urdf_root)
    if articulation_api:
        articulation_api.CreateSolverPositionIterationCountAttr(64)
        articulation_api.CreateSolverVelocityIterationCountAttr(32)

    drive_prims = {}
    root_prim = stage.GetPrimAtPath(robot_root)
    for prim in Usd.PrimRange(root_prim):
        if not prim.HasAPI(UsdPhysics.DriveAPI):
            continue
        name = prim.GetName()
        path = str(prim.GetPath())
        current = drive_prims.get(name)
        if current is None or len(path) < len(str(current.GetPath())):
            drive_prims[name] = prim

    configured = 0
    angular_stiffness = math.radians(5.0e6)
    angular_damping = math.radians(5.0e5)
    max_force = 1.0e7
    for joint_name, angle_rad in pose.items():
        joint_prim = drive_prims.get(joint_name)
        if joint_prim is None or not joint_prim.IsValid():
            continue
        drive = UsdPhysics.DriveAPI.Get(joint_prim, 'angular')
        if not drive:
            continue
        _set_drive_parameters(drive, math.degrees(angle_rad), angular_stiffness, angular_damping, max_force)
        configured += 1
    return configured


def _apply_pose_to_usd_skeleton(stage, skeleton, local_matrices) -> str:
    from pxr import UsdSkel

    binding_prim = skeleton.GetPrim()
    current = skeleton.GetPrim()
    while current and current.IsValid():
        if current.IsA(UsdSkel.Root):
            binding_prim = current
            break
        current = current.GetParent()

    anim = UsdSkel.Animation.Define(stage, binding_prim.GetPath().AppendChild('ParallelPoseAnim'))
    anim.GetJointsAttr().Set(skeleton.GetJointsAttr().Get())
    translations, rotations, scales = _trs_arrays_from_local_matrices(local_matrices)
    anim.GetTranslationsAttr().Set(translations)
    anim.GetRotationsAttr().Set(rotations)
    anim.GetScalesAttr().Set(scales)
    binding = UsdSkel.BindingAPI.Apply(binding_prim)
    if not binding.GetSkeletonRel().GetTargets():
        binding.CreateSkeletonRel().SetTargets([skeleton.GetPath()])
    binding.CreateAnimationSourceRel().SetTargets([anim.GetPrim().GetPath()])
    if binding_prim != skeleton.GetPrim():
        UsdSkel.BindingAPI.Apply(skeleton.GetPrim()).CreateAnimationSourceRel().SetTargets([anim.GetPrim().GetPath()])
    return str(anim.GetPrim().GetPath())


def _capture_rgba(camera, sim_app, path: Path, steps: int = 12) -> None:
    from PIL import Image

    for _ in range(steps):
        sim_app.update()
    rgba = camera.get_rgba()
    image = np.asarray(rgba[:, :, :3], dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(path)


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm < 1e-12:
        raise ValueError('Cannot normalize a near-zero vector.')
    return vec / norm


def _look_at_rotation_world_from_camera_usd(
    camera_pos_world: np.ndarray,
    target_world: np.ndarray,
    world_up: np.ndarray = np.array([0.0, 0.0, 1.0]),
) -> np.ndarray:
    forward = _normalize(target_world - camera_pos_world)
    z_axis = -forward
    x_axis = np.cross(world_up, z_axis)
    if np.linalg.norm(x_axis) < 1e-8:
        x_axis = np.cross(np.array([0.0, 1.0, 0.0]), z_axis)
    x_axis = _normalize(x_axis)
    y_axis = _normalize(np.cross(z_axis, x_axis))
    return np.column_stack((x_axis, y_axis, z_axis))


def _quat_wxyz_from_matrix(matrix: np.ndarray) -> np.ndarray:
    trace = float(matrix[0, 0] + matrix[1, 1] + matrix[2, 2])
    if trace > 0.0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (matrix[2, 1] - matrix[1, 2]) * s
        y = (matrix[0, 2] - matrix[2, 0]) * s
        z = (matrix[1, 0] - matrix[0, 1]) * s
    elif matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
        s = 2.0 * np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2])
        w = (matrix[2, 1] - matrix[1, 2]) / s
        x = 0.25 * s
        y = (matrix[0, 1] + matrix[1, 0]) / s
        z = (matrix[0, 2] + matrix[2, 0]) / s
    elif matrix[1, 1] > matrix[2, 2]:
        s = 2.0 * np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2])
        w = (matrix[0, 2] - matrix[2, 0]) / s
        x = (matrix[0, 1] + matrix[1, 0]) / s
        y = 0.25 * s
        z = (matrix[1, 2] + matrix[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1])
        w = (matrix[1, 0] - matrix[0, 1]) / s
        x = (matrix[0, 2] + matrix[2, 0]) / s
        y = (matrix[1, 2] + matrix[2, 1]) / s
        z = 0.25 * s
    quat = np.array([w, x, y, z], dtype=float)
    return quat / np.linalg.norm(quat)


def _set_camera_view(camera, eye: np.ndarray, target: np.ndarray) -> None:
    rotation = _look_at_rotation_world_from_camera_usd(eye, target)
    quat_wxyz = _quat_wxyz_from_matrix(rotation)
    camera.set_world_pose(position=eye.tolist(), orientation=quat_wxyz.tolist(), camera_axes='usd')


def _collect_link_prim_paths(stage, root_path: str, link_names: set[str]) -> dict:
    from pxr import Usd

    root_prim = stage.GetPrimAtPath(root_path)
    best = {}
    for prim in Usd.PrimRange(root_prim):
        name = prim.GetName()
        if name not in link_names:
            continue
        current = best.get(name)
        candidate = str(prim.GetPath())
        if current is None or len(candidate) < len(current):
            best[name] = candidate
    return best


def _prim_world_matrix(stage, prim_path: str) -> np.ndarray:
    from pxr import UsdGeom

    prim = stage.GetPrimAtPath(prim_path)
    matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform()
    return np.array([[float(matrix[r][c]) for c in range(4)] for r in range(4)], dtype=float).T


def _offline_summary_payload(output_dir: Path) -> dict:
    offline_path = output_dir / 'offline_transform_comparison.json'
    if not offline_path.exists():
        return {
            'comparison_mode': 'offline_unavailable',
            'offline_transform_comparison_path': str(offline_path),
        }
    payload = json.loads(offline_path.read_text(encoding='utf-8'))
    payload['comparison_mode'] = 'offline_root_relative_kinematics'
    payload['offline_transform_comparison_path'] = str(offline_path)
    return payload


def main() -> None:
    args = _parse_args()
    if args.stay_open and args.headless:
        raise RuntimeError('--stay-open requires GUI mode. Remove --headless when using this option.')
    portable_root = args.portable_root.resolve()
    home_root = portable_root / 'home'
    (home_root / 'Documents').mkdir(parents=True, exist_ok=True)
    screenshot_dir = portable_root / 'documents' / 'Kit' / 'shared' / 'screenshots'
    screenshot_dir.mkdir(parents=True, exist_ok=True)
    _ensure_gui_environment(args.headless)
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
        import omni.kit.app
        import omni.timeline
        import omni.usd
        from isaacsim.core.utils.stage import add_reference_to_stage
        from pxr import Sdf, UsdLux

        if not args.urdf_path.exists():
            raise RuntimeError(f'URDF does not exist yet: {args.urdf_path}')

        args.output_dir.mkdir(parents=True, exist_ok=True)
        _wait_for_app_ready(app)
        stage = omni.usd.get_context().get_stage()
        _log(f'[VAL] app ready: {omni.kit.app.get_app().is_app_ready()}')
        _write_checkpoint(args.output_dir, 'app_ready')
        _upsert_physics_scene(stage)
        _add_ground_plane(stage)

        light = UsdLux.DistantLight.Define(stage, Sdf.Path('/World/DistantLight'))
        light.CreateIntensityAttr(3500)

        usd_root = '/World/UsdCharacter'
        add_reference_to_stage(usd_path=str(args.usd_path), prim_path=usd_root)
        _wait_for_prim(app, stage, usd_root)
        _write_checkpoint(args.output_dir, 'usd_loaded')
        sanitized_domes = _sanitize_dome_lights(stage, usd_root)
        usd_skel = _find_first_skeleton(stage, usd_root)
        if usd_skel is None:
            raise RuntimeError('Referenced USD did not expose a skeleton under /World/UsdCharacter')

        extracted = extract_skeleton_records(usd_skel)
        records = extracted['records']
        pose = build_demo_pose(records)
        local_posed = apply_pose_to_local_matrices(records, pose)
        posed_world_local = world_matrices_from_local(records, local_posed)
        rest_base_z = root_height_offset(records)
        posed_base_z = max(rest_base_z, root_height_offset_from_world_matrices(posed_world_local))

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
            urdf_path=str(args.urdf_path),
            import_config=import_config,
            get_articulation_root=True,
        )
        if not status:
            raise RuntimeError('URDF import failed.')
        _wait_for_prim(app, stage, urdf_root)
        _write_checkpoint(args.output_dir, 'urdf_imported')
        for _ in range(max(args.post_import_warmup_steps, 0)):
            app.update()
        robot_root = _robot_root_path(str(urdf_root))

        timeline = omni.timeline.get_timeline_interface()
        timeline.play()
        for _ in range(8):
            app.update()
        _write_checkpoint(args.output_dir, 'timeline_playing')

        usd_offset = np.array([-0.55, 0.0, posed_base_z], dtype=float)
        urdf_offset = np.array([0.55, 0.0, posed_base_z], dtype=float)
        _set_translate(stage, usd_root, usd_offset)
        driven_joint_count = _configure_urdf_pose(stage, str(urdf_root), urdf_offset, pose)
        _log(f'[VAL] configured URDF drives: {driven_joint_count}')
        _write_checkpoint(args.output_dir, 'urdf_pose_applied')
        usd_anim_path = _apply_pose_to_usd_skeleton(stage, usd_skel, local_posed)
        _log(f'[VAL] usd animation: {usd_anim_path}')
        _write_checkpoint(args.output_dir, 'usd_pose_applied')

        for _ in range(20):
            app.update()
        _write_checkpoint(args.output_dir, 'pose_settled')

        summary = {
            'usd_path': str(args.usd_path),
            'urdf_path': str(args.urdf_path),
            'usd_root_prim': usd_root,
            'usd_animation_prim': usd_anim_path,
            'urdf_root_prim': robot_root,
            'urdf_articulation_prim': str(urdf_root),
            'pose_radians': pose,
            'rest_root_height_offset_m': rest_base_z,
            'posed_root_height_offset_m': posed_base_z,
            'sanitized_dome_light_count': sanitized_domes,
            'live_prim_comparison_enabled': bool(args.live_prim_comparison),
        }
        if args.live_prim_comparison:
            usd_world_local = world_matrices_from_local(records, local_posed)
            usd_root_inv = np.linalg.inv(usd_world_local[0])
            usd_by_name = {
                record['name']: usd_root_inv @ usd_world_local[record['index']]
                for record in records
            }

            link_names = {record['name'] for record in records}
            urdf_link_paths = _collect_link_prim_paths(stage, robot_root, link_names)
            if 'root_x' not in urdf_link_paths:
                raise RuntimeError(f'Unable to resolve imported URDF link prims under {robot_root}')

            urdf_root_world = _prim_world_matrix(stage, urdf_link_paths['root_x'])
            urdf_root_inv = np.linalg.inv(urdf_root_world)
            urdf_by_name = {
                name: urdf_root_inv @ _prim_world_matrix(stage, prim_path)
                for name, prim_path in urdf_link_paths.items()
            }

            per_link = []
            for record in records:
                name = record['name']
                if name not in urdf_by_name:
                    continue
                usd_matrix = usd_by_name[name]
                urdf_matrix = urdf_by_name[name]
                pos_error = float(np.linalg.norm(usd_matrix[:3, 3] - urdf_matrix[:3, 3]))
                rot_error = float(rotation_error_radians(usd_matrix, urdf_matrix))
                per_link.append(
                    {
                        'name': name,
                        'position_error_m': pos_error,
                        'rotation_error_rad': rot_error,
                        'usd_xyz': [float(v) for v in usd_matrix[:3, 3]],
                        'urdf_xyz': [float(v) for v in urdf_matrix[:3, 3]],
                        'urdf_prim_path': urdf_link_paths[name],
                    }
                )
            per_link.sort(key=lambda item: item['position_error_m'], reverse=True)
            summary.update(
                {
                    'comparison_mode': 'live_isaac_prim_transforms',
                    'comparison_link_count': len(per_link),
                    'max_position_error_m': max((item['position_error_m'] for item in per_link), default=0.0),
                    'mean_position_error_m': float(np.mean([item['position_error_m'] for item in per_link])) if per_link else 0.0,
                    'max_rotation_error_rad': max((item['rotation_error_rad'] for item in per_link), default=0.0),
                    'mean_rotation_error_rad': float(np.mean([item['rotation_error_rad'] for item in per_link])) if per_link else 0.0,
                    'worst_links': per_link[:15],
                }
            )
        else:
            summary.update(_offline_summary_payload(args.output_dir))

        save_json(args.output_dir / 'transform_comparison.json', summary)
        _log(f'[VAL] usd root: {usd_root}')
        _log(f'[VAL] urdf root: {urdf_root}')
        if args.live_prim_comparison:
            _log(f"[VAL] compared links: {summary['comparison_link_count']}")
            _log(f"[VAL] max position error: {summary['max_position_error_m']:.4f} m")
            _log(f"[VAL] mean position error: {summary['mean_position_error_m']:.4f} m")
            _log(f"[VAL] max rotation error: {summary['max_rotation_error_rad']:.4f} rad")
        else:
            _log(f"[VAL] comparison mode: {summary['comparison_mode']}")
            _log(f"[VAL] offline comparison: {summary.get('offline_transform_comparison_path')}")
        _log(f'[VAL] dome lights sanitized: {sanitized_domes}')
        _log(f"[VAL] wrote: {args.output_dir / 'transform_comparison.json'}")
        _write_checkpoint(args.output_dir, 'done')
        if args.stay_open:
            _log('[VAL] stay-open enabled. Close the Isaac Sim window to exit.')
            while app.is_running():
                app.update()
    finally:
        try:
            import omni.timeline

            omni.timeline.get_timeline_interface().stop()
        except Exception:
            pass
        app.close()


if __name__ == '__main__':
    main()
