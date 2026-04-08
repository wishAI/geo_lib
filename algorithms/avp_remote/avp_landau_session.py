from __future__ import annotations

import argparse
import sys
from pathlib import Path


MODULE_ROOT = Path(__file__).resolve().parent
MODULE_ROOT_STR = str(MODULE_ROOT)
if MODULE_ROOT_STR in sys.path:
    sys.path.remove(MODULE_ROOT_STR)
sys.path.insert(0, MODULE_ROOT_STR)


def _log_import(message: str) -> None:
    print(f"[AVP-IMPORT] {message}", flush=True)


def _parse_args(argv: list[str]):
    parser = argparse.ArgumentParser(
        description="Drive the copied Landau URDF from AVP tracking and mirror the solved pose onto the USD character.",
    )
    parser.add_argument(
        "--tracking-source",
        choices=("bridge", "snapshot"),
        default="bridge",
        help="Use live bridge tracking or a saved snapshot payload.",
    )
    parser.add_argument(
        "--snapshot-path",
        default=None,
        help="Snapshot file used for --tracking-source snapshot and calibration scaling.",
    )
    parser.add_argument(
        "--refresh-inputs",
        action="store_true",
        help="Re-copy URDF/USD/STL inputs from algorithms/usd_parallel_urdf before launch.",
    )
    parser.add_argument(
        "--show-urdf",
        action="store_true",
        help="Leave the imported URDF articulation visible instead of showing only the USD character. Requires --import-stage-urdf.",
    )
    parser.add_argument(
        "--import-stage-urdf",
        action="store_true",
        help="Attempt to import the URDF articulation into the live Isaac stage. Disabled by default because that importer is unstable here.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run Isaac Sim headless.",
    )
    parser.add_argument(
        "--baseline-urdf-path",
        "--h1-2-urdf-path",
        "--g1-urdf-path",
        dest="baseline_urdf_path",
        default=None,
        help="Optional third-baseline URDF. Defaults to Unitree H1_2 from the cloned xr_teleoperate helper repo.",
    )
    parser.add_argument(
        "--no-baseline",
        "--no-h1-2",
        "--no-g1",
        dest="no_baseline",
        action="store_true",
        help="Disable the third articulated baseline in the compare scene.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Optional frame limit for smoke tests. Zero means run until the app closes.",
    )
    return parser.parse_args(argv)


ARGS = _parse_args(sys.argv[1:])

_log_import("parsed args")
import numpy as np
_log_import("imported numpy")
import omni.kit.commands
_log_import("imported omni.kit.commands")
import omni.kit.app
_log_import("imported omni.kit.app")
from isaacsim.core.api import World
_log_import("imported isaacsim.core.api.World")
from isaacsim.core.prims import Articulation
_log_import("imported isaacsim.core.prims.Articulation")
from isaacsim.core.utils.stage import get_current_stage
_log_import("imported isaacsim.core.utils.stage")
from pxr import Sdf, UsdGeom, UsdLux, UsdPhysics
_log_import("imported pxr symbols")

from avp_marker_visualizer import HandMarkerSetVisualizer, HandStyle, MarkerStyle, MarkerVisualizer
_log_import("imported avp_marker_visualizer")
from avp_g1_pose import estimate_urdf_root_height, h1_2_hand_pose_summary, load_joint_limits, map_landau_pose_to_h1_2_pose
_log_import("imported avp_g1_pose")
from dex_hand_retargeting import DexHandRetargetingClient
_log_import("imported dex_hand_retargeting")
from asset_setup import prepare_landau_inputs
_log_import("imported asset_setup")
from avp_config import AVP_DEX_RETARGET_PYTHON, AVP_H1_2_URDF_PATH, AVP_SNAPSHOT_PATH
_log_import("imported avp_config")
from avp_tracking_schema import HAND_JOINT_NAMES
_log_import("imported avp_tracking_schema")
from avp_transform_utils import TransformOptions, build_xyz_transform, to_usd_world
_log_import("imported avp_transform_utils")
from landau_pose import LandauRawMeshPoseDriver, LandauUsdPoseDriver, set_visibility, world_map_from_pose
_log_import("imported landau_pose")
from landau_retarget import LandauUpperBodyRetargeter
_log_import("imported landau_retarget")
from tracking_source import TrackingSourceError, TrackingStream
_log_import("imported tracking_source")

APP = omni.kit.app.get_app()
_log_import("resolved omni app")


RAW_MARKER_OPTIONS = TransformOptions(
    column_major=True,
    pretransform=None,
    posttransform=build_xyz_transform(
        (0.0, 0.0, 180.0),
        (0.0, -0.13, 0.13),
        scale_xyz=(0.6, 0.6, 0.6),
    ),
)
RAW_VISUAL_OFFSET_XYZ = (-0.8, 0.0, 0.0)
SOLVED_VISUAL_OFFSET_XYZ = (0.8, 0.0, 0.0)
BASELINE_VISUAL_OFFSET_XYZ = (2.4, 0.0, 0.0)
# The H1_2 URDF faces across the compare row in importer-default orientation,
# so rotate it to face the same way as the solved USD character.
BASELINE_VISUAL_YAW_DEGREES = -90.0
BASELINE_LIVE_POSE_FILTER_ALPHA = 0.20
SOLVED_HAND_BASENAMES = (
    "hand",
    "thumb1",
    "thumb2",
    "thumb3",
    "index1_base",
    "index1",
    "index2",
    "index3",
    "middle1_base",
    "middle1",
    "middle2",
    "middle3",
    "ring1_base",
    "ring1",
    "ring2",
    "ring3",
    "pinky1_base",
    "pinky1",
    "pinky2",
    "pinky3",
    "forearm_stretch",
    "arm_stretch",
)


def _side_names(side: str, basenames: tuple[str, ...]) -> tuple[str, ...]:
    suffix = "l" if side == "left" else "r"
    return tuple(f"{basename}_{suffix}" for basename in basenames)


def _row_offset_stack(stack: np.ndarray | None, offset: np.ndarray) -> np.ndarray | None:
    if stack is None:
        return None
    return np.stack([np.asarray(mat, dtype=float) @ offset for mat in stack], axis=0)


def _tracking_stack_to_usd_row(stack) -> np.ndarray | None:
    if stack is None:
        return None
    return np.stack(
        [np.asarray(to_usd_world(mat, options=RAW_MARKER_OPTIONS), dtype=float) for mat in stack],
        axis=0,
    )


def _solved_hand_stack(records, pose_by_name: dict[str, float], side: str) -> np.ndarray:
    world_map = world_map_from_pose(records, pose_by_name)
    names = _side_names(side, SOLVED_HAND_BASENAMES)
    return np.stack([np.asarray(world_map[name], dtype=float).T for name in names], axis=0)


def _row_offset_matrix(translate_xyz) -> np.ndarray:
    return build_xyz_transform((0.0, 0.0, 0.0), tuple(float(value) for value in translate_xyz))


def _log(message: str) -> None:
    print(f"[AVP] {message}", flush=True)


def _add_scene_lighting(stage) -> None:
    key_light = UsdLux.DistantLight.Define(stage, Sdf.Path("/World/KeyLight"))
    key_light.CreateIntensityAttr(2000.0)
    fill_light = UsdLux.DomeLight.Define(stage, Sdf.Path("/World/FillLight"))
    fill_light.CreateIntensityAttr(200.0)


def _ensure_physics_scene(stage) -> None:
    scene_path = Sdf.Path("/World/physicsScene")
    if not stage.GetPrimAtPath(scene_path).IsValid():
        scene = UsdPhysics.Scene.Define(stage, scene_path)
        scene.CreateGravityDirectionAttr().Set((0.0, 0.0, -1.0))
        scene.CreateGravityMagnitudeAttr().Set(9.81)


def _add_local_ground(stage) -> None:
    ground_path = Sdf.Path("/World/Ground")
    if stage.GetPrimAtPath(ground_path).IsValid():
        return

    ground = UsdGeom.Cube.Define(stage, ground_path)
    ground.CreateSizeAttr(1.0)
    xform_api = UsdGeom.XformCommonAPI(ground)
    xform_api.SetScale((10.0, 10.0, 0.02))
    xform_api.SetTranslate((0.0, 0.0, -0.02))
    ground.CreateDisplayColorAttr([(0.18, 0.18, 0.18)])
    UsdPhysics.CollisionAPI.Apply(ground.GetPrim())


def _robot_root_path(articulation_root_path: str) -> str:
    return str(Sdf.Path(articulation_root_path).GetParentPath())


def _set_root_transform(stage, prim_path: str, xyz, yaw_degrees: float = 0.0) -> None:
    from pxr import Gf

    prim = stage.GetPrimAtPath(prim_path)
    xformable = UsdGeom.Xformable(prim)
    matrix = Gf.Matrix4d(1.0)
    if abs(float(yaw_degrees)) > 1.0e-6:
        matrix.SetRotateOnly(Gf.Rotation(Gf.Vec3d(0.0, 0.0, 1.0), float(yaw_degrees)))
    matrix.SetTranslateOnly(Gf.Vec3d(float(xyz[0]), float(xyz[1]), float(xyz[2])))
    xformable.ClearXformOpOrder()
    transform_op = xformable.AddTransformOp(opSuffix="livePose", precision=UsdGeom.XformOp.PrecisionDouble)
    transform_op.Set(matrix)


def _position_imported_robot(stage, articulation_root_path: str, xyz, yaw_degrees: float = 0.0) -> None:
    _set_root_transform(stage, _robot_root_path(articulation_root_path), xyz, yaw_degrees=yaw_degrees)


def _pump_app_startup(update_count: int = 3) -> None:
    if update_count <= 0:
        return
    _log(f"Pumping {update_count} startup app update(s)")
    for _ in range(update_count):
        APP.update()


def _ensure_articulation_initialized(articulation, label: str, update_attempts: int = 8) -> None:
    last_error = None
    for attempt in range(max(update_attempts, 1)):
        if articulation.is_physics_handle_valid():
            return
        if attempt == 0:
            _log(f"{label} articulation was not ready after reset; initializing explicitly")
        try:
            articulation.initialize()
        except Exception as exc:  # pragma: no cover - requires Isaac runtime
            last_error = exc
        APP.update()
    if articulation.is_physics_handle_valid():
        return
    if last_error is not None:
        raise RuntimeError(f"{label} articulation failed to initialize") from last_error
    raise RuntimeError(f"{label} articulation failed to initialize")


def _disable_articulation_self_collisions(articulation, label: str) -> None:
    articulation.set_enabled_self_collisions(np.asarray([False], dtype=bool))
    _log(f"{label} self-collision disabled")


def _disable_articulation_gravity(articulation, label: str) -> None:
    body_count = int(getattr(articulation, "num_bodies", 0) or 0)
    if body_count <= 0:
        return
    articulation.set_body_disable_gravity(np.ones((1, body_count), dtype=bool))
    _log(f"{label} gravity disabled for {body_count} bodies")


def _blend_pose(
    previous_pose: dict[str, float] | None,
    target_pose: dict[str, float],
    *,
    alpha: float,
) -> dict[str, float]:
    if previous_pose is None or alpha >= 1.0:
        return {joint_name: float(value) for joint_name, value in target_pose.items()}
    alpha = float(np.clip(alpha, 0.0, 1.0))
    blended = {}
    for joint_name, target_value in target_pose.items():
        prev_value = float(previous_pose.get(joint_name, target_value))
        blended[joint_name] = prev_value + alpha * (float(target_value) - prev_value)
    return blended


def _apply_articulation_pose(articulation, dof_names: list[str], pose_by_name: dict[str, float]) -> None:
    if not dof_names:
        return
    joint_positions = np.asarray(
        [[float(pose_by_name.get(joint_name, 0.0)) for joint_name in dof_names]],
        dtype=float,
    )
    articulation.set_joint_positions(joint_positions)
    articulation.set_joint_velocities(np.zeros_like(joint_positions))


def _set_import_config_value(import_config, attr_name: str, setter_name: str, value) -> None:
    setter = getattr(import_config, setter_name, None)
    if callable(setter):
        setter(value)
        return
    if hasattr(import_config, attr_name):
        setattr(import_config, attr_name, value)
        return
    raise AttributeError(f"URDF import config does not support {attr_name}/{setter_name}")


def _configure_urdf_import_config(import_config) -> None:
    _set_import_config_value(import_config, "merge_fixed_joints", "set_merge_fixed_joints", False)
    _set_import_config_value(import_config, "convex_decomp", "set_convex_decomp", False)
    _set_import_config_value(import_config, "import_inertia_tensor", "set_import_inertia_tensor", True)
    _set_import_config_value(import_config, "fix_base", "set_fix_base", True)
    _set_import_config_value(import_config, "self_collision", "set_self_collision", False)
    _set_import_config_value(import_config, "distance_scale", "set_distance_scale", 1.0)
    if hasattr(import_config, "set_create_physics_scene"):
        import_config.set_create_physics_scene(False)


def _try_create_import_config(update_attempts: int = 24):
    last_error = None
    for _ in range(max(update_attempts, 1)):
        try:
            status, import_config = omni.kit.commands.execute("URDFCreateImportConfig")
        except Exception as exc:  # pragma: no cover - requires Isaac runtime
            status = False
            import_config = None
            last_error = exc
        if status:
            return import_config
        APP.update()
    if last_error is not None:
        raise RuntimeError("Failed to create URDF import config") from last_error
    raise RuntimeError("Failed to create URDF import config")


def _create_standard_urdf_import_config():
    from isaacsim.core.utils.extensions import enable_extension

    extension_name = "isaacsim.asset.importer.urdf"
    extension_manager = APP.get_extension_manager()
    if not extension_manager.is_extension_enabled(extension_name):
        _log(f"Enabling {extension_name}")
        enable_extension(extension_name)
    import_config = _try_create_import_config()
    _log("Created URDF import config via Isaac Sim extension")
    return import_config


def _create_headless_urdf_import_config():
    _log("Loading headless URDF shim")
    from headless_urdf_commands import register_urdf_commands

    _log("Registering headless URDF commands")
    register_urdf_commands()
    import_config = _try_create_import_config(update_attempts=4)
    _log("Created URDF import config via headless shim")
    return import_config


def _import_urdf(urdf_path: str, *, allow_shim_fallback: bool) -> str:
    try:
        import_config = _create_standard_urdf_import_config()
    except Exception as exc:
        if not allow_shim_fallback:
            raise RuntimeError("Isaac Sim URDF importer extension is unavailable") from exc
        _log(f"Standard URDF importer unavailable, falling back to headless shim: {exc}")
        import_config = _create_headless_urdf_import_config()

    _configure_urdf_import_config(import_config)

    _log("Executing URDFParseAndImportFile")
    status, prim_path = omni.kit.commands.execute(
        "URDFParseAndImportFile",
        urdf_path=urdf_path,
        import_config=import_config,
        get_articulation_root=True,
    )
    _log(f"URDFParseAndImportFile returned status={status} prim_path={prim_path!r}")
    if not status:
        raise RuntimeError(f"Failed to import URDF: {urdf_path}")
    return prim_path


def main() -> None:
    snapshot_path = ARGS.snapshot_path or str(AVP_SNAPSHOT_PATH)
    _log(f"Preparing inputs from snapshot source {snapshot_path}")
    prepared = prepare_landau_inputs(refresh=ARGS.refresh_inputs)

    _log("Creating Isaac world")
    world = World(stage_units_in_meters=1.0)
    _log("Isaac world created")
    if not ARGS.headless:
        from isaacsim.core.utils.viewports import set_camera_view

        set_camera_view(
            eye=[4.2, -4.2, 1.55],
            target=[0.8, 0.0, 0.85],
            camera_prim_path="/OmniverseKit_Persp",
        )

    stage = get_current_stage()
    _ensure_physics_scene(stage)
    _add_local_ground(stage)
    _add_scene_lighting(stage)
    _log("Stage lighting and local ground ready")
    raw_driver = LandauRawMeshPoseDriver(
        stage,
        urdf_path=prepared.urdf_path,
        skeleton_json_path=prepared.skeleton_json_path,
        visual_root_path="/World/Compare/RawVisual",
        root_offset_xyz=RAW_VISUAL_OFFSET_XYZ,
    )
    usd_driver = LandauUsdPoseDriver(
        stage,
        usd_path=prepared.usd_path,
        skeleton_json_path=prepared.skeleton_json_path,
        visual_root_path="/World/Compare/SolvedVisual",
        visual_asset_path="/World/Compare/SolvedVisual/Asset",
        root_offset_xyz=SOLVED_VISUAL_OFFSET_XYZ,
        anim_name="SolvedPoseAnim",
    )
    _log(f"Raw URDF and USD compare drivers ready for {prepared.usd_path}")
    baseline_urdf_path = (
        Path(ARGS.baseline_urdf_path).expanduser().resolve()
        if ARGS.baseline_urdf_path
        else AVP_H1_2_URDF_PATH.resolve()
    )
    baseline_enabled = (not ARGS.no_baseline) and baseline_urdf_path.exists()
    baseline_joint_limits = load_joint_limits(baseline_urdf_path) if baseline_enabled else None
    baseline_root_xyz = np.array(
        (
            float(BASELINE_VISUAL_OFFSET_XYZ[0]),
            float(BASELINE_VISUAL_OFFSET_XYZ[1]),
            estimate_urdf_root_height(baseline_urdf_path) if baseline_enabled else 0.0,
        ),
        dtype=float,
    )
    dex_hand_client = None
    if AVP_DEX_RETARGET_PYTHON.exists():
        try:
            dex_hand_client = DexHandRetargetingClient(
                helper_python=AVP_DEX_RETARGET_PYTHON,
                landau_urdf_path=prepared.urdf_path,
                snapshot_path=Path(snapshot_path),
                baseline_urdf_path=baseline_urdf_path if baseline_urdf_path.exists() else None,
            )
            _log(f"Dex hand retargeting helper ready via {AVP_DEX_RETARGET_PYTHON}")
        except Exception as exc:
            _log(f"Dex hand retargeting helper unavailable, continuing with heuristic hands: {exc}")
    else:
        _log(f"Dex hand retargeting helper was not found: {AVP_DEX_RETARGET_PYTHON}")
    retargeter = LandauUpperBodyRetargeter(
        urdf_path=prepared.urdf_path,
        skeleton_json_path=prepared.skeleton_json_path,
        snapshot_path=snapshot_path,
        hand_retargeting_client=dex_hand_client,
    )
    _log("Retargeter initialized")
    raw_hand_style = HandStyle(
        wrist=MarkerStyle(mode="sphere", color=(0.1, 0.85, 1.0), radius=0.012),
        joint=MarkerStyle(mode="sphere", color=(0.1, 0.85, 1.0), radius=0.008),
    )
    solved_hand_style = HandStyle(
        wrist=MarkerStyle(mode="sphere", color=(1.0, 0.55, 0.15), radius=0.012),
        joint=MarkerStyle(mode="sphere", color=(1.0, 0.55, 0.15), radius=0.008),
    )
    raw_head_marker = MarkerVisualizer(
        stage,
        "/World/Compare/Raw/Head",
        style=MarkerStyle(mode="sphere", color=(1.0, 1.0, 0.2), radius=0.03),
        label="raw_head",
        print_debug=False,
    )
    raw_left_markers = HandMarkerSetVisualizer.for_hand(
        stage,
        "/World/Compare/Raw/LeftHand",
        HAND_JOINT_NAMES,
        style=raw_hand_style,
        label="raw_left_hand",
        print_debug=False,
    )
    raw_right_markers = HandMarkerSetVisualizer.for_hand(
        stage,
        "/World/Compare/Raw/RightHand",
        HAND_JOINT_NAMES,
        style=raw_hand_style,
        label="raw_right_hand",
        print_debug=False,
    )
    solved_left_markers = HandMarkerSetVisualizer.for_hand(
        stage,
        "/World/Compare/Solved/LeftHand",
        _side_names("left", SOLVED_HAND_BASENAMES),
        style=solved_hand_style,
        label="solved_left_hand",
        print_debug=False,
    )
    solved_right_markers = HandMarkerSetVisualizer.for_hand(
        stage,
        "/World/Compare/Solved/RightHand",
        _side_names("right", SOLVED_HAND_BASENAMES),
        style=solved_hand_style,
        label="solved_right_hand",
        print_debug=False,
    )
    raw_offset = _row_offset_matrix(raw_driver.root_translate_xyz)
    solved_offset = _row_offset_matrix(usd_driver.root_translate_xyz)

    use_stage_urdf = ARGS.import_stage_urdf
    urdf_prim_path: str | None = None
    if use_stage_urdf:
        _log(f"Importing URDF {prepared.urdf_path}")
        urdf_prim_path = _import_urdf(str(prepared.urdf_path), allow_shim_fallback=ARGS.headless)
        _log(f"URDF imported at {urdf_prim_path}")
    else:
        _log("Skipping stage URDF import by default and using USD-only visualization")

    baseline_prim_path: str | None = None
    if baseline_enabled:
        try:
            _log(f"Importing H1_2 URDF {baseline_urdf_path}")
            baseline_prim_path = _import_urdf(str(baseline_urdf_path), allow_shim_fallback=ARGS.headless)
            _position_imported_robot(stage, baseline_prim_path, baseline_root_xyz, yaw_degrees=BASELINE_VISUAL_YAW_DEGREES)
            _log(f"H1_2 URDF imported at {baseline_prim_path}")
        except Exception as exc:
            baseline_enabled = False
            baseline_joint_limits = None
            baseline_prim_path = None
            _log(f"H1_2 import failed, continuing without the third baseline: {exc}")
    elif ARGS.no_baseline:
        _log("Skipping the third baseline because --no-baseline was requested")
    else:
        _log(f"Skipping the third baseline because the URDF was not found: {baseline_urdf_path}")

    robot = None
    if urdf_prim_path is not None:
        robot = world.scene.add(Articulation(urdf_prim_path, name="stage_urdf_baseline"))

    baseline_robot = None
    if baseline_prim_path is not None:
        baseline_robot = world.scene.add(Articulation(baseline_prim_path, name="h1_2_baseline"))

    world.reset()
    _log("World reset complete")

    robot_dof_names: list[str] = []
    current_pose: dict[str, float] = {}
    if robot is not None:
        _ensure_articulation_initialized(robot, "Stage URDF")
        robot_dof_names = list(robot.dof_names or [])
        current_pose = {joint_name: 0.0 for joint_name in robot_dof_names}
        _disable_articulation_gravity(robot, "Stage URDF")

        if not ARGS.show_urdf:
            set_visibility(stage, urdf_prim_path, False)
    elif ARGS.show_urdf:
        _log("--show-urdf was requested, but no stage URDF was imported. Pass --import-stage-urdf to enable it.")

    baseline_dof_names: list[str] = []
    if baseline_robot is not None:
        _ensure_articulation_initialized(baseline_robot, "H1_2")
        baseline_dof_names = list(baseline_robot.dof_names or [])
        baseline_joint_names = list(baseline_robot.joint_names or [])
        _disable_articulation_self_collisions(baseline_robot, "H1_2")
        _disable_articulation_gravity(baseline_robot, "H1_2")
        _position_imported_robot(stage, baseline_prim_path, baseline_root_xyz, yaw_degrees=BASELINE_VISUAL_YAW_DEGREES)
        _log(f"H1_2 articulation initialized with {len(baseline_dof_names)} dofs and {len(baseline_joint_names)} joints")

    tracking_stream = TrackingStream(
        tracking_source=ARGS.tracking_source,
        snapshot_path=snapshot_path,
    )
    _log(f"Tracking stream ready: {ARGS.tracking_source}")
    _pump_app_startup()

    frame_count = 0
    have_landau_pose = False
    baseline_command_pose: dict[str, float] | None = None
    snapshot_pose_applied = False
    _log(f"Entering main update loop with app_running={APP.is_running()}")
    while True:
        first_iteration = frame_count == 0
        if first_iteration:
            _log(f"Starting first main-loop iteration with app_running={APP.is_running()}")
        frame = None if (ARGS.tracking_source == "snapshot" and snapshot_pose_applied) else tracking_stream.get_tracking_frame()
        if frame is not None:
            if first_iteration:
                _log("Acquired first tracking frame")
            raw_left_markers.update(_row_offset_stack(_tracking_stack_to_usd_row(frame.get("left_arm")), raw_offset))
            raw_right_markers.update(_row_offset_stack(_tracking_stack_to_usd_row(frame.get("right_arm")), raw_offset))
            raw_head = frame.get("head")
            if raw_head is not None:
                raw_head_marker.update(np.asarray(to_usd_world(raw_head, options=RAW_MARKER_OPTIONS), dtype=float) @ raw_offset, now=0.0)
            pose_by_name = retargeter.retarget_frame(frame)
            if first_iteration:
                _log("Retargeted first tracking frame onto Landau pose")
            current_pose.update(pose_by_name)
            have_landau_pose = True
            baseline_pose = None
            if baseline_robot is not None:
                baseline_pose = map_landau_pose_to_h1_2_pose(
                    current_pose,
                    hand_pose_override=retargeter.h1_2_hand_pose_overrides(),
                    joint_limits=baseline_joint_limits,
                )
                baseline_command_pose = _blend_pose(
                    baseline_command_pose,
                    baseline_pose,
                    alpha=1.0 if ARGS.tracking_source == "snapshot" else BASELINE_LIVE_POSE_FILTER_ALPHA,
                )
                if first_iteration:
                    _log("Mapped first Landau pose onto H1_2")
            if frame_count == 0:
                if baseline_pose is not None:
                    _log(f"H1_2 hand left summary: {h1_2_hand_pose_summary(baseline_pose, 'left')}")
                    _log(f"H1_2 hand right summary: {h1_2_hand_pose_summary(baseline_pose, 'right')}")
                if ARGS.tracking_source == "snapshot" and ARGS.max_frames == 0:
                    _log("Snapshot mode keeps the current pose applied and leaves Isaac Sim running until you close the window.")
            if ARGS.tracking_source == "snapshot":
                snapshot_pose_applied = True
        elif first_iteration:
            _log("No tracking frame was available on the first main-loop iteration")

        if have_landau_pose:
            if robot is not None:
                _apply_articulation_pose(robot, robot_dof_names, current_pose)
                if first_iteration:
                    _log(f"Applied first pose to imported stage URDF with {len(robot_dof_names)} dofs")
            raw_driver.apply_pose(current_pose)
            usd_driver.apply_pose(current_pose)
            solved_left_markers.update(_row_offset_stack(_solved_hand_stack(retargeter.records, current_pose, "left"), solved_offset))
            solved_right_markers.update(_row_offset_stack(_solved_hand_stack(retargeter.records, current_pose, "right"), solved_offset))

        if baseline_robot is not None and baseline_command_pose is not None:
            _apply_articulation_pose(baseline_robot, baseline_dof_names, baseline_command_pose)
            if first_iteration:
                _log(f"Applied first pose to H1_2 with {len(baseline_dof_names)} dofs")

        if frame_count == 0 and have_landau_pose:
            _log("Applied first tracking frame")

        if (robot is not None or baseline_robot is not None) and not (ARGS.tracking_source == "snapshot" and snapshot_pose_applied):
            world.step(render=not ARGS.headless)
        else:
            APP.update()
        if first_iteration:
            _log(f"Completed first frame update with app_running={APP.is_running()}")
        frame_count += 1
        if ARGS.max_frames > 0 and frame_count >= ARGS.max_frames:
            _log(f"Reached frame limit {ARGS.max_frames}")
            break
        if not APP.is_running():
            if snapshot_pose_applied:
                _log(f"Isaac app stopped after {frame_count} frame(s)")
            else:
                _log("Isaac app stopped before the first tracking frame was applied")
            break

    if ARGS.max_frames > 0:
        _log("Posting app quit")
        APP.post_quit()
    if dex_hand_client is not None:
        dex_hand_client.close()
    _log("Session main exiting")


if __name__ == "__main__":
    try:
        main()
    except TrackingSourceError as exc:
        print(f"[AVP] {exc}")
        APP.post_quit()
        raise SystemExit(1) from exc
