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
from asset_setup import prepare_landau_inputs
_log_import("imported asset_setup")
from avp_config import AVP_SNAPSHOT_PATH
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


def _import_landau_urdf(urdf_path: str) -> str:
    _log("Loading headless URDF shim")
    from headless_urdf_commands import register_urdf_commands

    _log("Registering headless URDF commands")
    register_urdf_commands()
    _log("Creating URDF import config")
    status, import_config = omni.kit.commands.execute("URDFCreateImportConfig")
    if not status:
        raise RuntimeError("Failed to create URDF import config")
    _log("Created URDF import config")

    import_config.merge_fixed_joints = False
    import_config.convex_decomp = False
    import_config.import_inertia_tensor = True
    import_config.fix_base = True
    import_config.distance_scale = 1.0

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
            eye=[2.8, -3.0, 1.45],
            target=[0.0, 0.0, 0.85],
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
    retargeter = LandauUpperBodyRetargeter(
        urdf_path=prepared.urdf_path,
        skeleton_json_path=prepared.skeleton_json_path,
        snapshot_path=snapshot_path,
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
        urdf_prim_path = _import_landau_urdf(str(prepared.urdf_path))
        _log(f"URDF imported at {urdf_prim_path}")
    else:
        _log("Skipping stage URDF import by default and using USD-only visualization")

    world.reset()
    _log("World reset complete")

    robot = None
    joint_names: list[str] = []
    current_pose: dict[str, float] = {}
    if urdf_prim_path is not None:
        robot = Articulation(urdf_prim_path)
        robot.initialize()
        joint_names = list(robot.joint_names)
        current_pose = {joint_name: 0.0 for joint_name in joint_names}

        if not ARGS.show_urdf:
            set_visibility(stage, urdf_prim_path, False)
    elif ARGS.show_urdf:
        _log("--show-urdf was requested, but no stage URDF was imported. Pass --import-stage-urdf to enable it.")

    tracking_stream = TrackingStream(
        tracking_source=ARGS.tracking_source,
        snapshot_path=snapshot_path,
    )
    _log(f"Tracking stream ready: {ARGS.tracking_source}")

    frame_count = 0
    snapshot_pose_applied = False
    while APP.is_running():
        frame = None if (ARGS.tracking_source == "snapshot" and snapshot_pose_applied) else tracking_stream.get_tracking_frame()
        if frame is not None:
            raw_left_markers.update(_row_offset_stack(_tracking_stack_to_usd_row(frame.get("left_arm")), raw_offset))
            raw_right_markers.update(_row_offset_stack(_tracking_stack_to_usd_row(frame.get("right_arm")), raw_offset))
            raw_head = frame.get("head")
            if raw_head is not None:
                raw_head_marker.update(np.asarray(to_usd_world(raw_head, options=RAW_MARKER_OPTIONS), dtype=float) @ raw_offset, now=0.0)
            pose_by_name = retargeter.retarget_frame(frame)
            current_pose.update(pose_by_name)
            if robot is not None:
                joint_positions = np.asarray(
                    [[float(current_pose.get(joint_name, 0.0)) for joint_name in joint_names]],
                    dtype=float,
                )
                robot.set_joint_positions(joint_positions)
            raw_driver.apply_pose(current_pose)
            usd_driver.apply_pose(current_pose)
            solved_left_markers.update(_row_offset_stack(_solved_hand_stack(retargeter.records, current_pose, "left"), solved_offset))
            solved_right_markers.update(_row_offset_stack(_solved_hand_stack(retargeter.records, current_pose, "right"), solved_offset))
            if frame_count == 0:
                _log("Applied first tracking frame")
                if ARGS.tracking_source == "snapshot" and ARGS.max_frames == 0:
                    _log("Snapshot mode keeps the current pose applied and leaves Isaac Sim running until you close the window.")
            if ARGS.tracking_source == "snapshot":
                snapshot_pose_applied = True

        if robot is not None:
            world.step(render=not ARGS.headless)
        else:
            APP.update()
        frame_count += 1
        if ARGS.max_frames > 0 and frame_count >= ARGS.max_frames:
            _log(f"Reached frame limit {ARGS.max_frames}")
            break

    if ARGS.max_frames > 0:
        _log("Posting app quit")
        APP.post_quit()


if __name__ == "__main__":
    try:
        main()
    except TrackingSourceError as exc:
        print(f"[AVP] {exc}")
        APP.post_quit()
        raise SystemExit(1) from exc
