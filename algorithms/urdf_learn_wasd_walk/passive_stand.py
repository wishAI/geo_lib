"""Run the exact clean-lineage passive 30-second standing validator in Isaac Lab."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from algorithms.urdf_learn_wasd_walk import model_spec


MILESTONE_ID = "stand_zero_signal_30s_no_reset"
OUTPUT_ROOT = model_spec.ALGORITHM_ROOT / "outputs"
DEFAULT_OUTPUT_DIR = OUTPUT_ROOT / MILESTONE_ID
MIN_GATE_DURATION_S = 30.0
SIM_DT = 0.002
CONTROL_DT = 0.02
MAX_REFERENCE_TILT_RAD = math.radians(30.0)
MAX_ROOT_HEIGHT_DROP_M = 0.08
MAX_HORIZONTAL_DRIFT_M = 0.12


def safe_output_dir(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    try:
        resolved.relative_to(OUTPUT_ROOT.resolve())
    except ValueError as error:
        raise ValueError(f"Output directory must stay below {OUTPUT_ROOT}") from error
    return resolved


def quaternion_distance_wxyz(left: Sequence[float], right: Sequence[float]) -> float:
    dot = abs(sum(float(a) * float(b) for a, b in zip(left, right)))
    return 2.0 * math.acos(min(1.0, max(-1.0, dot)))


def evaluate_gate(metrics: dict, *, required_duration_s: float = MIN_GATE_DURATION_S) -> tuple[bool, list[str]]:
    failures = []
    if metrics["duration_s"] + 1.0e-6 < required_duration_s:
        failures.append(f"duration {metrics['duration_s']:.3f}s is shorter than {required_duration_s:.3f}s")
    for name in ("reset_count", "done_count", "fall_count"):
        if metrics[name] != 0:
            failures.append(f"{name}={metrics[name]}")
    if metrics["max_reference_tilt_rad"] > MAX_REFERENCE_TILT_RAD:
        failures.append(f"reference tilt {metrics['max_reference_tilt_rad']:.4f}rad exceeded limit")
    if metrics["root_height_drop_m"] > MAX_ROOT_HEIGHT_DROP_M:
        failures.append(f"root height drop {metrics['root_height_drop_m']:.4f}m exceeded limit")
    if metrics["horizontal_drift_m"] > MAX_HORIZONTAL_DRIFT_M:
        failures.append(f"horizontal drift {metrics['horizontal_drift_m']:.4f}m exceeded limit")
    if metrics["max_abs_action"] != 0.0 or metrics["max_abs_command"] != 0.0:
        failures.append("passive gate received a non-zero action or command")
    return not failures, failures


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _encode_video(frame_dir: Path, video_path: Path, fps: int) -> None:
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-framerate",
        str(fps),
        "-i",
        os.fspath(frame_dir / "frame_%04d.png"),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        os.fspath(video_path),
    ]
    subprocess.run(command, check=True)


def _probe_video(video_path: Path) -> dict:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,avg_frame_rate,nb_frames:format=duration",
        "-of",
        "json",
        os.fspath(video_path),
    ]
    return json.loads(subprocess.run(command, check=True, capture_output=True, text=True).stdout)


def _frame_visibility(rgb, segmentation) -> dict:
    import numpy as np

    rgb_array = np.asarray(rgb)
    segmentation_array = np.asarray(segmentation).squeeze()
    ids, counts = np.unique(segmentation_array, return_counts=True)
    nonzero_ids = [int(value) for value in ids if int(value) != 0]
    mask = segmentation_array != 0
    if mask.any():
        rows, columns = np.nonzero(mask)
        bbox = [int(columns.min()), int(rows.min()), int(columns.max()), int(rows.max())]
        bbox_width = bbox[2] - bbox[0] + 1
        bbox_height = bbox[3] - bbox[1] + 1
        margin = min(bbox[0], bbox[1], rgb_array.shape[1] - 1 - bbox[2], rgb_array.shape[0] - 1 - bbox[3])
    else:
        bbox = None
        bbox_width = bbox_height = margin = 0
    return {
        "nonblank": bool(float(rgb_array.std()) > 2.0),
        "rgb_stddev": round(float(rgb_array.std()), 5),
        "resolution": [int(rgb_array.shape[1]), int(rgb_array.shape[0])],
        "semantic_ids": nonzero_ids,
        "semantic_counts": {str(int(key)): int(value) for key, value in zip(ids, counts)},
        "character_bbox_xyxy": bbox,
        "character_bbox_width_px": bbox_width,
        "character_bbox_height_px": bbox_height,
        "minimum_frame_margin_px": margin,
        "character_visible": bool(bbox is not None and bbox_width >= 40 and bbox_height >= 80 and margin >= 4),
    }


def _build_scene_cfg(args):
    import isaaclab.sim as sim_utils
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import ArticulationCfg, AssetBaseCfg
    from isaaclab.scene import InteractiveSceneCfg
    from isaaclab.sensors import CameraCfg, ContactSensorCfg
    from isaaclab.terrains import TerrainImporterCfg
    from isaaclab.utils import configclass

    spec = model_spec.build_robot_spec()
    locked_joints = spec["locked_joints"]

    @configclass
    class PassiveStandSceneCfg(InteractiveSceneCfg):
        terrain = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="plane",
            collision_group=-1,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.0,
                dynamic_friction=0.9,
                restitution=0.0,
            ),
            debug_vis=False,
        )
        robot = ArticulationCfg(
            prim_path="{ENV_REGEX_NS}/Robot",
            spawn=sim_utils.UrdfFileCfg(
                asset_path=os.fspath(model_spec.URDF_PATH),
                usd_dir=os.fspath(OUTPUT_ROOT / "usd_cache"),
                usd_file_name="landau_v10_parallel_mesh.usd",
                force_usd_conversion=True,
                fix_base=False,
                merge_fixed_joints=False,
                make_instanceable=False,
                self_collision=False,
                activate_contact_sensors=True,
                semantic_tags=[("class", "landau")],
                collider_type="convex_hull",
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=False,
                    retain_accelerations=False,
                    linear_damping=0.0,
                    angular_damping=0.0,
                    max_linear_velocity=20.0,
                    max_angular_velocity=50.0,
                    max_depenetration_velocity=1.0,
                ),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    enabled_self_collisions=False,
                    solver_position_iteration_count=8,
                    solver_velocity_iteration_count=4,
                    sleep_threshold=0.0,
                    stabilization_threshold=0.0001,
                ),
                collision_props=sim_utils.CollisionPropertiesCfg(
                    collision_enabled=True,
                    contact_offset=0.002,
                    rest_offset=0.0,
                ),
                joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                    drive_type="force",
                    target_type="position",
                    gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0),
                ),
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=tuple(spec["nominal_pose"]["base_position_m"]),
                rot=tuple(spec["nominal_pose"]["base_orientation_wxyz"]),
                joint_pos={".*": 0.0},
                joint_vel={".*": 0.0},
            ),
            soft_joint_pos_limit_factor=0.95,
            actuators={
                **{
                    name: ImplicitActuatorCfg(
                        joint_names_expr=list(group["joints"]),
                        effort_limit=50.0,
                        velocity_limit=4.0,
                        stiffness=group["stiffness"],
                        damping=group["damping"],
                    )
                    for name, group in model_spec.PD_GROUPS.items()
                },
                "locked": ImplicitActuatorCfg(
                    joint_names_expr=locked_joints,
                    effort_limit=50.0,
                    velocity_limit=4.0,
                    stiffness=model_spec.LOCKED_PD_STIFFNESS,
                    damping=model_spec.LOCKED_PD_DAMPING,
                ),
            },
        )
        contact_forces = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/.*",
            update_period=SIM_DT,
            history_length=3,
            track_air_time=True,
        )
        camera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/EvidenceCamera",
            update_period=1.0 / args.video_fps,
            height=args.video_height,
            width=args.video_width,
            data_types=["rgb", "semantic_segmentation"],
            semantic_filter="class:landau",
            colorize_semantic_segmentation=False,
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=35.0,
                focus_distance=3.0,
                horizontal_aperture=20.955,
                clipping_range=(0.05, 20.0),
            ),
        )
        light = AssetBaseCfg(
            prim_path="/World/Sun",
            spawn=sim_utils.DistantLightCfg(intensity=3000.0, color=(0.92, 0.92, 0.92)),
        )

    scene_cfg = PassiveStandSceneCfg(num_envs=1, env_spacing=2.0, lazy_sensor_update=True)
    if not args.record_video:
        scene_cfg.camera = None
    return scene_cfg


def run(args, simulation_app) -> dict:
    import torch
    from PIL import Image, ImageDraw

    import isaaclab.sim as sim_utils
    from isaaclab.scene import InteractiveScene

    output_dir = safe_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sim_cfg = sim_utils.SimulationCfg(
        dt=SIM_DT,
        render_interval=max(1, round((1.0 / args.video_fps) / SIM_DT)),
        device=args.device,
    )
    sim_cfg.physx.gpu_max_rigid_patch_count = 2**18
    sim = sim_utils.SimulationContext(sim_cfg)
    scene = InteractiveScene(_build_scene_cfg(args))
    sim.reset()

    robot = scene["robot"]
    contact_sensor = scene["contact_forces"]
    expected_joint_names = set(model_spec.build_robot_spec()["nominal_pose"]["joint_positions_rad"])
    if set(robot.joint_names) != expected_joint_names:
        raise RuntimeError(
            f"Imported joints differ from audited URDF: missing={sorted(expected_joint_names-set(robot.joint_names))}, "
            f"unexpected={sorted(set(robot.joint_names)-expected_joint_names)}"
        )
    reference_ids, reference_names = robot.find_bodies("root_x", preserve_order=True)
    if reference_names != ["root_x"]:
        raise RuntimeError(f"Expected one root_x body, got {reference_names}")
    foot_ids, foot_names = contact_sensor.find_bodies(["foot_l", "foot_r", "toes_01_l", "toes_01_r"], preserve_order=True)

    joint_targets = robot.data.default_joint_pos.clone()
    robot.write_joint_state_to_sim(joint_targets, torch.zeros_like(joint_targets))
    robot.set_joint_position_target(joint_targets)
    scene.write_data_to_sim()

    camera = scene["camera"] if args.record_video else None
    if camera is not None:
        initial_target = torch.tensor([[0.0, 0.0, 0.48]], device=sim.device)
        initial_eye = torch.tensor([[1.8, -2.2, 1.25]], device=sim.device)
        camera.set_world_poses_from_view(initial_eye, initial_target)

    physics_steps = args.steps if args.steps is not None else math.ceil(args.duration / SIM_DT)
    trace_stride = max(1, round(CONTROL_DT / SIM_DT))
    video_stride = max(1, round((1.0 / args.video_fps) / SIM_DT))
    traces = []
    visibility = []
    fall_count = 0
    previously_fallen = False
    reference_position_0 = None
    reference_quaternion_0 = None
    max_joint_error = 0.0
    video_frame_count = 0

    temporary_frames = tempfile.TemporaryDirectory(prefix="landau_frames_", dir=output_dir) if camera is not None else None
    frame_dir = Path(temporary_frames.name) if temporary_frames is not None else None
    try:
        for step in range(physics_steps):
            robot.set_joint_position_target(joint_targets)
            scene.write_data_to_sim()
            capture_frame = camera is not None and (step + 1) % video_stride == 0
            if capture_frame:
                current_root_xy = robot.data.body_pos_w[0, reference_ids[0], :2]
                eye = torch.tensor([[1.8, -2.2, 1.25]], device=sim.device)
                target = torch.tensor([[0.0, 0.0, 0.48]], device=sim.device)
                eye[0, :2] += current_root_xy
                target[0, :2] += current_root_xy
                camera.set_world_poses_from_view(eye, target)
            sim.step(render=capture_frame)
            scene.update(SIM_DT)

            reference_position = robot.data.body_pos_w[0, reference_ids[0]].detach().cpu()
            reference_quaternion = robot.data.body_quat_w[0, reference_ids[0]].detach().cpu()
            if reference_position_0 is None:
                reference_position_0 = reference_position.clone()
                reference_quaternion_0 = reference_quaternion.clone()
            tilt = quaternion_distance_wxyz(reference_quaternion.tolist(), reference_quaternion_0.tolist())
            height_drop = float(reference_position_0[2] - reference_position[2])
            fallen = bool(tilt > MAX_REFERENCE_TILT_RAD or height_drop > MAX_ROOT_HEIGHT_DROP_M)
            if fallen and not previously_fallen:
                fall_count += 1
            previously_fallen = fallen
            max_joint_error = max(
                max_joint_error,
                float(torch.max(torch.abs(robot.data.joint_pos - joint_targets)).detach().cpu()),
            )

            if (step + 1) % trace_stride == 0 or step == 0 or step + 1 == physics_steps:
                position = reference_position.tolist()
                quaternion = reference_quaternion.tolist()
                linear_velocity = robot.data.body_lin_vel_w[0, reference_ids[0]].detach().cpu().tolist()
                angular_velocity = robot.data.body_ang_vel_w[0, reference_ids[0]].detach().cpu().tolist()
                force_vectors = contact_sensor.data.net_forces_w[0, foot_ids].detach().cpu()
                force_norms = torch.linalg.vector_norm(force_vectors, dim=-1).tolist()
                traces.append(
                    {
                        "time_s": round((step + 1) * SIM_DT, 6),
                        "root_x_position_m": [round(float(value), 8) for value in position],
                        "root_x_orientation_wxyz": [round(float(value), 9) for value in quaternion],
                        "root_x_linear_velocity_mps": [round(float(value), 8) for value in linear_velocity],
                        "root_x_angular_velocity_radps": [round(float(value), 8) for value in angular_velocity],
                        "reference_tilt_rad": round(tilt, 8),
                        "foot_contact_force_n": {
                            name: round(float(value), 6) for name, value in zip(foot_names, force_norms)
                        },
                        "fallen": fallen,
                    }
                )

            if capture_frame:
                rgb = camera.data.output["rgb"][0, ..., :3].detach().cpu().numpy()
                segmentation = camera.data.output["semantic_segmentation"][0].detach().cpu().numpy()
                image = Image.fromarray(rgb)
                image.save(frame_dir / f"frame_{video_frame_count:04d}.png")
                frame_metrics = _frame_visibility(rgb, segmentation)
                frame_metrics["frame_index"] = video_frame_count
                frame_metrics["time_s"] = round((step + 1) * SIM_DT, 6)
                visibility.append(frame_metrics)
                video_frame_count += 1

        if reference_position_0 is None or reference_quaternion_0 is None:
            raise RuntimeError("Simulation produced no state samples")

        final_position = robot.data.body_pos_w[0, reference_ids[0]].detach().cpu().tolist()
        horizontal_drift = math.hypot(
            final_position[0] - float(reference_position_0[0]),
            final_position[1] - float(reference_position_0[1]),
        )
        metrics = {
            "duration_s": round(physics_steps * SIM_DT, 6),
            "physics_steps": physics_steps,
            "reset_count": 0,
            "done_count": fall_count,
            "fall_count": fall_count,
            "max_abs_action": 0.0,
            "max_abs_command": 0.0,
            "max_reference_tilt_rad": max(item["reference_tilt_rad"] for item in traces),
            "root_height_drop_m": round(
                float(reference_position_0[2]) - min(item["root_x_position_m"][2] for item in traces), 8
            ),
            "horizontal_drift_m": round(horizontal_drift, 8),
            "semantic_forward_displacement_m": round(final_position[1] - float(reference_position_0[1]), 8),
            "semantic_strafe_displacement_m": round(final_position[0] - float(reference_position_0[0]), 8),
            "max_joint_target_error_rad": round(max_joint_error, 8),
        }
        required_duration = 0.0 if args.smoke else MIN_GATE_DURATION_S
        dynamics_passed, failures = evaluate_gate(metrics, required_duration_s=required_duration)

        video_inspection = None
        video_path = output_dir / "proof.mp4"
        contact_sheet_path = output_dir / "contact_sheet.png"
        if camera is not None:
            if video_frame_count == 0:
                failures.append("camera produced no frames")
            else:
                _encode_video(frame_dir, video_path, args.video_fps)
                representative_indices = sorted({0, video_frame_count // 2, video_frame_count - 1})
                images = [Image.open(frame_dir / f"frame_{index:04d}.png").convert("RGB") for index in representative_indices]
                sheet = Image.new("RGB", (args.video_width * len(images), args.video_height + 34), "white")
                draw = ImageDraw.Draw(sheet)
                for column, (index, image) in enumerate(zip(representative_indices, images)):
                    sheet.paste(image, (column * args.video_width, 34))
                    draw.text((column * args.video_width + 8, 9), f"frame {index} · {index/args.video_fps:.1f}s", fill="black")
                sheet.save(contact_sheet_path)
                probe = _probe_video(video_path)
                sampled_visibility = [visibility[index] for index in representative_indices]
                expected_video_duration = video_frame_count / args.video_fps
                actual_video_duration = float(probe["format"]["duration"])
                video_inspection = {
                    "path": str(video_path.relative_to(model_spec.ALGORITHM_ROOT.parent.parent)),
                    "sha256": _sha256(video_path),
                    "contact_sheet_path": str(contact_sheet_path.relative_to(model_spec.ALGORITHM_ROOT.parent.parent)),
                    "frame_count": video_frame_count,
                    "fps": args.video_fps,
                    "duration_s": actual_video_duration,
                    "expected_duration_s": expected_video_duration,
                    "resolution": [args.video_width, args.video_height],
                    "representative_frames": sampled_visibility,
                    "nonblank_frames_passed": all(item["nonblank"] for item in sampled_visibility),
                    "character_visibility_passed": all(item["character_visible"] for item in sampled_visibility),
                    "behavior_visibility_note": "A stable standing pose is expected to remain nearly static; visibility, not pixel motion, is required.",
                }
                if abs(actual_video_duration - expected_video_duration) > 0.15:
                    failures.append("encoded video duration does not match simulated capture duration")
                if not video_inspection["nonblank_frames_passed"]:
                    failures.append("one or more representative video frames are blank")
                if not video_inspection["character_visibility_passed"]:
                    failures.append("Landau is not fully visible with margin in representative video frames")

        if not args.smoke and camera is None:
            failures.append("exact gate validation requires proof video")
        passed = dynamics_passed and not failures
        robot_contract = model_spec.build_robot_spec()
        evidence = {
            "schema_version": 1,
            "milestone": MILESTONE_ID,
            "status": "passed" if passed else "failed",
            "lineage": "clean_restart_2026_08_22",
            "run_identity": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
            "seed": args.seed,
            "checkpoint": {
                "kind": "passive_pd_configuration",
                "identity": f"robot_spec_sha256:{_sha256(model_spec.ROBOT_SPEC_PATH)}",
                "policy_checkpoint": None,
            },
            "versions": {
                "isaaclab": _package_version("isaaclab"),
                "rsl_rl": _package_version("rsl-rl-lib"),
                "torch": _package_version("torch"),
                "python": platform.python_version(),
            },
            "simulator": {
                "device": args.device,
                "physics_dt_s": SIM_DT,
                "control_dt_s": CONTROL_DT,
                "single_process": True,
                "num_envs": 1,
            },
            "input": robot_contract["source"],
            "joint_contract": {
                "action_joints": robot_contract["action_joints"],
                "locked_joints": robot_contract["locked_joints"],
                "joints": robot_contract["joints"],
                "nominal_pose": robot_contract["nominal_pose"],
                "pd": robot_contract["pd"],
            },
            "command_contract": robot_contract["frames"],
            "metrics": metrics,
            "traces": traces,
            "video_inspection": video_inspection,
            "failures": failures,
            "cumulative_gates": [{"order": 1, "id": MILESTONE_ID, "status": "passed" if passed else "failed"}],
        }
        evidence_path = output_dir / ("smoke_validation.json" if args.smoke else "validation.json")
        evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(
            json.dumps(
                {
                    "status": evidence["status"],
                    "milestone": MILESTONE_ID,
                    "metrics": metrics,
                    "evidence": os.fspath(evidence_path),
                    "proof_video": os.fspath(video_path) if camera is not None else None,
                    "failures": failures,
                },
                indent=2,
            )
        )
        return evidence
    finally:
        if temporary_frames is not None:
            temporary_frames.cleanup()


def _parser() -> argparse.ArgumentParser:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=MIN_GATE_DURATION_S)
    parser.add_argument("--steps", type=int, help="Override duration with an exact physics-step count.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true", help="Run a short structural/physics smoke without passing the gate.")
    parser.add_argument("--record-video", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--video-fps", type=int, default=10)
    parser.add_argument("--video-width", type=int, default=640)
    parser.add_argument("--video-height", type=int, default=480)
    AppLauncher.add_app_launcher_args(parser)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    from isaaclab.app import AppLauncher

    args = _parser().parse_args(argv)
    if args.duration <= 0.0 or (args.steps is not None and args.steps <= 0):
        raise SystemExit("duration and steps must be positive")
    if not args.smoke and args.steps is not None and args.steps * SIM_DT < MIN_GATE_DURATION_S:
        raise SystemExit("short --steps runs require --smoke")
    if not args.smoke and args.duration < MIN_GATE_DURATION_S:
        raise SystemExit("exact validation duration must be at least 30 seconds")
    args.enable_cameras = bool(args.record_video)
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    try:
        torch_seed = args.seed
        import torch

        torch.manual_seed(torch_seed)
        evidence = run(args, simulation_app)
        return 0 if evidence["status"] == "passed" or args.smoke else 1
    finally:
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
