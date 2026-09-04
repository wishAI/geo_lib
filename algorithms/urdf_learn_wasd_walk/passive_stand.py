"""Run one fail-closed component of the clean Landau passive-stand gate.

The dynamics component never enables rendering. The proof component runs in a
separate Isaac process and captures the active viewport directly, avoiding the
Replicator/synthetic-data camera graph.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
import traceback
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
MIN_IMPORTED_AXIS_COSINE = 0.995


def safe_output_dir(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    try:
        resolved.relative_to(OUTPUT_ROOT.resolve())
    except ValueError as error:
        raise ValueError(f"Output directory must stay below {OUTPUT_ROOT}") from error
    return resolved


def component_artifact_name(phase: str, smoke: bool) -> str:
    return f"{phase}{'_smoke' if smoke else ''}_validation.json"


def failure_artifact_name(phase: str, smoke: bool) -> str:
    return f"{phase}{'_smoke' if smoke else ''}_failure.json"


def traceback_artifact_name(phase: str, smoke: bool) -> str:
    return f"{phase}{'_smoke' if smoke else ''}_traceback.log"


def _set_runtime_stage(args, stage: str) -> None:
    args.runtime_stage = stage
    print(f"[landau-passive] stage={stage}", file=sys.stderr, flush=True)


def build_failure_evidence(args, error: Exception, traceback_text: str) -> dict:
    return {
        "schema_version": 1,
        "milestone": MILESTONE_ID,
        "component": getattr(args, "phase", "unknown"),
        "status": "failed_to_execute",
        "gate_eligible": False,
        "milestone_status_changed": False,
        "experiment_id": (
            "zero_pose_upper_body_authority_probe_v1"
            if getattr(args, "authority_probe", False)
            else (
                "gravity_static_pose_release_v1"
                if getattr(args, "static_pose_probe", False)
                else None
            )
        ),
        "run_identity": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ"),
        "runtime_stage": getattr(args, "runtime_stage", "unknown"),
        "exception": {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback_text,
        },
        "input": {
            "urdf_path": str(model_spec.URDF_PATH.relative_to(model_spec.ALGORITHM_ROOT.parent.parent)),
            "urdf_sha256": model_spec.EXPECTED_URDF_SHA256,
            "robot_spec_sha256": _sha256(model_spec.ROBOT_SPEC_PATH),
        },
        "simulator_request": {
            "device": getattr(args, "device", None),
            "steps": getattr(args, "steps", None),
            "duration_s": getattr(args, "duration", None),
            "headless": getattr(args, "headless", None),
            "rendering_enabled": getattr(args, "phase", None) == "proof",
            "single_process": True,
        },
        "argv": list(sys.argv),
    }


def write_failure_evidence(args, error: Exception, traceback_text: str) -> tuple[Path, Path]:
    output_dir = safe_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    evidence_path = output_dir / failure_artifact_name(args.phase, args.smoke)
    traceback_path = output_dir / traceback_artifact_name(args.phase, args.smoke)
    traceback_path.write_text(traceback_text, encoding="utf-8")
    evidence = build_failure_evidence(args, error, traceback_text)
    evidence["traceback_path"] = str(
        traceback_path.relative_to(model_spec.ALGORITHM_ROOT.parent.parent)
    )
    evidence["traceback_sha256"] = _sha256(traceback_path)
    evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return evidence_path, traceback_path


def quaternion_distance_wxyz(left: Sequence[float], right: Sequence[float]) -> float:
    dot = abs(sum(float(a) * float(b) for a, b in zip(left, right)))
    return 2.0 * math.acos(min(1.0, max(-1.0, dot)))


def semantic_projected_gravity_wxyz(
    current_root: Sequence[float], upright_root: Sequence[float]
) -> list[float]:
    """Project unit gravity into a frame that is identity at the audited upright pose."""

    def multiply(left: Sequence[float], right: Sequence[float]) -> tuple[float, float, float, float]:
        lw, lx, ly, lz = map(float, left)
        rw, rx, ry, rz = map(float, right)
        return (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        )

    upright_inverse = (
        float(upright_root[0]), -float(upright_root[1]),
        -float(upright_root[2]), -float(upright_root[3]),
    )
    semantic_world = multiply(current_root, upright_inverse)
    inverse = (semantic_world[0], -semantic_world[1], -semantic_world[2], -semantic_world[3])
    rotated = multiply(multiply(inverse, (0.0, 0.0, 0.0, -1.0)), semantic_world)
    return [rotated[1], rotated[2], rotated[3]]


def summarize_joint_tracking(
    joint_names: Sequence[str],
    max_errors: Sequence[float],
    max_error_times: Sequence[float],
    max_velocities: Sequence[float],
    max_computed_torques: Sequence[float],
    max_applied_torques: Sequence[float],
    torque_saturation_counts: Sequence[int],
    physics_steps: int,
) -> list[dict]:
    """Build a compact, deterministic per-joint diagnostic in worst-error order."""
    columns = (
        max_errors,
        max_error_times,
        max_velocities,
        max_computed_torques,
        max_applied_torques,
        torque_saturation_counts,
    )
    if physics_steps <= 0 or any(len(column) != len(joint_names) for column in columns):
        raise ValueError("joint diagnostic columns must match and physics_steps must be positive")
    records = [
        {
            "name": name,
            "max_target_error_rad": round(float(max_errors[index]), 8),
            "max_target_error_time_s": round(float(max_error_times[index]), 6),
            "max_abs_velocity_radps": round(float(max_velocities[index]), 8),
            "max_abs_computed_torque_nm": round(float(max_computed_torques[index]), 8),
            "max_abs_applied_torque_nm": round(float(max_applied_torques[index]), 8),
            "torque_saturation_step_fraction": round(
                int(torque_saturation_counts[index]) / physics_steps, 8
            ),
        }
        for index, name in enumerate(joint_names)
    ]
    return sorted(records, key=lambda item: (-item["max_target_error_rad"], item["name"]))


def audit_and_clamp_derived_limit(
    name: str,
    raw_desired_rad: float,
    raw_target_rad: float,
    limits_rad: Sequence[float],
    *,
    tolerance_rad: float = model_spec.DERIVED_POSE_FINGER_LIMIT_TOLERANCE_RAD,
) -> tuple[float, float, dict]:
    """Clamp negligible finger-only noise while preserving meaningful failures.

    Two milliradians is 0.115 degrees and moves a point on a 40 mm finger
    segment by at most 80 micrometers. The tolerance never applies to an action
    joint or another non-locomotion joint.
    """

    lower, upper = map(float, limits_rad)
    if lower > upper or tolerance_rad < 0.0:
        raise ValueError("joint limits and tolerance must be valid")
    is_tolerant_finger = name in model_spec.FINGER_JOINTS

    def audit_field(raw: float) -> tuple[float, float, bool, bool]:
        legal = min(upper, max(lower, float(raw)))
        excess = abs(float(raw) - legal)
        outside = excess > 0.0
        accepted = not outside or (is_tolerant_finger and excess <= tolerance_rad)
        effective = legal if outside and accepted else float(raw)
        return effective, excess, outside and accepted, accepted

    desired, desired_excess, desired_clamped, desired_accepted = audit_field(raw_desired_rad)
    target, target_excess, target_clamped, target_accepted = audit_field(raw_target_rad)
    passed = desired_accepted and target_accepted
    record = {
        "name": name,
        "limits_rad": [lower, upper],
        "tolerance_rad": tolerance_rad if is_tolerant_finger else 0.0,
        "tolerance_scope": "non_locomotion_finger_only" if is_tolerant_finger else "none",
        "raw_desired_rad": float(raw_desired_rad),
        "clamped_desired_rad": desired,
        "desired_limit_excess_rad": desired_excess,
        "desired_was_clamped": desired_clamped,
        "raw_target_rad": float(raw_target_rad),
        "clamped_target_rad": target,
        "target_limit_excess_rad": target_excess,
        "target_was_clamped": target_clamped,
        "passed": passed,
    }
    return desired, target, record


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


def evaluate_proof(video: dict, *, required_duration_s: float = MIN_GATE_DURATION_S) -> tuple[bool, list[str]]:
    failures = []
    if video["duration_s"] + 0.15 < required_duration_s:
        failures.append("proof video is shorter than the required simulated behavior")
    if not video["nonblank_frames_passed"]:
        failures.append("one or more representative proof frames are blank")
    if not video["character_visibility_passed"]:
        failures.append("Landau link geometry is not fully framed with margin")
    if not video["temporal_progression_visible"]:
        failures.append("proof frames do not visibly establish temporal progression")
    return not failures, failures


def preflight_proof(output_dir: Path, *, smoke: bool, seed: int) -> dict:
    """Require a current, passing dynamics component before starting a renderer."""
    path = output_dir / component_artifact_name("dynamics", smoke)
    if not path.is_file():
        raise ValueError(f"proof requires prior camera-free dynamics evidence: {path}")
    evidence = json.loads(path.read_text(encoding="utf-8"))
    expected_checkpoint = f"robot_spec_sha256:{_sha256(model_spec.ROBOT_SPEC_PATH)}"
    if evidence.get("component") != "dynamics" or evidence.get("status") != "passed":
        raise ValueError("proof requires a passing dynamics component")
    if evidence.get("seed") != seed:
        raise ValueError("proof seed must match dynamics evidence")
    if evidence.get("input", {}).get("urdf_sha256") != model_spec.EXPECTED_URDF_SHA256:
        raise ValueError("dynamics evidence does not use the required current URDF")
    if evidence.get("checkpoint", {}).get("identity") != expected_checkpoint:
        raise ValueError("dynamics evidence robot contract is stale")
    imported = evidence.get("joint_contract", {}).get("runtime_importer_axis_evidence")
    if not imported or not imported.get("passed"):
        raise ValueError("dynamics evidence lacks a passing runtime importer-axis audit")
    return evidence


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
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-framerate", str(fps),
            "-i", os.fspath(frame_dir / "frame_%04d.png"), "-c:v", "libx264", "-pix_fmt",
            "yuv420p", "-movflags", "+faststart", os.fspath(video_path),
        ],
        check=True,
    )


def _probe_video(video_path: Path) -> dict:
    return json.loads(
        subprocess.run(
            [
                "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                "stream=width,height,avg_frame_rate,nb_frames:format=duration", "-of", "json",
                os.fspath(video_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    )


def _build_scene_cfg(args):
    import isaaclab.sim as sim_utils
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import ArticulationCfg, AssetBaseCfg
    from isaaclab.scene import InteractiveSceneCfg
    from isaaclab.sensors import ContactSensorCfg
    from isaaclab.terrains import TerrainImporterCfg
    from isaaclab.utils import configclass

    spec = model_spec.build_robot_spec()
    actuator_groups = model_spec.passive_actuator_groups(
        [joint["name"] for joint in spec["joints"]],
        authority_probe=args.authority_probe,
    )

    @configclass
    class PassiveStandSceneCfg(InteractiveSceneCfg):
        terrain = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="plane",
            collision_group=-1,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply", restitution_combine_mode="multiply",
                static_friction=1.0, dynamic_friction=0.9, restitution=0.0,
            ),
            debug_vis=False,
        )
        robot = ArticulationCfg(
            prim_path="{ENV_REGEX_NS}/Robot",
            spawn=sim_utils.UrdfFileCfg(
                asset_path=os.fspath(model_spec.URDF_PATH),
                usd_dir=os.fspath(OUTPUT_ROOT / "usd_cache"),
                usd_file_name="landau_v10_parallel_mesh.usd",
                force_usd_conversion=args.phase == "dynamics" and not args.reuse_usd_cache,
                fix_base=False,
                merge_fixed_joints=False,
                make_instanceable=False,
                self_collision=False,
                activate_contact_sensors=True,
                collider_type="convex_hull",
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=False, retain_accelerations=False, linear_damping=0.0,
                    angular_damping=0.0, max_linear_velocity=20.0, max_angular_velocity=50.0,
                    max_depenetration_velocity=1.0,
                ),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    enabled_self_collisions=False, solver_position_iteration_count=8,
                    solver_velocity_iteration_count=4, sleep_threshold=0.0,
                    stabilization_threshold=0.0001,
                ),
                joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                    drive_type="force", target_type="position",
                    gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0),
                ),
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=tuple(spec["nominal_pose"]["base_position_m"]),
                rot=tuple(spec["nominal_pose"]["base_orientation_wxyz"]),
                joint_pos={".*": 0.0}, joint_vel={".*": 0.0},
            ),
            soft_joint_pos_limit_factor=0.95,
            actuators={
                **{
                    name: ImplicitActuatorCfg(
                        joint_names_expr=list(group["joints"]), effort_limit_sim=50.0,
                        velocity_limit_sim=4.0, stiffness=group["stiffness"], damping=group["damping"],
                    )
                    for name, group in actuator_groups.items()
                },
            },
        )
        contact_forces = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/.*", update_period=SIM_DT,
            history_length=3, track_air_time=True,
        )
        light = AssetBaseCfg(
            prim_path="/World/Sun",
            spawn=sim_utils.DistantLightCfg(intensity=3000.0, color=(0.92, 0.92, 0.92)),
        )

    return PassiveStandSceneCfg(num_envs=1, env_spacing=2.0, lazy_sensor_update=True)


def _quaternion_values(quaternion) -> list[float]:
    imaginary = quaternion.GetImaginary()
    return [round(float(quaternion.GetReal()), 9), *[round(float(value), 9) for value in imaginary]]


def _inspect_imported_joint_contract(stage, robot_path: str, contract: dict) -> dict:
    """Reconstruct PhysX/USD joint axes after importer local-frame rotation."""
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    source = {record["name"]: record for record in contract["joints"]}
    root = stage.GetPrimAtPath(robot_path)
    if not root.IsValid():
        raise RuntimeError(f"Imported robot prim is missing: {robot_path}")
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    primary = {"X": Gf.Vec3d(1, 0, 0), "Y": Gf.Vec3d(0, 1, 0), "Z": Gf.Vec3d(0, 0, 1)}
    records = []
    found = set()
    for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        if not prim.IsA(UsdPhysics.RevoluteJoint) or prim.GetName() not in source:
            continue
        name = prim.GetName()
        joint = UsdPhysics.RevoluteJoint(prim)
        body0 = joint.GetBody0Rel().GetTargets()
        body1 = joint.GetBody1Rel().GetTargets()
        if len(body0) != 1 or len(body1) != 1:
            raise RuntimeError(
                f"{name} has body target counts ({len(body0)}, {len(body1)}); expected (1, 1)"
            )
        axis_attribute = joint.GetAxisAttr()
        authored_axis = str(axis_attribute.Get())
        # Isaac Sim 4.5 omits physics:axis after aligning arbitrary axes to
        # PhysX's X default. An empty USD token therefore has effective axis X.
        axis_token = authored_axis or "X"
        if axis_token not in primary:
            raise RuntimeError(f"{name} imported unsupported primary axis {axis_token!r}")
        local_rotation = joint.GetLocalRot0Attr().Get()
        body_axis = Gf.Rotation(local_rotation).TransformDir(primary[axis_token])
        world_axis = cache.GetLocalToWorldTransform(stage.GetPrimAtPath(body0[0])).TransformDir(body_axis)
        norm = math.sqrt(sum(float(world_axis[index]) ** 2 for index in range(3)))
        imported = [float(world_axis[index]) / norm for index in range(3)]
        expected = source[name]["axis_world_zero_pose"]
        cosine = sum(float(a) * float(b) for a, b in zip(imported, expected))
        records.append(
            {
                "name": name,
                "usd_prim_path": str(prim.GetPath()),
                "body0_path": str(body0[0]),
                "body1_path": str(body1[0]),
                "imported_primary_axis": axis_token,
                "imported_primary_axis_authored": bool(authored_axis),
                "imported_local_rotation0_wxyz": _quaternion_values(local_rotation),
                "imported_local_rotation1_wxyz": _quaternion_values(joint.GetLocalRot1Attr().Get()),
                "imported_world_axis_zero_pose": [round(value, 9) for value in imported],
                "source_world_axis_zero_pose": expected,
                "source_axis_is_primary": source[name]["source_axis_is_primary"],
                "directed_axis_cosine": round(cosine, 9),
                "alignment_passed": cosine >= MIN_IMPORTED_AXIS_COSINE,
            }
        )
        found.add(name)
    missing = sorted(set(source) - found)
    failed = sorted(record["name"] for record in records if not record["alignment_passed"])
    return {
        "method": "UsdPhysics primary axis transformed by localRot0 and body0 zero-pose world transform",
        "minimum_directed_axis_cosine": MIN_IMPORTED_AXIS_COSINE,
        "joint_count": len(records),
        "missing_joints": missing,
        "misaligned_joints": failed,
        "passed": not missing and not failed and len(records) == len(source),
        "joints": sorted(records, key=lambda item: item["name"]),
    }


def _capture_viewport_frame(viewport, sim, path: Path) -> None:
    """Capture viewport LdrColor without creating a synthetic-data graph."""
    import omni.renderer_capture
    from omni.kit.viewport.utility import capture_viewport_to_file

    path.unlink(missing_ok=True)
    helper = capture_viewport_to_file(viewport, file_path=os.fspath(path), is_hdr=False)
    if helper is None:
        raise RuntimeError("viewport capture API returned no capture helper")
    for _ in range(8):
        sim.render()
        if path.is_file() and path.stat().st_size > 0:
            break
    omni.renderer_capture.acquire_renderer_capture_interface().wait_async_capture()
    for _ in range(2):
        sim.render()
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"viewport capture did not produce {path}")


def _projected_link_bbox(viewport, positions, width: int, height: int) -> dict:
    from pxr import Gf

    projected = [viewport.world_to_ndc.Transform(Gf.Vec3d(*map(float, position))) for position in positions]
    pixels = [((float(p[0]) + 1) * 0.5 * width, (1 - float(p[1])) * 0.5 * height) for p in projected]
    min_x, max_x = min(p[0] for p in pixels), max(p[0] for p in pixels)
    min_y, max_y = min(p[1] for p in pixels), max(p[1] for p in pixels)
    margin = min(min_x, min_y, width - max_x, height - max_y)
    front = min(float(p[2]) for p in projected) >= 0.0
    visible = front and margin >= 12 and max_x - min_x >= 45 and max_y - min_y >= 100
    return {
        "projected_link_origin_bbox_xyxy": [round(v, 2) for v in (min_x, min_y, max_x, max_y)],
        "minimum_projected_margin_px": round(margin, 2),
        "all_link_origins_in_front": front,
        "character_visible": visible,
    }


def _inspect_frame(path: Path, bbox: dict, index: int, time_s: float) -> dict:
    import numpy as np
    from PIL import Image, ImageDraw

    image = Image.open(path).convert("RGB")
    pixels = np.asarray(image)
    draw = ImageDraw.Draw(image)
    draw.rectangle((7, 7, 190, 33), fill=(245, 245, 245), outline=(30, 30, 30))
    draw.text((13, 13), f"Landau passive · t={time_s:05.1f}s", fill=(10, 10, 10))
    image.save(path)
    return {
        "frame_index": index, "time_s": round(time_s, 6),
        "nonblank": bool(float(pixels.std()) > 2.0),
        "rgb_stddev": round(float(pixels.std()), 5),
        "resolution": [image.width, image.height], **bbox,
    }


def _versions() -> dict:
    try:
        from isaacsim.core.version import get_version
        isaac_sim = ".".join(str(item) for item in get_version())
    except Exception as error:  # pragma: no cover
        isaac_sim = f"unavailable:{type(error).__name__}"
    return {
        "isaac_sim": isaac_sim, "isaaclab": _package_version("isaaclab"),
        "rsl_rl": _package_version("rsl-rl-lib"), "torch": _package_version("torch"),
        "numpy": _package_version("numpy"), "python": platform.python_version(),
    }


def _run(args, simulation_app) -> dict:
    import torch
    import isaaclab.sim as sim_utils
    from isaaclab.scene import InteractiveScene

    _set_runtime_stage(args, "simulation_configuration")
    output_dir = safe_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sim_cfg = sim_utils.SimulationCfg(
        dt=SIM_DT, render_interval=max(1, round((1 / args.video_fps) / SIM_DT)), device=args.device,
    )
    sim_cfg.physx.gpu_max_rigid_patch_count = 2**18
    sim = sim_utils.SimulationContext(sim_cfg)
    _set_runtime_stage(args, "scene_construction")
    scene = InteractiveScene(_build_scene_cfg(args))
    _set_runtime_stage(args, "simulation_reset")
    sim.reset()
    _set_runtime_stage(args, "post_reset_contract")
    robot = scene["robot"]
    contacts = scene["contact_forces"]
    contract = model_spec.build_robot_spec()
    expected = set(contract["nominal_pose"]["joint_positions_rad"])
    if set(robot.joint_names) != expected:
        raise RuntimeError(
            f"Imported joints differ from audited URDF: missing={sorted(expected-set(robot.joint_names))}, "
            f"unexpected={sorted(set(robot.joint_names)-expected)}"
        )
    reference_ids, reference_names = robot.find_bodies("root_x", preserve_order=True)
    if reference_names != ["root_x"]:
        raise RuntimeError(f"Expected one root_x body, got {reference_names}")
    foot_ids, foot_names = contacts.find_bodies(
        ["foot_l", "foot_r", "toes_01_l", "toes_01_r"], preserve_order=True
    )
    configured_groups = model_spec.passive_actuator_groups(
        robot.joint_names, authority_probe=args.authority_probe
    )
    expected_gains = {
        name: (group["stiffness"], group["damping"])
        for group in configured_groups.values()
        for name in group["joints"]
    }
    static_pose_seed = None
    initial_joint_positions = robot.data.default_joint_pos.clone()
    targets = robot.data.default_joint_pos.clone()
    initial_root_state = robot.data.default_root_state.clone()
    if args.static_pose_probe:
        static_pose_seed = model_spec.derive_static_pose()
        for index, name in enumerate(robot.joint_names):
            desired = float(static_pose_seed["joint_positions_rad"].get(name, 0.0))
            gravity = float(static_pose_seed["fixed_root_gravity_torque_nm"][name])
            stiffness = float(expected_gains[name][0])
            initial_joint_positions[0, index] = desired
            # The gravity-torque sign is the free acceleration direction; a
            # position-drive preload must oppose it.
            targets[0, index] = desired - gravity / stiffness
        initial_root_state[:, 2] += float(static_pose_seed["ground_aligned_base_z_m"])
        initial_root_state[:, 7:] = 0.0
        robot.write_root_state_to_sim(initial_root_state)
    robot.write_joint_state_to_sim(initial_joint_positions, torch.zeros_like(targets))
    robot.set_joint_position_target(targets)
    scene.write_data_to_sim()
    _set_runtime_stage(args, "mass_diagnostic_setup")
    # Isaac Lab 0.36.1 does not normalize `default_mass` onto the asset
    # device when it clones the PhysX mass view. Make the transfer explicit
    # before combining masses with CUDA-backed body COM positions.
    body_masses = robot.data.default_mass[0].to(device=robot.device, dtype=targets.dtype)
    total_mass = torch.sum(body_masses)
    robot_weight_n = float(total_mass.detach().cpu()) * 9.81
    support_aabb = contract["nominal_pose"]["zero_pose_ground_support"]["support_aabb_xy_m"]
    support_reference = "audited_zero_pose_collision_support_aabb"

    _set_runtime_stage(args, "runtime_actuator_audit")
    target_values = robot.data.joint_pos_target[0].detach().cpu().tolist()
    stiffness_values = robot.data.joint_stiffness[0].detach().cpu().tolist()
    damping_values = robot.data.joint_damping[0].detach().cpu().tolist()
    effort_limit_values = robot.data.joint_effort_limits[0].detach().cpu().tolist()
    velocity_limit_values = robot.data.joint_vel_limits[0].detach().cpu().tolist()
    position_limits = robot.data.joint_pos_limits[0].detach().cpu().tolist()
    actuator_mismatches = []
    for index, name in enumerate(robot.joint_names):
        expected_stiffness, expected_damping = expected_gains[name]
        actual = (
            target_values[index], stiffness_values[index], damping_values[index],
            effort_limit_values[index], velocity_limit_values[index],
        )
        expected_actual = (
            float(targets[0, index].detach().cpu()), expected_stiffness,
            expected_damping, 50.0, 4.0,
        )
        if any(abs(float(a) - float(b)) > 1.0e-5 for a, b in zip(actual, expected_actual)):
            actuator_mismatches.append(
                {"name": name, "actual_target_stiffness_damping_effort_velocity": actual,
                 "expected_target_stiffness_damping_effort_velocity": expected_actual}
            )
    runtime_actuators = {
        "authority": "values read from Isaac Lab ArticulationData after actuator initialization",
        "passed": not actuator_mismatches,
        "mismatches": actuator_mismatches,
        "joint_order": list(robot.joint_names),
        "joints": [
            {
                "name": name,
                "target_position_rad": round(float(target_values[index]), 9),
                "stiffness_nm_per_rad": round(float(stiffness_values[index]), 9),
                "damping_nm_s_per_rad": round(float(damping_values[index]), 9),
                "effort_limit_nm": round(float(effort_limit_values[index]), 9),
                "velocity_limit_radps": round(float(velocity_limit_values[index]), 9),
                "position_limits_rad": [round(float(value), 9) for value in position_limits[index]],
            }
            for index, name in enumerate(robot.joint_names)
        ],
    }

    imported = None
    if args.phase == "dynamics":
        _set_runtime_stage(args, "importer_axis_audit")
        imported = _inspect_imported_joint_contract(sim.stage, "/World/envs/env_0/Robot", contract)

    viewport = None
    frame_context = None
    frame_dir = None
    if args.phase == "proof":
        from omni.kit.viewport.utility import get_active_viewport
        viewport = get_active_viewport()
        if viewport is None:
            raise RuntimeError("proof phase has no active viewport")
        viewport.updates_enabled = True
        viewport.resolution = (args.video_width, args.video_height)
        sim.set_camera_view(eye=(1.8, -2.4, 1.35), target=(0.0, 0.0, 0.52))
        for _ in range(12):
            sim.render()
        if tuple(map(int, viewport.resolution)) != (args.video_width, args.video_height):
            raise RuntimeError(f"viewport resolution did not settle: {viewport.resolution}")
        frame_context = tempfile.TemporaryDirectory(prefix="landau_viewport_frames_", dir=output_dir)
        frame_dir = Path(frame_context.name)

    static_pose_derivation = None
    if args.static_pose_probe:
        _set_runtime_stage(args, "fixed_root_gravity_settling")
        settle_window_steps = min(args.settle_steps, max(1, round(0.5 / SIM_DT)))
        position_sum = torch.zeros_like(targets[0])
        torque_sum = torch.zeros_like(targets[0])
        velocity_sum = torch.zeros_like(targets[0])
        support_force_sum_n = 0.0
        settle_samples = 0
        settling_trace = []
        for settle_step in range(args.settle_steps):
            # Installed Isaac Lab 0.36.1's supported root-state writer updates
            # both PhysX and cached root state. Rewriting zero velocity makes
            # this a bounded fixed-root settling phase without a second asset
            # or Isaac process.
            robot.write_root_state_to_sim(initial_root_state)
            robot.set_joint_position_target(targets)
            scene.write_data_to_sim()
            sim.step(render=False)
            scene.update(SIM_DT)
            if settle_step >= args.settle_steps - settle_window_steps:
                position_sum += robot.data.joint_pos[0]
                torque_sum += robot.data.applied_torque[0]
                velocity_sum += torch.abs(robot.data.joint_vel[0])
                support_force_sum_n += float(
                    torch.sum(
                        torch.linalg.vector_norm(
                            contacts.data.net_forces_w[0, foot_ids], dim=-1
                        )
                    ).detach().cpu()
                )
                settle_samples += 1
            if (
                settle_step == 0
                or (settle_step + 1) % max(1, round(0.5 / SIM_DT)) == 0
                or settle_step + 1 == args.settle_steps
            ):
                settling_trace.append(
                    {
                        "time_s": round((settle_step + 1) * SIM_DT, 6),
                        "max_abs_joint_velocity_radps": round(
                            float(torch.max(torch.abs(robot.data.joint_vel[0])).detach().cpu()), 8
                        ),
                        "support_force_n": round(
                            float(
                                torch.sum(
                                    torch.linalg.vector_norm(
                                        contacts.data.net_forces_w[0, foot_ids], dim=-1
                                    )
                                ).detach().cpu()
                            ),
                            8,
                        ),
                    }
                )
        if settle_samples == 0:
            raise RuntimeError("fixed-root settling produced no averaging samples")
        settled_positions = position_sum / settle_samples
        required_torques = torque_sum / settle_samples
        mean_abs_velocities = velocity_sum / settle_samples
        stiffness_tensor = robot.data.joint_stiffness[0]
        release_targets = settled_positions + required_torques / stiffness_tensor
        violations = []
        torque_records = []
        limit_audits = []
        for index, name in enumerate(robot.joint_names):
            raw_desired = float(settled_positions[index].detach().cpu())
            raw_target = float(release_targets[index].detach().cpu())
            desired, target, limit_audit = audit_and_clamp_derived_limit(
                name, raw_desired, raw_target, position_limits[index]
            )
            settled_positions[index] = desired
            release_targets[index] = target
            limit_audits.append(limit_audit)
            required_torque = float(required_torques[index].detach().cpu())
            effort_limit = float(effort_limit_values[index])
            if not limit_audit["passed"]:
                violations.append(limit_audit)
            torque_records.append(
                {
                    "name": name,
                    "raw_settled_position_rad": round(raw_desired, 12),
                    "settled_position_rad": round(desired, 9),
                    "raw_released_target_rad": round(raw_target, 12),
                    "released_target_rad": round(target, 9),
                    "raw_required_torque_nm": round(required_torque, 9),
                    "required_torque_nm": round(required_torque, 9),
                    "released_pd_preload_after_clamp_nm": round(
                        float(stiffness_tensor[index].detach().cpu()) * (target - desired), 9
                    ),
                    "effort_limit_nm": round(effort_limit, 9),
                    "required_torque_limit_fraction": round(abs(required_torque) / effort_limit, 9),
                    "limit_audit": limit_audit,
                    "mean_abs_settling_velocity_radps": round(
                        float(mean_abs_velocities[index].detach().cpu()), 9
                    ),
                }
            )
        if violations:
            raise RuntimeError(f"derived static pose or preload exceeds joint limits: {violations}")
        settled_pose = {
            name: float(settled_positions[index].detach().cpu())
            for index, name in enumerate(robot.joint_names)
        }
        settled_geometry = model_spec.analyze_pose_geometry(settled_pose)
        static_pose_derivation = {
            "id": "gravity_static_pose_release_v1",
            "fixed_root_duration_s": round(args.settle_steps * SIM_DT, 6),
            "averaging_window_s": round(settle_window_steps * SIM_DT, 6),
            "seed_geometry": static_pose_seed,
            "settled_geometry": settled_geometry,
            "mean_support_force_n": round(support_force_sum_n / settle_samples, 8),
            "mean_support_force_body_weight_ratio": round(
                support_force_sum_n / settle_samples / robot_weight_n, 8
            ),
            "maximum_required_torque_limit_fraction": max(
                item["required_torque_limit_fraction"] for item in torque_records
            ),
            "maximum_mean_abs_settling_velocity_radps": max(
                item["mean_abs_settling_velocity_radps"] for item in torque_records
            ),
            "joint_order": list(robot.joint_names),
            "joints": torque_records,
            "limit_audit": {
                "tolerance_rad": model_spec.DERIVED_POSE_FINGER_LIMIT_TOLERANCE_RAD,
                "scope": "non_locomotion_finger_only",
                "finger_joints": list(model_spec.FINGER_JOINTS),
                "clamped_joint_count": sum(
                    audit["desired_was_clamped"] or audit["target_was_clamped"]
                    for audit in limit_audits
                ),
                "clamped_desired_count": sum(
                    audit["desired_was_clamped"] for audit in limit_audits
                ),
                "clamped_target_count": sum(
                    audit["target_was_clamped"] for audit in limit_audits
                ),
                "records": limit_audits,
            },
            "settling_trace": settling_trace,
            "hands_and_fingers_use_baseline_locked_pd": True,
            "high_authority_profile_used": False,
        }
        support_aabb = settled_geometry["support_aabb_xy_m"]
        support_reference = "settled_pose_exact_fk_collision_support_aabb"
        release_root_state = robot.data.default_root_state.clone()
        release_root_state[:, 2] += float(settled_geometry["ground_aligned_base_z_m"])
        release_root_state[:, 7:] = 0.0
        robot.write_root_state_to_sim(release_root_state)
        robot.write_joint_state_to_sim(
            settled_positions.unsqueeze(0), torch.zeros_like(targets)
        )
        targets = release_targets.unsqueeze(0)
        robot.set_joint_position_target(targets)
        scene.write_data_to_sim()
        _set_runtime_stage(args, "free_root_release")

    _set_runtime_stage(args, "diagnostic_buffer_setup")
    steps = args.steps if args.steps is not None else math.ceil(args.duration / SIM_DT)
    trace_stride = max(1, round(CONTROL_DT / SIM_DT))
    video_stride = max(1, round((1 / args.video_fps) / SIM_DT))
    traces, frames = [], []
    fall_count, max_joint_error = 0, 0.0
    previously_fallen = False
    initial_position = initial_quaternion = None
    first_fall_time_s = None
    initial_system_com = None
    first_static_support_exit_time_s = None
    minimum_positive_y_support_margin_m = math.inf
    max_projected_gravity_horizontal = 0.0
    peak_total_support_force_n = 0.0
    peak_total_support_force_time_s = None
    joint_count = len(robot.joint_names)
    max_errors = torch.zeros(joint_count, device=robot.device)
    max_error_times = torch.zeros(joint_count, device=robot.device)
    max_velocities = torch.zeros(joint_count, device=robot.device)
    max_computed_torques = torch.zeros(joint_count, device=robot.device)
    max_applied_torques = torch.zeros(joint_count, device=robot.device)
    torque_saturation_counts = torch.zeros(joint_count, dtype=torch.int64, device=robot.device)
    contact_threshold_n = 1.0
    contact_timing = {
        name: {"first_contact_time_s": None, "last_contact_time_s": None, "peak_force_n": 0.0,
               "peak_force_time_s": None}
        for name in foot_names
    }
    try:
        for step in range(steps):
            if step == 0:
                _set_runtime_stage(args, "first_physics_step")
            robot.set_joint_position_target(targets)
            scene.write_data_to_sim()
            capture = viewport is not None and (step + 1) % video_stride == 0
            sim.step(render=capture)
            scene.update(SIM_DT)
            if step == 0:
                _set_runtime_stage(args, "first_root_state_sample")
            position = robot.data.body_pos_w[0, reference_ids[0]].detach().cpu()
            quaternion = robot.data.body_quat_w[0, reference_ids[0]].detach().cpu()
            if initial_position is None:
                initial_position, initial_quaternion = position.clone(), quaternion.clone()
            tilt = quaternion_distance_wxyz(quaternion.tolist(), initial_quaternion.tolist())
            drop = float(initial_position[2] - position[2])
            fallen = tilt > MAX_REFERENCE_TILT_RAD or drop > MAX_ROOT_HEIGHT_DROP_M
            if fallen and not previously_fallen:
                fall_count += 1
                if first_fall_time_s is None:
                    first_fall_time_s = (step + 1) * SIM_DT
            previously_fallen = fallen
            absolute_errors = torch.abs(robot.data.joint_pos[0] - targets[0])
            new_error_maxima = absolute_errors > max_errors
            max_errors = torch.maximum(max_errors, absolute_errors)
            max_error_times = torch.where(
                new_error_maxima,
                torch.full_like(max_error_times, (step + 1) * SIM_DT),
                max_error_times,
            )
            max_velocities = torch.maximum(max_velocities, torch.abs(robot.data.joint_vel[0]))
            computed_torques = torch.abs(robot.data.computed_torque[0])
            applied_torques = torch.abs(robot.data.applied_torque[0])
            max_computed_torques = torch.maximum(max_computed_torques, computed_torques)
            max_applied_torques = torch.maximum(max_applied_torques, applied_torques)
            torque_saturation_counts += (
                computed_torques > robot.data.joint_effort_limits[0] + 1.0e-6
            ).to(torch.int64)
            max_joint_error = max(max_joint_error, float(torch.max(absolute_errors).detach().cpu()))
            force_norms_tensor = torch.linalg.vector_norm(
                contacts.data.net_forces_w[0, foot_ids].detach().cpu(), dim=-1
            )
            time_s = (step + 1) * SIM_DT
            if step == 0:
                _set_runtime_stage(args, "first_system_com_sample")
            system_com_tensor = torch.sum(
                robot.data.body_com_pos_w[0] * body_masses.unsqueeze(-1), dim=0
            ) / total_mass
            system_com = system_com_tensor.detach().cpu().tolist()
            if initial_system_com is None:
                initial_system_com = list(system_com)
            positive_y_margin = float(support_aabb["maximum"][1]) - float(system_com[1])
            minimum_positive_y_support_margin_m = min(
                minimum_positive_y_support_margin_m, positive_y_margin
            )
            if positive_y_margin < 0.0 and first_static_support_exit_time_s is None:
                first_static_support_exit_time_s = time_s
            imported_root_gravity = robot.data.projected_gravity_b[0].detach().cpu().tolist()
            projected_gravity = semantic_projected_gravity_wxyz(
                quaternion.tolist(), initial_quaternion.tolist()
            )
            max_projected_gravity_horizontal = max(
                max_projected_gravity_horizontal,
                math.hypot(float(projected_gravity[0]), float(projected_gravity[1])),
            )
            total_support_force = sum(float(force) for force in force_norms_tensor.tolist())
            if step == 0:
                _set_runtime_stage(args, "physics_loop")
            if total_support_force > peak_total_support_force_n:
                peak_total_support_force_n = total_support_force
                peak_total_support_force_time_s = time_s
            for name, force in zip(foot_names, force_norms_tensor.tolist()):
                timing = contact_timing[name]
                if force >= contact_threshold_n:
                    if timing["first_contact_time_s"] is None:
                        timing["first_contact_time_s"] = round(time_s, 6)
                    timing["last_contact_time_s"] = round(time_s, 6)
                if force > timing["peak_force_n"]:
                    timing["peak_force_n"] = round(float(force), 6)
                    timing["peak_force_time_s"] = round(time_s, 6)
            if (step + 1) % trace_stride == 0 or step == 0 or step + 1 == steps:
                force_norms = force_norms_tensor.tolist()
                traces.append(
                    {
                        "time_s": round((step + 1) * SIM_DT, 6),
                        "root_x_position_m": [round(float(v), 8) for v in position.tolist()],
                        "root_x_orientation_wxyz": [round(float(v), 9) for v in quaternion.tolist()],
                        "root_x_linear_velocity_mps": [round(float(v), 8) for v in robot.data.body_lin_vel_w[0, reference_ids[0]].detach().cpu().tolist()],
                        "root_x_angular_velocity_radps": [round(float(v), 8) for v in robot.data.body_ang_vel_w[0, reference_ids[0]].detach().cpu().tolist()],
                        "reference_tilt_rad": round(tilt, 8),
                        "system_center_of_mass_m": [round(float(v), 8) for v in system_com],
                        "support_reference": support_reference,
                        "support_positive_y_margin_m": round(positive_y_margin, 8),
                        "zero_pose_support_positive_y_margin_m": round(positive_y_margin, 8),
                        "projected_gravity_semantic_body_frame": [round(float(v), 8) for v in projected_gravity],
                        "projected_gravity_imported_root_frame": [
                            round(float(v), 8) for v in imported_root_gravity
                        ],
                        "foot_contact_force_n": {name: round(float(v), 6) for name, v in zip(foot_names, force_norms)},
                        "joint_position_rad": [round(float(v), 8) for v in robot.data.joint_pos[0].detach().cpu().tolist()],
                        "joint_velocity_radps": [round(float(v), 8) for v in robot.data.joint_vel[0].detach().cpu().tolist()],
                        "joint_target_error_rad": [round(float(v), 8) for v in (targets[0] - robot.data.joint_pos[0]).detach().cpu().tolist()],
                        "computed_torque_nm": [round(float(v), 8) for v in robot.data.computed_torque[0].detach().cpu().tolist()],
                        "applied_torque_nm": [round(float(v), 8) for v in robot.data.applied_torque[0].detach().cpu().tolist()],
                        "fallen": fallen,
                    }
                )
            if capture:
                root_xy = robot.data.body_pos_w[0, reference_ids[0], :2].detach().cpu().tolist()
                sim.set_camera_view(
                    eye=(1.8 + root_xy[0], -2.4 + root_xy[1], 1.35),
                    target=(root_xy[0], root_xy[1], 0.52),
                )
                sim.render()
                index = len(frames)
                path = frame_dir / f"frame_{index:04d}.png"
                _capture_viewport_frame(viewport, sim, path)
                bbox = _projected_link_bbox(
                    viewport, robot.data.body_pos_w[0].detach().cpu().tolist(),
                    args.video_width, args.video_height,
                )
                frames.append(_inspect_frame(path, bbox, index, (step + 1) * SIM_DT))

        if initial_position is None or initial_system_com is None:
            raise RuntimeError("simulation produced no state samples")
        final_position = robot.data.body_pos_w[0, reference_ids[0]].detach().cpu().tolist()
        joint_tracking = summarize_joint_tracking(
            robot.joint_names,
            max_errors.detach().cpu().tolist(),
            max_error_times.detach().cpu().tolist(),
            max_velocities.detach().cpu().tolist(),
            max_computed_torques.detach().cpu().tolist(),
            max_applied_torques.detach().cpu().tolist(),
            torque_saturation_counts.detach().cpu().tolist(),
            steps,
        )
        metrics = {
            "duration_s": round(steps * SIM_DT, 6), "physics_steps": steps,
            "reset_count": 0, "done_count": fall_count, "fall_count": fall_count,
            "max_abs_action": 0.0, "max_abs_command": 0.0,
            "max_reference_tilt_rad": max(item["reference_tilt_rad"] for item in traces),
            "root_height_drop_m": round(float(initial_position[2]) - min(item["root_x_position_m"][2] for item in traces), 8),
            "horizontal_drift_m": round(math.hypot(final_position[0] - float(initial_position[0]), final_position[1] - float(initial_position[1])), 8),
            "semantic_forward_displacement_m": round(final_position[1] - float(initial_position[1]), 8),
            "semantic_strafe_displacement_m": round(final_position[0] - float(initial_position[0]), 8),
            "max_joint_target_error_rad": round(max_joint_error, 8),
            "first_fall_time_s": round(first_fall_time_s, 6) if first_fall_time_s is not None else None,
            "initial_system_center_of_mass_m": [round(float(v), 8) for v in initial_system_com],
            "minimum_zero_pose_support_positive_y_margin_m": round(
                minimum_positive_y_support_margin_m, 8
            ),
            "support_reference": support_reference,
            "minimum_support_positive_y_margin_m": round(
                minimum_positive_y_support_margin_m, 8
            ),
            "first_zero_pose_support_exit_time_s": (
                round(first_static_support_exit_time_s, 6)
                if first_static_support_exit_time_s is not None else None
            ),
            "first_support_exit_time_s": (
                round(first_static_support_exit_time_s, 6)
                if first_static_support_exit_time_s is not None else None
            ),
            "max_projected_gravity_horizontal_norm": round(max_projected_gravity_horizontal, 8),
            "peak_total_support_force_n": round(peak_total_support_force_n, 8),
            "peak_total_support_force_time_s": (
                round(peak_total_support_force_time_s, 6)
                if peak_total_support_force_time_s is not None else None
            ),
            "peak_support_force_body_weight_ratio": round(
                peak_total_support_force_n / robot_weight_n, 8
            ),
            "worst_joint_target_errors": joint_tracking[:10],
        }
        required = 0.0 if args.smoke else MIN_GATE_DURATION_S
        passed, failures = evaluate_gate(metrics, required_duration_s=required)
        if imported is not None and not imported["passed"]:
            failures.append("imported PhysX/USD axes do not preserve the current URDF axis contract")
        if not runtime_actuators["passed"]:
            failures.append("runtime actuator targets, gains, or limits differ from the robot contract")
        if static_pose_derivation is not None:
            if static_pose_derivation["settled_geometry"]["support_margin_m"] <= 0.0:
                failures.append("derived pose COM is outside its exact collision support hull")
            if static_pose_derivation["maximum_required_torque_limit_fraction"] > 1.0:
                failures.append("derived static torque exceeds an imported actuator effort limit")
            support_ratio = static_pose_derivation["mean_support_force_body_weight_ratio"]
            if not 0.8 <= support_ratio <= 1.2:
                failures.append(
                    "fixed-root settling contact force does not support 0.8-1.2 body weights"
                )
            if static_pose_derivation["maximum_mean_abs_settling_velocity_radps"] > 0.5:
                failures.append("fixed-root pose did not settle below 0.5 rad/s mean joint speed")

        video = None
        if args.phase == "proof":
            if not frames:
                failures.append("viewport produced no proof frames")
            else:
                from PIL import Image, ImageChops, ImageDraw, ImageStat
                video_path = output_dir / ("proof_smoke.mp4" if args.smoke else "proof.mp4")
                sheet_path = output_dir / ("contact_sheet_smoke.png" if args.smoke else "contact_sheet.png")
                _encode_video(frame_dir, video_path, args.video_fps)
                indices = sorted({0, len(frames) // 2, len(frames) - 1})
                images = [Image.open(frame_dir / f"frame_{index:04d}.png").convert("RGB") for index in indices]
                sheet = Image.new("RGB", (args.video_width * len(images), args.video_height + 34), "white")
                draw = ImageDraw.Draw(sheet)
                for column, (index, image) in enumerate(zip(indices, images)):
                    sheet.paste(image, (column * args.video_width, 34))
                    draw.text((column * args.video_width + 8, 9), f"frame {index} · sim {frames[index]['time_s']:.1f}s", fill="black")
                sheet.save(sheet_path)
                differences = [
                    round(sum(ImageStat.Stat(ImageChops.difference(left, right)).mean) / 3, 6)
                    for left, right in zip(images, images[1:])
                ]
                probe = _probe_video(video_path)
                sampled = [frames[index] for index in indices]
                video = {
                    "capture_path": "active_viewport_LdrColor_AOV",
                    "synthetic_data_camera_graph_used": False,
                    "path": str(video_path.relative_to(model_spec.ALGORITHM_ROOT.parent.parent)),
                    "sha256": _sha256(video_path),
                    "contact_sheet_path": str(sheet_path.relative_to(model_spec.ALGORITHM_ROOT.parent.parent)),
                    "contact_sheet_sha256": _sha256(sheet_path),
                    "frame_count": len(frames), "fps": args.video_fps,
                    "duration_s": float(probe["format"]["duration"]),
                    "expected_duration_s": len(frames) / args.video_fps,
                    "resolution": [args.video_width, args.video_height],
                    "representative_frames": sampled,
                    "representative_mean_absolute_differences": differences,
                    "nonblank_frames_passed": all(item["nonblank"] for item in sampled),
                    "character_visibility_passed": all(item["character_visible"] for item in sampled),
                    "temporal_progression_visible": len(sampled) >= 3 and all(v > 0.005 for v in differences),
                    "behavior_visibility_note": "Stable pose motion is small; embedded time and start/middle/end frames establish full-duration progression.",
                }
                proof_passed, proof_failures = evaluate_proof(video, required_duration_s=required)
                passed = passed and proof_passed
                failures.extend(proof_failures)

        passed = passed and not failures
        experiment = None
        if args.authority_probe:
            stable = (
                metrics["fall_count"] == 0
                and metrics["first_zero_pose_support_exit_time_s"] is None
                and metrics["max_reference_tilt_rad"] <= MAX_REFERENCE_TILT_RAD
            )
            experiment = {
                "id": "zero_pose_upper_body_authority_probe_v1",
                "diagnostic_only": True,
                "independent_variable": "waist and upper-body PD stiffness/damping",
                "held_constant": [
                    "zero joint targets", "ground-aligned root pose", "lower-body PD gains",
                    "effort and velocity limits", "collision geometry", "contact material",
                    "solver timestep and iterations",
                ],
                "runtime_actuator_groups": configured_groups,
                "result_supports_controller_authority_hypothesis": stable,
                "result_supports_posture_or_contact_hypothesis": not stable,
                "interpretation": (
                    "The zero pose remained inside its measured support region with high upper-body holding; "
                    "promote the explicit holding contract before exact validation."
                    if stable
                    else "High upper-body holding did not retain the zero-pose COM over support; keep the "
                    "ground alignment and next change nominal posture/contact equilibrium."
                ),
            }
        elif args.static_pose_probe:
            stable = (
                metrics["fall_count"] == 0
                and metrics["first_support_exit_time_s"] is None
                and metrics["max_reference_tilt_rad"] <= MAX_REFERENCE_TILT_RAD
            )
            experiment = {
                "id": "gravity_static_pose_release_v1",
                "diagnostic_only": True,
                "independent_variable": "derived nonzero standing pose and measured baseline-PD preload",
                "held_constant": [
                    "exact URDF and imported axes", "baseline PD gains and effort limits",
                    "low-authority hand/finger locks", "contact material",
                    "solver timestep and iterations",
                ],
                "derivation": static_pose_derivation,
                "comparison_to_archived_runs": {
                    "ground_aligned_zero_pose": {
                        "first_fall_time_s": 1.212,
                        "first_support_exit_time_s": 0.696,
                        "horizontal_drift_m": 0.40758578,
                    },
                    "broad_authority_rejected": {
                        "first_fall_time_s": 1.156,
                        "first_support_exit_time_s": 0.662,
                        "horizontal_drift_m": 0.43666864,
                    },
                },
                "result_supports_static_pose_hypothesis": stable,
                "interpretation": (
                    "The derived pose retained support after free-root release; adopt it for a longer bounded test."
                    if stable
                    else "The derived pose did not retain support after release; use its settled-pose, torque, "
                    "and contact diagnostics to revise the equilibrium method rather than increasing authority."
                ),
            }
        evidence = {
            "schema_version": 3, "milestone": MILESTONE_ID, "component": args.phase,
            "scope": (
                "diagnostic_experiment"
                if args.authority_probe or args.static_pose_probe
                else "component_only"
            ),
            "status": "passed" if passed else "failed",
            "gate_eligible": not args.smoke and not args.authority_probe and not args.static_pose_probe,
            "milestone_status_changed": False,
            "experiment": experiment,
            "lineage": "clean_restart_2026_08_22",
            "run_identity": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ"),
            "seed": args.seed,
            "checkpoint": {
                "kind": (
                    "derived_static_pose_diagnostic"
                    if args.static_pose_probe
                    else "passive_pd_configuration"
                ),
                "identity": f"robot_spec_sha256:{_sha256(model_spec.ROBOT_SPEC_PATH)}",
                "policy_checkpoint": None,
            },
            "versions": _versions(),
            "simulator": {
                "device": args.device, "physics_dt_s": SIM_DT, "control_dt_s": CONTROL_DT,
                "physics_steps_per_control_step": round(CONTROL_DT / SIM_DT),
                "physx_substeps": 1,
                "solver_position_iterations": 8, "solver_velocity_iterations": 4,
                "single_process": True, "num_envs": 1, "rendering_enabled": args.phase == "proof",
                "camera_sensor_created": False, "usd_cache_reused": args.phase == "proof" or args.reuse_usd_cache,
                "fixed_root_settling_steps": args.settle_steps if args.static_pose_probe else 0,
                "free_root_validation_steps": steps,
            },
            "input": contract["source"],
            "joint_contract": {
                "action_joints": contract["action_joints"], "locked_joints": contract["locked_joints"],
                "joints": contract["joints"], "nominal_pose": contract["nominal_pose"],
                "pd": contract["pd"], "importer_axis_contract": contract["importer_axis_contract"],
                "runtime_importer_axis_evidence": imported,
                "runtime_actuators": runtime_actuators,
                "runtime_actuator_groups": configured_groups,
                "joint_trace_order": list(robot.joint_names),
                "joint_tracking_summary": joint_tracking,
            },
            "contact_timing": {"threshold_n": contact_threshold_n, "bodies": contact_timing},
            "command_contract": contract["frames"], "metrics": metrics, "traces": traces,
            "video_inspection": video, "failures": failures,
            "cumulative_gates": [{"order": 1, "id": MILESTONE_ID, "status": "unresolved_pending_final_assembly"}],
        }
        evidence_path = output_dir / component_artifact_name(args.phase, args.smoke)
        evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({
            "status": evidence["status"], "component": args.phase, "milestone": MILESTONE_ID,
            "metrics": metrics, "evidence": os.fspath(evidence_path),
            "proof_video": video["path"] if video else None, "failures": failures,
        }, indent=2))
        return evidence
    finally:
        if frame_context is not None:
            frame_context.cleanup()


def _parser() -> argparse.ArgumentParser:
    from isaaclab.app import AppLauncher
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("dynamics", "proof"), required=True)
    parser.add_argument("--duration", type=float, default=MIN_GATE_DURATION_S)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--reuse-usd-cache", action="store_true")
    parser.add_argument(
        "--authority-probe",
        action="store_true",
        help="Diagnostic only: keep zero pose/lower-body settings and increase explicit upper-body holding.",
    )
    parser.add_argument(
        "--static-pose-probe",
        action="store_true",
        help="Diagnostic only: settle the derived pose with a pinned root, measure preload, then release.",
    )
    parser.add_argument(
        "--settle-steps",
        type=int,
        default=2000,
        help="Pinned-root settling steps before a --static-pose-probe release (default: 2000).",
    )
    parser.add_argument("--video-fps", type=int, default=5)
    parser.add_argument("--video-width", type=int, default=640)
    parser.add_argument("--video-height", type=int, default=480)
    AppLauncher.add_app_launcher_args(parser)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    from isaaclab.app import AppLauncher
    args = _parser().parse_args(argv)
    if args.duration <= 0 or (args.steps is not None and args.steps <= 0):
        raise SystemExit("duration and steps must be positive")
    if not args.smoke and args.steps is not None and args.steps * SIM_DT < MIN_GATE_DURATION_S:
        raise SystemExit("short --steps runs require --smoke")
    if not args.smoke and args.duration < MIN_GATE_DURATION_S:
        raise SystemExit("exact validation duration must be at least 30 seconds")
    if args.reuse_usd_cache and not args.smoke:
        raise SystemExit("--reuse-usd-cache is restricted to non-promotable smoke runs")
    if args.authority_probe and (args.phase != "dynamics" or not args.smoke):
        raise SystemExit("--authority-probe requires --phase dynamics and --smoke")
    if args.static_pose_probe and (args.phase != "dynamics" or not args.smoke):
        raise SystemExit("--static-pose-probe requires --phase dynamics and --smoke")
    if args.authority_probe and args.static_pose_probe:
        raise SystemExit("authority and static-pose probes are mutually exclusive")
    if args.settle_steps <= 0:
        raise SystemExit("--settle-steps must be positive")
    output_dir = safe_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / component_artifact_name(args.phase, args.smoke)).unlink(missing_ok=True)
    (output_dir / failure_artifact_name(args.phase, args.smoke)).unlink(missing_ok=True)
    (output_dir / traceback_artifact_name(args.phase, args.smoke)).unlink(missing_ok=True)
    if args.phase == "proof":
        (output_dir / ("proof_smoke.mp4" if args.smoke else "proof.mp4")).unlink(missing_ok=True)
        (output_dir / ("contact_sheet_smoke.png" if args.smoke else "contact_sheet.png")).unlink(missing_ok=True)
        preflight_proof(output_dir, smoke=args.smoke, seed=args.seed)
    # Force the phase contract even if the caller inherited ENABLE_CAMERAS.
    os.environ["ENABLE_CAMERAS"] = "1" if args.phase == "proof" else "0"
    # `video` keeps the viewport active headlessly; no CameraCfg/annotator exists.
    args.enable_cameras = args.phase == "proof"
    args.video = args.phase == "proof"
    launcher = None
    try:
        _set_runtime_stage(args, "app_launcher")
        launcher = AppLauncher(args)
        import torch
        torch.manual_seed(args.seed)
        _set_runtime_stage(args, "validator_entry")
        evidence = _run(args, launcher.app)
        return 0 if evidence["status"] == "passed" else 1
    except Exception as error:
        traceback_text = traceback.format_exc()
        print(traceback_text, file=sys.stderr, flush=True)
        try:
            evidence_path, traceback_path = write_failure_evidence(args, error, traceback_text)
            print(
                f"[landau-passive] failure evidence={evidence_path} traceback={traceback_path}",
                file=sys.stderr,
                flush=True,
            )
        except Exception:
            print("[landau-passive] failed to write failure evidence", file=sys.stderr, flush=True)
            traceback.print_exc(file=sys.stderr)
        return 1
    finally:
        if launcher is not None:
            launcher.app.close()


if __name__ == "__main__":
    raise SystemExit(main())
