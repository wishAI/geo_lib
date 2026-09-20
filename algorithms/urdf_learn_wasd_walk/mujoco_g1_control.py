"""Simulation-only playback of Unitree's bundled G1 velocity ONNX policy.

Uses deployment YAML and unchanged scene_g1.xml, without DDS, SDK or hardware code.
The observation and action conventions are traced to the pinned Unitree C++ headers;
this is a positive control, never Landau milestone evidence or proof of new training.
"""
from __future__ import annotations

import argparse
from collections import deque
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import onnxruntime as ort
import yaml


ROOT = Path(__file__).resolve().parents[2]
ALGORITHM = Path(__file__).resolve().parent
EXPECTED_REVISION = "1425b15f73bd4095f0df53709d7c389c3eb9e790"
TERMS = ("base_ang_vel", "projected_gravity", "velocity_commands", "gait_phase",
         "joint_pos_rel", "joint_vel_rel", "last_action")
REFERENCE_FILES = (
    "deploy/include/isaaclab/envs/mdp/observations/observations.h",
    "deploy/include/isaaclab/manager/observation_manager.h",
    "deploy/include/isaaclab/manager/manager_term_cfg.h",
    "deploy/include/isaaclab/envs/mdp/actions/joint_actions.h",
    "deploy/include/isaaclab/envs/manager_based_rl_env.h",
    "deploy/include/unitree_articulation.h",
    "deploy/include/FSM/State_RLBase.h",
    "simulate/src/unitree_sdk2_bridge.h",
    "simulate/config.yaml",
)


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


class DeploymentObservations:
    """C++ term order, clip-before-scale, oldest-first per-term history, reset phase."""
    def __init__(self, cfg):
        self.cfg = cfg
        self.phase = np.float32(0.0)
        self.buffers = {}
        if tuple(cfg["observations"]) != TERMS:
            raise ValueError("Unexpected official observation order; review deployment headers")
        for name, term in cfg["observations"].items():
            self.buffers[name] = deque(maxlen=int(term.get("history_length", 1)))

    def compute(self, angular_velocity, gravity, command, joint_pos, joint_vel,
                last_action, *, reset=False):
        if reset:
            self.phase = np.float32(0.0)
        period = self.cfg["observations"]["gait_phase"]["params"]["period"]
        delta = np.float32(self.cfg["step_dt"]) * (np.float32(1.0) / np.float32(period))
        self.phase = np.fmod(self.phase + delta, np.float32(1.0))
        phase = np.array([math.sin(float(self.phase)*2*math.pi),
                          math.cos(float(self.phase)*2*math.pi)], dtype=np.float32)
        if np.linalg.norm(command) < 0.1:
            phase[:] = 0
        values = dict(zip(TERMS, (angular_velocity, gravity, command, phase,
                      joint_pos - self.cfg["default_joint_pos"], joint_vel, last_action)))
        for name, term in self.cfg["observations"].items():
            value = np.asarray(values[name], dtype=np.float32)
            scale = np.asarray(term.get("scale", 1.0), dtype=np.float32)
            clip = term.get("clip")
            if term.get("scale_first", False):
                value = value * scale
            if clip is not None:
                value = np.clip(value, clip[0], clip[1])
            if not term.get("scale_first", False):
                value = value * scale
            if reset:
                self.buffers[name].clear()
                self.buffers[name].extend([value.copy()] * self.buffers[name].maxlen)
            else:
                self.buffers[name].append(value.copy())
        return np.concatenate([v for name in TERMS for v in self.buffers[name]]).astype(np.float32)


def load_control(repo):
    revision = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    if revision != EXPECTED_REVISION:
        raise ValueError(f"Unreviewed baseline revision {revision}")
    policy_dir = repo / "deploy/robots/g1/config/policy/velocity/v0"
    yaml_path = policy_dir / "params/deploy.yaml"
    onnx_path = policy_dir / "exported/policy.onnx"
    scene_path = repo / "src/assets/robots/unitree_g1/xmls/scene_g1.xml"
    cfg = yaml.safe_load(yaml_path.read_text())
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    ids = np.asarray(cfg["joint_ids_map"], dtype=int)
    if model.nu != 29 or sorted(ids.tolist()) != list(range(29)):
        raise ValueError("Expected complete 29-motor G1 mapping")
    if model.neq or not np.allclose(model.opt.gravity, [0, 0, -9.81]):
        raise ValueError("Unexpected support constraints or gravity")
    joints = model.actuator_trnid[ids, 0]
    names = [model.joint(int(j)).name for j in joints]
    # The official bridge obtains q/dq from the first two motor-sized sensor blocks.
    for motor in range(model.nu):
        if int(model.sensor_objid[motor]) != int(model.actuator_trnid[motor, 0]):
            raise ValueError("Deployment motor/sensor joint order mismatch")
        if int(model.sensor_type[motor]) != int(mujoco.mjtSensor.mjSENS_JOINTPOS):
            raise ValueError("Unexpected deployment joint-position sensor")
        if int(model.sensor_objid[motor+model.nu]) != int(model.actuator_trnid[motor, 0]):
            raise ValueError("Deployment velocity sensor order mismatch")
    qadr, vadr = model.jnt_qposadr[joints], model.jnt_dofadr[joints]
    data.qpos[qadr] = cfg["default_joint_pos"]
    mujoco.mj_forward(model, data)
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    session = ort.InferenceSession(str(onnx_path), sess_options=opts,
                                   providers=["CPUExecutionProvider"])
    input_meta, output_meta = session.get_inputs(), session.get_outputs()
    if len(input_meta) != 1 or input_meta[0].shape != [1, 98] or output_meta[0].shape != [1, 29]:
        raise ValueError(f"Unexpected policy interface: {[(i.name, i.shape) for i in input_meta]}")
    action_cfg = cfg["actions"]["JointPositionAction"]
    if action_cfg["joint_ids"] is not None:
        raise ValueError("Partial action mapping is not supported")
    files = [yaml_path, onnx_path, scene_path] + [repo / p for p in REFERENCE_FILES]
    def include_files(path, seen=None):
        seen = set() if seen is None else seen
        path = path.resolve()
        if path in seen:
            return seen
        seen.add(path)
        for node in ET.parse(path).iter("include"):
            include_files(path.parent / node.attrib["file"], seen)
        return seen
    files.extend(sorted(include_files(scene_path)))
    files.append(scene_path.parent / "g1.xml")  # Training asset, distinct from deployment scene.
    mesh_files = sorted((scene_path.parent / "assets").glob("*"))
    mesh_hashes = {p.name: file_hash(p) for p in mesh_files if p.is_file()}
    provenance = {
        "source_url": "https://github.com/unitreerobotics/unitree_rl_mjlab",
        "revision": revision, "files_sha256": {str(p.relative_to(repo)): file_hash(p) for p in files},
        "mesh_files_sha256": mesh_hashes,
        "mesh_manifest_sha256": hashlib.sha256(json.dumps(mesh_hashes, sort_keys=True).encode()).hexdigest(),
        "script_sha256": file_hash(__file__),
        "packages": {n: importlib.metadata.version(n) for n in ("mujoco", "onnxruntime", "numpy", "PyYAML")},
        "joint_names_policy_order": names, "actuator_ids_policy_order": ids.tolist(),
        "qpos_addresses": qadr.tolist(), "qvel_addresses": vadr.tolist(),
        "physics_dt_s": model.opt.timestep, "control_dt_s": cfg["step_dt"],
        "gravity": model.opt.gravity.tolist(), "equality_constraints": model.neq,
        "total_mass_kg": float(model.body_mass.sum()), "initial_base_height_m": float(data.qpos[2]),
        "policy_input": {"name": input_meta[0].name, "shape": input_meta[0].shape},
        "policy_output": {"name": output_meta[0].name, "shape": output_meta[0].shape},
        "motor_pd": {"kp": cfg["stiffness"], "kd": cfg["damping"],
                     "ctrlrange": model.actuator_ctrlrange[ids].tolist(),
                     "joint_actuatorfrcrange": model.jnt_actfrcrange[joints].tolist()},
        "observation_semantics": "C++ reset computes phase once; first policy observation advances it again. Body-frame IMU angular velocity; inverse IMU quaternion times world down; raw last action. YAML insertion order, per-term oldest-first history, clip then scale.",
        "physics_semantics": "Unmodified official deployment scene; default MuJoCo timestep. PD torque every physics step; model motor and joint limits remain active. No auxiliary support, resets, external forces, or SDK.",
        "baseline_kind": "bundled_official_pretrained_policy_playback_not_new_training",
    }
    return cfg, model, data, session, ids, qadr, vadr, provenance


def run(args):
    start = time.perf_counter()
    repo = args.repo.resolve()
    out = args.output.resolve()
    if not out.is_relative_to(ALGORITHM / "outputs"):
        raise ValueError("Evidence must stay inside this isolated algorithm outputs directory")
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Preserving existing evidence; choose a new output directory: {out}")
    out.mkdir(parents=True, exist_ok=False)
    np.random.seed(args.seed)
    cfg, model, data, session, ids, qadr, vadr, provenance = load_control(repo)
    command = np.asarray([args.forward, args.strafe, args.yaw], dtype=np.float32)
    ranges = cfg["commands"]["base_velocity"]["ranges"]
    for k, name in enumerate(("lin_vel_x", "lin_vel_y", "ang_vel_z")):
        if not ranges[name][0] <= command[k] <= ranges[name][1]:
            raise ValueError(f"Command outside official deployment range: {name}")
    decimation = round(cfg["step_dt"] / model.opt.timestep)
    if not math.isclose(decimation * model.opt.timestep, cfg["step_dt"], abs_tol=1e-10):
        raise ValueError("Control timestep must be an integer number of physics steps")
    obs_builder = DeploymentObservations(cfg)
    previous_action = np.zeros(29, dtype=np.float32)
    def observation(reset=False):
        quat = data.sensor("imu_quat").data
        rotation = np.empty(9)
        mujoco.mju_quat2Mat(rotation, quat)
        gravity = rotation.reshape(3, 3).T @ np.array([0., 0., -1.])
        return obs_builder.compute(data.sensor("imu_gyro").data.copy(), gravity,
                                   command, data.qpos[qadr].copy(), data.qvel[vadr].copy(),
                                   previous_action, reset=reset)
    observation(reset=True)
    first_obs = observation()
    if first_obs.shape != (98,):
        raise ValueError("Observation does not match bundled policy")
    provenance["first_observation"] = first_obs.tolist()
    provenance["reset_phase"] = float(obs_builder.phase)
    provenance["exact_command"] = sys.argv
    provenance["seed"] = args.seed
    provenance["command"] = command.tolist()
    provenance["requested_duration_s"] = args.duration
    provenance["startup_s"] = time.perf_counter() - start
    write_json(out / "provenance.json", provenance)
    if args.audit_only:
        write_json(out / "result.json", {"status": "audit_only_no_simulation", "simulation_steps": 0,
                   "provenance": str(out / "provenance.json"), "gate_passed": False})
        return 0
    if not 0 < args.duration <= 120:
        raise ValueError("Playback must be bounded to (0, 120] seconds")
    if not math.isclose(round(args.duration/cfg["step_dt"])*cfg["step_dt"], args.duration, abs_tol=1e-9):
        raise ValueError("Duration must be a multiple of control dt")
    renderer = writer = None
    video_error = None
    if args.video:
        import imageio.v2 as imageio
        try:
            renderer = mujoco.Renderer(model, height=480, width=640)
            writer = imageio.get_writer(out / "proof.mp4", fps=25, codec="libx264")
        except Exception as exc:
            if renderer is not None:
                renderer.close()
            renderer = None
            video_error = f"{type(exc).__name__}: {exc}"
    camera = mujoco.MjvCamera()
    camera.distance, camera.azimuth, camera.elevation = 3.2, 135, -18
    kp, kd = np.asarray(cfg["stiffness"]), np.asarray(cfg["damping"])
    action_cfg = cfg["actions"]["JointPositionAction"]
    action_scale, offset = np.asarray(action_cfg["scale"]), np.asarray(action_cfg["offset"])
    floor = model.geom("floor").id
    foot_bodies = [model.body(f"{side}_ankle_roll_link").id for side in ("left", "right")]
    initial_pos = data.qpos[:3].copy()
    rows, observations, actions, positions = [], [], [], []
    max_external = max_torque = max_tilt = 0.
    min_height = float(data.qpos[2])
    contact_transitions = np.zeros(2, dtype=int)
    previous_contact = np.ones(2, dtype=bool)
    swing_durations = np.zeros(2)
    completed_swings = np.zeros(2, dtype=int)
    maximum_swing_s = np.zeros(2)
    slip_square_sum = np.zeros(2)
    slip_count = np.zeros(2, dtype=int)
    maximum_slip = np.zeros(2)
    maximum_foot_lift = np.zeros(2)
    initial_foot_z = data.xpos[foot_bodies, 2].copy()
    current_flight_s = max_flight_s = total_flight_s = 0.0
    physics_trace = []
    jacp, jacr = np.empty((3, model.nv)), np.empty((3, model.nv))
    failure = None
    tilt = 0.0
    run_start = time.perf_counter()
    cpu_start = time.process_time()
    inference_s = physics_s = render_s = 0.
    physics_steps = 0
    def record_frame(warm=False):
        nonlocal render_s, renderer, writer, video_error
        if renderer is not None and writer is not None:
            t = time.perf_counter()
            try:
                camera.lookat[:] = data.qpos[:3]
                renderer.update_scene(data, camera=camera)
                frame = renderer.render()
                if not warm:
                    writer.append_data(frame)
            except Exception as exc:
                video_error = f"{type(exc).__name__}: {exc}"
                writer.close()
                renderer.close()
                renderer = writer = None
            render_s += time.perf_counter()-t
    try:
        record_frame(warm=True)  # Warm offscreen context before the first proof frame.
        record_frame()
        for control_step in range(round(args.duration/cfg["step_dt"])):
            obs = first_obs if control_step == 0 else observation()
            t = time.perf_counter()
            action = session.run(None, {session.get_inputs()[0].name: obs[None]})[0][0]
            inference_s += time.perf_counter()-t
            if not np.isfinite(action).all():
                failure = "nonfinite_action"
                break
            target = action*action_scale+offset
            if action_cfg["clip"] is not None:
                limits = np.asarray(action_cfg["clip"])
                target = np.clip(target, limits[:, 0], limits[:, 1])
            previous_action = action.copy()
            observations.append(obs.copy()); actions.append(action.copy())
            t = time.perf_counter()
            for _ in range(decimation):
                data.ctrl[ids] = kp*(target-data.qpos[qadr])-kd*data.qvel[vadr]
                mujoco.mj_step(model, data)
                physics_steps += 1
                max_external = max(max_external, float(np.max(np.abs(data.xfrc_applied))),
                                   float(np.max(np.abs(data.qfrc_applied))))
                max_torque = max(max_torque, float(np.max(np.abs(data.actuator_force))))
                if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                    failure = "nonfinite_state"
                    break
                quat = data.qpos[3:7]
                tilt = math.acos(float(np.clip(1-2*(quat[1]**2+quat[2]**2), -1, 1)))
                max_tilt = max(max_tilt, tilt)
                min_height = min(min_height, float(data.qpos[2]))
                if data.qpos[2] < 0.45 or tilt > 1.0:
                    failure = "fall_height_or_orientation"
                    break
                contacts = np.zeros(2, dtype=bool)
                slip_samples = [[], []]
                for c in data.contact:
                    if floor not in (c.geom1, c.geom2):
                        continue
                    other = c.geom2 if c.geom1 == floor else c.geom1
                    body = int(model.geom_bodyid[other])
                    if body not in foot_bodies:
                        failure = "nonfoot_ground_contact"
                    for i, foot in enumerate(foot_bodies):
                        if body == foot:
                            contacts[i] = True
                            mujoco.mj_jac(model, data, jacp, jacr, c.pos, body)
                            slip_samples[i].append(float(np.linalg.norm((jacp @ data.qvel)[:2])))
                slip = np.asarray([max(samples, default=0.) for samples in slip_samples])
                maximum_slip = np.maximum(maximum_slip, slip)
                slip_square_sum += slip**2 * contacts
                slip_count += contacts.astype(int)
                contact_transitions += contacts != previous_contact
                previous_contact = contacts.copy()
                for i in range(2):
                    if contacts[i]:
                        if swing_durations[i] >= 0.05:
                            completed_swings[i] += 1
                        swing_durations[i] = 0.
                    else:
                        swing_durations[i] += model.opt.timestep
                maximum_swing_s = np.maximum(maximum_swing_s, swing_durations)
                if not contacts.any():
                    current_flight_s += model.opt.timestep
                    total_flight_s += model.opt.timestep
                    max_flight_s = max(max_flight_s, current_flight_s)
                else:
                    current_flight_s = 0.
                lift = data.xpos[foot_bodies, 2] - initial_foot_z
                maximum_foot_lift = np.maximum(maximum_foot_lift, lift)
                physics_trace.append([data.time, *contacts.astype(int), *slip, *lift])
                if failure:
                    break
                if any(w.number for w in data.warning):
                    failure = "mujoco_warning"
                    break
            physics_s += time.perf_counter()-t
            rows.append([data.time, *data.qpos[:3], tilt, *contacts.astype(int), max_external])
            positions.append(data.qpos[qadr].copy())
            if (control_step+1) % 2 == 0:
                record_frame()
            if failure:
                break
    finally:
        if writer is not None:
            writer.close()
        if renderer is not None:
            renderer.close()
    elapsed = time.perf_counter()-run_start
    cpu_s = time.process_time()-cpu_start
    displacement = data.qpos[:3]-initial_pos
    ranges_q = np.ptp(np.array(positions), axis=0) if len(positions)>1 else np.zeros(29)
    moving = np.linalg.norm(command) >= 0.1
    slip_rms = np.sqrt(slip_square_sum / np.maximum(slip_count, 1))
    dynamics_pass = (failure is None and data.time >= args.duration-1e-8 and max_external == 0)
    if moving:
        dynamics_pass = bool(dynamics_pass and displacement[0] >= min(5., args.forward*args.duration*0.6)
                            and min(completed_swings) >= 3 and np.count_nonzero(ranges_q[:12] > 0.1) >= 4
                            and max_flight_s <= 0.12 and max(slip_rms) <= 0.2
                            and min(maximum_foot_lift) >= 0.025)
    np.savez_compressed(out / "trajectory.npz", state=np.asarray(rows), observations=np.asarray(observations),
                        actions=np.asarray(actions), joint_pos=np.asarray(positions),
                        physics_metrics=np.asarray(physics_trace))
    result = {"status": "completed" if failure is None else "failed", "failure": failure,
              "positive_control_dynamics_passed": bool(dynamics_pass), "landau_gate_passed": False,
              "visual_review": "pending" if args.video and not video_error else "unavailable",
              "provenance": str(out/"provenance.json"), "video": str(out/"proof.mp4") if args.video and not video_error else None,
              "video_error": video_error, "duration_s": float(data.time), "requested_duration_s": args.duration,
              "reset_count": 0, "done_count": int(failure is not None), "fall_count": int(failure is not None and 'fall' in failure),
              "displacement_m": displacement.tolist(), "max_tilt_rad": max_tilt, "min_base_height_m": min_height,
              "contact_transitions": contact_transitions.tolist(),
              "completed_swings_at_least_50ms": completed_swings.tolist(),
              "maximum_single_foot_swing_s": maximum_swing_s.tolist(),
              "foot_contact_slip_rms_m_s": slip_rms.tolist(), "max_foot_slip_m_s": maximum_slip.tolist(),
              "max_foot_lift_m": maximum_foot_lift.tolist(),
              "max_both_feet_flight_s": max_flight_s, "total_both_feet_flight_s": total_flight_s,
              "criteria": {"min_completed_swings_each": 3, "max_flight_s": 0.12,
                           "max_slip_rms_m_s": 0.2, "min_foot_lift_m": 0.025}, "joint_peak_to_peak_rad": ranges_q.tolist(),
              "max_auxiliary_force_or_torque": max_external, "assistance_coefficient": 0.,
              "max_motor_torque_nm": max_torque, "physics_steps": physics_steps,
              "control_transitions": len(rows), "startup_s": provenance["startup_s"], "compilation_s": 0.,
              "rollout_wall_s": elapsed, "inference_s": inference_s, "physics_s": physics_s, "render_s": render_s,
              "physics_steps_per_s_excluding_render": physics_steps/max(elapsed-render_s, 1e-9),
              "control_transitions_per_s_excluding_render": len(rows)/max(elapsed-render_s, 1e-9),
              "process_cpu_percent": 100*cpu_s/max(elapsed, 1e-9),
              "trajectory_sha256": file_hash(out/"trajectory.npz")}
    try:
        import resource
        result["max_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    except ImportError:
        pass
    write_json(out / "result.json", result)
    print(json.dumps(result, indent=2))
    return 0 if dynamics_pass else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=ROOT/"helper_repos/unitree_rl_mjlab")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--duration", type=float, default=30.)
    parser.add_argument("--forward", type=float, default=0.5)
    parser.add_argument("--strafe", type=float, default=0.)
    parser.add_argument("--yaw", type=float, default=0.)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--video", action="store_true")
    args = parser.parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
