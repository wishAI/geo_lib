"""Isolated M2 diagnostic: installed official G1 task versus current Landau.

No result from this runner can promote a Landau milestone. No foreign checkpoint
is accepted. Imports requiring Kit are intentionally inside the runtime path.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithms.urdf_learn_wasd_walk import model_spec

OUTPUT = ROOT / "algorithms/urdf_learn_wasd_walk/outputs/model_comparison"
MODELS = ("unitree_g1", "landau_current")
PROTOCOL = "installed_manager_rsl_rl_comparison_v1"
DEADLINE = "2026-09-06T21:30:00+08:00"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def output_dir(model, experiment):
    if model not in MODELS or not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", experiment):
        raise ValueError("model and experiment must be allowlisted identifiers")
    path = OUTPUT / model / experiment
    path.resolve().relative_to(OUTPUT.resolve())
    return path


def identity(model):
    if model not in MODELS:
        raise ValueError("unknown model")
    return {
        "model": model, "protocol": PROTOCOL, "landau_gate_eligible": False,
        "lineage": f"m2_diagnostic/{PROTOCOL}/{model}/"
                   + (model_spec.EXPECTED_MESH_TREE_SHA256 if model == "landau_current" else "official_g1_minimal"),
        "task": "Isaac-Velocity-Flat-G1-v0" if model == "unitree_g1" else "Landau-M2-stand",
        "semantic_forward_axis": "+X" if model == "unitree_g1" else "+Y",
        "asset_format": "official USD" if model == "unitree_g1" else "URDF + 68 STL meshes",
    }


def validate_checkpoint(evidence, model, directory):
    if any(evidence.get(k) != v for k, v in identity(model).items()):
        raise ValueError("checkpoint model/protocol/lineage differs")
    if evidence.get("status") != "completed_not_promoted":
        raise ValueError("training did not complete")
    checkpoint = (ROOT / evidence["checkpoint"]["path"]).resolve()
    checkpoint.relative_to(directory.resolve())
    if digest(checkpoint) != evidence["checkpoint"]["sha256"]:
        raise ValueError("checkpoint hash differs")
    return checkpoint


def observed_contact(previous, current, airborne_steps, liftoffs):
    """Integer transitions from directly observed forces, independent of float timers."""
    for i, touching in enumerate(current):
        liftoffs[i] += int(previous[i] and not touching)
        airborne_steps[i] = 0 if touching else airborne_steps[i] + 1
    return list(current)


def safe_json(value):
    if isinstance(value, dict):
        return {str(k): safe_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe_json(x) for x in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if callable(value):
        return f"{value.__module__}.{value.__qualname__}"
    return str(value)


def write(path, payload):
    path.write_text(json.dumps(safe_json(payload), indent=2, allow_nan=False) + "\n")


@contextmanager
def exclusive_host():
    # A sandbox PID namespace cannot prove the host is idle.
    if Path("/proc/1/comm").read_text().strip() not in {"init", "systemd"}:
        raise RuntimeError("host process visibility required before starting Isaac")
    deadline = datetime.fromisoformat(DEADLINE)
    if (deadline - datetime.now(deadline.tzinfo)).total_seconds() < 600:
        raise RuntimeError("less than ten minutes remain before the stop boundary")
    if not Path("/dev/nvidiactl").exists():
        raise RuntimeError("host NVIDIA device required")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / ".isaac.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ancestors = {os.getpid()}
        parent = os.getpid()
        while parent > 1:
            stat = Path(f"/proc/{parent}/stat").read_text()
            parent = int(stat.rsplit(")", 1)[1].split()[1])
            ancestors.add(parent)
        conflicts = []
        for path in Path("/proc").glob("[0-9]*/cmdline"):
            # isaaclab.sh and Geo legitimately retain this script in argv.
            # Ignore only our ancestry, never another Isaac child/sibling.
            if int(path.parent.name) in ancestors:
                continue
            try:
                argv = path.read_bytes().decode(errors="replace").split("\0")
            except FileNotFoundError:
                continue
            scripts = [a for a in argv if a.endswith(".py")]
            if (Path(argv[0]).name == "kit" or any(
                Path(s).name in {"passive_stand.py", "policy_stand.py", "forward_walk.py",
                                "forward_reference_probe.py", "model_comparison.py"}
                or "/IsaacLab/" in s for s in scripts)):
                conflicts.append(int(path.parent.name))
        if conflicts:
            raise RuntimeError(f"existing Isaac process: {conflicts}")
        print(json.dumps({"isaac_preflight": "host_idle", "pid": os.getpid()}), flush=True)
        yield


def usd_asset_identity(uri):
    """Hash every composed USD layer and non-layer dependency; never just the root."""
    import omni.client
    from pxr import Sdf, UsdUtils

    layers, assets, unresolved = UsdUtils.ComputeAllDependencies(Sdf.AssetPath(uri))
    if unresolved or not layers:
        raise ValueError(f"official G1 USD has unresolved dependencies: {unresolved}")
    records = []
    for layer in layers:
        records.append({"path": layer.identifier, "kind": "canonical_usda",
                        "sha256": hashlib.sha256(layer.ExportToString().encode()).hexdigest()})
    for asset in assets:
        result, _, content = omni.client.read_file(asset)
        if result != omni.client.Result.OK:
            raise ValueError(f"unreadable G1 dependency: {asset}")
        records.append({"path": asset, "kind": "bytes",
                        "sha256": hashlib.sha256(bytes(content)).hexdigest()})
    records.sort(key=lambda x: x["path"])
    return {"uri": uri, "hash_method": "sorted resolved USD layer text and external asset bytes",
            "asset_tree_sha256": hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest(),
            "dependencies": records, "missing_dependencies": []}


def configs(args):
    if args.model == "unitree_g1":
        from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.flat_env_cfg import G1FlatEnvCfg
        from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.agents.rsl_rl_ppo_cfg import G1FlatPPORunnerCfg
        cfg, agent = G1FlatEnvCfg(), G1FlatPPORunnerCfg()
        asset = usd_asset_identity(cfg.scene.robot.spawn.usd_path)
    else:
        from algorithms.urdf_learn_wasd_walk.policy_stand_env import build_env_cfg
        from algorithms.urdf_learn_wasd_walk.policy_stand import _runner_cfg
        cfg = build_env_cfg(num_envs=args.num_envs, seed=args.seed,
                            training=args.mode == "train", force_usd_conversion=True)
        agent = _runner_cfg(args.seed, args.iterations)
        asset = model_spec.build_robot_spec()["source"]
    cfg.scene.num_envs = args.num_envs if args.mode == "train" else 1
    cfg.seed = agent.seed = args.seed
    cfg.sim.device = agent.device = args.device
    agent.max_iterations = args.iterations
    if args.mode == "evaluate":
        cfg.episode_length_s = max(60., args.steps * cfg.sim.dt * cfg.decimation + 1)
        cfg.observations.policy.enable_corruption = False
        if args.model == "unitree_g1":
            cfg.events.base_external_force_torque = None
            cfg.events.push_robot = None
            cmd = cfg.commands.base_velocity
            cmd.heading_command = False
            cmd.rel_standing_envs = 0.0
            cmd.resampling_time_range = (1000., 1000.)
            cmd.ranges.lin_vel_x = (0.5, 0.5)
            cmd.ranges.lin_vel_y = (0.0, 0.0)
            cmd.ranges.ang_vel_z = (0.0, 0.0)
            cfg.events.reset_base.params["pose_range"] = {}
    return cfg, agent, asset


def joint_frames(stage):
    from pxr import UsdPhysics
    records = []
    for prim in stage.Traverse():
        if not str(prim.GetPath()).startswith("/World/envs/env_0/Robot/") or not prim.IsA(UsdPhysics.RevoluteJoint):
            continue
        joint = UsdPhysics.RevoluteJoint(prim)
        def quat(value):
            return [float(value.GetReal()), *map(float, value.GetImaginary())]
        records.append({"name": prim.GetName(), "axis": str(joint.GetAxisAttr().Get()),
                        "local_rotation_0_wxyz": quat(joint.GetLocalRot0Attr().Get()),
                        "local_rotation_1_wxyz": quat(joint.GetLocalRot1Attr().Get()),
                        "body0": list(map(str, joint.GetBody0Rel().GetTargets())),
                        "body1": list(map(str, joint.GetBody1Rel().GetTargets()))})
    if not records:
        raise ValueError("no imported joint frames found for model axis audit")
    return records


def execute(args, directory):
    import torch
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from rsl_rl.runners import OnPolicyRunner

    args.runtime_stage = "official_configuration_and_asset_dependencies"
    print(f"[model-comparison] {args.runtime_stage}", flush=True)
    cfg, agent, asset = configs(args)
    training = None
    if args.mode == "evaluate":
        training = json.loads((directory / "training.json").read_text())
        checkpoint = validate_checkpoint(training, args.model, directory)
        if training["asset"] != asset:
            raise ValueError("evaluation asset dependencies differ from training")
    write(directory / f"{args.mode}_config.json", {"environment": cfg.to_dict(), "ppo": agent.to_dict(), "asset": asset})
    args.runtime_stage = "manager_environment_construction"
    print(f"[model-comparison] {args.runtime_stage}", flush=True)
    env = ManagerBasedRLEnv(cfg=cfg, render_mode=None)
    try:
        initialization = {"method": "installed_official_default_pose_and_reset", "fixed_root_steps": 0}
        if args.model == "landau_current":
            from algorithms.urdf_learn_wasd_walk.policy_stand_initialization import initialize
            from algorithms.urdf_learn_wasd_walk.policy_stand_contract import load_prior_gate
            initialization = initialize(env, load_prior_gate())
        wrapped = RslRlVecEnvWrapper(env)
        robot = env.scene["robot"]
        term = env.action_manager.get_term("joint_pos")
        contract = {
            "comparison_scope": "same execution loop; native model-specific MDP and PPO parameters recorded, not claimed identical",
            "imported_joint_frames": joint_frames(env.sim.stage),
            "initialization": initialization,
            "joint_names": robot.joint_names, "action_joint_names": term._joint_names,
            "joint_positions": robot.data.default_joint_pos[0].tolist(),
            "joint_limits": robot.data.joint_pos_limits[0].tolist(),
            "stiffness": robot.data.joint_stiffness[0].tolist(),
            "damping": robot.data.joint_damping[0].tolist(),
            "effort_limits": robot.data.joint_effort_limits[0].tolist(),
            "observation_terms": env.observation_manager.active_terms,
            "observation_dimensions": env.observation_manager.group_obs_dim,
            "action_dimension": env.action_manager.total_action_dim,
            "rewards": env.reward_manager.active_terms,
            "terminations": env.termination_manager.active_terms,
            "physics_dt_s": env.physics_dt, "control_dt_s": env.step_dt,
        }
        run_dir = directory / "runs"
        runner = OnPolicyRunner(wrapped, agent.to_dict(), log_dir=str(run_dir) if args.mode == "train" else None,
                                device=args.device)
        payload = {**identity(args.model), "asset": asset, "runtime_contract": contract,
                   "run_identity": datetime.now().astimezone().isoformat(), "mode": args.mode,
                   "experiment": args.experiment, "seed": args.seed}
        from algorithms.urdf_learn_wasd_walk.policy_stand import _versions
        payload["versions"] = _versions()
        if args.mode == "train":
            args.runtime_stage = "ppo_training"
            try:
                runner.learn(num_learning_iterations=args.iterations, init_at_random_ep_len=True)
            except KeyboardInterrupt:
                runner.save(str(run_dir / "interrupted_checkpoint.pt"))
                raise
            checkpoint = max(run_dir.glob("model_*.pt"), key=lambda p: int(p.stem.split("_")[-1]))
            payload.update(status="completed_not_promoted", num_envs=args.num_envs, iterations=args.iterations,
                           checkpoint={"path": str(checkpoint.relative_to(ROOT)), "sha256": digest(checkpoint),
                                       "size_bytes": checkpoint.stat().st_size})
            write(directory / "training.json", payload)
        else:
            args.runtime_stage = "checkpoint_evaluation"
            runner.load(str(checkpoint))
            policy = runner.get_inference_policy(device=args.device)
            obs, _ = wrapped.get_observations()
            sensor = env.scene["contact_forces"]
            expressions = (["left_ankle_roll_link", "right_ankle_roll_link"] if args.model == "unitree_g1"
                           else ["foot_l", "foot_r"])
            ids, names = sensor.find_bodies(expressions, preserve_order=True)
            if len(ids) != 2:
                raise ValueError(f"bilateral contact mapping differs: {names}")
            start = robot.data.root_pos_w[0].clone()
            forward = 0 if args.model == "unitree_g1" else 1
            trace, dones, falls = [], 0, 0
            prior_contact, air, liftoffs, max_air = [False]*2, [0]*2, [0]*2, [0]*2
            max_error = max_tilt = max_drop = 0.
            for step in range(args.steps):
                with torch.inference_mode():
                    action = policy(obs)
                    if not bool(torch.isfinite(action).all()):
                        raise ValueError("non-finite actor output")
                    if args.model == "landau_current":
                        action = torch.clamp(action, -1.0, 1.0)  # Preserve M2's deployable actor contract.
                    obs, _, done, _ = wrapped.step(action)
                dones += int(done.sum())
                falls += int(env.reset_terminated.sum())
                force = sensor.data.net_forces_w[0, ids].norm(dim=-1).tolist()
                prior_contact = observed_contact(prior_contact, [f > 1.0 for f in force], air, liftoffs)
                max_air = [max(a, b) for a, b in zip(max_air, air)]
                max_error = max(max_error, float((robot.data.joint_pos_target-robot.data.joint_pos).abs().max()))
                gravity = robot.data.projected_gravity_b[0]
                max_tilt = max(max_tilt, math.acos(max(-1., min(1., -float(gravity[2])))))
                max_drop = max(max_drop, float(start[2]-robot.data.root_pos_w[0, 2]))
                trace.append({"time_s": (step+1)*env.step_dt, "root_position": robot.data.root_pos_w[0].tolist(),
                              "root_quaternion": robot.data.root_quat_w[0].tolist(),
                              "root_velocity_body": robot.data.root_lin_vel_b[0].tolist(),
                              "support_forces_n": force, "direct_contact": prior_contact,
                              "projected_gravity": robot.data.projected_gravity_b[0].tolist()})
                if dones:
                    break  # The installed RL env already reset; never count post-reset travel.
            displacement = float((robot.data.root_pos_w[0]-start)[forward])
            metrics = {"duration_s": len(trace)*env.step_dt, "reset_count": dones, "done_count": dones,
                       "fall_count": falls, "forward_axis_world_displacement_m": displacement if not dones else None,
                       "left_liftoffs": liftoffs[0], "right_liftoffs": liftoffs[1],
                       "max_air_steps": max_air, "max_joint_target_error_rad": max_error,
                       "max_root_tilt_rad": max_tilt, "root_height_drop_m": max_drop}
            failures = ["reset/done occurred"] if dones else []
            if args.model == "unitree_g1" and (displacement < .1 or min(liftoffs) < 1):
                failures.append("no demonstrated forward bilateral stepping")
            payload.update(status="failed" if failures else "diagnostic_passed", metrics=metrics,
                           failures=failures, trace=trace, checkpoint=training["checkpoint"])
            write(directory / "evaluation.json", payload)
        return payload
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--experiment", default="official_smoke_20260906")
    parser.add_argument("--reference-experiment", default="official_smoke_20260906")
    parser.add_argument("--mode", choices=("train", "evaluate"), required=True)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    directory = output_dir(args.model, args.experiment)
    if not (1 <= args.num_envs <= 512 and 1 <= args.iterations <= 200 and 1 <= args.steps <= 1500):
        raise ValueError("diagnostic exceeds bounded budget")
    artifact = directory / ("training.json" if args.mode == "train" else "evaluation.json")
    if artifact.exists():
        raise ValueError("experiment already recorded; inspect it, do not overwrite")
    if args.model == "landau_current":
        reference = json.loads((output_dir("unitree_g1", args.reference_experiment) / "evaluation.json").read_text())
        if reference.get("status") != "diagnostic_passed" or reference.get("model") != "unitree_g1":
            raise ValueError("run and diagnose the official G1 baseline before Landau transfer")
    if args.iterations > 2:
        smoke_dir = output_dir(args.model, "official_smoke_20260906")
        smoke = json.loads((smoke_dir / "training.json").read_text())
        validate_checkpoint(smoke, args.model, smoke_dir)
    with exclusive_host():
        directory.mkdir(parents=True, exist_ok=True)
        app = None
        try:
            def interrupt(signum, frame):
                raise KeyboardInterrupt(f"graceful stop signal {signum}")
            signal.signal(signal.SIGTERM, interrupt)
            signal.signal(signal.SIGALRM, interrupt)
            remaining = (datetime.fromisoformat(DEADLINE) - datetime.now().astimezone()).total_seconds()
            signal.alarm(max(1, int(remaining - 30)))
            args.enable_cameras = False
            app = AppLauncher(args)
            payload = execute(args, directory)
            return 1 if payload["status"] == "failed" else 0
        except (Exception, KeyboardInterrupt) as error:
            write(directory / f"{args.mode}_failure.json", {**identity(args.model), "status": "failed_to_execute",
                  "run_identity": datetime.now().astimezone().isoformat(),
                  "runtime_stage": getattr(args, "runtime_stage", "app_launcher"),
                  "exception": str(error), "traceback": traceback.format_exc()})
            raise
        finally:
            signal.alarm(0)
            if app is not None:
                app.app.close()


if __name__ == "__main__":
    raise SystemExit(main())
