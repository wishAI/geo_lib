"""M2 reset baseline derived with the promoted M1 gravity-settling procedure.

Only ``initialize`` imports torch. The limit/provenance audit is usable without Kit.
"""

from __future__ import annotations

import math

from algorithms.urdf_learn_wasd_walk import model_spec, passive_stand

METHOD = "m1_derived_static_settling_policy_initialization_v1"
SETTLE_STEPS = 2000
AVERAGING_STEPS = 250
PHYSICS_DT = 0.002


def protocol() -> dict:
    return {
        "method": METHOD,
        "source_method": "gravity_static_pose_release_v1",
        "fixed_root_steps": SETTLE_STEPS,
        "fixed_root_duration_s": SETTLE_STEPS * PHYSICS_DT,
        "averaging_steps": AVERAGING_STEPS,
        "averaging_duration_s": AVERAGING_STEPS * PHYSICS_DT,
        "seed_target": "derived_position - free_gravity_torque / stiffness",
        "release_target": "mean_settled_position + mean_applied_torque / stiffness",
        "reset": "restore each environment's measured baseline before existing perturbations",
        "settling_counts_toward_policy_duration": False,
        "fixed_root_contact_load_is_gating": False,
        "locked_joint_authority": "unchanged baseline PD",
    }


def validate_report(report: dict, *, num_envs: int, prior: dict, source: dict) -> None:
    """Reject evidence made with a missing/different initialization before checkpoint use."""
    if any(report.get(key) != value for key, value in protocol().items()):
        raise ValueError("policy initialization protocol differs from promoted M1 settling")
    if (report.get("status") != "initialized_not_validated"
            or report.get("policy_steps_during_settling") != 0
            or report.get("prior_gate") != prior or report.get("input") != source):
        raise ValueError("policy initialization provenance or episode clock differs")
    environments = report.get("environments", [])
    if num_envs <= 0 or [x.get("env_id") for x in environments] != list(range(num_envs)):
        raise ValueError("policy initialization lacks per-environment release evidence")
    for item in environments:
        margin = item.get("support_margin_m", math.nan)
        if not math.isfinite(margin) or margin <= 0:
            raise ValueError("policy initialization support margin failed")


def audit_release(names, positions, torques, velocities, stiffness, limits, efforts):
    """Reject invalid mechanics; clamp only the same negligible finger noise as M1."""
    arrays = (positions, torques, velocities, stiffness, limits, efforts)
    if len(set(names)) != len(names) or any(len(x) != len(names) for x in arrays):
        raise ValueError("settling joint arrays are not uniquely aligned")
    records = []
    for name, q, tau, speed, kp, bounds, effort in zip(names, *arrays):
        if not all(math.isfinite(x) for x in (q, tau, speed, kp, effort, *bounds)):
            raise ValueError(f"non-finite settling sample: {name}")
        if kp <= 0 or effort <= 0 or speed < 0 or bounds[0] > bounds[1]:
            raise ValueError(f"invalid settling mechanical contract: {name}")
        if abs(tau) > effort or speed > 0.5:
            raise ValueError(f"unsettled or torque-limited joint: {name}")
        target = q + tau / kp
        desired, target, audit = passive_stand.audit_and_clamp_derived_limit(
            name, q, target, bounds
        )
        if not audit["passed"]:
            raise ValueError(f"derived pose or preload exceeds joint limits: {name}")
        records.append({
            "name": name, "raw_settled_position_rad": q,
            "raw_released_target_rad": q + tau / kp,
            "settled_position_rad": desired, "released_target_rad": target,
            "required_torque_nm": tau, "required_torque_limit_fraction": abs(tau) / effort,
            "mean_abs_settling_velocity_radps": speed, "limit_audit": audit,
        })
    return records


def restore_settled_baseline(env, env_ids):
    """Isaac reset event, before the unchanged root/action-joint perturbation events."""
    robot = env.scene["robot"]
    if not hasattr(env, "_landau_settled_targets"):
        raise RuntimeError("policy reset occurred before gravity-settling initialization")
    ids = slice(None) if env_ids is None else env_ids
    root = robot.data.default_root_state[ids].clone()
    root[:, :3] += env.scene.env_origins[ids]
    robot.write_root_state_to_sim(root, env_ids=env_ids)
    robot.write_joint_state_to_sim(
        robot.data.default_joint_pos[ids], robot.data.default_joint_vel[ids], env_ids=env_ids
    )
    robot.set_joint_position_target(env._landau_settled_targets[ids], env_ids=env_ids)


def initialize(env, prior: dict) -> dict:
    """Settle all clones once before the wrapper's first reset; never call env.step."""
    import torch

    if hasattr(env, "_landau_settled_targets"):
        raise RuntimeError("policy initialization may run only once per environment")
    if abs(env.physics_dt - PHYSICS_DT) > 1e-12:
        raise ValueError("M1 settling requires the audited 0.002 s physics timestep")
    robot = env.scene["robot"]
    spec = model_spec.build_robot_spec()
    seed = model_spec.derive_static_pose()
    names = list(robot.joint_names)
    if set(names) != set(spec["nominal_pose"]["joint_positions_rad"]):
        raise ValueError("settling articulation joint set differs")
    kp = robot.data.joint_stiffness.clone()
    if not bool(torch.all(torch.isfinite(kp) & (kp > 0))):
        raise ValueError("settling requires finite positive stiffness for every joint")
    q = torch.tensor([seed["joint_positions_rad"].get(n, 0.0) for n in names],
                     device=env.device).repeat(env.num_envs, 1)
    gravity = torch.tensor([seed["fixed_root_gravity_torque_nm"][n] for n in names],
                           device=env.device)
    targets = q - gravity / kp
    root = robot.data.default_root_state.clone()
    root[:, :3] += env.scene.env_origins
    root[:, 2] += seed["ground_aligned_base_z_m"] - spec["nominal_pose"]["base_position_m"][2]
    root[:, 7:] = 0
    robot.write_joint_state_to_sim(q, torch.zeros_like(q))
    sums = [torch.zeros_like(q) for _ in range(3)]
    trace = []
    for step in range(SETTLE_STEPS):
        robot.write_root_state_to_sim(root)
        robot.set_joint_position_target(targets)
        env.scene.write_data_to_sim()
        env.sim.step(render=False)
        env.scene.update(PHYSICS_DT)
        if step >= SETTLE_STEPS - AVERAGING_STEPS:
            sums[0] += robot.data.joint_pos
            sums[1] += robot.data.applied_torque
            sums[2] += torch.abs(robot.data.joint_vel)
        if (step + 1) % AVERAGING_STEPS == 0:
            trace.append({"time_s": (step + 1) * PHYSICS_DT,
                          "max_abs_joint_velocity_radps": float(robot.data.joint_vel.abs().max())})
    means = [(value / AVERAGING_STEPS).detach().cpu().tolist() for value in sums]
    stiffness = kp.detach().cpu().tolist()
    limits = robot.data.joint_pos_limits.detach().cpu().tolist()
    efforts = robot.data.joint_effort_limits.detach().cpu().tolist()
    records = []
    for index in range(env.num_envs):
        joints = audit_release(names, *(values[index] for values in means),
                               stiffness[index], limits[index], efforts[index])
        positions = {j["name"]: j["settled_position_rad"] for j in joints}
        # Match M1's collision-derived release height for each clone.
        geometry = model_spec.analyze_pose_geometry(positions)
        margin = geometry["support_margin_m"]
        if not math.isfinite(margin) or margin <= 0:
            raise ValueError(f"settled COM outside support hull in environment {index}: {margin}")
        q[index] = torch.tensor(list(positions.values()), device=env.device)
        targets[index] = torch.tensor([j["released_target_rad"] for j in joints], device=env.device)
        robot.data.default_root_state[index, 2] = geometry["ground_aligned_base_z_m"]
        records.append({"env_id": index, "support_margin_m": margin,
                        "ground_aligned_base_z_m": geometry["ground_aligned_base_z_m"],
                        "support_hull_xy_m": geometry["support_hull_xy_m"],
                        "joints": joints})
    robot.data.default_joint_pos[:] = q
    robot.data.default_joint_vel.zero_()
    robot.data.default_root_state[:, 7:] = 0
    env._landau_settled_targets = targets.clone()
    action = env.action_manager.get_term("joint_pos")
    if list(action._joint_names) != list(model_spec.ACTION_JOINTS):
        raise ValueError("settled action offset order differs")
    # Isaac Lab 0.36.1 clones this offset during action construction.
    action._offset = targets[:, action._joint_ids].clone()
    restore_settled_baseline(env, None)
    env.scene.write_data_to_sim()
    return {**protocol(), "status": "initialized_not_validated", "prior_gate": prior,
            "input": spec["source"], "seed_geometry": seed,
            "environments": records, "settling_trace": trace,
            "policy_steps_during_settling": 0}
