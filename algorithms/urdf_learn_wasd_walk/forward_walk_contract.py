"""Pure-Python contract for Landau's cumulative 5 m forward gate."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from algorithms.urdf_learn_wasd_walk import model_spec


MILESTONE_ID = "gate_5m_no_reset"
LINEAGE = model_spec.LINEAGE
OUTPUT_ROOT = model_spec.ALGORITHM_ROOT / "outputs"
DEFAULT_OUTPUT_DIR = OUTPUT_ROOT / MILESTONE_ID
TRAINING_EVIDENCE = "training.json"
PARENT_MILESTONE_ID = "stand_30s_no_reset"
PARENT_CHECKPOINT_SHA256 = "8fd649e9458caa02d4fb176cf3af5ad4da33d2974268e222210cd3080d54cb64"
PHYSICS_DT_S = 0.002
CONTROL_DT_S = 0.02
ACTION_SCALE_RAD = 0.12
ACTION_CLIP = 1.0
TRAINING_METHOD_ID = "phase_gait_l2_v2"
GAIT_PERIOD_S = 0.8
TARGET_FORWARD_SPEED_MPS = 0.4
TRAIN_FORWARD_SPEED_RANGE_MPS = (0.25, 0.45)
TARGET_DISTANCE_M = 5.0
MAX_GATE_DURATION_S = 30.0
STAND_DURATION_S = 30.0
MAX_REFERENCE_TILT_RAD = math.radians(30.0)
MAX_ROOT_HEIGHT_DROP_M = 0.08
MAX_STRAFE_DISPLACEMENT_M = 0.75
MIN_LEG_JOINT_EXCURSION_RAD = 0.05
MAX_MEAN_CONTACT_SLIP_MPS = 0.3
MAX_SIMULTANEOUS_AIR_FRACTION = 0.05
FORWARD_PROGRESS_VELOCITY_EPS_MPS = 0.01
MIN_TRAINING_DIAGNOSTIC_PROGRESS_M = 0.1

NEXT_TRAINING_HYPOTHESIS = {
    "id": "command_gated_phase_reference_residual_v3",
    "hypothesis": (
        "phase observations and sparse contact rewards do not provide enough exploration from the "
        "stable stand policy; a small command-gated, phase-conditioned leg reference can seed "
        "alternating weight transfer while PPO learns bounded residuals"
    ),
    "first_experiment": (
        "camera-free short open-loop reference probe before any PPO run, requiring bilateral "
        "liftoff, positive +Y progress, no reset/fall, and bounded target error"
    ),
    "stand_retention": "the phase reference amplitude is exactly zero for a zero command",
}

ACTOR_OBSERVATION_TERMS = (
    ("base_linear_velocity", 3),
    ("base_angular_velocity", 3),
    ("projected_gravity", 3),
    ("action_joint_position_error", len(model_spec.ACTION_JOINTS)),
    ("action_joint_velocity", len(model_spec.ACTION_JOINTS)),
    ("previous_action", len(model_spec.ACTION_JOINTS)),
    ("semantic_velocity_command", 3),
    ("gait_phase", 2),
)
ACTOR_OBSERVATION_DIM = sum(width for _, width in ACTOR_OBSERVATION_TERMS)

REWARD_CONTRACT = (
    {"name": "termination", "weight": -100.0},
    {"name": "forward_velocity_tracking_l2", "weight": -2.0},
    {"name": "signed_forward_progress", "weight": 1.0},
    {"name": "yaw_velocity_l2", "weight": -0.5},
    {"name": "upright_posture", "weight": -2.0},
    {"name": "base_height", "weight": -1.0},
    {"name": "alternating_single_support", "weight": 1.0},
    {"name": "feet_air_time", "weight": 0.5},
    {"name": "foot_slip", "weight": -0.15},
    {"name": "natural_posture", "weight": -0.1},
    {"name": "action_rate", "weight": -0.01},
    {"name": "action_magnitude", "weight": -0.01},
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_output_dir(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    try:
        resolved.relative_to(OUTPUT_ROOT.resolve())
    except ValueError as error:
        raise ValueError(f"Output directory must stay below {OUTPUT_ROOT}") from error
    return resolved


def semantic_command_to_sim(forward: float, strafe: float = 0.0, yaw: float = 0.0) -> tuple[float, float, float]:
    return model_spec.semantic_to_sim_command(forward, strafe, yaw)


def episode_phase_step_buffer(env):
    """Return the per-env episode clock only after Isaac Lab has created it."""

    return getattr(env, "episode_length_buf", None)


def summarize_forward_velocity_samples(
    samples_mps: list[float], command_samples_mps: list[float], *, control_dt_s: float
) -> dict:
    """Summarize measured body +Y velocity without changing gate acceptance."""

    if not samples_mps or len(samples_mps) != len(command_samples_mps):
        raise ValueError("forward velocity and command samples must be non-empty and aligned")
    if control_dt_s <= 0.0:
        raise ValueError("control_dt_s must be positive")
    positive = [value > FORWARD_PROGRESS_VELOCITY_EPS_MPS for value in samples_mps]
    reverse = [value < -FORWARD_PROGRESS_VELOCITY_EPS_MPS for value in samples_mps]
    first_positive_index = next((index for index, value in enumerate(positive) if value), None)
    count = len(samples_mps)
    return {
        "mean_semantic_forward_velocity_mps": round(sum(samples_mps) / count, 8),
        "minimum_semantic_forward_velocity_mps": round(min(samples_mps), 8),
        "maximum_semantic_forward_velocity_mps": round(max(samples_mps), 8),
        "mean_abs_semantic_forward_velocity_tracking_error_mps": round(
            sum(abs(value - command) for value, command in zip(samples_mps, command_samples_mps))
            / count,
            8,
        ),
        "forward_progress_step_fraction": round(sum(positive) / count, 8),
        "reverse_motion_step_fraction": round(sum(reverse) / count, 8),
        "first_positive_forward_velocity_time_s": (
            round((first_positive_index + 1) * control_dt_s, 6)
            if first_positive_index is not None
            else None
        ),
    }


def analyze_failed_forward_evidence(evidence: dict) -> dict:
    """Extract the failure signature that justifies a changed training method."""

    if (
        evidence.get("status") != "failed"
        or evidence.get("milestone") != MILESTONE_ID
        or evidence.get("component") != "forward"
    ):
        raise ValueError("resume diagnosis requires failed exact forward evidence")
    action_traces = evidence.get("action_traces", [])
    if len(action_traces) < 2:
        raise ValueError("resume diagnosis requires action traces")
    joint_order = action_traces[0].get("joint_order", [])
    if joint_order != list(model_spec.ACTION_JOINTS):
        raise ValueError("resume diagnosis action order differs from the robot contract")
    flips = {}
    for index, name in enumerate(joint_order):
        values = [float(item["raw_policy_action"][index]) for item in action_traces]
        flips[name] = round(
            sum(left * right < 0.0 for left, right in zip(values, values[1:]))
            / (len(values) - 1),
            8,
        )
    sagittal_names = [
        name for name in joint_order
        if name.endswith(("hip_pitch_joint", "knee_joint", "ankle_pitch_joint"))
    ]
    metrics = evidence.get("metrics", {})
    peak_sagittal_flip_fraction = max(flips[name] for name in sagittal_names)
    no_liftoff = (
        int(metrics.get("left_foot_liftoff_count", -1)) == 0
        and int(metrics.get("right_foot_liftoff_count", -1)) == 0
    )
    period_two = peak_sagittal_flip_fraction >= 0.9
    return {
        "classification": "period_two_no_liftoff" if period_two and no_liftoff else "other",
        "period_two_sagittal_action_oscillation": period_two,
        "both_feet_without_liftoff": no_liftoff,
        "peak_sagittal_action_sign_flip_fraction": peak_sagittal_flip_fraction,
        "action_sign_flip_fraction_by_joint": flips,
        "semantic_forward_displacement_m": metrics.get("semantic_forward_displacement_m"),
        "mean_semantic_forward_velocity_mps": metrics.get("mean_semantic_forward_velocity_mps"),
        "reverse_motion_step_fraction": metrics.get("reverse_motion_step_fraction"),
        "actor_output_clipped_value_count": metrics.get("actor_output_clipped_value_count"),
        "hypothesis": (
            "the bounded instantaneous exponential tracking reward admits a profitable alternating "
            "velocity exploit and supplies no direct alternating swing schedule"
        ),
        "next_method": TRAINING_METHOD_ID,
    }


def evaluate_training_smoke_diagnostic(metrics: dict) -> list[str]:
    """Judge gait emergence separately from a relaxed smoke harness status."""

    failures = []
    if float(metrics.get("semantic_forward_displacement_m", -math.inf)) < MIN_TRAINING_DIAGNOSTIC_PROGRESS_M:
        failures.append("training smoke produced less than 0.1 m semantic +Y progress")
    if float(metrics.get("mean_semantic_forward_velocity_mps", -math.inf)) <= 0.0:
        failures.append("training smoke mean semantic +Y velocity was not positive")
    if int(metrics.get("left_foot_liftoff_count", 0)) < 1:
        failures.append("training smoke had no left-side liftoff")
    if int(metrics.get("right_foot_liftoff_count", 0)) < 1:
        failures.append("training smoke had no right-side liftoff")
    for name in ("reset_count", "done_count", "fall_count"):
        if int(metrics.get(name, -1)) != 0:
            failures.append(f"training smoke {name}={metrics.get(name)}")
    return failures


def load_cumulative_prior() -> tuple[list[dict], dict]:
    """Hash-check gates 1–2 and return the exact stand checkpoint for initialization."""

    ledger = json.loads((model_spec.ALGORITHM_ROOT / "milestones.json").read_text(encoding="utf-8"))
    if ledger.get("lineage") != LINEAGE:
        raise ValueError("milestone ledger belongs to another lineage")
    records = {item["id"]: item for item in ledger.get("milestones", [])}
    prior = []
    for order, milestone_id in ((1, "stand_zero_signal_30s_no_reset"), (2, PARENT_MILESTONE_ID)):
        record = records.get(milestone_id)
        if record is None or record.get("status") != "passed":
            raise ValueError(f"prior milestone {milestone_id} is not passed")
        declaration = next(
            (item for item in record.get("evidence", []) if item.get("kind") == "validation"), None
        )
        if declaration is None:
            raise ValueError(f"prior milestone {milestone_id} lacks validation evidence")
        path = (model_spec.ALGORITHM_ROOT.parent.parent / declaration["path"]).resolve()
        if not path.is_file() or sha256(path) != declaration["sha256"]:
            raise ValueError(f"prior milestone {milestone_id} validation hash differs")
        evidence = json.loads(path.read_text(encoding="utf-8"))
        if evidence.get("status") != "passed" or evidence.get("milestone") != milestone_id:
            raise ValueError(f"prior milestone {milestone_id} validation is not passing")
        if evidence.get("input", {}).get("urdf_sha256") != model_spec.EXPECTED_URDF_SHA256:
            raise ValueError(f"prior milestone {milestone_id} uses another URDF")
        if evidence.get("input", {}).get("mesh_tree_sha256") != model_spec.EXPECTED_MESH_TREE_SHA256:
            raise ValueError(f"prior milestone {milestone_id} uses another visual/collision mesh package")
        prior.append({
            "order": order,
            "id": milestone_id,
            "status": "passed",
            "path": declaration["path"],
            "sha256": declaration["sha256"],
            "checkpoint": record.get("checkpoint"),
        })
    parent = records[PARENT_MILESTONE_ID]["checkpoint"]
    parent_path = (model_spec.ALGORITHM_ROOT.parent.parent / parent["path"]).resolve()
    if (
        parent.get("sha256") != PARENT_CHECKPOINT_SHA256
        or not parent_path.is_file()
        or sha256(parent_path) != PARENT_CHECKPOINT_SHA256
    ):
        raise ValueError("canonical policy-stand checkpoint is absent or differs")
    return prior, {**parent, "resolved_path": str(parent_path)}


def training_contract(
    *, seed: int, num_envs: int, iterations: int, initialization_source: dict | None = None
) -> dict:
    if seed < 0 or num_envs <= 0 or iterations <= 0:
        raise ValueError("seed must be non-negative and sizes positive")
    prior, parent = load_cumulative_prior()
    if initialization_source is None:
        initialization_source = {
            "kind": "expanded_observation_transfer",
            "path": parent["path"],
            "sha256": parent["sha256"],
            "actor_observation_dim": 60,
            "source_milestone": PARENT_MILESTONE_ID,
        }
    source_width = int(initialization_source["actor_observation_dim"])
    if source_width not in {60, 63, ACTOR_OBSERVATION_DIM}:
        raise ValueError(f"unsupported initialization observation width: {source_width}")
    return {
        "schema_version": 1,
        "lineage": LINEAGE,
        "milestone": MILESTONE_ID,
        "algorithm": "RSL-RL PPO",
        "training_method": TRAINING_METHOD_ID,
        "seed": seed,
        "num_envs": num_envs,
        "iterations": iterations,
        "num_steps_per_env": 24,
        "sample_count": num_envs * iterations * 24,
        "initialization": {
            **initialization_source,
            "preserved_observation_prefix_width": source_width,
            "new_input_columns_zero_initialized": ACTOR_OBSERVATION_DIM - source_width,
            "optimizer_state_loaded": False,
        },
        "environment": {
            "kind": "Isaac Lab ManagerBasedRLEnv",
            "terrain": "flat_plane",
            "physics_dt_s": PHYSICS_DT_S,
            "control_dt_s": CONTROL_DT_S,
            "episode_length_s": 16.0,
            "semantic_command_order": list(model_spec.SEMANTIC_COMMAND_ORDER),
            "sim_command_mapping": {"forward": "linear_y", "strafe": "linear_x", "yaw": "angular_z"},
            "training_forward_speed_range_mps": list(TRAIN_FORWARD_SPEED_RANGE_MPS),
            "standing_environment_fraction": 0.25,
            "action_scale_rad": ACTION_SCALE_RAD,
            "action_joints": list(model_spec.ACTION_JOINTS),
            "actor_observations": [
                {"name": name, "width": width} for name, width in ACTOR_OBSERVATION_TERMS
            ],
            "actor_observation_dim": ACTOR_OBSERVATION_DIM,
            "gait_phase_period_s": GAIT_PERIOD_S,
            "gait_phase_observation": ["sin_phase", "cos_phase"],
            "privileged_critic_observations": False,
            "reward_terms": list(REWARD_CONTRACT),
            "rough_terrain": False,
            "curriculum": None,
            "symmetry_augmentation": False,
        },
        "ppo": {
            "actor_hidden_dims": [128, 128],
            "critic_hidden_dims": [128, 128],
            "activation": "elu",
            "initial_action_noise_std": 0.2,
            "learning_rate": 0.001,
            "gamma": 0.99,
            "lambda": 0.95,
            "clip_parameter": 0.2,
            "learning_epochs": 5,
            "mini_batches": 4,
        },
        "source": model_spec.build_robot_spec()["source"],
        "robot_spec_sha256": sha256(model_spec.ROBOT_SPEC_PATH),
        "cumulative_prior": prior,
    }


def load_training_evidence(output_dir: Path) -> dict:
    output_dir = safe_output_dir(output_dir)
    path = output_dir / TRAINING_EVIDENCE
    if not path.is_file():
        raise ValueError(f"missing forward training evidence: {path}")
    evidence = json.loads(path.read_text(encoding="utf-8"))
    if (
        evidence.get("status") != "completed_not_promoted"
        or evidence.get("milestone") != MILESTONE_ID
        or evidence.get("lineage") != LINEAGE
    ):
        raise ValueError("forward training evidence has the wrong status or lineage")
    if evidence.get("input", {}).get("urdf_sha256") != model_spec.EXPECTED_URDF_SHA256:
        raise ValueError("forward training evidence belongs to another URDF")
    if evidence.get("input", {}).get("mesh_tree_sha256") != model_spec.EXPECTED_MESH_TREE_SHA256:
        raise ValueError("forward training evidence belongs to another visual/collision mesh package")
    if evidence.get("robot_spec_sha256") != sha256(model_spec.ROBOT_SPEC_PATH):
        raise ValueError("forward training evidence belongs to another robot contract")
    checkpoint_record = evidence.get("checkpoint", {})
    checkpoint = (model_spec.ALGORITHM_ROOT.parent.parent / checkpoint_record.get("path", "")).resolve()
    try:
        checkpoint.relative_to(output_dir)
    except ValueError as error:
        raise ValueError("forward checkpoint escaped its milestone output directory") from error
    if checkpoint.suffix != ".pt" or not checkpoint.is_file():
        raise ValueError("forward checkpoint is missing or has the wrong file type")
    if sha256(checkpoint) != checkpoint_record.get("sha256"):
        raise ValueError("forward checkpoint differs from training evidence")
    evidence["checkpoint"]["resolved_path"] = str(checkpoint)
    return evidence


def evaluate_repassed_stand(
    metrics: dict, *, required_duration_s: float = STAND_DURATION_S
) -> list[str]:
    failures = []
    if float(metrics.get("duration_s", -1.0)) + 1.0e-6 < required_duration_s:
        failures.append("candidate checkpoint stand replay is shorter than 30 seconds")
    for name in ("reset_count", "done_count", "fall_count"):
        if int(metrics.get(name, -1)) != 0:
            failures.append(f"stand {name}={metrics.get(name)}")
    if float(metrics.get("max_abs_command", math.inf)) != 0.0:
        failures.append("stand replay received a non-zero command")
    if float(metrics.get("max_reference_tilt_rad", math.inf)) > MAX_REFERENCE_TILT_RAD:
        failures.append("stand replay tilt exceeded 30 degrees")
    if float(metrics.get("root_height_drop_m", math.inf)) > MAX_ROOT_HEIGHT_DROP_M:
        failures.append("stand replay root height drop exceeded 0.08 m")
    if float(metrics.get("horizontal_drift_m", math.inf)) > 0.12:
        failures.append("stand replay horizontal drift exceeded 0.12 m")
    if float(metrics.get("max_abs_action", math.inf)) > ACTION_CLIP + 1.0e-6:
        failures.append("stand replay action escaped its clip contract")
    if int(metrics.get("policy_inference_steps", -1)) != int(metrics.get("control_steps", -2)):
        failures.append("stand replay checkpoint actor did not produce every action")
    if required_duration_s > 0.0:
        if metrics.get("first_support_exit_time_s") is not None:
            failures.append("stand replay COM exited the canonical support hull")
        if float(metrics.get("minimum_support_polygon_margin_m", -math.inf)) <= 0.0:
            failures.append("stand replay support margin is not positive")
        if float(metrics.get("peak_support_force_body_weight_ratio", math.inf)) > 3.0:
            failures.append("stand replay peak support force exceeded 3 body weights")
        mean_support = float(metrics.get("mean_support_force_body_weight_ratio", math.inf))
        if not 0.5 <= mean_support <= 1.5:
            failures.append("stand replay mean support force is outside 0.5-1.5 body weights")
    return failures


def evaluate_forward_gate(
    metrics: dict,
    *,
    required_distance_m: float = TARGET_DISTANCE_M,
    require_gait: bool = True,
) -> list[str]:
    failures = []
    for name in ("reset_count", "done_count", "fall_count"):
        if int(metrics.get(name, -1)) != 0:
            failures.append(f"{name}={metrics.get(name)}")
    if float(metrics.get("duration_s", math.inf)) > MAX_GATE_DURATION_S + CONTROL_DT_S:
        failures.append("5 m gate exceeded its 30-second bound")
    if float(metrics.get("semantic_forward_displacement_m", -math.inf)) < required_distance_m:
        failures.append("semantic +Y forward displacement did not reach 5 m")
    if abs(float(metrics.get("semantic_strafe_displacement_m", math.inf))) > MAX_STRAFE_DISPLACEMENT_M:
        failures.append("strafe drift exceeded 0.75 m")
    if float(metrics.get("max_reference_tilt_rad", math.inf)) > MAX_REFERENCE_TILT_RAD:
        failures.append("reference tilt exceeded 30 degrees")
    if float(metrics.get("root_height_drop_m", math.inf)) > MAX_ROOT_HEIGHT_DROP_M:
        failures.append("root height drop exceeded 0.08 m")
    if float(metrics.get("max_abs_command", math.inf)) <= 0.0:
        failures.append("forward gate did not receive a command")
    if float(metrics.get("max_abs_action", math.inf)) > ACTION_CLIP + 1.0e-6:
        failures.append("forward action escaped its clip contract")
    if require_gait:
        if int(metrics.get("left_foot_liftoff_count", 0)) < 1 or int(metrics.get("right_foot_liftoff_count", 0)) < 1:
            failures.append("both feet did not demonstrate a swing phase")
        excursions = metrics.get("leg_joint_excursion_rad", {})
        for name in ("left_hip_pitch_joint", "right_hip_pitch_joint", "left_knee_joint", "right_knee_joint"):
            if float(excursions.get(name, -math.inf)) < MIN_LEG_JOINT_EXCURSION_RAD:
                failures.append(f"{name} excursion is too small for walking")
        if float(metrics.get("mean_contact_foot_slip_mps", math.inf)) > MAX_MEAN_CONTACT_SLIP_MPS:
            failures.append("mean contact-foot slip exceeded 0.3 m/s")
        if float(metrics.get("simultaneous_air_fraction", math.inf)) > MAX_SIMULTANEOUS_AIR_FRACTION:
            failures.append("simultaneous foot flight indicates hopping rather than walking")
    if int(metrics.get("policy_inference_steps", -1)) != int(metrics.get("control_steps", -2)):
        failures.append("checkpoint actor did not produce every action")
    return failures


def component_artifact_name(mode: str, smoke: bool = False) -> str:
    names = {
        "stand": "stand_dynamics",
        "forward": "forward_dynamics",
        "proof": "proof",
    }
    if mode not in names:
        raise ValueError(f"unsupported forward component: {mode}")
    return f"{names[mode]}{'_smoke' if smoke else ''}_validation.json"
