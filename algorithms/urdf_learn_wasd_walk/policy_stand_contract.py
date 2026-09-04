"""Pure-Python contract and evidence checks for the policy stand gate."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from algorithms.urdf_learn_wasd_walk import model_spec


MILESTONE_ID = "stand_30s_no_reset"
PRIOR_MILESTONE_ID = "stand_zero_signal_30s_no_reset"
LINEAGE = "clean_restart_2026_08_22"
OUTPUT_ROOT = model_spec.ALGORITHM_ROOT / "outputs"
DEFAULT_OUTPUT_DIR = OUTPUT_ROOT / MILESTONE_ID
TRAINING_EVIDENCE = "training.json"
MIN_GATE_DURATION_S = 30.0
PHYSICS_DT_S = 0.002
CONTROL_DT_S = 0.02
ACTION_SCALE_RAD = 0.12
ACTION_CLIP = 1.0
MAX_REFERENCE_TILT_RAD = math.radians(30.0)
MAX_ROOT_HEIGHT_DROP_M = 0.08
MAX_HORIZONTAL_DRIFT_M = 0.12

ACTOR_OBSERVATION_TERMS = (
    ("base_linear_velocity", 3),
    ("base_angular_velocity", 3),
    ("projected_gravity", 3),
    ("action_joint_position_error", len(model_spec.ACTION_JOINTS)),
    ("action_joint_velocity", len(model_spec.ACTION_JOINTS)),
    ("previous_action", len(model_spec.ACTION_JOINTS)),
)
ACTOR_OBSERVATION_DIM = sum(width for _, width in ACTOR_OBSERVATION_TERMS)

REWARD_CONTRACT = (
    {"name": "alive", "weight": 1.0},
    {"name": "termination", "weight": -100.0},
    {"name": "planar_velocity_l2", "weight": -2.0},
    {"name": "angular_velocity_l2", "weight": -0.1},
    {"name": "upright_l2", "weight": -4.0},
    {"name": "base_height_l2", "weight": -2.0},
    {"name": "bilateral_foot_contact", "weight": 0.5},
    {"name": "foot_slip", "weight": -0.1},
    {"name": "natural_action_joint_posture_l1", "weight": -0.2},
    {"name": "action_rate_l2", "weight": -0.02},
    {"name": "action_l2", "weight": -0.05},
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


def component_artifact_name(phase: str, smoke: bool = False) -> str:
    if phase not in {"dynamics", "proof"}:
        raise ValueError(f"unsupported policy evidence phase: {phase}")
    return f"{phase}{'_smoke' if smoke else ''}_validation.json"


def failure_artifact_name(mode: str, smoke: bool = False) -> str:
    stem = f"{mode}{'_smoke' if smoke else ''}"
    return f"{stem}_failure.json"


def training_contract(*, seed: int, num_envs: int, iterations: int) -> dict:
    if num_envs <= 0 or iterations <= 0:
        raise ValueError("num_envs and iterations must be positive")
    spec = model_spec.build_robot_spec()
    return {
        "schema_version": 1,
        "lineage": LINEAGE,
        "milestone": MILESTONE_ID,
        "algorithm": "RSL-RL PPO",
        "seed": seed,
        "num_envs": num_envs,
        "iterations": iterations,
        "num_steps_per_env": 24,
        "sample_count": num_envs * iterations * 24,
        "environment": {
            "kind": "Isaac Lab ManagerBasedRLEnv",
            "terrain": "flat_plane",
            "physics_dt_s": PHYSICS_DT_S,
            "control_dt_s": CONTROL_DT_S,
            "episode_length_s": 10.0,
            "commands": {"forward": 0.0, "strafe": 0.0, "yaw": 0.0},
            "action_scale_rad": ACTION_SCALE_RAD,
            "action_clip": [-ACTION_CLIP, ACTION_CLIP],
            "action_joints": list(model_spec.ACTION_JOINTS),
            "locked_joints": spec["locked_joints"],
            "locked_target": "canonical nominal joint position with explicit passive PD actuator",
            "actor_observations": [
                {"name": name, "width": width} for name, width in ACTOR_OBSERVATION_TERMS
            ],
            "actor_observation_dim": ACTOR_OBSERVATION_DIM,
            "privileged_critic_observations": False,
            "reset_perturbations": {
                "root_roll_pitch_rad": [-0.035, 0.035],
                "root_linear_velocity_mps": [-0.05, 0.05],
                "root_angular_velocity_radps": [-0.1, 0.1],
                "action_joint_position_rad": [-0.02, 0.02],
                "action_joint_velocity_radps": [-0.05, 0.05],
            },
            "reward_terms": list(REWARD_CONTRACT),
            "curriculum": None,
            "rough_terrain": False,
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
        "source": spec["source"],
        "robot_spec_sha256": sha256(model_spec.ROBOT_SPEC_PATH),
    }


def evaluate_policy_gate(metrics: dict, *, required_duration_s: float = MIN_GATE_DURATION_S) -> list[str]:
    failures: list[str] = []
    if float(metrics.get("duration_s", -1.0)) + 1.0e-6 < required_duration_s:
        failures.append("policy evaluation is shorter than the required duration")
    for name in ("reset_count", "done_count", "fall_count"):
        if int(metrics.get(name, -1)) != 0:
            failures.append(f"{name}={metrics.get(name)}")
    if float(metrics.get("max_reference_tilt_rad", math.inf)) > MAX_REFERENCE_TILT_RAD:
        failures.append("reference tilt exceeded 30 degrees")
    if float(metrics.get("root_height_drop_m", math.inf)) > MAX_ROOT_HEIGHT_DROP_M:
        failures.append("root height drop exceeded 0.08 m")
    if float(metrics.get("horizontal_drift_m", math.inf)) > MAX_HORIZONTAL_DRIFT_M:
        failures.append("horizontal drift exceeded 0.12 m")
    if float(metrics.get("max_abs_command", math.inf)) != 0.0:
        failures.append("policy stand received a non-zero command")
    action = float(metrics.get("max_abs_action", math.inf))
    if not math.isfinite(action) or action > ACTION_CLIP + 1.0e-6:
        failures.append("policy action was non-finite or escaped its clip contract")
    if int(metrics.get("policy_inference_steps", -1)) != int(metrics.get("control_steps", -2)):
        failures.append("the checkpoint actor did not produce every evaluated action")
    return failures


def load_prior_gate() -> dict:
    """Load and hash-check the exact passed gate-1 evidence declared in milestones."""

    milestones_path = model_spec.ALGORITHM_ROOT / "milestones.json"
    milestones = json.loads(milestones_path.read_text(encoding="utf-8"))
    prior = next((item for item in milestones["milestones"] if item["id"] == PRIOR_MILESTONE_ID), None)
    if prior is None or prior.get("status") != "passed":
        raise ValueError("prior passive gate is not recorded as passed")
    declaration = next(
        (item for item in prior.get("evidence", []) if item.get("kind") == "validation"), None
    )
    if declaration is None:
        raise ValueError("prior passive gate lacks a declared validation artifact")
    path = (model_spec.ALGORITHM_ROOT.parent.parent / declaration["path"]).resolve()
    if not path.is_file() or sha256(path) != declaration["sha256"]:
        raise ValueError("prior passive validation is absent or differs from its declared hash")
    evidence = json.loads(path.read_text(encoding="utf-8"))
    if evidence.get("status") != "passed" or evidence.get("milestone") != PRIOR_MILESTONE_ID:
        raise ValueError("prior passive validation is not a passing gate-1 artifact")
    if evidence.get("input", {}).get("urdf_sha256") != model_spec.EXPECTED_URDF_SHA256:
        raise ValueError("prior passive validation belongs to another URDF")
    if evidence.get("checkpoint", {}).get("identity") != prior.get("checkpoint"):
        raise ValueError("prior passive validation robot specification differs from milestones")
    return {
        "order": 1,
        "id": PRIOR_MILESTONE_ID,
        "status": "passed",
        "path": declaration["path"],
        "sha256": declaration["sha256"],
        "checkpoint": prior["checkpoint"],
        "urdf_sha256": model_spec.EXPECTED_URDF_SHA256,
    }


def load_training_evidence(output_dir: Path) -> dict:
    output_dir = safe_output_dir(output_dir)
    path = output_dir / TRAINING_EVIDENCE
    if not path.is_file():
        raise ValueError(f"missing policy training evidence: {path}")
    evidence = json.loads(path.read_text(encoding="utf-8"))
    if evidence.get("status") != "completed_not_promoted" or evidence.get("milestone") != MILESTONE_ID:
        raise ValueError("policy training evidence is not a completed milestone-2 run")
    if evidence.get("lineage") != LINEAGE:
        raise ValueError("policy training evidence belongs to another lineage")
    if evidence.get("input", {}).get("urdf_sha256") != model_spec.EXPECTED_URDF_SHA256:
        raise ValueError("policy training evidence belongs to another URDF")
    if evidence.get("robot_spec_sha256") != sha256(model_spec.ROBOT_SPEC_PATH):
        raise ValueError("policy training evidence belongs to another robot contract")
    checkpoint_record = evidence.get("checkpoint", {})
    checkpoint = (model_spec.ALGORITHM_ROOT.parent.parent / checkpoint_record.get("path", "")).resolve()
    try:
        checkpoint.relative_to(output_dir)
    except ValueError as error:
        raise ValueError("policy checkpoint escaped its milestone output directory") from error
    if checkpoint.suffix != ".pt" or not checkpoint.is_file():
        raise ValueError("policy checkpoint is missing or has the wrong file type")
    if sha256(checkpoint) != checkpoint_record.get("sha256"):
        raise ValueError("policy checkpoint differs from its training evidence hash")
    evidence["checkpoint"]["resolved_path"] = str(checkpoint)
    return evidence
