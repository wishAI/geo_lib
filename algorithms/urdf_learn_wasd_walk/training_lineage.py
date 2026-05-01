from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .asset_paths import outputs_dir
from .isaac_lock import assert_no_active_isaac_lock


_LANDAU_STAGE_SUFFIX = {
    "stand": "stand",
    "full": "flat",
    "fwd_only": "fwd_only",
    "fwd_yaw": "fwd_yaw",
    "game": "game",
}

_STAGE_OBJECTIVES = {
    "stand": "recover and hold an upright idle pose first under zero signal, then under policy control",
    "fwd_only": "walk forward on flat ground and pass the 10 m A-to-B gait gate",
    "fwd_yaw": "add direction changes while preserving forward walking quality",
    "game": "handle rough terrain and obstacles with forward-biased game control",
    "full": "general flat-ground locomotion",
}

_GOAL_LADDER = (
    {
        "id": "stand_zero_signal_30s_no_reset",
        "stage": "stand",
        "title": "Stand for 30 s with zero signal",
        "summary": "Hold an upright idle pose for 30 seconds with zero action and zero commanded motion.",
        "evidence_record_kind": "diagnostic",
        "scenario": {
            "kind": "diagnose",
            "steps": 600,
            "action_mode": "zero",
            "command_vx": 0.0,
            "command_vy": 0.0,
            "command_yaw": 0.0,
        },
        "pass_criteria": {
            "duration_s": 30.0,
            "max_done_count_total": 0,
            "max_hard_resets": 0,
            "require_upright": True,
            "require_zero_falls": True,
            "require_all_prior_milestones": True,
            "require_zero_signal": True,
            "auto_fail_on_hard_reset": True,
        },
    },
    {
        "id": "stand_30s_no_reset",
        "stage": "stand",
        "title": "Stand for 30 s without reset",
        "summary": "Hold an upright idle pose for 30 seconds without a hard reset or fall.",
        "evidence_record_kind": "diagnostic",
        "scenario": {
            "kind": "diagnose",
            "steps": 600,
            "action_mode": "policy",
        },
        "pass_criteria": {
            "duration_s": 30.0,
            "max_done_count_total": 0,
            "max_hard_resets": 0,
            "require_upright": True,
            "require_zero_falls": True,
            "require_all_prior_milestones": True,
            "auto_fail_on_hard_reset": True,
        },
    },
    {
        "id": "gate_5m_no_reset",
        "stage": "fwd_only",
        "title": "Pass the 5 m flat gate",
        "summary": "Walk 5 meters forward on flat ground without falling, resetting, or losing a walking pose.",
        "evidence_record_kind": "evaluation",
        "scenario": {
            "kind": "eval",
            "path_preset": "gate",
            "gate_direction": "forward",
            "path_distance_m": 5.0,
            "terrain_mode": "flat",
        },
        "pass_criteria": {
            "target_distance_m": 5.0,
            "max_done_count_total": 0,
            "max_hard_resets": 0,
            "require_walk_completed": True,
            "require_walking_pose": True,
            "require_arm_swing_review": True,
            "require_all_prior_milestones": True,
            "auto_fail_on_hard_reset": True,
        },
    },
    {
        "id": "gate_10m_no_reset",
        "stage": "fwd_only",
        "title": "Pass the 10 m flat gate",
        "summary": "Walk 10 meters forward on flat ground without falling, resetting, or breaking the walking pose.",
        "evidence_record_kind": "evaluation",
        "scenario": {
            "kind": "eval",
            "path_preset": "gate",
            "gate_direction": "forward",
            "path_distance_m": 10.0,
            "terrain_mode": "flat",
        },
        "pass_criteria": {
            "target_distance_m": 10.0,
            "max_done_count_total": 0,
            "max_hard_resets": 0,
            "require_walk_completed": True,
            "require_walking_pose": True,
            "require_arm_swing_review": True,
            "require_all_prior_milestones": True,
            "auto_fail_on_hard_reset": True,
        },
    },
    {
        "id": "yaw_turn_90deg_hold",
        "stage": "fwd_yaw",
        "title": "Turn then hold heading",
        "summary": "Turn toward the commanded direction, settle, and hold without falling. Angle tolerance may stay loose.",
        "evidence_record_kind": "validation",
        "scenario": {
            "kind": "validate",
            "command_vx": 0.35,
            "command_yaw": 0.30,
        },
        "pass_criteria": {
            "target_heading_change_deg": 90.0,
            "max_done_count_total": 0,
            "max_hard_resets": 0,
            "require_heading_response": True,
            "require_hold_after_turn_s": 5.0,
            "require_all_prior_milestones": True,
            "auto_fail_on_hard_reset": True,
        },
    },
    {
        "id": "teleop_60s_forward_turn",
        "stage": "fwd_yaw",
        "title": "Teleop for 60 s with forward-biased control",
        "summary": "Respond to forward and turn inputs for 60 seconds without falling or locking the joints.",
        "evidence_record_kind": "manual",
        "scenario": {
            "kind": "teleop",
            "duration_s": 60.0,
        },
        "pass_criteria": {
            "duration_s": 60.0,
            "max_hard_resets": 0,
            "require_joint_motion": True,
            "prefer_forward_turning": True,
            "deprioritize_backwards_and_strafe": True,
            "require_all_prior_milestones": True,
            "auto_fail_on_hard_reset": True,
        },
    },
    {
        "id": "gate_10m_four_directions_no_reset",
        "stage": "fwd_yaw",
        "title": "Pass the 10 m gate in four directions",
        "summary": (
            "From the same checkpoint, pass 10 m gates in forward, left, right, and backward directions "
            "without a fall, hard reset, or walking-pose collapse."
        ),
        "evidence_record_kind": "evaluation",
        "scenario": {
            "kind": "eval_suite",
            "path_preset": "gate",
            "gate_directions": ["forward", "left", "right", "backward"],
            "path_distance_m": 10.0,
            "terrain_mode": "flat",
        },
        "pass_criteria": {
            "target_distance_m": 10.0,
            "max_done_count_total": 0,
            "max_hard_resets": 0,
            "require_walking_pose": True,
            "require_arm_swing_review": True,
            "require_all_directions": True,
            "require_all_prior_milestones": True,
            "auto_fail_on_hard_reset": True,
        },
    },
    {
        "id": "triangle_path_follow_no_reset",
        "stage": "fwd_yaw",
        "title": "Follow a triangle path within tolerance",
        "summary": (
            "Use joystick-style forward and yaw commands derived from waypoints to follow a closed triangle path "
            "without falling or leaving the arrival tolerance at each corner."
        ),
        "evidence_record_kind": "evaluation",
        "scenario": {
            "kind": "eval",
            "path_preset": "triangle",
            "path_edge_length_m": 3.5,
            "path_arrival_radius": 0.35,
            "terrain_mode": "flat",
        },
        "pass_criteria": {
            "path_shape": "triangle",
            "path_tolerance_m": 0.35,
            "max_done_count_total": 0,
            "max_hard_resets": 0,
            "require_path_follow": True,
            "require_walking_pose": True,
            "require_all_prior_milestones": True,
            "auto_fail_on_hard_reset": True,
        },
    },
    {
        "id": "square_path_follow_no_reset",
        "stage": "fwd_yaw",
        "title": "Follow a square path within tolerance",
        "summary": (
            "Use joystick-style forward and yaw commands derived from waypoints to follow a closed square path "
            "without falling or leaving the arrival tolerance at each corner."
        ),
        "evidence_record_kind": "evaluation",
        "scenario": {
            "kind": "eval",
            "path_preset": "square",
            "path_edge_length_m": 3.0,
            "path_arrival_radius": 0.35,
            "terrain_mode": "flat",
        },
        "pass_criteria": {
            "path_shape": "square",
            "path_tolerance_m": 0.35,
            "max_done_count_total": 0,
            "max_hard_resets": 0,
            "require_path_follow": True,
            "require_walking_pose": True,
            "require_all_prior_milestones": True,
            "auto_fail_on_hard_reset": True,
        },
    },
    {
        "id": "terrain_5m_no_reset",
        "stage": "game",
        "title": "Traverse 5 m of rough terrain",
        "summary": "Cross the small rough-terrain map for 5 meters before scaling map size or clutter.",
        "evidence_record_kind": "evaluation",
        "scenario": {
            "kind": "eval",
            "path_preset": "gate",
            "gate_direction": "forward",
            "path_distance_m": 5.0,
            "terrain_mode": "game",
            "obstacles": False,
        },
        "pass_criteria": {
            "target_distance_m": 5.0,
            "max_done_count_total": 0,
            "max_hard_resets": 0,
            "require_small_map_first": True,
            "require_walking_pose": True,
            "require_all_prior_milestones": True,
            "auto_fail_on_hard_reset": True,
        },
    },
    {
        "id": "obstacle_stop_before_collision",
        "stage": "game",
        "title": "Stop before obstacle collision",
        "summary": "Approach an obstacle, brake, and stop before collision without a hard reset.",
        "evidence_record_kind": "evaluation",
        "scenario": {
            "kind": "eval",
            "path_preset": "gate",
            "gate_direction": "forward",
            "path_distance_m": 5.0,
            "terrain_mode": "game",
            "obstacles": True,
            "obstacle_brake": True,
        },
        "pass_criteria": {
            "max_hard_resets": 0,
            "require_obstacle_brake": True,
            "require_no_collision": True,
            "require_all_prior_milestones": True,
            "auto_fail_on_hard_reset": True,
        },
    },
    {
        "id": "game_10m_no_reset",
        "stage": "game",
        "title": "Pass the 10 m game gate",
        "summary": "Complete the mixed game gate for 10 meters with forward-biased commands, terrain, and obstacles.",
        "evidence_record_kind": "evaluation",
        "scenario": {
            "kind": "eval",
            "path_preset": "gate",
            "gate_direction": "forward",
            "path_distance_m": 10.0,
            "terrain_mode": "game",
            "obstacles": True,
        },
        "pass_criteria": {
            "target_distance_m": 10.0,
            "max_done_count_total": 0,
            "max_hard_resets": 0,
            "require_walking_pose": True,
            "require_arm_swing_review": True,
            "require_obstacle_brake": True,
            "require_all_prior_milestones": True,
            "auto_fail_on_hard_reset": True,
        },
    },
)

_STAGE_REQUIRED_MILESTONES = {
    stage: [goal["id"] for goal in _GOAL_LADDER if goal["stage"] == stage]
    for stage in _LANDAU_STAGE_SUFFIX
}

_HISTORY_README = """# History Index

This directory is the active source of truth for the current Landau training lineage.

## Layout

- `active_lineage.json`
  - the lineage currently used by default commands when `--experiment_name` is omitted
- `../TRAINING_RULES.md`
  - tracked staged goal ladder and restart rules for future coding agents
- `lineages/*.json`
  - immutable lineage manifests with stage-to-experiment mapping and acceptance goals
- `experiments/*.json`
  - one summary per experiment with the latest training / validation / evaluation / diagnostic records
- `refs/milestones.json`
  - compact milestone registry showing which exact checkpoints first passed the staged gates
- `refs/stage_history.json`
  - append-only stage transition trail for the active lineage
- `archives/*/reset_manifest.json`
  - archived broken or superseded history/log state that was rotated out before a fresh start
- `training_runs.jsonl`
- `validation_runs.jsonl`
- `evaluation_runs.jsonl`
- `diagnostic_runs.jsonl`
- `checkpoint_registry.json`
- `refs/index.json`

## Read Order

1. `active_lineage.json`
2. `refs/index.json`
3. `../TRAINING_RULES.md`
4. the matching lineage file under `lineages/`
5. raw JSONL ledgers for append-only detail

## Common Commands

```bash
./geo walk reset --lineage-name <name>
./geo walk curriculum --from-start --lineage-name <name>
./geo walk refs
./geo walk gate --stage game
./geo walk milestone --milestone-id stand_zero_signal_30s_no_reset --stage stand --load_run <run> --checkpoint model_<n>.pt
```
"""


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _timestamp_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def history_root() -> Path:
    return outputs_dir() / "history"


def lineages_dir() -> Path:
    path = history_root() / "lineages"
    path.mkdir(parents=True, exist_ok=True)
    return path


def archives_dir() -> Path:
    path = outputs_dir() / "archives"
    path.mkdir(parents=True, exist_ok=True)
    return path


def active_lineage_path() -> Path:
    return history_root() / "active_lineage.json"


def _lineage_path(lineage_name: str) -> Path:
    return lineages_dir() / f"{_slugify(lineage_name)}.json"


def _slugify(value: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value.strip())
    cleaned = cleaned.strip("_")
    return cleaned or "fresh_start"


def goal_ladder() -> list[dict[str, Any]]:
    return [dict(goal) for goal in _GOAL_LADDER]


def goal_ladder_ids() -> tuple[str, ...]:
    return tuple(goal["id"] for goal in _GOAL_LADDER)


def _default_training_rule() -> dict[str, Any]:
    return {
        "version": 3,
        "summary": (
            "Train sequentially. Do not hand off a later-stage checkpoint until all earlier milestones are passed "
            "and recorded with an exact checkpoint plus evidence."
        ),
        "goal_order": list(goal_ladder_ids()),
        "goal_ladder": goal_ladder(),
        "recording_requirements": [
            "exact checkpoint path",
            "stage and run name",
            "evidence record kind",
            "evidence summary or metrics",
            "manual review note for arm swing / walking pose when the robot is moving",
        ],
        "hard_rules": [
            "Zero-signal standing is the first gate. No policy-standing or walking-stage promotion before stand_zero_signal_30s_no_reset is recorded.",
            "Do not start stand-policy rescue training until the zero-signal stand gate is stable enough to diagnose cleanly.",
            "Record the first checkpoint that passes each milestone; do not overwrite history with prose only.",
            "A checkpoint only counts for a new milestone after that same checkpoint re-passes every earlier milestone in the ladder.",
            "Any hard reset, fall, or done event during a ladder test is an automatic failure.",
            "Backwards motion and strafing stay low priority until the forward 10 m gate and yaw gate pass.",
            "Game-stage work starts on the small map before larger terrain or obstacle layouts.",
            "A moving model only counts as walking if the pose still looks like walking and the arms help balance.",
            "Any later-stage regression that breaks an earlier milestone blocks promotion until the regression is fixed.",
        ],
        "restart_read_order": [
            "algorithms/urdf_learn_wasd_walk/TRAINING_RULES.md",
            "algorithms/urdf_learn_wasd_walk/outputs/history/active_lineage.json",
            "algorithms/urdf_learn_wasd_walk/outputs/history/refs/index.json",
            "algorithms/urdf_learn_wasd_walk/outputs/history/refs/milestones.json",
            "algorithms/urdf_learn_wasd_walk/outputs/history/checkpoint_registry.json",
            "algorithms/urdf_learn_wasd_walk/outputs/history/training_runs.jsonl",
            "algorithms/urdf_learn_wasd_walk/outputs/history/validation_runs.jsonl",
            "algorithms/urdf_learn_wasd_walk/outputs/history/evaluation_runs.jsonl",
            "algorithms/urdf_learn_wasd_walk/outputs/history/diagnostic_runs.jsonl",
        ],
    }


def _default_milestone_registry() -> dict[str, dict[str, Any]]:
    registry: dict[str, dict[str, Any]] = {}
    for order, goal in enumerate(_GOAL_LADDER, start=1):
        prerequisite_ids = [previous_goal["id"] for previous_goal in _GOAL_LADDER[: order - 1]]
        registry[str(goal["id"])] = {
            "milestone_id": goal["id"],
            "order": order,
            "stage": goal["stage"],
            "title": goal["title"],
            "summary": goal["summary"],
            "status": "pending",
            "checkpoint_path": None,
            "run_name": None,
            "recorded_at": None,
            "recorded_by": None,
            "evidence_record_kind": goal.get("evidence_record_kind"),
            "scenario": goal.get("scenario"),
            "prerequisite_ids": prerequisite_ids,
            "evidence": None,
            "manual_review": None,
            "notes": None,
        }
    return registry


def _next_pending_milestone_id(registry: dict[str, Any] | None) -> str | None:
    if not isinstance(registry, dict):
        return None
    for goal in _GOAL_LADDER:
        entry = registry.get(goal["id"])
        if not isinstance(entry, dict):
            return goal["id"]
        if entry.get("status") != "passed":
            return goal["id"]
    return None


def _next_pending_stage_milestone_id(stage: str, registry: dict[str, Any] | None) -> str | None:
    if not isinstance(registry, dict):
        return next(iter(_STAGE_REQUIRED_MILESTONES.get(stage, [])), None)
    for milestone_id in _STAGE_REQUIRED_MILESTONES.get(stage, []):
        entry = registry.get(milestone_id)
        if not isinstance(entry, dict) or entry.get("status") != "passed":
            return milestone_id
    return None


def _normalize_lineage_payload(payload: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    changed = False
    experiments = payload.get("experiments")
    if not isinstance(experiments, dict):
        experiments = {}
        payload["experiments"] = experiments
        changed = True

    stage_state = payload.get("stage_state")
    if not isinstance(stage_state, dict):
        stage_state = _default_stage_state(experiments)
        payload["stage_state"] = stage_state
        changed = True

    training_rule = payload.get("training_rule")
    if not isinstance(training_rule, dict):
        payload["training_rule"] = _default_training_rule()
        changed = True

    milestone_registry = payload.get("milestone_registry")
    if not isinstance(milestone_registry, dict):
        payload["milestone_registry"] = _default_milestone_registry()
        changed = True
    else:
        default_registry = _default_milestone_registry()
        stale_milestone_ids = [milestone_id for milestone_id in milestone_registry if milestone_id not in default_registry]
        for milestone_id in stale_milestone_ids:
            del milestone_registry[milestone_id]
            changed = True
        for milestone_id, default_entry in default_registry.items():
            current_entry = milestone_registry.get(milestone_id)
            if not isinstance(current_entry, dict):
                milestone_registry[milestone_id] = dict(default_entry)
                changed = True
                continue
            for key, value in default_entry.items():
                if key not in current_entry:
                    current_entry[key] = value
                    changed = True
            for static_key in (
                "order",
                "stage",
                "title",
                "summary",
                "evidence_record_kind",
                "scenario",
                "prerequisite_ids",
            ):
                if current_entry.get(static_key) != default_entry.get(static_key):
                    current_entry[static_key] = default_entry.get(static_key)
                    changed = True

    default_training_rule = _default_training_rule()
    training_rule = payload.get("training_rule")
    if isinstance(training_rule, dict):
        for key, value in default_training_rule.items():
            if training_rule.get(key) != value:
                training_rule[key] = value
                changed = True

    milestone_history = payload.get("milestone_history")
    if not isinstance(milestone_history, list):
        payload["milestone_history"] = []
        changed = True

    for stage, experiment_name in experiments.items():
        stage_entry = stage_state.get(stage)
        if not isinstance(stage_entry, dict):
            stage_state[stage] = {
                "status": "pending",
                "experiment_name": experiment_name,
                "objective": _STAGE_OBJECTIVES.get(stage),
                "required_milestones": list(_STAGE_REQUIRED_MILESTONES.get(stage, [])),
                "current_target_milestone": None,
                "last_training": None,
                "last_validation": None,
                "last_evaluation": None,
                "last_diagnostic": None,
                "selected_checkpoint": None,
                "produced_checkpoint": None,
                "achieved_capability": None,
                "failure": None,
            }
            stage_entry = stage_state[stage]
            changed = True
        if "objective" not in stage_entry:
            stage_entry["objective"] = _STAGE_OBJECTIVES.get(stage)
            changed = True
        if "experiment_name" not in stage_entry:
            stage_entry["experiment_name"] = experiment_name
            changed = True
        required_milestones = list(_STAGE_REQUIRED_MILESTONES.get(stage, []))
        if stage_entry.get("required_milestones") != required_milestones:
            stage_entry["required_milestones"] = required_milestones
            changed = True
        current_target = _next_pending_stage_milestone_id(stage, payload.get("milestone_registry"))
        if stage_entry.get("current_target_milestone") != current_target:
            stage_entry["current_target_milestone"] = current_target
            changed = True

    current_target_milestone = _next_pending_milestone_id(payload.get("milestone_registry"))
    if payload.get("current_target_milestone") != current_target_milestone:
        payload["current_target_milestone"] = current_target_milestone
        changed = True

    if int(payload.get("schema_version", 0) or 0) < 5:
        payload["schema_version"] = 5
        changed = True
    return payload, changed


def legacy_landau_experiment_name(stage: str) -> str:
    suffix = _LANDAU_STAGE_SUFFIX.get(stage or "full")
    if suffix is None:
        raise ValueError(f"Unsupported Landau stage '{stage}'.")
    return f"geo_landau_{suffix}"


def build_landau_experiment_names(lineage_name: str) -> dict[str, str]:
    lineage_slug = _slugify(lineage_name)
    return {
        stage: f"geo_landau_{lineage_slug}_{suffix}"
        for stage, suffix in _LANDAU_STAGE_SUFFIX.items()
    }


def load_active_lineage() -> dict[str, Any] | None:
    path = active_lineage_path()
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    payload, changed = _normalize_lineage_payload(payload)
    if changed:
        _write_lineage_payload(payload)
    return payload


def resolve_landau_experiment_name(stage: str, explicit_experiment_name: str | None = None) -> str:
    if explicit_experiment_name:
        return explicit_experiment_name
    active_lineage = load_active_lineage()
    if isinstance(active_lineage, dict):
        experiments = active_lineage.get("experiments")
        if isinstance(experiments, dict):
            experiment_name = experiments.get(stage)
            if isinstance(experiment_name, str) and experiment_name:
                return experiment_name
    return legacy_landau_experiment_name(stage)


def _ensure_history_layout() -> None:
    root = history_root()
    root.mkdir(parents=True, exist_ok=True)
    for relative in (
        "refs/experiments",
        "workflows",
        "diagnostics",
        "pose_candidates",
        "experiments",
        "lineages",
    ):
        (root / relative).mkdir(parents=True, exist_ok=True)
    for ledger_name in (
        "training_runs.jsonl",
        "validation_runs.jsonl",
        "evaluation_runs.jsonl",
        "diagnostic_runs.jsonl",
    ):
        ledger_path = root / ledger_name
        if not ledger_path.exists():
            ledger_path.write_text("", encoding="utf-8")
    registry_path = root / "checkpoint_registry.json"
    if not registry_path.exists():
        registry_path.write_text(json.dumps({"experiments": {}}, indent=2, sort_keys=True), encoding="utf-8")
    readme_path = root / "README.md"
    readme_path.write_text(_HISTORY_README, encoding="utf-8")


def _default_stage_state(experiments: dict[str, str]) -> dict[str, dict[str, Any]]:
    return {
        stage: {
            "status": "pending",
            "experiment_name": experiment_name,
            "objective": _STAGE_OBJECTIVES.get(stage),
            "required_milestones": list(_STAGE_REQUIRED_MILESTONES.get(stage, [])),
            "current_target_milestone": next(iter(_STAGE_REQUIRED_MILESTONES.get(stage, [])), None),
            "last_training": None,
            "last_validation": None,
            "last_evaluation": None,
            "last_diagnostic": None,
            "selected_checkpoint": None,
            "produced_checkpoint": None,
            "achieved_capability": None,
            "failure": None,
        }
        for stage, experiment_name in experiments.items()
    }


def _write_lineage_payload(payload: dict[str, Any]) -> None:
    lineage_path = _lineage_path(str(payload["lineage_name"]))
    serialized = json.dumps(payload, indent=2, sort_keys=True)
    lineage_path.write_text(serialized, encoding="utf-8")
    active_lineage_path().write_text(serialized, encoding="utf-8")


def initialize_lineage(
    *,
    lineage_name: str,
    note: str | None = None,
    archived_from: str | None = None,
) -> dict[str, Any]:
    _ensure_history_layout()
    experiments = build_landau_experiment_names(lineage_name)
    payload = {
        "record_kind": "training_lineage",
        "schema_version": 5,
        "lineage_name": _slugify(lineage_name),
        "created_at": _timestamp(),
        "note": note,
        "archived_from": archived_from,
        "acceptance": {
            "accept_stage": "fwd_only",
            "accept_terrain_mode": "flat",
            "path_follow_distance_m": 10.0,
            "require_walking_pose": True,
            "require_zero_falls": True,
        },
        "experiments": experiments,
        "current_stage": "stand",
        "current_run": None,
        "current_checkpoint": None,
        "current_workflow_id": None,
        "promoted_default": None,
        "training_rule": _default_training_rule(),
        "milestone_registry": _default_milestone_registry(),
        "milestone_history": [],
        "current_target_milestone": _next_pending_milestone_id(_default_milestone_registry()),
        "stage_state": _default_stage_state(experiments),
        "stage_history": [],
        "status": "active",
    }
    _write_lineage_payload(payload)
    from .run_history import refresh_history_refs

    refresh_history_refs()
    return payload


def update_lineage_stage_state(
    *,
    stage: str,
    status: str | None = None,
    workflow_id: str | None = None,
    run_name: str | None = None,
    selected_checkpoint: str | None = None,
    produced_checkpoint: str | None = None,
    last_training: dict[str, Any] | None = None,
    last_validation: dict[str, Any] | None = None,
    last_evaluation: dict[str, Any] | None = None,
    last_diagnostic: dict[str, Any] | None = None,
    achieved_capability: dict[str, Any] | str | None = None,
    failure: dict[str, Any] | str | None = None,
) -> dict[str, Any]:
    active_lineage = load_active_lineage()
    if not isinstance(active_lineage, dict):
        raise RuntimeError("No active training lineage is initialized.")
    active_lineage, _ = _normalize_lineage_payload(active_lineage)

    stage_state = active_lineage.get("stage_state")
    experiments = active_lineage.get("experiments", {})
    if not isinstance(stage_state, dict):
        stage_state = _default_stage_state(experiments if isinstance(experiments, dict) else {})
        active_lineage["stage_state"] = stage_state

    stage_entry = stage_state.setdefault(
        stage,
        {
            "status": "pending",
            "experiment_name": experiments.get(stage) if isinstance(experiments, dict) else None,
            "objective": _STAGE_OBJECTIVES.get(stage),
            "required_milestones": list(_STAGE_REQUIRED_MILESTONES.get(stage, [])),
            "current_target_milestone": next(iter(_STAGE_REQUIRED_MILESTONES.get(stage, [])), None),
            "last_training": None,
            "last_validation": None,
            "last_evaluation": None,
            "last_diagnostic": None,
            "selected_checkpoint": None,
            "produced_checkpoint": None,
            "achieved_capability": None,
            "failure": None,
        },
    )

    def _normalize_failure_payload(value: dict[str, Any] | str | None) -> dict[str, Any] | None:
        if value is None:
            return None
        if isinstance(value, dict):
            payload = dict(value)
            if "message" not in payload and "reason" in payload:
                payload["message"] = payload["reason"]
            return payload
        return {"message": str(value)}

    def _record_ref(record: dict[str, Any] | None) -> dict[str, Any] | None:
        if not isinstance(record, dict):
            return None
        checkpoint_path = record.get("checkpoint_path")
        if not isinstance(checkpoint_path, str):
            checkpoint_path = record.get("checkpoints", {}).get("latest")
        task = record.get("task")
        return {
            "record_kind": record.get("record_kind"),
            "recorded_at": record.get("recorded_at"),
            "status": record.get("status"),
            "workflow_id": record.get("workflow_id"),
            "experiment_name": task.get("experiment_name") if isinstance(task, dict) else None,
            "checkpoint_path": checkpoint_path,
            "failure": _normalize_failure_payload(record.get("failure")),
        }

    if status is not None:
        stage_entry["status"] = status
    if last_training is not None:
        stage_entry["last_training"] = last_training
    if last_validation is not None:
        stage_entry["last_validation"] = last_validation
    if last_evaluation is not None:
        stage_entry["last_evaluation"] = last_evaluation
    if last_diagnostic is not None:
        stage_entry["last_diagnostic"] = last_diagnostic
    if selected_checkpoint is not None:
        stage_entry["selected_checkpoint"] = selected_checkpoint
        active_lineage["current_checkpoint"] = selected_checkpoint
    if produced_checkpoint is not None:
        stage_entry["produced_checkpoint"] = produced_checkpoint
        active_lineage["current_checkpoint"] = produced_checkpoint
    if achieved_capability is not None:
        stage_entry["achieved_capability"] = achieved_capability
    if failure is not None:
        stage_entry["failure"] = _normalize_failure_payload(failure)
    elif status in {"completed", "running"}:
        stage_entry["failure"] = None

    active_lineage["current_stage"] = stage
    active_lineage["current_run"] = run_name
    if workflow_id is not None:
        active_lineage["current_workflow_id"] = workflow_id
    stage_entry["current_target_milestone"] = _next_pending_stage_milestone_id(
        stage,
        active_lineage.get("milestone_registry"),
    )
    active_lineage["current_target_milestone"] = _next_pending_milestone_id(active_lineage.get("milestone_registry"))
    active_lineage["updated_at"] = _timestamp()
    stage_history = active_lineage.setdefault("stage_history", [])
    if isinstance(stage_history, list):
        stage_history.append(
            {
                "recorded_at": active_lineage["updated_at"],
                "stage": stage,
                "status": stage_entry.get("status"),
                "objective": stage_entry.get("objective"),
                "workflow_id": workflow_id,
                "run_name": run_name,
                "selected_checkpoint": stage_entry.get("selected_checkpoint"),
                "produced_checkpoint": stage_entry.get("produced_checkpoint"),
                "last_training": _record_ref(stage_entry.get("last_training")),
                "last_validation": _record_ref(stage_entry.get("last_validation")),
                "last_evaluation": _record_ref(stage_entry.get("last_evaluation")),
                "last_diagnostic": _record_ref(stage_entry.get("last_diagnostic")),
                "achieved_capability": achieved_capability,
                "failure": stage_entry.get("failure"),
            }
        )
    _write_lineage_payload(active_lineage)

    from .run_history import refresh_history_refs

    refresh_history_refs()
    return active_lineage


def record_lineage_milestone(
    *,
    milestone_id: str,
    checkpoint_path: str | None,
    run_name: str | None = None,
    stage: str | None = None,
    status: str = "passed",
    evidence: dict[str, Any] | None = None,
    manual_review: dict[str, Any] | str | None = None,
    notes: str | None = None,
    recorded_by: str | None = None,
) -> dict[str, Any]:
    active_lineage = load_active_lineage()
    if not isinstance(active_lineage, dict):
        raise RuntimeError("No active training lineage is initialized.")
    active_lineage, _ = _normalize_lineage_payload(active_lineage)

    registry = active_lineage.get("milestone_registry")
    if not isinstance(registry, dict):
        registry = _default_milestone_registry()
        active_lineage["milestone_registry"] = registry
    milestone_entry = registry.get(milestone_id)
    if not isinstance(milestone_entry, dict):
        valid_ids = ", ".join(goal_ladder_ids())
        raise ValueError(f"Unknown milestone_id '{milestone_id}'. Expected one of: {valid_ids}")

    expected_stage = str(milestone_entry.get("stage"))
    if stage is None:
        stage = expected_stage
    if stage != expected_stage:
        raise ValueError(f"Milestone '{milestone_id}' belongs to stage '{expected_stage}', not '{stage}'.")

    resolved_checkpoint = None if checkpoint_path is None else str(Path(checkpoint_path).expanduser().resolve())
    normalized_manual_review: dict[str, Any] | None
    if isinstance(manual_review, dict):
        normalized_manual_review = dict(manual_review)
    elif manual_review is None:
        normalized_manual_review = None
    else:
        normalized_manual_review = {"summary": str(manual_review)}

    milestone_entry.update(
        {
            "status": status,
            "checkpoint_path": resolved_checkpoint,
            "run_name": run_name,
            "recorded_at": _timestamp(),
            "recorded_by": recorded_by,
            "evidence": evidence,
            "manual_review": normalized_manual_review,
            "notes": notes,
        }
    )

    stage_state = active_lineage.get("stage_state")
    if not isinstance(stage_state, dict):
        stage_state = _default_stage_state(active_lineage.get("experiments", {}))
        active_lineage["stage_state"] = stage_state
    stage_entry = stage_state.setdefault(
        stage,
        {
            "status": "pending",
            "experiment_name": active_lineage.get("experiments", {}).get(stage) if isinstance(active_lineage.get("experiments"), dict) else None,
            "objective": _STAGE_OBJECTIVES.get(stage),
            "required_milestones": list(_STAGE_REQUIRED_MILESTONES.get(stage, [])),
            "current_target_milestone": _next_pending_stage_milestone_id(stage, registry),
            "last_training": None,
            "last_validation": None,
            "last_evaluation": None,
            "last_diagnostic": None,
            "selected_checkpoint": None,
            "produced_checkpoint": None,
            "achieved_capability": None,
            "failure": None,
        },
    )
    stage_entry["current_target_milestone"] = _next_pending_stage_milestone_id(stage, registry)
    if status == "passed":
        stage_entry["achieved_capability"] = {
            "milestone_id": milestone_id,
            "checkpoint_path": resolved_checkpoint,
            "run_name": run_name,
            "recorded_at": milestone_entry.get("recorded_at"),
            "title": milestone_entry.get("title"),
        }

    milestone_history = active_lineage.setdefault("milestone_history", [])
    if isinstance(milestone_history, list):
        milestone_history.append(
            {
                "recorded_at": milestone_entry.get("recorded_at"),
                "milestone_id": milestone_id,
                "stage": stage,
                "status": status,
                "checkpoint_path": resolved_checkpoint,
                "run_name": run_name,
                "recorded_by": recorded_by,
                "notes": notes,
                "manual_review": normalized_manual_review,
                "evidence_record_kind": evidence.get("record_kind") if isinstance(evidence, dict) else None,
            }
        )

    active_lineage["current_target_milestone"] = _next_pending_milestone_id(registry)
    active_lineage["updated_at"] = _timestamp()
    _write_lineage_payload(active_lineage)

    from .run_history import refresh_history_refs

    refresh_history_refs()
    return active_lineage


def reset_landau_training_state(
    *,
    lineage_name: str | None = None,
    archive_tag: str | None = None,
    note: str | None = None,
    archive_logs: bool = True,
) -> dict[str, Any]:
    assert_no_active_isaac_lock()
    lineage_slug = _slugify(lineage_name or f"fresh_start_{_timestamp_slug().lower()}")
    archive_slug = f"{_timestamp_slug()}_{_slugify(archive_tag or lineage_slug)}"
    archive_root = archives_dir() / archive_slug
    archive_root.mkdir(parents=True, exist_ok=True)

    moved_paths: list[dict[str, str]] = []
    active_history_root = history_root()
    if active_history_root.exists() and any(active_history_root.iterdir()):
        archived_history_root = archive_root / "history"
        shutil.move(str(active_history_root), str(archived_history_root))
        moved_paths.append({"kind": "history", "from": str(active_history_root), "to": str(archived_history_root)})

    logs_root = _repo_root() / "logs" / "rsl_rl"
    archived_logs_root = archive_root / "logs" / "rsl_rl"
    if archive_logs and logs_root.is_dir():
        archived_logs_root.mkdir(parents=True, exist_ok=True)
        for experiment_dir in sorted(logs_root.glob("geo_landau*")):
            if not experiment_dir.is_dir():
                continue
            destination = archived_logs_root / experiment_dir.name
            shutil.move(str(experiment_dir), str(destination))
            moved_paths.append({"kind": "log_dir", "from": str(experiment_dir), "to": str(destination)})

    manifest = {
        "record_kind": "reset_manifest",
        "recorded_at": _timestamp(),
        "lineage_name": lineage_slug,
        "archive_root": str(archive_root),
        "archive_tag": archive_tag,
        "note": note,
        "archive_logs": archive_logs,
        "moved_paths": moved_paths,
    }
    manifest_path = archive_root / "reset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    active_lineage = initialize_lineage(
        lineage_name=lineage_slug,
        note=note,
        archived_from=str(manifest_path),
    )
    manifest["active_lineage"] = active_lineage
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "archive_manifest_path": str(manifest_path),
        "archive_root": str(archive_root),
        "active_lineage": active_lineage,
    }
