from __future__ import annotations

from dataclasses import replace

from .robot_specs import G1_TASK_SPEC, RobotTaskSpec, load_landau_robot_spec


def supported_robot_keys() -> tuple[str, ...]:
    return ("g1", "landau")


def resolve_robot_task_spec(robot_key: str) -> RobotTaskSpec:
    normalized = robot_key.lower()
    if normalized == "g1":
        return G1_TASK_SPEC
    if normalized == "landau":
        return load_landau_robot_spec()
    raise KeyError(f"Unsupported robot '{robot_key}'.")


def with_play_task(spec: RobotTaskSpec) -> RobotTaskSpec:
    return replace(spec, train_task_id=spec.play_task_id)

