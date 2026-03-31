from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .asset_paths import landau_urdf_path
from .urdf_utils import (
    JointGroups,
    classify_joint_groups,
    detect_primary_foot_links,
    detect_support_links,
    detect_termination_links,
    estimate_root_height,
    find_missing_meshes,
    load_urdf_model,
)


@dataclass(frozen=True)
class RobotTaskSpec:
    key: str
    display_name: str
    train_task_id: str
    play_task_id: str
    experiment_name: str


@dataclass(frozen=True)
class LandauRobotSpec(RobotTaskSpec):
    urdf_path: Path
    root_link_name: str
    primary_foot_links: tuple[str, ...]
    support_link_names: tuple[str, ...]
    termination_link_names: tuple[str, ...]
    joint_groups: JointGroups
    default_joint_positions: dict[str, float]
    init_root_height: float
    missing_meshes: tuple[Path, ...]


G1_TASK_SPEC = RobotTaskSpec(
    key="g1",
    display_name="Unitree G1",
    train_task_id="Geo-Velocity-Flat-G1-v0",
    play_task_id="Geo-Velocity-Flat-G1-Play-v0",
    experiment_name="geo_g1_flat",
)


def _build_landau_default_joint_positions() -> dict[str, float]:
    return {
        "spine_01_x": 0.03,
        "spine_02_x": 0.08,
        "neck_x": -0.05,
        "thigh_stretch_l": 0.20,
        "thigh_stretch_r": 0.20,
        "leg_stretch_l": 0.54,
        "leg_stretch_r": 0.54,
        "foot_l": -0.07,
        "foot_r": -0.07,
        "toes_01_l": 0.05,
        "toes_01_r": 0.05,
        "forearm_stretch_l": 0.21,
        "forearm_stretch_r": 0.21,
    }


def load_landau_robot_spec(urdf_path: Path | None = None) -> LandauRobotSpec:
    resolved_urdf = Path(urdf_path or landau_urdf_path()).resolve()
    model = load_urdf_model(resolved_urdf)
    root_link_name = model.root_links[0] if model.root_links else "base_link"
    joint_groups = classify_joint_groups(model)
    default_joint_positions = _build_landau_default_joint_positions()
    primary_feet = detect_primary_foot_links(model)
    support_links = detect_support_links(model)
    termination_links = detect_termination_links(model)
    init_root_height = estimate_root_height(
        model=model,
        root_link_name=root_link_name,
        support_link_names=support_links,
        joint_positions=default_joint_positions,
        clearance=0.01,
    )
    return LandauRobotSpec(
        key="landau",
        display_name="Landau v10 URDF",
        train_task_id="Geo-Velocity-Flat-Landau-v0",
        play_task_id="Geo-Velocity-Flat-Landau-Play-v0",
        experiment_name="geo_landau_flat",
        urdf_path=resolved_urdf,
        root_link_name=root_link_name,
        primary_foot_links=primary_feet,
        support_link_names=support_links,
        termination_link_names=termination_links,
        joint_groups=joint_groups,
        default_joint_positions=default_joint_positions,
        init_root_height=init_root_height,
        missing_meshes=find_missing_meshes(model),
    )


def task_spec_for_robot(robot_key: str) -> RobotTaskSpec:
    normalized = robot_key.lower()
    if normalized == "g1":
        return G1_TASK_SPEC
    if normalized == "landau":
        return load_landau_robot_spec()
    raise KeyError(f"Unsupported robot '{robot_key}'. Expected one of: g1, landau.")
