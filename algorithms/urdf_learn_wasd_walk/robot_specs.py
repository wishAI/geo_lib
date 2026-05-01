from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .asset_paths import landau_urdf_path
from .urdf_utils import (
    JointGroups,
    classify_joint_groups,
    compute_link_world_transforms,
    detect_primary_foot_links,
    detect_support_links,
    detect_termination_links,
    estimate_root_height,
    find_missing_meshes,
    load_urdf_model,
    support_surface_world_z,
    transform_point,
)


@dataclass(frozen=True)
class RobotTaskSpec:
    key: str
    display_name: str
    train_task_id: str
    play_task_id: str
    experiment_name: str
    forward_body_axis: str = "x"  # "x" or "y" — which body-frame axis is semantic forward
    control_root_link: str | None = None  # explicit control root; None means use root_link_name


@dataclass(frozen=True)
class LandauRobotSpec(RobotTaskSpec):
    urdf_path: Path = Path()
    root_link_name: str = ""
    primary_foot_links: tuple[str, ...] = ()
    primary_foot_contact_up_vectors: tuple[tuple[float, float, float], ...] = ()
    support_link_names: tuple[str, ...] = ()
    gait_guard_link_names: tuple[str, ...] = ()
    termination_link_names: tuple[str, ...] = ()
    joint_groups: JointGroups | None = None
    default_joint_positions: dict[str, float] | None = None
    init_root_height: float = 0.0
    nominal_control_root_height: float = 0.0
    nominal_stance_width: float = 0.0
    missing_meshes: tuple[Path, ...] = ()


G1_TASK_SPEC = RobotTaskSpec(
    key="g1",
    display_name="Unitree G1",
    train_task_id="Geo-Velocity-Flat-G1-v0",
    play_task_id="Geo-Velocity-Flat-G1-Play-v0",
    experiment_name="geo_g1_flat",
)


def _build_landau_default_joint_positions(model) -> dict[str, float]:
    # The refreshed Landau URDF is noticeably more stable with a straighter lower-body
    # preload than with the original bent-knee handoff pose. Zero-action diagnostics
    # showed the old pose collapsing through the ankles within ~25 steps.
    child_link_targets = {
        "spine_01_x": 0.03,
        "spine_02_x": 0.08,
        "neck_x": -0.05,
        "thigh_stretch_l": 0.08,
        "thigh_stretch_r": 0.08,
        "leg_stretch_l": 0.24,
        "leg_stretch_r": 0.24,
        # The slightly plantar-flexed preload still trains better than the flatter
        # whole-foot stance, even though the flatter stance survives a little longer
        # under raw PD holding. Keep the training pose on the better learning track.
        "foot_l": -0.12,
        "foot_r": -0.12,
        "toes_01_l": -0.11,
        "toes_01_r": -0.11,
        "forearm_stretch_l": 0.21,
        "forearm_stretch_r": 0.21,
    }
    return {
        model.child_joint_by_link[child_link]: value
        for child_link, value in child_link_targets.items()
        if child_link in model.child_joint_by_link
    }


def _build_landau_gait_guard_links(
    model,
    *,
    support_links: tuple[str, ...],
    termination_links: tuple[str, ...],
) -> tuple[str, ...]:
    excluded_prefixes = (
        "thumb",
        "index",
        "middle",
        "ring",
        "pinky",
        "shoulder",
        "arm",
        "forearm",
        "hand",
    )
    return tuple(
        name
        for name in model.links
        if name not in support_links
        and name not in termination_links
        and not name.startswith(excluded_prefixes)
    )


def _local_vector_for_world_axis(
    transform: tuple[tuple[float, float, float, float], ...],
    world_axis: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Return a world-space direction expressed in the link-local frame."""

    return (
        transform[0][0] * world_axis[0] + transform[1][0] * world_axis[1] + transform[2][0] * world_axis[2],
        transform[0][1] * world_axis[0] + transform[1][1] * world_axis[1] + transform[2][1] * world_axis[2],
        transform[0][2] * world_axis[0] + transform[1][2] * world_axis[1] + transform[2][2] * world_axis[2],
    )


def _select_primary_support_links(
    *,
    model,
    world: dict[str, tuple[tuple[float, float, float, float], ...]],
    support_links: tuple[str, ...],
) -> tuple[str, ...]:
    selected: list[str] = []
    for suffix in ("_l", "_r"):
        candidates = [name for name in support_links if name.endswith(suffix) and name in world]
        if not candidates:
            continue
        selected.append(min(candidates, key=lambda name: support_surface_world_z(model, world, name)))
    return tuple(selected)


def _select_landau_primary_feet(
    *,
    model,
    world: dict[str, tuple[tuple[float, float, float, float], ...]],
    support_links: tuple[str, ...],
) -> tuple[str, str]:
    preferred = detect_primary_foot_links(model)
    if len(preferred) == 2 and all(link_name in support_links for link_name in preferred):
        return preferred  # type: ignore[return-value]

    selected = _select_primary_support_links(model=model, world=world, support_links=support_links)
    if len(selected) != 2:
        raise RuntimeError(f"Unable to resolve a biped primary-foot pair from support links: {support_links}")
    return tuple(selected)  # type: ignore[return-value]


_LANDAU_STAGE_TASK_IDS: dict[str, dict[str, str]] = {
    "stand": {
        "train": "Geo-Velocity-Flat-Landau-Stand-v0",
        "play": "Geo-Velocity-Flat-Landau-Stand-Play-v0",
        "experiment": "geo_landau_stand",
    },
    "full": {
        "train": "Geo-Velocity-Flat-Landau-v0",
        "play": "Geo-Velocity-Flat-Landau-Play-v0",
        "experiment": "geo_landau_flat",
    },
    "fwd_only": {
        "train": "Geo-Velocity-Flat-Landau-FwdOnly-v0",
        "play": "Geo-Velocity-Flat-Landau-FwdOnly-Play-v0",
        "experiment": "geo_landau_fwd_only",
    },
    "fwd_yaw": {
        "train": "Geo-Velocity-Flat-Landau-FwdYaw-v0",
        "play": "Geo-Velocity-Flat-Landau-FwdYaw-Play-v0",
        "experiment": "geo_landau_fwd_yaw",
    },
    "game": {
        "train": "Geo-Velocity-Rough-Landau-Game-v0",
        "play": "Geo-Velocity-Rough-Landau-Game-Play-v0",
        "experiment": "geo_landau_game",
    },
}


def load_landau_robot_spec(
    urdf_path: Path | None = None, stage: str = "full"
) -> LandauRobotSpec:
    if stage not in _LANDAU_STAGE_TASK_IDS:
        raise ValueError(
            f"Unknown Landau stage '{stage}'. "
            f"Expected one of: {', '.join(_LANDAU_STAGE_TASK_IDS)}"
        )
    ids = _LANDAU_STAGE_TASK_IDS[stage]
    resolved_urdf = Path(urdf_path or landau_urdf_path()).resolve()
    if not resolved_urdf.is_file():
        raise FileNotFoundError(
            f"Missing prepared Landau URDF: {resolved_urdf}. "
            "Keep inputs/landau_v10/ in place before running the Landau workflow."
        )
    model = load_urdf_model(resolved_urdf)
    discovered_root_link = model.root_links[0] if model.root_links else "base_link"
    joint_groups = classify_joint_groups(model)
    default_joint_positions = _build_landau_default_joint_positions(model)
    support_links = detect_support_links(model)
    forward_body_axis = "y"
    # Keep the imported articulation rooted at the URDF root. Landau exposes `root_x`
    # as the semantic control body through a fixed joint under `base_link`, and forcing
    # the importer root onto that child body regressed stand stability immediately.
    control_root_link = "root_x" if "root_x" in model.links else discovered_root_link
    root_link_name = discovered_root_link
    termination_links = detect_termination_links(model)
    gait_guard_links = _build_landau_gait_guard_links(
        model,
        support_links=support_links,
        termination_links=termination_links,
    )
    init_root_height = estimate_root_height(
        model=model,
        root_link_name=root_link_name,
        support_link_names=support_links,
        joint_positions=default_joint_positions,
        clearance=0.01,
    )
    world = compute_link_world_transforms(
        model,
        joint_positions=default_joint_positions,
        root_link=discovered_root_link,
    )
    primary_feet = _select_landau_primary_feet(model=model, world=world, support_links=support_links)
    primary_foot_contact_up_vectors = tuple(
        _local_vector_for_world_axis(world[link_name], (0.0, 0.0, 1.0)) if link_name in world else (0.0, 0.0, 1.0)
        for link_name in primary_feet
    )
    support_points = [support_surface_world_z(model, world, link_name) for link_name in support_links if link_name in world]
    nominal_control_root_height = 0.0
    if support_points and control_root_link in world:
        nominal_control_root_height = transform_point(world[control_root_link], (0.0, 0.0, 0.0))[2] - min(support_points)
    left_support_points = [
        transform_point(world[link_name], (0.0, 0.0, 0.0))
        for link_name in support_links
        if link_name.endswith("_l") and link_name in world
    ]
    right_support_points = [
        transform_point(world[link_name], (0.0, 0.0, 0.0))
        for link_name in support_links
        if link_name.endswith("_r") and link_name in world
    ]
    nominal_stance_width = 0.0
    if left_support_points and right_support_points:
        left_center = tuple(
            sum(point[index] for point in left_support_points) / len(left_support_points) for index in range(3)
        )
        right_center = tuple(
            sum(point[index] for point in right_support_points) / len(right_support_points) for index in range(3)
        )
        lateral_axis = 0 if forward_body_axis == "y" else 1
        nominal_stance_width = abs(left_center[lateral_axis] - right_center[lateral_axis])
    return LandauRobotSpec(
        key="landau",
        display_name="Landau v10 URDF",
        train_task_id=ids["train"],
        play_task_id=ids["play"],
        experiment_name=ids["experiment"],
        forward_body_axis=forward_body_axis,
        control_root_link=control_root_link,
        urdf_path=resolved_urdf,
        root_link_name=root_link_name,
        primary_foot_links=primary_feet,
        primary_foot_contact_up_vectors=primary_foot_contact_up_vectors,
        support_link_names=support_links,
        gait_guard_link_names=gait_guard_links,
        termination_link_names=termination_links,
        joint_groups=joint_groups,
        default_joint_positions=default_joint_positions,
        init_root_height=init_root_height,
        nominal_control_root_height=nominal_control_root_height,
        nominal_stance_width=nominal_stance_width,
        missing_meshes=find_missing_meshes(model),
    )


def task_spec_for_robot(robot_key: str) -> RobotTaskSpec:
    normalized = robot_key.lower()
    if normalized == "g1":
        return G1_TASK_SPEC
    if normalized == "landau":
        return load_landau_robot_spec()
    raise KeyError(f"Unsupported robot '{robot_key}'. Expected one of: g1, landau.")
