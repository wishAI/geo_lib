from __future__ import annotations

from copy import deepcopy
import re

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.terrains import TerrainGeneratorCfg
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from .landau_rewards import body_height_below_min_termination
from .robot_specs import load_landau_robot_spec
from .urdf_utils import estimate_root_height, load_urdf_model


LANDAU_SPEC = load_landau_robot_spec()
LANDAU_MODEL = load_urdf_model(LANDAU_SPEC.urdf_path)

LANDAU_ACTUATED_JOINTS = tuple(
    joint_name for joint_name, joint in LANDAU_MODEL.joints.items() if joint.joint_type != "fixed"
)
LANDAU_LOWER_BODY_JOINTS = tuple((*LANDAU_SPEC.joint_groups.leg_joints, *LANDAU_SPEC.joint_groups.foot_joints))
LANDAU_ANKLE_JOINTS = tuple(joint_name for joint_name in LANDAU_SPEC.joint_groups.foot_joints if "ankle" in joint_name)
LANDAU_TOE_JOINTS = tuple(joint_name for joint_name in LANDAU_SPEC.joint_groups.foot_joints if "toe" in joint_name)
LANDAU_TRUNK_CONTROLLED_JOINTS = tuple(
    joint_name
    for joint_name in LANDAU_SPEC.joint_groups.torso_joints
    if any(token in joint_name for token in ("waist", "spine"))
)
LANDAU_CORE_CONTROLLED_JOINTS = tuple((*LANDAU_LOWER_BODY_JOINTS, *LANDAU_TRUNK_CONTROLLED_JOINTS))
LANDAU_WALK_CONTROLLED_JOINTS = LANDAU_CORE_CONTROLLED_JOINTS
LANDAU_PRIMARY_FOOT_LINKS = tuple(LANDAU_SPEC.primary_foot_links)
LANDAU_SUPPORT_LINKS = tuple(LANDAU_SPEC.support_link_names)
LANDAU_GAIT_GUARD_LINKS = tuple(LANDAU_SPEC.gait_guard_link_names)
LANDAU_CONTROL_ROOT_LINK = LANDAU_SPEC.control_root_link or LANDAU_SPEC.root_link_name
LANDAU_CONTROL_ROOT_HEIGHT_FLOOR = max(0.17, float(LANDAU_SPEC.nominal_control_root_height) * 0.65)

LANDAU_RAW_INIT_JOINT_POS: dict[str, float] = {}
LANDAU_STAND_INIT_JOINT_POS = dict(LANDAU_SPEC.default_joint_positions or {})
LANDAU_STAND_INIT_JOINT_POS.update(
    {
        "waist_yaw_joint": 0.0,
        "waist_roll_joint": 0.0,
        "left_ankle_pitch_joint": -0.12,
        "right_ankle_pitch_joint": -0.12,
        "left_toe_joint": 0.08,
        "right_toe_joint": 0.08,
    }
)
LANDAU_WALK_INIT_JOINT_POS = dict(LANDAU_SPEC.default_joint_positions or {})
LANDAU_RAW_INIT_ROOT_HEIGHT = estimate_root_height(
    model=LANDAU_MODEL,
    root_link_name=LANDAU_SPEC.root_link_name,
    support_link_names=LANDAU_SUPPORT_LINKS,
    joint_positions=LANDAU_RAW_INIT_JOINT_POS,
    clearance=0.01,
)
LANDAU_STAND_INIT_ROOT_HEIGHT = float(LANDAU_SPEC.init_root_height)
LANDAU_WALK_INIT_ROOT_HEIGHT = float(LANDAU_SPEC.init_root_height)

LANDAU_HEIGHT_SCAN_SIZE = (1.6, 1.0)
LANDAU_HEIGHT_SCAN_RESOLUTION = 0.1

LANDAU_GAME_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "obstacles": terrain_gen.HfDiscreteObstaclesTerrainCfg(
            proportion=0.25,
            obstacle_width_range=(0.3, 1.1),
            obstacle_height_range=(0.08, 0.35),
            num_obstacles=18,
            platform_width=2.5,
        ),
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.20,
            grid_width=0.45,
            grid_height_range=(0.05, 0.2),
            platform_width=2.2,
        ),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.20,
            noise_range=(0.02, 0.08),
            noise_step=0.02,
            border_width=0.25,
        ),
        "stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.20,
            step_height_range=(0.05, 0.16),
            step_width=0.35,
            platform_width=2.8,
            border_width=1.0,
            holes=False,
        ),
        "slopes": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.15,
            slope_range=(0.0, 0.25),
            platform_width=2.5,
            border_width=0.25,
        ),
    },
)

LANDAU_GAME_PLAY_TERRAINS_CFG = deepcopy(LANDAU_GAME_TERRAINS_CFG)
LANDAU_GAME_PLAY_TERRAINS_CFG.size = (3.0, 3.0)
LANDAU_GAME_PLAY_TERRAINS_CFG.border_width = 2.5
LANDAU_GAME_PLAY_TERRAINS_CFG.num_rows = 1
LANDAU_GAME_PLAY_TERRAINS_CFG.num_cols = 2
LANDAU_GAME_PLAY_TERRAINS_CFG.curriculum = False
LANDAU_GAME_PLAY_TERRAINS_CFG.sub_terrains["obstacles"].proportion = 0.10
LANDAU_GAME_PLAY_TERRAINS_CFG.sub_terrains["obstacles"].obstacle_width_range = (0.22, 0.55)
LANDAU_GAME_PLAY_TERRAINS_CFG.sub_terrains["obstacles"].obstacle_height_range = (0.04, 0.16)
LANDAU_GAME_PLAY_TERRAINS_CFG.sub_terrains["obstacles"].num_obstacles = 4
LANDAU_GAME_PLAY_TERRAINS_CFG.sub_terrains["obstacles"].platform_width = 1.8
LANDAU_GAME_PLAY_TERRAINS_CFG.sub_terrains["boxes"].proportion = 0.15
LANDAU_GAME_PLAY_TERRAINS_CFG.sub_terrains["boxes"].grid_width = 0.35
LANDAU_GAME_PLAY_TERRAINS_CFG.sub_terrains["boxes"].grid_height_range = (0.03, 0.10)
LANDAU_GAME_PLAY_TERRAINS_CFG.sub_terrains["boxes"].platform_width = 1.9
LANDAU_GAME_PLAY_TERRAINS_CFG.sub_terrains["random_rough"].proportion = 0.50
LANDAU_GAME_PLAY_TERRAINS_CFG.sub_terrains["random_rough"].noise_range = (0.01, 0.05)
LANDAU_GAME_PLAY_TERRAINS_CFG.sub_terrains["stairs"].proportion = 0.05
LANDAU_GAME_PLAY_TERRAINS_CFG.sub_terrains["stairs"].step_height_range = (0.02, 0.08)
LANDAU_GAME_PLAY_TERRAINS_CFG.sub_terrains["stairs"].platform_width = 2.4
LANDAU_GAME_PLAY_TERRAINS_CFG.sub_terrains["slopes"].proportion = 0.25
LANDAU_GAME_PLAY_TERRAINS_CFG.sub_terrains["slopes"].slope_range = (0.0, 0.12)


def _apply_actuator_overrides(env_cfg, overrides: dict[str, dict[str, float]]) -> None:
    actuators = getattr(getattr(getattr(env_cfg, "scene", None), "robot", None), "actuators", None)
    if not isinstance(actuators, dict):
        return
    for actuator_name, values in overrides.items():
        actuator_cfg = actuators.get(actuator_name)
        if actuator_cfg is None:
            continue
        for field_name, value in values.items():
            setattr(actuator_cfg, field_name, value)


def configure_landau_stand_actuators(env_cfg) -> None:
    _apply_actuator_overrides(
        env_cfg,
        {
            "legs": {"effort_limit_sim": 320.0, "velocity_limit_sim": 20.0, "stiffness": 460.0, "damping": 60.0},
            "ankles": {"effort_limit_sim": 280.0, "velocity_limit_sim": 20.0, "stiffness": 420.0, "damping": 54.0},
            "toes": {"effort_limit_sim": 220.0, "velocity_limit_sim": 20.0, "stiffness": 320.0, "damping": 42.0},
            "torso": {"effort_limit_sim": 260.0, "velocity_limit_sim": 12.0, "stiffness": 320.0, "damping": 40.0},
            "arms": {"effort_limit_sim": 90.0, "velocity_limit_sim": 12.0, "stiffness": 110.0, "damping": 16.0},
            "fingers": {"effort_limit_sim": 12.0, "velocity_limit_sim": 10.0, "stiffness": 15.0, "damping": 2.0},
        },
    )


def configure_landau_walk_actuators(env_cfg) -> None:
    _apply_actuator_overrides(
        env_cfg,
        {
            "legs": {"effort_limit_sim": 240.0, "velocity_limit_sim": 20.0, "stiffness": 260.0, "damping": 28.0},
            "ankles": {"effort_limit_sim": 180.0, "velocity_limit_sim": 20.0, "stiffness": 180.0, "damping": 22.0},
            "toes": {"effort_limit_sim": 90.0, "velocity_limit_sim": 20.0, "stiffness": 90.0, "damping": 12.0},
            "torso": {"effort_limit_sim": 200.0, "velocity_limit_sim": 12.0, "stiffness": 220.0, "damping": 24.0},
            "arms": {"effort_limit_sim": 70.0, "velocity_limit_sim": 12.0, "stiffness": 70.0, "damping": 10.0},
            "fingers": {"effort_limit_sim": 12.0, "velocity_limit_sim": 10.0, "stiffness": 15.0, "damping": 2.0},
        },
    )


def configure_landau_playback_actuators(env_cfg) -> None:
    _apply_actuator_overrides(
        env_cfg,
        {
            "legs": {"effort_limit_sim": 220.0, "velocity_limit_sim": 20.0, "stiffness": 240.0, "damping": 30.0},
            "ankles": {"effort_limit_sim": 170.0, "velocity_limit_sim": 20.0, "stiffness": 170.0, "damping": 24.0},
            "toes": {"effort_limit_sim": 85.0, "velocity_limit_sim": 20.0, "stiffness": 90.0, "damping": 14.0},
            "torso": {"effort_limit_sim": 180.0, "velocity_limit_sim": 12.0, "stiffness": 200.0, "damping": 26.0},
            "arms": {"effort_limit_sim": 60.0, "velocity_limit_sim": 12.0, "stiffness": 60.0, "damping": 10.0},
            "fingers": {"effort_limit_sim": 12.0, "velocity_limit_sim": 10.0, "stiffness": 15.0, "damping": 2.0},
        },
    )


def build_landau_action_scale(
    *,
    leg_scale: float,
    foot_scale: float,
    toe_scale: float,
    torso_scale: float,
    arm_scale: float,
    hand_scale: float,
    head_scale: float = 0.1,
    controlled_joint_names: tuple[str, ...],
) -> dict[str, float]:
    pattern_scales = {
        "thigh_.*": leg_scale,
        "leg_.*": leg_scale,
        ".*hip.*": leg_scale,
        ".*knee.*": leg_scale,
        ".*shin.*": leg_scale,
        "foot_.*": foot_scale,
        ".*ankle.*": foot_scale,
        "toes_.*": toe_scale,
        ".*toe.*": toe_scale,
        "spine_.*": torso_scale,
        "waist_.*": torso_scale,
        "neck_.*": torso_scale,
        "head_.*": head_scale,
        "shoulder_.*": arm_scale,
        "arm_.*": arm_scale,
        "forearm_.*": arm_scale,
        ".*shoulder.*": arm_scale,
        ".*upper_arm.*": arm_scale,
        ".*elbow.*": arm_scale,
        ".*forearm.*": arm_scale,
        "hand_.*": hand_scale,
        ".*wrist.*": hand_scale,
    }
    matched_patterns: dict[str, float] = {}
    for pattern, scale in pattern_scales.items():
        regex = re.compile(pattern)
        if any(regex.fullmatch(joint_name) for joint_name in controlled_joint_names):
            matched_patterns[pattern] = scale
    return matched_patterns


def apply_landau_init_pose(env_cfg, joint_targets: dict[str, float], root_height: float) -> None:
    init_state_cfg = getattr(getattr(getattr(env_cfg, "scene", None), "robot", None), "init_state", None)
    if init_state_cfg is None:
        return
    joint_pos = {joint_name: 0.0 for joint_name in LANDAU_ACTUATED_JOINTS}
    joint_pos.update({joint_name: float(value) for joint_name, value in joint_targets.items() if joint_name in joint_pos})
    init_state_cfg.pos = (0.0, 0.0, float(root_height))
    init_state_cfg.joint_pos = joint_pos


def build_landau_articulation_cfg() -> ArticulationCfg:
    joint_defaults = {joint_name: 0.0 for joint_name in LANDAU_ACTUATED_JOINTS}
    return ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UrdfFileCfg(
            asset_path=str(LANDAU_SPEC.urdf_path),
            root_link_name=LANDAU_SPEC.root_link_name,
            fix_base=False,
            merge_fixed_joints=False,
            make_instanceable=False,
            collision_from_visuals=False,
            self_collision=False,
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=False,
                linear_damping=0.05,
                angular_damping=0.1,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=8,
            ),
            joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=None, damping=None)
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, float(LANDAU_RAW_INIT_ROOT_HEIGHT)),
            joint_pos=joint_defaults,
            joint_vel={".*": 0.0},
        ),
        soft_joint_pos_limit_factor=0.9,
        actuators={
            "legs": ImplicitActuatorCfg(
                joint_names_expr=list(LANDAU_SPEC.joint_groups.leg_joints),
                effort_limit_sim=220.0,
                velocity_limit_sim=20.0,
                stiffness=260.0,
                damping=28.0,
            ),
            "ankles": ImplicitActuatorCfg(
                joint_names_expr=list(LANDAU_ANKLE_JOINTS),
                effort_limit_sim=160.0,
                velocity_limit_sim=20.0,
                stiffness=180.0,
                damping=22.0,
            ),
            "toes": ImplicitActuatorCfg(
                joint_names_expr=list(LANDAU_TOE_JOINTS),
                effort_limit_sim=90.0,
                velocity_limit_sim=20.0,
                stiffness=90.0,
                damping=12.0,
            ),
            "torso": ImplicitActuatorCfg(
                joint_names_expr=list(LANDAU_SPEC.joint_groups.torso_joints),
                effort_limit_sim=200.0,
                velocity_limit_sim=12.0,
                stiffness=220.0,
                damping=24.0,
            ),
            "arms": ImplicitActuatorCfg(
                joint_names_expr=list((*LANDAU_SPEC.joint_groups.arm_joints, *LANDAU_SPEC.joint_groups.hand_joints)),
                effort_limit_sim=70.0,
                velocity_limit_sim=12.0,
                stiffness=70.0,
                damping=10.0,
            ),
            "fingers": ImplicitActuatorCfg(
                joint_names_expr=list(LANDAU_SPEC.joint_groups.finger_joints),
                effort_limit_sim=12.0,
                velocity_limit_sim=10.0,
                stiffness=15.0,
                damping=2.0,
            ),
        },
    )


def build_landau_height_scanner_cfg() -> RayCasterCfg:
    return RayCasterCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Robot/{LANDAU_CONTROL_ROOT_LINK}",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(
            resolution=LANDAU_HEIGHT_SCAN_RESOLUTION,
            size=list(LANDAU_HEIGHT_SCAN_SIZE),
        ),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )


def build_landau_height_scan_obs_term() -> ObsTerm:
    return ObsTerm(
        func=mdp.height_scan,
        params={"sensor_cfg": SceneEntityCfg("height_scanner")},
        noise=Unoise(n_min=-0.05, n_max=0.05),
        clip=(-1.0, 1.0),
    )


def configure_landau_controlled_joints(env_cfg, joint_names: tuple[str, ...]) -> None:
    env_cfg.actions.joint_pos.joint_names = list(joint_names)
    env_cfg.actions.joint_pos.preserve_order = True
    env_cfg.observations.policy.joint_pos.params = {"asset_cfg": SceneEntityCfg("robot", joint_names=list(joint_names))}
    env_cfg.observations.policy.joint_vel.params = {"asset_cfg": SceneEntityCfg("robot", joint_names=list(joint_names))}


def configure_landau_joint_action_offset(env_cfg, joint_targets: dict[str, float]) -> None:
    joint_pos_cfg = getattr(getattr(env_cfg, "actions", None), "joint_pos", None)
    if joint_pos_cfg is None:
        return
    controlled_joint_names = tuple(getattr(joint_pos_cfg, "joint_names", []))
    joint_pos_cfg.use_default_offset = False
    joint_pos_cfg.offset = {
        joint_name: float(joint_targets.get(joint_name, 0.0)) for joint_name in controlled_joint_names
    }


def configure_landau_policy_observations(
    env_cfg,
    *,
    include_base_lin_vel: bool,
    include_velocity_commands: bool,
    include_height_scan: bool,
) -> None:
    env_cfg.observations.policy.enable_corruption = False
    env_cfg.observations.policy.base_lin_vel = env_cfg.observations.policy.base_lin_vel if include_base_lin_vel else None
    env_cfg.observations.policy.velocity_commands = (
        env_cfg.observations.policy.velocity_commands if include_velocity_commands else None
    )
    if include_height_scan:
        env_cfg.scene.height_scanner = build_landau_height_scanner_cfg()
        env_cfg.observations.policy.height_scan = build_landau_height_scan_obs_term()
    else:
        env_cfg.scene.height_scanner = None
        env_cfg.observations.policy.height_scan = None


def configure_landau_common_commands(env_cfg) -> None:
    env_cfg.commands.base_velocity.heading_command = False
    env_cfg.commands.base_velocity.rel_heading_envs = 0.0
    env_cfg.commands.base_velocity.ranges.heading = None
    env_cfg.commands.base_velocity.debug_vis = False


def configure_landau_common_events(env_cfg) -> None:
    env_cfg.events.push_robot = None
    env_cfg.events.add_base_mass = None
    env_cfg.events.base_external_force_torque = None
    env_cfg.events.physics_material.params["static_friction_range"] = (1.0, 1.0)
    env_cfg.events.physics_material.params["dynamic_friction_range"] = (1.0, 1.0)
    env_cfg.events.physics_material.params["restitution_range"] = (0.0, 0.0)
    env_cfg.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
    env_cfg.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)


def configure_landau_common_terminations(env_cfg, *, min_root_height: float = 0.15) -> None:
    env_cfg.terminations.base_contact.params["sensor_cfg"].body_names = list(LANDAU_SPEC.termination_link_names)
    env_cfg.terminations.bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 1.0})
    env_cfg.terminations.root_height = DoneTerm(
        func=body_height_below_min_termination,
        params={
            "min_height": float(min_root_height),
            "asset_cfg": SceneEntityCfg("robot", body_names=[LANDAU_CONTROL_ROOT_LINK]),
            "reference_asset_cfg": SceneEntityCfg("robot", body_names=list(LANDAU_SUPPORT_LINKS)),
        },
    )


def configure_landau_base_env(
    env_cfg,
    *,
    num_envs: int,
    env_spacing: float,
    terrain_type: str = "plane",
    terrain_generator=None,
    max_init_terrain_level: int | None = None,
    enable_height_scan: bool = False,
    enable_terrain_curriculum: bool = False,
) -> None:
    env_cfg.scene.num_envs = num_envs
    env_cfg.scene.env_spacing = env_spacing
    env_cfg.scene.robot = build_landau_articulation_cfg()
    env_cfg.scene.terrain.terrain_type = terrain_type
    env_cfg.scene.terrain.terrain_generator = terrain_generator
    env_cfg.scene.terrain.max_init_terrain_level = max_init_terrain_level
    configure_landau_policy_observations(
        env_cfg,
        include_base_lin_vel=True,
        include_velocity_commands=True,
        include_height_scan=enable_height_scan,
    )
    env_cfg.curriculum.terrain_levels = CurrTerm(func=mdp.terrain_levels_vel) if enable_terrain_curriculum else None
    configure_landau_common_commands(env_cfg)
    configure_landau_common_events(env_cfg)
    configure_landau_common_terminations(env_cfg)


def configure_landau_reset(
    env_cfg,
    *,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]] | None = None,
) -> None:
    env_cfg.events.reset_base.params = {
        "pose_range": pose_range,
        "velocity_range": velocity_range
        or {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        },
    }


def configure_landau_playback_reset(env_cfg) -> None:
    env_cfg.scene.num_envs = 1
    env_cfg.scene.env_spacing = 2.5
    configure_landau_reset(
        env_cfg,
        pose_range={"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)},
    )


def configure_landau_base_play_env(env_cfg, *, episode_length_s: float) -> None:
    configure_landau_playback_reset(env_cfg)
    env_cfg.episode_length_s = episode_length_s
    env_cfg.events.base_external_force_torque = None
    env_cfg.events.push_robot = None
    env_cfg.commands.base_velocity.resampling_time_range = (1.0e9, 1.0e9)


def configure_landau_game_playback_scene(
    env_cfg,
    *,
    terrain_mode: str,
    obstacles_enabled: bool,
) -> None:
    env_cfg.scene.env_spacing = 3.0
    if terrain_mode == "flat":
        env_cfg.scene.terrain.terrain_type = "plane"
        env_cfg.scene.terrain.terrain_generator = None
        env_cfg.curriculum.terrain_levels = None
        return
    if terrain_mode != "game":
        raise ValueError(f"Unsupported terrain_mode '{terrain_mode}'.")

    terrain_cfg = deepcopy(LANDAU_GAME_PLAY_TERRAINS_CFG)
    if not obstacles_enabled:
        terrain_cfg.sub_terrains["obstacles"].proportion = 0.0
        terrain_cfg.sub_terrains["boxes"].proportion = 0.0
        terrain_cfg.sub_terrains["random_rough"].proportion = 0.7
        terrain_cfg.sub_terrains["stairs"].proportion = 0.1
        terrain_cfg.sub_terrains["slopes"].proportion = 0.2

    env_cfg.scene.terrain.terrain_type = "generator"
    env_cfg.scene.terrain.terrain_generator = terrain_cfg
    env_cfg.scene.terrain.max_init_terrain_level = 0
    env_cfg.curriculum.terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)
