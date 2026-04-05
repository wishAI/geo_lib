from __future__ import annotations

import isaaclab.sim as sim_utils
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg, RewardsCfg

from .asset_setup import prepare_landau_inputs
from .custom_rewards import (
    alternating_biped_async_reward,
    body_height_below_min,
    command_aware_root_planar_speed_penalty,
    contact_body_alignment_penalty,
    feet_contact_state_observation,
    feet_mode_time_observation,
    feet_positions_in_root_frame,
    gait_phase_clock_observation,
    grouped_support_air_time_positive_biped,
    grouped_support_double_stance_time_penalty,
    grouped_support_flight_time_penalty,
    grouped_support_first_contact_biped,
    landing_step_ahead_reward,
    phase_clock_alternating_foot_contact_reward,
    primary_single_support_reward,
    secondary_contact_force_share_penalty,
    secondary_contact_without_primary_penalty,
    single_support_root_straddle_reward,
    support_width_above_max,
    support_width_deviation,
    swing_foot_ahead_of_stance_reward,
    touchdown_step_length_deficit_penalty,
    touchdown_support_width_excess_penalty,
    touchdown_root_straddle_reward,
    swing_height_difference_below_min,
)
from .robot_specs import load_landau_robot_spec
from .urdf_utils import load_urdf_model


prepare_landau_inputs(refresh=False)
LANDAU_SPEC = load_landau_robot_spec()
LANDAU_MODEL = load_urdf_model(LANDAU_SPEC.urdf_path)
LANDAU_ACTUATED_JOINTS = tuple(
    joint_name for joint_name, joint in LANDAU_MODEL.joints.items() if joint.joint_type != "fixed"
)
LANDAU_LOWER_BODY_JOINTS = tuple(
    (*LANDAU_SPEC.joint_groups.leg_joints, *LANDAU_SPEC.joint_groups.foot_joints)
)
LANDAU_CONTROLLED_JOINTS = tuple(
    (
        *LANDAU_LOWER_BODY_JOINTS,
        *LANDAU_SPEC.joint_groups.torso_joints,
        *LANDAU_SPEC.joint_groups.arm_joints,
        *LANDAU_SPEC.joint_groups.hand_joints,
    )
)
LANDAU_LEG_TWIST_JOINTS = tuple(joint_name for joint_name in LANDAU_SPEC.joint_groups.leg_joints if "twist" in joint_name)
LANDAU_UPPER_BODY_JOINTS = tuple(
    (*LANDAU_SPEC.joint_groups.arm_joints, *LANDAU_SPEC.joint_groups.hand_joints)
)
LANDAU_PRIMARY_FOOT_LINKS = tuple(LANDAU_SPEC.primary_foot_links)
LANDAU_PRIMARY_FOOT_CONTACT_UP_VECTORS = tuple(LANDAU_SPEC.primary_foot_contact_up_vectors)
LANDAU_LEFT_PRIMARY_FOOT_LINKS = tuple(name for name in LANDAU_PRIMARY_FOOT_LINKS if name.endswith("_l"))
LANDAU_RIGHT_PRIMARY_FOOT_LINKS = tuple(name for name in LANDAU_PRIMARY_FOOT_LINKS if name.endswith("_r"))
LANDAU_SUPPORT_LINKS = tuple(LANDAU_SPEC.support_link_names)
LANDAU_LEFT_SUPPORT_LINKS = tuple(name for name in LANDAU_SUPPORT_LINKS if name.endswith("_l"))
LANDAU_RIGHT_SUPPORT_LINKS = tuple(name for name in LANDAU_SUPPORT_LINKS if name.endswith("_r"))
LANDAU_LEFT_AUX_SUPPORT_LINKS = tuple(name for name in LANDAU_LEFT_SUPPORT_LINKS if name not in LANDAU_LEFT_PRIMARY_FOOT_LINKS)
LANDAU_RIGHT_AUX_SUPPORT_LINKS = tuple(name for name in LANDAU_RIGHT_SUPPORT_LINKS if name not in LANDAU_RIGHT_PRIMARY_FOOT_LINKS)
LANDAU_CONTROL_ROOT_LINK = LANDAU_SPEC.control_root_link or LANDAU_SPEC.root_link_name
LANDAU_GAIT_GUARD_LINKS = tuple(LANDAU_SPEC.gait_guard_link_names)
LANDAU_CONTROL_ROOT_HEIGHT_FLOOR = max(0.17, float(LANDAU_SPEC.nominal_control_root_height) * 0.65)
LANDAU_UPRIGHT_CONTROL_ROOT_HEIGHT = max(
    LANDAU_CONTROL_ROOT_HEIGHT_FLOOR + 0.03,
    float(LANDAU_SPEC.nominal_control_root_height) * 0.9,
)
LANDAU_NOMINAL_STANCE_WIDTH = max(0.12, float(LANDAU_SPEC.nominal_stance_width))
LANDAU_MAX_WALK_STANCE_WIDTH = max(LANDAU_NOMINAL_STANCE_WIDTH * 1.3, LANDAU_NOMINAL_STANCE_WIDTH + 0.06)


def build_landau_action_scale(
    *,
    leg_scale: float,
    foot_scale: float,
    toe_scale: float,
    torso_scale: float,
    arm_scale: float,
    hand_scale: float,
    head_scale: float = 0.1,
) -> dict[str, float]:
    return {
        "thigh_.*": leg_scale,
        "leg_.*": leg_scale,
        "foot_.*": foot_scale,
        "toes_.*": toe_scale,
        "spine_.*": torso_scale,
        "neck_.*": torso_scale,
        "head_.*": head_scale,
        "shoulder_.*": arm_scale,
        "arm_.*": arm_scale,
        "forearm_.*": arm_scale,
        "hand_.*": hand_scale,
    }


def build_landau_articulation_cfg() -> ArticulationCfg:
    joint_defaults = {joint_name: 0.0 for joint_name in LANDAU_ACTUATED_JOINTS}
    joint_defaults.update(LANDAU_SPEC.default_joint_positions)
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
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=4,
            ),
            joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=None, damping=None)
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, float(LANDAU_SPEC.init_root_height)),
            joint_pos=joint_defaults,
            joint_vel={".*": 0.0},
        ),
        soft_joint_pos_limit_factor=0.9,
        actuators={
            "legs": ImplicitActuatorCfg(
                joint_names_expr=list(LANDAU_SPEC.joint_groups.leg_joints),
                effort_limit_sim=120.0,
                velocity_limit_sim=20.0,
                stiffness=140.0,
                damping=10.0,
            ),
            "feet": ImplicitActuatorCfg(
                joint_names_expr=list(LANDAU_SPEC.joint_groups.foot_joints),
                effort_limit_sim=80.0,
                velocity_limit_sim=20.0,
                stiffness=60.0,
                damping=6.0,
            ),
            "torso": ImplicitActuatorCfg(
                joint_names_expr=list(LANDAU_SPEC.joint_groups.torso_joints),
                effort_limit_sim=80.0,
                velocity_limit_sim=12.0,
                stiffness=80.0,
                damping=8.0,
            ),
            "arms": ImplicitActuatorCfg(
                joint_names_expr=list(LANDAU_UPPER_BODY_JOINTS),
                effort_limit_sim=40.0,
                velocity_limit_sim=12.0,
                stiffness=40.0,
                damping=4.0,
            ),
            "fingers": ImplicitActuatorCfg(
                joint_names_expr=list(LANDAU_SPEC.joint_groups.finger_joints),
                effort_limit_sim=12.0,
                velocity_limit_sim=10.0,
                stiffness=15.0,
                damping=1.5,
            ),
        },
    )


@configclass
class LandauRewards(RewardsCfg):
    """Starting reward weights for the custom flat-ground biped task."""

    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=1.5,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    phase_clock_gait = RewTerm(
        func=phase_clock_alternating_foot_contact_reward,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "phase_sharpness": 4.0,
            "slow_speed": 0.3,
            "fast_speed": 0.8,
            "slow_period": 0.6,
            "fast_period": 0.4,
            "left_sensor_cfg": SceneEntityCfg("contact_forces", body_names=list(LANDAU_LEFT_PRIMARY_FOOT_LINKS)),
            "right_sensor_cfg": SceneEntityCfg("contact_forces", body_names=list(LANDAU_RIGHT_PRIMARY_FOOT_LINKS)),
        },
    )
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=0.75,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=list(LANDAU_PRIMARY_FOOT_LINKS)),
            "threshold": 0.4,
        },
    )
    gait_async_timing = RewTerm(
        func=alternating_biped_async_reward,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "std": 0.03,
            "max_err": 0.25,
            "left_sensor_cfg": SceneEntityCfg("contact_forces", body_names=list(LANDAU_LEFT_SUPPORT_LINKS)),
            "right_sensor_cfg": SceneEntityCfg("contact_forces", body_names=list(LANDAU_RIGHT_SUPPORT_LINKS)),
        },
    )
    primary_single_support = RewTerm(
        func=primary_single_support_reward,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "overlap_grace_period": 0.08,
            "left_sensor_cfg": SceneEntityCfg("contact_forces", body_names=list(LANDAU_LEFT_PRIMARY_FOOT_LINKS)),
            "right_sensor_cfg": SceneEntityCfg("contact_forces", body_names=list(LANDAU_RIGHT_PRIMARY_FOOT_LINKS)),
        },
    )
    feet_step_contact = RewTerm(
        func=grouped_support_first_contact_biped,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "threshold": 0.4,
            "left_sensor_cfg": SceneEntityCfg("contact_forces", body_names=list(LANDAU_LEFT_SUPPORT_LINKS)),
            "right_sensor_cfg": SceneEntityCfg("contact_forces", body_names=list(LANDAU_RIGHT_SUPPORT_LINKS)),
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=list(LANDAU_PRIMARY_FOOT_LINKS)),
            "asset_cfg": SceneEntityCfg("robot", body_names=list(LANDAU_PRIMARY_FOOT_LINKS)),
        },
    )
    non_support_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-0.5,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=list(LANDAU_GAIT_GUARD_LINKS)),
            "threshold": 5.0,
        },
    )
    non_support_contact_force = RewTerm(
        func=mdp.contact_forces,
        weight=-0.01,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=list(LANDAU_GAIT_GUARD_LINKS)),
            "threshold": 5.0,
        },
    )
    control_root_height_floor = RewTerm(
        func=body_height_below_min,
        weight=-2.0,
        params={
            "min_height": LANDAU_CONTROL_ROOT_HEIGHT_FLOOR,
            "asset_cfg": SceneEntityCfg("robot", body_names=[LANDAU_CONTROL_ROOT_LINK]),
        },
    )
    control_root_height_target = RewTerm(
        func=body_height_below_min,
        weight=0.0,
        params={
            "min_height": LANDAU_UPRIGHT_CONTROL_ROOT_HEIGHT,
            "asset_cfg": SceneEntityCfg("robot", body_names=[LANDAU_CONTROL_ROOT_LINK]),
        },
    )
    idle_root_planar_speed = RewTerm(
        func=command_aware_root_planar_speed_penalty,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "max_command_speed": 0.05,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    stance_width_deviation = RewTerm(
        func=support_width_deviation,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "target_width": LANDAU_NOMINAL_STANCE_WIDTH,
            "tolerance": 0.04,
            "forward_body_axis": LANDAU_SPEC.forward_body_axis,
            "left_asset_cfg": SceneEntityCfg("robot", body_names=list(LANDAU_LEFT_SUPPORT_LINKS)),
            "right_asset_cfg": SceneEntityCfg("robot", body_names=list(LANDAU_RIGHT_SUPPORT_LINKS)),
        },
    )
    stance_width_excess = RewTerm(
        func=support_width_above_max,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "max_width": LANDAU_MAX_WALK_STANCE_WIDTH,
            "forward_body_axis": LANDAU_SPEC.forward_body_axis,
            "left_asset_cfg": SceneEntityCfg("robot", body_names=list(LANDAU_LEFT_SUPPORT_LINKS)),
            "right_asset_cfg": SceneEntityCfg("robot", body_names=list(LANDAU_RIGHT_SUPPORT_LINKS)),
        },
    )
    touchdown_step_length_deficit = RewTerm(
        func=touchdown_step_length_deficit_penalty,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "min_step_length": 0.04,
            "min_air_time": 0.08,
            "max_penalized_deficit": 0.12,
            "forward_body_axis": LANDAU_SPEC.forward_body_axis,
            "left_sensor_cfg": SceneEntityCfg("contact_forces", body_names=list(LANDAU_LEFT_PRIMARY_FOOT_LINKS)),
            "right_sensor_cfg": SceneEntityCfg("contact_forces", body_names=list(LANDAU_RIGHT_PRIMARY_FOOT_LINKS)),
            "left_asset_cfg": SceneEntityCfg("robot", body_names=list(LANDAU_LEFT_PRIMARY_FOOT_LINKS)),
            "right_asset_cfg": SceneEntityCfg("robot", body_names=list(LANDAU_RIGHT_PRIMARY_FOOT_LINKS)),
        },
    )
    touchdown_support_width_excess = RewTerm(
        func=touchdown_support_width_excess_penalty,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "max_width": max(LANDAU_NOMINAL_STANCE_WIDTH * 1.25, LANDAU_NOMINAL_STANCE_WIDTH + 0.05),
            "min_air_time": 0.05,
            "max_penalized_excess": 0.08,
            "forward_body_axis": LANDAU_SPEC.forward_body_axis,
            "left_sensor_cfg": SceneEntityCfg("contact_forces", body_names=list(LANDAU_LEFT_SUPPORT_LINKS)),
            "right_sensor_cfg": SceneEntityCfg("contact_forces", body_names=list(LANDAU_RIGHT_SUPPORT_LINKS)),
            "left_asset_cfg": SceneEntityCfg("robot", body_names=list(LANDAU_LEFT_SUPPORT_LINKS)),
            "right_asset_cfg": SceneEntityCfg("robot", body_names=list(LANDAU_RIGHT_SUPPORT_LINKS)),
        },
    )
    landing_step_ahead = RewTerm(
        func=landing_step_ahead_reward,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "step_length_threshold": 0.05,
            "min_air_time": 0.18,
            "max_rewarded_step_length": 0.08,
            "forward_body_axis": LANDAU_SPEC.forward_body_axis,
            "left_sensor_cfg": SceneEntityCfg("contact_forces", body_names=list(LANDAU_LEFT_PRIMARY_FOOT_LINKS)),
            "right_sensor_cfg": SceneEntityCfg("contact_forces", body_names=list(LANDAU_RIGHT_PRIMARY_FOOT_LINKS)),
            "left_asset_cfg": SceneEntityCfg("robot", body_names=list(LANDAU_LEFT_PRIMARY_FOOT_LINKS)),
            "right_asset_cfg": SceneEntityCfg("robot", body_names=list(LANDAU_RIGHT_PRIMARY_FOOT_LINKS)),
        },
    )
    touchdown_root_straddle = RewTerm(
        func=touchdown_root_straddle_reward,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "landing_margin": 0.03,
            "stance_margin": 0.01,
            "min_air_time": 0.12,
            "max_rewarded_margin": 0.12,
            "forward_body_axis": LANDAU_SPEC.forward_body_axis,
            "root_asset_cfg": SceneEntityCfg("robot", body_names=[LANDAU_CONTROL_ROOT_LINK]),
            "left_sensor_cfg": SceneEntityCfg("contact_forces", body_names=list(LANDAU_LEFT_PRIMARY_FOOT_LINKS)),
            "right_sensor_cfg": SceneEntityCfg("contact_forces", body_names=list(LANDAU_RIGHT_PRIMARY_FOOT_LINKS)),
            "left_asset_cfg": SceneEntityCfg("robot", body_names=list(LANDAU_LEFT_PRIMARY_FOOT_LINKS)),
            "right_asset_cfg": SceneEntityCfg("robot", body_names=list(LANDAU_RIGHT_PRIMARY_FOOT_LINKS)),
        },
    )
    swing_foot_ahead_of_stance = RewTerm(
        func=swing_foot_ahead_of_stance_reward,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "min_step_length": 0.03,
            "max_rewarded_step_length": 0.1,
            "forward_body_axis": LANDAU_SPEC.forward_body_axis,
            "left_sensor_cfg": SceneEntityCfg("contact_forces", body_names=list(LANDAU_LEFT_PRIMARY_FOOT_LINKS)),
            "right_sensor_cfg": SceneEntityCfg("contact_forces", body_names=list(LANDAU_RIGHT_PRIMARY_FOOT_LINKS)),
            "left_asset_cfg": SceneEntityCfg("robot", body_names=list(LANDAU_LEFT_PRIMARY_FOOT_LINKS)),
            "right_asset_cfg": SceneEntityCfg("robot", body_names=list(LANDAU_RIGHT_PRIMARY_FOOT_LINKS)),
        },
    )
    single_support_root_straddle = RewTerm(
        func=single_support_root_straddle_reward,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "root_margin": 0.015,
            "max_rewarded_margin": 0.08,
            "forward_body_axis": LANDAU_SPEC.forward_body_axis,
            "root_asset_cfg": SceneEntityCfg("robot", body_names=[LANDAU_CONTROL_ROOT_LINK]),
            "left_sensor_cfg": SceneEntityCfg("contact_forces", body_names=list(LANDAU_LEFT_PRIMARY_FOOT_LINKS)),
            "right_sensor_cfg": SceneEntityCfg("contact_forces", body_names=list(LANDAU_RIGHT_PRIMARY_FOOT_LINKS)),
            "left_asset_cfg": SceneEntityCfg("robot", body_names=list(LANDAU_LEFT_PRIMARY_FOOT_LINKS)),
            "right_asset_cfg": SceneEntityCfg("robot", body_names=list(LANDAU_RIGHT_PRIMARY_FOOT_LINKS)),
        },
    )
    flight_time_penalty = RewTerm(
        func=grouped_support_flight_time_penalty,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "threshold": 0.04,
            "left_sensor_cfg": SceneEntityCfg("contact_forces", body_names=list(LANDAU_LEFT_PRIMARY_FOOT_LINKS)),
            "right_sensor_cfg": SceneEntityCfg("contact_forces", body_names=list(LANDAU_RIGHT_PRIMARY_FOOT_LINKS)),
        },
    )
    double_support_time_penalty = RewTerm(
        func=grouped_support_double_stance_time_penalty,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "threshold": 0.12,
            "left_sensor_cfg": SceneEntityCfg("contact_forces", body_names=list(LANDAU_LEFT_PRIMARY_FOOT_LINKS)),
            "right_sensor_cfg": SceneEntityCfg("contact_forces", body_names=list(LANDAU_RIGHT_PRIMARY_FOOT_LINKS)),
        },
    )
    swing_height_difference_floor = RewTerm(
        func=swing_height_difference_below_min,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "min_height_difference": 0.025,
            "left_sensor_cfg": SceneEntityCfg("contact_forces", body_names=list(LANDAU_LEFT_PRIMARY_FOOT_LINKS)),
            "right_sensor_cfg": SceneEntityCfg("contact_forces", body_names=list(LANDAU_RIGHT_PRIMARY_FOOT_LINKS)),
            "left_asset_cfg": SceneEntityCfg("robot", body_names=list(LANDAU_LEFT_PRIMARY_FOOT_LINKS)),
            "right_asset_cfg": SceneEntityCfg("robot", body_names=list(LANDAU_RIGHT_PRIMARY_FOOT_LINKS)),
        },
    )
    primary_foot_flat_contact = RewTerm(
        func=contact_body_alignment_penalty,
        weight=0.0,
        params={
            "min_cosine": 0.9,
            "contact_threshold": 5.0,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=list(LANDAU_PRIMARY_FOOT_LINKS)),
            "asset_cfg": SceneEntityCfg("robot", body_names=list(LANDAU_PRIMARY_FOOT_LINKS)),
            "local_reference_vectors": LANDAU_PRIMARY_FOOT_CONTACT_UP_VECTORS,
        },
    )
    aux_support_without_primary = RewTerm(
        func=secondary_contact_without_primary_penalty,
        weight=0.0,
        params={
            "threshold": 5.0,
            "left_primary_sensor_cfg": SceneEntityCfg("contact_forces", body_names=list(LANDAU_LEFT_PRIMARY_FOOT_LINKS)),
            "right_primary_sensor_cfg": SceneEntityCfg("contact_forces", body_names=list(LANDAU_RIGHT_PRIMARY_FOOT_LINKS)),
            "left_aux_sensor_cfg": SceneEntityCfg("contact_forces", body_names=list(LANDAU_LEFT_AUX_SUPPORT_LINKS)),
            "right_aux_sensor_cfg": SceneEntityCfg("contact_forces", body_names=list(LANDAU_RIGHT_AUX_SUPPORT_LINKS)),
        },
    )
    aux_support_force_share = RewTerm(
        func=secondary_contact_force_share_penalty,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "threshold": 5.0,
            "min_primary_force_share": 0.7,
            "left_primary_sensor_cfg": SceneEntityCfg("contact_forces", body_names=list(LANDAU_LEFT_PRIMARY_FOOT_LINKS)),
            "right_primary_sensor_cfg": SceneEntityCfg("contact_forces", body_names=list(LANDAU_RIGHT_PRIMARY_FOOT_LINKS)),
            "left_aux_sensor_cfg": SceneEntityCfg("contact_forces", body_names=list(LANDAU_LEFT_AUX_SUPPORT_LINKS)),
            "right_aux_sensor_cfg": SceneEntityCfg("contact_forces", body_names=list(LANDAU_RIGHT_AUX_SUPPORT_LINKS)),
        },
    )
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-0.5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=list(LANDAU_SPEC.joint_groups.foot_joints))},
    )
    joint_deviation_upper = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.03,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=list(LANDAU_UPPER_BODY_JOINTS))},
    )
    joint_deviation_leg_twist = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=list(LANDAU_LEG_TWIST_JOINTS))},
    )
    joint_deviation_feet = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=list(LANDAU_SPEC.joint_groups.foot_joints))},
    )
    joint_deviation_fingers = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.05,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=list(LANDAU_SPEC.joint_groups.finger_joints))},
    )
    joint_deviation_torso = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.02,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=list(LANDAU_SPEC.joint_groups.torso_joints))},
    )


@configclass
class LandauFlatEnvCfg(LocomotionVelocityRoughEnvCfg):
    rewards: LandauRewards = LandauRewards()

    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 1024
        self.scene.env_spacing = 2.5
        self.scene.robot = build_landau_articulation_cfg()

        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        self.curriculum.terrain_levels = None

        self.actions.joint_pos.joint_names = list(LANDAU_CONTROLLED_JOINTS)
        self.actions.joint_pos.preserve_order = True
        self.actions.joint_pos.scale = build_landau_action_scale(
            leg_scale=0.35,
            foot_scale=0.25,
            toe_scale=0.2,
            torso_scale=0.25,
            arm_scale=0.2,
            hand_scale=0.15,
        )
        self.observations.policy.joint_pos.params = {
            "asset_cfg": SceneEntityCfg("robot", joint_names=list(LANDAU_CONTROLLED_JOINTS))
        }
        self.observations.policy.joint_vel.params = {
            "asset_cfg": SceneEntityCfg("robot", joint_names=list(LANDAU_CONTROLLED_JOINTS))
        }
        self.observations.policy.gait_phase = ObsTerm(
            func=gait_phase_clock_observation,
            params={
                "command_name": "base_velocity",
                "slow_speed": 0.3,
                "fast_speed": 0.8,
                "slow_period": 0.6,
                "fast_period": 0.4,
            },
        )
        self.observations.policy.foot_positions = ObsTerm(
            func=feet_positions_in_root_frame,
            params={
                "root_asset_cfg": SceneEntityCfg("robot", body_names=[LANDAU_CONTROL_ROOT_LINK]),
                "left_asset_cfg": SceneEntityCfg("robot", body_names=list(LANDAU_LEFT_PRIMARY_FOOT_LINKS)),
                "right_asset_cfg": SceneEntityCfg("robot", body_names=list(LANDAU_RIGHT_PRIMARY_FOOT_LINKS)),
            },
        )
        self.observations.policy.foot_contact_state = ObsTerm(
            func=feet_contact_state_observation,
            params={
                "left_sensor_cfg": SceneEntityCfg("contact_forces", body_names=list(LANDAU_LEFT_PRIMARY_FOOT_LINKS)),
                "right_sensor_cfg": SceneEntityCfg("contact_forces", body_names=list(LANDAU_RIGHT_PRIMARY_FOOT_LINKS)),
            },
        )
        self.observations.policy.foot_mode_time = ObsTerm(
            func=feet_mode_time_observation,
            params={
                "left_sensor_cfg": SceneEntityCfg("contact_forces", body_names=list(LANDAU_LEFT_PRIMARY_FOOT_LINKS)),
                "right_sensor_cfg": SceneEntityCfg("contact_forces", body_names=list(LANDAU_RIGHT_PRIMARY_FOOT_LINKS)),
                "time_window": 1.0,
            },
        )
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.rel_heading_envs = 0.0
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.ranges.heading = None
        self.commands.base_velocity.debug_vis = False
        self.commands.base_velocity.ranges.lin_vel_x = (-0.5, 0.5)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.75, 0.75)

        self.events.push_robot = None
        self.events.add_base_mass = None
        self.events.base_external_force_torque = None
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.25, 0.25), "y": (-0.25, 0.25), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }

        self.rewards.lin_vel_z_l2.weight = -0.2
        self.rewards.ang_vel_xy_l2.weight = -0.05
        self.rewards.flat_orientation_l2.weight = -1.0
        self.rewards.action_rate_l2.weight = -0.005
        self.rewards.dof_acc_l2.weight = -1.0e-7
        self.rewards.dof_acc_l2.params["asset_cfg"] = SceneEntityCfg("robot", joint_names=list(LANDAU_LOWER_BODY_JOINTS))
        self.rewards.dof_torques_l2.weight = -5.0e-7
        self.rewards.dof_torques_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=list(LANDAU_LOWER_BODY_JOINTS)
        )
        self.rewards.undesired_contacts = None

        self.terminations.base_contact.params["sensor_cfg"].body_names = list(LANDAU_SPEC.termination_link_names)


@configclass
class LandauFlatEnvCfg_PLAY(LandauFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 8
        self.scene.env_spacing = 2.5
        self.episode_length_s = 60.0
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.resampling_time_range = (1.0e9, 1.0e9)


# ---------------------------------------------------------------------------
# Staged curriculum configs
# ---------------------------------------------------------------------------
# Stage A: forward-only — learn to translate before anything else.
# Landau forward = body Y, so we command lin_vel_y only.


@configclass
class LandauFwdOnlyEnvCfg(LandauFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.actions.joint_pos.scale = build_landau_action_scale(
            leg_scale=0.4,
            foot_scale=0.24,
            toe_scale=0.10,
            torso_scale=0.26,
            arm_scale=0.18,
            hand_scale=0.12,
        )
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        # Keep the command band energetic enough to encourage real stepping, but slightly narrower than the original v1.
        self.commands.base_velocity.ranges.lin_vel_y = (0.4, 0.7)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.rel_standing_envs = 0.0
        # Keep forward speed dominant, but use a soft command gate so the clock reward still teaches stepping early on.
        self.rewards.track_lin_vel_xy_exp.weight = 5.0
        self.rewards.track_lin_vel_xy_exp.params["std"] = 0.2
        self.rewards.track_ang_vel_z_exp.weight = 2.0
        self.rewards.phase_clock_gait.params["phase_sharpness"] = 4.0
        self.rewards.phase_clock_gait.params["slow_speed"] = 0.3
        self.rewards.phase_clock_gait.params["fast_speed"] = 0.8
        self.rewards.phase_clock_gait.params["slow_period"] = 0.6
        self.rewards.phase_clock_gait.params["fast_period"] = 0.4
        self.rewards.phase_clock_gait.params["velocity_gate_std"] = 0.55
        self.rewards.phase_clock_gait.params["velocity_gate_floor"] = 0.25
        self.rewards.phase_clock_gait.params["yaw_rate_gate_std"] = 0.6
        self.rewards.phase_clock_gait.params["yaw_rate_gate_floor"] = 0.25
        self.rewards.phase_clock_gait.params["asset_cfg"] = SceneEntityCfg("robot")
        self.rewards.phase_clock_gait.weight = 5.0
        self.rewards.gait_async_timing.weight = 0.0
        self.rewards.primary_single_support.weight = 0.5
        self.rewards.feet_air_time.weight = 0.0
        self.rewards.feet_step_contact.weight = 0.0
        self.rewards.feet_slide.params = {
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=list(LANDAU_PRIMARY_FOOT_LINKS)),
            "asset_cfg": SceneEntityCfg("robot", body_names=list(LANDAU_PRIMARY_FOOT_LINKS)),
        }
        self.rewards.feet_slide.weight = -0.08
        self.rewards.non_support_contacts.weight = -1.5
        self.rewards.non_support_contact_force.weight = -0.02
        self.rewards.control_root_height_floor.weight = -15.0
        self.rewards.control_root_height_target.weight = 0.0
        self.rewards.idle_root_planar_speed.params["max_command_speed"] = 0.05
        self.rewards.idle_root_planar_speed.weight = 0.0
        self.rewards.stance_width_deviation.weight = -0.25
        self.rewards.stance_width_excess.weight = -3.0
        self.rewards.touchdown_step_length_deficit.params["min_step_length"] = 0.03
        self.rewards.touchdown_step_length_deficit.params["min_air_time"] = 0.05
        self.rewards.touchdown_step_length_deficit.params["max_penalized_deficit"] = 0.16
        self.rewards.touchdown_step_length_deficit.weight = -1.0
        self.rewards.touchdown_support_width_excess.weight = -2.0
        self.rewards.landing_step_ahead.params["step_length_threshold"] = 0.04
        self.rewards.landing_step_ahead.params["min_air_time"] = 0.08
        self.rewards.landing_step_ahead.params["max_rewarded_step_length"] = 0.1
        self.rewards.landing_step_ahead.weight = 2.5
        self.rewards.touchdown_root_straddle.weight = 0.0
        self.rewards.swing_foot_ahead_of_stance.weight = 0.0
        self.rewards.single_support_root_straddle.weight = 0.0
        self.rewards.flight_time_penalty.weight = 0.0
        self.rewards.double_support_time_penalty.weight = 0.0
        self.rewards.swing_height_difference_floor.weight = 0.0
        self.rewards.primary_foot_flat_contact.weight = 0.0
        self.rewards.aux_support_without_primary.weight = 0.0
        self.rewards.aux_support_force_share.weight = 0.0
        self.rewards.dof_pos_limits.weight = -0.25
        self.rewards.joint_deviation_upper.weight = -0.03
        self.rewards.joint_deviation_leg_twist.weight = -0.12
        self.rewards.joint_deviation_feet.weight = -0.06
        self.rewards.joint_deviation_torso.weight = -0.03
        self.rewards.action_rate_l2.weight = -0.005
        self.terminations.gait_guard_contact = DoneTerm(
            func=mdp.illegal_contact,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=list(LANDAU_GAIT_GUARD_LINKS)),
                "threshold": 10.0,
            },
        )


@configclass
class LandauFwdOnlyEnvCfg_PLAY(LandauFwdOnlyEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 8
        self.episode_length_s = 60.0
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.resampling_time_range = (1.0e9, 1.0e9)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.2, 1.0)


# Stage B: forward + yaw — add heading control once forward is proven.


@configclass
class LandauFwdYawEnvCfg(LandauFwdOnlyEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.actions.joint_pos.scale = build_landau_action_scale(
            leg_scale=0.5,
            foot_scale=0.35,
            toe_scale=0.25,
            torso_scale=0.3,
            arm_scale=0.18,
            hand_scale=0.12,
        )
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.45, 1.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.75, 0.75)
        self.rewards.track_lin_vel_xy_exp.weight = 2.0
        self.rewards.track_lin_vel_xy_exp.params["std"] = 0.25


@configclass
class LandauFwdYawEnvCfg_PLAY(LandauFwdYawEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 8
        self.episode_length_s = 60.0
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.resampling_time_range = (1.0e9, 1.0e9)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.2, 1.0)
