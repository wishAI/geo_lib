from __future__ import annotations

import isaaclab.sim as sim_utils
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg, RewardsCfg

from .asset_setup import prepare_landau_inputs
from .custom_rewards import grouped_support_first_contact_biped
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
LANDAU_UPPER_BODY_JOINTS = tuple(
    (*LANDAU_SPEC.joint_groups.arm_joints, *LANDAU_SPEC.joint_groups.hand_joints)
)
LANDAU_PRIMARY_FOOT_LINKS = tuple(LANDAU_SPEC.primary_foot_links)
LANDAU_SUPPORT_LINKS = tuple(LANDAU_SPEC.support_link_names)
LANDAU_LEFT_SUPPORT_LINKS = tuple(name for name in LANDAU_SUPPORT_LINKS if name.endswith("_l"))
LANDAU_RIGHT_SUPPORT_LINKS = tuple(name for name in LANDAU_SUPPORT_LINKS if name.endswith("_r"))


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
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=0.75,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=list(LANDAU_PRIMARY_FOOT_LINKS)),
            "threshold": 0.4,
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
            leg_scale=0.5,
            foot_scale=0.35,
            toe_scale=0.25,
            torso_scale=0.3,
            arm_scale=0.18,
            hand_scale=0.12,
        )
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        # Bias Stage A toward a faster gait instead of converging to one comfortable walk speed.
        self.commands.base_velocity.ranges.lin_vel_y = (0.55, 1.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        # Keep forward speed dominant, but make uncontrolled yaw and slip materially expensive.
        self.rewards.track_lin_vel_xy_exp.weight = 3.0
        self.rewards.track_lin_vel_xy_exp.params["std"] = 0.2
        self.rewards.track_ang_vel_z_exp.weight = 1.0
        self.rewards.feet_air_time.weight = 0.5
        self.rewards.feet_step_contact.weight = 1.0
        self.rewards.feet_slide.weight = -0.2
        self.rewards.joint_deviation_upper.weight = -0.06
        self.rewards.joint_deviation_torso.weight = -0.05
        self.rewards.action_rate_l2.weight = -0.0025


@configclass
class LandauFwdOnlyEnvCfg_PLAY(LandauFlatEnvCfg_PLAY):
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
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 1.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)


# Stage B: forward + yaw — add heading control once forward is proven.


@configclass
class LandauFwdYawEnvCfg(LandauFlatEnvCfg):
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
        self.commands.base_velocity.ranges.lin_vel_y = (0.55, 1.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.75, 0.75)
        self.rewards.track_lin_vel_xy_exp.weight = 2.0
        self.rewards.track_lin_vel_xy_exp.params["std"] = 0.25


@configclass
class LandauFwdYawEnvCfg_PLAY(LandauFlatEnvCfg_PLAY):
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
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 1.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.75, 0.75)
