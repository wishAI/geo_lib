from __future__ import annotations

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.classic.humanoid.mdp.rewards import upright_posture_bonus
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg, RewardsCfg

from .landau_common import (
    LANDAU_CONTROL_ROOT_HEIGHT_FLOOR,
    LANDAU_CONTROL_ROOT_LINK,
    LANDAU_CORE_CONTROLLED_JOINTS,
    LANDAU_GAIT_GUARD_LINKS,
    LANDAU_STAND_INIT_JOINT_POS,
    LANDAU_STAND_INIT_ROOT_HEIGHT,
    LANDAU_SUPPORT_LINKS,
    apply_landau_init_pose,
    build_landau_action_scale,
    configure_landau_base_env,
    configure_landau_base_play_env,
    configure_landau_controlled_joints,
    configure_landau_joint_action_offset,
    configure_landau_policy_observations,
    configure_landau_reset,
    configure_landau_stand_actuators,
)
from .landau_rewards import control_root_height_floor, non_support_contacts_count


@configclass
class LandauStandRewards(RewardsCfg):
    track_lin_vel_xy_exp = None
    track_ang_vel_z_exp = None
    lin_vel_z_l2 = None
    ang_vel_xy_l2 = None
    dof_torques_l2 = None
    dof_acc_l2 = None
    feet_air_time = None
    undesired_contacts = None
    flat_orientation_l2 = None

    upright_posture = RewTerm(func=upright_posture_bonus, weight=4.0, params={"threshold": 0.95})
    control_root_height_floor = RewTerm(
        func=control_root_height_floor,
        weight=-12.0,
        params={
            "min_height": LANDAU_CONTROL_ROOT_HEIGHT_FLOOR,
            "asset_cfg": SceneEntityCfg("robot", body_names=[LANDAU_CONTROL_ROOT_LINK]),
            "reference_asset_cfg": SceneEntityCfg("robot", body_names=list(LANDAU_SUPPORT_LINKS)),
        },
    )
    non_support_contacts = RewTerm(
        func=non_support_contacts_count,
        weight=-0.5,
        params={"threshold": 5.0, "sensor_cfg": SceneEntityCfg("contact_forces", body_names=list(LANDAU_GAIT_GUARD_LINKS))},
    )
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.002)
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-0.05,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=list(LANDAU_CORE_CONTROLLED_JOINTS))},
    )


@configclass
class LandauStandEnvCfg(LocomotionVelocityRoughEnvCfg):
    rewards: LandauStandRewards = LandauStandRewards()

    def __post_init__(self):
        super().__post_init__()
        configure_landau_base_env(self, num_envs=1024, env_spacing=2.5)
        configure_landau_controlled_joints(self, LANDAU_CORE_CONTROLLED_JOINTS)
        configure_landau_policy_observations(
            self,
            include_base_lin_vel=False,
            include_velocity_commands=False,
            include_height_scan=False,
        )
        apply_landau_init_pose(self, LANDAU_STAND_INIT_JOINT_POS, LANDAU_STAND_INIT_ROOT_HEIGHT)
        configure_landau_joint_action_offset(self, LANDAU_STAND_INIT_JOINT_POS)
        configure_landau_stand_actuators(self)
        self.actions.joint_pos.scale = build_landau_action_scale(
            leg_scale=0.10,
            foot_scale=0.06,
            toe_scale=0.04,
            torso_scale=0.05,
            arm_scale=0.0,
            hand_scale=0.0,
            controlled_joint_names=LANDAU_CORE_CONTROLLED_JOINTS,
        )
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.rel_standing_envs = 1.0
        self.commands.base_velocity.resampling_time_range = (1.0e9, 1.0e9)
        configure_landau_reset(
            self,
            pose_range={"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)},
        )


@configclass
class LandauStandEnvCfg_PLAY(LandauStandEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        configure_landau_base_play_env(self, episode_length_s=60.0)
        configure_landau_stand_actuators(self)
