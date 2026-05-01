from __future__ import annotations

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg, RewardsCfg

from .landau_common import (
    LANDAU_CONTROL_ROOT_HEIGHT_FLOOR,
    LANDAU_CONTROL_ROOT_LINK,
    LANDAU_GAIT_GUARD_LINKS,
    LANDAU_PRIMARY_FOOT_LINKS,
    LANDAU_SUPPORT_LINKS,
    LANDAU_WALK_CONTROLLED_JOINTS,
    LANDAU_WALK_INIT_JOINT_POS,
    LANDAU_WALK_INIT_ROOT_HEIGHT,
    apply_landau_init_pose,
    build_landau_action_scale,
    configure_landau_base_env,
    configure_landau_base_play_env,
    configure_landau_controlled_joints,
    configure_landau_policy_observations,
    configure_landau_playback_actuators,
    configure_landau_reset,
    configure_landau_walk_actuators,
)
from .landau_rewards import control_root_height_floor, non_support_contacts_count


@configclass
class LandauFwdOnlyRewards(RewardsCfg):
    lin_vel_z_l2 = None
    ang_vel_xy_l2 = None
    dof_torques_l2 = None
    dof_acc_l2 = None
    undesired_contacts = None
    flat_orientation_l2 = None

    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=2.5,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    track_ang_vel_z_exp = None
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=0.4,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=list(LANDAU_PRIMARY_FOOT_LINKS)),
            "threshold": 0.4,
        },
    )
    control_root_height_floor = RewTerm(
        func=control_root_height_floor,
        weight=-4.0,
        params={
            "min_height": LANDAU_CONTROL_ROOT_HEIGHT_FLOOR,
            "asset_cfg": SceneEntityCfg("robot", body_names=[LANDAU_CONTROL_ROOT_LINK]),
            "reference_asset_cfg": SceneEntityCfg("robot", body_names=list(LANDAU_SUPPORT_LINKS)),
        },
    )
    non_support_contacts = RewTerm(
        func=non_support_contacts_count,
        weight=-0.4,
        params={"threshold": 5.0, "sensor_cfg": SceneEntityCfg("contact_forces", body_names=list(LANDAU_GAIT_GUARD_LINKS))},
    )
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.005)
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-0.05,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=list(LANDAU_WALK_CONTROLLED_JOINTS))},
    )


@configclass
class LandauFwdYawRewards(LandauFwdOnlyRewards):
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=1.5,
        params={"command_name": "base_velocity", "std": 0.5},
    )


def _configure_walk_base(env_cfg) -> None:
    configure_landau_base_env(env_cfg, num_envs=1024, env_spacing=2.5)
    configure_landau_controlled_joints(env_cfg, LANDAU_WALK_CONTROLLED_JOINTS)
    configure_landau_policy_observations(
        env_cfg,
        include_base_lin_vel=True,
        include_velocity_commands=True,
        include_height_scan=False,
    )
    apply_landau_init_pose(env_cfg, LANDAU_WALK_INIT_JOINT_POS, LANDAU_WALK_INIT_ROOT_HEIGHT)
    configure_landau_walk_actuators(env_cfg)
    env_cfg.actions.joint_pos.scale = build_landau_action_scale(
        leg_scale=0.35,
        foot_scale=0.18,
        toe_scale=0.10,
        torso_scale=0.12,
        arm_scale=0.0,
        hand_scale=0.0,
        controlled_joint_names=LANDAU_WALK_CONTROLLED_JOINTS,
    )
    configure_landau_reset(
        env_cfg,
        pose_range={"x": (-0.25, 0.25), "y": (-0.25, 0.25), "yaw": (-3.14, 3.14)},
    )
    env_cfg.commands.base_velocity.rel_standing_envs = 0.0


@configclass
class LandauFwdOnlyEnvCfg(LocomotionVelocityRoughEnvCfg):
    rewards: LandauFwdOnlyRewards = LandauFwdOnlyRewards()

    def __post_init__(self):
        super().__post_init__()
        _configure_walk_base(self)
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.20, 0.55)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)


@configclass
class LandauFwdOnlyEnvCfg_PLAY(LandauFwdOnlyEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        configure_landau_base_play_env(self, episode_length_s=60.0)
        configure_landau_playback_actuators(self)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.25, 0.9)


@configclass
class LandauFwdYawEnvCfg(LocomotionVelocityRoughEnvCfg):
    rewards: LandauFwdYawRewards = LandauFwdYawRewards()

    def __post_init__(self):
        super().__post_init__()
        _configure_walk_base(self)
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.20, 0.55)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.7, 0.7)


@configclass
class LandauFwdYawEnvCfg_PLAY(LandauFwdYawEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        configure_landau_base_play_env(self, episode_length_s=60.0)
        configure_landau_playback_actuators(self)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.25, 0.9)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)


@configclass
class LandauFlatEnvCfg(LandauFwdYawEnvCfg):
    pass


@configclass
class LandauFlatEnvCfg_PLAY(LandauFwdYawEnvCfg_PLAY):
    pass
