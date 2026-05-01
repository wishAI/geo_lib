from __future__ import annotations

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg

from .landau_common import (
    LANDAU_CONTROL_ROOT_HEIGHT_FLOOR,
    LANDAU_CONTROL_ROOT_LINK,
    LANDAU_GAME_TERRAINS_CFG,
    LANDAU_GAIT_GUARD_LINKS,
    LANDAU_PRIMARY_FOOT_LINKS,
    LANDAU_SPEC,
    LANDAU_SUPPORT_LINKS,
    LANDAU_WALK_CONTROLLED_JOINTS,
    LANDAU_WALK_INIT_JOINT_POS,
    LANDAU_WALK_INIT_ROOT_HEIGHT,
    apply_landau_init_pose,
    build_landau_action_scale,
    configure_landau_base_env,
    configure_landau_base_play_env,
    configure_landau_controlled_joints,
    configure_landau_game_playback_scene,
    configure_landau_playback_actuators,
    configure_landau_reset,
    configure_landau_walk_actuators,
)
from .landau_rewards import control_root_height_floor, non_support_contacts_count, obstacle_brake_penalty
from .landau_walk_cfg import LandauFwdYawRewards


@configclass
class LandauGameRewards(LandauFwdYawRewards):
    dof_pos_limits = None
    obstacle_brake_penalty = RewTerm(
        func=obstacle_brake_penalty,
        weight=-0.5,
        params={
            "sensor_cfg": SceneEntityCfg("height_scanner"),
            "asset_cfg": SceneEntityCfg("robot", body_names=[LANDAU_CONTROL_ROOT_LINK]),
            "forward_body_axis": LANDAU_SPEC.forward_body_axis,
        },
    )


@configclass
class LandauGameEnvCfg(LocomotionVelocityRoughEnvCfg):
    rewards: LandauGameRewards = LandauGameRewards()

    def __post_init__(self):
        super().__post_init__()
        configure_landau_base_env(
            self,
            num_envs=768,
            env_spacing=5.0,
            terrain_type="generator",
            terrain_generator=LANDAU_GAME_TERRAINS_CFG,
            max_init_terrain_level=4,
            enable_height_scan=True,
            enable_terrain_curriculum=True,
        )
        configure_landau_controlled_joints(self, LANDAU_WALK_CONTROLLED_JOINTS)
        apply_landau_init_pose(self, LANDAU_WALK_INIT_JOINT_POS, LANDAU_WALK_INIT_ROOT_HEIGHT)
        configure_landau_walk_actuators(self)
        self.actions.joint_pos.scale = build_landau_action_scale(
            leg_scale=0.32,
            foot_scale=0.18,
            toe_scale=0.10,
            torso_scale=0.12,
            arm_scale=0.0,
            hand_scale=0.0,
            controlled_joint_names=LANDAU_WALK_CONTROLLED_JOINTS,
        )
        configure_landau_reset(
            self,
            pose_range={"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
        )
        self.curriculum.terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.10, 0.60)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.rel_standing_envs = 0.1
        self.rewards.control_root_height_floor.params["min_height"] = max(0.15, LANDAU_CONTROL_ROOT_HEIGHT_FLOOR - 0.02)
        self.rewards.non_support_contacts.params["sensor_cfg"] = SceneEntityCfg(
            "contact_forces", body_names=list(LANDAU_GAIT_GUARD_LINKS)
        )
        self.rewards.feet_air_time.params["sensor_cfg"] = SceneEntityCfg(
            "contact_forces", body_names=list(LANDAU_PRIMARY_FOOT_LINKS)
        )
        self.rewards.control_root_height_floor.params["asset_cfg"] = SceneEntityCfg(
            "robot", body_names=[LANDAU_CONTROL_ROOT_LINK]
        )
        self.rewards.control_root_height_floor.params["reference_asset_cfg"] = SceneEntityCfg(
            "robot", body_names=list(LANDAU_SUPPORT_LINKS)
        )


@configclass
class LandauGameEnvCfg_PLAY(LandauGameEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        configure_landau_base_play_env(self, episode_length_s=120.0)
        configure_landau_playback_actuators(self)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.2, 0.9)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        configure_landau_game_playback_scene(self, terrain_mode="flat", obstacles_enabled=False)
