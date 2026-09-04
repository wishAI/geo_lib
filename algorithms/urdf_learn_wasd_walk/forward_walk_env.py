"""Manager-based flat forward-walking extension of the proven Landau stand task.

Import only after Isaac Lab's AppLauncher has started Kit.
"""

from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedRLEnvCfg, mdp
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.locomotion.velocity import mdp as locomotion_mdp

from algorithms.urdf_learn_wasd_walk import forward_walk_contract as contract
from algorithms.urdf_learn_wasd_walk import model_spec
from algorithms.urdf_learn_wasd_walk.policy_stand_env import (
    ACTION_ASSET,
    ActionsCfg,
    EventsCfg,
    LandauStandSceneCfg,
    TerminationsCfg,
    make_landau_articulation_cfg,
)


SUPPORT_BODIES = ["foot_l", "toes_01_l", "foot_r", "toes_01_r"]


def semantic_velocity_command(env) -> torch.Tensor:
    """Expose command values in semantic forward, strafe, yaw order."""

    command = env.command_manager.get_command("base_velocity")
    return command[:, [1, 0, 2]]


def gait_phase(env) -> torch.Tensor:
    """Deployable periodic phase reset with each episode."""

    phase = 2.0 * torch.pi * env.episode_length_buf * env.step_dt / contract.GAIT_PERIOD_S
    return torch.stack((torch.sin(phase), torch.cos(phase)), dim=-1)


def forward_velocity_tracking_l2(env) -> torch.Tensor:
    """Unbounded quadratic tracking makes opposite velocity errors costly."""

    robot = env.scene["robot"]
    command = env.command_manager.get_command("base_velocity")
    return torch.square(command[:, 1] - robot.data.root_lin_vel_b[:, 1])


def signed_forward_progress(env) -> torch.Tensor:
    robot = env.scene["robot"]
    command = env.command_manager.get_command("base_velocity")[:, 1]
    moving = torch.abs(command) > 0.1
    return torch.clamp(torch.sign(command) * robot.data.root_lin_vel_b[:, 1], -1.0, 1.0) * moving


def yaw_velocity_l2(env) -> torch.Tensor:
    robot = env.scene["robot"]
    command = env.command_manager.get_command("base_velocity")
    return torch.square(command[:, 2] - robot.data.root_ang_vel_b[:, 2])


def alternating_single_support(
    env, left_cfg: SceneEntityCfg, right_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward the scheduled stance contact and scheduled swing clearance."""

    sensor: ContactSensor = env.scene.sensors[left_cfg.name]
    force_history = sensor.data.net_forces_w_history
    left_force = torch.linalg.vector_norm(force_history[:, :, left_cfg.body_ids], dim=-1)
    right_force = torch.linalg.vector_norm(force_history[:, :, right_cfg.body_ids], dim=-1)
    left_contact = torch.any(torch.any(left_force > 1.0, dim=1), dim=1)
    right_contact = torch.any(torch.any(right_force > 1.0, dim=1), dim=1)
    left_stance = gait_phase(env)[:, 0] >= 0.0
    stance_contact = torch.where(left_stance, left_contact, right_contact)
    swing_clear = torch.where(left_stance, ~right_contact, ~left_contact)
    moving = torch.abs(env.command_manager.get_command("base_velocity")[:, 1]) > 0.1
    return 0.5 * (stance_contact.float() + swing_clear.float()) * moving


@configclass
class CommandsCfg:
    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(8.0, 12.0),
        rel_standing_envs=0.25,
        rel_heading_envs=0.0,
        heading_command=False,
        debug_vis=False,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 0.0),
            lin_vel_y=contract.TRAIN_FORWARD_SPEED_RANGE_MPS,
            ang_vel_z=(0.0, 0.0),
            heading=None,
        ),
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        base_linear_velocity = ObsTerm(func=mdp.base_lin_vel)
        base_angular_velocity = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        action_joint_position_error = ObsTerm(
            func=mdp.joint_pos_rel, params={"asset_cfg": ACTION_ASSET}
        )
        action_joint_velocity = ObsTerm(
            func=mdp.joint_vel_rel, params={"asset_cfg": ACTION_ASSET}
        )
        previous_action = ObsTerm(func=mdp.last_action)
        semantic_velocity_command = ObsTerm(func=semantic_velocity_command)
        gait_phase = ObsTerm(func=gait_phase)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    termination = RewTerm(func=mdp.is_terminated, weight=-100.0)
    forward_velocity_tracking_l2 = RewTerm(func=forward_velocity_tracking_l2, weight=-2.0)
    signed_forward_progress = RewTerm(func=signed_forward_progress, weight=1.0)
    yaw_velocity_l2 = RewTerm(func=yaw_velocity_l2, weight=-0.5)
    upright_posture = RewTerm(func=mdp.flat_orientation_l2, weight=-2.0)
    base_height = RewTerm(
        func=mdp.base_height_l2,
        weight=-1.0,
        params={"target_height": model_spec.build_robot_spec()["nominal_pose"]["base_position_m"][2]},
    )
    alternating_single_support = RewTerm(
        func=alternating_single_support,
        weight=1.0,
        params={
            "left_cfg": SceneEntityCfg(
                "contact_forces", body_names=["foot_l", "toes_01_l"]
            ),
            "right_cfg": SceneEntityCfg(
                "contact_forces", body_names=["foot_r", "toes_01_r"]
            ),
        },
    )
    feet_air_time = RewTerm(
        func=locomotion_mdp.feet_air_time_positive_biped,
        weight=0.5,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["foot_l", "foot_r"]),
            "threshold": 0.25,
        },
    )
    foot_slip = RewTerm(
        func=locomotion_mdp.feet_slide,
        weight=-0.15,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=SUPPORT_BODIES),
            "asset_cfg": SceneEntityCfg("robot", body_names=SUPPORT_BODIES),
        },
    )
    natural_posture = RewTerm(
        func=mdp.joint_deviation_l1, weight=-0.1, params={"asset_cfg": ACTION_ASSET}
    )
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    action_magnitude = RewTerm(func=mdp.action_l2, weight=-0.01)


@configclass
class LandauForwardEnvCfg(ManagerBasedRLEnvCfg):
    scene: LandauStandSceneCfg = LandauStandSceneCfg(num_envs=512, env_spacing=2.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()
    curriculum = None

    def __post_init__(self):
        self.decimation = round(contract.CONTROL_DT_S / contract.PHYSICS_DT_S)
        self.episode_length_s = 16.0
        self.sim.dt = contract.PHYSICS_DT_S
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 2**18
        self.ui_window_class_type = None
        self.rerender_on_reset = False


def build_env_cfg(
    *,
    num_envs: int,
    seed: int,
    training: bool,
    evaluation_command: str | None = None,
    force_usd_conversion: bool,
) -> LandauForwardEnvCfg:
    if num_envs <= 0:
        raise ValueError("num_envs must be positive")
    if evaluation_command not in {None, "stand", "forward"}:
        raise ValueError(f"unsupported evaluation command: {evaluation_command}")
    cfg = LandauForwardEnvCfg()
    cfg.scene.num_envs = num_envs
    cfg.scene.robot = make_landau_articulation_cfg(force_usd_conversion=force_usd_conversion)
    cfg.seed = seed
    if not training:
        cfg.events.reset_base = None
        cfg.events.reset_action_joints = None
        cfg.episode_length_s = contract.MAX_GATE_DURATION_S + contract.CONTROL_DT_S * 2
        cfg.commands.base_velocity.resampling_time_range = (1000.0, 1000.0)
        cfg.commands.base_velocity.rel_standing_envs = 1.0 if evaluation_command == "stand" else 0.0
        speed = 0.0 if evaluation_command == "stand" else contract.TARGET_FORWARD_SPEED_MPS
        cfg.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        cfg.commands.base_velocity.ranges.lin_vel_y = (speed, speed)
        cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
    return cfg
