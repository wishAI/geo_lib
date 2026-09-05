"""Isaac Lab 0.36.1 manager-based configuration for Landau policy standing.

Import this module only after :class:`isaaclab.app.AppLauncher` has started Kit.
"""

from __future__ import annotations

import os
from dataclasses import MISSING

import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.envs import mdp
from isaaclab_tasks.manager_based.locomotion.velocity import mdp as locomotion_mdp

from algorithms.urdf_learn_wasd_walk import model_spec
from algorithms.urdf_learn_wasd_walk import policy_stand_contract as contract


ACTION_ASSET = SceneEntityCfg(
    "robot", joint_names=list(model_spec.ACTION_JOINTS), preserve_order=True
)
SUPPORT_BODIES = ["foot_l", "toes_01_l", "foot_r", "toes_01_r"]


def planar_velocity_l2(env) -> torch.Tensor:
    robot = env.scene["robot"]
    return torch.sum(torch.square(robot.data.root_lin_vel_b[:, :2]), dim=1)


def angular_velocity_l2(env) -> torch.Tensor:
    robot = env.scene["robot"]
    return torch.sum(torch.square(robot.data.root_ang_vel_b), dim=1)


def bilateral_foot_contact(env, left_cfg: SceneEntityCfg, right_cfg: SceneEntityCfg) -> torch.Tensor:
    sensor: ContactSensor = env.scene.sensors[left_cfg.name]
    force_history = sensor.data.net_forces_w_history
    left = torch.max(torch.linalg.vector_norm(force_history[:, :, left_cfg.body_ids], dim=-1), dim=1)[0]
    right = torch.max(torch.linalg.vector_norm(force_history[:, :, right_cfg.body_ids], dim=-1), dim=1)[0]
    return (torch.any(left > 1.0, dim=1) & torch.any(right > 1.0, dim=1)).float()


def make_landau_articulation_cfg(*, force_usd_conversion: bool) -> ArticulationCfg:
    spec = model_spec.build_robot_spec()
    actuator_groups = model_spec.passive_actuator_groups([joint["name"] for joint in spec["joints"]])
    return ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UrdfFileCfg(
            asset_path=os.fspath(model_spec.URDF_PATH),
            usd_dir=os.fspath(contract.OUTPUT_ROOT / "usd_cache"),
            usd_file_name="landau_v10_parallel_mesh.usd",
            force_usd_conversion=force_usd_conversion,
            fix_base=False,
            merge_fixed_joints=False,
            make_instanceable=False,
            self_collision=False,
            activate_contact_sensors=True,
            collider_type="convex_hull",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=False,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=20.0,
                max_angular_velocity=50.0,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=4,
                sleep_threshold=0.0,
                stabilization_threshold=0.0001,
            ),
            joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                drive_type="force",
                target_type="position",
                gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                    stiffness=0.0, damping=0.0
                ),
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=tuple(spec["nominal_pose"]["base_position_m"]),
            rot=tuple(spec["nominal_pose"]["base_orientation_wxyz"]),
            joint_pos=spec["nominal_pose"]["joint_positions_rad"],
            joint_vel={".*": 0.0},
        ),
        soft_joint_pos_limit_factor=0.95,
        actuators={
            name: ImplicitActuatorCfg(
                joint_names_expr=list(group["joints"]),
                effort_limit_sim=50.0,
                velocity_limit_sim=4.0,
                stiffness=group["stiffness"],
                damping=group["damping"],
            )
            for name, group in actuator_groups.items()
        },
    )


@configclass
class LandauStandSceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=0.9,
            restitution=0.0,
        ),
        debug_vis=False,
    )
    robot: ArticulationCfg = MISSING
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        update_period=contract.PHYSICS_DT_S,
        history_length=3,
        track_air_time=True,
    )
    light = AssetBaseCfg(
        prim_path="/World/Sun",
        spawn=sim_utils.DistantLightCfg(intensity=3000.0, color=(0.92, 0.92, 0.92)),
    )


@configclass
class ActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=list(model_spec.ACTION_JOINTS),
        scale=contract.ACTION_SCALE_RAD,
        use_default_offset=True,
        preserve_order=True,
        clip={".*": (-contract.ACTION_CLIP, contract.ACTION_CLIP)},
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        base_linear_velocity = ObsTerm(func=mdp.base_lin_vel)
        base_angular_velocity = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        action_joint_position_error = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": ACTION_ASSET})
        action_joint_velocity = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": ACTION_ASSET})
        previous_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventsCfg:
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
                "roll": (-0.035, 0.035), "pitch": (-0.035, 0.035), "yaw": (0.0, 0.0),
            },
            "velocity_range": {
                "x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.02, 0.02),
                "roll": (-0.1, 0.1), "pitch": (-0.1, 0.1), "yaw": (-0.1, 0.1),
            },
        },
    )
    reset_action_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.02, 0.02),
            "velocity_range": (-0.05, 0.05),
            "asset_cfg": ACTION_ASSET,
        },
    )


@configclass
class RewardsCfg:
    alive = RewTerm(func=mdp.is_alive, weight=1.0)
    termination = RewTerm(func=mdp.is_terminated, weight=-100.0)
    planar_velocity_l2 = RewTerm(func=planar_velocity_l2, weight=-2.0)
    angular_velocity_l2 = RewTerm(func=angular_velocity_l2, weight=-0.1)
    upright_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-4.0)
    base_height_l2 = RewTerm(
        func=mdp.base_height_l2,
        weight=-2.0,
        params={"target_height": model_spec.build_robot_spec()["nominal_pose"]["base_position_m"][2]},
    )
    bilateral_foot_contact = RewTerm(
        func=bilateral_foot_contact,
        weight=0.5,
        params={
            "left_cfg": SceneEntityCfg("contact_forces", body_names=["foot_l", "toes_01_l"]),
            "right_cfg": SceneEntityCfg("contact_forces", body_names=["foot_r", "toes_01_r"]),
        },
    )
    foot_slip = RewTerm(
        func=locomotion_mdp.feet_slide,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=SUPPORT_BODIES),
            "asset_cfg": SceneEntityCfg("robot", body_names=SUPPORT_BODIES),
        },
    )
    natural_action_joint_posture_l1 = RewTerm(
        func=mdp.joint_deviation_l1, weight=-0.2, params={"asset_cfg": ACTION_ASSET}
    )
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.02)
    action_l2 = RewTerm(func=mdp.action_l2, weight=-0.05)


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    bad_orientation = DoneTerm(
        func=mdp.bad_orientation, params={"limit_angle": contract.MAX_REFERENCE_TILT_RAD}
    )
    root_height = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={
            "minimum_height": (
                model_spec.build_robot_spec()["nominal_pose"]["base_position_m"][2]
                - contract.MAX_ROOT_HEIGHT_DROP_M
            )
        },
    )


@configclass
class LandauStandEnvCfg(ManagerBasedRLEnvCfg):
    scene: LandauStandSceneCfg = LandauStandSceneCfg(num_envs=512, env_spacing=1.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands = None
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()
    curriculum = None

    def __post_init__(self):
        self.decimation = round(contract.CONTROL_DT_S / contract.PHYSICS_DT_S)
        self.episode_length_s = 10.0
        self.sim.dt = contract.PHYSICS_DT_S
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 2**18
        self.ui_window_class_type = None
        self.rerender_on_reset = False


def build_env_cfg(
    *, num_envs: int, seed: int, training: bool, force_usd_conversion: bool
) -> LandauStandEnvCfg:
    if num_envs <= 0:
        raise ValueError("num_envs must be positive")
    cfg = LandauStandEnvCfg()
    cfg.scene.num_envs = num_envs
    cfg.scene.robot = make_landau_articulation_cfg(force_usd_conversion=force_usd_conversion)
    cfg.seed = seed
    if not training:
        cfg.events.reset_base = None
        cfg.events.reset_action_joints = None
        cfg.episode_length_s = contract.MIN_GATE_DURATION_S + contract.CONTROL_DT_S * 2
    return cfg
