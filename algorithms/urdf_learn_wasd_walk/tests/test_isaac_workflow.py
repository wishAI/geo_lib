from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch


MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT.parent.parent) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT.parent.parent))


def _install_isaaclab_tasks_stubs() -> None:
    if "torch" not in sys.modules:
        try:
            import torch  # noqa: F401
        except ImportError:
            sys.modules["torch"] = types.ModuleType("torch")
    if "isaaclab_tasks.utils" not in sys.modules:
        utils_module = types.ModuleType("isaaclab_tasks.utils")
        utils_module.get_checkpoint_path = lambda *args, **kwargs: ""
        utils_module.parse_env_cfg = lambda *args, **kwargs: None
        sys.modules["isaaclab_tasks.utils"] = utils_module
    if "isaaclab_tasks.utils.parse_cfg" not in sys.modules:
        parse_cfg_module = types.ModuleType("isaaclab_tasks.utils.parse_cfg")
        parse_cfg_module.load_cfg_from_registry = lambda *args, **kwargs: None
        sys.modules["isaaclab_tasks.utils.parse_cfg"] = parse_cfg_module


_install_isaaclab_tasks_stubs()

import algorithms.urdf_learn_wasd_walk.isaac_workflow as isaac_workflow
import algorithms.urdf_learn_wasd_walk.checkpoint_preflight as checkpoint_preflight
from algorithms.urdf_learn_wasd_walk.isaac_workflow import (
    _is_default_checkpoint_selector,
    apply_checkpoint_playback_compat,
    build_init_pose_action,
    clamp_base_velocity_command,
    load_runner_checkpoint,
    resolve_checkpoint_selection,
)


class ClampBaseVelocityCommandTests(unittest.TestCase):
    def test_preserves_idle_for_positive_only_stage(self) -> None:
        env_cfg = types.SimpleNamespace(
            commands=types.SimpleNamespace(
                base_velocity=types.SimpleNamespace(
                    ranges=types.SimpleNamespace(
                        lin_vel_x=(0.0, 0.0),
                        lin_vel_y=(0.45, 0.85),
                        ang_vel_z=(0.0, 0.0),
                    )
                )
            )
        )

        self.assertEqual(clamp_base_velocity_command(env_cfg, (0.0, 0.0, 0.0)), (0.0, 0.0, 0.0))
        self.assertEqual(clamp_base_velocity_command(env_cfg, (0.0, -0.5, 0.0)), (0.0, 0.0, 0.0))

    def test_clamps_non_idle_requests_into_valid_range(self) -> None:
        env_cfg = types.SimpleNamespace(
            commands=types.SimpleNamespace(
                base_velocity=types.SimpleNamespace(
                    ranges=types.SimpleNamespace(
                        lin_vel_x=(-0.3, 0.3),
                        lin_vel_y=(0.45, 0.85),
                        ang_vel_z=(-0.4, 0.4),
                    )
                )
            )
        )

        self.assertEqual(clamp_base_velocity_command(env_cfg, (0.4, 0.2, 0.9)), (0.3, 0.45, 0.4))


class PreflightDefaultCheckpointSelectionTests(unittest.TestCase):
    def test_falls_back_to_same_stage_active_lineage_current_checkpoint(self) -> None:
        original_resolve_recommended = checkpoint_preflight.resolve_recommended_checkpoint
        original_load_active_lineage = checkpoint_preflight.load_active_lineage
        try:
            checkpoint_preflight.resolve_recommended_checkpoint = lambda experiment_name: None
            checkpoint_preflight.load_active_lineage = lambda: {
                "lineage_name": "restart_case",
                "current_stage": "game",
                "current_run": "2026-04-18_10-41-48_game_v11_zero_init_200",
                "current_checkpoint": "/tmp/model_150.pt",
                "experiments": {
                    "stand": "geo_landau_restart_case_stand",
                    "game": "geo_landau_restart_case_game",
                },
            }
            args = types.SimpleNamespace(
                robot="landau",
                stage="game",
                latest=False,
                load_run=None,
                checkpoint=None,
                experiment_name=None,
            )
            checkpoint_preflight.preflight_default_checkpoint_selection(args, mode="play")
        finally:
            checkpoint_preflight.resolve_recommended_checkpoint = original_resolve_recommended
            checkpoint_preflight.load_active_lineage = original_load_active_lineage

        self.assertEqual(args.stage, "game")
        self.assertEqual(args.load_run, "2026-04-18_10-41-48_game_v11_zero_init_200")
        self.assertEqual(args.checkpoint, "model_150.pt")
        self.assertEqual(args.experiment_name, "geo_landau_restart_case_game")

    def test_raises_when_only_other_stage_checkpoint_exists(self) -> None:
        original_resolve_recommended = checkpoint_preflight.resolve_recommended_checkpoint
        original_load_active_lineage = checkpoint_preflight.load_active_lineage
        try:
            checkpoint_preflight.resolve_recommended_checkpoint = lambda experiment_name: None
            checkpoint_preflight.load_active_lineage = lambda: {
                "lineage_name": "restart_case",
                "current_stage": "stand",
                "current_run": "2026-04-18_19-24-17_stand_stage0_rebuild_v3_armdefaults_150",
                "current_checkpoint": "/tmp/model_149.pt",
                "experiments": {
                    "stand": "geo_landau_restart_case_stand",
                    "game": "geo_landau_restart_case_game",
                },
            }
            args = types.SimpleNamespace(
                robot="landau",
                stage="game",
                latest=False,
                load_run=None,
                checkpoint=None,
                experiment_name=None,
            )
            with self.assertRaises(SystemExit) as ctx:
                checkpoint_preflight.preflight_default_checkpoint_selection(args, mode="play")
        finally:
            checkpoint_preflight.resolve_recommended_checkpoint = original_resolve_recommended
            checkpoint_preflight.load_active_lineage = original_load_active_lineage

        message = str(ctx.exception)
        self.assertIn("Requested stage 'game' does not match the active lineage stage 'stand'", message)

    def test_raises_clear_error_when_no_fallback_checkpoint_exists(self) -> None:
        original_resolve_recommended = checkpoint_preflight.resolve_recommended_checkpoint
        original_load_active_lineage = checkpoint_preflight.load_active_lineage
        try:
            checkpoint_preflight.resolve_recommended_checkpoint = lambda experiment_name: None
            checkpoint_preflight.load_active_lineage = lambda: {
                "lineage_name": "restart_case",
                "current_stage": "stand",
                "current_run": None,
                "current_checkpoint": None,
            }
            args = types.SimpleNamespace(
                robot="landau",
                stage="game",
                latest=False,
                load_run=None,
                checkpoint=None,
                experiment_name=None,
            )
            with self.assertRaises(SystemExit) as ctx:
                checkpoint_preflight.preflight_default_checkpoint_selection(args, mode="play")
        finally:
            checkpoint_preflight.resolve_recommended_checkpoint = original_resolve_recommended
            checkpoint_preflight.load_active_lineage = original_load_active_lineage

        message = str(ctx.exception)
        self.assertIn("No promoted checkpoint is registered", message)
        self.assertIn("restart_case", message)
        self.assertIn("Current stage: stand", message)
        self.assertIn("--latest", message)

    def test_skips_error_for_explicit_checkpoint_request(self) -> None:
        args = types.SimpleNamespace(
            robot="landau",
            stage="game",
            latest=False,
            load_run="some_run",
            checkpoint="model_42.pt",
            experiment_name=None,
        )

        checkpoint_preflight.preflight_default_checkpoint_selection(args, mode="teleop")


class ApplyCheckpointPlaybackCompatTests(unittest.TestCase):
    def test_restores_action_reset_and_actuator_compat_from_checkpoint_env(self) -> None:
        with TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "run"
            params_dir = run_dir / "params"
            params_dir.mkdir(parents=True)
            checkpoint_path = run_dir / "model_1.pt"
            checkpoint_path.write_text("", encoding="utf-8")
            (params_dir / "env.yaml").write_text(
                "\n".join(
                    (
                        "actions:",
                        "  joint_pos:",
                        "    scale:",
                        "      thigh_.*: 0.5",
                        "      foot_.*: 0.35",
                        "    joint_names: [left_hip_pitch_joint, left_knee_joint]",
                        "    preserve_order: true",
                        "    use_default_offset: false",
                        "    offset:",
                        "      left_hip_pitch_joint: 0.1",
                        "      left_knee_joint: -0.05",
                        "commands:",
                        "  base_velocity:",
                        "    ranges:",
                        "      lin_vel_x: [0.0, 0.0]",
                        "      lin_vel_y: [0.45, 1.0]",
                        "      ang_vel_z: [0.0, 0.0]",
                        "scene:",
                        "  robot:",
                        "    init_state:",
                        "      pos: [0.0, 0.0, 0.02]",
                        "      rot: [1.0, 0.0, 0.0, 0.0]",
                        "      lin_vel: [0.0, 0.0, 0.0]",
                        "      ang_vel: [0.0, 0.0, 0.0]",
                        "      joint_pos:",
                        "        left_hip_pitch_joint: 0.2",
                        "        left_knee_joint: 0.54",
                        "        left_ankle_pitch_joint: -0.07",
                        "        left_toe_joint: 0.05",
                        "      joint_vel:",
                        "        .*: 0.0",
                        "    actuators:",
                        "      legs:",
                        "        stiffness: 140.0",
                        "        damping: 10.0",
                        "        effort_limit_sim: 120.0",
                        "        velocity_limit_sim: 20.0",
                        "      feet:",
                        "        stiffness: 60.0",
                        "        damping: 6.0",
                        "        effort_limit_sim: 80.0",
                        "        velocity_limit_sim: 20.0",
                    )
                ),
                encoding="utf-8",
            )
            env_cfg = types.SimpleNamespace(
                actions=types.SimpleNamespace(
                    joint_pos=types.SimpleNamespace(
                        scale={"thigh_.*": 0.4},
                        joint_names=["left_ankle_pitch_joint"],
                        preserve_order=False,
                        use_default_offset=True,
                        offset=0.0,
                    )
                ),
                commands=types.SimpleNamespace(
                    base_velocity=types.SimpleNamespace(
                        ranges=types.SimpleNamespace(
                            lin_vel_x=(-0.2, 0.2),
                            lin_vel_y=(-0.2, 1.0),
                            ang_vel_z=(-0.5, 0.5),
                        )
                    )
                ),
                scene=types.SimpleNamespace(
                    robot=types.SimpleNamespace(
                        init_state=types.SimpleNamespace(
                            pos=(0.0, 0.0, 0.01),
                            rot=(1.0, 0.0, 0.0, 0.0),
                            lin_vel=(0.0, 0.0, 0.0),
                            ang_vel=(0.0, 0.0, 0.0),
                            joint_vel={".*": 0.1},
                            joint_pos={
                                "left_hip_pitch_joint": 0.08,
                                "left_knee_joint": 0.24,
                                "left_ankle_pitch_joint": -0.12,
                                "left_toe_joint": -0.06,
                            },
                        ),
                        actuators={
                            "legs": types.SimpleNamespace(
                                stiffness=220.0,
                                damping=16.0,
                                effort_limit_sim=180.0,
                                velocity_limit_sim=20.0,
                            ),
                            "ankles": types.SimpleNamespace(
                                stiffness=320.0,
                                damping=28.0,
                                effort_limit_sim=220.0,
                                velocity_limit_sim=20.0,
                            ),
                            "toes": types.SimpleNamespace(
                                stiffness=260.0,
                                damping=24.0,
                                effort_limit_sim=180.0,
                                velocity_limit_sim=20.0,
                            ),
                        },
                    )
                ),
            )

            applied = apply_checkpoint_playback_compat(env_cfg, str(checkpoint_path))

            self.assertTrue(applied)
            self.assertEqual(env_cfg.actions.joint_pos.scale["thigh_.*"], 0.5)
            self.assertEqual(env_cfg.actions.joint_pos.scale["foot_.*"], 0.35)
            self.assertEqual(env_cfg.actions.joint_pos.joint_names, ["left_hip_pitch_joint", "left_knee_joint"])
            self.assertTrue(env_cfg.actions.joint_pos.preserve_order)
            self.assertFalse(env_cfg.actions.joint_pos.use_default_offset)
            self.assertEqual(env_cfg.actions.joint_pos.offset["left_hip_pitch_joint"], 0.1)
            self.assertEqual(env_cfg.commands.base_velocity.ranges.lin_vel_x, (0.0, 0.0))
            self.assertEqual(env_cfg.commands.base_velocity.ranges.lin_vel_y, (0.45, 1.0))
            self.assertEqual(env_cfg.commands.base_velocity.ranges.ang_vel_z, (0.0, 0.0))
            self.assertEqual(env_cfg.scene.robot.init_state.pos, (0.0, 0.0, 0.02))
            self.assertEqual(env_cfg.scene.robot.init_state.joint_pos["left_hip_pitch_joint"], 0.2)
            self.assertEqual(env_cfg.scene.robot.init_state.joint_pos["left_knee_joint"], 0.54)
            self.assertEqual(env_cfg.scene.robot.init_state.joint_pos["left_ankle_pitch_joint"], -0.07)
            self.assertEqual(env_cfg.scene.robot.init_state.joint_pos["left_toe_joint"], 0.05)
            self.assertEqual(env_cfg.scene.robot.init_state.joint_vel[".*"], 0.0)
            self.assertEqual(env_cfg.scene.robot.actuators["legs"].stiffness, 140.0)
            self.assertEqual(env_cfg.scene.robot.actuators["legs"].damping, 10.0)
            self.assertEqual(env_cfg.scene.robot.actuators["ankles"].stiffness, 60.0)
            self.assertEqual(env_cfg.scene.robot.actuators["ankles"].damping, 6.0)
            self.assertEqual(env_cfg.scene.robot.actuators["ankles"].effort_limit_sim, 80.0)
            self.assertEqual(env_cfg.scene.robot.actuators["toes"].stiffness, 60.0)
            self.assertEqual(env_cfg.scene.robot.actuators["toes"].damping, 6.0)
            self.assertEqual(env_cfg.scene.robot.actuators["toes"].effort_limit_sim, 80.0)

    def test_control_only_mode_skips_init_state_and_actuators(self) -> None:
        with TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "run"
            params_dir = run_dir / "params"
            params_dir.mkdir(parents=True)
            checkpoint_path = run_dir / "model_1.pt"
            checkpoint_path.write_text("", encoding="utf-8")
            (params_dir / "env.yaml").write_text(
                "\n".join(
                    (
                        "actions:",
                        "  joint_pos:",
                        "    scale:",
                        "      thigh_.*: 0.5",
                        "    joint_names: [left_hip_pitch_joint, left_knee_joint]",
                        "commands:",
                        "  base_velocity:",
                        "    ranges:",
                        "      lin_vel_x: [0.0, 0.0]",
                        "      lin_vel_y: [0.45, 1.0]",
                        "      ang_vel_z: [0.0, 0.0]",
                        "scene:",
                        "  robot:",
                        "    init_state:",
                        "      joint_pos:",
                        "        left_hip_pitch_joint: 0.2",
                        "    actuators:",
                        "      feet:",
                        "        stiffness: 60.0",
                    )
                ),
                encoding="utf-8",
            )
            env_cfg = types.SimpleNamespace(
                actions=types.SimpleNamespace(
                    joint_pos=types.SimpleNamespace(
                        scale={"thigh_.*": 0.4},
                        joint_names=["left_ankle_pitch_joint"],
                    )
                ),
                commands=types.SimpleNamespace(
                    base_velocity=types.SimpleNamespace(
                        ranges=types.SimpleNamespace(
                            lin_vel_x=(-0.2, 0.2),
                            lin_vel_y=(-0.2, 1.0),
                            ang_vel_z=(-0.5, 0.5),
                        )
                    )
                ),
                scene=types.SimpleNamespace(
                    robot=types.SimpleNamespace(
                        init_state=types.SimpleNamespace(joint_pos={"left_hip_pitch_joint": 0.08}, joint_vel={".*": 0.1}),
                        actuators={"ankles": types.SimpleNamespace(stiffness=320.0)},
                    )
                ),
            )

            applied = apply_checkpoint_playback_compat(env_cfg, str(checkpoint_path), mode="control_only")

            self.assertTrue(applied)
            self.assertEqual(env_cfg.actions.joint_pos.scale["thigh_.*"], 0.5)
            self.assertEqual(env_cfg.actions.joint_pos.joint_names, ["left_hip_pitch_joint", "left_knee_joint"])
            self.assertEqual(env_cfg.commands.base_velocity.ranges.lin_vel_y, (0.45, 1.0))
            self.assertEqual(env_cfg.scene.robot.init_state.joint_pos["left_hip_pitch_joint"], 0.08)
            self.assertEqual(env_cfg.scene.robot.actuators["ankles"].stiffness, 320.0)

    def test_off_mode_is_noop(self) -> None:
        with TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "run"
            params_dir = run_dir / "params"
            params_dir.mkdir(parents=True)
            checkpoint_path = run_dir / "model_1.pt"
            checkpoint_path.write_text("", encoding="utf-8")
            (params_dir / "env.yaml").write_text(
                "\n".join(
                    (
                        "actions:",
                        "  joint_pos:",
                        "    scale:",
                        "      thigh_.*: 0.5",
                    )
                ),
                encoding="utf-8",
            )
            env_cfg = types.SimpleNamespace(
                actions=types.SimpleNamespace(
                    joint_pos=types.SimpleNamespace(scale={"thigh_.*": 0.4}, joint_names=["left_ankle_pitch_joint"])
                ),
                commands=types.SimpleNamespace(
                    base_velocity=types.SimpleNamespace(
                        ranges=types.SimpleNamespace(
                            lin_vel_x=(-0.2, 0.2),
                            lin_vel_y=(-0.2, 1.0),
                            ang_vel_z=(-0.5, 0.5),
                        )
                    )
                ),
                scene=types.SimpleNamespace(
                    robot=types.SimpleNamespace(
                        init_state=types.SimpleNamespace(joint_pos={"left_hip_pitch_joint": 0.08}, joint_vel={".*": 0.1}),
                        actuators={"ankles": types.SimpleNamespace(stiffness=320.0)},
                    )
                ),
            )

            applied = apply_checkpoint_playback_compat(env_cfg, str(checkpoint_path), mode="off")

            self.assertFalse(applied)
            self.assertEqual(env_cfg.actions.joint_pos.scale["thigh_.*"], 0.4)

    def test_noop_when_checkpoint_has_no_saved_env(self) -> None:
        with TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "model_1.pt"
            checkpoint_path.write_text("", encoding="utf-8")
            env_cfg = types.SimpleNamespace(
                actions=types.SimpleNamespace(
                    joint_pos=types.SimpleNamespace(scale={"thigh_.*": 0.4}, joint_names=["left_ankle_pitch_joint"])
                ),
                commands=types.SimpleNamespace(
                    base_velocity=types.SimpleNamespace(
                        ranges=types.SimpleNamespace(
                            lin_vel_x=(-0.2, 0.2),
                            lin_vel_y=(-0.2, 1.0),
                            ang_vel_z=(-0.5, 0.5),
                        )
                    )
                ),
                scene=types.SimpleNamespace(
                    robot=types.SimpleNamespace(
                        init_state=types.SimpleNamespace(joint_pos={"left_hip_pitch_joint": 0.08}, joint_vel={".*": 0.1}),
                        actuators={"ankles": types.SimpleNamespace(stiffness=320.0)},
                    )
                ),
            )

            applied = apply_checkpoint_playback_compat(env_cfg, str(checkpoint_path))

            self.assertFalse(applied)
            self.assertEqual(env_cfg.actions.joint_pos.scale["thigh_.*"], 0.4)
            self.assertEqual(env_cfg.actions.joint_pos.joint_names, ["left_ankle_pitch_joint"])
            self.assertEqual(env_cfg.commands.base_velocity.ranges.lin_vel_y, (-0.2, 1.0))
            self.assertEqual(env_cfg.scene.robot.init_state.joint_pos["left_hip_pitch_joint"], 0.08)
            self.assertEqual(env_cfg.scene.robot.init_state.joint_vel[".*"], 0.1)
            self.assertEqual(env_cfg.scene.robot.actuators["ankles"].stiffness, 320.0)


class BuildInitPoseActionTests(unittest.TestCase):
    def test_targets_configured_init_pose_relative_to_asset_defaults(self) -> None:
        env = types.SimpleNamespace()
        env.num_envs = 1
        env.device = "cpu"
        joint_pos_term = types.SimpleNamespace(
            _joint_names=["left_hip_pitch_joint", "left_knee_joint"],
            _joint_ids=[0, 1],
        )
        env.action_manager = types.SimpleNamespace(
            total_action_dim=2,
            active_terms=["joint_pos"],
            action_term_dim=[2],
            get_term=lambda name: joint_pos_term,
        )
        robot = types.SimpleNamespace(
            joint_names=["left_hip_pitch_joint", "left_knee_joint"],
            data=types.SimpleNamespace(default_joint_pos=torch.tensor([[0.0, 0.1]], dtype=torch.float32)),
        )
        env.scene = {"robot": robot}

        env_cfg = types.SimpleNamespace(
            actions=types.SimpleNamespace(
                joint_pos=types.SimpleNamespace(
                    joint_names=["left_hip_pitch_joint", "left_knee_joint"],
                    scale={"left_hip_pitch_joint": 0.5, "left_knee_joint": 0.2},
                )
            ),
            scene=types.SimpleNamespace(
                robot=types.SimpleNamespace(
                    init_state=types.SimpleNamespace(
                        joint_pos={"left_hip_pitch_joint": 0.25, "left_knee_joint": 0.3}
                    )
                )
            ),
        )

        action = build_init_pose_action(env, env_cfg)

        self.assertEqual(tuple(action.shape), (1, 2))
        self.assertAlmostEqual(float(action[0, 0]), 0.5, places=6)
        self.assertAlmostEqual(float(action[0, 1]), 1.0, places=6)

    def test_targets_configured_init_pose_with_explicit_offsets(self) -> None:
        env = types.SimpleNamespace()
        env.num_envs = 1
        env.device = "cpu"
        joint_pos_term = types.SimpleNamespace(
            _joint_names=["left_hip_pitch_joint", "left_knee_joint"],
            _joint_ids=[0, 1],
        )
        env.action_manager = types.SimpleNamespace(
            total_action_dim=2,
            active_terms=["joint_pos"],
            action_term_dim=[2],
            get_term=lambda name: joint_pos_term,
        )
        robot = types.SimpleNamespace(
            joint_names=["left_hip_pitch_joint", "left_knee_joint"],
            data=types.SimpleNamespace(default_joint_pos=torch.tensor([[0.0, 0.1]], dtype=torch.float32)),
        )
        env.scene = {"robot": robot}

        env_cfg = types.SimpleNamespace(
            actions=types.SimpleNamespace(
                joint_pos=types.SimpleNamespace(
                    joint_names=["left_hip_pitch_joint", "left_knee_joint"],
                    scale={"left_hip_pitch_joint": 0.5, "left_knee_joint": 0.2},
                    offset={"left_hip_pitch_joint": 0.1, "left_knee_joint": -0.05},
                    use_default_offset=False,
                )
            ),
            scene=types.SimpleNamespace(
                robot=types.SimpleNamespace(
                    init_state=types.SimpleNamespace(
                        joint_pos={"left_hip_pitch_joint": 0.35, "left_knee_joint": 0.15}
                    )
                )
            ),
        )

        action = build_init_pose_action(env, env_cfg)

        self.assertEqual(tuple(action.shape), (1, 2))
        self.assertAlmostEqual(float(action[0, 0]), 0.5, places=6)
        self.assertAlmostEqual(float(action[0, 1]), 1.0, places=6)


class ResolveCheckpointSelectionTests(unittest.TestCase):
    def test_prefers_recommended_checkpoint_when_no_explicit_request(self) -> None:
        with TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "model_7.pt"
            checkpoint_path.write_text("", encoding="utf-8")
            original = isaac_workflow.resolve_recommended_checkpoint
            try:
                isaac_workflow.resolve_recommended_checkpoint = lambda experiment_name: {
                    "checkpoint_path": str(checkpoint_path),
                    "reason": "promoted",
                }
                selection = resolve_checkpoint_selection(
                    "unused",
                    types.SimpleNamespace(
                        experiment_name="geo_landau_game",
                        load_run=None,
                        load_checkpoint=None,
                    ),
                )
            finally:
                isaac_workflow.resolve_recommended_checkpoint = original

            self.assertEqual(selection["path"], str(checkpoint_path))
            self.assertEqual(selection["source"], "recommended")
            self.assertEqual(selection["entry"]["reason"], "promoted")

    def test_uses_latest_when_requested(self) -> None:
        with TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "model_9.pt"
            checkpoint_path.write_text("", encoding="utf-8")
            original_recommended = isaac_workflow.resolve_recommended_checkpoint
            original_get_checkpoint_path = isaac_workflow.get_checkpoint_path
            try:
                isaac_workflow.resolve_recommended_checkpoint = lambda experiment_name: {
                    "checkpoint_path": str(Path(temp_dir) / "model_1.pt"),
                }
                isaac_workflow.get_checkpoint_path = lambda *args, **kwargs: str(checkpoint_path)
                selection = resolve_checkpoint_selection(
                    "unused",
                    types.SimpleNamespace(
                        experiment_name="geo_landau_game",
                        load_run=None,
                        load_checkpoint=None,
                    ),
                    prefer_latest=True,
                )
            finally:
                isaac_workflow.resolve_recommended_checkpoint = original_recommended
                isaac_workflow.get_checkpoint_path = original_get_checkpoint_path

            self.assertEqual(selection["path"], str(checkpoint_path))
            self.assertEqual(selection["source"], "latest")

    def test_wildcard_defaults_are_not_treated_as_explicit_checkpoint_request(self) -> None:
        self.assertTrue(_is_default_checkpoint_selector(".*", "model_.*.pt"))
        self.assertTrue(_is_default_checkpoint_selector(None, None))
        self.assertFalse(_is_default_checkpoint_selector("2026-04-18_run", "model_150.pt"))

        with TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "model_7.pt"
            checkpoint_path.write_text("", encoding="utf-8")
            original_recommended = isaac_workflow.resolve_recommended_checkpoint
            original_get_checkpoint_path = isaac_workflow.get_checkpoint_path
            try:
                isaac_workflow.resolve_recommended_checkpoint = lambda experiment_name: {
                    "checkpoint_path": str(checkpoint_path),
                    "reason": "promoted",
                }
                isaac_workflow.get_checkpoint_path = lambda *args, **kwargs: str(Path(temp_dir) / "model_99.pt")
                selection = resolve_checkpoint_selection(
                    "unused",
                    types.SimpleNamespace(
                        experiment_name="geo_landau_game",
                        load_run=".*",
                        load_checkpoint="model_.*.pt",
                    ),
                )
            finally:
                isaac_workflow.resolve_recommended_checkpoint = original_recommended
                isaac_workflow.get_checkpoint_path = original_get_checkpoint_path

            self.assertEqual(selection["path"], str(checkpoint_path))
            self.assertEqual(selection["source"], "recommended")

    def test_landau_requires_recommended_checkpoint_by_default(self) -> None:
        original_recommended = isaac_workflow.resolve_recommended_checkpoint
        try:
            isaac_workflow.resolve_recommended_checkpoint = lambda experiment_name: None
            with self.assertRaises(ValueError):
                resolve_checkpoint_selection(
                    "unused",
                    types.SimpleNamespace(
                        experiment_name="geo_landau_game",
                        load_run=None,
                        load_checkpoint=None,
                    ),
                )
        finally:
            isaac_workflow.resolve_recommended_checkpoint = original_recommended


class LoadRunnerCheckpointTests(unittest.TestCase):
    def test_skips_optimizer_restore_when_disabled(self) -> None:
        class FakeOptimizer:
            def __init__(self) -> None:
                self.loaded_state_dict = None

            def load_state_dict(self, state_dict) -> None:
                self.loaded_state_dict = state_dict

        actor_critic = torch.nn.Sequential(torch.nn.Linear(3, 2), torch.nn.ELU(), torch.nn.Linear(2, 1))
        checkpoint_model = {name: tensor.clone() for name, tensor in actor_critic.state_dict().items()}
        optimizer_state = {"state": {0: {"momentum_buffer": torch.ones(1)}}, "param_groups": [{"lr": 1.0e-3}]}

        runner = types.SimpleNamespace(
            alg=types.SimpleNamespace(
                actor_critic=actor_critic,
                optimizer=FakeOptimizer(),
                rnd=None,
            ),
            empirical_normalization=False,
            current_learning_iteration=0,
        )

        with TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "model_1.pt"
            torch.save(
                {
                    "model_state_dict": checkpoint_model,
                    "optimizer_state_dict": optimizer_state,
                    "iter": 37,
                },
                checkpoint_path,
            )

            metadata = load_runner_checkpoint(runner, str(checkpoint_path), load_optimizer=False)

        self.assertEqual(metadata["mode"], "strict")
        self.assertFalse(metadata["optimizer_loaded"])
        self.assertIsNone(runner.alg.optimizer.loaded_state_dict)
        self.assertEqual(runner.current_learning_iteration, 37)


if __name__ == "__main__":
    unittest.main()
