from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT.parent.parent) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT.parent.parent))


def _install_isaaclab_tasks_stubs() -> None:
    if "torch" not in sys.modules:
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

from algorithms.urdf_learn_wasd_walk.isaac_workflow import apply_checkpoint_playback_compat, clamp_base_velocity_command


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


class ApplyCheckpointPlaybackCompatTests(unittest.TestCase):
    def test_restores_action_scale_and_command_ranges_from_checkpoint_env(self) -> None:
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
                        "commands:",
                        "  base_velocity:",
                        "    ranges:",
                        "      lin_vel_x: [0.0, 0.0]",
                        "      lin_vel_y: [0.45, 1.0]",
                        "      ang_vel_z: [0.0, 0.0]",
                    )
                ),
                encoding="utf-8",
            )
            env_cfg = types.SimpleNamespace(
                actions=types.SimpleNamespace(joint_pos=types.SimpleNamespace(scale={"thigh_.*": 0.4})),
                commands=types.SimpleNamespace(
                    base_velocity=types.SimpleNamespace(
                        ranges=types.SimpleNamespace(
                            lin_vel_x=(-0.2, 0.2),
                            lin_vel_y=(-0.2, 1.0),
                            ang_vel_z=(-0.5, 0.5),
                        )
                    )
                ),
            )

            applied = apply_checkpoint_playback_compat(env_cfg, str(checkpoint_path))

            self.assertTrue(applied)
            self.assertEqual(env_cfg.actions.joint_pos.scale["thigh_.*"], 0.5)
            self.assertEqual(env_cfg.actions.joint_pos.scale["foot_.*"], 0.35)
            self.assertEqual(env_cfg.commands.base_velocity.ranges.lin_vel_x, (0.0, 0.0))
            self.assertEqual(env_cfg.commands.base_velocity.ranges.lin_vel_y, (0.45, 1.0))
            self.assertEqual(env_cfg.commands.base_velocity.ranges.ang_vel_z, (0.0, 0.0))

    def test_noop_when_checkpoint_has_no_saved_env(self) -> None:
        with TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "model_1.pt"
            checkpoint_path.write_text("", encoding="utf-8")
            env_cfg = types.SimpleNamespace(
                actions=types.SimpleNamespace(joint_pos=types.SimpleNamespace(scale={"thigh_.*": 0.4})),
                commands=types.SimpleNamespace(
                    base_velocity=types.SimpleNamespace(
                        ranges=types.SimpleNamespace(
                            lin_vel_x=(-0.2, 0.2),
                            lin_vel_y=(-0.2, 1.0),
                            ang_vel_z=(-0.5, 0.5),
                        )
                    )
                ),
            )

            applied = apply_checkpoint_playback_compat(env_cfg, str(checkpoint_path))

            self.assertFalse(applied)
            self.assertEqual(env_cfg.actions.joint_pos.scale["thigh_.*"], 0.4)
            self.assertEqual(env_cfg.commands.base_velocity.ranges.lin_vel_y, (-0.2, 1.0))


if __name__ == "__main__":
    unittest.main()
