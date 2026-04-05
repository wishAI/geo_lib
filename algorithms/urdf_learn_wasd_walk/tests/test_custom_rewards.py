from __future__ import annotations

import sys
import unittest
import importlib.util
from pathlib import Path

MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT.parent.parent) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT.parent.parent))

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
if TORCH_AVAILABLE:
    import torch

    from algorithms.urdf_learn_wasd_walk.custom_rewards import (
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
        grouped_support_first_contact_reward,
        grouped_support_mode_time,
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


@unittest.skipUnless(TORCH_AVAILABLE, "torch is not available in the system test interpreter")
class GroupedSupportModeTimeTests(unittest.TestCase):
    def test_contact_mode_uses_longest_contact_timer(self) -> None:
        contact_time = torch.tensor([[0.4, 0.1], [0.0, 0.3]], dtype=torch.float32)
        air_time = torch.tensor([[0.0, 0.0], [0.2, 0.0]], dtype=torch.float32)

        in_contact, mode_time = grouped_support_mode_time(contact_time, air_time)

        self.assertTrue(torch.equal(in_contact, torch.tensor([True, True])))
        self.assertTrue(torch.allclose(mode_time, torch.tensor([0.4, 0.3])))

    def test_air_mode_uses_shortest_air_timer(self) -> None:
        contact_time = torch.tensor([[0.0, 0.0]], dtype=torch.float32)
        air_time = torch.tensor([[0.7, 0.5]], dtype=torch.float32)

        in_contact, mode_time = grouped_support_mode_time(contact_time, air_time)

        self.assertTrue(torch.equal(in_contact, torch.tensor([False])))
        self.assertTrue(torch.allclose(mode_time, torch.tensor([0.5])))


@unittest.skipUnless(TORCH_AVAILABLE, "torch is not available in the system test interpreter")
class GroupedSupportFirstContactRewardTests(unittest.TestCase):
    def test_rewards_group_landing_after_threshold(self) -> None:
        first_contact = torch.tensor([[True, False], [False, False]], dtype=torch.bool)
        last_air_time = torch.tensor([[0.7, 0.6], [0.9, 0.8]], dtype=torch.float32)

        reward = grouped_support_first_contact_reward(first_contact, last_air_time, threshold=0.4)

        self.assertTrue(torch.allclose(reward, torch.tensor([0.2, 0.0])))

    def test_requires_whole_side_airborne(self) -> None:
        first_contact = torch.tensor([[False, True]], dtype=torch.bool)
        last_air_time = torch.tensor([[0.45, 0.8]], dtype=torch.float32)

        reward = grouped_support_first_contact_reward(first_contact, last_air_time, threshold=0.5)

        self.assertTrue(torch.allclose(reward, torch.tensor([0.0])))


@unittest.skipUnless(TORCH_AVAILABLE, "torch is not available in the system test interpreter")
class GroupedSupportAirTimePositiveBipedTests(unittest.TestCase):
    def _make_env(
        self,
        *,
        command_xy: tuple[float, float],
        contact_time: torch.Tensor,
        air_time: torch.Tensor,
    ):
        env = type("Env", (), {})()
        env.step_dt = 0.02
        env.command_manager = type(
            "CommandManager",
            (),
            {
                "get_command": lambda self, name: torch.tensor(
                    [[command_xy[0], command_xy[1], 0.0]], dtype=torch.float32
                )
            },
        )()
        sensor = type("Sensor", (), {})()
        sensor.data = type(
            "Data",
            (),
            {
                "current_contact_time": contact_time,
                "current_air_time": air_time,
            },
        )()
        env.scene = type("Scene", (), {"sensors": {"contact_forces": sensor}})()
        return env

    def test_rewards_true_single_stance_on_grouped_support_links(self) -> None:
        env = self._make_env(
            command_xy=(0.5, 0.0),
            contact_time=torch.tensor([[0.3, 0.1, 0.0, 0.0]], dtype=torch.float32),
            air_time=torch.tensor([[0.0, 0.0, 0.45, 0.5]], dtype=torch.float32),
        )
        left_cfg = type("Cfg", (), {"name": "contact_forces", "body_ids": [0, 1]})()
        right_cfg = type("Cfg", (), {"name": "contact_forces", "body_ids": [2, 3]})()

        reward = grouped_support_air_time_positive_biped(
            env,
            command_name="base_velocity",
            threshold=0.4,
            left_sensor_cfg=left_cfg,
            right_sensor_cfg=right_cfg,
        )

        self.assertTrue(torch.allclose(reward, torch.tensor([0.3])))

    def test_returns_zero_for_double_support(self) -> None:
        env = self._make_env(
            command_xy=(0.5, 0.0),
            contact_time=torch.tensor([[0.3, 0.1, 0.2, 0.1]], dtype=torch.float32),
            air_time=torch.zeros((1, 4), dtype=torch.float32),
        )
        left_cfg = type("Cfg", (), {"name": "contact_forces", "body_ids": [0, 1]})()
        right_cfg = type("Cfg", (), {"name": "contact_forces", "body_ids": [2, 3]})()

        reward = grouped_support_air_time_positive_biped(
            env,
            command_name="base_velocity",
            threshold=0.4,
            left_sensor_cfg=left_cfg,
            right_sensor_cfg=right_cfg,
        )

        self.assertTrue(torch.allclose(reward, torch.tensor([0.0])))

    def test_deadband_removes_tiny_single_stance_blips(self) -> None:
        env = self._make_env(
            command_xy=(0.5, 0.0),
            contact_time=torch.tensor([[0.12, 0.1, 0.0, 0.0]], dtype=torch.float32),
            air_time=torch.tensor([[0.0, 0.0, 0.12, 0.1]], dtype=torch.float32),
        )
        left_cfg = type("Cfg", (), {"name": "contact_forces", "body_ids": [0, 1]})()
        right_cfg = type("Cfg", (), {"name": "contact_forces", "body_ids": [2, 3]})()

        reward = grouped_support_air_time_positive_biped(
            env,
            command_name="base_velocity",
            threshold=0.4,
            min_single_stance_time=0.15,
            left_sensor_cfg=left_cfg,
            right_sensor_cfg=right_cfg,
        )

        self.assertTrue(torch.allclose(reward, torch.tensor([0.0])))


@unittest.skipUnless(TORCH_AVAILABLE, "torch is not available in the system test interpreter")
class GaitPhaseClockObservationTests(unittest.TestCase):
    def test_scales_phase_with_command_speed(self) -> None:
        env = type("Env", (), {})()
        env.step_dt = 0.02
        env.episode_length_buf = torch.tensor([15.0, 15.0], dtype=torch.float32)
        env.command_manager = type(
            "CommandManager",
            (),
            {
                "get_command": lambda self, name: torch.tensor(
                    [[0.3, 0.0, 0.0], [0.8, 0.0, 0.0]],
                    dtype=torch.float32,
                )
            },
        )()

        observation = gait_phase_clock_observation(
            env,
            command_name="base_velocity",
            slow_speed=0.3,
            fast_speed=0.8,
            slow_period=0.6,
            fast_period=0.4,
        )

        expected = torch.tensor(
            [[0.0, -1.0], [-1.0, 0.0]],
            dtype=torch.float32,
        )
        self.assertTrue(torch.allclose(observation, expected, atol=1.0e-5))


@unittest.skipUnless(TORCH_AVAILABLE, "torch is not available in the system test interpreter")
class FeetPositionsInRootFrameTests(unittest.TestCase):
    def test_returns_left_and_right_centers_in_root_frame(self) -> None:
        env = type("Env", (), {})()
        robot = type("Robot", (), {})()
        robot.data = type(
            "Data",
            (),
            {
                "body_pos_w": torch.tensor(
                    [
                        [
                            [1.0, 2.0, 3.0],
                            [1.2, 2.1, 3.4],
                            [0.9, 2.4, 3.1],
                        ]
                    ],
                    dtype=torch.float32,
                ),
                "body_quat_w": torch.tensor(
                    [[[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]],
                    dtype=torch.float32,
                ),
            },
        )()

        class Scene(dict):
            def __init__(self):
                super().__init__({"robot": robot})
                self.sensors = {}

        env.scene = Scene()
        root_cfg = type("Cfg", (), {"name": "robot", "body_ids": [0]})()
        left_cfg = type("Cfg", (), {"name": "robot", "body_ids": [1]})()
        right_cfg = type("Cfg", (), {"name": "robot", "body_ids": [2]})()

        observation = feet_positions_in_root_frame(env, root_cfg, left_cfg, right_cfg)

        expected = torch.tensor([[0.2, 0.1, 0.4, -0.1, 0.4, 0.1]], dtype=torch.float32)
        self.assertTrue(torch.allclose(observation, expected, atol=1.0e-5))


@unittest.skipUnless(TORCH_AVAILABLE, "torch is not available in the system test interpreter")
class FeetContactStateObservationTests(unittest.TestCase):
    def test_reports_binary_per_foot_contact_state(self) -> None:
        env = type("Env", (), {})()
        sensor = type("Sensor", (), {})()
        sensor.data = type(
            "Data",
            (),
            {
                "current_contact_time": torch.tensor([[0.2, 0.0]], dtype=torch.float32),
            },
        )()
        env.scene = type("Scene", (), {"sensors": {"contact_forces": sensor}})()
        left_cfg = type("Cfg", (), {"name": "contact_forces", "body_ids": [0]})()
        right_cfg = type("Cfg", (), {"name": "contact_forces", "body_ids": [1]})()

        observation = feet_contact_state_observation(env, left_cfg, right_cfg)

        self.assertTrue(torch.allclose(observation, torch.tensor([[1.0, 0.0]], dtype=torch.float32)))


@unittest.skipUnless(TORCH_AVAILABLE, "torch is not available in the system test interpreter")
class FeetModeTimeObservationTests(unittest.TestCase):
    def test_returns_normalized_time_since_last_contact_switch(self) -> None:
        env = type("Env", (), {})()
        sensor = type("Sensor", (), {})()
        sensor.data = type(
            "Data",
            (),
            {
                "current_contact_time": torch.tensor([[0.3, 0.0]], dtype=torch.float32),
                "current_air_time": torch.tensor([[0.0, 0.4]], dtype=torch.float32),
            },
        )()
        env.scene = type("Scene", (), {"sensors": {"contact_forces": sensor}})()
        left_cfg = type("Cfg", (), {"name": "contact_forces", "body_ids": [0]})()
        right_cfg = type("Cfg", (), {"name": "contact_forces", "body_ids": [1]})()

        observation = feet_mode_time_observation(env, left_cfg, right_cfg, time_window=1.0)

        self.assertTrue(torch.allclose(observation, torch.tensor([[0.3, 0.4]], dtype=torch.float32), atol=1.0e-5))


@unittest.skipUnless(TORCH_AVAILABLE, "torch is not available in the system test interpreter")
class PhaseClockAlternatingFootContactRewardTests(unittest.TestCase):
    def _make_env(
        self,
        *,
        current_contact_time: torch.Tensor,
        root_lin_vel_b: torch.Tensor | None = None,
        root_ang_vel_w: torch.Tensor | None = None,
    ):
        env = type("Env", (), {})()
        env.step_dt = 0.02
        env.episode_length_buf = torch.tensor([0.0], dtype=torch.float32)
        env.command_manager = type(
            "CommandManager",
            (),
            {
                "get_command": lambda self, name: torch.tensor(
                    [[0.5, 0.0, 0.0]], dtype=torch.float32
                )
            },
        )()
        sensor = type("Sensor", (), {})()
        sensor.data = type("Data", (), {"current_contact_time": current_contact_time})()
        if root_lin_vel_b is None:
            root_lin_vel_b = torch.tensor([[0.0, 0.5, 0.0]], dtype=torch.float32)
        if root_ang_vel_w is None:
            root_ang_vel_w = torch.zeros((1, 3), dtype=torch.float32)
        robot = type("Robot", (), {})()
        robot.data = type(
            "Data",
            (),
            {
                "root_lin_vel_b": root_lin_vel_b,
                "root_ang_vel_w": root_ang_vel_w,
            },
        )()

        class Scene(dict):
            def __init__(self):
                super().__init__({"robot": robot})
                self.sensors = {"contact_forces": sensor}

        env.scene = Scene()
        return env

    def test_rewards_when_left_stance_matches_phase_clock(self) -> None:
        env = self._make_env(current_contact_time=torch.tensor([[0.2, 0.0]], dtype=torch.float32))
        left_cfg = type("Cfg", (), {"name": "contact_forces", "body_ids": [0]})()
        right_cfg = type("Cfg", (), {"name": "contact_forces", "body_ids": [1]})()

        reward = phase_clock_alternating_foot_contact_reward(
            env,
            command_name="base_velocity",
            phase_sharpness=4.0,
            slow_speed=0.3,
            fast_speed=0.8,
            slow_period=0.6,
            fast_period=0.4,
            left_sensor_cfg=left_cfg,
            right_sensor_cfg=right_cfg,
        )

        self.assertGreater(float(reward[0]), 0.97)

    def test_penalizes_double_support_when_clock_expects_single_support(self) -> None:
        env = self._make_env(current_contact_time=torch.tensor([[0.2, 0.2]], dtype=torch.float32))
        left_cfg = type("Cfg", (), {"name": "contact_forces", "body_ids": [0]})()
        right_cfg = type("Cfg", (), {"name": "contact_forces", "body_ids": [1]})()

        reward = phase_clock_alternating_foot_contact_reward(
            env,
            command_name="base_velocity",
            phase_sharpness=4.0,
            slow_speed=0.3,
            fast_speed=0.8,
            slow_period=0.6,
            fast_period=0.4,
            left_sensor_cfg=left_cfg,
            right_sensor_cfg=right_cfg,
        )

        self.assertLess(float(reward[0]), 0.2)

    def test_velocity_gate_suppresses_reward_for_backward_motion(self) -> None:
        env = self._make_env(
            current_contact_time=torch.tensor([[0.2, 0.0]], dtype=torch.float32),
            root_lin_vel_b=torch.tensor([[0.0, -0.4, 0.0]], dtype=torch.float32),
        )
        left_cfg = type("Cfg", (), {"name": "contact_forces", "body_ids": [0]})()
        right_cfg = type("Cfg", (), {"name": "contact_forces", "body_ids": [1]})()

        reward = phase_clock_alternating_foot_contact_reward(
            env,
            command_name="base_velocity",
            phase_sharpness=4.0,
            slow_speed=0.3,
            fast_speed=0.8,
            slow_period=0.6,
            fast_period=0.4,
            left_sensor_cfg=left_cfg,
            right_sensor_cfg=right_cfg,
            velocity_gate_std=0.3,
        )

        self.assertLess(float(reward[0]), 0.05)

    def test_velocity_gate_floor_keeps_some_phase_signal(self) -> None:
        env = self._make_env(
            current_contact_time=torch.tensor([[0.2, 0.0]], dtype=torch.float32),
            root_lin_vel_b=torch.tensor([[0.0, -0.4, 0.0]], dtype=torch.float32),
        )
        left_cfg = type("Cfg", (), {"name": "contact_forces", "body_ids": [0]})()
        right_cfg = type("Cfg", (), {"name": "contact_forces", "body_ids": [1]})()

        reward = phase_clock_alternating_foot_contact_reward(
            env,
            command_name="base_velocity",
            phase_sharpness=4.0,
            slow_speed=0.3,
            fast_speed=0.8,
            slow_period=0.6,
            fast_period=0.4,
            left_sensor_cfg=left_cfg,
            right_sensor_cfg=right_cfg,
            velocity_gate_std=0.3,
            velocity_gate_floor=0.25,
        )

        self.assertGreater(float(reward[0]), 0.2)


@unittest.skipUnless(TORCH_AVAILABLE, "torch is not available in the system test interpreter")
class AlternatingBipedAsyncRewardTests(unittest.TestCase):
    def _make_env(
        self,
        *,
        command_xy: tuple[float, float],
        contact_time: torch.Tensor,
        air_time: torch.Tensor,
    ):
        env = type("Env", (), {})()
        env.command_manager = type(
            "CommandManager",
            (),
            {
                "get_command": lambda self, name: torch.tensor(
                    [[command_xy[0], command_xy[1], 0.0]], dtype=torch.float32
                )
            },
        )()
        sensor = type("Sensor", (), {})()
        sensor.data = type(
            "Data",
            (),
            {
                "current_contact_time": contact_time,
                "current_air_time": air_time,
            },
        )()
        env.scene = type("Scene", (), {"sensors": {"contact_forces": sensor}})()
        return env

    def test_rewards_matched_alternating_support_timing(self) -> None:
        env = self._make_env(
            command_xy=(0.5, 0.0),
            contact_time=torch.tensor([[0.0, 0.0, 0.2, 0.18]], dtype=torch.float32),
            air_time=torch.tensor([[0.22, 0.2, 0.0, 0.0]], dtype=torch.float32),
        )
        left_cfg = type("Cfg", (), {"name": "contact_forces", "body_ids": [0, 1]})()
        right_cfg = type("Cfg", (), {"name": "contact_forces", "body_ids": [2, 3]})()

        reward = alternating_biped_async_reward(
            env,
            command_name="base_velocity",
            std=0.03,
            max_err=0.25,
            left_sensor_cfg=left_cfg,
            right_sensor_cfg=right_cfg,
        )

        self.assertTrue(torch.allclose(reward, torch.tensor([1.0]), atol=1.0e-5))

    def test_penalizes_double_support_shuffle_timing(self) -> None:
        env = self._make_env(
            command_xy=(0.5, 0.0),
            contact_time=torch.tensor([[0.2, 0.18, 0.2, 0.18]], dtype=torch.float32),
            air_time=torch.zeros((1, 4), dtype=torch.float32),
        )
        left_cfg = type("Cfg", (), {"name": "contact_forces", "body_ids": [0, 1]})()
        right_cfg = type("Cfg", (), {"name": "contact_forces", "body_ids": [2, 3]})()

        reward = alternating_biped_async_reward(
            env,
            command_name="base_velocity",
            std=0.03,
            max_err=0.25,
            left_sensor_cfg=left_cfg,
            right_sensor_cfg=right_cfg,
        )

        self.assertLess(float(reward[0]), 0.1)


@unittest.skipUnless(TORCH_AVAILABLE, "torch is not available in the system test interpreter")
class PrimarySingleSupportRewardTests(unittest.TestCase):
    def _make_env(self, *, command_xy: tuple[float, float], current_contact_time: torch.Tensor):
        env = type("Env", (), {})()
        env.command_manager = type(
            "CommandManager",
            (),
            {
                "get_command": lambda self, name: torch.tensor(
                    [[command_xy[0], command_xy[1], 0.0]], dtype=torch.float32
                )
            },
        )()
        sensor = type("Sensor", (), {})()
        sensor.data = type("Data", (), {"current_contact_time": current_contact_time})()
        env.scene = type("Scene", (), {"sensors": {"contact_forces": sensor}})()
        return env

    def test_rewards_true_single_support(self) -> None:
        env = self._make_env(command_xy=(0.5, 0.0), current_contact_time=torch.tensor([[0.2, 0.0]], dtype=torch.float32))
        left_cfg = type("Cfg", (), {"name": "contact_forces", "body_ids": [0]})()
        right_cfg = type("Cfg", (), {"name": "contact_forces", "body_ids": [1]})()

        reward = primary_single_support_reward(
            env,
            command_name="base_velocity",
            overlap_grace_period=0.08,
            left_sensor_cfg=left_cfg,
            right_sensor_cfg=right_cfg,
        )

        self.assertTrue(torch.allclose(reward, torch.tensor([1.0]), atol=1.0e-5))

    def test_rewards_brief_double_support_overlap(self) -> None:
        env = self._make_env(command_xy=(0.5, 0.0), current_contact_time=torch.tensor([[0.20, 0.04]], dtype=torch.float32))
        left_cfg = type("Cfg", (), {"name": "contact_forces", "body_ids": [0]})()
        right_cfg = type("Cfg", (), {"name": "contact_forces", "body_ids": [1]})()

        reward = primary_single_support_reward(
            env,
            command_name="base_velocity",
            overlap_grace_period=0.08,
            left_sensor_cfg=left_cfg,
            right_sensor_cfg=right_cfg,
        )

        self.assertTrue(torch.allclose(reward, torch.tensor([1.0]), atol=1.0e-5))

    def test_rejects_prolonged_double_support(self) -> None:
        env = self._make_env(command_xy=(0.5, 0.0), current_contact_time=torch.tensor([[0.20, 0.12]], dtype=torch.float32))
        left_cfg = type("Cfg", (), {"name": "contact_forces", "body_ids": [0]})()
        right_cfg = type("Cfg", (), {"name": "contact_forces", "body_ids": [1]})()

        reward = primary_single_support_reward(
            env,
            command_name="base_velocity",
            overlap_grace_period=0.08,
            left_sensor_cfg=left_cfg,
            right_sensor_cfg=right_cfg,
        )

        self.assertTrue(torch.allclose(reward, torch.tensor([0.0]), atol=1.0e-5))


@unittest.skipUnless(TORCH_AVAILABLE, "torch is not available in the system test interpreter")
class GroupedSupportDoubleStanceTimePenaltyTests(unittest.TestCase):
    def _make_env(
        self,
        *,
        command_xy: tuple[float, float],
        contact_time: torch.Tensor,
    ):
        env = type("Env", (), {})()
        env.command_manager = type(
            "CommandManager",
            (),
            {
                "get_command": lambda self, name: torch.tensor(
                    [[command_xy[0], command_xy[1], 0.0]], dtype=torch.float32
                )
            },
        )()
        sensor = type("Sensor", (), {})()
        sensor.data = type(
            "Data",
            (),
            {
                "current_contact_time": contact_time,
            },
        )()
        env.scene = type("Scene", (), {"sensors": {"contact_forces": sensor}})()
        return env

    def test_penalizes_prolonged_double_support(self) -> None:
        env = self._make_env(
            command_xy=(0.5, 0.0),
            contact_time=torch.tensor([[0.22, 0.2, 0.24, 0.21]], dtype=torch.float32),
        )
        left_cfg = type("Cfg", (), {"name": "contact_forces", "body_ids": [0, 1]})()
        right_cfg = type("Cfg", (), {"name": "contact_forces", "body_ids": [2, 3]})()

        penalty = grouped_support_double_stance_time_penalty(
            env,
            command_name="base_velocity",
            threshold=0.12,
            left_sensor_cfg=left_cfg,
            right_sensor_cfg=right_cfg,
        )

        self.assertTrue(torch.allclose(penalty, torch.tensor([0.1])))

    def test_allows_brief_double_support_transition(self) -> None:
        env = self._make_env(
            command_xy=(0.5, 0.0),
            contact_time=torch.tensor([[0.08, 0.06, 0.09, 0.07]], dtype=torch.float32),
        )
        left_cfg = type("Cfg", (), {"name": "contact_forces", "body_ids": [0, 1]})()
        right_cfg = type("Cfg", (), {"name": "contact_forces", "body_ids": [2, 3]})()

        penalty = grouped_support_double_stance_time_penalty(
            env,
            command_name="base_velocity",
            threshold=0.12,
            left_sensor_cfg=left_cfg,
            right_sensor_cfg=right_cfg,
        )

        self.assertTrue(torch.allclose(penalty, torch.tensor([0.0])))


@unittest.skipUnless(TORCH_AVAILABLE, "torch is not available in the system test interpreter")
class SupportWidthDeviationTests(unittest.TestCase):
    def _make_env(
        self,
        *,
        command_xy: tuple[float, float],
        root_quat_w: torch.Tensor,
        body_pos_w: torch.Tensor,
    ):
        env = type("Env", (), {})()
        env.command_manager = type(
            "CommandManager",
            (),
            {
                "get_command": lambda self, name: torch.tensor(
                    [[command_xy[0], command_xy[1], 0.0]], dtype=torch.float32
                )
            },
        )()
        robot = type("Robot", (), {})()
        robot.data = type("Data", (), {"root_quat_w": root_quat_w, "body_pos_w": body_pos_w})()
        env.scene = {"robot": robot}
        return env

    def test_uses_body_frame_lateral_axis_for_landau_width(self) -> None:
        yaw_quarter_turn = torch.tensor([[0.70710677, 0.0, 0.0, 0.70710677]], dtype=torch.float32)
        body_pos_w = torch.tensor(
            [
                [
                    [0.0, 0.125, 0.0],
                    [0.0, 0.135, 0.0],
                    [0.0, -0.115, 0.0],
                    [0.0, -0.125, 0.0],
                ]
            ],
            dtype=torch.float32,
        )
        env = self._make_env(command_xy=(0.5, 0.0), root_quat_w=yaw_quarter_turn, body_pos_w=body_pos_w)
        left_cfg = type("Cfg", (), {"name": "robot", "body_ids": [0, 1]})()
        right_cfg = type("Cfg", (), {"name": "robot", "body_ids": [2, 3]})()

        penalty = support_width_deviation(
            env,
            command_name="base_velocity",
            target_width=0.23,
            tolerance=0.01,
            forward_body_axis="y",
            left_asset_cfg=left_cfg,
            right_asset_cfg=right_cfg,
        )

        self.assertTrue(torch.allclose(penalty, torch.tensor([0.01]), atol=1.0e-5))

    def test_returns_zero_when_robot_is_not_commanded_to_move(self) -> None:
        env = self._make_env(
            command_xy=(0.0, 0.0),
            root_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
            body_pos_w=torch.tensor(
                [[[0.12, 0.0, 0.0], [0.13, 0.0, 0.0], [-0.12, 0.0, 0.0], [-0.13, 0.0, 0.0]]],
                dtype=torch.float32,
            ),
        )
        left_cfg = type("Cfg", (), {"name": "robot", "body_ids": [0, 1]})()
        right_cfg = type("Cfg", (), {"name": "robot", "body_ids": [2, 3]})()

        penalty = support_width_deviation(
            env,
            command_name="base_velocity",
            target_width=0.23,
            tolerance=0.01,
            forward_body_axis="y",
            left_asset_cfg=left_cfg,
            right_asset_cfg=right_cfg,
        )

        self.assertTrue(torch.allclose(penalty, torch.tensor([0.0])))


@unittest.skipUnless(TORCH_AVAILABLE, "torch is not available in the system test interpreter")
class SupportWidthAboveMaxTests(unittest.TestCase):
    def _make_env(
        self,
        *,
        command_xy: tuple[float, float],
        root_quat_w: torch.Tensor,
        body_pos_w: torch.Tensor,
    ):
        env = type("Env", (), {})()
        env.command_manager = type(
            "CommandManager",
            (),
            {
                "get_command": lambda self, name: torch.tensor(
                    [[command_xy[0], command_xy[1], 0.0]], dtype=torch.float32
                )
            },
        )()
        robot = type("Robot", (), {})()
        robot.data = type("Data", (), {"root_quat_w": root_quat_w, "body_pos_w": body_pos_w})()
        env.scene = {"robot": robot}
        return env

    def test_penalizes_only_width_excess(self) -> None:
        env = self._make_env(
            command_xy=(0.5, 0.0),
            root_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
            body_pos_w=torch.tensor(
                [[[0.16, 0.0, 0.0], [0.14, 0.0, 0.0], [-0.16, 0.0, 0.0], [-0.14, 0.0, 0.0]]],
                dtype=torch.float32,
            ),
        )
        left_cfg = type("Cfg", (), {"name": "robot", "body_ids": [0, 1]})()
        right_cfg = type("Cfg", (), {"name": "robot", "body_ids": [2, 3]})()

        penalty = support_width_above_max(
            env,
            command_name="base_velocity",
            max_width=0.25,
            forward_body_axis="y",
            left_asset_cfg=left_cfg,
            right_asset_cfg=right_cfg,
        )

        self.assertTrue(torch.allclose(penalty, torch.tensor([0.05]), atol=1.0e-5))


@unittest.skipUnless(TORCH_AVAILABLE, "torch is not available in the system test interpreter")
class LandingStepAheadRewardTests(unittest.TestCase):
    def _make_env(
        self,
        *,
        command_xy: tuple[float, float],
        current_contact_time: torch.Tensor,
        last_air_time: torch.Tensor,
        first_contact: torch.Tensor,
        root_quat_w: torch.Tensor,
        body_pos_w: torch.Tensor,
    ):
        env = type("Env", (), {})()
        env.step_dt = 0.02
        env.command_manager = type(
            "CommandManager",
            (),
            {
                "get_command": lambda self, name: torch.tensor(
                    [[command_xy[0], command_xy[1], 0.0]], dtype=torch.float32
                )
            },
        )()
        sensor = type("Sensor", (), {})()
        sensor.data = type(
            "Data",
            (),
            {
                "current_contact_time": current_contact_time,
                "last_air_time": last_air_time,
            },
        )()
        sensor.compute_first_contact = lambda dt: first_contact
        robot = type("Robot", (), {})()
        robot.data = type("Data", (), {"root_quat_w": root_quat_w, "body_pos_w": body_pos_w})()

        class Scene(dict):
            def __init__(self):
                super().__init__({"robot": robot})
                self.sensors = {"contact_forces": sensor}

        env.scene = Scene()
        return env

    def test_rewards_touchdown_ahead_of_opposite_support(self) -> None:
        env = self._make_env(
            command_xy=(0.5, 0.0),
            current_contact_time=torch.tensor([[0.2, 0.2, 0.3, 0.3]], dtype=torch.float32),
            last_air_time=torch.tensor([[0.24, 0.25, 0.0, 0.0]], dtype=torch.float32),
            first_contact=torch.tensor([[True, False, False, False]], dtype=torch.bool),
            root_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
            body_pos_w=torch.tensor(
                [[[0.0, 0.10, 0.0], [0.0, 0.12, 0.0], [0.0, -0.05, 0.0], [0.0, -0.07, 0.0]]],
                dtype=torch.float32,
            ),
        )
        left_sensor_cfg = type("Cfg", (), {"name": "contact_forces", "body_ids": [0, 1]})()
        right_sensor_cfg = type("Cfg", (), {"name": "contact_forces", "body_ids": [2, 3]})()
        left_asset_cfg = type("Cfg", (), {"name": "robot", "body_ids": [0, 1]})()
        right_asset_cfg = type("Cfg", (), {"name": "robot", "body_ids": [2, 3]})()

        reward = landing_step_ahead_reward(
            env,
            command_name="base_velocity",
            step_length_threshold=0.08,
            min_air_time=0.2,
            max_rewarded_step_length=0.2,
            forward_body_axis="y",
            left_sensor_cfg=left_sensor_cfg,
            right_sensor_cfg=right_sensor_cfg,
            left_asset_cfg=left_asset_cfg,
            right_asset_cfg=right_asset_cfg,
        )

        self.assertTrue(torch.allclose(reward, torch.tensor([0.09]), atol=1.0e-5))


@unittest.skipUnless(TORCH_AVAILABLE, "torch is not available in the system test interpreter")
class TouchdownStepLengthDeficitPenaltyTests(unittest.TestCase):
    def _make_env(
        self,
        *,
        command_xy: tuple[float, float],
        current_contact_time: torch.Tensor,
        last_air_time: torch.Tensor,
        first_contact: torch.Tensor,
        root_quat_w: torch.Tensor,
        body_pos_w: torch.Tensor,
    ):
        env = type("Env", (), {})()
        env.step_dt = 0.02
        env.command_manager = type(
            "CommandManager",
            (),
            {
                "get_command": lambda self, name: torch.tensor(
                    [[command_xy[0], command_xy[1], 0.0]], dtype=torch.float32
                )
            },
        )()
        sensor = type("Sensor", (), {})()
        sensor.data = type(
            "Data",
            (),
            {
                "current_contact_time": current_contact_time,
                "last_air_time": last_air_time,
            },
        )()
        sensor.compute_first_contact = lambda dt: first_contact
        robot = type("Robot", (), {})()
        robot.data = type("Data", (), {"root_quat_w": root_quat_w, "body_pos_w": body_pos_w})()

        class Scene(dict):
            def __init__(self):
                super().__init__({"robot": robot})
                self.sensors = {"contact_forces": sensor}

        env.scene = Scene()
        return env

    def test_penalizes_touchdown_that_lands_behind_stance_side(self) -> None:
        env = self._make_env(
            command_xy=(0.5, 0.0),
            current_contact_time=torch.tensor([[0.2, 0.0, 0.3, 0.0]], dtype=torch.float32),
            last_air_time=torch.tensor([[0.24, 0.0, 0.0, 0.0]], dtype=torch.float32),
            first_contact=torch.tensor([[True, False, False, False]], dtype=torch.bool),
            root_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
            body_pos_w=torch.tensor(
                [[[0.0, -0.02, 0.0], [0.0, -0.02, 0.0], [0.0, 0.06, 0.0], [0.0, 0.06, 0.0]]],
                dtype=torch.float32,
            ),
        )
        left_sensor_cfg = type("Cfg", (), {"name": "contact_forces", "body_ids": [0]})()
        right_sensor_cfg = type("Cfg", (), {"name": "contact_forces", "body_ids": [2]})()
        left_asset_cfg = type("Cfg", (), {"name": "robot", "body_ids": [0]})()
        right_asset_cfg = type("Cfg", (), {"name": "robot", "body_ids": [2]})()

        penalty = touchdown_step_length_deficit_penalty(
            env,
            command_name="base_velocity",
            min_step_length=0.04,
            min_air_time=0.2,
            max_penalized_deficit=0.2,
            forward_body_axis="y",
            left_sensor_cfg=left_sensor_cfg,
            right_sensor_cfg=right_sensor_cfg,
            left_asset_cfg=left_asset_cfg,
            right_asset_cfg=right_asset_cfg,
        )

        self.assertTrue(torch.allclose(penalty, torch.tensor([0.12]), atol=1.0e-5))


@unittest.skipUnless(TORCH_AVAILABLE, "torch is not available in the system test interpreter")
class TouchdownSupportWidthExcessPenaltyTests(unittest.TestCase):
    def _make_env(
        self,
        *,
        command_xy: tuple[float, float],
        current_contact_time: torch.Tensor,
        last_air_time: torch.Tensor,
        first_contact: torch.Tensor,
        root_quat_w: torch.Tensor,
        body_pos_w: torch.Tensor,
    ):
        env = type("Env", (), {})()
        env.step_dt = 0.02
        env.command_manager = type(
            "CommandManager",
            (),
            {
                "get_command": lambda self, name: torch.tensor(
                    [[command_xy[0], command_xy[1], 0.0]], dtype=torch.float32
                )
            },
        )()
        sensor = type("Sensor", (), {})()
        sensor.data = type(
            "Data",
            (),
            {
                "current_contact_time": current_contact_time,
                "last_air_time": last_air_time,
            },
        )()
        sensor.compute_first_contact = lambda dt: first_contact
        robot = type("Robot", (), {})()
        robot.data = type("Data", (), {"root_quat_w": root_quat_w, "body_pos_w": body_pos_w})()

        class Scene(dict):
            def __init__(self):
                super().__init__({"robot": robot})
                self.sensors = {"contact_forces": sensor}

        env.scene = Scene()
        return env

    def test_penalizes_wide_touchdown_events(self) -> None:
        env = self._make_env(
            command_xy=(0.5, 0.0),
            current_contact_time=torch.tensor([[0.2, 0.2, 0.3, 0.3]], dtype=torch.float32),
            last_air_time=torch.tensor([[0.12, 0.0, 0.0, 0.0]], dtype=torch.float32),
            first_contact=torch.tensor([[True, False, False, False]], dtype=torch.bool),
            root_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
            body_pos_w=torch.tensor(
                [[[0.17, 0.0, 0.0], [0.15, 0.0, 0.0], [-0.16, 0.0, 0.0], [-0.14, 0.0, 0.0]]],
                dtype=torch.float32,
            ),
        )
        left_sensor_cfg = type("Cfg", (), {"name": "contact_forces", "body_ids": [0]})()
        right_sensor_cfg = type("Cfg", (), {"name": "contact_forces", "body_ids": [2]})()
        left_asset_cfg = type("Cfg", (), {"name": "robot", "body_ids": [0, 1]})()
        right_asset_cfg = type("Cfg", (), {"name": "robot", "body_ids": [2, 3]})()

        penalty = touchdown_support_width_excess_penalty(
            env,
            command_name="base_velocity",
            max_width=0.25,
            min_air_time=0.1,
            max_penalized_excess=0.2,
            forward_body_axis="y",
            left_sensor_cfg=left_sensor_cfg,
            right_sensor_cfg=right_sensor_cfg,
            left_asset_cfg=left_asset_cfg,
            right_asset_cfg=right_asset_cfg,
        )

        self.assertTrue(torch.allclose(penalty, torch.tensor([0.06]), atol=1.0e-5))


@unittest.skipUnless(TORCH_AVAILABLE, "torch is not available in the system test interpreter")
class TouchdownRootStraddleRewardTests(unittest.TestCase):
    def _make_env(
        self,
        *,
        command_xy: tuple[float, float],
        current_contact_time: torch.Tensor,
        last_air_time: torch.Tensor,
        first_contact: torch.Tensor,
        body_pos_w: torch.Tensor,
        root_quat_w: torch.Tensor,
    ):
        env = type("Env", (), {})()
        env.step_dt = 0.02
        env.command_manager = type(
            "CommandManager",
            (),
            {
                "get_command": lambda self, name: torch.tensor(
                    [[command_xy[0], command_xy[1], 0.0]], dtype=torch.float32
                )
            },
        )()
        sensor = type("Sensor", (), {})()
        sensor.data = type(
            "Data",
            (),
            {
                "current_contact_time": current_contact_time,
                "last_air_time": last_air_time,
            },
        )()
        sensor.compute_first_contact = lambda dt: first_contact
        robot = type("Robot", (), {})()
        robot.data = type("Data", (), {"body_pos_w": body_pos_w, "root_quat_w": root_quat_w})()

        class Scene(dict):
            def __init__(self):
                super().__init__({"robot": robot})
                self.sensors = {"contact_forces": sensor}

        env.scene = Scene()
        return env

    def test_rewards_touchdown_when_root_is_between_feet(self) -> None:
        env = self._make_env(
            command_xy=(0.5, 0.0),
            current_contact_time=torch.tensor([[0.2, 0.2, 0.3, 0.3, 0.0]], dtype=torch.float32),
            last_air_time=torch.tensor([[0.24, 0.25, 0.0, 0.0, 0.0]], dtype=torch.float32),
            first_contact=torch.tensor([[True, False, False, False, False]], dtype=torch.bool),
            body_pos_w=torch.tensor(
                [[[0.0, 0.10, 0.0], [0.0, 0.12, 0.0], [0.0, -0.08, 0.0], [0.0, -0.06, 0.0], [0.0, 0.01, 0.2]]],
                dtype=torch.float32,
            ),
            root_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
        )
        root_cfg = type("Cfg", (), {"name": "robot", "body_ids": [4]})()
        left_sensor_cfg = type("Cfg", (), {"name": "contact_forces", "body_ids": [0, 1]})()
        right_sensor_cfg = type("Cfg", (), {"name": "contact_forces", "body_ids": [2, 3]})()
        left_asset_cfg = type("Cfg", (), {"name": "robot", "body_ids": [0, 1]})()
        right_asset_cfg = type("Cfg", (), {"name": "robot", "body_ids": [2, 3]})()

        reward = touchdown_root_straddle_reward(
            env,
            command_name="base_velocity",
            landing_margin=0.05,
            stance_margin=0.04,
            min_air_time=0.2,
            max_rewarded_margin=0.2,
            forward_body_axis="y",
            root_asset_cfg=root_cfg,
            left_sensor_cfg=left_sensor_cfg,
            right_sensor_cfg=right_sensor_cfg,
            left_asset_cfg=left_asset_cfg,
            right_asset_cfg=right_asset_cfg,
        )

        self.assertTrue(torch.allclose(reward, torch.tensor([0.04]), atol=1.0e-5))


@unittest.skipUnless(TORCH_AVAILABLE, "torch is not available in the system test interpreter")
class SingleSupportRootStraddleRewardTests(unittest.TestCase):
    def _make_env(
        self,
        *,
        command_xy: tuple[float, float],
        current_contact_time: torch.Tensor,
        body_pos_w: torch.Tensor,
        root_quat_w: torch.Tensor,
    ):
        env = type("Env", (), {})()
        env.command_manager = type(
            "CommandManager",
            (),
            {
                "get_command": lambda self, name: torch.tensor(
                    [[command_xy[0], command_xy[1], 0.0]], dtype=torch.float32
                )
            },
        )()
        sensor = type("Sensor", (), {})()
        sensor.data = type("Data", (), {"current_contact_time": current_contact_time})()
        robot = type("Robot", (), {})()
        robot.data = type("Data", (), {"body_pos_w": body_pos_w, "root_quat_w": root_quat_w})()

        class Scene(dict):
            def __init__(self):
                super().__init__({"robot": robot})
                self.sensors = {"contact_forces": sensor}

        env.scene = Scene()
        return env

    def test_rewards_single_support_when_swing_foot_moves_ahead_of_root(self) -> None:
        env = self._make_env(
            command_xy=(0.5, 0.0),
            current_contact_time=torch.tensor([[0.0, 0.0, 0.2, 0.2]], dtype=torch.float32),
            body_pos_w=torch.tensor(
                [
                    [
                        [0.0, 0.10, 0.0],
                        [0.0, 0.11, 0.0],
                        [0.0, -0.08, 0.0],
                        [0.0, -0.09, 0.0],
                        [0.0, 0.02, 0.2],
                    ]
                ],
                dtype=torch.float32,
            ),
            root_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
        )
        root_cfg = type("Cfg", (), {"name": "robot", "body_ids": [4]})()
        left_sensor_cfg = type("Cfg", (), {"name": "contact_forces", "body_ids": [0, 1]})()
        right_sensor_cfg = type("Cfg", (), {"name": "contact_forces", "body_ids": [2, 3]})()
        left_asset_cfg = type("Cfg", (), {"name": "robot", "body_ids": [0]})()
        right_asset_cfg = type("Cfg", (), {"name": "robot", "body_ids": [2]})()

        reward = single_support_root_straddle_reward(
            env,
            command_name="base_velocity",
            root_margin=0.02,
            max_rewarded_margin=0.1,
            forward_body_axis="y",
            root_asset_cfg=root_cfg,
            left_sensor_cfg=left_sensor_cfg,
            right_sensor_cfg=right_sensor_cfg,
            left_asset_cfg=left_asset_cfg,
            right_asset_cfg=right_asset_cfg,
        )

        self.assertTrue(torch.allclose(reward, torch.tensor([0.06]), atol=1.0e-5))


@unittest.skipUnless(TORCH_AVAILABLE, "torch is not available in the system test interpreter")
class GroupedSupportFlightTimePenaltyTests(unittest.TestCase):
    def _make_env(
        self,
        *,
        command_xy: tuple[float, float],
        current_contact_time: torch.Tensor,
        current_air_time: torch.Tensor,
    ):
        env = type("Env", (), {})()
        env.command_manager = type(
            "CommandManager",
            (),
            {
                "get_command": lambda self, name: torch.tensor(
                    [[command_xy[0], command_xy[1], 0.0]], dtype=torch.float32
                )
            },
        )()
        sensor = type("Sensor", (), {})()
        sensor.data = type(
            "Data",
            (),
            {
                "current_contact_time": current_contact_time,
                "current_air_time": current_air_time,
            },
        )()
        env.scene = type("Scene", (), {"sensors": {"contact_forces": sensor}})()
        return env

    def test_penalizes_sustained_flight_only_after_threshold(self) -> None:
        env = self._make_env(
            command_xy=(0.5, 0.0),
            current_contact_time=torch.zeros((1, 4), dtype=torch.float32),
            current_air_time=torch.tensor([[0.08, 0.09, 0.07, 0.06]], dtype=torch.float32),
        )
        left_cfg = type("Cfg", (), {"name": "contact_forces", "body_ids": [0, 1]})()
        right_cfg = type("Cfg", (), {"name": "contact_forces", "body_ids": [2, 3]})()

        penalty = grouped_support_flight_time_penalty(
            env,
            command_name="base_velocity",
            threshold=0.04,
            left_sensor_cfg=left_cfg,
            right_sensor_cfg=right_cfg,
        )

        self.assertTrue(torch.allclose(penalty, torch.tensor([0.02]), atol=1.0e-5))


@unittest.skipUnless(TORCH_AVAILABLE, "torch is not available in the system test interpreter")
class SwingHeightDifferenceBelowMinTests(unittest.TestCase):
    def _make_env(
        self,
        *,
        command_xy: tuple[float, float],
        current_contact_time: torch.Tensor,
        body_pos_w: torch.Tensor,
    ):
        env = type("Env", (), {})()
        env.command_manager = type(
            "CommandManager",
            (),
            {
                "get_command": lambda self, name: torch.tensor(
                    [[command_xy[0], command_xy[1], 0.0]], dtype=torch.float32
                )
            },
        )()
        sensor = type("Sensor", (), {})()
        sensor.data = type("Data", (), {"current_contact_time": current_contact_time})()
        robot = type("Robot", (), {})()
        robot.data = type("Data", (), {"body_pos_w": body_pos_w})()

        class Scene(dict):
            def __init__(self):
                super().__init__({"robot": robot})
                self.sensors = {"contact_forces": sensor}

        env.scene = Scene()
        return env

    def test_penalizes_unloaded_side_that_stays_low(self) -> None:
        env = self._make_env(
            command_xy=(0.5, 0.0),
            current_contact_time=torch.tensor([[0.0, 0.0, 0.2, 0.2]], dtype=torch.float32),
            body_pos_w=torch.tensor(
                [[[0.0, 0.0, 0.09], [0.0, 0.0, 0.10], [0.0, 0.0, 0.08], [0.0, 0.0, 0.09]]],
                dtype=torch.float32,
            ),
        )
        left_sensor_cfg = type("Cfg", (), {"name": "contact_forces", "body_ids": [0, 1]})()
        right_sensor_cfg = type("Cfg", (), {"name": "contact_forces", "body_ids": [2, 3]})()
        left_asset_cfg = type("Cfg", (), {"name": "robot", "body_ids": [0, 1]})()
        right_asset_cfg = type("Cfg", (), {"name": "robot", "body_ids": [2, 3]})()

        penalty = swing_height_difference_below_min(
            env,
            command_name="base_velocity",
            min_height_difference=0.03,
            left_sensor_cfg=left_sensor_cfg,
            right_sensor_cfg=right_sensor_cfg,
            left_asset_cfg=left_asset_cfg,
            right_asset_cfg=right_asset_cfg,
        )

        self.assertTrue(torch.allclose(penalty, torch.tensor([0.02]), atol=1.0e-5))


@unittest.skipUnless(TORCH_AVAILABLE, "torch is not available in the system test interpreter")
class ContactBodyAlignmentPenaltyTests(unittest.TestCase):
    def _make_env(
        self,
        *,
        body_quat_w: torch.Tensor,
        net_forces_w_history: torch.Tensor,
    ):
        env = type("Env", (), {})()
        sensor = type("Sensor", (), {})()
        sensor.data = type("Data", (), {"net_forces_w_history": net_forces_w_history})()
        robot = type("Robot", (), {})()
        robot.data = type("Data", (), {"body_quat_w": body_quat_w})()

        class Scene(dict):
            def __init__(self):
                super().__init__({"robot": robot})
                self.sensors = {"contact_forces": sensor}

        env.scene = Scene()
        return env

    def test_penalizes_contacted_body_that_is_not_flat(self) -> None:
        quarter_roll = torch.tensor(
            [[[0.70710678, 0.70710678, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]],
            dtype=torch.float32,
        )
        net_forces = torch.tensor(
            [[[[10.0, 0.0, 0.0], [10.0, 0.0, 0.0]]]],
            dtype=torch.float32,
        )
        env = self._make_env(body_quat_w=quarter_roll, net_forces_w_history=net_forces)
        sensor_cfg = type("Cfg", (), {"name": "contact_forces", "body_ids": [0, 1]})()
        asset_cfg = type("Cfg", (), {"name": "robot", "body_ids": [0, 1]})()

        penalty = contact_body_alignment_penalty(
            env,
            min_cosine=0.8,
            contact_threshold=5.0,
            sensor_cfg=sensor_cfg,
            asset_cfg=asset_cfg,
            local_reference_vectors=((0.0, 0.0, 1.0), (0.0, 0.0, 1.0)),
        )

        self.assertTrue(torch.allclose(penalty, torch.tensor([0.4]), atol=1.0e-5))


@unittest.skipUnless(TORCH_AVAILABLE, "torch is not available in the system test interpreter")
class SecondaryContactWithoutPrimaryPenaltyTests(unittest.TestCase):
    def test_penalizes_toe_only_support(self) -> None:
        env = type("Env", (), {})()
        sensor = type("Sensor", (), {})()
        sensor.data = type(
            "Data",
            (),
            {
                "net_forces_w_history": torch.tensor(
                    [[[[0.0, 0.0, 0.0], [0.0, 0.0, 8.0], [0.0, 0.0, 9.0], [0.0, 0.0, 0.0]]]],
                    dtype=torch.float32,
                )
            },
        )()
        env.scene = type("Scene", (), {"sensors": {"contact_forces": sensor}})()

        left_primary_cfg = type("Cfg", (), {"name": "contact_forces", "body_ids": [0]})()
        right_primary_cfg = type("Cfg", (), {"name": "contact_forces", "body_ids": [2]})()
        left_aux_cfg = type("Cfg", (), {"name": "contact_forces", "body_ids": [1]})()
        right_aux_cfg = type("Cfg", (), {"name": "contact_forces", "body_ids": [3]})()

        penalty = secondary_contact_without_primary_penalty(
            env,
            threshold=5.0,
            left_primary_sensor_cfg=left_primary_cfg,
            right_primary_sensor_cfg=right_primary_cfg,
            left_aux_sensor_cfg=left_aux_cfg,
            right_aux_sensor_cfg=right_aux_cfg,
        )

        self.assertTrue(torch.allclose(penalty, torch.tensor([1.0]), atol=1.0e-5))


@unittest.skipUnless(TORCH_AVAILABLE, "torch is not available in the system test interpreter")
class SecondaryContactForceSharePenaltyTests(unittest.TestCase):
    def test_penalizes_toe_dominant_support_share(self) -> None:
        env = type("Env", (), {})()
        env.command_manager = type(
            "CommandManager",
            (),
            {
                "get_command": lambda self, name: torch.tensor(
                    [[0.5, 0.0, 0.0]], dtype=torch.float32
                )
            },
        )()
        sensor = type("Sensor", (), {})()
        sensor.data = type(
            "Data",
            (),
            {
                "net_forces_w_history": torch.tensor(
                    [[[[0.0, 0.0, 2.0], [0.0, 0.0, 8.0], [0.0, 0.0, 9.0], [0.0, 0.0, 1.0]]]],
                    dtype=torch.float32,
                )
            },
        )()
        env.scene = type("Scene", (), {"sensors": {"contact_forces": sensor}})()

        left_primary_cfg = type("Cfg", (), {"name": "contact_forces", "body_ids": [0]})()
        right_primary_cfg = type("Cfg", (), {"name": "contact_forces", "body_ids": [2]})()
        left_aux_cfg = type("Cfg", (), {"name": "contact_forces", "body_ids": [1]})()
        right_aux_cfg = type("Cfg", (), {"name": "contact_forces", "body_ids": [3]})()

        penalty = secondary_contact_force_share_penalty(
            env,
            command_name="base_velocity",
            threshold=1.0,
            min_primary_force_share=0.75,
            left_primary_sensor_cfg=left_primary_cfg,
            right_primary_sensor_cfg=right_primary_cfg,
            left_aux_sensor_cfg=left_aux_cfg,
            right_aux_sensor_cfg=right_aux_cfg,
        )

        self.assertTrue(torch.allclose(penalty, torch.tensor([0.55]), atol=1.0e-5))


@unittest.skipUnless(TORCH_AVAILABLE, "torch is not available in the system test interpreter")
class CommandAwareRootPlanarSpeedPenaltyTests(unittest.TestCase):
    def test_penalizes_root_speed_only_for_idle_commands(self) -> None:
        env = type("Env", (), {})()
        env.command_manager = type(
            "CommandManager",
            (),
            {
                "get_command": lambda self, name: torch.tensor(
                    [[0.0, 0.0, 0.0], [0.3, 0.0, 0.0]], dtype=torch.float32
                )
            },
        )()
        env.scene = {
            "robot": type(
                "Robot",
                (),
                {
                    "data": type(
                        "Data",
                        (),
                        {
                            "root_lin_vel_w": torch.tensor(
                                [[0.1, 0.2, 0.0], [0.4, 0.0, 0.0]],
                                dtype=torch.float32,
                            )
                        },
                    )()
                },
            )()
        }
        asset_cfg = type("Cfg", (), {"name": "robot"})()

        penalty = command_aware_root_planar_speed_penalty(
            env,
            command_name="base_velocity",
            max_command_speed=0.05,
            asset_cfg=asset_cfg,
        )

        self.assertTrue(torch.allclose(penalty, torch.tensor([0.2236068, 0.0]), atol=1.0e-5))


@unittest.skipUnless(TORCH_AVAILABLE, "torch is not available in the system test interpreter")
class SwingFootAheadOfStanceRewardTests(unittest.TestCase):
    def _make_env(
        self,
        *,
        command_xy: tuple[float, float],
        current_contact_time: torch.Tensor,
        root_quat_w: torch.Tensor,
        body_pos_w: torch.Tensor,
    ):
        env = type("Env", (), {})()
        env.command_manager = type(
            "CommandManager",
            (),
            {
                "get_command": lambda self, name: torch.tensor(
                    [[command_xy[0], command_xy[1], 0.0]], dtype=torch.float32
                )
            },
        )()
        sensor = type("Sensor", (), {})()
        sensor.data = type("Data", (), {"current_contact_time": current_contact_time})()
        robot = type("Robot", (), {})()
        robot.data = type("Data", (), {"root_quat_w": root_quat_w, "body_pos_w": body_pos_w})()

        class Scene(dict):
            def __init__(self):
                super().__init__({"robot": robot})
                self.sensors = {"contact_forces": sensor}

        env.scene = Scene()
        return env

    def test_rewards_swing_foot_ahead_in_single_support(self) -> None:
        env = self._make_env(
            command_xy=(0.5, 0.0),
            current_contact_time=torch.tensor([[0.0, 0.2]], dtype=torch.float32),
            root_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
            body_pos_w=torch.tensor(
                [[[0.0, 0.12, 0.0], [0.0, 0.0, 0.0]]],
                dtype=torch.float32,
            ),
        )
        left_sensor_cfg = type("Cfg", (), {"name": "contact_forces", "body_ids": [0]})()
        right_sensor_cfg = type("Cfg", (), {"name": "contact_forces", "body_ids": [1]})()
        left_asset_cfg = type("Cfg", (), {"name": "robot", "body_ids": [0]})()
        right_asset_cfg = type("Cfg", (), {"name": "robot", "body_ids": [1]})()

        reward = swing_foot_ahead_of_stance_reward(
            env,
            command_name="base_velocity",
            min_step_length=0.03,
            max_rewarded_step_length=0.1,
            forward_body_axis="y",
            left_sensor_cfg=left_sensor_cfg,
            right_sensor_cfg=right_sensor_cfg,
            left_asset_cfg=left_asset_cfg,
            right_asset_cfg=right_asset_cfg,
        )

        self.assertTrue(torch.allclose(reward, torch.tensor([0.09]), atol=1.0e-5))

    def test_returns_zero_outside_single_support(self) -> None:
        env = self._make_env(
            command_xy=(0.5, 0.0),
            current_contact_time=torch.tensor([[0.2, 0.2]], dtype=torch.float32),
            root_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
            body_pos_w=torch.tensor(
                [[[0.0, 0.12, 0.0], [0.0, 0.0, 0.0]]],
                dtype=torch.float32,
            ),
        )
        left_sensor_cfg = type("Cfg", (), {"name": "contact_forces", "body_ids": [0]})()
        right_sensor_cfg = type("Cfg", (), {"name": "contact_forces", "body_ids": [1]})()
        left_asset_cfg = type("Cfg", (), {"name": "robot", "body_ids": [0]})()
        right_asset_cfg = type("Cfg", (), {"name": "robot", "body_ids": [1]})()

        reward = swing_foot_ahead_of_stance_reward(
            env,
            command_name="base_velocity",
            min_step_length=0.03,
            max_rewarded_step_length=0.1,
            forward_body_axis="y",
            left_sensor_cfg=left_sensor_cfg,
            right_sensor_cfg=right_sensor_cfg,
            left_asset_cfg=left_asset_cfg,
            right_asset_cfg=right_asset_cfg,
        )

        self.assertTrue(torch.allclose(reward, torch.tensor([0.0]), atol=1.0e-5))


@unittest.skipUnless(TORCH_AVAILABLE, "torch is not available in the system test interpreter")
class BodyHeightBelowMinTests(unittest.TestCase):
    def test_penalizes_only_height_below_floor(self) -> None:
        env = type("Env", (), {})()
        env.scene = {
            "robot": type(
                "Robot",
                (),
                {
                    "data": type(
                        "Data",
                        (),
                        {
                            "body_pos_w": torch.tensor(
                                [
                                    [[0.0, 0.0, 0.30], [0.0, 0.0, 0.18]],
                                    [[0.0, 0.0, 0.12], [0.0, 0.0, 0.22]],
                                ],
                                dtype=torch.float32,
                            )
                        },
                    )()
                },
            )()
        }
        asset_cfg = type("Cfg", (), {"name": "robot", "body_ids": [0, 1]})()

        penalty = body_height_below_min(env, min_height=0.2, asset_cfg=asset_cfg)

        self.assertTrue(torch.allclose(penalty, torch.tensor([0.02, 0.08])))


if __name__ == "__main__":
    unittest.main()
