from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT.parent.parent) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT.parent.parent))

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
if TORCH_AVAILABLE:
    import torch

    from algorithms.urdf_learn_wasd_walk.landau_rewards import (
        body_height_below_min_termination,
        control_root_height_floor,
        non_support_contacts_count,
        obstacle_brake_penalty,
    )


@unittest.skipUnless(TORCH_AVAILABLE, "torch is not available in the system test interpreter")
class ControlRootHeightFloorTests(unittest.TestCase):
    def test_penalizes_height_below_support_floor(self) -> None:
        robot = type("Robot", (), {})()
        robot.data = type(
            "Data",
            (),
            {
                "body_pos_w": torch.tensor(
                    [
                        [[0.0, 0.0, 0.30], [0.0, 0.0, 0.10], [0.0, 0.0, 0.12]],
                        [[0.0, 0.0, 0.26], [0.0, 0.0, 0.10], [0.0, 0.0, 0.11]],
                    ],
                    dtype=torch.float32,
                ),
            },
        )()
        env = type("Env", (), {"scene": {"robot": robot}})()
        root_cfg = type("Cfg", (), {"name": "robot", "body_ids": [0]})()
        support_cfg = type("Cfg", (), {"name": "robot", "body_ids": [1, 2]})()

        penalty = control_root_height_floor(
            env,
            min_height=0.18,
            asset_cfg=root_cfg,
            reference_asset_cfg=support_cfg,
        )

        self.assertTrue(torch.allclose(penalty, torch.tensor([0.0, 0.02])))

    def test_termination_matches_height_floor(self) -> None:
        robot = type("Robot", (), {})()
        robot.data = type("Data", (), {"body_pos_w": torch.tensor([[[0.0, 0.0, 0.15]]], dtype=torch.float32)})()
        env = type("Env", (), {"scene": {"robot": robot}})()
        cfg = type("Cfg", (), {"name": "robot", "body_ids": [0]})()

        terminated = body_height_below_min_termination(env, min_height=0.16, asset_cfg=cfg)

        self.assertTrue(torch.equal(terminated, torch.tensor([True])))


@unittest.skipUnless(TORCH_AVAILABLE, "torch is not available in the system test interpreter")
class NonSupportContactsCountTests(unittest.TestCase):
    def test_counts_bodies_over_contact_threshold(self) -> None:
        sensor = type("Sensor", (), {})()
        sensor.data = type(
            "Data",
            (),
            {
                "net_forces_w_history": torch.tensor(
                    [[[[0.0, 0.0, 6.0], [0.0, 0.0, 0.5], [0.0, 0.0, 7.0]]]],
                    dtype=torch.float32,
                )
            },
        )()
        env = type("Env", (), {"scene": type("Scene", (), {"sensors": {"contact_forces": sensor}})()})()
        cfg = type("Cfg", (), {"name": "contact_forces", "body_ids": [0, 1, 2]})()

        penalty = non_support_contacts_count(env, threshold=5.0, sensor_cfg=cfg)

        self.assertTrue(torch.allclose(penalty, torch.tensor([2.0])))


@unittest.skipUnless(TORCH_AVAILABLE, "torch is not available in the system test interpreter")
class ObstacleBrakePenaltyTests(unittest.TestCase):
    def test_penalizes_forward_speed_when_obstacle_is_close_ahead(self) -> None:
        sensor = type("Sensor", (), {})()
        sensor.data = type(
            "Data",
            (),
            {
                "ray_hits_w": torch.tensor(
                    [[[0.10, 0.55, 0.35], [0.20, 0.80, 0.30], [1.50, 0.20, -0.10]]],
                    dtype=torch.float32,
                )
            },
        )()
        robot = type("Robot", (), {})()
        robot.data = type(
            "Data",
            (),
            {
                "body_pos_w": torch.tensor([[[0.0, 0.0, 0.8]]], dtype=torch.float32),
                "body_quat_w": torch.tensor([[[1.0, 0.0, 0.0, 0.0]]], dtype=torch.float32),
                "root_lin_vel_b": torch.tensor([[0.0, 0.4, 0.0]], dtype=torch.float32),
            },
        )()
        scene = type("Scene", (), {"sensors": {"height_scanner": sensor}, "__getitem__": lambda self, key: robot})()
        env = type("Env", (), {"scene": scene})()
        sensor_cfg = type("Cfg", (), {"name": "height_scanner"})()
        root_cfg = type("Cfg", (), {"name": "robot", "body_ids": [0]})()

        penalty = obstacle_brake_penalty(
            env,
            sensor_cfg=sensor_cfg,
            asset_cfg=root_cfg,
            forward_body_axis="y",
        )

        self.assertTrue(torch.allclose(penalty, torch.tensor([0.4])))


if __name__ == "__main__":
    unittest.main()
