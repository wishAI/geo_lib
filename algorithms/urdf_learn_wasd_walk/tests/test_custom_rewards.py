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
        body_height_below_min,
        grouped_support_first_contact_reward,
        grouped_support_mode_time,
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

        self.assertTrue(torch.allclose(reward, torch.tensor([0.3, 0.0])))

    def test_uses_longest_air_time_in_group(self) -> None:
        first_contact = torch.tensor([[False, True]], dtype=torch.bool)
        last_air_time = torch.tensor([[0.45, 0.8]], dtype=torch.float32)

        reward = grouped_support_first_contact_reward(first_contact, last_air_time, threshold=0.5)

        self.assertTrue(torch.allclose(reward, torch.tensor([0.3])))


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
