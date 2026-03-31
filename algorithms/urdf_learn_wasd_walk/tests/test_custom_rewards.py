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

    from algorithms.urdf_learn_wasd_walk.custom_rewards import grouped_support_mode_time


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


if __name__ == "__main__":
    unittest.main()
