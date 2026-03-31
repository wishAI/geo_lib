from __future__ import annotations

import unittest

from algorithms.urdf_learn_wasd_walk.command_frame import semantic_command_to_env_command, semantic_forward_dir_xy


class CommandFrameTests(unittest.TestCase):
    def test_g1_commands_remain_identity(self) -> None:
        self.assertEqual(semantic_command_to_env_command("g1", (0.5, -0.25, 0.1)), (0.5, -0.25, 0.1))

    def test_landau_semantic_command_remaps_visual_forward_to_env_frame(self) -> None:
        self.assertEqual(semantic_command_to_env_command("landau", (0.5, -0.25, 0.1)), (-0.25, 0.5, 0.1))

    def test_landau_forward_dir_uses_body_y_axis(self) -> None:
        self.assertEqual(semantic_forward_dir_xy("landau", (1.0, 0.0, 0.0, 0.0)), (0.0, 1.0))


if __name__ == "__main__":
    unittest.main()
