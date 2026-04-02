from __future__ import annotations

import unittest

from algorithms.urdf_learn_wasd_walk.command_frame import semantic_command_to_env_command, semantic_forward_dir_xy


class CommandFrameTests(unittest.TestCase):
    def test_x_axis_forward_commands_remain_identity(self) -> None:
        self.assertEqual(semantic_command_to_env_command("x", (0.5, -0.25, 0.1)), (0.5, -0.25, 0.1))

    def test_y_axis_forward_remaps_to_env_frame(self) -> None:
        self.assertEqual(semantic_command_to_env_command("y", (0.5, -0.25, 0.1)), (-0.25, 0.5, 0.1))

    def test_y_axis_forward_dir_uses_body_y(self) -> None:
        self.assertEqual(semantic_forward_dir_xy("y", (1.0, 0.0, 0.0, 0.0)), (0.0, 1.0))

    def test_x_axis_forward_dir_uses_body_x(self) -> None:
        self.assertAlmostEqual(semantic_forward_dir_xy("x", (1.0, 0.0, 0.0, 0.0))[0], 1.0)
        self.assertAlmostEqual(semantic_forward_dir_xy("x", (1.0, 0.0, 0.0, 0.0))[1], 0.0)


if __name__ == "__main__":
    unittest.main()
