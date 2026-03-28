import math
import unittest

import numpy as np

import usd_utils


def _rot_x(deg):
    rad = math.radians(deg)
    c, s = math.cos(rad), math.sin(rad)
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, c, -s],
            [0.0, s, c],
        ],
        dtype=float,
    )


def _rot_y(deg):
    rad = math.radians(deg)
    c, s = math.cos(rad), math.sin(rad)
    return np.array(
        [
            [c, 0.0, s],
            [0.0, 1.0, 0.0],
            [-s, 0.0, c],
        ],
        dtype=float,
    )


def _rot_z(deg):
    rad = math.radians(deg)
    c, s = math.cos(rad), math.sin(rad)
    return np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def _compose_xyz(x_deg, y_deg, z_deg):
    # Matches usd_utils.mat4_to_euler_xyz convention:
    # R = Rz @ Ry @ Rx
    r = _rot_z(z_deg) @ _rot_y(y_deg) @ _rot_x(x_deg)
    mat = np.eye(4, dtype=float)
    mat[:3, :3] = r
    return mat


def _wrap_deg(deg):
    return (deg + 180.0) % 360.0 - 180.0


class TestUsdUtilsMath(unittest.TestCase):
    def test_mat4_to_euler_xyz_identity(self):
        mat = np.eye(4, dtype=float)
        euler = usd_utils.mat4_to_euler_xyz(mat)
        self.assertTrue(np.allclose(euler, [0.0, 0.0, 0.0], atol=1e-6))

    def test_mat4_to_euler_xyz_known_rotation(self):
        x_deg, y_deg, z_deg = 20.0, 30.0, -40.0
        mat = _compose_xyz(x_deg, y_deg, z_deg)
        euler = usd_utils.mat4_to_euler_xyz(mat)
        euler = np.array([_wrap_deg(v) for v in euler], dtype=float)
        expected = np.array([x_deg, y_deg, z_deg], dtype=float)
        self.assertTrue(np.allclose(euler, expected, atol=1e-6))

    def test_mat4_rotation_angle_deg_identity(self):
        mat = np.eye(4, dtype=float)
        angle = usd_utils.mat4_rotation_angle_deg(mat)
        self.assertAlmostEqual(angle, 0.0, places=6)

    def test_mat4_rotation_angle_deg_180(self):
        mat = _compose_xyz(180.0, 0.0, 0.0)
        angle = usd_utils.mat4_rotation_angle_deg(mat)
        self.assertAlmostEqual(angle, 180.0, places=6)

    def test_mat4_rotation_angle_deg_90(self):
        mat = _compose_xyz(0.0, 0.0, 90.0)
        angle = usd_utils.mat4_rotation_angle_deg(mat)
        self.assertAlmostEqual(angle, 90.0, places=6)


if __name__ == "__main__":
    unittest.main()
