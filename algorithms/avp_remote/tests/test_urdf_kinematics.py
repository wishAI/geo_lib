import math
import sys
import unittest
from pathlib import Path

import numpy as np


MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

import landau_pose
import urdf_kinematics


def _rot_x(angle_rad: float) -> np.ndarray:
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, c, -s],
            [0.0, s, c],
        ],
        dtype=float,
    )


def _rot_y(angle_rad: float) -> np.ndarray:
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return np.array(
        [
            [c, 0.0, s],
            [0.0, 1.0, 0.0],
            [-s, 0.0, c],
        ],
        dtype=float,
    )


def _rot_z(angle_rad: float) -> np.ndarray:
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


class TestUrdfKinematics(unittest.TestCase):
    def test_rpy_matrix_matches_urdf_convention(self):
        roll = 2.748947
        pitch = -1.256079
        yaw = 0.394193
        expected = _rot_z(yaw) @ _rot_y(pitch) @ _rot_x(roll)

        np.testing.assert_allclose(urdf_kinematics.rpy_matrix(roll, pitch, yaw), expected, atol=1.0e-9)
        np.testing.assert_allclose(landau_pose.rpy_matrix(roll, pitch, yaw), expected, atol=1.0e-9)

    def test_left_hip_pitch_origin_uses_urdf_rpy_order(self):
        urdf_path = MODULE_ROOT / "inputs" / "landau_v10" / "landau_v10_parallel_mesh.urdf"
        specs = urdf_kinematics.load_urdf_joint_specs(urdf_path)
        left_hip_pitch = specs["left_hip_pitch_joint"]

        expected_row0 = np.array([0.28580513040669403, 0.01889961027516769, 0.9581013684181123, 0.059407])
        np.testing.assert_allclose(left_hip_pitch.origin[0], expected_row0, atol=5.0e-6)


if __name__ == "__main__":
    unittest.main()
