from __future__ import annotations

import math

import numpy as np

from algorithms.ikfast_urdf_solver.solver import joint_distance, pose_error


def test_joint_distance_wraps_continuous_joints() -> None:
    candidate = np.array([math.pi - 0.01, 1.0], dtype=np.float64)
    reference = np.array([-math.pi + 0.01, 1.0], dtype=np.float64)
    distance = joint_distance(candidate, reference, np.array([True, False]))
    assert distance < 0.05


def test_pose_error_detects_translation_and_rotation() -> None:
    target_position = np.zeros(3, dtype=np.float64)
    target_rotation = np.eye(3, dtype=np.float64)
    actual_position = np.array([0.1, 0.0, 0.0], dtype=np.float64)
    actual_rotation = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    position_error, rotation_error = pose_error(
        target_position,
        target_rotation,
        actual_position,
        actual_rotation,
    )
    assert math.isclose(position_error, 0.1, rel_tol=0.0, abs_tol=1e-9)
    assert math.isclose(rotation_error, math.pi / 2.0, rel_tol=0.0, abs_tol=1e-9)
