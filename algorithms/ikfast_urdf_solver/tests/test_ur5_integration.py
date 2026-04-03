from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from algorithms.ikfast_urdf_solver import IkFastSolver, get_builtin_config
from algorithms.ikfast_urdf_solver.benchmark import MujocoChain
from algorithms.ikfast_urdf_solver.solver import pose_error


@pytest.mark.skipif(
    not Path("helper_repos/ikfastpy/ikfast61.cpp").exists() or not Path("algorithms/ikfast_urdf_solver/inputs/ur5/ur5_kinematics.urdf").exists(),
    reason="UR5 helper assets are not available.",
)
def test_ur5_helper_round_trip_matches_mujoco() -> None:
    solver = IkFastSolver(get_builtin_config("ur5_helper_sample"))
    chain = MujocoChain.from_solver(solver)
    q = np.array([0.2, -1.1, 1.0, -0.8, 0.6, 0.1], dtype=np.float64)
    target_position, target_rotation = chain.forward_kinematics(q)
    result = solver.ik(target_position, target_rotation, seed_joint_values=q)
    assert result.success
    assert result.joint_values is not None
    actual_position, actual_rotation = chain.forward_kinematics(result.joint_values)
    position_error, rotation_error = pose_error(
        target_position,
        target_rotation,
        actual_position,
        actual_rotation,
    )
    assert position_error < 1e-6
    assert rotation_error < 1e-6
