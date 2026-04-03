from __future__ import annotations

import math
from collections.abc import Sequence


def semantic_command_to_env_command(
    forward_body_axis: str, command: tuple[float, float, float]
) -> tuple[float, float, float]:
    """Map user-facing `(forward, strafe, yaw)` commands into the robot's native body frame.

    *forward_body_axis* is ``"x"`` or ``"y"`` and comes from robot metadata
    (``RobotTaskSpec.forward_body_axis``).
    """
    forward, strafe, yaw = command
    if forward_body_axis == "y":
        return (strafe, forward, yaw)
    return command


def semantic_forward_dir_xy(
    forward_body_axis: str, quat_wxyz: Sequence[float]
) -> tuple[float, float]:
    """Return the world-frame planar direction corresponding to user-facing forward motion."""

    w, x, y, z = (float(value) for value in quat_wxyz)
    body_x = (
        1.0 - 2.0 * (y * y + z * z),
        2.0 * (x * y + z * w),
    )
    body_y = (
        2.0 * (x * y - z * w),
        1.0 - 2.0 * (x * x + z * z),
    )
    if forward_body_axis == "y":
        forward = body_y
    else:
        forward = body_x
    norm = math.hypot(*forward)
    if norm < 1.0e-8:
        return (1.0, 0.0)
    return (forward[0] / norm, forward[1] / norm)
