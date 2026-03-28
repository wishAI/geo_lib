import math
from dataclasses import dataclass

import numpy as np


def _rot_x_deg(deg):
    rad = math.radians(float(deg))
    c, s = math.cos(rad), math.sin(rad)
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, c, -s],
            [0.0, s, c],
        ],
        dtype=float,
    )


def _rot_y_deg(deg):
    rad = math.radians(float(deg))
    c, s = math.cos(rad), math.sin(rad)
    return np.array(
        [
            [c, 0.0, s],
            [0.0, 1.0, 0.0],
            [-s, 0.0, c],
        ],
        dtype=float,
    )


def _rot_z_deg(deg):
    rad = math.radians(float(deg))
    c, s = math.cos(rad), math.sin(rad)
    return np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def build_xyz_transform(rotate_xyz_deg, translate_m, scale_xyz=(1.0, 1.0, 1.0)):
    """Build a 4x4 transform from XYZ Euler degrees, translation, and XYZ scale."""
    x_deg, y_deg, z_deg = rotate_xyz_deg
    tx, ty, tz = translate_m
    sx, sy, sz = scale_xyz

    mat = np.eye(4, dtype=float)
    rot = _rot_x_deg(x_deg) @ _rot_y_deg(y_deg) @ _rot_z_deg(z_deg)
    scale = np.diag([float(sx), float(sy), float(sz)])
    mat[:3, :3] = rot @ scale
    # mat[:3, 3] = [float(tx), float(ty), float(tz)]
    mat[3, :3] = [float(tx), float(ty), float(tz)]
    return mat


@dataclass(frozen=True)
class TransformOptions:
    column_major: bool = True
    pretransform: np.ndarray | None = None
    posttransform: np.ndarray | None = None


def to_usd_world(mat, options: TransformOptions):
    if mat is None:
        return None

    mat_world = np.asarray(mat, dtype=float)
    if mat_world.shape != (4, 4):
        raise ValueError(f"Expected 4x4 matrix, got shape {mat_world.shape}")

    if options.column_major:
        mat_world = mat_world.T
    if options.pretransform is not None:
        mat_world = np.asarray(options.pretransform, dtype=float) @ mat_world
    if options.posttransform is not None:
        mat_world = mat_world @ np.asarray(options.posttransform, dtype=float)
    return mat_world
