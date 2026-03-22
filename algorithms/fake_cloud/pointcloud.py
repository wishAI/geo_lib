from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float


def intrinsics_from_fovy(width: int, height: int, fov_y_deg: float) -> CameraIntrinsics:
    fov_y = np.deg2rad(fov_y_deg)
    fy = height / (2.0 * np.tan(fov_y / 2.0))
    fx = fy
    cx = (width - 1.0) / 2.0
    cy = (height - 1.0) / 2.0
    return CameraIntrinsics(fx=fx, fy=fy, cx=cx, cy=cy)


def depth_to_points_camera(
    depth_m: np.ndarray,
    intr: CameraIntrinsics,
    depth_min_m: float,
    depth_max_m: float,
) -> np.ndarray:
    rows, cols = depth_m.shape
    v_coords, u_coords = np.indices((rows, cols))

    valid = np.isfinite(depth_m)
    valid &= depth_m >= depth_min_m
    valid &= depth_m <= depth_max_m

    z = depth_m
    # Project each valid depth pixel to camera-frame XYZ using pinhole geometry.
    x = (u_coords - intr.cx) * z / intr.fx
    y = (v_coords - intr.cy) * z / intr.fy

    points = np.stack((x, y, z), axis=-1)
    return points[valid]


def transform_points_world(
    points_camera: np.ndarray,
    rotation_world_from_camera: np.ndarray,
    position_world: np.ndarray,
) -> np.ndarray:
    return points_camera @ rotation_world_from_camera.T + position_world


def write_ply(path: str | Path, points_xyz: np.ndarray) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    with target.open("w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points_xyz)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")
        for point in points_xyz:
            f.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f}\n")


def read_ply(path: str | Path) -> np.ndarray:
    src = Path(path)
    with src.open("r", encoding="utf-8") as f:
        lines = f.readlines()

    vertex_count = 0
    data_start = 0
    for i, line in enumerate(lines):
        if line.startswith("element vertex"):
            vertex_count = int(line.strip().split()[2])
        if line.strip() == "end_header":
            data_start = i + 1
            break

    coords = []
    for line in lines[data_start : data_start + vertex_count]:
        if not line.strip():
            continue
        x_str, y_str, z_str = line.strip().split()[:3]
        coords.append((float(x_str), float(y_str), float(z_str)))
    return np.array(coords, dtype=float)
