from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .config import CameraConfig


@dataclass
class CameraPose:
    name: str
    position_world: np.ndarray
    rotation_world_from_camera_mj: np.ndarray
    rotation_world_from_camera_cv: np.ndarray

    @property
    def transform_world_from_camera_cv(self) -> np.ndarray:
        transform = np.eye(4)
        transform[:3, :3] = self.rotation_world_from_camera_cv
        transform[:3, 3] = self.position_world
        return transform


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm < 1e-12:
        raise ValueError("Cannot normalize near-zero vector")
    return vec / norm


def look_at_rotation_world_from_camera_mj(
    camera_pos_world: np.ndarray,
    target_world: np.ndarray,
    world_up: np.ndarray = np.array([0.0, 0.0, 1.0]),
) -> np.ndarray:
    forward = _normalize(target_world - camera_pos_world)
    z_axis = -forward
    x_axis = np.cross(world_up, z_axis)
    if np.linalg.norm(x_axis) < 1e-8:
        fallback_up = np.array([0.0, 1.0, 0.0])
        x_axis = np.cross(fallback_up, z_axis)
    x_axis = _normalize(x_axis)
    y_axis = _normalize(np.cross(z_axis, x_axis))
    return np.column_stack((x_axis, y_axis, z_axis))


def sample_cameras(camera_cfg: CameraConfig, target_world: np.ndarray) -> list[CameraPose]:
    num_views = int(camera_cfg.num_views)
    if not 2 <= num_views <= 3:
        raise ValueError("num_views must be in [2, 3]")

    azimuths_deg = [camera_cfg.first_view_deg + i * camera_cfg.view_step_deg for i in range(num_views)]

    cameras: list[CameraPose] = []
    for idx, azimuth_deg in enumerate(azimuths_deg):
        azimuth = math.radians(azimuth_deg)
        r = camera_cfg.ring_radius_m
        position = np.array(
            [
                target_world[0] + r * math.cos(azimuth),
                target_world[1] + r * math.sin(azimuth),
                target_world[2] + camera_cfg.ring_height_m,
            ],
            dtype=float,
        )
        rotation_mj = look_at_rotation_world_from_camera_mj(position, target_world)

        # Convert MuJoCo camera frame (x right, y up, -z forward) to CV-style
        # camera frame (x right, y down, z forward).
        rotation_cv = rotation_mj @ np.diag([1.0, -1.0, -1.0])

        cameras.append(
            CameraPose(
                name=f"view_{idx:03d}",
                position_world=position,
                rotation_world_from_camera_mj=rotation_mj,
                rotation_world_from_camera_cv=rotation_cv,
            )
        )

    return cameras
