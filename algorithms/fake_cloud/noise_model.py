from __future__ import annotations

import math

import numpy as np

from .config import NoiseConfig


def apply_depth_noise(
    depth_m: np.ndarray,
    noise_cfg: NoiseConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    if not noise_cfg.enable_depth_noise:
        return depth_m.copy()

    sigma = noise_cfg.depth_noise_a_m + noise_cfg.depth_noise_b_m * np.square(depth_m)
    jitter = rng.normal(0.0, 1.0, size=depth_m.shape) * sigma
    noisy = depth_m + jitter
    noisy = np.where(np.isfinite(depth_m), noisy, np.nan)
    return np.clip(noisy, 1e-4, None)


def _rodrigues(rotation_vec: np.ndarray) -> np.ndarray:
    theta = np.linalg.norm(rotation_vec)
    if theta < 1e-12:
        return np.eye(3)
    axis = rotation_vec / theta
    x, y, z = axis
    k = np.array(
        [
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0],
        ]
    )
    return np.eye(3) + math.sin(theta) * k + (1.0 - math.cos(theta)) * (k @ k)


def perturb_camera_pose(
    rotation_world_from_camera: np.ndarray,
    position_world: np.ndarray,
    noise_cfg: NoiseConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    if not noise_cfg.enable_pose_noise:
        return rotation_world_from_camera.copy(), position_world.copy()

    translation_noise = rng.normal(
        loc=0.0, scale=noise_cfg.pose_translation_sigma_m, size=3
    )
    rot_sigma = math.radians(noise_cfg.pose_rotation_sigma_deg)
    small_angle_vec = rng.normal(loc=0.0, scale=rot_sigma, size=3)
    rotation_noise = _rodrigues(small_angle_vec)

    perturbed_rotation = rotation_world_from_camera @ rotation_noise
    perturbed_position = position_world + translation_noise
    return perturbed_rotation, perturbed_position
