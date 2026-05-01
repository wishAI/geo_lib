from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.managers import SceneEntityCfg


def _rotate_body_vectors(quat_wxyz: torch.Tensor, vectors: torch.Tensor) -> torch.Tensor:
    quat_vec = quat_wxyz[..., 1:]
    uv = torch.cross(quat_vec, vectors, dim=-1)
    uuv = torch.cross(quat_vec, uv, dim=-1)
    return vectors + 2.0 * (quat_wxyz[..., :1] * uv + uuv)


def _rotate_world_vectors_into_body(quat_wxyz: torch.Tensor, vectors: torch.Tensor) -> torch.Tensor:
    quat_conj = torch.cat((quat_wxyz[..., :1], -quat_wxyz[..., 1:]), dim=-1)
    return _rotate_body_vectors(quat_conj, vectors)


def control_root_height_floor(
    env,
    min_height: float,
    asset_cfg: "SceneEntityCfg",
    reference_asset_cfg: "SceneEntityCfg" | None = None,
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    body_height = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
    min_body_height = torch.min(body_height, dim=1).values
    if reference_asset_cfg is not None:
        reference_asset = env.scene[reference_asset_cfg.name]
        reference_height = reference_asset.data.body_pos_w[:, reference_asset_cfg.body_ids, 2]
        min_body_height = min_body_height - torch.min(reference_height, dim=1).values
    return torch.clamp(min_height - min_body_height, min=0.0)


def body_height_below_min_termination(
    env,
    min_height: float,
    asset_cfg: "SceneEntityCfg",
    reference_asset_cfg: "SceneEntityCfg" | None = None,
) -> torch.Tensor:
    return control_root_height_floor(
        env,
        min_height=min_height,
        asset_cfg=asset_cfg,
        reference_asset_cfg=reference_asset_cfg,
    ) > 0.0


def non_support_contacts_count(
    env,
    threshold: float,
    sensor_cfg: "SceneEntityCfg",
) -> torch.Tensor:
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    net_forces = contact_sensor.data.net_forces_w_history[:, 0, sensor_cfg.body_ids]
    contact_mask = torch.linalg.norm(net_forces, dim=-1) > threshold
    return torch.sum(contact_mask.to(torch.float32), dim=1)


def obstacle_brake_penalty(
    env,
    sensor_cfg: "SceneEntityCfg",
    asset_cfg: "SceneEntityCfg",
    forward_body_axis: str = "y",
    min_forward_distance: float = 0.25,
    max_forward_distance: float = 1.1,
    lateral_window: float = 0.35,
    min_obstacle_local_z: float = -0.65,
) -> torch.Tensor:
    sensor = env.scene.sensors[sensor_cfg.name]
    asset = env.scene[asset_cfg.name]

    root_pos = asset.data.body_pos_w[:, asset_cfg.body_ids[0], :3]
    root_quat = asset.data.body_quat_w[:, asset_cfg.body_ids[0]]
    hit_positions = sensor.data.ray_hits_w
    hit_offsets_w = hit_positions - root_pos.unsqueeze(1)
    hit_offsets_b = _rotate_world_vectors_into_body(root_quat.unsqueeze(1), hit_offsets_w)

    forward_axis = 1 if forward_body_axis == "y" else 0
    lateral_axis = 0 if forward_body_axis == "y" else 1
    finite_mask = torch.isfinite(hit_offsets_b).all(dim=-1)
    forward_mask = (hit_offsets_b[..., forward_axis] >= min_forward_distance) & (
        hit_offsets_b[..., forward_axis] <= max_forward_distance
    )
    lateral_mask = torch.abs(hit_offsets_b[..., lateral_axis]) <= lateral_window
    obstacle_height_mask = hit_offsets_b[..., 2] >= min_obstacle_local_z
    obstacle_ahead = torch.any(finite_mask & forward_mask & lateral_mask & obstacle_height_mask, dim=1)

    root_forward_speed = torch.clamp(asset.data.root_lin_vel_b[:, forward_axis], min=0.0)
    return root_forward_speed * obstacle_ahead.to(root_forward_speed.dtype)
