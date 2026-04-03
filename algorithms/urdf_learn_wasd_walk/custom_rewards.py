from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.managers import SceneEntityCfg


def grouped_support_mode_time(
    contact_time: torch.Tensor,
    air_time: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reduce multiple support-link timers into a single side-level stance/swing timer."""

    in_contact = torch.any(contact_time > 0.0, dim=1)
    contact_mode_time = torch.max(contact_time, dim=1).values
    air_mode_time = torch.min(air_time, dim=1).values
    mode_time = torch.where(in_contact, contact_mode_time, air_mode_time)
    return in_contact, mode_time


def grouped_support_first_contact_reward(
    first_contact: torch.Tensor,
    last_air_time: torch.Tensor,
    threshold: float,
) -> torch.Tensor:
    """Reward a grouped support side only when it lands after sufficient air time."""

    group_first_contact = torch.any(first_contact, dim=1)
    group_last_air_time = torch.max(last_air_time, dim=1).values
    return torch.clamp(group_last_air_time - threshold, min=0.0) * group_first_contact


def grouped_support_air_time_positive_biped(
    env,
    command_name: str,
    threshold: float,
    left_sensor_cfg: "SceneEntityCfg",
    right_sensor_cfg: "SceneEntityCfg",
) -> torch.Tensor:
    """Reward biped stepping using grouped support links per side.

    Landau exposes both `foot_*` and `toes_*` contacts per side. The stock Isaac helper assumes one
    contact body per side, so it never sees a clean left/right stance signal on this robot.
    """

    contact_sensor = env.scene.sensors[left_sensor_cfg.name]

    left_contact_time = contact_sensor.data.current_contact_time[:, left_sensor_cfg.body_ids]
    left_air_time = contact_sensor.data.current_air_time[:, left_sensor_cfg.body_ids]
    right_contact_time = contact_sensor.data.current_contact_time[:, right_sensor_cfg.body_ids]
    right_air_time = contact_sensor.data.current_air_time[:, right_sensor_cfg.body_ids]

    left_in_contact, left_mode_time = grouped_support_mode_time(left_contact_time, left_air_time)
    right_in_contact, right_mode_time = grouped_support_mode_time(right_contact_time, right_air_time)

    single_stance = torch.logical_xor(left_in_contact, right_in_contact)
    reward = torch.minimum(left_mode_time, right_mode_time)
    reward = torch.where(single_stance, reward, torch.zeros_like(reward))
    reward = torch.clamp(reward, max=threshold)
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def grouped_support_first_contact_biped(
    env,
    command_name: str,
    threshold: float,
    left_sensor_cfg: "SceneEntityCfg",
    right_sensor_cfg: "SceneEntityCfg",
) -> torch.Tensor:
    """Reward real left/right landings for grouped support links on a biped."""

    contact_sensor = env.scene.sensors[left_sensor_cfg.name]
    first_contact = contact_sensor.compute_first_contact(env.step_dt)

    left_reward = grouped_support_first_contact_reward(
        first_contact[:, left_sensor_cfg.body_ids],
        contact_sensor.data.last_air_time[:, left_sensor_cfg.body_ids],
        threshold,
    )
    right_reward = grouped_support_first_contact_reward(
        first_contact[:, right_sensor_cfg.body_ids],
        contact_sensor.data.last_air_time[:, right_sensor_cfg.body_ids],
        threshold,
    )

    reward = left_reward + right_reward
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def body_height_below_min(
    env,
    min_height: float,
    asset_cfg: "SceneEntityCfg",
) -> torch.Tensor:
    """Penalize configured bodies when they sink below a minimum world-space height."""

    asset = env.scene[asset_cfg.name]
    body_height = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
    min_body_height = torch.min(body_height, dim=1).values
    return torch.clamp(min_height - min_body_height, min=0.0)
