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
    # Require the whole side to have been airborne, not just one body in the bundle.
    group_last_air_time = torch.min(last_air_time, dim=1).values
    return torch.clamp(group_last_air_time - threshold, min=0.0) * group_first_contact


def _normalized_planar_body_axes(quat_wxyz: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return unit-length planar body-frame axes expressed in world XY."""

    w, x, y, z = quat_wxyz.unbind(dim=1)
    body_x = torch.stack((1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + z * w)), dim=1)
    body_y = torch.stack((2.0 * (x * y - z * w), 1.0 - 2.0 * (x * x + z * z)), dim=1)
    body_x = body_x / torch.clamp(torch.linalg.norm(body_x, dim=1, keepdim=True), min=1.0e-8)
    body_y = body_y / torch.clamp(torch.linalg.norm(body_y, dim=1, keepdim=True), min=1.0e-8)
    return body_x, body_y


def _rotate_body_vectors(quat_wxyz: torch.Tensor, vectors: torch.Tensor) -> torch.Tensor:
    """Rotate body-frame vectors into world space for batched quaternion tensors."""

    quat_vec = quat_wxyz[..., 1:]
    uv = torch.cross(quat_vec, vectors, dim=-1)
    uuv = torch.cross(quat_vec, uv, dim=-1)
    return vectors + 2.0 * (quat_wxyz[..., :1] * uv + uuv)


def _rotate_world_vectors_into_body(quat_wxyz: torch.Tensor, vectors: torch.Tensor) -> torch.Tensor:
    """Rotate world-frame vectors into the corresponding body frame."""

    quat_conj = torch.cat((quat_wxyz[..., :1], -quat_wxyz[..., 1:]), dim=-1)
    return _rotate_body_vectors(quat_conj, vectors)


def _contact_mask_for_bodies(contact_sensor, body_ids: list[int], threshold: float) -> torch.Tensor:
    net_forces = contact_sensor.data.net_forces_w_history[:, 0, body_ids]
    return torch.linalg.norm(net_forces, dim=-1) > threshold


def _command_speed(env, command_name: str) -> torch.Tensor:
    return torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1)


def _command_speed_mask(env, command_name: str) -> torch.Tensor:
    return _command_speed(env, command_name) > 0.1


def _command_idle_mask(env, command_name: str, max_command_speed: float) -> torch.Tensor:
    return _command_speed(env, command_name) <= max_command_speed


def _adaptive_gait_period(
    command_speed: torch.Tensor,
    *,
    slow_speed: float,
    fast_speed: float,
    slow_period: float,
    fast_period: float,
) -> torch.Tensor:
    if fast_speed <= slow_speed:
        return torch.full_like(command_speed, slow_period)
    speed_alpha = torch.clamp((command_speed - slow_speed) / (fast_speed - slow_speed), min=0.0, max=1.0)
    return slow_period + speed_alpha * (fast_period - slow_period)


def _gait_phase(
    env,
    command_name: str,
    *,
    slow_speed: float,
    fast_speed: float,
    slow_period: float,
    fast_period: float,
    phase_offset: float = 0.0,
) -> torch.Tensor:
    command_speed = _command_speed(env, command_name)
    gait_period = _adaptive_gait_period(
        command_speed,
        slow_speed=slow_speed,
        fast_speed=fast_speed,
        slow_period=slow_period,
        fast_period=fast_period,
    )
    episode_length_buf = getattr(env, "episode_length_buf", None)
    if episode_length_buf is None:
        episode_length = torch.full_like(command_speed, float(getattr(env, "common_step_counter", 0.0)))
    else:
        episode_length = episode_length_buf.to(device=command_speed.device, dtype=command_speed.dtype)
    phase = torch.remainder(episode_length * env.step_dt / torch.clamp(gait_period, min=1.0e-6), 1.0)
    return torch.remainder(phase * (2.0 * torch.pi) + phase_offset, 2.0 * torch.pi)


def _support_centers_xy(asset, left_asset_cfg: "SceneEntityCfg", right_asset_cfg: "SceneEntityCfg") -> tuple[torch.Tensor, torch.Tensor]:
    left_center_xy = asset.data.body_pos_w[:, left_asset_cfg.body_ids, :2].mean(dim=1)
    right_center_xy = asset.data.body_pos_w[:, right_asset_cfg.body_ids, :2].mean(dim=1)
    return left_center_xy, right_center_xy


def _support_mean_heights(asset, left_asset_cfg: "SceneEntityCfg", right_asset_cfg: "SceneEntityCfg") -> tuple[torch.Tensor, torch.Tensor]:
    left_height = asset.data.body_pos_w[:, left_asset_cfg.body_ids, 2].mean(dim=1)
    right_height = asset.data.body_pos_w[:, right_asset_cfg.body_ids, 2].mean(dim=1)
    return left_height, right_height


def _scene_entity(scene, name: str):
    if hasattr(scene, "sensors") and name in scene.sensors:
        return scene.sensors[name]
    return scene[name]


def gait_phase_clock_observation(
    env,
    command_name: str,
    slow_speed: float,
    fast_speed: float,
    slow_period: float,
    fast_period: float,
) -> torch.Tensor:
    """Return a 2D gait clock that speeds up as the commanded speed increases."""

    phase = _gait_phase(
        env,
        command_name,
        slow_speed=slow_speed,
        fast_speed=fast_speed,
        slow_period=slow_period,
        fast_period=fast_period,
    )
    return torch.stack((torch.sin(phase), torch.cos(phase)), dim=1)


def feet_positions_in_root_frame(
    env,
    root_asset_cfg: "SceneEntityCfg",
    left_asset_cfg: "SceneEntityCfg",
    right_asset_cfg: "SceneEntityCfg",
) -> torch.Tensor:
    """Return left/right foot centers expressed in the configured root body frame."""

    asset = env.scene[left_asset_cfg.name]
    root_pos = asset.data.body_pos_w[:, root_asset_cfg.body_ids, :3].mean(dim=1)
    root_quat = asset.data.body_quat_w[:, root_asset_cfg.body_ids[0]]
    left_center = asset.data.body_pos_w[:, left_asset_cfg.body_ids, :3].mean(dim=1)
    right_center = asset.data.body_pos_w[:, right_asset_cfg.body_ids, :3].mean(dim=1)
    left_pos_b = _rotate_world_vectors_into_body(root_quat, left_center - root_pos)
    right_pos_b = _rotate_world_vectors_into_body(root_quat, right_center - root_pos)
    return torch.cat((left_pos_b, right_pos_b), dim=1)


def feet_contact_state_observation(
    env,
    left_sensor_cfg: "SceneEntityCfg",
    right_sensor_cfg: "SceneEntityCfg",
) -> torch.Tensor:
    """Return binary per-foot contact state observations."""

    contact_sensor = env.scene.sensors[left_sensor_cfg.name]
    left_in_contact = torch.any(contact_sensor.data.current_contact_time[:, left_sensor_cfg.body_ids] > 0.0, dim=1)
    right_in_contact = torch.any(contact_sensor.data.current_contact_time[:, right_sensor_cfg.body_ids] > 0.0, dim=1)
    return torch.stack((left_in_contact.to(torch.float32), right_in_contact.to(torch.float32)), dim=1)


def feet_mode_time_observation(
    env,
    left_sensor_cfg: "SceneEntityCfg",
    right_sensor_cfg: "SceneEntityCfg",
    time_window: float,
) -> torch.Tensor:
    """Return normalized time since the current contact mode started for each foot."""

    contact_sensor = env.scene.sensors[left_sensor_cfg.name]
    _, left_mode_time = grouped_support_mode_time(
        contact_sensor.data.current_contact_time[:, left_sensor_cfg.body_ids],
        contact_sensor.data.current_air_time[:, left_sensor_cfg.body_ids],
    )
    _, right_mode_time = grouped_support_mode_time(
        contact_sensor.data.current_contact_time[:, right_sensor_cfg.body_ids],
        contact_sensor.data.current_air_time[:, right_sensor_cfg.body_ids],
    )
    return torch.clamp(
        torch.stack((left_mode_time, right_mode_time), dim=1) / max(time_window, 1.0e-6),
        min=0.0,
        max=1.0,
    )


def phase_clock_alternating_foot_contact_reward(
    env,
    command_name: str,
    phase_sharpness: float,
    slow_speed: float,
    fast_speed: float,
    slow_period: float,
    fast_period: float,
    left_sensor_cfg: "SceneEntityCfg",
    right_sensor_cfg: "SceneEntityCfg",
    velocity_gate_std: float | None = None,
    velocity_gate_floor: float = 0.0,
    yaw_rate_gate_std: float | None = None,
    yaw_rate_gate_floor: float = 0.0,
    asset_cfg: "SceneEntityCfg | None" = None,
) -> torch.Tensor:
    """Reward foot contacts that match an explicit left/right alternating gait clock."""

    contact_sensor = env.scene.sensors[left_sensor_cfg.name]
    left_in_contact = torch.any(contact_sensor.data.current_contact_time[:, left_sensor_cfg.body_ids] > 0.0, dim=1)
    right_in_contact = torch.any(contact_sensor.data.current_contact_time[:, right_sensor_cfg.body_ids] > 0.0, dim=1)
    left_in_contact = left_in_contact.to(torch.float32)
    right_in_contact = right_in_contact.to(torch.float32)

    left_phase = _gait_phase(
        env,
        command_name,
        slow_speed=slow_speed,
        fast_speed=fast_speed,
        slow_period=slow_period,
        fast_period=fast_period,
    )
    right_phase = _gait_phase(
        env,
        command_name,
        slow_speed=slow_speed,
        fast_speed=fast_speed,
        slow_period=slow_period,
        fast_period=fast_period,
        phase_offset=torch.pi,
    )
    left_stance_target = torch.sigmoid(phase_sharpness * torch.cos(left_phase))
    right_stance_target = torch.sigmoid(phase_sharpness * torch.cos(right_phase))

    left_match = left_stance_target * left_in_contact + (1.0 - left_stance_target) * (1.0 - left_in_contact)
    right_match = right_stance_target * right_in_contact + (1.0 - right_stance_target) * (1.0 - right_in_contact)

    reward = torch.sqrt(torch.clamp(left_match * right_match, min=0.0))
    if velocity_gate_std and velocity_gate_std > 0.0:
        asset_name = asset_cfg.name if asset_cfg is not None else "robot"
        asset = env.scene[asset_name]
        command = env.command_manager.get_command(command_name)
        lin_vel_error = torch.sum(torch.square(command[:, :2] - asset.data.root_lin_vel_b[:, :2]), dim=1)
        lin_gate = torch.exp(-lin_vel_error / (velocity_gate_std**2))
        lin_gate_floor = max(0.0, min(float(velocity_gate_floor), 1.0))
        if lin_gate_floor > 0.0:
            lin_gate = lin_gate_floor + (1.0 - lin_gate_floor) * lin_gate
        reward *= lin_gate
    if yaw_rate_gate_std and yaw_rate_gate_std > 0.0:
        asset_name = asset_cfg.name if asset_cfg is not None else "robot"
        asset = env.scene[asset_name]
        command = env.command_manager.get_command(command_name)
        yaw_rate_error = torch.square(command[:, 2] - asset.data.root_ang_vel_w[:, 2])
        yaw_gate = torch.exp(-yaw_rate_error / (yaw_rate_gate_std**2))
        yaw_gate_floor = max(0.0, min(float(yaw_rate_gate_floor), 1.0))
        if yaw_gate_floor > 0.0:
            yaw_gate = yaw_gate_floor + (1.0 - yaw_gate_floor) * yaw_gate
        reward *= yaw_gate
    reward *= _command_speed_mask(env, command_name)
    return reward


def contact_body_alignment_penalty(
    env,
    min_cosine: float,
    contact_threshold: float,
    sensor_cfg: "SceneEntityCfg",
    asset_cfg: "SceneEntityCfg",
    local_reference_vectors: tuple[tuple[float, float, float], ...],
    world_reference_vector: tuple[float, float, float] = (0.0, 0.0, 1.0),
) -> torch.Tensor:
    """Penalize contacted bodies whose designated local axis tilts away from the world target axis."""

    asset = env.scene[asset_cfg.name]
    contact_sensor = _scene_entity(env.scene, sensor_cfg.name)
    body_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids]
    local_vectors = torch.tensor(
        local_reference_vectors,
        device=body_quat_w.device,
        dtype=body_quat_w.dtype,
    ).unsqueeze(0).expand_as(body_quat_w[..., 1:])
    world_vectors = _rotate_body_vectors(body_quat_w, local_vectors)
    world_vectors = world_vectors / torch.clamp(torch.linalg.norm(world_vectors, dim=-1, keepdim=True), min=1.0e-8)
    world_target = torch.tensor(world_reference_vector, device=body_quat_w.device, dtype=body_quat_w.dtype)
    world_target = world_target / torch.clamp(torch.linalg.norm(world_target), min=1.0e-8)
    alignment = torch.sum(world_vectors * world_target.view(1, 1, 3), dim=-1)
    contact_mask = _contact_mask_for_bodies(contact_sensor, sensor_cfg.body_ids, contact_threshold).to(alignment.dtype)
    penalty = torch.clamp(min_cosine - alignment, min=0.0) * contact_mask
    return penalty.sum(dim=1) / torch.clamp(contact_mask.sum(dim=1), min=1.0)


def secondary_contact_without_primary_penalty(
    env,
    threshold: float,
    left_primary_sensor_cfg: "SceneEntityCfg",
    right_primary_sensor_cfg: "SceneEntityCfg",
    left_aux_sensor_cfg: "SceneEntityCfg",
    right_aux_sensor_cfg: "SceneEntityCfg",
) -> torch.Tensor:
    """Penalize support phases where auxiliary links touch while the primary foot on that side is unloaded."""

    contact_sensor = _scene_entity(env.scene, left_primary_sensor_cfg.name)
    left_primary_in_contact = torch.any(
        _contact_mask_for_bodies(contact_sensor, left_primary_sensor_cfg.body_ids, threshold),
        dim=1,
    )
    right_primary_in_contact = torch.any(
        _contact_mask_for_bodies(contact_sensor, right_primary_sensor_cfg.body_ids, threshold),
        dim=1,
    )
    left_aux_in_contact = torch.any(
        _contact_mask_for_bodies(contact_sensor, left_aux_sensor_cfg.body_ids, threshold),
        dim=1,
    )
    right_aux_in_contact = torch.any(
        _contact_mask_for_bodies(contact_sensor, right_aux_sensor_cfg.body_ids, threshold),
        dim=1,
    )
    penalty = (left_aux_in_contact & ~left_primary_in_contact).to(torch.float32)
    penalty += (right_aux_in_contact & ~right_primary_in_contact).to(torch.float32)
    return penalty


def secondary_contact_force_share_penalty(
    env,
    command_name: str,
    threshold: float,
    min_primary_force_share: float,
    left_primary_sensor_cfg: "SceneEntityCfg",
    right_primary_sensor_cfg: "SceneEntityCfg",
    left_aux_sensor_cfg: "SceneEntityCfg",
    right_aux_sensor_cfg: "SceneEntityCfg",
) -> torch.Tensor:
    """Penalize support that carries too much load on auxiliary foot contacts."""

    contact_sensor = _scene_entity(env.scene, left_primary_sensor_cfg.name)
    left_primary_force = torch.linalg.norm(
        contact_sensor.data.net_forces_w_history[:, 0, left_primary_sensor_cfg.body_ids], dim=-1
    ).sum(dim=1)
    right_primary_force = torch.linalg.norm(
        contact_sensor.data.net_forces_w_history[:, 0, right_primary_sensor_cfg.body_ids], dim=-1
    ).sum(dim=1)
    left_aux_force = torch.linalg.norm(
        contact_sensor.data.net_forces_w_history[:, 0, left_aux_sensor_cfg.body_ids], dim=-1
    ).sum(dim=1)
    right_aux_force = torch.linalg.norm(
        contact_sensor.data.net_forces_w_history[:, 0, right_aux_sensor_cfg.body_ids], dim=-1
    ).sum(dim=1)

    left_total_force = left_primary_force + left_aux_force
    right_total_force = right_primary_force + right_aux_force
    left_primary_share = left_primary_force / torch.clamp(left_total_force, min=1.0e-6)
    right_primary_share = right_primary_force / torch.clamp(right_total_force, min=1.0e-6)

    left_penalty = torch.clamp(min_primary_force_share - left_primary_share, min=0.0)
    left_penalty *= left_total_force > threshold
    right_penalty = torch.clamp(min_primary_force_share - right_primary_share, min=0.0)
    right_penalty *= right_total_force > threshold

    penalty = left_penalty + right_penalty
    penalty *= _command_speed_mask(env, command_name)
    return penalty


def command_aware_root_planar_speed_penalty(
    env,
    command_name: str,
    max_command_speed: float,
    asset_cfg: "SceneEntityCfg",
) -> torch.Tensor:
    """Penalize root planar motion when the commanded speed is effectively zero."""

    asset = env.scene[asset_cfg.name]
    root_planar_speed = torch.linalg.norm(asset.data.root_lin_vel_w[:, :2], dim=1)
    return root_planar_speed * _command_idle_mask(env, command_name, max_command_speed)


def grouped_support_air_time_positive_biped(
    env,
    command_name: str,
    threshold: float,
    left_sensor_cfg: "SceneEntityCfg",
    right_sensor_cfg: "SceneEntityCfg",
    min_single_stance_time: float = 0.0,
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
    reward = torch.clamp(reward - min_single_stance_time, min=0.0, max=threshold)
    reward *= _command_speed_mask(env, command_name)
    return reward


def primary_single_support_reward(
    env,
    command_name: str,
    overlap_grace_period: float,
    left_sensor_cfg: "SceneEntityCfg",
    right_sensor_cfg: "SceneEntityCfg",
) -> torch.Tensor:
    """Reward primary-foot single support while allowing a brief touchdown overlap."""

    contact_sensor = env.scene.sensors[left_sensor_cfg.name]
    left_contact_time = torch.max(contact_sensor.data.current_contact_time[:, left_sensor_cfg.body_ids], dim=1).values
    right_contact_time = torch.max(contact_sensor.data.current_contact_time[:, right_sensor_cfg.body_ids], dim=1).values
    left_in_contact = left_contact_time > 0.0
    right_in_contact = right_contact_time > 0.0

    single_support = torch.logical_xor(left_in_contact, right_in_contact)
    brief_overlap = (left_in_contact & right_in_contact) & (
        torch.minimum(left_contact_time, right_contact_time) <= overlap_grace_period
    )
    reward = (single_support | brief_overlap).to(torch.float32)
    reward *= _command_speed_mask(env, command_name)
    return reward


def alternating_biped_async_reward(
    env,
    command_name: str,
    std: float,
    max_err: float,
    left_sensor_cfg: "SceneEntityCfg",
    right_sensor_cfg: "SceneEntityCfg",
) -> torch.Tensor:
    """Reward left/right support timing that looks like alternating biped gait."""

    contact_sensor = env.scene.sensors[left_sensor_cfg.name]
    left_contact_time = torch.max(contact_sensor.data.current_contact_time[:, left_sensor_cfg.body_ids], dim=1).values
    right_contact_time = torch.max(contact_sensor.data.current_contact_time[:, right_sensor_cfg.body_ids], dim=1).values
    left_air_time = torch.min(contact_sensor.data.current_air_time[:, left_sensor_cfg.body_ids], dim=1).values
    right_air_time = torch.min(contact_sensor.data.current_air_time[:, right_sensor_cfg.body_ids], dim=1).values

    async_error_0 = torch.clamp(torch.square(left_air_time - right_contact_time), max=max_err * max_err)
    async_error_1 = torch.clamp(torch.square(right_air_time - left_contact_time), max=max_err * max_err)
    reward = torch.exp(-(async_error_0 + async_error_1) / std)
    reward *= _command_speed_mask(env, command_name)
    return reward


def support_width_deviation(
    env,
    command_name: str,
    target_width: float,
    tolerance: float,
    forward_body_axis: str,
    left_asset_cfg: "SceneEntityCfg",
    right_asset_cfg: "SceneEntityCfg",
) -> torch.Tensor:
    """Penalize overly narrow or wide left/right support spacing in the commanded body frame."""

    asset = env.scene[left_asset_cfg.name]
    left_center_xy = asset.data.body_pos_w[:, left_asset_cfg.body_ids, :2].mean(dim=1)
    right_center_xy = asset.data.body_pos_w[:, right_asset_cfg.body_ids, :2].mean(dim=1)
    body_x_xy, body_y_xy = _normalized_planar_body_axes(asset.data.root_quat_w)
    lateral_axis_xy = body_x_xy if forward_body_axis == "y" else body_y_xy
    support_delta_xy = left_center_xy - right_center_xy
    support_width = torch.abs(torch.sum(support_delta_xy * lateral_axis_xy, dim=1))
    penalty = torch.clamp(torch.abs(support_width - target_width) - tolerance, min=0.0)
    penalty *= _command_speed_mask(env, command_name)
    return penalty


def support_width_above_max(
    env,
    command_name: str,
    max_width: float,
    forward_body_axis: str,
    left_asset_cfg: "SceneEntityCfg",
    right_asset_cfg: "SceneEntityCfg",
) -> torch.Tensor:
    """Penalize only overly wide support spacing in the commanded body frame."""

    asset = env.scene[left_asset_cfg.name]
    left_center_xy, right_center_xy = _support_centers_xy(asset, left_asset_cfg, right_asset_cfg)
    body_x_xy, body_y_xy = _normalized_planar_body_axes(asset.data.root_quat_w)
    lateral_axis_xy = body_x_xy if forward_body_axis == "y" else body_y_xy
    support_delta_xy = left_center_xy - right_center_xy
    support_width = torch.abs(torch.sum(support_delta_xy * lateral_axis_xy, dim=1))
    penalty = torch.clamp(support_width - max_width, min=0.0)
    penalty *= _command_speed_mask(env, command_name)
    return penalty


def touchdown_step_length_deficit_penalty(
    env,
    command_name: str,
    min_step_length: float,
    min_air_time: float,
    max_penalized_deficit: float,
    forward_body_axis: str,
    left_sensor_cfg: "SceneEntityCfg",
    right_sensor_cfg: "SceneEntityCfg",
    left_asset_cfg: "SceneEntityCfg",
    right_asset_cfg: "SceneEntityCfg",
) -> torch.Tensor:
    """Penalize touchdown events that land beside or behind the opposite support side."""

    contact_sensor = env.scene.sensors[left_sensor_cfg.name]
    first_contact = contact_sensor.compute_first_contact(env.step_dt)
    left_group_first_contact = torch.any(first_contact[:, left_sensor_cfg.body_ids], dim=1)
    right_group_first_contact = torch.any(first_contact[:, right_sensor_cfg.body_ids], dim=1)
    left_last_air_time = torch.min(contact_sensor.data.last_air_time[:, left_sensor_cfg.body_ids], dim=1).values
    right_last_air_time = torch.min(contact_sensor.data.last_air_time[:, right_sensor_cfg.body_ids], dim=1).values
    left_in_contact = torch.any(contact_sensor.data.current_contact_time[:, left_sensor_cfg.body_ids] > 0.0, dim=1)
    right_in_contact = torch.any(contact_sensor.data.current_contact_time[:, right_sensor_cfg.body_ids] > 0.0, dim=1)

    asset = env.scene[left_asset_cfg.name]
    left_center_xy, right_center_xy = _support_centers_xy(asset, left_asset_cfg, right_asset_cfg)
    body_x_xy, body_y_xy = _normalized_planar_body_axes(asset.data.root_quat_w)
    forward_axis_xy = body_y_xy if forward_body_axis == "y" else body_x_xy
    left_step_length = torch.sum((left_center_xy - right_center_xy) * forward_axis_xy, dim=1)
    right_step_length = torch.sum((right_center_xy - left_center_xy) * forward_axis_xy, dim=1)

    left_penalty = torch.clamp(min_step_length - left_step_length, min=0.0, max=max_penalized_deficit)
    left_penalty *= left_group_first_contact & (left_last_air_time >= min_air_time) & right_in_contact
    right_penalty = torch.clamp(min_step_length - right_step_length, min=0.0, max=max_penalized_deficit)
    right_penalty *= right_group_first_contact & (right_last_air_time >= min_air_time) & left_in_contact
    penalty = left_penalty + right_penalty
    penalty *= _command_speed_mask(env, command_name)
    return penalty


def touchdown_support_width_excess_penalty(
    env,
    command_name: str,
    max_width: float,
    min_air_time: float,
    max_penalized_excess: float,
    forward_body_axis: str,
    left_sensor_cfg: "SceneEntityCfg",
    right_sensor_cfg: "SceneEntityCfg",
    left_asset_cfg: "SceneEntityCfg",
    right_asset_cfg: "SceneEntityCfg",
) -> torch.Tensor:
    """Penalize touchdown events that open the stance wider than the walk target."""

    contact_sensor = env.scene.sensors[left_sensor_cfg.name]
    first_contact = contact_sensor.compute_first_contact(env.step_dt)
    left_group_first_contact = torch.any(first_contact[:, left_sensor_cfg.body_ids], dim=1)
    right_group_first_contact = torch.any(first_contact[:, right_sensor_cfg.body_ids], dim=1)
    left_last_air_time = torch.min(contact_sensor.data.last_air_time[:, left_sensor_cfg.body_ids], dim=1).values
    right_last_air_time = torch.min(contact_sensor.data.last_air_time[:, right_sensor_cfg.body_ids], dim=1).values
    left_in_contact = torch.any(contact_sensor.data.current_contact_time[:, left_sensor_cfg.body_ids] > 0.0, dim=1)
    right_in_contact = torch.any(contact_sensor.data.current_contact_time[:, right_sensor_cfg.body_ids] > 0.0, dim=1)

    asset = env.scene[left_asset_cfg.name]
    left_center_xy, right_center_xy = _support_centers_xy(asset, left_asset_cfg, right_asset_cfg)
    body_x_xy, body_y_xy = _normalized_planar_body_axes(asset.data.root_quat_w)
    lateral_axis_xy = body_x_xy if forward_body_axis == "y" else body_y_xy
    support_width = torch.abs(torch.sum((left_center_xy - right_center_xy) * lateral_axis_xy, dim=1))

    width_penalty = torch.clamp(support_width - max_width, min=0.0, max=max_penalized_excess)
    event_mask = (
        (left_group_first_contact & (left_last_air_time >= min_air_time) & right_in_contact)
        | (right_group_first_contact & (right_last_air_time >= min_air_time) & left_in_contact)
    )
    penalty = width_penalty * event_mask
    penalty *= _command_speed_mask(env, command_name)
    return penalty


def landing_step_ahead_reward(
    env,
    command_name: str,
    step_length_threshold: float,
    min_air_time: float,
    max_rewarded_step_length: float,
    forward_body_axis: str,
    left_sensor_cfg: "SceneEntityCfg",
    right_sensor_cfg: "SceneEntityCfg",
    left_asset_cfg: "SceneEntityCfg",
    right_asset_cfg: "SceneEntityCfg",
) -> torch.Tensor:
    """Reward real touchdown events that land the swing side ahead of the opposite support side."""

    contact_sensor = env.scene.sensors[left_sensor_cfg.name]
    first_contact = contact_sensor.compute_first_contact(env.step_dt)
    left_group_first_contact = torch.any(first_contact[:, left_sensor_cfg.body_ids], dim=1)
    right_group_first_contact = torch.any(first_contact[:, right_sensor_cfg.body_ids], dim=1)
    left_last_air_time = torch.min(contact_sensor.data.last_air_time[:, left_sensor_cfg.body_ids], dim=1).values
    right_last_air_time = torch.min(contact_sensor.data.last_air_time[:, right_sensor_cfg.body_ids], dim=1).values
    left_in_contact = torch.any(contact_sensor.data.current_contact_time[:, left_sensor_cfg.body_ids] > 0.0, dim=1)
    right_in_contact = torch.any(contact_sensor.data.current_contact_time[:, right_sensor_cfg.body_ids] > 0.0, dim=1)

    asset = env.scene[left_asset_cfg.name]
    left_center_xy, right_center_xy = _support_centers_xy(asset, left_asset_cfg, right_asset_cfg)
    body_x_xy, body_y_xy = _normalized_planar_body_axes(asset.data.root_quat_w)
    forward_axis_xy = body_y_xy if forward_body_axis == "y" else body_x_xy
    left_step_length = torch.sum((left_center_xy - right_center_xy) * forward_axis_xy, dim=1)
    right_step_length = torch.sum((right_center_xy - left_center_xy) * forward_axis_xy, dim=1)

    left_reward = torch.clamp(left_step_length - step_length_threshold, min=0.0, max=max_rewarded_step_length)
    left_reward *= left_group_first_contact & (left_last_air_time >= min_air_time) & right_in_contact
    right_reward = torch.clamp(right_step_length - step_length_threshold, min=0.0, max=max_rewarded_step_length)
    right_reward *= right_group_first_contact & (right_last_air_time >= min_air_time) & left_in_contact
    reward = left_reward + right_reward
    reward *= _command_speed_mask(env, command_name)
    return reward


def touchdown_root_straddle_reward(
    env,
    command_name: str,
    landing_margin: float,
    stance_margin: float,
    min_air_time: float,
    max_rewarded_margin: float,
    forward_body_axis: str,
    root_asset_cfg: "SceneEntityCfg",
    left_sensor_cfg: "SceneEntityCfg",
    right_sensor_cfg: "SceneEntityCfg",
    left_asset_cfg: "SceneEntityCfg",
    right_asset_cfg: "SceneEntityCfg",
) -> torch.Tensor:
    """Reward touchdown only when the root ends up between the stance and landing feet."""

    contact_sensor = env.scene.sensors[left_sensor_cfg.name]
    first_contact = contact_sensor.compute_first_contact(env.step_dt)
    left_group_first_contact = torch.any(first_contact[:, left_sensor_cfg.body_ids], dim=1)
    right_group_first_contact = torch.any(first_contact[:, right_sensor_cfg.body_ids], dim=1)
    left_last_air_time = torch.min(contact_sensor.data.last_air_time[:, left_sensor_cfg.body_ids], dim=1).values
    right_last_air_time = torch.min(contact_sensor.data.last_air_time[:, right_sensor_cfg.body_ids], dim=1).values
    left_in_contact = torch.any(contact_sensor.data.current_contact_time[:, left_sensor_cfg.body_ids] > 0.0, dim=1)
    right_in_contact = torch.any(contact_sensor.data.current_contact_time[:, right_sensor_cfg.body_ids] > 0.0, dim=1)

    asset = env.scene[left_asset_cfg.name]
    root_xy = asset.data.body_pos_w[:, root_asset_cfg.body_ids, :2].mean(dim=1)
    left_center_xy, right_center_xy = _support_centers_xy(asset, left_asset_cfg, right_asset_cfg)
    body_x_xy, body_y_xy = _normalized_planar_body_axes(asset.data.root_quat_w)
    forward_axis_xy = body_y_xy if forward_body_axis == "y" else body_x_xy

    left_rel_root = torch.sum((left_center_xy - root_xy) * forward_axis_xy, dim=1)
    right_rel_root = torch.sum((right_center_xy - root_xy) * forward_axis_xy, dim=1)

    left_landing_ahead = torch.clamp(left_rel_root - landing_margin, min=0.0, max=max_rewarded_margin)
    right_stance_behind = torch.clamp(-right_rel_root - stance_margin, min=0.0, max=max_rewarded_margin)
    left_reward = torch.minimum(left_landing_ahead, right_stance_behind)
    left_reward *= left_group_first_contact & (left_last_air_time >= min_air_time) & right_in_contact

    right_landing_ahead = torch.clamp(right_rel_root - landing_margin, min=0.0, max=max_rewarded_margin)
    left_stance_behind = torch.clamp(-left_rel_root - stance_margin, min=0.0, max=max_rewarded_margin)
    right_reward = torch.minimum(right_landing_ahead, left_stance_behind)
    right_reward *= right_group_first_contact & (right_last_air_time >= min_air_time) & left_in_contact

    reward = left_reward + right_reward
    reward *= _command_speed_mask(env, command_name)
    return reward


def swing_foot_ahead_of_stance_reward(
    env,
    command_name: str,
    min_step_length: float,
    max_rewarded_step_length: float,
    forward_body_axis: str,
    left_sensor_cfg: "SceneEntityCfg",
    right_sensor_cfg: "SceneEntityCfg",
    left_asset_cfg: "SceneEntityCfg",
    right_asset_cfg: "SceneEntityCfg",
) -> torch.Tensor:
    """Reward single-support poses where the swing foot is clearly ahead of the stance foot."""

    contact_sensor = env.scene.sensors[left_sensor_cfg.name]
    left_in_contact = torch.any(contact_sensor.data.current_contact_time[:, left_sensor_cfg.body_ids] > 0.0, dim=1)
    right_in_contact = torch.any(contact_sensor.data.current_contact_time[:, right_sensor_cfg.body_ids] > 0.0, dim=1)

    asset = env.scene[left_asset_cfg.name]
    left_center_xy, right_center_xy = _support_centers_xy(asset, left_asset_cfg, right_asset_cfg)
    body_x_xy, body_y_xy = _normalized_planar_body_axes(asset.data.root_quat_w)
    forward_axis_xy = body_y_xy if forward_body_axis == "y" else body_x_xy

    left_step_length = torch.sum((left_center_xy - right_center_xy) * forward_axis_xy, dim=1)
    right_step_length = torch.sum((right_center_xy - left_center_xy) * forward_axis_xy, dim=1)

    left_reward = torch.clamp(left_step_length - min_step_length, min=0.0, max=max_rewarded_step_length)
    left_reward *= (~left_in_contact) & right_in_contact
    right_reward = torch.clamp(right_step_length - min_step_length, min=0.0, max=max_rewarded_step_length)
    right_reward *= (~right_in_contact) & left_in_contact

    reward = left_reward + right_reward
    reward *= _command_speed_mask(env, command_name)
    return reward


def single_support_root_straddle_reward(
    env,
    command_name: str,
    root_margin: float,
    max_rewarded_margin: float,
    forward_body_axis: str,
    root_asset_cfg: "SceneEntityCfg",
    left_sensor_cfg: "SceneEntityCfg",
    right_sensor_cfg: "SceneEntityCfg",
    left_asset_cfg: "SceneEntityCfg",
    right_asset_cfg: "SceneEntityCfg",
) -> torch.Tensor:
    """Reward single-support phases where the root is straddled by stance and swing feet."""

    contact_sensor = env.scene.sensors[left_sensor_cfg.name]
    left_in_contact = torch.any(contact_sensor.data.current_contact_time[:, left_sensor_cfg.body_ids] > 0.0, dim=1)
    right_in_contact = torch.any(contact_sensor.data.current_contact_time[:, right_sensor_cfg.body_ids] > 0.0, dim=1)

    asset = env.scene[left_asset_cfg.name]
    root_xy = asset.data.body_pos_w[:, root_asset_cfg.body_ids, :2].mean(dim=1)
    left_center_xy, right_center_xy = _support_centers_xy(asset, left_asset_cfg, right_asset_cfg)
    body_x_xy, body_y_xy = _normalized_planar_body_axes(asset.data.root_quat_w)
    forward_axis_xy = body_y_xy if forward_body_axis == "y" else body_x_xy

    left_rel_root = torch.sum((left_center_xy - root_xy) * forward_axis_xy, dim=1)
    right_rel_root = torch.sum((right_center_xy - root_xy) * forward_axis_xy, dim=1)

    left_reward = torch.minimum(
        torch.clamp(left_rel_root - root_margin, min=0.0, max=max_rewarded_margin),
        torch.clamp(-right_rel_root - root_margin, min=0.0, max=max_rewarded_margin),
    )
    left_reward *= (~left_in_contact) & right_in_contact
    right_reward = torch.minimum(
        torch.clamp(right_rel_root - root_margin, min=0.0, max=max_rewarded_margin),
        torch.clamp(-left_rel_root - root_margin, min=0.0, max=max_rewarded_margin),
    )
    right_reward *= (~right_in_contact) & left_in_contact

    reward = left_reward + right_reward
    reward *= _command_speed_mask(env, command_name)
    return reward


def grouped_support_flight_time_penalty(
    env,
    command_name: str,
    threshold: float,
    left_sensor_cfg: "SceneEntityCfg",
    right_sensor_cfg: "SceneEntityCfg",
) -> torch.Tensor:
    """Penalize sustained aerial phases when both support sides are off the ground."""

    contact_sensor = env.scene.sensors[left_sensor_cfg.name]
    left_in_contact = torch.any(contact_sensor.data.current_contact_time[:, left_sensor_cfg.body_ids] > 0.0, dim=1)
    right_in_contact = torch.any(contact_sensor.data.current_contact_time[:, right_sensor_cfg.body_ids] > 0.0, dim=1)
    left_air_time = torch.min(contact_sensor.data.current_air_time[:, left_sensor_cfg.body_ids], dim=1).values
    right_air_time = torch.min(contact_sensor.data.current_air_time[:, right_sensor_cfg.body_ids], dim=1).values

    penalty = torch.clamp(torch.minimum(left_air_time, right_air_time) - threshold, min=0.0)
    penalty *= (~left_in_contact) & (~right_in_contact)
    penalty *= _command_speed_mask(env, command_name)
    return penalty


def grouped_support_double_stance_time_penalty(
    env,
    command_name: str,
    threshold: float,
    left_sensor_cfg: "SceneEntityCfg",
    right_sensor_cfg: "SceneEntityCfg",
) -> torch.Tensor:
    """Penalize prolonged double-support dwell while still allowing brief walk transitions."""

    contact_sensor = env.scene.sensors[left_sensor_cfg.name]
    left_contact_time = torch.max(contact_sensor.data.current_contact_time[:, left_sensor_cfg.body_ids], dim=1).values
    right_contact_time = torch.max(contact_sensor.data.current_contact_time[:, right_sensor_cfg.body_ids], dim=1).values
    left_in_contact = left_contact_time > 0.0
    right_in_contact = right_contact_time > 0.0

    penalty = torch.clamp(torch.minimum(left_contact_time, right_contact_time) - threshold, min=0.0)
    penalty *= left_in_contact & right_in_contact
    penalty *= _command_speed_mask(env, command_name)
    return penalty


def swing_height_difference_below_min(
    env,
    command_name: str,
    min_height_difference: float,
    left_sensor_cfg: "SceneEntityCfg",
    right_sensor_cfg: "SceneEntityCfg",
    left_asset_cfg: "SceneEntityCfg",
    right_asset_cfg: "SceneEntityCfg",
) -> torch.Tensor:
    """Penalize swing phases that fail to lift the unloaded side above the stance side."""

    contact_sensor = env.scene.sensors[left_sensor_cfg.name]
    left_in_contact = torch.any(contact_sensor.data.current_contact_time[:, left_sensor_cfg.body_ids] > 0.0, dim=1)
    right_in_contact = torch.any(contact_sensor.data.current_contact_time[:, right_sensor_cfg.body_ids] > 0.0, dim=1)

    asset = env.scene[left_asset_cfg.name]
    left_height, right_height = _support_mean_heights(asset, left_asset_cfg, right_asset_cfg)
    left_swing_penalty = torch.clamp(min_height_difference - (left_height - right_height), min=0.0)
    left_swing_penalty *= (~left_in_contact) & right_in_contact
    right_swing_penalty = torch.clamp(min_height_difference - (right_height - left_height), min=0.0)
    right_swing_penalty *= (~right_in_contact) & left_in_contact
    penalty = left_swing_penalty + right_swing_penalty
    penalty *= _command_speed_mask(env, command_name)
    return penalty


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
    left_in_contact = torch.any(contact_sensor.data.current_contact_time[:, left_sensor_cfg.body_ids] > 0.0, dim=1)
    right_in_contact = torch.any(contact_sensor.data.current_contact_time[:, right_sensor_cfg.body_ids] > 0.0, dim=1)
    left_reward *= right_in_contact
    right_reward *= left_in_contact

    reward = left_reward + right_reward
    reward *= _command_speed_mask(env, command_name)
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
