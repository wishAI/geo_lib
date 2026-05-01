from __future__ import annotations

import math


def _nonfinite_tensor_count(state_dict) -> int:
    import torch

    total = 0
    for value in state_dict.values():
        if torch.is_tensor(value):
            total += int((~torch.isfinite(value)).sum().item())
    return total


def _clone_state_dict(state_dict):
    import torch

    cloned = {}
    for key, value in state_dict.items():
        if torch.is_tensor(value):
            cloned[key] = value.detach().clone()
        else:
            cloned[key] = value
    return cloned


def _sanitize_logged_loss(value):
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return value
    if math.isfinite(numeric):
        return numeric
    return 0.0


def _install_safe_distribution_patch() -> None:
    try:
        from rsl_rl.modules.actor_critic import ActorCritic
        from torch.distributions import Normal
        import torch
    except ImportError:
        return

    if getattr(ActorCritic, "_geo_safe_distribution_patch_installed", False):
        return

    original_update_distribution = ActorCritic.update_distribution

    def _safe_update_distribution(self, observations):
        mean = self.actor(observations)
        mean = torch.nan_to_num(mean, nan=0.0, posinf=1.0e3, neginf=-1.0e3)

        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            log_std = torch.nan_to_num(self.log_std, nan=math.log(0.8), posinf=2.0, neginf=-20.0)
            log_std = torch.clamp(log_std, min=-20.0, max=2.0)
            std = torch.exp(log_std).expand_as(mean)
        else:
            return original_update_distribution(self, observations)

        invalid_mask = ~torch.isfinite(std) | (std <= 0.0)
        if torch.any(invalid_mask):
            std = torch.nan_to_num(std, nan=1.0e-4, posinf=7.5, neginf=1.0e-4)
            std = torch.clamp(std, min=1.0e-4, max=7.5)
            if not getattr(self, "_geo_invalid_std_warned", False):
                invalid_count = int(invalid_mask.sum().item())
                print(
                    f"[SAFE_PPO] sanitized {invalid_count} invalid std values before constructing Normal().",
                    flush=True,
                )
                self._geo_invalid_std_warned = True

        self.distribution = Normal(mean, std)

    ActorCritic.update_distribution = _safe_update_distribution
    ActorCritic._geo_safe_distribution_patch_installed = True


def _install_safe_ppo_update_patch() -> None:
    try:
        from rsl_rl.algorithms.ppo import PPO
    except ImportError:
        return

    if getattr(PPO, "_geo_safe_update_patch_installed", False):
        return

    original_update = PPO.update

    def _restore_last_finite_update(self, snapshot, reason: str):
        self.actor_critic.load_state_dict(snapshot)
        if getattr(self, "optimizer", None) is not None:
            self.optimizer.state.clear()
            self.learning_rate = max(1.0e-5, float(self.learning_rate) / 2.0)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.learning_rate
        if getattr(self, "storage", None) is not None:
            self.storage.clear()
        recovery_count = int(getattr(self, "_geo_update_recovery_count", 0)) + 1
        self._geo_update_recovery_count = recovery_count
        print(
            f"[SAFE_PPO] rolled back non-finite PPO update ({reason}); "
            f"recovery_count={recovery_count} lr={self.learning_rate:.6g}",
            flush=True,
        )

    def _safe_update(self):
        snapshot = _clone_state_dict(self.actor_critic.state_dict())
        try:
            result = original_update(self)
        except Exception as exc:
            _restore_last_finite_update(self, snapshot, f"exception: {exc}")
            return 0.0, 0.0, 0.0, None, None

        bad_param_count = _nonfinite_tensor_count(self.actor_critic.state_dict())
        numeric_losses = [value for value in result if isinstance(value, (int, float))]
        bad_loss = any(not math.isfinite(float(value)) for value in numeric_losses)
        if bad_param_count > 0 or bad_loss:
            reason = f"nonfinite_params={bad_param_count}"
            if bad_loss:
                reason += " nonfinite_losses=1"
            _restore_last_finite_update(self, snapshot, reason)
            return tuple(_sanitize_logged_loss(value) for value in result)

        return result

    PPO.update = _safe_update
    PPO._geo_safe_update_patch_installed = True


def install_safe_actor_critic_distribution_patch() -> None:
    """Install RSL-RL safety patches used by train/play/validate entry points."""

    _install_safe_distribution_patch()
    _install_safe_ppo_update_patch()
