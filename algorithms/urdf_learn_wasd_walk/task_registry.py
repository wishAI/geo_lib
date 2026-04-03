from __future__ import annotations

import gymnasium as gym

from . import agents


TASK_IDS = {
    "g1": {
        "train": "Geo-Velocity-Flat-G1-v0",
        "play": "Geo-Velocity-Flat-G1-Play-v0",
    },
    "landau": {
        "train": "Geo-Velocity-Flat-Landau-v0",
        "play": "Geo-Velocity-Flat-Landau-Play-v0",
    },
    "landau_fwd_only": {
        "train": "Geo-Velocity-Flat-Landau-FwdOnly-v0",
        "play": "Geo-Velocity-Flat-Landau-FwdOnly-Play-v0",
    },
    "landau_fwd_yaw": {
        "train": "Geo-Velocity-Flat-Landau-FwdYaw-v0",
        "play": "Geo-Velocity-Flat-Landau-FwdYaw-Play-v0",
    },
}

LANDAU_CURRICULUM_STAGES = ("fwd_only", "fwd_yaw", "full")


def _register_if_missing(task_id: str, env_cfg_entry_point: str, rsl_rl_cfg_entry_point: str) -> None:
    if task_id in gym.registry:
        return
    gym.register(
        id=task_id,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": env_cfg_entry_point,
            "rsl_rl_cfg_entry_point": rsl_rl_cfg_entry_point,
        },
    )


def register_gym_envs() -> None:
    pkg = __name__.rsplit(".", 1)[0]
    agents_pkg = agents.__name__

    _register_if_missing(
        task_id=TASK_IDS["g1"]["train"],
        env_cfg_entry_point=f"{pkg}.g1_env_cfg:GeoG1FlatEnvCfg",
        rsl_rl_cfg_entry_point=f"{agents_pkg}.rsl_rl_ppo_cfg:GeoG1FlatPPORunnerCfg",
    )
    _register_if_missing(
        task_id=TASK_IDS["g1"]["play"],
        env_cfg_entry_point=f"{pkg}.g1_env_cfg:GeoG1FlatEnvCfg_PLAY",
        rsl_rl_cfg_entry_point=f"{agents_pkg}.rsl_rl_ppo_cfg:GeoG1FlatPPORunnerCfg",
    )
    # Landau full (Stage D / current default)
    _register_if_missing(
        task_id=TASK_IDS["landau"]["train"],
        env_cfg_entry_point=f"{pkg}.landau_env_cfg:LandauFlatEnvCfg",
        rsl_rl_cfg_entry_point=f"{agents_pkg}.rsl_rl_ppo_cfg:LandauFlatPPORunnerCfg",
    )
    _register_if_missing(
        task_id=TASK_IDS["landau"]["play"],
        env_cfg_entry_point=f"{pkg}.landau_env_cfg:LandauFlatEnvCfg_PLAY",
        rsl_rl_cfg_entry_point=f"{agents_pkg}.rsl_rl_ppo_cfg:LandauFlatPPORunnerCfg",
    )
    # Landau Stage A: forward-only
    _register_if_missing(
        task_id=TASK_IDS["landau_fwd_only"]["train"],
        env_cfg_entry_point=f"{pkg}.landau_env_cfg:LandauFwdOnlyEnvCfg",
        rsl_rl_cfg_entry_point=f"{agents_pkg}.rsl_rl_ppo_cfg:LandauFwdOnlyPPORunnerCfg",
    )
    _register_if_missing(
        task_id=TASK_IDS["landau_fwd_only"]["play"],
        env_cfg_entry_point=f"{pkg}.landau_env_cfg:LandauFwdOnlyEnvCfg_PLAY",
        rsl_rl_cfg_entry_point=f"{agents_pkg}.rsl_rl_ppo_cfg:LandauFwdOnlyPPORunnerCfg",
    )
    # Landau Stage B: forward + yaw
    _register_if_missing(
        task_id=TASK_IDS["landau_fwd_yaw"]["train"],
        env_cfg_entry_point=f"{pkg}.landau_env_cfg:LandauFwdYawEnvCfg",
        rsl_rl_cfg_entry_point=f"{agents_pkg}.rsl_rl_ppo_cfg:LandauFwdYawPPORunnerCfg",
    )
    _register_if_missing(
        task_id=TASK_IDS["landau_fwd_yaw"]["play"],
        env_cfg_entry_point=f"{pkg}.landau_env_cfg:LandauFwdYawEnvCfg_PLAY",
        rsl_rl_cfg_entry_point=f"{agents_pkg}.rsl_rl_ppo_cfg:LandauFwdYawPPORunnerCfg",
    )


def task_id_for_robot(robot_key: str, play: bool = False) -> str:
    normalized = robot_key.lower()
    mode = "play" if play else "train"
    if normalized not in TASK_IDS:
        raise KeyError(f"Unsupported robot '{robot_key}'.")
    return TASK_IDS[normalized][mode]
