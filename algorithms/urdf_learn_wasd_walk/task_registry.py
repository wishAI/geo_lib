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
}


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
    _register_if_missing(
        task_id=TASK_IDS["g1"]["train"],
        env_cfg_entry_point=f"{__name__.rsplit('.', 1)[0]}.g1_env_cfg:GeoG1FlatEnvCfg",
        rsl_rl_cfg_entry_point=f"{agents.__name__}.rsl_rl_ppo_cfg:GeoG1FlatPPORunnerCfg",
    )
    _register_if_missing(
        task_id=TASK_IDS["g1"]["play"],
        env_cfg_entry_point=f"{__name__.rsplit('.', 1)[0]}.g1_env_cfg:GeoG1FlatEnvCfg_PLAY",
        rsl_rl_cfg_entry_point=f"{agents.__name__}.rsl_rl_ppo_cfg:GeoG1FlatPPORunnerCfg",
    )
    _register_if_missing(
        task_id=TASK_IDS["landau"]["train"],
        env_cfg_entry_point=f"{__name__.rsplit('.', 1)[0]}.landau_env_cfg:LandauFlatEnvCfg",
        rsl_rl_cfg_entry_point=f"{agents.__name__}.rsl_rl_ppo_cfg:LandauFlatPPORunnerCfg",
    )
    _register_if_missing(
        task_id=TASK_IDS["landau"]["play"],
        env_cfg_entry_point=f"{__name__.rsplit('.', 1)[0]}.landau_env_cfg:LandauFlatEnvCfg_PLAY",
        rsl_rl_cfg_entry_point=f"{agents.__name__}.rsl_rl_ppo_cfg:LandauFlatPPORunnerCfg",
    )


def task_id_for_robot(robot_key: str, play: bool = False) -> str:
    normalized = robot_key.lower()
    mode = "play" if play else "train"
    if normalized not in TASK_IDS:
        raise KeyError(f"Unsupported robot '{robot_key}'.")
    return TASK_IDS[normalized][mode]

