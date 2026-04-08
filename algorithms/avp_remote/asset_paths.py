from __future__ import annotations

from pathlib import Path


def module_root() -> Path:
    return Path(__file__).resolve().parent


def repo_root() -> Path:
    return module_root().parents[1]


def inputs_dir() -> Path:
    return module_root() / "inputs"


def landau_input_dir() -> Path:
    return inputs_dir() / "landau_v10"


def landau_urdf_path() -> Path:
    return landau_input_dir() / "landau_v10_parallel_mesh.urdf"


def landau_mesh_root() -> Path:
    return landau_input_dir() / "mesh_collision_stl"


def landau_usd_path() -> Path:
    return landau_input_dir() / "landau_v10.usdc"


def landau_skeleton_json_path() -> Path:
    return landau_input_dir() / "landau_v10_skeleton.json"


def landau_texture_dir() -> Path:
    return landau_input_dir() / "textures"


def default_landau_source_urdf() -> Path:
    return repo_root() / "algorithms" / "usd_parallel_urdf" / "outputs" / "landau_v10_parallel_mesh.urdf"


def default_landau_source_mesh_root() -> Path:
    return repo_root() / "algorithms" / "usd_parallel_urdf" / "outputs" / "mesh_collision_stl"


def default_landau_source_usd() -> Path:
    return repo_root() / "algorithms" / "usd_parallel_urdf" / "inputs" / "landau_v10.usdc"


def default_landau_source_skeleton_json() -> Path:
    return repo_root() / "algorithms" / "usd_parallel_urdf" / "outputs" / "landau_v10_skeleton.json"


def default_landau_source_texture_dir() -> Path:
    return repo_root() / "algorithms" / "usd_parallel_urdf" / "inputs" / "textures"


def default_h1_2_urdf_path() -> Path:
    return repo_root() / "helper_repos" / "xr_teleoperate_shallow" / "assets" / "h1_2" / "h1_2.urdf"


def default_g1_urdf_path() -> Path:
    # Backward-compatible alias kept for existing launch scripts and tests.
    return default_h1_2_urdf_path()


def default_dex_retargeting_python_path() -> Path:
    return repo_root() / "helper_repos" / "dex_retargeting_env" / "bin" / "python"
