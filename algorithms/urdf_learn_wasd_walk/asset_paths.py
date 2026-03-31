from __future__ import annotations

from pathlib import Path


def module_root() -> Path:
    return Path(__file__).resolve().parent


def inputs_dir() -> Path:
    return module_root() / "inputs"


def outputs_dir() -> Path:
    return module_root() / "outputs"


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
    return module_root().parents[1] / "algorithms" / "usd_parallel_urdf" / "outputs" / "landau_v10_parallel_mesh.urdf"


def default_landau_source_mesh_root() -> Path:
    return module_root().parents[1] / "algorithms" / "usd_parallel_urdf" / "outputs" / "mesh_collision_stl"


def default_landau_source_usd() -> Path:
    return module_root().parents[1] / "algorithms" / "usd_parallel_urdf" / "inputs" / "landau_v10.usdc"


def default_landau_source_skeleton_json() -> Path:
    return module_root().parents[1] / "algorithms" / "usd_parallel_urdf" / "outputs" / "landau_v10_skeleton.json"


def default_landau_source_texture_dir() -> Path:
    return module_root().parents[1] / "algorithms" / "usd_parallel_urdf" / "inputs" / "textures"


def robot_output_dir(robot_key: str) -> Path:
    return outputs_dir() / robot_key
