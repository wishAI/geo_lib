from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from asset_paths import (
    default_landau_source_mesh_root,
    default_landau_source_skeleton_json,
    default_landau_source_texture_dir,
    default_landau_source_urdf,
    default_landau_source_usd,
    inputs_dir,
    landau_input_dir,
    landau_mesh_root,
    landau_skeleton_json_path,
    landau_texture_dir,
    landau_urdf_path,
    landau_usd_path,
)


@dataclass(frozen=True)
class PreparedAssetPaths:
    input_dir: Path
    urdf_path: Path
    mesh_root: Path
    usd_path: Path
    skeleton_json_path: Path
    texture_dir: Path


def prepare_landau_inputs(refresh: bool = False) -> PreparedAssetPaths:
    src_urdf = default_landau_source_urdf()
    src_mesh_root = default_landau_source_mesh_root()
    src_usd = default_landau_source_usd()
    src_skeleton_json = default_landau_source_skeleton_json()
    src_texture_dir = default_landau_source_texture_dir()
    dst_dir = landau_input_dir()
    dst_dir.mkdir(parents=True, exist_ok=True)

    if refresh or not landau_urdf_path().exists():
        if not src_urdf.exists():
            raise FileNotFoundError(f"Missing source URDF: {src_urdf}")
        shutil.copy2(src_urdf, landau_urdf_path())

    if refresh and landau_mesh_root().exists():
        shutil.rmtree(landau_mesh_root())
    if refresh or not landau_mesh_root().exists():
        if not src_mesh_root.exists():
            raise FileNotFoundError(f"Missing source mesh folder: {src_mesh_root}")
        shutil.copytree(src_mesh_root, landau_mesh_root(), dirs_exist_ok=True)

    if refresh or not landau_usd_path().exists():
        if not src_usd.exists():
            raise FileNotFoundError(f"Missing source USD: {src_usd}")
        shutil.copy2(src_usd, landau_usd_path())

    if refresh or not landau_skeleton_json_path().exists():
        if not src_skeleton_json.exists():
            raise FileNotFoundError(f"Missing source skeleton JSON: {src_skeleton_json}")
        shutil.copy2(src_skeleton_json, landau_skeleton_json_path())

    if refresh and landau_texture_dir().exists():
        shutil.rmtree(landau_texture_dir())
    if refresh or not landau_texture_dir().exists():
        if not src_texture_dir.exists():
            raise FileNotFoundError(f"Missing source texture folder: {src_texture_dir}")
        shutil.copytree(src_texture_dir, landau_texture_dir(), dirs_exist_ok=True)

    return PreparedAssetPaths(
        input_dir=dst_dir,
        urdf_path=landau_urdf_path(),
        mesh_root=landau_mesh_root(),
        usd_path=landau_usd_path(),
        skeleton_json_path=landau_skeleton_json_path(),
        texture_dir=landau_texture_dir(),
    )


def ensure_inputs_layout() -> None:
    inputs_dir().mkdir(parents=True, exist_ok=True)
    prepare_landau_inputs(refresh=False)
