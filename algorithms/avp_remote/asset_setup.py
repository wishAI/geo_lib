from __future__ import annotations

import filecmp
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


def _file_needs_sync(src_path: Path, dst_path: Path) -> bool:
    if not dst_path.exists():
        return True
    return not filecmp.cmp(src_path, dst_path, shallow=False)


def _tree_needs_sync(src_root: Path, dst_root: Path) -> bool:
    if not dst_root.exists():
        return True
    src_files = {
        path.relative_to(src_root)
        for path in src_root.rglob("*")
        if path.is_file()
    }
    dst_files = {
        path.relative_to(dst_root)
        for path in dst_root.rglob("*")
        if path.is_file()
    }
    if src_files != dst_files:
        return True
    return any(_file_needs_sync(src_root / rel_path, dst_root / rel_path) for rel_path in src_files)


def _sync_file(src_path: Path, dst_path: Path, *, refresh: bool) -> None:
    if refresh or _file_needs_sync(src_path, dst_path):
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_path, dst_path)


def _sync_tree(src_root: Path, dst_root: Path, *, refresh: bool) -> None:
    if refresh or _tree_needs_sync(src_root, dst_root):
        if dst_root.exists():
            shutil.rmtree(dst_root)
        shutil.copytree(src_root, dst_root)


def prepare_landau_inputs(refresh: bool = False) -> PreparedAssetPaths:
    src_urdf = default_landau_source_urdf()
    src_mesh_root = default_landau_source_mesh_root()
    src_usd = default_landau_source_usd()
    src_skeleton_json = default_landau_source_skeleton_json()
    src_texture_dir = default_landau_source_texture_dir()
    dst_dir = landau_input_dir()
    dst_dir.mkdir(parents=True, exist_ok=True)

    if not src_urdf.exists():
        raise FileNotFoundError(f"Missing source URDF: {src_urdf}")
    _sync_file(src_urdf, landau_urdf_path(), refresh=refresh)

    if not src_mesh_root.exists():
        raise FileNotFoundError(f"Missing source mesh folder: {src_mesh_root}")
    _sync_tree(src_mesh_root, landau_mesh_root(), refresh=refresh)

    if not src_usd.exists():
        raise FileNotFoundError(f"Missing source USD: {src_usd}")
    _sync_file(src_usd, landau_usd_path(), refresh=refresh)

    if not src_skeleton_json.exists():
        raise FileNotFoundError(f"Missing source skeleton JSON: {src_skeleton_json}")
    _sync_file(src_skeleton_json, landau_skeleton_json_path(), refresh=refresh)

    if not src_texture_dir.exists():
        raise FileNotFoundError(f"Missing source texture folder: {src_texture_dir}")
    _sync_tree(src_texture_dir, landau_texture_dir(), refresh=refresh)

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
