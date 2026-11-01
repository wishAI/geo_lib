from __future__ import annotations

import filecmp
import hashlib
import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from asset_paths import (
    default_landau_source_mesh_root,
    default_landau_source_skeleton_json,
    default_landau_source_texture_dir,
    default_landau_source_urdf,
    default_landau_source_usd,
    inputs_dir,
    repo_root,
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


def _tree_needs_sync(src_root: Path, dst_root: Path, ignored: frozenset[Path] = frozenset()) -> bool:
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
    src_files -= ignored
    dst_files -= ignored
    if src_files != dst_files:
        return True
    return any(_file_needs_sync(src_root / rel_path, dst_root / rel_path) for rel_path in src_files)


def _sync_file(src_path: Path, dst_path: Path, *, refresh: bool) -> None:
    if refresh or _file_needs_sync(src_path, dst_path):
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_path, dst_path)


def _sync_tree(src_root: Path, dst_root: Path, *, refresh: bool, replacements: dict[str, bytes] | None = None) -> None:
    replacements = replacements or {}
    ignored = frozenset(map(Path, replacements))
    changed = any(not (dst_root / name).is_file() or (dst_root / name).read_bytes() != content
                  for name, content in replacements.items())
    if not (refresh or changed or _tree_needs_sync(src_root, dst_root, ignored)):
        return
    # A failed cloud read must not delete the last working local asset tree.
    dst_root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.asset-handoff-', dir=dst_root.parent) as tmp:
        staged, previous = Path(tmp) / 'new', Path(tmp) / 'previous'
        shutil.copytree(src_root, staged)
        for name, content in replacements.items():
            target = staged / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
        if dst_root.exists():
            dst_root.rename(previous)
        try:
            staged.rename(dst_root)
        except OSError:
            if previous.exists():
                previous.rename(dst_root)
            raise


def mesh_overrides() -> dict:
    """Explicit file handoffs survive refreshes from older producer packages."""
    path = landau_input_dir() / 'mesh_overrides.json'
    if not path.is_file():
        raise FileNotFoundError(f'Missing approved mesh manifest: {path}')
    payload = json.loads(path.read_text())
    if payload.get('version') != 1:
        raise ValueError('Unsupported mesh override manifest')
    return payload['overrides']


def verified_mesh_overrides() -> dict[str, bytes]:
    repo = repo_root().resolve()
    result = {}
    for relative, spec in mesh_overrides().items():
        dst = (landau_mesh_root() / relative).resolve()
        if not dst.is_relative_to(landau_mesh_root().resolve()):
            raise ValueError(f'Mesh override escapes input directory: {relative}')
        src = (repo / spec['source']).resolve()
        # Reuse the verified consumer copy if the producer is absent or changed.
        if dst.is_file():
            content = dst.read_bytes()
            if hashlib.sha256(content).hexdigest() == spec['sha256']:
                result[relative] = content
                continue
        if not src.is_relative_to(repo) or not src.is_file():
            raise FileNotFoundError(f'Approved mesh handoff is missing: {src}')
        content = src.read_bytes()
        if hashlib.sha256(content).hexdigest() != spec['sha256']:
            raise ValueError(f'Approved mesh handoff changed; review mesh_overrides.json: {src}')
        result[relative] = content
    return result


def apply_mesh_overrides() -> None:
    for relative, content in verified_mesh_overrides().items():
        dst = landau_mesh_root() / relative
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(content)


def sync_landau_meshes(src: Path, *, refresh: bool = False) -> None:
    _sync_tree(src, landau_mesh_root(), refresh=refresh, replacements=verified_mesh_overrides())


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
    sync_landau_meshes(src_mesh_root, refresh=refresh)

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
