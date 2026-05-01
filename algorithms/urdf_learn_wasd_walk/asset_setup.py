from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

from .asset_paths import (
    default_landau_source_mesh_root,
    default_landau_source_skeleton_json,
    default_landau_source_texture_dir,
    default_landau_source_urdf,
    default_landau_source_usd,
    inputs_dir,
    landau_fixed_urdf_path,
    landau_input_dir,
    landau_mesh_root,
    landau_source_urdf_path,
    landau_skeleton_json_path,
    landau_texture_dir,
    landau_urdf_path,
    landau_usd_path,
)
from .scripts.rescale_landau_mass import MASS_RESCALE_VERSION, rescale_landau_urdf


@dataclass(frozen=True)
class PreparedAssetPaths:
    input_dir: Path
    urdf_path: Path
    mesh_root: Path
    usd_path: Path
    skeleton_json_path: Path
    texture_dir: Path


def _asset_manifest_path() -> Path:
    return landau_input_dir() / "asset_manifest.json"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_tree(root: Path) -> dict[str, object]:
    if not root.exists():
        return {"root": str(root), "exists": False, "file_count": 0, "sha256": None}
    digest = hashlib.sha256()
    file_count = 0
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rel_path = path.relative_to(root).as_posix()
        digest.update(rel_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256_file(path).encode("utf-8"))
        digest.update(b"\0")
        file_count += 1
    return {
        "root": str(root),
        "exists": True,
        "file_count": file_count,
        "sha256": digest.hexdigest(),
    }


def _source_manifest(
    *,
    src_urdf: Path,
    src_mesh_root: Path,
    src_usd: Path,
    src_skeleton_json: Path,
    src_texture_dir: Path,
) -> dict[str, object]:
    return {
        "mass_rescale_version": MASS_RESCALE_VERSION,
        "urdf": {"path": str(src_urdf), "sha256": _sha256_file(src_urdf)},
        "mesh_tree": _hash_tree(src_mesh_root),
        "usd": {"path": str(src_usd), "sha256": _sha256_file(src_usd)},
        "skeleton_json": {"path": str(src_skeleton_json), "sha256": _sha256_file(src_skeleton_json)},
        "texture_tree": _hash_tree(src_texture_dir),
    }


def _load_manifest(path: Path) -> dict[str, object] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def prepare_landau_inputs(refresh: bool = False) -> PreparedAssetPaths:
    """Copy the custom URDF handoff into this package's inputs directory."""

    src_urdf = default_landau_source_urdf()
    src_mesh_root = default_landau_source_mesh_root()
    src_usd = default_landau_source_usd()
    src_skeleton_json = default_landau_source_skeleton_json()
    src_texture_dir = default_landau_source_texture_dir()
    dst_dir = landau_input_dir()
    dst_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = _asset_manifest_path()

    if not src_urdf.exists():
        raise FileNotFoundError(f"Missing source URDF: {src_urdf}")
    if not src_mesh_root.exists():
        raise FileNotFoundError(f"Missing source mesh folder: {src_mesh_root}")
    if not src_usd.exists():
        raise FileNotFoundError(f"Missing source USD: {src_usd}")
    if not src_skeleton_json.exists():
        raise FileNotFoundError(f"Missing source skeleton JSON: {src_skeleton_json}")
    if not src_texture_dir.exists():
        raise FileNotFoundError(f"Missing source texture folder: {src_texture_dir}")

    source_manifest = _source_manifest(
        src_urdf=src_urdf,
        src_mesh_root=src_mesh_root,
        src_usd=src_usd,
        src_skeleton_json=src_skeleton_json,
        src_texture_dir=src_texture_dir,
    )
    existing_manifest = _load_manifest(manifest_path)
    refresh_required = refresh or existing_manifest != source_manifest

    if refresh_required:
        shutil.copy2(src_urdf, landau_source_urdf_path())
        rescale_landau_urdf(landau_source_urdf_path(), landau_fixed_urdf_path())
    elif not landau_fixed_urdf_path().exists():
        rescale_landau_urdf(landau_source_urdf_path(), landau_fixed_urdf_path())

    if refresh_required and landau_mesh_root().exists():
        shutil.rmtree(landau_mesh_root())
    if refresh_required or not landau_mesh_root().exists():
        shutil.copytree(src_mesh_root, landau_mesh_root(), dirs_exist_ok=True)

    if refresh_required or not landau_usd_path().exists():
        shutil.copy2(src_usd, landau_usd_path())

    if refresh_required or not landau_skeleton_json_path().exists():
        shutil.copy2(src_skeleton_json, landau_skeleton_json_path())

    if refresh_required and landau_texture_dir().exists():
        shutil.rmtree(landau_texture_dir())
    if refresh_required or not landau_texture_dir().exists():
        shutil.copytree(src_texture_dir, landau_texture_dir(), dirs_exist_ok=True)

    manifest_path.write_text(json.dumps(source_manifest, indent=2, sort_keys=True), encoding="utf-8")

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
