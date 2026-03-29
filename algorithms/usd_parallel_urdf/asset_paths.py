from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


DEFAULT_INPUT_USD_BASENAME = 'landau_v10.usdc'


def module_root() -> Path:
    return Path(__file__).resolve().parent


def inputs_dir() -> Path:
    return module_root() / 'inputs'


def legacy_default_usd_path() -> Path:
    return module_root().parents[1] / 'algorithms' / 'avp_remote' / DEFAULT_INPUT_USD_BASENAME


def default_usd_path() -> Path:
    candidate = inputs_dir() / DEFAULT_INPUT_USD_BASENAME
    if candidate.exists():
        return candidate
    legacy = legacy_default_usd_path()
    if legacy.exists():
        return legacy
    return candidate


def asset_tag(usd_path: Path) -> str:
    stem = usd_path.stem.strip() or 'asset'
    return re.sub(r'[^A-Za-z0-9._-]+', '_', stem)


@dataclass(frozen=True)
class AssetPaths:
    usd_path: Path
    output_dir: Path
    asset_tag: str
    primitive_robot_name: str
    mesh_robot_name: str
    skeleton_json: Path
    primitive_urdf: Path
    mesh_urdf: Path
    mesh_output_dir: Path
    mesh_summary: Path
    primitive_validation_dir: Path
    mesh_validation_dir: Path
    mesh_validation_gallery_dir: Path
    animation_capture_dir: Path


def resolve_asset_paths(
    usd_path: Path,
    output_dir: Path,
    robot_name: str | None = None,
    mesh_robot_name: str | None = None,
    mesh_output_dir: Path | None = None,
) -> AssetPaths:
    tag = asset_tag(usd_path)
    primitive_robot_name = robot_name or f'{tag}_parallel'
    resolved_mesh_robot_name = mesh_robot_name or f'{primitive_robot_name}_mesh'
    return AssetPaths(
        usd_path=usd_path,
        output_dir=output_dir,
        asset_tag=tag,
        primitive_robot_name=primitive_robot_name,
        mesh_robot_name=resolved_mesh_robot_name,
        skeleton_json=output_dir / f'{tag}_skeleton.json',
        primitive_urdf=output_dir / f'{primitive_robot_name}.urdf',
        mesh_urdf=output_dir / f'{resolved_mesh_robot_name}.urdf',
        mesh_output_dir=mesh_output_dir or (output_dir / 'mesh_collision_stl' / tag),
        mesh_summary=output_dir / f'{tag}_mesh_collision_summary.json',
        primitive_validation_dir=output_dir / f'validation_{tag}',
        mesh_validation_dir=output_dir / f'validation_mesh_{tag}',
        mesh_validation_gallery_dir=output_dir / f'validation_mesh_{tag}_gallery',
        animation_capture_dir=output_dir / f'animation_{tag}',
    )
