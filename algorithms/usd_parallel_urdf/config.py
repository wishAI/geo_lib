from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict, Tuple


@dataclass(frozen=True)
class LowpolyMeshConfig:
    target_face_ratio: float = 0.25
    max_faces: int = 1200
    max_hole_edges: int = 500
    component_keep_area_ratio: float = 0.05
    planar_deviation_ratio: float = 0.05
    force_fill_max_edges: int = 10
    target_cells: Tuple[int, ...] = (24, 22, 20, 18, 16, 14, 12)
    cluster_scales: Tuple[float, ...] = (0.55, 0.75, 0.95, 1.15, 1.35, 1.65, 2.0, 2.4)
    smooth_sigma: float = 0.45
    closing_iterations: int = 1
    dilation_iterations: int = 0
    padding_cells: int = 1
    max_grid_cells: int = 44
    max_sample_points: int = 45000
    sample_tolerance: float = 3e-4
    fit_margin_ratio: float = 0.03
    fit_margin_min: float = 8e-4
    max_extent_ratio_xyz: Tuple[float, float, float] = (1.08, 1.08, 1.08)
    smoothing_iterations: int = 0
    smoothing_lambda: float = 0.35


@dataclass(frozen=True)
class MeshBuildConfig:
    mesh_simplify_mode: str = 'lowpoly_surface'
    max_hull_faces: int = 48
    target_hull_points: int = 24
    min_thickness: float = 0.004
    lowpoly_default: LowpolyMeshConfig = field(default_factory=LowpolyMeshConfig)
    lowpoly_link_overrides: Dict[str, LowpolyMeshConfig] = field(default_factory=dict)


DEFAULT_LOWPOLY_CONFIG = LowpolyMeshConfig()

# Edit this file to tune STL generation without touching the mesh builder.
# `head_x` intentionally keeps a higher face budget and a tighter fit ratio
# because the head looked both too boxy and slightly oversized in earlier runs.
DEFAULT_MESH_BUILD_CONFIG = MeshBuildConfig(
    lowpoly_link_overrides={
        'head_x': replace(
            DEFAULT_LOWPOLY_CONFIG,
            max_faces=760,
            target_face_ratio=0.3,
            target_cells=(30, 28, 26, 24, 22, 20, 18),
            cluster_scales=(0.4, 0.55, 0.7, 0.85, 1.0, 1.15, 1.3, 1.5, 1.75),
            smooth_sigma=0.32,
            fit_margin_ratio=0.015,
            fit_margin_min=5e-4,
            max_extent_ratio_xyz=(1.04, 1.04, 1.04),
        ),
    }
)


def resolve_lowpoly_link_config(build_config: MeshBuildConfig, link_name: str) -> LowpolyMeshConfig:
    return build_config.lowpoly_link_overrides.get(link_name, build_config.lowpoly_default)
