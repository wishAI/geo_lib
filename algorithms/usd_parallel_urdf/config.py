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
    min_pitch: float = 0.0025
    max_sample_points: int = 45000
    sample_tolerance: float = 3e-4
    fit_margin_ratio: float = 0.03
    fit_margin_min: float = 8e-4
    max_extent_ratio_xyz: Tuple[float, float, float] = (1.08, 1.08, 1.08)
    smoothing_iterations: int = 0
    smoothing_lambda: float = 0.35
    alpha_radius_ratios: Tuple[float, ...] = (0.12, 0.16, 0.22, 0.3, 0.45, 0.7, 1.05, 1.6, 2.4, 4.0)
    alpha_max_points: int = 600


@dataclass(frozen=True)
class LinkMeshPolicy:
    # None means "use the command-level --mesh-simplify-mode".
    # Supported explicit methods are lowpoly_surface, alpha_shape, obb,
    # convex_hull, and rounded_cylinder.
    mesh_method: str | None = None

    # Marching-cube alignment policy for lowpoly/voxel remeshing.
    # none: use link-local axes as-is
    # local: infer the link's own bone/root axis in link-local space
    # world: infer the link's bone/root axis in world space, then express it in link-local space
    # custom_local: use marching_axis as a link-local direction
    # custom_world: use marching_axis as a world-space direction
    marching_axis_mode: str = 'none'
    marching_axis: Tuple[float, float, float] | None = None

    # Rounded-cylinder/capsule settings. The generated STL is a cylinder along
    # the resolved axis with hemispherical caps at both ends.
    capsule_segments: int = 12
    capsule_rings: int = 4
    capsule_radius_scale: float = 0.55
    capsule_min_radius: float = 0.0015


@dataclass(frozen=True)
class MeshBuildConfig:
    mesh_simplify_mode: str = 'alpha_shape'
    max_hull_faces: int = 48
    target_hull_points: int = 24
    min_thickness: float = 0.004
    lowpoly_default: LowpolyMeshConfig = field(default_factory=LowpolyMeshConfig)
    lowpoly_link_overrides: Dict[str, LowpolyMeshConfig] = field(default_factory=dict)
    mesh_policy_default: LinkMeshPolicy = field(default_factory=LinkMeshPolicy)
    mesh_policy_overrides: Dict[str, LinkMeshPolicy] = field(default_factory=dict)


DEFAULT_LOWPOLY_CONFIG = LowpolyMeshConfig()

# Edit this file to tune STL generation without touching the mesh builder.
# `head_x` intentionally keeps a higher face budget and a tighter fit ratio
# because the head looked both too boxy and slightly oversized in earlier runs.
DEFAULT_MESH_BUILD_CONFIG = MeshBuildConfig(
    lowpoly_link_overrides={
        # Torso/body links are broad enough to tolerate a coarser marching grid
        # than the head/fingers while still preserving the silhouette.
        'body_default': replace(
            DEFAULT_LOWPOLY_CONFIG,
            max_faces=620,
            target_face_ratio=0.22,
            target_cells=(26, 24, 22, 20, 18, 16),
            cluster_scales=(0.7, 0.9, 1.1, 1.35, 1.65, 2.0, 2.4),
            smooth_sigma=0.42,
            max_extent_ratio_xyz=(1.06, 1.06, 1.06),
        ),
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
        # Fingers are thin and visually sensitive. Keep more alpha-shape input
        # points so knuckle and taper silhouettes are preserved where the USD
        # skinning data has enough samples.
        'finger_default': replace(
            DEFAULT_LOWPOLY_CONFIG,
            max_faces=1200,
            target_face_ratio=0.45,
            target_cells=(42, 38, 34, 30, 26, 22),
            cluster_scales=(0.28, 0.36, 0.46, 0.58, 0.72, 0.9, 1.1),
            smooth_sigma=0.22,
            max_grid_cells=58,
            min_pitch=0.0012,
            fit_margin_ratio=0.012,
            fit_margin_min=2e-4,
            max_extent_ratio_xyz=(1.035, 1.035, 1.035),
            alpha_radius_ratios=(0.1, 0.14, 0.18, 0.24, 0.32, 0.45, 0.7, 1.05, 1.6, 2.4, 4.0),
            alpha_max_points=900,
        ),
    },
    mesh_policy_overrides={
        # These names are group keys resolved by `resolve_link_mesh_policy`.
        # Per-link keys such as "foot_l" or "head_x" override group keys.
        'body_default': LinkMeshPolicy(marching_axis_mode='none'),
        'head_default': LinkMeshPolicy(marching_axis_mode='none'),
        'hand_default': LinkMeshPolicy(marching_axis_mode='local'),
        'foot_default': LinkMeshPolicy(marching_axis_mode='local'),
        'toe_default': LinkMeshPolicy(marching_axis_mode='local'),
        'finger_default': LinkMeshPolicy(
            mesh_method='alpha_shape',
            marching_axis_mode='local',
        ),
    },
)


def _is_finger_link(link_name: str) -> bool:
    return link_name.startswith(('thumb', 'index', 'middle', 'ring', 'pinky'))


def _is_body_link(link_name: str) -> bool:
    return link_name == 'root_x' or link_name.startswith('spine_') or link_name == 'neck_x'


def _group_keys_for_link(link_name: str) -> Tuple[str, ...]:
    if _is_finger_link(link_name):
        return ('finger_default',)
    if link_name.startswith('toes_'):
        return ('toe_default',)
    if link_name.startswith('foot_'):
        return ('foot_default',)
    if link_name.startswith('hand_'):
        return ('hand_default',)
    if link_name == 'head_x':
        return ('head_default',)
    if _is_body_link(link_name):
        return ('body_default',)
    return ()


def resolve_lowpoly_link_config(build_config: MeshBuildConfig, link_name: str) -> LowpolyMeshConfig:
    direct = build_config.lowpoly_link_overrides.get(link_name)
    if direct is not None:
        return direct
    for group_key in _group_keys_for_link(link_name):
        grouped = build_config.lowpoly_link_overrides.get(group_key)
        if grouped is not None:
            return grouped
    return build_config.lowpoly_default


def resolve_link_mesh_policy(build_config: MeshBuildConfig, link_name: str) -> LinkMeshPolicy:
    direct = build_config.mesh_policy_overrides.get(link_name)
    if direct is not None:
        return direct
    for group_key in _group_keys_for_link(link_name):
        grouped = build_config.mesh_policy_overrides.get(group_key)
        if grouped is not None:
            return grouped
    return build_config.mesh_policy_default
