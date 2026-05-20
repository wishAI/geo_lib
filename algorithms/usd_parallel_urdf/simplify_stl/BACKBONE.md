# STL Simplification Backbone

This document describes how `usd_parallel_urdf/simplify_stl/` turns a skinned USD character into one collision STL per generated URDF link. It is intended to preserve the design model, method steps, and failure boundaries, not to be a command recipe.

## Inputs And Ownership

The source character is an articulated USD/USDC asset with a `UsdSkel.Skeleton` and one or more skinned `UsdGeom.Mesh` prims below the same `UsdSkel.Root`. The generated URDF keeps one link per skeleton joint, so the mesh builder needs to decide which source surface belongs to each joint.

The builder evaluates the source mesh in the skeleton rest pose, reads skinning influences per point, triangulates every source polygon, and assigns each triangle to the joint with the largest summed skinning weight across that triangle. This produces, per generated link:

- A surface point cloud in world space.
- A triangle soup in world space.
- A fragment mesh that preserves the original per-face topology for repair.
- A primitive fallback derived from the skeleton edge when surface data is missing.

Every final STL is written in the link-local URDF frame. The URDF link itself owns the transform; STL vertices should therefore stay local to the link and usually use zero visual/collision origin.

## Configuration Surface

`config.py` is the supported tuning point. It separates two ideas:

- `LowpolyMeshConfig` controls precision and fitting: marching-cube grid density, alpha-shape radii, minimum voxel pitch, face budget, clustering scales, smoothing, repair limits, and post-fit bounds.
- `LinkMeshPolicy` controls method and alignment: explicit mesh method override, marching-axis alignment mode, optional custom axis, and rounded-cylinder/capsule resolution.

Both config types are resolved per link. Exact link names win first, then group keys such as `body_default`, `head_default`, `hand_default`, `foot_default`, `toe_default`, and `finger_default`, then global defaults. This is where head/body precision is adjusted:

```python
DEFAULT_MESH_BUILD_CONFIG = MeshBuildConfig(
    lowpoly_link_overrides={
        "body_default": replace(DEFAULT_LOWPOLY_CONFIG, target_cells=(26, 24, 22, 20, 18, 16)),
        "head_x": replace(DEFAULT_LOWPOLY_CONFIG, target_cells=(30, 28, 26, 24, 22, 20, 18)),
    },
)
```

Method and axis policy are configured in the same object:

```python
MeshBuildConfig(
    mesh_policy_overrides={
        "foot_default": LinkMeshPolicy(marching_axis_mode="local"),
        "toe_default": LinkMeshPolicy(marching_axis_mode="local"),
        "finger_default": LinkMeshPolicy(mesh_method="rounded_cylinder", marching_axis_mode="local"),
        "thumb2_l": LinkMeshPolicy(marching_axis_mode="custom_local", marching_axis=(1.0, 0.0, 0.0)),
    },
)
```

`marching_axis_mode` has five meanings. `none` uses the existing link-local frame. `local` infers the link's child or parent bone vector in link-local coordinates. `world` infers the same vector in world coordinates and converts it into link-local coordinates. `custom_local` uses `marching_axis` as a link-local vector. `custom_world` uses `marching_axis` as a world vector and converts it into link-local coordinates.

## Method Rules

All methods follow the same output rules:

- Work in link-local coordinates after source triangles have been assigned to a dominant skinning joint.
- Respect the resolved `LinkMeshPolicy`; exact link policy wins over body-part policy, and body-part policy wins over global defaults.
- Keep STL vertices local to the URDF link. Do not bake the link world transform into the STL.
- Fit generated surfaces back toward the source link bounds unless the method is a deliberate primitive fallback.
- Record the resolved method, policy, face counts, fit details, and axis policy in `<asset>_mesh_collision_summary.json`.
- Fall back to simpler geometry instead of emitting empty or invalid STL data.

## Low-Poly Surface Method

The default `lowpoly_surface` method is surface-driven. It samples source triangle vertices, midpoints, and centroids, removes duplicate samples, then searches candidate marching-cube resolutions from `target_cells`. For each candidate:

- A voxel grid is placed around the link-local point cloud.
- Surface samples mark occupied cells.
- Morphological closing and hole filling turn sparse samples into a solid occupancy field.
- A Gaussian-smoothed scalar field is extracted with marching cubes.
- The resulting mesh is cleaned, vertex-clustered at several `cluster_scales`, and fit back toward the original source bounds.

The method selects the first candidate that satisfies `max_faces`, or the best lower-face candidate found. The output metadata records pitch, grid shape, face count, cluster scale, watertightness, and fit scale. These values are emitted in `<asset>_mesh_collision_summary.json`.

## Repair And Voxel Fallback

Before low-poly fallback, the builder tries a repair-first path for skinned fragments with enough topology. It preserves source faces for the link fragment, closes boundary loops against the original watertight character mesh, and simplifies while respecting the link config. If that repaired mesh exists, the builder tries to voxel-remesh the repaired closed surface before falling back to sparse-sample low-poly reconstruction.

This ordering matters. Sparse marching cubes are good for lightweight collision shells, but topology-preserving repair usually better respects thin surfaces and holes. Voxel remeshing of repaired input is the middle ground: watertightness from repair, simplified regularity from marching cubes.

## Alpha Shape Method

`alpha_shape` is a point-cloud boundary method for links where a convex hull is too loose but full low-poly marching cubes is unnecessary. It uses the same link-local point ownership as `lowpoly_surface`, then:

- Deduplicates and downsamples the link-local point cloud using `sample_tolerance` and `alpha_max_points`.
- Builds a 3D Delaunay tetrahedralization with SciPy.
- Tests candidate tetrahedra by circumsphere radius using each configured `alpha_radius_ratios` value times the point-cloud bounding-box diagonal.
- Keeps tetrahedra at or below the current radius and emits triangles that belong to exactly one kept tetrahedron.
- Cleans and orients the boundary mesh, then fits it back toward the source point bounds.

Small radius ratios preserve tighter concavities but can fragment or drop sparse regions. Large ratios approach a convex hull. Configure exact links first when a body part needs different tightness; avoid using alpha shapes for very sparse or mostly planar point sets because Delaunay may fail or produce a box fallback.

## Rounded Cylinder Finger Method

`rounded_cylinder` is a synthetic method for finger-like links. It does not preserve source surface detail. It fits a capsule around the link's resolved axis: a cylinder in the center and hemispherical caps on both ends.

The point cloud is first transformed into the configured marching basis, so the resolved axis becomes local `+X`. The capsule radius comes from the maximum transverse extent, scaled by `capsule_radius_scale` and clamped by `capsule_min_radius` and `min_thickness`. The capsule length spans the original axial bounds. The mesh is then transformed back into the link-local frame before STL export.

This method is intentionally stable for fingers because finger collision usually benefits more from smooth, predictable volumes than from preserving knuckle mesh detail. If a specific finger segment needs source-driven detail, override that link back to `mesh_method="lowpoly_surface"`.

## Convex Hull And OBB Methods

`convex_hull` downsamples the link-local point cloud, asks SciPy for a convex hull, and uses it only when the hull fits the configured `max_hull_faces` cap. If it cannot build a valid low-face hull, it falls back to an oriented box.

`obb` is the simple oriented-box path. It computes principal axes from the link-local point cloud and boxes the local extents with `min_thickness` clamps. Use it for debugging, conservative broad collision, or links with too little valid surface data.

## Axis Alignment Model

Marching cubes and capsule fitting are sensitive to the coordinate frame used for gridding. The builder can rotate link-local points into a temporary basis before reconstruction:

- The resolved axis becomes temporary `+X`.
- Reconstruction or capsule fitting happens in that temporary basis.
- Output vertices are rotated back into the original link-local frame.

This does not change URDF joint transforms. It only changes how the collision STL is fitted. Use it when a long thin body part is diagonal in link-local coordinates and an axis-aligned voxel grid would waste resolution or inflate the bounds.

## Output Contract

The generated STL for each link must be local to that link, should be closed when possible, and should not exceed source bounds by more than the configured fit allowance. The mesh summary is part of the output contract: it records the resolved low-poly config, resolved mesh policy, selected method, axis mode, axis vector, counts, and fit metadata for each link.

The primitive URDF and mesh URDF share the same skeleton and transforms. Low-poly STL generation only changes link geometry, not pose mapping or joint semantics.
