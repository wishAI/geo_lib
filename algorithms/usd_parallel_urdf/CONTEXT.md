# USD Parallel USD/URDF Context

## Goal

Build a parallel URDF representation of `landau_v10.usdc` that keeps the USD skeleton 1:1 while exposing practical collision geometry for Isaac Sim and other URDF-capable simulators.

This folder now supports two geometry variants over the same joint tree:

- primitive mode
  - boxes and spheres inferred from skeleton edges
- mesh mode
  - one STL per link derived from the corresponding USD skinned surface data

## Source Asset Choice

- `landau_v10.usdc` is the articulated asset with the full 68-joint humanoid plus fingers skeleton.
- `lantu.usdc` and `landau_v8.usdc` do not expose the full hand joint set needed for this workflow.

## Design Constraints

- Keep this folder self-contained; do not depend on runtime code inside `algorithms/avp_remote`.
- Use the Isaac Sim Python environment for USD and skeleton work:
  - `/home/wishai/vscode/IsaacLab/isaaclab.sh -p ...`
- Preserve the source standing orientation.
  - The generated URDF keeps the original root basis through a fixed `base_link -> root_x` joint.
- The source USD is a skinned mesh, not a rigid articulated robot.
  - Primitive mode uses skeleton-derived proxies.
  - Mesh mode extracts surface data per link and closes it into lightweight STL collision meshes.

## Mesh Mode Strategy

- Surface extraction:
  - find the `UsdSkel.Root`
  - use `UsdSkel.Cache` and the mesh skinning query
  - skin the source mesh points into skeleton space
  - assign each source triangle to the dominant joint by summed skinning weights
- Simplification:
  - collect the assigned points in link-local space
  - default mode `lowpoly_surface` reconstructs a watertight low-poly shell from the extracted link surface samples
  - the shell is then vertex-clustered to keep the face budget low enough for collision use
- Optional alternate mode:
  - `obb` and `convex_hull` are still available for fallback/debugging, but they are not the default

The important distinction from the primitive URDF is that the STL mode is still surface-driven: the collision mesh comes from the extracted USD faces/vertices for that link, not from the skeleton edge alone.

## Key Files

- `inspect_usd_skeleton.py`
  - prints and exports the USD skeleton hierarchy
- `build_parallel_urdf.py`
  - builds the skeleton JSON
  - writes the primitive URDF and/or the mesh URDF
  - writes `mesh_collision_summary.json`
- `mesh_collision_builder.py`
  - extracts per-link surface ownership from the skinned USD mesh
  - closes and simplifies the result into STL collision meshes
- `validate_parallel_scene.py`
  - loads the source USD and a generated URDF together in Isaac
  - useful for headless scene and import validation
- `render_parallel_scene.py`
  - one-shot renderer for rest or posed comparison images
- `compare_urdf_pose_offline.py`
  - deterministic FK comparison for the generated kinematic model
- `skeleton_common.py`
  - shared extraction, pose, geometry, and URDF generation logic

## Current Validation State

- The generated URDF stands upright and preserves the source root orientation.
- The primitive URDF matches the extracted USD skeleton numerically in offline FK within about `2.1e-6 m` max root-relative position error.
- The mesh URDF shares the exact same joints and transforms, so kinematics remain aligned; only the collision geometry differs.
- The current default STL mode generates very lightweight per-link meshes.
- in the latest build, all 68 links used `skinned_lowpoly_surface`
- the latest mesh build averaged about `387` faces per link, with a max of `478`
- Headless Isaac import of the mesh URDF now completes without the earlier `Invalid PhysX transform` warnings that appeared with more detailed convex-hull meshes.
- The validator/renderer include a short post-import warmup before applying poses so mesh-backed URDF imports have time to finish settling inside Isaac.
- GUI validation still depends on running from a real desktop session with working display environment variables.
