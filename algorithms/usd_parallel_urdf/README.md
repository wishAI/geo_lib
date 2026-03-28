# USD Parallel URDF

This folder builds two URDF variants from the articulated `landau_v10.usdc` character and validates them against the source USD skeleton.

## What It Produces

- `outputs/landau_v10_skeleton.json`
  - extracted 68-joint skeleton hierarchy from `landau_v10.usdc`
  - local/world rest transforms
  - inferred URDF joint axes and demo pose
- `outputs/usd_landau_parallel.urdf`
  - primitive collision URDF
  - one link per USD skeleton joint
  - simple box/sphere colliders derived from skeleton edges
- `outputs/usd_landau_parallel_mesh.urdf`
  - mesh-backed collision URDF
  - same joints and transforms as the primitive URDF
  - each link points at its own STL under `outputs/mesh_collision_stl/`
- `outputs/mesh_collision_stl/*.stl`
  - one closed simplified STL per link
  - generated from USD surface vertices/faces assigned to the dominant skinning joint
- `outputs/mesh_collision_summary.json`
  - per-link mesh build metadata
  - includes the resolved low-poly config used for each link
- `config.py`
  - user-editable mesh generation defaults
  - includes per-link overrides such as the higher-detail `head_x` profile
- `tests/*.py`
  - fast unit tests for config resolution, mesh-fit helpers, and skeleton/URDF utilities
- `outputs/validation/offline_transform_comparison.json`
  - deterministic FK comparison for the generated kinematic model
- `outputs/validation/*.png`
  - Isaac renders for rest and posed validation scenes

## Geometry Modes

- `primitives`
  - uses boxes and spheres inferred from the skeleton only
  - fastest and most conservative collision setup
- `mesh`
  - extracts skinned surface points per link from the USD mesh
  - closes the otherwise open surface data into a low-poly STL per link
  - current pipeline is repair-first:
    - preserve the source per-face topology for each joint fragment
    - close boundary loops against the original watertight character mesh
    - if simplification breaks watertightness, fall back to the repaired closed mesh or a voxel/lowpoly remesh of it
  - current default simplification is `lowpoly_surface`
  - it reconstructs a closed low-poly shell from the extracted per-link surface samples, then fits the shell back toward the source bounds so links do not balloon outward
  - `obb` and `convex_hull` remain available as fallback/debug modes
- `both`
  - writes both URDF variants in one build pass

## Commands

Run all commands from the repo root.

Print the USD joint/link information:

```bash
/home/wishai/vscode/IsaacLab/isaaclab.sh -p algorithms/usd_parallel_urdf/inspect_usd_skeleton.py --headless
```

Generate both URDF variants:

```bash
/home/wishai/vscode/IsaacLab/isaaclab.sh -p algorithms/usd_parallel_urdf/build_parallel_urdf.py
```

Generate only the STL-backed URDF:

```bash
/home/wishai/vscode/IsaacLab/isaaclab.sh -p algorithms/usd_parallel_urdf/build_parallel_urdf.py --geometry-mode mesh
```

Tune the STL build by editing `algorithms/usd_parallel_urdf/config.py`, then rebuild with the same command.

Switch the mesh simplifier to convex hulls instead of the default low-poly surface reconstruction:

```bash
/home/wishai/vscode/IsaacLab/isaaclab.sh -p algorithms/usd_parallel_urdf/build_parallel_urdf.py --geometry-mode mesh --mesh-simplify-mode convex_hull --max-hull-faces 48 --target-hull-points 24
```

Validate the primitive URDF in Isaac:

```bash
/home/wishai/vscode/IsaacLab/isaaclab.sh -p algorithms/usd_parallel_urdf/validate_parallel_scene.py --headless
```

Validate the STL-backed URDF in Isaac:

```bash
/home/wishai/vscode/IsaacLab/isaaclab.sh -p algorithms/usd_parallel_urdf/validate_parallel_scene.py --headless --urdf-path algorithms/usd_parallel_urdf/outputs/usd_landau_parallel_mesh.urdf --output-dir algorithms/usd_parallel_urdf/outputs/validation_mesh
```

Open the GUI and keep the scene open until you close Isaac yourself:

```bash
/home/wishai/vscode/IsaacLab/isaaclab.sh -p algorithms/usd_parallel_urdf/validate_parallel_scene.py --stay-open
```

Run the deterministic FK comparison without Isaac rendering:

```bash
python3 algorithms/usd_parallel_urdf/compare_urdf_pose_offline.py
```

Run the local unit tests:

```bash
python3 -m unittest discover -s algorithms/usd_parallel_urdf/tests
```

Render a posed overview:

```bash
/home/wishai/vscode/IsaacLab/isaaclab.sh -p algorithms/usd_parallel_urdf/render_parallel_scene.py --headless --posed --view overview --output-path algorithms/usd_parallel_urdf/outputs/validation/scene_pose.png
```

## Notes

- The selected articulated asset is `algorithms/avp_remote/landau_v10.usdc`.
- The generated URDF preserves the source standing basis by keeping the original root transform through a fixed `base_link -> root_x` joint.
- The build script uses a repo-local Kit portable root under `algorithms/usd_parallel_urdf/.kit_portable/` so it does not depend on `~/Documents/Kit/shared`.
- The mesh URDF keeps the same kinematics as the primitive URDF. Only the collision/visual geometry changes.
- The mesh builder now mixes three watertight outputs depending on the link: repaired surface, low-poly remesh, and voxelized closed-surface fallback. In the latest build all 68 links ended up watertight.
- `config.py` is the supported tuning point for mesh density and fit tolerance. `head_x` ships with a tighter fit limit and a higher face budget than the rest of the body.
- The validator and renderer now wait briefly after URDF import before applying poses. That warmup avoids an intermittent mesh-import timing race in Isaac where the articulated pose could be driven before mesh-backed links had fully settled.
- GUI validation still requires launching from a real desktop session with `DISPLAY` or `WAYLAND_DISPLAY` available.
