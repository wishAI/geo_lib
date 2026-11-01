# USD Parallel URDF

This folder builds two URDF variants from an articulated `.usd`/`.usdc` character and validates them against the source USD skeleton.

## What It Produces

- `inputs/landau_v10.usdc`
  - canonical local copy of the current source asset
  - new assets should also go under `inputs/`
- `outputs/<asset>_skeleton.json`
  - extracted 68-joint skeleton hierarchy from the selected input asset
  - local/world rest transforms
  - inferred URDF joint axes and demo pose
- `outputs/<asset>_parallel.urdf`
  - primitive collision URDF
  - one link per USD skeleton joint
  - simple box/sphere colliders derived from skeleton edges
- `outputs/urdf_packages/<asset>/`
  - self-contained URDF package for drag-and-drop robot viewers
  - contains `<asset>_parallel_mesh.urdf` plus the referenced `mesh_collision_stl/<asset>/` folder
  - this is the default mesh URDF path used by the repo launcher
- `outputs/urdf_packages/<asset>/<asset>_mesh_collision_summary.json`
  - per-link mesh build metadata
  - includes the resolved low-poly config used for each link
- `config.py`
  - user-editable mesh generation defaults
  - includes per-link and per-body-part precision/method/axis overrides
  - includes `body_default`, `head_x`, `foot_default`, `toe_default`, `hand_default`, and `finger_default`
- `simplify_stl/`
  - STL simplification, repair, postprocess packaging code, and `BACKBONE.md`
- `pose_mapping.md`
  - architecture notes for how USD skeleton pose mapping is preserved in URDF
- `tests/*.py`
  - fast unit tests for config resolution, mesh-fit helpers, and skeleton/URDF utilities
- `outputs/validation_<asset>/offline_transform_comparison.json`
  - deterministic FK comparison for the generated kinematic model
- `outputs/validation_<asset>/*.png`
  - Isaac renders for rest and posed validation scenes

## Geometry Modes

- `primitives`
  - uses boxes and spheres inferred from the skeleton only
  - fastest and most conservative collision setup
- `mesh`
  - extracts skinned surface points per link from the USD mesh
  - closes the otherwise open surface data into an STL per link
  - current default simplification is `alpha_shape`
  - it builds a Delaunay alpha complex from the extracted per-link surface samples, exports the boundary triangles, then fits the shell back toward the source bounds so links do not balloon outward
  - fragmented or non-watertight alpha candidates are rejected; the builder keeps searching larger alpha radii and falls back to the alpha-limit convex hull when needed
  - `lowpoly_surface`, `obb`, and `convex_hull` remain available as fallback/debug modes
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

The current supported knobs are:

- Head precision: edit the exact `head_x` entry in `DEFAULT_MESH_BUILD_CONFIG.lowpoly_link_overrides`.
- Body precision: edit the `body_default` entry; it applies to `root_x`, `spine_*`, and `neck_x` unless an exact link override exists.
- Finger method: edit `finger_default` in `mesh_policy_overrides`; it currently uses `mesh_method="alpha_shape"`.
- Alpha-shape tightness: edit `alpha_radius_ratios` and `alpha_max_points` in the resolved `LowpolyMeshConfig`.
- Foot, toe, hand, and finger alpha-shape axis alignment: edit `foot_default`, `toe_default`, `hand_default`, or `finger_default` in `mesh_policy_overrides`.
- Custom alpha-shape axis: use `LinkMeshPolicy(marching_axis_mode="custom_local", marching_axis=(...))` or `custom_world` on an exact link such as `foot_l`.

Generate/update the default URDF outputs through the repo launcher:

```bash
./geo usd build
```

Generate/update only the STL-backed mesh URDF:

```bash
./geo usd build-mesh
```

Build explicitly with alpha shapes:

```bash
/home/wishai/vscode/IsaacLab/isaaclab.sh -p algorithms/usd_parallel_urdf/build_parallel_urdf.py --geometry-mode mesh --mesh-simplify-mode alpha_shape
```

Switch the mesh simplifier to convex hulls instead of the default alpha-shape reconstruction:

```bash
/home/wishai/vscode/IsaacLab/isaaclab.sh -p algorithms/usd_parallel_urdf/build_parallel_urdf.py --geometry-mode mesh --mesh-simplify-mode convex_hull --max-hull-faces 48 --target-hull-points 24
```

Validate the primitive URDF in Isaac:

```bash
/home/wishai/vscode/IsaacLab/isaaclab.sh -p algorithms/usd_parallel_urdf/validate_parallel_scene.py --headless
```

Validate the STL-backed URDF in Isaac:

```bash
/home/wishai/vscode/IsaacLab/isaaclab.sh -p algorithms/usd_parallel_urdf/validate_parallel_scene.py --headless --urdf-path algorithms/usd_parallel_urdf/outputs/urdf_packages/landau_v10/landau_v10_parallel_mesh.urdf --output-dir algorithms/usd_parallel_urdf/outputs/validation_mesh_landau_v10
```

Open the GUI and keep the scene open until you close Isaac yourself:

```bash
/home/wishai/vscode/IsaacLab/isaaclab.sh -p algorithms/usd_parallel_urdf/validate_parallel_scene.py --stay-open
```

Play a looping synchronized animation in Isaac GUI so you can inspect the USD and URDF side by side:

```bash
/home/wishai/vscode/IsaacLab/isaaclab.sh -p algorithms/usd_parallel_urdf/play_parallel_animation.py --urdf-path algorithms/usd_parallel_urdf/outputs/urdf_packages/landau_v10/landau_v10_parallel_mesh.urdf --animation-clip walk_cycle --camera-view walk_side
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
/home/wishai/vscode/IsaacLab/isaaclab.sh -p algorithms/usd_parallel_urdf/render_parallel_scene.py --headless --posed --view overview --output-path algorithms/usd_parallel_urdf/outputs/validation_landau_v10/scene_pose.png
```

## Notes

- The canonical selected articulated asset is `algorithms/usd_parallel_urdf/inputs/landau_v10.usdc`.
- Output names are now derived from the input asset stem so multiple characters can coexist without overriding each other.
- The generated URDF preserves the source standing basis by keeping the original root transform through a fixed `base_link -> root_x` joint.
- The build script uses a repo-local Kit portable root under `algorithms/usd_parallel_urdf/.kit_portable/` so it does not depend on `~/Documents/Kit/shared`.
- The mesh URDF keeps the same kinematics as the primitive URDF. Only the collision/visual geometry changes.
- The mesh builder now mixes three watertight outputs depending on the link: repaired surface, low-poly remesh, and voxelized closed-surface fallback. In the latest build all 68 links ended up watertight.
- `config.py` is the supported tuning point for mesh density and fit tolerance. `head_x` ships with a tighter fit limit and a higher face budget than the rest of the body.
- The validator, renderer, and animation player now wait briefly after URDF import before applying poses. That warmup avoids an intermittent mesh-import timing race in Isaac where the articulated pose could be driven before mesh-backed links had fully settled.
- The current URDF has `67` controllable revolute joints plus the fixed `base_link -> root_x` root joint.
- GUI validation still requires launching from a real desktop session with `DISPLAY` or `WAYLAND_DISPLAY` available.
