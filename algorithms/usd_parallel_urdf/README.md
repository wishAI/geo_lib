# USD Parallel URDF

This folder builds a simplified collision-ready URDF from the articulated `landau_v10.usdc` character and validates it next to the original USD.

## What It Produces

- `outputs/landau_v10_skeleton.json`
  - extracted 68-joint skeleton hierarchy from `landau_v10.usdc`
  - local/world rest transforms
  - inferred URDF joint axes and demo pose
- `outputs/usd_landau_parallel.urdf`
  - simplified parallel URDF
  - one link per USD skeleton joint
  - simple box/sphere collision geometry derived from the skeleton edges
- `outputs/validation/offline_transform_comparison.json`
  - deterministic FK comparison between the generated URDF and the extracted USD skeleton records
- `outputs/validation/scene_rest.png`
  - headless Isaac rest-pose render of the USD asset next to the generated scene setup
- `outputs/validation/scene_pose.png`
  - posed headless Isaac render artifact
  - currently useful as a failure signal: driving the skinned USD by overwriting skeleton rest transforms in headless Isaac breaks the mesh skinning

## Why This URDF Is Approximate

The source asset is a skinned character USD, not a rigid articulated robot. It exposes the joint hierarchy and rest transforms, but not a ready-made rigid-body collider decomposition. This implementation therefore keeps the joint layout 1:1 with the USD skeleton and uses simplified collision boxes/spheres built from parent-child bone segments.

That was the more reliable path for Isaac / URDF import than trying to recover watertight collision meshes from the skinned surface.

## Commands

Run all commands from the repo root.

Print the USD joint/link information:

```bash
/home/wishai/vscode/IsaacLab/isaaclab.sh -p algorithms/usd_parallel_urdf/inspect_usd_skeleton.py --headless
```

Generate the simplified URDF:

```bash
/home/wishai/vscode/IsaacLab/isaaclab.sh -p algorithms/usd_parallel_urdf/build_parallel_urdf.py --headless
```

Try the Isaac-side paired scene validation:

```bash
/home/wishai/vscode/IsaacLab/isaaclab.sh -p algorithms/usd_parallel_urdf/validate_parallel_scene.py
```

Run the same validator headless:

```bash
/home/wishai/vscode/IsaacLab/isaaclab.sh -p algorithms/usd_parallel_urdf/validate_parallel_scene.py --headless
```

Generate the deterministic URDF/USD FK comparison without Isaac rendering:

```bash
python3 algorithms/usd_parallel_urdf/compare_urdf_pose_offline.py
```

Render a single posed overview or hands shot in headless Isaac:

```bash
/home/wishai/vscode/IsaacLab/isaaclab.sh -p algorithms/usd_parallel_urdf/render_parallel_scene.py --headless --posed --view overview --output-path algorithms/usd_parallel_urdf/outputs/validation/scene_pose.png
```

## Notes

- The selected articulated asset is `algorithms/avp_remote/landau_v10.usdc`.
- The imported URDF is fixed-base on purpose so it does not collapse or drift while validating collisions and joint mapping.
- The validator now forces Kit `--portable-root` into `algorithms/usd_parallel_urdf/.kit_portable/...` so it does not depend on `~/Documents/Kit/shared`.
- The validator/renderer now use Isaac Sim's full app experience and wait for `app ready` before touching the stage, which avoids the prior GUI window setup errors and the reset-time crash path.
- The generated joint axes are heuristic because the source skeleton does not encode URDF-style 1-DOF joint axes.
- The offline FK comparison currently shows the generated URDF matches the extracted USD skeleton within about `2.1e-6 m` max root-relative position error.
- Headless Isaac imports and renders the asset, but directly posing the skinned USD by rewriting skeleton rest transforms is not yet a safe way to produce a clean posed visual result.
