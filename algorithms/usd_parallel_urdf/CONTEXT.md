# USD Parallel USD/URDF Context

## Goal

Build a simplified URDF that tracks the `landau_v10.usdc` articulated skeleton closely enough to:

- expose collisions in Isaac Sim and other URDF-capable sims
- keep the skeleton hierarchy 1:1 with the USD
- drive the URDF joints and apply the same pose back onto the USD skeleton
- compare root-relative link transforms between the two representations

## Source Asset Choice

- `landau_v10.usdc` is the articulated asset with the full 68-joint humanoid + finger skeleton.
- `lantu.usdc` and `landau_v8.usdc` do not expose the full hand joint set needed for this workflow.

## Design Constraints

- Do not import code from `algorithms/avp_remote`; keep this folder self-contained.
- Use the Isaac Sim Python environment for all USD/Skeleton work:
  - `/home/wishai/vscode/IsaacLab/isaaclab.sh -p ...`
- The source USD is a skinned mesh, not a rigid articulated robot.
  - The URDF therefore uses simplified box/sphere colliders built from skeleton edges.
  - No attempt is made to recover exact watertight collision meshes from the skinned surface.

## Key Files

- `inspect_usd_skeleton.py`
  - one-shot printer/exporter for the USD skeleton hierarchy
- `build_parallel_urdf.py`
  - extracts the USD skeleton and writes the generated URDF
- `validate_parallel_scene.py`
  - attempts the full Isaac paired-scene validation path
  - now waits for Kit app readiness, uses direct stage commands for the ground plane, and avoids the prior `World.reset()` crash path
- `compare_urdf_pose_offline.py`
  - deterministic FK comparison between the generated URDF and the extracted USD skeleton records
  - this is the reliable numeric validation artifact right now
- `render_parallel_scene.py`
  - one-shot headless renderer for a single scene view
  - useful for probing camera placement and current USD posing behavior
- `skeleton_common.py`
  - shared extraction, pose, geometry, and URDF generation logic

## Current Validation State

- The URDF imports into Isaac with `fix_base=True` and does not collapse under gravity.
- The generated URDF matches the extracted USD skeleton numerically in offline FK within about `2.1e-6 m` max root-relative position error across 68 links.
- Headless Isaac renders the rest scene successfully.
- The validator/renderer use the full Isaac Sim experience instead of the IsaacLab app kits so GUI runs do not inherit the `sim-arm` / `standalone` source-extension warnings.
- Applying the demo pose back onto the skinned USD by writing skeleton rest transforms causes visible mesh breakup in the current headless Isaac path.
- If a future iteration needs a clean posed USD visual result, the next thing to investigate is using animation / skinning-compatible joint posing APIs rather than replacing rest transforms.
