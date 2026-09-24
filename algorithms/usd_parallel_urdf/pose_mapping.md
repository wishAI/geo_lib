# URDF/USD Pose Mapping Backbone

This document describes how pose meaning moves between the source USD skeleton and the generated URDF. It focuses on the conceptual pipeline and invariants a coding agent must preserve when modifying the code.

## Shared Skeleton Contract

The generated URDF is parallel to the USD skeleton. Each USD skeleton joint becomes one URDF link, and each parent-child skeleton edge becomes one URDF joint. A fixed `base_link -> root_x` joint preserves the source standing basis instead of normalizing the character into a new root orientation.

The important invariant is that the primitive URDF and mesh URDF have identical kinematics. Primitive boxes, mesh STLs, and repaired low-poly surfaces are only geometry variants. They must not change joint names, parent-child order, rest transforms, or inferred joint axes.

## Extraction Layer

`skeleton_common.py` extracts one record per USD skeleton joint. Each record stores:

- Joint index and skeleton path.
- Link name and parent name.
- Child names.
- Local and world rest matrices.
- Local/world rest translations.
- Inferred URDF axis.
- Demo pose values used for validation and rendering.

The records are serialized to `<asset>_skeleton.json`. That JSON is the bridge artifact between USD extraction, URDF generation, offline comparison, validation renders, animation playback, and downstream training/AVP asset setup.

## Rest Transform Mapping

For each skeleton edge, the child rest local transform defines the URDF joint origin relative to the parent link. The URDF link is named after the USD child joint. The generated joint axis is expressed in the parent/link frame expected by URDF.

The root is special. USD assets can carry a standing basis that is not equivalent to a clean robotics `base_link`. The generator keeps that basis by creating a fixed root joint from `base_link` to `root_x`. Downstream code should treat this as an intentional alignment anchor, not an accidental offset.

## Axis Inference

The USD skeleton provides transforms but not robot-style revolute joint semantics. `skeleton_common.py` infers one axis per generated URDF joint from link names, local frames, child vectors, and mirrored-pair expectations. This axis is the actuation basis for both URDF poses and authored USD skeleton animation.

The axis inference layer is heuristic. It must be validated numerically because small axis errors can make a pose look static or mirrored incorrectly. Stretch links, feet, toes, shoulders, hands, and fingers are particularly sensitive because their visually obvious bend direction may not be the same as their bone direction.

## Pose Application

A pose is represented as scalar values keyed by joint/link name. Applying a pose means:

- Start from the extracted rest local matrix for every joint.
- For each posed joint, build an axis-angle delta from the inferred axis and scalar value.
- Compose that delta with the joint's rest local rotation.
- Recompute world transforms down the skeleton hierarchy.

The USD path and URDF path must consume the same resolved scalar pose. USD validation authors local transforms into `UsdSkelAnimation`; URDF validation drives articulation joints or computes FK from the generated URDF. A mismatch should be treated as a mapping bug, not a mesh-generation bug.

## Semantic Pose Layer

`pose_semantics.py` separates pose intent from rig actuation. Semantic channels describe motions such as shoulder abduction, elbow flexion, wrist yaw, leg swing, foot pitch, and toe roll. The rig adapter maps those semantic channels into concrete joint scalar values using the current asset's naming and sign conventions.

This separation matters for mirrored limbs. A symmetric semantic pose does not imply identical signed joint values. Some mirrored pairs need equal signs, others need opposite signs. The adapter owns that choice so pose presets can express intent without duplicating rig-specific sign rules.

## Validation Layers

Pose correctness is checked in three layers:

- Rest baseline: USD and URDF world transforms should match at zero pose before any posed test matters.
- FK comparison: applying the same pose to both representations should produce matching root-relative joint transforms.
- Visual validation: Isaac renders or animation playback should confirm that the transformed geometry follows the expected skeleton.

The offline comparison is the most deterministic signal. Screenshots are useful after the numeric mapping is already sane, but they should not be the primary debugging mechanism for axis or sign errors.

## Failure Boundaries

If USD and URDF joint positions diverge while the mesh URDF and primitive URDF agree, the bug is in pose mapping or URDF transform generation. If the skeleton transforms agree but collision shapes look wrong, the bug is in geometry generation or STL fitting. If only mirrored limbs look wrong, the likely source is axis inference or semantic sign mapping. If only a single rendered mesh is offset but its link transform is correct, the likely source is STL local-frame export or visual/collision origin.

Keeping these boundaries clear prevents low-poly mesh changes from being blamed for kinematic errors and prevents pose mapping changes from being used to hide bad STL local frames.

## Downstream Asset Flow

The generated outputs under `algorithms/usd_parallel_urdf/outputs/` are the source artifacts. Downstream modules such as AVP and walking training copy or refresh those artifacts into their own `inputs/landau_v10/` directories. The copied files should be treated as deployment snapshots, not as the canonical generation source.

The canonical refresh set is the mesh URDF, the STL tree, the source USD, the skeleton JSON, and textures. The manifest hash used by downstream setup code exists to detect drift between generated outputs and local runtime inputs.
