# Pose Mapping Redesign Plan

## Current Design

The current mapping is a direct joint-angle pipeline, not a semantic pose pipeline.

- The USD skeleton rest local matrices are extracted and each joint gets one inferred revolute axis from heuristics in `skeleton_common.py`.
- Pose presets are plain `joint_name -> angle` dictionaries.
- Applying a pose means multiplying an axis-angle delta into each joint's rest local rotation.
- The same scalar angles are sent to the URDF articulation drives.
- The USD path is driven by authored `UsdSkelAnimation` local TRS.

This works when a named joint angle has the same motion meaning across both sides of the rig, but that assumption is not reliable for mirrored joints. The recent arm issue showed that clearly: some mirrored pairs need opposite signs, while others need the same sign.

## Main Problem

"Pose meaning" and "rig actuation" are currently coupled.

Right now a preset encodes both:

- what motion is intended
- how that motion must be actuated on this exact rig

That makes presets brittle. A pose like `open_arms` should express semantic intent such as "abduct both shoulders", but the current system instead hardcodes per-joint actuation values and assumes the sign convention is obvious.

## Known Issues in Current Code

### Axis inference is fragile

`infer_joint_axis` in `skeleton_common.py` uses keyword matching (`'twist' in name`, `'shoulder' in name`) to choose axes. It picks an axis but never validates that the chosen axis produces symmetric motion for mirrored pairs. This is the root cause of the mirror-sign problem.

### `_choose_bend_axis` can silently pick a wrong axis

The function picks the local cardinal axis whose world-space projection best aligns with the lateral axis, but only considers the 3 cardinal axes. If the rest pose has the joint rotated ~45 degrees off-cardinal, two candidates can score nearly equal and the choice becomes arbitrary. There is no confidence threshold or warning.

### Diagnostics only cover arm chains

`ARM_LINK_PAIRS` and `ARM_PROGRESSIVE_STEPS` in `pose_diagnostics.py` only cover arms. The walk poses have the same mirror problem for legs (`thigh_stretch`, `leg_stretch`, `foot`), but the diagnostic tooling ignores them.

### Duplicated rotation matrix code

`pose_diagnostics.py` contains a `_rotation_matrix` function that duplicates `axis_angle_matrix` from `skeleton_common.py`. These will drift over time.

### No rest-pose baseline validation

Nothing validates that the USD and URDF rest poses match before posed comparisons begin. If they diverge (e.g., from fixed root joint transforms or column/row-major confusion), every posed comparison carries a constant offset that masks real pose-mapping bugs.

### No sanity check on mirror matrix

`mirror_matrix_from_records` computes a reflection matrix from `infer_lateral_axis_world`, but there is no assertion that the result is a proper reflection (det = -1) or that the lateral axis is roughly unit-length. A wrong lateral axis silently corrupts all mirror metrics.

## Redesign Goals

1. Separate semantic pose intent from rig-specific joint actuation.
2. Make mirrored-pair behavior explicit and testable.
3. Validate pose mapping numerically before relying on rendered screenshots.
4. Make failures localizable to a specific joint stage in a chain.

## Redesign Plan

### 0. Validate Rest-Pose Baseline

Before any posed comparisons, confirm that the USD and URDF rest poses produce matching world matrices within tolerance.

- Compute root-relative world matrices for every joint in both USD and URDF at rest (zero angles).
- Report per-joint position and rotation error.
- Fail loudly if any joint exceeds a threshold (e.g., 1e-4 m position, 0.1 deg rotation).

This prevents constant offsets from masking real pose-mapping bugs in all later steps.

### 1. Add Confidence Thresholds to Axis Inference

Harden `_choose_bend_axis` and `infer_joint_axis`:

- After scoring cardinal axis candidates, require the best score to exceed a minimum threshold (e.g., 0.6).
- If below threshold, flag the joint as ambiguous in the skeleton metadata rather than silently picking a possibly wrong axis.
- Add a sanity assertion to `mirror_matrix_from_records`: verify `abs(det(mirror_matrix) - (-1)) < 1e-6` and that the lateral axis is roughly unit-length.

### 2. Mirror-Sign Calibration Per Joint Pair

For every left/right joint pair, determine the correct sign convention empirically:

- Apply `+angle` to both L and R, measure mirrored endpoint and orientation error.
- Apply `+angle` to L and `-angle` to R, measure the same errors.
- Store whichever sign combination minimizes mirror error as rig metadata.
- Save the result per asset alongside the skeleton JSON, reuse without re-probing.

This must be done before building semantic presets, since the semantic layer depends on knowing the correct sign per joint.

### 3. Freeze a Reference Rig Description

Create one canonical skeleton metadata file per USD asset that stores:

- topology
- rest local frames
- parent-child chain information
- inferred lateral axis
- left/right mirror partner for each joint
- calibrated mirror sign per joint pair (from Step 2)
- axis confidence scores (from Step 1)

This metadata should become the source of truth for pose mapping rather than raw preset dictionaries.

### 4. Split Pose Mapping Into Two Layers

Introduce two layers:

- `Pose semantics`
  - side-independent intent such as `shoulder_abduct`, `elbow_flex`, `wrist_yaw`, `spine_bend`
- `Rig adapter`
  - per-joint mapping from semantic channels into actual rig angles, using the calibrated sign conventions from rig metadata, including scale and limits

This prevents presets from directly encoding fragile rig-specific assumptions.

### 5. Replace Raw Named Presets With Semantic Presets

Instead of writing:

- `shoulder_l = 0.30`
- `shoulder_r = -0.30`
- `arm_stretch_l = 1.45`
- `arm_stretch_r = 1.45`

write semantic intent such as:

- `left_shoulder_abduct = 0.30`
- `right_shoulder_abduct = 0.30`
- `left_elbow_extend = 0.08`
- `right_elbow_extend = 0.08`

The adapter layer should convert semantic values into concrete rig actuation values using the calibrated rig metadata.

### 6. Build Chain-Level Diagnostics

Stop relying on screenshots as the primary debugging tool. Generalize diagnostics to cover arbitrary chain pairs, not just arms.

For each pose, print and save:

- mirrored position and rotation error for arm chain: `shoulder`, `upper arm`, `forearm`, `hand`
- mirrored position and rotation error for leg chain: `thigh`, `leg`, `foot`, `toes`
- USD-vs-URDF error for the same links
- a progressive scan per chain:
  - `rest`
  - `first joint only`
  - `first + second`
  - `+ third`
  - `+ fourth`

That makes it obvious where symmetry first breaks, in both arms and legs.

### 7. Add Three Validation Modes

Validation should be split into:

- `Rig correctness`
  - USD vs URDF root-relative FK match (starting from rest-pose baseline in Step 0)
- `Mirror correctness`
  - left/right symmetry for symmetric semantic poses
- `Visual correctness`
  - deterministic camera captures from front, 3/4, and side views

These should be treated as separate checks rather than one mixed result.

### 8. Make the Gallery Semantic-Aware

Each gallery pose should declare whether it is:

- `symmetric`
- `asymmetric`

For symmetric poses:

- always emit a symmetry report
- always capture a 3/4 view in addition to the frontal view

That avoids misreading symmetry from a single camera angle.

## Concrete Next Implementation Steps

1. Add rest-pose baseline validation (USD vs URDF at zero angles).
2. Add axis confidence thresholds and mirror-matrix sanity assertions.
3. Implement mirror-sign calibration and persist results in rig metadata.
4. Generalize diagnostic chain pairs to cover legs alongside arms.
5. Remove duplicated `_rotation_matrix` in `pose_diagnostics.py`, reuse `axis_angle_matrix` from `skeleton_common.py`.
6. Introduce `pose_semantics.py` layer with semantic channels and per-rig adapters built on calibrated metadata.
7. Move current presets to semantic presets and regenerate `open_arms`, `walk`, and `walk_right`.
8. Add tests for:
   - rest-pose USD-vs-URDF match at zero angles
   - axis confidence thresholds flagging ambiguous joints
   - per-joint mirror sign calibration
   - symmetric pose mirror error thresholds (arms and legs)
   - USD-vs-URDF chain error thresholds
   - semantic preset regression for `rest`, `open_arms`, and `walk`

## Success Criteria

- Rest-pose USD and URDF world matrices match within tolerance before any posed work.
- Axis inference flags ambiguous joints rather than silently guessing.
- Mirror-sign conventions are calibrated per joint pair, not assumed.
- Symmetric poses are symmetric by metric, not by eye, for both arms and legs.
- USD and URDF match numerically on the same semantic pose.
- A broken pose can be localized to one joint stage from one printed report.
