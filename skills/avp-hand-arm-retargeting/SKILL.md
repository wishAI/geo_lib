---
name: avp-hand-arm-retargeting
description: Diagnose and repair AVP head/hand retargeting, finger animation, arm skin deformation, and URDF-to-USD pose transfer in geo_lib's avp_remote and usd_parallel_urdf sandboxes. Use for unnatural palms, fingers, elbows or sleeves, and mapping regressions in live capture or recorded playback; not locomotion training.
---

# AVP Hand and Arm Retargeting

Use this repository skill for the Landau tracking-to-pose-to-skin pipeline. Read root `AGENTS.md` and `agent_guide.md`, then the affected algorithm's README. Paths and commands below are relative to the repository root. For formulas, prior failure mechanisms, and evidence limits, read [mapping details](references/mapping.md) when working on that layer.

## Locate the failing layer

Reproduce with the same pose, asset version, camera, and playback-remapping setting. A screenshot alone cannot distinguish a solver error from a skin error. Use a zero pose and a deterministic snapshot first; use an explicitly supplied or task-authorized recording when the issue depends on motion.

| Observed difference | Inspect first |
| --- | --- |
| Tracking markers misplaced in both models | `avp_remote/gui/pose-math.js`, `tracking-markers.js`, tracking pretransform and calibration |
| Both wrists reach reasonably but palms face the wrong way | `gui/live-retarget.js`, `landau_retarget.py`, anatomical palm orientation in arm IK |
| Thumb opposition, spread or distal fingers wrong | `gui/hand-rig.js`, measured segment directions and bind-relative rotations |
| URDF arm plausible, USD sleeve flattened or detached | `gui/skin-retarget.js`, geometry ownership, original bind frames and serial/sibling hierarchy |
| Native USD differs from browser or URDF | `avp_remote/landau_pose.py`; `usd_parallel_urdf/skeleton_common.py` and its display callers |
| Playback differs from live or changes after seeking | `gui/motion-clip.js`, `viewer.js`, saved `handPose`, remap caches and solver state |

The table's abbreviated paths are under `algorithms/`. Follow actual callers; do not assume all views use the same mapper. In particular, browser animation and native scalar joint control are different representations.

## Preserve the representation contracts

- WebXR has 25 hand joints; native snapshots may have 27. Do not synthesize the native forearm points for WebXR. Head/hands are measured; arm/body IK is estimated.
- WebXR is Y-up with forward -Z; the stored native pipeline uses Z-up/+Y. Reuse `nativeMatrix` and the exported tracking pretransform. Avoid adding another axis/sign conversion to compensate for a downstream bug.
- Browser pose keys are URDF child-link names. Native actuator commands can use joint names. Map explicitly; check units (radians) and loaded limits.
- The 38-bone `HandRig` animates fingers with local quaternions `[x,y,z,w]`. It is separate from the reduced scalar URDF articulation used by sliders, native Python and Unitree. A scalar curl fix cannot reproduce general thumb opposition or finger spread.
- Preserve source bone pivots, lengths, bind frames, skin weights and Landau's hand size for a mapping repair. A visual mismatch does not by itself justify editing the asset or the URDF axes, limits, masses or training inputs.
- Apply body FK, then hand rotations, then map to USD skin. Both browser USD and URDF previews consume the resulting hand worlds; native control is not automatically upgraded by a browser hand-rig change.
- Keep algorithm Python modules isolated. Handoff between sandboxes is files, never cross-folder Python imports. Equivalent native math can live independently in each sandbox.

## Repair by layer

**Palm and fingers.** Derive an anatomical basis from measured positions, not vendor joint axes: little metacarpal toward index metacarpal, and wrist toward the mean knuckle position. Match the same polarity in the model rest basis and arm solver. Position-only IK cannot observe roll about an axis through the wrist; include palm orientation while retaining reach/limit checks. Use every measured finger segment, including distal-to-tip. Preserve bind roll with hierarchical minimal-swing quaternion rotations. Omit missing or degenerate chains; verify they do not keep stale curls.

**Arm and skin.** Inspect the actual URDF parent links, visual ownership and USD hierarchy. Landau's serial URDF upper-arm twist link owns both stretch and twist geometry; the stretch link is empty. When the forearm is parented through that twist link, its motion must also drive the USD upper-arm stretch weights. Use rest-relative motion with each original skin bind frame, then reconstruct locals from the adjusted USD parent targets. Keep forearm/hand targets intact to avoid applying roll twice. Do not activate that ownership alias for a source-style sibling URDF. See the formula and regression in [mapping details](references/mapping.md#arm-motion-and-skin-ownership).

For native changes, check both `LandauUsdPoseDriver` and the conversion sandbox's playback/render/gallery paths. Keep `apply_pose_to_local_matrices` as the source-skeleton evaluator: pose semantics and independent FK diagnostics rely on it. Display mapping belongs in `UrdfSkinMapper`; do not redefine an independent diagnostic to use its own expected output.

**Recording compatibility.** Keep original recordings immutable. Corrected playback derives body/hand poses from saved tracking; disabling remapping must restore saved motion, including saved hand quaternions when present. Interpolate quaternions with slerp. Test old takes without `handPose`, new takes with it, partial/missing hands, nonsequential seeks and pose JSON import/export. A browser remap is not a migration of the user's stored take.

## Validation and evidence

Choose checks for the changed layer. Existing browser tests consume the prepared scene/GLB; if absent, follow the algorithm README's asset preparation workflow. Do not silently regenerate unrelated assets or treat a missing fixture as a passing check.

```sh
node --test algorithms/avp_remote/tests/test_*.mjs
python3 -m unittest discover -s algorithms/avp_remote/tests -p test_urdf_kinematics.py
python3 -m unittest discover -s algorithms/avp_remote/tests -p test_web_pose.py
python3 -m unittest discover -s algorithms/usd_parallel_urdf/tests
./geo storage audit
```

For a task-authorized take, with its real path substituted:

```sh
node algorithms/avp_remote/tests/evaluate_hand_recording.mjs /path/to/recording.json.gz
python3 algorithms/avp_remote/tests/browser_hand_recording.py /path/to/recording.json.gz
```

The second command needs Playwright and Chrome and uses an isolated recording store. Resolve available environments from the README/environment; do not rely on a previous agent's `/tmp` virtualenv. Some broader Python tests prepare external assets and may stall independently of mapping. Report that separately. `spawnSync ... EPERM` or localhost socket denial is an execution restriction; use the permitted escalation path rather than changing application logic or suppressing the test.

Validate observable invariants:

- Zero pose preserves original skin transforms; nonidentity bind bases still work.
- Both hands retain bone lengths/scale while matching measured segment directions, including thumb opposition, spread and distal segments.
- Both upper sleeves remain connected to the elbow under positive/negative serial rotation, including a transverse axis. Hand/forearm worlds remain correct without double rotation.
- Sibling topology does not accidentally acquire serial ownership; missing tracking and partial saved poses have deterministic fallback.
- Inspect before/after skin renders at the same pose/camera, from both sides for hands. Matching matrices alone cannot prove correct ownership or natural deformation.
- Exercise actual viewer replay/seek/import/export after serialization or rendering changes. For XR loop/input changes, inspect `tests/browser_xr_runtime.py` before running it: it defaults to a live LAN service and records synthetic takes. Use an isolated test server/store when testing capture.

Report numerical and visual evidence separately from limitations. Desktop Chrome simulation does not certify Vision Pro Safari stability, headset FPS, collision-free fingertips, or an Isaac native render. Generated comparisons, private takes and runtime assets stay out of Git; commit source, tests and this skill only when requested.
