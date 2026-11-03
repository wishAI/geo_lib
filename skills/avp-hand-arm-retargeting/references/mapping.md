# Mapping Details and Regressions

Read the section for the layer being changed. These are repository-specific lessons, not a requirement to reproduce one historical pose.

## Head look direction

The original mapper sent 55% of measured pitch to `neck_x` and 45% to `head_x`. The URDF actually defines `neck_pitch_joint` with world axis +X and `head_yaw_joint` with an approximately +Z axis tilted about 7.7 degrees. The avatar faces -Y; its head bone's local +Z is forward. Native tracking uses local +Y forward, and the existing pretransform rotates it into avatar space. The old split therefore reversed the nod and introduced an unrelated yaw.

Assigning negative pitch and yaw directly fixes pure nods but still misses mixed motion: the input Euler order and the serial joint order differ, and the yaw axis is tilted. `solve_look_direction` and browser `headPose` instead minimize the angle between the actual FK head-forward direction and the transformed measured forward vector. Coordinate descent projects both vectors onto each joint's rotation plane, computes a signed atan2 update, and clamps to that joint's real limits. It starts from zero for every frame and has a bounded iteration count, so a seek cannot inherit the previous head pose. This is a look-direction objective; roll is not independently representable by this two-joint articulation.

Native implementations live separately in `avp_remote/urdf_kinematics.py` and `usd_parallel_urdf/g1_pose.py`. The latter is also used by `avp_pose_debug.py`; load the actual URDF head chain there instead of assuming source-skeleton axes or looking up child-link limits in a joint-name dictionary. Preserve the existing head calibration in `viewer.js`: it already maps the first live frame into the reference snapshot frame. Subtracting snapshot pitch again would change both the recording and marker coordinate contract.

Existing `scene.json` packages bake the old `head pitch` rules and snapshot pose. The browser upgrades those identifiable legacy rules in memory and recomputes the reference head pose at load. Newly built packages use the new rule descriptions. Do not rebuild a GLB just to refresh head math; do not treat its old baked head values as the current native oracle. Dedicated tests compare against the current Python mapper and actual FK.

The 2026-11-03 20:44:13 take had 128 head-tracked frames. Comparing actual head-forward directions with measured tracking gave mean/max errors of 28.47/54.95 degrees for the saved pose and below 0.000002 degrees after correction, without hitting joint limits. The corrected/original playback toggle, nonsequential seeks and pose export/import were tested in an isolated browser/store; original bytes stayed unchanged. This result measures head-forward direction on that take, not full orientation, headset latency, or all extreme poses. New takes carry retarget version 4; old takes are corrected from tracking when remapping is enabled.

## Anatomical palm and finger mapping

`algorithms/avp_remote/gui/hand-rig.js` has 19 driven bones per hand: three thumb bones and four bones for each other finger (metacarpal/base plus three phalanges). The measured fingertip supplies the last bone's direction even though there is no additional driven tip bone.

For measured positions `p` in the hand's input space:

- `x = normalize(p[5] - p[20])`: little metacarpal toward index metacarpal.
- `y0 = (p[6] + p[11] + p[16]) / 3 - p[0]`: wrist toward knuckles.
- `y = normalize(y0 - x * dot(x,y0))`; `z = cross(x,y)`.
- Build the same basis `B_model` from the robot's rest points in wrist-local space. Rotate measured directions by `B_model * transpose(B_measured)`.

The opposite lateral polarity can produce nearly 180-degree palm errors. Raw vendor joint rotations do not necessarily use the model's anatomical axes. Inspect frame conventions before adding a sign patch.

For each bone, the rig computes its expected orientation from the parent's solved orientation and its original local bind rotation. The local bone +Y is the segment direction for this asset. A minimal-swing quaternion aligns that direction with the measured segment. Store the resulting rotation relative to the expected bind frame, not an absolute world orientation. This transports the bind roll through the chain and avoids unstable bend-plane normals when fingers straighten. A different asset may use a different bone axis; inspect it first.

No translated tracking point becomes a new bone pivot, and no measured human length rescales the avatar. Degenerate palm bases and collapsed chains are omitted. Applying a partial saved finger pose must still propagate through unspecified descendants using their bind rotations.

`LiveRetargeter.targetRotation` constrains the wrist using the anatomical palm basis as well as wrist position. Wrist and forearm roll can be invisible to a position-only solver. A good orientation fit can coexist with a position residual when an input target is unreachable under the avatar's size and limits; report both.

The browser's legacy scalar finger mapper remains for reduced robot/Unitree joint commands. Do not assume it represents the 38-bone animation rig or overwrite the latter with its Euler/scalar outputs.

## Arm motion and skin ownership

Source USD has sibling deformation bones; generated URDF uses serial articulation:

```text
USD:  arm_stretch -> arm_twist
                 -> forearm_stretch -> forearm_twist
                                    -> hand
URDF: arm_stretch -> arm_twist -> forearm_stretch -> forearm_twist -> hand
```

In `usd_parallel_urdf/skeleton_common.py::build_urdf_model`, the URDF `arm_stretch_*` link has no visual geometry and `arm_twist_*` owns both upper-arm source regions. Therefore same-name skin transfer is insufficient even when all matrices satisfy a same-name test.

Let `U_j(q)` be the actual URDF world frame, `U_j(0)` its zero-pose frame, and `S_i(0)` the original source skin frame. Let `owner(i)` normally be `i`, with upper-arm stretch mapped to upper-arm twist only when the loaded URDF actually has that serial chain:

```text
S_i(q) = U_owner(i)(q) * inverse(U_owner(i)(0)) * S_i(0)
L_i(q) = inverse(S_parent(i)(q)) * S_i(q)
```

For a root bone, use its skeleton/model-space target directly. In the browser, preserve any non-bone parent's model-relative transform when deriving the local matrix. Evaluate all targets before deriving locals; using already-mutated parent transforms introduces order dependence. GLB rest frames may differ in orientation from URDF frames; keep the full bind offset rather than assigning a raw URDF rotation to the GLB bone.

Native USD records contain small non-unit scales. Use full matrix inverses for these source frames instead of assuming transpose equals inverse. Browser `SkinRetargeter` and native `UrdfSkinMapper` cache rest offsets.

After aliasing upper-arm stretch, reconstruct forearm/hand locals from their own unchanged targets and the adjusted parent target. They must not inherit another copy of upper-arm rotation. Forearm twist and hand keep their own motion sources; mapping every twist bone to its parent would remove legitimate forearm roll.

The 2026-11-03 regression used the existing reference snapshot. The right upper-arm roll joint rotates around local Z, while its elbow lies along local Y. At 1.2 rad, using the pre-twist stretch frame left the sleeve's rest-elbow point about **0.102244 m** from the articulated elbow. Transferring the geometry owner's motion removed that disconnect in desktop skin renders. This identifies a mapping defect; it does not authorize changing the joint axis to match its name.

The conversion sandbox's native player had a separate inheritance error: applying angles directly to the source sibling skeleton omitted serial roll propagation. A diagnostic left upper-arm twist of 0.7 rad gave about 16.5 mm hand-position and 0.7 rad orientation disagreement. Display callers now use the loaded URDF's FK and rest-relative skin mapping:

- `avp_remote/gui/skin-retarget.js`: browser GLB skin.
- `avp_remote/landau_pose.py`: native `LandauUsdPoseDriver`.
- `usd_parallel_urdf/skeleton_common.py`: conversion sandbox mapper.
- `usd_parallel_urdf/play_parallel_animation.py`, `render_parallel_scene.py`, `validate_parallel_scene.py`: native display callers.

Keep source-only pose semantics and independent FK comparisons separate. In live imported-prim validation, compare each skin frame to its declared geometry owner's imported motion plus the bind offset; expose `motion_owner` in evidence so the exception is visible.

## Evidence from the hand and arm repairs

The hand repair is recorded in existing commit `0f1d783`, following the palm/bind mapping work in `003cfe6`. The code and tests are authoritative; do not use historical implementations to replace current source.

The hand README reports a 522-frame take with 19,836 checked bone directions, near-zero corrected direction error and unchanged lengths, plus open/fist skin inspection from both sides. `tests/evaluate_hand_recording.mjs` reproduces direction, length, determinant and solve-time checks for a supplied take. These measurements establish geometric agreement for that take, not physical finger collision avoidance or headset performance.

The arm repair was checked with zero-bind, bilateral rotation, sleeve-to-elbow and unchanged downstream-target regressions, same-pose desktop GLB renders, and actual viewer playback/seek/pose import/export. Its initial validation did not include an Isaac render or a Vision Pro device run. Full AVP Python discovery stalled at external asset preparation; targeted mapping tests passed. Future reports should replace these historical limitations with their own observed results, not carry forward a blanket pass claim.
