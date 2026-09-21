# 3d char details

Read [the local facial skill](skills/landau-face-rig/SKILL.md) before character work.
Keep edits inside this sandbox; other sandboxes may be edited concurrently.

## Current direction (September 11, 2026)

Checkpoint before the inspector/continuous-skin work: `27f46ad`. Its binary
snapshot is in `outputs/landau_v10/checkpoints/27f46ad/` (local, not Git).

Use `inputs/landau_v10/landau_body.fbx`, uniformly scaled by 0.79. Move the rig
joints to the supplied proportions. Do not compress separate axes to fit the
old skeleton or derive anatomy by shrinking clothes.

Revision 5 uses `continuous_skin.py` to join Head, Face_Cream and Body_Complete,
weld the shared rim and relax the neck band Z=.745–.816. Body_Complete now
includes the original head and original hands. Neck boundary edges: zero.
Original head vertex and shape-key coordinates above Z=.818 are exact; the
other 23 facial components, original lashes and fixed ocular surfaces remain
unchanged. Do not expect separate Head / Face_Cream objects after this step.
The remaining facial openings are intentional component interfaces; do not
claim the entire character is a closed manifold or production retopology.

Keep eight original garment meshes and boots, with original neutral geometry,
UVs, materials and weights. They start hidden and unfitted for manual fitting.
Dedicated tailoring controls replace the old accidental facial morph falloffs
on clothing. Do not automatically scale or refit the garments.

## Inspector

- Clothing has folded cards for Vest, Sleeves, Cuffs, Trousers and Boots.
  Visibility belongs in each card. Pairs link by default; unlink exposes the
  right-side card. Values live in `settings.outfit[part][morph]`; links live in
  `settings.links`. Per-part values must survive preset reload and GLB export.
- `gui/property-widgets.js`: compact rows, per-value reset, amber editing / violet
  animation controls, glyphs for operation and global/local scope. Tooltip and
  accessible text explain glyphs; no visible type badges.
- `gui/body-placement.js`: global below-head scale (.75–1.25) and lateral offset
  (-.05–.05). Transform rest vertices, morph deltas and rest joints together;
  recalculate inverse binds before restoring pose. Head stays fixed. Smooth
  transition at the neck, original global pose remains editable. Store values
  in presets and bake them into exported GLB geometry / inverse binds.
- New imported JS modules must be declared as syncOnly artifacts in manifest.

Build order: facial builder → integrate_fbx_body → refine_neck_transition →
refine_body_skin → match_neck_normals → body_adjustments → continuous_skin → finish_neck.

Validate with `validate_asset.py`, `validate_body.py` (Blender),
`verify_likeness.py` (Blender), and `validate_editor.py` (Node + bundled Three.js).
Inspect open/half/closed export proofs and test UI links, resets and persistence.
Deep hip bends still need corrective shapes. Original garments remain unfitted.

## Storage

Generated outputs and binaries stay out of Git. Never overwrite an archived
revision through a managed symlink. Archive to a new verified hash path only.
The latest local master, GLB and reports are newer than large_files.json's
archive entries. Final Nextcloud transfer was blocked by approval review in the
previous turn and remains pending user authorization. Keep local files and
`outputs/landau_v10/archive_pending.json`; do not hydrate over these revisions.

## Neck finish and preset history (September 11)

`finish_neck.py` follows the weld step: tapered Taubin relaxation changes only
Z .750–.812, preserving every original facial vertex/morph above .818. Explicit
area-weighted corner normals replace partition-dependent normals at the neck;
a shared color field crosses the former body/head material boundary. Compare
`neck_export_front` and `neck_export_oblique` against the old checkpoint. The
neck still has zero boundary edges. This is local cleanup, not full retopology.

`gui/preset-history.js` implements the Presets pane. Save preset writes a new
named snapshot through `/api/character/presets`; history supports restore,
rename, archive and unarchive. JSON import/export remains under a folded row.
The minimal shared server route delegates to this sandbox's `preset_store.py`.
Snapshots are stored atomically in `outputs/presets/history.json`. Renaming and
archiving never change snapshot settings; concurrent writes are serialized.
Browser localStorage remains the working draft, not the explicit saved history.
Known compatible previous GLB hashes live in `asset_report.preset_compatible_hashes`;
never accept an arbitrary different model's preset. Current fit was migrated
without changing any settings; only its asset hash was advanced.

Checks: `python3 -m unittest algorithms.3d_char_details.test_presets`, plus the
asset/editor checks above. The UI was tested through save, reload, restore,
rename, archive and unarchive; the QA half-blink version is archived.

## Requested next session: clothing reconstruction and running preview

The user now questions the eight existing garment partitions. Audit their
provenance; do not treat dominant bone weights or source texture colors as
garment boundaries. Use source 3D features, seams and connectivity to identify
real garments, which may span several bones; repair clothing material regions.
This supersedes preserving the current garment partitions/materials verbatim,
but retains the original shoe design and the accepted face/body work.

Research suitable industry approaches for seam continuity, skinning, corrective
deformation and selectively simulated cloth with body collision. Check
vest–sleeve–cuff and trouser–boot interfaces through motion: prevent visible gaps
and penetration, while retaining intentional concealed layering instead of
indiscriminately welding every garment. Preserve independently replaceable
garments, paired fitting controls and saved presets.

Inspect the actual animation clips in `~/Downloads/landau_body.fbx`; import and
retarget the supplied running clip to the current rig with correct axes, units,
rest pose and root motion. Add play/pause, loop, speed, scrub and reset in the
sandbox, then test body/garment deformation and interfaces throughout the clip.
The September 11 continuation below records the implemented work and the user's
subsequent restriction to segmentation and manual fitting.

## September 11 continuation from 6446c39: segmentation and running

The user clarified: **do not create a new clothing version or automatically fit
it**. Segment the existing geometry; leave fitting to the sandbox properties.
This supersedes the earlier request to implement corrective/cloth fitting now.
Do not resume the discarded automatic-fitting experiment.

- `segment_clothing.py` uses reviewed 15-degree crease components on the original
  source sculpt plus dihedral-weighted surface propagation. Placket/collar colors
  use geometry-based minimum cuts. No bone ownership, UV or texture classification.
- `rebuild_clothing.py` keeps exact original source positions and corner UV/normals,
  the existing rigid placements, full source skin weights, and existing tailoring
  keys. Ownership changes at seams; controls are transferred to newly owned
  vertices from the previous garment's nearest source vertex. Eight independently
  replaceable garment objects retain their stable GUI names. Shoes are original
  source geometry; no replacement, scaling or automatic fitting is applied.
- `retarget_running.py` transfers the supplied Downloads FBX clip (same SHA as the
  retained input) through source/target world rest frames. It is 33 frames at
  60 fps, an in-place 0.533333-second loop. All 71 target rest bones are unchanged.
- `gui/motion-player.js` adds play/pause, loop, speed, scrub and motion reset.
  Playback is separate from saved manual settings; reset restores the user's pose.
  Body-placement edits and unanimated manual bones are supported. The action is
  embedded as `Running` in GLB and retained as an unassigned fake-user action in
  Blender so the neutral master stays neutral. Browser edited-GLB export keeps
  the current posed character; it does not promise animation-clip export.
- `export_clothing.py` writes fresh local assets and keeps prior GLB hashes in the
  compatibility list. Saved preset history is untouched. Legacy garment material
  color names remain accepted and are mapped to the replacement clean palette.
- `clothing_research.md` records industry options as future guidance, not features
  currently implemented. There is no automatic cloth fit, correction or simulation.
  Unfitted clothes can intersect or have interface gaps in motion; the user owns
  manual fitting. Never describe these diagnostics as a collision-free pass.

Validation: `validate_asset.py`, `validate_editor.py`, `validate_motion.py`, plus
`python3 -m unittest algorithms.3d_char_details.test_presets`. Motion tests sample
65 times and cover repeated/backward scrub, pause, looping, speed, end clamping,
manual pose, body placement and reset. Protected face/body/neck data hashes match
before and after segmentation. Original garment Basis source-position error is 0.

The accepted starting local assets remain in `outputs/landau_v10/checkpoints/6446c39/`.
Use current local master/GLB. Do not hydrate older archive entries over them.
`discarded_autofit/` contains rejected diagnostic outputs only, not current assets.

## September 12: connected manual clothing fitting

Accepted segmentation/running source checkpoint: `3cf1e6e` (committed before
this work). Current local Blender/GLB assets remain unchanged. The user's new
request explicitly permits manual shared scaling, clothing joint-angle edits
and seam constraints; it supersedes the earlier segmentation-only GUI scope.

`gui/garment-fit.js` treats Vest → Sleeves → Cuffs and Trousers → Boots as two
logical outfits while keeping all eight meshes/material sections replaceable.
Neutral attachment uses each outfit parent's rigid placement for its children,
restoring original shared source boundaries without reshaping the garments.
Fit edits bake onto a fresh copy of rest positions, then child seam vertices
follow their parent positions and skin weights. Corrections spread over the
child's connectivity graph. The two outfits are not joined at the waist.

Each card shows its outfit scale (.5–2); child shared controls are locked by
default. Unlocking permits editing the same authoritative parent value, not
breaking the seam. Boot top width/depth/height follow trouser calf width/depth/
length. Cuffs inherit sleeve length/forearm room. Movement and shoulder controls
show their owner. Local child movement adjusts the free end with its attachment
held. Unlinking sleeves also exposes independent cuffs. Clothing-only shoulder,
elbow, hip and knee angles use audited original USD joints; they never rotate
the body rig. Matching sleeve angles mirror anatomically left/right.

Preset field `garmentFit` stores scales, angles and shared-control locks. Existing
history snapshots stay immutable; legacy presets still load, with shared
boundaries now governed by their parent values. Export bakes the fitted garment
positions/weights and stores fitting settings in extras, retaining the original
facial morphs and current skeletal pose. Clothing morph targets are omitted from
that edited export to avoid applying the same fit twice. The native master and
source GLB still retain the original tailoring targets.

`validate_garment_fit.py` checks all six interfaces through 65 run samples,
parent/child values, independent sides, scale, angles, body preservation,
repeatability, preset round trips, rest-frame edits and GLB export/reimport.
Neutral source error < 7e-8 after rigid attachment; sampled seam error 0;
export/reimport world-position error < 7e-8. These are attachment checks, not a
body-collision or cloth-simulation certificate. Body fit remains manual.
Safari was used for visual/GUI QA because the in-app browser failed WebGL context
creation. Shared scale/locks and running/reset were checked; the user's original
manual values were restored after the scale test. Saved history was not edited.

### September 13 control corrections

Clothing width/depth/room properties now accept -1…1 via `outfitMinimum()`;
zero is still the authored neutral, and negative values narrow the garment.
Both slider ranges and preset validation use the same limit. No source asset
or stored history was rebased. Chest narrowing is checked on the actual GLB.

Clothing angles have independent `garmentFit.jointLinks.arms/legs`, both true
by default, regardless of garment tailoring links. Each joint section offers
“Adjust both L / R”; disabling it exposes individual leg controls or permits
individual sleeve angles. Paired edits/resets write both sides. Legacy unequal
angles remain unchanged on load and show “mixed” until edited; explicitly
relinking uses the displayed side (left for trousers). Joint foldouts remain
open during link changes. Browser checks confirmed negative chest width and
both knee values; test values were restored afterward.

### Collar and independent front/back depth

The Vest card now has neckline width, front depth, back depth and collar height
controls, all signed -1…1. Source-space height and lateral falloffs confine these
edits to the collar/upper-neckline region, with zero change below source Z=.52.
The accepted face/body/master assets remain untouched.

All eight legacy clothing depth controls are replaced in the property cards by
front/back pairs. Their saved legacy keys remain accepted: normal two-sided
targets initialize both sides to the previous value; chest depth initializes
only the front, and heel depth only the back, matching their authored geometry.
The newly available chest-back and heel-front fields default to zero. No saved
fit is silently rebased. Boot shaft depth aliases preserve front/back ownership
through the trouser calf controls. Negative values remain supported.

Validation covers every depth pair, exact legacy-fit expansion, collar-region
isolation, front-depth independence from the rear, preset round trips, running
seams and export/reimport. Blender MCP inspection was unavailable (broken pipe)
in this turn; current local GLB geometry was used for all functional checks.

### Lower-neck body transition and sleeve attachment controls

The user clarified that “between neck and chest” means the **body**, not the
clothing collar. Shape now groups four `bodyTransition*` controls under
“Lower neck / upper chest”: width, front depth, back depth and height.
`gui/body-transition.js` installs zero-neutral morphs on the existing connected
skin before rest-frame capture; the band is Y=.710–.818 in glTF coordinates,
with a lateral fade that excludes the arms. Existing face/body morph indices,
source assets and saved values stay intact. The original Neck width control
is retained. Body placement also transports the new morph normals.

The Vest and Sleeve cards expose “Shoulder / sleeve join”: attachment outward,
height, front/back depth and opening height. These `vestArmhole*` fields use
source-shape falloffs on the outer upper vest. Vest owns the seam values;
sleeves show the shared values under the existing parent-control lock. Seam
constraints carry the sleeves/cuffs and remain active during running. No new
clothing geometry or automatic body fit was added. All nine fields are signed,
zero-neutral, individually resettable, preset-compatible and exported.

The current source Blender/GLB files are unchanged; editor GLB export includes
the added body morphs and bakes the clothing fit. Blender MCP was unavailable
for this turn. `validate_garment_fit.py` checks regional isolation, both signs,
shared values, running seams and GLB export/reimport on the current local asset.

The in-app browser loaded successfully for this check. Both new groups were
visible; transition width and attachment front depth were exercised, working
values survived reload, and the temporary values were reset afterward.
No saved history entries were changed.


## September 14: retain the existing body, open the underarm grooves

The user explicitly said not to rebuild the character; retain the gray-body
model shown in the structural previews. `repair_underarms.py` is a localized
repair of that mesh, not model generation. Do not return to full-body rebuilds.
It measures front/back concave cross-section features, follows their diagonal
underarm groove with a narrow curved relief, rounds the shoulder termination,
and closes/smooths only the local cut surfaces. Constant-X cuts through the
breast and broad nearest-point skin smoothing were rejected.

Current local master/GLB includes the repair. The prior asset/report snapshot is
`outputs/landau_v10/checkpoints/pre_underarm_20260914/`. Face, hands and anterior
body protection covers 40,376 vertices with zero position, morph and weight
error; both axillae have zero boundary edges. The existing 276 facial interface
boundary edges remain. This is local repair, not whole-character retopology.
Preserve exact weights for untouched vertices; snapshot numeric vertex-group
IDs before removing live Blender groups. New shape keys must explicitly start
at value zero: an intermediate preview with all shape controls active was
rejected and never became the current asset. Boolean cap faces use existing
body material 0; remove the cutter's empty material slot.

`gui/editing-pose.js` supplies the default T editing baseline without rebasing
rest joints, inverse binds, garment coordinates or the Running clip. Manual
bone values are offsets; `editingPose: "t" | "a"` is preset-persistent, with T
as the default for legacy presets. The source A pose is still selectable.
Running/scrubbing uses its authored pose and reset restores the editing pose.
Existing clothing source correspondence, fitting controls and immutable preset
history stay intact. Export uses the current editing pose and shared skeleton.

`export_clothing.py` accepts a validated localized repair and retains original
facial preservation checks, updated protected-body hashes and compatible preset
hashes. Checks passed on the new GLB: asset, placement, motion, connected clothing,
65 running samples, six seams, and T-pose export/reimport. Gray front/back proofs
are `underarm_gray_neutral*.png`; do not use the rejected intermediate previews.
Current large assets are local, still subject to the pending archive contract.

## September 21: shoulder pivots and garment shoulder weights

The user requested adjustable shoulder joints and a fix for the blue shoulder
patch exposed when switching the current manual fit from A to T.

- Shape → Shoulder joints → Shoulder joint height stores
  `bodyFrame.shoulderHeight` (-.03…+.03, zero default). It moves both upper-arm
  stretch/twist rest origins vertically, scaled with the body frame. Neutral
  mesh positions, elbow/wrist world landmarks and the head stay fixed. Rebind
  and recalculate the T editing baseline before restoring animation. This is
  a rotation-pivot adjustment, not a new body shape or clavicle-motion solver.
- The exposed patch was body penetration, not a disconnected garment seam.
  The original USD clothing weights could follow the upper arm where the
  fitted FBX body mainly followed the upper spine. `garment-fit.js` now updates
  shoulder weights from the closest body-rest triangle with barycentric
  interpolation and a smooth regional fade. It evaluates body morphs without
  current skeletal animation; repeated edits do not accumulate weights.
- Clothing → Upper outfit → Follow body at shoulders controls this correction
  (`garmentFit.shoulderFollow`, default true, false restores original weights).
  Updates happen after manual fit/body-shape edits. Exact seam weights are
  reapplied afterward. It does not change manual rest geometry, regenerate
  garments, shrink the skin or hide faces. Presets and edited GLB exports retain
  the new settings/binding/weights; the source Blender and GLB stay untouched.
- Current exported user fit: 1,801 shoulder samples; A-clear points penetrating
  in T dropped 186→3. At 65 run times ×451 sampled vertices, new penetration
  samples fell 2,329→1,019. One worst running depth increased by .000112 model
  units; residual intersections remain. Do not claim full collision-free fit.
  The large patch disappeared in matched oblique browser comparison.
- Checks: `validate_editor.py`, `validate_garment_fit.py`, `validate_motion.py`,
  `python3 -m unittest algorithms.3d_char_details.test_presets`, and
  `validate_shoulder.py <exported-preset.json> --output <report.json>`.
  Six seams remain exact through 328 tested poses, and nonzero shoulder
  GLB export/reimport position error is below 7e-8. Browser checks covered
  A/T, toggle off/on, shoulder height, reload persistence, running and reset.

### September 21 underarm regression correction

The broad shoulder transfer above introduced an axilla regression: adjacent
vest vertices chose opposite sides of the close arm/torso surfaces. A 1.1 mm
rest edge stretched to 29× in T-pose, creating the user's sharp underarm folds.
Do not restore the broad Y=.60–.66 transfer ramp or treat low penetration counts
alone as a pass. Shoulder transfer now starts at unscaled body Y=.70 and reaches
full weight at .73; the lower vest retains its original continuous weights.
The body triangle search remains broader than the target blend so the closest
source triangle is not discarded at the band's boundary.

For the current fit, all 5,160 lower-vest vertices retain exact original weights.
The 2,160 axilla edges return exactly to the original deformation (maximum stretch
2.31869×, 99th percentile 1.77861×), while new T shoulder penetrations remain
186→3. `shoulder-check.mjs` now guards both edge distortion and penetration, plus
all 26 protected body/facial meshes and morphs. Final 65-frame running results
are 2,329→2,263 new penetration samples, worst depth unchanged; the earlier 56%
running improvement belonged to the rejected broad transfer and no longer
applies. All six seams and nonzero-shoulder GLB export checks still pass.
Low-angle browser comparison confirms the introduced spikes are removed;
original fit folds/intersections are not a full cloth-collision pass.

## September 21: opening mouth, independent eyes and blink repair

The user explicitly authorized facial topology changes. `repair_face.py` is a
localized postprocess of `checkpoints/pre_face_20260921/landau_character.blend`;
do not rerun the obsolete full-body builder to reproduce this revision.

- A 297-face mouth-crease patch becomes an annulus with an actual lip opening,
  recessed oral bag, concealed upper/lower teeth and jaw-following tongue.
  The 39-vertex lip boundary matches the cavity at jaw 0/.5/1. Unused incision
  vertices are removed. The neck remains welded; other facial component
  interfaces and the separate mouth rim are intentional, not whole-body closure.
- `EyeShell_L/R` are independent closed smooth non-spherical eyeballs. The old
  raised `Iris_L/R` supports are removed; RoundIris/Pupil/glints follow one
  symmetric sclera-only surface with offsets .000025/.000050/.000075. Eye parts
  remain stationary during blink; gaze uses the ocular surface only.
- Original Lash_L/R neutral vertices and topology remain exact. Shared guides,
  mirrored target-surface correspondence and matched closed-lid surfaces repair
  the closure. `gui/facial-controls.js` clamps blink + .45*squint and evaluates
  `_blinkArcL/R = 4*b*(1-b)` for intermediate clearance. These corrective morphs
  are exported; another engine must evaluate the same rule when animating.
- `validate_face.py` measures actual geometry and compares against the checkpoint.
  Closed-lash mirrored distance p95 improved .008089 → .000396 model units;
  maximum residual is .001457, so do not claim mathematically exact symmetry.
  Five-state attachment error is zero; no lower lid or replacement lashes.
- `export_clothing.py(facial_repair=...)` records scoped preservation separately
  from the historical body-integration face hashes. 51,571 protected body
  vertices retain their positions/morphs. Clothing, rest rig and Running remain.
  Known-compatible presets ignore only retired Iris support part entries;
  immutable saved history is untouched.
- The exporter retains authored body split normals during jawDrop by omitting
  only that target's NORMAL delta. Check final shading in WebGL: Blender's GLB
  importer does not restore morph normals, so reimport renders check geometry
  but are not an exact shading oracle. Proof reports identify the asset hash.

Three user-supplied mouth references are preserved byte-for-byte under
`inputs/landau_v10/open_mouth/`, copied from `~/Downloads/open_mouth`, and listed
in the Ref picker. They guide the dark red interior, small tongue and concealed
teeth; this is a custom facial prototype, not a phoneme/production lip-sync rig.

Checks: `validate_asset.py`, Blender `validate_face.py`, `validate_body.py`,
`verify_likeness.py` (add `-- --glb` for actual GLB reimport proofs),
`validate_editor.py`, `validate_motion.py`, `validate_garment_fit.py`, preset
unit tests and `node algorithms/3d_char_details/gui/facial-controls-check.mjs`.
Local master/GLB supersede the archives; retain the checkpoint and do not hydrate
older large_files entries over this repair. Cloud archival remains pending.

### Close-up mouth and zoom-depth follow-up

The user's distant-eye screenshot exposed depth quantization in the editor's
fixed .005 near plane. `gui/ocular-rendering.js` adapts camera clipping to orbit
distance and uses small ordered depth-buffer offsets on iris/pupil/highlights.
Eye geometry stays separate and shallow; depth testing still lets lids occlude
it. These are viewport settings, not portable glTF polygon-offset properties.
`ocular-rendering-check.mjs` checks the complete supported zoom range. Actual
eye triangle/vertex sampling found positive separation, not intersecting layers.

The follow-up mouth replaces 342 local source faces and has a 176-vertex lip
boundary. Smooth curve coordinates, constrained lip strips and filtered depth
replace the earlier coarse 39-point annulus. `mouth_constraints.py` prevents
folds jointly across jawDrop [0,1] and mouthLength/mouthCurvature [-1,1], with
fixed outer/lip boundaries and a small interior Basis correction. It audits
the continuous parameter box after float32 key serialization. Do not remove
this gate or treat fixed normals alone as a geometry repair.

Shape → Mouth exposes length (±20% at the lip) and curvature (corners down/up).
They remain active across expression presets, persist with other shape edits,
and export as native morph targets. `validate_face.py` checks combined settings
and lip/cavity attachment; `validate_asset.py` also checks actual GLB triangles
at 45 combinations. There are 51,546 unchanged protected body vertices in this
larger local patch; the original rig, lashes, clothing and history remain.
The previous local model is retained in `checkpoints/pre_face_smooth_20260921/`.

### Rabbit muzzle likeness and slider performance correction

The user rejected both human-like lip relief and the subsequent completely
smooth muzzle. Preserve the original sculpt's projecting upper muzzle. The
mouth line now comes from `muzzle_features.py`: dense front-facing depth samples,
the lower concave foot of the strongest depth transition, mirrored averaging,
and sub-millimeter filtering. `repair_face.py` samples the original surface
along that measured crease; it does not replace the muzzle with a polynomial
or sinusoidal surface. Lip support vertices follow actual curve normals to
avoid crossing at the sloped mouth corners.

The user-provided `inputs/landau_v10/side_muzzle_reference.png` guides the profile:
a modest .0015 model-unit upper projection and up to .004 inward lower-muzzle/
chin adjustment, tapered to preserve surrounding geometry. Existing chin
topology remains; authored normals follow the deformation Jacobian. The prior
model is in `checkpoints/pre_rabbit_muzzle_20260921/`. Check neutral front,
oblique and true side profiles (`mouth_refined_profile`), plus half/full opening.
The full-width bowl and flattened open upper edge were also rejected. Jaw
opening keeps the measured upper mouth curve fixed and opens a smaller tapered
lower region beneath its two lobes. The lower rim independently relaxes into
one smooth arc rather than carrying a central W-shaped spike downward. The
outer cheek seams stay nearly closed. Keep signed mouth controls and cavity
attachment; fold checks alone do not certify likeness.

The new mouth controls had unnecessarily invoked full garment fitting on every
input. `editor.js` skips that path for mouthLength/mouthCurvature. In
`garment-fit.js`, a bounded spatial index and rest-surface cache replace repeated
brute-force shoulder searches; active control values are resolved once per mesh.
Actual-GLB comparison across eight settings preserved positions, normals and
weights exactly. Warm garment inputs improved from 573–719 ms to 86–98 ms;
body/rest-surface changes invalidate the cache. Existing seam, motion, reset and
export checks still apply. Side reference selection shows the entire image.
