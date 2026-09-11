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
These changes were requested for the next session and are not implemented yet.
