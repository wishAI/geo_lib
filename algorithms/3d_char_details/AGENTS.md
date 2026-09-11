# 3d char details

Read [the local facial skill](skills/landau-face-rig/SKILL.md) before character work.
Keep edits inside this sandbox; other sandboxes may be edited concurrently.

## Current accepted direction (September 10, 2026)

Use the user-supplied `inputs/landau_v10/landau_body.fbx` for the body.
Apply **one uniform scale: 0.79 on X, Y and Z**. Move the original skeleton's
joint locations to that body. Do not compress different body axes or independently
warp arms, torso and legs to force the old skeleton's proportions.

Keep the original face/head and original recovered hand geometry. The head is
rigidly translated, and hands are rigidly positioned at the new wrists. The body
has stitched wrists and an 82-vertex open neck rim paired to the original head.
Local facial geometry, UVs, normals, weights, materials and morphs stay exact.

Keep all eight original garment meshes, especially the original boots. Their
Basis geometry, UVs, normals, weights and materials are restored. Only rigid
placement follows the new rig. Zero-default tailoring controls are provided;
the user explicitly wants to perform the final clothing fitting. Clothing is
hidden initially, and it intersects the body until adjusted. Do not replace the
boots, automatically scale the outfit, or claim the zero-fit outfit is fitted.

`integrate_fbx_body.py` integrates the supplied sculpt; `rebuild_body.py` now
contains preservation/hand recovery/join helpers only. `refine_body_skin.py`
smooths weights without changing neutral shape; `match_neck_normals.py` matches
the new collar's shading to the retained head; `body_adjustments.py` adds 19 body
and 39 garment controls. `build_landau.py` runs these after the facial build.
`refine_neck_transition.py` adds five transition loops at the neck and fixes
body winding before skin/normal refinement. The final body has 38,965 vertices,
77,332 faces and the same 82-vertex paired neck rim. The final GLB has 141,403
triangles across 34 meshes; it excludes inspection scenes.

Validate exported GLB with `validate_asset.py`; inspect body poses using
`validate_body.py` and original blinks using `verify_likeness.py`. Deep hip bends
still need corrective shapes. The triangulated generated body is not certified
production retopology. Never present rejected older render files as current QA.

Archive updated binaries through new content-addressed Nextcloud paths and update
only this sandbox's `large_files.json` entries. Never write through existing
managed symlinks or overwrite an archived revision.

The latest local master, GLB and reports are newer than their archive manifest
entries. Automatic approval review blocked the final Nextcloud transfer; keep
these local files until the user authorizes archiving. `outputs/landau_v10/
archive_pending.json` records their hashes. Do not hydrate over these revisions.
