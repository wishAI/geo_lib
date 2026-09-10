---
name: landau-face-rig
description: Continue Landau v10 facial geometry, material segmentation, original eyelash animation and upper eyelid rigging in the local 3d_char_details sandbox. Use also to preserve the face during body or clothing reconstruction.
---

# Landau facial workflow

All paths below are relative to `algorithms/3d_char_details/` unless stated otherwise.
This records the September 2026 working prototype, not a final likeness or game-asset approval.

## Reference and acceptance

- Compare `inputs/landau_v10/reference.png` (original `~/Downloads/landau_v10.png`), `landau_test.png` and `landau_test2.png` before reshaping. The source texture is misleading: inspect clay, clean materials and geometry together.
- The four `closed_eyes_reference*.png` / `half_closed_reference*.png` images were supplied by the user from `~/Downloads/landau_facial`. Pale blue upper lids meet dark navy original eyelashes. Keep the soft character silhouette and round, detailed eyes.
- Animate the original `Lash_L` / `Lash_R` geometry. Do not add replacement lashes, paint a substitute navy band on the lid, or conceal the original lashes beneath overlays. Lashes stay navy throughout blinking.
- Use upper eyelids only. The fixed lower eye boundary is a closure target, not permission to add a lower eyelid mesh.
- Ocular surfaces, iris, pupil and highlights are separate components. Blink must not scale, flatten or deform them. Intentional eye-size editing is a separate control.
- Judge open, half-closed and closed states, including oblique views. A zero attachment gap does not by itself establish an attractive eyelash pose.

## Where to work

| File | Responsibility |
| --- | --- |
| `build_landau.py` | Import USD, partition source, clean facial materials, create eye details and morphs, skin and export |
| `segment_face_geometry.py` | Geometry-based facial region segmentation |
| `inspect_landau.py` | Source inspection and render helpers |
| `verify_likeness.py` | Original-lash attachment / stationary-eye checks and blink renders |
| `validate_asset.py` | Parse exported GLB skin, morph, source-preservation and embedded-asset data |
| `gui/editor.js`, `gui/editor.css` | Interactive orbit editor, controls, references, part visibility, export |
| `gui/manifest.json` | Sandbox registration, validation action and artifacts |

The imported model uses Z up and faces -Y. GLTF export converts to Three.js Y up / +Z front. Do not reuse the locomotion sandbox's coordinate assumptions.

## Geometry and material identification

The original USD is one connected sculpted mesh with 50,000 triangles, split UV vertices and a 68-bone rig. Loose-part separation alone cannot identify facial regions. This implementation uses reviewed Landau landmarks; it is not a general automatic semantic classifier.

`component_regions()` welds positions at approximately 1e-6 **for adjacency analysis only**, then uses landmark seeds and dihedral barriers (currently 15 degrees for cream and 35 degrees for several smaller features). `iris_cut()` uses adjacency graph cuts to distinguish the raised iris from its ocular surround. Inspect those functions before adjusting thresholds or seed coordinates.

Preserve original vertex positions, corner UVs and custom normals when constructing subsets. Do not collapse the render mesh merely because the analysis graph welds UV seams. Small enclosed residual regions near the mouth/philtrum were reassigned from neighboring geometry; broad texture-color classification previously produced blue spots and misaligned cream regions.

Current clean facial palette: head blue `#4b9fb7`, cream `#fff0c1`, ocular white `#fffbee`, lashes/brows navy `#182d5e`, nose `#781729`, inner ear `#e6a6b6`. These are sRGB values converted to linear material inputs. The face uses clean materials without the inaccurate original base-color/normal maps. Body and garment materials remain unfinished.

Round iris/pupil/highlights fit a smooth surface derived from ocular geometry only. Projecting onto the whole face previously placed red iris arcs on the cream face and exposed them through blinking lids. Preserve independent ocular components; extreme gaze combinations still need review.

## Upper lid and original lash motion

Read `curved_lids()` and `lash_beds()` before editing:

1. Extract the original welded lash boundary and eye aperture. Upper and fixed lower guide curves come from the source geometry.
2. Transport the actual lash vertices in the guide's tangent frame toward the closed boundary. The current motion includes out-of-plane roll (about 0.90 radians) and outward tip spread. Tune these against the reference silhouettes rather than translating or shrinking the entire lash.
3. Sweep the pale-blue upper lid to the transformed original lash attachment. The final blue row uses the same attachment positions as the source lash. Verify both eyes throughout interpolation; the current check samples 0, 0.5 and 1.
4. Relax interior lid rows while keeping the attachment constrained. An ocular-only clearance surface keeps lids in front of iris and glints; the present padding is 0.002 model units.
5. A lash-bed patch fills the hole left by moving the formerly fused source lashes. Its boundary follows the true concave attachment loop. It is supporting skin, not another eyelash. Avoid convex-hull patches that cover surrounding details.

Avoid nearly zero-width neutral strips: float32 row collapse caused inconsistent normals and black noise. Determine face winding from nondegenerate closed coordinates before assigning the neutral positions. Blender 5.2's `tessellate_polygon` result used here is indices; check the installed API before changing that code.

After programmatic key changes, update shape-key/data tags, the frame and the view layer before rendering. Validate `Lash_L` and `Lash_R` by hiding them independently in the GUI: their visible navy lash must disappear with no replacement underneath.

## Rebuild, inspect and preserve

From the repository root, restore declared assets with `./geo storage hydrate`. The original `inputs/landau_v10/source.usdc`, prepared master/GLB, report and large textures use the root `large_files.json` Nextcloud contract. Reference PNGs and source code are in Git. The original USD was copied from the existing repository Landau input, not generated from the reference image in this sandbox.

Use Blender MCP when available, or the local Blender executable. Read-only inspection should precede scene mutation. A full build replaces the active scene; use background Blender or save unrelated work first.

```bash
blender --background --python algorithms/3d_char_details/build_landau.py
blender --background algorithms/3d_char_details/outputs/landau_v10/landau_character.blend --python algorithms/3d_char_details/verify_likeness.py
python3 algorithms/3d_char_details/validate_asset.py
node --input-type=module --check < algorithms/3d_char_details/gui/editor.js
./geo storage audit
```

On this Mac the executable may be `/Applications/Blender.app/Contents/MacOS/Blender`. Current proof names are `geometry_open`, `geometry_half`, `geometry_closed`, `geometry_threequarter`, `geometry_full`, plus `export_half` and `export_closed` from GLB reimport. Older `likeness_*` and trial renders can depict rejected approaches. Regenerate proofs after changing geometry; do not present stale images as new validation.

Open `http://127.0.0.1:8767/#/sandbox/3d_char_details` if the GUI is already running; otherwise use `./geo gui`. Check orbit/zoom, clay and clean materials, independent part visibility, blink presets, auto blink, refresh persistence and export. Do not restart an active shared GUI merely to inspect this editor.

Reimport the exported GLB and inspect the same blink states. The data validator checks finite geometry/morphs, normalized maximum-four skin influences, source neutral preservation, original lash animation, stationary ocular blink parts, absence of lower lids and embedded assets. It does not certify likeness or production deformation. The current browser exporter produced a download link, but an actual browser-downloaded file was not independently verified on disk.

Keep generated outputs and large binaries out of Git. Archive revised masters and reports to new content-addressed Nextcloud paths, verify SHA-256 and update only this sandbox's manifest entries. The builder unlinks managed output symlinks before writing; preserve this behavior so rebuilding never overwrites an archived version. Proof renders can be regenerated from the master.

## Honest continuation state

- Original source neutral geometry is preserved; the prototype adds ocular details and upper-lid controls. Outer eye corners and lash silhouettes still require art review.
- Current export has 71 bones (68 source plus ear/tail additions) and custom facial controls. This is not a complete ARKit, VRM or FaceRig mapping. `jawDrop` does not provide an oral cavity, phonemes or production lip sync.
- `Body_UnderClothes` is a hidden rough capsule/voxel approximation and can intersect clothing. Clothing visibility currently does **not** satisfy the user's complete independent-body requirement.
- For subsequent clothing work, infer a coherent complete body from the character's anatomy and reference proportions; do not shrink garments to fake it. Body and clothes need independent meshes with compatible skinning, usable garment hiding/replacement, refined material boundaries, and joint-pose deformation checks. Preserve the facial components and expression behavior while doing that work.
- Do not claim universal game readiness: retargeting, extreme poses, mouth interiors, LOD, collisions and cloth behavior remain separate acceptance work.

Useful conceptual references: [Blender Rigify face rig types](https://docs.blender.org/manual/en/latest/addons/rigify/rig_types/face.html) and [Blender Studio facial-rigging eyes chapter](https://studio.blender.org/training/facial-rigging/chapter/eyes/). The public chapter overview was consulted; paid lesson contents were not reviewed.
