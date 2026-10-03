# Clothing deformation research — September 11, 2026

The segmentation keeps separate garment surfaces and records corresponding
source seam vertices. The September 12 GUI applies parent/child attachment
constraints during user-controlled fitting; it does not automatically fit to the body. Garment identity is established before assigning
weights. The source is a fused sculpt, so geometric adjacency alone does not imply
that every interface is physically sewn. Replaceable parts retain separate objects;
concealed overlap and seam constraints must be distinguished during motion QA.

## Industry methods and this sandbox

- [Blender Data Transfer](https://docs.blender.org/manual/en/5.2/modeling/modifiers/modify/data_transfer.html)
  interpolates weights across source faces. Use only after garment segmentation,
  with donor regions restricted so nearby fingers or another limb cannot become
  accidental influences. Weight ownership does not define a garment.
- [NVIDIA's hybrid cloth architecture](https://nvidiagameworks.github.io/APEX/1.4/docs/APEX_Clothing/Clothing_Module_Doc.html)
  separates a simplified physical mesh from the detailed render mesh. Animated
  attachments, bounded movement and collision constraints support local cloth
  simulation. This is a historical design reference, not a runtime dependency.
- [Blender cloth shape controls](https://docs.blender.org/manual/en/latest/physics/cloth/settings/shape.html)
  provide sewing springs and pin groups. The connected-fitting GUI
  matches source boundary positions and normalized skin vectors, with corrections
  spread over child surface connectivity. Independent
  materials and UV splits do not require a physical gap.
- [Lewis, Cordner and Fong, Pose Space Deformation](https://www.scribblethink.org/Work/PSD/PSD.pdf)
  describes correcting deformation as a function of pose. A possible later extension
  is garment-only corrective morphs; the user explicitly deferred fitting, so
  this change adds no corrective geometry or simulation.
- [Blender Corrective Smooth](https://docs.blender.org/manual/en/4.2/modeling/modifiers/deform/corrective_smooth.html)
  can reduce post-skinning distortion and preserve boundaries, but it is not a
  collision solver. Smoothing alone cannot establish clearance.
- [Unreal Clothing Tool](https://dev.epicgames.com/documentation/unreal-engine/clothing-tool-in-unreal-engine)
  combines maximum distances, backstop, animation drive and character collision.
  A future simulation layer should pin necklines/cuffs/tucked hems, permit motion
  mainly in loose panels, and use fitted body collision geometry. Over-large
  backstops can distort clothing or destroy intended layering.
- [Blender glTF animation](https://docs.blender.org/manual/en/4.4/addons/import_export/scene_gltf2.html)
  supports skeletal transforms and morph values; Blender physics is not exported
  as a live browser simulation. Deterministic scrubbing requires baking or an
  explicitly implemented runtime solver.
- [Three.js AnimationMixer](https://threejs.org/docs/pages/AnimationMixer.html)
  and [AnimationAction](https://threejs.org/docs/pages/AnimationAction.html)
  provide the playback primitives. The editor samples the retargeted skeleton
  while keeping the preset's manual settings separate.

## Interface contract and validation

Store seam correspondence separately from material/UV topology. Corresponding
samples need the same fitted position, full normalized skin vector and corrective
displacement. Preserve garment replacement: do not permanently merge vest, sleeves,
cuffs, trousers and boots into one editable object. A sewn shoulder seam can share
boundary deformation; a trouser hem tucked inside a boot should have a concealed
continuation and collision-aware opening rather than an exposed stretched bridge.

Future fitted-garment validation should sample vertices, edge midpoints and
triangle centers against the deformed body, with a separate seam-gap measurement. Negative nearest-surface
distance is a diagnostic, not a watertight signed-distance or continuous-collision
proof. Multi-view renders and between-frame tests are needed as well. Concealed
overlaps must be explicitly identified; do not relabel visible failures as layering.

The original sculpt shoe Basis and per-corner UVs are preserved. This change does not fit, scale, reshape or add collision corrections to shoes. Source texture colors are not used to identify shoe panels or piping.

The user initially restricted clothing work to segmentation, then explicitly
requested connected manual fitting on September 12. The GUI now shares scale,
exposes clothing-only rest angles and constrains attachment boundaries. Collision
response, pose-space correctives and cloth simulation remain future options.
The running preview uses the supplied skeletal animation and does not claim
collision-free clothing.
