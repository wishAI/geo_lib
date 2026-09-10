# 3d char details

Before changing Landau's face or integrating body/clothing changes, read
[the local facial skill](skills/landau-face-rig/SKILL.md).

Keep character work inside this sandbox. Shared GUI changes should only wire
this editor into the existing launcher; other sandboxes may be edited concurrently.

Current body/clothing separation is incomplete. `Body_UnderClothes` is a hidden
rough approximation, not an accepted independent body. Future work must infer
and construct a complete body beneath independently hideable/replaceable clothes;
shrinking the garment is not a body reconstruction. Preserve the facial work
while improving body topology, materials and deformation.
