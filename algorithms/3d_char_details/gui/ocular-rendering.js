// The conformal eye layers are only .000025 model units apart. A fixed .005 near
// plane loses that separation when orbiting out to a full-body view.
export function updateCameraDepth(camera,target) {
  const distance=camera.position.distanceTo(target);
  const near=Math.max(.002,distance*.05),far=Math.max(10,distance*2);
  if(Math.abs(camera.near-near)<1e-7&&Math.abs(camera.far-far)<1e-7)return;
  camera.near=near;camera.far=far;camera.updateProjectionMatrix();
}

export function eyeLayerDepth(material,partName) {
  const layer=/^RoundIris_[LR]$/.test(partName)?1:/^Pupil_[LR]$/.test(partName)?2:/^Catchlight(?:Small)?_[LR]$/.test(partName)?3:0;
  if(!layer)return;
  // Bias only by depth-buffer units, never by polygon slope. Keep depth testing
  // and writing enabled so real eyelids still occlude every ocular layer.
  // This is a viewport precision guard, not an exported geometry displacement.
  material.polygonOffset=true;material.polygonOffsetFactor=0;material.polygonOffsetUnits=-layer;
}
