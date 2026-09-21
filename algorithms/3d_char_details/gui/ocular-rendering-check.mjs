import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
const source=await readFile(new URL('./ocular-rendering.js',import.meta.url),'utf8');
const {updateCameraDepth,eyeLayerDepth}=await import('data:text/javascript;base64,'+Buffer.from(source).toString('base64'));

// Compare the actual perspective depth quantization at the eye, with a small
// additional allowance for the target-to-eye distance during orbiting.
const depthStep=(z,near,far)=>z*z*(far-near)/(far*near*((2**24)-1));
const legacyStep=depthStep(6,.005,30);
assert.ok(legacyStep>.000025,'Old camera must reproduce the insufficient precision.');
let updates=0;
for(const distance of [.18,.3,.72,1,2.4,4,6]) {
  const camera={position:{distanceTo:()=>distance},near:.005,far:30,updateProjectionMatrix(){updates++;}};
  updateCameraDepth(camera,{});
  assert.ok(camera.near<distance*.1,'Near plane must stay comfortably before the orbit target.');
  assert.ok(camera.far>distance+2,'Whole character and grid remain inside the far plane.');
  assert.ok(depthStep(distance+.2,camera.near,camera.far)<.000008,'Resolve shallow eye layers across the complete zoom range.');
  const before=updates;updateCameraDepth(camera,{});assert.equal(updates,before,'Stationary camera needs no projection update.');
}
for(const side of ['L','R'])for(const [prefix,units] of [['RoundIris',-1],['Pupil',-2],['Catchlight',-3],['CatchlightSmall',-3]]) {
  const material={depthTest:true,depthWrite:true};eyeLayerDepth(material,prefix+'_'+side);
  assert.equal(material.polygonOffset,true);assert.equal(material.polygonOffsetFactor,0);assert.equal(material.polygonOffsetUnits,units);
  assert.equal(material.depthTest,true);assert.equal(material.depthWrite,true);
}
for(const name of ['EyeShell_L','Lash_R','UpperLid_L','Body_Complete','Vest']) {
  const material={depthTest:true,depthWrite:true};eyeLayerDepth(material,name);
  assert.deepEqual(material,{depthTest:true,depthWrite:true});
}
console.log('Eye depth passed: full zoom range resolves conformal layers; bias is symmetric and retains eyelid occlusion.');
