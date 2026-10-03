import fs from 'node:fs';
import assert from 'node:assert/strict';
import * as T from './three.mjs';
import {GLTFLoader} from './loader.mjs';
import {bodyPlacement} from './placement.mjs';
globalThis.ProgressEvent=class{constructor(type,args){this.type=type;Object.assign(this,args);}};
const data=fs.readFileSync(process.argv[2]);
const jl=data.readUInt32LE(12),json=JSON.parse(data.subarray(20,20+jl));
const bin=data.subarray(28+jl);json.buffers[0].uri='data:application/octet-stream;base64,'+bin.toString('base64');
json.materials=json.materials.map(m=>({name:m.name}));delete json.images;delete json.textures;delete json.samplers;
const {scene}=await new GLTFLoader().parseAsync(JSON.stringify(json),'');
const bones=new Map(),meshes=[];scene.traverse(o=>{if(o.isBone)bones.set(o.name,{bone:o,quaternion:o.quaternion.clone(),scale:o.scale.clone()});if(o.isMesh)meshes.push(o);});
scene.updateMatrixWorld(true);
const original=meshes.map(o=>({o,positions:o.geometry.attributes.position.array.slice(),world:o.matrixWorld.clone()}));
const apply=bodyPlacement(scene,bones),v=new T.Vector3();
for(const side of ['l','r'])for(const prefix of ['arm_stretch_','arm_twist_','forearm_stretch_','hand_'])assert(bones.has(prefix+side),'Missing joint '+prefix+side);
let maxBindError=0,maxHeadDrift=0,maxResetError=0,maxPivotError=0,maxOtherJointDrift=0,maxShoulderSurfaceDrift=0,maxRepeatError=0;
const scenarios=[[.75,-.05,-.03],[1.25,.05,.03],[1.13,.025,-.015],[1,0,.03],[1,0,-.03],[1,0,0]];
const jointPositions=()=>{scene.updateMatrixWorld(true);return new Map([...bones].map(([name,{bone}])=>[name,bone.getWorldPosition(new T.Vector3())]));};
const surface=()=>meshes.flatMap(o=>[...o.geometry.attributes.position.array,...(o.geometry.morphAttributes.position||[]).flatMap(a=>Array.from(a.array))]);
const maxError=(a,b)=>a.reduce((error,value,i)=>Math.max(error,Math.abs(value-b[i])),0);
for(const [scale,offset,shoulderHeight] of scenarios){
 apply(scale,offset,0);const baselineJoints=jointPositions(),baselineSurface=surface();
 apply(scale,offset,shoulderHeight);scene.updateMatrixWorld(true);
 assert.equal(scene.userData.bodyFrame.shoulderHeight,shoulderHeight);
 maxShoulderSurfaceDrift=Math.max(maxShoulderSurfaceDrift,maxError(baselineSurface,surface()));
 for(const [name,p]of jointPositions()){
  const expected=baselineJoints.get(name).clone();
  if(/^arm_(stretch|twist)_[lr]$/.test(name)){expected.y+=shoulderHeight*scale;maxPivotError=Math.max(maxPivotError,p.distanceTo(expected));}
  else maxOtherJointDrift=Math.max(maxOtherJointDrift,p.distanceTo(expected));
 }
 for(const {o,positions,world}of original){
  o.skeleton?.update();const p=o.geometry.attributes.position;
  for(let i=0;i<p.count;i++){
   const baseline=new T.Vector3().fromArray(positions,i*3).applyMatrix4(world);
   const actual=new T.Vector3().fromBufferAttribute(p,i).applyMatrix4(world);
   assert(actual.toArray().every(Number.isFinite));
   if(baseline.y>.818)maxHeadDrift=Math.max(maxHeadDrift,actual.distanceTo(baseline));
   if(scale===1&&offset===0)maxResetError=Math.max(maxResetError,actual.distanceTo(baseline));
   if(o.isSkinnedMesh){o.getVertexPosition(i,v);v.applyMatrix4(o.matrixWorld);maxBindError=Math.max(maxBindError,v.distanceTo(actual));}
  }
 }
 const firstSurface=surface(),firstJoints=jointPositions();apply(scale,offset,shoulderHeight);
 maxRepeatError=Math.max(maxRepeatError,maxError(firstSurface,surface()));
 for(const [name,p]of jointPositions())maxRepeatError=Math.max(maxRepeatError,p.distanceTo(firstJoints.get(name)));
}
assert(maxHeadDrift<1e-6,{maxHeadDrift});assert(maxResetError<1e-6,{maxResetError});assert(maxBindError<2e-6,{maxBindError});
assert(maxPivotError<1e-7,{maxPivotError});assert(maxOtherJointDrift<1e-7,{maxOtherJointDrift});
assert.equal(maxShoulderSurfaceDrift,0,'Shoulder pivot adjustment changed neutral surface or morphs');assert(maxRepeatError<1e-7,{maxRepeatError});
for(const invalid of [NaN,Infinity,-Infinity,.031,-.031])assert.throws(()=>apply(1,0,invalid),/shoulder joint height/);
const result={scenarios:scenarios.length,meshes:meshes.length,bones:bones.size,maxHeadDrift,maxResetError,maxBindError,maxPivotError,maxOtherJointDrift,maxShoulderSurfaceDrift,maxRepeatError};
console.log(JSON.stringify(result,null,2));
fs.writeFileSync(process.argv[3],JSON.stringify(result,null,2));
