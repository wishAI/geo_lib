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
let maxBindError=0,maxHeadDrift=0,maxResetError=0;
for(const [scale,offset] of [[.75,-.05],[1.25,.05],[1.13,.025],[1,0]]){
 apply(scale,offset);scene.updateMatrixWorld(true);
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
}
assert(maxHeadDrift<1e-6,{maxHeadDrift});assert(maxResetError<1e-6,{maxResetError});assert(maxBindError<2e-6,{maxBindError});
const result={scenarios:4,meshes:meshes.length,bones:bones.size,maxHeadDrift,maxResetError,maxBindError};
console.log(JSON.stringify(result,null,2));
fs.writeFileSync(process.argv[3],JSON.stringify(result,null,2));
