// node tests/evaluate_hand_recording.mjs <explicit recording.json[.gz]>
// No private takes are committed; evaluate the actual supplied immutable file.
import {readFile} from 'node:fs/promises';
import {gunzipSync} from 'node:zlib';
import assert from 'node:assert/strict';
import * as THREE from '../../../webgui/static/vendor/three/three.module.js';
import {moduleURL} from './browser_modules.mjs';
const {HandRig}=await import(await moduleURL('hand-rig.js'));
const {worldTransforms,fromRows}=await import(await moduleURL('pose-math.js'));
const data=JSON.parse(await readFile(new URL('../outputs/web_scene/scene.json',import.meta.url)));
const path=process.argv[2];if(!path)throw new Error('Supply a recording path');
const bytes=await readFile(path),clip=JSON.parse(path.endsWith('.gz')?gunzipSync(bytes):bytes);
const rig=new HandRig(data.joints),rest=worldTransforms(data.joints,{}),errors={old:[],corrected:[]},times=[];
let maxLengthError=0,maxScaleError=0;
const pos=m=>new THREE.Vector3().setFromMatrixPosition(m);
for(const frame of clip.frames){
 const start=performance.now(),hands=rig.solve(frame.tracking);times.push(performance.now()-start);
 const old=worldTransforms(data.joints,frame.pose),corrected=rig.apply(worldTransforms(data.joints,frame.pose),hands);
 for(const [side,s] of Object.entries(rig.sides)){
  const stack=frame.tracking?.[side+'_arm'];if(!stack)continue;
  const p=stack.map(m=>pos(fromRows(m))),x=p[5].clone().sub(p[20]).normalize();
  const y=p[6].clone().add(p[11]).add(p[16]).multiplyScalar(1/3).sub(p[0]);y.addScaledVector(x,-y.dot(x)).normalize();
  const alignment=s.basis.clone().multiply(new THREE.Matrix4().makeBasis(x,y,new THREE.Vector3().crossVectors(x,y)).transpose());
  for(const chain of s.chains)for(const {name,parent,index,origin} of chain){
   if(!hands[name])continue;
   const target=p[index+1].clone().sub(p[index]).transformDirection(alignment);
   for(const [key,worlds] of [['old',old],['corrected',corrected]]){
    const local=worlds[s.hand].clone().invert().multiply(worlds[name]);
    errors[key].push(new THREE.Vector3(0,1,0).transformDirection(local).angleTo(target)*180/Math.PI);
   }
   maxLengthError=Math.max(maxLengthError,Math.abs(pos(corrected[name]).distanceTo(pos(corrected[parent]))-pos(origin).length()));
   maxScaleError=Math.max(maxScaleError,Math.abs(corrected[name].determinant()-rest[name].determinant()));
  }
 }
}
const stats=a=>{a.sort((x,y)=>x-y);return {mean:a.reduce((s,v)=>s+v,0)/a.length,p95:a[Math.floor(.95*a.length)],max:a.at(-1)};};
const report={createdAt:clip.createdAt,frames:clip.frames.length,segments:errors.corrected.length,
 directionErrorDegrees:{old:stats(errors.old),corrected:stats(errors.corrected)},maxLengthErrorMeters:maxLengthError,
 maxScaleDeterminantError:maxScaleError,solveMs:stats(times)};
assert.ok(report.directionErrorDegrees.corrected.max<.001);assert.ok(maxLengthError<1e-9);assert.ok(maxScaleError<1e-9);
console.log(JSON.stringify(report,null,2));
