// Evaluate an explicitly supplied immutable take; no recordings enter Git.
import {readFile} from 'node:fs/promises';
import {gunzipSync} from 'node:zlib';
import assert from 'node:assert/strict';
import * as THREE from '../../../webgui/static/vendor/three/three.module.js';
import {moduleURL} from './browser_modules.mjs';
const {LiveRetargeter}=await import(await moduleURL('live-retarget.js'));
const {worldTransforms,fromRows}=await import(await moduleURL('pose-math.js'));
const data=JSON.parse(await readFile(new URL('../outputs/web_scene/scene.json',import.meta.url)));
const path=process.argv[2];if(!path)throw new Error('Supply a recording path');
const bytes=await readFile(path),clip=JSON.parse(path.endsWith('.gz')?gunzipSync(bytes):bytes);
assert.equal(clip.urdfSha256,data.source.urdfSha256,'Recording must match current URDF');
const solver=new LiveRetargeter(data),errors={saved:[],corrected:[]};let limitedFrames=0,worstSaved={degrees:-1};
for(const [index,frame] of clip.frames.entries()){
 if(!frame.tracking?.head)continue;
 const corrected={...frame.pose,...solver.headPose(frame.tracking)};
 const target=new THREE.Vector3().setFromMatrixColumn(solver.transform.clone().multiply(fromRows(frame.tracking.head)),1).normalize();
 if(['head_x','neck_x'].some(n=>Math.abs(corrected[n]-solver.joints.get(n).lower)<1e-8||Math.abs(corrected[n]-solver.joints.get(n).upper)<1e-8))limitedFrames++;
 for(const [kind,pose] of [['saved',frame.pose],['corrected',corrected]]){
  const direction=new THREE.Vector3(0,0,1).transformDirection(worldTransforms(data.joints,pose).head_x);
  const error=direction.angleTo(target)*180/Math.PI;assert.ok(Number.isFinite(error));errors[kind].push(error);
  if(kind==='saved'&&error>worstSaved.degrees)worstSaved={index,t:frame.t,degrees:error};
 }
}
const stats=a=>{assert.ok(a.length,'No tracked head frames');a.sort((a,b)=>a-b);return {mean:a.reduce((s,v)=>s+v,0)/a.length,p95:a[Math.floor(.95*a.length)],max:a.at(-1)};};
console.log(JSON.stringify({createdAt:clip.createdAt,frames:clip.frames.length,trackedFrames:errors.saved.length,limitedFrames,
 headForwardErrorDegrees:{saved:stats(errors.saved),corrected:stats(errors.corrected)},worstSaved,
 limitation:'Look direction only; the physical two-joint head has no independent roll actuator.'},null,2));
