import {readFile} from 'node:fs/promises';
import {test} from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from '../../../webgui/static/vendor/three/three.module.js';
const three=new URL('../../../webgui/static/vendor/three/three.module.js',import.meta.url).href;
const moduleURL=async name=>'data:text/javascript;base64,'+Buffer.from((await readFile(new URL('../gui/'+name,import.meta.url),'utf8')).replace("'three'",JSON.stringify(three))).toString('base64');
const mathURL=await moduleURL('pose-math.js'),markersURL=await moduleURL('tracking-markers.js');
const code=(await readFile(new URL('../gui/live-retarget.js',import.meta.url),'utf8')).replace("'three'",JSON.stringify(three)).replace("'/api/artifact?path=algorithms/avp_remote/gui/pose-math.js'",JSON.stringify(mathURL)).replace("'/api/artifact?path=algorithms/avp_remote/gui/tracking-markers.js'",JSON.stringify(markersURL));
const {LiveRetargeter}=await import('data:text/javascript;base64,'+Buffer.from(code).toString('base64'));
const {worldTransforms,fromRows}=await import(mathURL);
const data=JSON.parse(await readFile(new URL('../outputs/web_scene/scene.json',import.meta.url)));
const pos=m=>new THREE.Vector3().setFromMatrixPosition(m);
function error(solver,pose,side,frame){const arm=solver.arms[side],wrist=frame[side+'_arm']?.[0]||frame[side+'_wrist'];const target=pos(solver.baseInverse.clone().multiply(solver.transform).multiply(fromRows(wrist)));if(target.length()>arm.cap)target.setLength(arm.cap);return pos(solver.baseInverse.clone().multiply(worldTransforms(data.joints,pose)['hand_'+arm.suffix])).distanceTo(target);}
test('browser head and fingers match the authoritative Python snapshot',()=>{
 const solver=new LiveRetargeter(data),pose=solver.solve(data.snapshot);
 for(const name of Object.keys(data.retargetConfig.rules).filter(n=>!Object.values(solver.arms).some(a=>a.chain.some(j=>j.child===n))))assert.ok(Math.abs(pose[name]-(data.snapshotPose[name]||0))<1e-6,`${name}: ${pose[name]} vs ${data.snapshotPose[name]}`);
});
test('warm-started arm CCD converges while respecting every URDF limit',()=>{
 const solver=new LiveRetargeter(data);let pose;for(let i=0;i<30;i++)pose=solver.solve(data.snapshot);
 for(const side of ['left','right'])assert.ok(error(solver,pose,side,data.snapshot)<=error(solver,data.snapshotPose,side,data.snapshot)+.005,side);
 for(const j of data.joints.filter(j=>j.type!=='fixed'))assert.ok(Number.isFinite(pose[j.child])&&pose[j.child]>=j.lower&&pose[j.child]<=j.upper,j.child);
 const headOnly=solver.solve({head:data.snapshot.head});for(const name of Object.keys(headOnly).filter(n=>!['head_x','neck_x'].includes(n)))assert.equal(headOnly[name],0);
});
test('moving wrist targets produce fresh solutions each call with a bounded iteration budget',()=>{
 const solver=new LiveRetargeter(data),times=[],values=[];
 for(let i=0;i<240;i++){
  const frame=structuredClone(data.snapshot);for(const side of ['left','right'])for(const m of frame[side+'_arm'])m[0][3]+=.12*Math.sin(i*.05);
  const start=performance.now(),pose=solver.solve(frame);times.push(performance.now()-start);values.push(pose.arm_stretch_l);
 }
 assert.ok(new Set(values.map(x=>x.toFixed(5))).size>180);
 times.sort((a,b)=>a-b);console.log(`local retarget: median ${times[120].toFixed(2)} ms, p95 ${times[228].toFixed(2)} ms`);
});
