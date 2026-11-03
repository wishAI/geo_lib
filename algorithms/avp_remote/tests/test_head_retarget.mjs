import {readFile} from 'node:fs/promises';
import {execFileSync} from 'node:child_process';
import {test} from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from '../../../webgui/static/vendor/three/three.module.js';
import {moduleURL} from './browser_modules.mjs';
const {LiveRetargeter}=await import(await moduleURL('live-retarget.js'));
const {worldTransforms,rows,fromRows,nativeMatrix}=await import(await moduleURL('pose-math.js'));
const data=JSON.parse(await readFile(new URL('../outputs/web_scene/scene.json',import.meta.url)));
const head=(pitch=0,yaw=0,roll=0)=>rows(new THREE.Matrix4().makeRotationZ(yaw).multiply(new THREE.Matrix4().makeRotationX(pitch)).multiply(new THREE.Matrix4().makeRotationY(roll)));
const forward=m=>new THREE.Vector3(0,0,1).transformDirection(m);

test('nods drive neck pitch in the actual model direction, turns drive head yaw',()=>{
 const solver=new LiveRetargeter(data);
 for(const pitch of [-.4,0,.4])for(const yaw of [-.4,0,.4]){
  const frame={head:head(pitch,yaw,.3)},pose=solver.headPose(frame);
  if(yaw===0){assert.ok(Math.abs(pose.neck_x+pitch)<1e-5);assert.ok(Math.abs(pose.head_x)<1e-5);}
  else assert.equal(Math.sign(pose.head_x),Math.sign(yaw));
  // Check the visible head's direction, not just agreement between mappers.
  const target=new THREE.Vector3().setFromMatrixColumn(solver.transform.clone().multiply(fromRows(frame.head)),1).normalize();
  const actual=forward(worldTransforms(data.joints,pose).head_x);
  // Fit look direction even when pitch/yaw are mixed and the yaw axis tilts.
  assert.ok(actual.angleTo(target)<1e-5,`pitch ${pitch}, yaw ${yaw}`);
 }
});

test('native conversion and head mapping preserve WebXR nod/turn signs and ignore roll',()=>{
 const solver=new LiveRetargeter(data);
 const nod=solver.headPose({head:rows(nativeMatrix(new THREE.Matrix4().makeRotationX(.4).elements))});
 assert.ok(Math.abs(nod.neck_x+.4)<1e-5);assert.ok(Math.abs(nod.head_x)<1e-5);
 const turn=solver.headPose({head:rows(nativeMatrix(new THREE.Matrix4().makeRotationY(.4).elements))});
 assert.ok(turn.head_x>0);
 const target=new THREE.Vector3(Math.sin(.4),-Math.cos(.4),0);
 assert.ok(forward(worldTransforms(data.joints,turn).head_x).angleTo(target)<1e-5);
 const roll=solver.headPose({head:head(0,0,.6)});
 assert.ok(Math.abs(roll.head_x)+Math.abs(roll.neck_x)<1e-5);
});

test('head solve is bounded and independent of prior frames, including missing/vertical tracking',()=>{
 const solver=new LiveRetargeter(data),expected=solver.headPose({head:head(.3,-.4)});
 for(const frame of [{head:head(1.2,1.1)},{head:head(Math.PI/2,.5)},{}]){
  const pose=solver.headPose(frame);
  for(const name of ['neck_x','head_x']){const j=solver.joints.get(name);assert.ok(Number.isFinite(pose[name])&&pose[name]>=j.lower&&pose[name]<=j.upper);}
 }
 assert.deepEqual(solver.headPose({}),{neck_x:0,head_x:0});
 assert.deepEqual(solver.headPose({head:head(.3,-.4)}),expected);

});

test('legacy baked head rules upgrade without mutating scene or requiring mesh conversion',()=>{
 const before=JSON.stringify(data),legacy=structuredClone(data),current=structuredClone(data);
 for(const [name,scale] of [['neck_x',.55],['head_x',.45]]){
  legacy.retargetConfig.rules[name]={source:'head',formula:'head pitch',scale,bias:0};
  current.retargetConfig.rules[name]={source:'head',formula:name==='neck_x'?'head look-direction IK pitch':'head look-direction IK yaw',scale:1,bias:0};
 }
 assert.deepEqual(new LiveRetargeter(legacy).headPose({head:head(.4,.3)}),new LiveRetargeter(current).headPose({head:head(.4,.3)}));
 assert.equal(JSON.stringify(data),before);
});

test('browser and current Python head mapping agree for synthetic motions and snapshot',()=>{
 const frames=[{head:data.snapshot.head},...[-1.1,-.4,0,.4,1.1].flatMap(p=>[-1.1,0,1.1].map(y=>({head:head(p,y,.2)})))];
 const code="import sys,json;from pathlib import Path;sys.path.insert(0,'algorithms/avp_remote');from landau_retarget import LandauUpperBodyRetargeter;r=LandauUpperBodyRetargeter(urdf_path=Path('algorithms/avp_remote/inputs/landau_v10/landau_v10_parallel_mesh.urdf'),skeleton_json_path=Path('algorithms/avp_remote/inputs/landau_v10/landau_v10_skeleton.json'),snapshot_path=Path('algorithms/avp_remote/avp_snapshot.json'),use_trac_ik=False);out=[]\nfor f in json.loads(sys.stdin.read()):\n p={};r._retarget_head(f,p);out.append(p)\nprint(json.dumps(out))";
 const native=JSON.parse(execFileSync('python3',['-c',code],{input:JSON.stringify(frames),encoding:'utf8',cwd:new URL('../../../',import.meta.url)}));
 const solver=new LiveRetargeter(data);
 frames.forEach((f,i)=>{for(const [name,value] of Object.entries(solver.headPose(f)))assert.ok(Math.abs(value-native[i][name])<1e-10);});
});
