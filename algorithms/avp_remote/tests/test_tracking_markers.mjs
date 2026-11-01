import {readFile} from 'node:fs/promises';
import {test} from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from '../../../webgui/static/vendor/three/three.module.js';
const code=(await readFile(new URL('../gui/tracking-markers.js',import.meta.url),'utf8')).replace("'three'",JSON.stringify(new URL('../../../webgui/static/vendor/three/three.module.js',import.meta.url).href));
const {TrackingMarkers,trackingRecords,XRHandMarkers}=await import('data:text/javascript;base64,'+Buffer.from(code).toString('base64'));
const scene=JSON.parse(await readFile(new URL('../outputs/web_scene/scene.json',import.meta.url)));
const transform=new THREE.Matrix4().set(...scene.trackingTransform.flat());

test('native snapshot renders 55 observed points and 52 hand/forearm segments',()=>{
  const layer=new TrackingMarkers();layer.update(scene.snapshot,transform);
  assert.equal(layer.points.count,55);assert.equal(layer.lines.geometry.drawRange.count,104);
  assert.deepEqual([...layer.axes.values()].map(axis=>axis.visible),[true,true,true]);
  assert.ok(layer.records.every(r=>!r.name.includes('elbow')&&!r.name.includes('knee')));
  const expected=new THREE.Vector3().setFromMatrixPosition(new THREE.Matrix4().set(...scene.snapshot.head.flat())).applyMatrix4(transform);
  assert.ok(layer.records.find(r=>r.name==='head').position.distanceTo(expected)<1e-10);
});
test('WebXR 25-joint hands render 51 points and never invent forearm joints',()=>{
  const payload=structuredClone(scene.snapshot);for(const side of ['left','right'])payload[side+'_arm']=payload[side+'_arm'].slice(0,25);
  const layer=new TrackingMarkers();layer.update(payload,transform);
  assert.equal(layer.points.count,51);assert.equal(layer.lines.geometry.drawRange.count,96);
  assert.ok(layer.records.every(r=>!r.name.includes('forearm')));
});
test('missing tracking clears previously rendered points, connections and axes',()=>{
  const layer=new TrackingMarkers();layer.update(scene.snapshot,transform);layer.update({head:scene.snapshot.head},transform);
  assert.equal(layer.points.count,1);assert.equal(layer.lines.geometry.drawRange.count,0);
  assert.equal(layer.axes.get('left.wrist').visible,false);
  layer.update(null,transform);assert.equal(layer.points.count,0);assert.equal(layer.axes.get('head').visible,false);
});
test('invalid joint matrices are omitted without connecting across missing joints',()=>{
  const payload=structuredClone(scene.snapshot);payload.left_arm[7][0][0]=NaN;
  const result=trackingRecords(payload,transform);assert.equal(result.records.length,54);
  assert.ok(!result.records.some(r=>r.name==='left.indexIntermediateBase'));
  assert.equal(result.connections.length,50);
});
test('marker toggles persist across new frames and pure wrist payloads work',()=>{
  const layer=new TrackingMarkers();layer.setOptions({visible:false,lines:false,axes:false});
  layer.update({left_wrist:scene.snapshot.left_wrist},transform);
  assert.equal(layer.visible,false);assert.equal(layer.points.count,1);assert.equal(layer.lines.visible,false);assert.equal(layer.axes.get('left.wrist').visible,false);
  layer.setOptions({visible:true,lines:true,axes:true});assert.equal(layer.axes.get('left.wrist').visible,true);
});

// XR-space tests deliberately use a translated, rotated hand, away from the robot origin.
test('first-person hands preserve WebXR meters and orientation, with no head marker',()=>{
  const names=Array.from({length:25},(_,i)=>String(i)),ref={};
  const frame={getJointPose:(joint,space)=>{assert.equal(space,ref);return {transform:{matrix:new THREE.Matrix4().makeRotationY(.7).setPosition(2+Number(joint)*.01,1.2,-3).elements}};}};
  const layer=new XRHandMarkers();layer.updateXR(frame,[{handedness:'left',hand:new Map(names.map(n=>[n,n]))}],ref,names);
  assert.equal(layer.points.count,25);assert.equal(layer.lines.geometry.drawRange.count,48);
  assert.ok(layer.records[0].position.distanceTo(new THREE.Vector3(2,1.2,-3))<1e-10);
  assert.ok(layer.records.every(r=>r.side==='left'));assert.equal(layer.visible,true);
  assert.ok([...layer.axes.values()].every(a=>!a.visible));
  layer.updateXR(null,[],ref,names);assert.equal(layer.points.count,0);assert.equal(layer.visible,false);
});
test('first-person hands hide missing joints and never bridge across a tracking gap',()=>{
  const names=Array.from({length:25},(_,i)=>String(i));let missing=false;
  const frame={getJointPose:joint=>missing&&joint==='2'?null:{transform:{matrix:new THREE.Matrix4().setPosition(Number(joint)*.01,1,-.4).elements}}};
  const source={handedness:'right',hand:new Map(names.map(n=>[n,n]))},layer=new XRHandMarkers();
  layer.updateXR(frame,[source],{},names);assert.equal(layer.records.length,25);
  missing=true;layer.updateXR(frame,[source],{},names);assert.equal(layer.records.length,24);assert.equal(layer.lines.geometry.drawRange.count,44);
  layer.updateXR(frame,[],{},names);assert.equal(layer.records.length,0);assert.equal(layer.lines.geometry.drawRange.count,0);
});

const poseCode=(await readFile(new URL('../gui/pose-math.js',import.meta.url),'utf8')).replace("'three'",JSON.stringify(new URL('../../../webgui/static/vendor/three/three.module.js',import.meta.url).href));
const {sampleXRFrame}=await import('data:text/javascript;base64,'+Buffer.from(poseCode).toString('base64'));
test('one XR frame shares native reads between own hands, comparison markers and input',()=>{
  let reads=0;const ref={},pose={transform:{matrix:new THREE.Matrix4().elements}};
  const frame={getViewerPose:()=>pose,getJointPose:()=>{reads++;return pose;}};
  const sampled=sampleXRFrame(frame,ref);
  const sources=[{handedness:'left',hand:{get:name=>name}}],names=Array.from({length:25},(_,i)=>'joint'+i);
  const hands=new XRHandMarkers();hands.updateXR(sampled,sources,ref,names);
  for(const name of names)assert.equal(sampled.getJointPose(name,ref),pose);
  assert.equal(reads,25);assert.equal(hands.records.length,25);
});
test('unavailable or malformed XR transforms are omitted and recover on the next frame',()=>{
  const errors=[],ref={};let broken=true,reads=0;
  const pose={transform:{matrix:new THREE.Matrix4().elements}};
  const frame={getViewerPose:()=>pose,getJointPose:()=>{reads++;if(broken)throw new Error('temporarily unavailable');return pose;}};
  const sampled=sampleXRFrame(frame,ref,(kind,error)=>errors.push([kind,error.message]));
  assert.equal(sampled.getJointPose('wrist'),null);assert.equal(sampled.getJointPose('wrist'),null);
  assert.equal(reads,1);assert.deepEqual(errors,[['joint','temporarily unavailable']]);
  broken=false;assert.equal(sampleXRFrame(frame,ref).getJointPose('wrist'),pose);
  frame.getJointPose=()=>({transform:{matrix:Array(16).fill(NaN)}});
  assert.equal(sampleXRFrame(frame,ref).getJointPose('wrist'),null);
  frame.getViewerPose=()=>{throw new Error('viewer unavailable');};
  assert.equal(sampleXRFrame(frame,ref).getViewerPose(),null);
  assert.equal(sampleXRFrame(frame,null).getJointPose('wrist'),null);
});
