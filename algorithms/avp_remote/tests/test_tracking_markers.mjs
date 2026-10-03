import {readFile} from 'node:fs/promises';
import {test} from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from '../../../webgui/static/vendor/three/three.module.js';
const code=(await readFile(new URL('../gui/tracking-markers.js',import.meta.url),'utf8')).replace("'three'",JSON.stringify(new URL('../../../webgui/static/vendor/three/three.module.js',import.meta.url).href));
const {TrackingMarkers,trackingRecords}=await import('data:text/javascript;base64,'+Buffer.from(code).toString('base64'));
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
