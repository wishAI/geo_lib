import {readFile} from 'node:fs/promises';
import {test} from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from '../../../webgui/static/vendor/three/three.module.js';
const code=(await readFile(new URL('../gui/pose-math.js',import.meta.url),'utf8')).replace("'three'",JSON.stringify(new URL('../../../webgui/static/vendor/three/three.module.js',import.meta.url).href));
const {worldTransforms,originMatrix,rows,fromRows,nativeMatrix,XR_HAND_NAMES}=await import('data:text/javascript;base64,'+Buffer.from(code).toString('base64'));
const scene=JSON.parse(await readFile(new URL('../outputs/web_scene/scene.json',import.meta.url)));
const skeleton=JSON.parse(await readFile(new URL('../inputs/landau_v10/landau_v10_skeleton.json',import.meta.url)));

test('browser FK matches all 68 source USD bone rest frames in meters',()=>{
  const world=worldTransforms(scene.joints);
  for(const record of skeleton.records){
    const source=fromRows(record.world_matrix);
    assert.ok(Math.max(...source.elements.map((v,i)=>Math.abs(v-world[record.name].elements[i])))<2e-5,record.name);
  }
});
test('posed USD locals reconstruct the same URDF worlds across inserted hip links',()=>{
  const pose={...scene.snapshotPose,left_hip_roll_link:.2,neck_x:.17};
  const world=worldTransforms(scene.joints,pose), rebuilt={};
  for(const record of skeleton.records){
    const parent=record.parent_index<0?null:skeleton.records[record.parent_index].name;
    const local=(parent?world[parent].clone().invert():new THREE.Matrix4()).multiply(world[record.name]);
    rebuilt[record.name]=(parent?rebuilt[parent].clone():new THREE.Matrix4()).multiply(local);
    assert.ok(Math.max(...world[record.name].elements.map((v,i)=>Math.abs(v-rebuilt[record.name].elements[i])))<1e-10);
  }
});
test('XR coordinate conversion preserves physical meter distances and matrix convention',()=>{
  assert.equal(XR_HAND_NAMES.length,25);assert.equal(new Set(XR_HAND_NAMES).size,25);
  const m=new THREE.Matrix4().makeTranslation(.1,1.6,-.4), converted=nativeMatrix(m.elements);
  const p=new THREE.Vector3().setFromMatrixPosition(converted);
  assert.ok(p.distanceTo(new THREE.Vector3(.1,.4,1.6))<1e-10);
  assert.deepEqual(fromRows(rows(m)).elements,m.elements);
});
test('joint limits are applied and invalid hierarchy rejected',()=>{
  const j=scene.joints.find(j=>j.child==='neck_x');
  assert.deepEqual(worldTransforms(scene.joints,{neck_x:1e9}).neck_x.elements,worldTransforms(scene.joints,{neck_x:j.upper}).neck_x.elements);
  assert.throws(()=>worldTransforms(scene.joints,{neck_x:NaN}));
  assert.throws(()=>worldTransforms([{...j,parent:'missing'}]));
});
test('actual exported GLB retains every USD bind frame, embedded textures and skin attributes',async()=>{
  const b=await readFile(new URL('../outputs/web_scene/character.glb',import.meta.url));
  const gltf=JSON.parse(b.subarray(20,20+b.readUInt32LE(12)));
  assert.equal(gltf.skins.length,1);assert.equal(gltf.skins[0].joints.length,68);
  assert.ok(gltf.images.every(i=>i.bufferView!==undefined&&!i.uri));
  assert.ok(gltf.meshes.every(m=>m.primitives.every(p=>p.attributes.JOINTS_0!==undefined&&p.attributes.WEIGHTS_0!==undefined)));
  const worlds={};
  function visit(index,parent){const node=gltf.nodes[index];const matrix=node.matrix?new THREE.Matrix4().fromArray(node.matrix):new THREE.Matrix4().compose(new THREE.Vector3(...(node.translation||[0,0,0])),new THREE.Quaternion(...(node.rotation||[0,0,0,1])),new THREE.Vector3(...(node.scale||[1,1,1])));const world=parent.clone().multiply(matrix);worlds[node.name]=world;for(const child of node.children||[])visit(child,world);}
  for(const index of gltf.scenes[gltf.scene||0].nodes)visit(index,new THREE.Matrix4());
  for(const record of skeleton.records){const expected=fromRows(record.world_matrix);assert.ok(Math.max(...expected.elements.map((v,i)=>Math.abs(v-worlds[record.name].elements[i])))<3e-5,record.name);}
});
