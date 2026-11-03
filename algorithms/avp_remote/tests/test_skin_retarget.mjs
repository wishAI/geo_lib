import {readFile} from 'node:fs/promises';
import {test} from 'node:test';
import assert from 'node:assert/strict';
import {moduleURL as loadModuleURL} from './browser_modules.mjs';
const {HandRig}=await import(await loadModuleURL('hand-rig.js'));
import * as THREE from '../../../webgui/static/vendor/three/three.module.js';
const three=new URL('../../../webgui/static/vendor/three/three.module.js',import.meta.url).href;
const moduleURL=async name=>'data:text/javascript;base64,'+Buffer.from((await readFile(new URL('../gui/'+name,import.meta.url),'utf8')).replace("'three'",JSON.stringify(three))).toString('base64');
const mathURL=await moduleURL('pose-math.js');
const code=(await readFile(new URL('../gui/skin-retarget.js',import.meta.url),'utf8')).replace("'three'",JSON.stringify(three)).replace("'/api/artifact?path=algorithms/avp_remote/gui/pose-math.js'",JSON.stringify(mathURL));
const {SkinRetargeter}=await import('data:text/javascript;base64,'+Buffer.from(code).toString('base64'));
const {worldTransforms}=await import(mathURL);
const data=JSON.parse(await readFile(new URL('../outputs/web_scene/scene.json',import.meta.url)));
const buffer=await readFile(new URL('../outputs/web_scene/character.glb',import.meta.url)),length=buffer.readUInt32LE(12),gltf=JSON.parse(buffer.subarray(20,20+length)),binary=buffer.subarray(length+28);
function fixture(){
 const jointIds=new Set(gltf.skins[0].joints),objects=gltf.nodes.map((n,i)=>{
  const o=jointIds.has(i)?new THREE.Bone():new THREE.Group();o.name=n.name;
  o.position.fromArray(n.translation||[0,0,0]);o.quaternion.fromArray(n.rotation||[0,0,0,1]);o.scale.fromArray(n.scale||[1,1,1]);return o;
 });
 gltf.nodes.forEach((n,i)=>(n.children||[]).forEach(c=>objects[i].add(objects[c])));
 const root=new THREE.Group();gltf.scenes[0].nodes.forEach(i=>root.add(objects[i]));root.updateMatrixWorld(true);
 return {root,objects,mapper:new SkinRetargeter(root,data.joints)};
}
const error=(a,b)=>Math.max(...a.elements.map((v,i)=>Math.abs(v-b.elements[i])));
test('zero URDF pose preserves every original USD bind transform',()=>{
 const {root,objects,mapper}=fixture(),original=objects.map(o=>o.matrixWorld.clone());
 mapper.apply(worldTransforms(data.joints,{}));root.updateMatrixWorld(true);
 objects.forEach((o,i)=>assert.ok(error(o.matrixWorld,original[i])<1e-10,o.name));
});
test('all 68 skin matrices follow URDF rest-relative motion despite sibling twist bones',()=>{
 const {root,objects,mapper}=fixture(),rest=worldTransforms(data.joints,{});
 assert.equal(root.getObjectByName('arm_twist_l').parent,root.getObjectByName('forearm_stretch_l').parent);
 const skin=gltf.skins[0],accessor=gltf.accessors[skin.inverseBindMatrices],view=gltf.bufferViews[accessor.bufferView];
 const poses=[data.snapshotPose,{arm_stretch_l:-.7,arm_twist_l:.5,forearm_stretch_l:.8,forearm_twist_l:-.4,hand_l:.3,left_hip_roll_link:.1}];
 for(const pose of poses){
  const worlds=worldTransforms(data.joints,pose);mapper.apply(worlds);root.updateMatrixWorld(true);
  skin.joints.forEach((id,i)=>{
   const offset=(view.byteOffset||0)+(accessor.byteOffset||0)+i*64;
   const inverse=new THREE.Matrix4().fromArray(Array.from({length:16},(_,n)=>binary.readFloatLE(offset+n*4)));
   const actual=objects[id].matrixWorld.clone().multiply(inverse);
   const expected=worlds[objects[id].name].clone().multiply(rest[objects[id].name].clone().invert());
   assert.ok(error(actual,expected)<2e-5,objects[id].name);
  });
 }
});

test('anatomical hand rotations deform the original skin using the same bind offsets',()=>{
 const {root,objects,mapper}=fixture(),rest=worldTransforms(data.joints,{});
 const originals=objects.map(o=>o.matrixWorld.clone()),rig=new HandRig(data.joints);
 const worlds=rig.apply(worldTransforms(data.joints,data.snapshotPose),rig.solve(data.snapshot));
 mapper.apply(worlds);root.updateMatrixWorld(true);
 for(const name of rig.names){
  const i=objects.findIndex(o=>o.name===name);
  const expected=worlds[name].clone().multiply(rest[name].clone().invert()).multiply(originals[i]);
  assert.ok(error(objects[i].matrixWorld,expected)<1e-10,name);
 }
});
