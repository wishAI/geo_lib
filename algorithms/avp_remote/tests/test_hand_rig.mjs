import {readFile} from 'node:fs/promises';
import {test} from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from '../../../webgui/static/vendor/three/three.module.js';
import {moduleURL} from './browser_modules.mjs';
const {HandRig,validateHandPose}=await import(await moduleURL('hand-rig.js'));
const {worldTransforms,rows}=await import(await moduleURL('pose-math.js'));
const data=JSON.parse(await readFile(new URL('../outputs/web_scene/scene.json',import.meta.url)));
const rig=new HandRig(data.joints),rest=worldTransforms(data.joints,{});
const pos=m=>new THREE.Vector3().setFromMatrixPosition(m),direction=m=>new THREE.Vector3(0,1,0).transformDirection(m);
function trackingFromWorlds(worlds){
  const frame={};
  for(const [side,s] of Object.entries(rig.sides)){
    const inverse=worlds[s.hand].clone().invert(),stack=Array(25);stack[0]=rows(new THREE.Matrix4());
    for(const chain of s.chains)for(const {name,index} of chain){
      const local=inverse.clone().multiply(worlds[name]);stack[index]=rows(local);
      if(index===chain.at(-1).index)stack[index+1]=rows(local.clone().multiply(new THREE.Matrix4().makeTranslation(0,.008,0)));
    }
    frame[side+'_arm']=stack;
  }
  return frame;
}
test('model rest is identity even with pre-bent thumb and pinky bind frames',()=>{
  const hands=rig.solve(trackingFromWorlds(rest));assert.equal(Object.keys(hands).length,38);
  for(const [name,q] of Object.entries(hands))assert.ok(new THREE.Quaternion().fromArray(q).angleTo(new THREE.Quaternion())<1e-6,name);
  const worlds=rig.apply(worldTransforms(data.joints,{}),hands);
  for(const name of rig.names)assert.ok(Math.max(...worlds[name].elements.map((x,i)=>Math.abs(x-rest[name].elements[i])))<1e-6,name);
});
test('open, curled, splayed and opposing fingers follow every measured segment with unchanged bone lengths',()=>{
  for(const curl of [0,.5,1.3,1.8]){
    const frame=trackingFromWorlds(rest);
    for(const [side,s] of Object.entries(rig.sides))for(const chain of s.chains){
      // Leave metacarpals defining the palm unchanged; prescribe arbitrary 3D
      // phalanx directions, including thumb opposition and little-finger curl.
      const stack=frame[side+'_arm'];let p=new THREE.Vector3(...stack[chain[1].index].slice(0,3).map(r=>r[3]));
      for(let i=1;i<chain.length;i++){
        const q=new THREE.Quaternion().setFromEuler(new THREE.Euler(-curl*i,.15*(side==='left'?1:-1),.2));
        p=p.clone().add(new THREE.Vector3(0,.025,0).applyQuaternion(q));stack[chain[i].index+1]=rows(new THREE.Matrix4().setPosition(p));
      }
    }
    const hands=rig.solve(frame),worlds=rig.apply(worldTransforms(data.joints,{}),hands);
    for(const [side,s] of Object.entries(rig.sides)){
      const inverse=rest[s.hand].clone().invert(),stack=frame[side+'_arm'];
      for(const chain of s.chains)for(const {name,parent,index,origin} of chain){
        const a=new THREE.Vector3(...stack[index].slice(0,3).map(r=>r[3])),b=new THREE.Vector3(...stack[index+1].slice(0,3).map(r=>r[3]));
        assert.ok(direction(inverse.clone().multiply(worlds[name])).angleTo(b.sub(a))<1e-6,name);
        assert.ok(Math.abs(pos(worlds[name]).distanceTo(pos(worlds[parent]))-pos(origin).length())<1e-10,name);
        assert.ok(Math.abs(worlds[name].determinant()-1)<1e-10,name);
      }
    }
  }
});
test('rig is rigid-motion invariant and deterministic after seek; missing or collapsed hands do not retain curls',()=>{
  const frame=trackingFromWorlds(rest),a=rig.solve(frame),m=new THREE.Matrix4().makeRotationZ(1.2).setPosition(3,4,5);
  const transformed=Object.fromEntries(Object.entries(frame).map(([k,stack])=>[k,stack.map(r=>rows(m.clone().multiply(new THREE.Matrix4().set(...r.flat()))))]));
  for(const [name,q] of Object.entries(rig.solve(transformed)))assert.ok(new THREE.Quaternion().fromArray(q).angleTo(new THREE.Quaternion().fromArray(a[name]))<1e-6);
  rig.solve(data.snapshot);assert.deepEqual(rig.solve(frame),a);
  assert.deepEqual(rig.solve(null),{});
  assert.deepEqual(rig.solve({left_arm:Array(25).fill(rows(new THREE.Matrix4()))}),{});
  assert.ok(validateHandPose(a));assert.ok(!validateHandPose({thumb1_l:[0,0,0,2]}));
});

test('partial saved finger poses preserve connected descendants and terminal bones',()=>{
  const worlds=rig.apply(worldTransforms(data.joints,{thumb2_l:-1}),{thumb1_l:new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0,0,1),.4).toArray()});
  const chain=rig.sides.left.chains[0];
  for(const {name,parent,origin} of chain.slice(1)){
    const local=worlds[parent].clone().invert().multiply(worlds[name]);
    assert.ok(Math.max(...local.elements.map((x,i)=>Math.abs(x-origin.elements[i])))<1e-10,name);
  }
});
