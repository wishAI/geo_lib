import {readFile} from 'node:fs/promises';
import {test} from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from '../../../webgui/static/vendor/three/three.module.js';
const code=(await readFile(new URL('../gui/xr-buttons.js',import.meta.url),'utf8')).replace("'three'",JSON.stringify(new URL('../../../webgui/static/vendor/three/three.module.js',import.meta.url).href));
const {XRButtonInput}=await import('data:text/javascript;base64,'+Buffer.from(code).toString('base64'));
function fixture(){
 const scene=new THREE.Scene(),panel=new THREE.Group();panel.position.set(1,1.9,-2);panel.rotation.y=.4;panel.scale.setScalar(.68);scene.add(panel);
 const buttons=['Record','Play'].map((name,i)=>{const b=new THREE.Mesh(new THREE.PlaneGeometry(.22,.1),new THREE.MeshBasicMaterial());b.position.x=i*.24;b.userData.action=name;panel.add(b);return b;});
 scene.updateMatrixWorld(true);const eye=new THREE.Vector3(1,1.6,0),target=buttons[0].getWorldPosition(new THREE.Vector3());
 const matrix=new THREE.Matrix4().lookAt(eye,target,new THREE.Vector3(0,1,0)).setPosition(eye),hits=[];
 const input=new XRButtonInput(scene,buttons,{activate:b=>hits.push(b.userData.action),enabled:b=>!b.userData.disabled});
 const source={targetRayMode:'transient-pointer',targetRaySpace:{}},event={inputSource:source,frame:{getPose:()=>({transform:{matrix:matrix.elements}})}};
 return {input,buttons,hits,source,event,matrix};
}
test('head reticle animates hover without triggering a click',()=>{
 const f=fixture();f.input.update({getViewerPose:()=>({transform:{matrix:f.matrix.elements}})},[],{});
 assert.equal(f.input.hover,f.buttons[0]);assert.ok(f.buttons[0].scale.x>1);assert.ok(f.buttons[0].userData.focusBorder.material.opacity>0);assert.equal(f.input.cursor.visible,true);assert.deepEqual(f.hits,[]);
});
test('pinch start locks target; moved or missing release ray still activates exactly once',()=>{
 const f=fixture();f.input.start(f.event,{});
 f.event.frame.getPose=()=>null;f.input.select(f.event,{});f.input.end(f.event);f.input.select(f.event,{});
 assert.deepEqual(f.hits,['Record']);
 // A new pinch of the same input source must be usable again.
 f.event.frame.getPose=()=>({transform:{matrix:f.matrix.elements}});f.input.start(f.event,{});f.input.select(f.event,{});assert.deepEqual(f.hits,['Record','Record']);
});
test('a cancelled gesture or an actual ray miss never activates the head-hovered button',()=>{
 const f=fixture();f.input.update({getViewerPose:()=>({transform:{matrix:f.matrix.elements}})},[],{});
 f.input.start(f.event,{});f.input.end(f.event);assert.deepEqual(f.hits,[]);
 f.event.frame.getPose=()=>({transform:{matrix:new THREE.Matrix4().makeTranslation(9,9,9).elements}});
 f.input.start(f.event,{});f.input.select(f.event,{});assert.deepEqual(f.hits,[]);
});
test('disabled targets do not activate and session reset clears focus',()=>{
 const f=fixture();f.input.start(f.event,{});f.buttons[0].userData.disabled=true;f.input.select(f.event,{});assert.deepEqual(f.hits,[]);
 f.input.reset();assert.equal(f.input.cursor.visible,false);assert.equal(f.input.presses.size,0);
});
