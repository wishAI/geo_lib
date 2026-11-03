import {readFile} from 'node:fs/promises';
import {execFileSync} from 'node:child_process';
import {test} from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from '../../../webgui/static/vendor/three/three.module.js';
import {moduleURL} from './browser_modules.mjs';
const {worldTransforms}=await import(await moduleURL('pose-math.js'));
const code=(await readFile(new URL('../gui/comparison.js',import.meta.url),'utf8')).replace("'three'",JSON.stringify(new URL('../../../webgui/static/vendor/three/three.module.js',import.meta.url).href));
const {bodySegments, unitreePose, UNITREE_DISPLAY_YAW}=await import('data:text/javascript;base64,'+Buffer.from(code).toString('base64'));
test('schematic has no body without head and drops missing hand connections',()=>{
  const records=[{name:'head',position:new THREE.Vector3(0,0,1.5)},{name:'left.wrist',position:new THREE.Vector3(-.4,.2,1)}];
  assert.equal(bodySegments(records).length,4);
  assert.equal(bodySegments(records.slice(1)).length,0);
  assert.equal(bodySegments(records.slice(0,1)).length,3);
  assert.deepEqual(records.map(r=>r.name),['head','left.wrist']);
});
test('browser H1-2 mapping matches authoritative Python including elbows and mimic joints',async()=>{
  const data=JSON.parse(await readFile(new URL('../outputs/web_scene/unitree.json',import.meta.url)));
  const scene=JSON.parse(await readFile(new URL('../outputs/web_scene/scene.json',import.meta.url)));
  const samples=[{},scene.snapshotPose,{forearm_stretch_l:3,forearm_stretch_r:-2,index1_l:-1.4,thumb2_r:-.9,shoulder_l:3,arm_stretch_r:-2}];
  const expected=JSON.parse(execFileSync('python3',['-c',`import sys,json;sys.path.insert(0,'algorithms/avp_remote');from avp_g1_pose import map_landau_pose_to_h1_2_pose;print(json.dumps([map_landau_pose_to_h1_2_pose(p) for p in json.loads(sys.argv[1])]))`,JSON.stringify(samples)],{encoding:'utf8'}));
  samples.forEach((sample,i)=>{
    const actual=unitreePose(data,sample);
    for(const j of data.joints){
      if(j.mimic){const parent=data.joints.find(p=>p.name===j.mimic.joint);assert.ok(Math.abs(actual[j.child]-THREE.MathUtils.clamp(actual[parent.child]*j.mimic.multiplier+j.mimic.offset,j.lower,j.upper))<1e-8);}
      else assert.ok(Math.abs(actual[j.child]-THREE.MathUtils.clamp(expected[i][j.name]||0,j.lower,j.upper))<1e-8,j.name);
    }
  });
});

test('negative Landau curl closes every H1-2 finger and moves linked meshes on both hands',async()=>{
  const data=JSON.parse(await readFile(new URL('../outputs/web_scene/unitree.json',import.meta.url)));
  const open=unitreePose(data,{}),rest=worldTransforms(data.joints,open,data.root);
  for(const [prefix,suffix] of [['L','l'],['R','r']])for(const finger of ['thumb','index','middle','ring','pinky']){
    const source={};for(const i of finger==='thumb'?[2,3]:[1,2,3])source[`${finger}${i}_${suffix}`]=-1.4;
    const pose=unitreePose(data,source),worlds=worldTransforms(data.joints,pose,data.root);
    const driver=finger==='thumb'?`${prefix}_thumb_proximal_pitch_joint`:`${prefix}_${finger}_proximal_joint`;
    const j=data.joints.find(j=>j.name===driver);
    assert.ok(j&&j.type==='revolute',driver);
    assert.ok(pose[j.child]>.35,`${driver} must close, not clamp to zero`);
    for(const linked of data.joints.filter(j=>j.mimic?.joint===driver)){
      assert.ok(pose[linked.child]>0,linked.name);
      assert.ok(new THREE.Vector3().setFromMatrixPosition(worlds[linked.child]).distanceTo(new THREE.Vector3().setFromMatrixPosition(rest[linked.child]))>.005,linked.child+' mesh must move');
    }
    // Curling one digit must not curl an unrelated finger or the other hand.
    for(const other of data.joints.filter(j=>/^[LR]_.*proximal_joint$/.test(j.name)&&j.name!==driver))assert.equal(pose[other.child],open[other.child]);
  }
});

test('Unitree front faces the same displayed direction as Landau',()=>{
  const forward=new THREE.Vector3(1,0,0).applyAxisAngle(new THREE.Vector3(0,0,1),UNITREE_DISPLAY_YAW).applyAxisAngle(new THREE.Vector3(1,0,0),-Math.PI/2);
  assert.ok(forward.distanceTo(new THREE.Vector3(0,0,1))<1e-9);
});
