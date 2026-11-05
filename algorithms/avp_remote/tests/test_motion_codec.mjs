import {test} from 'node:test';
import assert from 'node:assert/strict';
import {moduleURL} from './browser_modules.mjs';
const {packClip,unpackClip,compressedClip,readClipFile}=await import(await moduleURL('motion-codec.js'));
const identity=()=>[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]];
function fixture(){const matrix=identity(),raw={head:matrix,left_arm:Array.from({length:25},identity),left_wrist:matrix};return {
  format:'avp-motion-clip',version:1,retargetVersion:5,custom:'preserved',frames:[
    {t:0,pose:{neck:0,leg:.2},tracking:structuredClone(raw),rawTracking:raw,alignment:matrix,handPose:{}},
    {t:1,pose:{neck:.4,leg:.2},tracking:null,rawTracking:null,alignment:null},
  ]};}
test('lossless packed roundtrip preserves missing tracking, handPose and metadata',()=>{
  const clip=fixture(),before=structuredClone(clip),packed=packClip(clip);
  assert.equal(packed.frames[0].tracking,'derived');assert.equal(packed.frames[0].rawTracking.left_wrist,'arm0');
  assert.deepEqual(packed.constantPose,{leg:.2});assert.deepEqual(unpackClip(JSON.parse(JSON.stringify(packed))),clip);assert.deepEqual(clip,before);
});
test('differing tracking and nonexact affine bottom row use lossless fallback',()=>{
  const clip=fixture();clip.frames[0].tracking.head[0][3]=.1;clip.frames[0].rawTracking.head[3][0]=1e-8;
  const packed=packClip(clip);assert.notEqual(packed.frames[0].tracking,'derived');assert.equal(packed.frames[0].rawTracking.head.length,16);
  assert.deepEqual(unpackClip(packed),clip);
});
test('gzip export imports alongside legacy JSON without changing saved poses',async()=>{
  const clip=fixture();assert.deepEqual(await readClipFile(await compressedClip(clip)),clip);
  assert.deepEqual(await readClipFile(new Blob([JSON.stringify(clip)])),clip);
});
test('malformed dictionary, alias, matrix and derived tracking are rejected',()=>{
  for(const mutate of [p=>p.poseKeys.push('neck'),p=>p.frames[0].alignment=999,
    p=>p.frames[0].rawTracking.left_arm=[],p=>p.frames[0].rawTracking.head=[1],
    p=>p.frames[0].rawTracking=null,p=>p.frames[0].pose=[Infinity]]){
    const packed=packClip(fixture());mutate(packed);assert.throws(()=>unpackClip(packed));
  }
});
