import {readFile} from 'node:fs/promises';
import {test} from 'node:test';
import assert from 'node:assert/strict';
const code=await readFile(new URL('../gui/motion-clip.js',import.meta.url),'utf8');
const {createClip,appendFrame,sampleClip,validateClip,MAX_FRAMES}=await import('data:text/javascript;base64,'+Buffer.from(code).toString('base64'));
const joints=[{child:'neck',type:'revolute'}];
const identity=[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]];
function fixture(){const c=createClip('sha','local-floor');appendFrame(c,0,{neck:0},{head:identity},{head:identity},identity);appendFrame(c,1,{neck:1},null,null,null);return c;}
test('captures immutable measured samples at bounded rate, preserving tracking loss',()=>{
 const c=createClip('sha','local');const pose={neck:.2},tracking={head:structuredClone(identity)};
 assert.equal(appendFrame(c,0,pose,tracking,tracking,identity),true);
 pose.neck=8;tracking.head[0][3]=50;
 assert.equal(c.frames[0].pose.neck,.2);assert.equal(c.frames[0].tracking.head[0][3],0);
 assert.equal(appendFrame(c,.01,pose,null,null,null),false);
 assert.equal(appendFrame(c,.04,pose,null,null,null),true);
 assert.equal(c.frames[1].tracking,null);assert.equal(appendFrame(c,61,pose,null,null,null),false);
 assert.throws(()=>appendFrame(c,NaN,pose,null,null,null));
});
test('seek interpolates robot pose but never fabricates measured hand or head samples',()=>{
 const c=fixture();assert.equal(sampleClip(c,.5).pose.neck,.5);assert.deepEqual(sampleClip(c,.5).tracking,{head:identity});
 assert.equal(sampleClip(c,1).tracking,null);assert.equal(sampleClip(c,50).pose.neck,1);assert.equal(sampleClip(c,-1).pose.neck,0);
});
test('import rejects wrong model, bad timestamps, missing joints and malformed transforms',()=>{
 assert.equal(validateClip(fixture(),joints,'sha').frames.length,2);
 for(const mutate of [c=>c.urdfSha256='other',c=>c.frames[1].t=0,c=>c.frames[1].t=61,c=>c.frames[1].pose={},c=>c.frames[1].pose.neck=Infinity,c=>c.frames[1].tracking={head:[[1]]},c=>c.frames[1].rawTracking={left_arm:[]},c=>c.frames[1].alignment=[[0]],c=>c.frames=Array(MAX_FRAMES+1).fill(c.frames[0])]){
  const c=fixture();mutate(c);assert.throws(()=>validateClip(c,joints,'sha'));
 }
});
test('JSON round trip preserves timestamps, gaps, raw poses and alignment',()=>{
 const c=fixture(),loaded=validateClip(JSON.parse(JSON.stringify(c)),joints,'sha');assert.deepEqual(loaded,c);
});
