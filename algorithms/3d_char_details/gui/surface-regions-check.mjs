import fs from 'node:fs';
import assert from 'node:assert/strict';
import * as T from './three.mjs';
import {GLTFLoader} from './loader.mjs';
import {GLTFExporter} from './exporter.mjs';
import {surfaceRegions} from './surface-regions.mjs';
import {bodyPlacement} from './body-placement.mjs';
import {installBodyTransition,BODY_TRANSITION_CONTROLS} from './body-transition.mjs';
import {decodeAtlas,pickUV} from './influence-view.mjs';
import {validateRegions,editedWeight} from './uv-regions.mjs';
globalThis.ProgressEvent=class{constructor(type,args){Object.assign(this,args);}};
const data=fs.readFileSync(process.argv[2]),length=data.readUInt32LE(12),json=JSON.parse(data.subarray(20,20+length));
json.buffers[0].uri='data:application/octet-stream;base64,'+data.subarray(28+length).toString('base64');delete json.images;delete json.textures;delete json.samplers;
const gltf=await new GLTFLoader().parseAsync(JSON.stringify(json),''),{scene}=gltf,report=JSON.parse(fs.readFileSync(process.argv[3])),bones=new Map(),meshes=[];
scene.traverse(o=>{if(o.isBone)bones.set(o.name,{bone:o,quaternion:o.quaternion.clone(),scale:o.scale.clone()});if(o.isMesh)meshes.push(o);});
scene.traverse(o=>{if(o.isMesh){const a=gltf.parser.associations.get(o);o.userData.editAtlasId=`${a.meshes}:${a.primitives}`;}});
installBodyTransition(scene);const placement=bodyPlacement(scene,bones);
const names=new Set(['headWidth','faceWidth','earLength','eyeSize','cheekFullness','muzzleLength','jawRecess','mouthLength','mouthCurvature',...Object.keys(BODY_TRANSITION_CONTROLS),...Object.entries(report.body_reconstruction.adjustment_controls).filter(([,m])=>m.kind==='body').map(([n])=>n)]);
const surface=surfaceRegions(scene,names),snapshot=()=>meshes.map(m=>m.geometry.morphAttributes.position.map(a=>new Float32Array(a.array))),original=snapshot();
const atlasData=JSON.parse(fs.readFileSync(process.argv[4])),atlas=decodeAtlas(atlasData,report.glb_sha256,scene);surface.setAtlas(atlas);
assert.equal(atlas.size,meshes.length);assert.throws(()=>decodeAtlas(atlasData,'wrong',scene));
for(const [m,a]of atlas){for(let c=0;c<a.uv.length;c+=6){const [x,y,X,Y,u,v]=a.uv.slice(c,c+6);assert(Math.abs((X-x)*(v-y)-(Y-y)*(u-x))>1e-16,'Collapsed atlas triangle');}assert.equal(a.uv.length,(m.geometry.index?.count??m.geometry.attributes.position.count)*2);assert(a.uv.every(v=>Number.isFinite(v)&&v>=0&&v<=1));}
const error=(a,b)=>{let e=0;for(let i=0;i<a.length;i++)for(let j=0;j<a[i].length;j++)for(let k=0;k<a[i][j].length;k++)e=Math.max(e,Math.abs(a[i][j][k]-b[i][j][k]));return e;};
const full={enabled:true,projection:'front',feather:.05,points:[[0,0],[1,0],[1,1],[0,1]]},partial={...full,points:[[.1,.1],[.5,.1],[.5,.85],[.1,.85]]};
const active=[...names].filter(n=>surface.has(n));assert(surface.has('muzzleLength'));assert(surface.has('jawRecess'));assert(active.length>20);
// First opening has neither a saved region nor a cache entry. Exercise this
// before apply() populates the cache, then exercise the undefined-signature hit.
for(const name of active){
 const first=surface.influence(name),again=surface.influence(name,{});
 assert(first.length>0,'Missing first-open field: '+name);
 for(let i=0;i<first.length;i++){
  assert(first[i].atlas,'Missing first-open atlas: '+name);
  assert(first[i].weights.every(v=>v===1),'New field must preserve authored movement: '+name);
  assert.deepEqual(again[i].weights,first[i].weights,'Cached first-open field: '+name);
 }
}
assert.equal(error(original,snapshot()),0,'Inspecting an unedited field changed geometry');
// Seam correspondence is captured before masking and spans material primitives.
const groups=new Map();for(const mesh of meshes){if(mesh.userData.part_type==='clothing')continue;const p=mesh.geometry.attributes.position;for(let i=0;i<p.count;i++){const v=new T.Vector3().fromBufferAttribute(p,i).applyMatrix4(mesh.matrixWorld),key=v.toArray().map(x=>Math.round(x*1e6)).join(',');if(!groups.has(key))groups.set(key,[]);groups.get(key).push([mesh,i]);}}
const joins=[...groups.values()].filter(a=>a.length>1);
function seamDrift(){let drift=0;for(const group of joins){const [m,i]=group[0],a=m.getVertexPosition(i,new T.Vector3()).applyMatrix4(m.matrixWorld);for(const [n,j]of group.slice(1)){const b=n.getVertexPosition(j,new T.Vector3()).applyMatrix4(n.matrixWorld);drift=Math.max(drift,a.distanceTo(b));}}return drift;}
const initialSeam=seamDrift();let maxSeam=initialSeam,changed=0;
for(const name of active){surface.apply({['surface:'+name]:full});assert.equal(error(original,snapshot()),0,'Full region changed '+name);surface.apply({['surface:'+name]:partial});const a=snapshot();assert(error(original,a)>0,'Partial region did not affect '+name);changed++;surface.apply({['surface:'+name]:partial});assert.equal(error(a,snapshot()),0,'Mask accumulated '+name);surface.apply({});assert.equal(error(original,snapshot()),0,'Reset drift '+name);}
// A region entirely outside authored support must remove positions AND shading.
for(const name of ['muzzleLength','jawRecess','bodyChestWidth']){
 surface.apply({['surface:'+name]:{...full,feather:0,points:[[0,0],[.01,0],[.01,.01],[0,.01]]}});
 for(const m of meshes){const i=m.morphTargetDictionary?.[name];if(i===undefined)continue;const g=m.geometry,p=g.morphAttributes.position[i],n=g.morphAttributes.normal?.[i];if(!p.array.some(v=>v!==0)&&n)assert(n.array.every(v=>Math.abs(v)<1e-6),'Excluded region changed shading: '+name+' '+m.name+' '+n.array.reduce((a,v)=>Math.max(a,Math.abs(v)),0));}
 surface.restore();
}
const dab={type:'brush',sheet:'test',center:[.5,.5],radius:.2,value:.6,opacity:1};
assert(Math.abs(editedWeight({enabled:true,edits:[dab]},[{sheet:'test',u:.5,v:.5}])-.6)<1e-9);
assert(Math.abs(editedWeight({enabled:true,edits:[dab]},[{sheet:'test',u:.6,v:.5}])-.8)<1e-9);
assert.equal(editedWeight({enabled:true,edits:[dab]},[{sheet:'other',u:.5,v:.5}]),1);
assert.equal(editedWeight({enabled:false,edits:[dab]},[{sheet:'test',u:.5,v:.5}]),1);
assert.equal(editedWeight({enabled:true,edits:[dab]},[{sheet:'test',u:.5,v:.5},{sheet:'test',u:.5,v:.5}]),.6);
assert.throws(()=>validateRegions({'surface:muzzleLength':{enabled:true,edits:[{...dab,value:2}]}}));
assert.throws(()=>surface.checkRegion({enabled:true,atlasHash:'wrong',edits:[dab]}));
// The same fractional edit applies across every UV alias and material split.
const sheets=[...new Set([...atlas.values()].map(a=>a.sheet))],fractional={enabled:true,atlasHash:atlasData.atlasHash,edits:sheets.map(sheet=>({type:'polygon',sheet,points:[[0,0],[1,0],[1,1],[0,1]],feather:0,value:.6,opacity:1}))};
validateRegions({'surface:muzzleLength':fractional});surface.apply({'surface:muzzleLength':fractional});
for(let mi=0;mi<meshes.length;mi++){const m=meshes[mi],i=m.morphTargetDictionary?.muzzleLength;if(i===undefined)continue;const a=m.geometry.morphAttributes.position[i].array,b=original[mi][i];for(let j=0;j<a.length;j++)assert(Math.abs(a[j]-b[j]*.6)<1e-8);}
const field={entries:surface.influence('muzzleLength',{'surface:muzzleLength':fractional})};
for(const e of field.entries){assert(e.weights.every(v=>Math.abs(v-.6)<1e-6));const g=e.mesh.geometry,idx=g.index,ids=[0,1,2].map(i=>idx?idx.getX(i):i),vertices=ids.map(i=>e.mesh.getVertexPosition(i,new T.Vector3()).applyMatrix4(e.mesh.matrixWorld));if(new T.Triangle(...vertices).getArea()<1e-12)continue;const point=vertices[0].clone().add(vertices[1]).add(vertices[2]).multiplyScalar(1/3),hit={object:e.mesh,faceIndex:0,face:{a:ids[0],b:ids[1],c:ids[2]},point},picked=pickUV(field,hit);assert(Math.abs(picked.u-(e.atlas.uv[0]+e.atlas.uv[2]+e.atlas.uv[4])/3)<1e-5);}
surface.apply({});
const regions=validateRegions(Object.fromEntries(['muzzleLength','jawRecess','mouthLength','mouthCurvature'].map(n=>['surface:'+n,partial])),k=>k.startsWith('surface:'));
surface.apply(regions);
for(const jaw of [0,.5,1])for(const shape of [-1,0,1]){
 for(const m of meshes)for(const [n,i]of Object.entries(m.morphTargetDictionary||{}))m.morphTargetInfluences[i]=n==='jawDrop'?jaw:['muzzleLength','jawRecess','mouthLength','mouthCurvature'].includes(n)?shape:0;
 maxSeam=Math.max(maxSeam,seamDrift());
}
// Source already has some coincident-but-unjoined surfaces. Their baseline is
// checked with the same poses, rather than claiming every coincidence is a seam.
surface.restore();let baselineSeam=initialSeam;for(const jaw of [0,.5,1])for(const shape of [-1,0,1]){for(const m of meshes)for(const [n,i]of Object.entries(m.morphTargetDictionary||{}))m.morphTargetInfluences[i]=n==='jawDrop'?jaw:['muzzleLength','jawRecess','mouthLength','mouthCurvature'].includes(n)?shape:0;baselineSeam=Math.max(baselineSeam,seamDrift());}
assert(maxSeam<=baselineSeam+2e-6,'Projected masks separated shared facial vertices');
for(const m of meshes)m.morphTargetInfluences?.fill(0);
for(const [scale,offset]of [[.75,-.05],[1.25,.05],[1,0]]){surface.restore();placement(scale,offset,0);surface.capture();const unmasked=snapshot();surface.apply(regions);const masked=snapshot();surface.apply(JSON.parse(JSON.stringify(regions)));assert.equal(error(masked,snapshot()),0);surface.restore();assert.equal(error(unmasked,snapshot()),0);}
assert(error(original,snapshot())<1e-7,'Placement/reset drift');
regions['surface:muzzleLength']=fractional;surface.apply(regions);const before=snapshot();
for(const m of meshes)for(const a of m.geometry.morphAttributes.normal||[])assert(a.array.every(Number.isFinite));
globalThis.FileReader=class{readAsArrayBuffer(blob){blob.arrayBuffer().then(x=>{this.result=x;this.onloadend?.();});}readAsDataURL(blob){blob.arrayBuffer().then(x=>{this.result='data:application/octet-stream;base64,'+Buffer.from(x).toString('base64');this.onloadend?.();});}};
const exported=await new GLTFExporter().parseAsync(scene,{binary:true,trs:true,onlyVisible:false}),loaded=await new GLTFLoader().parseAsync(exported,'');
const reimport=new Map();loaded.scene.traverse(o=>{if(o.isMesh)reimport.set(o.name,o);});let exportError=0;for(const m of meshes){const n=reimport.get(m.name);assert(n,'Missing exported mesh '+m.name);for(const [name,i]of Object.entries(m.morphTargetDictionary||{})){const j=n.morphTargetDictionary[name];assert(j!==undefined);const a=m.geometry.morphAttributes.position[i].array,b=n.geometry.morphAttributes.position[j].array;for(let k=0;k<a.length;k++)exportError=Math.max(exportError,Math.abs(a[k]-b[k]));}}
assert(exportError<1e-7);assert.equal(error(before,snapshot()),0);surface.apply({});assert(error(original,snapshot())<1e-7);
console.log(JSON.stringify({status:'passed',shapeControls:changed,sharedVertexGroups:joins.length,maxSeam,baselineSeam,exportError,checks:['first-open and cached unedited fields for every shape','full and disabled exact equivalence','partial region changes geometry','repeat and reset','combined mouth/profile interfaces','placement extremes','finite normals and zero-influence shading','actual GLB export/reimport','atlas correspondence and 3D picking','fractional influence across UV seams']},null,2));
