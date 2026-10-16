import fs from 'node:fs';
import assert from 'node:assert/strict';
import * as T from './three.mjs';
import {GLTFLoader} from './loader.mjs';
import {GLTFExporter} from './exporter.mjs';
import {surfaceRegions} from './surface-regions.mjs';
import {bodyPlacement} from './body-placement.mjs';
import {installBodyTransition,BODY_TRANSITION_CONTROLS} from './body-transition.mjs';
import {decodeAtlas,originalAtlas,pickUV,affectedBounds,frameUVBounds} from './influence-view.mjs';
import {validateRegions,editedWeight,mirrorCorrespondence,volumeWeight,validateVolume,symmetricPolygon,centerVolume,displacementField} from './uv-regions.mjs';
globalThis.ProgressEvent=class{constructor(type,args){Object.assign(this,args);}};
const data=fs.readFileSync(process.argv[2]),length=data.readUInt32LE(12),json=JSON.parse(data.subarray(20,20+length));
json.buffers[0].uri='data:application/octet-stream;base64,'+data.subarray(28+length).toString('base64');delete json.images;delete json.textures;delete json.samplers;
const gltf=await new GLTFLoader().parseAsync(JSON.stringify(json),''),{scene}=gltf,report=JSON.parse(fs.readFileSync(process.argv[3])),bones=new Map(),meshes=[];
scene.traverse(o=>{if(o.isBone)bones.set(o.name,{bone:o,quaternion:o.quaternion.clone(),scale:o.scale.clone()});if(o.isMesh)meshes.push(o);});
scene.traverse(o=>{if(o.isMesh){const a=gltf.parser.associations.get(o);o.userData.editAtlasId=`${a.meshes}:${a.primitives}`;}});
installBodyTransition(scene);const placement=bodyPlacement(scene,bones);
const names=new Set(['headWidth','faceWidth','earLength','eyeSize','cheekFullness','muzzleLength','jawRecess','mouthLength','mouthCurvature',...Object.keys(BODY_TRANSITION_CONTROLS),...Object.entries(report.body_reconstruction.adjustment_controls).filter(([,m])=>m.kind==='body').map(([n])=>n)]);
const surface=surfaceRegions(scene,names,{faceNames:new Set(['headWidth','faceWidth','earLength','eyeSize','cheekFullness','muzzleLength','jawRecess','mouthLength','mouthCurvature']),faceCenter:report.body_reconstruction.head_rigid_translation[0]}),snapshot=()=>meshes.map(m=>m.geometry.morphAttributes.position.map(a=>new Float32Array(a.array))),original=snapshot();
const atlasData=JSON.parse(fs.readFileSync(process.argv[4])),atlas=decodeAtlas(atlasData,report.glb_sha256,scene);surface.setAtlas(atlas);
surface.setAtlas(new Map([...atlas].map(([m,a])=>[m,{original:a.original}])));assert(surface.influence('muzzleLength').every(e=>e.atlas.kind==='Original model UVs'));surface.setAtlas(atlas);
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
// Native display coordinates must match the GLB exactly, including UV seams.
for(const [mesh,descriptor]of atlas){const a=descriptor.original,g=mesh.geometry,source=g.attributes.uv;for(let c=0;c<a.uv.length/2;c++){const i=g.index?g.index.getX(c):c;assert.equal(a.uv[c*2],source?.getX(i)||0);assert.equal(a.uv[c*2+1],source?.getY(i)||0);}}
const nativeField={entries:surface.influence('muzzleLength'),peak:.014},nativeSheet='uv:0',bounds=affectedBounds(nativeField,nativeSheet);assert(bounds);
const framed=frameUVBounds(bounds);assert(framed.zoom>1,'Muzzle must open zoomed in');for(const u of [bounds.min[0],bounds.max[0]])for(const v of [bounds.min[1],bounds.max[1]]){const x=u*512*framed.zoom+framed.pan[0],y=(1-v)*512*framed.zoom+framed.pan[1];assert(x>=60&&x<=452&&y>=60&&y<=452,'Auto-frame clips the affected area');}
assert.equal(affectedBounds({entries:surface.influence('bodyChestWidth'),peak:1},nativeSheet),null,'Collapsed torso UVs must not masquerade as a usable unwrap');
// Reject repeated placeholder unit triangles, but retain a normal mirrored pair.
const placeholderGeometry=new T.BufferGeometry();placeholderGeometry.setAttribute('position',new T.BufferAttribute(new Float32Array(60*3),3));placeholderGeometry.setAttribute('uv',new T.BufferAttribute(Float32Array.from(Array.from({length:20},()=>[0,1,1,1,1,0]).flat()),2));
const placeholderMesh=new T.Mesh(placeholderGeometry);assert(originalAtlas(placeholderMesh,'test','test').valid.every(v=>v===0));
placeholderGeometry.setAttribute('position',new T.BufferAttribute(new Float32Array(6*3),3));placeholderGeometry.setAttribute('uv',new T.BufferAttribute(Float32Array.from([0,1,1,1,1,0,0,1,1,1,1,0]),2));assert(originalAtlas(placeholderMesh,'test','test').valid.every(v=>v===1));placeholderGeometry.dispose();placeholderMesh.material.dispose();
const nativeRegion={enabled:true,edits:[{type:'polygon',sheet:nativeSheet,points:[[0,0],[1,0],[1,1],[0,1]],feather:0,value:.6,opacity:1,symmetry:false}]};
const nativeResult=surface.influence('muzzleLength',{'surface:muzzleLength':nativeRegion});let nativeEdited=0;for(const e of nativeResult)for(let i=0;i<e.weights.length;i++)if(e.atlas.sheet===nativeSheet&&e.atlas.vertices[i]){assert(Math.abs(e.weights[i]-.6)<1e-6);nativeEdited++;}assert(nativeEdited>0);
assert(surface.influence('muzzleLength',{'surface:muzzleLength':{...nativeRegion,enabled:false}}).every(e=>e.weights.every(w=>w===1)));
// Mirroring follows an offset anatomical plane and rotated/separate UV charts.
const nodes=[{p:{x:-.015,y:0,z:0},samples:[{sheet:'uv:test',u:.2,v:.3}]},{p:{x:.005,y:0,z:0},samples:[{sheet:'uv:test',u:.8,v:.6}]}];mirrorCorrespondence(nodes,-.005,.001);assert.equal(nodes[0].mirror,nodes[1]);assert.equal(nodes[1].mirror,nodes[0]);
const symmetricEdit={type:'polygon',sheet:'uv:test',points:[[.1,.2],[.3,.2],[.3,.4],[.1,.4]],value:.6,opacity:.5,feather:0,symmetry:true},symmetricRegion={enabled:true,edits:[symmetricEdit]};
for(const n of nodes)assert(Math.abs(editedWeight(symmetricRegion,n.samples,1,n.mirror.samples)-.8)<1e-9);
assert.equal(editedWeight({...symmetricRegion,edits:[{...symmetricEdit,symmetry:false}]},nodes[1].samples,1,nodes[1].mirror.samples),1);
assert.equal(editedWeight(symmetricRegion,nodes[0].samples,1,nodes[0].samples),.8,'Midline opacity applied twice');
assert.deepEqual(validateRegions({'surface:muzzleLength':symmetricRegion})['surface:muzzleLength'],symmetricRegion);
assert.throws(()=>validateRegions({'surface:muzzleLength':{enabled:true,edits:[{...symmetricEdit,symmetry:'yes'}]}}));

// Volumes use the same neutral positions as the visible clay reference, with no UV dependency.
const vol={shape:'sphere',position:[0,0,0],rotation:[0,0,0],size:[1,1,1],feather:.1,symmetry:false,centerX:0};
const at=(x,y=0,z=0)=>new T.Vector3(x,y,z);
assert.equal(volumeWeight(vol,at(0)),1);assert(Math.abs(volumeWeight(vol,at(.45))-.5)<1e-8);assert.equal(volumeWeight(vol,at(.5)),0);assert.equal(volumeWeight(vol,at(.51)),0);
// Feather position moves only the falloff band. Centered mode keeps the half-strength boundary fixed at any width.
const centered={...vol,version:2,featherBias:.5};for(const width of [.05,.1,.3]){const v={...centered,feather:width};assert(Math.abs(volumeWeight(v,at(.5))-.5)<1e-8);assert.equal(volumeWeight(v,at(.5-width)),1);assert.equal(volumeWeight(v,at(.5+width)),0);}
assert.equal(volumeWeight({...centered,featherBias:0},at(.5)),0);assert.equal(volumeWeight({...centered,featherBias:1},at(.5)),1);assert(Math.abs(volumeWeight({...centered,featherBias:1},at(.55))-.5)<1e-8);
for(const bias of [-.1,1.1,NaN])assert.throws(()=>validateVolume({...centered,featherBias:bias}));assert.deepEqual(validateRegions({'surface:test':{enabled:true,volume:centered}})['surface:test'].volume,centered);
const rotated={...vol,shape:'box',size:[2,.2,.2],rotation:[0,0,Math.PI/2],feather:0};assert.equal(volumeWeight(rotated,at(0,.8)),1);assert.equal(volumeWeight(rotated,at(.8,0)),0);rotated.rotation=[0,0,0];assert.equal(volumeWeight(rotated,at(.8,0)),1,'Rotation cache stale after gizmo drag');
const mirrored={...vol,position:[-.2,0,0],size:[.1,.1,.1],symmetry:true,centerX:-.01};assert.equal(volumeWeight(mirrored,at(.18)),1);assert.equal(volumeWeight({...mirrored,symmetry:false},at(.18)),0);
const prism={...vol,shape:'polygon',points:[[0,0],[1,0],[1,.4],[.4,.4],[.4,1],[0,1]],feather:0};assert.equal(volumeWeight(prism,at(-.3,.3)),1);assert.equal(volumeWeight(prism,at(.3,.3)),0,'Concavity filled');assert.equal(volumeWeight(prism,at(-.3,.3,.6)),0,'Extrusion cap ignored');
for(const bad of [{...vol,size:[0,1,1]},{...vol,rotation:[NaN,0,0]},{...vol,feather:-.1},{...prism,points:[[0,0],[1,1],[0,1],[1,0]]}])assert.throws(()=>validateVolume(bad));
const muzzleBox=new T.Box3();for(const e of surface.influence('muzzleLength'))for(let i=0;i<e.native.length;i++)if(e.native[i]>.0001)muzzleBox.expandByPoint(e.regionPoints[i]);
const muzzleVolume={...vol,shape:'box',position:muzzleBox.getCenter(new T.Vector3()).toArray(),size:muzzleBox.getSize(new T.Vector3()).multiplyScalar(.8).toArray(),feather:.15,symmetry:true,centerX:report.body_reconstruction.head_rigid_translation[0]},volumeRegion={enabled:true,volume:muzzleVolume};
assert.deepEqual(validateRegions({'surface:muzzleLength':volumeRegion})['surface:muzzleLength'],volumeRegion);
surface.apply({'surface:muzzleLength':volumeRegion});let partialVolume=0,missingUVSelected=0;
for(const e of surface.influence('muzzleLength',{'surface:muzzleLength':volumeRegion})){const i=e.mesh.morphTargetDictionary.muzzleLength,mi=meshes.indexOf(e.mesh),out=e.mesh.geometry.morphAttributes.position[i].array;for(let j=0;j<e.weights.length;j++){const w=volumeWeight(muzzleVolume,e.regionPoints[j]);assert(Math.abs(e.weights[j]-w)<1e-6);if(w>0&&w<1)partialVolume++;if(w>0&&e.native[j]>0&&!e.atlas.vertices[j])missingUVSelected++;for(let k=0;k<3;k++)assert(Math.abs(out[j*3+k]-original[mi][i][j*3+k]*w)<1e-8);}}
assert(partialVolume>0,'Missing smooth feather');assert(missingUVSelected>100,'Volume still depends on UVs');const volumeSnapshot=snapshot();surface.apply(JSON.parse(JSON.stringify({'surface:muzzleLength':volumeRegion})));assert.equal(error(volumeSnapshot,snapshot()),0);surface.apply({});assert.equal(error(original,snapshot()),0);

// A new volume replaces the old support, including surfaces whose native delta is zero.
const asymmetric=[[.1,.1],[.9,.2],[.75,.85],[.3,.9]],symmetric=symmetricPolygon(asymmetric);assert(symmetric.every(([x,y])=>symmetric.some(([u,v])=>Math.abs(x+u-1)<1e-6&&Math.abs(y-v)<1e-6)));assert.deepEqual(symmetricPolygon(symmetric),symmetric,'Symmetry changed point order while dragging');
const single=centerVolume({...vol,version:2,shape:'polygon',points:asymmetric,symmetry:true,position:[.2,0,0],rotation:[.2,.3,.4]});assert.equal(single.position[0],0);assert.deepEqual(single.rotation,[.2,0,0]);for(const x of [.01,.1,.2,.3])assert(Math.abs(volumeWeight(single,at(x,.1,.1))-volumeWeight(single,at(-x,.1,.1)))<1e-10);
const linear=displacementField([-1,1].flatMap(x=>[-1,1].flatMap(y=>[-1,1].map(z=>({p:[x,y,z],d:[x*.01,0,.02]})))));assert(Math.abs(linear([.2,.3,.4])[0]-.002)<1e-7);assert(Math.abs(linear([.2,.3,.4])[2]-.02)<1e-7);
const expandedVolume={...vol,version:2,shape:'box',position:[0,.96,.07],size:[.16,.12,.16],feather:.2},expandedRegion={enabled:true,volume:expandedVolume};
surface.apply({'surface:muzzleLength':expandedRegion});let newlyMoved=0,allZeroMeshMoved=0;for(const e of surface.influence('muzzleLength')){const i=e.mesh.morphTargetDictionary.muzzleLength,d=e.mesh.geometry.morphAttributes.position[i],mi=meshes.indexOf(e.mesh),native=original[mi][i],wasZero=native.every(n=>n===0);for(let j=0;j<d.count;j++){const moved=Math.hypot(d.getX(j),d.getY(j),d.getZ(j));if(!e.native[j]&&moved>1e-6){newlyMoved++;if(wasZero)allZeroMeshMoved++;}if(volumeWeight(expandedVolume,e.regionPoints[j])===0)assert(moved<1e-9,'New volume leaked outside its boundary');}assert(d.array.every(Number.isFinite));assert(e.mesh.geometry.morphAttributes.normal[i].array.every(Number.isFinite));}
assert(newlyMoved>100,'Cannot select surface outside old support');assert(allZeroMeshMoved>0,'All-zero original components still excluded');const extendedSnapshot=snapshot();surface.apply({'surface:muzzleLength':expandedRegion});assert.equal(error(extendedSnapshot,snapshot()),0,'Extended field accumulated');surface.apply({});assert.equal(error(original,snapshot()),0,'Extended field reset drift');
// Canonical extension follows the same nonlinear body-placement endpoint map.
placement(.8,.02);surface.capture();surface.apply({'surface:muzzleLength':expandedRegion});assert(snapshot().every(a=>a.every(b=>b.every(Number.isFinite))));surface.restore();placement(1,0);surface.capture();surface.apply({});assert(error(original,snapshot())<1e-7);

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
for(const e of field.entries){
 assert(e.weights.every(v=>Math.abs(v-.6)<1e-6));const g=e.mesh.geometry,idx=g.index,faceIndex=e.atlas.valid.findIndex(v=>v),c=faceIndex*3;
 if(faceIndex<0){assert.equal(pickUV(field,{object:e.mesh,faceIndex:0}),null);continue;}
 const ids=[0,1,2].map(i=>idx?idx.getX(c+i):c+i),vertices=ids.map(i=>e.mesh.getVertexPosition(i,new T.Vector3()).applyMatrix4(e.mesh.matrixWorld));if(new T.Triangle(...vertices).getArea()<1e-12)continue;
 const point=vertices[0].clone().add(vertices[1]).add(vertices[2]).multiplyScalar(1/3),hit={object:e.mesh,faceIndex,face:{a:ids[0],b:ids[1],c:ids[2]},point},picked=pickUV(field,hit);assert(picked);assert(Math.abs(picked.u-(e.atlas.uv[c*2]+e.atlas.uv[c*2+2]+e.atlas.uv[c*2+4])/3)<1e-5);
}
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
regions['surface:muzzleLength']=expandedRegion;surface.apply(regions);const before=snapshot();
for(const m of meshes)for(const a of m.geometry.morphAttributes.normal||[])assert(a.array.every(Number.isFinite));
globalThis.FileReader=class{readAsArrayBuffer(blob){blob.arrayBuffer().then(x=>{this.result=x;this.onloadend?.();});}readAsDataURL(blob){blob.arrayBuffer().then(x=>{this.result='data:application/octet-stream;base64,'+Buffer.from(x).toString('base64');this.onloadend?.();});}};
const exported=await new GLTFExporter().parseAsync(scene,{binary:true,trs:true,onlyVisible:false}),loaded=await new GLTFLoader().parseAsync(exported,'');
const reimport=new Map();loaded.scene.traverse(o=>{if(o.isMesh)reimport.set(o.name,o);});let exportError=0;for(const m of meshes){const n=reimport.get(m.name);assert(n,'Missing exported mesh '+m.name);assert.deepEqual(n.geometry.attributes.uv?.array,m.geometry.attributes.uv?.array,'Native UVs changed on export');for(const [name,i]of Object.entries(m.morphTargetDictionary||{})){const j=n.morphTargetDictionary[name];assert(j!==undefined);const a=m.geometry.morphAttributes.position[i].array,b=n.geometry.morphAttributes.position[j].array;for(let k=0;k<a.length;k++)exportError=Math.max(exportError,Math.abs(a[k]-b[k]));}}
assert(exportError<1e-7);assert.equal(error(before,snapshot()),0);surface.apply({});assert(error(original,snapshot())<1e-7);
console.log(JSON.stringify({status:'passed',shapeControls:changed,newlyMoved,allZeroMeshMoved,sharedVertexGroups:joins.length,maxSeam,baselineSeam,exportError,checks:['first-open and cached unedited fields for every shape','full and disabled exact equivalence','partial region changes geometry','repeat and reset','combined mouth/profile interfaces','placement extremes','finite normals and zero-influence shading','extended volume actual GLB export/reimport','native UV equality, auto-framing, symmetry and original-UV picking','fractional influence across UV seams','sphere, rotated box, concave prism, feather, symmetry, no-UV muzzle coverage and volume export']},null,2));
