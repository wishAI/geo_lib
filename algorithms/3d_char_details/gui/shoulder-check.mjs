import fs from 'node:fs';
import assert from 'node:assert/strict';
import * as T from './three.mjs';
import {GLTFLoader} from './loader.mjs';
import {bodyPlacement} from './placement.mjs';
import {editingPose} from './editing.mjs';
import {installBodyTransition} from './transition.mjs';
import {motionPlayer} from './motion.mjs';
import {garmentFit,fitDefaults} from './fit.mjs';

globalThis.ProgressEvent=class { constructor(type,args){Object.assign(this,args);} };
const bytes=fs.readFileSync(process.argv[2]),length=bytes.readUInt32LE(12),json=JSON.parse(bytes.subarray(20,20+length));
json.buffers[0].uri='data:application/octet-stream;base64,'+bytes.subarray(28+length).toString('base64');
delete json.images;delete json.textures;delete json.samplers;
const {scene,animations}=await new GLTFLoader().parseAsync(JSON.stringify(json),'');
const report=JSON.parse(fs.readFileSync(process.argv[3])),settings=JSON.parse(fs.readFileSync(process.argv[4]));
settings.garmentFit=fitDefaults(settings.garmentFit);
const parts=new Map(),bones=new Map();
scene.traverse(o=>{if(o.isBone)bones.set(o.name,{bone:o,quaternion:o.quaternion.clone(),scale:o.scale.clone()});if(o.userData.part_type)parts.set(o.name,{object:o});});
installBodyTransition(scene);
let baseline,player,mode='a';
const applyPose=()=>{for(const [n,b]of bones){const cfg=settings.bones?.[n]||{rotation:[0,0,0],scale:1},q=new T.Quaternion().setFromEuler(new T.Euler(...cfg.rotation.map(x=>x*Math.PI/180),'XYZ'));b.bone.quaternion.copy(player?.quaternion(n)||baseline?.quaternion(n,mode)||b.quaternion).multiply(q);b.bone.scale.copy(b.scale).multiplyScalar(cfg.scale);}};
const place=bodyPlacement(scene,bones),fit=garmentFit(scene,parts,report);
player=motionPlayer(scene,animations,bones,applyPose);
player.beforeRestEdit();place(settings.bodyFrame?.scale??1,settings.bodyFrame?.offset??0,settings.bodyFrame?.shoulderHeight??0);fit.capture();
baseline=editingPose(scene,bones);player.afterRestEdit(settings.bodyFrame?.scale??1);
const bodyMeshes=[];parts.get('Body_Complete').object.traverse(o=>{if(o.isMesh)bodyMeshes.push(o);});
scene.traverse(o=>{if(!o.isMesh)return;let p=o;while(p&&!p.userData.part_type)p=p.parent;if(p?.userData.part_type==='clothing')return;for(const [n,k]of Object.entries(o.morphTargetDictionary||{}))if(n in (settings.morphs||{}))o.morphTargetInfluences[k]=settings.morphs[n];});
const refresh=()=>{scene.updateMatrixWorld(true);scene.traverse(o=>{if(o.skeleton)o.skeleton.update();});};
const restPose=()=>{mode='a';player.reset();applyPose();refresh();};
const poseT=()=>{mode='t';player.reset();applyPose();refresh();};
const snapshot=()=>[...fit.garments.values()].flatMap(p=>p.records.map(r=>({r,position:Array.from(r.g.attributes.position.array),index:Array.from(r.g.attributes.skinIndex.array),weight:Array.from(r.g.attributes.skinWeight.array)})));
const same=(a,b,label)=>{assert.equal(a.length,b.length,label);for(let i=0;i<a.length;i++)assert(Math.abs(a[i]-b[i])<1e-7,label+' at '+i);};
const protectedGeometry=[];
scene.traverse(o=>{if(!o.isMesh)return;let p=o;while(p&&!p.userData.part_type)p=p.parent;if(p?.userData.part_type==='clothing')return;const attributes=Object.fromEntries(Object.entries(o.geometry.attributes).map(([name,a])=>[name,Array.from(a.array)])),morphs=Object.fromEntries(Object.entries(o.geometry.morphAttributes).map(([name,arrays])=>[name,arrays.map(a=>Array.from(a.array))]));protectedGeometry.push({o,attributes,morphs});});
settings.garmentFit.shoulderFollow=false;restPose();fit.apply(settings);refresh();
const original=snapshot(),joint=bones.get('arm_stretch_l').bone.getWorldPosition(new T.Vector3()),nodes=[];
for(const name of ['Vest','Sleeve_L','Sleeve_R'])for(const node of fit.garments.get(name).nodes){
 const [r,i]=node.refs[0],p=r.o.getVertexPosition(i,new T.Vector3()).applyMatrix4(r.o.matrixWorld);
 if(Math.abs(p.x)>.03&&Math.abs(p.x)<.16&&p.y>joint.y-.065&&p.y<joint.y+.075)nodes.push({name,node,r,i,rest:p});
}
assert(nodes.length>20,'Shoulder measurement region is empty');
const vertex=c=>c.r.o.getVertexPosition(c.i,new T.Vector3()).applyMatrix4(c.r.o.matrixWorld);
const bodyCoordinates=p=>({x:Math.abs((p.x+.004865-(settings.bodyFrame?.offset||0))/(settings.bodyFrame?.scale??1)),y:.806+(p.y-.806)/(settings.bodyFrame?.scale??1)});
function axillaEdges(){
 const vest=fit.garments.get('Vest'),posed=new Map(vest.nodes.map(node=>{const [r,i]=node.refs[0];return [node,vertex({r,i})];})),stretch=[];
 for(const node of vest.nodes){const {x,y}=bodyCoordinates(node.current);if(x<=.04||x>=.18||y<=.60||y>=.745)continue;
  for(const other of node.edges){if(other.index<=node.index)continue;const length=node.current.distanceTo(other.current);if(length<.0001)continue;stretch.push(posed.get(node).distanceTo(posed.get(other))/length);}
 }
 stretch.sort((a,b)=>a-b);
 return {edges:stretch.length,maximum:stretch.at(-1)||0,p99:stretch[Math.floor(stretch.length*.99)]||0,overTwo:stretch.filter(x=>x>2).length};
}

// A small triangle BVH makes exact closest-surface checks affordable for 65 poses.
function surface(){
 const triangles=[];
 for(const o of bodyMeshes){
  const g=o.geometry,index=g.index,vertices=Array.from({length:g.attributes.position.count},(_,i)=>o.getVertexPosition(i,new T.Vector3()).applyMatrix4(o.matrixWorld));
  for(let i=0;i<(index?.count??vertices.length);i+=3){
   const abc=[0,1,2].map(k=>vertices[index?index.getX(i+k):i+k]),triangle=new T.Triangle(...abc);
   if(triangle.getArea()<1e-14)continue;
   const box=new T.Box3().setFromPoints(abc),center=box.getCenter(new T.Vector3());triangles.push({triangle,box,center});
  }
 }
 const build=items=>{
  const box=new T.Box3();for(const item of items)box.union(item.box);
  if(items.length<=12)return {box,items};
  const size=box.getSize(new T.Vector3()),axis=size.x>size.y?(size.x>size.z?'x':'z'):(size.y>size.z?'y':'z');
  items.sort((a,b)=>a.center[axis]-b.center[axis]);const mid=items.length>>1;
  return {box,left:build(items.slice(0,mid)),right:build(items.slice(mid))};
 };
 const root=build(triangles);
 return p=>{
  let best=Infinity,signed=0;const point=new T.Vector3(),normal=new T.Vector3();
  const visit=node=>{
   if(node.box.distanceToPoint(p)>best)return;
   if(node.items){for(const {triangle}of node.items){triangle.closestPointToPoint(p,point);const distance=point.distanceTo(p);if(distance<best){best=distance;signed=p.clone().sub(point).dot(triangle.getNormal(normal));}}return;}
   const a=node.left.box.distanceToPoint(p),b=node.right.box.distanceToPoint(p);
   if(a<b){visit(node.left);visit(node.right);}else{visit(node.right);visit(node.left);}
  };visit(root);return signed;
 };
}
const summarize=(distances,selected=nodes)=>({samples:distances.length,inside:distances.filter(x=>x<-.001).length,newInside:distances.filter((x,i)=>x<-.001&&selected[i].restDistance>=0).length,minimum:Math.min(...distances)});
let nearest=surface();for(const c of nodes)c.restDistance=nearest(c.rest);
const rest=summarize(nodes.map(c=>c.restDistance));
poseT();nearest=surface();const before=summarize(nodes.map(c=>nearest(vertex(c)))),axillaBefore=axillaEdges();
restPose();settings.garmentFit.shoulderFollow=true;fit.apply(settings);refresh();
const corrected=snapshot();
let validSkinVertices=0;
for(const {r}of corrected){const weights=r.g.attributes.skinWeight,index=r.g.attributes.skinIndex;assert.equal(weights.itemSize,4);assert.equal(index.itemSize,4);for(let i=0;i<weights.count;i++){let total=0;for(let k=0;k<4;k++){const w=weights.getComponent(i,k),j=index.getComponent(i,k);assert(Number.isFinite(w)&&w>=0,'Invalid skin weight');assert(Number.isInteger(j)&&j>=0&&j<r.o.skeleton.bones.length,'Invalid skin joint');total+=w;}assert(Math.abs(total-1)<1e-5,'Unnormalized skin weights');validSkinVertices++;}}
for(let i=0;i<original.length;i++)same(original[i].position,corrected[i].position,'Correction changed manual rest fit');
for(const {o,attributes,morphs}of protectedGeometry){for(const [name,data]of Object.entries(attributes))same(data,Array.from(o.geometry.attributes[name].array),'Correction changed protected body/face '+name);for(const [name,arrays]of Object.entries(morphs))for(let k=0;k<arrays.length;k++)same(arrays[k],Array.from(o.geometry.morphAttributes[name][k].array),'Correction changed protected body/face morph');}
let unchangedLowerVest=0;
for(let n=0;n<original.length;n++)if(original[n].r.part.name==='Vest')for(let i=0;i<original[n].r.g.attributes.position.count;i++){
 const p=new T.Vector3().fromBufferAttribute(original[n].r.g.attributes.position,i).applyMatrix4(original[n].r.world);
 if(bodyCoordinates(p).y>=.69999)continue;
 for(let k=0;k<4;k++){assert.equal(corrected[n].index[i*4+k],original[n].index[i*4+k],'Transferred weights below shoulder cap');assert.equal(corrected[n].weight[i*4+k],original[n].weight[i*4+k],'Changed lower vest weights');}unchangedLowerVest++;
}
let untouched=0;
for(let n=0;n<original.length;n++){
 const {r}=original[n];
 for(let i=0;i<r.g.attributes.position.count;i++){
  const p=new T.Vector3().fromBufferAttribute(r.g.attributes.position,i).applyMatrix4(r.world);
  if(p.y>.59&&Math.abs(p.x)<.22)continue;
  for(let k=0;k<4;k++){assert.equal(corrected[n].index[i*4+k],original[n].index[i*4+k],'Changed distant joint');assert.equal(corrected[n].weight[i*4+k],original[n].weight[i*4+k],'Changed distant weight');}untouched++;
 }
}
fit.apply(settings);const repeated=snapshot();
for(let i=0;i<corrected.length;i++){same(corrected[i].weight,repeated[i].weight,'Correction accumulated');same(corrected[i].position,repeated[i].position,'Fit accumulated');}
poseT();nearest=surface();const after=summarize(nodes.map(c=>nearest(vertex(c)))),axillaAfter=axillaEdges();
console.log('A-to-T shoulder measurements',JSON.stringify({rest,before,after,axillaBefore,axillaAfter}));
// Guard the demonstrated smooth-fit case without assuming arbitrary presets
// have a well-behaved axilla before this correction.
if(axillaBefore.edges>50&&axillaBefore.maximum<3&&axillaBefore.p99<2){assert(axillaAfter.maximum<=axillaBefore.maximum*1.15,'Shoulder correction introduced an axilla edge spike');assert(axillaAfter.p99<=axillaBefore.p99*1.15,'Shoulder correction corrugated the axilla');}
assert(after.newInside<=before.newInside,'Correction introduced additional A-clear/T-penetrating points');
if(before.newInside>=20)assert(after.newInside<=Math.ceil(before.newInside*.25),'Shoulder correction did not remove most pose-induced penetration');
let maxSeamGap=0;
function seams(){for(const seam of fit.seams)for(const [p,c]of seam.pairs){const [r,i]=p.refs[0],a=r.o.getVertexPosition(i,new T.Vector3()).applyMatrix4(r.o.matrixWorld);for(const [q,j]of c.refs)maxSeamGap=Math.max(maxSeamGap,a.distanceTo(q.o.getVertexPosition(j,new T.Vector3()).applyMatrix4(q.o.matrixWorld)));}}
seams();
const running=[],sampleNodes=nodes.filter((_,i)=>i%4===0);
const restoreWeights=records=>{for(const {r,index,weight}of records){r.g.attributes.skinIndex.array.set(index);r.g.attributes.skinWeight.array.set(weight);}};
for(let i=0;i<=64;i++){
 player.seek(player.duration*i/64);refresh();nearest=surface();
 restoreWeights(original);const before=summarize(sampleNodes.map(c=>nearest(vertex(c))),sampleNodes);seams();
 restoreWeights(corrected);const after=summarize(sampleNodes.map(c=>nearest(vertex(c))),sampleNodes);seams();
 running.push({time:player.duration*i/64,before,after});
}
assert(maxSeamGap<3e-6,'Shoulder correction opened a garment seam');
restPose();settings.garmentFit.shoulderFollow=false;fit.apply(settings);const disabled=snapshot();
for(let i=0;i<original.length;i++){same(original[i].position,disabled[i].position,'Reset changed fit');same(original[i].index,disabled[i].index,'Reset changed skin indices');same(original[i].weight,disabled[i].weight,'Reset changed weights');}
const result={preset:process.argv[4],rest,before,after,axillaBefore,axillaAfter,validSkinVertices,unchangedLowerVest,protectedBodyAndFaceMeshes:protectedGeometry.length,unchangedDistantVertices:untouched,maxSeamGap,running,
 limitation:'Signed nearest-surface samples diagnose shoulder penetration; they do not certify whole-outfit collision freedom.'};
if(process.argv[5])fs.writeFileSync(process.argv[5],JSON.stringify(result,null,2)+'\n');
console.log(JSON.stringify({...result,running:{samples:running.length,verticesPerSample:sampleNodes.length,...Object.fromEntries(['before','after'].map(key=>[key,{worstMinimum:Math.min(...running.map(r=>r[key].minimum)),maxNewInside:Math.max(...running.map(r=>r[key].newInside)),totalNewInside:running.reduce((s,r)=>s+r[key].newInside,0)}]))}},null,2));
player.dispose();
