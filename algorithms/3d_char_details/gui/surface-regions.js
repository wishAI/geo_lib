import * as T from 'three';
import {regionWeight,editedWeight} from '/api/artifact?path=algorithms/3d_char_details/gui/uv-regions.js';

// One immutable rest-space map per authored shape, shared across material
// splits and attached facial components. This is a projection, not a UV unwrap.
export function surfaceRegions(model,names){
 let atlasHash;const records=[],fields=new Map(),maps=new Map(),weightCache=new Map();model.updateMatrixWorld(true);
 model.traverse(mesh=>{
  if(!mesh.isMesh)return;let owner=mesh;while(owner&&!owner.userData.part_type)owner=owner.parent;
  if(owner?.userData.part_type==='clothing')return;
  const g=mesh.geometry,world=mesh.matrixWorld.clone(),rest=[];
  for(let i=0;i<g.attributes.position.count;i++)rest.push(new T.Vector3().fromBufferAttribute(g.attributes.position,i).applyMatrix4(world));
  const record={mesh,rest,keys:rest.map(p=>p.toArray().map(v=>Math.round(v*1e6)).join(',')),original:new Map(),masked:new Set()};
  for(const [name,index]of Object.entries(mesh.morphTargetDictionary||{})){
   if(!names.has(name)||!g.morphAttributes.position?.[index])continue;
   const delta=g.morphAttributes.position[index];let active=false;
   for(let i=0;i<delta.count;i++)if(Math.abs(delta.getX(i))+Math.abs(delta.getY(i))+Math.abs(delta.getZ(i))>1e-9){active=true;break;}
   if(!active)continue;
   if(!fields.has(name))fields.set(name,{box:new T.Box3(),records:[]});const field=fields.get(name);field.records.push([record,index]);
   for(let i=0;i<delta.count;i++)if(Math.abs(delta.getX(i))+Math.abs(delta.getY(i))+Math.abs(delta.getZ(i))>1e-9)field.box.expandByPoint(rest[i]);
   record.original.set(index,{position:delta.clone(),normal:g.morphAttributes.normal?.[index]?.clone()});
  }if(record.original.size)records.push(record);
 });
 for(const field of fields.values()){
  const size=field.box.getSize(new T.Vector3());field.box.expandByVector(size.multiplyScalar(.12).max(new T.Vector3(.003,.003,.003)));
 }
 function coordinates(name,p,projection='front'){
  const b=fields.get(name).box,axis=projection==='side'?'z':'x';return [(p[axis]-b.min[axis])/(b.max[axis]-b.min[axis]),(p.y-b.min.y)/(b.max.y-b.min.y)];
 }
 // WebGL morph textures are immutable after upload. Rebuild the geometry shell
 // while retaining untouched attributes, instead of duplicating every morph.
 function refresh(record){const old=record.mesh.geometry,g=new T.BufferGeometry();g.name=old.name;g.index=old.index;g.attributes={...old.attributes};g.morphAttributes=Object.fromEntries(Object.entries(old.morphAttributes).map(([k,v])=>[k,[...v]]));g.morphTargetsRelative=old.morphTargetsRelative;g.groups=old.groups.map(v=>({...v}));g.drawRange={...old.drawRange};g.userData={...old.userData};g.boundingBox=old.boundingBox?.clone()||null;g.boundingSphere=old.boundingSphere?.clone()||null;record.mesh.geometry=g;old.dispose();}
 function restore(){for(const r of records){const g=r.mesh.geometry;for(const i of r.masked){const original=r.original.get(i);g.morphAttributes.position[i]=original.position.clone();if(original.normal)g.morphAttributes.normal[i]=original.normal.clone();}if(r.masked.size)refresh(r);r.masked.clear();}}
 function capture(){for(const r of records)for(const [i]of r.original){const g=r.mesh.geometry;r.original.set(i,{position:g.morphAttributes.position[i].clone(),normal:g.morphAttributes.normal?.[i]?.clone()});}}
 function checkRegion(region){if(region?.edits?.length&&region.atlasHash!==atlasHash)throw new Error("Influence edits use a different UV atlas. Restore a matching atlas or clear those edits.");}
 function weightField(name,region){
  checkRegion(region);
  const field=fields.get(name),signature=JSON.stringify(region),old=weightCache.get(name);if(old&&old.signature===signature)return old.weights;
  if(!field.nodes){field.nodes=new Map();for(const [r]of field.records)for(let i=0;i<r.rest.length;i++){const key=r.keys[i];let n=field.nodes.get(key);if(!n){n={p:r.rest[i],samples:[]};field.nodes.set(key,n);}if(r.samples)n.samples.push(...r.samples[i]);}}
  const weights=new Map();for(const [key,n]of field.nodes){const [u,v]=coordinates(name,n.p,region?.projection);weights.set(key,editedWeight(region,n.samples,regionWeight(region,u,v)));}weightCache.set(name,{signature,weights});return weights;
 }
 function setAtlas(descriptors){
  atlasHash=descriptors.values().next().value?.atlasHash;
  for(const r of records){r.atlas=descriptors.get(r.mesh);r.samples=Array.from({length:r.rest.length},()=>[]);if(!r.atlas)continue;const idx=r.mesh.geometry.index;for(let c=0;c<r.atlas.uv.length/2;c++){const i=idx?idx.getX(c):c;r.samples[i].push({sheet:r.atlas.sheet,u:r.atlas.uv[c*2],v:r.atlas.uv[c*2+1]});}}
  weightCache.clear();for(const field of fields.values())field.nodes=null;
 }
 function influence(name,regions={}){
  const field=fields.get(name);if(!field)return [];const weights=weightField(name,regions['surface:'+name]);
  return field.records.map(([r,index])=>{const d=r.original.get(index).position,native=new Float32Array(d.count),mask=new Float32Array(d.count);for(let i=0;i<d.count;i++){native[i]=Math.hypot(d.getX(i),d.getY(i),d.getZ(i));mask[i]=weights.get(r.keys[i]);}return {mesh:r.mesh,atlas:r.atlas,native,weights:mask};});
 }
 function apply(regions={}){
  restore();
  for(const [name,field]of fields){const region=regions['surface:'+name];if(!region?.enabled)continue;
   const shared=weightField(name,region);
   for(const [r,index]of field.records){const g=r.mesh.geometry,base=g.attributes.position,original=r.original.get(index),target=original.position.clone(),weights=new Float32Array(base.count);let changed=false;
    for(let i=0;i<base.count;i++){
     const w=shared.get(r.keys[i]);weights[i]=w;
     const x=original.position.getX(i),y=original.position.getY(i),z=original.position.getZ(i);
     if(g.morphTargetsRelative){target.setXYZ(i,x*w,y*w,z*w);if(w!==1&&(x||y||z))changed=true;}
     else{target.setXYZ(i,base.getX(i)+(x-base.getX(i))*w,base.getY(i)+(y-base.getY(i))*w,base.getZ(i)+(z-base.getZ(i))*w);if(w!==1)changed=true;}
    }
    if(!changed)continue;
    g.morphAttributes.position[index]=target;
    if(original.normal&&g.attributes.normal)g.morphAttributes.normal[index]=maskedNormals(g,original.position,target,original.normal,weights);
    r.masked.add(index);
   }
  }
  for(const r of records)if(r.masked.size){refresh(r);r.mesh.geometry.computeBoundingBox();r.mesh.geometry.computeBoundingSphere();}
  model.userData.surfaceRegions=JSON.parse(JSON.stringify(Object.fromEntries(Object.entries(regions).filter(([k])=>k.startsWith('surface:')))));
 }
 return {has:name=>fields.has(name),restore,capture,apply,setAtlas,influence,checkRegion,
  map(name,projection='front'){
   const field=fields.get(name);if(!field)return null;const key=name+':'+projection;if(maps.has(key))return maps.get(key);
   const result={label:projection==='side'?'Side projection · depth / height':'Front projection · width / height',triangles:field.records.map(([r])=>({index:r.mesh.geometry.index,points:r.rest.map(p=>coordinates(name,p,projection))})),note:'Shared rest-space projection across all affected surfaces. Overlapping front/back surfaces share coordinates; influence stays within the authored shape.'};maps.set(key,result);return result;
  },pick(name,hit,projection='front'){
   const field=fields.get(name),record=field?.records.find(([r])=>r.mesh===hit.object)?.[0];if(!record||!hit.face)return null;
   const ids=[hit.face.a,hit.face.b,hit.face.c],vertices=ids.map(i=>hit.object.getVertexPosition(i,new T.Vector3()).applyMatrix4(hit.object.matrixWorld));
   const bary=new T.Triangle(...vertices).getBarycoord(hit.point,new T.Vector3());if(!bary)return null;
   const p=new T.Vector3();ids.forEach((id,i)=>p.addScaledVector(record.rest[id],bary.getComponent(i)));const [x,y]=coordinates(name,p,projection);return {x,y};
  }};
}

function maskedNormals(g,before,after,authored,weights){
 const base=g.attributes.position,normal=g.attributes.normal,relative=g.morphTargetsRelative,count=base.count;
 function geometric(target){
  const accum=new Float32Array(count*3),p=new Float32Array(count*3);for(let i=0;i<count;i++)for(let c=0;c<3;c++)p[i*3+c]=target.getComponent(i,c)+(relative?base.getComponent(i,c):0);
  const idx=g.index;for(let i=0;i<(idx?.count??count);i+=3){const a=(idx?idx.getX(i):i)*3,b=(idx?idx.getX(i+1):i+1)*3,c=(idx?idx.getX(i+2):i+2)*3,ux=p[b]-p[a],uy=p[b+1]-p[a+1],uz=p[b+2]-p[a+2],vx=p[c]-p[a],vy=p[c+1]-p[a+1],vz=p[c+2]-p[a+2],n=[uy*vz-uz*vy,uz*vx-ux*vz,ux*vy-uy*vx];for(const j of [a,b,c])for(let k=0;k<3;k++)accum[j+k]+=n[k];}return accum;
 }
 const zero=relative?new T.BufferAttribute(new Float32Array(count*3),3):base;
 const rest=geometric(zero),a=geometric(before),b=geometric(after),result=authored.clone(),q=new T.Quaternion(),na=new T.Vector3(),nb=new T.Vector3(),nr=new T.Vector3(),baseNormal=new T.Vector3(),n=new T.Vector3();
 for(let i=0;i<count;i++){
  // Blend authored split normals, then account for the polygon's spatial
  // gradient. Zero influence with unchanged neighbouring faces is exact base.
  const w=weights[i];baseNormal.fromBufferAttribute(normal,i);
  nr.fromArray(rest,i*3);na.fromArray(a,i*3);nb.fromArray(b,i*3);
  if(w===0&&nr.equals(nb)){result.setXYZ(i,relative?0:baseNormal.x,relative?0:baseNormal.y,relative?0:baseNormal.z);continue;}
  n.fromBufferAttribute(authored,i);if(relative)n.add(baseNormal);
  n.lerp(baseNormal,1-w).normalize();
  nr.normalize();na.normalize().lerp(nr,1-w).normalize();
  if(na.lengthSq()>1e-20&&nb.lengthSq()>1e-20)n.applyQuaternion(q.setFromUnitVectors(na,nb.normalize()));
  n.normalize();if(relative)n.sub(baseNormal);result.setXYZ(i,n.x,n.y,n.z);
 }return result;
}
