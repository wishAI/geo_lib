import * as T from 'three';

export function decodeAtlas(data,hash,model){
 if(data.version!==1||data.sourceHash!==hash)throw new Error('The editing atlas belongs to another model revision. Regenerate it for this character.');
 const descriptors=new Map();model.traverse(mesh=>{if(!mesh.isMesh)return;const p=data.primitives[mesh.userData.editAtlasId];if(!p)throw new Error('Editing atlas is missing a surface');const g=mesh.geometry,count=g.index?.count??g.attributes.position.count;if(p.vertices!==g.attributes.position.count||p.corners!==count)throw new Error('Editing atlas topology mismatch');
  const binary=atob(p.uv32),view=new DataView(Uint8Array.from(binary,c=>c.charCodeAt(0)).buffer),uv=new Float32Array(count*2);if(view.byteLength!==uv.length*4)throw new Error('Invalid editing atlas');for(let i=0;i<uv.length;i++)uv[i]=view.getFloat32(i*4,true);
  let owner=mesh;while(owner.parent&&!owner.userData.part_type)owner=owner.parent;
  const label=owner.name.replaceAll('_',' ');descriptors.set(mesh,{uv,atlasHash:data.atlasHash,sheet:p.sheet,label,kind:'Previous editing atlas',original:originalAtlas(mesh,'uv:'+p.sheet,label)});
 });return descriptors;
}
// Read TEXCOORD_0 verbatim. Degenerate triangles are unavailable, never unwrapped
// or projected silently. The display and the deformation use this same descriptor.
export function originalAtlas(mesh,sheet,label){
 const g=mesh.geometry,uv=g.attributes.uv,idx=g.index,count=idx?.count??g.attributes.position.count,out=new Float32Array(count*2),valid=new Uint8Array(count/3),vertices=new Uint8Array(g.attributes.position.count);
 for(let c=0;c<count;c++){const i=idx?idx.getX(c):c;out[c*2]=uv?.getX(i)||0;out[c*2+1]=uv?.getY(i)||0;}
 // Thousands of repeated unit triangles are placeholder coordinates, not
 // editable UV islands. Keep ordinary overlapping/mirrored charts intact.
 const keys=[],counts=new Map();
 for(let c=0;c<count;c+=3){const k=c*2,[x,y,X,Y,u,v]=out.subarray(k,k+6),area=Math.abs((X-x)*(v-y)-(Y-y)*(u-x));if(!Number.isFinite(area)||area<=1e-14)continue;valid[c/3]=1;if(area>.02){const key=[[x,y],[X,Y],[u,v]].map(p=>p.join(',')).sort().join(';');keys[c/3]=key;counts.set(key,(counts.get(key)||0)+1);}}
 for(let t=0;t<valid.length;t++){if(keys[t]&&counts.get(keys[t])>16)valid[t]=0;if(valid[t])for(let j=0;j<3;j++)vertices[idx?idx.getX(t*3+j):t*3+j]=1;}
 return {uv:out,valid,vertices,sheet,label,kind:'Original model UVs'};
}
export function garmentAtlas(entry){return originalAtlas(entry.mesh,entry.sheet,entry.sheet.replaceAll('_',' '));}
export function heatColor(w){w=Math.max(0,Math.min(1,w));const stops=[[.10,.24,.85],[0,.77,.92],[.18,.83,.40],[.98,.86,.14],[.95,.20,.14]],t=w*4,i=Math.min(3,Math.floor(t)),f=t-i;return stops[i].map((v,k)=>v+(stops[i+1][k]-v)*f);}
const srgb=x=>x<=.0031308?x*12.92:1.055*Math.pow(x,1/2.4)-.055;
export function prepareField(entries,materialInfo){
 let peak=0;for(const e of entries)for(const v of e.native)peak=Math.max(peak,v);
 for(const e of entries){e.atlas??=garmentAtlas(e);e.colors=new Float32Array(e.native.length*3);const g=e.mesh.geometry,material=Array.isArray(e.mesh.material)?e.mesh.material[0]:e.mesh.material,b=materialInfo(material)||material,c=b.color||material.color,vc=b.vertexColors?g.attributes.color:null;

  for(let i=0;i<e.native.length;i++)for(let k=0;k<3;k++)e.colors[i*3+k]=Math.max(0,Math.min(1,srgb([c.r,c.g,c.b][k]*(vc?vc.getComponent(i,k):1))));
 }
 return {entries,peak:peak||1,atlasHash:entries.find(e=>e.atlas?.atlasHash)?.atlas.atlasHash};
}
export const displayWeight=(entry,i,layer,peak)=>layer==='weights'?entry.weights[i]:entry.native[i]/peak*(layer==='result'?entry.weights[i]:1);
// Rasterize actual corner UV triangles with barycentric vertex fields. Material
// and vertex colors are the current asset's appearance; it has no image maps.
export function drawField(canvas,field,sheet,{layer='result',opacity=.55,colors=true,wire=false,zoom=1,pan=[0,0]}={}){
 const size=canvas.width,ctx=canvas.getContext('2d'),image=ctx.createImageData(size,size),pixels=image.data;
 for(let y=0;y<size;y++)for(let x=0;x<size;x++){const n=(y*size+x)*4,c=((x>>4)+(y>>4))%2?37:44;pixels.set([c,c+7,c+9,255],n);}
 for(const e of field.entries){if(e.atlas.sheet!==sheet)continue;const uv=e.atlas.uv,g=e.mesh.geometry,idx=g.index;
  for(let c=0;c<uv.length/2;c+=3){if(e.atlas.valid&&!e.atlas.valid[c/3])continue;const ids=[0,1,2].map(j=>idx?idx.getX(c+j):c+j),p=[0,1,2].map(j=>[uv[(c+j)*2]*size*zoom+pan[0],(1-uv[(c+j)*2+1])*size*zoom+pan[1]]),[a,b,d]=p,den=(b[1]-d[1])*(a[0]-d[0])+(d[0]-b[0])*(a[1]-d[1]);if(Math.abs(den)<1e-8)continue;
   const minX=Math.max(0,Math.floor(Math.min(...p.map(v=>v[0])))),maxX=Math.min(size-1,Math.ceil(Math.max(...p.map(v=>v[0])))),minY=Math.max(0,Math.floor(Math.min(...p.map(v=>v[1])))),maxY=Math.min(size-1,Math.ceil(Math.max(...p.map(v=>v[1]))));
   const ws=ids.map(i=>displayWeight(e,i,layer,field.peak));
   for(let y=minY;y<=maxY;y++)for(let x=minX;x<=maxX;x++){const u=((b[1]-d[1])*(x-d[0])+(d[0]-b[0])*(y-d[1]))/den,v=((d[1]-a[1])*(x-d[0])+(a[0]-d[0])*(y-d[1]))/den,w=1-u-v;if(u<-.001||v<-.001||w<-.001)continue;
    const bary=[u,v,w],weight=bary.reduce((s,t,k)=>s+t*ws[k],0),heat=heatColor(weight),off=(y*size+x)*4;
    for(let k=0;k<3;k++){const color=colors?bary.reduce((s,t,j)=>s+t*e.colors[ids[j]*3+k],0):.22;pixels[off+k]=255*(color*(1-opacity)+heat[k]*opacity);}pixels[off+3]=255;
   }
  }
 }ctx.putImageData(image,0,0);
 if(wire){ctx.strokeStyle='#101c2440';ctx.lineWidth=.35;ctx.beginPath();for(const e of field.entries){if(e.atlas.sheet!==sheet)continue;const uv=e.atlas.uv;for(let i=0;i<uv.length;i+=6){if(e.atlas.valid&&!e.atlas.valid[i/6])continue;ctx.moveTo(uv[i]*size*zoom+pan[0],(1-uv[i+1])*size*zoom+pan[1]);ctx.lineTo(uv[i+2]*size*zoom+pan[0],(1-uv[i+3])*size*zoom+pan[1]);ctx.lineTo(uv[i+4]*size*zoom+pan[0],(1-uv[i+5])*size*zoom+pan[1]);ctx.closePath();}}ctx.stroke();}
}
export function sampleField(field,sheet,u,v){
 let nearest=null,best=Infinity;for(const e of field.entries){if(e.atlas.sheet!==sheet)continue;const uv=e.atlas.uv,idx=e.mesh.geometry.index;for(let c=0;c<uv.length/2;c++){if(e.atlas.valid&&!e.atlas.valid[Math.floor(c/3)])continue;const x=uv[c*2],y=uv[c*2+1],d=(x-u)**2+(y-v)**2;if(d<best){best=d;const i=idx?idx.getX(c):c;nearest={mesh:e.mesh,u:x,v:y,distance:Math.sqrt(d),vertex:i,native:e.native[i]/field.peak,weight:e.weights[i],movement:e.native[i]};}}}return nearest;
}
export function pickUV(field,hit){const e=field.entries.find(e=>e.mesh===hit.object);if(!e||hit.faceIndex==null||(e.atlas.valid&&!e.atlas.valid[hit.faceIndex]))return null;const c=hit.faceIndex*3,ids=[hit.face.a,hit.face.b,hit.face.c],vertices=ids.map(i=>hit.object.getVertexPosition(i,new T.Vector3()).applyMatrix4(hit.object.matrixWorld)),b=new T.Triangle(...vertices).getBarycoord(hit.point,new T.Vector3());if(!b)return null;const uv=e.atlas.uv;return {sheet:e.atlas.sheet,u:b.x*uv[c*2]+b.y*uv[c*2+2]+b.z*uv[c*2+4],v:b.x*uv[c*2+1]+b.y*uv[c*2+3]+b.z*uv[c*2+5]};}
export function influencePreview(scene){
 const group=new T.Group();group.name='Editor influence preview';scene.add(group);const dot=new T.Mesh(new T.SphereGeometry(.0025,12,8),new T.MeshBasicMaterial({color:0xffffff,depthTest:false,toneMapped:false}));dot.renderOrder=40;dot.visible=false;group.add(dot);let items=[],selection=null;
 function clear(){for(const {overlay}of items){group.remove(overlay);overlay.geometry.dispose();overlay.material.dispose();}items=[];selection=null;dot.visible=false;}
 function show(field,layer,opacity){clear();for(const e of field.entries){const source=e.mesh,g=new T.BufferGeometry(),original=source.geometry;g.index=original.index;g.attributes={...original.attributes};g.morphAttributes={...original.morphAttributes};g.morphTargetsRelative=original.morphTargetsRelative;const colors=new Float32Array(e.native.length*3);for(let i=0;i<e.native.length;i++){const c=new T.Color().setRGB(...heatColor(displayWeight(e,i,layer,field.peak)),T.SRGBColorSpace);colors.set([c.r,c.g,c.b],i*3);}g.setAttribute('color',new T.BufferAttribute(colors,3));
   const material=new T.MeshBasicMaterial({vertexColors:true,transparent:true,opacity,depthWrite:false,polygonOffset:true,polygonOffsetFactor:-2,polygonOffsetUnits:-2,side:T.DoubleSide,toneMapped:false}),overlay=source.isSkinnedMesh?new T.SkinnedMesh(g,material):new T.Mesh(g,material);if(source.isSkinnedMesh){overlay.bind(source.skeleton,source.bindMatrix);overlay.bindMode=source.bindMode;}overlay.morphTargetInfluences=source.morphTargetInfluences;overlay.matrixAutoUpdate=false;overlay.frustumCulled=false;overlay.renderOrder=30;group.add(overlay);items.push({source,overlay});}
  tick();
 }
 function tick(){if(selection){const {mesh,index}=selection;mesh.updateWorldMatrix(true,false);mesh.getVertexPosition(index,dot.position).applyMatrix4(mesh.matrixWorld);dot.visible=true;for(let p=mesh;p;p=p.parent)if(!p.visible)dot.visible=false;}for(const {source,overlay}of items){source.updateWorldMatrix(true,false);overlay.matrix.copy(source.matrixWorld);overlay.morphTargetInfluences=source.morphTargetInfluences;overlay.visible=true;for(let p=source;p;p=p.parent)if(!p.visible)overlay.visible=false;}}
 return {show,clear,tick,select(mesh,index){selection={mesh,index};tick();},dispose(){clear();dot.geometry.dispose();dot.material.dispose();scene.remove(group);}};
}

// Bounds use only non-collapsed source-UV triangles that actually move.
export function affectedBounds(field,sheet){
 const min=[Infinity,Infinity],max=[-Infinity,-Infinity];let found=false;
 for(const e of field.entries){if(e.atlas.sheet!==sheet)continue;const {uv,valid}=e.atlas,idx=e.mesh.geometry.index;
  for(let c=0;c<uv.length/2;c++){if(valid&&!valid[Math.floor(c/3)])continue;const i=idx?idx.getX(c):c;if(e.native[i]<=field.peak*.025)continue;const u=uv[c*2],v=uv[c*2+1];if(!Number.isFinite(u)||!Number.isFinite(v))continue;found=true;min[0]=Math.min(min[0],u);min[1]=Math.min(min[1],v);max[0]=Math.max(max[0],u);max[1]=Math.max(max[1],v);}
 }return found?{min,max}:null;
}
export function frameUVBounds({min,max}){
 const zoom=Math.max(.5,Math.min(64,.72/Math.max(max[0]-min[0],max[1]-min[1],.004)));
 return {zoom,pan:[256-(min[0]+max[0])*256*zoom,256-(1-(min[1]+max[1])/2)*512*zoom]};
}
