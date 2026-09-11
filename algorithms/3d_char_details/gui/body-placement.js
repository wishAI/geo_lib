import * as THREE from 'three';

// Edit the rest geometry and rest joints together. Recompute inverse bind
// matrices in rest pose, then restore the user's pose. No runtime scale is
// left on the root bone, which would also resize the head.
export function bodyPlacement(model,bones){
 model.updateMatrixWorld(true);
 const pivot=new THREE.Vector3(-.004865,.806,-.032925);
 const low=.748,high=.808;
 const weight=y=>{const t=THREE.MathUtils.clamp((y-low)/(high-low),0,1);return 1-t*t*(3-2*t);};
 const map=(p,scale,offset)=>{const w=weight(p.y);return p.clone().addScaledVector(p.clone().sub(pivot),w*(scale-1)).add(new THREE.Vector3(w*offset,0,0));};
 const meshes=[],joints=[],skeletons=new Set();
 model.traverse(o=>{
  if(o.isBone)joints.push({bone:o,world:o.matrixWorld.clone(),position:o.position.clone(),quaternion:o.quaternion.clone(),scale:o.scale.clone()});
  if(!o.isMesh)return;
  o.geometry=o.geometry.clone();
  const g=o.geometry;meshes.push({o,world:o.matrixWorld.clone(),inverse:o.matrixWorld.clone().invert(),position:g.attributes.position.clone(),normal:g.attributes.normal?.clone(),morph:(g.morphAttributes.position||[]).map(a=>a.clone())});
  if(o.skeleton)skeletons.add(o.skeleton);
 });
 return (scale=1,offset=0)=>{
  model.userData.bodyFrame={scale,offset,pivot:pivot.toArray()};
  for(const m of meshes){
   const g=m.o.geometry,base=m.position,p=new THREE.Vector3(),q=new THREE.Vector3(),n=new THREE.Vector3();
   for(let i=0;i<base.count;i++){
    p.fromBufferAttribute(base,i).applyMatrix4(m.world);q.copy(map(p,scale,offset)).applyMatrix4(m.inverse);g.attributes.position.setXYZ(i,q.x,q.y,q.z);
    for(let k=0;k<m.morph.length;k++){
     const a=m.morph[k];q.fromBufferAttribute(a,i);if(g.morphTargetsRelative)q.add(new THREE.Vector3().fromBufferAttribute(base,i));q.applyMatrix4(m.world);q.copy(map(q,scale,offset)).applyMatrix4(m.inverse);
     if(g.morphTargetsRelative)q.sub(new THREE.Vector3().fromBufferAttribute(g.attributes.position,i));g.morphAttributes.position[k].setXYZ(i,q.x,q.y,q.z);
    }
    // Inverse-transpose of the smooth neck transition's Jacobian.
    if(m.normal){
     n.fromBufferAttribute(m.normal,i).transformDirection(m.world);
     const t=THREE.MathUtils.clamp((p.y-low)/(high-low),0,1),dw=(t>0&&t<1)?-6*t*(1-t)/(high-low):0,a=1+weight(p.y)*(scale-1);
     const bx=dw*((scale-1)*(p.x-pivot.x)+offset),by=dw*(scale-1)*(p.y-pivot.y),bz=dw*(scale-1)*(p.z-pivot.z);
     n.set(n.x/a,(n.y-(bx*n.x+bz*n.z)/a)/(a+by),n.z/a).normalize().transformDirection(m.inverse);g.attributes.normal.setXYZ(i,n.x,n.y,n.z);
    }
   }
   g.attributes.position.needsUpdate=true;if(g.attributes.normal)g.attributes.normal.needsUpdate=true;
   for(const a of g.morphAttributes.position||[])a.needsUpdate=true;
   g.computeBoundingBox();g.computeBoundingSphere();
  }
  const rest=new Map();
  for(const j of joints){const matrix=j.world.clone(),p=new THREE.Vector3().setFromMatrixPosition(matrix);matrix.setPosition(map(p,scale,offset));rest.set(j.bone,matrix);}
  for(const j of joints){const parent=rest.get(j.bone.parent)||j.bone.parent.matrixWorld;const local=parent.clone().invert().multiply(rest.get(j.bone));local.decompose(j.bone.position,j.bone.quaternion,j.bone.scale);}
  model.updateMatrixWorld(true);for(const s of skeletons)s.calculateInverses();
  // applyBone() restores animation after this call; rest rotations are unchanged.
  for(const j of joints){const item=bones.get(j.bone.name);if(item){item.quaternion.copy(j.bone.quaternion);item.scale.copy(j.bone.scale);}}
 };
}
