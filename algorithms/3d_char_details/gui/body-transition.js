import * as T from 'three';

export const BODY_TRANSITION_CONTROLS={
 bodyTransitionWidth:'Transition width',
 bodyTransitionFrontDepth:'Transition front depth',
 bodyTransitionBackDepth:'Transition back depth',
 bodyTransitionHeight:'Transition height'
};
const smooth=(a,b,x)=>{const t=T.MathUtils.clamp((x-a)/(b-a),0,1);return t*t*(3-2*t);};
export function transitionDelta(p,name){
 const x=p.x+.004865,w=smooth(.710,.747,p.y)*(1-smooth(.780,.818,p.y))*(1-smooth(.068,.105,Math.abs(x)));
 const front=smooth(-.036,-.021,p.z),d=new T.Vector3();
 if(name==='bodyTransitionWidth')d.x=x*.35*w;
 if(name==='bodyTransitionHeight')d.y=.015*w;
 if(name==='bodyTransitionFrontDepth')d.z=.014*w*front;
 if(name==='bodyTransitionBackDepth')d.z=-.014*w*(1-front);
 return d;
}

// Add regional, zero-neutral morphs before the editor captures its rest geometry.
// This preserves the source asset, all existing morph indices and saved values.
export function installBodyTransition(model){
 model.updateMatrixWorld(true);let meshes=0;
 model.traverse(o=>{
  if(!o.isMesh)return;let parent=o;while(parent&&parent.name!=='Body_Complete')parent=parent.parent;if(!parent)return;
  const g=o.geometry,base=g.attributes.position,normal=g.attributes.normal,world=o.matrixWorld.clone(),inverse=world.clone().invert();
  const normalWorld=new T.Matrix3().getNormalMatrix(world),normalLocal=new T.Matrix3().getNormalMatrix(inverse);
  g.morphAttributes.position??=[];const count=g.morphAttributes.position.length;
  if(normal&&!g.morphAttributes.normal)g.morphAttributes.normal=Array.from({length:count},()=>new T.Float32BufferAttribute(new Float32Array(base.count*3),3));
  o.morphTargetDictionary??={};o.morphTargetInfluences??=[];
  for(const name of Object.keys(BODY_TRANSITION_CONTROLS)){
   if(o.morphTargetDictionary[name]!==undefined)continue;
   const position=new T.Float32BufferAttribute(new Float32Array(base.count*3),3),normals=normal?new T.Float32BufferAttribute(new Float32Array(base.count*3),3):null;position.name=name;
   for(let i=0;i<base.count;i++){
    const original=new T.Vector3().fromBufferAttribute(base,i),p=original.clone().applyMatrix4(world),delta=transitionDelta(p,name);
    const target=delta.lengthSq()?p.clone().add(delta).applyMatrix4(inverse):original.clone();
    position.setXYZ(i,...(g.morphTargetsRelative?target.sub(original):target).toArray());
    if(normals){
     const n=new T.Vector3().fromBufferAttribute(normal,i),changed=n.clone();
     if(delta.lengthSq()){
      // Inverse-transpose of the deformation Jacobian keeps shading continuous
      // across the material primitives without replacing authored face normals.
      const cols=[];for(const axis of ['x','y','z']){const a=p.clone(),b=p.clone();a[axis]+=.00001;b[axis]-=.00001;const c=transitionDelta(a,name).sub(transitionDelta(b,name)).multiplyScalar(50000);c[axis]+=1;cols.push(c);}
      const [a,b,c]=cols,j=new T.Matrix3().set(a.x,b.x,c.x,a.y,b.y,c.y,a.z,b.z,c.z).invert().transpose();
      changed.applyNormalMatrix(normalWorld).applyNormalMatrix(j).applyNormalMatrix(normalLocal);
     }
     normals.setXYZ(i,...(g.morphTargetsRelative?changed.sub(n):changed).toArray());
    }
   }
   const index=g.morphAttributes.position.length;g.morphAttributes.position.push(position);if(normals)g.morphAttributes.normal.push(normals);
   o.morphTargetDictionary[name]=index;o.morphTargetInfluences[index]=0;
  }
  meshes++;
 });
 if(!meshes)throw new Error('Connected body skin is unavailable for transition controls');
}
