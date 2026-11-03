import * as THREE from 'three';
import {worldTransforms} from '/api/artifact?path=algorithms/avp_remote/gui/pose-math.js';

// Apply each URDF link's motion relative to rest to the corresponding USD bind
// frame. The exported USD twist bones are siblings, not URDF serial parents.
// In the serial arm, arm_twist owns the whole upper-arm geometry. Its motion
// must also drive the stretch skin weights, or the sleeve stays behind the elbow.
export class SkinRetargeter {
  constructor(model,joints){
    model.updateMatrixWorld(true);
    const inverseRoot=model.matrixWorld.clone().invert(),rest=worldTransforms(joints,{});
    const parents=new Map(joints.map(j=>[j.child,j.parent]));
    this.records=[];this.byBone=new Map();
    model.traverse(bone=>{
      if(!bone.isBone)return;
      bone.matrixAutoUpdate=false;
      if(!rest[bone.name])throw new Error('USD bone absent from URDF: '+bone.name);
      const bind=inverseRoot.clone().multiply(bone.matrixWorld);
      const twist=bone.name.replace(/^arm_stretch_/, 'arm_twist_');
      const forearm=bone.name.replace(/^arm_stretch_/, 'forearm_stretch_');
      const source=twist!==bone.name&&parents.get(twist)===bone.name&&parents.get(forearm)===twist?twist:bone.name;
      const record={bone,source,offset:rest[source].clone().invert().multiply(bind),target:new THREE.Matrix4(),
        parentRest:inverseRoot.clone().multiply(bone.parent.matrixWorld)};
      this.records.push(record);this.byBone.set(bone,record);
    });
    this.local=new THREE.Matrix4();
  }
  apply(worlds){
    for(const record of this.records)record.target.copy(worlds[record.source]).multiply(record.offset);
    for(const {bone,target,parentRest} of this.records){
      const parent=this.byBone.get(bone.parent)?.target||parentRest;
      this.local.copy(parent).invert().multiply(target);
      bone.matrix.copy(this.local);bone.matrix.decompose(bone.position,bone.quaternion,bone.scale);
      bone.matrixWorldNeedsUpdate=true;
    }
  }
}
