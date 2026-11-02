import * as THREE from 'three';
import {worldTransforms} from '/api/artifact?path=algorithms/avp_remote/gui/pose-math.js';

// Apply each URDF link's motion relative to rest to the corresponding USD bind
// frame. The exported USD twist bones are siblings, not URDF serial parents.
export class SkinRetargeter {
  constructor(model,joints){
    model.updateMatrixWorld(true);
    const inverseRoot=model.matrixWorld.clone().invert(),rest=worldTransforms(joints,{});
    this.records=[];this.byBone=new Map();
    model.traverse(bone=>{
      if(!bone.isBone)return;
      bone.matrixAutoUpdate=false;
      if(!rest[bone.name])throw new Error('USD bone absent from URDF: '+bone.name);
      const bind=inverseRoot.clone().multiply(bone.matrixWorld);
      const record={bone,offset:rest[bone.name].clone().invert().multiply(bind),target:new THREE.Matrix4(),
        parentRest:inverseRoot.clone().multiply(bone.parent.matrixWorld)};
      this.records.push(record);this.byBone.set(bone,record);
    });
    this.local=new THREE.Matrix4();
  }
  apply(worlds){
    for(const record of this.records)record.target.copy(worlds[record.bone.name]).multiply(record.offset);
    for(const {bone,target,parentRest} of this.records){
      const parent=this.byBone.get(bone.parent)?.target||parentRest;
      this.local.copy(parent).invert().multiply(target);
      bone.matrix.copy(this.local);bone.matrix.decompose(bone.position,bone.quaternion,bone.scale);
      bone.matrixWorldNeedsUpdate=true;
    }
  }
}
