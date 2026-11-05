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
    const descendant=(child,ancestor)=>{const seen=new Set();while(parents.has(child)&&!seen.has(child)){seen.add(child);child=parents.get(child);if(child===ancestor)return true;}return false;};
    const ownerFor=name=>{for(const [part,distal] of [['arm','forearm_stretch'],['thigh','leg_stretch'],['leg','foot']]){
      if(!name.startsWith(part+'_stretch_'))continue;const side=name.slice(-1),owner=part+'_twist_'+side;
      if(descendant(owner,name)&&descendant(distal+'_'+side,owner))return owner;
    }return name;};

    this.records=[];this.byBone=new Map();
    model.traverse(bone=>{
      if(!bone.isBone)return;
      bone.matrixAutoUpdate=false;
      if(!rest[bone.name])throw new Error('USD bone absent from URDF: '+bone.name);
      const bind=inverseRoot.clone().multiply(bone.matrixWorld);
      const source=ownerFor(bone.name);
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
