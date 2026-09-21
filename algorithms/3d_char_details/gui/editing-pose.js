import * as THREE from 'three';

// An editing baseline, independent of the authored bind pose and running clip.
// Solve each segment in world space, then retain its local quaternion. Twist
// children inherit the upper arm; rotating them again would double the lift.
export function editingPose(model,bones) {
  const result=new Map();
  model.updateMatrixWorld(true);
  const snapshot=new Map([...bones].map(([n,{bone}])=>[n,bone.quaternion.clone()]));
  for(const [side,sign]of [['l',1],['r',-1]]){
    for(const [name,end]of [['arm_stretch_','forearm_stretch_'],['forearm_stretch_','hand_']]){
      const bone=bones.get(name+side)?.bone,tip=bones.get(end+side)?.bone;
      if(!bone||!tip)throw new Error('Missing editing-pose arm joints');
      const direction=tip.getWorldPosition(new THREE.Vector3()).sub(bone.getWorldPosition(new THREE.Vector3())).normalize();
      const correction=new THREE.Quaternion().setFromUnitVectors(direction,new THREE.Vector3(sign,0,0));
      const world=bone.getWorldQuaternion(new THREE.Quaternion());
      const parent=bone.parent.getWorldQuaternion(new THREE.Quaternion());
      bone.quaternion.copy(parent.invert().multiply(correction).multiply(world));
      result.set(bone.name,bone.quaternion.clone());
      model.updateMatrixWorld(true);
    }
  }
  for(const [n,q]of snapshot)bones.get(n).bone.quaternion.copy(q);
  model.updateMatrixWorld(true);
  return {quaternion:(name,mode='t')=>mode==='t'?result.get(name):undefined};
}
