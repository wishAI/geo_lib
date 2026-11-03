import * as THREE from 'three';
import {originMatrix, worldTransforms, fromRows} from '/api/artifact?path=algorithms/avp_remote/gui/pose-math.js';

const position=m=>new THREE.Vector3().setFromMatrixPosition(m);
const Y=new THREE.Vector3(0,1,0),ONE=new THREE.Vector3(1,1,1);
const CHAINS=[['thumb',1,3],['index',5,4],['middle',10,4],['ring',15,4],['pinky',20,4]];

// Identical anatomical polarity to the arm solver: little -> index, wrist ->
// knuckles. Derive directions from positions, not vendor-dependent joint axes.
// https://www.w3.org/TR/webxr-hand-input-1/#skeleton-joints
function palmBasis(points){
  const x=points[5].clone().sub(points[20]);
  const y=points[6].clone().add(points[11]).add(points[16]).multiplyScalar(1/3).sub(points[0]);
  if(x.lengthSq()<1e-10)return null;
  x.normalize();y.addScaledVector(x,-y.dot(x));
  if(y.lengthSq()<1e-10)return null;
  y.normalize();return new THREE.Matrix4().makeBasis(x,y,new THREE.Vector3().crossVectors(x,y));
}

// Animation rig, independent of the reduced one-axis robot articulation. Every
// source phalanx drives its corresponding bone (including the distal -> tip
// direction). Only rotations change: original pivots, bone lengths, bind frames
// and skin weights remain intact, so Landau keeps its smaller hands.
export class HandRig {
  constructor(joints){
    const rest=worldTransforms(joints,{}),byName=new Map(joints.map(j=>[j.child,j]));
    this.sides={};this.names=new Set();
    for(const [side,suffix] of [['left','l'],['right','r']]){
      const hand='hand_'+suffix,inverse=rest[hand].clone().invert(),points=Array(25);
      points[0]=new THREE.Vector3();
      const chains=CHAINS.map(([finger,start,count])=>Array.from({length:count},(_,i)=>{
        const name=(count===4?(i===0?finger+'1_base':finger+i):finger+(i+1))+'_'+suffix;
        const joint=byName.get(name),origin=originMatrix(joint);
        points[start+i]=position(inverse.clone().multiply(rest[name]));this.names.add(name);
        return {name,parent:joint.parent,origin,index:start+i};
      }));
      this.sides[side]={hand,chains,basis:palmBasis(points)};
    }
  }
  solve(frame){
    const result={};
    for(const [side,rig] of Object.entries(this.sides)){
      const stack=frame?.[side+'_arm'];if(!stack||![25,27].includes(stack.length))continue;
      const points=stack.map(m=>position(fromRows(m)));
      if(points.some(p=>!p.toArray().every(Number.isFinite)))continue;
      const measured=palmBasis(points);if(!measured)continue;
      const alignment=rig.basis.clone().multiply(measured.transpose());
      const rotations={[rig.hand]:new THREE.Quaternion()};
      for(const chain of rig.chains){
        // Reject an incomplete/degenerate chain as a unit; never retain stale
        // curl from a previous frame or create a direction from a zero vector.
        const directions=chain.map(({index})=>points[index+1].clone().sub(points[index]));
        if(directions.some(v=>v.lengthSq()<1e-10))continue;
        chain.forEach(({name,parent,origin},i)=>{
          const target=directions[i].transformDirection(alignment);
          const before=rotations[parent].clone().multiply(new THREE.Quaternion().setFromRotationMatrix(origin));
          const predicted=Y.clone().applyQuaternion(before);
          // Parallel transport through the chain preserves bind roll without
          // unstable bend-plane normals when a finger is fully straight.
          const swing=new THREE.Quaternion().setFromUnitVectors(predicted,target);
          const after=swing.multiply(before);
          result[name]=before.clone().invert().multiply(after).normalize().toArray();
          rotations[name]=after;
        });
      }
    }
    return result;
  }
  apply(worlds,pose){
    for(const rig of Object.values(this.sides))for(const chain of rig.chains){
      if(!chain.some(({name})=>pose?.[name]))continue;
      // A partial imported pose still propagates through the whole chain;
      // unspecified descendants use their bind rotation, never stale worlds.
      for(const {name,parent,origin} of chain){
        const rotation=pose[name]?new THREE.Quaternion().fromArray(pose[name]):new THREE.Quaternion();
        worlds[name]=worlds[parent].clone().multiply(origin).multiply(new THREE.Matrix4().compose(new THREE.Vector3(),rotation,ONE));
      }
    }
    return worlds;
  }
}

export function validateHandPose(pose){
  if(!pose||typeof pose!=='object'||Array.isArray(pose))return false;
  return Object.entries(pose).every(([name,q])=>/^(thumb[123]|(?:index|middle|ring|pinky)(?:1_base|[123]))_[lr]$/.test(name)
    &&Array.isArray(q)&&q.length===4&&q.every(Number.isFinite)&&Math.abs(Math.hypot(...q)-1)<1e-4);
}

export function blendHandPoses(a={},b={},alpha=0){
  return Object.fromEntries(Object.entries(a).map(([name,q])=>[name,b[name]
    ?new THREE.Quaternion().fromArray(q).slerp(new THREE.Quaternion().fromArray(b[name]),alpha).toArray():q.slice()]));
}
