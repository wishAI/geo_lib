import * as THREE from 'three';
import {originMatrix, worldTransforms, fromRows} from '/api/artifact?path=algorithms/avp_remote/gui/pose-math.js';
import {HAND_NAMES} from '/api/artifact?path=algorithms/avp_remote/gui/tracking-markers.js';

const position=m=>new THREE.Vector3().setFromMatrixPosition(m);
const basis=(lateral,forward)=>{
  const x=lateral.clone().normalize(),y=forward.clone().addScaledVector(x,-x.dot(forward)).normalize();
  if(x.lengthSq()<1e-12||y.lengthSq()<1e-12)return new THREE.Matrix4();
  const z=new THREE.Vector3().crossVectors(x,y).normalize();y.crossVectors(z,x).normalize();
  return new THREE.Matrix4().makeBasis(x,y,z);
};
const angle=(a,b)=>a.lengthSq()<1e-16||b.lengthSq()<1e-16?0:a.angleTo(b);

// Bounded, warm-started CCD uses the same URDF and exported mapping rules as
// landau_retarget.py. Live samples never wait on HTTP or queue stale work.
export class LiveRetargeter {
  constructor(data,{iterations=10}={}){
    if(!data.retargetConfig)throw new Error('模型缺少实时映射配置，请刷新模型资源');
    this.config=data.retargetConfig;this.iterations=iterations;
    this.joints=new Map(data.joints.map(j=>[j.child,{...j,origin:originMatrix(j),axisVector:new THREE.Vector3(...j.axis).normalize()}]));
    this.zero=Object.fromEntries(data.joints.filter(j=>j.type!=='fixed').map(j=>[j.child,0]));
    this.transform=fromRows(data.trackingTransform);
    const rest=worldTransforms(data.joints,{});this.baseInverse=rest[this.config.base].clone().invert();this.arms={};
    for(const [side,suffix] of [['left','l'],['right','r']]){
      const tip='hand_'+suffix,chain=[];let name=tip;
      while(name!==this.config.base){const j=this.joints.get(name);if(!j)throw new Error('手臂关节链不完整');chain.unshift(j);name=j.parent;}
      const inverse=rest[tip].clone().invert(),local=name=>position(inverse.clone().multiply(rest[name+'_'+suffix]));
      const lateral=local('index1_base').sub(local('pinky1_base'));
      const forward=local('index1').add(local('middle1')).add(local('ring1')).multiplyScalar(1/3);
      this.arms[side]={suffix,chain,values:chain.map(()=>0),worlds:chain.map(()=>new THREE.Matrix4()),
        cap:position(this.baseInverse.clone().multiply(rest[tip])).length()*this.config.radiusCapScale,basis:basis(lateral,forward)};
    }
    this.motion=new THREE.Matrix4();this.identity=new THREE.Matrix4();
  }
  mapped(name,value){const rule=this.config.rules[name],j=this.joints.get(name);return THREE.MathUtils.clamp(value*(rule?.scale??1)+(rule?.bias??0),j?.lower??-Infinity,j?.upper??Infinity);}
  chainWorlds(arm){
    let parent=this.identity;
    arm.chain.forEach((j,i)=>{
      const m=arm.worlds[i].copy(parent).multiply(j.origin),q=arm.values[i];
      if(j.type==='prismatic')m.multiply(this.motion.makeTranslation(...j.axis.map(v=>v*q)));
      else if(j.type!=='fixed')m.multiply(this.motion.makeRotationAxis(j.axisVector,q));
      parent=m;
    });return arm.worlds;
  }
  solveArm(arm,wrist,pose){
    const target=position(this.baseInverse.clone().multiply(this.transform).multiply(fromRows(wrist)));
    if(target.length()>arm.cap)target.setLength(arm.cap);
    const tip=()=>position(arm.worlds.at(-1));
    for(let iteration=0;iteration<this.iterations;iteration++){
      this.chainWorlds(arm);if(tip().distanceToSquared(target)<1e-6)break;
      for(let i=arm.chain.length-1;i>=0;i--){
        const j=arm.chain[i];if(j.type==='fixed')continue;
        this.chainWorlds(arm);const m=arm.worlds[i],origin=position(m),axis=j.axisVector.clone().transformDirection(m);
        const a=tip().sub(origin),b=target.clone().sub(origin);a.addScaledVector(axis,-a.dot(axis));b.addScaledVector(axis,-b.dot(axis));
        if(a.lengthSq()<1e-16||b.lengthSq()<1e-16)continue;a.normalize();b.normalize();
        const delta=Math.atan2(axis.dot(new THREE.Vector3().crossVectors(a,b)),THREE.MathUtils.clamp(a.dot(b),-1,1));
        arm.values[i]=THREE.MathUtils.clamp(arm.values[i]+.85*delta,j.lower,j.upper);
      }
    }
    arm.chain.forEach((j,i)=>{if(j.type!=='fixed')pose[j.child]=this.mapped(j.child,arm.values[i]);});
  }
  fingers(arm,stack,pose){
    const wristInverse=fromRows(stack[0]).invert();
    const local=Object.fromEntries(stack.map((m,i)=>[HAND_NAMES[i],position(wristInverse.clone().multiply(fromRows(m)))]));
    const lateral=local.littleMetacarpal.clone().sub(local.indexMetacarpal);
    const forward=local.indexKnuckle.clone().add(local.middleKnuckle).add(local.ringKnuckle).multiplyScalar(1/3);
    const rotation=arm.basis.clone().multiply(basis(lateral,forward).transpose());
    for(const point of Object.values(local))point.applyMatrix4(rotation);
    const set=(name,value)=>{pose[name+'_'+arm.suffix]=this.mapped(name+'_'+arm.suffix,value);};
    for(const [finger,layout] of Object.entries(this.config.fingers)){
      const p=layout.avp.map(n=>local[n]);if(p.some(v=>!v))continue;
      const segments=p.slice(1).map((v,i)=>v.clone().sub(p[i]));
      if(finger==='thumb'){
        set(layout.robot[0],Math.atan2(p[0].x,Math.max(Math.abs(p[0].y),1e-6)));
        set(layout.robot[1],-1.15*angle(segments[0],segments[1]));set(layout.robot[2],-1.05*angle(segments[1],segments[2]));
      }else{
        const length=segments.reduce((sum,v)=>sum+v.length(),0),curl=length>1e-8?THREE.MathUtils.clamp(1-p.at(-1).distanceTo(p[0])/length,0,1):0;
        set(layout.robot[0],Math.atan2(segments[0].x,Math.max(Math.abs(segments[0].y),1e-6)));
        [.55,.75,.65].forEach((factor,i)=>set(layout.robot[i+1],-Math.max(angle(segments[i],segments[i+1]),factor*1.9*curl)));
      }
    }
  }
  solve(frame){
    const pose={...this.zero};
    if(frame?.head){const m=this.transform.clone().multiply(fromRows(frame.head)),forward=new THREE.Vector3().setFromMatrixColumn(m,1).normalize();
      const pitch=Math.atan2(forward.z,Math.max(Math.hypot(forward.x,forward.y),1e-6));for(const name of ['neck_x','head_x'])pose[name]=this.mapped(name,pitch);}
    for(const [side,arm] of Object.entries(this.arms)){
      const stack=frame?.[side+'_arm'],wrist=stack?.[0]||frame?.[side+'_wrist'];
      if(wrist)this.solveArm(arm,wrist,pose);else arm.values.fill(0);
      if(stack&&[25,27].includes(stack.length))this.fingers(arm,stack,pose);
    }
    return pose;
  }
}
