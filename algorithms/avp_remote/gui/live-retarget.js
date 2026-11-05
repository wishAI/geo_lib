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
    this.config={...data.retargetConfig,rules:{...data.retargetConfig.rules}};this.iterations=iterations;
    // Existing scene.json files baked the erroneous pitch split into both
    // rules. Upgrade those rules without rebuilding the unchanged GLB/meshes.
    for(const name of ['neck_x','head_x'])if(this.config.rules[name]?.formula==='head pitch')
      this.config.rules[name]={...this.config.rules[name],scale:1,formula:name==='neck_x'?'head look-direction IK pitch':'head look-direction IK yaw'};
    this.joints=new Map(data.joints.map(j=>[j.child,{...j,origin:originMatrix(j),axisVector:new THREE.Vector3(...j.axis).normalize()}]));
    this.zero=Object.fromEntries(data.joints.filter(j=>j.type!=='fixed').map(j=>[j.child,0]));
    this.transform=fromRows(data.trackingTransform);
    const rest=worldTransforms(data.joints,{});this.baseInverse=rest[this.config.base].clone().invert();this.arms={};
    this.head={chain:['neck_x','head_x'].map(n=>this.joints.get(n)),values:[0,0],worlds:[new THREE.Matrix4(),new THREE.Matrix4()]};
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
  targetRotation(arm,stack){
    if(!stack?.[20])return null;
    const p=i=>position(fromRows(stack[i]));
    const lateral=p(5).sub(p(20)),forward=p(6).add(p(11)).add(p(16)).multiplyScalar(1/3).sub(p(0));
    if(lateral.lengthSq()<1e-10||forward.lengthSq()<1e-10||new THREE.Vector3().crossVectors(lateral,forward).lengthSq()<1e-12)return null;
    const world=basis(lateral,forward).multiply(arm.basis.clone().transpose());
    return new THREE.Matrix4().extractRotation(this.baseInverse.clone().multiply(this.transform).multiply(world));
  }
  solveArm(arm,wrist,pose,rotation=null){
    const target=position(this.baseInverse.clone().multiply(this.transform).multiply(fromRows(wrist)));
    if(target.length()>arm.cap)target.setLength(arm.cap);
    const tip=()=>position(arm.worlds.at(-1));
    for(let iteration=0;iteration<this.iterations;iteration++){
      this.chainWorlds(arm);
      const rotationError=rotation?new THREE.Quaternion().setFromRotationMatrix(arm.worlds.at(-1)).angleTo(new THREE.Quaternion().setFromRotationMatrix(rotation)):0;
      if(tip().distanceToSquared(target)<1e-6&&rotationError<.025)break;
      for(let i=arm.chain.length-1;i>=0;i--){
        const j=arm.chain[i];if(j.type==='fixed')continue;
        this.chainWorlds(arm);const m=arm.worlds[i],origin=position(m),axis=j.axisVector.clone().transformDirection(m);
        const a=tip().sub(origin),b=target.clone().sub(origin);a.addScaledVector(axis,-a.dot(axis));b.addScaledVector(axis,-b.dot(axis));
        // Minimize wrist position plus palm-frame orientation. Position-only CCD
        // cannot observe wrist/forearm roll when their axis passes through the tip.
        let sin=axis.dot(new THREE.Vector3().crossVectors(a,b)),cos=a.dot(b);
        if(rotation)for(let column=0;column<3;column++){
          const u=new THREE.Vector3().setFromMatrixColumn(arm.worlds.at(-1),column),v=new THREE.Vector3().setFromMatrixColumn(rotation,column);
          u.addScaledVector(axis,-u.dot(axis));v.addScaledVector(axis,-v.dot(axis));
          sin+=.08**2*axis.dot(new THREE.Vector3().crossVectors(u,v));cos+=.08**2*u.dot(v);
        }
        if(Math.abs(sin)+Math.abs(cos)<1e-16)continue;
        const delta=Math.atan2(sin,cos);
        arm.values[i]=THREE.MathUtils.clamp(arm.values[i]+.85*delta,j.lower,j.upper);
      }
    }
    arm.chain.forEach((j,i)=>{if(j.type!=='fixed')pose[j.child]=this.mapped(j.child,arm.values[i]);});
  }
  fingers(arm,stack,pose){
    if(!stack?.[0])return;
    const wristInverse=fromRows(stack[0]).invert();
    const local=Object.fromEntries(stack.flatMap((m,i)=>m?[ [HAND_NAMES[i],position(wristInverse.clone().multiply(fromRows(m)))] ]:[]));
    const valid=p=>p&&p.toArray().every(Number.isFinite);
    if(!['indexMetacarpal','littleMetacarpal','indexKnuckle','middleKnuckle','ringKnuckle'].every(n=>valid(local[n])))return;
    const lateral=local.indexMetacarpal.clone().sub(local.littleMetacarpal);
    const forward=local.indexKnuckle.clone().add(local.middleKnuckle).add(local.ringKnuckle).multiplyScalar(1/3);
    if(lateral.lengthSq()<1e-16||new THREE.Vector3().crossVectors(lateral,forward).lengthSq()<1e-16)return;
    const rotation=arm.basis.clone().multiply(basis(lateral,forward).transpose());
    for(const point of Object.values(local))point.applyMatrix4(rotation);
    const set=(name,value)=>{pose[name+'_'+arm.suffix]=this.mapped(name+'_'+arm.suffix,value);};
    for(const [finger,layout] of Object.entries(this.config.fingers)){
      const p=layout.avp.map(n=>local[n]);if(!p.every(valid))continue;
      const segments=p.slice(1).map((v,i)=>v.clone().sub(p[i]));
      if(segments.some(v=>v.lengthSq()<1e-16))continue;
      if(finger==='thumb'){
        set(layout.robot[0],Math.atan2(p[0].x,Math.max(Math.abs(p[0].y),1e-6)));
        const flexSign=arm.suffix==='l'?1:-1;
        set(layout.robot[1],flexSign*1.15*angle(segments[0],segments[1]));set(layout.robot[2],flexSign*1.05*angle(segments[1],segments[2]));
      }else{
        const length=segments.reduce((sum,v)=>sum+v.length(),0),curl=length>1e-8?THREE.MathUtils.clamp(1-p.at(-1).distanceTo(p[0])/length,0,1):0;
        set(layout.robot[0],Math.atan2(segments[0].x,Math.max(Math.abs(segments[0].y),1e-6)));
        [.55,.75,.65].forEach((factor,i)=>set(layout.robot[i+1],-Math.max(angle(segments[i],segments[i+1]),factor*1.9*curl)));
      }
    }
  }
  headPose(frame){
    if(!frame?.head)return {neck_x:0,head_x:0};
    const m=this.transform.clone().multiply(fromRows(frame.head));
    const target=new THREE.Vector3().setFromMatrixColumn(m,1).transformDirection(this.baseInverse);
    // Native +Y maps to avatar forward -Y (head-local +Z). Fit that direction
    // using the actual serial pitch/yaw axes, not a pitch split or Euler order.
    const head=this.head;head.values.fill(0);
    const forward=()=>{this.chainWorlds(head);return new THREE.Vector3(0,0,1).transformDirection(head.worlds.at(-1));};
    for(let iteration=0;iteration<16;iteration++){
      if(forward().distanceTo(target)<1e-10)break;
      for(let i=head.chain.length-1;i>=0;i--){
        const current=forward(),j=head.chain[i],axis=j.axisVector.clone().transformDirection(head.worlds[i]);
        const a=current.addScaledVector(axis,-current.dot(axis)),b=target.clone().addScaledVector(axis,-target.dot(axis));
        const sin=axis.dot(new THREE.Vector3().crossVectors(a,b)),cos=a.dot(b);
        if(Math.abs(sin)+Math.abs(cos)>1e-16)head.values[i]=THREE.MathUtils.clamp(head.values[i]+Math.atan2(sin,cos),j.lower,j.upper);
      }
    }
    return Object.fromEntries(head.chain.map((j,i)=>[j.child,this.mapped(j.child,head.values[i])]));
  }
  solve(frame){
    const pose={...this.zero,...this.headPose(frame)};
    for(const [side,arm] of Object.entries(this.arms)){
      const stack=frame?.[side+'_arm'],wrist=stack?.[0]||frame?.[side+'_wrist'];
      if(wrist)this.solveArm(arm,wrist,pose,this.targetRotation(arm,stack));else arm.values.fill(0);
      if(stack&&[25,27].includes(stack.length))this.fingers(arm,stack,pose);
    }
    return pose;
  }
}
