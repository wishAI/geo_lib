import {regionWeight,polygonValid,validateRegions} from './uv-regions.mjs';
import fs from 'node:fs';
import assert from 'node:assert/strict';
import * as T from './three.mjs';
import {GLTFLoader} from './loader.mjs';
import {GLTFExporter} from './exporter.mjs';
import {editingPose} from './editing.mjs';
import {bodyPlacement} from './placement.mjs';
import {motionPlayer} from './motion.mjs';
import {garmentFit,fitDefaults,setOutfitValue,outfitValue,OUTFIT_ASSEMBLIES,ANGLES,outfitMinimum,setJointAngle,setJointLink,jointsLinked,DEPTH_CONTROLS,COLLAR_CONTROLS,SHOULDER_CONTROLS,depthNames,depthFrontWeight} from './fit.mjs';
import {installBodyTransition,BODY_TRANSITION_CONTROLS} from './transition.mjs';
globalThis.ProgressEvent=class{constructor(type,args){Object.assign(this,args);}};
const data=fs.readFileSync(process.argv[2]),jl=data.readUInt32LE(12),json=JSON.parse(data.subarray(20,20+jl));
json.buffers[0].uri='data:application/octet-stream;base64,'+data.subarray(28+jl).toString('base64');
delete json.images;delete json.textures;delete json.samplers;
const {scene,animations}=await new GLTFLoader().parseAsync(JSON.stringify(json),'');
const report=JSON.parse(fs.readFileSync(process.argv[3])),parts=new Map(),bones=new Map(),protectedGeometry=new Map();
scene.traverse(o=>{if(o.isBone)bones.set(o.name,{bone:o,quaternion:o.quaternion.clone(),scale:o.scale.clone()});if(o.userData.part_type)parts.set(o.name,{object:o});});
installBodyTransition(scene);
const body=bodyPlacement(scene,bones),fit=garmentFit(scene,parts,report);
scene.traverse(o=>{if(o.isMesh){let p=o;while(p&&!p.userData.part_type)p=p.parent;if(p?.userData.part_type!=='clothing')protectedGeometry.set(o,Array.from(o.geometry.attributes.position.array));}});
let player;player=motionPlayer(scene,animations,bones,()=>{for(const[n,b]of bones){b.bone.quaternion.copy(player?.quaternion(n)||b.quaternion);b.bone.scale.copy(b.scale);}});
const jointConfig=fitDefaults();assert(jointsLinked(jointConfig,'Sleeve_L'));assert(jointsLinked(jointConfig,'Trousers'));
assert.equal(jointConfig.shoulderFollow,true,'Legacy presets must enable shoulder following');
for(const shoulderFollow of [false,true])assert.equal(fitDefaults(JSON.parse(JSON.stringify(fitDefaults({shoulderFollow})))).shoulderFollow,shoulderFollow,'Shoulder following did not survive preset reload');
for(const shoulderFollow of ['true',1,null,{}])assert.throws(()=>fitDefaults({shoulderFollow}),/shoulder following/);
setJointAngle(jointConfig,'Trousers','hipLX',20);assert.equal(jointConfig.angles.Trousers.hipRX,20);
setJointAngle(jointConfig,'Trousers','hipLX',0);assert.equal(jointConfig.angles.Trousers.hipRX,0);
setJointAngle(jointConfig,'Sleeve_R','elbowZ',-15);assert.equal(jointConfig.angles.Sleeve_L.elbowZ,-15);
setJointLink(jointConfig,'Trousers',false);setJointAngle(jointConfig,'Trousers','kneeRX',35);assert.equal(jointConfig.angles.Trousers.kneeLX,undefined);
assert.equal(fitDefaults(JSON.parse(JSON.stringify(jointConfig))).jointLinks.legs,false);
setJointLink(jointConfig,'Trousers',true);assert.equal(jointConfig.angles.Trousers.kneeRX,0);
const legacy=fitDefaults({angles:{Trousers:{hipLX:5,hipRX:12}}});assert.equal(legacy.angles.Trousers.hipRX,12);assert(jointsLinked(legacy,'Trousers'));
assert.throws(()=>fitDefaults({jointLinks:{arms:'yes'}}));assert.equal(outfitMinimum('vestChestWidth',0),-1);
const skinMeshes=[];scene.traverse(o=>{if(o.isMesh&&o.morphTargetDictionary?.bodyTransitionWidth!==undefined)skinMeshes.push(o);});
const transitionChecks=[];
for(const name of Object.keys(BODY_TRANSITION_CONTROLS))for(const value of [-1,1]){
 let motion=0,protectedDrift=0;const joins=new Map();
 for(const o of skinMeshes){
  const index=o.morphTargetDictionary[name];o.morphTargetInfluences[index]=value;
  for(let i=0;i<o.geometry.attributes.position.count;i++){
   const p=new T.Vector3().fromBufferAttribute(o.geometry.attributes.position,i).applyMatrix4(o.matrixWorld),q=o.getVertexPosition(i,new T.Vector3()).applyMatrix4(o.matrixWorld),d=p.distanceTo(q);motion=Math.max(motion,d);
   if(p.y>=.818||p.y<=.710||Math.abs(p.x+.004865)>=.105)protectedDrift=Math.max(protectedDrift,d);
   const key=p.toArray().map(x=>Math.round(x*1e6)).join(',');if(joins.has(key))assert(q.distanceTo(joins.get(key))<3e-6,'Body transition opened a material seam');else joins.set(key,q);
  }
  assert(o.geometry.morphAttributes.normal[index].array.every(Number.isFinite));o.morphTargetInfluences[index]=0;
 }
 assert(motion>.003,'Inactive body transition '+name);assert(protectedDrift<1e-6,'Transition moved protected head/arms/lower body');transitionChecks.push({name,value,motion,protectedDrift});
}
const settings={outfit:{},morphs:{},links:{},bodyFrame:{scale:1,offset:0},garmentFit:fitDefaults()};
const geometry=()=>[...fit.garments.values()].flatMap(p=>p.records.flatMap(r=>Array.from(r.g.attributes.position.array)));
const maxError=(a,b)=>a.reduce((x,v,i)=>Math.max(x,Math.abs(v-b[i])),0);
const boneSnapshot=()=>{scene.updateMatrixWorld(true);return [...bones.values()].flatMap(x=>x.bone.matrixWorld.toArray());};
const neutralBones=boneSnapshot();fit.apply(settings);const neutral=geometry();
assert(maxError(neutralBones,boneSnapshot())===0,'Clothing fit moved body joints');
let neutralSourceMaxError=0;
for(const g of OUTFIT_ASSEMBLIES)for(const name of g.parts){const toWorld=v=>new T.Vector3(v[0],v[2],-v[1]),shift=toWorld(report.clothing_segmentation.garments[g.root].original_rigid_placement).sub(toWorld(report.clothing_segmentation.garments[name].original_rigid_placement));for(const r of fit.garments.get(name).records)for(let i=0;i<r.base.count;i++){const expected=new T.Vector3().fromBufferAttribute(r.original,i).applyMatrix4(r.world).add(shift),actual=new T.Vector3().fromBufferAttribute(r.g.attributes.position,i).applyMatrix4(r.world);neutralSourceMaxError=Math.max(neutralSourceMaxError,expected.distanceTo(actual));}}
assert(neutralSourceMaxError<2e-6,'Neutral attachment reshaped original clothing');
let maxGap=0,samples=0;
function seamGap(){
 scene.updateMatrixWorld(true);scene.traverse(o=>{if(o.skeleton)o.skeleton.update();});
 for(const s of fit.seams)for(const [p,c]of s.pairs){const [r,i]=p.refs[0],a=r.o.getVertexPosition(i,new T.Vector3()).applyMatrix4(r.o.matrixWorld);for(const [q,j]of c.refs){const b=q.o.getVertexPosition(j,new T.Vector3()).applyMatrix4(q.o.matrixWorld);maxGap=Math.max(maxGap,a.distanceTo(b));}}
 samples++;
}
seamGap();
for(const [owner,controls]of Object.entries(ANGLES))for(const name of controls){
 settings.garmentFit.angles={[owner]:{[name]:20}};fit.apply(settings);
 assert(maxError(neutral,geometry())>1e-4,`Inactive clothing angle ${owner}.${name}`);assert(maxError(neutralBones,boneSnapshot())===0);seamGap();
}
settings.garmentFit=fitDefaults();fit.apply(settings);
const chestWidth=()=>{const x=fit.garments.get('Vest').nodes.filter(n=>n.source.y>.45&&n.source.y<.52).map(n=>n.current.x);return Math.max(...x)-Math.min(...x);};
const neutralChestWidth=chestWidth();setOutfitValue(settings,'Vest','vestChestWidth',-.5);fit.apply(settings);const narrowedChestWidth=chestWidth();
assert(narrowedChestWidth<neutralChestWidth-.01,'Negative chest width did not narrow the original chest');seamGap();
// Existing depth presets are losslessly expanded, including one-sided targets.
const depthOwners={vestChestDepth:'Vest',vestWaistDepth:'Vest',cuffDepth:'Cuff_L',trouserWaistDepth:'Trousers',trouserThighDepth:'Trousers',trouserCalfDepth:'Trousers',bootHeelDepth:'Boot_L',bootShaftDepth:'Boot_L'};
const depthChecks=[];
for(const name of DEPTH_CONTROLS){
 const owner=depthOwners[name];settings.outfit={};setOutfitValue(settings,owner,name,.3);fit.apply(settings);const oldFit=geometry(),[front,back]=depthNames(name);
 assert.equal(outfitValue(settings,owner,front),name==='bootHeelDepth'?0:.3);
 assert.equal(outfitValue(settings,owner,back),name==='vestChestDepth'?0:.3);
 setOutfitValue(settings,owner,front,outfitValue(settings,owner,front));setOutfitValue(settings,owner,back,outfitValue(settings,owner,back));fit.apply(settings);assert(maxError(oldFit,geometry())<1e-7,'Legacy depth changed: '+name);
 for(const side of [front,back]){const before=geometry();setOutfitValue(settings,owner,side,-.4);fit.apply(settings);assert(maxError(before,geometry())>1e-5,'Inactive depth '+side);assert(geometry().every(Number.isFinite));seamGap();}
 depthChecks.push(name);
}
settings.outfit={};fit.apply(settings);
const baseVest=fit.garments.get('Vest').nodes.map(n=>n.current.clone());
setOutfitValue(settings,'Vest','vestWaistFrontDepth',.5);fit.apply(settings);
let rearDepthDrift=0,frontDepthMotion=0;
for(const n of fit.garments.get('Vest').nodes){const error=n.current.distanceTo(baseVest[n.index]);if(depthFrontWeight('Vest',n.source)===0)rearDepthDrift=Math.max(rearDepthDrift,error);else frontDepthMotion=Math.max(frontDepthMotion,error);}
assert(rearDepthDrift<1e-7&&frontDepthMotion>1e-3,'Front depth moved the rear');
settings.outfit={};fit.apply(settings);let collarLowerDrift=0;
for(const name of Object.keys(COLLAR_CONTROLS)){
 settings.outfit={};setOutfitValue(settings,'Vest',name,-.5);fit.apply(settings);let collarMotion=0;
 for(const n of fit.garments.get('Vest').nodes){const error=n.current.distanceTo(baseVest[n.index]);if(n.source.y<=.52)collarLowerDrift=Math.max(collarLowerDrift,error);else collarMotion=Math.max(collarMotion,error);}
 assert(collarMotion>1e-3,'Inactive collar control '+name);seamGap();
}
assert(collarLowerDrift<1e-7,'Collar adjustment reached lower chest');
let shoulderProtectedDrift=0;const shoulderChecks=[];
for(const name of Object.keys(SHOULDER_CONTROLS))for(const value of [-1,1]){
 settings.outfit={};setOutfitValue(settings,'Sleeve_L',name,value);assert.equal(outfitValue(settings,'Vest',name),value);assert.equal(outfitValue(settings,'Sleeve_R',name),value);fit.apply(settings);let motion=0;
 for(const n of fit.garments.get('Vest').nodes){const d=n.current.distanceTo(baseVest[n.index]);motion=Math.max(motion,d);if(n.source.y<=.455||Math.abs(n.source.x)<=.03)shoulderProtectedDrift=Math.max(shoulderProtectedDrift,d);}
 assert(motion>1e-3,'Inactive sleeve attachment control '+name);seamGap();shoulderChecks.push(name+':'+value);
}
assert(shoulderProtectedDrift<1e-7,'Shoulder attachment moved collar center/lower vest');

// Fractional atlas edits must match scaling the same fitting control, including
// procedural collar/shoulder and front/back fields, before seam constraints.
for(const name of ['vestChestWidth','vestCollarWidth','vestArmholeRaise','vestChestBackDepth']){
 settings.outfit={};settings.uvRegions={};setOutfitValue(settings,'Vest',name,.6);fit.apply(settings);const expected=geometry(),native=fit.influence('Vest',name,settings).map(e=>Array.from(e.native));
 setOutfitValue(settings,'Vest',name,1);settings.uvRegions={['Vest:'+name]:{enabled:true,edits:[{type:'polygon',sheet:'Vest',points:[[0,0],[1,0],[1,1],[0,1]],feather:0,value:.6,opacity:1}]}};fit.apply(settings);
 assert(maxError(expected,geometry())<1e-7,'Fractional fitting field mismatch '+name);assert.deepEqual(fit.influence('Vest',name,settings).map(e=>Array.from(e.native)),native,'Authored heatmap depends on slider or mask');seamGap();
}
settings.outfit={};settings.uvRegions={};

// UV fields restrict real authored deltas and remain exact through reset/export.
const region={enabled:true,feather:.06,points:[[.1,.1],[.5,.1],[.5,.5],[.1,.5]]};
assert.equal(regionWeight(region,.3,.3),1);assert.equal(regionWeight(region,.9,.9),0);
assert(Math.abs(regionWeight(region,.53,.3)-.5)<1e-10);
assert(!polygonValid([[0,0],[1,1],[0,1],[1,0]]));
assert.throws(()=>validateRegions({'Vest:vestChestWidth':{...region,feather:NaN}}));
assert.throws(()=>validateRegions({'Vest:vestChestWidth':{...region,points:[[0,0],[0,0],[1,1]]}}));
settings.outfit={};setOutfitValue(settings,'Vest','vestChestWidth',.6);fit.apply(settings);const fullRegionFit=geometry();
settings.uvRegions={'Vest:vestChestWidth':region};fit.apply(settings);const limitedRegionFit=geometry();
assert(maxError(fullRegionFit,limitedRegionFit)>1e-4,'UV polygon did not restrict actual fitting');
fit.apply(settings);assert.equal(maxError(limitedRegionFit,geometry()),0,'UV edit accumulated');seamGap();
settings.uvRegions=validateRegions(JSON.parse(JSON.stringify(settings.uvRegions)));fit.apply(settings);assert.equal(maxError(limitedRegionFit,geometry()),0,'UV preset reload drift');
settings.uvRegions['Vest:vestChestWidth'].enabled=false;fit.apply(settings);assert.equal(maxError(fullRegionFit,geometry()),0,'Disabling UV region did not restore authored fit');
settings.uvRegions={'Vest:vestChestWidth':region,'Vest:vestCollarWidth':region,'Vest:vestArmholeRaise':region,'Trousers:trouserCalfBackDepth':region};
settings.outfit={};setOutfitValue(settings,'Vest','vestCollarWidth',-.3);setOutfitValue(settings,'Vest','vestCollarFrontDepth',.2);setOutfitValue(settings,'Vest','vestChestBackDepth',.25);setOutfitValue(settings,'Trousers','trouserCalfBackDepth',.15);
setOutfitValue(settings,'Vest','vestArmholeRaise',.6);setOutfitValue(settings,'Vest','vestArmholeFrontDepth',-.5);setOutfitValue(settings,'Vest','vestArmholeOpeningHeight',.4);
setOutfitValue(settings,'Trousers','trouserLength',.42);
assert.equal(outfitValue(settings,'Boot_L','bootShaftHeight'),.42);assert.equal(settings.outfit.Boot_R.bootShaftHeight,.42);
setOutfitValue(settings,'Boot_R','bootShaftHeight',.17);assert.equal(settings.outfit.Trousers.trouserLength,.17);assert.equal(outfitValue(settings,'Boot_L','bootShaftHeight'),.17);
setOutfitValue(settings,'Sleeve_L','vestRaise',.12);assert.equal(outfitValue(settings,'Vest','vestRaise'),.12);assert.equal(outfitValue(settings,'Sleeve_R','vestRaise'),.12);
setOutfitValue(settings,'Vest','vestShoulderWidth',.3);assert.equal(outfitValue(settings,'Sleeve_R','vestShoulderWidth'),.3);
setOutfitValue(settings,'Cuff_R','sleeveLength',.5);assert.equal(outfitValue(settings,'Sleeve_L','sleeveLength'),.5);
settings.links.sleeves=false;setOutfitValue(settings,'Cuff_R','sleeveLength',.2);assert.equal(outfitValue(settings,'Sleeve_L','sleeveLength'),.5);assert.equal(outfitValue(settings,'Sleeve_R','sleeveLength'),.2);
settings.garmentFit.scales={upper:.78,lower:1.5};settings.garmentFit.angles={Sleeve_L:{shoulderZ:25,elbowX:35},Sleeve_R:{shoulderY:-20,elbowZ:35},Trousers:{hipLX:20,kneeLX:40,hipRZ:15,kneeRX:30}};
setOutfitValue(settings,'Boot_L','bootWidth',.3);setOutfitValue(settings,'Boot_R','bootForward',.2);setOutfitValue(settings,'Vest','vestRaise',.2);setOutfitValue(settings,'Cuff_L','cuffSlide',.2);
fit.apply(settings);const changed=geometry();assert(maxError(changed,neutral)>.05);assert(changed.every(Number.isFinite));assert(maxError(neutralBones,boneSnapshot())===0);
fit.apply(settings);assert(maxError(changed,geometry())===0,'Accumulated fit on repeated apply');
const restored=JSON.parse(JSON.stringify(settings));restored.garmentFit=fitDefaults(restored.garmentFit);fit.apply(restored);assert(maxError(changed,geometry())===0,'Preset reload differs');
for(let i=0;i<=64;i++){player.seek(player.duration*i/64);seamGap();}
const shoulderMotionChecks=[];let shoulderHorizontalError=0;
for(const [scale,offset,shoulderHeight]of [[.78,-.01,-.03],[1.25,.05,.03],[1,0,0]]){
 player.beforeRestEdit();body(scale,offset,shoulderHeight);fit.capture();settings.bodyFrame={scale,offset,shoulderHeight};fit.apply(settings);player.afterRestEdit(scale);seamGap();
 player.reset();const aMatrices=boneSnapshot(),adjustedEdit=editingPose(scene,bones);
 assert(maxError(aMatrices,boneSnapshot())<1e-7,'Editing-pose calculation changed adjusted A-pose');
 for(const[n,{bone,quaternion}]of bones)bone.quaternion.copy(adjustedEdit.quaternion(n,'a')||quaternion);
 assert(maxError(aMatrices,boneSnapshot())<1e-7,'A-pose baseline moved adjusted joints');seamGap();
 for(const[n,{bone,quaternion}]of bones)bone.quaternion.copy(adjustedEdit.quaternion(n)||quaternion);
 scene.updateMatrixWorld(true);seamGap();
 for(const side of ['l','r'])for(const[a,b]of [['arm_stretch_','forearm_stretch_'],['forearm_stretch_','hand_']]){
  const direction=bones.get(b+side).bone.getWorldPosition(new T.Vector3()).sub(bones.get(a+side).bone.getWorldPosition(new T.Vector3())).normalize();
  shoulderHorizontalError=Math.max(shoulderHorizontalError,Math.hypot(direction.y,direction.z));assert(direction.x*(side==='l'?1:-1)>.999);
 }
 for(let i=0;i<=64;i++){player.seek(player.duration*i/64);seamGap();}
 player.reset();assert(maxError(aMatrices,boneSnapshot())<1e-7,'Running reset lost adjusted shoulder rest pose');
 shoulderMotionChecks.push({scale,offset,shoulderHeight,animationSamples:65});
}
assert(shoulderHorizontalError<1e-6,'Adjusted editing arms are not horizontal');
player.reset();settings.outfit={};settings.morphs={};settings.uvRegions={};settings.garmentFit=fitDefaults();fit.apply(settings);assert(maxError(neutral,geometry())<1e-7,'Reset drift');
for(const [o,a]of protectedGeometry)assert(maxError(a,Array.from(o.geometry.attributes.position.array))<1e-7,'Protected skin changed');
// Export with a raised shoulder pivot and non-default scale/offset. Recompute
// the editing pose from this rest frame, as the interactive editor does.
const exportBodyFrame={scale:1.13,offset:.025,shoulderHeight:.02};
player.beforeRestEdit();body(exportBodyFrame.scale,exportBodyFrame.offset,exportBodyFrame.shoulderHeight);fit.capture();restored.bodyFrame=exportBodyFrame;fit.apply(restored);player.afterRestEdit(exportBodyFrame.scale);
player.reset();const edit=editingPose(scene,bones),restMatrices=boneSnapshot();
for(const[n,{bone,quaternion}]of bones)bone.quaternion.copy(edit.quaternion(n)||quaternion);
seamGap();let horizontalError=0;
for(const side of ['l','r'])for(const[a,b]of [['arm_stretch_','forearm_stretch_'],['forearm_stretch_','hand_']]){
 const direction=bones.get(b+side).bone.getWorldPosition(new T.Vector3()).sub(bones.get(a+side).bone.getWorldPosition(new T.Vector3())).normalize();
 horizontalError=Math.max(horizontalError,Math.hypot(direction.y,direction.z));assert(direction.x*(side==='l'?1:-1)>.999);
}
assert(horizontalError<1e-6,'Editing arms are not horizontal');
const tMatrices=boneSnapshot();player.seek(player.duration*.43);seamGap();player.reset();
assert(maxError(restMatrices,boneSnapshot())<1e-7,'Editing baseline entered the animation rest capture');
for(const[n,{bone,quaternion}]of bones)bone.quaternion.copy(edit.quaternion(n)||quaternion);
assert(maxError(tMatrices,boneSnapshot())<1e-7,'T-pose reset drift');seamGap();
const worldPositions=tree=>{const result={};tree.updateMatrixWorld(true);tree.traverse(o=>{if(o.skeleton)o.skeleton.update();});tree.traverse(o=>{if(!o.isMesh)return;let p=o;while(p&&!p.userData.part_type)p=p.parent;const n=p?.name||o.name;result[n]??=[];for(let i=0;i<o.geometry.attributes.position.count;i++)result[n].push(o.getVertexPosition(i,new T.Vector3()).applyMatrix4(o.matrixWorld).toArray());});return result;};
for(const o of skinMeshes)o.morphTargetInfluences[o.morphTargetDictionary.bodyTransitionFrontDepth]=.65;
const beforeExport=worldPositions(scene);
globalThis.FileReader=class{readAsArrayBuffer(blob){blob.arrayBuffer().then(x=>{this.result=x;this.onloadend?.();});}readAsDataURL(blob){blob.arrayBuffer().then(x=>{this.result='data:application/octet-stream;base64,'+Buffer.from(x).toString('base64');this.onloadend?.();});}};
const restoreExport=fit.prepareExport();assert.deepEqual(scene.userData.clothingSettings.uvRegions,restored.uvRegions,'Export lost UV settings');for(const p of fit.garments.values())for(const r of p.records)assert.equal(Object.keys(r.g.morphAttributes).length,0);const binary=await new GLTFExporter().parseAsync(scene,{binary:true,trs:true,onlyVisible:false});
fs.writeFileSync(process.argv[4].replace('.json','_export.glb'),Buffer.from(binary));
const imported=await new GLTFLoader().parseAsync(binary,''),afterExport=worldPositions(imported.scene);let exportMaxError=0;
for(const [name,points]of Object.entries(beforeExport)){assert(afterExport[name],name);assert.equal(points.length,afterExport[name].length);for(let i=0;i<points.length;i++)exportMaxError=Math.max(exportMaxError,...points[i].map((x,k)=>Math.abs(x-afterExport[name][i][k])));}
assert(exportMaxError<2e-6,`Export changed fitted pose: ${exportMaxError}`);
restoreExport();player.reset();player.beforeRestEdit();body(1,0,0);fit.capture();settings.bodyFrame={scale:1,offset:0,shoulderHeight:0};fit.apply(settings);player.afterRestEdit(1);assert(maxError(neutral,geometry())<1e-7);
assert(maxGap<2e-6,`Seam gap ${maxGap}`);
const result={status:'passed',horizontalError,shoulderHorizontalError,shoulderMotionChecks,exportBodyFrame,exportPose:'T-pose',transitionChecks,shoulderChecks,shoulderProtectedDrift,seams:fit.seams.map(s=>({parent:s.parent.name,child:s.child.name,samples:s.pairs.length})),animationSamples:65,angleControlsTested:Object.values(ANGLES).flat().length,checkedPoses:samples,neutralChestWidth,narrowedChestWidth,depthChecks,rearDepthDrift,collarLowerDrift,maxGap,exportMaxError,neutralSourceMaxError,checks:['UV field, polygon validation, actual restriction, disable/reset, persistence and export','collar regional controls','front/back depth independence','legacy depth preservation','negative chest width','default bilateral joints','independent joints','joint-link persistence','legacy asymmetric angle preservation','original source seam correspondence','shared parent/child values','unlinked sides','shared scale','clothing-only joint angles','body rig preservation','preset round trip','repeat fit','running seam continuity','body placement','shoulder height through A/T and running','reset','nonzero shoulder export bake/restore']};
fs.writeFileSync(process.argv[4],JSON.stringify(result,null,2));console.log(JSON.stringify(result,null,2));
