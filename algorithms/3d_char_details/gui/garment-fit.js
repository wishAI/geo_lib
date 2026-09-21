import * as T from 'three';

export const OUTFIT_ASSEMBLIES=[
 {id:'upper',label:'Upper outfit',root:'Vest',parts:['Vest','Sleeve_L','Sleeve_R','Cuff_L','Cuff_R']},
 {id:'lower',label:'Lower outfit',root:'Trousers',parts:['Trousers','Boot_L','Boot_R']}
];
export const PARENT={Sleeve_L:'Vest',Sleeve_R:'Vest',Cuff_L:'Sleeve_L',Cuff_R:'Sleeve_R',Boot_L:'Trousers',Boot_R:'Trousers'};
export const assemblyOf=part=>OUTFIT_ASSEMBLIES.find(g=>g.parts.includes(part));
export const ANGLES={
 Sleeve_L:['shoulderX','shoulderY','shoulderZ','elbowX','elbowY','elbowZ'],
 Sleeve_R:['shoulderX','shoulderY','shoulderZ','elbowX','elbowY','elbowZ'],
 Trousers:['hipLX','hipLY','hipLZ','kneeLX','kneeLY','kneeLZ','hipRX','hipRY','hipRZ','kneeRX','kneeRY','kneeRZ']
};
export const DEPTH_CONTROLS=['vestChestDepth','vestWaistDepth','cuffDepth','trouserWaistDepth','trouserThighDepth','trouserCalfDepth','bootHeelDepth','bootShaftDepth'];
export const COLLAR_CONTROLS={vestCollarWidth:'Neckline width',vestCollarFrontDepth:'Neckline front depth',vestCollarBackDepth:'Neckline back depth',vestCollarHeight:'Collar height'};
export const SHOULDER_CONTROLS={vestArmholeSpread:'Attachment outward',vestArmholeRaise:'Attachment height',vestArmholeFrontDepth:'Attachment front depth',vestArmholeBackDepth:'Attachment back depth',vestArmholeOpeningHeight:'Opening height'};
export const depthNames=name=>DEPTH_CONTROLS.includes(name)?['Front','Back'].map(side=>name.replace(/Depth$/,side+'Depth')):[name];
export function splitDepth(name){const match=name.match(/(Front|Back)Depth$/);if(!match)return null;const base=name.replace(/(Front|Back)Depth$/,'Depth');return DEPTH_CONTROLS.includes(base)?{base,side:match[1]}:null;}
// Width/depth/room targets are signed offsets from the authored neutral.
export const outfitMinimum=(name,authoredMin=0)=>(/(Width|Depth|Room)$/.test(name)||name in COLLAR_CONTROLS||name in SHOULDER_CONTROLS)?-1:authoredMin;
export const jointGroup=part=>part==='Trousers'?'legs':'arms';
export const jointsLinked=(cfg,part)=>cfg.jointLinks?.[jointGroup(part)]!==false;
export function jointPeer(part,name){return part==='Trousers'?[part,name.replace(/([LR])([XYZ])$/,(_,side,axis)=>(side==='L'?'R':'L')+axis)]:[part==='Sleeve_L'?'Sleeve_R':'Sleeve_L',name];}
export function setJointAngle(cfg,part,name,value){
 if(!ANGLES[part]?.includes(name)||!Number.isFinite(value)||Math.abs(value)>90)throw new Error('Invalid clothing joint angle');
 const targets=[[part,name]];if(jointsLinked(cfg,part))targets.push(jointPeer(part,name));
 for(const [p,n]of targets)cfg.angles[p]={...cfg.angles[p],[n]:value};
}
export function setJointLink(cfg,part,linked){
 cfg.jointLinks??={arms:true,legs:true};cfg.jointLinks[jointGroup(part)]=linked;
 if(linked)for(const n of ANGLES[part].filter(n=>part!=='Trousers'||!n.includes('R')))setJointAngle(cfg,part,n,cfg.angles[part]?.[n]||0);
}
export const angleLabel=n=>n.replace(/([A-Z])/g,' $1').replace(/^./,c=>c.toUpperCase())+' °';
export function inheritedControls(part){
 if(part.startsWith('Sleeve'))return ['vestRaise','vestForward','vestShoulderWidth',...Object.keys(SHOULDER_CONTROLS)];
 if(part.startsWith('Cuff'))return ['sleeveRaise','sleeveForward','sleeveSpread','sleeveLength','sleeveForearmRoom'];
 if(part.startsWith('Boot'))return ['trouserRaise','trouserForward','bootShaftWidth','bootShaftDepth','bootShaftHeight'];
 return [];
}
export function sharedOwner(part,name){
 const split=splitDepth(name);if(split){const [owner,base]=sharedOwner(part,split.base);return [owner,base.replace(/Depth$/,split.side+'Depth')];}
 if(part.startsWith('Sleeve')&&inheritedControls(part).includes(name))return ['Vest',name];
 if(part.startsWith('Cuff')&&inheritedControls(part).includes(name))return ['Sleeve_'+part.slice(-1),name];
 const hem={bootShaftWidth:'trouserCalfRoom',bootShaftDepth:'trouserCalfDepth',bootShaftHeight:'trouserLength'};
 if(part.startsWith('Boot')&&inheritedControls(part).includes(name))return ['Trousers',hem[name]||name];
 return [part,name];
}
export function fitDefaults(v={}){
 if(!v||typeof v!=='object'||Array.isArray(v))throw new Error('Invalid clothing fit');
 const out={version:1,scales:{upper:1,lower:1},angles:{},unlocked:{},jointLinks:{arms:true,legs:true},shoulderFollow:true};
 if(v.shoulderFollow!==undefined){if(typeof v.shoulderFollow!=='boolean')throw new Error('Invalid shoulder following');out.shoulderFollow=v.shoulderFollow;}
 if(v.version!==undefined&&v.version!==1)throw new Error('Unsupported clothing fit version');
 for(const [k,x]of Object.entries(v.scales||{})){if(!['upper','lower'].includes(k)||!Number.isFinite(x)||x<.5||x>2)throw new Error('Invalid outfit scale');out.scales[k]=x;}
 for(const [p,angles]of Object.entries(v.angles||{})){
  if(!ANGLES[p]||!angles||typeof angles!=='object')throw new Error('Invalid garment angle owner');out.angles[p]={};
  for(const [k,x]of Object.entries(angles)){if(!ANGLES[p].includes(k)||!Number.isFinite(x)||Math.abs(x)>90)throw new Error('Invalid garment angle');out.angles[p][k]=x;}
 }
 for(const [k,x]of Object.entries(v.jointLinks||{})){if(!['arms','legs'].includes(k)||typeof x!=='boolean')throw new Error('Invalid clothing joint link');out.jointLinks[k]=x;}
 for(const [p,x]of Object.entries(v.unlocked||{})){if(!PARENT[p]||typeof x!=='boolean')throw new Error('Invalid shared-control lock');out.unlocked[p]=x;}
 return out;
}
export function outfitValue(settings,part,name){
 const [owner,key]=sharedOwner(part,name),explicit=settings.outfit?.[owner]?.[key]??settings.morphs?.[key];if(explicit!==undefined)return explicit;
 const split=splitDepth(key);if(!split)return 0;
 // Original chest depth only moved the front; heel depth only moved the rear.
 // The other depth targets affected both sides. Preserve every old fit exactly.
 if((split.base==='vestChestDepth'&&split.side==='Back')||(split.base==='bootHeelDepth'&&split.side==='Front'))return 0;
 return settings.outfit?.[owner]?.[split.base]??settings.morphs?.[split.base]??0;
}
export function setOutfitValue(settings,part,name,value){
 const [owner,key]=sharedOwner(part,name),pair=owner.startsWith('Sleeve')?'sleeves':owner.startsWith('Cuff')?'cuffs':owner.startsWith('Boot')?'boots':null;
 const targets=pair&&settings.links?.[pair]!==false?[owner.slice(0,-1)+'L',owner.slice(0,-1)+'R']:[owner];
 settings.outfit??={};for(const p of targets)settings.outfit[p]={...settings.outfit[p],[key]:value};
 // One authoritative value, with explicit mirrors for saved snapshots and UI.
 for(const group of OUTFIT_ASSEMBLIES)for(const p of group.parts)for(const alias of inheritedControls(p).flatMap(n=>[n,...depthNames(n)])){
  const [op,ok]=sharedOwner(p,alias);if(op!==p&&targets.includes(op)&&ok===key)settings.outfit[p]={...settings.outfit[p],[alias]:value};
 }
}

// Landmarks read from the original source.usdc rig through Blender MCP.
// These are clothing rest joints, not the differently proportioned FBX rig.
const JOINTS={shoulder:[.06394642,.52382362,-.00994237],elbow:[.11979874,.45279932,-.01572367],wrist:[.17565103,.38177502,.00262408],hip:[.059407,.29964218,.01291146],knee:[.06185648,.17097059,.02826263],ankle:[.05452706,.04250674,.01084183]};
const v3=a=>new T.Vector3(...a),smooth=(a,b,x)=>{const t=T.MathUtils.clamp((x-a)/(b-a),0,1);return t*t*(3-2*t);};
const hash=p=>[p.x,p.y,p.z].map(x=>Math.round(x*1e5)).join(',');
const xyz=(a,i)=>new T.Vector3().fromBufferAttribute(a,i);

// Transfer only across the upper shoulder cap. Below it, the axilla's arm and
// torso surfaces are close together: nearest-face transfer can switch between
// them on adjacent garment vertices and turn tiny edges into spikes. Retain
// the continuous authored underarm weights instead. Coordinates follow the
// body frame so changing body scale/offset does not move the transition band.
export function shoulderFollowWeight(p,frame={scale:1,offset:0}){
 const scale=frame.scale??1,x=Math.abs((p.x+.004865-(frame.offset||0))/scale),y=.806+(p.y-.806)/scale;
 return smooth(.03,.05,x)*(1-smooth(.16,.20,x))*smooth(.70,.73,y)*(1-smooth(.79,.815,y));
}

function shoulderSkin(parts){
 const body=[];
 parts.get('Body_Complete')?.object.traverse(o=>{if(o.isSkinnedMesh)body.push({o,world:o.matrixWorld.clone()});});
 const closest=new T.Vector3(),bary=new T.Vector3();
 return (garments,frame)=>{
  const triangles=[];
  for(const {o,world}of body){
   const g=o.geometry,base=g.attributes.position,idx=g.index;
   // Read morph-deformed REST vertices. getVertexPosition() would also apply
   // the current pose and make editing/scrubbing change the fitted weights.
   const active=(o.morphTargetInfluences||[]).flatMap((w,k)=>w&&g.morphAttributes.position?.[k]?[[w,g.morphAttributes.position[k]]]:[]);
   const points=Array.from({length:base.count},(_,i)=>{
    const p=xyz(base,i);for(const [w,a]of active){const d=xyz(a,i);if(!g.morphTargetsRelative)d.sub(xyz(base,i));p.addScaledVector(d,w);}return p.applyMatrix4(world);
   });
   for(let i=0;i<(idx?.count??base.count);i+=3){
    const ids=[0,1,2].map(k=>idx?idx.getX(i+k):i+k),abc=ids.map(j=>points[j]);
    // Source search is deliberately broader than the target blend; otherwise
    // narrowing the blend would exclude the actual closest torso triangle.
    if(!abc.some(p=>{const y=.806+(p.y-.806)/frame.scale,x=Math.abs((p.x+.004865-(frame.offset||0))/frame.scale);return y>.57&&y<.84&&x<.25;}))continue;
    const triangle=new T.Triangle(...abc),box=new T.Box3().setFromPoints(abc);
    if(triangle.getArea()>1e-12)triangles.push({o,ids,triangle,box});
   }
  }
  if(!triangles.length)return;
  for(const name of ['Vest','Sleeve_L','Sleeve_R'])for(const node of garments.get(name).nodes){
   const amount=shoulderFollowWeight(node.current,frame);if(!amount)continue;
   let best=null,distance=Infinity;
   for(const t of triangles){
    // Bounds rejection avoids most triangle queries without a persistent
    // spatial cache that could go stale after a body or clothing shape edit.
    const p=node.current,b=t.box,dx=Math.max(b.min.x-p.x,0,p.x-b.max.x),dy=Math.max(b.min.y-p.y,0,p.y-b.max.y),dz=Math.max(b.min.z-p.z,0,p.z-b.max.z);
    if(dx*dx+dy*dy+dz*dz>=distance)continue;
    t.triangle.closestPointToPoint(p,closest);const d=closest.distanceToSquared(p);
    if(d<distance){distance=d;best=t;}
   }
   if(!best)continue;
   best.triangle.closestPointToPoint(node.current,closest);best.triangle.getBarycoord(closest,bary);
   const target=new Map(),g=best.o.geometry;
   for(let k=0;k<3;k++)for(let j=0;j<4;j++){
    const bone=best.o.skeleton.bones[g.attributes.skinIndex.getComponent(best.ids[k],j)].name;
    target.set(bone,(target.get(bone)||0)+Math.max(0,bary.getComponent(k))*g.attributes.skinWeight.getComponent(best.ids[k],j));
   }
   const [first,index]=node.refs[0],weights=new Map();
   for(let j=0;j<4;j++){const bone=first.o.skeleton.bones[first.skin.getComponent(index,j)].name;weights.set(bone,(weights.get(bone)||0)+(1-amount)*first.weights.getComponent(index,j));}
   for(const [bone,w]of target)weights.set(bone,(weights.get(bone)||0)+amount*w);
   const top=[...weights].filter(([,w])=>w>1e-8).sort((a,b)=>b[1]-a[1]).slice(0,4),total=top.reduce((s,[,w])=>s+w,0);
   for(const [r,i]of node.refs)for(let j=0;j<4;j++){
    r.g.attributes.skinIndex.setComponent(i,j,top[j]?r.o.skeleton.bones.findIndex(b=>b.name===top[j][0]):0);
    r.g.attributes.skinWeight.setComponent(i,j,top[j]?top[j][1]/total:0);
   }
  }
 };
}

// Source-space front is +Z after the glTF axis conversion. These fields affect
// clothing only and fade into the existing sculpt; they do not replace geometry.
export function depthFrontWeight(part,p){
 let center=part==='Vest'?-.005:.01;
 if(part.startsWith('Cuff')){const a=v3(JOINTS.shoulder),b=v3(JOINTS.wrist);a.x*=Math.sign(p.x)||1;b.x*=Math.sign(p.x)||1;const axis=b.clone().sub(a),t=T.MathUtils.clamp(p.clone().sub(a).dot(axis)/axis.lengthSq(),0,1);center=a.lerp(b,t).z;}
 return smooth(center-.005,center+.005,p.z);
}
export function collarDelta(source,name){
 const w=smooth(.52,.552,source.y)*(1-smooth(.055,.085,Math.abs(source.x))),d=new T.Vector3();
 if(name==='vestCollarWidth')d.x=source.x*.65*w;
 if(name==='vestCollarHeight')d.y=.035*w;
 if(name==='vestCollarFrontDepth')d.z=.03*w*smooth(-.005,.012,source.z);
 if(name==='vestCollarBackDepth')d.z=-.03*w*(1-smooth(-.018,-.005,source.z));
 return d;
}
export function shoulderDelta(source,name){
 const w=smooth(.455,.505,source.y)*(1-smooth(.553,.583,source.y))*smooth(.03,.066,Math.abs(source.x))*(1-smooth(.11,.14,Math.abs(source.x))),front=depthFrontWeight('Vest',source),d=new T.Vector3();
 if(name==='vestArmholeSpread')d.x=.025*Math.sign(source.x)*w;
 if(name==='vestArmholeRaise')d.y=.02*w;
 if(name==='vestArmholeFrontDepth')d.z=.02*w*front;
 if(name==='vestArmholeBackDepth')d.z=-.02*w*(1-front);
 if(name==='vestArmholeOpeningHeight')d.y=(source.y-.516)*.6*w;
 return d;
}
export function garmentFit(model,parts,report){
 model.updateMatrixWorld(true);
 const garments=new Map(),records=[],seams=[];
 const followShoulders=shoulderSkin(parts);
 const placements=Object.fromEntries(OUTFIT_ASSEMBLIES.flatMap(g=>g.parts).map(n=>{
  const p=report.clothing_segmentation.garments[n].original_rigid_placement;return [n,v3([p[0],p[2],-p[1]])];
 }));
 for(const group of OUTFIT_ASSEMBLIES)for(const name of group.parts){
  const part={name,group,nodes:[],lookup:new Map(),records:[]};garments.set(name,part);
  parts.get(name).object.traverse(o=>{
   if(!o.isMesh)return;
   const g=o.geometry;const r={o,g,part,world:o.matrixWorld.clone(),inverse:o.matrixWorld.clone().invert(),original:g.attributes.position.clone(),normal:g.attributes.normal?.clone(),skin:g.attributes.skinIndex?.clone(),weights:g.attributes.skinWeight?.clone(),base:g.attributes.position.clone(),morph:(g.morphAttributes.position||[]).map(a=>a.clone()),nodes:[]};
   part.records.push(r);records.push(r);
   for(let i=0;i<r.original.count;i++){
    const p=xyz(r.original,i).applyMatrix4(r.world),source=p.clone().sub(placements[name]),key=hash(source);
    let node=part.lookup.get(key);
    if(!node){node={source,refs:[],edges:new Set(),index:part.nodes.length};part.lookup.set(key,node);part.nodes.push(node);}
    node.refs.push([r,i]);r.nodes.push(node);
   }
   const idx=g.index;
   for(let i=0;i<(idx?.count??r.original.count);i+=3){const ns=[0,1,2].map(k=>r.nodes[idx?idx.getX(i+k):i+k]);for(let k=0;k<3;k++){ns[k].edges.add(ns[(k+1)%3]);ns[(k+1)%3].edges.add(ns[k]);}}
  });
 }
 // Match original surface coordinates, never current proximity or bone ownership.
 for(const [child,parent]of Object.entries(PARENT)){
  const c=garments.get(child),p=garments.get(parent),pairs=[];
  for(const n of c.nodes){let match=p.lookup.get(hash(n.source));if(!match){let best=2e-6;for(const q of p.nodes){const d=q.source.distanceTo(n.source);if(d<best){best=d;match=q;}}}if(match&&match.source.distanceTo(n.source)<2e-6)pairs.push([match,n]);}
  if(!pairs.length)throw new Error('Missing original clothing seam: '+parent+' → '+child);
  const pinned=new Set(pairs.map(([,n])=>n));
  // A graph band (not spatial nearest-neighbour projection) confines attachment
  // corrections to the connected surface around the seam; distal design stays free.
  let frontier=[...pinned];for(const n of c.nodes)n.distance=Infinity;for(const n of frontier)n.distance=0;
  while(frontier.length){const next=[];for(const n of frontier)for(const q of n.edges)if(q.distance>n.distance+1){q.distance=n.distance+1;next.push(q);}frontier=next;}
  seams.push({parent:p,child:c,pairs,pinned});
 }
 let lastSettings;
 const capture=()=>{for(const r of records){r.base=r.g.attributes.position.clone();r.morph=(r.g.morphAttributes.position||[]).map(a=>a.clone());}};
 function apply(settings){
  lastSettings=settings;const cfg=settings.garmentFit||fitDefaults(),frame=settings.bodyFrame||{scale:1,offset:0};
  const map=p=>p.clone().sub(v3([-.004865,.806,-.032925])).multiplyScalar(frame.scale).add(v3([-.004865+frame.offset,.806,-.032925]));
  const groupPivots={upper:map(v3([0,.578,0]).add(placements.Vest)),lower:map(v3([0,.365,0]).add(placements.Trousers))};
  function rotate(p,part,source){
   const sign=source.x>=0?1:-1,side=sign>0?'L':'R',arm=/Sleeve|Cuff/.test(part.name),leg=/Trousers|Boot/.test(part.name);
   if(!arm&&!leg)return p;
   const owner=arm?'Sleeve_'+side:'Trousers',a=cfg.angles[owner]||{},prox=arm?'shoulder':'hip',dist=arm?'elbow':'knee';
   const joint=k=>{const q=v3(JOINTS[k]);q.x*=sign;return map(q.add(placements[part.group.root]));};
   const quaternion=k=>new T.Quaternion().setFromEuler(new T.Euler(...['X','Y','Z'].map(axis=>{
    // Mirrored arms use anatomical signs: equal values move both sleeves together.
    const mirror=sign<0&&axis!=='X'?-1:1;return (a[k+(arm?'':side)+axis]||0)*Math.PI/180*mirror;
   }),'XYZ'));
   const d=joint(dist),e=joint(arm?'wrist':'ankle'),axis=e.clone().sub(d).normalize();
   const w=smooth(-.035*frame.scale,.025*frame.scale,p.clone().sub(d).dot(axis));
   const distal=p.clone().sub(d).applyQuaternion(quaternion(dist)).add(d);p.lerp(distal,w);
   const h=joint(prox),q=p.clone().sub(h).applyQuaternion(quaternion(prox)).add(h);
   return p.lerp(q,arm?1:smooth(0,.04,Math.abs(source.x)));
  }
  for(const r of records){
   const {part,g}=r,scale=cfg.scales[part.group.id],shift=placements[part.group.root].clone().sub(placements[part.name]).multiplyScalar(frame.scale);
   g.attributes.skinIndex.copy(r.skin);g.attributes.skinWeight.copy(r.weights);
   const rootMove=new T.Vector3(0,(settings.outfit?.[part.group.root]?.[part.group.root==='Vest'?'vestRaise':'trouserRaise']??settings.morphs?.[part.group.root==='Vest'?'vestRaise':'trouserRaise']??0)*(part.group.root==='Vest'?.075:.12)*frame.scale,(settings.outfit?.[part.group.root]?.[part.group.root==='Vest'?'vestForward':'trouserForward']??settings.morphs?.[part.group.root==='Vest'?'vestForward':'trouserForward']??0)*.07*frame.scale);
   for(let i=0;i<r.base.count;i++){
    const base=xyz(r.base,i).applyMatrix4(r.world).add(shift),p=base.clone();
    for(const [name,k]of Object.entries(r.o.morphTargetDictionary||{})){
     // Root translations are applied to the entire assembly after rotation.
     if(part.name===part.group.root&&/(Raise|Forward)$/.test(name))continue;
     // Child aliases are driven by the parent surface, not a second, unrelated morph.
     if(sharedOwner(part.name,name)[0]!==part.name)continue;
     if(!r.morph[k])continue;
     const delta=xyz(r.morph[k],i);if(!g.morphTargetsRelative)delta.sub(xyz(r.base,i));delta.applyMatrix3(new T.Matrix3().setFromMatrix4(r.world));
     if(DEPTH_CONTROLS.includes(name)){
      const [frontName,backName]=depthNames(name),front=outfitValue(settings,part.name,frontName),back=outfitValue(settings,part.name,backName),source=r.nodes[i].source;
      if(name==='vestChestDepth'){
       p.addScaledVector(delta,front);
       p.z-=back*.04*Math.exp(-Math.pow((source.y-.49)/.075,2))* (1-smooth(-.03,-.02,source.z))*frame.scale;
      }else if(name==='bootHeelDepth'){
       p.addScaledVector(delta,back);
       p.z+=front*.035*smooth(0,.025,source.z)*Math.exp(-Math.pow(source.z/.05,2))*(1-smooth(.065,.11,source.y))*frame.scale;
      }else{const w=depthFrontWeight(part.name,source);p.addScaledVector(delta,w*front+(1-w)*back);}
     }else p.addScaledVector(delta,outfitValue(settings,part.name,name));
    }
    if(part.name==='Vest')for(const name of Object.keys(SHOULDER_CONTROLS)){const value=outfitValue(settings,part.name,name);if(value)p.addScaledVector(shoulderDelta(r.nodes[i].source,name),value*frame.scale);}
    if(part.name==='Vest')for(const name of Object.keys(COLLAR_CONTROLS)){const value=outfitValue(settings,part.name,name);if(value)p.addScaledVector(collarDelta(r.nodes[i].source,name),value*frame.scale);}

    const transform=p=>rotate(p,part,r.nodes[i].source).sub(groupPivots[part.group.id]).multiplyScalar(scale).add(groupPivots[part.group.id]).add(rootMove);
    const node=r.nodes[i];if(node.refs[0][0]===r&&node.refs[0][1]===i){node.base=transform(base);node.current=transform(p);}
   }
   r.o.morphTargetInfluences?.fill(0);
  }
  for(const s of seams){
   // Parent displacement carries its child. Child-only edits remain free away
   // from the seam; changing a child never pushes geometry into its parent.
   const inherited=new T.Vector3();for(const [p]of s.pairs)inherited.add(p.current.clone().sub(p.base));inherited.multiplyScalar(1/s.pairs.length);
   for(const n of s.child.nodes){n.current.add(inherited);n.correction=new T.Vector3();}
   for(const [p,c]of s.pairs)c.correction.copy(p.current).sub(c.current);
   for(let it=0;it<64;it++){
    const next=s.child.nodes.map(n=>{
     if(s.pinned.has(n)||n.distance>14)return n.correction;
     const d=new T.Vector3();let total=0;for(const q of n.edges){const w=1/Math.max(1e-5,n.source.distanceTo(q.source));d.addScaledVector(q.correction,w);total+=w;}
     return d.multiplyScalar(1/(total*1.035||1));
    });for(let i=0;i<next.length;i++)s.child.nodes[i].correction=next[i];
   }
   for(const n of s.child.nodes)n.current.add(n.correction);
   for(const [p,c]of s.pairs){c.current.copy(p.current);const [pr,pi]=p.refs[0];for(const [r,i]of c.refs){for(let k=0;k<4;k++){r.g.attributes.skinIndex.setComponent(i,k,pr.g.attributes.skinIndex.getComponent(pi,k));r.g.attributes.skinWeight.setComponent(i,k,pr.g.attributes.skinWeight.getComponent(pi,k));}}}
  }
  if(cfg.shoulderFollow!==false)followShoulders(garments,frame);
  // Body transfer must not override the exact parent/child seam contract.
  for(const s of seams)for(const [p,c]of s.pairs){const [pr,pi]=p.refs[0];for(const [r,i]of c.refs)for(let k=0;k<4;k++){r.g.attributes.skinIndex.setComponent(i,k,pr.g.attributes.skinIndex.getComponent(pi,k));r.g.attributes.skinWeight.setComponent(i,k,pr.g.attributes.skinWeight.getComponent(pi,k));}}
  for(const r of records){
   for(let i=0;i<r.base.count;i++){const p=r.nodes[i].current.clone().applyMatrix4(r.inverse);r.g.attributes.position.setXYZ(i,p.x,p.y,p.z);}
   updateNormals(r);r.g.attributes.position.needsUpdate=true;r.g.attributes.skinIndex.needsUpdate=true;r.g.attributes.skinWeight.needsUpdate=true;r.g.computeBoundingBox();r.g.computeBoundingSphere();
  }
  model.userData.garmentFit=JSON.parse(JSON.stringify(cfg));
 }
 function updateNormals(r){
  if(!r.normal)return;
  // Rotate the original split normals by the local surface-normal change, so
  // authored creases survive and position/UV duplicates don't acquire cracks.
  const before=Array.from({length:r.base.count},()=>new T.Vector3()),after=before.map(()=>new T.Vector3()),idx=r.g.index;
  for(let j=0;j<(idx?.count??r.base.count);j+=3){const ids=[0,1,2].map(k=>idx?idx.getX(j+k):j+k);for(const [attr,dest]of [[r.original,before],[r.g.attributes.position,after]]){const [a,b,c]=ids.map(i=>xyz(attr,i)),n=b.sub(a).cross(c.sub(a));for(const i of ids)dest[i].add(n);}}
  for(let i=0;i<r.base.count;i++){const a=before[i],b=after[i],n=xyz(r.normal,i);if(a.lengthSq()>1e-16&&b.lengthSq()>1e-16)n.applyQuaternion(new T.Quaternion().setFromUnitVectors(a.normalize(),b.normalize()));r.g.attributes.normal.setXYZ(i,n.x,n.y,n.z);}r.g.attributes.normal.needsUpdate=true;
 }
 return {apply,capture,seams,garments,
  // Garment fitting is baked in exported positions; facial morphs remain live.
  prepareExport(){const saved=records.map(r=>[r,r.g.morphAttributes,r.o.morphTargetInfluences,r.o.morphTargetDictionary]);for(const [r]of saved){r.g.morphAttributes={};r.o.morphTargetInfluences=[];r.o.morphTargetDictionary={};}model.userData.clothingSettings=JSON.parse(JSON.stringify({outfit:lastSettings?.outfit,garmentFit:lastSettings?.garmentFit}));return ()=>{for(const [r,a,i,d]of saved){r.g.morphAttributes=a;r.o.morphTargetInfluences=i;r.o.morphTargetDictionary=d;}};}
 };
}
