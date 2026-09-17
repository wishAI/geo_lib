import * as THREE from 'three';

export const HAND_NAMES = ['wrist', 'thumbKnuckle', 'thumbIntermediateBase', 'thumbIntermediateTip', 'thumbTip',
  ...['index','middle','ring','little'].flatMap(f=>[`${f}Metacarpal`,`${f}Knuckle`,`${f}IntermediateBase`,`${f}IntermediateTip`,`${f}Tip`]),
  'forearmWrist','forearmArm'];
export const HAND_EDGES = [[0,1],[1,2],[2,3],[3,4],
  ...[5,10,15,20].flatMap(base=>[[0,base],[base,base+1],[base+1,base+2],[base+2,base+3],[base+3,base+4]]),[0,25],[25,26]];
const COLORS = {head:0xeab342,left:0x009ca8,right:0xc74286};

function matrix(value) {
  if(!Array.isArray(value)||value.length!==4||value.some(row=>!Array.isArray(row)||row.length!==4||row.some(v=>!Number.isFinite(v))))return null;
  if(value[3].some((v,i)=>Math.abs(v-(i===3?1:0))>1e-5))return null;
  const result=new THREE.Matrix4().set(...value.flat());
  return Math.abs(result.determinant())>1e-8?result:null;
}

// Uses the exact pretransform exported by the Python retargeter. No synthetic
// elbow/shoulder/leg markers are derived from solved robot joints here.
export function trackingRecords(payload, transform) {
  const records=[], connections=[], byName=new Map();
  function add(name,side,value){
    const raw=matrix(value);if(!raw)return;
    const mapped=transform.clone().multiply(raw);
    const item={name,side,raw,mapped,position:new THREE.Vector3().setFromMatrixPosition(mapped)};
    byName.set(name,records.length);records.push(item);
  }
  add('head','head',payload?.head);
  for(const side of ['left','right']){
    const stack=payload?.[side+'_arm'];
    if(Array.isArray(stack)&&[25,27].includes(stack.length)){
      stack.forEach((m,i)=>add(`${side}.${HAND_NAMES[i]}`,side,m));
      for(const [a,b] of HAND_EDGES){const ia=byName.get(`${side}.${HAND_NAMES[a]}`),ib=byName.get(`${side}.${HAND_NAMES[b]}`);if(ia!==undefined&&ib!==undefined)connections.push([ia,ib]);}
    }
    if(!byName.has(`${side}.wrist`))add(`${side}.wrist`,side,payload?.[side+'_wrist']);
  }
  return {records,connections};
}

export class TrackingMarkers extends THREE.Group {
  constructor(){
    super();this.name='AVP tracked input';this.records=[];
    this.points=new THREE.InstancedMesh(new THREE.SphereGeometry(1,10,8),new THREE.MeshBasicMaterial({depthTest:false,transparent:true,opacity:.95}),55);
    this.points.count=0;this.points.frustumCulled=false;this.points.renderOrder=12;this.points.userData.trackingMarkers=this;this.add(this.points);
    const geometry=new THREE.BufferGeometry();geometry.setAttribute('position',new THREE.Float32BufferAttribute(new Float32Array(52*6),3));geometry.setAttribute('color',new THREE.Float32BufferAttribute(new Float32Array(52*6),3));geometry.setDrawRange(0,0);
    this.lines=new THREE.LineSegments(geometry,new THREE.LineBasicMaterial({vertexColors:true,depthTest:false,transparent:true,opacity:.85}));this.lines.frustumCulled=false;this.lines.renderOrder=11;this.add(this.lines);
    this.axes=new Map();
    for(const name of ['head','left.wrist','right.wrist']){const axes=new THREE.AxesHelper(name==='head'?.08:.05);axes.material.depthTest=false;axes.renderOrder=13;axes.visible=false;this.axes.set(name,axes);this.add(axes);}
  }
  update(payload,transform){
    const {records,connections}=trackingRecords(payload,transform);this.records=records;
    this.points.count=records.length;this.points.boundingSphere=null;
    for(const [i,item] of records.entries()){
      const radius=item.name==='head'?.014:item.name.endsWith('.wrist')?.009:.0045;
      this.points.setMatrixAt(i,new THREE.Matrix4().makeScale(radius,radius,radius).setPosition(item.position));
      this.points.setColorAt(i,new THREE.Color(COLORS[item.side]));
    }
    this.points.instanceMatrix.needsUpdate=true;if(this.points.instanceColor)this.points.instanceColor.needsUpdate=true;
    const positions=this.lines.geometry.attributes.position,colors=this.lines.geometry.attributes.color;
    let vertex=0;
    for(const edge of connections)for(const index of edge){const item=records[index],color=new THREE.Color(COLORS[item.side]);positions.setXYZ(vertex,...item.position.toArray());colors.setXYZ(vertex,color.r,color.g,color.b);vertex++;}
    positions.needsUpdate=colors.needsUpdate=true;this.lines.geometry.setDrawRange(0,vertex);
    for(const [name,axes] of this.axes){const item=records.find(r=>r.name===name);axes.visible=Boolean(item)&&this.showAxes!==false;if(item){axes.position.copy(item.position);axes.quaternion.setFromRotationMatrix(new THREE.Matrix4().extractRotation(item.mapped));}}
  }
  setOptions({visible,lines,axes}){this.visible=visible;this.lines.visible=lines;this.showAxes=axes;for(const [name,object]of this.axes)object.visible=axes&&this.records.some(r=>r.name===name);}
}
