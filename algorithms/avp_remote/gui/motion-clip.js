// Timestamped samples preserve missing tracking; playback never re-solves IK.
export const MAX_FRAMES = 1800;
export const MAX_SECONDS = 60;
export const FORMAT = 'avp-motion-clip';
export function createClip(urdfSha256, referenceSpace) {
  return {format:FORMAT, version:1, name:new Date().toLocaleString(), createdAt:new Date().toISOString(),
    urdfSha256, referenceSpace, coordinates:'Z-up, +Y forward, meters; row-major 4x4',
    sampleHz:30, poseSource:'displayed joint-limited CCD pose, interpolated; not measured body tracking', frames:[]};
}
export function appendFrame(clip, t, pose, tracking, rawTracking, alignment) {
  if (!Number.isFinite(t) || t < 0) throw new Error('Invalid sample time');
  const last=clip.frames.at(-1);
  if (last && t-last.t < 1/30-0.001) return false;
  if (clip.frames.length >= MAX_FRAMES || t > MAX_SECONDS) return false;
  clip.frames.push(structuredClone({t,pose,tracking,rawTracking,alignment}));
  return true;
}
export function frameIndex(frames, time) {
  let lo=0, hi=frames.length-1;
  while(lo<hi){const mid=Math.ceil((lo+hi)/2);if(frames[mid].t<=time)lo=mid;else hi=mid-1;}
  return lo;
}
export function sampleClip(clip,time) {
  const i=frameIndex(clip.frames,time), a=clip.frames[i], b=clip.frames[i+1];
  if(!a)return null;
  const alpha=b?Math.max(0,Math.min(1,(time-a.t)/(b.t-a.t))):0;
  // Interpolate only the rendered robot; measured joints remain exact samples.
  const pose=Object.fromEntries(Object.entries(a.pose).map(([key,v])=>[key,v+((b?.pose[key]??v)-v)*alpha]));
  return {...a,pose,index:i};
}
function matrix(value) {
  return Array.isArray(value)&&value.length===4&&value.every(r=>Array.isArray(r)&&r.length===4&&r.every(Number.isFinite))&&
    value[3].every((v,i)=>Math.abs(v-(i===3?1:0))<1e-4);
}
function validTracking(value) {
  if(value===null)return true;
  if(!value||typeof value!=='object'||Array.isArray(value))return false;
  return Object.entries(value).every(([key,v])=>['head','left_wrist','right_wrist'].includes(key)?matrix(v):
    ['left_arm','right_arm'].includes(key)&&Array.isArray(v)&&[25,27].includes(v.length)&&v.every(matrix));
}
export function validateClip(value,joints,sha) {
  if(value?.format!==FORMAT||value.version!==1)throw new Error('不支持的动作文件格式');
  if(value.urdfSha256!==sha)throw new Error('动作文件与当前机器人模型不匹配');
  if(!Array.isArray(value.frames)||!value.frames.length||value.frames.length>MAX_FRAMES)throw new Error('动作帧数无效');
  const names=new Set(joints.filter(j=>j.type!=='fixed').map(j=>j.child));
  let previous=-1;
  for(const f of value.frames){
    if(!Number.isFinite(f.t)||f.t<0||f.t<=previous||f.t>MAX_SECONDS)throw new Error('动作时间戳必须递增且不超过 60 秒');
    if(!f.pose||Array.isArray(f.pose)||Object.keys(f.pose).length!==names.size||Object.entries(f.pose).some(([k,v])=>!names.has(k)||!Number.isFinite(v)))throw new Error('动作关节数据无效');
    if(!validTracking(f.tracking)||!validTracking(f.rawTracking)||!(f.alignment===null||matrix(f.alignment)))throw new Error('追踪矩阵无效');
    previous=f.t;
  }
  return value;
}
export async function clipStorage(value, key='latest') {
  const db=await new Promise((resolve,reject)=>{const r=indexedDB.open('avp-motion-studio',1);r.onupgradeneeded=()=>r.result.createObjectStore('clips');r.onsuccess=()=>resolve(r.result);r.onerror=()=>reject(r.error);});
  try{return await new Promise((resolve,reject)=>{
    const tx=db.transaction('clips',value===undefined?'readonly':'readwrite'),store=tx.objectStore('clips');
    const r=value===undefined?store.get(key):store.put(value,'latest');
    if(value!==undefined){
      store.put(value,'take:'+value.createdAt);
      store.put({key:'take:'+value.createdAt,name:value.name,createdAt:value.createdAt,frames:value.frames.length},'summary:'+value.createdAt);
    }
    tx.oncomplete=()=>resolve(r.result);tx.onerror=()=>reject(tx.error);tx.onabort=()=>reject(tx.error||new Error('Storage aborted'));
  });}finally{db.close();}
}
export async function clipSummaries() {
  const db=await new Promise((resolve,reject)=>{const r=indexedDB.open('avp-motion-studio',1);r.onupgradeneeded=()=>r.result.createObjectStore('clips');r.onsuccess=()=>resolve(r.result);r.onerror=()=>reject(r.error);});
  try{return await new Promise((resolve,reject)=>{
    const tx=db.transaction('clips','readonly');
    const r=tx.objectStore('clips').getAll(IDBKeyRange.bound('summary:','summary:\uffff'));
    tx.oncomplete=()=>resolve(r.result.reverse());tx.onerror=()=>reject(tx.error);tx.onabort=()=>reject(tx.error||new Error('Storage aborted'));
  });}finally{db.close();}
}
