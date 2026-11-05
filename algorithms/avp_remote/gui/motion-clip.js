import {blendHandPoses, validateHandPose} from '/api/artifact?path=algorithms/avp_remote/gui/hand-rig.js';
import {packClip,unpackClip} from '/api/artifact?path=algorithms/avp_remote/gui/motion-codec.js';

// Timestamped samples preserve missing tracking and the displayed hand rig.
export const MAX_FRAMES = 1800;
export const MAX_SECONDS = 60;
export const FORMAT = 'avp-motion-clip';
export function createClip(urdfSha256, referenceSpace) {
  return {format:FORMAT, version:1, name:new Date().toLocaleString(), createdAt:new Date().toISOString(),
    urdfSha256, referenceSpace, coordinates:'Z-up, +Y forward, meters; row-major 4x4',
    sampleHz:30, retargetVersion:5, poseSource:'joint-limited arm CCD, separate neck pitch/head yaw and bind-relative hand rotations; not measured body tracking', frames:[]};
}
export function appendFrame(clip, t, pose, tracking, rawTracking, alignment, handPose={}) {
  if (!Number.isFinite(t) || t < 0) throw new Error('Invalid sample time');
  const last=clip.frames.at(-1);
  if (last && t-last.t < 1/30-0.001) return false;
  if (clip.frames.length >= MAX_FRAMES || t > MAX_SECONDS) return false;
  clip.frames.push(structuredClone({t,pose,tracking,rawTracking,alignment,handPose}));
  return true;
}
export function frameIndex(frames, time) {
  let lo=0, hi=frames.length-1;
  while(lo<hi){const mid=Math.ceil((lo+hi)/2);if(frames[mid].t<=time)lo=mid;else hi=mid-1;}
  return lo;
}
export function sampleClip(clip,time,poseAt=null,handPoseAt=null) {
  const i=frameIndex(clip.frames,time), a=clip.frames[i], b=clip.frames[i+1];
  if(!a)return null;
  const alpha=b?Math.max(0,Math.min(1,(time-a.t)/(b.t-a.t))):0;
  // Interpolate only the rendered robot; measured joints remain exact samples.
  const from=poseAt?poseAt(i):a.pose,to=b?(poseAt?poseAt(i+1):b.pose):null;
  const pose=Object.fromEntries(Object.entries(from).map(([key,v])=>[key,v+((to?.[key]??v)-v)*alpha]));
  const hands=handPoseAt?handPoseAt(i):a.handPose;
  const nextHands=b?(handPoseAt?handPoseAt(i+1):b.handPose):hands;
  return {...a,pose,handPose:blendHandPoses(hands,nextHands,alpha),index:i};
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
  value=unpackClip(value);
  if(value?.format!==FORMAT||value.version!==1)throw new Error('不支持的动作文件格式');
  if(value.urdfSha256!==sha)throw new Error('动作文件与当前机器人模型不匹配');
  if(!Array.isArray(value.frames)||!value.frames.length||value.frames.length>MAX_FRAMES)throw new Error('动作帧数无效');
  const names=new Set(joints.filter(j=>j.type!=='fixed').map(j=>j.child));
  let previous=-1;
  for(const f of value.frames){
    if(!Number.isFinite(f.t)||f.t<0||f.t<=previous||f.t>MAX_SECONDS)throw new Error('动作时间戳必须递增且不超过 60 秒');
    if(!f.pose||Array.isArray(f.pose)||Object.keys(f.pose).length!==names.size||Object.entries(f.pose).some(([k,v])=>!names.has(k)||!Number.isFinite(v)))throw new Error('动作关节数据无效');
    if(f.handPose!==undefined&&!validateHandPose(f.handPose))throw new Error('动作手部旋转数据无效');
    if(!validTracking(f.tracking)||!validTracking(f.rawTracking)||!(f.alignment===null||matrix(f.alignment)))throw new Error('追踪矩阵无效');
    previous=f.t;
  }
  return value;
}
export async function clipStorage(value, key='latest') {
  const db=await new Promise((resolve,reject)=>{const r=indexedDB.open('avp-motion-studio',1);r.onupgradeneeded=()=>r.result.createObjectStore('clips');r.onsuccess=()=>resolve(r.result);r.onerror=()=>reject(r.error);});
  try{const result=await new Promise((resolve,reject)=>{
    const tx=db.transaction('clips',value===undefined?'readonly':'readwrite'),store=tx.objectStore('clips');
    let result;
    const r=value===undefined?store.get(key):store.put({format:'avp-clip-pointer',key:'take:'+value.createdAt},'latest');
    r.onsuccess=()=>{
      if(value===undefined&&r.result?.format==='avp-clip-pointer'){
        const take=store.get(r.result.key);take.onsuccess=()=>{result=take.result;};
      }else result=r.result;
    };
    if(value!==undefined){
      store.put(packClip(value),'take:'+value.createdAt);
      store.delete('remote:take:'+value.createdAt);
      store.put({key:'take:'+value.createdAt,name:value.name,createdAt:value.createdAt,frames:value.frames.length},'summary:'+value.createdAt);
    }
    tx.oncomplete=()=>resolve(result);tx.onerror=()=>reject(tx.error);tx.onabort=()=>reject(tx.error||new Error('Storage aborted'));
  });return value===undefined?unpackClip(result):result;}finally{db.close();}
}
export async function clipSummaries() {
  const db=await new Promise((resolve,reject)=>{const r=indexedDB.open('avp-motion-studio',1);r.onupgradeneeded=()=>r.result.createObjectStore('clips');r.onsuccess=()=>resolve(r.result);r.onerror=()=>reject(r.error);});
  try{return await new Promise((resolve,reject)=>{
    const tx=db.transaction('clips','readonly');
    const r=tx.objectStore('clips').getAll(IDBKeyRange.bound('summary:','summary:\uffff'));
    tx.oncomplete=()=>resolve(r.result.reverse());tx.onerror=()=>reject(tx.error);tx.onabort=()=>reject(tx.error||new Error('Storage aborted'));
  });}finally{db.close();}
}

export function syncLocalClips(){
  return new Promise((resolve,reject)=>{
    const worker=new Worker('/api/artifact?path=algorithms/avp_remote/gui/clip-sync.js',{type:'module'});
    worker.onmessage=event=>{worker.terminate();event.data.error?reject(new Error(event.data.error)):resolve(event.data);};
    worker.onerror=event=>{worker.terminate();reject(new Error(event.message||'录制同步失败'));};
    worker.postMessage({});
  });
}
export async function sharedClips(key,{signal,onProgress}={}){
  const controller=new AbortController(),timeoutMs=key?120000:20000;
  let timedOut=false;
  const cancel=()=>controller.abort();
  if(signal?.aborted)cancel();else signal?.addEventListener('abort',cancel,{once:true});
  const timer=setTimeout(()=>{timedOut=true;controller.abort();},timeoutMs);
  try{
    const response=await fetch('/api/avp/recordings'+(key?'/'+encodeURIComponent(key.replace(/^server:/,''))+'?packed=1':''),{signal:controller.signal});
    if(!response.ok){const error=await response.json();throw new Error(error.error||response.statusText);}
    let result;
    if(key&&response.body){
      const reader=response.body.getReader(),decoder=new TextDecoder(),chunks=[];
      const total=Number(response.headers.get('X-Uncompressed-Content-Length')||(!response.headers.get('Content-Encoding')&&response.headers.get('Content-Length')))||0;
      let loaded=0;
      onProgress?.({loaded,total});
      while(true){const {done,value}=await reader.read();if(done)break;loaded+=value.byteLength;chunks.push(decoder.decode(value,{stream:true}));onProgress?.({loaded,total});}
      chunks.push(decoder.decode());result=JSON.parse(chunks.join(''));
    }else result=await response.json();
    return key?unpackClip(result):result.recordings;
  }catch(error){
    if(timedOut)throw new Error(key?'录制下载超时（120 秒），请检查连接后重试':'录制列表请求超时，请重试');
    throw error;
  }finally{clearTimeout(timer);signal?.removeEventListener('abort',cancel);}
}
