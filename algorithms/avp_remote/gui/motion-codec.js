// Lossless storage/transport representation. Runtime clips remain v1 objects.
export const PACKED_FORMAT='avp-motion-clip-packed';
const equal=(a,b)=>JSON.stringify(a)===JSON.stringify(b);
const multiply=(a,b)=>Array.from({length:4},(_,r)=>Array.from({length:4},(_,c)=>a[r][0]*b[0][c]+a[r][1]*b[1][c]+a[r][2]*b[2][c]+a[r][3]*b[3][c]));
const matrix=m=>m[3].every((v,i)=>v===(i===3?1:0))?m.slice(0,3).flat():m.flat();
function unmatrix(m){if(!Array.isArray(m)||![12,16].includes(m.length)||!m.every(Number.isFinite))throw new Error('压缩矩阵无效');return [m.slice(0,4),m.slice(4,8),m.slice(8,12),m.length===16?m.slice(12):[0,0,0,1]];}
function packTracking(t){if(t===null)return null;const result={};for(const [k,v] of Object.entries(t)){
  const arm=k.replace('_wrist','_arm');
  result[k]=k.endsWith('_wrist')&&t[arm]?.length&&equal(v,t[arm][0])?'arm0':k.endsWith('_arm')?v.map(matrix):matrix(v);
}return result;}
function unpackTracking(t){if(t===null)return null;if(!t||typeof t!=='object'||Array.isArray(t))throw new Error('压缩追踪无效');const result={};
  for(const [k,v] of Object.entries(t)){if(v==='arm0')continue;
    if(['left_arm','right_arm'].includes(k)){if(!Array.isArray(v)||![25,27].includes(v.length))throw new Error('压缩手部无效');result[k]=v.map(unmatrix);}
    else if(['head','left_wrist','right_wrist'].includes(k))result[k]=unmatrix(v);else throw new Error('未知追踪字段');}
  for(const [k,v] of Object.entries(t))if(v==='arm0'){const arm=k.replace('_wrist','_arm');if(!k.endsWith('_wrist')||!result[arm])throw new Error('手腕别名无效');result[k]=structuredClone(result[arm][0]);}
  return result;
}
function derived(raw,alignment){if(!raw||!alignment)return null;return Object.fromEntries(Object.entries(raw).map(([k,v])=>[k,k.endsWith('_arm')?v.map(m=>multiply(alignment,m)):multiply(alignment,v)]));}

export function packClip(clip){
  if(clip.format===PACKED_FORMAT)return clip;
  if(clip.format!=='avp-motion-clip'||clip.version!==1||!clip.frames?.length)throw new Error('不支持的录制格式');
  const metadata={...clip};delete metadata.frames;
  const keys=[...new Set(clip.frames.flatMap(f=>Object.keys(f.pose)))].sort(),constantPose={};
  for(const k of keys)if(clip.frames.every(f=>Object.hasOwn(f.pose,k)&&f.pose[k]===clip.frames[0].pose[k]))constantPose[k]=clip.frames[0].pose[k];
  const poseKeys=keys.filter(k=>!Object.hasOwn(constantPose,k)),alignments=[],lookup=new Map();
  const frames=clip.frames.map(f=>{const result={...f,pose:poseKeys.map(k=>Object.hasOwn(f.pose,k)?f.pose[k]:null)};
    if(Object.hasOwn(f,'alignment')){const key=JSON.stringify(f.alignment);if(!lookup.has(key)){lookup.set(key,alignments.length);alignments.push(f.alignment===null?null:matrix(f.alignment));}result.alignment=lookup.get(key);}
    if(Object.hasOwn(f,'rawTracking'))result.rawTracking=packTracking(f.rawTracking);
    if(Object.hasOwn(f,'tracking')){const reconstructed=derived(f.rawTracking,f.alignment);
      const exact=reconstructed&&f.tracking&&Object.keys(reconstructed).length===Object.keys(f.tracking).length&&Object.entries(reconstructed).every(([k,v])=>Object.hasOwn(f.tracking,k)&&equal(v,f.tracking[k]));
      result.tracking=exact?'derived':packTracking(f.tracking);}
    return result;});
  return {format:PACKED_FORMAT,version:1,metadata,poseKeys,constantPose,alignments,frames};
}

export function unpackClip(value){
  if(value?.format!==PACKED_FORMAT)return value;
  if(value.version!==1||value.metadata?.format!=='avp-motion-clip'||value.metadata.version!==1||
      !Array.isArray(value.frames)||!value.frames.length||value.frames.length>1800||
      !Array.isArray(value.poseKeys)||value.poseKeys.length>256||new Set(value.poseKeys).size!==value.poseKeys.length||
      !Array.isArray(value.alignments)||value.alignments.length>1800)throw new Error('压缩动作结构无效');
  const constant=value.constantPose;if(!constant||Array.isArray(constant)||Object.keys(constant).length+value.poseKeys.length>256||
      !Object.values(constant).every(Number.isFinite)||value.poseKeys.some(k=>typeof k!=='string'||Object.hasOwn(constant,k)))throw new Error('压缩关节字典无效');
  const alignments=value.alignments.map(m=>m===null?null:unmatrix(m));
  return {...value.metadata,frames:value.frames.map(f=>{
    if(!Array.isArray(f.pose)||f.pose.length!==value.poseKeys.length)throw new Error('压缩关节数量无效');
    const pose={...constant};value.poseKeys.forEach((k,i)=>{if(f.pose[i]!==null){if(!Number.isFinite(f.pose[i]))throw new Error('压缩关节数值无效');Object.defineProperty(pose,k,{value:f.pose[i],enumerable:true,writable:true,configurable:true});}});
    const out={...f,pose};
    if(Object.hasOwn(f,'alignment')){if(!Number.isInteger(f.alignment)||f.alignment<0||f.alignment>=alignments.length)throw new Error('压缩对齐索引无效');out.alignment=structuredClone(alignments[f.alignment]);}
    if(Object.hasOwn(f,'rawTracking'))out.rawTracking=unpackTracking(f.rawTracking);
    if(Object.hasOwn(f,'tracking')){out.tracking=f.tracking==='derived'?derived(out.rawTracking,out.alignment):unpackTracking(f.tracking);if(f.tracking==='derived'&&!out.tracking)throw new Error('追踪派生条件不完整');}
    return out;
  })};
}

export async function compressedClip(clip){const data=JSON.stringify(packClip(clip));return new Response(new Blob([data]).stream().pipeThrough(new CompressionStream('gzip'))).blob();}
export async function readClipFile(file){
  const buffer=await file.arrayBuffer(),magic=new Uint8Array(buffer,0,Math.min(2,buffer.byteLength));
  if(magic[0]!==31||magic[1]!==139)return unpackClip(JSON.parse(new TextDecoder().decode(buffer)));
  const reader=new Blob([buffer]).stream().pipeThrough(new DecompressionStream('gzip')).getReader();let size=0;const chunks=[];
  while(true){const {value,done}=await reader.read();if(done)break;size+=value.length;if(size>96*1024*1024){await reader.cancel();throw new Error('动作解压超过96MiB');}chunks.push(value);}
  return unpackClip(JSON.parse(await new Blob(chunks).text()));
}
