import * as THREE from 'three';
import {OrbitControls} from '/vendor/three/modules/OrbitControls.js';
import {GLTFLoader} from '/vendor/three/modules/GLTFLoader.js';
import {DRACOLoader} from '/vendor/three/modules/DRACOLoader.js';

const url = path => '/api/artifact?path='+encodeURIComponent(path);
const MAX_BYTES = 128*1024*1024;
export async function hash(bytes) {
  if(globalThis.crypto?.subtle)return Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',bytes)),v=>v.toString(16).padStart(2,'0')).join('');
  // WebCrypto is unavailable on plain LAN HTTP. Keep integrity checks there too.
  const primes=[];for(let n=2;primes.length<64;n++)if(!primes.some(p=>p*p<=n&&n%p===0))primes.push(n);
  const k=primes.map(p=>(Math.cbrt(p)%1*2**32)>>>0),h=primes.slice(0,8).map(p=>(Math.sqrt(p)%1*2**32)>>>0);
  const input=new Uint8Array(bytes),padded=new Uint8Array(Math.ceil((input.length+9)/64)*64);padded.set(input);padded[input.length]=128;
  const view=new DataView(padded.buffer);view.setUint32(padded.length-4,input.length*8,false);
  const r=(v,n)=>(v>>>n)|(v<<(32-n)),w=new Uint32Array(64);
  for(let offset=0;offset<padded.length;offset+=64){
    for(let i=0;i<16;i++)w[i]=view.getUint32(offset+4*i,false);
    for(let i=16;i<64;i++){const a=w[i-15],b=w[i-2];w[i]=(w[i-16]+(r(a,7)^r(a,18)^(a>>>3))+w[i-7]+(r(b,17)^r(b,19)^(b>>>10)))>>>0;}
    let [a,b,c,d,e,f,g,v]=h;
    for(let i=0;i<64;i++){const t1=(v+(r(e,6)^r(e,11)^r(e,25))+((e&f)^(~e&g))+k[i]+w[i])>>>0,t2=((r(a,2)^r(a,13)^r(a,22))+((a&b)^(a&c)^(b&c)))>>>0;v=g;g=f;f=e;e=(d+t1)>>>0;d=c;c=b;b=a;a=(t1+t2)>>>0;}
    [a,b,c,d,e,f,g,v].forEach((x,i)=>h[i]=(h[i]+x)>>>0);
  }return h.map(v=>v.toString(16).padStart(8,'0')).join('');
}

async function bytes(path,signal,onProgress) {
  if(window.GeoAssets)return window.GeoAssets.read(url(path),{signal,onProgress});
  const response=await fetch(url(path),{signal});
  if(!response.ok)throw new Error(`${path.split('/').pop()} loading failed: ${response.status}`);
  const buffer=await response.arrayBuffer();
  if(buffer.byteLength>MAX_BYTES)throw new Error('Motion asset too large');
  return buffer;
}
async function gunzip(buffer,limit=MAX_BYTES) {
  const reader=new Blob([buffer]).stream().pipeThrough(new DecompressionStream('gzip')).getReader();
  const chunks=[];let size=0;
  while(true){const {value,done}=await reader.read();if(done)break;size+=value.length;if(size>limit){await reader.cancel();throw new Error('动作解压长度超限');}chunks.push(value);}
  const out=new Uint8Array(size);let offset=0;for(const chunk of chunks){out.set(chunk,offset);offset+=chunk.length;}return out.buffer;
}

export function decodeMotion(meta,buffer) {
  if(meta.format!=='landau-walk-motion'||meta.version!==1||meta.quaternion!=='wxyz'||
      !Number.isInteger(meta.frames)||meta.frames<2||meta.frames>100000||
      meta.qposWidth!==7+meta.joints.length||meta.qposWidth>256||
      meta.columns.length!==meta.qposWidth||buffer.byteLength!==meta.byteLength||buffer.byteLength>MAX_BYTES)
    throw new Error('动作格式或长度无效');
  const view=new DataView(buffer),columns=[];
  for(let i=0;i<meta.columns.length;i++){
    const c=meta.columns[i];if(c.qpos!==i)throw new Error('动作列映射无效');
    const values=new Float64Array(meta.frames);
    if(c.constant!==undefined){if(!Number.isFinite(c.constant))throw new Error('静态关节无效');values.fill(c.constant);}
    else {
      const width={f32:4,u16:2,u32:4}[c.dtype];
      if(!width||!Number.isInteger(c.byteOffset)||c.byteOffset<0||c.byteLength!==width*meta.frames||
          c.byteOffset+c.byteLength>buffer.byteLength||!Number.isFinite(c.offset)||!Number.isFinite(c.scale)||c.scale<=0)throw new Error('动作数据越界');
      for(let f=0;f<meta.frames;f++){
        const p=c.byteOffset+f*width;
        values[f]=(c.dtype==='f32'?view.getFloat32(p,true):c.dtype==='u16'?view.getUint16(p,true):view.getUint32(p,true))*c.scale+c.offset;
        if(!Number.isFinite(values[f]))throw new Error('动作含非有限数值');
      }
    }columns.push(values);
  }
  const times=new Float64Array(meta.frames),clock=meta.clock;
  for(let i=0;i<meta.frames;i++){
    times[i]=clock.step!==undefined?clock.start+i*clock.step:view.getFloat64(clock.byteOffset+8*i,true);
    if(!Number.isFinite(times[i])||(i&&times[i]<=times[i-1]))throw new Error('动作时钟无效');
    const norm=Math.hypot(...columns.slice(3,7).map(c=>c[i]));if(Math.abs(norm-1)>1e-3)throw new Error('根节点旋转无效');
  }
  if(Math.abs(times.at(-1)-times[0]-meta.duration)>1e-6)throw new Error('动作时长不匹配');
  return {meta,columns,times};
}

function origin({xyz,rpy}) {
  return new THREE.Matrix4().compose(new THREE.Vector3(...xyz),new THREE.Quaternion().setFromEuler(new THREE.Euler(...rpy,'ZYX')),new THREE.Vector3(1,1,1));
}
export function worldTransforms(joints,pose={}) {
  const worlds={base_link:new THREE.Matrix4()},pending=[...joints];
  while(pending.length){let progress=false;
    for(let i=pending.length-1;i>=0;i--){const j=pending[i];if(!worlds[j.parent])continue;
      const v=pose[j.name]??0,motion=new THREE.Matrix4();
      if(j.type==='prismatic')motion.makeTranslation(...j.axis.map(a=>a*v));
      else if(j.type!=='fixed')motion.makeRotationAxis(new THREE.Vector3(...j.axis).normalize(),v);
      worlds[j.child]=worlds[j.parent].clone().multiply(origin(j)).multiply(motion);pending.splice(i,1);progress=true;
    }if(!progress)throw new Error('模型骨骼层级不完整');
  }return worlds;
}

// Same bind-relative ownership rule as the AVP retargeting contract. No IK or
// tracking remap is applied to simulation joint states.
class SkinMapper {
  constructor(model,joints){
    model.updateMatrixWorld(true);const inverse=model.matrixWorld.clone().invert(),rest=worldTransforms(joints);
    const parents=new Map(joints.map(j=>[j.child,j.parent]));
    const descendant=(child,ancestor)=>{const seen=new Set();while(parents.has(child)&&!seen.has(child)){seen.add(child);child=parents.get(child);if(child===ancestor)return true;}return false;};
    const ownerFor=name=>{for(const [part,distal] of [['arm','forearm_stretch'],['thigh','leg_stretch'],['leg','foot']]){
      if(!name.startsWith(part+'_stretch_'))continue;const side=name.slice(-1),owner=part+'_twist_'+side;
      if(descendant(owner,name)&&descendant(distal+'_'+side,owner))return owner;
    }return name;};
this.records=[];this.byBone=new Map();
    model.traverse(bone=>{if(!bone.isBone)return;if(!rest[bone.name])throw new Error('USD骨骼不匹配：'+bone.name);
      bone.matrixAutoUpdate=false;
      const source=ownerFor(bone.name);
      const r={bone,source,offset:rest[source].clone().invert().multiply(inverse.clone().multiply(bone.matrixWorld)),target:new THREE.Matrix4(),parentRest:inverse.clone().multiply(bone.parent.matrixWorld)};
      this.records.push(r);this.byBone.set(bone,r);
    });
  }
  apply(worlds){for(const r of this.records)r.target.copy(worlds[r.source]).multiply(r.offset);
    for(const r of this.records){r.bone.matrix.copy(this.byBone.get(r.bone.parent)?.target||r.parentRest).invert().multiply(r.target);r.bone.matrixWorldNeedsUpdate=true;}}
}

export function mount(root,{motionPath,scenePath,usdPath,onError,onExpand,expandIcon}) {
  for(const key of ['ready','usdReady','error','frame','time'])delete root.dataset[key];
  root.innerHTML=`<div class="motion-toolbar"><label><select data-model aria-label="回放模型"><option value="usd">USD 蒙皮</option><option value="urdf">URDF</option></select></label><label><input type="checkbox" data-follow checked>跟随</label><label title="显示当前动作的关节位置与骨架连线"><input type="checkbox" data-joints>关节</label><button class="button button-light motion-reset" data-reset aria-label="重置镜头" title="重置镜头"><svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="1.6" aria-hidden="true"><path d="M4 10a8 8 0 1 1 1 8M4 4v6h6"/><circle cx="12" cy="12" r="3"/></svg></button></div><div class="motion-stage" title="拖动旋转 · 滚轮缩放 · 右键平移" aria-label="可旋转的3D动作回放">${onExpand?`<button class="viewport-expand" data-expand-viewer aria-label="Expand playback" title="Expand playback">${expandIcon}</button>`:''}<span class="motion-status" data-status>正在加载动作…</span><div class="viewer-load-overlay" data-load-overlay role="status"><span class="loading-spinner"></span><b data-load-label>Preparing playback…</b><progress data-load-progress aria-label="Asset download progress"></progress></div></div><div class="motion-timeline"><button class="button" data-play disabled>播放</button><input data-seek type="range" min="0" max="1" step="0.001" value="0" aria-label="动作时间轴" disabled><output data-time>0:00 / 0:00</output><select data-speed aria-label="回放速度"><option value="0.5">0.5×</option><option value="1" selected>1×</option><option value="2">2×</option></select></div><p class="muted motion-help">拖动旋转 · 滚轮或双指缩放 · 右键平移 · 模型切换保持当前动作和镜头 · 显示关节：橙色关节、浅色骨架</p>`;
  const $=s=>root.querySelector(s),stage=$('.motion-stage'),abort=new AbortController();
  $('[data-expand-viewer]')?.addEventListener('click',onExpand);
  const report=label=>progress=>{if(disposed)return;const text=label+' · '+window.GeoAssets.describe(progress);$('[data-load-label]').textContent=text;$('[data-status]').textContent=text;const bar=$('[data-load-progress]');if(progress.total){bar.max=progress.total;bar.value=progress.loaded;}else bar.removeAttribute('value');};
  let disposed=false,raf=0,playing=false,cursor=0,last=0,clip=null,sceneData=null,mapper=null,usdReady=false;
  const renderer=new THREE.WebGLRenderer({antialias:true,alpha:false});renderer.setPixelRatio(Math.min(devicePixelRatio,2));stage.append(renderer.domElement);
  renderer.outputColorSpace=THREE.SRGBColorSpace;renderer.shadowMap.enabled=true;renderer.shadowMap.type=THREE.PCFSoftShadowMap;
  const scene=new THREE.Scene();scene.background=new THREE.Color('#e9eee9');
  const camera=new THREE.PerspectiveCamera(38,1,.01,1000),controls=new OrbitControls(camera,renderer.domElement);controls.enableDamping=true;
  scene.add(new THREE.HemisphereLight(0xffffff,0x526672,2.5));const light=new THREE.DirectionalLight(0xffffff,3);light.position.set(2,4,3);light.castShadow=true;light.shadow.mapSize.set(1024,1024);light.shadow.camera.left=-2;light.shadow.camera.right=2;light.shadow.camera.top=2;light.shadow.camera.bottom=-2;light.shadow.bias=-.0001;scene.add(light,light.target);
  const checker=new THREE.DataTexture(new Uint8Array([51,77,102,255,26,51,77,255,26,51,77,255,51,77,102,255]),2,2);
  checker.colorSpace=THREE.SRGBColorSpace;checker.wrapS=checker.wrapT=THREE.RepeatWrapping;checker.repeat.set(100,100);checker.magFilter=THREE.NearestFilter;checker.needsUpdate=true;
  const floor=new THREE.Mesh(new THREE.PlaneGeometry(200,200),new THREE.MeshStandardMaterial({map:checker,roughness:1}));floor.rotation.x=-Math.PI/2;floor.position.y=-.001;floor.receiveShadow=true;scene.add(floor);
  const coordinate=new THREE.Group();coordinate.rotation.x=-Math.PI/2;scene.add(coordinate);
  const character=new THREE.Group(),urdf=new THREE.Group(),usd=new THREE.Group();coordinate.add(character);character.add(urdf,usd);
  const skeleton=new THREE.Group();skeleton.visible=false;character.add(skeleton);
  let jointPoints=null,boneLines=null,visibleJoints=[];
  function buildSkeleton(){
    visibleJoints=sceneData.joints.filter(j=>j.type!=='fixed');
    const dots=new THREE.BufferGeometry(),lines=new THREE.BufferGeometry();
    dots.setAttribute('position',new THREE.Float32BufferAttribute(new Float32Array(visibleJoints.length*3),3));
    lines.setAttribute('position',new THREE.Float32BufferAttribute(new Float32Array(sceneData.joints.length*6),3));
    jointPoints=new THREE.Points(dots,new THREE.PointsMaterial({color:0xffa326,size:6,sizeAttenuation:false,depthTest:false,depthWrite:false}));
    boneLines=new THREE.LineSegments(lines,new THREE.LineBasicMaterial({color:0xd9f4ff,transparent:true,opacity:.8,depthTest:false,depthWrite:false}));
    jointPoints.renderOrder=12;boneLines.renderOrder=11;jointPoints.frustumCulled=boneLines.frustumCulled=false;
    skeleton.add(boneLines,jointPoints);root.dataset.jointCount=String(visibleJoints.length);
  }
  function updateSkeleton(worlds){
    if(!jointPoints)return;
    const dots=jointPoints.geometry.attributes.position,lines=boneLines.geometry.attributes.position,p=new THREE.Vector3();
    visibleJoints.forEach((j,i)=>{p.setFromMatrixPosition(worlds[j.child]);dots.setXYZ(i,p.x,p.y,p.z);});
    sceneData.joints.forEach((j,i)=>{p.setFromMatrixPosition(worlds[j.parent]);lines.setXYZ(i*2,p.x,p.y,p.z);p.setFromMatrixPosition(worlds[j.child]);lines.setXYZ(i*2+1,p.x,p.y,p.z);});
    dots.needsUpdate=lines.needsUpdate=true;
  }
  const meshes=new Map(),geometries=new Map(),material=new THREE.MeshStandardMaterial({color:0xc6ad77,roughness:.72});
  const draco=new DRACOLoader().setDecoderPath('/vendor/three/draco/').setWorkerLimit(1);
  const loader=new GLTFLoader().setDRACOLoader(draco),lastPosition=new THREE.Vector3(),rootPosition=new THREE.Vector3();
  let centered=false;
  function reset(){camera.position.copy(lastPosition).add(new THREE.Vector3(1.65,1.15,2.05));controls.target.copy(lastPosition).add(new THREE.Vector3(0,.5,0));controls.update();}
  function display(){usd.visible=$('[data-model]').value==='usd'&&usdReady;urdf.visible=!usd.visible;root.dataset.model=usd.visible?'usd':'urdf';}
  function applyWorlds(pose){
    const worlds=worldTransforms(sceneData.joints,pose);
    for(const [name,group] of meshes)group.matrix.copy(worlds[name]);mapper?.apply(worlds);updateSkeleton(worlds);
  }
  function drawPose(){if(!clip||!sceneData)return;
    const {times,columns,meta}=clip,t=cursor+times[0];let lo=0,hi=times.length-1;
    while(lo<hi){const mid=Math.ceil((lo+hi)/2);if(times[mid]<=t)lo=mid;else hi=mid-1;}
    const next=Math.min(lo+1,times.length-1),alpha=next===lo?0:(t-times[lo])/(times[next]-times[lo]);
    const sample=i=>THREE.MathUtils.lerp(columns[i][lo],columns[i][next],alpha);
    const q=i=>new THREE.Quaternion(columns[4][i],columns[5][i],columns[6][i],columns[3][i]).normalize();
    character.position.set(sample(0),sample(1),sample(2));character.quaternion.copy(q(lo).slerp(q(next),alpha));
    const pose=Object.fromEntries(meta.joints.map(j=>[j.name,sample(j.qpos)]));applyWorlds(pose);
    coordinate.updateMatrixWorld(true);character.getWorldPosition(rootPosition);
    if(centered&&$('[data-follow]').checked){const delta=rootPosition.clone().sub(lastPosition);camera.position.add(delta);controls.target.add(delta);}
    lastPosition.copy(rootPosition);if(!centered){centered=true;reset();}
    light.position.copy(rootPosition).add(new THREE.Vector3(2,4,3));light.target.position.copy(rootPosition);
    $('[data-seek]').value=cursor;const fmt=s=>`${Math.floor(s/60)}:${Math.floor(s%60).toString().padStart(2,'0')}`;
    $('[data-time]').textContent=`${fmt(cursor)} / ${fmt(meta.duration)}`;root.dataset.frame=String(lo);root.dataset.time=cursor.toFixed(3);
  }
  function animate(now){if(disposed)return;const dt=last?(now-last)/1000:0;last=now;
    if(playing&&clip){cursor=Math.min(clip.meta.duration,cursor+dt*Number($('[data-speed]').value));if(cursor>=clip.meta.duration){playing=false;$('[data-play]').textContent='重播';}drawPose();}
    controls.update();renderer.render(scene,camera);raf=requestAnimationFrame(animate);
  }
  $('[data-model]').onchange=display;$('[data-reset]').onclick=reset;
  $('[data-joints]').onchange=e=>{skeleton.visible=e.target.checked;root.dataset.jointsVisible=String(skeleton.visible);};
  $('[data-play]').onclick=()=>{if(!clip)return;if(cursor>=clip.meta.duration)cursor=0;playing=!playing;$('[data-play]').textContent=playing?'暂停':'播放';drawPose();};
  $('[data-seek]').oninput=e=>{cursor=Number(e.target.value);drawPose();};
  const resize=new ResizeObserver(()=>{const w=stage.clientWidth,h=stage.clientHeight;if(w&&h){renderer.setSize(w,h);camera.aspect=w/h;camera.updateProjectionMatrix();}});resize.observe(stage);reset();raf=requestAnimationFrame(animate);
  async function load(){try{
    const sceneBuffer=await bytes(scenePath,abort.signal,report('Robot scene'));
    const decodeScene=async()=>JSON.parse(new TextDecoder().decode(await gunzip(sceneBuffer)));
    sceneData=window.GeoAssets?await window.GeoAssets.memo('scene:'+scenePath,decodeScene):await decodeScene();
    const meta=JSON.parse(new TextDecoder().decode(await bytes(motionPath,abort.signal,report('Motion metadata'))));
      if(meta.source.urdfSha256!==sceneData.source.urdfSha256||meta.payload!=='motion.bin.gz')throw new Error('动作与模型版本不匹配');
      const packed=await bytes(motionPath.replace(/[^/]+$/,'')+meta.payload,abort.signal,report('Motion frames'));
      if(await hash(packed)!==meta.payloadSha256)throw new Error('动作压缩文件校验失败');
      clip=decodeMotion(meta,await gunzip(packed,meta.byteLength));
    if(disposed)return;
    for(const [name,vertices] of Object.entries(sceneData.meshes)){const geometry=new THREE.BufferGeometry();geometry.setAttribute('position',new THREE.Float32BufferAttribute(vertices,3));geometry.computeVertexNormals();geometries.set(name,geometry);}
    for(const link of sceneData.links){const group=new THREE.Group();group.matrixAutoUpdate=false;meshes.set(link.name,group);urdf.add(group);
      for(const v of link.visuals){const mesh=new THREE.Mesh(geometries.get(v.mesh),material);mesh.castShadow=true;mesh.matrixAutoUpdate=false;mesh.matrix.copy(origin(v).multiply(new THREE.Matrix4().makeScale(...v.scale)));group.add(mesh);}}
    const known=new Set(sceneData.joints.map(j=>j.name));if(meta.joints.some(j=>!known.has(j.name)))throw new Error('动作关节不属于当前模型');
    $('[data-seek]').max=meta.duration;$('[data-seek]').disabled=false;$('[data-play]').disabled=false;
    buildSkeleton();drawPose();display();root.dataset.ready='true';
    const description=`${meta.frames.toLocaleString()} 帧 · ${(meta.compressedBytes/1024).toFixed(0)} KB`;
    $('[data-status]').textContent=description+' · 正在加载 USD';
    const glb=await bytes(usdPath,abort.signal,report('Character model'));if(await hash(glb)!==sceneData.source.usdGlbSha256)throw new Error('USD模型校验失败');
    if(disposed)return;
    $('[data-load-label]').textContent='Preparing character geometry…';$('[data-load-progress]').removeAttribute('value');
    const model=await loader.parseAsync(glb.slice(0),'');if(disposed){dispose(model.scene);return;}
    model.scene.traverse(o=>{if(o.isMesh)o.castShadow=true;if(o.isSkinnedMesh)o.frustumCulled=false;});mapper=new SkinMapper(model.scene,sceneData.joints);usd.add(model.scene);usdReady=true;display();drawPose();
    $('[data-status]').textContent=description+' · USD / URDF 已就绪';
    root.dataset.usdReady='true';$('[data-load-overlay]').hidden=true;
  }catch(error){if(!disposed){$('[data-status]').textContent=error.message;root.dataset.error=error.message;$('[data-load-overlay]').hidden=true;onError?.(error);}}}
  function dispose(object){object.traverse(o=>{o.geometry?.dispose();for(const m of [o.material].flat().filter(Boolean)){for(const v of Object.values(m))if(v?.isTexture)v.dispose();m.dispose();}});}
  void load();
  return {destroy(){disposed=true;abort.abort();cancelAnimationFrame(raf);resize.disconnect();controls.dispose();draco.dispose();dispose(scene);renderer.dispose();renderer.forceContextLoss();root.innerHTML='';}};
}
