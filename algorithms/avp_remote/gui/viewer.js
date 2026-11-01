import * as THREE from 'three';
import {OrbitControls} from '/vendor/three/modules/OrbitControls.js';
import {GLTFLoader} from '/vendor/three/modules/GLTFLoader.js';
import {DRACOLoader} from '/vendor/three/modules/DRACOLoader.js';
import {originMatrix, worldTransforms, XR_HAND_NAMES, nativeMatrix, rows, fromRows, sampleXRFrame} from '/api/artifact?path=algorithms/avp_remote/gui/pose-math.js';
import {TrackingMarkers, XRHandMarkers} from '/api/artifact?path=algorithms/avp_remote/gui/tracking-markers.js';
import {createClip, appendFrame, sampleClip, validateClip, clipStorage, clipSummaries, MAX_FRAMES, MAX_SECONDS, FORMAT} from '/api/artifact?path=algorithms/avp_remote/gui/motion-clip.js';
import {unitreePose, BodySchematic, UNITREE_DISPLAY_YAW} from '/api/artifact?path=algorithms/avp_remote/gui/comparison.js';
import {XRButtonInput} from '/api/artifact?path=algorithms/avp_remote/gui/xr-buttons.js';
import {LiveRetargeter} from '/api/artifact?path=algorithms/avp_remote/gui/live-retarget.js';
const asset = path => `/api/artifact?path=algorithms/avp_remote/${path}`;

export function mount(root) {
  root.innerHTML = `<link rel="stylesheet" href="${asset('gui/viewer.css')}">
  <div class="avp-shell"><header class="avp-toolbar">
    <a class="avp-home back-link" href="#/" aria-label="返回 Geo Library">← <span>Geo Library</span></a>
    <div class="avp-brand"><span class="avp-brandmark">◎</span><div><small>SPATIAL MOTION LAB</small><h2>AVP <span>Remote</span></h2></div></div>
    <span class="avp-mode">Snapshot</span><span class="avp-toolbar-note">Landau / 动作工作台</span>
    <button class="avp-primary" data-action="xr" disabled>开始头手追踪 ↗</button>
    <button data-settings-toggle aria-expanded="false" aria-controls="avp-settings">设置 ☷</button>
  </header>
  <section class="avp-capture-guide" data-capture-guide aria-label="实时追踪操作引导">
    <div class="avp-guide-copy"><small>LIVE CAPTURE / 实时采集</small><h3 data-guide-title>正在准备追踪入口…</h3><p data-guide-hint>打开页面不会自动采集。需要主动进入沉浸模式，并允许系统权限。</p>
    <ol class="avp-guide-steps"><li data-step="connect">1 · 开始追踪</li><li data-step="hands">2 · 允许权限，举起双手</li><li data-step="record">3 · 点击开始录制</li></ol></div>
    <div class="avp-guide-actions"><button class="avp-primary" data-action="begin" disabled>开始头手追踪 ↗</button><a data-open-secure hidden>打开 HTTPS 录制页 ↗</a>
    <div class="avp-sensor-status" aria-label="实时追踪状态"><span data-sensor="head">头部：未采集</span><span data-sensor="left">左手：未采集</span><span data-sensor="right">右手：未采集</span></div></div>
  </section>
  <div class="avp-grid"><div class="avp-viewport"><div class="avp-stage">
    <div class="avp-stage-top"><span><i></i> 3D VIEWPORT</span><span data-load-progress role="status">正在下载基础场景…</span><button data-action="fit" title="重置视角">重置视角 ⤢</button></div>
    <div class="avp-caption">拖动旋转 · 滚轮缩放 · 右键平移</div><div class="avp-stage-tag">MOTION COMPARE <span>LANDAU × UNITREE × INPUT</span></div>
  </div><div class="avp-stage-bottom"><span>● <b data-tracking-counts>等待追踪数据</b></span><span>头部 + 双手 / 米</span></div></div>
  <aside class="avp-side" id="avp-settings" aria-label="采集和场景设置">
    <section><div class="avp-section-title"><h3>采集设备</h3><span class="avp-mini">01 / INPUT</span></div>
      <div class="avp-device"><span>◉</span><div><strong>Apple Vision Pro</strong><small>Safari · WebXR</small></div><span data-device-state>待连接</span></div>
      <p class="avp-status" role="status">正在加载 Landau 模型…</p><p data-xr-note>正在检查沉浸式访问条件…</p><p><a href="/setup/vision-pro" style="color:var(--mint)">Vision Pro 局域网设置 ↗</a></p>
      <label class="avp-toggle"><input type="checkbox" data-live> 实时追踪（进入后自动开启）</label><button data-action="recenter">重新校准原点</button>
      <label><input type="checkbox" data-self-hands checked> 沉浸模式：显示自己的手</label><p>双手的实测关节点和线框显示在实际位置；回放时也能看到自己的手。</p>
    </section>
    <section><div class="avp-section-title"><h3>场景显示</h3><span class="avp-mini">02 / SCENE</span></div>
      <select data-display aria-label="Model display"><option value="all">Landau + Unitree + 采集结构</option><option value="unitree">Unitree H1-2</option><option value="both">USD 角色 + URDF 对照</option><option value="usd">USD 蒙皮角色</option><option value="urdf">URDF 网格</option><option value="tracking">采集结构 · 实测点 + 示意连线</option></select>
      <div class="avp-tracking"><label><input type="checkbox" data-markers checked> 追踪标记</label><label><input type="checkbox" data-marker-lines checked> 手部骨架连线</label><label><input type="checkbox" data-marker-axes checked> 头部 / 手腕方向轴</label>
      <p class="avp-marker-legend"><span>● 头部</span><span>● 左手</span><span>● 右手</span></p><p class="avp-capability-note">灰色虚线：躯干与肩臂结构示意，非实测全身骨架。Unitree 为姿态映射估计；H1-2 以 0.52× 缩放展示。</p></div>
    </section>
    <details><summary>追踪数据检查 <span>INSPECT</span></summary><p>点击场景中的追踪点查看坐标。</p><pre data-marker-detail>尚未选择追踪点</pre><p class="avp-capability-note">实测：头部、每手 25 个关节点。手臂 IK 为估计；不采集面部、全身或前臂点。</p></details>
    <details><summary>沉浸运行状态 <span>XR</span></summary><pre data-xr-diagnostics>尚未进入沉浸模式</pre><button data-action="xr-report">导出运行诊断</button></details>
    <details><summary>单帧姿态 <span>POSE</span></summary><div class="avp-button-row"><button data-action="snapshot">参考快照</button><button data-action="reset">零位姿态</button><button data-action="capture">捕获当前帧</button><button data-action="save">导出姿态</button><button data-action="png">保存图片</button></div></details>
    <details><summary>关节微调 <span>JOINTS</span></summary><div class="avp-joints"></div></details>
    <details><summary>资源信息 <span>ASSETS</span></summary><p class="avp-source"></p><button data-action="verify">检查模型来源</button><button data-action="prepare">刷新模型资源</button></details>
  </aside></div>
  <section class="avp-timeline" aria-label="动作录制与回放">
    <div class="avp-timeline-top"><div><small>MOTION TAKE</small><strong data-clip-name>尚未录制动作</strong></div><span data-clip-meta>进入 Vision Pro 后开始录制 · 最长 60 秒</span>
      <button class="avp-record" data-action="record" disabled><span>●</span> 开始录制</button>
      <select data-takes aria-label="已保存的录制"><option value="">已保存的录制</option></select>
      <label class="avp-file">导入 JSON<input data-import type="file" accept=".json,application/json" aria-label="导入动作或姿态 JSON"></label><button data-action="export" disabled>导出动作 ↗</button>
    </div>
    <div class="avp-transport"><button data-action="start" aria-label="回到起点" disabled>⏮</button><button class="avp-play" data-action="play" aria-label="播放动作" disabled>▶</button><output data-play-time>00:00.00</output>
      <div class="avp-scrub"><input data-seek type="range" min="0" max="1" step="0.001" value="0" aria-label="动作播放进度（秒）" disabled><div class="avp-ruler"><span>00:00</span><span data-mid-time>时间轴</span><span data-end-time>00:00</span></div></div>
      <output data-duration>00:00.00</output><select data-speed aria-label="回放速度"><option value="0.25">0.25×</option><option value="0.5">0.5×</option><option value="1" selected>1×</option><option value="2">2×</option></select><label class="avp-loop"><input data-loop type="checkbox">循环</label>
    </div><p data-save-note>录制结束后自动保存在当前浏览器；导出 JSON 可跨设备回放。</p>
  </section><footer class="avp-footer"><span>● 网页动作预览 · 无需 Teleop App</span><span>非物理仿真 · 腿部保持默认姿态</span></footer></div>`;
  root.querySelectorAll('button:not([data-action=prepare]),input').forEach(el=>el.disabled=true);
  const $ = sel => root.querySelector(sel), stage = $('.avp-stage');
  $('[data-settings-toggle]').disabled=false;
  $('[data-settings-toggle]').onclick=()=>{const open=$('.avp-shell').classList.toggle('settings-open');$('[data-settings-toggle]').setAttribute('aria-expanded',String(open));};
  let disposed = false, data, gltf, unitreeData, unitreeGltf, pose = {}, tracking, snapshotPose, snapshotTracking;
  let refreshTimer, liveTarget=null, lastRender=0, markerTransform=null, selectedMarker='head', trackingLabel="Snapshot";
  let referenceSpaceType='local-floor', xrSupport='checking', xrStarting=false, xrError='', xrPlaced=false, lastXRFrameAt=0;
  let live = false, epoch = 0, alignment = null, session = null, liveSolver=null;
  let clip=null, recording=false, playing=false, cursor=0, recordStart=0, rawTracking=null, lastUi=0, savePending=false, saved=true;
  let recordRequested=false,poseRateStart=0,poseRateFrames=0;
  const draco=new DRACOLoader().setDecoderPath('/vendor/three/draco/').setWorkerLimit(2);
  const modelLoader=new GLTFLoader().setDRACOLoader(draco);
  const abort = new AbortController();
  const status = (message, error = false) => { $('.avp-status').textContent = message; $('.avp-status').classList.toggle('avp-error', error); };
  const renderer = new THREE.WebGLRenderer({antialias:false, preserveDrawingBuffer:false, powerPreference:'high-performance'});
  renderer.setPixelRatio(Math.min(devicePixelRatio,2)); renderer.xr.enabled = true;
  renderer.xr.setFramebufferScaleFactor(.65);renderer.xr.setFoveation(1);
  renderer.xr.setReferenceSpaceType('local-floor'); renderer.outputColorSpace = THREE.SRGBColorSpace;
  stage.prepend(renderer.domElement);
  const scene = new THREE.Scene(); scene.background = new THREE.Color('#101c24'); scene.fog=new THREE.Fog('#101c24',5,12);
  const selfHands=new XRHandMarkers();scene.add(selfHands);
  const camera = new THREE.PerspectiveCamera(40,1,.01,100); camera.position.set(1.1,1.1,2.2);
  const controls = new OrbitControls(camera, renderer.domElement); controls.enableDamping=true; controls.target.set(0,.4,0);
  scene.add(new THREE.HemisphereLight(0xffffff,0x70867b,2.8));
  const sun = new THREE.DirectionalLight(0xffffff,3);sun.position.set(2,4,3);scene.add(sun);
  const grid = new THREE.GridHelper(8,80,0x38505b,0x20333e);scene.add(grid);
  const content = new THREE.Group();scene.add(content);
  const usd = new THREE.Group(), urdf = new THREE.Group(), trackingOnly=new THREE.Group(), unitree=new THREE.Group();
  const unitreeModel=new THREE.Group();unitreeModel.rotation.z=UNITREE_DISPLAY_YAW;unitree.add(unitreeModel);
  usd.rotation.x=urdf.rotation.x=trackingOnly.rotation.x=unitree.rotation.x=-Math.PI/2;content.add(usd,urdf,unitree,trackingOnly);
  const body=new BodySchematic();trackingOnly.add(body);
  const usdMarkers=new TrackingMarkers(),urdfMarkers=new TrackingMarkers(),soloMarkers=new TrackingMarkers();
  usd.add(usdMarkers);urdf.add(urdfMarkers);trackingOnly.add(soloMarkers);
  const markerLayers=[usdMarkers,urdfMarkers,soloMarkers];
  function markerOptions(){for(const layer of markerLayers)layer.setOptions({visible:$('[data-markers]').checked,lines:$('[data-marker-lines]').checked,axes:$('[data-marker-axes]').checked});}
  for(const selector of ['[data-markers]','[data-marker-lines]','[data-marker-axes]'])$(selector).onchange=markerOptions;
  function markerDetail(){
    const item=soloMarkers.records.find(r=>r.name===selectedMarker);
    if(!item){$('[data-marker-detail]').textContent=selectedMarker?`${selectedMarker} · not present in this frame`:'No point selected';return;}
    const coords=m=>new THREE.Vector3().setFromMatrixPosition(m).toArray().map(v=>v.toFixed(4)).join(', ');
    const quaternion=new THREE.Quaternion().setFromRotationMatrix(new THREE.Matrix4().extractRotation(item.mapped));
    $('[data-marker-detail]').textContent=`${item.name} · ${trackingLabel}\nInput Z-up m: ${coords(item.raw)}\nRobot Z-up m: ${coords(item.mapped)}\nRotation xyzw: ${quaternion.toArray().map(v=>v.toFixed(4)).join(', ')}`;
  }
  function updateMarkers(frame,label){
    if(!markerTransform)return;
    trackingLabel=label;
    soloMarkers.update(frame,markerTransform);
    for(const layer of [usdMarkers,urdfMarkers])layer.updateRecords(soloMarkers);
    const records=soloMarkers.records;body.update(records);
    const count=side=>records.filter(r=>r.side===side).length;
    const text=`${label==='Snapshot'?'参考快照（非实时）':label} · head ${count('head')} · left ${count('left')} · right ${count('right')}`;
    if($('[data-tracking-counts]').textContent!==text)$('[data-tracking-counts]').textContent=text;
    root.dataset.trackingMarkers=String(records.length);root.dataset.trackingSource=label;
    if(!session)markerDetail();
  }
  markerOptions();
  const visuals = new Map(), bones = [], geometries = new Map(), unitreeLinks=new Map();
  let allowedJoints=null;
  const robotMaterial = new THREE.MeshStandardMaterial({color:0x759b99,roughness:.74,metalness:.12,side:THREE.DoubleSide});
  const labels = [];
  function label(text, x) {
    const c=document.createElement('canvas');c.width=512;c.height=96;
    const ctx=c.getContext('2d');ctx.fillStyle='#a5c4cc';ctx.font='28px sans-serif';ctx.textAlign='center';ctx.fillText(text,256,55);
    const map=new THREE.CanvasTexture(c), sprite=new THREE.Sprite(new THREE.SpriteMaterial({map,depthTest:false}));
    sprite.scale.set(.72,.135,1);sprite.position.set(x,1.12,0);content.add(sprite);labels.push(sprite);return sprite;
  }
  const usdLabel=label('LANDAU / USD',0), urdfLabel=label('LANDAU / URDF',0), unitreeLabel=label('UNITREE H1-2 / mapped',0), trackingLabelSprite=label('INPUT / head + hands',0);
  function layout(){
    const choice=$('[data-display]').value;
    const entries=[[unitree,unitreeLabel,['all','unitree']],[usd,usdLabel,['all','both','usd']],[urdf,urdfLabel,['all','both','urdf']],[trackingOnly,trackingLabelSprite,['all','tracking']]];
    const active=entries.filter(([group,label,modes])=>modes.includes(choice));
    for(const [group,label,modes] of entries){group.visible=modes.includes(choice);label.visible=group.visible;}
    active.forEach(([group,label],i)=>{group.position.x=(i-(active.length-1)/2)*.85;label.position.x=group.position.x;});
    root.dataset.comparison=choice;
  }
  function resetView(){
    const count=[usd,urdf,unitree,trackingOnly].filter(g=>g.visible).length;
    controls.target.set(0,.52,0);
    const halfFov=THREE.MathUtils.degToRad(camera.fov/2);
    const distance=Math.max(1.4/2/Math.tan(halfFov),count*.85/2/Math.tan(halfFov)/camera.aspect)*1.15;
    camera.position.copy(controls.target).add(new THREE.Vector3(.06,.08,1).normalize().multiplyScalar(distance));
    controls.update();
  }
  $('[data-display]').onchange=()=>{layout();if(!renderer.xr.isPresenting)resetView();};layout();
  function applyPose(next) {
    if(!data)return;
    const allowed = allowedJoints||(allowedJoints=new Map(data.joints.map(j=>[j.child,j])));
    for(const [name,value] of Object.entries(next))if(!allowed.has(name)||!Number.isFinite(value))throw new Error(`Invalid pose joint: ${name}`);
    pose=Object.fromEntries(data.joints.filter(j=>j.type!=='fixed').map(j=>[j.child,THREE.MathUtils.clamp(next[j.child]??0,j.lower,j.upper)]));
    const worlds=worldTransforms(data.joints,pose);
    for(const [name,group] of visuals){group.matrix.copy(worlds[name]||new THREE.Matrix4());group.matrixWorldNeedsUpdate=true;}
    // USD skeleton parenting skips the URDF's extra hip-roll links. Use world
    // transforms, then recover each exported bone's own parent-relative matrix.
    for(const bone of bones){
      const world=worlds[bone.name];if(!world)throw new Error(`USD bone absent from URDF: ${bone.name}`);
      const parent=bone.parent?.isBone?worlds[bone.parent.name]:new THREE.Matrix4();
      bone.matrix.copy(parent.clone().invert().multiply(world));bone.matrix.decompose(bone.position,bone.quaternion,bone.scale);
    }
    gltf?.scene.updateMatrixWorld(true);
    if(unitreeData&&unitreeGltf){
      const mapped=unitreePose(unitreeData,pose), worlds=worldTransforms(unitreeData.joints,mapped,unitreeData.root);
      for(const name of unitreeData.links){let object=unitreeLinks.get(name);if(!object){object=unitreeGltf.scene.getObjectByName('unitree_'+name);if(object)unitreeLinks.set(name,object);}if(object){object.matrixAutoUpdate=false;object.matrix.copy(worlds[name]);object.matrixWorldNeedsUpdate=true;}}
      unitreeGltf.scene.updateMatrixWorld(true);root.dataset.unitreeJoints=String(Object.keys(mapped).length);
    }
    root.dataset.poseJointCount=String(Object.keys(pose).length);
    root.dataset.poseMode=$('.avp-mode').textContent;
  }
  function mode(name){$('.avp-mode').textContent=name;root.dataset.poseMode=name;}
  function stopLive(){recordRequested=false;if(recording)stopRecording();playing=false;live=false;liveTarget=null;$('[data-live]').checked=false;epoch++;}
  function jointControls(){
    const box=$('.avp-joints');box.replaceChildren();
    for(const j of data.joints.filter(j=>j.type!=='fixed')){
      const label=document.createElement('label');label.className='avp-joint';
      const title=document.createElement('span'), name=document.createElement('span'), value=document.createElement('output');name.textContent=j.child;title.append(name,value);
      const input=document.createElement('input');input.type='range';input.min=j.lower;input.max=j.upper;input.step=.005;input.value=pose[j.child]||0;input.setAttribute('aria-label',j.child);
      value.textContent=Number(input.value).toFixed(2);input.oninput=()=>{stopLive();updateMarkers(tracking,'Reference input · manual robot pose');mode('Manual pose');applyPose({...pose,[j.child]:Number(input.value)});value.textContent=Number(input.value).toFixed(2);status('Manual pose · radians');};label.append(title,input);box.append(label);
    }
  }
  async function solve(frame){
    const response=await fetch('/api/avp/pose',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(frame),signal:abort.signal});
    const result=await response.json();if(!response.ok||result.error)throw new Error(result.error||response.statusText);return result.pose;
  }
  function snapshot(){stopLive();if(!data)return;tracking=snapshotTracking;updateMarkers(tracking,'Snapshot');applyPose(snapshotPose);mode('Snapshot');jointControls();status('参考快照已载入 · 无需头显即可预览');}
  function download(blob,name){const url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
  async function loadFile(file){
    if(!file)return;stopLive();const request=++epoch;
    try{if(file.size>128*1024*1024)throw new Error('文件大小不能超过 128 MiB');const value=JSON.parse(await file.text());
      if(value.format===FORMAT){const imported=validateClip(value,data.joints,data.source.urdfSha256);if(disposed||request!==epoch)return;clip=imported;cursor=0;seek(0);void persistClip();status('已导入动作：'+file.name);return;}
      if(file.size>1000000)throw new Error('单帧姿态文件不能超过 1 MB');
      if(value.urdfSha256&&value.urdfSha256!==data.source.urdfSha256)throw new Error('Snapshot uses a different URDF; load its tracking data instead.');
      const next=value.pose||await solve(value.tracking||value);if(disposed||request!==epoch)return;
      applyPose(next);snapshotPose={...pose};snapshotTracking=value.tracking||(value.pose?null:value);tracking=snapshotTracking;updateMarkers(tracking,'Snapshot');mode('Snapshot');jointControls();status('Loaded '+file.name);
    }catch(error){if(!disposed)status(error.message,true);}
  }
  $('input[type=file]').onchange=e=>void loadFile(e.target.files[0]);
  $('[data-live]').onchange=e=>{if(!session){e.target.checked=false;void enterXR();return;}if(recording)stopRecording();playing=false;recordRequested=false;live=e.target.checked;epoch++;liveTarget=null;alignment=null;updateMarkers(tracking,live?'Waiting for WebXR':'Paused input');status(live?'Live armed · enter Vision Pro, then hold both hands in view.':'Live paused · pose held');};
  function recenter(){if(recording){status('请先结束录制，再重新校准。',true);return;}alignment=null;xrPlaced=false;epoch++;liveTarget=null;if(session&&live){tracking=null;updateMarkers(null,'Waiting to recenter');}status('Tracking origin resets on the next visible head frame.');}
  async function prepare(){
    const button=$('[data-action=prepare]');button.disabled=true;status('正在后台生成模型资源…');
    try{const response=await fetch('/api/jobs',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({sandbox:'avp_remote',example:'prepare_browser_scene',target:'local'}),signal:abort.signal});
      const job=await response.json();if(!response.ok)throw new Error(job.error||response.statusText);
      async function poll(){if(disposed)return;try{const res=await fetch('/api/jobs/'+job.id,{signal:abort.signal}),result=await res.json();
        if(['queued','running','starting'].includes(result.status)){refreshTimer=setTimeout(poll,1000);return;}
        if(result.status!=='succeeded')throw new Error(result.error||('Asset preparation failed: '+(result.log||'No build log').slice(-1200)));
        location.reload();
      }catch(e){if(!disposed){button.disabled=false;status(e.message,true);}}}void poll();
    }catch(e){if(!disposed){button.disabled=false;status(e.message,true);}}
  }
  async function verifyAssets(){
    const button=$('[data-action=verify]');button.disabled=true;
    try{const result=await (await fetch('/api/avp/assets',{signal:abort.signal})).json();if(!disposed)status(result.ready===true?'模型来源校验通过':result.ready===false?'模型来源已更新，请刷新模型资源':'来源校验暂不可用，当前模型仍可使用',result.ready===false);}
    catch(e){if(!disposed)status('来源校验暂不可用：'+e.message,true);}
    finally{if(!disposed)button.disabled=false;}
  }
  function beginCapture(){
    if(!session){void enterXR();return;}
    if(!live){live=true;playing=false;epoch++;liveTarget=null;rawTracking=null;lastXRFrameAt=0;$('[data-live]').checked=true;syncTimeline();return;}
    toggleRecord();
  }
  const actions={'xr-report':()=>download(new Blob([JSON.stringify(xrDiagnostics,null,2)],{type:'application/json'}),'avp-xr-diagnostics.json'),begin:beginCapture,verify:()=>void verifyAssets(),record:toggleRecord,play:togglePlay,start:()=>seek(0),export:exportClip,snapshot,prepare,capture:()=>{stopLive();snapshotPose={...pose};snapshotTracking=tracking;updateMarkers(tracking,'Snapshot');mode('Snapshot');jointControls();status('Current pose captured · save JSON to keep it');},reset:()=>{stopLive();tracking=null;updateMarkers(null,'No tracking · zero robot pose');applyPose({});mode('Zero pose');jointControls();status('Zero joint angles · source bind stance');},fit:resetView,recenter,
    save:()=>download(new Blob([JSON.stringify({version:1,urdfSha256:data.source.urdfSha256,pose,tracking},null,2)],{type:'application/json'}),'landau-pose.json'),
    png:()=>{if(renderer.xr.isPresenting){status('Leave immersive view before saving the desktop image.');return;}renderer.render(scene,camera);renderer.domElement.toBlob(blob=>{if(blob)download(blob,'landau-snapshot.png');});},xr:()=>void enterXR()};
  root.querySelectorAll('[data-action]').forEach(b=>b.onclick=()=>{try{actions[b.dataset.action]();}catch(e){status(e.message,true);}});

  // A small scene-space control strip remains usable inside immersive mode.
  const xrPanel=new THREE.Group();xrPanel.scale.setScalar(.68);xrPanel.visible=false;scene.add(xrPanel);
  const xrButtons=[];
  for(const [i,text] of ['Record','Play','Live / pause','Recenter','Exit'].entries()){
    const c=document.createElement('canvas');c.width=256;c.height=96;const ctx=c.getContext('2d');ctx.fillStyle='#245d53';ctx.fillRect(0,0,256,96);ctx.fillStyle='white';ctx.font='30px sans-serif';ctx.textAlign='center';ctx.fillText(text,128,60);
    const button=new THREE.Mesh(new THREE.PlaneGeometry(.22,.10),new THREE.MeshBasicMaterial({map:new THREE.CanvasTexture(c),side:THREE.DoubleSide}));button.position.x=(i-2)*.24;button.userData.action=text;xrPanel.add(button);xrButtons.push(button);
  }
  const hudCanvas=document.createElement('canvas');hudCanvas.width=1024;hudCanvas.height=224;
  const hudTexture=new THREE.CanvasTexture(hudCanvas);
  const xrHud=new THREE.Mesh(new THREE.PlaneGeometry(1.16,.20),new THREE.MeshBasicMaterial({map:hudTexture,side:THREE.DoubleSide,transparent:true}));
  xrHud.position.y=.16;xrPanel.add(xrHud);let hudText='';
  const raycaster=new THREE.Raycaster();
  let pointerStart=null;
  function pointerDown(e){pointerStart=[e.clientX,e.clientY];}
  function pointerUp(e){
    if(renderer.xr.isPresenting||!pointerStart)return;
    const delta=Math.hypot(e.clientX-pointerStart[0],e.clientY-pointerStart[1]);pointerStart=null;if(delta>5)return;
    const rect=renderer.domElement.getBoundingClientRect();raycaster.setFromCamera(new THREE.Vector2((e.clientX-rect.left)/rect.width*2-1,-(e.clientY-rect.top)/rect.height*2+1),camera);
    const layers=[trackingOnly.visible?soloMarkers:null,usd.visible?usdMarkers:null,urdf.visible?urdfMarkers:null].filter(layer=>layer?.visible);
    const hit=raycaster.intersectObjects(layers.map(layer=>layer.points))[0];
    if(hit){selectedMarker=hit.object.userData.trackingMarkers.records[hit.instanceId]?.name;markerDetail();}
  }
  renderer.domElement.addEventListener('pointerdown',pointerDown);renderer.domElement.addEventListener('pointerup',pointerUp);
  const xrInput=new XRButtonInput(scene,xrButtons,{
    enabled:button=>{const action=button.userData.action;return action==='Record'?(!savePending&&!recordRequested):action==='Play'?Boolean(clip?.frames.length&&!recording):action==='Recenter'?!recording:true;},
    activate:button=>{try{activateXR(button.userData.action);}catch(error){status(error.message,true);}}
  });
  function activateXR(action){
    root.dataset.xrSelection=action;
    if(action==='Record'){toggleRecord();return;}
    if(action==='Play'){togglePlay();return;}
    if(action==='Exit')void session.end();
    else if(action==='Live / pause'){if(recording)stopRecording();playing=false;recordRequested=false;live=!live;epoch++;liveTarget=null;$('[data-live]').checked=live;updateMarkers(tracking,live?'Waiting for WebXR':'Paused input');status(live?'Live tracking resumed':'Live paused · pose held');}
    else if(action==='Recenter')recenter();
  }
  let xrDiagnostics={version:20,state:'idle',errors:[],frames:0},lastDiagnosticsAt=0,frameFailures=0,renderFailures=0;
  try{xrDiagnostics=JSON.parse(localStorage.getItem('avp-xr-diagnostics'))||xrDiagnostics;}catch{}
  function saveXRDiagnostics(){
    const output=$('[data-xr-diagnostics]');if(output)output.textContent=JSON.stringify(xrDiagnostics,null,2);
    try{localStorage.setItem('avp-xr-diagnostics',JSON.stringify(xrDiagnostics));}catch{}
  }
  function xrFailure(stage,error){
    const message=String(error?.message||error).slice(0,240);
    const previous=xrDiagnostics.errors.at(-1);
    if(previous?.stage===stage&&previous.message===message)previous.count++;
    else {xrDiagnostics.errors.push({stage,message,count:1});if(xrDiagnostics.errors.length>8)xrDiagnostics.errors.shift();}
    root.dataset.xrLastError=stage+': '+message;
  }
  saveXRDiagnostics();
  renderer.domElement.addEventListener('webglcontextlost',event=>{
    event.preventDefault();xrFailure('context','WebGL context lost');xrDiagnostics.state='context-lost';saveXRDiagnostics();
    xrError='图形上下文已丢失，请刷新页面后重新进入沉浸模式。';
    if(session)void session.end().catch(()=>{});status(xrError,true);
  });
  async function enterXR(){
    if(xrStarting)return;
    try{if(session){await session.end();return;}
      xrStarting=true;xrError='';syncTimeline();
      const next=await navigator.xr.requestSession('immersive-vr',{requiredFeatures:['hand-tracking'],optionalFeatures:['local-floor']});
      if(disposed){await next.end();return;}session=next;rawTracking=null;lastXRFrameAt=0;xrPlaced=false;lastRender=0;frameFailures=renderFailures=0;
      xrDiagnostics={version:20,state:'starting',startedAt:new Date().toISOString(),userAgent:navigator.userAgent,scale:.65,frames:0,errors:[]};saveXRDiagnostics();
      next.addEventListener('selectstart',event=>{if(next.visibilityState==='visible')xrInput.start(event,renderer.xr.getReferenceSpace());});
      next.addEventListener('select',event=>{if(next.visibilityState==='visible')xrInput.select(event,renderer.xr.getReferenceSpace());});
      next.addEventListener('selectend',event=>xrInput.end(event));
      next.addEventListener('visibilitychange',()=>{if(next.visibilityState!=='visible'){xrInput.reset();if(recording)stopRecording();}});
      next.addEventListener('end',()=>{xrDiagnostics.state=xrDiagnostics.state==='context-lost'?'context-lost':'ended';xrDiagnostics.endedAt=new Date().toISOString();saveXRDiagnostics();if(recording)stopRecording();session=null;live=false;recordRequested=false;epoch++;liveTarget=null;alignment=null;if(disposed)return;$('[data-live]').checked=false;updateMarkers(tracking,'Last WebXR frame · paused');content.position.set(0,0,0);content.rotation.y=0;rawTracking=null;lastXRFrameAt=0;xrPanel.visible=false;xrInput.reset();controls.enabled=true;$('[data-action=xr]').textContent='开始头手追踪 ↗';$('[data-device-state]').textContent='已断开';syncTimeline();status(xrError||'沉浸模式已结束 · 动作已保留；运行状态可在设置中查看',Boolean(xrError));jointControls();},{once:true});
      let spaceType='local-floor';try{await next.requestReferenceSpace(spaceType);}catch{spaceType='local';}
      referenceSpaceType=spaceType;renderer.xr.setReferenceSpaceType(spaceType);
      controls.enabled=false;content.position.set(0,.65,-1.4);xrPanel.visible=true;
      await renderer.xr.setSession(next);xrDiagnostics.state='running';saveXRDiagnostics();live=true;playing=false;alignment=null;epoch++;$('[data-live]').checked=true;$('[data-device-state]').textContent='已连接';syncTimeline();$('[data-action=xr]').textContent='退出 Vision Pro';status('已进入沉浸模式。把双手举到胸前，看到追踪状态亮起后点击「开始录制」。');
    }catch(error){if(session)await session.end().catch(()=>{});xrError=error.name==='NotAllowedError'?'未获得权限。请重试并允许沉浸体验、手部追踪；若曾拒绝，请检查 Safari 的此网站权限。':'无法开启追踪：'+error.message;status(xrError,true);}
    finally{xrStarting=false;if(!disposed)syncTimeline();}
  }
  async function checkXR(){
    const note=$('[data-xr-note]');
    if(!window.isSecureContext){
      xrSupport='insecure';note.textContent='当前 HTTP 不能采集。请打开 HTTPS 录制页；首次使用需在局域网设置中信任证书。';
      void fetch('/api/connection',{signal:abort.signal}).then(r=>r.json()).then(info=>{if(disposed||!info.httpsUrl)return;const url=new URL(info.httpsUrl);if(url.protocol==='https:'){$('[data-open-secure]').href=url.origin+'/#/sandbox/avp_remote';$('[data-open-secure]').hidden=false;}}).catch(()=>{});
      syncTimeline();return;
    }
    if(!navigator.xr){xrSupport='unsupported';note.textContent='未检测到 WebXR。请使用 Vision Pro 的 Safari 打开此 HTTPS 页面。';syncTimeline();return;}
    try{const supported=await navigator.xr.isSessionSupported('immersive-vr');if(disposed)return;xrSupport=supported?'ready':'unsupported';note.textContent=supported?'点击「开始头手追踪」，在系统提示中允许沉浸体验和手部追踪。':'当前浏览器未开放沉浸模式。请在 Vision Pro Safari 中打开，并检查系统和 Safari 是否支持 WebXR。';}
    catch(e){if(!disposed){xrSupport='error';xrError=e.message;note.textContent=e.message;}}
    if(!disposed)syncTimeline();
  }
  function hasLiveHead(){return Boolean(session&&live&&rawTracking?.head&&performance.now()-lastXRFrameAt<600);}
  function captureGuide(){
    const head=hasLiveHead(),left=head&&Boolean(rawTracking?.left_arm),right=head&&Boolean(rawTracking?.right_arm);
    let title,hint,button='开始头手追踪 ↗',state='idle',disabled=false;
    if(!data){state='loading';title='正在加载采集所需的基础场景';hint='加载完后，点击这里开始追踪；无需等待全部对照模型。';disabled=true;}
    else if(xrStarting){state='connecting';title='请在系统提示中选择「允许」';hint='允许沉浸体验和手部追踪。进入后，双手放在胸前并保持可见。';button='等待系统权限…';disabled=true;}
    else if(!session){
      title='尚未开启实时追踪 · 当前仅为预览';hint='点击「开始头手追踪」→ 允许沉浸体验和手部追踪 → 举起双手。';
      if(xrSupport==='insecure'){state='insecure';title='当前是 HTTP 预览，不能采集动作';hint='点击右侧 HTTPS 入口；若提示证书问题，先完成「Vision Pro 局域网设置」。';disabled=true;}
      else if(xrSupport==='unsupported'){state='unsupported';title='当前浏览器没有提供沉浸式追踪';hint='请在 Vision Pro 的 Safari 打开 HTTPS 页面。普通电脑可预览、回放，但不能代替头显采集。';disabled=true;}
      else if(xrSupport==='checking'){title='正在检查 Safari 追踪支持…';disabled=true;}
      if(xrError){state='error';title='追踪未能启动';hint=xrError;button='重试开启追踪 ↗';}
    }else if(!live){state='paused';title=playing?'正在回放 · 实时追踪已暂停':'实时追踪已暂停';hint='点击「录制新一段」可自动恢复追踪并开始录制。';button='恢复实时追踪';}
    else if(!head){state='waiting';title=recording?'正在录制 · 暂时丢失追踪':'已进入沉浸模式，等待头部追踪';hint='请保持 Safari 沉浸界面处于前台；若一直无数据，退出沉浸后重新开启。';button=recording?'结束并保存':'等待追踪数据…';disabled=!recording;}
    else {state=recording?'recording':left&&right?'tracking':'hands-missing';title=recording?'正在录制动作':left&&right?'头部和双手正在实时追踪':'头部已追踪 · 等待手部';hint=recording?(left&&right?'正在保存头部与双手数据。点击「结束并保存」完成本段录制。':'正在录制；部分手部数据缺失，请把双手举到胸前。缺失数据会如实保留。'):left&&right?'已连接不等于已录制。点击「开始录制」，结束后可在底部回放。':'双手放到胸前、张开手指，避免遮挡。未检测到的手不会被记录；请确认已允许手部追踪。';button=recording?'结束并保存':'开始录制动作 ●';}
    if(recordRequested){title='准备录制新一段 · 等待实时追踪';hint='检测到头部数据后自动开始；上一段保留在录制列表。';button='等待追踪…';disabled=true;}
    else if(clip&&!recording&&head){button='录制新一段 ●';if(!savePending)hint='上一段已结束；可直接录制新一段，或在底部选择历史动作回放。';}
    $('[data-capture-guide]').dataset.state=state;root.dataset.captureState=state;
    $('[data-guide-title]').textContent=title;$('[data-guide-hint]').textContent=hint;
    $('[data-action=begin]').textContent=button;$('[data-action=begin]').disabled=disabled||savePending;
    $('[data-action=xr]').disabled=!data||xrStarting||(!session&&xrSupport!=='ready');
    $('[data-live]').disabled=!session;
    $('[data-device-state]').textContent=session?(head?'追踪中':'已进入'):'未连接';
    for(const [name,ok] of [['head',head],['left',left],['right',right]]){const el=$(`[data-sensor=${name}]`);el.dataset.active=String(Boolean(ok));el.textContent=({head:'头部',left:'左手',right:'右手'})[name]+'：'+(ok?'追踪中':session&&live?'未检测到':'未采集');}
    for(const [name,done] of [['connect',Boolean(session)],['hands',left&&right],['record',recording]])$(`[data-step=${name}]`).classList.toggle('done',Boolean(done));
    const message=`${title}\n头部 ${head?'●':'—'}   左手 ${left?'●':'—'}   右手 ${right?'●':'—'} · ${root.dataset.poseHz||'0'} Hz\n${recording?'录制中 '+timecode(cursor):left&&right?(clip?'看向「录制新一段」并捏合；上段已保留':'头部准星高亮按钮后捏合；也可直接看向按钮捏合'):'将双手举到胸前，等待手部状态亮起'}`;
    if(hudText!==message){hudText=message;const ctx=hudCanvas.getContext('2d');ctx.clearRect(0,0,1024,224);ctx.fillStyle='#102a24ee';ctx.fillRect(0,0,1024,224);ctx.textAlign='center';ctx.font='34px system-ui,sans-serif';ctx.fillStyle=recording?'#ffb5c1':head?'#a7ead2':'#ffda91';message.split('\n').forEach((line,i)=>ctx.fillText(line,512,53+i*65));hudTexture.needsUpdate=true;}
  }
  function xrTracking(frame){
    const ref=renderer.xr.getReferenceSpace(),viewer=frame.getViewerPose(ref);if(!viewer){rawTracking=null;return null;}
    if(!xrPlaced){
      const m=new THREE.Matrix4().fromArray(viewer.transform.matrix),position=new THREE.Vector3().setFromMatrixPosition(m);
      const forward=new THREE.Vector3(0,0,-1).transformDirection(m);forward.y=0;if(forward.lengthSq()<.001)forward.set(0,0,-1);forward.normalize();
      const yaw=Math.atan2(-forward.x,-forward.z);
      content.position.copy(position).addScaledVector(forward,1.8);content.position.y=position.y-.95;content.rotation.y=yaw;
      xrPanel.position.copy(position).addScaledVector(forward,1.55);xrPanel.position.y=position.y+.35;xrPanel.rotation.y=yaw;xrPlaced=true;
    }
    lastXRFrameAt=performance.now();
    const head=nativeMatrix(viewer.transform.matrix);rawTracking={head:rows(head)};
    if(!alignment)alignment=fromRows(data.snapshot.head).multiply(head.clone().invert());
    const output={head:rows(alignment.clone().multiply(head))};
    for(const source of session.inputSources){
      if(!source.hand||!['left','right'].includes(source.handedness))continue;
      const stack=[],rawStack=[];
      for(const name of XR_HAND_NAMES){const joint=source.hand.get(name),p=joint&&frame.getJointPose(joint,ref);if(!p){stack.length=0;break;}const raw=nativeMatrix(p.transform.matrix),aligned=alignment.clone().multiply(raw);rawStack.push(rows(raw));stack.push(rows(aligned));if(name==='wrist'){rawTracking[source.handedness+'_wrist']=rows(raw);output[source.handedness+'_wrist']=rows(aligned);}}
      if(stack.length===25){output[source.handedness+'_arm']=stack;rawTracking[source.handedness+'_arm']=rawStack;}
    }return output;
  }
  const resize=new ResizeObserver(()=>{if(disposed||renderer.xr.isPresenting)return;const r=stage.getBoundingClientRect();renderer.setSize(r.width,r.height,false);camera.aspect=r.width/r.height;camera.updateProjectionMatrix();resetView();});resize.observe(stage);
  renderer.setAnimationLoop((time,frame)=>{
    if(disposed)return;
    const frameStarted=performance.now();
    const active=Boolean(frame&&session&&session.visibilityState==='visible'&&renderer.xr.getReferenceSpace());
    const sampled=active?sampleXRFrame(frame,renderer.xr.getReferenceSpace(),xrFailure):null;
    try {
    // Keep first-person hands live during playback and paused robot capture.
    const handsFrame=active&&$('[data-self-hands]').checked?sampled:null;
    if(handsFrame||selfHands.visible)selfHands.updateXR(handsFrame,session?.inputSources,session?renderer.xr.getReferenceSpace():null,XR_HAND_NAMES);
    const handCount=String(selfHands.records.length);if(root.dataset.immersiveHandMarkers!==handCount)root.dataset.immersiveHandMarkers=handCount;
    // Read markers and solve the newest pose in the same XR frame, without HTTP.
    // Missing tracking clears points immediately rather than displaying stale data.
    if(active&&live&&data){
      const current=xrTracking(sampled);
      tracking=current;updateMarkers(current,current?'Live WebXR':'Tracking unavailable');
      if(recordRequested&&current)startRecording();
      if(!current){epoch++;liveTarget=null;poseRateStart=poseRateFrames=0;root.dataset.poseHz='0';status('Tracking unavailable · robot pose held');}
      else {
        const started=performance.now();liveTarget=liveSolver.solve(current);
        root.dataset.poseSolveMs=(performance.now()-started).toFixed(2);
        if(!poseRateStart)poseRateStart=performance.now();poseRateFrames++;
        if(performance.now()-poseRateStart>=500){root.dataset.poseHz=String(Math.round(poseRateFrames*1000/(performance.now()-poseRateStart)));poseRateFrames=0;poseRateStart=performance.now();}
        root.dataset.poseSolver='local';root.dataset.livePoseUpdates=String(Number(root.dataset.livePoseUpdates||0)+1);
      }
    }
    if(!live){poseRateStart=poseRateFrames=0;root.dataset.poseHz='0';}
    if(active&&xrPanel.visible){xrInput.update(sampled,session.inputSources,renderer.xr.getReferenceSpace());root.dataset.xrHover=xrInput.hover?.userData.action||'';}
    if(active&&live&&liveTarget){const alpha=1-Math.exp(-Math.max(0,Math.min((time-lastRender)/1000,.1))*45);const smooth={};for(const j of data.joints)if(j.type!=='fixed')smooth[j.child]=(pose[j.child]||0)+((liveTarget[j.child]||0)-(pose[j.child]||0))*alpha;applyPose(smooth);}
    if(recording){
      const elapsed=(performance.now()-recordStart)/1000;
      if(elapsed>MAX_SECONDS||clip.frames.length>=MAX_FRAMES)stopRecording();
      else if(active&&rawTracking?.head){appendFrame(clip,elapsed,pose,tracking,rawTracking,alignment?rows(alignment):null);cursor=elapsed;}
    }
    if(playing&&clip){
      cursor+=Math.max(0,(time-lastRender)/1000)*Number($('[data-speed]').value);
      const duration=clip.frames.at(-1).t;
      if(cursor>=duration){if($('[data-loop]').checked&&duration>0)cursor%=duration;else{cursor=duration;playing=false;}}
      showFrame(cursor);
    }
    if(time-lastUi>100){if(live&&hasLiveHead()){mode(recording?'Recording':'Live WebXR');status(`实时跟随 ${root.dataset.poseHz||'…'} Hz · 本地求解 ${root.dataset.poseSolveMs||'0'} ms`);}syncTimeline();markerDetail();lastUi=time;}
    frameFailures=0;
    } catch(error) {
      xrFailure('update',error);rawTracking=null;tracking=null;liveTarget=null;
      // An invalid sample is skipped; repeated failures pause capture but leave
      // the renderer and controls running so the user can exit or retry.
      if(++frameFailures>=3){live=false;recordRequested=false;if(recording)stopRecording();xrError='追踪更新出错，已暂停采集。请退出后重试；设置中可导出诊断。';status(xrError,true);}
    }
    lastRender=time;
    try {
      if(!renderer.xr.isPresenting)controls.update();renderer.render(scene,camera);renderFailures=0;
    } catch(error) {
      xrFailure('render',error);
      if(++renderFailures===3&&session){xrError='沉浸渲染出错，已保存诊断，请刷新后重试。';saveXRDiagnostics();void session.end().catch(()=>{});}
    }
    if(active){
      xrDiagnostics.frames++;xrDiagnostics.frameCpuMs=Number((performance.now()-frameStarted).toFixed(2));
      xrDiagnostics.drawCalls=renderer.info.render.calls;xrDiagnostics.triangles=renderer.info.render.triangles;
      if(time-lastDiagnosticsAt>2000){xrDiagnostics.poseHz=Number(root.dataset.poseHz||0);saveXRDiagnostics();lastDiagnosticsAt=time;}
    }
  });
  const downloads=new Map(),loadStarted=performance.now();
  function loadProgress(){
    if(disposed||root.dataset.modelsReady!==undefined)return;
    $('[data-load-progress]').textContent=[...downloads].map(([name,value])=>`${name} ${value}`).join(' · ');
  }
  async function fetchAsset(path,name,revision){
    const controller=new AbortController();
    const cancel=()=>controller.abort();abort.signal.addEventListener('abort',cancel,{once:true});
    const timer=setTimeout(cancel,45000);
    downloads.set(name,'下载中');loadProgress();
    try{
      // Content-addressed browser storage also works when HTTP cache revalidation
      // is disabled. A changed build hash can never reuse an old model.
      let cache;
      const cacheKey=revision?asset(path)+'&v='+revision:null;
      const hash=async bytes=>Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',bytes)),v=>v.toString(16).padStart(2,'0')).join('');
      if(cacheKey&&window.isSecureContext&&globalThis.caches){
        try{
          cache=await caches.open('avp-models-v1');const cached=await cache.match(cacheKey);
          if(cached){const bytes=await cached.arrayBuffer();if(await hash(bytes)===revision){downloads.set(name,'本地缓存');loadProgress();return bytes;}await cache.delete(cacheKey);}
        }catch{cache=null;} // Quota/private-mode restrictions must not block loading.
      }
      const response=await fetch(asset(path),{signal:controller.signal});
      if(!response.ok)throw new Error(`${name}: HTTP ${response.status}`);
      const reader=response.body?.getReader();
      let bytes;
      if(reader){
        const chunks=[];let length=0;
        for(;;){const {done,value}=await reader.read();if(done)break;chunks.push(value);length+=value.length;downloads.set(name,`${(length/1048576).toFixed(1)} MB`);loadProgress();}
        bytes=new Uint8Array(length);let offset=0;for(const chunk of chunks){bytes.set(chunk,offset);offset+=chunk.length;}
      }else bytes=new Uint8Array(await response.arrayBuffer());
      if(cache){
        if(await hash(bytes)!==revision)throw new Error(`${name} 版本已变化，请刷新页面`);
        try{
          await cache.put(cacheKey,new Response(bytes));
          const old=await cache.keys();for(const key of old){const url=new URL(key.url);if(url.searchParams.get('path')==='algorithms/avp_remote/'+path&&url.searchParams.get('v')!==revision)await cache.delete(key);}
        }catch{} // A full cache still permits this model to render.
      }
      downloads.set(name,'已下载');loadProgress();return bytes.buffer;
    }catch(error){downloads.set(name,'加载失败');loadProgress();throw new Error(error.name==='AbortError'?`${name} 下载超时，请检查网络后刷新`:error.message);}
    finally{clearTimeout(timer);abort.signal.removeEventListener('abort',cancel);}
  }
  async function load(){
    try{
      data=JSON.parse(new TextDecoder().decode(await fetchAsset('outputs/web_scene/scene.json','基础场景')));if(disposed)return;allowedJoints=null;
      if(!data.trackingTransform)throw new Error('请刷新模型资源以补齐追踪坐标变换');
      markerTransform=fromRows(data.trackingTransform);liveSolver=new LiveRetargeter(data);
      for(const [name,vertices] of Object.entries(data.meshes)){const geometry=new THREE.BufferGeometry();geometry.setAttribute('position',new THREE.Float32BufferAttribute(vertices,3));geometry.computeVertexNormals();geometries.set(name,geometry);}
      for(const link of data.links){const group=new THREE.Group();group.matrixAutoUpdate=false;urdf.add(group);visuals.set(link.name,group);for(const v of link.visuals){const mesh=new THREE.Mesh(geometries.get(v.mesh),robotMaterial);mesh.matrixAutoUpdate=false;mesh.matrix.copy(originMatrix(v).multiply(new THREE.Matrix4().makeScale(...v.scale)));group.add(mesh);}}
      snapshotPose=data.snapshotPose;snapshotTracking=data.snapshot;snapshot();
      $('.avp-source').textContent=`${data.source.source||data.source.urdf}\nURDF SHA-256 ${data.source.urdfSha256}\n${data.source.triangles.toLocaleString()} URDF triangles`;
      const headAsset=data.source.meshOverrides?.['landau_v10/head_x.stl'];
      if(headAsset)$('.avp-source').textContent+=`\nHead STL · ${headAsset.triangles} triangles\nSHA-256 ${headAsset.sha256}`;
      root.querySelectorAll('button:not([data-action=xr]),input').forEach(el=>el.disabled=false);
      layout();resetView();root.dataset.ready='true';syncTimeline();void checkXR();
      // Recording and URDF are usable now. Decode compressed geometry locally, without a CDN.
      draco.preload();
      const usdLoad=(async()=>{
        const model=await modelLoader.parseAsync(await fetchAsset('outputs/web_scene/character.glb','Landau USD',data.source.glbSha256),'');
        if(disposed){disposeObject(model.scene);return;}
        const modelBones=[];model.scene.traverse(o=>{if(o.isBone)modelBones.push(o);if(o.isSkinnedMesh)o.frustumCulled=false;});
        if(modelBones.length!==data.source.bones){disposeObject(model.scene);throw new Error('USD conversion lost skeleton joints');}
        gltf=model;bones.push(...modelBones);usd.add(model.scene);applyPose(pose);root.dataset.usdBones=String(bones.length);
        downloads.set('Landau USD','就绪');loadProgress();
      })();
      const unitreeLoad=(async()=>{
        const [metadata,buffer]=await Promise.all([fetchAsset('outputs/web_scene/unitree.json','Unitree 关节'),fetchAsset('outputs/web_scene/unitree.glb','Unitree 网格',data.source.unitreeGlbSha256)]);
        const model=await modelLoader.parseAsync(buffer,'');
        if(disposed){disposeObject(model.scene);return;}
        unitreeData=JSON.parse(new TextDecoder().decode(metadata));unitreeGltf=model;unitreeLinks.clear();
        unitreeModel.add(model.scene);model.scene.position.z=unitreeData.rootHeight;unitree.scale.setScalar(.52);applyPose(pose);
        downloads.set('Unitree 网格','就绪');loadProgress();
      })();
      void Promise.allSettled([usdLoad,unitreeLoad]).then(results=>{
        if(disposed)return;
        const failed=results.filter(r=>r.status==='rejected');
        root.dataset.modelsReady=String(!failed.length);
        if(failed.length)$('[data-load-progress]').textContent='部分模型未加载；头手采集和 URDF 可用。'+failed.map(r=>r.reason.message).join('；');
        else {root.dataset.modelLoadSeconds=((performance.now()-loadStarted)/1000).toFixed(1);$('[data-load-progress]').textContent=`四视图已就绪 · 加载 ${root.dataset.modelLoadSeconds} 秒 · 压缩模型 / 本地解码`;}
      });
      try{const stored=await clipStorage();if(!disposed&&stored&&!clip){clip=validateClip(stored,data.joints,data.source.urdfSha256);await clipStorage(clip);if(disposed)return;$('[data-save-note]').textContent='已恢复上次动作 · 点击播放或拖动时间轴';syncTimeline();}}catch(e){if(!disposed)$('[data-save-note]').textContent='本地动作未恢复：'+e.message;}
      await refreshTakes();
    }catch(e){if(!disposed)status(e.message,true);}
  }
  function timecode(seconds){const t=Math.max(0,seconds||0);return `${String(Math.floor(t/60)).padStart(2,'0')}:${(t%60).toFixed(2).padStart(5,'0')}`;}
  function syncTimeline(){
    captureGuide();
    const n=clip?.frames.length||0,duration=n?clip.frames.at(-1).t:0;
    $('[data-action=record]').disabled=!data||(!recording&&(!session||savePending||recordRequested));
    $('[data-action=record]').innerHTML=recording?'<span>■</span> 结束录制':savePending?'保存中…':recordRequested?'等待追踪…':clip?'<span>●</span> 录制新一段':'<span>●</span> 开始录制';
    $('[data-takes]').disabled=recording||savePending||recordRequested;
    $('.avp-shell').classList.toggle('is-recording',recording);
    for(const action of ['play','start','export'])$(`[data-action=${action}]`).disabled=!n||recording;
    $('[data-action=play]').textContent=playing?'Ⅱ':'▶';$('[data-action=play]').setAttribute('aria-label',playing?'暂停动作':'播放动作');
    $('[data-seek]').disabled=!n||recording;$('[data-seek]').max=String(Math.max(duration,.001));$('[data-seek]').value=String(cursor);
    $('[data-seek]').style.setProperty('--progress',`${duration?Math.min(100,cursor/duration*100):0}%`);
    $('[data-play-time]').textContent=timecode(cursor);$('[data-duration]').textContent=timecode(duration);
    $('[data-mid-time]').textContent=timecode(duration/2);$('[data-end-time]').textContent=timecode(duration);
    $('[data-clip-name]').textContent=clip?`Take / ${clip.name}`:'尚未录制动作';
    $('[data-clip-meta]').textContent=clip?`${recording?'● REC · ':''}${n} 帧 · ${timecode(duration)} · 目标 30 Hz`:'进入 Vision Pro 后开始录制 · 最长 60 秒';
    root.dataset.recording=String(recording);root.dataset.playing=String(playing);root.dataset.clipFrames=String(n);
    for(const button of xrButtons){const key=button.userData.action,label=key==='Record'?(recording?'结束 '+timecode(cursor):savePending?'保存中…':recordRequested?'等待追踪…':clip?'录制新一段':'开始录制'):key==='Play'?(playing?'暂停回放':'回放'):key==='Live / pause'?(live?'暂停追踪':'恢复追踪'):key==='Recenter'?'重新校准':'退出沉浸';
      if(button.userData.label===label)continue;button.userData.label=label;
      const texture=button.material.map,c=texture.image,ctx=c.getContext('2d');ctx.fillStyle=key==='Record'&&recording?'#b43c4c':'#245d53';ctx.fillRect(0,0,c.width,c.height);ctx.fillStyle='white';ctx.font='26px sans-serif';ctx.textAlign='center';ctx.fillText(label,128,60);texture.needsUpdate=true;
    }
  }
  function showFrame(t){const f=sampleClip(clip,t);if(!f)return;applyPose(f.pose);tracking=f.tracking;updateMarkers(tracking,'回放 · frame '+(f.index+1));mode('Playback');}
  function seek(t){if(recording||!clip?.frames.length)return;stopLive();cursor=Math.max(0,Math.min(t,clip.frames.at(-1).t));showFrame(cursor);jointControls();syncTimeline();}
  $('[data-seek]').oninput=e=>seek(Number(e.target.value));
  function togglePlay(){if(recording||!clip?.frames.length)return;const resume=!playing;stopLive();if(resume&&cursor>=clip.frames.at(-1).t)cursor=0;playing=resume;lastRender=performance.now();showFrame(cursor);syncTimeline();}
  function toggleRecord(){
    if(recording){stopRecording();return;}
    if(!session){status('请在 Vision Pro Safari 中进入沉浸模式后录制。',true);return;}
    if(savePending)return;
    if(!saved&&clip?.frames.length){status('上一段尚未保存，正在重试；保存完成后可录制新一段。',true);void persistClip();return;}
    if(!hasLiveHead()){
      recordRequested=true;playing=false;live=true;liveTarget=null;rawTracking=null;lastXRFrameAt=0;epoch++;$('[data-live]').checked=true;
      status('已准备录制新一段，检测到实时头部数据后自动开始。');syncTimeline();return;
    }
    startRecording();
  }
  function startRecording(){
    recordRequested=false;
    playing=false;live=true;liveTarget=null;epoch++;$('[data-live]').checked=true;
    clip=createClip(data.source.urdfSha256,referenceSpaceType);
    recording=true;saved=false;cursor=0;recordStart=performance.now();mode('Recording');
    $('[data-save-note]').textContent='正在录制头部与手部数据 · 结束后自动保存 · 最长 60 秒';syncTimeline();
  }
  function stopRecording(){if(!recording)return;recording=false;playing=false;mode('Recorded');if(clip?.frames.length){cursor=clip.frames.at(-1).t;void persistClip();}syncTimeline();}
  async function persistClip(){
    savePending=true;saved=false;const current=clip;
    try{await clipStorage(current);if(!disposed&&clip===current){saved=true;$('[data-save-note]').textContent='已保存 · 可直接录制新一段；在录制列表中切换历史动作。建议导出 JSON 保留副本。';await refreshTakes();}}
    catch(e){if(!disposed)$('[data-save-note]').textContent='自动保存失败，请导出 JSON：'+e.message;}
    finally{savePending=false;if(!disposed)syncTimeline();}
  }
  async function refreshTakes(){
    const takes=await clipSummaries();if(disposed)return;
    const select=$('[data-takes]');select.replaceChildren(new Option('已保存的录制',''));
    for(const take of takes)select.add(new Option(`${take.name} · ${take.frames} 帧`,take.key));
    select.value=clip?'take:'+clip.createdAt:'';
  }
  $('[data-takes]').onchange=async e=>{
    const key=e.target.value;if(!key||recording||savePending)return;
    stopLive();const request=epoch;
    try{const stored=await clipStorage(undefined,key);if(disposed||recording||request!==epoch)return;clip=validateClip(stored,data.joints,data.source.urdfSha256);saved=true;cursor=0;showFrame(0);syncTimeline();}
    catch(error){status('无法读取录制：'+error.message,true);}
  };
  function exportClip(){if(recording||!clip?.frames.length)return;download(new Blob([JSON.stringify(clip)],{type:'application/json'}),'landau-motion-'+clip.createdAt.replace(/[:.]/g,'-')+'.json');}
  function beforeUnload(e){if(recording||savePending||!saved){e.preventDefault();e.returnValue='';}}
  function visibility(){if(document.hidden&&recording)stopRecording();}
  window.addEventListener('beforeunload',beforeUnload);document.addEventListener('visibilitychange',visibility);
  function disposeObject(object){object.traverse(o=>{o.geometry?.dispose();for(const material of [o.material].flat().filter(Boolean)){for(const value of Object.values(material))if(value?.isTexture)value.dispose();material.dispose();}});}
  void load();
  return {destroy(){if(recording)stopRecording();window.removeEventListener('beforeunload',beforeUnload);document.removeEventListener('visibilitychange',visibility);disposed=true;epoch++;clearTimeout(refreshTimer);abort.abort();if(session)void session.end();renderer.setAnimationLoop(null);resize.disconnect();renderer.domElement.removeEventListener('pointerdown',pointerDown);renderer.domElement.removeEventListener('pointerup',pointerUp);controls.dispose();draco.dispose();disposeObject(scene);renderer.dispose();renderer.forceContextLoss();root.replaceChildren();}};
}
