import * as THREE from 'three';
import {OrbitControls} from '/vendor/three/modules/OrbitControls.js';
import {GLTFLoader} from '/vendor/three/modules/GLTFLoader.js';
import {originMatrix, worldTransforms, XR_HAND_NAMES, nativeMatrix, rows, fromRows} from '/api/artifact?path=algorithms/avp_remote/gui/pose-math.js';
import {TrackingMarkers} from '/api/artifact?path=algorithms/avp_remote/gui/tracking-markers.js';
const asset = path => `/api/artifact?path=algorithms/avp_remote/${path}`;

export function mount(root) {
  root.innerHTML = `<link rel="stylesheet" href="${asset('gui/viewer.css')}">
  <div class="avp-shell"><header class="avp-toolbar"><h2>Landau · spatial pose studio</h2><span class="avp-mode">Snapshot</span>
  <button data-action="snapshot">Snapshot mode</button><button data-action="reset">Zero pose</button><button data-action="fit">Reset view</button>
  <button class="avp-primary" data-action="xr" disabled>Open in Vision Pro</button></header>
  <div class="avp-grid"><div class="avp-stage"><div class="avp-caption">Drag to orbit · scroll to zoom · right-drag to pan</div></div>
  <aside class="avp-side"><div><h3>Pose source</h3><p class="avp-status" role="status">Loading current Landau assets…</p></div>
  <div><h3>Scene</h3><select aria-label="Model display"><option value="both">USD character + URDF comparison</option><option value="usd">USD character</option><option value="urdf">URDF meshes</option><option value="tracking">Tracking markers only</option></select></div>
  <div class="avp-tracking"><h3>Tracking input · before IK</h3><label><input type="checkbox" data-markers checked> Show tracking markers</label><label><input type="checkbox" data-marker-lines checked> Hand skeleton lines</label><label><input type="checkbox" data-marker-axes checked> Head and wrist axes</label><p class="avp-marker-legend"><span>● Head</span><span>● Left</span><span>● Right</span></p><p data-tracking-counts role="status">No tracking frame</p><p class="avp-marker-help">Click a point to inspect its input and mapped coordinates.</p><pre data-marker-detail>No point selected</pre><p class="avp-capability-note">Measured: head + hands. Arm IK: estimated. Face expressions and full-body tracking: unavailable from Vision Pro APIs.</p></div>
  <div><h3>Vision Pro</h3><label><input type="checkbox" data-live> Follow my head & hands</label><p data-xr-note>Checking immersive WebXR…</p><button data-action="recenter">Recenter tracking</button></div>
  <div><h3>Snapshot</h3><label class="avp-file">Load snapshot JSON<input type="file" accept=".json,application/json"></label><p>Inspect and edit a saved pose without a headset.</p><button data-action="capture">Capture current pose</button><button data-action="save">Save pose JSON</button><button data-action="png">Save view PNG</button></div>
  <div><h3>Joint controls</h3><div class="avp-joints"></div></div><details><summary>Asset provenance</summary><p class="avp-source"></p><button data-action="prepare">Refresh latest assets</button></details></aside></div>
  <footer class="avp-footer">Pose visualization · head, arms and fingers · legs stay at explicit defaults · no physics simulation</footer></div>`;
  root.querySelectorAll('button:not([data-action=prepare]),input').forEach(el=>el.disabled=true);
  const $ = sel => root.querySelector(sel), stage = $('.avp-stage');
  let disposed = false, data, gltf, pose = {}, tracking, snapshotPose, snapshotTracking;
  let refreshTimer, liveTarget=null, lastRender=0, markerTransform=null, selectedMarker='head', trackingLabel="Snapshot";
  let live = false, busy = false, epoch = 0, lastSend = 0, alignment = null, session = null;
  const abort = new AbortController();
  const status = (message, error = false) => { $('.avp-status').textContent = message; $('.avp-status').classList.toggle('avp-error', error); };
  const renderer = new THREE.WebGLRenderer({antialias:true, preserveDrawingBuffer:true});
  renderer.setPixelRatio(Math.min(devicePixelRatio,2)); renderer.xr.enabled = true;
  renderer.xr.setReferenceSpaceType('local-floor'); renderer.outputColorSpace = THREE.SRGBColorSpace;
  stage.prepend(renderer.domElement);
  const scene = new THREE.Scene(); scene.background = new THREE.Color('#dde6e2');
  const camera = new THREE.PerspectiveCamera(40,1,.01,100); camera.position.set(1.1,1.1,2.2);
  const controls = new OrbitControls(camera, renderer.domElement); controls.enableDamping=true; controls.target.set(0,.4,0);
  scene.add(new THREE.HemisphereLight(0xffffff,0x70867b,2.8));
  const sun = new THREE.DirectionalLight(0xffffff,3);sun.position.set(2,4,3);scene.add(sun);
  const grid = new THREE.GridHelper(8,80,0x9fb6ad,0xc2d0ca);scene.add(grid);
  const content = new THREE.Group();scene.add(content);
  const usd = new THREE.Group(), urdf = new THREE.Group(), trackingOnly=new THREE.Group();
  usd.rotation.x=urdf.rotation.x=trackingOnly.rotation.x=-Math.PI/2;content.add(usd,urdf,trackingOnly);
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
    for(const layer of markerLayers)layer.update(frame,markerTransform);
    const records=soloMarkers.records;
    const count=side=>records.filter(r=>r.side===side).length;
    const text=`${label} · head ${count('head')} · left ${count('left')} · right ${count('right')}`;
    if($('[data-tracking-counts]').textContent!==text)$('[data-tracking-counts]').textContent=text;
    root.dataset.trackingMarkers=String(records.length);root.dataset.trackingSource=label;
    markerDetail();
  }
  markerOptions();
  const visuals = new Map(), bones = [], geometries = new Map();
  const robotMaterial = new THREE.MeshStandardMaterial({color:0x759b99,roughness:.74,metalness:.12,side:THREE.DoubleSide});
  const labels = [];
  function label(text, x) {
    const c=document.createElement('canvas');c.width=512;c.height=96;
    const ctx=c.getContext('2d');ctx.fillStyle='#244c45';ctx.font='28px sans-serif';ctx.textAlign='center';ctx.fillText(text,256,55);
    const map=new THREE.CanvasTexture(c), sprite=new THREE.Sprite(new THREE.SpriteMaterial({map,depthTest:false}));
    sprite.scale.set(.55,.103,1);sprite.position.set(x,1.12,0);content.add(sprite);labels.push(sprite);return sprite;
  }
  const usdLabel=label('USD · skinned character',-.42), urdfLabel=label('URDF · current meshes',.42);
  function layout(){const choice=$('select').value;usd.visible=['both','usd'].includes(choice);urdf.visible=['both','urdf'].includes(choice);trackingOnly.visible=choice==='tracking';usd.position.x=choice==='both'?-.42:0;urdf.position.x=choice==='both'?.42:0;usdLabel.visible=usd.visible;urdfLabel.visible=urdf.visible;usdLabel.position.x=usd.position.x;urdfLabel.position.x=urdf.position.x;}
  function resetView(){
    controls.target.set(0,.4,0);camera.position.set(1.1,1.1,2.2);
    if($('select').value==='tracking'&&soloMarkers.records.length){
      trackingOnly.updateWorldMatrix(true,false);
      const bounds=new THREE.Box3().setFromPoints(soloMarkers.records.map(r=>r.position.clone().applyMatrix4(trackingOnly.matrixWorld)));
      bounds.expandByScalar(.07);bounds.getCenter(controls.target);
      const radius=bounds.getSize(new THREE.Vector3()).length()/2;
      const halfFov=Math.min(THREE.MathUtils.degToRad(camera.fov/2),Math.atan(Math.tan(THREE.MathUtils.degToRad(camera.fov/2))*camera.aspect));
      camera.position.copy(controls.target).add(new THREE.Vector3(.25,.12,1).normalize().multiplyScalar(radius/Math.sin(halfFov)));
    }
    controls.update();
  }
  $('select').onchange=()=>{layout();if(!renderer.xr.isPresenting)resetView();};layout();
  function applyPose(next) {
    if(!data)return;
    const allowed = new Map(data.joints.map(j=>[j.child,j]));
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
    root.dataset.poseJointCount=String(Object.keys(pose).length);
    root.dataset.poseMode=$('.avp-mode').textContent;
  }
  function mode(name){$('.avp-mode').textContent=name;root.dataset.poseMode=name;}
  function stopLive(){live=false;liveTarget=null;$('[data-live]').checked=false;epoch++;}
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
  function snapshot(){stopLive();if(!data)return;tracking=snapshotTracking;updateMarkers(tracking,'Snapshot');applyPose(snapshotPose);mode('Snapshot');jointControls();status('Snapshot loaded · no headset needed');}
  function download(blob,name){const url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
  async function loadFile(file){
    if(!file)return;stopLive();const request=++epoch;
    try{if(file.size>1000000)throw new Error('Snapshot must be smaller than 1 MB');const value=JSON.parse(await file.text());
      if(value.urdfSha256&&value.urdfSha256!==data.source.urdfSha256)throw new Error('Snapshot uses a different URDF; load its tracking data instead.');
      const next=value.pose||await solve(value.tracking||value);if(disposed||request!==epoch)return;
      applyPose(next);snapshotPose={...pose};snapshotTracking=value.tracking||(value.pose?null:value);tracking=snapshotTracking;updateMarkers(tracking,'Snapshot');mode('Snapshot');jointControls();status('Loaded '+file.name);
    }catch(error){if(!disposed)status(error.message,true);}
  }
  $('input[type=file]').onchange=e=>void loadFile(e.target.files[0]);
  $('[data-live]').onchange=e=>{live=e.target.checked;epoch++;liveTarget=null;alignment=null;updateMarkers(tracking,live?'Waiting for WebXR':'Paused input');status(live?'Live armed · enter Vision Pro, then hold both hands in view.':'Live paused · pose held');};
  function recenter(){alignment=null;epoch++;liveTarget=null;if(session&&live){tracking=null;updateMarkers(null,'Waiting to recenter');}status('Tracking origin resets on the next visible head frame.');}
  async function prepare(){
    const button=$('[data-action=prepare]');button.disabled=true;status('Preparing current URDF + USD assets…');
    try{const response=await fetch('/api/jobs',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({sandbox:'avp_remote',example:'prepare_browser_scene',target:'local'}),signal:abort.signal});
      const job=await response.json();if(!response.ok)throw new Error(job.error||response.statusText);
      async function poll(){if(disposed)return;try{const res=await fetch('/api/jobs/'+job.id,{signal:abort.signal}),result=await res.json();
        if(['queued','running','starting'].includes(result.status)){refreshTimer=setTimeout(poll,1000);return;}
        if(result.status!=='succeeded')throw new Error(result.error||('Asset preparation failed: '+(result.log||'No build log').slice(-1200)));
        location.reload();
      }catch(e){if(!disposed){button.disabled=false;status(e.message,true);}}}void poll();
    }catch(e){if(!disposed){button.disabled=false;status(e.message,true);}}
  }
  const actions={snapshot,prepare,capture:()=>{stopLive();snapshotPose={...pose};snapshotTracking=tracking;updateMarkers(tracking,'Snapshot');mode('Snapshot');jointControls();status('Current pose captured · save JSON to keep it');},reset:()=>{stopLive();tracking=null;updateMarkers(null,'No tracking · zero robot pose');applyPose({});mode('Zero pose');jointControls();status('Zero joint angles · source bind stance');},fit:resetView,recenter,
    save:()=>download(new Blob([JSON.stringify({version:1,urdfSha256:data.source.urdfSha256,pose,tracking},null,2)],{type:'application/json'}),'landau-pose.json'),
    png:()=>{if(renderer.xr.isPresenting){status('Leave immersive view before saving the desktop image.');return;}renderer.render(scene,camera);renderer.domElement.toBlob(blob=>{if(blob)download(blob,'landau-snapshot.png');});},xr:()=>void enterXR()};
  root.querySelectorAll('[data-action]').forEach(b=>b.onclick=()=>{try{actions[b.dataset.action]();}catch(e){status(e.message,true);}});

  // A small scene-space control strip remains usable inside immersive mode.
  const xrPanel=new THREE.Group();xrPanel.position.set(0,1.0,-.85);xrPanel.visible=false;scene.add(xrPanel);
  const xrButtons=[];
  for(const [i,text] of ['Live / pause','Snapshot','Recenter','Exit'].entries()){
    const c=document.createElement('canvas');c.width=256;c.height=96;const ctx=c.getContext('2d');ctx.fillStyle='#245d53';ctx.fillRect(0,0,256,96);ctx.fillStyle='white';ctx.font='30px sans-serif';ctx.textAlign='center';ctx.fillText(text,128,60);
    const button=new THREE.Mesh(new THREE.PlaneGeometry(.22,.083),new THREE.MeshBasicMaterial({map:new THREE.CanvasTexture(c),side:THREE.DoubleSide}));button.position.x=(i-1.5)*.24;button.userData.action=text;xrPanel.add(button);xrButtons.push(button);
  }
  const raycaster=new THREE.Raycaster();
  let pointerStart=null;
  function pointerDown(e){pointerStart=[e.clientX,e.clientY];}
  function pointerUp(e){
    if(renderer.xr.isPresenting||!pointerStart)return;
    const delta=Math.hypot(e.clientX-pointerStart[0],e.clientY-pointerStart[1]);pointerStart=null;if(delta>5)return;
    const rect=renderer.domElement.getBoundingClientRect();raycaster.setFromCamera(new THREE.Vector2((e.clientX-rect.left)/rect.width*2-1,-(e.clientY-rect.top)/rect.height*2+1),camera);
    const layers=[$('select').value==='tracking'?soloMarkers:null,usd.visible?usdMarkers:null,urdf.visible?urdfMarkers:null].filter(layer=>layer?.visible);
    const hit=raycaster.intersectObjects(layers.map(layer=>layer.points))[0];
    if(hit){selectedMarker=hit.object.userData.trackingMarkers.records[hit.instanceId]?.name;markerDetail();}
  }
  renderer.domElement.addEventListener('pointerdown',pointerDown);renderer.domElement.addEventListener('pointerup',pointerUp);
  function selectXR(event){
    const ref=renderer.xr.getReferenceSpace(),p=event.frame?.getPose(event.inputSource.targetRaySpace,ref);if(!p)return;
    const matrix=new THREE.Matrix4().fromArray(p.transform.matrix);raycaster.ray.origin.setFromMatrixPosition(matrix);raycaster.ray.direction.set(0,0,-1).transformDirection(matrix);
    const hit=raycaster.intersectObjects(xrButtons)[0];if(!hit)return;
    if(hit.object.userData.action==='Exit')void session.end();
    else if(hit.object.userData.action==='Live / pause'){live=!live;epoch++;liveTarget=null;$('[data-live]').checked=live;updateMarkers(tracking,live?'Waiting for WebXR':'Paused input');status(live?'Live tracking resumed':'Live paused · pose held');}
    else if(hit.object.userData.action==='Recenter')recenter();
    else {stopLive();snapshotPose={...pose};snapshotTracking=tracking;updateMarkers(tracking,'Snapshot');mode('Snapshot');status('Captured live pose · Snapshot mode');}
  }
  async function enterXR(){
    try{if(session){await session.end();return;}
      const next=await navigator.xr.requestSession('immersive-vr',{optionalFeatures:['local-floor','hand-tracking']});
      if(disposed){await next.end();return;}session=next;
      next.addEventListener('select',selectXR);
      next.addEventListener('end',()=>{session=null;epoch++;liveTarget=null;alignment=null;if(disposed)return;updateMarkers(tracking,'Last WebXR frame · paused');content.position.set(0,0,0);xrPanel.visible=false;controls.enabled=true;$('[data-action=xr]').textContent='Open in Vision Pro';status('Immersive view ended · current pose retained');jointControls();},{once:true});
      let spaceType='local-floor';try{await next.requestReferenceSpace(spaceType);}catch{spaceType='local';}
      renderer.xr.setReferenceSpaceType(spaceType);
      controls.enabled=false;content.position.set(0,.65,-1.4);xrPanel.visible=true;
      await renderer.xr.setSession(next);$('[data-action=xr]').textContent='Exit Vision Pro';status('Immersive scene · gaze and pinch the scene buttons.');
    }catch(error){if(session)await session.end().catch(()=>{});status('WebXR: '+error.message,true);}
  }
  async function checkXR(){
    const note=$('[data-xr-note]');
    if(!window.isSecureContext){note.textContent='Open this same page over trusted HTTPS on Vision Pro. HTTP on a LAN address cannot start WebXR.';return;}
    if(!navigator.xr){note.textContent='Desktop orbit and Snapshot work here. Open this HTTPS page in Vision Pro Safari for immersive view.';return;}
    try{const supported=await navigator.xr.isSessionSupported('immersive-vr');if(disposed)return;$('[data-action=xr]').disabled=!supported;note.textContent=supported?'Immersive view ready. Allow hand tracking to follow your pose.':'This browser does not offer immersive WebXR; Snapshot remains available.';}catch(e){if(!disposed)note.textContent=e.message;}
  }
  function xrTracking(frame){
    const ref=renderer.xr.getReferenceSpace(),viewer=frame.getViewerPose(ref);if(!viewer)return null;
    const head=nativeMatrix(viewer.transform.matrix);
    if(!alignment)alignment=fromRows(data.snapshot.head).multiply(head.clone().invert());
    const output={head:rows(alignment.clone().multiply(head))};
    for(const source of session.inputSources){
      if(!source.hand||!['left','right'].includes(source.handedness))continue;
      const stack=[];
      for(const name of XR_HAND_NAMES){const joint=source.hand.get(name),p=joint&&frame.getJointPose(joint,ref);if(!p){stack.length=0;break;}stack.push(rows(alignment.clone().multiply(nativeMatrix(p.transform.matrix))));}
      if(stack.length===25)output[source.handedness+'_arm']=stack;
    }return output;
  }
  const resize=new ResizeObserver(()=>{if(disposed||renderer.xr.isPresenting)return;const r=stage.getBoundingClientRect();renderer.setSize(r.width,r.height,false);camera.aspect=r.width/r.height;camera.updateProjectionMatrix();});resize.observe(stage);
  renderer.setAnimationLoop((time,frame)=>{
    if(disposed)return;
    // Read markers every XR frame, independently of asynchronous pose solving.
    // Missing tracking clears points immediately rather than displaying stale data.
    if(frame&&session&&live&&data){
      const current=xrTracking(frame);
      tracking=current;updateMarkers(current,current?'Live WebXR':'Tracking unavailable');
      if(!current){epoch++;liveTarget=null;status('Tracking unavailable · robot pose held');}
      else if(!busy&&time-lastSend>100){busy=true;lastSend=time;const request=epoch;
        void solve(current).then(next=>{if(disposed||!live||request!==epoch)return;liveTarget=next;mode('Live WebXR');const hands=['left','right'].filter(s=>current[s+'_arm']);status(hands.length?`Live · ${hands.join(' + ')} hand${hands.length>1?'s':''}`:'Head tracked · hands unavailable, resting arms');}).catch(e=>{if(!disposed&&request===epoch){stopLive();status(e.message,true);}}).finally(()=>{busy=false;});
      }
    }
    if(live&&liveTarget){const alpha=1-Math.exp(-Math.min((time-lastRender)/1000,.1)*20);const smooth={};for(const j of data.joints)if(j.type!=='fixed')smooth[j.child]=(pose[j.child]||0)+((liveTarget[j.child]||0)-(pose[j.child]||0))*alpha;applyPose(smooth);}
    lastRender=time;
    if(!renderer.xr.isPresenting)controls.update();renderer.render(scene,camera);
  });
  async function load(){
    try{const response=await fetch(asset('outputs/web_scene/scene.json'),{signal:abort.signal,cache:'no-store'});if(!response.ok)throw new Error('Browser assets are missing. Open Asset provenance → Refresh latest assets.');data=await response.json();if(disposed)return;
      if(!data.trackingTransform)throw new Error('Refresh latest assets to add the tracking coordinate transform.');
      markerTransform=fromRows(data.trackingTransform);
      for(const [name,vertices] of Object.entries(data.meshes)){const geometry=new THREE.BufferGeometry();geometry.setAttribute('position',new THREE.Float32BufferAttribute(vertices,3));geometry.computeVertexNormals();geometries.set(name,geometry);}
      for(const link of data.links){const group=new THREE.Group();group.matrixAutoUpdate=false;urdf.add(group);visuals.set(link.name,group);for(const v of link.visuals){const mesh=new THREE.Mesh(geometries.get(v.mesh),robotMaterial);mesh.matrixAutoUpdate=false;mesh.matrix.copy(originMatrix(v).multiply(new THREE.Matrix4().makeScale(...v.scale)));group.add(mesh);}}
      status('URDF ready · loading USD materials and skin…');
      gltf=await new GLTFLoader().loadAsync(asset('outputs/web_scene/character.glb'));
      if(disposed){disposeObject(gltf.scene);return;}
      gltf.scene.traverse(o=>{if(o.isBone)bones.push(o);if(o.isSkinnedMesh)o.frustumCulled=false;});
      if(bones.length!==data.source.bones)throw new Error('USD conversion lost skeleton joints');usd.add(gltf.scene);
      snapshotPose=data.snapshotPose;snapshotTracking=data.snapshot;snapshot();
      const sourceCheck=await fetch('/api/avp/assets',{signal:abort.signal,cache:'no-store'});
      if(sourceCheck.ok){const result=await sourceCheck.json();if(!result.ready)status('A source asset has changed. Use Asset provenance → Refresh latest assets.',true);}
      $('.avp-source').textContent=`${data.source.source||data.source.urdf}\nURDF SHA-256 ${data.source.urdfSha256}\n${data.source.triangles.toLocaleString()} URDF triangles · ${bones.length} USD bones\nSource PBR materials · embedded textures ≤ ${data.source.textureMaxSize||1024}px`;
      root.querySelectorAll('button:not([data-action=xr]),input').forEach(el=>el.disabled=false);
      root.dataset.ready='true';root.dataset.usdBones=String(bones.length);void checkXR();
    }catch(e){if(!disposed)status(e.message,true);}
  }
  function disposeObject(object){object.traverse(o=>{o.geometry?.dispose();for(const material of [o.material].flat().filter(Boolean)){for(const value of Object.values(material))if(value?.isTexture)value.dispose();material.dispose();}});}
  void load();
  return {destroy(){disposed=true;epoch++;clearTimeout(refreshTimer);abort.abort();if(session)void session.end();renderer.setAnimationLoop(null);resize.disconnect();renderer.domElement.removeEventListener('pointerdown',pointerDown);renderer.domElement.removeEventListener('pointerup',pointerUp);controls.dispose();disposeObject(scene);renderer.dispose();renderer.forceContextLoss();root.replaceChildren();}};
}
