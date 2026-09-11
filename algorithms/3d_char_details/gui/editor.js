import * as THREE from 'three';
import { OrbitControls } from '/vendor/three/modules/OrbitControls.js';
import { GLTFLoader } from '/vendor/three/modules/GLTFLoader.js';

const ROOT='algorithms/3d_char_details/';
const asset=p=>`/api/artifact?path=${encodeURIComponent(ROOT+p)}`;
const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const pretty=s=>s.replace(/([a-z])([A-Z])/g,'$1 $2').replace(/_/g,' ').replace(/([a-z])([LR])$/,'$1 · $2');
const SHAPES=new Set(['headWidth','bodyWidth','earLength','muzzleLength','faceWidth','clothingEase','eyeSize','cheekFullness']);
const OUTFIT_GROUPS=[
  ['Torso',{vestChestWidth:'Chest width',vestChestDepth:'Chest depth',vestWaistWidth:'Waist width',vestWaistDepth:'Waist depth',vestShoulderWidth:'Shoulder width',vestLength:'Vest length',skirtFlare:'Skirt flare'}],
  ['Arms',{sleeveUpperRoom:'Upper sleeve room',sleeveForearmRoom:'Forearm sleeve room',sleeveLength:'Sleeve length',cuffOpening:'Cuff opening'}],
  ['Trousers',{trouserRise:'Waist-to-crotch room',trouserThighRoom:'Thigh room',trouserCalfRoom:'Calf room',trouserLength:'Trouser length'}],
  ['Boots',{bootWidth:'Boot width',bootLength:'Boot length',bootInstep:'Instep height',bootShaftWidth:'Boot shaft width',bootShaftHeight:'Boot shaft height'}]
];
const OUTFIT_LABELS=Object.assign({},...OUTFIT_GROUPS.map(([,labels])=>labels));
const OUTFIT=new Set(Object.keys(OUTFIT_LABELS));
const COLORS=['#7fc9bd','#baa6db','#d4b078','#719ec3','#d29ca8','#9cbc80'];

export function mount(root) {
  const cssId='char-details-css';
  if(!document.getElementById(cssId)){const l=document.createElement('link');l.id=cssId;l.rel='stylesheet';l.href=asset('gui/editor.css');document.head.append(l);}
  let dead=false,model,report,renderer,frame,helper,selectedPart=null,selectedBone=null,animation=0,dirty=false,exportBusy=false,downloadUrl=null;
  const abort=new AbortController(),parts=new Map(),bones=new Map(),morphs=new Map(),baseMaterials=new Map();
  let adjustmentMeta={};
  let settings={version:1,assetHash:'',morphs:{},bones:{},parts:{}};
  const query=s=>root.querySelector(s);
  const announce=t=>{if(!dead)query('[data-status]').textContent=t;};
  root.innerHTML=`<div class="char-editor">
    <header class="char-bar"><div><span class="char-kicker">LANDAU V10 / CHARACTER WORKSHOP</span><h2>Shape. Pose. Express.</h2></div><div class="char-actions"><button data-action="reset">Reset all</button><button data-action="save">Save preset</button><button data-action="load">Load preset</button><button class="char-primary" data-action="export">Export edited GLB</button></div></header>
    <div class="char-layout"><section class="char-viewport"><canvas aria-label="Landau 3D editor — drag to orbit, right drag to pan, scroll to zoom"></canvas>
      <div class="char-viewtools"><button data-view="full">Full body</button><button data-view="face">Face</button><button data-view="side">Side</button><button data-view="back">Back</button><button data-action="focus">Expand editor</button><select aria-label="Shading" data-shading><option value="clay">Clay · structure</option><option value="textured" selected>Face material regions</option><option value="wire">Wireframe</option><option value="regions">Parts</option></select><label><input type="checkbox" data-skeleton> Skeleton</label></div>
      <div class="char-stage-label"><span data-selection>Landau v10</span><small>Drag to orbit · right drag to pan · scroll to zoom · click a part</small></div>
      <div class="char-loading" data-loading>Preparing the character…</div>
    </section><aside class="char-inspector"><nav class="char-tabs"><button class="active" data-tab="face">Face</button><button data-tab="shape">Shape</button><button data-tab="parts">Parts</button><button data-tab="rig">Rig</button><button data-tab="reference">Ref</button><button data-tab="asset">Asset</button></nav>
      <div class="char-pane" data-pane="face"><p class="char-hint">Blink moves the original eyelashes with the blue upper lid. The eye remains round and independent; the lower eye rim stays fixed.</p><div class="char-section"><h3>Expression presets</h3><div class="char-presets"><button data-expression="neutral">Neutral</button><button data-expression="happy">Happy</button><button data-expression="surprised">Surprised</button><button data-expression="half">Half closed</button><button data-expression="blink">Blink</button><button data-expression="look">Look left</button></div><label class="char-check"><input type="checkbox" data-blink> Play blink test</label></div><div data-face-sliders></div></div>
      <div class="char-pane" data-pane="shape" hidden><p class="char-hint">Rest-shape controls move the actual mesh. Test expressions again after changing proportions.</p><div data-shape-sliders></div><button data-action="reset-body">Reset body proportions</button><section class="char-section" data-outfit-section hidden><h3>Clothing fit</h3><p class="char-hint">Adjust local width, depth, length and openings. 0 restores the original garment shape. Width, depth, length and placement are independent; the outfit starts hidden for manual fitting. Check the fit after posing.</p><div data-outfit-sliders></div><button data-action="reset-outfit">Reset clothing fit</button></section></div>
      <div class="char-pane" data-pane="parts" hidden><p class="char-hint">Garments are independent objects. The supplied FBX body keeps a uniform scale. The original head and hands are retained; original garments are available for manual fitting.</p><div class="char-presets"><button data-action="undress">Hide clothing</button><button data-action="dress">Show clothing</button></div><div data-part-list></div><div class="char-section"><h3 data-part-title>Select a part</h3><div class="char-presets"><button data-action="isolate">Isolate selected</button><button data-action="show-character">Show character</button></div><label class="char-bone-select">Material <select aria-label="Selected material" data-material></select></label><label>Material color <input aria-label="Selected material color" type="color" value="#ffffff" data-tint></label><button data-action="untint">Reset material color</button></div></div>
      <div class="char-pane" data-pane="rig" hidden><p class="char-hint">71 bones: original body rig plus ears and tail. Angles use each bone’s local axes.</p><label class="char-bone-select">Bone <select aria-label="Bone" data-bone></select></label><div data-bone-sliders></div><button data-action="reset-bone">Reset selected bone</button><button data-action="reset-pose">Reset pose</button><p class="char-hint">Scale is a pose control; use Shape for body proportions. Extreme poses may need further weight painting.</p></div>
      <div class="char-pane" data-pane="reference" hidden><h3>Landau character references</h3><p class="char-hint">Compare the tall eyes, swept lashes, full cheeks and small smile. Clay removes color and normal maps.</p><select aria-label="Character reference" data-reference><option value="landau_test2.png">Landau test 2 · 3D reference</option><option value="internal_reference.png">Body structure · proportion reference</option><option value="closed_eyes_reference.png">Closed eyes · expression reference</option><option value="closed_eyes_reference2.png">Closed eyes · reference 2</option><option value="half_closed_reference.png">Half-closed eyes · expression</option><option value="half_closed_reference2.png">Half-closed eyes · reference 2</option><option value="landau_test.png">Landau test · illustration</option><option value="reference.png">Landau v10 · generation source</option></select><label class="char-check"><input type="checkbox" data-crop checked> Face close-up</label><div class="char-reference-crop closeup" data-reference-frame><img data-reference-image src="${asset('inputs/landau_v10/landau_test2.png')}" alt="Landau reference face"></div><a class="char-reference-link" data-reference-link href="${asset('inputs/landau_v10/landau_test2.png')}" target="_blank" rel="noopener">Open full reference</a><p class="char-hint">Each reference has different proportions. They guide likeness; the original mesh remains available at neutral.</p></div>
      <div class="char-pane" data-pane="asset" hidden><div data-asset-report></div><figure><img src="${asset('inputs/landau_v10/reference.png')}" alt="Original Landau v10 generation reference"><figcaption>Original generation reference</figcaption></figure><label class="char-check">Export textures <select aria-label="Export texture resolution" data-export-resolution><option value="4096">4096 · source detail</option><option value="2048">2048 · smaller file</option><option value="1024">1024 · compact</option></select></label><label class="char-check"><input type="checkbox" data-visible-only checked> Export visible parts only</label><div class="char-downloads"><a href="${asset('outputs/landau_v10/landau_character.blend')}&download=1" download>Blender master</a><a href="${asset('outputs/landau_v10/asset_report.json')}&download=1" download>Asset report</a></div></div>
    </aside></div><footer class="char-footer"><span data-status role="status">Loading the authored GLB…</span><a data-download hidden>Download prepared file</a><span data-stats></span></footer><input type="file" accept=".json,application/json" data-file hidden>
  </div>`;

  const stage=query('.char-viewport'),canvas=query('canvas');
  const scene=new THREE.Scene();scene.background=new THREE.Color('#202b32');
  const camera=new THREE.PerspectiveCamera(33,1,.005,30);camera.position.set(0,.64,2.4);
  renderer=new THREE.WebGLRenderer({canvas,antialias:true,preserveDrawingBuffer:false});renderer.setPixelRatio(Math.min(devicePixelRatio,2));renderer.outputColorSpace=THREE.SRGBColorSpace;renderer.toneMapping=THREE.ACESFilmicToneMapping;renderer.toneMappingExposure=1.25;
  const orbit=new OrbitControls(camera,canvas);orbit.target.set(0,.58,0);orbit.enableDamping=true;orbit.minDistance=.18;orbit.maxDistance=6;orbit.maxPolarAngle=Math.PI*.95;
  scene.add(new THREE.HemisphereLight(0xe7f3ff,0x666352,2.4));
  for(const [pos,color,intensity] of [[[1,2,2],0xffeed9,2.6],[[-2,1,1],0xd4eaff,1.7],[[0,2,-2],0x9ccfe0,2.2]]){const light=new THREE.DirectionalLight(color,intensity);light.position.set(...pos);scene.add(light);}
  const grid=new THREE.GridHelper(2,20,0x5a747d,0x36454e);grid.material.opacity=.35;grid.material.transparent=true;scene.add(grid);
  const resize=new ResizeObserver(()=>{if(dead)return;const w=stage.clientWidth,h=stage.clientHeight;renderer.setSize(w,h,false);camera.aspect=w/h;camera.updateProjectionMatrix();});resize.observe(stage);
  const released=new Set();
  function release(resource){if(resource&&!released.has(resource)){released.add(resource);resource.dispose?.();}}
  function releaseTree(tree){tree.traverse(o=>{release(o.geometry);if(o.skeleton)release(o.skeleton);for(const m of (Array.isArray(o.material)?o.material:[o.material]))if(m){for(const v of Object.values(m))if(v?.isTexture)release(v);release(m);}});}
  const dispose=()=>{if(dead)return;dead=true;abort.abort();if(downloadUrl)URL.revokeObjectURL(downloadUrl);cancelAnimationFrame(frame);resize.disconnect();orbit.dispose();releaseTree(scene);for(const m of baseMaterials.keys())release(m);renderer.dispose();};
  // Return a lifecycle handle immediately, including while the asynchronous loader runs.
  root.charDispose=dispose;
  function saveLocal(){try{localStorage.setItem('landau-char-v1',JSON.stringify(settings));dirty=false;}catch{announce('Browser storage unavailable; download a preset to keep your edits.');}}
  function touch(){dirty=true;saveLocal();}
  function setMorph(name,value,persist=true){
    if(!morphs.has(name))return;
    settings.morphs[name]=value;for(const [mesh,index] of morphs.get(name)||[])mesh.morphTargetInfluences[index]=value;
    const input=query(`input[data-morph="${name}"]`);if(input){input.value=value;input.nextElementSibling.value=Number(value).toFixed(2);}
    if(persist)touch();
  }
  function applyParts(){for(const [name,p] of parts){const cfg=settings.parts[name]||{};p.object.visible=cfg.visible??p.defaultVisible;for(const m of p.materials)m.color.set(cfg.materialColors?.[m.name]||baseMaterials.get(m).color);}}
  function applyBone(name){const item=bones.get(name);if(!item)return;const cfg=settings.bones[name]||{rotation:[0,0,0],scale:1};const q=new THREE.Quaternion().setFromEuler(new THREE.Euler(...cfg.rotation.map(v=>v*Math.PI/180),'XYZ'));item.bone.quaternion.copy(item.quaternion).multiply(q);item.bone.scale.copy(item.scale).multiplyScalar(cfg.scale);}
  function readBone(){if(!selectedBone)return;const v=settings.bones[selectedBone]||{rotation:[0,0,0],scale:1};for(const input of root.querySelectorAll('[data-bone-axis]')){const i=Number(input.dataset.boneAxis);input.value=v.rotation[i];input.nextElementSibling.value=v.rotation[i].toFixed(0)+'°';}const input=query('[data-bone-scale]');input.value=v.scale;input.nextElementSibling.value=v.scale.toFixed(2);}
  function readMaterial(){const p=parts.get(selectedPart);if(!p)return;const name=query('[data-material]').value;const m=[...p.materials].find(m=>m.name===name);if(m)query('[data-tint]').value=settings.parts[selectedPart]?.materialColors?.[name]||'#'+baseMaterials.get(m).color.getHexString();}
  function selectPart(name){if(!parts.has(name))return;selectedPart=name;query('[data-selection]').textContent=pretty(name);query('[data-part-title]').textContent=pretty(name);query('[data-material]').innerHTML=[...parts.get(name).materials].map(m=>`<option value="${esc(m.name)}">${esc(m.name)}</option>`).join('');readMaterial();root.querySelectorAll('[data-part]').forEach(el=>el.classList.toggle('selected',el.dataset.part===name));}
  function shading(){const mode=query('[data-shading]').value;let i=0;for(const [name,p] of parts){for(const m of p.materials){const b=baseMaterials.get(m);for(const [key,val]of Object.entries(b.maps))m[key]=(mode==='clay'||mode==='regions')?null:val;m.wireframe=mode==='wire';m.vertexColors=(mode==='clay'||mode==='regions')?false:b.vertexColors;m.roughness=mode==='clay'?.8:b.roughness;m.metalness=(mode==='clay'||mode==='regions')?0:b.metalness;if(mode==='clay')m.color.set('#c8d2d2');else if(mode==='regions')m.color.set(COLORS[i%COLORS.length]);else m.color.set(settings.parts[name]?.materialColors?.[m.name]||b.color);m.needsUpdate=true;}i++;}}
  function view(name){
    const reportedLift=report?.body_reconstruction?.head_rigid_lift;
    const lift=Number.isFinite(reportedLift)?reportedLift:0;
    const center=.58+lift/2,height=.64+lift/2,distance=2.4*(1+lift/1.16);
    orbit.target.set(0,name==='face'?.71+lift:center,name==='face'?.01:0);
    const positions={full:[0,height,distance],face:[0,.73+lift,.72],side:[distance,height,0],back:[0,height,-distance]};
    camera.position.set(...positions[name]);orbit.update();
  }
  function download(data,name,type){if(downloadUrl)URL.revokeObjectURL(downloadUrl);downloadUrl=URL.createObjectURL(new Blob([data],{type}));const a=query('[data-download]');a.href=downloadUrl;a.download=name;a.textContent='Download '+name;a.hidden=false;a.click();}
  function validatePreset(p){
    if(!p||p.version!==1||p.assetHash!==report.glb_sha256)throw new Error('This preset belongs to a different asset revision.');
    for(const [n,v] of Object.entries(p.morphs||{})){if(!morphs.has(n)||!Number.isFinite(v)||v<(adjustmentMeta[n]?.min??(SHAPES.has(n)?-1:0))||v>1)throw new Error('Invalid morph: '+n);}
    for(const [n,v] of Object.entries(p.bones||{})){if(!bones.has(n)||!Array.isArray(v.rotation)||v.rotation.length!==3||v.rotation.some(x=>!Number.isFinite(x)||Math.abs(x)>120)||!Number.isFinite(v.scale)||v.scale<.7||v.scale>1.3)throw new Error('Invalid bone: '+n);}
    for(const [n,v] of Object.entries(p.parts||{})){if(!parts.has(n)||!v||typeof v!=='object'||(v.visible!==undefined&&typeof v.visible!=='boolean')||(v.tint!==undefined&&!/^#[0-9a-f]{6}$/i.test(v.tint)))throw new Error('Invalid part: '+n);}
    for(const [n,v] of Object.entries(p.parts||{}))for(const [material,color]of Object.entries(v.materialColors||{}))if(![...parts.get(n).materials].some(m=>m.name===material)||!/^#[0-9a-f]{6}$/i.test(color))throw new Error('Invalid material color: '+material);
    return {version:1,assetHash:p.assetHash,morphs:p.morphs||{},bones:p.bones||{},parts:p.parts||{}};
  }
  function applyPreset(p,persist=true){query('[data-blink]').checked=false;settings=validatePreset(p);for(const n of morphs.keys())setMorph(n,settings.morphs[n]||0,false);for(const n of bones.keys())applyBone(n);applyParts();shading();root.querySelectorAll('[data-visible]').forEach(e=>e.checked=settings.parts[e.dataset.visible]?.visible??parts.get(e.dataset.visible).defaultVisible);readBone();readMaterial();if(persist)touch();}
  function expression(kind){query('[data-blink]').checked=false;for(const n of morphs.keys())if(!SHAPES.has(n)&&!OUTFIT.has(n))setMorph(n,0,false);const p={neutral:{},happy:{mouthSmile:.7,eyeSquintL:.15,eyeSquintR:.15},surprised:{jawDrop:.75,browUpL:.65,browUpR:.65,eyeWideL:.5,eyeWideR:.5},half:{eyeBlinkL:.5,eyeBlinkR:.5},blink:{eyeBlinkL:1,eyeBlinkR:1},look:{eyeLookOutL:.75,eyeLookInR:.75}}[kind];for(const [n,v]of Object.entries(p))setMorph(n,v,false);touch();announce('Expression: '+kind);}
  function control(name){const min=adjustmentMeta[name]?.min??(SHAPES.has(name)?-1:0),label=adjustmentMeta[name]?.label||OUTFIT_LABELS[name]||pretty(name);return `<label class="char-slider"><span>${esc(label)}</span><div><input aria-label="${esc(label)}" data-morph="${esc(name)}" type="range" min="${min}" max="1" step=".01" value="0"><output>0.00</output></div></label>`;}
  function tick(t){if(dead)return;frame=requestAnimationFrame(tick);if(query('[data-blink]').checked){const phase=(t/1000)%3.4;animation=Math.max(0,1-Math.abs(phase-.35)/.16);for(const n of ['eyeBlinkL','eyeBlinkR'])for(const[m,i]of morphs.get(n)||[])m.morphTargetInfluences[i]=animation;}orbit.update();renderer.render(scene,camera);}
  frame=requestAnimationFrame(tick);

  root.addEventListener('click',async e=>{
    const button=e.target.closest('button');if(!button||exportBusy)return;
    if(button.dataset.tab){root.querySelectorAll('[data-tab]').forEach(b=>b.classList.toggle('active',b===button));root.querySelectorAll('[data-pane]').forEach(p=>p.hidden=p.dataset.pane!==button.dataset.tab);}
    if(button.dataset.view)view(button.dataset.view);
    if(button.dataset.expression&&model)expression(button.dataset.expression);
    if(button.dataset.part)selectPart(button.dataset.part);
    const action=button.dataset.action;if(action==='focus'){const expanded=query('.char-editor').classList.toggle('char-expanded');button.textContent=expanded?'Exit expanded view':'Expand editor';return;}if(!action||!model)return;
    try{
      if(action==='reset'){query('[data-blink]').checked=false;applyPreset({version:1,assetHash:report.glb_sha256});announce('All character edits reset.');}
      if(action==='reset-body'){for(const [n,m]of Object.entries(adjustmentMeta))if(m.kind==='body')setMorph(n,0,false);touch();announce('Body proportions restored to the uniformly scaled FBX.');}
      if(action==='reset-outfit'){for(const n of OUTFIT)setMorph(n,0,false);touch();announce('Clothing fit reset.');}
      if(action==='save'){saveLocal();download(JSON.stringify(settings,null,2),'landau-v10-preset.json','application/json');announce('Preset saved with asset revision.');}
      if(action==='load')query('[data-file]').click();
      if(action==='undress'||action==='dress'){for(const [n,p]of parts)if(p.kind==='clothing')settings.parts[n]={...settings.parts[n],visible:action==='dress'};for(const [n,p]of parts)if(p.kind==='inferred_body')settings.parts[n]={...settings.parts[n],visible:true};applyParts();shading();root.querySelectorAll('[data-visible]').forEach(el=>el.checked=settings.parts[el.dataset.visible]?.visible??parts.get(el.dataset.visible).defaultVisible);touch();}
      if(action==='untint'&&selectedPart){const colors=settings.parts[selectedPart]?.materialColors||{};delete colors[query('[data-material]').value];shading();readMaterial();touch();}
      if((action==='isolate'&&selectedPart)||action==='show-character'){for(const [n,p]of parts)settings.parts[n]={...settings.parts[n],visible:action==='isolate'?n===selectedPart:p.defaultVisible};applyParts();shading();root.querySelectorAll('[data-visible]').forEach(e=>e.checked=settings.parts[e.dataset.visible].visible);touch();}
      if(action==='reset-bone'&&selectedBone){delete settings.bones[selectedBone];applyBone(selectedBone);readBone();touch();}
      if(action==='reset-pose'){settings.bones={};for(const n of bones.keys())applyBone(n);readBone();touch();}
      if(action==='export'){
        exportBusy=true;button.disabled=true;query('.char-layout').inert=true;query('.char-actions').inert=true;announce('Exporting skin, morph targets, textures and current settings…');
        query('[data-blink]').checked=false;for(const n of ['eyeBlinkL','eyeBlinkR'])setMorph(n,settings.morphs[n]||0,false);
        const old=query('[data-shading]').value;query('[data-shading]').value='textured';shading();
        try{const {GLTFExporter}=await import('/api/artifact?path=algorithms/3d_char_details/gui/vendor/GLTFExporter.js');if(dead)return;const result=await new GLTFExporter().parseAsync(model,{binary:true,trs:true,onlyVisible:query('[data-visible-only]').checked,maxTextureSize:Number(query('[data-export-resolution]').value)});if(dead)return;download(result,'landau-v10-edited.glb','model/gltf-binary');announce('Edited GLB exported. Facial morphs and skin are retained.');}finally{exportBusy=false;if(!dead){query('[data-shading]').value=old;shading();button.disabled=false;query('.char-layout').inert=false;query('.char-actions').inert=false;}}
      }
    }catch(err){announce(err.message);button.disabled=false;}
  },{signal:abort.signal});
  root.addEventListener('input',e=>{
    const el=e.target;if(!model||exportBusy)return;
    if(el.dataset.morph)setMorph(el.dataset.morph,Number(el.value));
    if(el.hasAttribute('data-bone-axis')||el.hasAttribute('data-bone-scale')){if(!selectedBone)return;const cfg=settings.bones[selectedBone]||{rotation:[0,0,0],scale:1};if(el.hasAttribute('data-bone-axis'))cfg.rotation[Number(el.dataset.boneAxis)]=Number(el.value);else cfg.scale=Number(el.value);settings.bones[selectedBone]=cfg;applyBone(selectedBone);readBone();touch();}
    if(el.hasAttribute('data-tint')&&selectedPart){const cfg=settings.parts[selectedPart]||{};settings.parts[selectedPart]={...cfg,materialColors:{...cfg.materialColors,[query('[data-material]').value]:el.value}};shading();touch();}
  },{signal:abort.signal});
  root.addEventListener('change',async e=>{
    const el=e.target;if(exportBusy)return;
    if(el.hasAttribute('data-reference')){const url=asset('inputs/landau_v10/'+el.value);query('[data-reference-image]').src=url;query('[data-reference-image]').alt=el.selectedOptions[0].textContent;query('[data-reference-link]').href=url;query('[data-reference-frame]').dataset.source=el.value;if(el.value==='internal_reference.png'){query('[data-crop]').checked=false;query('[data-reference-frame]').classList.remove('closeup');}}
    if(el.hasAttribute('data-crop'))query('[data-reference-frame]').classList.toggle('closeup',el.checked);
    if(el.hasAttribute('data-shading'))shading();
    if(el.hasAttribute('data-skeleton')&&helper)helper.visible=el.checked;
    if(el.dataset.visible){settings.parts[el.dataset.visible]={...settings.parts[el.dataset.visible],visible:el.checked};applyParts();shading();touch();}
    if(el.hasAttribute('data-material'))readMaterial();
    if(el.hasAttribute('data-bone')){selectedBone=el.value;readBone();}
    if(el.hasAttribute('data-blink')&&!el.checked)for(const n of ['eyeBlinkL','eyeBlinkR'])setMorph(n,settings.morphs[n]||0,false);
    if(el.hasAttribute('data-file')&&el.files[0]){try{if(el.files[0].size>200000)throw new Error('Preset is too large.');applyPreset(JSON.parse(await el.files[0].text()));announce('Preset restored.');}catch(err){announce(err.message);}el.value='';}
  },{signal:abort.signal});
  let down;
  canvas.addEventListener('pointerdown',e=>down=[e.clientX,e.clientY],{signal:abort.signal});
  canvas.addEventListener('pointerup',e=>{if(!model||!down||Math.hypot(e.clientX-down[0],e.clientY-down[1])>4)return;const r=canvas.getBoundingClientRect();const ray=new THREE.Raycaster();ray.setFromCamera(new THREE.Vector2((e.clientX-r.left)/r.width*2-1,-(e.clientY-r.top)/r.height*2+1),camera);const hits=ray.intersectObject(model,true).filter(h=>{let o=h.object;while(o){if(!o.visible)return false;o=o.parent;}return true;});if(hits.length){let o=hits[0].object;while(o&&!parts.has(o.name))o=o.parent;if(o)selectPart(o.name);}},{signal:abort.signal});
  const ready=(async()=>{try{
    const response=await fetch(asset('outputs/landau_v10/asset_report.json'),{signal:abort.signal});if(!response.ok)throw new Error('Asset report unavailable.');report=await response.json();
    adjustmentMeta=report.body_reconstruction?.adjustment_controls||{};
    for(const [n,m]of Object.entries(adjustmentMeta)){if(m.kind==='body')SHAPES.add(n);if(m.kind==='outfit')OUTFIT.add(n);}
    const gltf=await new GLTFLoader().loadAsync(asset('outputs/landau_v10/landau_character.glb'));if(dead){releaseTree(gltf.scene);return;}
    model=gltf.scene;model.name='Landau_v10';scene.add(model);
    model.traverse(o=>{
      if(o.isBone)bones.set(o.name,{bone:o,quaternion:o.quaternion.clone(),scale:o.scale.clone()});
      if(o.userData.part_type)parts.set(o.name,{object:o,kind:o.userData.part_type,defaultVisible:!o.userData.default_hidden,materials:new Set()});
    });
    model.traverse(o=>{if(!o.isMesh)return;let parent=o;while(parent&&!parts.has(parent.name))parent=parent.parent;if(!parent){parent=o;parts.set(o.name,{object:o,kind:'mesh',defaultVisible:true,materials:new Set()});}const p=parts.get(parent.name);
      const materials=(Array.isArray(o.material)?o.material:[o.material]).map(m=>{const copy=m.clone();baseMaterials.set(copy,{color:copy.color.clone(),map:copy.map,normalMap:copy.normalMap,roughness:copy.roughness,metalness:copy.metalness,vertexColors:copy.vertexColors,maps:Object.fromEntries(Object.entries(copy).filter(([k,v])=>k.endsWith('Map')||k==='map'))});p.materials.add(copy);release(m);return copy;});o.material=Array.isArray(o.material)?materials:materials[0];o.frustumCulled=false;
      for(const [name,index]of Object.entries(o.morphTargetDictionary||{})){if(!morphs.has(name))morphs.set(name,[]);morphs.get(name).push([o,index]);}
    });
    helper=new THREE.SkeletonHelper(model);helper.visible=false;helper.material.depthTest=false;helper.renderOrder=10;scene.add(helper);
    query('[data-face-sliders]').innerHTML=[...morphs.keys()].filter(n=>!SHAPES.has(n)&&!OUTFIT.has(n)).sort().map(control).join('');
    query('[data-shape-sliders]').innerHTML=[...morphs.keys()].filter(n=>SHAPES.has(n)).sort().map(control).join('');
    const outfitControls=[...OUTFIT].filter(n=>morphs.has(n));
    query('[data-outfit-sliders]').innerHTML=(Object.keys(adjustmentMeta).length?[...new Set(Object.values(adjustmentMeta).filter(m=>m.kind==='outfit').map(m=>m.group))].map(group=>[group,Object.fromEntries(Object.entries(adjustmentMeta).filter(([,m])=>m.kind==='outfit'&&m.group===group).map(([n,m])=>[n,m.label]))]):OUTFIT_GROUPS).map(([title,labels])=>{const names=Object.keys(labels).filter(n=>morphs.has(n));return names.length?`<section class="char-outfit-group"><h4>${esc(title)}</h4>${names.map(control).join('')}</section>`:'';}).join('');
    query('[data-outfit-section]').hidden=outfitControls.length===0;
    query('[data-part-list]').innerHTML=[...parts].map(([n,p])=>`<div class="char-part"><input type="checkbox" aria-label="Show ${esc(pretty(n))}" data-visible="${esc(n)}" ${p.defaultVisible?'checked':''}><button data-part="${esc(n)}">${esc(pretty(n))}<small>${esc(p.kind.replace('_',' '))}</small></button></div>`).join('');
    query('[data-bone]').innerHTML=[...bones.keys()].map(n=>`<option value="${esc(n)}">${esc(pretty(n))}</option>`).join('');selectedBone=bones.has('head_x')?'head_x':bones.keys().next().value;query('[data-bone]').value=selectedBone;
    query('[data-bone-sliders]').innerHTML=['X','Y','Z'].map((a,i)=>`<label class="char-slider"><span>Rotation ${a}</span><div><input type="range" aria-label="Bone rotation ${a}" data-bone-axis="${i}" min="-120" max="120" step="1" value="0"><output>0°</output></div></label>`).join('')+`<label class="char-slider"><span>Scale</span><div><input aria-label="Bone scale" type="range" data-bone-scale min=".7" max="1.3" step=".01" value="1"><output>1.00</output></div></label>`;
    const v=report.validation;query('[data-asset-report]').innerHTML=`<h3>Structure and rig preview</h3><dl class="char-facts"><dt>Meshes</dt><dd>${v.mesh_count}</dd><dt>Triangles</dt><dd>${v.triangles.toLocaleString()}</dd><dt>Bones</dt><dd>${v.bones}</dd><dt>Facial + shape controls</dt><dd>${morphs.size}</dd><dt>Invalid skin weights</dt><dd>${v.invalid_skin_vertices}</dd></dl><h3>Remaining production work</h3><ul>${report.limitations.map(s=>`<li>${esc(s)}</li>`).join('')}</ul>`;
    query('[data-stats]').textContent=`${parts.size} parts · ${bones.size} bones · ${morphs.size} controls`;
    settings.assetHash=report.glb_sha256;
    // Read saved edits before applying defaults; applying a preset may persist it.
    let restored=false;
    try{const saved=localStorage.getItem('landau-char-v1');if(saved){applyPreset(JSON.parse(saved),false);restored=true;}}catch{announce('An incompatible saved preset was skipped.');}
    if(!restored)applyPreset(settings,false);
    query('[data-loading]').hidden=true;selectPart('Head');readBone();shading();view(report.body_reconstruction?.revision>=3?'full':'face');announce('Ready. Edits are saved in this browser; download a preset for backup.');
  }catch(err){if(!dead){query('[data-loading]').textContent=err.message;announce('Could not load character: '+err.message);}}
  })();
  return {destroy:dispose,ready};
}
