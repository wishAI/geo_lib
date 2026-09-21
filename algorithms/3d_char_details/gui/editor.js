import * as THREE from 'three';
import { OrbitControls } from '/vendor/three/modules/OrbitControls.js';
import { GLTFLoader } from '/vendor/three/modules/GLTFLoader.js';
import {icon,widget,GARMENT_CARDS} from '/api/artifact?path=algorithms/3d_char_details/gui/property-widgets.js';
import {garmentFit,OUTFIT_ASSEMBLIES,PARENT,ANGLES,angleLabel,assemblyOf,sharedOwner,inheritedControls,fitDefaults,outfitMinimum,COLLAR_CONTROLS,SHOULDER_CONTROLS,depthNames,splitDepth,jointsLinked,jointPeer,setJointAngle,setJointLink,outfitValue as linkedOutfitValue,setOutfitValue} from '/api/artifact?path=algorithms/3d_char_details/gui/garment-fit.js';
import {editingPose} from '/api/artifact?path=algorithms/3d_char_details/gui/editing-pose.js';
import {bodyPlacement} from '/api/artifact?path=algorithms/3d_char_details/gui/body-placement.js';
import {applyBlink,migrateFacePreset} from '/api/artifact?path=algorithms/3d_char_details/gui/facial-controls.js';
import {updateCameraDepth,eyeLayerDepth} from '/api/artifact?path=algorithms/3d_char_details/gui/ocular-rendering.js';

import {motionPlayer,motionControls} from '/api/artifact?path=algorithms/3d_char_details/gui/motion-player.js';
import {presetHistory} from '/api/artifact?path=algorithms/3d_char_details/gui/preset-history.js';

import {BODY_TRANSITION_CONTROLS,installBodyTransition} from '/api/artifact?path=algorithms/3d_char_details/gui/body-transition.js';

const ROOT='algorithms/3d_char_details/';
const asset=p=>`/api/artifact?path=${encodeURIComponent(ROOT+p)}`;
const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const pretty=s=>s.replace(/([a-z])([A-Z])/g,'$1 $2').replace(/_/g,' ').replace(/([a-z])([LR])$/,'$1 · $2');
const SHAPES=new Set(['headWidth','bodyWidth','earLength','muzzleLength','faceWidth','eyeSize','cheekFullness','mouthLength','mouthCurvature']);
const OUTFIT_GROUPS=[
  ['Torso',{vestChestWidth:'Chest width',vestChestDepth:'Chest depth',vestWaistWidth:'Waist width',vestWaistDepth:'Waist depth',vestShoulderWidth:'Shoulder width',vestLength:'Vest length',skirtFlare:'Skirt flare'}],
  ['Arms',{sleeveUpperRoom:'Upper sleeve room',sleeveForearmRoom:'Forearm sleeve room',sleeveLength:'Sleeve length',cuffOpening:'Cuff opening'}],
  ['Trousers',{trouserRise:'Waist-to-crotch room',trouserThighRoom:'Thigh room',trouserCalfRoom:'Calf room',trouserLength:'Trouser length'}],
  ['Boots',{bootWidth:'Boot width',bootLength:'Boot length',bootInstep:'Instep height',bootShaftWidth:'Boot shaft width',bootShaftHeight:'Boot shaft height'}]
];
const OUTFIT_LABELS=Object.assign({},...OUTFIT_GROUPS.map(([,labels])=>labels));
const OUTFIT=new Set([...Object.keys(OUTFIT_LABELS),'clothingEase']);
const COLORS=['#7fc9bd','#baa6db','#d4b078','#719ec3','#d29ca8','#9cbc80'];

export function mount(root) {
  const cssId='char-details-css';
  if(!document.getElementById(cssId)){const l=document.createElement('link');l.id=cssId;l.rel='stylesheet';l.href=asset('gui/editor.css');document.head.append(l);}
  let dead=false,model,report,renderer,frame,helper,selectedPart=null,selectedBone=null,animation=0,history,dirty=false,exportBusy=false,downloadUrl=null;
  const abort=new AbortController(),parts=new Map(),bones=new Map(),morphs=new Map(),baseMaterials=new Map();
  let adjustmentMeta={},placeBody,editPose,frameScale=1,activeView='full',player,fit,updateMotion,lastTick=0;
  let settings={version:1,assetHash:'',editingPose:'t',morphs:{},bones:{},parts:{},outfit:{},links:{},bodyFrame:{scale:1,offset:0},garmentFit:fitDefaults()};
  const query=s=>root.querySelector(s);
  const announce=t=>{if(!dead)query('[data-status]').textContent=t;};
  root.innerHTML=`<div class="char-editor">
    <header class="char-bar"><div><span class="char-kicker">LANDAU V10 / CHARACTER WORKSHOP</span><h2>Shape. Pose. Express.</h2></div><div class="char-actions"><button data-action="reset">Reset all</button><button data-action="save">Save preset</button><button data-tab="presets">History</button><button class="char-primary" data-action="export">Export edited GLB</button></div></header>
    <div class="char-layout"><section class="char-viewport"><canvas aria-label="Landau 3D editor — drag to orbit, right drag to pan, scroll to zoom"></canvas>
      <div class="char-viewtools"><button data-view="full">Full body</button><button data-view="face">Face</button><button data-view="side">Side</button><button data-view="back">Back</button><select aria-label="Editing pose" data-editing-pose><option value="t">T-pose · editing</option><option value="a">A-pose · source</option></select><button data-action="focus">Expand editor</button><select aria-label="Shading" data-shading><option value="clay">Clay · structure</option><option value="textured" selected>Face material regions</option><option value="wire">Wireframe</option><option value="regions">Parts</option></select><label><input type="checkbox" data-skeleton> Skeleton</label></div>
      <div class="char-stage-label"><span data-selection>Landau v10</span><small>Drag to orbit · right drag to pan · scroll to zoom · click a part</small></div>
      <div class="char-motion" data-motion hidden></div><div class="char-loading" data-loading>Preparing the character…</div>
    </section><aside class="char-inspector"><nav class="char-tabs"><button class="active" data-tab="face">Face</button><button data-tab="shape">Shape</button><button data-tab="clothing">Clothing</button><button data-tab="parts">Parts</button><button data-tab="rig">Rig</button><button data-tab="reference">Ref</button><button data-tab="asset">Asset</button><button data-tab="presets">Presets</button></nav>
      <div class="char-pane" data-pane="face"><p class="char-hint">Open the mouth with Jaw drop. Adjust mouth length and curvature in Shape. Independent eyes stay fixed during blinking; the original lashes follow the upper lids.</p><div class="char-section"><h3>Expression presets</h3><div class="char-presets"><button data-expression="neutral">Neutral</button><button data-expression="happy">Happy</button><button data-expression="surprised">Surprised</button><button data-expression="mouth">Open mouth</button><button data-expression="half">Half closed</button><button data-expression="blink">Blink</button><button data-expression="look">Look left</button></div><label class="char-check"><input type="checkbox" data-blink> Play blink test</label></div><div data-face-sliders></div></div>
      <div class="char-pane" data-pane="shape" hidden><section class="char-frame"><div data-frame-sliders></div></section><div data-shape-sliders></div><button data-action="reset-body">Reset body proportions</button></div>
      <div class="char-pane" data-pane="clothing" hidden><div class="char-presets char-clothing-toolbar"><button data-action="dress">Show all</button><button data-action="undress">Hide all</button><button data-action="reset-outfit">Reset fit</button></div><div data-garment-cards></div></div>
      <div class="char-pane" data-pane="parts" hidden><p class="char-hint">Garments retain separate material and visibility sections within connected upper and lower outfits. The supplied FBX body keeps a uniform scale. The original head and hands are retained; original garments are available for manual fitting.</p><div data-part-list></div><div class="char-section"><h3 data-part-title>Select a part</h3><div class="char-presets"><button data-action="isolate">Isolate selected</button><button data-action="show-character">Show character</button></div><label class="char-bone-select">Material <select aria-label="Selected material" data-material></select></label><label>Material color <input aria-label="Selected material color" type="color" value="#ffffff" data-tint></label><button data-action="untint">Reset material color</button></div></div>
      <div class="char-pane" data-pane="rig" hidden><p class="char-hint">71 bones: original body rig plus ears and tail. T-pose is the default editing baseline. Angles use each bone’s local axes; running uses its authored pose.</p><label class="char-bone-select">Bone <select aria-label="Bone" data-bone></select></label><div data-bone-sliders></div><button data-action="reset-bone">Reset selected bone</button><button data-action="reset-pose">Reset pose</button><p class="char-hint">Scale is a pose control; use Shape for body proportions. Extreme poses may need further weight painting.</p></div>
      <div class="char-pane" data-pane="reference" hidden><h3>Landau character references</h3><p class="char-hint">Compare the tall eyes, swept lashes, full cheeks and small smile. Clay removes color and normal maps.</p><select aria-label="Character reference" data-reference><option value="landau_test2.png">Landau test 2 · 3D reference</option><option value="open_mouth/截屏2026-09-21 18.09.28.png">Open mouth reference 1</option><option value="open_mouth/截屏2026-09-21 18.10.11.png">Open mouth reference 2</option><option value="open_mouth/截屏2026-09-21 18.11.17.png">Open mouth reference 3</option><option value="side_muzzle_reference.png">Side muzzle · rabbit profile</option><option value="internal_reference.png">Body structure · proportion reference</option><option value="closed_eyes_reference.png">Closed eyes · expression reference</option><option value="closed_eyes_reference2.png">Closed eyes · reference 2</option><option value="half_closed_reference.png">Half-closed eyes · expression</option><option value="half_closed_reference2.png">Half-closed eyes · reference 2</option><option value="landau_test.png">Landau test · illustration</option><option value="reference.png">Landau v10 · generation source</option></select><label class="char-check"><input type="checkbox" data-crop checked> Face close-up</label><div class="char-reference-crop closeup" data-reference-frame><img data-reference-image src="${asset('inputs/landau_v10/landau_test2.png')}" alt="Landau reference face"></div><a class="char-reference-link" data-reference-link href="${asset('inputs/landau_v10/landau_test2.png')}" target="_blank" rel="noopener">Open full reference</a><p class="char-hint">Each reference has different proportions. They guide likeness; the original mesh remains available at neutral.</p></div>
      <div class="char-pane" data-pane="presets" hidden></div>
      <div class="char-pane" data-pane="asset" hidden><div data-asset-report></div><figure><img src="${asset('inputs/landau_v10/reference.png')}" alt="Original Landau v10 generation reference"><figcaption>Original generation reference</figcaption></figure><label class="char-check">Export textures <select aria-label="Export texture resolution" data-export-resolution><option value="4096">4096 · source detail</option><option value="2048">2048 · smaller file</option><option value="1024">1024 · compact</option></select></label><label class="char-check"><input type="checkbox" data-visible-only checked> Export visible parts only</label><div class="char-downloads"><a href="${asset('outputs/landau_v10/landau_character.blend')}&download=1" download>Blender master</a><a href="${asset('outputs/landau_v10/asset_report.json')}&download=1" download>Asset report</a></div></div>
    </aside></div><footer class="char-footer"><span data-status role="status">Loading the authored GLB…</span><a data-download hidden>Download prepared file</a><span data-stats></span></footer><input type="file" accept=".json,application/json" data-file hidden>
  </div>`;

  const stage=query('.char-viewport'),canvas=query('canvas');
  const scene=new THREE.Scene();scene.background=new THREE.Color('#202b32');
  const camera=new THREE.PerspectiveCamera(33,1,.12,10);camera.position.set(0,.64,2.4);
  renderer=new THREE.WebGLRenderer({canvas,antialias:true,preserveDrawingBuffer:false});renderer.setPixelRatio(Math.min(devicePixelRatio,2));renderer.outputColorSpace=THREE.SRGBColorSpace;renderer.toneMapping=THREE.ACESFilmicToneMapping;renderer.toneMappingExposure=1.25;
  const orbit=new OrbitControls(camera,canvas);orbit.target.set(0,.58,0);orbit.enableDamping=true;orbit.minDistance=.18;orbit.maxDistance=6;orbit.maxPolarAngle=Math.PI*.95;
  scene.add(new THREE.HemisphereLight(0xe7f3ff,0x666352,2.4));
  for(const [pos,color,intensity] of [[[1,2,2],0xffeed9,2.6],[[-2,1,1],0xd4eaff,1.7],[[0,2,-2],0x9ccfe0,2.2]]){const light=new THREE.DirectionalLight(color,intensity);light.position.set(...pos);scene.add(light);}
  const grid=new THREE.GridHelper(2,20,0x5a747d,0x36454e);grid.material.opacity=.35;grid.material.transparent=true;scene.add(grid);
  const resize=new ResizeObserver(()=>{if(dead)return;const w=stage.clientWidth,h=stage.clientHeight;renderer.setSize(w,h,false);camera.aspect=w/h;camera.updateProjectionMatrix();});resize.observe(stage);
  const released=new Set();
  function release(resource){if(resource&&!released.has(resource)){released.add(resource);resource.dispose?.();}}
  function releaseTree(tree){tree.traverse(o=>{release(o.geometry);if(o.skeleton)release(o.skeleton);for(const m of (Array.isArray(o.material)?o.material:[o.material]))if(m){for(const v of Object.values(m))if(v?.isTexture)release(v);release(m);}});}
  const dispose=()=>{if(dead)return;dead=true;abort.abort();if(downloadUrl)URL.revokeObjectURL(downloadUrl);cancelAnimationFrame(frame);player?.dispose();resize.disconnect();orbit.dispose();releaseTree(scene);for(const m of baseMaterials.keys())release(m);renderer.dispose();};
  // Return a lifecycle handle immediately, including while the asynchronous loader runs.
  root.charDispose=dispose;
  function saveLocal(){try{localStorage.setItem('landau-char-v1',JSON.stringify(settings));dirty=false;}catch{announce('Browser storage unavailable; save a version in sandbox history to keep your edits.');}}
  function touch(){dirty=true;saveLocal();}
  function setMorph(name,value,persist=true){
    if(!morphs.has(name))return;
    if(!name.startsWith('_'))settings.morphs[name]=value;for(const [mesh,index] of morphs.get(name)||[])mesh.morphTargetInfluences[index]=value;
    applyBlink(morphs,settings.morphs);
    for(const input of root.querySelectorAll(`input[data-morph="${name}"]`)){input.value=value;input.nextElementSibling.value=Number(value).toFixed(2);}
    // The mouth targets change only the face. Rebuilding clothing and its
    // shoulder weights here stalled every mouth-slider input for over 500 ms.
    if(persist){if(SHAPES.has(name)&&name!=='mouthLength'&&name!=='mouthCurvature')applyFit();touch();}
  }
  function partColor(name,m){const colors=settings.parts[name]?.materialColors||{};const aliases=report?.clothing_segmentation?.garments?.[name]?.legacy_material_names||[];return colors[m.name]||aliases.map(n=>colors[n]).find(Boolean)||baseMaterials.get(m).color;}
  function applyParts(){for(const [name,p] of parts){const cfg=settings.parts[name]||{};p.object.visible=cfg.visible??p.defaultVisible;for(const m of p.materials)m.color.set(partColor(name,m));}}
  function applyBone(name){const item=bones.get(name);if(!item)return;const cfg=settings.bones[name]||{rotation:[0,0,0],scale:1};const q=new THREE.Quaternion().setFromEuler(new THREE.Euler(...cfg.rotation.map(v=>v*Math.PI/180),'XYZ'));item.bone.quaternion.copy(player?.quaternion(name)||editPose?.quaternion(name,settings.editingPose)||item.quaternion).multiply(q);item.bone.scale.copy(item.scale).multiplyScalar(cfg.scale);}
  function readBone(){if(!selectedBone)return;const v=settings.bones[selectedBone]||{rotation:[0,0,0],scale:1};for(const input of root.querySelectorAll('[data-bone-axis]')){const i=Number(input.dataset.boneAxis);input.value=v.rotation[i];input.nextElementSibling.value=v.rotation[i].toFixed(0)+'°';}const input=query('[data-bone-scale]');input.value=v.scale;input.nextElementSibling.value=v.scale.toFixed(2);}
  function readMaterial(){const p=parts.get(selectedPart);if(!p)return;const name=query('[data-material]').value;const m=[...p.materials].find(m=>m.name===name);if(m)query('[data-tint]').value='#'+new THREE.Color(partColor(selectedPart,m)).getHexString();}
  function selectPart(name){if(!parts.has(name))return;selectedPart=name;query('[data-selection]').textContent=pretty(name);query('[data-part-title]').textContent=pretty(name);query('[data-material]').innerHTML=[...parts.get(name).materials].map(m=>`<option value="${esc(m.name)}">${esc(m.name)}</option>`).join('');readMaterial();root.querySelectorAll('[data-part]').forEach(el=>el.classList.toggle('selected',el.dataset.part===name));}
  function shading(){const mode=query('[data-shading]').value;let i=0;for(const [name,p] of parts){for(const m of p.materials){const b=baseMaterials.get(m);for(const [key,val]of Object.entries(b.maps))m[key]=(mode==='clay'||mode==='regions')?null:val;m.wireframe=mode==='wire';m.vertexColors=(mode==='clay'||mode==='regions')?false:b.vertexColors;m.roughness=mode==='clay'?.8:b.roughness;m.metalness=(mode==='clay'||mode==='regions')?0:b.metalness;if(mode==='clay')m.color.set('#c8d2d2');else if(mode==='regions')m.color.set(COLORS[i%COLORS.length]);else m.color.set(partColor(name,m));m.needsUpdate=true;}i++;}}
  function view(name){
    activeView=name;
    const reportedLift=report?.body_reconstruction?.head_rigid_lift;
    const lift=Number.isFinite(reportedLift)?reportedLift:0;
    const extra=.806*((settings.bodyFrame?.scale??1)-1),center=.58+lift/2-extra/2,height=.64+lift/2-extra/2,distance=2.4*(1+(lift+extra)/1.16);
    orbit.target.set(0,name==='face'?.71+lift:center,name==='face'?.01:0);
    const positions={full:[0,height,distance],face:[0,.73+lift,.72],side:[distance,height,0],back:[0,height,-distance]};
    camera.position.set(...positions[name]);orbit.update();
  }
  function download(data,name,type){if(downloadUrl)URL.revokeObjectURL(downloadUrl);downloadUrl=URL.createObjectURL(new Blob([data],{type}));const a=query('[data-download]');a.href=downloadUrl;a.download=name;a.textContent='Download '+name;a.hidden=false;a.click();}
  function compatiblePreset(hash){return hash===report.glb_sha256||(report.preset_compatible_hashes||[]).includes(hash);}
  function validatePreset(p){
    if(!p||p.version!==1||!compatiblePreset(p.assetHash))throw new Error('This preset belongs to a different asset revision.');
    p=migrateFacePreset(p,report);
    for(const [n,v] of Object.entries(p.morphs||{})){if(!morphs.has(n)||!Number.isFinite(v)||v<(OUTFIT.has(n)?outfitMinimum(n,adjustmentMeta[n]?.min??0):(adjustmentMeta[n]?.min??(SHAPES.has(n)?-1:0)))||v>1)throw new Error('Invalid morph: '+n);}
    for(const [n,v] of Object.entries(p.bones||{})){if(!bones.has(n)||!Array.isArray(v.rotation)||v.rotation.length!==3||v.rotation.some(x=>!Number.isFinite(x)||Math.abs(x)>120)||!Number.isFinite(v.scale)||v.scale<.7||v.scale>1.3)throw new Error('Invalid bone: '+n);}
    for(const [n,v] of Object.entries(p.parts||{})){if(!parts.has(n)||!v||typeof v!=='object'||(v.visible!==undefined&&typeof v.visible!=='boolean')||(v.tint!==undefined&&!/^#[0-9a-f]{6}$/i.test(v.tint)))throw new Error('Invalid part: '+n);}
    for(const [n,v] of Object.entries(p.parts||{}))for(const [material,color]of Object.entries(v.materialColors||{}))if((![...parts.get(n).materials].some(m=>m.name===material)&&!(report.clothing_segmentation?.garments?.[n]?.legacy_material_names||[]).includes(material))||!/^#[0-9a-f]{6}$/i.test(color))throw new Error('Invalid material color: '+material);
    for(const [part,values] of Object.entries(p.outfit||{}))for(const [n,v] of Object.entries(values)){if(!garmentNames(part).includes(n)||!Number.isFinite(v)||v<outfitMinimum(n,adjustmentMeta[n]?.min??0)||v>1)throw new Error('Invalid clothing adjustment');}
    for(const [n,v]of Object.entries(p.links||{}))if(!GARMENT_CARDS.some(g=>g.id===n)||typeof v!=='boolean')throw new Error('Invalid pair link');
    const frame={scale:1,offset:0,shoulderHeight:0,...p.bodyFrame};if(!Number.isFinite(frame.scale)||frame.scale<.75||frame.scale>1.25||!Number.isFinite(frame.offset)||Math.abs(frame.offset)>.05||!Number.isFinite(frame.shoulderHeight)||Math.abs(frame.shoulderHeight)>.03)throw new Error('Invalid body placement');
    if(p.editingPose!==undefined&&!['t','a'].includes(p.editingPose))throw new Error('Invalid editing pose');
    return {version:1,assetHash:report.glb_sha256,editingPose:p.editingPose??'t',morphs:p.morphs||{},bones:p.bones||{},parts:p.parts||{},outfit:p.outfit||{},links:p.links||{},bodyFrame:frame,garmentFit:fitDefaults(p.garmentFit)};
  }
  function applyPreset(p,persist=true){player?.reset();query('[data-blink]').checked=false;settings=validatePreset(p);query('[data-editing-pose]').value=settings.editingPose;for(const n of morphs.keys())setMorph(n,settings.morphs[n]||0,false);for(const n of bones.keys())applyBone(n);applyParts();shading();root.querySelectorAll('[data-visible]').forEach(e=>e.checked=settings.parts[e.dataset.visible]?.visible??parts.get(e.dataset.visible).defaultVisible);for(const g of GARMENT_CARDS)for(const n of g.parts)for(const name of garmentNames(n)){const v=outfitValue(n,name);parts.get(n)?.object.traverse(o=>{const i=o.morphTargetDictionary?.[name];if(i!==undefined)o.morphTargetInfluences[i]=v;});}applyBodyFrame();renderGarments();readBone();readMaterial();if(persist)touch();}
  function expression(kind){query('[data-blink]').checked=false;for(const n of morphs.keys())if(!SHAPES.has(n)&&!OUTFIT.has(n))setMorph(n,0,false);const p={neutral:{},mouth:{jawDrop:1},happy:{mouthSmile:.7,eyeSquintL:.15,eyeSquintR:.15},surprised:{jawDrop:.75,browUpL:.65,browUpR:.65,eyeWideL:.5,eyeWideR:.5},half:{eyeBlinkL:.5,eyeBlinkR:.5},blink:{eyeBlinkL:1,eyeBlinkR:1},look:{eyeLookOutL:.75,eyeLookInR:.75}}[kind];for(const [n,v]of Object.entries(p))setMorph(n,v,false);touch();announce('Expression: '+kind);}
  function control(name,part=''){
    const original=splitDepth(name),baseName=original?.base||name;const min=part?outfitMinimum(name,adjustmentMeta[baseName]?.min??0):(adjustmentMeta[name]?.min??(SHAPES.has(name)?-1:0)),label=esc(({bootShaftWidth:'Top opening width',bootShaftDepth:'Top opening depth',bootShaftHeight:'Top / trouser hem height',bootRaise:'Foot up / down',bootForward:'Foot forward / back',bootSpread:'Foot outward',sleeveRaise:'Sleeve free end up / down',sleeveForward:'Sleeve free end forward',sleeveSpread:'Sleeve free end outward'})[baseName]||COLLAR_CONTROLS[name]||SHOULDER_CONTROLS[name]||adjustmentMeta[baseName]?.label||OUTFIT_LABELS[baseName]||pretty(name));
    const shownLabel=original?label.replace(/(?:depth|room)/i,original.side.toLowerCase()+' depth'):label;
    return widget({label:shownLabel,min,attrs:part?`data-outfit="${esc(name)}" data-garment="${esc(part)}"`:`data-morph="${esc(name)}"`,reset:part?`data-reset-outfit="${esc(name)}" data-garment="${esc(part)}"`:`data-reset-morph="${esc(name)}"`,life:SHAPES.has(name)||part?'edit':'live',op:part&&/(Raise|Forward|Spread)$/.test(name)?'move':'morph'});
  }
  function garmentNames(part){const found=new Set();parts.get(part)?.object.traverse(o=>{for(const n of Object.keys(o.morphTargetDictionary||{}))if(OUTFIT.has(n)||n==='clothingEase')found.add(n);});for(const n of inheritedControls(part))found.add(n);for(const n of [...found])for(const side of depthNames(n))found.add(side);if(part==='Vest')for(const n of Object.keys({...COLLAR_CONTROLS,...SHOULDER_CONTROLS}))found.add(n);return [...found];}
  function garmentGroup(part){return GARMENT_CARDS.find(g=>g.parts.includes(part));}
  function linked(group){return settings.links[group.id]!==false;}
  function outfitValue(part,name){return linkedOutfitValue(settings,part,name);}
  function applyFit(){fit?.apply(settings);}
  function setOutfit(part,name,value,persist=true){
    setOutfitValue(settings,part,name,value);applyFit();syncGarmentControls();if(persist)touch();
  }
  function syncGarmentControls(){
    root.querySelectorAll('[data-outfit]').forEach(el=>{const p=el.dataset.garment,n=el.dataset.outfit,v=outfitValue(p,n),[owner]=sharedOwner(p,n);el.value=v;el.nextElementSibling.value=Number(v).toFixed(2);const locked=owner!==p&&!settings.garmentFit.unlocked[p];el.disabled=locked;el.closest('.char-property').classList.toggle('inherited',owner!==p);el.closest('.char-property').querySelector('button').disabled=locked;});
    root.querySelectorAll('[data-fit-scale]').forEach(el=>{el.value=settings.garmentFit.scales[el.dataset.fitScale];el.nextElementSibling.value=Number(el.value).toFixed(2);const locked=!!PARENT[el.dataset.garment]&&!settings.garmentFit.unlocked[el.dataset.garment];el.disabled=locked;el.closest('.char-property').querySelector('button').disabled=locked;});
    root.querySelectorAll('[data-fit-angle]').forEach(el=>{const part=el.dataset.garment,name=el.dataset.fitAngle,v=settings.garmentFit.angles[part]?.[name]||0,[peer,key]=jointPeer(part,name),other=settings.garmentFit.angles[peer]?.[key]||0,mixed=jointsLinked(settings.garmentFit,part)&&v!==other;el.value=v;el.nextElementSibling.value=mixed?'mixed':Number(v).toFixed(0)+'°';el.title=mixed?`Current values: ${v}° / ${other}°. Adjust to set both sides.`:'';});
    root.querySelectorAll('[data-garment-visible]').forEach(el=>{const group=garmentGroup(el.dataset.garmentVisible);const names=group&&linked(group)?group.parts:[el.dataset.garmentVisible];el.checked=names.every(n=>settings.parts[n]?.visible??parts.get(n)?.defaultVisible);el.indeterminate=names.some(n=>settings.parts[n]?.visible??parts.get(n)?.defaultVisible)&&!el.checked;});
  }
  function fitScaleControl(part){const g=assemblyOf(part);return widget({label:g.label+' scale',attrs:`data-fit-scale="${g.id}" data-garment="${part}"`,min:.5,max:2,value:1,reset:`data-reset-fit-scale="${g.id}" data-garment="${part}"`,op:'scale'});}
  function fittingAngles(part,open){
    if(!ANGLES[part])return '';const together=jointsLinked(settings.garmentFit,part),names=ANGLES[part].filter(n=>part!=='Trousers'||!together||!n.includes('R'));
    return `<details class="char-angle-controls" data-joint-card="${part}" ${open?'open':''}><summary>Clothing joint angles</summary><label class="char-shared-unlock"><input type="checkbox" data-joint-link="${part}" ${together?'checked':''}> Adjust both L / R</label><p class="char-hint">Rest-fit angles only. The body pose stays unchanged.${part.startsWith('Sleeve')?' Cuffs follow the sleeve.':' Boots follow their leg.'}</p>${names.map(n=>widget({label:angleLabel(together?n.replace(/L([XYZ])$/,'$1'):n),attrs:`data-fit-angle="${n}" data-garment="${part}"`,min:-90,max:90,step:1,reset:`data-reset-fit-angle="${n}" data-garment="${part}"`,op:'rotate',axis:n.slice(-1).toLowerCase()})).join('')}</details>`;
  }
  function renderGarments(){
    const open=new Set([...root.querySelectorAll('[data-garment-card][open]')].map(e=>e.dataset.garmentCard)),angleOpen=new Set([...root.querySelectorAll('[data-joint-card][open]')].map(e=>e.dataset.jointCard));
    query('[data-garment-cards]').innerHTML=OUTFIT_ASSEMBLIES.map(assembly=>`<section class="char-outfit-assembly"><h3>${assembly.label}</h3><p class="char-hint">${assembly.id==='upper'?'Vest → sleeves → cuffs':'Trousers → boots'} · shared scale and connected seams</p>${assembly.id==='upper'?`<label class="char-check"><input type="checkbox" data-shoulder-follow ${settings.garmentFit.shoulderFollow?'checked':''}> Follow body at shoulders</label><p class="char-hint">Updates shoulder skin weights after fit edits. Garment shape stays manual.</p>`:''}${GARMENT_CARDS.filter(g=>assembly.parts.includes(g.parts[0])).map(g=>{
      const pair=g.parts.length===2,together=linked(g);
      return `<section class="char-garment-group">${g.parts.filter((n,i)=>!i||!together).map((n,i)=>`<details class="char-garment-card" data-garment-card="${n}" ${open.has(n)?'open':''}><summary><span class="char-disclosure">${icon('chevron')}</span>${icon('shirt')}<span class="char-garment-name">${g.label}${pair&&!together?` · ${i?'R':'L'}`:''}</span>${pair&&!i?`<label class="char-icon-toggle char-link-toggle" title="Mirror left and right adjustments"><input type="checkbox" aria-label="Link ${g.label.toLowerCase()} left and right" data-garment-link="${g.id}" ${together?'checked':''}>${icon('link')}</label>`:''}<label class="char-icon-toggle char-eye-toggle" title="Show ${g.label}${pair&&!together?` ${i?'right':'left'}`:''}"><input type="checkbox" aria-label="Show ${g.label}${pair&&!together?` ${i?'right':'left'}`:''}" data-garment-visible="${n}">${icon('eye')}</label></summary><div class="char-garment-options">${PARENT[n]?`<p class="char-hint">Attached to ${pretty(PARENT[n])}. Shared values below edit the parent too.</p><label class="char-shared-unlock"><input type="checkbox" data-fit-unlock="${n}" ${settings.garmentFit.unlocked[n]?'checked':''}> Edit shared parent controls</label>`:`<p class="char-hint">Parent section · changes carry connected children.</p>`}${fitScaleControl(n)}${n==='Vest'?`<div class="char-collar-controls"><h4>Collar / neckline</h4>${Object.keys(COLLAR_CONTROLS).map(name=>control(name,n)).join('')}</div>`:''}${n==='Vest'||n.startsWith('Sleeve')?`<div class="char-collar-controls"><h4>Shoulder / sleeve join</h4>${n==='Vest'?'':'<p class="char-inheritance-note">Shared with Vest</p>'}${Object.keys(SHOULDER_CONTROLS).map(name=>control(name,n)).join('')}</div>`:''}${fittingAngles(n,angleOpen.has(n))}${garmentNames(n).filter(name=>!splitDepth(name)&&!(name in COLLAR_CONTROLS)&&!(name in SHOULDER_CONTROLS)).flatMap(depthNames).map(name=>{const [owner]=sharedOwner(n,name);return (owner!==n?`<div class="char-inheritance-note">From ${pretty(owner)}</div>`:'')+control(name,n);}).join('')}</div></details>`).join('')}</section>`;
    }).join('')}</section>`).join('');syncGarmentControls();
  }
  function setFitAngle(part,name,value){setJointAngle(settings.garmentFit,part,name,value);applyFit();syncGarmentControls();touch();}
  function applyBodyFrame(){
    const scale=settings.bodyFrame.scale;
    if(activeView!=='face'){
      const height=1.16+(report?.body_reconstruction?.head_rigid_lift||0),ratio=(height+.806*(scale-1))/(height+.806*(frameScale-1)),direction=camera.position.clone().sub(orbit.target);
      orbit.target.y-=.806*(scale-frameScale)/2;camera.position.copy(orbit.target).addScaledVector(direction,ratio);
    }
    frameScale=scale;grid.position.y=.806*(1-scale);
    player?.beforeRestEdit();placeBody?.(scale,settings.bodyFrame.offset,settings.bodyFrame.shoulderHeight||0);editPose=editingPose(model,bones);fit?.capture();applyFit();player?.afterRestEdit(scale);for(const n of bones.keys())applyBone(n);for(const input of root.querySelectorAll('[data-body-frame]')){input.value=settings.bodyFrame[input.dataset.bodyFrame]??0;input.nextElementSibling.value=Number(input.value).toFixed(input.dataset.bodyFrame==='scale'?2:3);}}
  function tick(t){if(dead)return;frame=requestAnimationFrame(tick);player?.tick(lastTick?(t-lastTick)/1000:0);lastTick=t;updateMotion?.();if(query('[data-blink]').checked){const phase=(t/1000)%3.4;animation=Math.max(0,1-Math.abs(phase-.35)/.16);applyBlink(morphs,settings.morphs,animation);}else applyBlink(morphs,settings.morphs);orbit.update();updateCameraDepth(camera,orbit.target);renderer.render(scene,camera);}
  frame=requestAnimationFrame(tick);

  function showTab(name){root.querySelectorAll('[data-tab]').forEach(b=>b.classList.toggle('active',b.dataset.tab===name));root.querySelectorAll('[data-pane]').forEach(p=>p.hidden=p.dataset.pane!==name);}
  root.addEventListener('click',async e=>{
    if(e.target.closest('.char-icon-toggle')){e.stopPropagation();return;}
    const button=e.target.closest('button');if(!button||exportBusy)return;
    if(button.dataset.resetMorph){setMorph(button.dataset.resetMorph,0);return;}
    if(button.dataset.resetFitScale){settings.garmentFit.scales[button.dataset.resetFitScale]=1;applyFit();syncGarmentControls();touch();return;}
    if(button.dataset.resetFitAngle){setFitAngle(button.dataset.garment,button.dataset.resetFitAngle,0);return;}
    if(button.dataset.resetOutfit){setOutfit(button.dataset.garment,button.dataset.resetOutfit,0);return;}
    if(button.dataset.resetFrame){settings.bodyFrame[button.dataset.resetFrame]=button.dataset.resetFrame==='scale'?1:0;applyBodyFrame();touch();return;}
    if(button.dataset.resetJoint){const c=settings.bones[selectedBone]||{rotation:[0,0,0],scale:1};if(button.dataset.resetJoint==='scale')c.scale=1;else c.rotation[Number(button.dataset.resetJoint)]=0;settings.bones[selectedBone]=c;applyBone(selectedBone);readBone();touch();return;}
    if(button.dataset.tab){showTab(button.dataset.tab);if(button.dataset.tab==='presets')void history?.refresh();}
    if(button.dataset.view)view(button.dataset.view);
    if(button.dataset.expression&&model)expression(button.dataset.expression);
    if(button.dataset.part)selectPart(button.dataset.part);
    const action=button.dataset.action;if(action==='focus'){const expanded=query('.char-editor').classList.toggle('char-expanded');button.textContent=expanded?'Exit expanded view':'Expand editor';return;}if(!action||!model)return;
    try{
      if(action==='reset'){query('[data-blink]').checked=false;applyPreset({version:1,assetHash:report.glb_sha256});announce('All character edits reset.');}
      if(action==='reset-body'){settings.bodyFrame={scale:1,offset:0};applyBodyFrame();for(const [n,m]of Object.entries(adjustmentMeta))if(m.kind==='body')setMorph(n,0,false);applyFit();touch();announce('Body proportions restored to the uniformly scaled FBX.');}
      if(action==='reset-outfit'){settings.outfit={};settings.garmentFit=fitDefaults();for(const n of OUTFIT)settings.morphs[n]=0;applyFit();renderGarments();touch();announce('Connected clothing fit reset.');}
      if(action==='save'){showTab('presets');await history.save();}
      if(action==='preset-download')download(JSON.stringify(settings,null,2),'landau-v10-preset.json','application/json');
      if(action==='load')query('[data-file]').click();
      if(action==='undress'||action==='dress'){for(const [n,p]of parts)if(p.kind==='clothing')settings.parts[n]={...settings.parts[n],visible:action==='dress'};for(const [n,p]of parts)if(p.kind==='inferred_body')settings.parts[n]={...settings.parts[n],visible:true};applyParts();shading();root.querySelectorAll('[data-visible]').forEach(el=>el.checked=settings.parts[el.dataset.visible]?.visible??parts.get(el.dataset.visible).defaultVisible);syncGarmentControls();touch();}
      if(action==='untint'&&selectedPart){const colors=settings.parts[selectedPart]?.materialColors||{};delete colors[query('[data-material]').value];shading();readMaterial();touch();}
      if((action==='isolate'&&selectedPart)||action==='show-character'){for(const [n,p]of parts)settings.parts[n]={...settings.parts[n],visible:action==='isolate'?n===selectedPart:p.defaultVisible};applyParts();shading();root.querySelectorAll('[data-visible]').forEach(e=>e.checked=settings.parts[e.dataset.visible].visible);syncGarmentControls();touch();}
      if(action==='reset-bone'&&selectedBone){delete settings.bones[selectedBone];applyBone(selectedBone);readBone();touch();}
      if(action==='reset-pose'){settings.bones={};for(const n of bones.keys())applyBone(n);readBone();touch();}
      if(action==='export'){
        exportBusy=true;button.disabled=true;query('.char-layout').inert=true;query('.char-actions').inert=true;announce('Exporting skin, morph targets, textures and current settings…');
        player?.pause();query('[data-blink]').checked=false;for(const n of ['eyeBlinkL','eyeBlinkR'])setMorph(n,settings.morphs[n]||0,false);
        const old=query('[data-shading]').value;query('[data-shading]').value='textured';shading();
        const restoreClothing=fit?.prepareExport();
        try{const {GLTFExporter}=await import('/api/artifact?path=algorithms/3d_char_details/gui/vendor/GLTFExporter.js');if(dead)return;const result=await new GLTFExporter().parseAsync(model,{binary:true,trs:true,onlyVisible:query('[data-visible-only]').checked,maxTextureSize:Number(query('[data-export-resolution]').value)});if(dead)return;download(result,'landau-v10-edited.glb','model/gltf-binary');announce('Edited GLB exported. Facial morphs, skin and the current motion pose are retained.');}finally{restoreClothing?.();exportBusy=false;if(!dead){query('[data-shading]').value=old;shading();button.disabled=false;query('.char-layout').inert=false;query('.char-actions').inert=false;}}
      }
    }catch(err){announce(err.message);button.disabled=false;}
  },{signal:abort.signal});
  root.addEventListener('input',e=>{
    const el=e.target;if(!model||exportBusy)return;
    if(el.dataset.fitScale){settings.garmentFit.scales[el.dataset.fitScale]=Number(el.value);applyFit();syncGarmentControls();touch();}
    if(el.dataset.fitAngle)setFitAngle(el.dataset.garment,el.dataset.fitAngle,Number(el.value));
    if(el.dataset.outfit)setOutfit(el.dataset.garment,el.dataset.outfit,Number(el.value));
    if(el.dataset.bodyFrame){settings.bodyFrame[el.dataset.bodyFrame]=Number(el.value);applyBodyFrame();touch();}
    if(el.dataset.morph)setMorph(el.dataset.morph,Number(el.value));
    if(el.hasAttribute('data-bone-axis')||el.hasAttribute('data-bone-scale')){if(!selectedBone)return;const cfg=settings.bones[selectedBone]||{rotation:[0,0,0],scale:1};if(el.hasAttribute('data-bone-axis'))cfg.rotation[Number(el.dataset.boneAxis)]=Number(el.value);else cfg.scale=Number(el.value);settings.bones[selectedBone]=cfg;applyBone(selectedBone);readBone();touch();}
    if(el.hasAttribute('data-tint')&&selectedPart){const cfg=settings.parts[selectedPart]||{};settings.parts[selectedPart]={...cfg,materialColors:{...cfg.materialColors,[query('[data-material]').value]:el.value}};shading();touch();}
  },{signal:abort.signal});
  root.addEventListener('change',async e=>{
    const el=e.target;if(exportBusy)return;
    if(el.hasAttribute('data-reference')){const url=asset('inputs/landau_v10/'+el.value);query('[data-reference-image]').src=url;query('[data-reference-image]').alt=el.selectedOptions[0].textContent;query('[data-reference-link]').href=url;query('[data-reference-frame]').dataset.source=el.value;if(['internal_reference.png','side_muzzle_reference.png'].includes(el.value)){query('[data-crop]').checked=false;query('[data-reference-frame]').classList.remove('closeup');}}
    if(el.hasAttribute('data-crop'))query('[data-reference-frame]').classList.toggle('closeup',el.checked);
    if(el.hasAttribute('data-shading'))shading();
    if(el.hasAttribute('data-skeleton')&&helper)helper.visible=el.checked;
    if(el.hasAttribute('data-shoulder-follow')){settings.garmentFit.shoulderFollow=el.checked;applyFit();touch();}
    if(el.dataset.jointLink){setJointLink(settings.garmentFit,el.dataset.jointLink,el.checked);applyFit();renderGarments();touch();}
    if(el.dataset.fitUnlock){const g=garmentGroup(el.dataset.fitUnlock);for(const p of linked(g)?g.parts:[el.dataset.fitUnlock])settings.garmentFit.unlocked[p]=el.checked;renderGarments();touch();}
    if(el.dataset.garmentLink){const group=GARMENT_CARDS.find(g=>g.id===el.dataset.garmentLink);settings.links[group.id]=el.checked;if(group.id==='sleeves'&&!el.checked)settings.links.cuffs=false;if(el.checked){for(const n of garmentNames(group.parts[0]))setOutfitValue(settings,group.parts[0],n,outfitValue(group.parts[0],n));const visible=settings.parts[group.parts[0]]?.visible??parts.get(group.parts[0]).defaultVisible;for(const n of group.parts)settings.parts[n]={...settings.parts[n],visible};applyParts();}applyFit();renderGarments();touch();}
    if(el.dataset.garmentVisible){const g=garmentGroup(el.dataset.garmentVisible);for(const n of linked(g)?g.parts:[el.dataset.garmentVisible])settings.parts[n]={...settings.parts[n],visible:el.checked};applyParts();syncGarmentControls();touch();}
    if(el.dataset.visible){settings.parts[el.dataset.visible]={...settings.parts[el.dataset.visible],visible:el.checked};applyParts();shading();touch();}
    if(el.hasAttribute('data-editing-pose')){settings.editingPose=el.value;player?.reset();for(const n of bones.keys())applyBone(n);updateMotion?.();touch();}
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
    adjustmentMeta={...report.body_reconstruction?.adjustment_controls,mouthLength:{label:'Mouth length',min:-1,max:1},mouthCurvature:{label:'Mouth curvature',min:-1,max:1}};
    for(const [n,m]of Object.entries(adjustmentMeta)){if(m.kind==='body')SHAPES.add(n);if(m.kind==='outfit')OUTFIT.add(n);}
    const gltf=await new GLTFLoader().loadAsync(asset('outputs/landau_v10/landau_character.glb'));if(dead){releaseTree(gltf.scene);return;}
    model=gltf.scene;model.name='Landau_v10';scene.add(model);
    installBodyTransition(model);
    for(const [n,label]of Object.entries(BODY_TRANSITION_CONTROLS)){SHAPES.add(n);adjustmentMeta[n]={label,kind:'body',group:'Lower neck / upper chest',min:-1,max:1};}
    model.traverse(o=>{
      if(o.isBone)bones.set(o.name,{bone:o,quaternion:o.quaternion.clone(),scale:o.scale.clone()});
      if(o.userData.part_type)parts.set(o.name,{object:o,kind:o.userData.part_type,defaultVisible:!o.userData.default_hidden,materials:new Set()});
    });
    model.traverse(o=>{if(!o.isMesh)return;let parent=o;while(parent&&!parts.has(parent.name))parent=parent.parent;if(!parent){parent=o;parts.set(o.name,{object:o,kind:'mesh',defaultVisible:true,materials:new Set()});}const p=parts.get(parent.name);
      const materials=(Array.isArray(o.material)?o.material:[o.material]).map(m=>{const copy=m.clone();eyeLayerDepth(copy,parent.name);baseMaterials.set(copy,{color:copy.color.clone(),map:copy.map,normalMap:copy.normalMap,roughness:copy.roughness,metalness:copy.metalness,vertexColors:copy.vertexColors,maps:Object.fromEntries(Object.entries(copy).filter(([k,v])=>k.endsWith('Map')||k==='map'))});p.materials.add(copy);release(m);return copy;});o.material=Array.isArray(o.material)?materials:materials[0];o.frustumCulled=false;
      for(const [name,index]of Object.entries(o.morphTargetDictionary||{})){if(!morphs.has(name))morphs.set(name,[]);morphs.get(name).push([o,index]);}
    });
    helper=new THREE.SkeletonHelper(model);helper.visible=false;helper.material.depthTest=false;helper.renderOrder=10;scene.add(helper);
    query('[data-face-sliders]').innerHTML=[...morphs.keys()].filter(n=>!n.startsWith('_')&&!SHAPES.has(n)&&!OUTFIT.has(n)).sort().map(n=>control(n)).join('');
    query('[data-shape-sliders]').innerHTML=`<section class="char-collar-controls"><h4>Mouth</h4><p class="char-hint">Length: shorter to wider. Curvature: corners down to corners up. Jaw opening is in Face.</p>${['mouthLength','mouthCurvature'].filter(n=>morphs.has(n)).map(n=>control(n)).join('')}</section><section class="char-collar-controls"><h4>Lower neck / upper chest</h4>${Object.keys(BODY_TRANSITION_CONTROLS).map(n=>control(n)).join('')}</section>`+[...morphs.keys()].filter(n=>SHAPES.has(n)&&!(n in BODY_TRANSITION_CONTROLS)&&!['mouthLength','mouthCurvature'].includes(n)).sort().map(n=>control(n)).join('');
    query('[data-part-list]').innerHTML=[...parts].filter(([,p])=>p.kind!=='clothing').map(([n,p])=>`<div class="char-part"><input type="checkbox" aria-label="Show ${esc(pretty(n))}" data-visible="${esc(n)}" ${p.defaultVisible?'checked':''}><button data-part="${esc(n)}">${esc(n==='Body_Complete'?'Connected skin':pretty(n))}</button></div>`).join('');
    query('[data-frame-sliders]').innerHTML=widget({label:'Body scale',attrs:'data-body-frame="scale"',min:.75,max:1.25,value:1,reset:'data-reset-frame="scale"',op:'scale',scope:'world'})+widget({label:'Body left / right',attrs:'data-body-frame="offset"',min:-.05,max:.05,step:.001,reset:'data-reset-frame="offset"',op:'move',scope:'world',axis:'x'})+`<h4>Shoulder joints</h4>`+widget({label:'Shoulder joint height',attrs:'data-body-frame="shoulderHeight"',min:-.03,max:.03,step:.001,reset:'data-reset-frame="shoulderHeight"',op:'move',scope:'world',axis:'y'})+`<p class="char-hint">Raises or lowers both arm pivots. The neutral A-pose surface stays fixed; inspect the change in T-pose or motion.</p>`;
    editPose=editingPose(model,bones);placeBody=bodyPlacement(model,bones);fit=garmentFit(model,parts,report);
    query('[data-bone]').innerHTML=[...bones.keys()].map(n=>`<option value="${esc(n)}">${esc(pretty(n))}</option>`).join('');selectedBone=bones.has('head_x')?'head_x':bones.keys().next().value;query('[data-bone]').value=selectedBone;
    query('[data-bone-sliders]').innerHTML=['X','Y','Z'].map((a,i)=>widget({label:`Rotation ${a}`,attrs:`data-bone-axis="${i}"`,min:-120,max:120,step:1,reset:`data-reset-joint="${i}"`,life:'live',op:'rotate',axis:a.toLowerCase()})).join('')+widget({label:'Scale',attrs:'data-bone-scale',min:.7,max:1.3,value:1,reset:'data-reset-joint="scale"',life:'live',op:'scale'});
    const v=report.validation;query('[data-asset-report]').innerHTML=`<h3>Structure and rig preview</h3><dl class="char-facts"><dt>Meshes</dt><dd>${v.mesh_count}</dd><dt>Triangles</dt><dd>${v.triangles.toLocaleString()}</dd><dt>Bones</dt><dd>${v.bones}</dd><dt>Facial + shape controls</dt><dd>${morphs.size}</dd><dt>Invalid skin weights</dt><dd>${v.invalid_skin_vertices}</dd></dl><h3>Remaining production work</h3><ul>${report.limitations.map(s=>`<li>${esc(s)}</li>`).join('')}</ul>`;
    query('[data-stats]').textContent=`${parts.size} parts · ${bones.size} bones · ${morphs.size} controls`;
    player=motionPlayer(model,gltf.animations,bones,()=>{for(const n of bones.keys())applyBone(n);});
    query('[data-motion]').hidden=false;updateMotion=motionControls(query('[data-motion]'),player,abort.signal);
    settings.assetHash=report.glb_sha256;
    history=presetHistory(root,{read:()=>structuredClone(settings),apply:applyPreset,compatible:compatiblePreset,announce,signal:abort.signal});
    // Read saved edits before applying defaults; applying a preset may persist it.
    let restored=false;
    try{const saved=localStorage.getItem('landau-char-v1');if(saved){applyPreset(JSON.parse(saved),false);restored=true;}}catch{announce('An incompatible saved preset was skipped.');}
    if(!restored)applyPreset(settings,false);
    query('[data-loading]').hidden=true;selectPart(parts.has('Head')?'Head':'Body_Complete');readBone();shading();view(report.body_reconstruction?.revision>=3?'full':'face');announce('Ready. Working edits resume in this browser; Save preset keeps a version in sandbox history.');
  }catch(err){if(!dead){query('[data-loading]').textContent=err.message;announce('Could not load character: '+err.message);}}
  })();
  return {destroy:dispose,ready};
}
