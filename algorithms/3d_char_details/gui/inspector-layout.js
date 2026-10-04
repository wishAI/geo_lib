import {icon,propertyVisibility} from '/api/artifact?path=algorithms/3d_char_details/gui/property-widgets.js';
import {propertyGroup} from '/api/artifact?path=algorithms/3d_char_details/gui/workspace.js';
const escape=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
export const FACE_SHAPES=new Set(['headWidth','faceWidth','earLength','eyeSize','cheekFullness','muzzleLength','jawRecess','mouthLength','mouthCurvature']);
const partLabel=n=>({Body_Complete:'Connected skin · face & body',Mouth_Interior:'Mouth interior',RoundIris_L:'Iris · left',RoundIris_R:'Iris · right',LashBed_L:'Lash support · left',LashBed_R:'Lash support · right'}[n]||n.replace(/([a-z])([A-Z])/g,'$1 $2').replace(/_L$/,' · left').replace(/_R$/,' · right').replaceAll('_',' '));
function jointGroup(n){if(/^(head|ear)_/.test(n))return 'Head & ears';if(/twist/.test(n))return 'Twist joints';if(/thumb|index|middle|ring|pinky/.test(n))return 'Fingers';if(/hand/.test(n))return 'Wrists';if(/shoulder|arm|forearm/.test(n))return 'Shoulders & arms';if(/thigh|leg/.test(n))return 'Hips & knees';if(/foot|toes/.test(n))return 'Ankles & toes';if(/tail/.test(n))return 'Tail';return 'Neck & torso';}
function jointLabel(n){return n.replace(/_([lr])$/,(m,s)=>s==='l'?' · left':' · right').replace(/_x$/,'').replace('forearm_stretch','Forearm').replace('arm_stretch','Upper arm').replace('thigh_stretch','Hip').replace('leg_stretch','Knee').replace('spine_03','Upper spine').replace('spine_02','Middle spine').replace('spine_01','Lower spine').replace('toes_01','Toes').replaceAll('_',' ').replace(/^./,s=>s.toUpperCase());}

export function inspectorLayout(root,signal){
 const q=s=>root.querySelector(s),face=q('[data-pane=face]'),body=q('[data-pane=shape]');
 q('.char-tabs').innerHTML='<button class="active" data-tab="face">Face</button><button data-tab="body">Body</button><button data-tab="clothing">Clothing</button><button data-tab="resources">Reference & model</button><button data-tab="presets">Presets</button>';
 const faceMotion=document.createElement('section');while(face.firstChild)faceMotion.append(face.firstChild);faceMotion.dataset.sectionDomain='face';faceMotion.dataset.modePanel='joints';faceMotion.hidden=true;
 faceMotion.querySelector('.char-hint').textContent='Head and ear joints, plus coordinated eyelid, gaze and mouth motion. Expression controls keep their authored component attachments.';
 body.dataset.pane='body';body.querySelector(':scope > .char-hint')?.remove();
 const bodyShape=document.createElement('section');while(body.firstChild)bodyShape.append(body.firstChild);bodyShape.dataset.sectionDomain='body';bodyShape.dataset.modePanel='shape';
 const faceShape=document.createElement('section');faceShape.dataset.sectionDomain='face';faceShape.dataset.modePanel='shape';faceShape.innerHTML='<div data-face-shapes></div>';
 const bodyJoints=document.createElement('section');bodyJoints.dataset.sectionDomain='body';bodyJoints.dataset.modePanel='joints';bodyJoints.hidden=true;
 for(const [domain,host,shape,joints]of [['face',face,faceShape,faceMotion],['body',body,bodyShape,bodyJoints]]){
  const nav=document.createElement('nav');nav.className='char-mode-tabs';nav.setAttribute('aria-label',domain+' adjustment type');nav.innerHTML=`<button data-domain="${domain}" data-mode="shape" aria-pressed="true">Shape adjustments</button><button data-domain="${domain}" data-mode="joints" aria-pressed="false">Joints & motion</button>`;host.append(nav,shape,joints);
 }
 const rig=q('[data-pane=rig]'),jointEditor=document.createElement('section');jointEditor.className='char-joint-editor';jointEditor.innerHTML='<h3>Joint adjustment</h3><p class="char-hint">Choose an anatomical joint. Rotation uses its local axes; Shape adjustments change the surface.</p>';
 for(const child of [...rig.children])if(!child.matches('p'))jointEditor.append(child);rig.remove();bodyJoints.append(jointEditor);
 jointEditor.querySelector('label').firstChild.textContent='Joint ';jointEditor.querySelector('select').setAttribute('aria-label','Anatomical joint');
 const partPane=q('[data-pane=parts]'),materials=document.createElement('details');materials.className='char-material-editor';materials.innerHTML='<summary>Selected surface · materials</summary>';
 const section=partPane.querySelector('.char-section');materials.append(section);bodyShape.append(materials);partPane.remove();
 const reference=q('[data-pane=reference]'),asset=q('[data-pane=asset]'),info=document.createElement('details');reference.dataset.pane='resources';info.className='char-control-group';info.innerHTML='<summary>Model details & export</summary><div class="char-group-body"></div>';while(asset.firstChild)info.lastElementChild.append(asset.firstChild);reference.append(info);asset.remove();
 let bones=null,onJoint=null;
 function joints(domain){if(!bones)return;const select=q('[data-bone]'),selected=select.value,groups=new Map();for(const n of bones.keys()){if((jointGroup(n)==='Head & ears')!==(domain==='face'))continue;const g=jointGroup(n);if(!groups.has(g))groups.set(g,[]);groups.get(g).push(n);}
  select.innerHTML=[...groups].map(([g,names])=>`<optgroup label="${g}">${names.map(n=>`<option value="${escape(n)}">${escape(jointLabel(n))}</option>`).join('')}</optgroup>`).join('');if([...select.options].some(o=>o.value===selected))select.value=selected;
  (domain==='face'?faceMotion:bodyJoints).prepend(jointEditor);onJoint(select.value);
 }
 root.addEventListener('click',e=>{const b=e.target.closest('[data-mode]');if(!b)return;const domain=b.dataset.domain,mode=b.dataset.mode;root.querySelectorAll(`[data-section-domain="${domain}"]`).forEach(p=>p.hidden=p.dataset.modePanel!==mode);root.querySelectorAll(`[data-domain="${domain}"]`).forEach(p=>p.setAttribute('aria-pressed',String(p.dataset.mode===mode)));if(mode==='joints')joints(domain);},{signal});
 return {
  setup({parts,shapes,control,jointMap,selectJoint}){
   bones=jointMap;onJoint=selectJoint;
   function visibility(names){return '<div class="char-context-parts">'+names.filter(n=>parts.has(n)).map(n=>`<div class="char-context-part"><button data-part="${escape(n)}" title="Edit ${escape(partLabel(n))} materials">${escape(partLabel(n))}</button>${propertyVisibility(`data-visible="${escape(n)}"`,partLabel(n),parts.get(n).defaultVisible)}</div>`).join('')+'</div>';}
   function group(title,names,visible,open=false){return propertyGroup(title,visibility(visible)+names.filter(n=>shapes.has(n)).map(n=>control(n)).join(''),open);}
   q('[data-face-shapes]').innerHTML=group('Muzzle & mouth',['muzzleLength','jawRecess','mouthLength','mouthCurvature'],['Nose','Mouth_Interior','Teeth_Upper','Teeth_Lower','Tongue'],true)+group('Head & cheeks',['headWidth','faceWidth','cheekFullness'],['Body_Complete'])+group('Eyes & brows',['eyeSize'],[...parts.keys()].filter(n=>/Eye|Iris|Pupil|Catchlight|Lash|Lid|Brow/.test(n)))+group('Ears',['earLength'],['InnerEar_L','InnerEar_R']);
   const bodyNames=[...shapes].filter(n=>!FACE_SHAPES.has(n)),groups=[['Neck & upper chest',n=>n.startsWith('bodyTransition')],['Arms & hands',n=>/arm|hand|shoulder/i.test(n)],['Legs & feet',n=>/leg|thigh|calf|foot|feet|paw|heel/i.test(n)],['Torso & proportions',()=>true]];let remaining=bodyNames;
   q('[data-shape-sliders]').innerHTML=visibility(['Body_Complete'])+groups.map(([title,test])=>{const names=remaining.filter(test);remaining=remaining.filter(n=>!test(n));return names.length?group(title,names,[],title==='Neck & upper chest'):'';}).join('');
   // Visibility also lives beside coordinated facial motion, not in a Parts tab.
   for(const group of faceMotion.querySelectorAll('.char-control-group')){const title=group.querySelector('summary').textContent,names=[...parts.keys()].filter(n=>title==='Brows'?/Brow/.test(n):title==='Gaze'?/Eye|Iris|Pupil|Catchlight/.test(n):title==='Eyelids'?/Lash|Lid/.test(n):/Mouth|Teeth|Tongue/.test(n));group.querySelector('.char-group-body').prepend(document.createRange().createContextualFragment(visibility(names)));}
   const pivots=q('[data-frame-sliders] [data-body-frame=shoulderHeight]')?.closest('.char-property');if(pivots){const h=pivots.previousElementSibling,note=pivots.nextElementSibling;bodyJoints.prepend(h,pivots,note);}
   joints('body');
  },
  material(name){
   const active=q('[data-pane]:not([hidden])'),rows=[...root.querySelectorAll('[data-part]')].filter(e=>e.dataset.part===name),row=rows.find(e=>active?.contains(e)&&!e.closest('[data-mode-panel][hidden]'));
   // Keep the shared editor outside garment cards, which rebuild after linking.
   const host=row&&!row.closest('[data-garment-card]')?row.closest('.char-group-body,.char-context-parts'):active;
   if(host){host.append(materials);for(let p=host;p&&p!==root;p=p.parentElement)if(p.tagName==='DETAILS')p.open=true;materials.open=true;materials.scrollIntoView({block:'nearest'});}
  },
  domain(name){if((name==='face'||name==='body')&&!q(`[data-section-domain="${name}"][data-mode-panel=joints]`).hidden)joints(name);}
 };
}
