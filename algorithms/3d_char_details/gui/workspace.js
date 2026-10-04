// In-page window: the viewport stays put and only the inspector scrolls.
export function floatingWorkspace(root,signal){
 const editor=root.querySelector('.char-editor'),bar=editor.querySelector('.char-bar');
 const tools=document.createElement('div');tools.className='char-window-tools';
 tools.innerHTML='<button data-window="float">Float editor</button><button data-window="maximize" hidden>Maximize</button><button data-window="dock" hidden>Dock</button>';
 bar.append(tools);let placeholder=null,previousOverflow='',maximized=false,drag=null;
 function constrain(){if(!placeholder||maximized)return;const r=editor.getBoundingClientRect();editor.style.width=Math.min(r.width,innerWidth-16)+'px';editor.style.height=Math.min(r.height,innerHeight-16)+'px';editor.style.left=Math.max(8,Math.min(r.left,innerWidth-editor.offsetWidth-8))+'px';editor.style.top=Math.max(8,Math.min(r.top,innerHeight-editor.offsetHeight-8))+'px';}
 function dock(){if(!placeholder)return;editor.classList.remove('char-floating','char-maximized');editor.removeAttribute('style');placeholder.remove();placeholder=null;document.body.style.overflow=previousOverflow;maximized=false;tools.querySelector('[data-window=float]').hidden=false;for(const n of ['dock','maximize'])tools.querySelector(`[data-window=${n}]`).hidden=true;tools.querySelector('[data-window=float]').focus();}
 function float(){if(placeholder)return;placeholder=document.createElement('div');placeholder.style.height=editor.offsetHeight+'px';editor.before(placeholder);previousOverflow=document.body.style.overflow;document.body.style.overflow='hidden';editor.classList.add('char-floating');editor.style.cssText=`left:24px;top:24px;width:${innerWidth-48}px;height:${innerHeight-48}px`;tools.querySelector('[data-window=float]').hidden=true;for(const n of ['dock','maximize'])tools.querySelector(`[data-window=${n}]`).hidden=false;tools.querySelector('[data-window=maximize]').textContent='Maximize';constrain();tools.querySelector('[data-window=dock]').focus();}
 tools.addEventListener('click',e=>{const action=e.target.closest('[data-window]')?.dataset.window;if(action==='float')float();if(action==='dock')dock();if(action==='maximize'){maximized=!maximized;editor.classList.toggle('char-maximized',maximized);e.target.textContent=maximized?'Restore size':'Maximize';if(!maximized)constrain();}},{signal});
 bar.addEventListener('pointerdown',e=>{if(!placeholder||maximized||e.target.closest('button,input,select')||e.button!==0)return;const r=editor.getBoundingClientRect();drag=[e.clientX,e.clientY,r.left,r.top];bar.setPointerCapture(e.pointerId);e.preventDefault();},{signal});
 bar.addEventListener('pointermove',e=>{if(!drag)return;editor.style.left=drag[2]+e.clientX-drag[0]+'px';editor.style.top=drag[3]+e.clientY-drag[1]+'px';constrain();},{signal});
 for(const event of ['pointerup','pointercancel','lostpointercapture'])bar.addEventListener(event,()=>drag=null,{signal});
 window.addEventListener('resize',constrain,{signal});
 document.addEventListener('keydown',e=>{if(e.key==='Escape'&&placeholder){dock();e.preventDefault();}},{signal});
 signal.addEventListener('abort',dock,{once:true});return {float,dock};
}

export function propertyGroup(title,body,open=false){return `<details class="char-control-group" ${open?'open':''}><summary>${title}</summary><div class="char-group-body">${body}</div></details>`;}
export function groupedControls(names,control,kind){
 const buckets=new Map();for(const name of names){let title;
 if(kind==='face')title=/^eyeLook/.test(name)?'Gaze':/^eye/.test(name)?'Eyelids':/^brow/.test(name)?'Brows':'Mouth & expression';
 else title=/mouth|muzzle|jaw/i.test(name)?'Muzzle & mouth':/^bodyTransition/.test(name)?'Neck & upper chest':/head|face|eye|ear|cheek/i.test(name)?'Head & features':/arm|hand|shoulder/i.test(name)?'Arms & shoulders':/leg|thigh|calf|foot|feet|paw|heel/i.test(name)?'Legs & feet':'Torso & proportions';
 if(!buckets.has(title))buckets.set(title,[]);buckets.get(title).push(name);}
 const order=kind==='face'?['Eyelids','Gaze','Brows','Mouth & expression']:['Head & features','Muzzle & mouth','Neck & upper chest','Torso & proportions','Arms & shoulders','Legs & feet'];
 return order.filter(t=>buckets.has(t)).map((t,i)=>propertyGroup(t,buckets.get(t).map(control).join(''),i===0)).join('');
}
