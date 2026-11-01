import * as THREE from 'three';
export const boundKeys=['x0','y0','z0','x1','y1','z1'];
const faces=[['x0','X min · west'],['x1','X max · east'],['y0','Y min · bottom'],['y1','Y max · top'],['z0','Z min · north'],['z1','Z max · south']];
export class BoundsEditor {
  constructor(root,{townBounds,onPreview,onSave,onRestore}) {
    this.root=root;this.townBounds=townBounds;this.onPreview=onPreview;this.onSave=onSave;this.onRestore=onRestore;
    root.innerHTML=`<h3>Bounding box · six sides</h3><p class="bounds-hint">Drag a colored face handle, use a slider, or enter a coordinate. Maximum values mark the outer edge.</p><div class="bounds-fields">${faces.map(([key,label])=>`<label class="bound-row" data-axis="${key[0]}"><span>${label}</span><input type="range" step="1" data-range="${key}" aria-label="${label} slider"><input type="number" step="1" data-number="${key}" aria-label="${label} coordinate"></label>`).join('')}</div><div class="bounds-buttons"><button type="button" data-save-bounds>Save bounds</button><button type="button" data-reset-bounds>Restore detected</button></div><p data-bounds-status role="status"></p>`;
    this.status=root.querySelector('[data-bounds-status]');
    for(const [key] of faces) {
      const lo=key[0]==='y'?0:townBounds[key[0]+'0'],hi=key[0]==='y'?256:townBounds[key[0]+'1'];
      for(const selector of ['data-range','data-number']) {
        const el=root.querySelector(`[${selector}="${key}"]`);el.min=lo;el.max=hi;
        el.addEventListener('input',()=>this.change(key,el.value,false));
        el.addEventListener('change',()=>this.change(key,el.value,true));
      }
    }
    root.querySelector('[data-save-bounds]').onclick=async()=>{const q=Object.fromEntries(boundKeys.map(k=>{const v=root.querySelector(`[data-number="${k}"]`).value;return [k,v===''?NaN:Number(v)];}));if(!this.valid(q))return;try{await onSave(q);this.status.textContent='Bounds saved on TK2.';}catch(e){this.status.textContent=e.message;}};
    root.querySelector('[data-reset-bounds]').onclick=async()=>{try{await onRestore();}catch(e){this.status.textContent=e.message;}};
  }
  valid(q) {
    if(!q||boundKeys.some(k=>!Number.isInteger(q[k]))) {this.status.textContent='Enter a whole number for all six coordinates.';return false;}
    for(const axis of 'xyz') {
      const lo=axis==='y'?0:this.townBounds[axis+'0'],hi=axis==='y'?256:this.townBounds[axis+'1'];
      if(q[axis+'0']<lo||q[axis+'1']>hi||q[axis+'0']>=q[axis+'1']){this.status.textContent=`${axis.toUpperCase()}: minimum must be below maximum, within ${lo}…${hi}.`;return false;}
    }
    if((q.x1-q.x0)*(q.z1-q.z0)>131072){this.status.textContent='Reduce X/Z size: maximum 131,072 columns per building.';return false;}
    return true;
  }
  select(building,saved) {this.building=building;this.draft={...(saved||building.bounds)};this.sync();this.status.textContent=saved?'Saved box · adjustable crop':'Detected footprint · adjust a side to preview a rectangular crop';}
  sync() {for(const k of boundKeys)for(const attr of ['data-range','data-number'])this.root.querySelector(`[${attr}="${k}"]`).value=this.draft[k];this.updateHandles();}
  change(key,value,commit=true) {
    const parsed=value===''?NaN:Number(value),q={...this.draft,[key]:parsed};
    if(!this.valid(q))return false;
    this.draft=q;this.sync();this.status.textContent='Unsaved box · preview only';this.onPreview({...q},commit);return true;
  }
  attach(scene,camera,renderer,controls) {
    this.scene=scene;this.camera=camera;this.renderer=renderer;this.controls=controls;
    this.group=new THREE.Group();this.group.name='bounds-face-handles';scene.add(this.group);
    const geometry=new THREE.SphereGeometry(1,12,8);
    for(const [key] of faces) {const mesh=new THREE.Mesh(geometry,new THREE.MeshBasicMaterial({color:{x:0xe9867f,y:0x99d48c,z:0x82bfff}[key[0]],depthTest:false}));mesh.userData.key=key;mesh.renderOrder=100;this.group.add(mesh);}
    this.ray=new THREE.Raycaster();this.plane=new THREE.Plane();this.point=new THREE.Vector3();this.updateHandles();
    const el=renderer.domElement;
    const ray=e=>{const r=el.getBoundingClientRect();this.ray.setFromCamera(new THREE.Vector2((e.clientX-r.left)/r.width*2-1,-(e.clientY-r.top)/r.height*2+1),camera);};
    el.addEventListener('pointerdown',e=>{
      if(!this.group.visible||!this.draft)return;ray(e);const hit=this.ray.intersectObjects(this.group.children)[0];if(!hit)return;
      const key=hit.object.userData.key,axis=new THREE.Vector3(key[0]==='x'?1:0,key[0]==='y'?1:0,key[0]==='z'?1:0);
      const normal=camera.getWorldDirection(new THREE.Vector3());normal.addScaledVector(axis,-normal.dot(axis));if(normal.lengthSq()<.001)return;normal.normalize();
      this.plane.setFromNormalAndCoplanarPoint(normal,hit.object.position);if(!this.ray.ray.intersectPlane(this.plane,this.point))return;
      this.drag={key,axis,start:this.point.clone(),value:this.draft[key],id:e.pointerId};controls.enabled=false;el.setPointerCapture(e.pointerId);e.preventDefault();e.stopImmediatePropagation();
    },true);
    el.addEventListener('pointermove',e=>{
      if(!this.drag)return;ray(e);if(!this.ray.ray.intersectPlane(this.plane,this.point))return;
      const d=this.drag,value=Math.round(d.value+this.point.clone().sub(d.start).dot(d.axis));this.change(d.key,value,false);e.preventDefault();e.stopImmediatePropagation();
    },true);
    const finish=e=>{if(!this.drag)return;this.drag=null;controls.enabled=true;if(el.hasPointerCapture(e.pointerId))el.releasePointerCapture(e.pointerId);this.onPreview({...this.draft},true);e.preventDefault();e.stopImmediatePropagation();};
    el.addEventListener('pointerup',finish,true);el.addEventListener('pointercancel',finish,true);
  }
  updateHandles() {
    if(!this.group||!this.draft)return;const q=this.draft,b=this.townBounds,center=new THREE.Vector3((q.x0+q.x1)/2-b.x0,(q.y0+q.y1)/2,(q.z0+q.z1)/2);
    const size=Math.max(.5,Math.min(2.5,Math.max(q.x1-q.x0,q.y1-q.y0,q.z1-q.z0)*.018));
    for(const mesh of this.group.children){const key=mesh.userData.key;mesh.position.copy(center);mesh.position[key[0]]=q[key]-(key[0]==='x'?b.x0:key[0]==='z'?b.z0:0);mesh.scale.setScalar(size);}
  }
  scaleHandles() {
    if(!this.group?.visible)return;
    const pixels=2*Math.tan(this.camera.fov*Math.PI/360)/Math.max(1,this.renderer.domElement.clientHeight);
    for(const mesh of this.group.children)mesh.scale.setScalar(Math.max(.4,this.camera.position.distanceTo(mesh.position)*pixels*5));
  }
  visible(value) {if(this.group)this.group.visible=value;}
}
