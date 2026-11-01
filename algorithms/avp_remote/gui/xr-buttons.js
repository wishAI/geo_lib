import * as THREE from 'three';

// visionOS exposes the eye-directed transient ray at pinch start, not continuously.
// Keep that target through select/end; a head-directed reticle provides idle hover.
export class XRButtonInput {
  constructor(scene,buttons,{activate,enabled=()=>true}){
    this.buttons=buttons;this.activate=activate;this.enabled=enabled;this.presses=new Map();this.ray=new THREE.Raycaster();this.hover=null;this.lastHeadHit=null;
    this.cursor=new THREE.Mesh(new THREE.RingGeometry(.003,.005,24),new THREE.MeshBasicMaterial({color:0xa7ead2,depthTest:false,transparent:true,opacity:.7}));
    this.cursor.renderOrder=30;this.cursor.visible=false;scene.add(this.cursor);
    for(const button of buttons){
      const border=new THREE.LineSegments(new THREE.EdgesGeometry(button.geometry),new THREE.LineBasicMaterial({color:0xa7ead2,depthTest:false,transparent:true,opacity:0}));
      border.position.z=.001;border.renderOrder=25;button.add(border);button.userData.focusBorder=border;button.material.transparent=true;
    }
  }
  hit(matrix){
    // Event poses can arrive between renders, including the first placed frame.
    for(const button of this.buttons)button.updateWorldMatrix(true,false);
    this.ray.ray.origin.setFromMatrixPosition(matrix);this.ray.ray.direction.set(0,0,-1).transformDirection(matrix);
    return this.ray.intersectObjects(this.buttons,false).find(hit=>this.enabled(hit.object))||null;
  }
  eventHit(event,ref){
    try{const pose=event.frame?.getPose(event.inputSource?.targetRaySpace,ref);
      if(pose)return {available:true,hit:this.hit(new THREE.Matrix4().fromArray(pose.transform.matrix))};
    }catch{}return {available:false,hit:null};
  }
  start(event,ref){
    const result=this.eventHit(event,ref),hit=result.available?result.hit:(performance.now()-(this.headAt||0)<200?this.lastHeadHit:null);
    this.presses.set(event.inputSource,{target:hit?.object||null,pressed:true,selected:false,at:performance.now()});
  }
  select(event,ref){
    if(!this.presses.has(event.inputSource))this.start(event,ref);
    const press=this.presses.get(event.inputSource);if(press.selected)return;
    press.selected=true;press.pressed=false;press.at=performance.now();
    if(press.target&&this.enabled(press.target)){press.target.userData.flashUntil=performance.now()+250;this.activate(press.target);}
  }
  end(event){const press=this.presses.get(event.inputSource);if(press){press.pressed=false;press.at=performance.now();}}
  update(frame,sources,ref){
    const now=performance.now();let hit=null,viewer;
    try{viewer=frame?.getViewerPose(ref);}catch{}
    if(viewer){
      const matrix=new THREE.Matrix4().fromArray(viewer.transform.matrix);hit=this.hit(matrix);this.lastHeadHit=hit;this.headAt=now;
      this.cursor.position.copy(hit?.point||new THREE.Vector3(0,0,-1.55).applyMatrix4(matrix));this.cursor.quaternion.setFromRotationMatrix(matrix);this.cursor.visible=true;
      for(const source of sources||[]){
        if(source.targetRayMode!=='transient-pointer')continue;
        const result=this.eventHit({frame,inputSource:source},ref);if(result.available){hit=result.hit;if(hit)this.cursor.position.copy(hit.point);break;}
      }
    }else{this.cursor.visible=false;this.lastHeadHit=null;}
    const activePress=[...this.presses.values()].find(p=>p.pressed&&p.target);
    this.hover=activePress?.target||hit?.object||null;
    if(activePress)activePress.target.getWorldPosition(this.cursor.position);
    for(const [source,press] of this.presses)if(!press.pressed&&now-press.at>500)this.presses.delete(source);
    for(const button of this.buttons){
      const pressed=[...this.presses.values()].some(p=>p.pressed&&p.target===button),focused=button===this.hover,flash=(button.userData.flashUntil||0)>now,enabled=this.enabled(button);
      button.scale.lerp(new THREE.Vector3().setScalar(pressed?.96:focused?1.07:1),.35);
      button.userData.focusBorder.material.opacity=pressed||flash?1:focused?.85:0;
      button.material.color.set(pressed?0x7bffc8:flash?0xbaffdc:0xffffff);button.material.opacity=enabled?1:.42;
      button.userData.xrHovered=focused;button.userData.xrPressed=pressed;
    }
    this.cursor.material.opacity=hit?1:.45;this.cursor.scale.setScalar(hit?1.35:1);
  }
  reset(){this.presses.clear();this.hover=null;this.lastHeadHit=null;this.cursor.visible=false;for(const b of this.buttons){b.scale.setScalar(1);b.userData.flashUntil=0;b.userData.xrHovered=b.userData.xrPressed=false;b.userData.focusBorder.material.opacity=0;}}
}
