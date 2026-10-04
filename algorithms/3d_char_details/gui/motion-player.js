import * as THREE from 'three';

// A single deterministic clock drives skeleton.
// Playback never writes the user's facial, fitting, pose or preset settings.
export function motionPlayer(model, clips, bones, applyPose) {
  const clip=clips.find(c=>c.name==='Running')||clips.find(c=>/Running/.test(c.name));
  if(!clip)throw new Error('The running clip is missing from this asset.');
  const mixer=new THREE.AnimationMixer(model),action=mixer.clipAction(clip);
  const rest=new Map(),original=new Map(),pose=new Map();
  const positionBones=new Set(clip.tracks.filter(t=>t.name.endsWith('.position')).map(t=>THREE.PropertyBinding.parseTrackName(t.name).nodeName));
  for(const [n,{bone,quaternion,scale}] of bones){
    rest.set(n,{position:bone.position.clone(),quaternion:quaternion.clone(),scale:scale.clone()});
    original.set(n,bone.position.clone());
  }
  let time=0,playing=false,active=false,loop=true,speed=1,bodyScale=1;
  action.setLoop(THREE.LoopOnce,1);action.clampWhenFinished=true;
  function evaluate(){
    pose.clear();
    if(active){
      // PropertyMixer skips identical samples; release its cached state before
      // restoring the edited rest pose, including on repeated/backward scrubs.
      action.stop();
      for(const [n,{bone}]of bones){const r=rest.get(n);bone.position.copy(r.position);bone.quaternion.copy(r.quaternion);bone.scale.copy(r.scale);}
      action.reset().play();mixer.setTime(time);
      for(const [n,{bone}]of bones){
        pose.set(n,bone.quaternion.clone());
        if(positionBones.has(n))bone.position.sub(original.get(n)).multiplyScalar(bodyScale).add(rest.get(n).position);
      }
    }else{
      for(const [n,{bone}]of bones){const r=rest.get(n);bone.position.copy(r.position);bone.quaternion.copy(r.quaternion);bone.scale.copy(r.scale);}
    }
    applyPose();model.updateMatrixWorld(true);
  }
  function seek(t){time=THREE.MathUtils.clamp(Number.isFinite(t)?t:0,0,clip.duration);active=true;evaluate();}
  function reset(){playing=false;active=false;time=0;action.stop();evaluate();}
  return {
    duration:clip.duration,clip,
    get state(){return {time,playing,active,loop,speed};},
    quaternion:name=>pose.get(name),
    play(){if(time>=clip.duration)time=0;active=true;playing=true;evaluate();},
    pause(){playing=false;},reset,seek,
    setLoop(v){loop=Boolean(v);},
    setSpeed(v){if(Number.isFinite(v)&&v>=.1&&v<=3)speed=v;},
    tick(dt){if(!playing)return;time+=Math.min(Math.max(0,dt),.25)*speed;if(time>=clip.duration){if(loop)time%=clip.duration;else{time=clip.duration;playing=false;}}evaluate();},
    beforeRestEdit(){pose.clear();for(const [n,{bone}]of bones){const r=rest.get(n);bone.position.copy(r.position);bone.quaternion.copy(r.quaternion);bone.scale.copy(r.scale);}},
    afterRestEdit(scale){bodyScale=scale;for(const[n,{bone,quaternion,scale:s}]of bones)rest.set(n,{position:bone.position.clone(),quaternion:quaternion.clone(),scale:s.clone()});evaluate();},
    dispose(){action.stop();mixer.uncacheRoot(model);},
  };
}

export function motionControls(container,player,signal){
  container.innerHTML=`<summary>Running preview</summary><div class="char-motion-row"><button data-motion-play aria-label="Play running animation">Play run</button><button data-motion-reset title="Return to your saved manual pose and garment fit">Reset motion</button><label><input type="checkbox" data-motion-loop checked> Loop</label><label>Speed <select aria-label="Running speed" data-motion-speed><option value="0.25">0.25×</option><option value="0.5">0.5×</option><option value="1" selected>1×</option><option value="1.5">1.5×</option><option value="2">2×</option></select></label></div><div class="char-motion-row"><input aria-label="Running timeline" data-motion-time type="range" min="0" max="${player.duration}" value="0" step="0.001"><output data-motion-readout></output></div>`;
  const q=s=>container.querySelector(s);
  q('[data-motion-play]').addEventListener('click',()=>{player.state.playing?player.pause():player.play();update();},{signal});
  q('[data-motion-reset]').addEventListener('click',()=>{player.reset();update();},{signal});
  q('[data-motion-time]').addEventListener('input',e=>{player.pause();player.seek(Number(e.target.value));update();},{signal});
  q('[data-motion-loop]').addEventListener('change',e=>player.setLoop(e.target.checked),{signal});
  q('[data-motion-speed]').addEventListener('change',e=>player.setSpeed(Number(e.target.value)),{signal});
  function update(){const s=player.state;q('[data-motion-play]').textContent=s.playing?'Pause':'Play run';q('[data-motion-play]').setAttribute('aria-label',s.playing?'Pause running animation':'Play running animation');q('[data-motion-time]').value=s.time;q('[data-motion-readout]').textContent=`${s.time.toFixed(3)} / ${player.duration.toFixed(3)} s · ${Math.round(s.time/player.duration*32)+1}/33`;}
  update();return update;
}
