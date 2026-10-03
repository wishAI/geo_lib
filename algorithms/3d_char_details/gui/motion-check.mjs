import fs from 'node:fs';
import assert from 'node:assert/strict';
import * as T from './three.mjs';
import {GLTFLoader} from './loader.mjs';
import {bodyPlacement} from './placement.mjs';
import {motionPlayer} from './motion.mjs';
globalThis.ProgressEvent=class{constructor(type,args){this.type=type;Object.assign(this,args);}};
const data=fs.readFileSync(process.argv[2]),jl=data.readUInt32LE(12),json=JSON.parse(data.subarray(20,20+jl));
json.buffers[0].uri='data:application/octet-stream;base64,'+data.subarray(28+jl).toString('base64');
json.materials=json.materials.map(m=>({name:m.name}));delete json.images;delete json.textures;delete json.samplers;
const {scene,animations}=await new GLTFLoader().parseAsync(JSON.stringify(json),'');
const bones=new Map();scene.traverse(o=>{if(o.isBone)bones.set(o.name,{bone:o,quaternion:o.quaternion.clone(),scale:o.scale.clone()});});
const placement=bodyPlacement(scene,bones);
const manual=new T.Quaternion().setFromAxisAngle(new T.Vector3(1,0,0),.25);
let player;
const pose=()=>{for(const[n,{bone,quaternion,scale}]of bones){bone.quaternion.copy(player?.quaternion(n)||quaternion);bone.scale.copy(scale);if(n==='ear_l')bone.quaternion.multiply(manual);}};
player=motionPlayer(scene,animations,bones,pose);
assert(Math.abs(player.duration-32/60)<1e-5,'Wrong source duration/fps');
const positions=()=>{scene.updateMatrixWorld(true);return [...bones.values()].flatMap(({bone})=>bone.matrixWorld.toArray());};
const error=(a,b)=>Math.max(...a.map((v,i)=>Math.abs(v-b[i])));
player.reset();const rest=positions();
let maxRepeatedError=0,maxLoopError=0,finite=true;
for(let i=0;i<=64;i++){
  player.seek(player.duration*i/64);const a=positions();
  player.seek(player.duration*i/64);const b=positions();maxRepeatedError=Math.max(maxRepeatedError,error(a,b));
  finite&&=a.every(Number.isFinite);
}
assert(finite);assert(maxRepeatedError<1e-6,{maxRepeatedError});
player.seek(0);const first=positions();player.seek(player.duration);maxLoopError=error(first,positions());
assert(maxLoopError<1e-4,{maxLoopError});
player.seek(player.duration*.3);const sample=positions();player.pause();player.tick(.1);assert(error(sample,positions())<1e-8);
player.seek(player.duration*.8);player.seek(player.duration*.3);assert(error(sample,positions())<1e-7);
player.reset();assert(error(rest,positions())<1e-7,'Reset changed manual pose');
player.setLoop(false);player.setSpeed(2);player.play();for(let i=0;i<4;i++)player.tick(.1);assert(!player.state.playing&&player.state.time===player.duration);
player.setLoop(true);player.play();player.tick(.1);assert(Math.abs(player.state.time-.2)<1e-7);
player.pause();player.seek(player.duration*.4);
for(const [scale,offset]of [[.78,0],[1.25,.05],[1,0]]){
  player.beforeRestEdit();placement(scale,offset);player.afterRestEdit(scale);
  const a=positions();player.seek(player.state.time);assert(error(a,positions())<1e-7,'Repeated scrub changed body placement');
  assert(a.every(Number.isFinite));
}
player.reset();assert(error(rest,positions())<1e-6,'Body placement reset changed manual pose');
const result={status:'passed',clip:player.clip.name,duration:player.duration,bones:bones.size,
  samples:65,maxRepeatedError,maxLoopError,checks:['scrub','repeated sample','reverse scrub','pause','loop boundary','loop','speed','end clamp','manual pose preservation','body placement','reset']};
fs.writeFileSync(process.argv[3],JSON.stringify(result,null,2));console.log(JSON.stringify(result,null,2));
