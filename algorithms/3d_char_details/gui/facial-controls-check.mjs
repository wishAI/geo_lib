import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';

const source=await readFile(new URL('./facial-controls.js',import.meta.url),'utf8');
const {applyBlink,migrateFacePreset}=await import('data:text/javascript;base64,'+Buffer.from(source).toString('base64'));
const morphs=new Map();
for(const side of ['L','R'])for(const prefix of ['eyeBlink','eyeSquint','_blinkArc']) {
  const meshes=[{morphTargetInfluences:[99]},{morphTargetInfluences:[99]}];
  morphs.set(prefix+side,meshes.map(mesh=>[mesh,0]));
}
function expect(side,blink,arc) {
  for(const [prefix,value] of [['eyeBlink',blink],['eyeSquint',0],['_blinkArc',arc]])
    for(const [mesh,i] of morphs.get(prefix+side))assert.ok(Math.abs(mesh.morphTargetInfluences[i]-value)<1e-12);
}
applyBlink(morphs,{eyeBlinkL:.9,eyeSquintL:1,eyeBlinkR:-1});expect('L',1,0);expect('R',0,0);
applyBlink(morphs,{eyeSquintL:1});expect('L',.45,.99);expect('R',0,0);
applyBlink(morphs,{eyeBlinkL:1,eyeBlinkR:0},.5);expect('L',.5,1);expect('R',.5,1);
applyBlink(morphs,{eyeBlinkL:1,eyeBlinkR:1},0);expect('L',0,0);expect('R',0,0);
applyBlink(morphs,{eyeSquintL:1},.8);expect('L',1,0);expect('R',.8,.64);
applyBlink(morphs,{});expect('L',0,0);expect('R',0,0);
applyBlink(new Map(),{},.5);
const saved={assetHash:'legacy',parts:{Iris_L:{visible:false},EyeShell_L:{visible:true},Vest:{visible:true}},outfit:{Vest:{vestChestWidth:-.3}}};
const copy=JSON.stringify(saved),migrated=migrateFacePreset(saved,{facial_repair:{removed_objects:['Iris_L','Iris_R']}});
assert.equal(JSON.stringify(saved),copy);assert.equal(migrated.parts.Iris_L,undefined);
assert.deepEqual(migrated.parts.Vest,saved.parts.Vest);assert.deepEqual(migrated.parts.EyeShell_L,saved.parts.EyeShell_L);assert.deepEqual(migrated.outfit,saved.outfit);
assert.equal(migrateFacePreset(saved,{}),saved);
console.log('Facial controls passed: paired application, saturation, squint, animation override, zero override, reset and missing targets.');
