import test from 'node:test';
import assert from 'node:assert/strict';
import vm from 'node:vm';
import {readFileSync} from 'node:fs';
const source=readFileSync(new URL('../static/shared_assets.js',import.meta.url),'utf8');
function cache(fetch) {
  const context={window:{},location:new URL('https://gui.example/'),URL,fetch,DOMException,TextDecoder,Uint8Array,Promise};
  vm.runInNewContext(source,context);
  return context.window.GeoAssets;
}

test('concurrent viewers share bytes, cached reads report progress, and abort is per viewer',async()=>{
  let finish,calls=0;
  const assets=cache(()=>{calls++;return new Promise(resolve=>finish=resolve);});
  const controller=new AbortController(),progress=[];
  const first=assets.read('/mesh',{signal:controller.signal});
  const second=assets.read('/mesh',{onProgress:p=>progress.push(p)});
  const aborted=assert.rejects(first,{name:'AbortError'});
  controller.abort();await aborted;
  finish(new Response(new Uint8Array([1,2,3]),{headers:{'Content-Length':'3'}}));
  const bytes=await second;
  assert.deepEqual([...new Uint8Array(bytes)],[1,2,3]);
  assert.equal(progress.at(-1).loaded,3);
  assert.equal(progress.at(-1).state,'ready');
  const cachedProgress=[];
  assert.equal(await assets.read('/mesh',{onProgress:p=>cachedProgress.push(p)}),bytes);
  assert.equal(cachedProgress[0].state,'cached');assert.equal(calls,1);
});

test('different URDF variants reuse the same resolved relative mesh',async()=>{
  let calls=0;const assets=cache(async()=>{calls++;return new Response('mesh');});
  const a='/api/robot/asset?urdf=inputs/v1/model.urdf&uri=../shared/leg.stl';
  const b='/api/robot/asset?uri=../shared/leg.stl&urdf=inputs/v2/model.urdf';
  assert.equal(await assets.read(a),await assets.read(b));assert.equal(calls,1);
  assert.notEqual(assets.keyFor(a),assets.keyFor('/api/robot/asset?urdf=other/model.urdf&uri=leg.stl'));
});

test('a failed shared request can be retried',async()=>{
  let calls=0;const assets=cache(async()=>++calls===1?new Response('',{status:403}):new Response('ok'));
  await assert.rejects(assets.read('/asset'),/403/);
  assert.equal(new TextDecoder().decode(await assets.read('/asset')),'ok');assert.equal(calls,2);
});

test('compressed transfers do not show a misleading percentage from compressed length',async()=>{
  const progress=[];
  const assets=cache(async()=>new Response('decoded body',{headers:{'Content-Encoding':'gzip','Content-Length':'2'}}));
  await assets.read('/compressed',{onProgress:p=>progress.push(p)});
  assert.equal(progress.find(p=>p.loaded>0&&p.state==='loading').total,null);
});

test('JSON and parsed geometry are shared, failed parsing is retryable',async()=>{
  const assets=cache(async()=>new Response('{"joints":[]}'));
  const [a,b]=await Promise.all([assets.json('/robot'),assets.json('/robot')]);assert.equal(a,b);
  await assert.rejects(assets.memo('geometry',()=>{throw new Error('parse failed');}),/parse failed/);
  assert.equal(await assets.memo('geometry',()=>42),42);
});
