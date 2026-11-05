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

test('identical content under different model paths shares download and parsed mesh key',async()=>{
  let calls=0;const assets=cache(async()=>{calls++;return new Response('same mesh');});
  const sha256='a'.repeat(64),a='/api/robot/asset?urdf=one.urdf&uri=package://one/leg.stl',b='/api/robot/asset?urdf=two.urdf&uri=file:///copy/leg.stl';
  const [first,second]=await Promise.all([assets.file(a,{sha256}),assets.file(b,{sha256})]);
  assert.equal(first,second);assert.equal(calls,1);assert.equal(assets.keyFor(a),assets.keyFor(b));
  const parsed=await assets.memo('stl:'+assets.keyFor(a),()=>({mesh:true}));
  assert.equal(await assets.memo('stl:'+assets.keyFor(b),()=>assert.fail('parsed twice')),parsed);
});

test('same filename with different contents never shares a cache entry',async()=>{
  let calls=0;const assets=cache(async()=>new Response(String(++calls)));
  const a=await assets.file('/one/mesh.stl',{sha256:'a'.repeat(64)});
  const b=await assets.file('/two/mesh.stl',{sha256:'b'.repeat(64)});
  assert.notEqual(a,b);assert.equal(calls,2);
});

test('arbitrary shared files resolve small metadata before downloading the content once',async()=>{
  let metadata=0,downloads=0;
  const assets=cache(async url=>{
    if(new URL(url).searchParams.has('metadata')){metadata++;return Response.json({sha256:'c'.repeat(64)});}
    downloads++;return new Response('shared character');
  });
  const a='/api/artifact?path=model-a/character.glb',b='/api/artifact?path=model-b/character.glb';
  assert.equal(await assets.file(a),await assets.file(b));
  await assets.file(a);assert.equal(metadata,2);assert.equal(downloads,1);
});

test('a changed file refreshes its identity once instead of caching the old contents',async()=>{
  let calls=0;
  const assets=cache(async url=>{
    calls++;const request=new URL(url);
    if(request.searchParams.has('metadata'))return Response.json({sha256:'b'.repeat(64)});
    return request.searchParams.get('sha256')==='a'.repeat(64)?new Response('',{status:409}):new Response('updated');
  });
  assert.equal(new TextDecoder().decode(await assets.file('/api/artifact?path=robot.glb',{sha256:'a'.repeat(64)})),'updated');
  assert.equal(calls,3);
});
