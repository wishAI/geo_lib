import {test} from 'node:test';
import assert from 'node:assert/strict';
import {moduleURL} from './browser_modules.mjs';
const {sharedClips}=await import(await moduleURL('motion-clip.js'));

test('streamed recordings decode split UTF-8 and report decoded size instead of gzip transfer size',async t=>{
  const clip={name:'昨天的录制',frames:[{t:0}]},bytes=new TextEncoder().encode(JSON.stringify(clip)),progress=[];
  t.mock.method(globalThis,'fetch',async()=>new Response(new ReadableStream({start(controller){
    for(let i=0;i<bytes.length;i+=2)controller.enqueue(bytes.slice(i,i+2));controller.close();
  }}),{headers:{'Content-Encoding':'gzip','Content-Length':'10','X-Uncompressed-Content-Length':String(bytes.length)}}));
  assert.deepEqual(await sharedClips('server:abc',{onProgress:p=>progress.push(p)}),clip);
  assert.deepEqual(progress.at(-1),{loaded:bytes.length,total:bytes.length});
  assert.ok(progress.length>2);
});

test('recording transfers receive a longer deadline and a useful timeout error',async t=>{
  let expire,deadline;
  t.mock.method(globalThis,'setTimeout',(fn,ms)=>{expire=fn;deadline=ms;return 1;});
  t.mock.method(globalThis,'clearTimeout',()=>{});
  t.mock.method(globalThis,'fetch',(_,options)=>new Promise((resolve,reject)=>{
    options.signal.addEventListener('abort',()=>reject(new DOMException('Fetch is aborted','AbortError')));
  }));
  const request=sharedClips('server:abc');assert.equal(deadline,120000);expire();
  await assert.rejects(request,/录制下载超时（120 秒）/);
  const list=sharedClips();assert.equal(deadline,20000);expire();
  await assert.rejects(list,/录制列表请求超时/);
});

test('switching takes cancels a transfer without reporting a network timeout',async t=>{
  t.mock.method(globalThis,'fetch',(_,options)=>new Promise((resolve,reject)=>{
    options.signal.addEventListener('abort',()=>reject(new DOMException('Cancelled','AbortError')));
  }));
  const controller=new AbortController(),request=sharedClips('server:abc',{signal:controller.signal});
  controller.abort();await assert.rejects(request,{name:'AbortError'});
});

test('server error messages are preserved and list responses remain compatible',async t=>{
  t.mock.method(globalThis,'fetch',async()=>Response.json({error:'Recording not found'},{status:404}));
  await assert.rejects(sharedClips('server:missing'),/Recording not found/);
  globalThis.fetch=async()=>Response.json({recordings:[{id:'abc'}]});
  assert.deepEqual(await sharedClips(),[{id:'abc'}]);
});
