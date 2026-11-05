/* Shared immutable byte buffers for viewers. HTTP validators handle page reloads. */
(function () {
  'use strict';
  const entries = new Map(), parsed = new Map();
  const MAX_BYTES = 128 * 1024 * 1024;
  let retainedBytes = 0;
  function keyFor(url) {
    const target = new URL(url, location.href);
    // Different URDF variants can refer to the same mesh through relative paths.
    if (target.origin === location.origin && target.pathname === '/api/robot/asset') {
      const urdf = target.searchParams.get('urdf'), uri = target.searchParams.get('uri');
      if (urdf && uri && !uri.includes('://') && !uri.startsWith('/')) {
        return 'robot-mesh:' + new URL(uri, new URL(urdf, location.origin + '/')).pathname;
      }
    }
    target.searchParams.sort();
    return target.href;
  }
  function trim() {
    for (const [key, entry] of entries) {
      if (retainedBytes <= MAX_BYTES && entries.size <= 512) break;
      if (!entry.buffer) continue;
      retainedBytes -= entry.buffer.byteLength;
      entries.delete(key);
    }
  }
  function read(url, {signal, onProgress} = {}) {
    if (signal?.aborted) return Promise.reject(new DOMException('Aborted', 'AbortError'));
    const key = keyFor(url);
    let entry = entries.get(key);
    if (!entry) {
      entry = {listeners: new Set(), progress: {state:'loading', loaded:0, total:null}};
      entries.set(key, entry);
      const emit = progress => { entry.progress=progress; for(const listener of entry.listeners) listener(progress); };
      entry.promise = (async () => {
        const response = await fetch(url, {cache:'default'});
        if (!response.ok) throw new Error(`Resource loading failed (${response.status}) · ${new URL(url,location.href).searchParams.get('path')?.split('/').pop() || 'model asset'}`);
        const decoded = Number(response.headers.get('X-Uncompressed-Content-Length'));
        const total = decoded || (!response.headers.get('Content-Encoding') && Number(response.headers.get('Content-Length'))) || null;
        const chunks = []; let loaded=0;
        const reader=response.body?.getReader();
        if(reader) {
          while(true) {
            const {value,done}=await reader.read(); if(done)break;
            loaded+=value.byteLength;
            if(loaded>MAX_BYTES){await reader.cancel();throw new Error('Model asset exceeds the 128 MiB limit');}
            chunks.push(value);emit({state:'loading',loaded,total});
          }
        } else { const buffer=await response.arrayBuffer();loaded=buffer.byteLength;if(loaded>MAX_BYTES)throw new Error('Model asset exceeds the 128 MiB limit');chunks.push(new Uint8Array(buffer)); }
        const buffer=new Uint8Array(loaded);let offset=0;
        for(const chunk of chunks){buffer.set(chunk,offset);offset+=chunk.byteLength;}
        entry.buffer=buffer.buffer;retainedBytes+=loaded;emit({state:'ready',loaded,total:loaded});trim();
        return entry.buffer;
      })().catch(error=>{entries.delete(key);throw error;});
    } else { entries.delete(key);entries.set(key,entry); }
    return new Promise((resolve,reject)=>{
      const progress=value=>{if(!signal?.aborted)onProgress?.(value);};
      const cleanup=()=>{entry.listeners.delete(progress);signal?.removeEventListener('abort',abort);};
      const abort=()=>{cleanup();reject(new DOMException('Aborted','AbortError'));};
      entry.listeners.add(progress);signal?.addEventListener('abort',abort,{once:true});
      progress(entry.buffer?{state:'cached',loaded:entry.buffer.byteLength,total:entry.buffer.byteLength}:entry.progress);
      entry.promise.then(value=>{cleanup();if(!signal?.aborted)resolve(value);},error=>{cleanup();reject(error);});
    });
  }
  function memo(key, factory) {
    if (!parsed.has(key)) {
      const promise=Promise.resolve().then(factory).catch(error=>{parsed.delete(key);throw error;});
      parsed.set(key,promise);
      if(parsed.size>96)parsed.delete(parsed.keys().next().value);
    }
    return parsed.get(key);
  }
  async function json(url, options) {
    const bytes=await read(url,options);
    return memo('json:'+keyFor(url),()=>JSON.parse(new TextDecoder().decode(bytes)));
  }
  function describe(progress) {
    if(progress.state==='cached')return 'Reusing shared asset';
    const kb=Math.ceil(progress.loaded/1024);
    return progress.total?`${Math.min(100,Math.round(progress.loaded/progress.total*100))}% · ${kb.toLocaleString()} KB`:`${kb.toLocaleString()} KB received`;
  }
  window.GeoAssets={read,json,memo,keyFor,describe,stats:()=>({entries:entries.size,retainedBytes})};
})();
