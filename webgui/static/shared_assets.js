/* Shared content-addressed assets across models, viewers and page reloads. */
(function () {
  'use strict';
  const entries = new Map(), parsed = new Map();
  const MAX_BYTES = 128 * 1024 * 1024;
  let retainedBytes = 0;
  const identities=new Map(),aliases=new Map();
  let database;
  function disk() {
    if(!window.indexedDB)return Promise.resolve(null);
    if(!database)database=new Promise(resolve=>{
      let request;
      try {request=window.indexedDB.open('geo-shared-assets',1);} catch {resolve(null);return;}
      request.onupgradeneeded=()=>{const db=request.result;db.createObjectStore('assets',{keyPath:'key'}).createIndex('storedAt','storedAt');db.createObjectStore('meta');};
      request.onsuccess=()=>{const db=request.result;db.onversionchange=()=>{db.close();database=null;};resolve(db);};
      request.onerror=request.onblocked=()=>resolve(null);
    });
    return database;
  }
  async function diskRead(key) {
    const db=await disk();if(!db)return null;
    return new Promise(resolve=>{try {
      const request=db.transaction('assets').objectStore('assets').get(key);
      request.onsuccess=()=>resolve(request.result?.buffer||null);request.onerror=()=>resolve(null);
    }catch{resolve(null);}});
  }
  async function diskWrite(key,buffer) {
    const db=await disk();if(!db)return;
    await new Promise(resolve=>{try {
      const tx=db.transaction(['assets','meta'],'readwrite'),assets=tx.objectStore('assets'),meta=tx.objectStore('meta');
      tx.oncomplete=tx.onerror=tx.onabort=()=>resolve();
      const existing=assets.get(key);
      existing.onsuccess=()=>{if(existing.result)return;
        const sizeRequest=meta.get('bytes');sizeRequest.onsuccess=()=>{
          let size=(sizeRequest.result||0)+buffer.byteLength;
          assets.put({key,buffer,storedAt:Date.now(),size:buffer.byteLength});
          if(size<=MAX_BYTES*2){meta.put(size,'bytes');return;}
          const oldest=assets.index('storedAt').openCursor();oldest.onsuccess=()=>{
            const cursor=oldest.result;
            if(cursor&&size>MAX_BYTES*2){size-=cursor.value.size;cursor.delete();cursor.continue();}
            else meta.put(size,'bytes');
          };
        };
      };
    }catch{resolve();}});
  }
  function absolute(url) {const target=new URL(url,location.href);target.searchParams.sort();return target.href;}
  async function file(url, {sha256, ...options} = {}, retried = false) {
    const target=new URL(url,location.href),original=absolute(url);
    if(!sha256){
      if(!identities.has(original))identities.set(original,(async()=>{
        const metadata=new URL(target);metadata.searchParams.set('metadata','1');
        const response=await fetch(metadata.href,{cache:'no-cache'});
        if(!response.ok)throw new Error(`Asset identity unavailable (${response.status})`);
        return (await response.json()).sha256;
      })().catch(error=>{identities.delete(original);throw error;}));
      sha256=await identities.get(original);
    }
    if(!/^[a-f0-9]{64}$/.test(sha256))throw new Error('Invalid asset content identity');
    aliases.set(original,'sha256:'+sha256);
    target.searchParams.set('sha256',sha256);
    try {return await read(target.href,{...options,identity:sha256});}
    catch(error){if(error.status!==409||retried)throw error;identities.delete(original);aliases.delete(original);return file(url,options,true);}
  }
  function keyFor(url) {
    const target = new URL(url, location.href);
    if(aliases.has(absolute(url)))return aliases.get(absolute(url));
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
  function read(url, {signal, onProgress, identity} = {}) {
    if (signal?.aborted) return Promise.reject(new DOMException('Aborted', 'AbortError'));
    const key = identity ? 'sha256:'+identity : keyFor(url);
    let entry = entries.get(key);
    if (!entry) {
      entry = {listeners: new Set(), progress: {state:'loading', loaded:0, total:null}};
      entries.set(key, entry);
      const emit = progress => { entry.progress=progress; for(const listener of entry.listeners) listener(progress); };
      entry.promise = (async () => {
        if(identity){const cached=await diskRead(key);if(cached){entry.buffer=cached;retainedBytes+=cached.byteLength;emit({state:'cached',loaded:cached.byteLength,total:cached.byteLength});trim();return cached;}}
        const response = await fetch(url, {cache:'default'});
        if (!response.ok) {const error=new Error(`Resource loading failed (${response.status}) · ${new URL(url,location.href).searchParams.get('path')?.split('/').pop() || 'model asset'}`);error.status=response.status;throw error;}
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
        if(identity)await diskWrite(key,entry.buffer);
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
  window.GeoAssets={read,file,json,memo,keyFor,describe,stats:()=>({entries:entries.size,retainedBytes})};
})();
