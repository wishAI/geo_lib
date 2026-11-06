/* Small, revalidated lineage indexes; full node records are fetched on demand. */
(function () {
  'use strict';
  const indexes = new Map(), details = new Map(), pending = new Map();
  async function read(url, onProgress) {
    const controller = new AbortController(), timeout = setTimeout(() => controller.abort(), 30000);
    try {
      onProgress?.({loaded:0, total:null, state:'connecting'});
      const response = await fetch(url, {cache:'no-cache', signal:controller.signal});
      if (!response.ok) throw new Error(`Lineage request failed (${response.status})`);
      const total = Number(response.headers.get('X-Uncompressed-Content-Length')) ||
        (!response.headers.get('Content-Encoding') && Number(response.headers.get('Content-Length'))) || null;
      const reader = response.body?.getReader(), chunks = [];
      let loaded = 0;
      if (!reader) return await response.json();
      while (true) {
        const {done, value} = await reader.read();
        if (done) break;
        chunks.push(value); loaded += value.byteLength;
        onProgress?.({loaded, total, state:'loading'});
      }
      const buffer = new Uint8Array(loaded);
      let offset = 0;
      for (const chunk of chunks) { buffer.set(chunk, offset); offset += chunk.byteLength; }
      onProgress?.({loaded, total:loaded, state:'ready'});
      return JSON.parse(new TextDecoder().decode(buffer));
    } finally { clearTimeout(timeout); }
  }
  function request(key, url, onProgress) {
    if (!pending.has(key)) {
      const entry = {listeners:new Set(), progress:null};
      pending.set(key, entry);
      entry.promise = read(url, progress => {
        entry.progress = progress;
        for (const listener of entry.listeners) listener(progress);
      }).finally(() => pending.delete(key));
    }
    const entry = pending.get(key);
    if (onProgress) { entry.listeners.add(onProgress); if (entry.progress) onProgress(entry.progress); }
    return entry.promise.finally(() => entry.listeners.delete(onProgress));
  }
  async function index(sandbox, {force=false, onProgress} = {}) {
    const cached = indexes.get(sandbox);
    if (!force && cached && Date.now() - cached.time < 60000) return cached.data;
    const data = await request(sandbox, '/api/lineage?sandbox='+encodeURIComponent(sandbox), onProgress);
    indexes.set(sandbox, {data, time:Date.now()});
    return data;
  }
  async function node(sandbox, revision, id, onProgress) {
    const key = JSON.stringify([sandbox, revision, id]);
    if (details.has(key)) return details.get(key);
    const payload = await request(key, '/api/lineage?sandbox='+encodeURIComponent(sandbox)+'&node='+encodeURIComponent(id), onProgress);
    if (payload.revision !== revision) throw new Error('Lineage changed. Refresh the tree to load this record.');
    details.set(key, payload.node);
    if (details.size > 64) details.delete(details.keys().next().value);
    return payload.node;
  }
  window.GeoLineage = {index, node, peek:sandbox=>indexes.get(sandbox)?.data};
})();
