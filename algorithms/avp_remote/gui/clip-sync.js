// Keep large JSON serialization and network uploads off the immersive render thread.
self.onmessage=async()=>{
 let db;
 try{
  db=await new Promise((resolve,reject)=>{const r=indexedDB.open('avp-motion-studio',1);r.onupgradeneeded=()=>r.result.createObjectStore('clips');r.onsuccess=()=>resolve(r.result);r.onerror=()=>reject(r.error);});
  const access=(key,value)=>new Promise((resolve,reject)=>{const tx=db.transaction('clips',value===undefined?'readonly':'readwrite'),s=tx.objectStore('clips'),r=value===undefined?s.get(key):s.put(value,key);tx.oncomplete=()=>resolve(r.result);tx.onerror=()=>reject(tx.error);tx.onabort=()=>reject(tx.error||new Error('Storage aborted'));});
  const request=async(url,options)=>{const r=await fetch(url,{...options,signal:AbortSignal.timeout(120000)});const data=await r.json();if(!r.ok)throw new Error(data.error||r.statusText);return data;};
  const remote=(await request('/api/avp/recordings')).recordings,ids=new Set(remote.map(t=>t.id));
  const summaries=await new Promise((resolve,reject)=>{const tx=db.transaction('clips'),r=tx.objectStore('clips').getAll(IDBKeyRange.bound('summary:','summary:\uffff'));tx.oncomplete=()=>resolve(r.result);tx.onerror=()=>reject(tx.error);});
  let uploaded=0;
  for(const summary of summaries){
   const previous=await access('remote:'+summary.key);if(previous&&ids.has(previous))continue;
   const clip=await access(summary.key);if(!clip?.frames?.length)continue;
   const saved=await request('/api/avp/recordings',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(clip)});
   await access('remote:'+summary.key,saved.id);ids.add(saved.id);uploaded++;
  }
  self.postMessage({uploaded});
 }catch(error){self.postMessage({error:error.message});}finally{db?.close();self.close();}
};
