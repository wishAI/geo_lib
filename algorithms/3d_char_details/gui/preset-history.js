const API='/api/character/presets';
const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));

export function presetHistory(root,{read,apply,compatible,announce,signal}){
  const pane=root.querySelector('[data-pane="presets"]');
  pane.innerHTML=`<div class="char-preset-save"><label>Preset name<input data-preset-name aria-label="Preset name" maxlength="80" value="Landau"></label><button data-history-save>Save version</button></div><p class="char-hint">Each save keeps a separate version in this sandbox. Restoring a version keeps the history intact.</p><div class="char-history-tools"><label><input type="checkbox" data-history-archived> Archived</label><button data-history-refresh>Refresh</button></div><div data-history-list>Loading history…</div><details class="char-preset-files"><summary>JSON files</summary><div class="char-presets"><button data-action="load">Import JSON</button><button data-action="preset-download">Download current</button></div></details>`;
  let busy=false,entries=[];
  const name=pane.querySelector('[data-preset-name]'),list=pane.querySelector('[data-history-list]');
  const archived=pane.querySelector('[data-history-archived]');
  try{name.value=localStorage.getItem('landau-preset-name')||'Landau';}catch{}
  const rememberName=()=>{try{localStorage.setItem('landau-preset-name',name.value);}catch{}};
  name.addEventListener('change',rememberName,{signal});
  async function request(body,id){
    const res=await fetch(API+(id?'?id='+encodeURIComponent(id):''),{signal,...(body?{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}:{})});
    const data=await res.json();if(!res.ok)throw new Error(data.error||'Could not access preset history');return data;
  }
  function render(){
    const shown=entries.filter(e=>e.archived===archived.checked);
    list.innerHTML=shown.length?shown.map(e=>`<article class="char-history-entry" data-history-id="${e.id}"><div class="char-history-title"><input aria-label="Version name ${e.id}" value="${esc(e.name)}" maxlength="80"><button data-history-rename="${e.id}" title="Rename version" aria-label="Rename ${esc(e.name)}">✎</button></div><time datetime="${e.createdAt}">${esc(new Date(e.createdAt).toLocaleString())}</time><div class="char-history-row"><span title="Asset revision ${e.assetHash}">${compatible(e.assetHash)?'Compatible':'Different model'} · ${e.assetHash.slice(0,7)}</span><button data-history-restore="${e.id}" ${compatible(e.assetHash)?'':'disabled'}>Restore</button><button data-history-archive="${e.id}" data-archived="${!e.archived}">${e.archived?'Unarchive':'Archive'}</button></div></article>`).join(''):`<p class="char-hint">${archived.checked?'No archived versions.':'No saved versions yet.'}</p>`;
  }
  async function refresh(){try{entries=(await request()).entries;if(!signal.aborted)render();}catch(err){if(!signal.aborted){list.textContent='History unavailable. Your working edits remain in this browser.';announce(err.message);}}}
  async function run(fn){if(busy)return;busy=true;pane.setAttribute('aria-busy','true');try{await fn();}catch(err){if(!signal.aborted)announce(err.message);}finally{busy=false;pane.removeAttribute('aria-busy');}}
  async function save(){await run(async()=>{const title=name.value.trim();if(!title)throw new Error('Enter a preset name.');const entry=await request({action:'save',name:title,settings:read()});rememberName();await refresh();announce(`Saved “${entry.name}” in sandbox history.`);});}
  pane.addEventListener('click',e=>{
    const b=e.target.closest('button');if(!b)return;
    if(b.hasAttribute('data-history-save'))void save();
    if(b.hasAttribute('data-history-refresh'))void refresh();
    if(b.dataset.historyRestore)void run(async()=>{const entry=await request(null,b.dataset.historyRestore);apply(entry.settings);name.value=entry.name;rememberName();announce(`Restored “${entry.name}” · ${new Date(entry.createdAt).toLocaleString()}`);});
    if(b.dataset.historyRename)void run(async()=>{await request({action:'rename',id:b.dataset.historyRename,name:b.closest('article').querySelector('input').value});await refresh();announce('Version renamed.');});
    if(b.dataset.historyArchive)void run(async()=>{await request({action:'archive',id:b.dataset.historyArchive,archived:b.dataset.archived==='true'});await refresh();announce(b.dataset.archived==='true'?'Version archived. It can be recovered under Archived.':'Version returned to history.');});
  },{signal});
  archived.addEventListener('change',render,{signal});
  name.addEventListener('keydown',e=>{if(e.key==='Enter'){e.preventDefault();void save();}},{signal});
  return {refresh,save};
}
