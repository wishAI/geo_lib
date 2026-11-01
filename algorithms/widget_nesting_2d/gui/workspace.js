const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const url = path => '/api/artifact?path=' + encodeURIComponent(path);
const num = value => Number.isFinite(value) ? value.toLocaleString('en', {maximumFractionDigits:2}) : '—';
const pct = value => Number.isFinite(value) ? num(value * 100) + '%' : '—';
const palette = ['#087f8c','#dd7045','#8774b4','#c39c35','#497cc0','#639946'];
const color = id => palette[[...String(id)].reduce((sum,c) => sum + c.charCodeAt(0),0) % palette.length];
const pathData = polygon => [polygon.shell, ...(polygon.holes || [])].map(ring => 'M ' + ring.map(p => p.join(',')).join(' L ') + ' Z').join(' ');
const metrics = row => `<dl class="nw-metrics"><div><dt>Placed parts</dt><dd>${num(row.placed_count)} / ${num(row.requested_count)}</dd></div><div><dt>Requested area fitted</dt><dd>${pct(row.score?.placed_ratio_vs_requested)}</dd></div><div><dt>Board utilization</dt><dd>${pct(row.score?.board_utilization_ratio)}</dd></div><div><dt>Largest scored remainder</dt><dd>${num(row.score?.max_rest_rectangle_area)}</dd></div><div><dt>Solve + adapter</dt><dd>${num(row.solve_seconds)} s</dd></div><div><dt>Outcome</dt><dd>${esc(row.status)}</dd></div></dl>`;

// An input snapshot inside one recorded run owns every algorithm and seed result.
// Case names alone cannot establish identical geometry or constraints.
export function groupInputs(rows) {
  const groups = new Map();
  for (const row of rows) {
    const id = JSON.stringify([row.report || row.provenance?.id || row.id, row.problem]);
    if (!groups.has(id)) groups.set(id, {...row, id, trials: []});
    const group = groups.get(id);
    group.trials.push(row);
    if ((row.modifiedAt || '') > (group.modifiedAt || '')) group.modifiedAt = row.modifiedAt;
  }
  return [...groups.values()].sort((a,b) => (b.modifiedAt || '').localeCompare(a.modifiedAt || '') || a.case_id.localeCompare(b.case_id));
}
export function comparisonRows(group, seed) {
  return (group?.trials || []).filter(r => String(r.seed ?? '') === String(seed ?? ''));
}
export function preferredSeed(group) {
  const seeds = [...new Set(group.trials.map(r => String(r.seed ?? '')))];
  return seeds.sort((a,b) => new Set(comparisonRows(group,b).map(r=>r.engine)).size - new Set(comparisonRows(group,a).map(r=>r.engine)).size || (a==='42'?-1:b==='42'?1:Number(a)-Number(b)))[0];
}
export function recordedTime(row) {
  const date = new Date(row.modifiedAt);
  if (!Number.isFinite(date.getTime())) return 'Time unavailable';
  return `${row.timeKind === 'started' ? 'Started' : 'Updated'} ${date.toLocaleDateString('en-CA')} ${date.toLocaleTimeString('en-GB', {hour12:false})}`;
}
const engineName = engine => ({nfp:'Our solver',opennest:'OpenNest',spyrrow:'Sparrow',contact:'Contact'})[engine] || engine;

export function mount(root, {sandbox, api}) {
  const style = document.createElement('link'); style.rel='stylesheet'; style.href=url('algorithms/widget_nesting_2d/gui/workspace.css'); document.head.append(style);
  let saved = {};
  try { saved = JSON.parse(localStorage.getItem('geo-nesting-workspace') || '{}'); } catch {}
  let files=[], rows=[], inputs=[], selectedInput=null, seed=null, selected=null, job=null, destroyed=false, timer, sequence=0, offset=0;
  const jsonCache=new Map();
  const example=sandbox.examples[0];
  root.innerHTML=`<a class="back-link" href="#/">← All sandboxes</a><header class="nw-heading"><div><p class="eyebrow">IRREGULAR STOCK LAYOUT</p><h1>2D nesting workspace</h1><p>Configure a solve, compare algorithms on the same captured input.</p></div><button class="button" data-refresh>Refresh history</button></header>
  <div class="nw-layout"><aside class="nw-controls"><section class="nw-pane"><h2>Configuration</h2><form data-config>${example.parameters.map(p => `<label>${esc(p.label)}${p.type==='select'?`<select name="${esc(p.id)}" aria-label="${esc(p.label)}">${p.choices.map(v=>`<option value="${esc(v)}">${esc(p.choiceLabels?.[v] || v.replaceAll('_',' '))}</option>`).join('')}</select>`:`<input name="${esc(p.id)}" aria-label="${esc(p.label)}" type="number" min="${p.min}" max="${p.max}" step="${p.step || (p.type==='integer'?1:0.1)}" required>`}</label>`).join('')}<label>Execution target<select name="target" aria-label="Execution target"><option value="tk2">TK2 remote</option><option value="local">Local Mac</option></select></label><p class="nw-muted">New run folder · input & settings preserved.</p><button class="button button-primary" type="submit" data-run>Run & compare</button><button class="button" type="button" data-reset>Reset defaults</button></form></section><section class="nw-pane nw-running"><header><h2>Running</h2><button class="button button-danger" data-stop hidden>Stop</button></header><p data-job-state role="status">Ready to run</p><small data-job-id></small><pre data-log>No active job.</pre></section></aside>
  <section class="nw-pane nw-diagram"><header><h2>Diagram</h2><span data-input-note></span></header><div class="nw-diagram-tools"><label>Evidence<select data-evidence aria-label="Evidence"><option value="">Selected trial geometry</option></select></label><label>Seed<select data-seed aria-label="Comparison seed"></select></label><label><input type="checkbox" data-compare checked> Side by side</label></div><div class="nw-algorithms" data-algorithms></div><div class="nw-canvases" data-canvases><p class="nw-empty">Loading saved runs…</p></div><div class="nw-comparison" data-comparison>Select an input to compare its captured algorithm results.</div><footer data-diagram-note>Exact placement polygons · transparent holes · select a part to inspect it</footer><p data-error role="alert"></p></section>
  <aside class="nw-pane nw-properties"><h2>Selection</h2><div data-selection><p>Select a saved trial.</p></div><h3>Part properties</h3><label>Part<select data-object-list aria-label="Part"><option value="">Select a part</option></select></label><div data-object>Click a polygon in the diagram.</div><details><summary>Captured configuration & provenance</summary><pre data-provenance></pre></details></aside>
  <section class="nw-pane nw-history"><header><div><h2>History</h2><p class="nw-muted">One input · all algorithms and seeds</p></div><span data-count></span><button class="button" data-fold-history aria-expanded="true" aria-label="Fold history" title="Fold history">›</button></header><div class="nw-filters"><label>Run set<select data-set aria-label="Run set"><option value="">All run sets</option></select></label><label>Case<select data-case aria-label="Case"><option value="">All cases</option></select></label><label>Find<input type="search" data-search placeholder="Input, algorithm, outcome…"></label></div><div class="nw-table-scroll"><table><thead><tr><th>Input · recorded time · algorithms</th></tr></thead><tbody data-history></tbody></table></div><footer class="nw-pagination"><button class="button" data-prev>Previous</button><span data-page></span><button class="button" data-next>Next</button></footer></section></div>`;
  const q=s=>root.querySelector(s), form=q('[data-config]');
  for(const p of example.parameters) form.elements[p.id].value=saved.parameters?.[p.id] ?? p.default;
  form.elements.target.value=saved.target || 'tk2';
  const persist=()=>localStorage.setItem('geo-nesting-workspace',JSON.stringify({parameters:Object.fromEntries(example.parameters.map(p=>[p.id,form.elements[p.id].value])),target:form.elements.target.value,selected:selected?.id,input:selectedInput?.id,seed,sideBySide:q('[data-compare]').checked,set:q('[data-set]').value,case:q('[data-case]').value,historyFolded:q('.nw-layout').classList.contains('history-folded')}));
  function foldHistory(folded){q('.nw-layout').classList.toggle('history-folded',folded);const b=q('[data-fold-history]');b.textContent=folded?'History ‹':'›';b.setAttribute('aria-expanded',String(!folded));b.setAttribute('aria-label',folded?'Unfold history':'Fold history');b.title=folded?'Unfold history':'Fold history';}
  foldHistory(Boolean(saved.historyFolded));
  q('[data-compare]').checked=saved.sideBySide !== false;
  q('[data-fold-history]').onclick=()=>{foldHistory(!q('.nw-layout').classList.contains('history-folded'));persist();};
  form.addEventListener('change',persist);
  q('[data-reset]').onclick=()=>{for(const p of example.parameters) form.elements[p.id].value=p.default;persist();};
  const filtered=()=>inputs.filter(r=>(!q('[data-set]').value||r.set===q('[data-set]').value)&&(!q('[data-case]').value||r.case_id===q('[data-case]').value)&&(!q('[data-search]').value||JSON.stringify([r.case_id,r.set,r.trials.map(t=>[t.engine,engineName(t.engine),t.seed,t.status])]).toLowerCase().includes(q('[data-search]').value.toLowerCase())));
  function table(){
    const list=filtered(); offset=Math.max(0,Math.min(offset,Math.floor(Math.max(0,list.length-1)/75)*75));
    q('[data-count]').textContent=`${list.length} / ${inputs.length} inputs`;
    q('[data-history]').innerHTML=list.slice(offset,offset+75).map(r=>{
      const engines=[...new Set(r.trials.map(t=>t.engine))], seeds=new Set(r.trials.map(t=>t.seed));
      return `<tr class="${r.id===selectedInput?.id?'selected':''}"><td><button class="nw-input-select" data-select="${esc(r.id)}" aria-pressed="${r.id===selectedInput?.id}"><strong>${esc(r.case_id)}</strong><time datetime="${esc(r.modifiedAt || '')}">${esc(recordedTime(r))}</time><small>${esc(r.set)} · ${seeds.size} seed${seeds.size===1?'':'s'}</small><span class="nw-history-engines">${engines.map(e=>`<span>${esc(engineName(e))}</span>`).join('')}</span></button></td></tr>`;
    }).join('') || '<tr><td>No matching inputs. Refresh history or run a comparison.</td></tr>';
    q('[data-page]').textContent=list.length?`${offset+1}–${Math.min(offset+75,list.length)} of ${list.length}`:'0 inputs';q('[data-prev]').disabled=offset===0;q('[data-next]').disabled=offset+75>=list.length;
    root.querySelectorAll('[data-select]').forEach(b=>b.onclick=()=>selectInput(inputs.find(r=>r.id===b.dataset.select)));
  }
  function comparison(){
    const trials=comparisonRows(selectedInput,seed);
    q('[data-algorithms]').innerHTML=trials.map(r=>`<button class="button ${r.id===selected?.id?'active':''}" data-algorithm="${esc(r.id)}" aria-pressed="${r.id===selected?.id}">${esc(engineName(r.engine))} <small>${esc(r.status)}</small></button>`).join('');
    root.querySelectorAll('[data-algorithm]').forEach(b=>b.onclick=()=>select(rows.find(r=>r.id===b.dataset.algorithm)));
    q('[data-input-note]').textContent=selectedInput?`${selectedInput.case_id} · ${trials.length} algorithm${trials.length===1?'':'s'}`:'';
    q('[data-comparison]').innerHTML=trials.length?`<b>Same captured input · seed ${esc(seed || '—')} · ${esc(recordedTime(selectedInput))}</b><div class="nw-table-scroll"><table><thead><tr><th>Algorithm</th><th>Outcome</th><th>Parts</th><th>Area fitted</th><th>Remainder</th><th>Solve + adapter</th></tr></thead><tbody>${trials.map(r=>`<tr class="${r.id===selected?.id?'selected':''}"><td><button class="nw-row-select" data-focus="${esc(r.id)}">${esc(engineName(r.engine))}</button></td><td title="${esc(r.reason)}">${esc(r.status)}</td><td>${num(r.placed_count)} / ${num(r.requested_count)}</td><td>${pct(r.score?.placed_ratio_vs_requested)}</td><td>${num(r.score?.max_rest_rectangle_area)}</td><td>${num(r.solve_seconds)} s</td></tr>`).join('')}</tbody></table></div><small>Only results at this seed are shown. Sparrow is an all-items strip feasibility probe; an oversized strip is inconclusive, with no partial-area score.</small>`:'No algorithm result recorded at this seed.';
    root.querySelectorAll('[data-focus]').forEach(b=>b.onclick=()=>select(rows.find(r=>r.id===b.dataset.focus)));
  }
  async function loadJSON(path){
    if(!files.some(f=>f.path===path))throw new Error('Missing artifact: '+path);
    if(!jsonCache.has(path)) jsonCache.set(path,api(url(path)).catch(error=>{jsonCache.delete(path);throw error;}));
    return jsonCache.get(path);
  }
  async function geometry(row){
    const [problem,solution]=await Promise.all([loadJSON(row.problem),loadJSON(row.solution)]);
    return {problem,solution};
  }
  function svgBoards(data,row){
    return `<h3>${esc(engineName(row.engine))} · seed ${esc(row.seed ?? '—')}</h3>`+data.problem.boards.map(board=>{
      const points=board.polygon.shell,xs=points.map(p=>p[0]),ys=points.map(p=>p[1]);const x0=Math.min(...xs),y0=Math.min(...ys),w=Math.max(...xs)-x0,h=Math.max(...ys)-y0,pad=Math.max(w,h)*.045;
      const placed=data.solution.placements.filter(p=>p.board_id===board.id);
      return `<div class="nw-board"><span>${esc(board.id)} · ${esc(data.problem.units || '')}</span><svg viewBox="${x0-pad} ${y0-pad} ${w+pad*2} ${h+pad*2}" role="group" aria-label="${esc(board.id)} nesting layout"><path d="${pathData(board.polygon)}" fill="#ede8dc" fill-rule="evenodd" stroke="#69746f" vector-effect="non-scaling-stroke"/>${placed.map(p=>`<path data-part="${esc(p.item_instance_id)}" data-row="${esc(row.id)}" tabindex="0" role="button" aria-label="Part ${esc(p.item_instance_id)}" d="${pathData(p.polygon)}" fill="${color(p.item_instance_id)}" fill-rule="evenodd" stroke="#263c42" stroke-width=".6" vector-effect="non-scaling-stroke"><title>${esc(p.item_instance_id)} · ${num(p.rotation_degrees)}°</title></path>`).join('')}</svg><div class="nw-zoom"><button class="button" data-zoom="in">+</button><button class="button" data-zoom="out">−</button><button class="button" data-zoom="fit">Fit</button></div></div>`;
    }).join('');
  }
  async function diagram(){
    const token=++sequence;q('[data-error]').textContent='';
    const evidence=files.find(f=>f.path===q('[data-evidence]').value);
    if(evidence){q('[data-canvases]').innerHTML=`<figure><img class="nw-evidence" src="${url(evidence.path)}" alt="${esc(evidence.path)}"><figcaption>${esc(evidence.path)}</figcaption></figure>`;return;}
    const views=q('[data-compare]').checked?comparisonRows(selectedInput,seed):[selected].filter(Boolean);
    q('[data-canvases]').innerHTML='<p class="nw-empty" role="status">Loading captured layouts…</p>';
    const loaded=await Promise.all(views.map(async r=>{try{return {row:r,data:await geometry(r)};}catch(error){return {row:r,error:error.message};}}));
    if(destroyed||token!==sequence)return;
    q('[data-canvases]').classList.toggle('split',views.length>1);
    q('[data-canvases]').style.setProperty('--nw-view-count',views.length);
    q('[data-canvases]').innerHTML=loaded.map(v=>`<div class="nw-canvas">${v.data?svgBoards(v.data,v.row):`<h3>${esc(engineName(v.row.engine))}</h3><p class="nw-empty">${esc(v.row.reason || v.error)}</p><p>Outcome: ${esc(v.row.status)}. Select an evidence image above if one is available.</p>`}</div>`).join('')||'<p class="nw-empty">No saved trial selected. Run nesting or select an evidence image.</p>';
    root.querySelectorAll('.nw-board').forEach(board=>{
      const svg=board.querySelector('svg'),original=svg.getAttribute('viewBox').split(' ').map(Number);
      const zoom=factor=>{const [x,y,w,h]=svg.getAttribute('viewBox').split(' ').map(Number);svg.setAttribute('viewBox',[x+w*(1-factor)/2,y+h*(1-factor)/2,w*factor,h*factor].join(' '));};
      board.querySelectorAll('[data-zoom]').forEach(b=>{b.setAttribute('aria-label',b.dataset.zoom==='fit'?'Fit board':b.dataset.zoom==='in'?'Zoom in':'Zoom out');b.onclick=()=>b.dataset.zoom==='fit'?svg.setAttribute('viewBox',original.join(' ')):zoom(b.dataset.zoom==='in'?.75:1.333);});
      svg.addEventListener('wheel',e=>{e.preventDefault();zoom(e.deltaY<0?.85:1.176);},{passive:false});
    });
    const parts=loaded.find(v=>v.row.id===selected?.id)?.data?.solution.placements || [];
    q('[data-object-list]').innerHTML='<option value="">Select a part</option>'+parts.map(p=>`<option value="${esc(p.item_instance_id)}">${esc(p.item_instance_id)}</option>`).join('');
    q('[data-object-list]').onchange=()=>{const el=[...root.querySelectorAll('[data-part]')].find(el=>el.dataset.row===selected?.id&&el.dataset.part===q('[data-object-list]').value);el?.click();};
    root.querySelectorAll('[data-part]').forEach(el=>{
      const show=()=>{q('[data-object-list]').value=el.dataset.part;root.querySelectorAll('[data-part]').forEach(p=>p.classList.toggle('part-selected',p.dataset.part===el.dataset.part));const view=loaded.find(v=>v.row.id===el.dataset.row);const p=view.data.solution.placements.find(p=>p.item_instance_id===el.dataset.part);q('[data-object]').innerHTML=`<b>${esc(engineName(view.row.engine))} · ${esc(p.item_instance_id)}</b><dl><dt>Widget</dt><dd>${esc(p.widget_id)}</dd><dt>Board</dt><dd>${esc(p.board_id)}</dd><dt>Rotation</dt><dd>${num(p.rotation_degrees)}°</dd><dt>Centroid</dt><dd>${p.centroid?.map(num).join(', ')}</dd><dt>Holes</dt><dd>${p.polygon.holes?.length || 0}</dd></dl>`;};el.onclick=show;el.onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();show();}};
    });
    q('[data-diagram-note]').textContent='Exact captured polygons · matching part colors across algorithms · select an algorithm or a part to inspect';
    if(loaded.some(view=>view.data) && !root.dataset.comparisonReadyMs) root.dataset.comparisonReadyMs=String(Math.round(performance.now()));
  }
  function select(row){selected=row;q('[data-evidence]').value='';q('[data-selection]').innerHTML=`<h3>${esc(row.case_id)}</h3><p>${esc(row.set)} · ${esc(engineName(row.engine))} · seed ${esc(row.seed ?? '—')}</p>${metrics(row)}${files.some(f=>f.path===row.solution)?`<a href="${url(row.solution)}" target="_blank" rel="noopener">Raw solution JSON ↗</a>`:''}`;q('[data-object]').textContent='Click or focus a polygon to inspect it.';q('[data-provenance]').textContent=JSON.stringify(row.provenance || {source:row.report,contract:row.contract,reason:row.reason},null,2);persist();table();comparison();void diagram();}
  function selectInput(input, requestedSeed, focusId){
    selectedInput=input;
    const seeds=[...new Set(input.trials.map(r=>String(r.seed ?? '')))].sort((a,b)=>Number(a)-Number(b));
    seed=seeds.includes(String(requestedSeed))?String(requestedSeed):preferredSeed(input);
    q('[data-seed]').innerHTML=seeds.map(s=>`<option value="${esc(s)}">${esc(s || '—')} · ${comparisonRows(input,s).length} algorithm${comparisonRows(input,s).length===1?'':'s'}</option>`).join('');
    q('[data-seed]').value=seed;
    const trials=comparisonRows(input,seed);
    select(trials.find(r=>r.id===focusId)||trials.find(r=>r.engine==='nfp')||trials[0]);
  }
  async function history(){
    q('[data-refresh]').disabled=true;q('[data-count]').textContent='Loading inputs…';
    const payload=await api('/api/nesting/history?include=reports');if(destroyed)return;files=payload.files;jsonCache.clear();
    for(const [path,data] of Object.entries(payload.documents || {})) jsonCache.set(path,Promise.resolve(data));
    for(const error of payload.errors || []) q('[data-error]').textContent=`${error.path}: ${error.error}`;
    const reports=files.filter(f=>/\/(analysis|index)\.json$/.test(f.path)).filter(f=>!f.path.endsWith('/index.json')||!files.some(a=>a.path===f.path.replace('/index.json','/analysis.json')));
    const groups=await Promise.all(reports.map(async f=>{try{if(payload.documents && !payload.documents[f.path]) throw new Error('Unable to read captured report: '+f.path);const data=await loadJSON(f.path);const dir=f.path.slice(0,f.path.lastIndexOf('/'));return (data.results||[]).map(r=>({...r,id:`${dir}/${r.id||r.case_id}/${r.engine}_${r.seed}`,set:dir.split('/').pop(),modifiedAt:data.startedAt||f.modifiedAt,timeKind:data.startedAt?'started':'updated',report:f.path,contract:data.contract,problem:r.problemPath||`${dir}/${r.case_id}/problem.json`,solution:r.solutionPath||`${dir}/${r.case_id}/${r.engine}_${r.seed}.solution.json`}));}catch(error){q('[data-error]').textContent=error.message;return [];}}));
    const runs=files.filter(f=>f.path.endsWith('/run.json'));
    groups.push((await Promise.all(runs.map(async f=>{
      const run=await loadJSON(f.path), dir=f.path.slice(0,-9);
      const common={modifiedAt:run.startedAt||f.modifiedAt,timeKind:run.startedAt?'started':'updated',set:'GUI runs',case_id:run.caseId,report:f.path,contract:run.contract,problem:dir+'/problem.json',provenance:run};
      if(run.results?.length) return run.results.map(r=>({...common,...r,id:`${dir}/${r.engine}_${r.seed}`,solution:`${dir}/${r.engine}_${r.seed}.solution.json`}));
      let solution={};try{solution=await loadJSON(dir+'/solution.json');}catch{}
      return [{...common,id:dir,engine:run.parameters?.candidate_mode||'nfp',seed:run.parameters?.seed,status:run.status,reason:run.error,score:solution.score,placed_count:solution.placements?.length,requested_count:solution.placements?solution.placements.length+(solution.skipped_item_ids?.length||0):undefined,solve_seconds:solution.search_stats?.solve_seconds,solution:dir+'/solution.json'}];
    }))).flat());
    if(destroyed)return;rows=groups.flat().sort((a,b)=>(b.modifiedAt||'').localeCompare(a.modifiedAt||'')||a.case_id.localeCompare(b.case_id)||String(a.engine).localeCompare(String(b.engine))||Number(a.seed)-Number(b.seed));
    for(const [sel,key,initial] of [['[data-set]','set',saved.set],['[data-case]','case_id',saved.case]]){const el=q(sel),value=el.value||initial||'';el.innerHTML=`<option value="">All ${key==='set'?'run sets':'cases'}</option>`+[...new Set(rows.map(r=>r[key]))].sort().map(v=>`<option value="${esc(v)}">${esc(v)}</option>`).join('');el.value=value;}
    const images=files.filter(f=>f.kind==='image').sort((a,b)=>b.modifiedAt.localeCompare(a.modifiedAt));const imageValue=q('[data-evidence]').value;q('[data-evidence]').innerHTML='<option value="">Selected trial geometry</option>'+images.map(f=>`<option value="${esc(f.path)}">${esc(f.path.split('/outputs/')[1])}</option>`).join('');q('[data-evidence]').value=imageValue;
    inputs=groupInputs(rows);
    const next=inputs.find(r=>r.id===(selectedInput?.id||saved.input)||r.trials.some(t=>t.id===(selected?.id||saved.selected)))||inputs.find(r=>r.trials.some(t=>t.engine==='nfp')&&new Set(r.trials.map(t=>t.engine)).size>1)||inputs[0];
    if(next)selectInput(next,seed??saved.seed,selected?.id||saved.selected);else{table();comparison();void diagram();}
    q('[data-refresh]').disabled=false;
  }

  async function refresh(){
    try{const payload=await api('/api/jobs');if(destroyed)return;const jobs=payload.jobs.filter(j=>j.sandbox===sandbox.id);const next=jobs.find(j=>['queued','running','cancelling'].includes(j.status))||jobs[0];const finished=job&&next?.id===job.id&&job.status!==next.status&&!['queued','running','cancelling'].includes(next.status);job=next;
      const active=job&&['queued','running','cancelling'].includes(job.status);q('[data-run]').disabled=Boolean(active);q('[data-stop]').hidden=!active;q('[data-stop]').disabled=job?.status==='cancelling';q('[data-job-state]').textContent=job?`${job.status} · ${job.target}`:'Ready to run';q('[data-job-id]').textContent=job?`Run ${job.id}`:'';q('[data-log]').textContent=job?.log||'No active job.';q('[data-log]').classList.toggle('idle',!job);
      if(finished){await history();const r=rows.find(r=>r.provenance?.id===job.id);if(r){q('[data-set]').value='GUI runs';q('[data-case]').value=r.case_id;q('[data-search]').value='';offset=0;selectInput(inputs.find(i=>i.trials.some(t=>t.id===r.id)),r.seed,r.id);}}
      clearTimeout(timer);if(active&&document.visibilityState==='visible')timer=setTimeout(refresh,1600);
    }catch(error){if(!destroyed)q('[data-error]').textContent=error.message;}
  }
  form.onsubmit=async e=>{e.preventDefault();q('[data-run]').disabled=true;try{persist();const parameters=Object.fromEntries(example.parameters.map(p=>[p.id,form.elements[p.id].value]));job=await api('/api/jobs',{method:'POST',body:JSON.stringify({sandbox:sandbox.id,example:example.id,target:form.elements.target.value,parameters})});await refresh();}catch(error){q('[data-error]').textContent=error.message;q('[data-run]').disabled=false;}};
  q('[data-stop]').onclick=async()=>{try{await api(`/api/jobs/${job.id}/cancel`,{method:'POST',body:'{}'});await refresh();}catch(error){q('[data-error]').textContent=error.message;}};
  q('[data-refresh]').onclick=()=>void history().catch(e=>{q('[data-error]').textContent=e.message;q('[data-refresh]').disabled=false;});
  q('[data-seed]').onchange=()=>selectInput(selectedInput,q('[data-seed]').value);
  for(const s of ['[data-set]','[data-case]','[data-search]'])q(s).oninput=()=>{offset=0;persist();table();};
  q('[data-prev]').onclick=()=>{offset-=75;table();};q('[data-next]').onclick=()=>{offset+=75;table();};q('[data-compare]').onchange=()=>{persist();void diagram();};q('[data-evidence]').onchange=()=>void diagram();
  const visible=()=>{if(document.visibilityState==='visible')void refresh();else clearTimeout(timer);};document.addEventListener('visibilitychange',visible);
  // Document-relative readiness timings make cold-open performance inspectable.
  root.dataset.controlsReadyMs=String(Math.round(performance.now()));
  void history().catch(e=>{q('[data-error]').textContent=e.message;q('[data-refresh]').disabled=false;});void refresh();
  return {refresh,destroy(){destroyed=true;clearTimeout(timer);sequence++;style.remove();document.removeEventListener('visibilitychange',visible);}};
}
