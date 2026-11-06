(function () {
  'use strict';

  const escapeHtml = value => String(value ?? '').replace(/[&<>'"]/g, character => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;' })[character]);
  const formatBytes = value => value == null ? '—' : new Intl.NumberFormat('en', { notation: 'compact', maximumFractionDigits: 1 }).format(value) + 'B';
  const formatNumber = value => typeof value === 'number' ? new Intl.NumberFormat('en', { maximumFractionDigits: 4 }).format(value) : '—';
  const compactNumber = value => typeof value === 'number' ? new Intl.NumberFormat('en', { notation: 'compact', maximumFractionDigits: 1 }).format(value) : '—';

  const displayLabel = node => String(node?.label || 'No runs recorded').replace(/^anatomical(?: mass)? \d+ /i, '').replaceAll('_', ' ');

  const svgPaths = {
    model:'<path d="m12 3 8 4.5v9L12 21l-8-4.5v-9L12 3Z"/><path d="m4 7.5 8 4.5 8-4.5M12 12v9"/>',
    joint:'<circle cx="7" cy="7" r="3"/><circle cx="17" cy="17" r="3"/><path d="m9 9 6 6M4 15l5 5M15 4l5 5"/>',
    storage:'<ellipse cx="12" cy="5" rx="8" ry="3"/><path d="M4 5v14c0 4 16 4 16 0V5M4 12c0 4 16 4 16 0"/>',
    mass:'<path d="M8 7h8l4 14H4L8 7Z"/><circle cx="12" cy="4" r="3"/>',
    branch:'<circle cx="6" cy="4" r="2"/><circle cx="6" cy="20" r="2"/><circle cx="18" cy="5" r="2"/><path d="M6 6v12M6 14c8 0 12-2 12-7"/>',
    check:'<path d="m5 12 4 4L19 6"/>',
    warning:'<path d="m12 3 10 18H2L12 3ZM12 9v5M12 17v1"/>',
    play:'<path d="m8 4 12 8-12 8V4Z"/>',
    file:'<path d="M14 2H5v20h14V7l-5-5ZM14 2v6h5M8 12h8M8 16h6"/>',
    clock:'<circle cx="12" cy="12" r="9"/><path d="M12 6v6l4 2"/>',
    expand:'<path d="M8 3H3v5M16 3h5v5M21 16v5h-5M8 21H3v-5"/>',
    chevron:'<path d="m9 5 7 7-7 7"/>',
    close:'<path d="m6 6 12 12M6 18 18 6"/>',
    settings:'<path d="M4 7h16M4 17h16"/><circle cx="9" cy="7" r="3"/><circle cx="15" cy="17" r="3"/>',
  };
  const icon = name => `<svg class="walk-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${svgPaths[name]||svgPaths.file}</svg>`;
  const modelLabel = model => model?.urdfPath?.includes('anatomical_legs_mass_v2')?'Anatomical v2':model?.urdfPath?.includes('anatomical_legs_v1')?'Anatomical v1':model?.urdfPath?.includes('landau_v10')?'Landau v10':model?.label||'Unknown model';
  const loading = label => `<div class="walk-loading" role="status"><span class="loading-spinner"></span><span>${escapeHtml(label)}</span></div>`;

  const modelPreviewImages = new Map();
  function modelPreview(host, model) {
    const key=model.id+':'+model.urdfPath;
    const showImage=source=>{const img=new Image();img.src=source;img.alt=modelLabel(model)+' · URDF neutral pose';host.replaceChildren(img);host.dataset.previewReady='true';};
    if(modelPreviewImages.has(key)){showImage(modelPreviewImages.get(key));return ()=>{};}
    host.innerHTML='<canvas data-preview-mesh></canvas><canvas data-preview-overlay></canvas><span class="model-preview-loading" role="status"><i class="loading-spinner"></i><span>Loading model preview…</span></span>';
    let viewer,stopped=false;
    const dispose=()=>{if(stopped)return;stopped=true;if(viewer){viewer.destroy();viewer.gl?.getExtension('WEBGL_lose_context')?.loseContext();}};
    void (async()=>{try{
      viewer=new window.MotionUrdfViewer({meshCanvas:host.querySelector('[data-preview-mesh]'),overlayCanvas:host.querySelector('[data-preview-overlay]'),assetUrl:'/api/robot/asset?urdf='+encodeURIComponent(model.urdfPath),onStatus:text=>{const label=host.querySelector('.model-preview-loading span');if(label)label.textContent=text;}});
      viewer.showInspection=true;viewer.trainingJoints=Array.isArray(model.actionJoints)?new Set(model.actionJoints):null;
      const stats=await viewer.load('/api/robot/urdf?path='+encodeURIComponent(model.urdfPath),[],[]);
      if(stopped||!host.isConnected)return;
      if(!stats.loaded)throw new Error('Model meshes unavailable');
      // Capture immediately after drawing, while the WebGL framebuffer is valid.
      viewer.render();const canvas=document.createElement('canvas');canvas.width=viewer.meshCanvas.width;canvas.height=viewer.meshCanvas.height;
      const ctx=canvas.getContext('2d');ctx.drawImage(viewer.meshCanvas,0,0);ctx.drawImage(viewer.overlayCanvas,0,0);
      const source=canvas.toDataURL('image/webp',.88);
      if(!stats.failed)modelPreviewImages.set(key,source);
      showImage(source);
    }catch(error){if(!stopped){host.innerHTML=`<span class="model-preview-unavailable">${icon('model')}<span>Preview unavailable</span></span>`;host.title=error.message;}}
    finally{dispose();}})();
    return dispose;
  }
  function recordedTime(value) {
    if(!value)return null;
    let stamp=String(value);
    const compact=stamp.match(/^(20\d{2})(\d{2})(\d{2})[_T-](\d{2})(\d{2})(\d{2})/);
    if(compact)stamp=`${compact[1]}-${compact[2]}-${compact[3]}T${compact[4]}:${compact[5]}:${compact[6]}`;
    const date=new Date(stamp);if(!Number.isFinite(date.getTime()))return null;
    return {value:date.getTime(),raw:String(value),label:new Intl.DateTimeFormat(undefined,{year:'numeric',month:'short',day:'2-digit',hour:'2-digit',minute:'2-digit'}).format(date)};
  }

  function modelPicker(container, models, raw, current, choose, trigger) {
    const dialog=document.createElement('dialog');dialog.className='walk-model-picker';dialog.setAttribute('aria-label','Choose robot model');
    const card=m=>{
      const runs=raw.filter(n=>n.modelId===m.id),passed=(m.milestones||[]).filter(g=>g.status==='passed').length;
      const mass=Object.entries(m.massGroupsKg||{});
      const times=runs.map(n=>recordedTime(n.startedAt)).filter(Boolean).sort((a,b)=>a.value-b.value);
      const timeRow=(label,time)=>`<div><span>${label}</span><time title="${escapeHtml(time?.raw||'')}">${escapeHtml(time?.label||'Not recorded')}</time></div>`;
      return `<button class="model-choice ${m.id===current?'selected':''}" data-choose-model="${escapeHtml(m.id)}"><div class="model-choice-top"><span class="model-emblem">${icon('model')}</span><span class="model-choice-badge">${m.id===current?'Selected':m.urdfPath?'URDF available':'Historical record'}</span></div>${m.urdfPath?`<div class="model-preview" data-model-preview="${escapeHtml(m.id)}"></div>`:`<div class="model-preview"><span class="model-preview-unavailable">${icon('model')}<span>No verified URDF preview</span></span></div>`}<h3>${escapeHtml(modelLabel(m))}</h3><p>${escapeHtml(m.label)}</p><div class="model-choice-stats"><span>${icon('mass')}<b>${m.massKg?m.massKg+' kg':'Unknown mass'}</b></span><span>${icon('joint')}<b>${m.actionJoints?.length??'—'} actions</b></span><span>${icon('branch')}<b>${runs.length} nodes</b></span></div><div class="model-recorded-times">${icon('clock')}<div>${timeRow('First record',times[0])}${timeRow('Latest record',times.at(-1))}</div></div><div class="model-choice-progress"><span>${m.milestones?.length?`${passed} / ${m.milestones.length} certified gates`:'No certification ledger'}</span><progress value="${passed}" max="${m.milestones?.length||12}"></progress></div><div class="model-mass-chips">${mass.map(([k,v])=>`<span>${escapeHtml(k.replaceAll('_',' '))} <b>${formatNumber(v)} kg</b></span>`).join('')}</div><footer><code>${escapeHtml(m.urdfPath?.split('/').slice(-2).join('/')||'URDF mapping not recorded')}</code>${icon('chevron')}</footer></button>`;
    };
    const known=models.filter(m=>m.urdfPath),history=models.filter(m=>!m.urdfPath);
    dialog.innerHTML=`<header><div><p class="eyebrow">MODEL LIBRARY</p><h2>Choose a robot model</h2><p>Each model keeps its own experiments and certification record.</p></div><button class="button button-light" data-close-models aria-label="Close model picker">${icon('close')}</button></header><div class="model-picker-scroll"><div class="model-card-grid">${known.map(card).join('')}</div>${history.length?`<details class="historical-models"><summary>Historical & comparison models <span>${history.length}</span></summary><div class="model-card-grid">${history.map(card).join('')}</div></details>`:''}</div>`;
    const previews=[];
    const close=()=>{previews.forEach(dispose=>dispose());dialog.close();dialog.remove();container._closePicker=null;trigger?.focus({preventScroll:true});};
    container._closePicker=close;container.append(dialog);
    dialog.querySelector('[data-close-models]').onclick=close;
    dialog.addEventListener('cancel',e=>{e.preventDefault();close();});
    dialog.addEventListener('click',e=>{if(e.target===dialog)close();});
    dialog.querySelectorAll('[data-choose-model]').forEach(button=>button.onclick=()=>{const id=button.dataset.chooseModel;close();choose(id);});
    dialog.showModal();
    dialog.querySelectorAll('[data-model-preview]').forEach(host=>previews.push(modelPreview(host,known.find(m=>m.id===host.dataset.modelPreview))));
  }

  function detailsPane(node, robot, nodes, milestones) {
    if(!node)return '<p class="evolution-empty">No recorded runs for this model.</p>';
    const status=node.kind==='milestone'&&node.status==='completed'?'Certified':node.status==='failed'?'Needs work':node.status==='running'?'In progress':'Recorded';
    const gate=milestones.find(m=>m.id===node.milestoneId);
    const stats=[['joint','Policy actions',node.actionJoints?.length??'—'],['storage','Checkpoint',formatBytes(node.diskBytes)],['mass','Body mass',robot?.massKg?robot.massKg+' kg':'—']];
    if (node.trainingProgress?.kind==='training') stats.push(['clock','Training updates',`${node.trainingProgress.completedIterations??0} / ${node.trainingProgress.requestedIterations??'—'}`]);
    if (node.metrics?.mean_forward_speed_mps != null) stats.push(['clock','Mean forward speed',formatNumber(node.metrics.mean_forward_speed_mps)+' m/s']);
    if (node.metrics?.sustained_forward_speed_mps != null) stats.push(['clock','Sustained speed',formatNumber(node.metrics.sustained_forward_speed_mps)+' m/s']);
    if (gate?.targetSpeedMps) stats.push(['clock','Gate target',gate.targetSpeedMps+' m/s']);
    if (node.diagnosticCurriculumLevel != null) stats.push(['branch','Diagnostic gait level',(100*node.diagnosticCurriculumLevel).toFixed(1)+'% · not certifiable']);
    if (node.diagnosticPhaseTemplate) stats.push(['branch','M7 phase retarget',`${node.diagnosticPhaseTemplate.period_s} s cycle · motor diagnostic · not certifiable`]);
    if (node.metrics?.curriculum_mean_level != null) stats.push(['branch','Average gait level',(100*node.metrics.curriculum_mean_level).toFixed(1)+'%']);
    if (node.metrics?.curriculum_speed_target_mean_mps != null) stats.push(['clock','Current training target',formatNumber(node.metrics.curriculum_speed_target_mean_mps)+' m/s']);
    if (node.metrics?.recent_episode_survival_fraction != null) stats.push(['check','Recent training survival',(100*node.metrics.recent_episode_survival_fraction).toFixed(1)+'%']);
    if (node.metrics?.curriculum_target_fraction != null) stats.push(['branch','Worlds at target gait',(100*node.metrics.curriculum_target_fraction).toFixed(1)+'%']);
    const mass=Object.entries(robot?.massGroupsKg||{}),total=mass.reduce((v,[,n])=>v+Number(n),0);
    const colors=['#427b86','#77a68c','#bfa06a','#a18daf','#cb8c78'];
    const metrics=Object.entries(node.metrics||{}).filter(([,v])=>typeof v==='number').slice(0,4);
    return `<div class="node-detail-design"><section class="node-outcome ${escapeHtml(node.status)}"><div>${icon(node.status==='failed'?'warning':'check')}<span>${status}</span>${gate?`<small>${escapeHtml(gate.displayLabel || `M${gate.order}`)} · ${escapeHtml(gate.label)}</small>`:''}</div><p>${escapeHtml(node.result||'No observed result recorded.')}</p></section><div class="node-stat-grid">${stats.map(([i,k,v])=>`<div>${icon(i)}<strong>${escapeHtml(v)}</strong><span>${k}</span></div>`).join('')}</div><section class="node-detail-section"><h4>${icon('model')}Robot configuration</h4><div class="node-model-identity"><span class="model-emblem">${icon('model')}</span><div><b>${escapeHtml(modelLabel(robot))}</b><small>${escapeHtml(node.bodySettings?.['mass profile']||'Model configuration from recorded evidence')}</small></div></div><div class="node-setting-chips">${Object.entries(node.bodySettings||{}).filter(([k,v])=>!k.includes('SHA')&&k!=='mass profile'&&!Array.isArray(v)).map(([k,v])=>`<span>${escapeHtml(k)}<b>${escapeHtml(v)}</b></span>`).join('')}</div></section><section class="node-detail-section"><h4>${icon('branch')}Checkpoint ancestry</h4><div class="node-parent-links">${(node.parentIds||[]).length?node.parentIds.map(id=>`<button data-select-parent="${escapeHtml(id)}" ${nodes.has(id)?'':'disabled'}>${icon('branch')}<span>${escapeHtml(displayLabel(nodes.get(id)||{label:id}))}</span>${icon('chevron')}</button>`).join(''):'<p>No parent checkpoint recorded.</p>'}</div></section>${mass.length?`<section class="node-detail-section"><h4>${icon('mass')}Mass distribution <small>kg</small></h4><div class="mass-distribution-bar">${mass.map(([k,v],i)=>`<span style="flex:${Number(v)/total};background:${colors[i%colors.length]}" title="${escapeHtml(k)}: ${v} kg"></span>`).join('')}</div><div class="mass-distribution-legend">${mass.map(([k,v],i)=>`<div><i style="background:${colors[i%colors.length]}"></i><span>${escapeHtml(k.replaceAll('_',' '))}</span><b>${formatNumber(v)}</b></div>`).join('')}</div></section>`:''}${metrics.length?`<section class="node-detail-section"><h4>${icon('settings')}Observed metrics</h4><div class="node-metric-grid">${metrics.map(([k,v])=>`<div><b>${formatNumber(v)}</b><span>${escapeHtml(k.replaceAll('_',' '))}</span></div>`).join('')}</div></section>`:''}<section class="node-detail-section"><h4>${icon('file')}Evidence & provenance</h4><div class="node-evidence-actions">${(node.artifacts||[]).filter(a=>['video','json','image'].includes(a.kind)||a.path.endsWith('.json')).slice(0,4).map(a=>`<button data-detail-artifact="${escapeHtml(a.path)}" data-kind="${escapeHtml(a.kind)}">${icon(a.kind==='video'?'play':'file')}<span>${escapeHtml(a.path.split('/').pop())}</span>${icon('chevron')}</button>`).join('')}</div><details class="node-provenance"><summary>Checkpoint path & all metrics</summary><code>${escapeHtml(typeof node.checkpointPath==='string'?node.checkpointPath:'No policy checkpoint attached')}</code>${node.checkpointSha256?`<p>SHA-256</p><code>${escapeHtml(node.checkpointSha256)}</code>`:''}<div class="evolution-parameters">${keyValueGrid(node.metrics,'No metrics recorded.')}</div></details></section></div>`;
  }

  function parentsOf(nodes, id) { return nodes.get(id)?.parentIds || []; }

  function ancestorSet(nodes, id) {
    const result = new Set();
    const queue = [id];
    while (queue.length) {
      const current = queue.shift();
      if (!current || result.has(current)) continue;
      result.add(current);
      queue.push(...parentsOf(nodes, current));
    }
    return result;
  }

  const nodeColumnWidth=112;

  // Milestone bands share coordinates with their nodes; branches reuse rows.
  function layout(visible, milestones = [], availableHeight = 364) {
    const byId = new Map(visible.map(n => [n.id, n]));
    const stages = [{id: '', label: 'Experiments', order: '·', status: 'diagnostic'}, ...milestones];
    const stageIndex = new Map(stages.map((m, i) => [m.id, i]));
    const positions = new Map(), visiting = new Set();
    const column = node => {
      if (positions.has(node.id)) return positions.get(node.id);
      const stage = stageIndex.get(node.milestoneId) || 0;
      if (visiting.has(node.id)) return {stage, depth: 0};
      visiting.add(node.id);
      const parents = (node.parentIds || []).map(id => byId.get(id)).filter(Boolean).map(column);
      const depth = Math.max(0, ...parents.filter(p => p.stage === stage).map(p => p.depth + 1));
      const result = {stage, depth}; positions.set(node.id, result); visiting.delete(node.id); return result;
    };
    visible.forEach(column);
    const occupied = new Map();
    // Three lanes keep the graph inside a laptop viewport; overflow grows rightward.
    const assigned = new Map();
    [...visible].sort((a,b)=>positions.get(a.id).depth-positions.get(b.id).depth).forEach(node => {
      const p = positions.get(node.id);
      const parentColumns = (node.parentIds || []).map(id => assigned.get(id)).filter(q => q?.stage === p.stage).map(q => q.column + 1);
      let column = Math.max(p.depth, 0, ...parentColumns), key = `${p.stage}:${column}`;
      while ((occupied.get(key) || 0) >= 3) { column++; key = `${p.stage}:${column}`; }
      const row = occupied.get(key) || 0; occupied.set(key, row + 1);
      assigned.set(node.id, {stage:p.stage,column,row});
    });
    let offset = 16;
    const bands = stages.map((stage,index) => {
      const columns = Math.max(1,...[...assigned.values()].filter(p=>p.stage===index).map(p=>p.column+1));
      const band={...stage,x:offset,width:columns*nodeColumnWidth};offset+=band.width;return band;
    });
    const height=Math.max(300,availableHeight);
    const points=new Map([...assigned].map(([id,p])=>[id,{x:bands[p.stage].x+(p.column+.5)*nodeColumnWidth-42,y:(p.row+.5)*height/3-42.5}]));
    return {width:offset+16,height,bands,point:id=>points.get(id)};
  }

  function progressLabel(node) {
    const progress = node.trainingProgress || {};
    if (progress.kind === 'merged') return `${progress.runCount || node.collapsedCount || 0} runs · seq ${progress.fromSequence ?? '—'}–${progress.toSequence ?? '—'}`;
    if (progress.kind === 'training') {
      const iterations = progress.completedIterations == null ? '—' : progress.requestedIterations == null ? progress.completedIterations : `${progress.completedIterations}/${progress.requestedIterations}`;
      return `iter ${iterations} · ${compactNumber(progress.sampleCount)} samples`;
    }
    if (progress.kind === 'diagnostic') {
      const steps = progress.physicsSteps ?? progress.controlSteps;
      const kind = progress.physicsSteps != null ? 'physics steps' : 'control steps';
      return steps == null ? `${formatNumber(progress.durationSeconds)} s diagnostic` : `${compactNumber(steps)} ${kind} · ${formatNumber(progress.durationSeconds)} s`;
    }
    return node.kind === 'root' || node.kind === 'milestone' ? 'milestone gate' : `run sequence ${node.step ?? '—'}`;
  }

  function outcomeLabel(node, primaryMetric, unit = 'm') {
    if (node.collapsedCount) return `${node.collapsedCount} runs merged`;
    if (node.trainingProgress) return progressLabel(node);
    const value = node.metrics?.[primaryMetric];
    if (value != null) return `${formatNumber(value)} ${unit}`;
    const falls = node.metrics?.fall_count;
    const resets = node.metrics?.reset_count;
    if (falls != null || resets != null) return `${falls ?? '—'} falls · ${resets ?? '—'} resets`;
    return node.status === 'running' ? 'current gate' : node.status;
  }

  function keyValueGrid(values, empty) {
    const rows = Object.entries(values || {}).map(([key, value]) => `<div><span>${escapeHtml(key)}</span><b>${escapeHtml(typeof value === 'number' ? formatNumber(value) : typeof value === 'object' ? JSON.stringify(value) : value)}</b></div>`).join('');
    return rows || `<p class="evolution-muted">${escapeHtml(empty)}</p>`;
  }

  async function inspectJoints(root, node, robot, onExpand) {
    const path = node.urdfPath || robot?.urdfPath;
    if (!path) { root.innerHTML = '<p class="evolution-empty">No verified URDF mapping for this record.</p>'; return; }
    root.classList.add('joint-inspector-body');
    const actions = node.actionJoints || [], mappingKnown = Array.isArray(node.actionJoints);
    root.innerHTML = `<div class="joint-legend"><span class="policy-dot">Policy action</span><span class="held-dot">Held / outside policy</span><span class="unknown-dot">Unknown</span></div><div class="lineage-robot-stage"><canvas data-mesh></canvas><canvas data-overlay></canvas><button class="viewport-expand" data-expand-viewer aria-label="Expand URDF inspector" title="Expand URDF inspector">${icon('expand')}</button><div class="viewer-load-overlay" data-urdf-loading role="status"><span class="loading-spinner"></span><b data-urdf-progress>Reading joint definitions…</b></div><span data-robot-status>Loading URDF…</span></div><div class="lineage-joint-list"><p>${mappingKnown ? `${actions.length} policy joints · exact run metadata` : 'Policy joint mapping not recorded for this run.'}</p><div class="lineage-joint-tools"><button class="button button-light" data-reset-joints>Reset pose</button><span>rad · preview only</span></div><p data-joint-hint>Move a slider to highlight its moving links in amber.</p><div data-joints></div></div>`;
    let viewer;
    try {
      root.querySelector('[data-expand-viewer]').addEventListener('click',onExpand);
      const payload = await window.GeoAssets.json('/api/robot/urdf?path=' + encodeURIComponent(path),{onProgress:p=>{
        if(root.isConnected)root.querySelector('[data-urdf-progress]').textContent='Joint definitions · '+window.GeoAssets.describe(p);
      }}); if (!root.isConnected) return;
      viewer = new window.MotionUrdfViewer({meshCanvas: root.querySelector('[data-mesh]'), overlayCanvas: root.querySelector('[data-overlay]'), assetUrl: '/api/robot/asset?urdf=' + encodeURIComponent(path), onStatus: text => { if(root.isConnected){root.querySelector('[data-robot-status]').textContent=text;root.querySelector('[data-urdf-progress]').textContent=text;} }});
      root._dispose = () => viewer.destroy(); viewer.showInspection = true;
      viewer.trainingJoints = mappingKnown ? new Set(actions) : null;
      const recorded=node.defaultJointPositions||{};
      const defaults=new Map(payload.joints.map(j=>[j.name,Math.max(j.lower,Math.min(j.upper,Number.isFinite(recorded[j.name])?recorded[j.name]:0))]));
      const row=j=>`<div class="lineage-joint ${mappingKnown ? actions.includes(j.name) ? 'policy' : 'held' : 'unknown'}" data-name="${escapeHtml(j.name)}"><button class="joint-select" type="button"><i></i><span>${escapeHtml(j.name.replace(/_joint$/, ''))}</span><small>${mappingKnown ? actions.includes(j.name) ? 'action' : 'hold' : '?'}</small></button><div class="joint-adjust"><input type="range" min="${j.lower}" max="${j.upper}" step="0.005" value="${defaults.get(j.name)}" data-angle aria-label="${escapeHtml(j.name)} angle" disabled><input type="number" min="${j.lower}" max="${j.upper}" step="0.005" value="${defaults.get(j.name).toFixed(3)}" data-angle-number aria-label="${escapeHtml(j.name)} radians" disabled></div></div>`;
      const policy=actions.map(name=>payload.joints.find(j=>j.name===name)).filter(Boolean),groups=new Map();
      for(const joint of payload.joints.filter(j=>!actions.includes(j.name))){
        const group=/(thumb|index|middle|ring|pinky|finger)/.test(joint.name)?(joint.group?.startsWith('left')?'Left hand':'Right hand'):/neck|head/.test(joint.name)?'Head & neck':joint.groupLabel||'Other joints';
        if(!groups.has(group))groups.set(group,[]);groups.get(group).push(joint);
      }
      root.querySelector('[data-joints]').innerHTML=`<div class="training-joint-controls">${policy.length?`<h4>Training joints <span>${policy.length}</span></h4>${policy.map(row).join('')}`:`<p>${mappingKnown?'No policy-controlled joints in this run.':'Training joint mapping is unavailable.'}</p>`}</div><div class="folded-joint-controls"><h4>${mappingKnown?'Outside policy':'Other joints'}</h4>${[...groups].map(([name,joints])=>`<details class="held-joint-group"><summary><span>${escapeHtml(name)}</span><b>${joints.length}</b>${joints.every(j=>Number.isFinite(recorded[j.name]))?'<small>defaults applied</small>':''}</summary><div>${joints.map(row).join('')}</div></details>`).join('')}</div>`;
      const hasDefaults=Object.keys(recorded).length>0;
      root.querySelector('[data-reset-joints]').textContent=hasDefaults?'Reset training pose':'Reset pose';
      const note=document.createElement('p');note.className='joint-default-note';
      const missingDefaults=payload.joints.filter(j=>!Number.isFinite(recorded[j.name])).length;
      note.textContent=hasDefaults?`Recorded training pose${node.bodySettings?.hands==='closed_fists_v1'?' · closed fists':''}${missingDefaults?` · ${missingDefaults} unrecorded joints shown at zero`:''}`:'Training defaults not recorded; preview starts at zero.';
      root.querySelector('[data-joint-hint]').before(note);
      const contactPanel=document.createElement('pre');contactPanel.className='joint-contact-properties';contactPanel.hidden=true;
      root.querySelector('[data-joint-hint]').after(contactPanel);
      const stage = root.querySelector('.lineage-robot-stage'); let pointer;
      const resize = new ResizeObserver(() => { viewer.fitPending = true; viewer.draw(); });
      resize.observe(stage);
      root._dispose = () => { resize.disconnect(); viewer.destroy(); };
      stage.onpointerdown = e => { if(e.target.closest('button'))return; pointer = [e.clientX, e.clientY]; stage.setPointerCapture(e.pointerId); };
      stage.onpointermove = e => { if (!pointer) return; viewer.orbit(e.clientX-pointer[0], e.clientY-pointer[1]); pointer=[e.clientX,e.clientY]; };
      stage.onpointerup = stage.onpointercancel = () => { pointer=null; };
      stage.addEventListener('wheel', e => {e.preventDefault(); viewer.dolly(e.deltaY);}, {passive:false});
      const selectJoint = row => {
        viewer.selectJoint(row.dataset.name);
        root.querySelectorAll('[data-name]').forEach(item=>item.classList.toggle('selected',item===row));
        const frame=viewer.jointFrame(row.dataset.name);
        root.querySelector('[data-joint-hint]').textContent=frame ? `${frame.parent} → ${frame.child} · amber = moving links · purple = joint axis` : 'Loading joint frame…';
        const contact=robot?.contactProperties;
        contactPanel.hidden=!contact||!/ankle|toe/.test(row.dataset.name);
        if(!contactPanel.hidden)contactPanel.textContent=[contact.backend,
          'Foot friction: '+contact.footFriction.join(' / '),'Ground friction: '+contact.groundFriction.join(' / '),
          'Order: sliding / torsional / rolling','Contact dimension: '+contact.condim+' · '+contact.activeFriction,
          'Mixing: '+contact.combineRule,'Solver solref: '+contact.solref.join(' / '),
          'Solver solimp: '+contact.solimp.join(' / '),contact.footModel].join('\n');
      };
      root.querySelectorAll('[data-name]').forEach(row => {
        row.querySelector('.joint-select').onclick=()=>selectJoint(row);
        const slider=row.querySelector('[data-angle]'), number=row.querySelector('[data-angle-number]');
        const change=input=>{
          if(!Number.isFinite(input.valueAsNumber))return;
          const value=Math.max(Number(slider.min),Math.min(Number(slider.max),input.valueAsNumber));
          slider.value=String(value);number.value=value.toFixed(3);
          viewer.setJoints([row.dataset.name],[value],false);selectJoint(row);
          root.dataset.changedJoint=row.dataset.name;root.dataset.jointAngle=String(value);
        };
        slider.oninput=()=>change(slider);number.oninput=()=>change(number);
        slider.onfocus=()=>selectJoint(row);
      });
      root.querySelector('[data-reset-joints]').onclick=()=>{
        if(defaults.size)viewer.setJoints([...defaults.keys()],[...defaults.values()]);
        root.querySelectorAll('[data-name]').forEach(row=>{const value=defaults.get(row.dataset.name);row.querySelector('[data-angle]').value=String(value);row.querySelector('[data-angle-number]').value=value.toFixed(3);});
        root.dataset.jointAngle=String(defaults.get(root.dataset.changedJoint)||0);
      };
      await viewer.load('/api/robot/urdf?path='+encodeURIComponent(path), [], []);
      if(!root.isConnected)return;
      if(defaults.size)viewer.setJoints([...defaults.keys()],[...defaults.values()]);
      for(const section of root.querySelectorAll('.training-joint-controls,.held-joint-group>div')){
        const rows=[...section.children].filter(element=>element.matches('[data-name]'));
        const joints=rows.map(element=>payload.joints.find(j=>j.name===element.dataset.name));
        for(const cluster of viewer.groupJointPivots(joints)){
          if(cluster.joints.length<2)continue;
          const members=cluster.joints.map(j=>rows.find(element=>element.dataset.name===j.name));
          const group=document.createElement('fieldset');group.className='joint-pivot-group';
          const legend=document.createElement('legend');legend.textContent=`Shared pivot · ${members.length} axes`;
          group.append(legend);members[0].before(group);members.forEach(member=>group.append(member));
        }
      }
      root.dataset.trainingDefaults=String(hasDefaults);
      if(root.isConnected){root.querySelectorAll('[data-angle],[data-angle-number]').forEach(input=>input.disabled=false);root.dataset.jointsReady='true';root.querySelector('[data-urdf-loading]').hidden=true;}

    } catch (error) { if(root.isConnected){root.querySelector('[data-robot-status]').textContent=error.message;root.querySelector('[data-urdf-loading]').hidden=true;} }
  }

  function expandInspector(container, body, title, trigger) {
    if (container._closeExpanded) return;
    const placeholder = document.createComment('inspector position');
    body.before(placeholder);
    const dialog = document.createElement('dialog');
    dialog.className = 'lineage-viewer-dialog';
    dialog.setAttribute('aria-label', title);
    dialog.innerHTML = `<header><h2>${escapeHtml(title)}</h2><button class="button button-light" data-close-expanded aria-label="Close expanded viewer">Close ✕</button></header>`;
    container.append(dialog);
    dialog.append(body);
    let restored = false;
    const restore = () => {
      if (restored) return;
      restored = true;
      placeholder.replaceWith(body);
      dialog.close();
      dialog.remove();
      container._closeExpanded = null;
      trigger?.focus({preventScroll:true});
    };
    container._closeExpanded = restore;
    dialog.querySelector('[data-close-expanded]').onclick = restore;
    dialog.addEventListener('cancel', event => { event.preventDefault(); restore(); });
    dialog.addEventListener('close', restore);
    dialog.addEventListener('click', event => { if(event.target === dialog) restore(); });
    dialog.showModal();
  }

  function destroy(container) { container._disposeGraph?.(); container._disposeGraph=null; container._closePicker?.(); container._closeExpanded?.(); container.querySelector('[data-inspector-body]')?._dispose?.(); }

  const workspaceStateKey='geo.landau.workspace.v1';
  function restoredWorkspace() {
    try {
      const saved=JSON.parse(localStorage.getItem(workspaceStateKey)||'{}'),state={};
      for(const key of ['modelId','selectedId','search'])if(typeof saved?.[key]==='string')state[key]=saved[key];
      if(['details','playback','joints'].includes(saved?.tab))state.tab=saved.tab;
      if(['overview','all'].includes(saved?.mode))state.mode=saved.mode;
      state.autoLoadPlayback=saved?.autoLoadPlayback===true;
      return state;
    } catch { return {}; }
  }
  function saveWorkspace(model) {
    try {localStorage.setItem(workspaceStateKey,JSON.stringify(model));} catch { /* Storage may be disabled. */ }
  }

  function mount(container, options) {
    const previousScroll=container.querySelector('.lineage-scroll');
    const scrollPosition=previousScroll?{left:previousScroll.scrollLeft,top:previousScroll.scrollTop}:null;
    const previousModelId=container._renderedModelId;
    destroy(container);
    const data = options.data || {}, raw = data.nodes || [], models = data.models || [];
    const model = container._evolutionModel || {modelId: data.activeModelId || raw[0]?.modelId, selectedId: data.currentNodeId, tab: 'details', mode: 'overview', search: '',...restoredWorkspace()};
    if(models.length&&!models.some(item=>item.id===model.modelId))model.modelId=data.activeModelId||models[0].id;
    container._evolutionModel = model;
    container._renderedModelId = model.modelId;
    const robot = models.find(m => m.id === model.modelId);
    const scoped = raw.filter(n => !models.length || n.modelId === model.modelId);
    const nodes = new Map(scoped.map(n => [n.id, n]));
    if (!nodes.has(model.selectedId)) model.selectedId = scoped.at(-1)?.id;
    saveWorkspace(model);
    const selected = nodes.get(model.selectedId);
    const ancestry = ancestorSet(nodes, model.selectedId);
    let visible = scoped;
    if (model.mode !== 'all') {
      const recent = [...scoped].sort((a,b)=>String(b.startedAt||'').localeCompare(String(a.startedAt||'')));
      const ids = new Set([...ancestry, ...scoped.filter(n=>n.kind==='milestone').map(n=>n.id), ...recent.slice(0,12).map(n=>n.id)]);
      visible = scoped.filter(n=>ids.has(n.id));
    }
    if (model.search) visible = scoped.filter(n => (n.label+' '+n.id+' '+(n.result||'')).toLowerCase().includes(model.search.toLowerCase()));
    const milestones = robot?.milestones?.length ? robot.milestones : (data.milestones || []).map(m=>({...m,status:'unverified'}));
    let graph = layout(visible, milestones);
    const visibleIds = new Set(visible.map(n=>n.id));
    const connections=visible.flatMap(n=>(n.parentIds||[]).filter(id=>visibleIds.has(id)).map(id=>({parent:id,child:n.id})));
    const edgePath=({parent,child})=>{
      const a=graph.point(parent),b=graph.point(child);
      return Math.abs(b.x-a.x)<nodeColumnWidth*1.5?`M ${a.x+60} ${a.y+26} C ${a.x+84} ${a.y+26}, ${b.x} ${b.y+26}, ${b.x+24} ${b.y+26}`:`M ${a.x+60} ${a.y+26} Q ${a.x+80} ${a.y+26} ${a.x+80} ${a.y-8} L ${b.x+3} ${a.y-8} Q ${b.x+3} ${b.y+26} ${b.x+24} ${b.y+26}`;
    };
    const edges=connections.map(edge=>`<path class="evolution-edge${ancestry.has(edge.parent)&&ancestry.has(edge.child)?' active':''}" d="${edgePath(edge)}"/>`).join('');
    const cards = visible.map(n=>{
      const p=graph.point(n.id);
      const statusLabel=n.kind==='milestone'&&n.status==='completed'?'certified':n.status;
      const statusWidth=Math.max(44,statusLabel.length*4.9+14);
      const symbol=n.kind==='milestone'?(n.status==='completed'?'✓':'◇'):n.kind==='checkpoint'?'⬡':n.playback?'▶':'•';
      return `<g class="evolution-node ${escapeHtml(n.status)}${n.id===model.selectedId?' selected':''}${ancestry.has(n.id)?' ancestor':''}" transform="translate(${p.x} ${p.y})" role="button" tabindex="0" data-node="${escapeHtml(n.id)}" aria-label="${escapeHtml(n.label)}"><title>${escapeHtml(n.label+' — '+(n.result||''))}</title><rect class="node-hit" x="-10" y="3" width="104" height="82" rx="8"/><circle class="node-disc" cx="42" cy="26" r="18"/><text class="node-symbol" x="42" y="31" text-anchor="middle">${symbol}</text><circle class="node-state-dot" cx="57" cy="12" r="4"/><text class="node-label" x="42" y="61" text-anchor="middle">${escapeHtml(displayLabel(n).length>20?displayLabel(n).slice(0,19)+'…':displayLabel(n))}</text><rect class="node-status-badge" x="${42-statusWidth/2}" y="65" width="${statusWidth}" height="16" rx="8"/><text class="node-metric" x="42" y="76" text-anchor="middle">${escapeHtml(statusLabel)}</text></g>`;
    }).join('');
    const bands = graph.bands.map((b,i)=>`<div class="lineage-band ${escapeHtml(b.status)}" style="left:${b.x}px;width:${b.width}px;--band-hue:${[205,155,175,35,26,270,290,310,210,225,95,65,0][i%13]}"><span>${escapeHtml(b.id?(b.displayLabel || 'M'+b.order):'LAB')}</span><b>${escapeHtml(b.label)}</b><small>${escapeHtml(b.status.replaceAll('_',' '))}</small></div>`).join('');
    const video = selected?.artifacts?.find(a=>a.kind==='video');
    container.innerHTML = `<div class="lineage-workspace"><div class="lineage-toolbar"><button class="model-picker-trigger" data-model-picker aria-haspopup="dialog">${icon('model')}<span><small>MODEL</small><b>${escapeHtml(modelLabel(robot))}</b></span>${icon('chevron')}</button><span class="lineage-count">${visible.length} / ${scoped.length} nodes${data.speedGoal && model.modelId===data.activeModelId ? ` · Final walking target ${escapeHtml(data.speedGoal.targetSpeedMps)} m/s` : ''}</span><input type="search" data-search placeholder="Find a run…" aria-label="Search lineage" value="${escapeHtml(model.search)}"><button class="button button-light" data-mode>${model.mode==='all'?'Compact':'All runs'}</button><button class="button button-light" data-fit>Locate</button>${options.onRefresh?'<button class="button button-light lineage-icon-button" data-refresh aria-label="Refresh lineage" title="Refresh">↻</button>':''}${options.onTools?'<button class="button button-light lineage-icon-button" data-tools aria-label="Tools and logs" title="Tools and logs">⋯</button>':''}</div><div class="lineage-body"><section class="lineage-graph"><div class="lineage-scroll"><div class="lineage-band-header" style="width:${graph.width}px">${bands}</div><svg width="${graph.width}" height="${graph.height}" role="group" aria-label="Checkpoint evolution by milestone">${graph.bands.map((b,i)=>`<rect x="${b.x}" width="${b.width}" height="${graph.height}" fill="${i%2?'#eaf1ec':'#f5f6f1'}"/>`).join('')}<g>${edges}</g><g>${cards}</g></svg></div><footer class="lineage-footer"><span>● certified gate &nbsp; ◌ experiment &nbsp; ▶ recorded 3D</span><span>Columns = target gate · only certified nodes prove a pass</span></footer></section><aside class="lineage-inspector"><header><p class="eyebrow">SELECTED NODE</p><h3>${escapeHtml(displayLabel(selected))}</h3></header><nav class="lineage-tabs" aria-label="Node inspector">${['details','playback','joints'].map(tab=>`<button data-tab="${tab}" class="${model.tab===tab?'active':''}" aria-pressed="${model.tab===tab}">${{details:'Details',playback:'Playback',joints:'Joints'}[tab]}</button>`).join('')}</nav><div data-inspector-body></div></aside></div></div>`;
    // Fit the hover target around the icon and its actual caption, centered together.
    container.querySelectorAll('[data-node]').forEach(node=>{
      const width=Math.max(52,node.querySelector('.node-label').getBBox().width+12,node.querySelector('.node-status-badge').width.baseVal.value+12);
      const hit=node.querySelector('.node-hit');hit.setAttribute('x',42-width/2);hit.setAttribute('width',width);
    });
    const body=container.querySelector('[data-inspector-body]');
    const onExpand=event=>expandInspector(container,body,`${model.tab==='joints'?'URDF inspector':'3D playback'} · ${displayLabel(selected)}`,event.currentTarget);
    if(model.tab==='details') {
      body.innerHTML=detailsPane(selected,robot,nodes,milestones);
      body.querySelectorAll('[data-select-parent]').forEach(button=>button.onclick=()=>{model.selectedId=button.dataset.selectParent;mount(container,options);});
      body.querySelectorAll('[data-detail-artifact]').forEach(button=>button.onclick=()=>options.onPreview(button.dataset.detailArtifact,button.dataset.kind));
    } else if(model.tab==='joints') { if(selected) void inspectJoints(body,selected,robot,onExpand); }
    else {
      body.classList.add('playback-inspector-body');
      body.innerHTML=`<div class="lineage-playback">${selected?.playback?'<button class="button" data-load-motion>Load 3D playback</button><p>Downloads only this node’s recorded motion.</p><div data-motion></div>':'<p>No 3D recording attached to this node.</p>'}${video?`<button class="button button-light" data-video>Open proof video</button>`:''}<p class="muted">Playback belongs to the selected run. A recording alone is not a milestone certificate.</p></div>`;
      body.querySelector('[data-video]')?.addEventListener('click',()=>options.onPreview(video.path,'video'));
      const loadPlayback=async()=>{
        const button=body.querySelector('[data-load-motion]');
        if(!button||button.disabled)return;
        model.autoLoadPlayback=true;saveWorkspace(model);
        button.disabled=true;button.textContent='Loading…';
        try { const module=await import('/character_motion_viewer.js?v=13'); if(!body.isConnected)return;
          body._dispose?.();
          const viewer=module.mount(body.querySelector('[data-motion]'), {...selected.playback,onExpand,expandIcon:icon('expand'),onError:()=>{
            if(body.isConnected){button.hidden=false;button.disabled=false;button.textContent='Retry 3D playback';}
          }});body._dispose=()=>viewer.destroy();button.hidden=true;body.querySelector('.lineage-playback').classList.add('is-loaded');
        } catch(error) {if(body.isConnected){const button=body.querySelector('[data-load-motion]');button.disabled=false;button.textContent='Retry 3D · '+error.message;}}
      };
      body.querySelector('[data-load-motion]')?.addEventListener('click',loadPlayback);
      if(model.autoLoadPlayback)void loadPlayback();
    }
    const redraw=()=>mount(container,options);
    container.querySelector('[data-model-picker]').onclick=event=>modelPicker(container,models,raw,model.modelId,id=>{model.modelId=id;model.selectedId='';model.search='';redraw();},event.currentTarget);
    container.querySelector('[data-mode]').onclick=()=>{model.mode=model.mode==='all'?'overview':'all';redraw();};
    const search=container.querySelector('[data-search]');search.onchange=()=>{model.search=search.value;redraw();};
    container.querySelectorAll('[data-tab]').forEach(b=>b.onclick=()=>{model.tab=b.dataset.tab;redraw();});
    const select=id=>{model.selectedId=id;redraw();};
    container.querySelectorAll('[data-node]').forEach(b=>{b.onclick=()=>select(b.dataset.node);b.onkeydown=e=>{const id=b.dataset.node,index=visible.findIndex(n=>n.id===id);let next;
      if(e.key==='Enter'||e.key===' ')next=id;
      if(e.key==='ArrowLeft')next=(nodes.get(id)?.parentIds||[]).find(parent=>visibleIds.has(parent));
      if(e.key==='ArrowRight')next=visible.find(n=>(n.parentIds||[]).includes(id))?.id;
      if(e.key==='ArrowUp')next=visible[Math.max(0,index-1)]?.id;
      if(e.key==='ArrowDown')next=visible[Math.min(visible.length-1,index+1)]?.id;
      if(next){e.preventDefault();select(next);queueMicrotask(()=>container.querySelector(`[data-node="${CSS.escape(next)}"]`)?.focus({preventScroll:true}));}};});
    const scroll=container.querySelector('.lineage-scroll'),svg=scroll.querySelector('svg');
    const fitGraph=()=>{
      const height=Math.max(300,Math.floor(scroll.clientHeight-scroll.querySelector('.lineage-band-header').offsetHeight));
      if(graph.height===height)return;
      graph=layout(visible,milestones,height);
      svg.setAttribute('height',height);
      svg.querySelectorAll(':scope > rect').forEach(rect=>rect.setAttribute('height',height));
      svg.querySelectorAll('[data-node]').forEach(node=>{const p=graph.point(node.dataset.node);node.setAttribute('transform',`translate(${p.x} ${p.y})`);});
      svg.querySelectorAll('.evolution-edge').forEach((edge,i)=>edge.setAttribute('d',edgePath(connections[i])));
    };
    fitGraph();
    const graphResize=new ResizeObserver(fitGraph);graphResize.observe(scroll);
    container._disposeGraph=()=>graphResize.disconnect();
    const locate=()=>{const p=graph.point(model.selectedId);if(p)container.querySelector('.lineage-scroll').scrollTo({left:Math.max(0,p.x-container.querySelector('.lineage-scroll').clientWidth+104),top:0});};
    container.querySelector('[data-fit]').onclick=locate;
    // Selection and inspector changes rebuild the DOM, but must not move the graph.
    if(scrollPosition&&previousModelId===model.modelId)container.querySelector('.lineage-scroll').scrollTo(scrollPosition);
    else locate();
    container.querySelector('[data-refresh]')?.addEventListener('click',options.onRefresh);
    container.querySelector('[data-tools]')?.addEventListener('click',options.onTools);
  }

  window.GeoEvolutionTree = {mount, destroy, layout, icon, loading};
})();
