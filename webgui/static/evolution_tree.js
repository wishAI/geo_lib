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
  const isCertified = node => node.kind === 'milestone' && node.status === 'completed';
  let graphSequence=0;

  // Stable topological columns, with children preferring their parent's lane.
  // Folding changes only the display mapping, never the recorded ancestry.
  function layout(visible, milestones = [], availableHeight = 364, availableWidth = 1000, expanded = new Set()) {
    const byId = new Map(visible.map(n => [n.id, n]));
    const stages = [{id: '', label: 'Experiments', order: '·', status: 'diagnostic'}, ...milestones];
    const stageIndex = new Map(stages.map((m, i) => [m.id, i]));
    const height=Math.max(90,availableHeight), lanes=Math.max(1,Math.floor(height/104));
    const assigned=new Map(), occupied=new Map(), visiting=new Set();
    const assign=node=>{
      if(assigned.has(node.id))return assigned.get(node.id);
      if(visiting.has(node.id))return null;
      visiting.add(node.id);
      const stage=stageIndex.get(node.milestoneId)||0;
      const parents=(node.parentIds||[]).map(id=>byId.get(id)).filter(Boolean).map(assign).filter(Boolean);
      let column=Math.max(0,...parents.filter(p=>p.stage===stage).map(p=>p.column+1));
      const preferred=parents[0]?.row||0;
      let row;
      while(row===undefined){
        const key=`${stage}:${column}`,used=occupied.get(key)||new Set();
        row=Array.from({length:lanes},(_,i)=>i).sort((a,b)=>Math.abs(a-preferred)-Math.abs(b-preferred)||a-b).find(r=>!used.has(r));
        if(row===undefined){column++;continue;}
        used.add(row);occupied.set(key,used);
      }
      const value={stage,column,row};assigned.set(node.id,value);visiting.delete(node.id);return value;
    };
    [...visible].sort((a,b)=>String(a.startedAt||'').localeCompare(String(b.startedAt||''))||a.id.localeCompare(b.id)).forEach(assign);
    const cap=Math.max(1,Math.floor(availableWidth*2/3/nodeColumnWidth));
    const points=new Map(), representatives=new Map(), shown=[], folds=[];
    let offset=0;
    const bands=stages.map((stage,index)=>{
      const members=visible.filter(n=>assigned.get(n.id).stage===index);
      // Certificates are permanent section endpoints, including when later
      // experiments descend from them. Preserve those real edges unchanged.
      const certified=members.filter(isCertified).sort((a,b)=>assigned.get(a.id).column-assigned.get(b.id).column||a.id.localeCompare(b.id));
      const experiments=members.filter(n=>!isCertified(n));
      const experimentColumns=[...new Set(experiments.map(n=>assigned.get(n.id).column))].sort((a,b)=>a-b);
      const columnIndex=new Map(experimentColumns.map((column,i)=>[column,i]));
      const columns=experimentColumns.length, certificateColumns=Math.ceil(certified.length/lanes);
      const budget=Math.max(experiments.length?1:0,cap-certificateColumns);
      const folded=columns>budget&&!expanded.has(stage.id);
      const count=folded?budget:columns, head=folded?Math.floor((count-1)/2):columns, tail=folded?count-1-head:0;
      const band={...stage,x:offset,width:Math.max(1,count+certificateColumns)*nodeColumnWidth,folded};offset+=band.width;
      const hidden=[];
      const point=(column,row)=>({x:band.x+(column+.5)*nodeColumnWidth-42,y:(row+.5)*height/lanes-42.5});
      for(const node of experiments){
        const p=assigned.get(node.id);
        const originalColumn=columnIndex.get(p.column);
        if(folded && originalColumn>=head && originalColumn<columns-tail){hidden.push(node);continue;}
        const column=folded&&originalColumn>=columns-tail?head+1+originalColumn-(columns-tail):originalColumn;
        points.set(node.id,point(column,p.row));representatives.set(node.id,node.id);shown.push(node);
      }
      const certificateRows=new Map();
      certified.forEach((node,i)=>{
        const column=count+Math.floor(i/lanes),preferred=assigned.get(node.id).row;
        const used=certificateRows.get(column)||new Set();
        const row=Array.from({length:lanes},(_,r)=>r).sort((a,b)=>Math.abs(a-preferred)-Math.abs(b-preferred)||a-b).find(r=>!used.has(r));
        used.add(row);certificateRows.set(column,used);
        points.set(node.id,point(column,row));representatives.set(node.id,node.id);shown.push(node);
      });
      if(hidden.length){
        const id='__fold__:'+index, fold={id,kind:'fold',status:'folded',label:`${hidden.length} hidden runs`,members:hidden,stageId:stage.id};
        points.set(id,point(head,Math.floor(lanes/2)));representatives.set(id,id);shown.push(fold);folds.push(fold);
        for(const n of hidden){points.set(n.id,points.get(id));representatives.set(n.id,id);}
      }
      return band;
    });
    const seen=new Set(),connections=[];
    for(const n of visible)for(const parent of n.parentIds||[]){
      const a=representatives.get(parent),b=representatives.get(n.id),key=JSON.stringify([a,b]);
      if(!a||!b||a===b||seen.has(key))continue;
      seen.add(key);connections.push({parent:a,child:b});
    }
    // Both ports have rightward tangents: leave the source's right rim and
    // approach the destination's left rim. Only obstructed/backward edges detour.
    const ports=edge=>{
      const a=points.get(edge.parent),b=points.get(edge.child);
      return {x1:a.x+60,y1:a.y+26,x2:b.x+24,y2:b.y+26};
    };
    const curveHandle=({x1,x2})=>Math.min(80,(x2-x1)/2);
    const clearDirect=edge=>{
      const p=ports(edge),{x1,y1,x2,y2}=p;
      if(x2<=x1)return false;
      const handle=curveHandle(p);
      // A forward cubic is monotone in both coordinates. Invert x at each
      // intervening card's sides to test its full y interval, including labels.
      const yAtX=x=>{
        let lo=0,hi=1;
        for(let i=0;i<18;i++){
          const t=(lo+hi)/2,u=1-t;
          const value=u*u*u*x1+3*u*u*t*(x1+handle)+3*u*t*t*(x2-handle)+t*t*t*x2;
          if(value<x)lo=t;else hi=t;
        }
        const t=(lo+hi)/2;return y1+(y2-y1)*t*t*(3-2*t);
      };
      return shown.every(node=>{
        if(node.id===edge.parent||node.id===edge.child)return true;
        const point=points.get(node.id),left=Math.max(x1,point.x-10),right=Math.min(x2,point.x+94);
        if(left>=right)return true;
        const a=yAtX(left),b=yAtX(right);
        return Math.max(a,b)<point.y||Math.min(a,b)>point.y+88;
      });
    };
    const tracks=[];
    for(const edge of connections){
      const a=points.get(edge.parent),b=points.get(edge.child);
      if(clearDirect(edge))continue;
      const left=Math.min(a.x,b.x),right=Math.max(a.x,b.x);
      let track=tracks.findIndex(intervals=>intervals.every(([l,r])=>right<l||left>r));
      if(track<0){track=tracks.length;tracks.push([]);}
      tracks[track].push([left,right]);edge.track=track;
    }
    const edgePath=edge=>{
      const a=points.get(edge.parent),b=points.get(edge.child),p=ports(edge),{x1,y1,x2,y2}=p;
      if(edge.track===undefined){
        if(y1===y2)return `M ${x1} ${y1} L ${x2} ${y2}`;
        const handle=curveHandle(p);
        return `M ${x1} ${y1} C ${x1+handle} ${y1}, ${x2-handle} ${y2}, ${x2} ${y2}`;
      }
      const gutter=4+(edge.track%Math.max(1,Math.floor((height/lanes-85)/3)))*3;
      const y=Math.max(3,Math.min(a.y,b.y)-gutter);
      const right=a.x+94,left=b.x-12,direction=Math.sign(left-right)||1,r=Math.min(9,Math.abs(left-right)/2);
      return `M ${x1} ${y1} L ${right-r} ${y1} Q ${right} ${y1} ${right} ${y1-r} L ${right} ${y+r} Q ${right} ${y} ${right+direction*r} ${y} L ${left-direction*r} ${y} Q ${left} ${y} ${left} ${y+r} L ${left} ${y2-r} Q ${left} ${y2} ${left+r} ${y2} L ${x2} ${y2}`;
    };
    return {width:offset,height,bands,folds,visible:shown,connections,edgePath,point:id=>points.get(id),representative:id=>representatives.get(id)};
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
      await viewer.load('/api/robot/urdf?path='+encodeURIComponent(path), [], [], () => {
      if(!root.isConnected)return;
      for(const section of root.querySelectorAll('.training-joint-controls,.held-joint-group>div')){
        const rows=[...section.children].filter(element=>element.matches('[data-name]'));
        const joints=rows.map(element=>payload.joints.find(j=>j.name===element.dataset.name));
        for(const cluster of viewer.groupJointPivots(joints)){
          if(cluster.joints.length<2)continue;
          const members=cluster.joints.map(j=>rows.find(element=>element.dataset.name===j.name));
          const group=document.createElement('fieldset');group.className='joint-pivot-group';
          const legend=document.createElement('legend');legend.textContent=window.MotionUrdfViewer.pivotLabel(cluster);
          group.append(legend);members[0].before(group);members.forEach(member=>{
            const label=member.querySelector('.joint-select span');label.textContent=window.MotionUrdfViewer.axisLabel(member.dataset.name,cluster);member.title=member.dataset.name;group.append(member);
          });
        }
      }
      if(defaults.size)viewer.setJoints([...defaults.keys()],[...defaults.values()]);
      root.dataset.trainingDefaults=String(hasDefaults);
      if(root.isConnected){root.querySelectorAll('[data-angle],[data-angle-number]').forEach(input=>input.disabled=false);root.dataset.jointsReady='true';}

      });
      if(root.isConnected)root.querySelector('[data-urdf-loading]').hidden=true;

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
    if (model.search) visible = scoped.filter(n => isCertified(n) || (n.label+' '+n.id+' '+(n.result||'')).toLowerCase().includes(model.search.toLowerCase()));
    const milestones = robot?.milestones?.length ? robot.milestones : (data.milestones || []).map(m=>({...m,status:'unverified'}));
    const graphWidth=previousScroll?.clientWidth || Math.max(112,container.clientWidth-350);
    const graphHeight=previousScroll?Math.max(90,previousScroll.clientHeight-49):Math.max(90,container.clientHeight-122);
    const expanded=new Set(model.mode==='all'||model.search?['',...milestones.map(m=>m.id)]:[]);
    let graph = layout(visible, milestones, graphHeight, graphWidth, expanded);
    const visibleIds = new Set(visible.map(n=>n.id));
    const connections=graph.connections;
    const arrowId=`lineage-arrow-${++graphSequence}`;
    const edges=connections.map(edge=>`<path class="evolution-edge${(edge.parent===graph.representative(model.selectedId)||edge.child===graph.representative(model.selectedId))?' active':''}" data-parent="${escapeHtml(edge.parent)}" data-child="${escapeHtml(edge.child)}" marker-end="url(#${arrowId})" d="${graph.edgePath(edge)}"/>`).join('');
    const cards = graph.visible.map(n=>{
      const p=graph.point(n.id);
      const statusLabel=n.kind==='fold'?'show runs':n.kind==='milestone'&&n.status==='completed'?'certified':n.status;
      const statusWidth=Math.max(44,statusLabel.length*4.9+14);
      const symbol=n.kind==='fold'?'⋯':n.kind==='milestone'?(n.status==='completed'?'✓':'◇'):n.kind==='checkpoint'?'⬡':(n.playback||n.hasPlayback)?'▶':'•';
      return `<g class="evolution-node ${escapeHtml(n.status)}${n.id===graph.representative(model.selectedId)?' selected':''}${ancestry.has(n.id)?' ancestor':''}" transform="translate(${p.x} ${p.y})" role="button" tabindex="0" data-node="${escapeHtml(n.id)}" aria-label="${escapeHtml(n.label)}"><title>${escapeHtml(n.label+' — '+(n.result||''))}</title><rect class="node-hit" x="-10" y="3" width="104" height="82" rx="8"/><circle class="node-disc" cx="42" cy="26" r="18"/><text class="node-symbol" x="42" y="31" text-anchor="middle">${symbol}</text><circle class="node-state-dot" cx="57" cy="12" r="4"/><text class="node-label" x="42" y="61" text-anchor="middle">${escapeHtml(displayLabel(n).length>20?displayLabel(n).slice(0,19)+'…':displayLabel(n))}</text><rect class="node-status-badge" x="${42-statusWidth/2}" y="65" width="${statusWidth}" height="16" rx="8"/><text class="node-metric" x="42" y="76" text-anchor="middle">${escapeHtml(statusLabel)}</text></g>`;
    }).join('');
    const bands = graph.bands.map((b,i)=>`<div class="lineage-band ${escapeHtml(b.status)}" style="left:${b.x}px;width:${b.width}px;--band-hue:${[205,155,175,35,26,270,290,310,210,225,95,65,0][i%13]}"><span>${escapeHtml(b.id?(b.displayLabel || 'M'+b.order):'LAB')}</span><b>${escapeHtml(b.label)}</b><small>${escapeHtml(b.status.replaceAll('_',' '))}</small></div>`).join('');
    container.innerHTML = `<div class="lineage-workspace"><div class="lineage-toolbar"><button class="model-picker-trigger" data-model-picker aria-haspopup="dialog">${icon('model')}<span><small>MODEL</small><b>${escapeHtml(modelLabel(robot))}</b></span>${icon('chevron')}</button><span class="lineage-count">${graph.visible.filter(n=>n.kind!=='fold').length} / ${scoped.length} nodes${data.speedGoal && model.modelId===data.activeModelId ? ` · Final walking target ${escapeHtml(data.speedGoal.targetSpeedMps)} m/s` : ''}</span><input type="search" data-search placeholder="Find a run…" aria-label="Search lineage" value="${escapeHtml(model.search)}"><button class="button button-light" data-mode>${model.mode==='all'?'Compact':'All runs'}</button><button class="button button-light" data-fit>Locate</button>${options.onRefresh?'<button class="button button-light lineage-icon-button" data-refresh aria-label="Refresh lineage" title="Refresh">↻</button>':''}${options.onTools?'<button class="button button-light lineage-icon-button" data-tools aria-label="Tools and logs" title="Tools and logs">⋯</button>':''}</div><div class="lineage-body"><section class="lineage-graph"><div class="lineage-scroll"><div class="lineage-band-header" style="width:${graph.width}px">${bands}</div><svg width="${graph.width}" height="${graph.height}" role="group" aria-label="Checkpoint evolution by milestone"><defs><marker id="${arrowId}" viewBox="0 0 7 6" refX="7" refY="3" markerWidth="7" markerHeight="6" markerUnits="userSpaceOnUse" orient="auto"><path d="M 0 0 L 7 3 L 0 6 Z" fill="context-stroke"/></marker></defs>${graph.bands.map((b,i)=>`<rect x="${b.x}" width="${b.width}" height="${graph.height}" fill="${i%2?'#eaf1ec':'#f5f6f1'}"/>`).join('')}<g>${edges}</g><g>${cards}</g></svg></div><footer class="lineage-footer"><span>Wheel to pan · ⋯ show folded runs</span><span>Columns = target gate · only certified nodes prove a pass</span></footer></section><aside class="lineage-inspector"><header><p class="eyebrow">SELECTED NODE</p><h3>${escapeHtml(displayLabel(selected))}</h3></header><nav class="lineage-tabs" aria-label="Node inspector">${['details','playback','joints'].map(tab=>`<button data-tab="${tab}" class="${model.tab===tab?'active':''}" aria-pressed="${model.tab===tab}">${{details:'Details',playback:'Playback',joints:'Joints'}[tab]}</button>`).join('')}</nav><div data-inspector-body></div></aside></div></div>`;
    // Fit the hover target around the icon and its actual caption, centered together.
    container.querySelectorAll('[data-node]').forEach(node=>{
      const width=Math.max(52,node.querySelector('.node-label').getBBox().width+12,node.querySelector('.node-status-badge').width.baseVal.value+12);
      const hit=node.querySelector('.node-hit');hit.setAttribute('x',42-width/2);hit.setAttribute('width',width);
    });
    const renderInspector=(selected,body)=>{
      body._dispose?.();body._dispose=null;
      body.className='';body.innerHTML='';
      const video=selected?.artifacts?.find(a=>a.kind==='video');
      const onExpand=event=>expandInspector(container,body,`${model.tab==='joints'?'URDF inspector':'3D playback'} · ${displayLabel(selected)}`,event.currentTarget);
      if(model.tab==='details') {
        body.innerHTML=detailsPane(selected,robot,nodes,milestones);
        body.querySelectorAll('[data-select-parent]').forEach(button=>button.onclick=()=>select(button.dataset.selectParent));
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
    };
    let inspectorRequest=0;
    const loadInspector=async()=>{
      const request=++inspectorRequest, id=model.selectedId;
      const previous=container.querySelector('[data-inspector-body]');
      previous._dispose?.();
      const body=document.createElement('div');body.dataset.inspectorBody='';previous.replaceWith(body);
      if(!data.summaryOnly || !options.loadNode || !id){renderInspector(nodes.get(id),body);return;}
      body._dispose?.();body._dispose=null;body.className='';
      body.innerHTML=loading('Loading selected run…');
      try {
        const detail=await options.loadNode(id,p=>{
          if(body.isConnected&&request===inspectorRequest)body.innerHTML=loading(p.state==='connecting'?'Connecting to run details…':'Run details · '+window.GeoAssets.describe(p));
        });
        if(body.isConnected&&request===inspectorRequest)renderInspector(detail,body);
      } catch(error){if(body.isConnected&&request===inspectorRequest){
        body.innerHTML=`<p class="evolution-empty">${escapeHtml(error.message)}</p><button class="button" data-retry-node>Retry details</button>`;
        body.querySelector('[data-retry-node]').onclick=loadInspector;
      }}
    };
    const redraw=()=>mount(container,options);
    container.querySelector('[data-model-picker]').onclick=event=>modelPicker(container,models,raw,model.modelId,id=>{model.modelId=id;model.selectedId='';model.search='';redraw();},event.currentTarget);
    container.querySelector('[data-mode]').onclick=()=>{model.mode=model.mode==='all'?'overview':'all';redraw();};
    const search=container.querySelector('[data-search]');search.onchange=()=>{model.search=search.value;redraw();};
    container.querySelectorAll('[data-tab]').forEach(b=>b.onclick=()=>{
      model.tab=b.dataset.tab;saveWorkspace(model);
      container.querySelectorAll('[data-tab]').forEach(tab=>{const active=tab===b;tab.classList.toggle('active',active);tab.setAttribute('aria-pressed',String(active));});
      void loadInspector();
    });
    const select=id=>{
      const fold=graph.folds.find(n=>n.id===id);
      if(fold){
        const dialog=document.createElement('dialog');dialog.className='lineage-fold-dialog';
        dialog.innerHTML=`<header><b>${fold.members.length} folded runs</b><button class="button" data-close>Close</button></header><div>${fold.members.map(n=>`<button data-fold-node="${escapeHtml(n.id)}"><span>${escapeHtml(displayLabel(n))}</span><small>${escapeHtml(n.status)}</small></button>`).join('')}</div>`;
        container.append(dialog);dialog.querySelector('[data-close]').onclick=()=>dialog.close();
        dialog.onclose=()=>dialog.remove();dialog.showModal();
        dialog.querySelectorAll('[data-fold-node]').forEach(button=>button.onclick=()=>{dialog.close();select(button.dataset.foldNode);});return;
      }
      model.selectedId=id;saveWorkspace(model);
      const ancestors=ancestorSet(nodes,id);
      container.querySelector('.lineage-inspector h3').textContent=displayLabel(nodes.get(id));
      svg.querySelectorAll('[data-node]').forEach(el=>{
        el.classList.toggle('selected',el.dataset.node===graph.representative(id));
        el.classList.toggle('ancestor',ancestors.has(el.dataset.node));
      });
      svg.querySelectorAll('.evolution-edge').forEach((el,i)=>el.classList.toggle('active',connections[i].parent===graph.representative(id)||connections[i].child===graph.representative(id)));
      void loadInspector();
    };
    container.querySelectorAll('[data-node]').forEach(b=>{b.onclick=()=>select(b.dataset.node);b.onkeydown=e=>{const id=b.dataset.node,index=visible.findIndex(n=>n.id===id);let next;
      if(e.key==='Enter'||e.key===' ')next=id;
      if(e.key==='ArrowLeft')next=(nodes.get(id)?.parentIds||[]).find(parent=>visibleIds.has(parent));
      if(e.key==='ArrowRight')next=visible.find(n=>(n.parentIds||[]).includes(id))?.id;
      if(e.key==='ArrowUp')next=visible[Math.max(0,index-1)]?.id;
      if(e.key==='ArrowDown')next=visible[Math.min(visible.length-1,index+1)]?.id;
      if(next){e.preventDefault();select(next);queueMicrotask(()=>container.querySelector(`[data-node="${CSS.escape(next)}"]`)?.focus({preventScroll:true}));}};});
    const scroll=container.querySelector('.lineage-scroll'),svg=scroll.querySelector('svg');
    let resizeFrame;
    const graphResize=new ResizeObserver(()=>{
      const height=Math.max(90,Math.floor(scroll.clientHeight-scroll.querySelector('.lineage-band-header').offsetHeight));
      if(Math.abs(height-graph.height)<2 && Math.abs(scroll.clientWidth-graphWidth)<2)return;
      cancelAnimationFrame(resizeFrame);
      resizeFrame=requestAnimationFrame(()=>mount(container,options));
    });graphResize.observe(scroll);
    container._disposeGraph=()=>{graphResize.disconnect();cancelAnimationFrame(resizeFrame);};
    scroll.addEventListener('wheel',event=>{
      if(event.ctrlKey||event.metaKey)return;
      const delta=(Math.abs(event.deltaX)>Math.abs(event.deltaY)?event.deltaX:event.deltaY)*(event.deltaMode===1?16:event.deltaMode===2?scroll.clientWidth:1);
      if(scroll.scrollWidth<=scroll.clientWidth)return;
      event.preventDefault();scroll.scrollLeft+=delta;
    },{passive:false});
    void loadInspector();
    const locate=()=>{const p=graph.point(model.selectedId);if(p)container.querySelector('.lineage-scroll').scrollTo({left:Math.max(0,p.x-container.querySelector('.lineage-scroll').clientWidth+104),top:0});};
    container.querySelector('[data-fit]').onclick=locate;
    // Explicit mode and viewport changes preserve the horizontal position.
    if(scrollPosition&&previousModelId===model.modelId)container.querySelector('.lineage-scroll').scrollTo(scrollPosition);
    else locate();
    container.querySelector('[data-refresh]')?.addEventListener('click',options.onRefresh);
    container.querySelector('[data-tools]')?.addEventListener('click',options.onTools);
  }

  window.GeoEvolutionTree = {mount, destroy, layout, icon, loading};
})();
