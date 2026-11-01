(function () {
  'use strict';

  const app = document.querySelector('#app');
  const statusRack = document.querySelector('#status-rack');
  const storageDrawer = document.querySelector('#storage-drawer');
  const storageContent = document.querySelector('#storage-content');
  const artifactDialog = document.querySelector('#artifact-dialog');
  const artifactPreview = document.querySelector('#artifact-preview');
  const urdfDialog = document.querySelector('#urdf-dialog');
  const urdfDialogContent = document.querySelector('#urdf-dialog-content');
  const meshDialog = document.querySelector('#mesh-dialog');
  const meshDialogContent = document.querySelector('#mesh-dialog-content');
  const evolutionDialog = document.querySelector('#evolution-dialog');
  const evolutionDialogContent = document.querySelector('#evolution-dialog-content');
  const state = { catalog: [], status: null, jobs: [], artifacts: {}, evolution: {}, route: '', robotViewer: null, robotPath: '', meshViewers: [], meshPart: '', shipDesigner: null, rimworldPrepare: null, charDetails: null, minecraftMap: null, pollTimer: null };

  const escapeHtml = value => String(value ?? '').replace(/[&<>'"]/g, character => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;' })[character]);
  const formatBytes = value => value == null ? '—' : new Intl.NumberFormat('en', { notation: 'compact', maximumFractionDigits: 1 }).format(value) + 'B';
  const statusWord = value => value ? 'online' : 'offline';
  const ICON_NAMES = new Set(['logo', 'mac', 'tk2', 'cloud', 'refresh', 'sync', 'search', 'arrow', 'back', 'close', 'play', 'stop', 'terminal', 'result', 'file', 'check', 'warning', 'layers', 'cube', 'robot', 'joint', 'focus', 'sliders', 'milestones', 'headset', 'point-cloud', 'arm', 'route', 'map', 'vector', 'walk', 'nest', 'ship', 'pawn']);
  const icon = (name, className = '') => {
    const resolved = ICON_NAMES.has(name) ? name : 'cube';
    return `<svg class="icon ${escapeHtml(className)}" aria-hidden="true"><use href="/icons.svg?v=2#icon-${resolved}"></use></svg>`;
  };
  const targetIcon = target => String(target || '').startsWith('tk2') ? 'tk2' : 'mac';
  const artifactIcon = kind => kind === 'image' ? 'result' : kind === 'video' ? 'play' : kind === 'json' ? 'file' : 'layers';
  const visibleExamples = sandbox => (sandbox.examples || []).filter(example => example.surface !== 'meshWorkbench');
  const declaredArtifactCount = sandbox => new Set([
    ...(sandbox.artifacts || []).filter(item => !item.syncOnly).map(item => item.path),
    ...(sandbox.examples || []).flatMap(example => (example.artifacts || []).filter(item => !item.syncOnly).map(item => item.path)),
  ]).size;
  const JOINT_GROUPS = [
    ['left_arm', 'Left arm'], ['right_arm', 'Right arm'], ['left_leg', 'Left leg'], ['right_leg', 'Right leg'], ['body', 'Body'],
  ];

  async function api(url, options = {}) {
    const method = String(options.method || 'GET').toUpperCase();
    const attempts = method === 'GET' ? 2 : 1;
    let lastError = null;
    for (let attempt = 0; attempt < attempts; attempt += 1) {
      const response = await fetch(url, { cache: 'no-store', ...options, headers: { 'Content-Type': 'application/json', ...(options.headers || {}) } });
      const payload = await response.json().catch(() => ({}));
      if (response.ok) return payload;
      lastError = new Error(payload.error || `Request failed (${response.status})`);
      if (attempt + 1 < attempts && [404, 502, 503].includes(response.status)) {
        await new Promise(resolve => setTimeout(resolve, 300));
        continue;
      }
      break;
    }
    throw lastError || new Error('Request failed');
  }

  function toast(message, error = false) {
    const item = document.createElement('div');
    item.className = `toast${error ? ' error' : ''}`;
    item.textContent = message;
    document.querySelector('#toast-stack').append(item);
    setTimeout(() => item.remove(), 4500);
  }

  function renderStatus() {
    const status = state.status;
    const cloud = status?.nextcloud;
    const chips = [
      ['Mac', 'mac', true],
      ['TK2', 'tk2', Boolean(status?.tk2?.online)],
      ['Cloud', 'cloud', Boolean(cloud?.available)],
    ];
    statusRack.innerHTML = chips.map(([label, iconName, online]) => {
      const health = label === 'Mac' || status ? statusWord(online) : 'checking';
      return `<span class="status-chip ${health}" title="${escapeHtml(label)} ${health}">${icon(iconName)}<i${health === 'checking' ? ' style="background:#92a39a"' : ''}></i><span>${label}</span></span>`;
    }).join('');
  }

  function homeView() {
    const cards = state.catalog.map((sandbox, index) => `
      <a class="sandbox-card" style="--accent:${escapeHtml(sandbox.accent || '#1f5b4b')}" href="#/sandbox/${encodeURIComponent(sandbox.id)}" data-search="${escapeHtml(`${sandbox.name} ${sandbox.summary} ${(sandbox.capabilities || []).join(' ')}`.toLowerCase())}">
        <div class="card-head"><span class="number">${String(index + 1).padStart(2, '0')}</span><span class="runtime-badge">${icon(targetIcon(sandbox.runtime))}<span>${escapeHtml((sandbox.runtime || 'local').replace('-clean', ''))}</span></span></div>
        <div class="card-visual"><span class="card-icon">${icon(sandbox.icon || 'cube')}</span><span class="card-orbit" aria-hidden="true"></span></div>
        <div class="card-title"><h3>${escapeHtml(sandbox.name)}</h3><span class="card-arrow">${icon('arrow')}</span></div>
        <span class="sr-only">${escapeHtml(sandbox.summary)}</span>
        <div class="card-foot"><span title="Runnable examples">${icon('play')}<b>${visibleExamples(sandbox).length}</b></span><span title="Declared results">${icon('result')}<b>${declaredArtifactCount(sandbox)}</b></span><div class="capabilities">${(sandbox.capabilities || []).slice(0, 2).map(value => `<span class="capability">${escapeHtml(value)}</span>`).join('')}</div></div>
      </a>`).join('');
    const runnable = state.catalog.filter(item => visibleExamples(item).length).length;
    const remote = state.catalog.filter(item => (item.runtime || '').includes('tk2')).length;
    app.innerHTML = `
      <section class="hero">
        <div class="hero-copy-block">
          <p class="eyebrow">LOCAL MAC · REMOTE TK2 · NEXTCLOUD</p>
          <h1>Build. Run.<br><em>See it.</em></h1>
          <p class="hero-copy">Local visual workbenches for geometry, robot assets, simulation, game modding, and remote compute.</p>
        </div>
        <div class="hero-visual" aria-label="Geo Lab runtime map">
          <span class="hero-grid" aria-hidden="true"></span>
          <div class="hero-core">${icon('logo')}<span>GEO</span></div>
          <div class="runtime-node node-mac online">${icon('mac')}<span>Mac</span><i></i></div>
          <div class="runtime-node node-tk2 ${statusWord(state.status?.tk2?.online)}">${icon('tk2')}<span>TK2</span><i></i></div>
          <div class="runtime-node node-cloud ${statusWord(state.status?.nextcloud?.available)}">${icon('cloud')}<span>Cloud</span><i></i></div>
          <div class="hero-metrics">
            <div class="metric">${icon('cube')}<b>${state.catalog.length}</b><span>Labs</span></div>
            <div class="metric">${icon('play')}<b>${runnable}</b><span>Run</span></div>
            <div class="metric">${icon('tk2')}<b>${remote}</b><span>Remote</span></div>
          </div>
        </div>
      </section>
      <section>
        <div class="catalog-toolbar"><div class="section-heading"><span class="section-icon">${icon('cube')}</span><div><p class="eyebrow">WORKBENCHES</p><h2>Choose a sandbox</h2></div></div><label class="search-shell">${icon('search')}<input id="sandbox-search" class="search" type="search" placeholder="Search" aria-label="Search sandboxes"></label></div>
        <div class="sandbox-grid" id="sandbox-grid">${cards}</div>
      </section>`;
    document.querySelector('#sandbox-search').addEventListener('input', event => {
      const query = event.target.value.trim().toLowerCase();
      document.querySelectorAll('.sandbox-card').forEach(card => { card.hidden = query && !card.dataset.search.includes(query); });
    });
  }

  function exampleCard(sandbox, example) {
    const targets = example.targets || [example.target || 'local'];
    const parameters = (example.parameters || []).map(parameter => {
      const attributes = `data-parameter="${escapeHtml(parameter.id)}"`;
      const label = `<label for="param-${escapeHtml(example.id)}-${escapeHtml(parameter.id)}">${escapeHtml(parameter.label || parameter.id)}</label>`;
      if (parameter.type === 'select') {
        return `<div class="field">${label}<select id="param-${escapeHtml(example.id)}-${escapeHtml(parameter.id)}" ${attributes}>${parameter.choices.map(choice => `<option value="${escapeHtml(choice)}" data-detail="${escapeHtml(parameter.choiceDetails?.[choice] || '')}" ${String(choice) === String(parameter.default) ? 'selected' : ''}>${escapeHtml(parameter.choiceLabels?.[choice] || choice)}</option>`).join('')}</select><small data-choice-detail>${escapeHtml(parameter.choiceDetails?.[parameter.default] || '')}</small></div>`;
      }
      return `<div class="field">${label}<input id="param-${escapeHtml(example.id)}-${escapeHtml(parameter.id)}" ${attributes} type="${parameter.type === 'number' || parameter.type === 'integer' ? 'number' : 'text'}" value="${escapeHtml(parameter.default ?? '')}" ${parameter.min != null ? `min="${escapeHtml(parameter.min)}"` : ''} ${parameter.max != null ? `max="${escapeHtml(parameter.max)}"` : ''} ${parameter.step != null ? `step="${escapeHtml(parameter.step)}"` : ''}></div>`;
    }).join('');
    return `<article class="example" data-example="${escapeHtml(example.id)}">
      <div class="example-top">
        <span class="example-icon">${icon('play')}</span>
        <div class="example-copy"><h3>${escapeHtml(example.name)}</h3><p>${escapeHtml(example.description)}</p><div class="example-meta"><span>${icon(targetIcon(targets[0]))}${escapeHtml(targets.join(' / '))}</span>${example.resource ? `<span>${icon('tk2')}single ${escapeHtml(example.resource)}</span>` : ''}<span>${icon('result')}${example.artifacts?.length || 0}</span></div></div>
        <button class="button run-button button-with-icon" type="button" data-run="${escapeHtml(example.id)}">${icon('play')}<span>Run</span></button>
      </div>
      ${parameters ? `<div class="parameter-grid">${parameters}</div>` : ''}
      ${targets.length > 1 ? `<div class="parameter-grid"><div class="field"><label>Execution target</label><select data-target>${targets.map(target => `<option value="${escapeHtml(target)}" ${target === example.target ? 'selected' : ''}>${target === 'tk2' ? 'TK2 remote' : 'Local Mac'}</option>`).join('')}</select></div></div>` : ''}
    </article>`;
  }

  function milestonePanel(sandbox) {
    if (!sandbox.milestones?.length) return '';
    return `<section class="panel"><header class="panel-head"><div class="panel-title"><span class="section-icon">${icon('milestones')}</span><div><p class="eyebrow">CLEAN RESTART</p><h2>Milestone ladder</h2></div></div><span class="runtime-badge">${icon('warning')}history removed</span></header><div class="panel-body milestone-list">${sandbox.milestones.map((item, index) => `<div class="milestone"><span class="milestone-number">${index + 1}</span><div><b>${escapeHtml(item.name || item.id)}</b><small>${escapeHtml(item.passWhen || item.stage || '')}</small></div><span class="milestone-state" title="${escapeHtml(item.status || 'not started')}">${icon('milestones')}</span></div>`).join('')}</div></section>`;
  }

  function artifactPanel(sandbox) {
    const artifacts = state.artifacts[sandbox.id] || [];
    return `<section class="panel"><header class="panel-head"><div class="panel-title"><span class="section-icon">${icon('result')}</span><h3>Results</h3></div><button class="button button-light button-icon-only" type="button" data-refresh-artifacts aria-label="Refresh results" title="Refresh results">${icon('refresh')}</button></header><div class="panel-body"><div class="artifact-grid">${artifacts.length ? artifacts.map(artifact => `<button type="button" class="artifact${artifact.exists ? '' : ' unavailable'}" ${artifact.exists ? `data-artifact="${escapeHtml(artifact.path)}" data-kind="${escapeHtml(artifact.kind || '')}"` : 'disabled'}><span class="artifact-icon">${icon(artifactIcon(artifact.kind))}</span><span class="artifact-copy"><b>${escapeHtml(artifact.label || artifact.path.split('/').pop())}</b><small>${artifact.exists ? `${escapeHtml(artifact.source)} · ${formatBytes(artifact.size)}` : 'Waiting for a run'}</small></span>${artifact.exists ? icon('arrow', 'artifact-arrow') : ''}</button>`).join('') : `<div class="empty-state">${icon('result')}<span>No declared results</span></div>`}</div></div></section>`;
  }

  function latestJobFor(sandbox) {
    return state.jobs.find(job => job.sandbox === sandbox.id) || null;
  }

  function consolePanel(sandbox) {
    const job = latestJobFor(sandbox);
    return `<section class="panel run-panel"><header class="panel-head"><div class="panel-title"><span class="section-icon">${icon('terminal')}</span><h3>Live run</h3></div>${job && ['queued', 'running', 'cancelling'].includes(job.status) ? `<button class="button button-danger button-with-icon" type="button" data-cancel-job>${icon('stop')}<span>Stop</span></button>` : ''}</header>${job ? `<div class="job-strip"><span class="job-status ${escapeHtml(job.status)}"><i></i>${escapeHtml(job.status)}</span><span class="job-target">${icon(targetIcon(job.target))}${escapeHtml(job.target)}</span></div><pre class="job-console" id="job-console">${escapeHtml(job.log || 'Starting…')}</pre>` : `<div class="empty-console visual-empty">${icon('terminal')}<b>Ready</b><span>Run an example to stream output here.</span></div>`}</section>`;
  }

  function minecraftMapPanel(sandbox) {
      const maps = sandbox.viewer.maps || [];
      return `<section class="panel minecraft-map-panel"><header class="panel-head"><div class="panel-title"><span class="section-icon">${icon('map')}</span><div><p class="eyebrow">OVERWORLD SATELLITE</p><h2>2D block map</h2></div></div><span class="runtime-badge">${icon('cloud')}Nextcloud index</span></header><div class="minecraft-map-toolbar"><label>${icon('layers')}<select data-minecraft-map aria-label="Minecraft map">${maps.map(item => `<option value="${escapeHtml(item.id)}">${escapeHtml(item.label)}</option>`).join('')}</select></label><div class="minecraft-map-actions"><button class="button button-map-region button-with-icon" type="button" data-map-region-mode aria-pressed="false">${icon('focus')}<span>Define region</span></button><button class="button button-light button-icon-only" type="button" data-map-zoom-out aria-label="Zoom out">−</button><output data-map-zoom>100%</output><button class="button button-light button-icon-only" type="button" data-map-zoom-in aria-label="Zoom in">+</button><button class="button button-light button-with-icon" type="button" data-map-reset>${icon('focus')}<span>Fit</span></button></div></div><form class="minecraft-region-form" data-region-form hidden><label><span>Name this region</span><input data-region-name maxlength="80" autocomplete="off" placeholder="Town, district, castle…" required></label><div><button class="button button-primary" type="submit">Save region</button><button class="button button-light" type="button" data-region-cancel>Cancel</button></div></form><div class="minecraft-map-stage" data-map-stage><div class="minecraft-map-canvas" data-map-canvas><img data-map-image alt=""><div class="minecraft-map-regions" data-map-regions aria-hidden="true"></div></div><div class="minecraft-map-loading" data-map-loading>${icon('map')}<span>Loading compact map index…</span></div></div><footer class="minecraft-map-readout"><span data-map-detail></span><span data-map-interaction>Drag to pan · wheel to zoom · world save read-only</span></footer><aside class="minecraft-region-index"><header><div><b>Named regions</b><small>Small coordinate records saved in Nextcloud</small></div><span data-region-count>0</span></header><div data-region-list><p>No regions on this map yet.</p></div></aside></section>`;
  }

  function visualToolsPanel(sandbox) {
    if (sandbox.viewer?.type === 'minecraftMap') {
      return `<section class="panel visual-tools-panel"><header class="panel-head"><h2>Minecraft · floating windows</h2></header><div class="panel-body visual-tool-grid"><button class="visual-tool" data-open-minecraft="map"><span class="visual-tool-icon">${icon('map')}</span><span><b>2D block map</b><small>Explore the overworld and edit named town regions.</small></span></button><button class="visual-tool" data-open-minecraft="buildings"><span class="visual-tool-icon">${icon('cube')}</span><span><b>Luna Town · building viewer</b><small>Largest first · rotate one building · adjust all six bounds.</small></span></button></div></section>`;
    }
    if (sandbox.viewer?.type !== 'urdf' && !sandbox.meshWorkbench) return '';
    return `<section class="panel visual-tools-panel"><header class="panel-head"><div class="panel-title"><span class="section-icon">${icon('cube')}</span><div><p class="eyebrow">FLOATING WINDOWS</p><h2>3D visual tools</h2></div></div></header><div class="panel-body visual-tool-grid">
      ${sandbox.viewer?.type === 'urdf' ? `<button class="visual-tool" type="button" data-open-workbench="urdf"><span class="visual-tool-icon">${icon('robot')}</span><span><b>URDF workbench</b><small>Orbit the produced robot and tune grouped joints.</small></span>${icon('arrow')}</button>` : ''}
      ${sandbox.meshWorkbench ? `<button class="visual-tool" type="button" data-open-workbench="mesh"><span class="visual-tool-icon">${icon('layers')}</span><span><b>Mesh → STL lab</b><small>Compare each body-part mesh and generated STL, then apply a method.</small></span>${icon('arrow')}</button>` : ''}
    </div></section>`;
  }

  function evolutionPreviewPanel(sandbox) {
    if (sandbox.inspector?.type !== 'evolutionTree') return '';
    return `<section class="panel evolution-preview-panel"><header class="panel-head"><div class="panel-title"><span class="section-icon">${icon('milestones')}</span><div><p class="eyebrow">REAL CHECKPOINT LINEAGE</p><h2>${escapeHtml(sandbox.inspector.label || 'Training evolution')}</h2></div></div><button class="button button-with-icon" type="button" data-open-workbench="evolution">${icon('arrow')}<span>Open interactive tree</span></button></header><div class="panel-body evolution-preview-body" data-evolution-preview><div class="evolution-preview-loading">${icon('milestones')}<span>Loading real lineage…</span></div></div></section>`;
  }

  function designerPanel(sandbox) {
    if (sandbox.designer?.type === 'avpRemote') return `<section class="panel"><div id="avp-remote"></div></section>`;
    if (sandbox.designer?.type === 'charDetails') return `<section class="panel"><div id="char-details"></div></section>`;
    if (sandbox.designer?.type === 'stellarisShipDesigner') return `<section class="panel ship-designer-panel"><div id="stellaris-ship-designer" aria-live="polite"></div></section>`;
    if (sandbox.designer?.type === 'rimworldPrepare') return `<section class="panel rimworld-prepare-panel"><div id="rimworld-prepare" aria-live="polite"></div></section>`;
    return '';
  }

  function evolutionPreviewMarkup(data) {
    const all = Array.isArray(data?.nodes) ? data.nodes : [];
    const byId = new Map(all.map(node => [node.id, node]));
    const current = byId.get(data?.currentNodeId) || all.at(-1);
    const branch = [];
    const seen = new Set();
    let cursor = current;
    while (cursor && !seen.has(cursor.id)) {
      branch.push(cursor);
      seen.add(cursor.id);
      cursor = byId.get(cursor.parentIds?.[0]);
    }
    const visible = branch.reverse().slice(-4);
    const progress = node => {
      const value = node?.trainingProgress || {};
      if (value.kind === 'training') return `iter ${value.completedIterations ?? '—'}/${value.requestedIterations ?? '—'} · ${Number(value.sampleCount || 0).toLocaleString()} samples`;
      if (value.kind === 'diagnostic') return `${Number(value.physicsSteps || value.controlSteps || 0).toLocaleString()} steps · ${value.durationSeconds ?? '—'} s`;
      if (value.kind === 'merged') return `${value.runCount || node.collapsedCount || 0} runs merged`;
      return node?.status || 'unknown';
    };
    const nodeMarkup = visible.map(node => `<button class="evolution-preview-node ${escapeHtml(node.status || '')}${node.id === current?.id ? ' current' : ''}" type="button" data-open-workbench="evolution"><i aria-hidden="true"></i><span><small>${escapeHtml(node.kind || 'checkpoint')}</small><b>${escapeHtml(node.label || node.id)}</b></span><em>${escapeHtml(progress(node))}</em></button>`).join('<span class="evolution-preview-edge" aria-hidden="true"></span>');
    const overviewCount = Number(data?.summary?.overviewNodeCount || data?.overviewNodes?.length || all.length);
    return `<div class="evolution-preview-summary"><div><span>Current change</span><b>${escapeHtml(current?.changeSummary || current?.approach || 'No change recorded')}</b></div><div><span>Current result</span><b>${escapeHtml(current?.result || current?.label || 'No current node')}</b></div><div><span>Smart overview</span><b>${overviewCount} visible · ${all.length} recorded</b></div></div><div class="evolution-preview-branch" aria-label="Current checkpoint ancestry">${nodeMarkup || '<p>No lineage nodes recorded.</p>'}</div>`;
  }

  async function loadEvolutionPreview(sandbox) {
    const container = document.querySelector('[data-evolution-preview]');
    if (!container) return;
    try {
      const path = sandbox.inspector?.path;
      if (!path) throw new Error('Evolution artifact path is missing.');
      const payload = state.evolution[sandbox.id] || await api(`/api/artifact?path=${encodeURIComponent(path)}`);
      state.evolution[sandbox.id] = payload;
      if (container.isConnected) container.innerHTML = evolutionPreviewMarkup(payload);
    } catch (error) {
      if (container.isConnected) container.innerHTML = `<div class="warning-box">Evolution lineage unavailable: ${escapeHtml(error.message)}</div>`;
    }
  }

  function sandboxView(sandbox) {
    if (sandbox.workspace?.type === 'nesting') {
      if (state.nestingRoot?.isConnected) { state.nestingWorkspace?.refresh?.(); return; }
      app.innerHTML = '<div id="nesting-workspace"></div>';
      state.nestingRoot = app.querySelector('#nesting-workspace');
      const root = state.nestingRoot;
      void import('/api/artifact?path=' + encodeURIComponent(sandbox.workspace.module)).then(module => {
        if (root.isConnected) state.nestingWorkspace = module.mount(root, { sandbox, api, onStart: ensureJobPolling });
      }).catch(error => { if (root.isConnected) root.textContent = 'Nesting workspace unavailable: ' + error.message; });
      return;
    }
    state.shipDesigner?.destroy?.();
    state.rimworldPrepare?.destroy?.();
    state.charDetails?.destroy?.();
    state.avpRemote?.destroy?.();
    state.avpRemote = null;
    state.shipDesigner = null;
    state.rimworldPrepare = null;
    state.charDetails = null;
    state.minecraftMap = null;
    if (sandbox.id === 'motion_anim_generate') {
      app.innerHTML = '<div id="motion-animation-viewer"></div>';
      const root = app.querySelector('#motion-animation-viewer');
      void import('/api/artifact?path=algorithms/motion_anim_generate/gui/viewer.js').then(module => {
        if (root.isConnected) return module.mount(root, {
          onPreview: previewArtifact,
        });
      }).catch(error => { if (root.isConnected) root.textContent = 'Animation viewer unavailable: ' + error.message; });
      return;
    }
    const resultCount = declaredArtifactCount(sandbox);
    const examples = visibleExamples(sandbox);
    app.innerHTML = `
      <a href="#/" class="back-link">${icon('back')}<span>All sandboxes</span></a>
      <section class="sandbox-hero${sandbox.designer ? ' designer-hero' : ''}" style="--accent:${escapeHtml(sandbox.accent || '#1f5b4b')}">
        <div class="sandbox-identity"><span class="sandbox-identity-icon">${icon(sandbox.icon || 'cube')}</span><div><p class="eyebrow">${escapeHtml(sandbox.eyebrow || 'ALGORITHM SANDBOX')}</p><h1>${escapeHtml(sandbox.name)}</h1><p class="sandbox-summary">${escapeHtml(sandbox.summary)}</p></div></div>
        <div class="sandbox-stats">
          <div>${icon(targetIcon(sandbox.runtime))}<span>Runtime</span><b>${escapeHtml(sandbox.runtimeLabel || sandbox.runtime || 'Local Mac')}</b></div>
          ${sandbox.designer?.type === 'stellarisShipDesigner' ? `<div>${icon('layers')}<span>Models</span><b>${sandbox.designer.designs?.length || 1} designs</b></div><div>${icon('focus')}<span>Markers</span><b>Live XYZ</b></div>` : sandbox.designer?.type === 'rimworldPrepare' ? `<div>${icon('layers')}<span>Source</span><b>Yuran core</b></div><div>${icon('pawn')}<span>Output</span><b>Playable pawn</b></div>` : `<div>${icon('play')}<span>Examples</span><b>${examples.length}</b></div><div>${icon('result')}<span>Results</span><b>${resultCount}</b></div>`}
        </div>
      </section>
      ${designerPanel(sandbox)}
      ${sandbox.designer ? '' : `<div class="workspace-grid">
        <div class="workspace-main">
          ${evolutionPreviewPanel(sandbox)}
          ${visualToolsPanel(sandbox)}
          ${examples.length ? `<section class="panel"><header class="panel-head"><div class="panel-title"><span class="section-icon">${icon('play')}</span><div><p class="eyebrow">ALLOWLISTED</p><h2>Runnable examples</h2></div></div><span class="panel-count">${examples.length}</span></header><div class="panel-body example-list">${examples.map(example => exampleCard(sandbox, example)).join('')}</div></section>` : milestonePanel(sandbox)}
          ${examples.length && sandbox.milestones?.length ? milestonePanel(sandbox) : ''}
        </div>
        <aside class="workspace-side">${consolePanel(sandbox)}${artifactPanel(sandbox)}</aside>
      </div>`}`;
    bindSandbox(sandbox);
    if (sandbox.designer?.type === 'stellarisShipDesigner') {
      const designerRoot = document.querySelector('#stellaris-ship-designer');
      if (designerRoot && window.StellarisShipDesigner) state.shipDesigner = window.StellarisShipDesigner.mount(designerRoot, { sandbox: sandbox.id });
    }
    if (sandbox.designer?.type === 'rimworldPrepare') {
      const prepareRoot = document.querySelector('#rimworld-prepare');
      if (prepareRoot && window.RimWorldPrepare) state.rimworldPrepare = window.RimWorldPrepare.mount(prepareRoot, { sandbox: sandbox.id, designer: sandbox.designer });
    }
    if (sandbox.designer?.type === 'charDetails') {
      const root = document.querySelector('#char-details');
      void import('/api/artifact?path=algorithms/3d_char_details/gui/editor.js').then(module => {
        if (root?.isConnected) state.charDetails = module.mount(root);
      }).catch(error => { if (root?.isConnected) root.textContent = 'Character editor unavailable: ' + error.message; });
    }
    if (sandbox.designer?.type === 'avpRemote') {
      const root = document.querySelector('#avp-remote');
      void import('/api/artifact?path=algorithms/avp_remote/gui/viewer.js').then(module => {
        if (root?.isConnected) state.avpRemote = module.mount(root);
      }).catch(error => { if (root?.isConnected) root.textContent = 'AVP viewer unavailable: ' + error.message; });
    }
    document.querySelectorAll('[data-open-minecraft]').forEach(button=>button.onclick=()=>openMinecraftWindow(sandbox,button.dataset.openMinecraft));
    void loadEvolutionPreview(sandbox);
    const consoleElement = document.querySelector('#job-console');
    if (consoleElement) consoleElement.scrollTop = consoleElement.scrollHeight;
  }

  let minecraftWindowZ=110;
  function openMinecraftWindow(sandbox,kind) {
    const id='minecraft-window-'+kind;
    let panel=document.getElementById(id);
    if(panel){panel.hidden=false;panel.style.zIndex=++minecraftWindowZ;state.minecraftMap?.fit?.();return;}
    panel=document.createElement('section');panel.id=id;panel.className='minecraft-floating-window';panel.setAttribute('role','dialog');panel.setAttribute('aria-label',kind==='map'?'2D block map window':'Luna Town building viewer window');
    panel.innerHTML=`<header class="minecraft-window-title"><b>${kind==='map'?'2D block map':'Luna Town · building viewer'}</b><span><button type="button" data-window-max aria-label="Maximize window">□</button><button type="button" data-window-close aria-label="Close window">×</button></span></header><div class="minecraft-window-content">${kind==='map'?minecraftMapPanel(sandbox):`<iframe title="Luna Town building viewer" src="${escapeHtml(sandbox.viewer.townViewer.url)}?embedded=1" allow="fullscreen"></iframe>`}</div>`;
    app.appendChild(panel);panel.style.zIndex=++minecraftWindowZ;
    panel.addEventListener('pointerdown',()=>panel.style.zIndex=++minecraftWindowZ);
    panel.querySelector('[data-window-close]').onclick=()=>{panel.hidden=true;};
    panel.querySelector('[data-window-max]').onclick=()=>{panel.classList.toggle('maximized');requestAnimationFrame(()=>state.minecraftMap?.fit?.());};
    const header=panel.querySelector('.minecraft-window-title');let drag;
    header.addEventListener('pointerdown',e=>{if(e.target.closest('button')||panel.classList.contains('maximized'))return;const r=panel.getBoundingClientRect();drag={x:e.clientX,y:e.clientY,left:r.left,top:r.top};header.setPointerCapture(e.pointerId);});
    header.addEventListener('pointermove',e=>{if(!drag)return;panel.style.left=Math.max(0,Math.min(innerWidth-120,drag.left+e.clientX-drag.x))+'px';panel.style.top=Math.max(0,Math.min(innerHeight-45,drag.top+e.clientY-drag.y))+'px';});
    header.addEventListener('pointerup',()=>drag=null);header.addEventListener('pointercancel',()=>drag=null);
    if(kind==='map')setupMinecraftMap(sandbox);
  }

  function setupMinecraftMap(sandbox) {
    const stage = document.querySelector('[data-map-stage]');
    const canvas = document.querySelector('[data-map-canvas]');
    const image = document.querySelector('[data-map-image]');
    const loading = document.querySelector('[data-map-loading]');
    const select = document.querySelector('[data-minecraft-map]');
    const zoomOutput = document.querySelector('[data-map-zoom]');
    const detail = document.querySelector('[data-map-detail]');
    const interaction = document.querySelector('[data-map-interaction]');
    const regionModeButton = document.querySelector('[data-map-region-mode]');
    const regionForm = document.querySelector('[data-region-form]');
    const regionName = document.querySelector('[data-region-name]');
    const regionsLayer = document.querySelector('[data-map-regions]');
    const regionList = document.querySelector('[data-region-list]');
    const regionCount = document.querySelector('[data-region-count]');
    if (!stage || !canvas || !image || !select) return;
    const maps = sandbox.viewer.maps || [];
    let transform = { scale: 1, x: 0, y: 0 };
    let pointer = null;
    let currentMap = null;
    let regionMode = false;
    let pendingRegion = null;
    let regions = [];
    const unavailable = new Set();
    const render = () => {
      canvas.style.transform = `translate(${transform.x}px,${transform.y}px) scale(${transform.scale})`;
      regionsLayer?.style.setProperty('--inverse-map-scale', String(1 / transform.scale));
      zoomOutput.textContent = `${Math.round(transform.scale * 100)}%`;
    };
    const regionMarkup = (region, pending = false) => {
      const bounds = region.normalized;
      const left = bounds.x0 * 100;
      const top = bounds.y0 * 100;
      const width = (bounds.x1 - bounds.x0) * 100;
      const height = (bounds.y1 - bounds.y0) * 100;
      return `<span class="minecraft-map-region${pending ? ' pending' : ''}" style="left:${left}%;top:${top}%;width:${width}%;height:${height}%">${pending ? '' : `<b>${escapeHtml(region.name)}</b>`}</span>`;
    };
    const renderRegions = () => {
      if (!regionsLayer || !currentMap) return;
      const visible = regions.filter(region => region.mapId === currentMap.id);
      regionsLayer.innerHTML = visible.map(region => regionMarkup(region)).join('') + (pendingRegion ? regionMarkup(pendingRegion, true) : '');
      if (regionCount) regionCount.textContent = String(visible.length);
      if (!regionList) return;
      regionList.innerHTML = visible.length ? visible.map(region => {
        const bounds = region.world
          ? `X ${Math.round(region.world.x0)}…${Math.round(region.world.x1)} · Z ${Math.round(region.world.z0)}…${Math.round(region.world.z1)}`
          : `Pixels ${Math.round(region.pixels.x0)},${Math.round(region.pixels.y0)} → ${Math.round(region.pixels.x1)},${Math.round(region.pixels.y1)}`;
        return `<article><span><b>${escapeHtml(region.name)}</b><small>${escapeHtml(bounds)}</small></span><button type="button" data-region-delete="${escapeHtml(region.id)}" aria-label="Remove ${escapeHtml(region.name)}">Remove</button></article>`;
      }).join('') : '<p>No regions on this map yet.</p>';
      regionList.querySelectorAll('[data-region-delete]').forEach(button => button.addEventListener('click', async () => {
        const selected = regions.find(region => region.id === button.dataset.regionDelete);
        if (!selected || !window.confirm(`Remove the region “${selected.name}”?`)) return;
        try {
          await api('/api/minecraft/regions/delete', { method: 'POST', body: JSON.stringify({ id: selected.id }) });
          regions = regions.filter(region => region.id !== selected.id);
          renderRegions();
          toast(`Removed region: ${selected.name}`);
        } catch (error) { toast(error.message, true); }
      }));
    };
    const setRegionMode = enabled => {
      regionMode = enabled;
      stage.classList.toggle('region-mode', enabled);
      regionModeButton?.classList.toggle('active', enabled);
      regionModeButton?.setAttribute('aria-pressed', String(enabled));
      if (interaction) interaction.textContent = enabled ? 'Drag a rectangle on the map · Esc to cancel · world save read-only' : 'Drag to pan · wheel to zoom · world save read-only';
    };
    const cancelPending = () => {
      pendingRegion = null;
      if (regionForm) regionForm.hidden = true;
      if (regionName) regionName.value = '';
      renderRegions();
    };
    const imagePoint = event => {
      const bounds = stage.getBoundingClientRect();
      return {
        x: Math.max(0, Math.min(image.naturalWidth, (event.clientX - bounds.left - transform.x) / transform.scale)),
        y: Math.max(0, Math.min(image.naturalHeight, (event.clientY - bounds.top - transform.y) / transform.scale)),
      };
    };
    const rectangle = (start, end) => ({
      x0: Math.min(start.x, end.x) / image.naturalWidth,
      y0: Math.min(start.y, end.y) / image.naturalHeight,
      x1: Math.max(start.x, end.x) / image.naturalWidth,
      y1: Math.max(start.y, end.y) / image.naturalHeight,
    });
    const fit = () => {
      if (!image.naturalWidth || !image.naturalHeight) return;
      const bounds = stage.getBoundingClientRect();
      transform.scale = Math.min((bounds.width - 24) / image.naturalWidth, (bounds.height - 24) / image.naturalHeight, 1);
      transform.x = (bounds.width - image.naturalWidth * transform.scale) / 2;
      transform.y = (bounds.height - image.naturalHeight * transform.scale) / 2;
      render();
    };
    const zoom = (factor, clientX = null, clientY = null) => {
      const previous = transform.scale;
      const next = Math.max(0.08, Math.min(12, previous * factor));
      const bounds = stage.getBoundingClientRect();
      const anchorX = clientX == null ? bounds.width / 2 : clientX - bounds.left;
      const anchorY = clientY == null ? bounds.height / 2 : clientY - bounds.top;
      transform.x = anchorX - (anchorX - transform.x) * next / previous;
      transform.y = anchorY - (anchorY - transform.y) * next / previous;
      transform.scale = next;
      render();
    };
    const load = id => {
      const selected = maps.find(item => item.id === id) || maps[0];
      if (!selected) return;
      currentMap = selected;
      cancelPending();
      setRegionMode(false);
      loading.hidden = false;
      detail.textContent = selected.detail || selected.label;
      image.alt = `${selected.label} top-down Minecraft block map`;
      image.onload = () => { loading.hidden = true; fit(); renderRegions(); };
      image.onerror = () => {
        unavailable.add(selected.id);
        const option = select.querySelector(`option[value="${CSS.escape(selected.id)}"]`);
        if (option) { option.disabled = true; option.textContent = `${selected.label} · preparing`; }
        const fallback = maps.find(item => !unavailable.has(item.id));
        if (fallback) {
          select.value = fallback.id;
          load(fallback.id);
          return;
        }
        loading.innerHTML = `${icon('warning')}<span>No compact map asset is available in Nextcloud.</span>`;
        loading.hidden = false;
      };
      image.src = `/api/artifact?path=${encodeURIComponent(selected.path)}&v=${Date.now()}`;
    };
    select.addEventListener('change', () => load(select.value));
    document.querySelector('[data-map-zoom-in]')?.addEventListener('click', () => zoom(1.35));
    document.querySelector('[data-map-zoom-out]')?.addEventListener('click', () => zoom(1 / 1.35));
    document.querySelector('[data-map-reset]')?.addEventListener('click', fit);
    regionModeButton?.addEventListener('click', () => { cancelPending(); setRegionMode(!regionMode); });
    document.querySelector('[data-region-cancel]')?.addEventListener('click', () => { cancelPending(); setRegionMode(false); });
    regionForm?.addEventListener('submit', async event => {
      event.preventDefault();
      if (!pendingRegion || !currentMap || !regionName?.value.trim()) return;
      try {
        const saved = await api('/api/minecraft/regions', { method: 'POST', body: JSON.stringify({
          name: regionName.value.trim(), mapId: currentMap.id, normalized: pendingRegion.normalized,
          imageSize: { width: image.naturalWidth, height: image.naturalHeight },
        }) });
        regions.push(saved);
        const savedName = saved.name;
        cancelPending();
        setRegionMode(false);
        renderRegions();
        toast(`Saved region: ${savedName}`);
      } catch (error) { toast(error.message, true); }
    });
    stage.addEventListener('pointerdown', event => {
      if (!image.naturalWidth || !image.naturalHeight) return;
      if (regionMode) {
        const start = imagePoint(event);
        pointer = { id: event.pointerId, mode: 'region', start, end: start };
        pendingRegion = { normalized: rectangle(start, start) };
        renderRegions();
      } else {
        pointer = { id: event.pointerId, mode: 'pan', x: event.clientX, y: event.clientY };
      }
      stage.setPointerCapture(event.pointerId);
    });
    stage.addEventListener('pointermove', event => {
      if (!pointer || pointer.id !== event.pointerId) return;
      if (pointer.mode === 'region') {
        pointer.end = imagePoint(event);
        pendingRegion = { normalized: rectangle(pointer.start, pointer.end) };
        renderRegions();
      } else {
        transform.x += event.clientX - pointer.x;
        transform.y += event.clientY - pointer.y;
        pointer.x = event.clientX; pointer.y = event.clientY;
        render();
      }
    });
    stage.addEventListener('pointerup', event => {
      if (!pointer || pointer.id !== event.pointerId) return;
      if (pointer.mode === 'region') {
        pointer.end = imagePoint(event);
        const normalized = rectangle(pointer.start, pointer.end);
        const wideEnough = (normalized.x1 - normalized.x0) * image.naturalWidth >= 4;
        const tallEnough = (normalized.y1 - normalized.y0) * image.naturalHeight >= 4;
        if (wideEnough && tallEnough) {
          pendingRegion = { normalized };
          if (regionForm) regionForm.hidden = false;
          regionName?.focus();
        } else cancelPending();
        renderRegions();
      }
      pointer = null;
    });
    stage.addEventListener('pointercancel', () => { pointer = null; cancelPending(); });
    stage.addEventListener('wheel', event => { event.preventDefault(); zoom(event.deltaY < 0 ? 1.18 : 1 / 1.18, event.clientX, event.clientY); }, { passive: false });
    document.addEventListener('keydown', event => { if (event.key === 'Escape' && (regionMode || pendingRegion)) { cancelPending(); setRegionMode(false); } });
    void api('/api/minecraft/regions').then(payload => {
      if (!stage.isConnected) return;
      regions = Array.isArray(payload.regions) ? payload.regions : [];
      renderRegions();
    }).catch(error => toast(`Named regions unavailable: ${error.message}`, true));
    state.minecraftMap = { fit, load };
    select.value=maps.find(item=>item.id==='bilicraft-overworld')?.id||maps[0]?.id;
    load(select.value);
  }

  function bindSandbox(sandbox) {
    document.querySelectorAll('select[data-parameter]').forEach(select => select.addEventListener('change', () => {
      const detail = select.closest('.field')?.querySelector('[data-choice-detail]');
      if (detail) detail.textContent = select.selectedOptions[0]?.dataset.detail || '';
    }));
    document.querySelectorAll('[data-run]').forEach(button => button.addEventListener('click', async () => {
      const card = button.closest('[data-example]');
      const exampleId = card.dataset.example;
      const parameters = {};
      card.querySelectorAll('[data-parameter]').forEach(input => { parameters[input.dataset.parameter] = input.value; });
      const target = card.querySelector('[data-target]')?.value;
      button.disabled = true;
      try {
        const job = await api('/api/jobs', { method: 'POST', body: JSON.stringify({ sandbox: sandbox.id, example: exampleId, target, parameters }) });
        state.jobs = [job, ...state.jobs.filter(item => item.id !== job.id)];
        sandboxView(sandbox);
        ensureJobPolling();
      } catch (error) { toast(error.message, true); button.disabled = false; }
    }));
    document.querySelector('[data-cancel-job]')?.addEventListener('click', async () => {
      const job = latestJobFor(sandbox);
      if (!job) return;
      try { await api(`/api/jobs/${job.id}/cancel`, { method: 'POST', body: '{}' }); ensureJobPolling(); } catch (error) { toast(error.message, true); }
    });
    document.querySelector('[data-refresh-artifacts]')?.addEventListener('click', () => void loadArtifacts(sandbox, true));
    document.querySelectorAll('[data-artifact]').forEach(button => button.addEventListener('click', () => void previewArtifact(button.dataset.artifact, button.dataset.kind)));
    document.querySelector('[data-open-workbench="urdf"]')?.addEventListener('click', () => void openRobotWorkbench(sandbox));
    document.querySelector('[data-open-workbench="mesh"]')?.addEventListener('click', () => void openMeshWorkbench(sandbox));
    document.querySelectorAll('[data-open-workbench="evolution"]').forEach(button => button.addEventListener('click', () => void openEvolutionTree(sandbox)));
  }

  async function openEvolutionTree(sandbox) {
    evolutionDialog.dataset.sandbox = sandbox.id;
    evolutionDialogContent.innerHTML = `<div class="viewer-loading">${icon('milestones')}<span>Loading real lineage…</span></div>`;
    evolutionDialog.showModal();
    try {
      const path = sandbox.inspector?.path;
      if (!path) throw new Error('This sandbox does not declare an evolution artifact.');
      const payload = await api(`/api/artifact?path=${encodeURIComponent(path)}`);
      state.evolution[sandbox.id] = payload;
      if (!window.GeoEvolutionTree) throw new Error('Evolution Tree renderer is unavailable.');
      window.GeoEvolutionTree.mount(evolutionDialogContent, {
        data: payload,
        artifacts: state.artifacts[sandbox.id] || [],
        onPreview: (artifactPath, kind) => void previewArtifact(artifactPath, kind),
      });
    } catch (error) {
      evolutionDialogContent.innerHTML = `<div class="workbench-retry workbench-retry-centered"><div class="warning-box">Evolution data could not be loaded. ${escapeHtml(error.message)}</div><button class="button button-with-icon" type="button" data-retry-evolution>${icon('refresh')}<span>Retry</span></button></div>`;
      evolutionDialogContent.querySelector('[data-retry-evolution]')?.addEventListener('click', () => void openEvolutionTree(sandbox));
    }
  }

  async function previewArtifact(path, kind) {
    const url = `/api/artifact?path=${encodeURIComponent(path)}`;
    artifactPreview.innerHTML = '<p>Loading preview…</p>';
    artifactDialog.showModal();
    try {
      if (kind === 'image' || /\.(png|jpg|jpeg|gif|pgm)$/i.test(path)) {
        artifactPreview.innerHTML = `<img src="${url}" alt="${escapeHtml(path)}">`;
      } else if (kind === 'video' || /\.(mp4|webm|mov)$/i.test(path)) {
        artifactPreview.innerHTML = `<video controls autoplay muted playsinline preload="metadata" src="${url}" aria-label="${escapeHtml(path)}"></video>`;
      } else {
        const response = await fetch(url, { cache: 'no-store' });
        const text = await response.text();
        let formatted = text;
        try { formatted = JSON.stringify(JSON.parse(text), null, 2); } catch {}
        artifactPreview.innerHTML = `<pre>${escapeHtml(formatted)}</pre>`;
      }
    } catch (error) { artifactPreview.innerHTML = `<p>${escapeHtml(error.message)}</p>`; }
  }

  async function loadArtifacts(sandbox, announce = false) {
    try {
      const payload = await api(`/api/artifacts/${encodeURIComponent(sandbox.id)}`);
      state.artifacts[sandbox.id] = payload.artifacts;
      if (currentSandbox()?.id === sandbox.id) sandboxView(sandbox);
      if (announce) toast('Results refreshed');
    } catch (error) { toast(error.message, true); }
  }

  async function openRobotWorkbench(sandbox) {
    urdfDialogContent.innerHTML = `<div class="viewer-loading">${icon('robot')}<span>Loading robot…</span></div>`;
    urdfDialog.dataset.sandbox = sandbox.id;
    urdfDialog.showModal();
    await setupRobotViewer(sandbox);
  }

  async function setupRobotViewer(sandbox) {
    const container = urdfDialogContent;
    if (!container) return;
    try {
      const payload = await api('/api/robot/catalog');
      const robots = payload.robots.filter(item => sandbox.viewer.urdfCandidates.some(candidate => candidate.path === item.path));
      container.innerHTML = `<div class="viewer-layout"><div class="viewer-stage" id="viewer-stage"><canvas id="robot-mesh"></canvas><canvas id="robot-overlay"></canvas><div class="viewer-tools"><label class="viewer-select">${icon('robot')}<select id="robot-select" aria-label="Robot URDF">${robots.map(robot => `<option value="${escapeHtml(robot.path)}" ${robot.exists ? '' : 'disabled'}>${escapeHtml(robot.label)}${robot.exists ? '' : ' · unavailable'}</option>`).join('')}</select></label><button class="button button-light button-with-icon" id="reset-view" type="button">${icon('focus')}<span>Reset view</span></button></div><span class="viewer-status" id="viewer-status">Select a URDF</span></div><div class="joint-controls"><div class="joint-controls-head"><span class="section-icon">${icon('joint')}</span><div><p class="eyebrow">ARTICULATION</p><h3>Joint controls</h3></div></div><div class="joint-toolbar"><label class="search-shell">${icon('search')}<input id="joint-search" type="search" placeholder="Find joint" aria-label="Filter joints"></label><button class="button button-light button-icon-only" id="zero-joints" type="button" aria-label="Zero all joints" title="Zero all joints">${icon('refresh')}</button></div><div id="joint-list" aria-live="polite"><div class="joint-list-state">Loading joints…</div></div></div></div>`;
      const select = document.querySelector('#robot-select');
      const first = robots.find(robot => robot.exists);
      if (!first) { document.querySelector('#viewer-status').textContent = 'No declared URDF is available'; return; }
      select.value = first.path;
      select.addEventListener('change', () => void loadRobot(select.value));
      document.querySelector('#reset-view').addEventListener('click', () => state.robotViewer?.resetView());
      document.querySelector('#zero-joints').addEventListener('click', () => {
        document.querySelectorAll('.joint-row input[type="range"]').forEach(input => { input.value = '0'; input.dispatchEvent(new Event('input')); });
      });
      document.querySelector('#joint-search').addEventListener('input', event => {
        const query = event.target.value.toLowerCase();
        document.querySelectorAll('.joint-row').forEach(row => { row.hidden = query && !row.dataset.name.includes(query); });
        document.querySelectorAll('.joint-group').forEach(group => {
          const visible = Array.from(group.querySelectorAll('.joint-row')).some(row => !row.hidden);
          group.hidden = !visible;
          if (query && visible) group.open = true;
        });
      });
      bindViewerGestures(document.querySelector('#viewer-stage'), () => state.robotViewer);
      try {
        await loadRobot(first.path);
      } catch (error) {
        showRobotLoadError(first.path, error);
      }
    } catch (error) { container.innerHTML = `<div class="warning-box">${escapeHtml(error.message)}</div>`; }
  }

  function showRobotLoadError(path, error) {
    const status = document.querySelector('#viewer-status');
    const list = document.querySelector('#joint-list');
    if (status) status.textContent = error.message;
    if (!list) return;
    list.innerHTML = `<div class="workbench-retry"><div class="warning-box">Joint definitions could not be loaded. ${escapeHtml(error.message)}</div><button class="button button-with-icon" id="retry-robot-load" type="button">${icon('refresh')}<span>Retry joints and model</span></button></div>`;
    document.querySelector('#retry-robot-load')?.addEventListener('click', async event => {
      event.currentTarget.disabled = true;
      try { await loadRobot(path); } catch (retryError) { showRobotLoadError(path, retryError); }
    });
  }

  async function loadRobot(path) {
    const status = document.querySelector('#viewer-status');
    const list = document.querySelector('#joint-list');
    status.textContent = 'Reading URDF…';
    list.setAttribute('aria-busy', 'true');
    list.innerHTML = '<div class="joint-list-state">Loading joint definitions…</div>';
    const payload = await api(`/api/robot/urdf?path=${encodeURIComponent(path)}`);
    state.robotViewer?.destroy?.();
    state.robotPath = path;
    state.robotViewer = new window.MotionUrdfViewer({
      meshCanvas: document.querySelector('#robot-mesh'),
      overlayCanvas: document.querySelector('#robot-overlay'),
      assetUrl: `/api/robot/asset?urdf=${encodeURIComponent(path)}`,
      onStatus: message => { status.textContent = message; },
    });
    const joints = payload.joints;
    const knownGroups = new Set(JOINT_GROUPS.map(([group]) => group));
    const groups = [...JOINT_GROUPS];
    if (joints.some(joint => !knownGroups.has(joint.group))) groups.push(['other', 'Other']);
    list.innerHTML = groups.map(([group, label]) => {
      const grouped = joints.filter(joint => group === 'other' ? !knownGroups.has(joint.group) : joint.group === group);
      if (!grouped.length) return '';
      return `<details class="joint-group" data-joint-group="${group}" open><summary><span>${escapeHtml(label)}</span><small>${grouped.length}</small></summary><div>${grouped.map(joint => `<div class="joint-row" data-name="${escapeHtml(joint.name.toLowerCase())}"><div class="joint-label"><span>${escapeHtml(joint.name)}</span><output>0.000</output></div><input type="range" min="${joint.lower}" max="${joint.upper}" step="0.001" value="0" data-joint="${escapeHtml(joint.name)}" aria-label="${escapeHtml(joint.name)} angle"></div>`).join('')}</div></details>`;
    }).join('');
    if (!joints.length) list.innerHTML = '<div class="warning-box">This URDF contains no controllable joints.</div>';
    list.removeAttribute('aria-busy');
    document.querySelectorAll('[data-joint]').forEach(input => input.addEventListener('input', () => {
      input.closest('.joint-row').querySelector('output').textContent = Number(input.value).toFixed(3);
      state.robotViewer.setJoints([input.dataset.joint], [Number(input.value)]);
    }));
    await state.robotViewer.load(`/api/robot/urdf?path=${encodeURIComponent(path)}`, joints.map(item => item.name), joints.map(() => 0));
  }

  function bindViewerGestures(stage, viewerProvider) {
    if (!stage || stage.dataset.viewerGesturesBound) return;
    stage.dataset.viewerGesturesBound = 'true';
    let pointer = null;
    stage.addEventListener('pointerdown', event => { pointer = { id: event.pointerId, x: event.clientX, y: event.clientY }; stage.setPointerCapture(event.pointerId); });
    stage.addEventListener('pointermove', event => {
      const viewer = viewerProvider();
      if (!pointer || pointer.id !== event.pointerId || !viewer) return;
      viewer.orbit(event.clientX - pointer.x, event.clientY - pointer.y);
      pointer.x = event.clientX; pointer.y = event.clientY;
    });
    stage.addEventListener('pointerup', () => { pointer = null; });
    stage.addEventListener('wheel', event => { event.preventDefault(); viewerProvider()?.dolly(event.deltaY); }, { passive: false });
  }

  function meshParameterFields(catalog) {
    return catalog.parameters.map(parameter => {
      const value = catalog.current[parameter.id] ?? parameter.default ?? '';
      const applies = parameter.id === 'method' ? 'all' : ['target_face_ratio', 'max_faces'].includes(parameter.id) ? 'lowpoly_surface' : ['max_hull_faces', 'target_hull_points'].includes(parameter.id) ? 'convex_hull' : 'all';
      if (parameter.type === 'select') {
        return `<div class="field mesh-field" data-methods="${applies}"><label>${escapeHtml(parameter.label || parameter.id)}</label><select data-mesh-parameter="${escapeHtml(parameter.id)}">${parameter.choices.map(choice => `<option value="${escapeHtml(choice)}" ${String(choice) === String(value) ? 'selected' : ''}>${escapeHtml(parameter.choiceLabels?.[choice] || String(choice).replaceAll('_', ' '))}</option>`).join('')}</select></div>`;
      }
      return `<div class="field mesh-field" data-methods="${applies}"><label>${escapeHtml(parameter.label || parameter.id)}</label><input data-mesh-parameter="${escapeHtml(parameter.id)}" type="number" value="${escapeHtml(value)}" ${parameter.min != null ? `min="${escapeHtml(parameter.min)}"` : ''} ${parameter.max != null ? `max="${escapeHtml(parameter.max)}"` : ''} ${parameter.step != null ? `step="${escapeHtml(parameter.step)}"` : ''}></div>`;
    }).join('');
  }

  function meshPartOptions(parts) {
    return JOINT_GROUPS.map(([group, label]) => {
      const grouped = parts.filter(part => part.group === group);
      return grouped.length ? `<optgroup label="${escapeHtml(label)}">${grouped.map(part => `<option value="${escapeHtml(part.name)}">${escapeHtml(part.name)}</option>`).join('')}</optgroup>` : '';
    }).join('');
  }

  async function openMeshWorkbench(sandbox) {
    meshDialogContent.innerHTML = `<div class="viewer-loading">${icon('layers')}<span>Loading mesh lab…</span></div>`;
    meshDialog.dataset.sandbox = sandbox.id;
    meshDialog.showModal();
    await setupMeshWorkbench(sandbox, state.meshPart);
  }

  async function setupMeshWorkbench(sandbox, preferredPart = '') {
    try {
      const catalog = await api(`/api/mesh/catalog?sandbox=${encodeURIComponent(sandbox.id)}`);
      if (!catalog.parts.length) throw new Error('No generated body-part STL files are available yet. Run the mesh build first.');
      const selected = catalog.parts.some(part => part.name === preferredPart) ? preferredPart : catalog.parts[0].name;
      state.meshPart = selected;
      meshDialogContent.innerHTML = `<div class="mesh-lab-layout">
        <aside class="mesh-settings">
          <div class="mesh-settings-intro"><span class="section-icon">${icon('sliders')}</span><div><p class="eyebrow">PER-LINK PREVIEW</p><h3>Build settings</h3></div></div>
          <div class="field"><label>Body part</label><select id="mesh-part-select">${meshPartOptions(catalog.parts)}</select></div>
          <div class="field"><label>Build on</label><select id="mesh-target-select">${catalog.targets.map(target => `<option value="${escapeHtml(target)}" ${target === catalog.target ? 'selected' : ''}>${target === 'local' ? 'This Mac · Blender' : 'TK2 · Isaac'}</option>`).join('')}</select></div>
          <div class="mesh-parameter-grid">${meshParameterFields(catalog)}</div>
          <div class="mesh-apply-note">Applying rebuilds every per-link STL as a closed surface, checks manifold topology, and updates the produced mesh URDF package.</div>
          <button class="button button-with-icon mesh-apply" type="button" id="apply-mesh-settings">${icon('play')}<span>Apply to produced URDF</span></button>
          <div class="mesh-apply-status" id="mesh-apply-status" aria-live="polite"></div>
        </aside>
        <section class="mesh-comparison">
          <header class="mesh-comparison-head"><div><p class="eyebrow">SYNCHRONIZED VIEW</p><h3 id="mesh-part-title">${escapeHtml(selected)}</h3></div><button class="button button-light button-with-icon" id="reset-mesh-views" type="button">${icon('focus')}<span>Reset views</span></button></header>
          <div class="mesh-compare-grid">
            <div class="mesh-compare-pane"><div class="mesh-pane-label"><b>Source mesh</b><span id="source-mesh-state">Loading…</span></div><div class="mesh-stage" id="source-mesh-stage"><canvas id="source-mesh-canvas"></canvas><canvas id="source-mesh-overlay"></canvas></div></div>
            <div class="mesh-compare-pane"><div class="mesh-pane-label"><b>Generated STL</b><span id="stl-mesh-state">Loading…</span></div><div class="mesh-stage" id="stl-mesh-stage"><canvas id="stl-mesh-canvas"></canvas><canvas id="stl-mesh-overlay"></canvas></div></div>
          </div>
        </section>
      </div>`;
      const partSelect = document.querySelector('#mesh-part-select');
      partSelect.value = selected;
      partSelect.addEventListener('change', () => void loadMeshPart(sandbox, catalog, partSelect.value));
      const methodSelect = document.querySelector('[data-mesh-parameter="method"]');
      const updateMethodFields = () => {
        const method = methodSelect.value;
        document.querySelectorAll('.mesh-field[data-methods]').forEach(field => {
          const applies = field.dataset.methods;
          field.hidden = applies !== 'all' && applies !== method;
        });
      };
      methodSelect.addEventListener('change', updateMethodFields);
      updateMethodFields();
      document.querySelector('#reset-mesh-views').addEventListener('click', () => state.meshViewers.forEach(viewer => viewer.resetView()));
      document.querySelector('#apply-mesh-settings').addEventListener('click', async event => {
        const button = event.currentTarget;
        const parameters = {};
        const target = document.querySelector('#mesh-target-select').value;
        document.querySelectorAll('[data-mesh-parameter]').forEach(input => { parameters[input.dataset.meshParameter] = input.value; });
        button.disabled = true;
        document.querySelector('#mesh-apply-status').textContent = target === 'local' ? 'Building watertight STL assets on this Mac…' : 'Syncing source, then rebuilding on TK2…';
        try {
          const job = await api('/api/jobs', { method: 'POST', body: JSON.stringify({ sandbox: sandbox.id, example: catalog.applyExample, target, parameters }) });
          state.jobs = [job, ...state.jobs.filter(item => item.id !== job.id)];
          const targetLabel = target === 'local' ? 'this Mac' : 'TK2';
          document.querySelector('#mesh-apply-status').textContent = `Build started on ${targetLabel}. The comparison refreshes when it completes.`;
          toast(`Mesh → STL build started on ${targetLabel}`);
          ensureJobPolling();
        } catch (error) {
          document.querySelector('#mesh-apply-status').textContent = error.message;
          toast(error.message, true);
          button.disabled = false;
        }
      });
      await loadMeshPart(sandbox, catalog, selected);
    } catch (error) {
      meshDialogContent.innerHTML = `<div class="workbench-retry workbench-retry-centered"><div class="warning-box">Mesh data could not be loaded. ${escapeHtml(error.message)}</div><button class="button button-with-icon" id="retry-mesh-load" type="button">${icon('refresh')}<span>Retry mesh lab</span></button></div>`;
      document.querySelector('#retry-mesh-load')?.addEventListener('click', event => {
        event.currentTarget.disabled = true;
        void setupMeshWorkbench(sandbox, preferredPart);
      });
    }
  }

  async function loadMeshPart(sandbox, catalog, part) {
    state.meshViewers.forEach(viewer => viewer.destroy?.());
    state.meshPart = part;
    document.querySelector('#mesh-part-title').textContent = part;
    const selected = catalog.parts.find(item => item.name === part);
    const sourceStatus = document.querySelector('#source-mesh-state');
    const stlStatus = document.querySelector('#stl-mesh-state');
    sourceStatus.textContent = selected?.sourceExists ? 'Original extracted surface' : 'No weighted USD surface exported for this part';
    stlStatus.textContent = selected?.stlExists ? 'Current URDF geometry' : 'Unavailable';
    const specs = [
      { variant: 'source', mesh: '#source-mesh-canvas', overlay: '#source-mesh-overlay', stage: '#source-mesh-stage', status: sourceStatus },
      { variant: 'stl', mesh: '#stl-mesh-canvas', overlay: '#stl-mesh-overlay', stage: '#stl-mesh-stage', status: stlStatus },
    ];
    for (const spec of specs) {
      if (spec.variant === 'source' && !selected?.sourceExists) continue;
      if (spec.variant === 'stl' && !selected?.stlExists) continue;
      spec.viewer = new window.MotionUrdfViewer({
        meshCanvas: document.querySelector(spec.mesh),
        overlayCanvas: document.querySelector(spec.overlay),
        assetUrl: `/api/mesh/asset?sandbox=${encodeURIComponent(sandbox.id)}&part=${encodeURIComponent(part)}&variant=${spec.variant}&v=${encodeURIComponent(selected?.[`${spec.variant}Version`] || '')}`,
        onStatus: message => { spec.status.textContent = message; },
      });
    }
    state.meshViewers = specs.map(spec => spec.viewer).filter(Boolean);
    specs.forEach(spec => bindViewerGestures(document.querySelector(spec.stage), () => ({
      orbit: (dx, dy) => state.meshViewers.forEach(viewer => viewer.orbit(dx, dy)),
      dolly: delta => state.meshViewers.forEach(viewer => viewer.dolly(delta)),
    })));
    await Promise.all(specs.filter(spec => spec.viewer).map(spec => spec.viewer.load(
      `/api/mesh/urdf?sandbox=${encodeURIComponent(sandbox.id)}&part=${encodeURIComponent(part)}&variant=${spec.variant}`,
      [], [],
    )));
  }

  function currentSandbox() {
    const match = location.hash.match(/^#\/sandbox\/([^/]+)/);
    return match ? state.catalog.find(item => item.id === decodeURIComponent(match[1])) : null;
  }

  function renderRoute() {
    state.nestingWorkspace?.destroy?.();
    state.nestingWorkspace = null;
    state.nestingRoot = null;
    const sandbox = currentSandbox();
    state.robotViewer?.destroy?.();
    state.meshViewers.forEach(viewer => viewer.destroy?.());
    state.shipDesigner?.destroy?.();
    state.rimworldPrepare?.destroy?.();
    state.charDetails?.destroy?.();
    state.avpRemote?.destroy?.();
    state.avpRemote = null;
    state.robotViewer = null;
    state.meshViewers = [];
    state.shipDesigner = null;
    state.rimworldPrepare = null;
    state.charDetails = null;
    if (urdfDialog.open && urdfDialog.dataset.sandbox !== sandbox?.id) urdfDialog.close();
    if (meshDialog.open && meshDialog.dataset.sandbox !== sandbox?.id) meshDialog.close();
    if (evolutionDialog.open && evolutionDialog.dataset.sandbox !== sandbox?.id) evolutionDialog.close();
    if (sandbox) {
      sandboxView(sandbox);
      if (sandbox.workspace?.type !== 'nesting' && !state.artifacts[sandbox.id]) void loadArtifacts(sandbox);
    } else {
      homeView();
    }
    app.focus({ preventScroll: true });
  }

  async function refreshJobs() {
    const payload = await api('/api/jobs');
    const previous = state.jobs;
    state.jobs = payload.jobs;
    const sandbox = currentSandbox();
    if (sandbox) {
      const before = previous.find(job => job.sandbox === sandbox.id);
      const after = state.jobs.find(job => job.sandbox === sandbox.id);
      if (JSON.stringify(before) !== JSON.stringify(after)) {
        sandboxView(sandbox);
        if (after && ['succeeded', 'failed', 'cancelled'].includes(after.status)) {
          if (sandbox.workspace?.type !== 'nesting') void loadArtifacts(sandbox);
          if (meshDialog.open && after.example === sandbox.meshWorkbench?.applyExample) {
            if (after.status === 'succeeded') void setupMeshWorkbench(sandbox, state.meshPart);
            else document.querySelector('#mesh-apply-status')?.append(` Build ${after.status}.`);
          }
        }
      }
    }
    ensureJobPolling();
  }

  function ensureJobPolling() {
    clearTimeout(state.pollTimer);
    const running = state.jobs.some(job => ['queued', 'running', 'cancelling'].includes(job.status));
    if (running && document.visibilityState === 'visible') state.pollTimer = setTimeout(() => void refreshJobs().catch(error => toast(error.message, true)), 1500);
  }

  async function refreshAll(announce = false) {
    // Health checks can involve SSH; the local workspace needs only its catalog.
    const status = api('/api/status').then(payload => {
      state.status = payload;
      renderStatus();
    }).catch(error => { if (announce) toast(error.message, true); });
    const jobs = api('/api/jobs').then(payload => {
      state.jobs = payload.jobs;
      ensureJobPolling();
    }).catch(error => { if (announce) toast(error.message, true); });
    try {
      const catalog = await api('/api/catalog');
      state.catalog = catalog.sandboxes;
      renderStatus();
      renderRoute();
      if (announce) {
        await Promise.all([status, jobs]);
        toast('Mac, TK2, storage, jobs, and results refreshed');
      }
    } catch (error) {
      app.innerHTML = `<section class="loading-view"><p class="eyebrow">CONNECTION ERROR</p><h1>Geo Lab is not ready.</h1><p>${escapeHtml(error.message)}</p></section>`;
      toast(error.message, true);
    }
  }

  async function openStorage() {
    storageDrawer.classList.add('open');
    storageDrawer.setAttribute('aria-hidden', 'false');
    storageContent.innerHTML = `<div class="drawer-loading">${icon('sync')}<span>Refreshing storage…</span></div>`;
    try {
      const payload = await api('/api/storage');
      const storage = payload.storage;
      const summary = storage.summary;
      storageContent.innerHTML = `
        <div class="storage-summary"><div>${icon('cloud')}<b>${summary.available}/${summary.total}</b><span>Cloud</span></div><div>${icon('sync')}<b>${summary.hydrated}</b><span>Linked</span></div><div>${icon(payload.audit.ok ? 'check' : 'warning')}<b>${payload.audit.oversizedTrackedFiles.length}</b><span>Oversized</span></div></div>
        <div class="storage-route" aria-label="Storage route"><span>${icon('mac')}Mac</span>${icon('arrow')}<span>${icon('cloud')}Nextcloud</span>${icon('arrow')}<span>${icon('tk2')}TK2</span></div>
        <p class="storage-note"><code>${escapeHtml(storage.root)}</code><span>Explicit sync keeps both machines predictable.</span></p>
        ${payload.audit.ok ? '' : `<div class="warning-box">${payload.audit.oversizedTrackedFiles.length} tracked files exceed the 5 MB repository limit.</div>`}
        <div class="storage-actions"><button class="button button-with-icon" data-storage-action="hydrate">${icon('layers')}<span>Hydrate</span></button><button class="button button-light button-with-icon" data-storage-action="audit">${icon('check')}<span>Audit</span></button><button class="button button-with-icon" data-storage-action="sync-code-tk2">${icon('tk2')}<span>Code → TK2</span></button><button class="button button-with-icon" data-storage-action="push-tk2">${icon('cloud')}<span>Cloud → TK2</span></button><button class="button button-light button-with-icon" data-storage-action="pull-tk2">${icon('sync')}<span>TK2 → Cloud</span></button></div>
        <div class="file-list">${storage.entries.map(entry => `<div class="file-row"><span class="file-icon ${entry.valid ? 'valid' : 'invalid'}">${icon(entry.valid ? 'check' : 'warning')}</span><div><b>${escapeHtml(entry.repoPath)}</b><span>${entry.sourceExists ? formatBytes(entry.size) : 'missing'} · ${entry.hydrated ? 'linked' : 'not linked'}</span></div></div>`).join('')}</div>`;
      storageContent.querySelectorAll('[data-storage-action]').forEach(button => button.addEventListener('click', async () => {
        button.disabled = true;
        try {
          const job = await api(`/api/storage/${button.dataset.storageAction}`, { method: 'POST', body: '{}' });
          state.jobs = [job, ...state.jobs];
          toast(`${button.textContent} started; refresh Shared files to see the completed state.`);
          ensureJobPolling();
        } catch (error) { toast(error.message, true); button.disabled = false; }
      }));
    } catch (error) { storageContent.innerHTML = `<div class="warning-box">${escapeHtml(error.message)}</div>`; }
  }

  document.querySelector('#refresh-button').addEventListener('click', () => void refreshAll(true));
  document.querySelector('#storage-button').addEventListener('click', () => void openStorage());
  document.querySelectorAll('[data-close-drawer]').forEach(element => element.addEventListener('click', () => { storageDrawer.classList.remove('open'); storageDrawer.setAttribute('aria-hidden', 'true'); }));
  artifactDialog.querySelector('.dialog-close').addEventListener('click', () => artifactDialog.close());
  artifactDialog.addEventListener('click', event => { if (event.target === artifactDialog) artifactDialog.close(); });
  document.querySelector('[data-close-workbench="urdf"]').addEventListener('click', () => urdfDialog.close());
  document.querySelector('[data-close-workbench="mesh"]').addEventListener('click', () => meshDialog.close());
  document.querySelector('[data-close-workbench="evolution"]').addEventListener('click', () => evolutionDialog.close());
  for (const dialog of [urdfDialog, meshDialog, evolutionDialog]) dialog.addEventListener('click', event => { if (event.target === dialog) dialog.close(); });
  urdfDialog.addEventListener('close', () => { state.robotViewer?.destroy?.(); state.robotViewer = null; urdfDialogContent.innerHTML = ''; });
  meshDialog.addEventListener('close', () => { state.meshViewers.forEach(viewer => viewer.destroy?.()); state.meshViewers = []; meshDialogContent.innerHTML = ''; });
  evolutionDialog.addEventListener('close', () => { evolutionDialogContent.innerHTML = ''; });
  window.addEventListener('hashchange', renderRoute);
  document.addEventListener('visibilitychange', () => { if (document.visibilityState === 'visible') void refreshAll(); else clearTimeout(state.pollTimer); });
  void refreshAll();
})();
