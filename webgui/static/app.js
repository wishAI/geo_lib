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
  const state = { catalog: [], status: null, jobs: [], artifacts: {}, route: '', robotViewer: null, robotPath: '', meshViewers: [], meshPart: '', pollTimer: null };

  const escapeHtml = value => String(value ?? '').replace(/[&<>'"]/g, character => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;' })[character]);
  const formatBytes = value => value == null ? '—' : new Intl.NumberFormat('en', { notation: 'compact', maximumFractionDigits: 1 }).format(value) + 'B';
  const statusWord = value => value ? 'online' : 'offline';
  const ICON_NAMES = new Set(['logo', 'mac', 'tk2', 'cloud', 'refresh', 'sync', 'search', 'arrow', 'back', 'close', 'play', 'stop', 'terminal', 'result', 'file', 'check', 'warning', 'layers', 'cube', 'robot', 'joint', 'focus', 'sliders', 'milestones', 'headset', 'point-cloud', 'arm', 'route', 'map', 'vector', 'walk', 'nest']);
  const icon = (name, className = '') => {
    const resolved = ICON_NAMES.has(name) ? name : 'cube';
    return `<svg class="icon ${escapeHtml(className)}" aria-hidden="true"><use href="/icons.svg?v=2#icon-${resolved}"></use></svg>`;
  };
  const targetIcon = target => String(target || '').startsWith('tk2') ? 'tk2' : 'mac';
  const artifactIcon = kind => kind === 'image' ? 'result' : kind === 'video' ? 'play' : kind === 'json' ? 'file' : 'layers';
  const visibleExamples = sandbox => (sandbox.examples || []).filter(example => example.surface !== 'meshWorkbench');
  const declaredArtifactCount = sandbox => new Set((sandbox.examples || []).flatMap(example => (example.artifacts || []).filter(item => !item.syncOnly).map(item => item.path))).size;
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
    statusRack.innerHTML = chips.map(([label, iconName, online]) => `<span class="status-chip ${statusWord(online)}" title="${escapeHtml(label)} ${statusWord(online)}">${icon(iconName)}<i></i><span>${label}</span></span>`).join('');
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
          <p class="hero-copy">Nine visual workbenches for geometry, robot assets, simulation, and remote compute.</p>
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
        return `<div class="field">${label}<select id="param-${escapeHtml(example.id)}-${escapeHtml(parameter.id)}" ${attributes}>${parameter.choices.map(choice => `<option value="${escapeHtml(choice)}" ${String(choice) === String(parameter.default) ? 'selected' : ''}>${escapeHtml(choice)}</option>`).join('')}</select></div>`;
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

  function visualToolsPanel(sandbox) {
    if (sandbox.viewer?.type !== 'urdf' && !sandbox.meshWorkbench) return '';
    return `<section class="panel visual-tools-panel"><header class="panel-head"><div class="panel-title"><span class="section-icon">${icon('cube')}</span><div><p class="eyebrow">FLOATING WINDOWS</p><h2>3D visual tools</h2></div></div></header><div class="panel-body visual-tool-grid">
      ${sandbox.viewer?.type === 'urdf' ? `<button class="visual-tool" type="button" data-open-workbench="urdf"><span class="visual-tool-icon">${icon('robot')}</span><span><b>URDF workbench</b><small>Orbit the produced robot and tune grouped joints.</small></span>${icon('arrow')}</button>` : ''}
      ${sandbox.meshWorkbench ? `<button class="visual-tool" type="button" data-open-workbench="mesh"><span class="visual-tool-icon">${icon('layers')}</span><span><b>Mesh → STL lab</b><small>Compare each body-part mesh and generated STL, then apply a method.</small></span>${icon('arrow')}</button>` : ''}
    </div></section>`;
  }

  function sandboxView(sandbox) {
    const resultCount = declaredArtifactCount(sandbox);
    const examples = visibleExamples(sandbox);
    app.innerHTML = `
      <a href="#/" class="back-link">${icon('back')}<span>All sandboxes</span></a>
      <section class="sandbox-hero" style="--accent:${escapeHtml(sandbox.accent || '#1f5b4b')}">
        <div class="sandbox-identity"><span class="sandbox-identity-icon">${icon(sandbox.icon || 'cube')}</span><div><p class="eyebrow">${escapeHtml(sandbox.eyebrow || 'ALGORITHM SANDBOX')}</p><h1>${escapeHtml(sandbox.name)}</h1><p class="sandbox-summary">${escapeHtml(sandbox.summary)}</p></div></div>
        <div class="sandbox-stats">
          <div>${icon(targetIcon(sandbox.runtime))}<span>Runtime</span><b>${escapeHtml(sandbox.runtimeLabel || sandbox.runtime || 'Local Mac')}</b></div>
          <div>${icon('play')}<span>Examples</span><b>${examples.length}</b></div>
          <div>${icon('result')}<span>Results</span><b>${resultCount}</b></div>
        </div>
      </section>
      <div class="workspace-grid">
        <div class="workspace-main">
          ${visualToolsPanel(sandbox)}
          ${examples.length ? `<section class="panel"><header class="panel-head"><div class="panel-title"><span class="section-icon">${icon('play')}</span><div><p class="eyebrow">ALLOWLISTED</p><h2>Runnable examples</h2></div></div><span class="panel-count">${examples.length}</span></header><div class="panel-body example-list">${examples.map(example => exampleCard(sandbox, example)).join('')}</div></section>` : milestonePanel(sandbox)}
          ${examples.length && sandbox.milestones?.length ? milestonePanel(sandbox) : ''}
        </div>
        <aside class="workspace-side">${consolePanel(sandbox)}${artifactPanel(sandbox)}</aside>
      </div>`;
    bindSandbox(sandbox);
    const consoleElement = document.querySelector('#job-console');
    if (consoleElement) consoleElement.scrollTop = consoleElement.scrollHeight;
  }

  function bindSandbox(sandbox) {
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
    const sandbox = currentSandbox();
    state.robotViewer?.destroy?.();
    state.meshViewers.forEach(viewer => viewer.destroy?.());
    state.robotViewer = null;
    state.meshViewers = [];
    if (urdfDialog.open && urdfDialog.dataset.sandbox !== sandbox?.id) urdfDialog.close();
    if (meshDialog.open && meshDialog.dataset.sandbox !== sandbox?.id) meshDialog.close();
    if (sandbox) {
      sandboxView(sandbox);
      if (!state.artifacts[sandbox.id]) void loadArtifacts(sandbox);
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
          void loadArtifacts(sandbox);
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
    try {
      const [catalog, status, jobs] = await Promise.all([api('/api/catalog'), api('/api/status'), api('/api/jobs')]);
      state.catalog = catalog.sandboxes;
      state.status = status;
      state.jobs = jobs.jobs;
      renderStatus();
      renderRoute();
      ensureJobPolling();
      if (announce) toast('Mac, TK2, storage, jobs, and results refreshed');
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
  for (const dialog of [urdfDialog, meshDialog]) dialog.addEventListener('click', event => { if (event.target === dialog) dialog.close(); });
  urdfDialog.addEventListener('close', () => { state.robotViewer?.destroy?.(); state.robotViewer = null; urdfDialogContent.innerHTML = ''; });
  meshDialog.addEventListener('close', () => { state.meshViewers.forEach(viewer => viewer.destroy?.()); state.meshViewers = []; meshDialogContent.innerHTML = ''; });
  window.addEventListener('hashchange', renderRoute);
  document.addEventListener('visibilitychange', () => { if (document.visibilityState === 'visible') void refreshAll(); else clearTimeout(state.pollTimer); });
  void refreshAll();
})();
