(function () {
  'use strict';

  const escapeHtml = value => String(value ?? '').replace(/[&<>'"]/g, character => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;' })[character]);
  const formatBytes = value => value == null ? '—' : new Intl.NumberFormat('en', { notation: 'compact', maximumFractionDigits: 1 }).format(value) + 'B';
  const formatNumber = value => typeof value === 'number' ? new Intl.NumberFormat('en', { maximumFractionDigits: 4 }).format(value) : '—';
  const compactNumber = value => typeof value === 'number' ? new Intl.NumberFormat('en', { notation: 'compact', maximumFractionDigits: 1 }).format(value) : '—';

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

  function commonAncestor(nodes, left, right) {
    const leftAncestors = ancestorSet(nodes, left);
    let current = right;
    const seen = new Set();
    while (current && !seen.has(current)) {
      if (leftAncestors.has(current)) return nodes.get(current);
      seen.add(current);
      current = parentsOf(nodes, current)[0];
    }
    return null;
  }

  function layout(visible) {
    const byId = new Map(visible.map(node => [node.id, node]));
    const depths = new Map();
    const depth = (node, stack = new Set()) => {
      if (depths.has(node.id)) return depths.get(node.id);
      if (stack.has(node.id)) return 0;
      stack.add(node.id);
      const parentDepths = (node.parentIds || []).map(id => byId.get(id)).filter(Boolean).map(parent => depth(parent, stack) + 1);
      const value = parentDepths.length ? Math.max(...parentDepths) : 0;
      depths.set(node.id, value);
      stack.delete(node.id);
      return value;
    };
    visible.forEach(node => depth(node));
    const lanes = new Map();
    visible.forEach((node, index) => lanes.set(node.id, index));
    return {
      width: Math.max(900, ...visible.map(node => 260 + depths.get(node.id) * 270)),
      height: Math.max(430, 115 + visible.length * 112),
      point: id => ({ x: 38 + (depths.get(id) || 0) * 270, y: 48 + (lanes.get(id) || 0) * 112 }),
    };
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

  function outcomeLabel(node, primaryMetric) {
    if (node.collapsedCount) return `${node.collapsedCount} runs merged`;
    if (node.trainingProgress) return progressLabel(node);
    const value = node.metrics?.[primaryMetric];
    if (value != null) return `${formatNumber(value)} m`;
    const falls = node.metrics?.fall_count;
    const resets = node.metrics?.reset_count;
    if (falls != null || resets != null) return `${falls ?? '—'} falls · ${resets ?? '—'} resets`;
    return node.status === 'running' ? 'current gate' : node.status;
  }

  function milestoneStrip(data, nodes) {
    const milestones = Array.isArray(data.milestones) ? data.milestones : [];
    if (!milestones.length) return '';
    const passed = milestones.filter(item => item.status === 'passed').length;
    const current = milestones.find(item => item.status === 'in_progress');
    const cards = milestones.map(item => {
      const nodeId = `milestone:${item.id}`;
      const selectable = nodes.has(nodeId);
      const status = item.status || 'not_started';
      return `<button class="evolution-milestone ${escapeHtml(status)}" type="button" ${selectable ? `data-evolution-milestone="${escapeHtml(nodeId)}"` : 'disabled'}><span>M${escapeHtml(item.order)}</span><b>${escapeHtml(item.label)}</b><small>${escapeHtml(status.replace('_', ' '))}</small></button>`;
    }).join('');
    return `<section class="evolution-milestone-strip"><header><div><p class="eyebrow">MILESTONE OUTCOMES</p><h3>${passed}/${milestones.length} passed</h3></div><b>${current ? `Current · M${escapeHtml(current.order)} ${escapeHtml(current.label)}` : 'No active milestone'}</b></header><div>${cards}</div></section>`;
  }

  function keyValueGrid(values, empty) {
    const rows = Object.entries(values || {}).map(([key, value]) => `<div><span>${escapeHtml(key)}</span><b>${escapeHtml(typeof value === 'number' ? formatNumber(value) : value)}</b></div>`).join('');
    return rows || `<p class="evolution-muted">${escapeHtml(empty)}</p>`;
  }

  function propertyPane(node, nodes, artifacts, primaryMetric, onPreview) {
    if (!node) return '<div class="evolution-empty">Select a lineage node.</div>';
    const available = new Map((artifacts || []).map(item => [item.path, item]));
    const artifactRows = (node.artifacts || []).map(artifact => {
      const inventory = available.get(artifact.path);
      const canPreview = Boolean(inventory?.exists);
      return `<button class="evolution-artifact" type="button" ${canPreview ? `data-evolution-artifact="${escapeHtml(artifact.path)}" data-kind="${escapeHtml(artifact.kind)}"` : 'disabled'}><span>${escapeHtml(artifact.kind)}</span><b>${escapeHtml(artifact.path.split('/').pop())}</b><small>${canPreview ? `${escapeHtml(inventory.source)} · ${formatBytes(inventory.size)}` : 'metadata only / not synced for preview'}</small></button>`;
    }).join('');
    const parentLabels = (node.parentIds || []).map(id => nodes.get(id)?.label || id).join(', ') || 'none';
    const storage = node.checkpointStorage;
    const checkpointPath = typeof node.checkpointPath === 'string' ? node.checkpointPath : node.checkpointPath?.path || node.checkpointPath?.identity || null;
    const checkpoint = checkpointPath
      ? `<div class="evolution-checkpoint"><span>${node.kind === 'range' ? 'Latest model · merged storage' : 'Model / configuration'} · ${formatBytes(node.diskBytes)}</span><code>${escapeHtml(checkpointPath)}</code>${node.checkpointSha256 ? `<small>SHA · ${escapeHtml(node.checkpointSha256)}</small>` : ''}<small>${escapeHtml(storage ? `${storage.provider} · ${storage.macHydration}` : 'identity only; no policy model')}</small><button type="button" data-evolution-copy-path>Copy path</button></div>`
      : '<div class="evolution-checkpoint evolution-checkpoint-empty"><span>Model / configuration</span><b>No policy model for this node</b><small>Passive or milestone-only node</small></div>';
    const changes = (node.parameterChanges || []).map(item => `<div><span>${escapeHtml(item.key)}</span><b>${escapeHtml(item.from == null ? '—' : item.from)} → ${escapeHtml(item.to)}</b></div>`).join('');
    const failures = (node.failureBreakdown || []).map(item => `<li><b>${escapeHtml(item.count)}×</b><span>${escapeHtml(item.result)}</span></li>`).join('');
    const milestoneBreakdown = (node.milestoneBreakdown || []).map(item => `<li><b>${escapeHtml(item.count)}×</b><span>${escapeHtml(item.milestone)}</span></li>`).join('');
    queueMicrotask(() => document.querySelectorAll('[data-evolution-artifact]').forEach(button => button.addEventListener('click', () => onPreview(button.dataset.evolutionArtifact, button.dataset.kind))));
    queueMicrotask(() => document.querySelector('[data-evolution-copy-path]')?.addEventListener('click', async () => { if (checkpointPath) await navigator.clipboard.writeText(checkpointPath); }));
    return `<div class="evolution-properties">
      <div class="evolution-node-heading"><span class="evolution-status ${escapeHtml(node.status)}">${escapeHtml(node.status)}</span><p>${escapeHtml(node.kind)} · ${escapeHtml(progressLabel(node))}</p><h3>${escapeHtml(node.label)}</h3></div>
      <section class="evolution-change"><h4>Changed in this node</h4><p>${escapeHtml(node.changeSummary || node.approach || 'No parameter change recorded')}</p>${changes ? `<div class="evolution-change-grid">${changes}</div>` : ''}</section>
      <dl><div><dt>Milestone</dt><dd>${escapeHtml(node.milestoneId || 'lineage setup')}</dd></div><div><dt>Observed result</dt><dd>${escapeHtml(node.result || 'No result recorded')}</dd></div><div><dt>Parent</dt><dd>${escapeHtml(parentLabels)}</dd></div><div><dt>Source revision</dt><dd><code>${escapeHtml(node.sourceRevision || 'not recorded')}</code></dd></div></dl>
      ${checkpoint}
      ${node.memberNodeIds?.length ? `<button class="button evolution-inspect-runs" type="button" data-evolution-expand-group="${escapeHtml(node.id)}">Inspect ${node.memberNodeIds.length} runs</button>` : ''}
      <section><h4>Run progress / steps</h4><div class="evolution-metrics">${keyValueGrid(node.trainingProgress, 'No step metadata recorded.')}</div></section>
      <section><h4>Parameters</h4><div class="evolution-parameters">${keyValueGrid(node.experimentParameters, 'No experiment parameters recorded.')}</div></section>
      <section><h4>Observed metrics</h4><div class="evolution-metrics">${keyValueGrid(node.metrics, `${primaryMetric} not recorded.`)}</div></section>
      ${milestoneBreakdown ? `<section><h4>Merged milestone stages</h4><ol class="evolution-failures evolution-stage-breakdown">${milestoneBreakdown}</ol></section>` : ''}
      ${failures ? `<section><h4>Merged failure outcomes</h4><ol class="evolution-failures">${failures}</ol></section>` : ''}
      <section><h4>Artifacts</h4><div class="evolution-artifacts">${artifactRows || '<p class="evolution-muted">No artifacts recorded for this node.</p>'}</div></section>
    </div>`;
  }

  function comparisonPane(selected, compared, nodes, primaryMetric) {
    if (!selected || !compared || selected.id === compared.id) return '<p class="evolution-compare-empty">Choose another visible node to compare parameters, metrics, and ancestry.</p>';
    const fields = (type, leftValues, rightValues) => [...new Set([...Object.keys(leftValues || {}), ...Object.keys(rightValues || {})])].map(key => {
      const left = leftValues?.[key];
      const right = rightValues?.[key];
      const delta = typeof left === 'number' && typeof right === 'number' ? left - right : null;
      return `<tr><th>${escapeHtml(type)} · ${escapeHtml(key)}</th><td>${escapeHtml(left ?? '—')}</td><td>${escapeHtml(right ?? '—')}</td><td>${delta == null ? (left === right ? 'same' : 'changed') : formatNumber(delta)}</td></tr>`;
    }).join('');
    const rows = fields('parameter', selected.experimentParameters, compared.experimentParameters) + fields('metric', selected.metrics, compared.metrics);
    const ancestor = commonAncestor(nodes, selected.id, compared.id);
    return `<div class="evolution-compare-summary"><span>Common ancestor</span><b>${escapeHtml(ancestor?.label || 'none')}</b><span>Primary Δ</span><b>${formatNumber((selected.metrics?.[primaryMetric] ?? 0) - (compared.metrics?.[primaryMetric] ?? 0))}</b></div><div class="evolution-table-wrap"><table><thead><tr><th>Field</th><th>Selected</th><th>Compare</th><th>Δ</th></tr></thead><tbody>${rows || '<tr><td colspan="4">No parameters or metrics to compare.</td></tr>'}</tbody></table></div>`;
  }

  function mount(container, options) {
    const data = options.data || {};
    const raw = Array.isArray(data.nodes) ? data.nodes : [];
    const overview = Array.isArray(data.overviewNodes) && data.overviewNodes.length ? data.overviewNodes : raw.filter(node => (data.defaultVisibleNodeIds || []).includes(node.id));
    const combined = new Map([...raw, ...overview].map(node => [node.id, node]));
    const currentPath = ancestorSet(combined, data.currentNodeId);
    const orderedOverview = overview.map((node, index) => ({ node, index })).sort((left, right) => {
      const pathRank = Number(!currentPath.has(left.node.id)) - Number(!currentPath.has(right.node.id));
      return pathRank || left.index - right.index;
    }).map(item => item.node);
    const model = container._evolutionModel || { selectedId: data.currentNodeId || raw.at(-1)?.id, compareId: '', mode: 'overview', groupId: '' };
    container._evolutionModel = model;
    let visible = orderedOverview;
    if (model.mode === 'all') visible = raw;
    if (model.mode === 'group') {
      const group = orderedOverview.find(node => node.id === model.groupId);
      const members = new Set(group?.memberNodeIds || []);
      const parentPath = ancestorSet(combined, group?.parentIds?.[0]);
      visible = [...orderedOverview.filter(node => parentPath.has(node.id)), ...raw.filter(node => members.has(node.id))];
    }
    const visibleIds = new Set(visible.map(node => node.id));
    if (!combined.has(model.selectedId) || !visibleIds.has(model.selectedId)) model.selectedId = data.currentNodeId || visible.at(-1)?.id;
    const selected = combined.get(model.selectedId);
    const graph = layout(visible);
    const highlighted = ancestorSet(combined, model.selectedId);
    const milestoneOrder = new Map((data.milestones || []).map(item => [item.id, item.order]));
    const edges = visible.flatMap(node => (node.parentIds || []).filter(parent => visibleIds.has(parent)).map(parent => {
      const from = graph.point(parent);
      const to = graph.point(node.id);
      const active = highlighted.has(parent) && highlighted.has(node.id);
      return `<path class="evolution-edge${active ? ' active' : ''}" d="M ${from.x + 214} ${from.y + 41} C ${from.x + 244} ${from.y + 41}, ${to.x - 30} ${to.y + 41}, ${to.x} ${to.y + 41}"/>`;
    })).join('');
    const nodeMarkup = visible.map(node => {
      const point = graph.point(node.id);
      const isSelected = node.id === model.selectedId;
      const path = highlighted.has(node.id);
      const milestone = milestoneOrder.get(node.milestoneId);
      const eyebrow = `${milestone ? `M${milestone} · ` : ''}${node.kind}${node.step != null ? ` · seq ${node.step}` : ''}`;
      return `<g class="evolution-node ${escapeHtml(node.status)}${node.kind === 'range' ? ' merged' : ''}${isSelected ? ' selected' : ''}${path ? ' path' : ''}" data-evolution-node="${escapeHtml(node.id)}" tabindex="0" role="button" aria-label="${escapeHtml(`${node.label}, ${node.status}`)}" transform="translate(${point.x} ${point.y})"><rect width="214" height="82" rx="13"/><circle cx="17" cy="16" r="5"/><text class="node-kind" x="29" y="19">${escapeHtml(eyebrow.slice(0, 34))}</text><text class="node-label" x="14" y="43">${escapeHtml(node.label.slice(0, 31))}</text><text class="node-metric" x="14" y="66">${escapeHtml(outcomeLabel(node, data.primaryMetric).slice(0, 39))}</text>${node.diskBytes ? `<text class="node-size" x="200" y="66" text-anchor="end">${escapeHtml(formatBytes(node.diskBytes))}</text>` : ''}</g>`;
    }).join('');
    const compareOptions = visible.map(node => `<option value="${escapeHtml(node.id)}" ${model.compareId === node.id ? 'selected' : ''}>${escapeHtml(node.label)}</option>`).join('');
    const modeButton = model.mode === 'overview' ? '' : '<button class="button button-light" type="button" data-evolution-overview>Smart overview</button>';
    container.innerHTML = `<div class="evolution-shell">
      ${milestoneStrip(data, combined)}
      <div class="evolution-layout">
        <section class="evolution-main">
          <header class="evolution-toolbar"><div><p class="eyebrow">REAL ARTIFACT LINEAGE</p><b>Clean milestone lineage</b><span>${data.summary?.overviewNodeCount || overview.length} overview · ${raw.length} recorded · ${data.summary?.failedGroupCount || 0} merged groups · ${formatBytes(data.summary?.checkpointBytes)}</span></div><div>${modeButton}<button class="button button-light" type="button" data-evolution-fit>Fit current</button>${model.mode !== 'all' && raw.length > overview.length ? `<button class="button" type="button" data-evolution-show-all>All ${raw.length}</button>` : ''}</div></header>
          <div class="evolution-canvas" tabindex="0"><svg width="${graph.width}" height="${graph.height}" viewBox="0 0 ${graph.width} ${graph.height}" aria-label="Checkpoint evolution tree"><g>${edges}</g><g>${nodeMarkup}</g></svg></div>
          <section class="evolution-compare"><header><div><p class="eyebrow">NODE COMPARISON</p><h3>What changed?</h3></div><label>Against<select data-evolution-compare><option value="">Choose visible node</option>${compareOptions}</select></label></header>${comparisonPane(selected, combined.get(model.compareId), combined, data.primaryMetric)}</section>
        </section>
        <aside class="evolution-side">${propertyPane(selected, combined, options.artifacts, data.primaryMetric, options.onPreview)}</aside>
      </div>
    </div>`;

    const centerSelected = behavior => {
      const canvas = container.querySelector('.evolution-canvas');
      const point = graph.point(model.selectedId);
      if (!canvas || !point) return;
      canvas.scrollTo({
        left: Math.max(0, point.x + 107 - canvas.clientWidth / 2),
        top: Math.max(0, point.y + 41 - canvas.clientHeight / 2),
        behavior,
      });
    };
    queueMicrotask(() => centerSelected('auto'));
    const selectNode = id => { model.selectedId = id; mount(container, options); };
    const ordered = visible.map(node => node.id);
    container.querySelectorAll('[data-evolution-node]').forEach(element => {
      element.addEventListener('click', () => selectNode(element.dataset.evolutionNode));
      element.addEventListener('keydown', event => {
        const id = element.dataset.evolutionNode;
        const index = ordered.indexOf(id);
        let next = null;
        if (event.key === 'ArrowLeft') next = parentsOf(combined, id).find(parent => visibleIds.has(parent));
        if (event.key === 'ArrowRight') next = visible.find(node => (node.parentIds || []).includes(id))?.id;
        if (event.key === 'ArrowUp') next = ordered[Math.max(0, index - 1)];
        if (event.key === 'ArrowDown') next = ordered[Math.min(ordered.length - 1, index + 1)];
        if (event.key === 'Enter' || event.key === ' ') next = id;
        if (next) { event.preventDefault(); selectNode(next); queueMicrotask(() => container.querySelector(`[data-evolution-node="${CSS.escape(next)}"]`)?.focus()); }
      });
    });
    container.querySelectorAll('[data-evolution-milestone]').forEach(button => button.addEventListener('click', () => selectNode(button.dataset.evolutionMilestone)));
    container.querySelector('[data-evolution-expand-group]')?.addEventListener('click', event => { model.mode = 'group'; model.groupId = event.currentTarget.dataset.evolutionExpandGroup; model.selectedId = combined.get(model.groupId)?.memberNodeIds?.at(-1); mount(container, options); });
    container.querySelector('[data-evolution-show-all]')?.addEventListener('click', () => { model.mode = 'all'; model.groupId = ''; mount(container, options); });
    container.querySelector('[data-evolution-overview]')?.addEventListener('click', () => { model.mode = 'overview'; model.groupId = ''; model.selectedId = data.currentNodeId; mount(container, options); });
    container.querySelector('[data-evolution-fit]')?.addEventListener('click', () => centerSelected(matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth'));
    container.querySelector('[data-evolution-compare]')?.addEventListener('change', event => { model.compareId = event.target.value; mount(container, options); });
  }

  window.GeoEvolutionTree = { mount };
})();
