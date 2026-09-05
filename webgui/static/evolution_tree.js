(function () {
  'use strict';

  const escapeHtml = value => String(value ?? '').replace(/[&<>'"]/g, character => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;' })[character]);
  const formatBytes = value => value == null ? '—' : new Intl.NumberFormat('en', { notation: 'compact', maximumFractionDigits: 1 }).format(value) + 'B';
  const formatMetric = value => typeof value === 'number' ? new Intl.NumberFormat('en', { maximumFractionDigits: 4 }).format(value) : '—';

  function parentsOf(nodes, id) {
    return nodes.get(id)?.parentIds || [];
  }

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
      width: Math.max(760, ...visible.map(node => 210 + depths.get(node.id) * 220)),
      height: Math.max(420, 100 + visible.length * 94),
      point: id => ({ x: 40 + (depths.get(id) || 0) * 220, y: 45 + (lanes.get(id) || 0) * 94 }),
    };
  }

  function metricLabel(node, primaryMetric) {
    const value = node.metrics?.[primaryMetric];
    return value == null ? 'awaiting metric' : `${formatMetric(value)} m`;
  }

  function propertyPane(node, nodes, artifacts, primaryMetric, onPreview) {
    if (!node) return '<div class="evolution-empty">Select a lineage node.</div>';
    const available = new Map((artifacts || []).map(item => [item.path, item]));
    const artifactRows = (node.artifacts || []).map(artifact => {
      const inventory = available.get(artifact.path);
      const canPreview = Boolean(inventory?.exists);
      return `<button class="evolution-artifact" type="button" ${canPreview ? `data-evolution-artifact="${escapeHtml(artifact.path)}" data-kind="${escapeHtml(artifact.kind)}"` : 'disabled'}><span>${escapeHtml(artifact.kind)}</span><b>${escapeHtml(artifact.path.split('/').pop())}</b><small>${canPreview ? `${escapeHtml(inventory.source)} · ${formatBytes(inventory.size)}` : 'metadata only / not synced for preview'}</small></button>`;
    }).join('');
    const metrics = Object.entries(node.metrics || {}).map(([key, value]) => `<div><span>${escapeHtml(key)}</span><b>${formatMetric(value)}</b></div>`).join('');
    const parentLabels = (node.parentIds || []).map(id => nodes.get(id)?.label || id).join(', ') || 'none';
    const storage = node.checkpointStorage;
    const checkpoint = node.checkpointPath ? `<div class="evolution-checkpoint"><span>Checkpoint · ${formatBytes(node.diskBytes)}</span><code>${escapeHtml(node.checkpointPath)}</code><small>${escapeHtml(storage ? `${storage.provider} · ${storage.macHydration} · local preview disabled` : 'storage not recorded')}</small></div>` : '';
    queueMicrotask(() => document.querySelectorAll('[data-evolution-artifact]').forEach(button => button.addEventListener('click', () => onPreview(button.dataset.evolutionArtifact, button.dataset.kind))));
    return `<div class="evolution-properties">
      <div class="evolution-node-heading"><span class="evolution-status ${escapeHtml(node.status)}">${escapeHtml(node.status)}</span><p>${escapeHtml(node.kind)} · step ${escapeHtml(node.step)}</p><h3>${escapeHtml(node.label)}</h3></div>
      <dl><div><dt>Observed result</dt><dd>${escapeHtml(node.result || 'No result recorded')}</dd></div><div><dt>Approach</dt><dd>${escapeHtml(node.approach || 'Not recorded')}</dd></div><div><dt>Parent</dt><dd>${escapeHtml(parentLabels)}</dd></div><div><dt>Source revision</dt><dd><code>${escapeHtml(node.sourceRevision || 'not recorded')}</code></dd></div></dl>
      ${checkpoint}
      <section><h4>Observed metrics</h4><div class="evolution-metrics">${metrics || `<div><span>${escapeHtml(primaryMetric)}</span><b>—</b></div>`}</div></section>
      <section><h4>Artifacts</h4><div class="evolution-artifacts">${artifactRows || '<p>No artifacts recorded for this node.</p>'}</div></section>
    </div>`;
  }

  function comparisonPane(selected, compared, nodes, primaryMetric) {
    if (!selected || !compared || selected.id === compared.id) return '<p class="evolution-compare-empty">Choose another node to compare parameters, metrics, and ancestry.</p>';
    const keys = [...new Set([...Object.keys(selected.metrics || {}), ...Object.keys(compared.metrics || {})])];
    const rows = keys.map(key => {
      const left = selected.metrics?.[key];
      const right = compared.metrics?.[key];
      const delta = typeof left === 'number' && typeof right === 'number' ? left - right : null;
      return `<tr><th>${escapeHtml(key)}</th><td>${formatMetric(left)}</td><td>${formatMetric(right)}</td><td>${delta == null ? '—' : formatMetric(delta)}</td></tr>`;
    }).join('');
    const ancestor = commonAncestor(nodes, selected.id, compared.id);
    return `<div class="evolution-compare-summary"><span>Common ancestor</span><b>${escapeHtml(ancestor?.label || 'none')}</b><span>Primary Δ</span><b>${formatMetric((selected.metrics?.[primaryMetric] ?? 0) - (compared.metrics?.[primaryMetric] ?? 0))}</b></div><div class="evolution-table-wrap"><table><thead><tr><th>Metric</th><th>Selected</th><th>Compare</th><th>Δ</th></tr></thead><tbody>${rows || '<tr><td colspan="4">No shared numeric metrics.</td></tr>'}</tbody></table></div>`;
  }

  function mount(container, options) {
    const data = options.data || {};
    const all = Array.isArray(data.nodes) ? data.nodes : [];
    const nodes = new Map(all.map(node => [node.id, node]));
    const model = container._evolutionModel || { selectedId: data.currentNodeId || all.at(-1)?.id, compareId: '', showAll: false };
    if (!nodes.has(model.selectedId)) model.selectedId = data.currentNodeId || all.at(-1)?.id;
    container._evolutionModel = model;
    const visibleIds = model.showAll ? new Set(all.map(node => node.id)) : new Set(data.defaultVisibleNodeIds || all.map(node => node.id));
    const visible = all.filter(node => visibleIds.has(node.id));
    const graph = layout(visible);
    const selected = nodes.get(model.selectedId);
    const highlighted = ancestorSet(nodes, model.selectedId);
    const edges = visible.flatMap(node => (node.parentIds || []).filter(parent => visibleIds.has(parent)).map(parent => {
      const from = graph.point(parent);
      const to = graph.point(node.id);
      const active = highlighted.has(parent) && highlighted.has(node.id);
      return `<path class="evolution-edge${active ? ' active' : ''}" d="M ${from.x + 168} ${from.y + 30} C ${from.x + 194} ${from.y + 30}, ${to.x - 26} ${to.y + 30}, ${to.x} ${to.y + 30}"/>`;
    })).join('');
    const nodeMarkup = visible.map(node => {
      const point = graph.point(node.id);
      const isSelected = node.id === model.selectedId;
      const path = highlighted.has(node.id);
      return `<g class="evolution-node ${escapeHtml(node.status)}${isSelected ? ' selected' : ''}${path ? ' path' : ''}" data-evolution-node="${escapeHtml(node.id)}" tabindex="0" role="button" aria-label="${escapeHtml(`${node.label}, ${node.status}`)}" transform="translate(${point.x} ${point.y})"><rect width="168" height="61" rx="12"/><circle cx="17" cy="18" r="5"/><text class="node-label" x="30" y="22">${escapeHtml(node.label.slice(0, 23))}</text><text class="node-metric" x="14" y="45">${escapeHtml(metricLabel(node, data.primaryMetric))}</text></g>`;
    }).join('');
    const optionsMarkup = all.map(node => `<option value="${escapeHtml(node.id)}" ${model.compareId === node.id ? 'selected' : ''}>${escapeHtml(node.label)}</option>`).join('');
    container.innerHTML = `<div class="evolution-layout">
      <section class="evolution-main">
        <header class="evolution-toolbar"><div><p class="eyebrow">REAL ARTIFACT LINEAGE</p><b>${escapeHtml(data.lineage || 'unknown')}</b><span>${all.length} nodes · ${data.summary?.failedCount || 0} rejected · ${formatBytes(data.summary?.checkpointBytes)}</span></div><div><button class="button button-light" type="button" data-evolution-fit>Fit current</button>${all.length > visible.length ? `<button class="button" type="button" data-evolution-show-all>Show all ${all.length}</button>` : ''}</div></header>
        <div class="evolution-canvas" tabindex="0"><svg width="${graph.width}" height="${graph.height}" viewBox="0 0 ${graph.width} ${graph.height}" aria-label="Checkpoint evolution tree"><g>${edges}</g><g>${nodeMarkup}</g></svg></div>
        <section class="evolution-compare"><header><div><p class="eyebrow">NODE COMPARISON</p><h3>Compare outcomes</h3></div><label>Against<select data-evolution-compare><option value="">Choose node</option>${optionsMarkup}</select></label></header>${comparisonPane(selected, nodes.get(model.compareId), nodes, data.primaryMetric)}</section>
      </section>
      <aside class="evolution-side">${propertyPane(selected, nodes, options.artifacts, data.primaryMetric, options.onPreview)}</aside>
    </div>`;

    const selectNode = id => { model.selectedId = id; mount(container, options); };
    const ordered = visible.map(node => node.id);
    container.querySelectorAll('[data-evolution-node]').forEach(element => {
      element.addEventListener('click', () => selectNode(element.dataset.evolutionNode));
      element.addEventListener('keydown', event => {
        const id = element.dataset.evolutionNode;
        const index = ordered.indexOf(id);
        let next = null;
        if (event.key === 'ArrowLeft') next = parentsOf(nodes, id).find(parent => visibleIds.has(parent));
        if (event.key === 'ArrowRight') next = visible.find(node => (node.parentIds || []).includes(id))?.id;
        if (event.key === 'ArrowUp') next = ordered[Math.max(0, index - 1)];
        if (event.key === 'ArrowDown') next = ordered[Math.min(ordered.length - 1, index + 1)];
        if (event.key === 'Enter' || event.key === ' ') next = id;
        if (next) { event.preventDefault(); selectNode(next); queueMicrotask(() => container.querySelector(`[data-evolution-node="${CSS.escape(next)}"]`)?.focus()); }
      });
    });
    container.querySelector('[data-evolution-show-all]')?.addEventListener('click', () => { model.showAll = true; mount(container, options); });
    container.querySelector('[data-evolution-fit]')?.addEventListener('click', () => {
      const selectedElement = container.querySelector('.evolution-node.selected');
      selectedElement?.scrollIntoView({ block: 'center', inline: 'center', behavior: matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth' });
    });
    container.querySelector('[data-evolution-compare]')?.addEventListener('change', event => { model.compareId = event.target.value; mount(container, options); });
  }

  window.GeoEvolutionTree = { mount };
})();
