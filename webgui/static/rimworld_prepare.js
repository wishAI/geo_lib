(function () {
  'use strict';

  const SKILLS = ['Animals', 'Artistic', 'Construction', 'Cooking', 'Crafting', 'Intellectual', 'Medical', 'Melee', 'Mining', 'Plants', 'Shooting', 'Social'];
  const TRAITS = ['Industrious', 'FastLearner', 'Kind', 'Sanguine', 'Tough', 'Jogger', 'GreatMemory', 'Nimble', 'IronWilled'];
  const DIRECTIONS = ['south', 'east', 'north', 'west'];
  const HEAD = { south: [0, -45], east: [15, -45], north: [0, -45], west: [-15, -45] };
  const EARS = {
    south: [[0, -57], [2, -57]], east: [[2, -54], [2, -54]],
    north: [[0, -57], [0, -57]], west: [[-2, -54], [-2, -54]],
  };
  const TAIL = { south: [72, 38], east: [5, -10], north: [0, 2], west: [-5, -10] };
  const esc = value => String(value ?? '').replace(/[&<>'"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;' })[c]);
  const clone = value => JSON.parse(JSON.stringify(value));

  async function api(url, options = {}) {
    const response = await fetch(url, { cache: 'no-store', ...options, headers: { 'Content-Type': 'application/json', ...(options.headers || {}) } });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(payload.error || `Request failed (${response.status})`);
    return payload;
  }

  function mount(root, options) {
    const sandbox = options.sandbox;
    const designer = options.designer || {};
    const state = { workspace: null, catalog: null, skinImport: null, selected: '', tab: sessionStorage.getItem('rimworld-prepare-tab') || 'character', assetFilter: 'all', search: '', job: null, uploading: false, destroyed: false, images: new Map() };
    root.innerHTML = '<div class="rw-loading"><span></span><b>Reading the installed Yuran 1.6 assets…</b></div>';

    const assetUrl = (path, token = '') => `/api/rimworld/asset?sandbox=${encodeURIComponent(sandbox)}&path=${encodeURIComponent(path)}${token ? `&v=${encodeURIComponent(token)}` : ''}`;
    const byId = id => state.catalog.assets.find(item => item.id === id);
    const current = () => state.workspace.characters.find(item => item.id === state.selected) || state.workspace.characters[0];
    const fileFor = (asset, direction) => {
      if (!asset) return null;
      if (asset.files[direction]) return { path: asset.files[direction], mirror: false };
      if (direction === 'west' && asset.files.east) return { path: asset.files.east, mirror: true };
      return asset.files.south ? { path: asset.files.south, mirror: false } : null;
    };
    const loadImage = path => {
      if (!state.images.has(path)) state.images.set(path, new Promise((resolve, reject) => {
        const image = new Image(); image.onload = () => resolve(image); image.onerror = reject; image.src = assetUrl(path);
      }));
      return state.images.get(path);
    };
    async function drawAsset(ctx, asset, direction, offset = [0, 0]) {
      const file = fileFor(asset, direction);
      if (!file) return;
      try {
        const image = await loadImage(file.path);
        ctx.save();
        if (file.mirror) { ctx.translate(256 + offset[0], offset[1]); ctx.scale(-1, 1); ctx.drawImage(image, 0, 0, 256, 256); }
        else ctx.drawImage(image, offset[0], offset[1], 256, 256);
        ctx.restore();
      } catch (_) { /* a missing optional texture is a transparent layer */ }
    }
    function apparelAsset(item) {
      const prefix = item.wornGraphicPath.replace(/^Yuran\/Yuranlike\//, '');
      const body = item.layers.includes('Overhead') ? prefix : `${prefix}_Thin`;
      return byId(body) || byId(prefix);
    }
    async function drawImportedPart(ctx, kind, direction, offset = [0, 0]) {
      const chosen = direction === 'west' ? 'east' : direction;
      const base = kind === 'face' ? 'Heads/Female_YR_head' : 'Bodies/Naked_Thin';
      const path = `outputs/imports/DragonSkin/${base}_${chosen}.png`;
      try {
        const image = await loadImage(path);
        ctx.save();
        if (direction === 'west') { ctx.translate(256 + offset[0], offset[1]); ctx.scale(-1, 1); ctx.drawImage(image, 0, 0, 256, 256); }
        else ctx.drawImage(image, offset[0], offset[1], 256, 256);
        ctx.restore();
        return true;
      } catch (_) { return false; }
    }
    async function renderPawn() {
      const canvas = root.querySelector('#rw-pawn-canvas');
      if (!canvas || !state.catalog) return;
      const person = current();
      const direction = person.direction || 'south';
      const ctx = canvas.getContext('2d');
      ctx.clearRect(0, 0, 256, 256);
      await drawAsset(ctx, byId('tail/YR_tail'), direction, TAIL[direction]);
      if (!(person.dragonSkinInstalled && await drawImportedPart(ctx, 'body', direction))) await drawAsset(ctx, byId('Bodies/Naked_Thin'), direction);
      const worn = person.apparel.map(def => state.catalog.apparel.find(item => item.defName === def)).filter(Boolean);
      for (const item of worn.filter(item => !item.layers.includes('Overhead'))) await drawAsset(ctx, apparelAsset(item), direction);
      if (!(person.dragonSkinInstalled && await drawImportedPart(ctx, 'face', direction, HEAD[direction]))) await drawAsset(ctx, byId('Heads/Female_YR_head'), direction, HEAD[direction]);
      await drawAsset(ctx, byId('Ear/L/YR_earL'), direction, EARS[direction][0]);
      await drawAsset(ctx, byId('Ear/R/YR_earR'), direction, EARS[direction][1]);
      await drawAsset(ctx, byId('AddonHair/YR_hair'), direction, HEAD[direction]);
      for (const item of worn.filter(item => item.layers.includes('Overhead'))) await drawAsset(ctx, apparelAsset(item), direction, HEAD[direction]);
    }

    function characterList() {
      return `<aside class="rw-roster"><header><b>Colonists</b><button type="button" data-add-character title="Duplicate selected colonist">＋</button></header>${state.workspace.characters.map(person => `<button type="button" class="rw-roster-item ${person.id === state.selected ? 'active' : ''}" data-character="${esc(person.id)}"><span>${esc(person.name.nick || person.name.first)}</span><small>${esc(person.biologicalAge)} · Dragon Yuran</small></button>`).join('')}</aside>`;
    }
    function identityPane(person) {
      return `<section class="rw-fields"><h3>Identity</h3><div class="rw-name-grid">
        <label>First<input data-path="name.first" value="${esc(person.name.first)}"></label><label>Nickname<input data-path="name.nick" value="${esc(person.name.nick)}"></label><label>Last<input data-path="name.last" value="${esc(person.name.last)}"></label>
      </div><div class="rw-name-grid"><label>Biological age<input type="number" min="13" max="200" data-path="biologicalAge" value="${esc(person.biologicalAge)}"></label><label>Chronological age<input type="number" min="13" max="10000" data-path="chronologicalAge" value="${esc(person.chronologicalAge)}"></label><label>Race<input value="Dragon Yuran" disabled></label></div>
      <h3>Skills</h3><div class="rw-skill-grid">${SKILLS.map(name => `<label><span>${esc(name)}</span><input type="number" min="0" max="20" data-skill="${name}" value="${person.skills[name].level}"><select data-passion="${name}">${['None', 'Minor', 'Major'].map(value => `<option ${person.skills[name].passion === value ? 'selected' : ''}>${value}</option>`).join('')}</select></label>`).join('')}</div>
      <h3>Traits</h3><div class="rw-check-grid">${TRAITS.map(name => `<label><input type="checkbox" data-trait="${name}" ${person.traits.some(item => item.def === name) ? 'checked' : ''}><span>${esc(name.replace(/([A-Z])/g, ' $1').trim())}</span></label>`).join('')}</div>
      <h3>Health and worn apparel</h3><div class="rw-check-grid"><label class="rw-dragon-check"><input type="checkbox" data-dragon-skin ${person.dragonSkinInstalled ? 'checked' : ''}><span>Dragon Skin installed</span></label>${state.workspace.mod.selectedApparel.map(def => { const item = state.catalog.apparel.find(value => value.defName === def); return `<label><input type="checkbox" data-worn="${esc(def)}" ${person.apparel.includes(def) ? 'checked' : ''}><span>${esc(item?.label || def)}</span></label>`; }).join('')}</div></section>`;
    }
    function characterTab() {
      const person = current();
      return `<div class="rw-character-layout">${characterList()}<section class="rw-preview"><div class="rw-canvas-shell"><canvas id="rw-pawn-canvas" width="256" height="256" aria-label="Layered Yuran pawn preview"></canvas></div><div class="rw-directions" role="group" aria-label="View direction">${DIRECTIONS.map(value => `<button type="button" data-direction="${value}" class="${person.direction === value ? 'active' : ''}">${value}</button>`).join('')}</div><p>Uses RimWorld 1.6 Thin head offset plus the Yuran 1.6 HAR addon offsets. This is a deterministic texture composite, not a Unity screenshot.</p></section>${identityPane(person)}</div>`;
    }
    function apparelTab() {
      const selected = new Set(state.workspace.mod.selectedApparel);
      return `<div class="rw-tab-head"><div><h2>Clothing to include</h2><p>Checked items are copied into Dragon Yuran and receive recipes at both vanilla tailoring benches.</p></div><b>${selected.size} / ${state.catalog.apparel.length}</b></div><div class="rw-apparel-grid">${state.catalog.apparel.map(item => `<label class="rw-apparel-card ${selected.has(item.defName) ? 'selected' : ''}"><input type="checkbox" data-include-apparel="${esc(item.defName)}" ${selected.has(item.defName) ? 'checked' : ''}><img loading="lazy" src="${assetUrl(item.preview)}" alt=""><span><b>${esc(item.label)}</b><small>${esc(item.defName)}</small><em>${esc(item.layers.join(' · '))}</em></span></label>`).join('')}</div>`;
    }
    function assetsTab() {
      const categories = ['all', ...new Set(state.catalog.assets.map(item => item.category))];
      const query = state.search.toLowerCase();
      const assets = state.catalog.assets.filter(item => (state.assetFilter === 'all' || item.category === state.assetFilter) && (!query || `${item.id} ${item.category}`.toLowerCase().includes(query)));
      return `<div class="rw-tab-head"><div><h2>Race asset inventory</h2><p>Only race-facing textures are listed; fiction, buildings, factions, weapons, research, Miko and Black Snake content are excluded from the output mod.</p></div><b>${assets.length} / ${state.catalog.assets.length}</b></div><div class="rw-asset-toolbar"><div>${categories.map(category => `<button type="button" data-asset-filter="${esc(category)}" class="${state.assetFilter === category ? 'active' : ''}">${esc(category)}</button>`).join('')}</div><input type="search" data-asset-search value="${esc(state.search)}" placeholder="Filter texture paths"></div><div class="rw-asset-grid">${assets.map(item => `<article><img loading="lazy" src="${assetUrl(item.preview)}" alt="${esc(item.label)}"><span><b>${esc(item.label)}</b><small>${esc(item.id)}</small><em>${esc(item.directions.join(' · ') || 'icon')}</em></span></article>`).join('')}</div>`;
    }
    function dragonTab() {
      const template = `algorithms/${sandbox}/${state.catalog.sheets.skin}`;
      const imported = state.skinImport;
      const stages = imported?.stages || [
        { id: 'upload', order: 1, label: 'Upload generated sheet', status: 'pending' },
        { id: 'restore', order: 2, label: 'Restore joined views', status: 'pending' },
        { id: 'mask', order: 3, label: 'Project body/head masks', status: 'pending' },
        { id: 'compare', order: 4, label: 'Compare originals', status: 'pending' },
      ];
      const comparisons = imported?.files?.map(item => `<article class="rw-compare-card"><header><div><b>${esc(item.label)}</b><small>${esc(item.gameOutput)}</small></div><em>${esc(item.metrics.changedPercent)}% changed</em></header><div class="rw-triptych"><figure><img src="${assetUrl(item.original)}" alt="Original ${esc(item.label)}"><figcaption>Original</figcaption></figure><figure><img src="${assetUrl(item.generated, imported.run.id)}" alt="Generated ${esc(item.label)}"><figcaption>Generated</figcaption></figure><figure class="diff"><img src="${assetUrl(item.diff, imported.run.id)}" alt="Difference ${esc(item.label)}"><figcaption>Difference ×4</figcaption></figure></div><footer><span>${esc(item.metrics.changedPixels)} / ${esc(item.metrics.maskedPixels)} silhouette pixels</span><span>mean Δ ${esc(item.metrics.meanDelta)}</span></footer></article>`).join('') || '';
      return `<div class="rw-dragon-workspace"><section class="rw-template-panel"><p class="rw-kicker">IMG2IMG SOURCE</p><h2>Joined body + hairless-face sheet</h2><p>Each direction is one continuous, engine-aligned character from feet through neck to face. South, East and North fill most of three 640×640 cells; RimWorld mirrors East for West. Hair and ears are intentionally absent. On import, the joined result is projected back into six separate body/head game textures.</p><img src="/api/artifact?path=${encodeURIComponent(template)}" alt="Joined Yuran body and hairless face sheet"><div class="rw-inline-actions"><a class="button" href="/api/artifact?path=${encodeURIComponent(template)}" download="dragon_yuran_joined_body_face_img2img.png">Download PNG</a></div></section><aside class="rw-upload-panel"><p class="rw-kicker">GENERATED RESULT</p><h2>Upload joined dragon-scale bodies</h2><label class="rw-drop-zone ${state.uploading ? 'uploading' : ''}" data-skin-drop><input type="file" data-import-skin accept="image/png" hidden><strong>${state.uploading ? 'Importing and comparing…' : 'Drop generated PNG here'}</strong><span>or click to choose the unchanged-size PNG</span></label><h3>Automatic pipeline</h3><div class="rw-skin-pipeline">${stages.map(stage => `<span class="${stage.status}"><i>${stage.order}</i><b>${esc(stage.label)}</b></span>`).join('')}</div><div class="rw-import-state">${imported ? `<b>${esc(imported.run.label)}</b><span>${esc(imported.summary.assetCount)} raw assets · ${esc(imported.summary.changedPercent)}% silhouette pixels changed</span>` : '<span>No generated sheet imported yet. Original assets remain untouched.</span>'}</div><h3>Activation rule</h3><p>The generated body and face textures activate only while the custom <b>Dragon Skin</b> implant is installed. Hair and ears remain separate original layers.</p></aside>${imported ? `<section class="rw-comparison"><div class="rw-tab-head"><div><p class="rw-kicker">LIVE ARTIFACT COMPARISON</p><h2>Original → generated → difference</h2><p>Observed pixel metrics are measured after the joined view is projected back through each original Yuran alpha mask. The ×4 difference view makes subtle scale changes visible.</p></div><b>${esc(imported.files.length)} assets</b></div><div class="rw-compare-grid">${comparisons}</div></section>` : ''}</div>`;
    }
    function deployTab() {
      return `<div class="rw-deploy"><section><p class="rw-kicker">BUILD CONTENT</p><h2>Clean Dragon Yuran package</h2><ul><li>Race: DragonYuran_Race + colonist kind</li><li>Selected apparel: ${state.workspace.mod.selectedApparel.length}, craftable at Hand/Electric Tailoring Bench</li><li>Prepare Carefully v5 files: ${state.workspace.characters.length}</li><li>Excluded: fiction, factions, buildings, weapons, research, Miko, Black Snake</li></ul></section><section><p class="rw-kicker">TK2 DESTINATION</p><h2>Deploy without launching</h2><code>RimWorld/Mods/DragonYuran</code><code>SaveData/PrepareCarefully/*.pcc</code><p>Deployment does not start RimWorld and does not change the active mod list.</p><div class="rw-deploy-actions"><button class="button button-light" type="button" data-action="build">Build locally</button><button class="button" type="button" data-action="deploy">Build & deploy to TK2</button></div></section></div>`;
    }
    function tabBody() {
      if (state.tab === 'apparel') return apparelTab();
      if (state.tab === 'assets') return assetsTab();
      if (state.tab === 'dragon') return dragonTab();
      if (state.tab === 'deploy') return deployTab();
      return characterTab();
    }
    function render() {
      if (state.destroyed) return;
      const source = state.catalog.source;
      root.innerHTML = `<div class="rw-shell"><div class="rw-pipeline"><span class="complete"><i>1</i><b>Installed source</b><small>${state.catalog.assets.length} race assets</small></span><span class="complete"><i>2</i><b>Race core</b><small>Yuran → Dragon Yuran</small></span><span class="complete"><i>3</i><b>Characters</b><small>${state.workspace.characters.length} Prepare Carefully</small></span><span class="${state.job?.status === 'succeeded' && state.job?.example === designer.deployExample ? 'complete' : ''}"><i>4</i><b>TK2</b><small>${state.job?.status || 'ready to deploy'}</small></span></div><nav class="rw-tabs" aria-label="RimWorld Prepare sections">${[['character', 'Character'], ['apparel', 'Clothing'], ['assets', 'Assets'], ['dragon', 'Dragon skin'], ['deploy', 'Build & deploy']].map(([id, label]) => `<button type="button" data-tab="${id}" class="${state.tab === id ? 'active' : ''}">${label}</button>`).join('')}<span></span><button class="rw-save" type="button" data-action="save">Save workspace</button><button class="rw-sync" type="button" data-action="sync">Refresh from TK2</button></nav><main class="rw-body">${tabBody()}</main><footer class="rw-status"><span class="${state.job?.status || 'ready'}"><i></i>${esc(state.job ? `${state.job.status}: ${state.job.example}` : 'Workspace ready')}</span><small>Workshop ${esc(source.workshopId)} · RimWorld ${esc(designer.gameVersion || source.version)} · game not launched</small></footer></div>`;
      bind();
      if (state.tab === 'character') void renderPawn();
    }
    function setPath(person, path, value) {
      const parts = path.split('.'); let target = person;
      while (parts.length > 1) target = target[parts.shift()];
      target[parts[0]] = value;
    }
    async function save(announce = true) {
      const result = await api('/api/rimworld/workspace', { method: 'POST', body: JSON.stringify({ sandbox, workspace: state.workspace }) });
      if (announce) setStatus(`Saved ${result.path}`);
      return result;
    }
    function setStatus(message, error = false) {
      const status = root.querySelector('.rw-status small');
      if (status) { status.textContent = message; status.classList.toggle('error', error); }
    }
    async function run(example) {
      await save(false);
      state.job = await api('/api/jobs', { method: 'POST', body: JSON.stringify({ sandbox, example }) });
      render();
      while (!state.destroyed && ['queued', 'running', 'cancelling'].includes(state.job.status)) {
        await new Promise(resolve => setTimeout(resolve, 1200));
        const jobs = await api('/api/jobs');
        state.job = jobs.jobs.find(item => item.id === state.job.id) || state.job;
        const indicator = root.querySelector('.rw-status span');
        if (indicator) { indicator.className = state.job.status; indicator.innerHTML = `<i></i>${esc(`${state.job.status}: ${state.job.example}`)}`; }
      }
      if (!state.destroyed) { setStatus(state.job.status === 'succeeded' ? 'Completed without launching RimWorld.' : (state.job.log || `${state.job.status}`), state.job.status !== 'succeeded'); render(); }
    }
    async function importSkin(file) {
      if (!file) return;
      state.uploading = true; render();
      try {
        const data = await new Promise((resolve, reject) => { const reader = new FileReader(); reader.onload = () => resolve(String(reader.result).split(',')[1]); reader.onerror = reject; reader.readAsDataURL(file); });
        const result = await api('/api/rimworld/import-skin', { method: 'POST', body: JSON.stringify({ sandbox, name: file.name, data }) });
        state.images.clear(); state.skinImport = result.result; state.uploading = false; render();
        setStatus(`Imported, restored and compared ${result.result.files.length} raw body/face textures.`);
      } catch (error) {
        state.uploading = false; render(); throw error;
      }
    }
    function bind() {
      root.querySelectorAll('[data-tab]').forEach(button => button.addEventListener('click', () => { state.tab = button.dataset.tab; sessionStorage.setItem('rimworld-prepare-tab', state.tab); render(); }));
      root.querySelectorAll('[data-character]').forEach(button => button.addEventListener('click', () => { state.selected = button.dataset.character; state.workspace.selectedCharacterId = state.selected; render(); }));
      root.querySelector('[data-add-character]')?.addEventListener('click', () => {
        const person = clone(current()); person.id = `dragon-yuran-${Date.now()}`; person.name.nick = `${person.name.nick} II`; state.workspace.characters.push(person); state.selected = person.id; state.workspace.selectedCharacterId = person.id; render();
      });
      root.querySelectorAll('[data-direction]').forEach(button => button.addEventListener('click', () => { current().direction = button.dataset.direction; render(); }));
      root.querySelectorAll('[data-path]').forEach(input => input.addEventListener('input', () => { setPath(current(), input.dataset.path, input.type === 'number' ? Number(input.value) : input.value); }));
      root.querySelectorAll('[data-skill]').forEach(input => input.addEventListener('input', () => { current().skills[input.dataset.skill].level = Number(input.value); }));
      root.querySelectorAll('[data-passion]').forEach(input => input.addEventListener('change', () => { current().skills[input.dataset.passion].passion = input.value; }));
      root.querySelectorAll('[data-trait]').forEach(input => input.addEventListener('change', () => { const traits = current().traits; const index = traits.findIndex(item => item.def === input.dataset.trait); if (input.checked && index < 0) traits.push({ def: input.dataset.trait, degree: 0 }); if (!input.checked && index >= 0) traits.splice(index, 1); }));
      root.querySelector('[data-dragon-skin]')?.addEventListener('change', event => { current().dragonSkinInstalled = event.target.checked; void renderPawn(); });
      root.querySelectorAll('[data-worn]').forEach(input => input.addEventListener('change', () => { const items = current().apparel; const def = input.dataset.worn; if (input.checked && !items.includes(def)) items.push(def); if (!input.checked) current().apparel = items.filter(item => item !== def); void renderPawn(); }));
      root.querySelectorAll('[data-include-apparel]').forEach(input => input.addEventListener('change', () => { const selected = state.workspace.mod.selectedApparel; const def = input.dataset.includeApparel; if (input.checked && !selected.includes(def)) selected.push(def); if (!input.checked) { state.workspace.mod.selectedApparel = selected.filter(item => item !== def); state.workspace.characters.forEach(person => { person.apparel = person.apparel.filter(item => item !== def); }); } render(); }));
      root.querySelectorAll('[data-asset-filter]').forEach(button => button.addEventListener('click', () => { state.assetFilter = button.dataset.assetFilter; render(); }));
      root.querySelector('[data-asset-search]')?.addEventListener('change', event => { state.search = event.target.value; render(); });
      root.querySelector('[data-import-skin]')?.addEventListener('change', event => void importSkin(event.target.files[0]).catch(error => setStatus(error.message, true)));
      const drop = root.querySelector('[data-skin-drop]');
      if (drop) {
        ['dragenter', 'dragover'].forEach(name => drop.addEventListener(name, event => { event.preventDefault(); drop.classList.add('dragging'); }));
        ['dragleave', 'drop'].forEach(name => drop.addEventListener(name, event => { event.preventDefault(); drop.classList.remove('dragging'); }));
        drop.addEventListener('drop', event => void importSkin(event.dataTransfer.files[0]).catch(error => setStatus(error.message, true)));
      }
      root.querySelectorAll('[data-action]').forEach(button => button.addEventListener('click', () => {
        button.disabled = true;
        const action = button.dataset.action;
        const promise = action === 'save' ? save() : run(action === 'sync' ? designer.syncExample : action === 'deploy' ? designer.deployExample : designer.buildExample);
        void promise.catch(error => { setStatus(error.message, true); button.disabled = false; });
      }));
    }

    void api(`/api/rimworld/workspace?sandbox=${encodeURIComponent(sandbox)}`).then(payload => {
      if (state.destroyed) return;
      state.workspace = payload.workspace; state.catalog = payload.catalog; state.skinImport = payload.skinImport || null; state.selected = state.workspace.selectedCharacterId || state.workspace.characters[0].id; render();
    }).catch(error => { root.innerHTML = `<div class="rw-error"><b>RimWorld Prepare could not load</b><p>${esc(error.message)}</p></div>`; });
    return { destroy() { state.destroyed = true; state.images.clear(); } };
  }

  window.RimWorldPrepare = { mount };
})();
