(function () {
  'use strict';

  const SKILLS = ['Animals', 'Artistic', 'Construction', 'Cooking', 'Crafting', 'Intellectual', 'Medical', 'Melee', 'Mining', 'Plants', 'Shooting', 'Social'];
  const TRAIT_ALIASES = { Industrious: ['Industriousness', 2], HardWorker: ['Industriousness', 1], Jogger: ['SpeedOffset', 2], Sanguine: ['NaturalMood', 2], Optimist: ['NaturalMood', 1], IronWilled: ['Nerves', 2], PsychicallyDeaf: ['PsychicSensitivity', -2], PsychicallyHypersensitive: ['PsychicSensitivity', 2], CarefulShooter: ['ShootingAccuracy', 1], TriggerHappy: ['ShootingAccuracy', -1] };
  const QUALITIES = ['Awful', 'Poor', 'Normal', 'Good', 'Excellent', 'Masterwork', 'Legendary'];
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
    const state = { workspace: null, catalog: null, skinImport: null, selected: '', tab: sessionStorage.getItem('rimworld-prepare-tab') || 'character', assetFilter: 'all', search: '', geneSearch: '', geneCategory: '', previewVersion: 0, job: null, uploading: false, destroyed: false, images: new Map() };
    root.innerHTML = '<div class="rw-loading"><span></span><b>Reading the installed Yuran 1.6 assets…</b></div>';

    const assetUrl = (path, token = '') => `/api/rimworld/asset?sandbox=${encodeURIComponent(sandbox)}&path=${encodeURIComponent(path)}${token ? `&v=${encodeURIComponent(token)}` : ''}`;
    const characterOptions = () => state.catalog.characterOptions || {};
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
    async function drawAsset(ctx, asset, direction, offset = [0, 0], color = null) {
      const file = fileFor(asset, direction);
      if (!file) return;
      try {
        let image = await loadImage(file.path);
        if (color && color !== '#ffffff') {
          const tinted = document.createElement('canvas'); tinted.width = image.width; tinted.height = image.height;
          const paint = tinted.getContext('2d'); paint.drawImage(image, 0, 0);
          paint.globalCompositeOperation = 'multiply'; paint.fillStyle = color; paint.fillRect(0, 0, tinted.width, tinted.height);
          paint.globalCompositeOperation = 'destination-in'; paint.drawImage(image, 0, 0); image = tinted;
        }
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
    async function drawImportedPart(ctx, kind, direction, offset = [0, 0], color = null) {
      const chosen = direction === 'west' ? 'east' : direction;
      const base = kind === 'face' ? 'Heads/Female_YR_head' : 'Bodies/Naked_Thin';
      const path = `outputs/imports/DragonSkin/${base}_${chosen}.png`;
      try {
        let image = await loadImage(path);
        if (color && color !== '#ffffff') {
          const tinted = document.createElement('canvas'); tinted.width = image.width; tinted.height = image.height;
          const paint = tinted.getContext('2d'); paint.drawImage(image, 0, 0);
          paint.globalCompositeOperation = 'multiply'; paint.fillStyle = color; paint.fillRect(0, 0, tinted.width, tinted.height);
          paint.globalCompositeOperation = 'destination-in'; paint.drawImage(image, 0, 0); image = tinted;
        }
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
      const version = ++state.previewVersion;
      const buffer = document.createElement('canvas'); buffer.width = 256; buffer.height = 256;
      const ctx = buffer.getContext('2d');
      await drawAsset(ctx, byId('tail/YR_tail'), direction, TAIL[direction], person.hairColor);
      if (!(person.dragonSkinInstalled && await drawImportedPart(ctx, 'body', direction, [0, 0], person.skinColor))) await drawAsset(ctx, byId('Bodies/Naked_Thin'), direction, [0, 0], person.skinColor);
      const worn = person.apparel.map(def => state.catalog.apparel.find(item => item.defName === def)).filter(Boolean);
      for (const item of worn.filter(item => !item.layers.includes('Overhead'))) await drawAsset(ctx, apparelAsset(item), direction, [0, 0], person.apparelDetails[item.defName]?.color);
      if (!(person.dragonSkinInstalled && await drawImportedPart(ctx, 'face', direction, HEAD[direction], person.skinColor))) await drawAsset(ctx, byId('Heads/Female_YR_head'), direction, HEAD[direction], person.skinColor);
      await drawAsset(ctx, byId('Ear/L/YR_earL'), direction, EARS[direction][0], person.hairColor);
      await drawAsset(ctx, byId('Ear/R/YR_earR'), direction, EARS[direction][1], person.hairColor);
      const hair = characterOptions().hairStyles?.find(item => item.defName === person.hairDef);
      if (hair?.asset) await drawAsset(ctx, byId(hair.asset), direction, HEAD[direction], person.hairColor);
      for (const item of worn.filter(item => item.layers.includes('Overhead'))) await drawAsset(ctx, apparelAsset(item), direction, HEAD[direction], person.apparelDetails[item.defName]?.color);
      if (!state.destroyed && version === state.previewVersion && canvas.isConnected) { const target = canvas.getContext('2d'); target.clearRect(0, 0, 256, 256); target.drawImage(buffer, 0, 0); }
    }

    function characterList() {
      return `<aside class="rw-roster"><header><b>Colonists</b><button type="button" data-add-character title="Duplicate selected colonist">＋</button></header>${state.workspace.characters.map(person => `<button type="button" class="rw-roster-item ${person.id === state.selected ? 'active' : ''}" data-character="${esc(person.id)}"><span>${esc(person.name.nick || person.name.first)}</span><small>${esc(person.biologicalAge)} · Dragon Yuran</small></button>`).join('')}</aside>`;
    }
    function selectField(label, path, value, entries, blank = 'None') {
      return `<label>${esc(label)}<select data-path="${esc(path)}">${['hairDef', 'genes.xenotypeDef'].includes(path) ? '' : `<option value="">${esc(blank)}</option>`}${entries.map(item => `<option value="${esc(item.defName)}" ${item.defName === value ? 'selected' : ''}>${esc(item.label)}${item.source && item.source !== 'Core' ? ` · ${esc(item.source)}` : ''}</option>`).join('')}</select></label>`;
    }
    function backstoryField(person, slot) {
      return `<label>${slot === 'childhood' ? 'Childhood' : 'Adulthood'}<input data-path="${slot}" list="rw-${slot}" value="${esc(person[slot])}" placeholder="Search title or definition"><datalist id="rw-${slot}">${(characterOptions().backstories || []).filter(item => item.slot.toLowerCase() === slot).map(item => `<option value="${esc(item.defName)}">${esc(item.label)}</option>`).join('')}</datalist></label>`;
    }
    function genesPane(person) {
      const genes = person.genes, options = characterOptions(), defs = options.genes || [];
      const picked = [...genes.endogenes, ...genes.xenogenes].map(name => defs.find(item => item.defName === name)).filter(Boolean);
      const totals = ['complexity', 'metabolism', 'archites'].map(key => `${key}: ${picked.reduce((sum, item) => sum + item[key], 0)}`).join(' · ');
      const conflicts = [];
      for (let i = 0; i < picked.length; i++) {
        const item = picked[i];
        if (item.prerequisite && !picked.some(other => other.defName === item.prerequisite)) conflicts.push(`${item.label} needs ${item.prerequisite}`);
        for (const other of picked.slice(i + 1)) if (item.exclusionTags.some(tag => other.exclusionTags.includes(tag))) conflicts.push(`${item.label} / ${other.label}`);
      }
      const selected = kind => `<div class="rw-gene-selected"><b>${kind === 'endogenes' ? 'Inherited genes' : 'Implanted genes'}</b>${genes[kind].map(name => `<button type="button" data-remove-gene="${esc(name)}" data-kind="${kind}" title="Remove gene">${esc(defs.find(item => item.defName === name)?.label || name)} ×</button>`).join('') || '<small>None</small>'}</div>`;
      return `<h3>Genes · Biotech</h3><label class="rw-check-label"><input type="checkbox" data-genes-enabled ${genes.enabled ? 'checked' : ''} ${!options.installedExpansions?.includes('Biotech') ? 'disabled' : ''}>Export gene configuration</label>
      <p class="rw-help">Requires Biotech enabled in RimWorld. Endogenes are inherited; xenogenes are implanted. Gene effects and tattoos appear in the game, not in this texture preview.</p>
      ${genes.enabled ? `<div class="rw-name-grid">${selectField('Xenotype preset', 'genes.xenotypeDef', genes.xenotypeDef, options.xenotypes || [])}<label>Custom xenotype name<input data-path="genes.customXenotypeName" maxlength="40" value="${esc(genes.customXenotypeName)}"></label><button type="button" data-apply-xenotype>Replace genes with preset</button></div>
      <p class="rw-help">Choosing a name preserves your current genes; “Replace genes with preset” replaces both lists. Editing a preset's genes creates a custom xenotype.</p>
      ${selected('endogenes')}${selected('xenogenes')}<p class="rw-gene-totals">Selected totals (before overrides) · ${esc(totals)}</p>
      ${conflicts.length ? `<p class="rw-warning">Possible overrides / prerequisites: ${esc(conflicts.join('; '))}</p>` : ''}
      <div class="rw-gene-filter"><input type="search" data-gene-search placeholder="Search genes" value="${esc(state.geneSearch)}"><select data-gene-category><option value="">All categories</option>${[...new Set(defs.map(item => item.category))].sort().map(category => `<option ${category === state.geneCategory ? 'selected' : ''}>${esc(category)}</option>`).join('')}</select></div>
      <div class="rw-gene-list">${defs.map(item => `<article data-gene-row data-search="${esc(`${item.label} ${item.defName} ${item.description}`.toLowerCase())}" data-category="${esc(item.category)}"><span title="${esc(item.description)}"><b>${esc(item.label)}</b><small>${esc(item.category)} · C ${item.complexity} / M ${item.metabolism} / A ${item.archites}</small></span><button type="button" data-add-gene="${esc(item.defName)}" data-kind="endogenes" ${picked.some(p => p.defName === item.defName) ? 'disabled' : ''}>Inherited +</button><button type="button" data-add-gene="${esc(item.defName)}" data-kind="xenogenes" ${picked.some(p => p.defName === item.defName) ? 'disabled' : ''}>Implanted +</button></article>`).join('')}</div>` : ''}
      <p class="rw-help">${esc(options.scope || 'Refresh source from TK2 to load character definitions.')}</p>`;
    }
    function identityPane(person) {
      const options = characterOptions();
      const traits = (options.traits || []).flatMap(item => item.degrees.map(degree => ({ ...degree, defName: item.defName })));
      return `<section class="rw-fields"><h3>Identity</h3><div class="rw-name-grid">
        <label>First<input data-path="name.first" value="${esc(person.name.first)}"></label><label>Nickname<input data-path="name.nick" value="${esc(person.name.nick)}"></label><label>Last<input data-path="name.last" value="${esc(person.name.last)}"></label>
      </div><div class="rw-name-grid"><label>Biological age<input type="number" min="13" max="1000" data-path="biologicalAge" value="${esc(person.biologicalAge)}"></label><label>Chronological age<input type="number" min="13" max="1000" data-path="chronologicalAge" value="${esc(person.chronologicalAge)}"></label><label>Race<input value="Dragon Yuran" disabled></label></div>
      <h3>Appearance</h3><div class="rw-name-grid">${selectField('Hairstyle', 'hairDef', person.hairDef, options.hairStyles || [])}<label>Hair / ears / tail color<input type="color" data-path="hairColor" value="${esc(person.hairColor)}"></label><label>Skin color<input type="color" data-path="skinColor" value="${esc(person.skinColor)}"></label></div>
      <p class="rw-help">Race artwork: female · Thin body · Yuran head. These silhouettes are fixed by the race's available artwork.</p>
      <div class="rw-name-grid">${selectField('Beard', 'beard', person.beard, options.beards || [])}${selectField('Face tattoo · Ideology', 'faceTattoo', person.faceTattoo, (options.tattoos || []).filter(item => item.tattooType === 'Face'))}${selectField('Body tattoo · Ideology', 'bodyTattoo', person.bodyTattoo, (options.tattoos || []).filter(item => item.tattooType === 'Body'))}</div>
      <div class="rw-name-grid">${selectField('Favorite color', 'favoriteColorDef', person.favoriteColorDef, options.favoriteColors || [], 'Generated default')}</div>
      <h3>Backstories</h3><div class="rw-name-grid rw-two-columns">${backstoryField(person, 'childhood')}${backstoryField(person, 'adulthood')}</div><p class="rw-help">Backstories can disable work types and alter effective skills in game. Clear a field to leave its generated backstory.</p>
      <h3>Skills</h3><div class="rw-skill-grid">${SKILLS.map(name => `<label><span>${esc(name)}</span><input type="number" min="0" max="20" data-skill="${name}" value="${person.skills[name].level}"><select data-passion="${name}">${['None', 'Minor', 'Major'].map(value => `<option ${person.skills[name].passion === value ? 'selected' : ''}>${value}</option>`).join('')}</select></label>`).join('')}</div>
      <h3>Traits</h3><div class="rw-selected-traits">${person.traits.map((trait, index) => `<button type="button" data-remove-trait="${index}">${esc(traits.find(item => item.defName === trait.def && item.degree === trait.degree)?.label || `${trait.def} (${trait.degree})`)} ×</button>`).join('')}</div><div class="rw-gene-filter"><select aria-label="Trait to add" data-trait-choice>${traits.map(item => `<option value="${esc(item.defName)}:${item.degree}">${esc(item.label)} · ${esc(item.defName)} (${item.degree})</option>`).join('')}</select><button type="button" data-add-trait>Add trait</button></div>
      ${genesPane(person)}
      <h3>Health and worn apparel</h3><div class="rw-check-grid"><label class="rw-dragon-check"><input type="checkbox" data-dragon-skin ${person.dragonSkinInstalled ? 'checked' : ''}><span>Dragon Skin installed</span></label><label class="rw-dragon-check"><input type="checkbox" data-dragon-bone ${person.dragonBoneInstalled ? 'checked' : ''}><span>Dragon Bone installed · spine</span></label>${state.workspace.mod.selectedApparel.map(def => { const item = state.catalog.apparel.find(value => value.defName === def); return `<label><input type="checkbox" data-worn="${esc(def)}" ${person.apparel.includes(def) ? 'checked' : ''}><span>${esc(item?.label || def)}</span></label>`; }).join('')}</div>
      ${person.apparel.map(def => { const detail = person.apparelDetails[def] || {}; return `<div class="rw-apparel-detail"><b>${esc(state.catalog.apparel.find(item => item.defName === def)?.label || def)}</b><label>Quality<select data-apparel-def="${esc(def)}" data-apparel-field="quality">${QUALITIES.map(q => `<option ${q === (detail.quality || 'Normal') ? 'selected' : ''}>${q}</option>`).join('')}</select></label><label>Condition %<input type="number" min="1" max="100" data-apparel-def="${esc(def)}" data-apparel-field="condition" value="${detail.condition ?? 100}"></label><label>Color<input type="color" data-apparel-def="${esc(def)}" data-apparel-field="color" value="${esc(detail.color || '#ffffff')}"></label></div>`; }).join('')}</section>`;
    }
    function characterTab() {
      const person = current();
      return `<div class="rw-character-layout">${characterList()}<section class="rw-preview"><div class="rw-canvas-shell"><canvas id="rw-pawn-canvas" width="256" height="256" aria-label="Layered Yuran pawn preview"></canvas></div><div class="rw-directions" role="group" aria-label="View direction">${DIRECTIONS.map(value => `<button type="button" data-direction="${value}" class="${person.direction === value ? 'active' : ''}">${value}</button>`).join('')}</div><p>Uses RimWorld 1.6 Thin head offset plus the Yuran 1.6 HAR addon offsets. Hair, skin and clothing colors update here. Beards, tattoos and gene effects require an in-game check.</p></section>${identityPane(person)}</div>`;
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
    function dragonBoneTab() {
      const scale = Number(state.workspace.mod.dragonBone.giantScale || 5);
      const giantHeight = Math.min(250, Math.round(78 * scale));
      const bodySize = (0.9 * scale).toFixed(2);
      return `<div class="rw-bone-workspace"><section class="rw-bone-controls"><p class="rw-kicker">CYBERNETIC SPINE</p><h2>Dragon Bone · giant form</h2><p>When this implant is installed on a Dragon Yuran spine, it grants an undrafted <b>Dragon Giant Form</b> command. The form stays active until you use its cancel command to transform back.</p><label class="rw-scale-control"><span><b>Giant body-size multiplier</b><output data-giant-scale-value>${scale.toFixed(2)}×</output></span><input type="range" min="1.25" max="10" step="0.05" value="${scale}" data-giant-scale><input type="number" min="1.25" max="10" step="0.05" value="${scale}" data-giant-scale></label><div class="rw-bone-facts"><span><small>Normal body size</small><b>0.90</b></span><span><small>Giant body size</small><b data-giant-body-size>${bodySize}</b></span><span><small>Activation</small><b>Manual toggle</b></span></div></section><section class="rw-scale-preview"><p class="rw-kicker">SIZE COMPARISON</p><div class="rw-scale-stage"><figure><i class="rw-size-pawn" style="--rw-pawn-height:78px"></i><figcaption>Normal · 1×</figcaption></figure><figure><i class="rw-size-pawn giant" style="--rw-pawn-height:${giantHeight}px"></i><figcaption data-giant-caption>Giant · ${scale.toFixed(2)}×</figcaption></figure></div><p>The chosen value is written to Framework stat <code>SM_BodySizeMultiplier</code>. Big and Small Framework controls body size, health scaling and rendering; its large-body visual curve may compress extreme values on screen.</p></section><aside class="rw-framework-card"><p class="rw-kicker">DEPENDENCY CONTRACT</p><h3>Framework only</h3><ul><li><b>Required:</b> Big and Small - Framework</li><li><b>Required:</b> Harmony + Humanoid Alien Races</li><li><b>Not used:</b> Big and Small - Genes & More</li></ul><p>The Dragon Bone ability and Hediffs live in Dragon Yuran. No True Form gene or Genes & More assembly is referenced.</p></aside></div>`;
    }
    function deployTab() {
      return `<div class="rw-deploy"><section><p class="rw-kicker">BUILD CONTENT</p><h2>Clean Dragon Yuran package</h2><ul><li>Race: DragonYuran_Race + colonist kind</li><li>Dragon Bone: spine implant + ${Number(state.workspace.mod.dragonBone.giantScale).toFixed(2)}× toggleable giant form</li><li>Selected apparel: ${state.workspace.mod.selectedApparel.length}, craftable at Hand/Electric Tailoring Bench</li><li>Prepare Carefully v5 files: ${state.workspace.characters.length}</li><li>Excluded: fiction, factions, buildings, weapons, research, Miko, Black Snake, Big and Small Genes</li></ul></section><section><p class="rw-kicker">TK2 DESTINATION</p><h2>Deploy without launching</h2><code>RimWorld/Mods/DragonYuran</code><code>SaveData/PrepareCarefully/*.pcc</code><p>Deployment does not start RimWorld and does not change the active mod list.</p><div class="rw-deploy-actions"><button class="button button-light" type="button" data-action="build">Build locally</button><button class="button" type="button" data-action="deploy">Build & deploy to TK2</button></div></section></div>`;
    }
    function tabBody() {
      if (state.tab === 'apparel') return apparelTab();
      if (state.tab === 'assets') return assetsTab();
      if (state.tab === 'dragon') return dragonTab();
      if (state.tab === 'bone') return dragonBoneTab();
      if (state.tab === 'deploy') return deployTab();
      return characterTab();
    }
    function render() {
      const scrollTop = root.querySelector('.rw-fields')?.scrollTop || 0;
      if (state.destroyed) return;
      const source = state.catalog.source;
      root.innerHTML = `<div class="rw-shell"><div class="rw-pipeline"><span class="complete"><i>1</i><b>Installed source</b><small>${state.catalog.assets.length} race assets</small></span><span class="complete"><i>2</i><b>Race core</b><small>Yuran → Dragon Yuran</small></span><span class="complete"><i>3</i><b>Characters</b><small>${state.workspace.characters.length} Prepare Carefully</small></span><span class="${state.job?.status === 'succeeded' && state.job?.example === designer.deployExample ? 'complete' : ''}"><i>4</i><b>TK2</b><small>${state.job?.status || 'ready to deploy'}</small></span></div><nav class="rw-tabs" aria-label="RimWorld Prepare sections">${[['character', 'Character'], ['apparel', 'Clothing'], ['assets', 'Assets'], ['dragon', 'Dragon skin'], ['bone', 'Dragon bone'], ['deploy', 'Build & deploy']].map(([id, label]) => `<button type="button" data-tab="${id}" class="${state.tab === id ? 'active' : ''}">${label}</button>`).join('')}<span></span><button class="rw-save" type="button" data-action="save">Save workspace</button><button class="rw-sync" type="button" data-action="sync">Refresh from TK2</button></nav><main class="rw-body">${tabBody()}</main><footer class="rw-status"><span class="${state.job?.status || 'ready'}"><i></i>${esc(state.job ? `${state.job.status}: ${state.job.example}` : 'Workspace ready')}</span><small>Workshop ${esc(source.workshopId)} · RimWorld ${esc(designer.gameVersion || source.version)} · game not launched</small></footer></div>`;
      bind();
      if (root.querySelector('.rw-fields')) root.querySelector('.rw-fields').scrollTop = scrollTop;
      filterGenes();
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
      if (!state.destroyed && state.job.status === 'succeeded' && example === designer.syncExample) { const updated = await api(`/api/rimworld/workspace?sandbox=${encodeURIComponent(sandbox)}`); state.catalog = updated.catalog; state.images.clear(); }
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
    function filterGenes() {
      root.querySelectorAll('[data-gene-row]').forEach(row => { row.hidden = !row.dataset.search.includes(state.geneSearch.toLowerCase()) || Boolean(state.geneCategory && row.dataset.category !== state.geneCategory); });
    }
    function normalizeWorkspace() {
      state.workspace.characters.forEach(person => {
        person.hairDef ??= `DragonYuran_Hair_${person.hairVariant || 0}`;
        for (const key of ['childhood', 'adulthood', 'beard', 'faceTattoo', 'bodyTattoo', 'favoriteColor', 'favoriteColorDef']) person[key] ??= '';
        person.apparelDetails ||= {};
        person.genes ||= { enabled: false, xenotypeDef: 'Baseliner', customXenotypeName: '', endogenes: [], xenogenes: [] };
        person.traits.forEach(trait => { const alias = TRAIT_ALIASES[trait.def]; if (alias) [trait.def, trait.degree] = alias; });
      });
    }
    function bind() {
      root.querySelectorAll('[data-tab]').forEach(button => button.addEventListener('click', () => { state.tab = button.dataset.tab; sessionStorage.setItem('rimworld-prepare-tab', state.tab); render(); }));
      root.querySelectorAll('[data-character]').forEach(button => button.addEventListener('click', () => { state.selected = button.dataset.character; state.workspace.selectedCharacterId = state.selected; render(); }));
      root.querySelector('[data-add-character]')?.addEventListener('click', () => {
        if (state.workspace.characters.length >= 24) { setStatus('Maximum 24 characters.', true); return; }
        const person = clone(current()); person.id = `dragon-yuran-${Date.now()}`; person.name.nick = `${person.name.nick} II`; state.workspace.characters.push(person); state.selected = person.id; state.workspace.selectedCharacterId = person.id; render();
      });
      root.querySelectorAll('[data-direction]').forEach(button => button.addEventListener('click', () => { current().direction = button.dataset.direction; render(); }));
      root.querySelectorAll('[data-path]').forEach(input => input.addEventListener('input', () => { setPath(current(), input.dataset.path, input.type === 'number' ? Number(input.value) : input.value); if (['hairDef', 'hairColor', 'skinColor'].includes(input.dataset.path)) void renderPawn(); }));
      root.querySelectorAll('[data-skill]').forEach(input => input.addEventListener('input', () => { current().skills[input.dataset.skill].level = Number(input.value); }));
      root.querySelectorAll('[data-passion]').forEach(input => input.addEventListener('change', () => { current().skills[input.dataset.passion].passion = input.value; }));
      root.querySelector('[data-add-trait]')?.addEventListener('click', () => {
        const value = root.querySelector('[data-trait-choice]').value; if (!value) return;
        const [def, degree] = value.split(':'); const existing = current().traits.find(item => item.def === def);
        if (existing) existing.degree = Number(degree); else if (current().traits.length < 8) current().traits.push({ def, degree: Number(degree) });
        else { setStatus('Maximum 8 traits per character.', true); return; } render();
      });
      root.querySelectorAll('[data-remove-trait]').forEach(button => button.addEventListener('click', () => { current().traits.splice(Number(button.dataset.removeTrait), 1); render(); }));
      root.querySelector('[data-genes-enabled]')?.addEventListener('change', event => { current().genes.enabled = event.target.checked; render(); });
      root.querySelector('[data-apply-xenotype]')?.addEventListener('click', () => {
        const preset = characterOptions().xenotypes.find(item => item.defName === current().genes.xenotypeDef);
        if (!preset) { setStatus('Choose a xenotype preset first.', true); return; }
        const known = new Set(characterOptions().genes.map(item => item.defName));
        if (preset.genes.some(name => !known.has(name))) { setStatus('This preset contains unavailable genes; refresh TK2 definitions.', true); return; }
        Object.assign(current().genes, { customXenotypeName: '', endogenes: preset.inheritable ? [...preset.genes] : [], xenogenes: preset.inheritable ? [] : [...preset.genes] }); render();
      });
      const markCustomGenes = () => { current().genes.xenotypeDef = 'Baseliner'; current().genes.customXenotypeName ||= 'Custom Yuran'; };
      root.querySelectorAll('[data-add-gene]').forEach(button => button.addEventListener('click', () => { current().genes[button.dataset.kind].push(button.dataset.addGene); markCustomGenes(); render(); }));
      root.querySelectorAll('[data-remove-gene]').forEach(button => button.addEventListener('click', () => { current().genes[button.dataset.kind] = current().genes[button.dataset.kind].filter(name => name !== button.dataset.removeGene); markCustomGenes(); render(); }));
      root.querySelector('[data-gene-search]')?.addEventListener('input', event => { state.geneSearch = event.target.value; filterGenes(); });
      root.querySelector('[data-gene-category]')?.addEventListener('change', event => { state.geneCategory = event.target.value; filterGenes(); });
      root.querySelectorAll('[data-apparel-field]').forEach(input => input.addEventListener('input', () => { current().apparelDetails[input.dataset.apparelDef] ||= {}; current().apparelDetails[input.dataset.apparelDef][input.dataset.apparelField] = input.type === 'number' ? Number(input.value) : input.value; if (input.dataset.apparelField === 'color') void renderPawn(); }));
      root.querySelector('[data-dragon-skin]')?.addEventListener('change', event => { current().dragonSkinInstalled = event.target.checked; void renderPawn(); });
      root.querySelector('[data-dragon-bone]')?.addEventListener('change', event => { current().dragonBoneInstalled = event.target.checked; });
      root.querySelectorAll('[data-giant-scale]').forEach(input => input.addEventListener('input', event => {
        const scale = Math.max(1.25, Math.min(10, Number(event.target.value) || 5));
        state.workspace.mod.dragonBone.giantScale = Math.round(scale * 100) / 100;
        root.querySelectorAll('[data-giant-scale]').forEach(other => { if (other !== event.target) other.value = scale; });
        const value = root.querySelector('[data-giant-scale-value]'); if (value) value.textContent = `${scale.toFixed(2)}×`;
        const size = root.querySelector('[data-giant-body-size]'); if (size) size.textContent = (0.9 * scale).toFixed(2);
        const giant = root.querySelector('.rw-size-pawn.giant'); if (giant) giant.style.setProperty('--rw-pawn-height', `${Math.min(250, Math.round(78 * scale))}px`);
        const caption = root.querySelector('[data-giant-caption]'); if (caption) caption.textContent = `Giant · ${scale.toFixed(2)}×`;
      }));
      root.querySelectorAll('[data-worn]').forEach(input => input.addEventListener('change', () => { const items = current().apparel; const def = input.dataset.worn; if (input.checked && !items.includes(def)) items.push(def); if (!input.checked) current().apparel = items.filter(item => item !== def); render(); }));
      root.querySelectorAll('[data-include-apparel]').forEach(input => input.addEventListener('change', () => { const selected = state.workspace.mod.selectedApparel; const def = input.dataset.includeApparel; if (input.checked && !selected.includes(def)) selected.push(def); if (!input.checked) { state.workspace.mod.selectedApparel = selected.filter(item => item !== def); state.workspace.characters.forEach(person => { person.apparel = person.apparel.filter(item => item !== def); delete person.apparelDetails[def]; }); } render(); }));
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
        void promise.catch(error => { setStatus(error.message, true); }).finally(() => { button.disabled = false; });
      }));
    }

    void api(`/api/rimworld/workspace?sandbox=${encodeURIComponent(sandbox)}`).then(payload => {
      if (state.destroyed) return;
      state.workspace = payload.workspace; state.catalog = payload.catalog; state.skinImport = payload.skinImport || null;
      state.workspace.mod.dragonBone ||= { giantScale: 5 };
      state.workspace.characters.forEach(person => { if (typeof person.dragonBoneInstalled !== 'boolean') person.dragonBoneInstalled = false; });
      normalizeWorkspace();
      state.selected = state.workspace.selectedCharacterId || state.workspace.characters[0].id; render();
    }).catch(error => { root.innerHTML = `<div class="rw-error"><b>RimWorld Prepare could not load</b><p>${esc(error.message)}</p></div>`; });
    return { destroy() { state.destroyed = true; state.images.clear(); } };
  }

  window.RimWorldPrepare = { mount };
})();
