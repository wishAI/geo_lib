(function () {
  'use strict';

  const escapeHtml = value => String(value ?? '').replace(/[&<>'"]/g, character => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;' })[character]);
  const clone = value => JSON.parse(JSON.stringify(value));
  const radians = degrees => Number(degrees || 0) * Math.PI / 180;
  const clamp = (value, minimum, maximum) => Math.min(maximum, Math.max(minimum, value));

  class VanillaModelRenderer {
    static async create(canvas, stats, design, sandbox, designId) {
      const [THREE, loaderModule, controlsModule] = await Promise.all([
        import('/vendor/three/three.module.js'),
        import('/vendor/three/modules/GLTFLoader.js'),
        import('/vendor/three/modules/OrbitControls.js'),
      ]);
      const instance = new VanillaModelRenderer(canvas, stats, design, sandbox, designId, THREE, loaderModule.GLTFLoader, controlsModule.OrbitControls);
      await instance.initialize();
      return instance;
    }

    constructor(canvas, stats, design, sandbox, designId, THREE, GLTFLoader, OrbitControls) {
      this.canvas = canvas;
      this.stats = stats;
      this.design = design;
      this.sandbox = sandbox;
      this.designId = designId;
      this.THREE = THREE;
      this.GLTFLoader = GLTFLoader;
      this.OrbitControls = OrbitControls;
      this.renderer = null;
      this.scene = null;
      this.camera = null;
      this.controls = null;
      this.model = null;
      this.markerGroup = null;
      this.skeletonHelper = null;
      this.mixer = null;
      this.actions = new Map();
      this.playing = true;
      this.showMarkers = true;
      this.showLabels = true;
      this.showSkeleton = false;
      this.selected = null;
      this.frameRequest = 0;
      this.last = performance.now();
      this.frames = 0;
      this.fpsStarted = performance.now();
      this.resizeObserver = new ResizeObserver(() => this.invalidate());
    }

    async initialize() {
      const THREE = this.THREE;
      this.renderer = new THREE.WebGLRenderer({ canvas: this.canvas, antialias: true, alpha: true, powerPreference: 'high-performance' });
      this.renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
      this.renderer.outputColorSpace = THREE.SRGBColorSpace;
      this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
      this.renderer.toneMappingExposure = 1.15;
      this.scene = new THREE.Scene();
      this.camera = new THREE.PerspectiveCamera(44, 1, .01, 5000);
      this.camera.position.set(13, 9, 18);
      this.controls = new this.OrbitControls(this.camera, this.canvas);
      this.controls.enableDamping = false;
      this.controls.screenSpacePanning = true;
      this.controls.addEventListener('change', () => this.invalidate());
      this.scene.add(new THREE.HemisphereLight(0xbfd4ff, 0x101625, 2.4));
      const key = new THREE.DirectionalLight(0xffffff, 3.4); key.position.set(6, 11, 8); this.scene.add(key);
      const rim = new THREE.DirectionalLight(0x5b87ff, 2.2); rim.position.set(-8, 2, -10); this.scene.add(rim);
      const grid = new THREE.GridHelper(80, 32, 0x35558e, 0x1c3158); grid.position.y = -5; grid.material.transparent = true; grid.material.opacity = .22; this.scene.add(grid);
      this.resizeObserver.observe(this.canvas);
      const gltf = await new Promise((resolve, reject) => new this.GLTFLoader().load(
        `/api/designer/model?sandbox=${encodeURIComponent(this.sandbox)}&design=${encodeURIComponent(this.designId)}&v=${encodeURIComponent(this.design.model?.sha256 || '')}`,
        resolve, undefined, reject,
      ));
      this.model = gltf.scene;
      this.model.name = this.design.model?.label || 'Vanilla Stellaris battleship';
      this.scene.add(this.model);
      this.mixer = gltf.animations.length ? new THREE.AnimationMixer(this.model) : null;
      for (const clip of gltf.animations) this.actions.set(clip.name, this.mixer.clipAction(clip));
      this.skeletonHelper = new THREE.SkeletonHelper(this.model);
      this.skeletonHelper.visible = false;
      this.skeletonHelper.material.linewidth = 2;
      this.scene.add(this.skeletonHelper);
      this.applyDesignTransforms();
      this.rebuildMarkers();
      this.fitView();
      this.selectClip(this.design.animation.selected);
      this.invalidate();
    }

    fitView() {
      const THREE = this.THREE;
      const box = new THREE.Box3().setFromObject(this.model);
      const center = box.getCenter(new THREE.Vector3());
      const size = box.getSize(new THREE.Vector3());
      const radius = Math.max(size.x, size.y, size.z, 1);
      this.controls.target.copy(center);
      this.camera.near = Math.max(.001, radius / 1000);
      this.camera.far = radius * 100;
      this.camera.position.copy(center).add(new THREE.Vector3(radius * .9, radius * .58, radius * 1.15));
      this.camera.updateProjectionMatrix();
      this.controls.update();
    }

    applyDesignTransforms() {
      if (!this.model) return;
      const scale = Number(this.design.ship.entityScale || 1);
      this.model.scale.setScalar(scale);
      for (const section of this.design.sections || []) {
        for (const part of section.parts || []) {
          if (!part.sourceNode) continue;
          const object = this.model.getObjectByName(part.sourceNode);
          if (!object) continue;
          object.visible = part.visible !== false;
          if (part.editTransform === true) {
            object.position.fromArray(part.position);
            object.rotation.set(radians(part.rotation?.[0]), radians(part.rotation?.[1]), radians(part.rotation?.[2]));
            object.scale.fromArray(part.scale);
          }
        }
      }
      for (const locator of this.design.locators || []) {
        if (!locator.sourceNode || locator.editTransform !== true) continue;
        const object = this.model.getObjectByName(locator.sourceNode);
        if (!object) continue;
        object.position.fromArray(locator.position);
        object.rotation.set(radians(locator.rotation?.[0]), radians(locator.rotation?.[1]), radians(locator.rotation?.[2]));
      }
      this.model.updateMatrixWorld(true);
    }

    rebuildMarkers() {
      if (!this.scene) return;
      if (this.markerGroup) {
        this.scene.remove(this.markerGroup);
        this.markerGroup.traverse(object => {
          object.geometry?.dispose?.();
          if (object.material?.map) object.material.map.dispose();
          object.material?.dispose?.();
        });
      }
      this.markerGroup = new this.THREE.Group();
      this.markerGroup.name = 'Editable locator markers';
      for (const locator of this.design.locators || []) {
        if (locator.visible === false) continue;
        const scale = Number(locator.markerScale || .6);
        const marker = new this.THREE.Group();
        marker.name = locator.id;
        marker.userData.locatorId = locator.id;
        marker.userData.source = locator.sourceNode ? this.model?.getObjectByName(locator.sourceNode) : null;
        const helper = new this.THREE.AxesHelper(scale);
        helper.material.depthTest = false;
        helper.material.transparent = true;
        helper.material.opacity = .98;
        helper.renderOrder = 8;
        marker.add(helper);
        if (locator.kind === 'fire_origin') {
          const dot = new this.THREE.Mesh(
            new this.THREE.SphereGeometry(scale * .13, 10, 8),
            new this.THREE.MeshBasicMaterial({ color: 0xffd05a, depthTest: false }),
          );
          dot.renderOrder = 9;
          marker.add(dot);
          const halo = new this.THREE.Mesh(
            new this.THREE.SphereGeometry(scale * .24, 12, 8),
            new this.THREE.MeshBasicMaterial({ color: 0xffd05a, wireframe: true, transparent: true, opacity: .75, depthTest: false }),
          );
          halo.userData.fireOriginPulse = true;
          halo.renderOrder = 9;
          marker.add(halo);
        }
        if (locator.fireAxis) marker.add(this.createFireGuide(locator, scale));
        const selected = this.selected?.kind === 'locator' && this.selected.id === locator.id;
        if (locator.labelVisible !== false && (selected || ['weapon', 'fire_origin'].includes(locator.kind))) {
          marker.add(this.createMarkerLabel(locator, scale));
        }
        if (!marker.userData.source) {
          marker.position.fromArray(locator.position || [0, 0, 0]);
          marker.rotation.set(radians(locator.rotation?.[0]), radians(locator.rotation?.[1]), radians(locator.rotation?.[2]));
        }
        this.markerGroup.add(marker);
      }
      this.markerGroup.visible = this.showMarkers;
      this.scene.add(this.markerGroup);
      this.updateMarkerTransforms();
    }

    createMarkerLabel(locator, scale) {
      const canvas = document.createElement('canvas');
      canvas.width = 768;
      canvas.height = 152;
      const context = canvas.getContext('2d');
      const official = locator.usage === 'official_slot_binding';
      context.fillStyle = official ? 'rgba(65,45,5,.94)' : 'rgba(5,17,39,.92)';
      context.fillRect(0, 0, canvas.width, canvas.height);
      context.fillStyle = official ? '#ffd66b' : '#f3f7ff';
      context.font = '700 42px ui-monospace, SFMono-Regular, Menlo, monospace';
      context.fillText(String(locator.name || locator.id).slice(0, 31), 28, 58);
      const detail = official
        ? locator.fireAxis ? `OFFICIAL SLOT · FIRING GUIDE ${locator.fireAxis}` : `OFFICIAL SLOT ORIGIN · DYNAMIC AIM`
        : locator.usage === 'embedded_unbound'
          ? 'EMBEDDED LOCATOR · NOT BOUND BY THIS SECTION'
          : `${String(locator.kind || 'LOCATOR').toUpperCase()} · ${locator.linkedSlot || 'NO SLOT BINDING'}`;
      context.fillStyle = official ? '#e9bd51' : '#91a9d8';
      context.font = '700 22px ui-monospace, SFMono-Regular, Menlo, monospace';
      context.fillText(detail.slice(0, 58), 28, 105);
      context.fillStyle = '#ff554d'; context.fillText('+X', 28, 137);
      context.fillStyle = '#54e987'; context.fillText('+Y', 78, 137);
      context.fillStyle = '#5793ff'; context.fillText('+Z', 128, 137);
      context.fillStyle = '#9eafd0'; context.fillText('LOCAL POSE', 190, 137);
      const texture = new this.THREE.CanvasTexture(canvas);
      texture.colorSpace = this.THREE.SRGBColorSpace;
      texture.minFilter = this.THREE.LinearFilter;
      texture.generateMipmaps = false;
      const sprite = new this.THREE.Sprite(new this.THREE.SpriteMaterial({ map: texture, transparent: true, depthTest: false, sizeAttenuation: true }));
      sprite.userData.markerLabel = true;
      sprite.visible = this.showLabels;
      sprite.position.set(0, scale * 1.35, 0);
      sprite.scale.set(scale * 5.8, scale * 1.15, 1);
      sprite.renderOrder = 10;
      return sprite;
    }

    createFireGuide(locator, scale) {
      const axes = {
        '+X': [1, 0, 0], '-X': [-1, 0, 0], '+Y': [0, 1, 0], '-Y': [0, -1, 0], '+Z': [0, 0, 1], '-Z': [0, 0, -1],
      };
      const direction = new this.THREE.Vector3(...(axes[locator.fireAxis] || axes['+Z']));
      const length = scale * 2.7;
      const guide = new this.THREE.Group();
      guide.userData.fireGuide = true;
      const geometry = new this.THREE.BufferGeometry().setFromPoints([new this.THREE.Vector3(), direction.clone().multiplyScalar(length)]);
      const line = new this.THREE.Line(geometry, new this.THREE.LineDashedMaterial({ color: 0xffd05a, dashSize: scale * .16, gapSize: scale * .12, transparent: true, opacity: .78, depthTest: false }));
      line.computeLineDistances();
      line.renderOrder = 9;
      guide.add(line);
      const pulse = new this.THREE.Mesh(
        new this.THREE.SphereGeometry(scale * .1, 10, 8),
        new this.THREE.MeshBasicMaterial({ color: 0xffe7a1, transparent: true, opacity: .95, depthTest: false }),
      );
      pulse.userData.firePulse = { direction, length, phase: Math.random() };
      pulse.renderOrder = 10;
      guide.add(pulse);
      return guide;
    }

    updateFireGuides(now) {
      this.markerGroup?.traverse(object => {
        if (object.userData.firePulse) {
          const { direction, length, phase } = object.userData.firePulse;
          const progress = ((now * .00058) + phase) % 1;
          object.position.copy(direction).multiplyScalar(length * progress);
          object.material.opacity = Math.sin(progress * Math.PI) * .9;
        }
        if (object.userData.fireOriginPulse) {
          const progress = (now * .0007) % 1;
          object.scale.setScalar(.7 + progress * 1.8);
          object.material.opacity = (1 - progress) * .72;
        }
      });
    }

    updateMarkerTransforms() {
      if (!this.markerGroup) return;
      for (const marker of this.markerGroup.children) {
        const source = marker.userData.source;
        if (!source) continue;
        source.getWorldPosition(marker.position);
        source.getWorldQuaternion(marker.quaternion);
      }
    }

    selectClip(requested) {
      if (!this.mixer) return;
      for (const action of this.actions.values()) action.stop();
      const aliases = this.design.model?.animationMap || {};
      const action = this.actions.get(aliases[requested] || requested) || this.actions.values().next().value;
      if (action) { action.reset(); action.setLoop(this.design.animation.loop ? this.THREE.LoopRepeat : this.THREE.LoopOnce); action.play(); action.paused = !this.playing; }
    }

    setDesign(design) { this.design = design; this.applyDesignTransforms(); this.rebuildMarkers(); this.selectClip(design.animation.selected); this.invalidate(); }
    setSelection(selection) { this.selected = selection; this.rebuildMarkers(); const object = selection?.kind === 'locator' ? this.markerGroup?.getObjectByName(selection.id) : null; if (object) object.scale.setScalar(1.65); this.invalidate(); }
    setPlaying(value) { this.playing = Boolean(value); for (const action of this.actions.values()) action.paused = !this.playing; this.last = performance.now(); this.invalidate(); }
    setMarkers(value) { this.showMarkers = Boolean(value); if (this.markerGroup) this.markerGroup.visible = this.showMarkers; this.invalidate(); }
    setLabels(value) { this.showLabels = Boolean(value); this.markerGroup?.traverse(object => { if (object.userData.markerLabel) object.visible = this.showLabels; }); this.invalidate(); }
    setSkeleton(value) { this.showSkeleton = Boolean(value); if (this.skeletonHelper) this.skeletonHelper.visible = this.showSkeleton; this.invalidate(); }
    resetView() { this.fitView(); this.invalidate(); }

    invalidate() { if (!this.frameRequest) this.frameRequest = requestAnimationFrame(now => this.render(now)); }

    render(now) {
      this.frameRequest = 0;
      const width = Math.max(1, this.canvas.clientWidth);
      const height = Math.max(1, this.canvas.clientHeight);
      this.renderer.setSize(width, height, false);
      this.camera.aspect = width / height;
      this.camera.updateProjectionMatrix();
      const delta = Math.min(.05, (now - this.last) / 1000);
      this.last = now;
      if (this.playing && this.mixer) this.mixer.update(delta * Number(this.design.animation.speed || 1));
      this.updateMarkerTransforms();
      this.updateFireGuides(now);
      this.renderer.render(this.scene, this.camera);
      this.frames += 1;
      if (now - this.fpsStarted > 700) {
        let triangles = 0; let draws = 0;
        this.model.traverse(object => { if (object.isMesh) { draws += 1; const index = object.geometry.index; triangles += index ? index.count / 3 : object.geometry.attributes.position.count / 3; } });
        this.stats.textContent = `${Math.round(this.frames * 1000 / (now - this.fpsStarted))} fps · ${Math.round(triangles).toLocaleString()} tris · ${draws} meshes`;
        this.frames = 0; this.fpsStarted = now;
      }
      if (this.playing && this.mixer && document.visibilityState === 'visible') this.invalidate();
    }

    destroy() {
      cancelAnimationFrame(this.frameRequest);
      this.resizeObserver.disconnect();
      this.controls?.dispose();
      this.renderer?.dispose();
      this.renderer?.forceContextLoss();
    }
  }

  const field = (label, path, value, options = {}) => {
    const id = `ship-field-${path.replace(/[^a-z0-9]+/gi, '-')}`;
    if (options.type === 'checkbox') return `<label class="ship-check"><input id="${id}" type="checkbox" data-design-path="${escapeHtml(path)}" ${value ? 'checked' : ''}><span>${escapeHtml(label)}</span></label>`;
    if (options.choices) return `<label class="ship-field" for="${id}"><span>${escapeHtml(label)}</span><select id="${id}" data-design-path="${escapeHtml(path)}">${options.choices.map(choice => `<option value="${escapeHtml(choice)}" ${String(choice) === String(value) ? 'selected' : ''}>${escapeHtml(choice)}</option>`).join('')}</select></label>`;
    const type = options.type || (typeof value === 'number' ? 'number' : options.color ? 'color' : 'text');
    return `<label class="ship-field" for="${id}"><span>${escapeHtml(label)}</span><input id="${id}" type="${type}" data-design-path="${escapeHtml(path)}" value="${escapeHtml(value)}" ${type === 'number' ? `step="${options.step || '0.1'}"` : ''}></label>`;
  };

  const vectorFields = (label, path, values) => `<fieldset class="ship-vector"><legend>${escapeHtml(label)}</legend>${['X', 'Y', 'Z'].map((axis, index) => field(axis, `${path}.${index}`, values[index], { step: .05 })).join('')}</fieldset>`;
  const panelGroup = (title, hint, content) => `<section class="ship-property-group"><header><div><b>${escapeHtml(title)}</b>${hint ? `<small>${escapeHtml(hint)}</small>` : ''}</div></header><div class="ship-property-group-body">${content}</div></section>`;
  const factGrid = facts => `<dl class="ship-fact-grid">${facts.filter(([, value]) => value !== undefined && value !== null && value !== '').map(([label, value]) => `<div><dt>${escapeHtml(label)}</dt><dd>${escapeHtml(Array.isArray(value) ? value.join(' · ') : value)}</dd></div>`).join('')}</dl>`;
  const sourceBadge = (label, tone = '') => `<span class="ship-source-badge ${escapeHtml(tone)}">${escapeHtml(label)}</span>`;

  class Designer {
    constructor(root, options) {
      this.root = root;
      this.options = options;
      this.design = null;
      this.cleanDesign = null;
      this.designId = '';
      this.designs = [];
      this.renderer = null;
      this.tab = 'ship';
      this.selection = { section: 0, part: 0, slot: 0, locator: 0 };
      this.dirty = false;
    }

    async mount() {
      await this.loadDesign('');
    }

    async loadDesign(designId) {
      this.renderer?.destroy();
      this.renderer = null;
      this.root.innerHTML = '<div class="ship-designer-loading"><span></span><b>Loading original Stellaris assets…</b></div>';
      try {
        const response = await fetch(`/api/designer/config?sandbox=${encodeURIComponent(this.options.sandbox)}&design=${encodeURIComponent(designId)}`, { cache: 'no-store' });
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.error || 'Design could not be loaded.');
        this.design = payload.design;
        this.designId = payload.designId;
        this.designs = payload.designs || [];
        this.cleanDesign = clone(this.design);
        this.selection = { section: 0, part: 0, slot: 0, locator: 0 };
        this.dirty = false;
        this.render();
      } catch (error) {
        this.root.innerHTML = `<div class="warning-box">${escapeHtml(error.message)}</div>`;
      }
    }

    render() {
      this.renderer?.destroy();
      const clips = this.design.animation.clips || [];
      const section = this.design.sections[this.selection.section] || this.design.sections[0];
      this.root.innerHTML = `<div class="ship-designer">
        <header class="ship-designer-bar"><div><span class="ship-kicker">ORIGINAL GAME ASSET · ${escapeHtml(this.design.source?.dlc || 'VANILLA')}</span><h2>${escapeHtml(this.design.ship.name)}</h2></div><div class="ship-actions"><label class="ship-example-select"><span>Example</span><select data-design>${this.designs.map(item => `<option value="${escapeHtml(item.id)}" ${item.id === this.designId ? 'selected' : ''}>${escapeHtml(item.label)}</option>`).join('')}</select></label><span class="ship-save-state" data-save-state>${this.dirty ? 'Unsaved changes' : 'Saved on Mac'}</span><button class="button button-light" type="button" data-ship-reset>Reload</button><button class="button" type="button" data-ship-save>Save + sync TK2</button></div></header>
        <div class="ship-workspace">
          <section class="ship-stage"><canvas data-ship-canvas aria-label="Interactive three-dimensional original Stellaris ship preview"></canvas><div class="ship-stage-tools"><label><span>Animation</span><select data-clip>${clips.map(clip => `<option value="${escapeHtml(clip.id)}" ${clip.id === this.design.animation.selected ? 'selected' : ''}>${escapeHtml(clip.name)}</option>`).join('')}</select></label><button class="ship-icon-button" type="button" data-play aria-label="Pause animation">Ⅱ</button><button class="ship-icon-button" type="button" data-camera aria-label="Reset camera">⌖</button><div class="ship-view-toggles"><label class="ship-marker-toggle"><input type="checkbox" data-markers checked><span>Axes</span></label><label class="ship-marker-toggle"><input type="checkbox" data-labels checked><span>Labels</span></label><label class="ship-marker-toggle"><input type="checkbox" data-skeleton><span>Rig</span></label></div></div><div class="ship-marker-help"><b>What the marker means</b><span>RGB is the animated local pose. A yellow travelling pulse is a pose-derived muzzle/rest direction. A yellow expanding pulse means the game defines only the origin and its turret resolves aim dynamically.</span></div><div class="ship-stage-readout"><span data-render-stats>Preparing GPU…</span><span>${escapeHtml(section?.name || '')}</span></div><div class="ship-axis-key"><i class="x"></i>+X red <i class="y"></i>+Y green <i class="z"></i>+Z blue <strong>local pose</strong></div></section>
          <aside class="ship-properties"><nav class="ship-tabs" aria-label="Ship properties">${[['ship', 'Ship'], ['sections', 'Sections'], ['parts', 'Parts'], ['slots', 'Slots'], ['locators', 'Locators'], ['rig', 'Rig'], ['motion', 'Motion'], ['json', 'JSON']].map(([id, name]) => `<button type="button" data-tab="${id}" class="${this.tab === id ? 'active' : ''}">${name}</button>`).join('')}</nav><div class="ship-property-body" data-property-body>${this.propertyMarkup()}</div></aside>
        </div>
      </div>`;
      void this.startRenderer();
      this.bind();
    }

    async startRenderer() {
      const stage = this.root.querySelector('.ship-stage');
      if (!this.design.model?.available) {
        stage.insertAdjacentHTML('beforeend', '<div class="ship-model-wait"><b>Original model is unavailable</b><span>Hydrate the declared project asset, then reload this example.</span></div>');
        return;
      }
      try {
        const renderer = await VanillaModelRenderer.create(this.root.querySelector('[data-ship-canvas]'), this.root.querySelector('[data-render-stats]'), this.design, this.options.sandbox, this.designId);
        if (!this.root.isConnected) { renderer.destroy(); return; }
        this.renderer = renderer;
        this.renderer.setSelection(this.rendererSelection());
      } catch (error) {
        stage.insertAdjacentHTML('beforeend', `<div class="ship-model-wait error"><b>Original model could not load</b><span>${escapeHtml(error.message)}</span></div>`);
      }
    }

    rendererSelection() {
      if (this.tab === 'parts') return { kind: 'part', id: this.currentPart()?.id };
      if (this.tab === 'locators') return { kind: 'locator', id: this.currentLocator()?.id };
      return null;
    }

    currentSection() { return this.design.sections[this.selection.section] || this.design.sections[0]; }
    currentPart() { return this.currentSection()?.parts?.[this.selection.part] || this.currentSection()?.parts?.[0]; }
    currentSlot() { return this.currentSection()?.slots?.[this.selection.slot] || this.currentSection()?.slots?.[0]; }
    currentLocator() { return this.design.locators[this.selection.locator] || this.design.locators[0]; }

    itemList(kind, items, selected, addLabel) {
      return `<div class="ship-item-browser"><label><span>Selected ${escapeHtml(kind)}</span><select data-select-list="${escapeHtml(kind)}">${items.map((item, index) => `<option value="${index}" ${index === selected ? 'selected' : ''}>${escapeHtml(item.name || item.id)} · ${escapeHtml(item.key || item.size || item.kind || item.primitive || item.id)}</option>`).join('')}</select></label><button type="button" class="ship-add" data-add="${kind}">+ ${escapeHtml(addLabel)}</button></div>`;
    }

    propertyMarkup() {
      const ship = this.design.ship;
      const rules = this.design.officialRules || {};
      const section = this.currentSection();
      const part = this.currentPart();
      const slot = this.currentSlot();
      const locator = this.currentLocator();
      if (this.tab === 'ship') return `<div class="ship-pane-heading"><p>Ship definition</p><h3>${escapeHtml(ship.name)}</h3></div><div class="ship-source-proof"><b>${escapeHtml(this.design.source?.status || '')}</b><small>${escapeHtml(rules.sourceFile || '')} · ${escapeHtml(rules.variablesFile || '')}</small></div>${panelGroup('Installed game rules', 'Read directly from ship_size and scripted-variable files', `${sourceBadge('SOURCE VERIFIED', 'verified')}${factGrid([
        ['Max hitpoints', rules.maxHitpoints], ['Max speed', rules.maxSpeed], ['Acceleration', rules.acceleration], ['Rotation speed', rules.rotationSpeed], ['Evasion', rules.evasion], ['Collision radius', rules.collisionRadius], ['Visual size multiplier', rules.sizeMultiplier], ['Fleet slots', rules.fleetSlotSize], ['Target locators', rules.targetLocators], ['Build time', rules.baseBuildTime], ['Default behavior', rules.defaultBehavior], ['Allowed roles', rules.roles], ['Ship category', rules.shipCategory], ['Bio-ship', rules.isBioShip ? 'yes' : 'no'], ['Growth threshold', rules.growthThreshold], ['Upgrades to', rules.upgradesTo], ['Native ship-browser 3D', rules.inGame3dView ? 'enabled' : 'disabled'], ['Section base cost', rules.sectionBaseCost], ['Base upkeep', rules.upkeep], ['Logistics', rules.logistics],
      ])}`)}${panelGroup('Editable preview identity', 'Changes here belong to this sandbox design', `<div class="ship-form-grid">${[field('Design id', 'ship.id', ship.id), field('Name', 'ship.name', ship.name), field('Hull size', 'ship.size', ship.size, { choices: ['corvette', 'frigate', 'destroyer', 'cruiser', 'battleship', 'titan', 'juggernaut', 'mauler_stage_1', 'custom'] }), field('Role', 'ship.role', ship.role, { choices: ['artillery', 'carrier', 'brawler', 'screen', 'torpedo', 'custom'] }), field('Entity scale', 'ship.entityScale', ship.entityScale, { step: .05 })].join('')}</div>`)}${panelGroup('Ship-size overrides', 'Initialized from the installed rules above; edit these when building a modded variant', `<div class="ship-form-grid">${[field('Hull points', 'ship.hullPoints', ship.hullPoints, { step: 10 }), field('Max speed', 'ship.combatSpeed', ship.combatSpeed), field('Rotation speed', 'ship.rotationSpeed', ship.rotationSpeed), field('Evasion', 'ship.evasion', ship.evasion), field('Fleet slots', 'ship.fleetSize', ship.fleetSize), field('Alloy cost field', 'ship.alloyCost', ship.alloyCost, { step: 10 })].join('')}</div>`)}${panelGroup('Loadout-dependent values', 'Zero means no component has been selected in this example; these are not base ship-size rules', `<div class="ship-form-grid">${[field('Armor', 'ship.armor', ship.armor, { step: 10 }), field('Shields', 'ship.shields', ship.shields, { step: 10 }), field('Power', 'ship.power', ship.power, { step: 10 }), field('Power use', 'ship.powerUse', ship.powerUse, { step: 10 }), field('Combat computer', 'ship.computer', ship.computer), field('Reactor / set', 'ship.reactor', ship.reactor), field('Thrusters / set', 'ship.thrusters', ship.thrusters), field('Sensor / set', 'ship.sensor', ship.sensor), field('FTL drive / set', 'ship.ftlDrive', ship.ftlDrive)].join('')}</div>`)}`;
      if (this.tab === 'sections') return `${this.itemList('section', this.design.sections, this.selection.section, 'Add section')}${section ? `<div class="ship-pane-heading"><p>Hull section</p><h3>${escapeHtml(section.name)}</h3></div>${panelGroup('Game template', this.design.source?.sectionTemplateFile || '', `${sourceBadge('SECTION SOURCE', 'verified')}${field('ID', `sections.${this.selection.section}.id`, section.id)}${field('Name', `sections.${this.selection.section}.name`, section.name)}${field('Game key', `sections.${this.selection.section}.key`, section.key)}${section.assetVariant ? field('Entity / asset variant', `sections.${this.selection.section}.assetVariant`, section.assetVariant) : ''}`)}${panelGroup('Assembly transform', 'Sandbox placement of this section', `${vectorFields('Position', `sections.${this.selection.section}.position`, section.position)}${vectorFields('Rotation · degrees', `sections.${this.selection.section}.rotation`, section.rotation)}`)}<button class="ship-delete" type="button" data-delete="section" ${this.design.sections.length <= 1 ? 'disabled' : ''}>Delete section</button>` : ''}`;
      if (this.tab === 'parts') return `${this.itemList('part', section.parts || [], this.selection.part, 'Add source part')}${part ? `<div class="ship-pane-heading"><p>${escapeHtml(section.name)}</p><h3>${escapeHtml(part.name)}</h3></div>${panelGroup('Original asset link', 'The GLB node is mapped back to the decoded game mesh', `${sourceBadge('ORIGINAL ASSET', 'verified')}${field('ID', `sections.${this.selection.section}.parts.${this.selection.part}.id`, part.id)}${field('Name', `sections.${this.selection.section}.parts.${this.selection.part}.name`, part.name)}${field('Original GLB node', `sections.${this.selection.section}.parts.${this.selection.part}.sourceNode`, part.sourceNode || '')}${field('Original game file', `sections.${this.selection.section}.parts.${this.selection.part}.sourceFile`, part.sourceFile || '')}`)}${panelGroup('Preview controls', 'Enable transform override only when intentionally changing the source pose', `${field('Visible', `sections.${this.selection.section}.parts.${this.selection.part}.visible`, part.visible, { type: 'checkbox' })}${field('Override source transform', `sections.${this.selection.section}.parts.${this.selection.part}.editTransform`, part.editTransform, { type: 'checkbox' })}${vectorFields('Position', `sections.${this.selection.section}.parts.${this.selection.part}.position`, part.position)}${vectorFields('Rotation · degrees', `sections.${this.selection.section}.parts.${this.selection.part}.rotation`, part.rotation)}${vectorFields('Scale', `sections.${this.selection.section}.parts.${this.selection.part}.scale`, part.scale)}`)}<button class="ship-delete" type="button" data-delete="part">Delete part</button>` : '<p class="ship-empty">No original source part has been imported for this section.</p>'}`;
      if (this.tab === 'slots') return `${this.itemList('slot', section.slots || [], this.selection.slot, 'Add slot')}${slot ? `<div class="ship-pane-heading"><p>${escapeHtml(section.name)}</p><h3>${escapeHtml(slot.name)}</h3></div>${slot.sourceTemplateLocator ? `<div class="ship-binding-proof">${sourceBadge('GAME BINDING', 'verified')}<b>Fires/spawns at “${escapeHtml(slot.sourceTemplateLocator)}”</b><small>${escapeHtml(slot.sourceTemplate || 'component template')} · ${escapeHtml(this.design.source?.sectionTemplateFile || '')}</small>${slot.embeddedLocatorCandidate ? `<small>The nearby mesh candidate ${escapeHtml(slot.embeddedLocatorCandidate)} exists but is not the shipped section binding.</small>` : ''}</div>` : ''}${panelGroup('Slot definition', 'Section-template and current component selection', `${field('ID', `sections.${this.selection.section}.slots.${this.selection.slot}.id`, slot.id)}${field('Name', `sections.${this.selection.section}.slots.${this.selection.slot}.name`, slot.name)}${field('Type', `sections.${this.selection.section}.slots.${this.selection.slot}.type`, slot.type, { choices: ['weapon', 'utility', 'auxiliary', 'strike_craft', 'point_defense', 'guided', 'custom'] })}${field('Size', `sections.${this.selection.section}.slots.${this.selection.slot}.size`, slot.size, { choices: ['S', 'M', 'L', 'X', 'T', 'G', 'H', 'P', 'A', 'W'] })}${field('Selected component', `sections.${this.selection.section}.slots.${this.selection.slot}.component`, slot.component)}${field('3D marker id', `sections.${this.selection.section}.slots.${this.selection.slot}.locatorId`, slot.locatorId || '')}${slot.sourceTemplate !== undefined ? field('Game turret template', `sections.${this.selection.section}.slots.${this.selection.slot}.sourceTemplate`, slot.sourceTemplate) : ''}${slot.sourceTemplateLocator !== undefined ? field('Game locatorname', `sections.${this.selection.section}.slots.${this.selection.slot}.sourceTemplateLocator`, slot.sourceTemplateLocator) : ''}${field('Enabled', `sections.${this.selection.section}.slots.${this.selection.slot}.enabled`, slot.enabled, { type: 'checkbox' })}`)}<button class="ship-delete" type="button" data-delete="slot">Delete slot</button>` : '<p class="ship-empty">Add the first slot.</p>'}`;
      if (this.tab === 'locators') return `${this.itemList('locator', this.design.locators, this.selection.locator, 'Add locator')}${locator ? `<div class="ship-pane-heading"><p>3D locator inspection</p><h3>${escapeHtml(locator.name)}</h3></div><div class="ship-binding-proof ${locator.usage === 'embedded_unbound' ? 'caution' : ''}">${sourceBadge(locator.usage === 'official_slot_binding' ? 'OFFICIAL SLOT ORIGIN' : locator.usage === 'embedded_unbound' ? 'EMBEDDED · UNBOUND' : 'MESH LOCATOR', locator.usage === 'embedded_unbound' ? 'caution' : 'verified')}<b>${locator.usage === 'official_slot_binding' ? 'Used by the installed section template' : locator.usage === 'embedded_unbound' ? 'Present in the mesh, not used by this section' : (locator.linkedSlot ? `Linked to ${escapeHtml(locator.linkedSlot)}` : 'No component-slot binding')}</b><small>${escapeHtml(locator.description || 'The RGB lines show this node’s animated local coordinate frame.')}</small></div>${panelGroup('Source and binding', 'Names preserved from the decoded game data', `${field('ID / marker key', `locators.${this.selection.locator}.id`, locator.id)}${field('Display name', `locators.${this.selection.locator}.name`, locator.name)}${field('Original GLB node', `locators.${this.selection.locator}.sourceNode`, locator.sourceNode || '')}${field('Original game file', `locators.${this.selection.locator}.sourceFile`, locator.sourceFile || '')}${field('Parent bone', `locators.${this.selection.locator}.parentBone`, locator.parentBone || '')}${field('Kind', `locators.${this.selection.locator}.kind`, locator.kind, { choices: ['fire_origin', 'weapon', 'engine', 'effect', 'light', 'camera', 'custom'] })}${field('Linked slot(s)', `locators.${this.selection.locator}.linkedSlot`, locator.linkedSlot || '')}`)}${panelGroup('3D marker and pose', 'Origin plus local +X red, +Y green, +Z blue; the yellow firing guide is separately declared', `${field('Visible', `locators.${this.selection.locator}.visible`, locator.visible, { type: 'checkbox' })}${field('Show 3D label', `locators.${this.selection.locator}.labelVisible`, locator.labelVisible !== false, { type: 'checkbox' })}${field('Marker scale', `locators.${this.selection.locator}.markerScale`, locator.markerScale, { step: .05 })}${field('Firing guide axis', `locators.${this.selection.locator}.fireAxis`, locator.fireAxis || '', { choices: ['', '+X', '-X', '+Y', '-Y', '+Z', '-Z'] })}${locator.fireAxisEvidence ? field('Direction evidence', `locators.${this.selection.locator}.fireAxisEvidence`, locator.fireAxisEvidence) : ''}${field('Override original transform', `locators.${this.selection.locator}.editTransform`, locator.editTransform, { type: 'checkbox' })}${vectorFields('Position · parent-local', `locators.${this.selection.locator}.position`, locator.position)}${vectorFields('Rotation · degrees', `locators.${this.selection.locator}.rotation`, locator.rotation)}`)}<button class="ship-delete" type="button" data-delete="locator">Delete locator</button>` : '<p class="ship-empty">Add the first locator.</p>'}`;
      if (this.tab === 'rig') return `<div class="ship-pane-heading"><p>Original decoded hierarchy</p><h3>${(this.design.model.bones || []).length} bones · ${(this.design.model.nodes || []).length} nodes</h3></div><div class="ship-clip-list">${(this.design.model.bones || []).map(bone => `<article><b>${escapeHtml(bone.name || bone.id)}</b><small>parent: ${escapeHtml(bone.parent || 'root')}</small></article>`).join('')}${(this.design.model.nodes || []).map(node => `<article><b>${escapeHtml(node.name || node.id)}</b><small>${escapeHtml(node.type || 'node')} · parent: ${escapeHtml(node.parent || 'root')}</small></article>`).join('') || '<p class="ship-empty">The source hierarchy will appear after the vanilla import.</p>'}</div>`;
      if (this.tab === 'motion') return `<div class="ship-pane-heading"><p>Animation player</p><h3>${escapeHtml(this.design.animation.selected)}</h3></div>${field('Playback speed', 'animation.speed', this.design.animation.speed, { step: .1 })}${field('Loop', 'animation.loop', this.design.animation.loop, { type: 'checkbox' })}<div class="ship-clip-list">${this.design.animation.clips.map((clip, index) => `<article class="${clip.officialBinding === false ? 'source-disabled' : ''}"><b>${escapeHtml(clip.name)}</b><em>${clip.officialBinding === false ? 'SOURCE FILE · BINDING DISABLED' : 'BOUND BY GAME ENTITY'}</em><small>${escapeHtml(clip.description)}</small><small>${escapeHtml(clip.sourceFile || '')} · ${escapeHtml(clip.frames || '?')} frames @ ${escapeHtml(clip.fps || '?')} fps</small>${field('Duration', `animation.clips.${index}.duration`, clip.duration, { step: .1 })}</article>`).join('')}</div>`;
      if (this.tab === 'json') return `<div class="ship-pane-heading"><p>Complete document</p><h3>Raw JSON</h3></div><textarea class="ship-json" data-json>${escapeHtml(JSON.stringify(this.design, null, 2))}</textarea><button class="button" type="button" data-apply-json>Apply JSON to preview</button>`;
      return '';
    }

    bind() {
      this.root.querySelectorAll('[data-tab]').forEach(button => button.addEventListener('click', () => { this.tab = button.dataset.tab; this.updateProperties(); }));
      this.root.querySelector('[data-design]')?.addEventListener('change', event => {
        if (this.dirty && !window.confirm('Switch examples and discard unsaved changes?')) { event.target.value = this.designId; return; }
        void this.loadDesign(event.target.value);
      });
      this.root.querySelector('[data-clip]')?.addEventListener('change', event => { this.design.animation.selected = event.target.value; this.markDirty(); this.renderer?.selectClip(event.target.value); this.renderer?.invalidate(); });
      this.root.querySelector('[data-play]')?.addEventListener('click', event => { const playing = event.currentTarget.textContent !== '▶'; event.currentTarget.textContent = playing ? '▶' : 'Ⅱ'; event.currentTarget.setAttribute('aria-label', playing ? 'Play animation' : 'Pause animation'); this.renderer?.setPlaying(!playing); });
      this.root.querySelector('[data-camera]')?.addEventListener('click', () => this.renderer?.resetView());
      this.root.querySelector('[data-markers]')?.addEventListener('change', event => this.renderer?.setMarkers(event.target.checked));
      this.root.querySelector('[data-labels]')?.addEventListener('change', event => this.renderer?.setLabels(event.target.checked));
      this.root.querySelector('[data-skeleton]')?.addEventListener('change', event => this.renderer?.setSkeleton(event.target.checked));
      this.root.querySelector('[data-ship-save]')?.addEventListener('click', event => void this.save(event.currentTarget));
      this.root.querySelector('[data-ship-reset]')?.addEventListener('click', () => { this.design = clone(this.cleanDesign); this.dirty = false; this.render(); });
      this.bindPropertyEvents();
    }

    bindPropertyEvents() {
      this.root.querySelectorAll('[data-select-list]').forEach(select => select.addEventListener('change', () => {
        const kind = select.dataset.selectList;
        if (kind === 'section') { this.selection.section = Number(select.value); this.selection.part = 0; this.selection.slot = 0; }
        else this.selection[kind] = Number(select.value);
        this.updateProperties();
      }));
      this.root.querySelectorAll('[data-select-kind]').forEach(button => button.addEventListener('click', () => {
        const kind = button.dataset.selectKind;
        if (kind === 'section') { this.selection.section = Number(button.dataset.selectIndex); this.selection.part = 0; this.selection.slot = 0; }
        else this.selection[kind] = Number(button.dataset.selectIndex);
        this.updateProperties();
      }));
      this.root.querySelectorAll('[data-design-path]').forEach(input => input.addEventListener('input', () => {
        const raw = input.type === 'checkbox' ? input.checked : input.value;
        const previous = this.getPath(input.dataset.designPath);
        const value = typeof previous === 'number' ? Number(raw) : raw;
        if (typeof previous === 'number' && !Number.isFinite(value)) return;
        this.setPath(input.dataset.designPath, value);
        this.markDirty();
        this.renderer?.setDesign(this.design);
        this.renderer?.setSelection(this.rendererSelection());
        if (input.dataset.designPath === 'ship.name') this.root.querySelector('.ship-designer-bar h2').textContent = value;
      }));
      this.root.querySelectorAll('[data-add]').forEach(button => button.addEventListener('click', () => this.addItem(button.dataset.add)));
      this.root.querySelectorAll('[data-delete]').forEach(button => button.addEventListener('click', () => this.deleteItem(button.dataset.delete)));
      this.root.querySelector('[data-apply-json]')?.addEventListener('click', () => {
        try { this.design = JSON.parse(this.root.querySelector('[data-json]').value); this.markDirty(); this.render(); } catch (error) { this.setSaveState(`JSON error: ${error.message}`, true); }
      });
    }

    updateProperties() {
      this.root.querySelectorAll('[data-tab]').forEach(button => button.classList.toggle('active', button.dataset.tab === this.tab));
      this.root.querySelector('[data-property-body]').innerHTML = this.propertyMarkup();
      this.bindPropertyEvents();
      this.renderer?.setSelection(this.rendererSelection());
    }

    getPath(path) { return path.split('.').reduce((value, key) => value?.[Number.isInteger(Number(key)) && key !== '' ? Number(key) : key], this.design); }
    setPath(path, value) { const keys = path.split('.'); const last = keys.pop(); const parent = keys.reduce((item, key) => item[Number.isInteger(Number(key)) && key !== '' ? Number(key) : key], this.design); parent[Number.isInteger(Number(last)) && last !== '' ? Number(last) : last] = value; }
    markDirty() { this.dirty = true; this.setSaveState('Unsaved changes'); }
    setSaveState(message, error = false) { const state = this.root.querySelector('[data-save-state]'); if (state) { state.textContent = message; state.classList.toggle('error', error); } }

    addItem(kind) {
      const stamp = Date.now().toString(36).slice(-5);
      if (kind === 'section') { this.design.sections.push({ id: `section_${stamp}`, name: 'New Section', key: 'CUSTOM_SECTION', position: [0, 0, 0], rotation: [0, 0, 0], parts: [], slots: [] }); this.selection.section = this.design.sections.length - 1; }
      if (kind === 'part') { const list = this.currentSection().parts; list.push({ id: `part_${stamp}`, name: 'New source part', sourceNode: '', sourceFile: '', position: [0, 0, 0], rotation: [0, 0, 0], scale: [1, 1, 1], visible: true, editTransform: false }); this.selection.part = list.length - 1; }
      if (kind === 'slot') { const list = this.currentSection().slots; list.push({ id: `slot_${stamp}`, name: 'New slot', type: 'weapon', size: 'M', component: '', locatorId: '', enabled: true }); this.selection.slot = list.length - 1; }
      if (kind === 'locator') { this.design.locators.push({ id: `locator_${stamp}`, name: 'New locator', kind: 'weapon', linkedSlot: '', position: [0, 0, 0], rotation: [0, 0, 0], markerScale: .6, visible: true }); this.selection.locator = this.design.locators.length - 1; }
      this.markDirty(); this.updateProperties(); this.renderer?.invalidate();
    }

    deleteItem(kind) {
      if (kind === 'section' && this.design.sections.length > 1) { this.design.sections.splice(this.selection.section, 1); this.selection.section = clamp(this.selection.section, 0, this.design.sections.length - 1); }
      if (kind === 'part') { this.currentSection().parts.splice(this.selection.part, 1); this.selection.part = clamp(this.selection.part, 0, Math.max(0, this.currentSection().parts.length - 1)); }
      if (kind === 'slot') { this.currentSection().slots.splice(this.selection.slot, 1); this.selection.slot = clamp(this.selection.slot, 0, Math.max(0, this.currentSection().slots.length - 1)); }
      if (kind === 'locator') { this.design.locators.splice(this.selection.locator, 1); this.selection.locator = clamp(this.selection.locator, 0, Math.max(0, this.design.locators.length - 1)); }
      this.markDirty(); this.updateProperties(); this.renderer?.invalidate();
    }

    async save(button) {
      button.disabled = true;
      this.setSaveState('Saving on Mac + syncing TK2…');
      try {
        const response = await fetch('/api/designer/config', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ sandbox: this.options.sandbox, designId: this.designId, design: this.design, syncTk2: true }) });
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.error || 'Save failed.');
        if (payload.sync && !payload.sync.ok) throw new Error('Saved on Mac, but TK2 sync failed.');
        this.cleanDesign = clone(this.design);
        this.dirty = false;
        this.setSaveState(payload.sync ? 'Saved · TK2 synced' : 'Saved on Mac');
      } catch (error) { this.setSaveState(error.message, true); }
      finally { button.disabled = false; }
    }

    destroy() { this.renderer?.destroy(); this.renderer = null; this.root.innerHTML = ''; }
  }

  window.StellarisShipDesigner = {
    mount(root, options) { const designer = new Designer(root, options); void designer.mount(); return designer; },
  };
})();
