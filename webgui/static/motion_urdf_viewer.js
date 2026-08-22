(function () {
  'use strict';

  const I4 = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
  const DEFAULT_MESH_COLOR = '#aeb8bd';

  class MotionUrdfViewer {
    constructor(options) {
      this.meshCanvas = options.meshCanvas;
      this.overlayCanvas = options.overlayCanvas;
      this.ctx = this.overlayCanvas.getContext('2d');
      this.assetUrl = options.assetUrl;
      this.onStatus = options.onStatus || (() => {});
      this.gl = this.meshCanvas.getContext('webgl', { antialias: true, alpha: true });
      this.webgl = null;
      try { this.webgl = this.gl ? initWebgl(this.gl) : null; } catch (error) { console.warn(error); }
      this.model = null;
      this.armJointNames = [];
      this.meshCache = new Map();
      this.meshStats = { visuals: 0, loaded: 0, failed: 0, triangles: 0 };
      this.overlayBounds = [];
      this.overlayDrawer = null;
      this.target = [0, 0, 0.38];
      this.yaw = -0.72;
      this.pitch = 0.42;
      this.distance = 3.2;
      this.zoom = 1;
      this.viewScale = 500;
      this.drawQueued = false;
      this.fitPending = true;
      this.destroyed = false;
      document.body.dataset.robotRenderMode = 'loading';
    }

    async load(urdfUrl, jointNames, jointValues) {
      this.onStatus('Loading robot URDF');
      let response = await fetch(urdfUrl, { cache: 'no-store' });
      if (!response.ok && [404, 502, 503].includes(response.status) && !this.destroyed) {
        await new Promise(resolve => setTimeout(resolve, 300));
        response = await fetch(urdfUrl, { cache: 'no-store' });
      }
      const payload = await response.json();
      if (this.destroyed) return null;
      if (!response.ok || payload.error) throw new Error(payload.error || response.statusText);
      this.model = parseUrdf(payload.urdf);
      this.armJointNames = Array.from(jointNames || []);
      const values = Array.from(jointValues || []);
      if (values.length !== this.armJointNames.length || values.some(value => !Number.isFinite(Number(value)))) {
        throw new Error('Robot joint names and initial values must be finite and have matching lengths');
      }
      const urdfJointNames = new Set(this.model.joints.map(joint => joint.name));
      const missingJoints = this.armJointNames.filter(name => !urdfJointNames.has(name));
      if (missingJoints.length) throw new Error(`Robot profile joints missing from URDF: ${missingJoints.join(', ')}`);
      for (const joint of this.model.joints) this.model.jointValues[joint.name] = 0;
      if (this.armJointNames.length) this.setJoints(this.armJointNames, values, false);
      this.updateLinkWorlds();
      this.fitPending = true;
      this.draw();
      this.onStatus('Loading URDF meshes');
      await this.loadMeshes();
      if (this.destroyed) return null;
      this.fitPending = true;
      this.draw();
      document.body.dataset.robotMeshVisuals = String(this.meshStats.visuals);
      document.body.dataset.robotMeshLoaded = String(this.meshStats.loaded);
      document.body.dataset.robotMeshTriangles = String(this.meshStats.triangles);
      document.body.dataset.robotRenderable = String(this.meshStats.loaded > 0 || this.model.joints.length > 0);
      document.body.dataset.robotRenderMode = this.webgl && this.meshStats.loaded > 0 ? 'webgl-meshes-with-link-fallback' : 'link-fallback';
      this.onStatus(`${this.meshStats.loaded}/${this.meshStats.visuals} URDF meshes · ${this.meshStats.triangles.toLocaleString()} triangles`);
      return { ...this.meshStats, webgl: Boolean(this.webgl) };
    }

    setJoints(names, values, redraw = true) {
      if (!this.model) return;
      const jointNames = Array.from(names || []);
      const jointValues = Array.from(values || [], Number);
      if (!jointNames.length || jointNames.length !== jointValues.length || jointValues.some(value => !Number.isFinite(value))) {
        throw new Error('Robot joint names and values must be finite and have matching nonzero lengths');
      }
      const missingJoints = jointNames.filter(name => !Object.prototype.hasOwnProperty.call(this.model.jointValues, name));
      if (missingJoints.length) throw new Error(`Robot joints missing from loaded model: ${missingJoints.join(', ')}`);
      jointNames.forEach((name, index) => { this.model.jointValues[name] = jointValues[index]; });
      this.updateLinkWorlds();
      if (redraw) this.draw();
    }

    setOverlay({ boundsPoints, draw, refit = false }) {
      this.overlayBounds = Array.isArray(boundsPoints) ? boundsPoints : [];
      this.overlayDrawer = typeof draw === 'function' ? draw : null;
      if (refit) this.fitPending = true;
      this.draw();
    }

    resetView() {
      this.yaw = -0.72;
      this.pitch = 0.42;
      this.distance = 3.2;
      this.zoom = 1;
      this.fitPending = true;
      this.draw();
    }

    orbit(dx, dy) {
      this.yaw += Number(dx) * 0.008;
      this.pitch = Math.max(-1.25, Math.min(1.25, this.pitch + Number(dy) * 0.008));
      this.draw();
    }

    dolly(delta) {
      this.zoom = Math.max(0.35, Math.min(3.2, this.zoom * Math.exp(-Number(delta) * 0.001)));
      this.draw();
    }

    linkMatrix(name) {
      return this.model?.linkWorld?.[name] || I4;
    }

    linkPose(name) {
      const matrix = Array.from(this.linkMatrix(name));
      return {
        positionM: transformPoint(matrix, [0, 0, 0]),
        quaternionXyzw: matrixQuaternion(matrix),
      };
    }

    updateLinkWorlds() {
      if (!this.model) return;
      this.model.linkWorld = {};
      const roots = this.model.rootLinks.length ? this.model.rootLinks : Object.keys(this.model.links).slice(0, 1);
      const visit = linkName => {
        for (const joint of this.model.childrenByParent[linkName] || []) {
          const parent = this.model.linkWorld[linkName] || I4;
          const value = this.model.jointValues[joint.name] || 0;
          this.model.linkWorld[joint.child] = matMul(parent, jointMatrix(joint, value));
          visit(joint.child);
        }
      };
      for (const root of roots) {
        this.model.linkWorld[root] = I4;
        visit(root);
      }
    }

    async loadMeshes() {
      if (!this.model) return;
      const visuals = [];
      for (const link of Object.values(this.model.links)) {
        for (const visual of link.visuals) visuals.push(visual);
      }
      let nextVisual = 0;
      const worker = async () => {
        while (!this.destroyed && nextVisual < visuals.length) {
          const visual = visuals[nextVisual];
          nextVisual += 1;
          await this.loadVisualMesh(visual);
        }
      };
      await Promise.all(Array.from({ length: Math.min(8, visuals.length) }, () => worker()));
      const stats = { visuals: 0, loaded: 0, failed: 0, triangles: 0 };
      for (const link of Object.values(this.model.links)) {
        for (const visual of link.visuals) {
          stats.visuals += 1;
          if (visual.mesh?.triangleCount > 0) { stats.loaded += 1; stats.triangles += visual.mesh.triangleCount; }
          if (visual.error) stats.failed += 1;
        }
      }
      this.meshStats = stats;
    }

    destroy() {
      this.destroyed = true;
      if (this.gl) {
        for (const mesh of this.meshCache.values()) {
          if (mesh.buffers?.positions) this.gl.deleteBuffer(mesh.buffers.positions);
          if (mesh.buffers?.normals) this.gl.deleteBuffer(mesh.buffers.normals);
          mesh.buffers = null;
        }
      }
      this.meshCache.clear();
      if (this.gl) {
        this.gl.clearColor(0, 0, 0, 0);
        this.gl.clear(this.gl.COLOR_BUFFER_BIT | this.gl.DEPTH_BUFFER_BIT);
      }
      this.ctx.clearRect(0, 0, this.overlayCanvas.width, this.overlayCanvas.height);
      this.model = null;
      this.overlayDrawer = null;
    }

    async loadVisualMesh(visual) {
      if (!visual.uri) return;
      try {
        if (!this.meshCache.has(visual.uri)) {
          const separator = this.assetUrl.includes('?') ? '&' : '?';
          const assetRequest = `${this.assetUrl}${separator}uri=${encodeURIComponent(visual.uri)}`;
          let response = await fetch(assetRequest);
          if (!response.ok && [404, 502, 503].includes(response.status) && !this.destroyed) {
            await new Promise(resolve => setTimeout(resolve, 300));
            response = await fetch(assetRequest, { cache: 'reload' });
          }
          if (!response.ok) throw new Error(response.statusText || `HTTP ${response.status}`);
          const mesh = parseStl(await response.arrayBuffer());
          if (this.destroyed) return;
          if (!mesh.triangleCount) throw new Error('Mesh contains no renderable triangles');
          this.prepareBuffers(mesh);
          this.meshCache.set(visual.uri, mesh);
        }
        visual.mesh = this.meshCache.get(visual.uri);
      } catch (error) {
        visual.error = String(error);
      }
    }

    prepareBuffers(mesh) {
      if (!this.webgl || !mesh?.positions?.length || mesh.buffers) return;
      const positions = this.gl.createBuffer();
      this.gl.bindBuffer(this.gl.ARRAY_BUFFER, positions);
      this.gl.bufferData(this.gl.ARRAY_BUFFER, mesh.positions, this.gl.STATIC_DRAW);
      const normals = this.gl.createBuffer();
      this.gl.bindBuffer(this.gl.ARRAY_BUFFER, normals);
      this.gl.bufferData(this.gl.ARRAY_BUFFER, mesh.normals, this.gl.STATIC_DRAW);
      mesh.buffers = { positions, normals };
    }

    draw() {
      if (this.destroyed || this.drawQueued) return;
      this.drawQueued = true;
      requestAnimationFrame(() => {
        this.drawQueued = false;
        if (this.destroyed) return;
        this.render();
      });
    }

    render() {
      this.fitCanvases();
      if (this.fitPending) this.fitView();
      this.drawMeshes();
      this.ctx.clearRect(0, 0, this.overlayCanvas.width, this.overlayCanvas.height);
      this.drawGrid();
      this.drawLinkTree(this.webgl && this.meshStats.loaded > 0);
      this.drawArmJoints();
      if (this.overlayDrawer) this.overlayDrawer(this);
    }

    fitCanvases() {
      const rect = this.overlayCanvas.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      const width = Math.max(320, Math.floor(rect.width * dpr));
      const height = Math.max(260, Math.floor(rect.height * dpr));
      for (const canvas of [this.meshCanvas, this.overlayCanvas]) {
        if (canvas.width !== width || canvas.height !== height) { canvas.width = width; canvas.height = height; }
      }
      if (this.gl) this.gl.viewport(0, 0, width, height);
    }

    sceneBounds() {
      const bounds = emptyBounds();
      let seen = false;
      if (this.model) {
        for (const [linkName, link] of Object.entries(this.model.links)) {
          const linkWorld = this.model.linkWorld[linkName] || I4;
          for (const visual of link.visuals) {
            if (!visual.mesh?.bounds) continue;
            const transform = matMul(linkWorld, visual.origin);
            for (const corner of boundsCorners(visual.mesh.bounds)) {
              includeBounds(bounds, transformPoint(transform, mulVec(corner, visual.scale)));
              seen = true;
            }
          }
        }
        if (!seen) {
          for (const matrix of Object.values(this.model.linkWorld)) { includeBounds(bounds, transformPoint(matrix, [0, 0, 0])); seen = true; }
        }
      }
      for (const point of this.overlayBounds) {
        if (Array.isArray(point) && point.length === 3) { includeBounds(bounds, point); seen = true; }
      }
      return seen ? bounds : null;
    }

    fitView() {
      const bounds = this.sceneBounds();
      if (!bounds) return;
      this.target = scale(add(bounds.min, bounds.max), 0.5);
      const extent = Math.max(bounds.max[0] - bounds.min[0], bounds.max[1] - bounds.min[1], bounds.max[2] - bounds.min[2], 0.15);
      this.viewScale = Math.min(this.overlayCanvas.width, this.overlayCanvas.height) * 0.82 / extent;
      this.fitPending = false;
    }

    project(point) {
      const p = sub(point, this.target);
      const cy = Math.cos(this.yaw), sy = Math.sin(this.yaw);
      const cp = Math.cos(this.pitch), sp = Math.sin(this.pitch);
      const x1 = cy * p[0] - sy * p[1];
      const y1 = sy * p[0] + cy * p[1];
      const y2 = cp * y1 - sp * p[2];
      const z2 = sp * y1 + cp * p[2];
      const perspective = this.distance / Math.max(0.8, this.distance - z2);
      return [
        this.overlayCanvas.width * 0.5 + x1 * this.viewScale * this.zoom * perspective,
        this.overlayCanvas.height * 0.48 - y2 * this.viewScale * this.zoom * perspective,
        z2,
      ];
    }

    line(a, b, color, width = 1, dash = []) {
      const pa = this.project(a), pb = this.project(b);
      this.ctx.beginPath();
      this.ctx.setLineDash(dash);
      this.ctx.strokeStyle = color;
      this.ctx.lineWidth = width * (window.devicePixelRatio || 1);
      this.ctx.moveTo(pa[0], pa[1]); this.ctx.lineTo(pb[0], pb[1]); this.ctx.stroke();
      this.ctx.setLineDash([]);
    }

    polyline(points, color, width, dash = []) {
      if (!points || points.length < 2) return;
      this.ctx.beginPath();
      this.ctx.setLineDash(dash);
      this.ctx.strokeStyle = color;
      this.ctx.lineWidth = width * (window.devicePixelRatio || 1);
      this.ctx.lineJoin = 'round'; this.ctx.lineCap = 'round';
      points.forEach((point, index) => { const p = this.project(point); index ? this.ctx.lineTo(p[0], p[1]) : this.ctx.moveTo(p[0], p[1]); });
      this.ctx.stroke(); this.ctx.setLineDash([]);
    }

    point(point, radius, fill, stroke = '#fffaf0') {
      const p = this.project(point), dpr = window.devicePixelRatio || 1;
      this.ctx.beginPath(); this.ctx.arc(p[0], p[1], radius * dpr, 0, Math.PI * 2);
      this.ctx.fillStyle = fill; this.ctx.fill();
      this.ctx.strokeStyle = stroke; this.ctx.lineWidth = 1.2 * dpr; this.ctx.stroke();
    }

    drawGrid() {
      for (let index = -8; index <= 8; index += 1) {
        this.line([index * 0.1, -0.8, 0], [index * 0.1, 0.8, 0], 'rgba(147,161,161,.25)', 0.7);
        this.line([-0.8, index * 0.1, 0], [0.8, index * 0.1, 0], 'rgba(147,161,161,.25)', 0.7);
      }
      this.line([-0.8, 0, 0], [0.8, 0, 0], 'rgba(101,123,131,.55)', 1);
      this.line([0, -0.8, 0], [0, 0.8, 0], 'rgba(101,123,131,.55)', 1);
    }

    drawArmJoints() {
      if (!this.model) return;
      if (this.armJointNames.length > 32) return;
      const dpr = window.devicePixelRatio || 1;
      const important = this.armJointNames.filter(name => /waist|neck|head|shoulder|elbow|wrist|hip|knee|ankle|toe|shin/.test(name));
      const candidates = important.length ? important : this.armJointNames;
      const stride = Math.max(1, Math.ceil(candidates.length / 18));
      candidates.filter((_, index) => index % stride === 0).forEach(name => {
        const joint = this.model.joints.find(item => item.name === name);
        const matrix = joint ? this.model.linkWorld[joint.child] : null;
        if (!matrix) return;
        const origin = transformPoint(matrix, [0, 0, 0]);
        this.point(origin, 3.1, '#fdf6e3', '#268bd2');
        const screen = this.project(origin);
        this.ctx.fillStyle = '#165d91';
        this.ctx.font = `bold ${7.5 * dpr}px ui-monospace, SFMono-Regular, Menlo, monospace`;
        this.ctx.fillText(name.replace(/_joint$/, ''), screen[0] + 4 * dpr, screen[1] + 10 * dpr);
      });
    }

    drawLinkTree(subtle = false) {
      if (!this.model) return;
      for (const joint of this.model.joints) {
        const parent = this.model.linkWorld[joint.parent], child = this.model.linkWorld[joint.child];
        if (parent && child) this.line(transformPoint(parent, [0, 0, 0]), transformPoint(child, [0, 0, 0]), subtle ? 'rgba(38,139,210,.22)' : 'rgba(38,139,210,.75)', subtle ? 1.2 : 2.5);
      }
    }

    drawMeshes() {
      if (!this.webgl || !this.model) {
        if (this.gl) { this.gl.clearColor(0, 0, 0, 0); this.gl.clear(this.gl.COLOR_BUFFER_BIT); }
        return;
      }
      const gl = this.gl, program = this.webgl;
      gl.clearColor(0, 0, 0, 0); gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
      gl.useProgram(program.program); gl.enable(gl.DEPTH_TEST); gl.disable(gl.CULL_FACE);
      gl.uniform3fv(program.uniforms.target, new Float32Array(this.target));
      gl.uniform1f(program.uniforms.yaw, this.yaw); gl.uniform1f(program.uniforms.pitch, this.pitch);
      gl.uniform1f(program.uniforms.distance, this.distance); gl.uniform1f(program.uniforms.viewScale, this.viewScale * this.zoom);
      gl.uniform2f(program.uniforms.canvasSize, this.meshCanvas.width, this.meshCanvas.height);
      gl.uniform1f(program.uniforms.depthSpan, Math.max(10, this.distance * 4));
      for (const [linkName, link] of Object.entries(this.model.links)) {
        const linkWorld = this.model.linkWorld[linkName] || I4;
        for (const visual of link.visuals) {
          const mesh = visual.mesh;
          if (!mesh?.buffers || !mesh.triangleCount) continue;
          const matrix = matMul(linkWorld, matMul(visual.origin, scaleMatrix(visual.scale)));
          gl.uniform4f(program.uniforms.row0, matrix[0], matrix[1], matrix[2], matrix[3]);
          gl.uniform4f(program.uniforms.row1, matrix[4], matrix[5], matrix[6], matrix[7]);
          gl.uniform4f(program.uniforms.row2, matrix[8], matrix[9], matrix[10], matrix[11]);
          gl.uniform3fv(program.uniforms.color, new Float32Array(hexToRgb01(visual.color || colorForName(linkName) || DEFAULT_MESH_COLOR)));
          gl.bindBuffer(gl.ARRAY_BUFFER, mesh.buffers.positions);
          gl.vertexAttribPointer(program.attributes.position, 3, gl.FLOAT, false, 0, 0); gl.enableVertexAttribArray(program.attributes.position);
          gl.bindBuffer(gl.ARRAY_BUFFER, mesh.buffers.normals);
          gl.vertexAttribPointer(program.attributes.normal, 3, gl.FLOAT, false, 0, 0); gl.enableVertexAttribArray(program.attributes.normal);
          gl.drawArrays(gl.TRIANGLES, 0, mesh.triangleCount * 3);
        }
      }
    }
  }

  function parseUrdf(xmlText) {
    const doc = new DOMParser().parseFromString(xmlText, 'application/xml');
    const error = doc.querySelector('parsererror');
    if (error) throw new Error(error.textContent || 'Invalid URDF XML');
    const materials = {};
    for (const node of doc.querySelectorAll('robot > material')) {
      const color = node.querySelector('color'), name = node.getAttribute('name') || '';
      if (name && color) materials[name] = colorFromRgba(color.getAttribute('rgba'));
    }
    const links = {};
    for (const linkNode of doc.querySelectorAll('robot > link')) {
      const name = linkNode.getAttribute('name') || '';
      const visuals = [];
      for (const visualNode of linkNode.querySelectorAll(':scope > visual')) {
        const meshNode = visualNode.querySelector('geometry > mesh');
        if (!meshNode) continue;
        const material = visualNode.querySelector('material'), color = material?.querySelector('color');
        visuals.push({
          uri: meshNode.getAttribute('filename') || '',
          scale: parseVector(meshNode.getAttribute('scale'), [1, 1, 1]),
          origin: parseOrigin(visualNode.querySelector('origin')),
          color: color ? colorFromRgba(color.getAttribute('rgba')) : (materials[material?.getAttribute('name') || ''] || null),
          mesh: null, error: '',
        });
      }
      links[name] = { name, visuals };
    }
    const joints = Array.from(doc.querySelectorAll('robot > joint')).map(node => ({
      name: node.getAttribute('name') || '', type: node.getAttribute('type') || 'fixed',
      parent: node.querySelector('parent')?.getAttribute('link') || '', child: node.querySelector('child')?.getAttribute('link') || '',
      origin: parseOrigin(node.querySelector('origin')), axis: norm(parseVector(node.querySelector('axis')?.getAttribute('xyz'), [1, 0, 0])),
    }));
    const childrenByParent = {};
    for (const joint of joints) (childrenByParent[joint.parent] ||= []).push(joint);
    const childLinks = new Set(joints.map(joint => joint.child));
    return { links, joints, childrenByParent, rootLinks: Object.keys(links).filter(name => !childLinks.has(name)), jointValues: {}, linkWorld: {} };
  }

  function parseStl(buffer) {
    const view = new DataView(buffer);
    if (buffer.byteLength >= 84) {
      const count = view.getUint32(80, true);
      if (84 + count * 50 === buffer.byteLength) {
        const positions = new Float32Array(count * 9), normals = new Float32Array(count * 9), bounds = emptyBounds();
        for (let index = 0; index < count; index += 1) {
          const offset = 84 + index * 50;
          const normal = [view.getFloat32(offset, true), view.getFloat32(offset + 4, true), view.getFloat32(offset + 8, true)];
          for (let vertex = 0; vertex < 3; vertex += 1) {
            const source = offset + 12 + vertex * 12, target = index * 9 + vertex * 3;
            const point = [view.getFloat32(source, true), view.getFloat32(source + 4, true), view.getFloat32(source + 8, true)];
            positions.set(point, target); normals.set(normal, target); includeBounds(bounds, point);
          }
        }
        return { positions, normals, triangleCount: count, bounds, buffers: null };
      }
    }
    const vertices = Array.from(new TextDecoder('utf-8').decode(buffer).matchAll(/vertex\s+([^\s]+)\s+([^\s]+)\s+([^\s]+)/g)).map(match => [Number(match[1]), Number(match[2]), Number(match[3])]);
    const count = Math.floor(vertices.length / 3), positions = new Float32Array(count * 9), normals = new Float32Array(count * 9), bounds = emptyBounds();
    for (let index = 0; index < count; index += 1) {
      const tri = [vertices[index * 3], vertices[index * 3 + 1], vertices[index * 3 + 2]];
      const normal = norm(cross(sub(tri[1], tri[0]), sub(tri[2], tri[0])));
      tri.forEach((point, vertex) => { positions.set(point, index * 9 + vertex * 3); normals.set(normal, index * 9 + vertex * 3); includeBounds(bounds, point); });
    }
    return { positions, normals, triangleCount: count, bounds, buffers: null };
  }

  function initWebgl(gl) {
    const vertex = compileShader(gl, gl.VERTEX_SHADER, `
      attribute vec3 a_position; attribute vec3 a_normal;
      uniform vec4 u_row0; uniform vec4 u_row1; uniform vec4 u_row2;
      uniform vec3 u_target; uniform float u_yaw; uniform float u_pitch; uniform float u_distance;
      uniform float u_viewScale; uniform vec2 u_canvasSize; uniform float u_depthSpan; varying float v_light;
      void main() {
        vec4 local=vec4(a_position,1.0); vec3 world=vec3(dot(u_row0,local),dot(u_row1,local),dot(u_row2,local));
        vec4 nl=vec4(a_normal,0.0); vec3 nw=normalize(vec3(dot(u_row0,nl),dot(u_row1,nl),dot(u_row2,nl)));
        vec3 p=world-u_target; float cy=cos(u_yaw),sy=sin(u_yaw),cp=cos(u_pitch),sp=sin(u_pitch);
        float x1=cy*p.x-sy*p.y; float y1=sy*p.x+cy*p.y; float y2=cp*y1-sp*p.z; float z2=sp*y1+cp*p.z;
        float perspective=u_distance/max(0.8,u_distance-z2);
        gl_Position=vec4(x1*u_viewScale*perspective/(u_canvasSize.x*.5), .04+y2*u_viewScale*perspective/(u_canvasSize.y*.5), clamp(.5-z2/u_depthSpan,.001,.999), 1.0);
        vec3 lightDir=normalize(vec3(-.35,-.45,.82)); v_light=.58+max(dot(nw,lightDir),0.0)*.34+max(nw.z,0.0)*.08;
      }`);
    const fragment = compileShader(gl, gl.FRAGMENT_SHADER, `precision mediump float; uniform vec3 u_color; varying float v_light; void main(){gl_FragColor=vec4(u_color*v_light,1.0);}`);
    const program = gl.createProgram(); gl.attachShader(program, vertex); gl.attachShader(program, fragment); gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(program) || 'WebGL link failed');
    const uniform = name => gl.getUniformLocation(program, name);
    return { program, attributes: { position: gl.getAttribLocation(program, 'a_position'), normal: gl.getAttribLocation(program, 'a_normal') }, uniforms: { row0: uniform('u_row0'), row1: uniform('u_row1'), row2: uniform('u_row2'), target: uniform('u_target'), yaw: uniform('u_yaw'), pitch: uniform('u_pitch'), distance: uniform('u_distance'), viewScale: uniform('u_viewScale'), canvasSize: uniform('u_canvasSize'), depthSpan: uniform('u_depthSpan'), color: uniform('u_color') } };
  }

  function compileShader(gl, type, source) {
    const shader = gl.createShader(type); gl.shaderSource(shader, source); gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(shader) || 'WebGL shader failed');
    return shader;
  }

  function jointMatrix(joint, value) { return matMul(joint.origin, joint.type === 'revolute' || joint.type === 'continuous' ? axisAngle(joint.axis, value) : joint.type === 'prismatic' ? translation(scale(joint.axis, value)) : I4); }
  function parseOrigin(node) { return matMul(translation(parseVector(node?.getAttribute('xyz'), [0, 0, 0])), rpyMatrix(parseVector(node?.getAttribute('rpy'), [0, 0, 0]))); }
  function parseVector(text, fallback) { if (!text) return Array.from(fallback); const values = String(text).trim().split(/\s+/).map(Number).filter(Number.isFinite); return values.length ? values : Array.from(fallback); }
  function colorFromRgba(value) { const rgb = parseVector(value, [.68, .72, .74, 1]).slice(0, 3).map(number => Math.round(Math.max(0, Math.min(1, number)) * 255)); return `#${rgb.map(number => number.toString(16).padStart(2, '0')).join('')}`; }
  function colorForName(name) { let hash = 0; for (let index = 0; index < name.length; index += 1) hash = (hash * 31 + name.charCodeAt(index)) >>> 0; return ['#9fb7d5','#d5aa73','#9dc7a8','#c3a6d9','#d2c06f','#9cc7c9'][hash % 6]; }
  function hexToRgb01(hex) { const value = hex.replace('#', ''); return [0, 2, 4].map(index => parseInt(value.slice(index, index + 2), 16) / 255); }
  function emptyBounds() { return { min: [Infinity, Infinity, Infinity], max: [-Infinity, -Infinity, -Infinity] }; }
  function includeBounds(bounds, point) { bounds.min = bounds.min.map((value, index) => Math.min(value, point[index])); bounds.max = bounds.max.map((value, index) => Math.max(value, point[index])); }
  function boundsCorners(bounds) { const a=bounds.min,b=bounds.max; return [[a[0],a[1],a[2]],[b[0],a[1],a[2]],[a[0],b[1],a[2]],[b[0],b[1],a[2]],[a[0],a[1],b[2]],[b[0],a[1],b[2]],[a[0],b[1],b[2]],[b[0],b[1],b[2]]]; }
  function add(a,b){return [a[0]+b[0],a[1]+b[1],a[2]+b[2]];} function sub(a,b){return [a[0]-b[0],a[1]-b[1],a[2]-b[2]];} function scale(a,s){return [a[0]*s,a[1]*s,a[2]*s];} function mulVec(a,b){return [a[0]*b[0],a[1]*b[1],a[2]*b[2]];}
  function cross(a,b){return [a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];} function norm(a){const length=Math.hypot(a[0],a[1],a[2])||1;return scale(a,1/length);}
  function matMul(a,b){const out=new Array(16).fill(0);for(let row=0;row<4;row+=1)for(let col=0;col<4;col+=1)out[row*4+col]=a[row*4]*b[col]+a[row*4+1]*b[4+col]+a[row*4+2]*b[8+col]+a[row*4+3]*b[12+col];return out;}
  function translation(v){return [1,0,0,v[0],0,1,0,v[1],0,0,1,v[2],0,0,0,1];} function scaleMatrix(v){return [v[0],0,0,0,0,v[1],0,0,0,0,v[2],0,0,0,0,1];}
  function rpyMatrix(rpy){const [r,p,y]=rpy,sr=Math.sin(r),cr=Math.cos(r),sp=Math.sin(p),cp=Math.cos(p),sy=Math.sin(y),cy=Math.cos(y);return [cy*cp,cy*sp*sr-sy*cr,cy*sp*cr+sy*sr,0,sy*cp,sy*sp*sr+cy*cr,sy*sp*cr-cy*sr,0,-sp,cp*sr,cp*cr,0,0,0,0,1];}
  function axisAngle(axis,angle){const [x,y,z]=norm(axis),c=Math.cos(angle),s=Math.sin(angle),t=1-c;return [t*x*x+c,t*x*y-s*z,t*x*z+s*y,0,t*x*y+s*z,t*y*y+c,t*y*z-s*x,0,t*x*z-s*y,t*y*z+s*x,t*z*z+c,0,0,0,0,1];}
  function matrixQuaternion(m){const trace=m[0]+m[5]+m[10];let x,y,z,w;if(trace>0){const s=Math.sqrt(trace+1)*2;w=.25*s;x=(m[9]-m[6])/s;y=(m[2]-m[8])/s;z=(m[4]-m[1])/s;}else if(m[0]>m[5]&&m[0]>m[10]){const s=Math.sqrt(1+m[0]-m[5]-m[10])*2;w=(m[9]-m[6])/s;x=.25*s;y=(m[1]+m[4])/s;z=(m[2]+m[8])/s;}else if(m[5]>m[10]){const s=Math.sqrt(1+m[5]-m[0]-m[10])*2;w=(m[2]-m[8])/s;x=(m[1]+m[4])/s;y=.25*s;z=(m[6]+m[9])/s;}else{const s=Math.sqrt(1+m[10]-m[0]-m[5])*2;w=(m[4]-m[1])/s;x=(m[2]+m[8])/s;y=(m[6]+m[9])/s;z=.25*s;}return[x,y,z,w];}
  function transformPoint(m,p){return [m[0]*p[0]+m[1]*p[1]+m[2]*p[2]+m[3],m[4]*p[0]+m[5]*p[1]+m[6]*p[2]+m[7],m[8]*p[0]+m[9]*p[1]+m[10]*p[2]+m[11]];}

  window.MotionUrdfViewer = MotionUrdfViewer;
})();
