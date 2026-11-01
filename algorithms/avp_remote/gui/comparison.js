import * as THREE from 'three';

// H1-2 mesh front (+X) faces the same displayed direction as Landau (-Y).
export const UNITREE_DISPLAY_YAW = -Math.PI / 2;

// This is a display mapping, not a measured H1-2 pose or a robot command.
export function unitreePose(data, landau) {
  const values = {};
  for (const [name, rule] of Object.entries(data.rules)) {
    let value = rule.bias;
    for (const [key, weight] of Object.entries(rule.weights)) value += (landau[key] || 0) * weight;
    values[name] = Math.max(rule.minimum ?? -Infinity, value);
  }
  const pending = [...data.joints], result = {}, resolved = new Set();
  while (pending.length) {
    let changed = false;
    for (let i = pending.length - 1; i >= 0; i--) {
      const j = pending[i];
      if (j.mimic && !resolved.has(j.mimic.joint)) continue;
      const value = j.mimic ? values[j.mimic.joint] * j.mimic.multiplier + j.mimic.offset : values[j.name] || 0;
      values[j.name] = THREE.MathUtils.clamp(value, j.lower, j.upper);
      result[j.child] = values[j.name];
      resolved.add(j.name);
      pending.splice(i, 1); changed = true;
    }
    if (!changed) throw new Error('Unresolved Unitree mimic joints');
  }
  return result;
}

// These unmeasured lines only explain the head + hand tracking topology.
// They never become tracked records, IK inputs, or recorded measurements.
export function bodySegments(records) {
  const head = records.find(r => r.name === 'head');
  if (!head) return [];
  const h = head.position, neck = h.clone().add(new THREE.Vector3(0, 0, -.16));
  const hip = neck.clone().add(new THREE.Vector3(0, 0, -.42));
  const left = neck.clone().add(new THREE.Vector3(.18, 0, -.04));
  const right = neck.clone().add(new THREE.Vector3(-.18, 0, -.04));
  const segments = [[h, neck], [neck, hip], [left, right]];
  for (const [side, shoulder] of [['left', left], ['right', right]]) {
    const wrist = records.find(r => r.name === side + '.wrist');
    if (wrist) segments.push([shoulder, wrist.position]);
  }
  return segments;
}

export class BodySchematic extends THREE.LineSegments {
  constructor() {
    super(new THREE.BufferGeometry(), new THREE.LineDashedMaterial({color:0x91a5b2, dashSize:.018, gapSize:.012, transparent:true, opacity:.8}));
    this.frustumCulled = false;
  }
  update(records) {
    const points = bodySegments(records).flat();
    this.geometry.dispose();
    this.geometry = new THREE.BufferGeometry().setFromPoints(points);
    if (points.length) this.computeLineDistances();
    this.visible = points.length > 0;
  }
}
