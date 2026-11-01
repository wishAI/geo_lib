import * as THREE from 'three';

export function originMatrix({xyz, rpy}) {
  return new THREE.Matrix4().compose(new THREE.Vector3(...xyz),
    new THREE.Quaternion().setFromEuler(new THREE.Euler(...rpy, 'ZYX')), new THREE.Vector3(1,1,1));
}

export function worldTransforms(joints, pose = {}, root = 'base_link') {
  const worlds = {[root]: new THREE.Matrix4()};
  const remaining = [...joints];
  while (remaining.length) {
    let progressed = false;
    for (let i = remaining.length - 1; i >= 0; i--) {
      const j = remaining[i];
      if (!worlds[j.parent]) continue;
      const value = Number(pose[j.child] ?? 0);
      if (!Number.isFinite(value)) throw new Error(`Invalid joint angle: ${j.child}`);
      const angle = j.type === 'fixed' ? 0 : THREE.MathUtils.clamp(value, j.lower, j.upper);
      const motion = new THREE.Matrix4();
      if (j.type === 'prismatic') motion.makeTranslation(...j.axis.map(v => v * angle));
      else motion.makeRotationAxis(new THREE.Vector3(...j.axis).normalize(), angle);
      worlds[j.child] = worlds[j.parent].clone().multiply(originMatrix(j)).multiply(motion);
      remaining.splice(i, 1); progressed = true;
    }
    if (!progressed) throw new Error('URDF contains an unresolved link hierarchy');
  }
  return worlds;
}

export const XR_HAND_NAMES = ['wrist', 'thumb-metacarpal', 'thumb-phalanx-proximal', 'thumb-phalanx-distal', 'thumb-tip',
  ...['index', 'middle', 'ring', 'pinky'].flatMap(f => [`${f}-finger-metacarpal`, `${f}-finger-phalanx-proximal`, `${f}-finger-phalanx-intermediate`, `${f}-finger-phalanx-distal`, `${f}-finger-tip`])];
export function rows(matrix) {
  const e = matrix.elements;
  return [0,1,2,3].map(r => [0,1,2,3].map(c => e[c*4+r]));
}
export function fromRows(value) { return new THREE.Matrix4().set(...value.flat()); }
// WebXR is Y-up/-Z forward; the recorded AVP pipeline is Z-up/+Y forward.
const XR_TO_NATIVE = new THREE.Matrix4().makeRotationX(Math.PI / 2);
export function nativeMatrix(xrMatrix) {
  return XR_TO_NATIVE.clone().multiply(new THREE.Matrix4().fromArray(xrMatrix)).multiply(XR_TO_NATIVE.clone().invert());
}

// Cache native reads for one active XR frame. A temporarily unavailable joint
// must not escape the callback and prevent Three.js from requesting another frame.
export function sampleXRFrame(frame, referenceSpace, onError = () => {}) {
  const joints = new Map();
  const read = (kind, fn) => {
    if (!frame || !referenceSpace) return null;
    try {
      const pose = fn();
      if (!pose) return null;
      const m = pose.transform?.matrix;
      if (!m || m.length !== 16 || !Array.from(m).every(Number.isFinite)) throw new Error('Invalid XR transform');
      return pose;
    } catch (error) { onError(kind, error); return null; }
  };
  const viewer = read('viewer', () => frame.getViewerPose(referenceSpace));
  return {
    getViewerPose: () => viewer,
    getJointPose(joint) {
      if (!joints.has(joint)) joints.set(joint, read('joint', () => frame.getJointPose(joint, referenceSpace)));
      return joints.get(joint);
    },
    getPose: (space) => read('pointer', () => frame.getPose(space, referenceSpace)),
  };
}
