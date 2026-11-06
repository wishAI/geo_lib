import test from 'node:test';
import assert from 'node:assert/strict';
import vm from 'node:vm';
import {readFileSync} from 'node:fs';
const context={window:{}};
vm.runInNewContext(readFileSync(new URL('../static/motion_urdf_viewer.js',import.meta.url),'utf8'),context);
const method=context.window.MotionUrdfViewer.prototype.groupJointPivots;
test('coincident world pivots group multi-axis joints without merging nearby joints',()=>{
  const points={hip_roll:[1,2,3],hip_pitch:[1,2,3.000001],hip_yaw:[1,2,3],knee:[1,2,2.9],unknown:null};
  const viewer={jointFrame:name=>points[name]&&{pivot:points[name]}};
  const groups=method.call(viewer,Object.keys(points).map(name=>({name})));
  assert.deepEqual(Array.from(groups,g=>Array.from(g.joints,j=>j.name)),[['hip_roll','hip_pitch','hip_yaw'],['knee'],['unknown']]);
});
