import assert from 'node:assert/strict';
import {test} from 'node:test';
import {groupInputs,comparisonRows,preferredSeed,recordedTime} from '../gui/workspace.js';
const row=(engine,seed,extra={})=>({id:`${engine}_${seed}`,report:'run/index.json',problem:'run/case/problem.json',case_id:'case',engine,seed,status:'valid',modifiedAt:'2026-10-10T01:02:03Z',...extra});
test('one input retains every algorithm and seed, including unsuccessful results',()=>{
  const groups=groupInputs([row('nfp',7),row('nfp',42),row('opennest',42),row('spyrrow',42,{status:'unsupported'})]);
  assert.equal(groups.length,1);assert.equal(groups[0].trials.length,4);
  assert.equal(preferredSeed(groups[0]),'42');
  assert.deepEqual(comparisonRows(groups[0],'42').map(r=>r.engine),['nfp','opennest','spyrrow']);
  assert.equal(comparisonRows(groups[0],7).length,1);
});
test('same case name never merges different snapshots or runs',()=>{
  const groups=groupInputs([row('nfp',42),row('opennest',42,{problem:'run/variant/problem.json'}),row('spyrrow',42,{report:'other/index.json'})]);
  assert.equal(groups.length,3);
});
test('stable input identity survives result ordering and additional algorithms',()=>{
  const first=groupInputs([row('nfp',42)]);
  const next=groupInputs([row('spyrrow',42),row('nfp',7),row('nfp',42)]);
  assert.equal(first[0].id,next[0].id);
});
test('seed comparison never substitutes a result from another seed',()=>{
  const group=groupInputs([row('nfp',7),row('opennest',42)])[0];
  assert.deepEqual(comparisonRows(group,7).map(r=>r.engine),['nfp']);
  assert.deepEqual(comparisonRows(group,123),[]);
});
test('timestamps distinguish recorded start from artifact modification and missing time',()=>{
  assert.match(recordedTime(row('nfp',7,{timeKind:'started'})),/^Started .*\d{2}:\d{2}:\d{2}$/);
  assert.match(recordedTime(row('nfp',7)),/^Updated /);
  assert.equal(recordedTime({}),'Time unavailable');
});
