import test from 'node:test';
import assert from 'node:assert/strict';
import vm from 'node:vm';
import {readFileSync} from 'node:fs';

const context={window:{}};
vm.runInNewContext(readFileSync(new URL('../static/evolution_tree.js',import.meta.url),'utf8'),context);
const layout=context.window.GeoEvolutionTree.layout;
const gates=[{id:'gate',label:'Gate',status:'in_progress'}];
const chain=Array.from({length:80},(_,i)=>({id:'n'+i,label:'Run '+i,milestoneId:'gate',parentIds:i?['n'+(i-1)]:[]}));

test('long sections fold their center, preserve ancestry and fit the viewport height',()=>{
  const graph=layout(chain,gates,340,900);
  assert.ok(graph.bands[1].width<=900*2/3);
  assert.equal(graph.visible.filter(n=>n.kind==='fold').length,1);
  assert.equal(graph.representative('n0'),'n0');
  assert.equal(graph.representative('n79'),'n79');
  assert.notEqual(graph.representative('n40'),'n40');
  assert.equal(graph.folds[0].members.length+graph.visible.length-1,80);
  assert.ok(graph.connections.some(e=>e.parent==='n0'));
  assert.ok(graph.connections.some(e=>e.child==='n79'));
  for(const n of graph.visible){const p=graph.point(n.id);assert.ok(p.y>=0&&p.y+85<=340);}
  const all=layout(chain,gates,340,900,new Set(['gate']));
  assert.equal(all.visible.length,80);
  assert.equal(all.folds.length,0);
});

test('branch lanes use available height without overlapping node boxes',()=>{
  const nodes=[chain[0],...Array.from({length:40},(_,i)=>({id:'b'+i,milestoneId:'gate',parentIds:['n0']}))];
  const graph=layout(nodes,gates,640,1600,new Set(['gate']));
  const points=nodes.map(n=>graph.point(n.id));
  assert.equal(new Set(points.map(p=>p.y)).size,6);
  for(let i=0;i<points.length;i++)for(let j=i+1;j<points.length;j++){
    assert.ok(Math.abs(points[i].x-points[j].x)>=104||Math.abs(points[i].y-points[j].y)>=85);
  }
  const reordered=layout([...nodes].reverse(),gates,640,1600,new Set(['gate']));
  for(const n of nodes)assert.deepEqual(graph.point(n.id),reordered.point(n.id));
});

test('certificates remain visible at the right edge in folded and expanded sections',()=>{
  const nodes=chain.map((n,i)=>({...n,kind:i===30||i===45?'milestone':'checkpoint',status:'completed'}));
  // Include an unfinished milestone: only genuine certificates are pinned.
  nodes[50].kind='milestone';nodes[50].status='running';
  for(const expanded of [new Set(),new Set(['gate'])]){
    const graph=layout(nodes,gates,340,900,expanded);
    for(const id of ['n30','n45']){
      assert.equal(graph.representative(id),id);
      assert.ok(graph.visible.some(n=>n.id===id));
      assert.equal(graph.point(id).x+42,graph.bands[1].x+graph.bands[1].width-56);
      for(const n of graph.visible.filter(n=>!['n30','n45'].includes(n.id)))assert.ok(graph.point(n.id).x<graph.point(id).x);
    }
    assert.notEqual(graph.point('n30').y,graph.point('n45').y);
    assert.ok(graph.connections.some(e=>e.parent==='n30'&&e.child===graph.representative('n31')));
    if(!expanded.size){
      assert.ok(graph.bands[1].width<=900*2/3);
      assert.ok(graph.folds.every(f=>f.members.every(n=>!['n30','n45'].includes(n.id))));
    }
  }
});

test('narrow sections retain every certificate without stacking node boxes',()=>{
  const nodes=chain.slice(0,8).map((n,i)=>({...n,kind:i<5?'milestone':'checkpoint',status:'completed'}));
  const graph=layout(nodes,gates,220,300);
  const certificates=nodes.filter(n=>n.kind==='milestone');
  for(const n of certificates){
    assert.equal(graph.representative(n.id),n.id);
    assert.ok(graph.point(n.id).x>graph.point('n7').x);
  }
  for(let i=0;i<graph.visible.length;i++)for(let j=i+1;j<graph.visible.length;j++){
    const a=graph.point(graph.visible[i].id),b=graph.point(graph.visible[j].id);
    assert.ok(Math.abs(a.x-b.x)>=104||Math.abs(a.y-b.y)>=85);
  }
});

test('every connection leaves the right rim and enters the left rim horizontally',()=>{
  const nodes=chain.slice(0,20).map((n,i)=>({...n,parentIds:[...n.parentIds],kind:i===8?'milestone':'checkpoint',status:'completed'}));
  nodes[19].parentIds.push('n0');
  for(const expanded of [new Set(),new Set(['gate'])]){
    const graph=layout(nodes,gates,340,900,expanded);
    for(const edge of graph.connections){
      const a=graph.point(edge.parent),b=graph.point(edge.child);
      const values=graph.edgePath(edge).match(/-?\d+(?:\.\d+)?/g).map(Number);
      assert.deepEqual(values.slice(0,2),[a.x+60,a.y+26]);
      assert.deepEqual(values.slice(-2),[b.x+24,b.y+26]);
      assert.ok(values[2]>values[0]);
      assert.equal(values[3],values[1]);
      assert.ok(values.at(-4)<values.at(-2));
      assert.equal(values.at(-3),values.at(-1));
    }
    const backward=graph.connections.find(e=>e.parent==='n8');
    assert.ok(graph.edgePath(backward).includes(' Q '),'Backward links keep rounded clearance bends');
  }
});

test('clear long links stay direct while links crossing nodes take rounded detours',()=>{
  const stages=['first','empty','last'].map(id=>({id,label:id}));
  const graph=layout([
    {id:'source',milestoneId:'first'},
    {id:'target',milestoneId:'last',parentIds:['source']},
  ],stages,340,900);
  assert.equal(graph.connections[0].track,undefined);
  assert.match(graph.edgePath(graph.connections[0]),/^M [\d.]+ [\d.]+ L [\d.]+ [\d.]+$/);
  const nodes=chain.slice(0,6).map(n=>({...n,parentIds:[...n.parentIds]}));
  nodes[5].parentIds.push('n0');
  const blocked=layout(nodes,gates,340,900,new Set(['gate']));
  const edge=blocked.connections.find(e=>e.parent==='n0'&&e.child==='n5');
  assert.notEqual(edge.track,undefined);
  assert.ok(blocked.edgePath(edge).includes(' Q '));
});

test('index requests coalesce, cache, report progress and retry failures',async()=>{
  let calls=0, fail=false;
  const payload={revision:'r1',nodes:[]};
  const context={window:{},AbortController,TextDecoder,Uint8Array,setTimeout,clearTimeout,
    fetch:async url=>{
      calls++;
      await new Promise(resolve=>setTimeout(resolve,5));
      if(fail)return new Response('',{status:503});
      const value=url.includes('&node=')?{revision:'r1',node:{id:'a',metrics:{value:42}}}:payload;
      const text=JSON.stringify(value);
      return new Response(text,{headers:{'X-Uncompressed-Content-Length':String(text.length)}});
    }};
  vm.runInNewContext(readFileSync(new URL('../static/lineage_data.js',import.meta.url),'utf8'),context);
  const data=context.window.GeoLineage,progress=[];
  await Promise.all([data.index('walk',{onProgress:p=>progress.push(p)}),data.index('walk')]);
  assert.equal(calls,1);
  assert.equal(progress.at(-1).loaded,progress.at(-1).total);
  await data.index('walk');assert.equal(calls,1);
  await data.node('walk','r1','a');await data.node('walk','r1','a');assert.equal(calls,2);
  fail=true;await assert.rejects(data.index('walk',{force:true}));
  assert.equal(data.peek('walk').revision,'r1');
  fail=false;await data.index('walk',{force:true});assert.equal(calls,4);
  await assert.rejects(data.node('walk','old-revision','a'),/Lineage changed/);
});
