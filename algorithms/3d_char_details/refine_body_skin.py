"""Smooth spatial skin gradients without changing the supplied neutral geometry."""
import bpy,bmesh,runpy,json,math
from pathlib import Path
from mathutils import Vector
from mathutils.kdtree import KDTree
ROOT=Path(__file__).resolve().parent

def run():
 o=bpy.data.objects['Body_Complete'];rig=bpy.data.objects['Landau_Rig'];api=runpy.run_path(str(ROOT/'rebuild_body.py'))
 bm=bmesh.new();bm.from_mesh(o.data);bm.verts.ensure_lookup_table();locked={v.index for v in bm.verts if v.is_boundary};bm.free()
 for v in o.data.vertices:
  if abs(v.co.x)>.14 and v.co.z>.30:
   suffix='l' if v.co.x>0 else 'r';b=rig.data.bones['hand_'+suffix]
   if (v.co-b.head_local).dot((b.tail_local-b.head_local).normalized())>-.003:locked.add(v.index)
 kd=KDTree(len(o.data.vertices))
 for v in o.data.vertices:kd.insert(v.co,v.index)
 kd.balance();neighbors=[]
 for v in o.data.vertices:
  candidates=[]
  for p,i,d in kd.find_n(v.co,32):
   if d>.022 or (v.co.z<.50 and abs(v.co.x)>.012 and v.co.x*p.x<0):continue
   candidates.append((i,math.exp(-2*(d/.018)**2)))
  total=sum(w for i,w in candidates);neighbors.append([(i,w/total) for i,w in candidates])
 ws=[{o.vertex_groups[g.group].name:g.weight for g in v.groups} for v in o.data.vertices]
 for step in range(5):
  nxt=[]
  for i,w in enumerate(ws):
   if i in locked:nxt.append(w);continue
   d={n:v*.35 for n,v in w.items()}
   for j,factor in neighbors[i]:
    for n,v in ws[j].items():d[n]=d.get(n,0)+v*.65*factor
   d=dict(sorted(d.items(),key=lambda p:-p[1])[:4]);total=sum(d.values());nxt.append({n:v/total for n,v in d.items()})
  ws=nxt
 api['assign'](o,ws)
 result={'method':'Five spatial averaging passes, 32 nearest surface samples, maximum radius 0.022','locked_neck_and_hand_vertices':len(locked),'neutral_geometry_changed':False}
 path=ROOT/'outputs/landau_v10/body_build.json'
 if path.exists():
  report=json.loads(path.read_text());report['skin_refinement']=result;path.write_text(json.dumps(report,indent=2))
 print(result);return result
if __name__=='__main__':run()
