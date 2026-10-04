"""Build a separate, source-hashed UV atlas for editing; never change the GLB.
Run with Blender --background --python algorithms/3d_char_details/build_edit_atlas.py.
Smart UV Project splits connected charts at angle boundaries and packs islands.
Reference: https://docs.blender.org/manual/en/latest/modeling/meshes/editing/uv.html
"""
import base64
import hashlib
import json
import math
import struct
from pathlib import Path
import bpy

ROOT=Path(__file__).resolve().parent
source=ROOT/'outputs/landau_v10/landau_character.glb'
data=source.read_bytes();length=struct.unpack_from('<I',data,12)[0]
doc=json.loads(data[20:20+length]);binary=data[28+length:]
formats={5126:('f',4),5125:('I',4),5123:('H',2),5121:('B',1)}
widths={'SCALAR':1,'VEC2':2,'VEC3':3,'VEC4':4}
def rows(accessor):
 a=doc['accessors'][accessor];view=doc['bufferViews'][a['bufferView']];fmt,size=formats[a['componentType']];width=widths[a['type']]
 start=view.get('byteOffset',0)+a.get('byteOffset',0);stride=view.get('byteStride',width*size)
 return [struct.unpack_from('<'+fmt*width,binary,start+i*stride) for i in range(a['count'])]
result={'version':1,'sourceHash':hashlib.sha256(data).hexdigest(),'method':'Blender Smart UV Project · 66° · separate editing atlas','primitives':{}}
for mi,mesh in enumerate(doc['meshes']):
 vertices=[];faces=[];lookup={};spans=[]
 for pi,p in enumerate(mesh['primitives']):
  pos=rows(p['attributes']['POSITION']);indices=[r[0] for r in rows(p['indices'])] if 'indices' in p else list(range(len(pos)))
  remap=[]
  for point in pos:
   key=tuple(point)
   if key not in lookup:lookup[key]=len(vertices);vertices.append(point)
   remap.append(lookup[key])
  start=len(faces);faces.extend(tuple(remap[k] for k in indices[i:i+3]) for i in range(0,len(indices),3));spans.append((pi,start,len(faces),len(pos)))
 m=bpy.data.meshes.new('edit-atlas');m.from_pydata(vertices,[],faces);m.update();o=bpy.data.objects.new('edit-atlas',m);bpy.context.collection.objects.link(o)
 bpy.ops.object.select_all(action='DESELECT');o.select_set(True);bpy.context.view_layer.objects.active=o
 bpy.ops.object.mode_set(mode='EDIT');bpy.ops.mesh.select_all(action='SELECT')
 bpy.ops.uv.smart_project(angle_limit=math.radians(66),island_margin=.012,area_weight=.2,scale_to_bounds=True)
 bpy.ops.object.mode_set(mode='OBJECT');assert len(m.polygons)==len(faces)
 uv=m.uv_layers.active.data
 # Reserve a thin strip for tiny numerical-degeneracy fallback islands. Never
 # discard a source triangle or silently make it impossible to paint.
 mapped=[];tiny=[]
 for polygon in m.polygons:
  loops=list(polygon.loop_indices);assert sorted(m.loops[i].vertex_index for i in loops)==sorted(faces[polygon.index]), 'Unwrap reordered source faces'
  by_vertex={m.loops[i].vertex_index:tuple(uv[i].uv) for i in loops}
  points=[(by_vertex[i][0]*.97,by_vertex[i][1]) for i in faces[polygon.index]]
  area=abs((points[1][0]-points[0][0])*(points[2][1]-points[0][1])-(points[1][1]-points[0][1])*(points[2][0]-points[0][0]))
  if area<1e-14:tiny.append(polygon.index)
  mapped.append(points)
 for j,index in enumerate(tiny):
  y=(j+.2)/max(1,len(tiny));h=.5/max(1,len(tiny));mapped[index]=[(.98,y),(.995,y),(.98,y+h)]
 for pi,start,end,count in spans:
  values=[];collapsed=0
  for polygon in m.polygons[start:end]:
   assert len(polygon.loop_indices)==3
   points=mapped[polygon.index]
   if abs((points[1][0]-points[0][0])*(points[2][1]-points[0][1])-(points[1][1]-points[0][1])*(points[2][0]-points[0][0]))<1e-14:collapsed+=1
   for point in points:values.extend(max(0,min(1,v)) for v in point)
  result['primitives'][f'{mi}:{pi}']={'sheet':str(mi),'label':mesh.get('name',str(mi)),'vertices':count,'corners':len(values)//2,'uv32':base64.b64encode(struct.pack('<'+'f'*len(values),*values)).decode(),'degenerate':collapsed,'tinyFallbacks':sum(start<=i<end for i in tiny)}
 bpy.data.objects.remove(o,do_unlink=True);bpy.data.meshes.remove(m)
 print('Unwrapped',mesh.get('name',mi),len(faces),'triangles',flush=True)
result['atlasHash']=hashlib.sha256(json.dumps(result['primitives'],sort_keys=True,separators=(',',':')).encode()).hexdigest()
out=ROOT/'outputs/landau_v10/editor_uv_atlas.json'
# Replace our derivative only; no source mesh, material or UV data is written.
if out.is_symlink():out.unlink()
out.write_text(json.dumps(result,separators=(',',':')))
assert hashlib.sha256(source.read_bytes()).hexdigest()==result['sourceHash']
print('Atlas saved:',out,out.stat().st_size)
