"""Match the new collar's terminal shading normals to the preserved head rim."""
import bpy,bmesh
from mathutils import Vector

def run():
 body=bpy.data.objects['Body_Complete'];reference={}
 for name in ['Head','Face_Cream']:
  o=bpy.data.objects[name];mat=o.matrix_world.to_3x3()
  for loop,normal in zip(o.data.loops,o.data.corner_normals):
   p=o.matrix_world@o.data.vertices[loop.vertex_index].co
   if p.z>.82:continue
   key=tuple(round(x,6) for x in p);reference.setdefault(key,[]).append(mat@normal.vector)
 bm=bmesh.new();bm.from_mesh(body.data);bm.verts.ensure_lookup_table();boundary={v.index for v in bm.verts if v.is_boundary};bm.free()
 normals=[];matched=0
 for loop,n in zip(body.data.loops,body.data.corner_normals):
  p=body.data.vertices[loop.vertex_index].co;key=tuple(round(x,6) for x in p)
  if loop.vertex_index in boundary and key in reference:
   normals.append(sum(reference[key],Vector()).normalized());matched+=1
  else:normals.append(n.vector[:])
 body.data.normals_split_custom_set(normals);body.data.update();print('Matched neck corner normals',matched)
if __name__=='__main__':run()
