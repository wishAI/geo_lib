"""Blender diagnostic; run via Blender MCP or Blender --background --python."""
import bpy
import json
from pathlib import Path
from collections import Counter
from mathutils import Vector
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'outputs/landau_v10'
OUT.mkdir(parents=True, exist_ok=True)

def setup_render():
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.cycles.samples = 16
    scene.cycles.use_denoising = True
    scene.render.resolution_x = 800
    scene.render.resolution_y = 1000
    scene.render.resolution_percentage = 100
    scene.world.color = (0.3, 0.3, 0.3)
    scene.view_settings.view_transform = 'Standard'
    for o in list(bpy.data.objects):
        if o.type in {'LIGHT', 'CAMERA'}:
            bpy.data.objects.remove(o, do_unlink=True)
    for name, loc, power, size in [('Key', (1,-2,2),180,2),('Fill',(-1,-1,1),100,2),('Rim',(0,2,2),180,2)]:
        data=bpy.data.lights.new(name,'AREA'); data.energy=power; data.shape='DISK'; data.size=size
        obj=bpy.data.objects.new(name,data); scene.collection.objects.link(obj); obj.location=loc
        obj.rotation_euler=(Vector((0,0,0.6))-obj.location).to_track_quat('-Z','Y').to_euler()
    data=bpy.data.cameras.new('Inspection'); cam=bpy.data.objects.new('Inspection',data)
    scene.collection.objects.link(cam); scene.camera=cam; data.type='ORTHO'; data.ortho_scale=1.28
    scene.render.image_settings.file_format='PNG'
    return cam

def render_view(name, location, target=(0,0,0.58), scale=1.28):
    cam=bpy.context.scene.camera
    cam.location=location; cam.rotation_euler=(Vector(target)-cam.location).to_track_quat('-Z','Y').to_euler()
    cam.data.ortho_scale=scale
    bpy.context.scene.render.filepath=str(OUT / (name+'.png'))
    bpy.ops.render.render(write_still=True)

def inspect():
    m=next(o for o in bpy.context.scene.objects if o.type=='MESH')
    verts=np.array([v.co[:] for v in m.data.vertices]); parent=list(range(len(verts)))
    def find(i):
        while parent[i]!=i: parent[i]=parent[parent[i]]; i=parent[i]
        return i
    for e in m.data.edges:
        x,y=map(find,e.vertices); parent[x]=y
    # USD duplicates vertices at UV/normal seams; weld positions for connectivity analysis.
    positions={}
    for i,v in enumerate(verts):
        key=tuple(np.round(v,5))
        if key in positions: parent[find(i)]=find(positions[key])
        else: positions[key]=i
    comps={}
    for i in range(len(verts)): comps.setdefault(find(i),[]).append(i)
    items=[]
    for ids in sorted(comps.values(),key=len,reverse=True):
        p=verts[ids]; items.append({'count':len(ids),'min':p.min(0).tolist(),'max':p.max(0).tolist()})
    result={'vertices':len(verts),'triangles':len(m.data.polygons),'components':items,'bones':len(next(o for o in bpy.context.scene.objects if o.type=='ARMATURE').data.bones)}
    (OUT/'source_diagnostic.json').write_text(json.dumps(result,indent=2))
    np.savez_compressed(OUT/'source_geometry.npz',vertices=verts,faces=np.array([p.vertices[:] for p in m.data.polygons]))
    print(json.dumps(result))

if __name__=='__main__':
    inspect()
    setup_render()
    render_view('source_front',(0,-3,0.58))
    render_view('source_side',(3,0,0.58))
