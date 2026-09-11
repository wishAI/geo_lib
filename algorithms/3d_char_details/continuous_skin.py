"""Weld the supplied body to the original head; relax only the neck junction."""
import bpy,bmesh,json,runpy
from pathlib import Path
from mathutils import Vector
from mathutils.kdtree import KDTree
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'outputs/landau_v10'

def run():
    body=bpy.data.objects['Body_Complete']
    if body.get('continuous_head'):raise RuntimeError('Head is already joined')
    preservation=runpy.run_path(str(ROOT/'rebuild_body.py'))['face_snapshot']
    others={n:h for n,h in preservation().items() if n not in {'Head','Face_Cream'}}
    protected=[]
    # Retain proof of every original head/cream vertex and shape outside the
    # small neck blend band. Other facial objects are untouched by this operation.
    for name in ['Head','Face_Cream']:
        o=bpy.data.objects[name]
        for v in o.data.vertices:
            p=o.matrix_world@v.co
            if p.z>.818:
                protected.append((p,{k.name:o.matrix_world@k.data[v.index].co for k in o.data.shape_keys.key_blocks}))
    bpy.ops.object.select_all(action='DESELECT')
    for o in [body,bpy.data.objects['Head'],bpy.data.objects['Face_Cream']]:o.hide_set(False);o.select_set(True)
    bpy.context.view_layer.objects.active=body;bpy.ops.object.join()
    mesh=body.data
    normals=[n.vector.copy() for n in mesh.corner_normals]
    normal_map={}
    for loop,n in zip(mesh.loops,normals):
        p=mesh.vertices[loop.vertex_index].co
        normal_map.setdefault(tuple(round(x,6) for x in p),[]).append(n)
    bm=bmesh.new();bm.from_mesh(mesh);bm.verts.ensure_lookup_table()
    # Include coincident UV/partition copies of the head, otherwise relaxing
    # only one copy opens the original material seam immediately above the neck.
    rim=[v for v in bm.verts if v.co.z>.773]
    initial=len(bm.verts);bmesh.ops.remove_doubles(bm,verts=rim,dist=1e-6)
    # The joined mesh carries all original shape keys as BMesh shape layers.
    layers=list(bm.verts.layers.shape.values())
    band=[v for v in bm.verts if .745<v.co.z<.816 and abs(v.co.x)<.055]
    original={v:v.co.copy() for v in band}
    for _ in range(40):
        updates={}
        for v in band:
            t=(original[v].z-.745)/(.816-.745);weight=(4*t*(1-t))**2*.55
            neighbors=[e.other_vert(v) for e in v.link_edges]
            updates[v]=v.co.lerp(sum((q.co for q in neighbors),Vector())/len(neighbors),weight)
        for v,p in updates.items():v.co=p
    for v in band:
        delta=v.co-original[v]
        for layer in layers:v[layer]+=delta
    bm.normal_update()
    after=len(bm.verts);neck_boundary=sum(e.is_boundary and all(.752<v.co.z<.81 for v in e.verts) for e in bm.edges)
    bm.to_mesh(mesh);bm.free()
    for p in mesh.polygons:
        if any(.745<mesh.vertices[i].co.z<.816 and abs(mesh.vertices[i].co.x)<.055 for i in p.vertices):p.use_smooth=True
    # Smooth only the junction; preserve authored split normals over the face.
    final=[]
    for loop,n in zip(mesh.loops,mesh.corner_normals):
        p=mesh.vertices[loop.vertex_index].co;key=tuple(round(x,6) for x in p)
        saved=normal_map.get(key)
        final.append(sum(saved,Vector()).normalized() if saved and not (.745<p.z<.816 and abs(p.x)<.055) else (0,0,0))
    mesh.normals_split_custom_set(final);mesh.update()
    # Match the cream neck to the retained cream muzzle instead of leaving a
    # narrow painted strip that makes the welded surface look like two pieces.
    def smooth(a,b,x):
        t=max(0,min(1,(x-a)/(b-a)));return t*t*(3-2*t)
    cream=bpy.data.materials['Cream face'].node_tree.nodes.get('Principled BSDF').inputs['Base Color'].default_value[:]
    color=mesh.color_attributes.get('FurColor')
    for p in mesh.polygons:
        if 'Supplied body' not in mesh.materials[p.material_index].name:continue
        for i in p.loop_indices:
            x,y,z=mesh.vertices[mesh.loops[i].vertex_index].co
            if .745<z<.816:
                w=smooth(.748,.785,z)*(1-smooth(.020,.058,y))*(1-smooth(.035,.058,abs(x)))
                color.data[i].color=tuple(a*(1-w)+b*w for a,b in zip(color.data[i].color,cream))
    tree=KDTree(len(mesh.vertices))
    for v in mesh.vertices:tree.insert(v.co,v.index)
    tree.balance();error=0.0
    for p,keys in protected:
        _,i,d=tree.find(p);error=max(error,d)
        for name,co in keys.items():error=max(error,(mesh.shape_keys.key_blocks[name].data[i].co-co).length)
    assert error<2e-6,('Protected head changed',error)
    assert neck_boundary==0,('Open edges at welded neck',neck_boundary)
    assert preservation()==others, 'A separate facial component changed'
    body['continuous_head']=True;body['body_revision']=5
    body['body_frame_pivot']=[-.004865,.032925,.806]
    result={'joined_objects':['Head','Face_Cream'],'target':'Body_Complete','welded_vertices':initial-after,'neck_boundary_edges':neck_boundary,'protected_head_vertices':len(protected),'protected_head_morph_max_error':error,'blend_band_z':[.745,.816],'method':'Shared welded neck vertices and relaxed transition; original face above neck preserved'}
    report=json.loads((OUT/'body_build.json').read_text());report['revision']=5;report['continuous_skin']=result
    for p in [OUT/'body_build.json']:
        if p.is_symlink():p.unlink()
    (OUT/'body_build.json').write_text(json.dumps(report,indent=2));print(json.dumps(result));return report

if __name__=='__main__':run()
