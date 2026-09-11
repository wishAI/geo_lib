"""Loft a short, smoothly sampled collar to the unchanged original head rim."""
import bpy, bmesh, runpy
from pathlib import Path
from mathutils import Vector
ROOT=Path(__file__).resolve().parent

def run():
    helpers=runpy.run_path(str(ROOT/'integrate_fbx_body.py'));api=helpers['api']
    o=bpy.data.objects['Body_Complete'];before=api['face_snapshot']()
    bm=bmesh.new();bm.from_mesh(o.data);bm.verts.ensure_lookup_table()
    rim=helpers['loops'](bm)[0];boundary=set(rim)
    collar={f for v in rim for f in v.link_faces}
    lower={v for f in collar for v in f.verts if v not in boundary}
    assert max(v.co.z for v in lower)-min(v.co.z for v in lower)<.0001
    oldweights=[{o.vertex_groups[g.group].name:g.weight for g in v.groups} for v in o.data.vertices]
    cage=api['Cage']();remap={}
    for v in bm.verts:
        if v not in boundary:remap[v.index]=cage.point(v.co,w=oldweights[v.index])
    for f in bm.faces:
        if f not in collar:cage.face([remap[v.index] for v in f.verts])
    # Follow the lower ring's actual mesh edges, then match perimeter distance.
    perimeter={e for f in collar for e in f.edges if all(v in lower for v in e.verts)}
    start=min(lower,key=lambda v:v.co.x);seq=[start];prev=None
    while True:
        choices=[e.other_vert(seq[-1]) for e in seq[-1].link_edges if e in perimeter and e.other_vert(seq[-1])!=prev]
        nxt=choices[0]
        if nxt==start:break
        prev=seq[-1];seq.append(nxt)
    assert len(seq)==len(lower), (len(seq),len(lower))
    a=[remap[v.index] for v in seq]
    b=[cage.point(v.co,w=oldweights[v.index]) for v in rim];b=cage.align(a,b)
    top=[Vector(cage.v[i]) for i in b]
    # Uniform angular progression at the lower ring, independent of irregular
    # tessellation density on either source boundary.
    lengths=[0.0]
    for i in range(len(a)):lengths.append(lengths[-1]+(Vector(cage.v[a[(i+1)%len(a)]])-Vector(cage.v[a[i]])).length)
    def sample(t):
        d=t*lengths[-1];j=next((j for j in range(len(a)) if lengths[j+1]>=d),len(a)-1)
        f=(d-lengths[j])/(lengths[j+1]-lengths[j]);return Vector(cage.v[a[j]]).lerp(Vector(cage.v[a[(j+1)%len(a)]]),f),a[j]
    smooth=[p.copy() for p in top]
    for _ in range(8):smooth=[smooth[i]*.5+(smooth[i-1]+smooth[(i+1)%len(smooth)])*.25 for i in range(len(smooth))]
    rings=[]
    for step in range(1,6):
        t=step/6;ring=[]
        for i,p in enumerate(top):
            low,li=sample(i/len(top));target=smooth[i].lerp(p,t*t)
            pos=low.lerp(target,t);wa=cage.ws[li];wb=cage.ws[b[i]]
            w={n:wa.get(n,0)*(1-t)+wb.get(n,0)*t for n in set(wa)|set(wb)}
            ring.append(cage.point(pos,w=w))
        rings.append(ring)
    cage.bridge(a,rings[0])
    for first,second in zip(rings,rings[1:]+[b]):cage.bridge(first,second)
    bm.free();mesh=bpy.data.meshes.new('Supplied body with lofted original-head collar');mesh.from_pydata(cage.v,[],cage.f);mesh.update();o.data=mesh
    api['bind'](o,bpy.data.objects['Landau_Rig'],cage.ws)
    bm=bmesh.new();bm.from_mesh(mesh);bmesh.ops.recalc_face_normals(bm,faces=list(bm.faces))
    if bm.calc_volume(signed=True)<0:bmesh.ops.reverse_faces(bm,faces=list(bm.faces))
    bm.to_mesh(mesh);bm.free()
    for p in mesh.polygons:p.use_smooth=True
    helpers['paint_body'](o)
    assert api['face_snapshot']()==before
    print('Lofted collar: preserved head and source body below neck, added five transition loops')

if __name__=='__main__':run()
