"""Shared preservation, original-hand recovery and mesh-join helpers.
Run in the prepared facial master through Blender MCP, or call rebuild() in builder.
Hidden anatomy is authored from landmarks, never offset/copied garment geometry.
"""
from pathlib import Path
import bpy, bmesh, math, json, hashlib
import numpy as np
import runpy
from mathutils import Vector

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'outputs/landau_v10'
GARMENTS=('Vest','Sleeve_L','Sleeve_R','Cuff_L','Cuff_R','Trousers','Boot_L','Boot_R')

def face_snapshot():
    result={}
    for o in bpy.context.scene.objects:
        if o.type!='MESH' or o.name in GARMENTS or o.name.startswith(('Body_','Hand_','Tail')):continue
        h=hashlib.sha256()
        for seq in [[v.co[:] for v in o.data.vertices],[list(p.vertices) for p in o.data.polygons],
                    [[(g.group,g.weight) for g in v.groups] for v in o.data.vertices],
                    [list(v.vector) for v in o.data.corner_normals],
                    [[list(v.uv) for v in l.data] for l in o.data.uv_layers],
                    [[k.name,[v.co[:] for v in k.data]] for k in o.data.shape_keys.key_blocks] if o.data.shape_keys else [],
                    [(m.name,tuple(m.diffuse_color),[(n.name,[(i.name,str(i.default_value)) for i in n.inputs if hasattr(i,'default_value')]) for n in m.node_tree.nodes]) for m in o.data.materials]]:
            h.update(json.dumps(seq).encode())
        result[o.name]=h.hexdigest()
    return result

def smooth(a,b,t):
    t=max(0,min(1,(t-a)/(b-a)));return t*t*(3-2*t)

def palette(name,hexcolor,rough):
    m=bpy.data.materials.get(name) or bpy.data.materials.new(name);m.use_nodes=True
    rgb=[int(hexcolor[i:i+2],16)/255 for i in (1,3,5)]
    c=tuple(v/12.92 if v<=.04045 else ((v+.055)/1.055)**2.4 for v in rgb)
    m.diffuse_color=(*c,1);p=m.node_tree.nodes.get('Principled BSDF')
    p.inputs['Base Color'].default_value=(*c,1);p.inputs['Roughness'].default_value=rough
    return m

def weights(p,region=None):
    x,y,z=p;side='l' if x>0 else 'r';ax=abs(x)
    if region=='arm' or (region is None and ax>.082 and z>.36):
        shoulder=Vector((.063946,.00994,.523824));elbow=Vector((.119799,.015724,.452799));wrist=Vector((.175651,-.002624,.381775))
        q=Vector((ax,y,z));t=(q-elbow).dot((wrist-elbow).normalized())
        e=smooth(-.022,.023,t);h=smooth(.069,.099,t)
        return {f'arm_stretch_{side}':1-e,f'forearm_stretch_{side}':e*(1-h),f'hand_{side}':e*h}
    if region=='leg' or (region is None and z<.32):
        knee=smooth(.148,.199,z);hip=smooth(.287,.346,z);foot=1-smooth(.035,.071,z)
        return {'root_x':hip,f'thigh_stretch_{side}':(1-hip)*knee,f'leg_stretch_{side}':(1-hip)*(1-knee)*(1-foot),f'foot_{side}':(1-hip)*(1-knee)*foot}
    levels=[(.35,'root_x'),(.40,'spine_01_x'),(.48,'spine_02_x'),(.533,'spine_03_x'),(.571,'neck_x'),(.602,'head_x')]
    if z<=levels[0][0]:return {'root_x':1}
    for (a,n),(b,m) in zip(levels,levels[1:]):
        if z<=b:
            t=smooth(a,b,z);return {n:1-t,m:t}
    return {'head_x':1}

def assign(o,ws):
    o.vertex_groups.clear()
    for i,w in enumerate(ws):
        w=sorted(((n,v) for n,v in w.items() if v>1e-6),key=lambda p:-p[1])[:4];total=sum(v for n,v in w)
        for n,v in w:
            g=o.vertex_groups.get(n) or o.vertex_groups.new(name=n);g.add([i],v/total,'REPLACE')

def bind(o,rig,ws):
    mat=o.matrix_world.copy();o.parent=rig;o.matrix_world=mat
    for m in list(o.modifiers):
        if m.type=='ARMATURE':o.modifiers.remove(m)
    m=o.modifiers.new('Shared skeleton · linear skin','ARMATURE');m.object=rig;m.use_deform_preserve_volume=False
    assign(o,ws)

def boundary_loop(mesh):
    bm=bmesh.new();bm.from_mesh(mesh);bm.verts.ensure_lookup_table()
    es=[e for e in bm.edges if e.is_boundary];adj={}
    for e in es:
        a,b=[v.index for v in e.verts];adj.setdefault(a,[]).append(b);adj.setdefault(b,[]).append(a)
    assert all(len(v)==2 for v in adj.values()),'non-loop hand boundary'
    loops=[]
    while adj:
        start=next(iter(adj));loop=[start];prev=None;cur=start
        while True:
            nxt=next(n for n in adj[cur] if n!=prev);prev,cur=cur,nxt
            if cur==start:break
            loop.append(cur)
        for i in loop:del adj[i]
        loops.append(loop)
    bm.free();return max(loops,key=len)

def repair_hands():
    # Recover palms/digits from the immutable sculpt, then cut a true wrist plane.
    # The previous semantic hand boundary wandered into the thumb and cuff.
    for side,sign in [('L',1),('R',-1)]:
        vs=[];fs=[];ws=[];lookup={}
        for name in ['Hand_'+side,'Cuff_'+side,'Sleeve_'+side,'Vest']:
            o=bpy.data.objects[name]
            for p in o.data.polygons:
                center=o.matrix_world@p.center
                if sign*center.x<.145 or center.z>.42:continue
                f=[]
                for idx in p.vertices:
                    v=o.data.vertices[idx];co=o.matrix_world@v.co;key=tuple(round(t,6) for t in co)
                    if key not in lookup:
                        lookup[key]=len(vs);vs.append(tuple(co));ws.append({o.vertex_groups[g.group].name:g.weight for g in v.groups})
                    f.append(lookup[key])
                fs.append(f)
        old=bpy.data.objects['Hand_'+side];bpy.data.objects.remove(old,do_unlink=True)
        mesh=bpy.data.meshes.new('Hand_'+side);mesh.from_pydata(vs,[],fs);mesh.update()
        o=bpy.data.objects.new('Hand_'+side,mesh);bpy.context.scene.collection.objects.link(o);assign(o,ws)
        bm=bmesh.new();bm.from_mesh(mesh)
        bmesh.ops.bisect_plane(bm,geom=list(bm.verts)+list(bm.edges)+list(bm.faces),dist=1e-7,plane_co=Vector((sign*.187,-.003,.366)),plane_no=Vector((sign*.62,-.10,-.78)),clear_inner=True,clear_outer=False)
        bmesh.ops.dissolve_degenerate(bm,edges=list(bm.edges),dist=1e-7)
        bmesh.ops.delete(bm,geom=[v for v in bm.verts if not v.link_faces],context='VERTS')
        bmesh.ops.recalc_face_normals(bm,faces=list(bm.faces));bm.to_mesh(mesh);bm.free()
        o['part_type']='body';mesh.materials.append(bpy.data.materials['Cream face'])
        for p in mesh.polygons:p.use_smooth=True

class Cage:
    def __init__(self):self.v=[];self.f=[];self.ws=[];self.mat=[]
    def point(self,p,region=None,w=None):
        self.v.append(tuple(p));self.ws.append(weights(p,region) if w is None else w);return len(self.v)-1
    def ring(self,points,region=None):return [self.point(p,region) for p in points]
    def face(self,ids,mat=0):self.f.append(ids);self.mat.append(mat)
    def bridge(self,a,b,mat=0):
        # Perimeter zipper allows exact irregular source hand boundary without retopologizing fingers.
        i=j=0
        while i<len(a) or j<len(b):
            ta=(i+1)/len(a) if i<len(a) else 2;tb=(j+1)/len(b) if j<len(b) else 2
            if abs(ta-tb)<1e-8:self.face([a[i%len(a)],a[(i+1)%len(a)],b[(j+1)%len(b)],b[j%len(b)]],mat);i+=1;j+=1
            elif ta<tb:self.face([a[i%len(a)],a[(i+1)%len(a)],b[j%len(b)]],mat);i+=1
            else:self.face([a[i%len(a)],b[(j+1)%len(b)],b[j%len(b)]],mat);j+=1
    def align(self,a,b):
        # Find winding and cyclic start with the smallest total bridge distance.
        choices=[]
        for seq in [b,list(reversed(b))]:
            for k in range(len(seq)):
                r=seq[k:]+seq[:k]
                d=sum((Vector(self.v[v])-Vector(self.v[r[round(i*len(r)/len(a))%len(r)]])).length_squared for i,v in enumerate(a))
                choices.append((d,r))
        return min(choices,key=lambda p:p[0])[1]

