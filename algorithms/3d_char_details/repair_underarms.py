"""Open the fused axillary web along measured surface grooves, preserving the rig.

Run in background Blender on the current master. --publish writes a new local
master/GLB only after preservation/topology checks; otherwise saves a candidate.
The head, original hands, clothing, rest joints and Running action are retained.
"""
from pathlib import Path
import hashlib
import json
import math
import runpy
import shutil
import sys
import bpy
import bmesh
import numpy as np
from mathutils import Matrix, Vector
from mathutils.bvhtree import BVHTree

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'outputs/landau_v10'


def smooth(a, b, x):
    t = max(0, min(1, (x-a)/(b-a)))
    return t*t*(3-2*t)


def topology(mesh):
    bm = bmesh.new(); bm.from_mesh(mesh)
    result = dict(vertices=len(bm.verts), faces=len(bm.faces),
                  boundary=sum(e.is_boundary for e in bm.edges),
                  nonmanifold=sum(not e.is_manifold for e in bm.edges),
                  axilla_boundary=sum(e.is_boundary and any(.59<v.co.z<.74 and .04<abs(v.co.x)<.15 for v in e.verts) for e in bm.edges))
    bm.free()
    return result


def measure_grooves(mesh):
    """Track concave front/back valleys of the posterior cross-section interval.

    A breast can add a separate anterior interval. It must not be mistaken for
    the arm/torso cleft. The landmark windows deliberately target this asset.
    """
    mesh.calc_loop_triangles()
    vertices = np.array([v.co[:] for v in mesh.vertices])
    triangles = vertices[np.array([t.vertices[:] for t in mesh.loop_triangles])]
    guides = {}
    for sign in [-1, 1]:
        rows = [[.615,.073,.015,.068,.035], [.635,.072,.012,.066,.036]]
        for z in [.65,.66,.675,.685]:
            ts = triangles[(triangles[:,:,2].min(1)<z)&(triangles[:,:,2].max(1)>z)]
            segments = []
            for triangle in ts:
                points = []
                for a,b in zip(triangle,np.roll(triangle,-1,axis=0)):
                    if (a[2]-z)*(b[2]-z)<0:
                        points.append(a+(b-a)*(z-a[2])/(b[2]-a[2]))
                if len(points)==2: segments.append(points)
            segments = np.array(segments); profile = []
            for x in np.linspace(.052,.086,341):
                dx = segments[:,:,0]-sign*x
                crossing = segments[(dx.min(1)<0)&(dx.max(1)>0)]
                ys = sorted(a[1]+(b[1]-a[1])*(sign*x-a[0])/(b[0]-a[0]) for a,b in crossing)
                if len(ys)>=2: profile.append((x,ys[-2],ys[-1]))
            profile = np.array(profile)
            front = profile[(profile[:,0]>.067)&(profile[:,0]<.081)]
            back = profile[(profile[:,0]>.055)&(profile[:,0]<.074)]
            f = front[front[:,1].argmax()]; b = back[back[:,2].argmin()]
            assert -.012<f[1]<.025 and .025<b[2]<.075, 'Unexpected underarm features'
            rows.append([z,float(f[0]),float(f[1]),float(b[0]),float(b[2])])
        guides[sign] = np.array(rows)
    return guides


def groove(guides, sign, z):
    rows = guides[sign]
    return [float(np.interp(z,rows[:,0],rows[:,i])) for i in range(1,5)]


def center(guides, sign, y, z):
    xf,yf,xb,yb = groove(guides,sign,z)
    t = max(0,min(1,(y-yf)/(yb-yf)))
    return xf+(xb-xf)*t


def cut_web(body, guides):
    for sign in [-1,1]:
        verts=[]; faces=[]; nz=32; ny=20
        for z in np.linspace(.615,.683,nz):
            xf,yf,xb,yb=groove(guides,sign,z)
            width=.0025*math.sqrt(max(.002,1-(max(0,z-.677)/.006)**2))
            for dx in [-width,width]:
                for y in np.linspace(yf-.008,yb+.008,ny):
                    verts.append((sign*(center(guides,sign,y,z)+dx),y,z))
        for k in range(nz-1):
            for side in range(2):
                for j in range(ny-1):
                    a=k*2*ny+side*ny+j; faces.append([a,a+1,a+2*ny+1,a+2*ny])
            for j in [0,ny-1]:
                a=k*2*ny+j; faces.append([a,a+ny,a+3*ny,a+2*ny])
        for k in [0,nz-1]:
            for j in range(ny-1):
                a=k*2*ny+j; faces.append([a,a+1,a+ny+1,a+ny])
        mesh=bpy.data.meshes.new('Measured axillary groove'); mesh.from_pydata(verts,[],faces)
        bm=bmesh.new(); bm.from_mesh(mesh); bmesh.ops.recalc_face_normals(bm,faces=list(bm.faces)); bm.to_mesh(mesh); bm.free()
        cutter=bpy.data.objects.new('Temporary axillary relief',mesh); bpy.context.scene.collection.objects.link(cutter)
        bpy.context.view_layer.objects.active=body
        mod=body.modifiers.new('Open underarm groove','BOOLEAN'); mod.operation='DIFFERENCE'; mod.solver='EXACT'; mod.object=cutter
        while body.modifiers.find(mod.name)>0: bpy.ops.object.modifier_move_up(modifier=mod.name)
        bpy.ops.object.modifier_apply(modifier=mod.name)
        bpy.data.objects.remove(cutter,do_unlink=True); bpy.data.meshes.remove(mesh)
    bm=bmesh.new(); bm.from_mesh(body.data)
    edges=[e for e in bm.edges if e.calc_length()>.008 and all(.062<abs(v.co.x)<.086 and .62<v.co.z<.682 for v in e.verts)]
    bmesh.ops.subdivide_edges(bm,edges=edges,cuts=4,use_grid_fill=True)
    bmesh.ops.triangulate(bm,faces=[f for f in bm.faces if len(f.verts)>3 and all(.59<v.co.z<.74 for v in f.verts)])
    for _ in range(22):
        updates={}
        for v in bm.verts:
            x,y,z=v.co; ax=abs(x)
            w=smooth(-.018,.004,y)*(1-smooth(.062,.08,y))*smooth(.047,.065,ax)*(1-smooth(.097,.12,ax))*smooth(.60,.63,z)*(1-smooth(.685,.722,z))
            if w:
                mean=sum((e.other_vert(v).co for e in v.link_edges),Vector())/len(v.link_edges)
                updates[v]=v.co.lerp(mean,.42*w)
        for v,p in updates.items(): v.co=p
    bm.to_mesh(body.data); bm.free(); body.data.update()
    # Boolean cutters carry an empty material slot; new skin uses body material 0.
    for p in body.data.polygons:
        if body.data.materials[p.material_index] is None: p.material_index=0
    for i in range(len(body.data.materials)-1,-1,-1):
        if body.data.materials[i] is None: body.data.materials.pop(index=i)


def assign_weights(body, weights, guides, exact):
    neighbors=[[] for v in body.data.vertices]; factors=[]; result=[]
    for edge in body.data.edges:
        a,b=edge.vertices; neighbors[a].append(b); neighbors[b].append(a)
    for v,ws in zip(body.data.vertices,weights):
        x,y,z=v.co; ax=abs(x); sign=1 if x>0 else -1; suffix='l' if sign>0 else 'r'
        f=smooth(-.028,-.004,y)*(1-smooth(.065,.09,y))*smooth(.04,.055,ax)*(1-smooth(.125,.16,ax))*smooth(.575,.61,z)*(1-smooth(.70,.745,z))
        arm=smooth(-.0015,.0015,ax-center(guides,sign,y,z))
        target={k:t*(1-f) for k,t in ws.items()}
        target['arm_stretch_'+suffix]=target.get('arm_stretch_'+suffix,0)+arm*f
        target['spine_03_x']=target.get('spine_03_x',0)+(1-arm)*f
        result.append(target); factors.append(f)
    # Connectivity smoothing cannot jump across the newly opened air gap.
    for _ in range(18):
        nxt=[]
        for i,ws in enumerate(result):
            f=factors[i]*.5; d={k:t*(1-f) for k,t in ws.items()}
            if f:
                for j in neighbors[i]:
                    for k,t in result[j].items(): d[k]=d.get(k,0)+t*f/len(neighbors[i])
            nxt.append(d)
        result=nxt
    for v,ws,f,is_exact in zip(body.data.vertices,result,factors,exact):
        if f==0 and is_exact:
            for gi in [g.group for g in v.groups]: body.vertex_groups[gi].remove([v.index])
            for name,w in weights[v.index].items(): body.vertex_groups[name].add([v.index],w,'REPLACE')
            continue
        # Snapshot numeric IDs before mutating Blender's live group collection.
        for gi in [g.group for g in v.groups]: body.vertex_groups[gi].remove([v.index])
        ws=dict(sorted(((k,t) for k,t in ws.items() if t>1e-8),key=lambda item:-item[1])[:4]); total=sum(ws.values())
        assert total>0
        for name,w in ws.items(): body.vertex_groups[name].add([v.index],w/total,'REPLACE')


def run(publish=False):
    scene=bpy.data.scenes['Scene']; bpy.context.window.scene=scene
    body=bpy.data.objects['Body_Complete']; rig=bpy.data.objects['Landau_Rig']
    assert not body.get('underarm_repair'), 'Underarms already repaired; use the preserved baseline to rebuild'
    rig.animation_data.action=None
    for p in rig.pose.bones: p.matrix_basis=Matrix.Identity(4)
    scene.frame_set(0); bpy.context.view_layer.update()
    for key in body.data.shape_keys.key_blocks: key.value=0
    old=body.data.copy(); old.calc_loop_triangles()
    original_topology=topology(old); guides=measure_grooves(old)
    old_positions=[v.co.copy() for v in old.vertices]
    old_weights=[{body.vertex_groups[g.group].name:g.weight for g in v.groups} for v in old.vertices]
    keys=[(k.name,[p.co.copy() for p in k.data],k.slider_min,k.slider_max) for k in old.shape_keys.key_blocks]
    triangles=[tuple(t.vertices) for t in old.loop_triangles]
    tree=BVHTree.FromPolygons(old_positions,triangles,all_triangles=True)
    provenance=body.data.attributes.new('_axilla_source_vertex','INT','POINT')
    for i,d in enumerate(provenance.data): d.value=i+1
    face_ids=body.data.attributes.new('_axilla_source_face','INT','FACE')
    for i,d in enumerate(face_ids.data): d.value=i+1
    body.shape_key_clear(); cut_web(body,guides)
    mesh=body.data; provenance=mesh.attributes['_axilla_source_vertex']
    source=[]; weights=[]; new_count=0
    for v in mesh.vertices:
        i=provenance.data[v.index].value-1
        if 0<=i<len(old_positions) and (old_positions[i]-v.co).length<1e-7:
            mapping=[(i,1.)]
        else:
            point,normal,face,distance=tree.find_nearest(v.co); ids=triangles[face]
            a,b,c=(old_positions[j] for j in ids); u=b-a; w=c-a; p=point-a
            denom=u.dot(u)*w.dot(w)-u.dot(w)**2
            beta=(w.dot(w)*p.dot(u)-u.dot(w)*p.dot(w))/denom if abs(denom)>1e-20 else 0
            gamma=(u.dot(u)*p.dot(w)-u.dot(w)*p.dot(u))/denom if abs(denom)>1e-20 else 0
            values=[max(0,1-beta-gamma),max(0,beta),max(0,gamma)]; total=sum(values)
            mapping=[(j,t/total) for j,t in zip(ids,values)]; new_count+=1
        source.append(mapping); ws={}
        for j,t in mapping:
            for n,w in old_weights[j].items(): ws[n]=ws.get(n,0)+t*w
        weights.append(ws)
    for name,positions,lo,hi in keys:
        key=body.shape_key_add(name=name,from_mix=False); key.slider_min=lo; key.slider_max=hi; key.value=0
        for v,mapping in zip(mesh.vertices,source):
            delta=sum(((positions[j]-old_positions[j])*t for j,t in mapping),Vector())
            key.data[v.index].co=v.co+delta
    assign_weights(body,weights,guides,[len(m)==1 for m in source])
    # Preserve original corner fields exactly outside the repaired band.
    old_loops={(p.index,old.loops[li].vertex_index):li for p in old.polygons for li in p.loop_indices}
    areas=[Vector() for v in mesh.vertices]
    for p in mesh.polygons:
        for i in p.vertices: areas[i]+=p.normal*p.area
    normals=[]; colors={a.name:a for a in old.color_attributes}; face_ids=mesh.attributes['_axilla_source_face']
    for p in mesh.polygons:
        source_face=face_ids.data[p.index].value-1
        for li in p.loop_indices:
            vi=mesh.loops[li].vertex_index; mapping=source[vi]; v=mesh.vertices[vi]
            original_loop=old_loops.get((source_face,mapping[0][0])) if len(mapping)==1 else None
            local=.045<abs(v.co.x)<.125 and .595<v.co.z<.73 and -.03<v.co.y<.09
            if original_loop is not None and not local:
                normals.append(old.corner_normals[original_loop].vector[:])
            else: normals.append(areas[vi].normalized()[:])
            if original_loop is None:
                _,_,fi,_=tree.find_nearest(v.co); donor=old.loop_triangles[fi]
                original_loop=min(donor.loops,key=lambda j:(old_positions[old.loops[j].vertex_index]-v.co).length)
            for name,attr in colors.items():
                if attr.domain=='CORNER': mesh.color_attributes[name].data[li].color=attr.data[original_loop].color
            for old_uv in old.uv_layers:
                mesh.uv_layers[old_uv.name].data[li].uv=old_uv.data[original_loop].uv
        p.use_smooth=True
    mesh.normals_split_custom_set(normals)
    mesh.attributes.remove(mesh.attributes['_axilla_source_vertex']); mesh.attributes.remove(mesh.attributes['_axilla_source_face'])
    after=topology(mesh)
    assert after['boundary']==original_topology['boundary'] and after['nonmanifold']==original_topology['nonmanifold'] and after['axilla_boundary']==0, 'Repair opened the skin'
    assert all(len(v.groups)<=4 and abs(sum(g.weight for g in v.groups)-1)<1e-5 for v in mesh.vertices)
    # Every original protected vertex and every corresponding morph must survive.
    protected={i for i,p in enumerate(old_positions) if p.z>=.818 or p.z<.575 or abs(p.x)>.16 or p.y<-.03}
    retained={}
    for i,mapping in enumerate(source):
        if len(mapping)==1 and mapping[0][1]==1:
            j=mapping[0][0]
            if j not in retained or (mesh.vertices[i].co-old_positions[j]).length<(mesh.vertices[retained[j]].co-old_positions[j]).length: retained[j]=i
    assert protected<=retained.keys(), 'Protected original geometry was removed'
    error=max((mesh.vertices[retained[i]].co-old_positions[i]).length for i in protected)
    morph_error=max((mesh.shape_keys.key_blocks[name].data[retained[i]].co-positions[i]).length for name,positions,_,_ in keys for i in protected)
    weight_error=max(abs({body.vertex_groups[g.group].name:g.weight for g in mesh.vertices[retained[i]].groups}.get(n,0)-w) for i in protected for n,w in old_weights[i].items())
    assert weight_error==0, 'Protected skin weights changed'
    assert error<1e-7 and morph_error<1e-7, 'Protected face, hands or anterior body changed'
    body['underarm_repair']='measured_front_back_grooves_v1'
    report=dict(method='Curved 5 mm relief through measured anterior/posterior concavity valleys, rounded shoulder termination; local cap relaxation and connectivity-only weights',
                source='Current accepted body geometry; no image regeneration',source_glb_sha256=hashlib.sha256((OUT/'landau_character.glb').read_bytes()).hexdigest(),guides={str(k):v.tolist() for k,v in guides.items()},
                before=original_topology,after=after,new_interpolated_vertices=new_count,protected_vertices=len(protected),
                protected_position_error=error,protected_morph_error=morph_error,protected_weight_error=weight_error,rest_joints_changed=False,clothing_changed=False,
                default_editing_pose='T',bind_pose='Original A-pose retained for fitting and Running')
    (OUT/'underarm_repair.json').write_text(json.dumps(report,indent=2))
    if publish:
        checkpoint=OUT/'checkpoints/pre_underarm_20260914'; checkpoint.mkdir(parents=True,exist_ok=True)
        for name in ['landau_character.blend','landau_character.glb','asset_report.json','body_build.json']:
            if not (checkpoint/name).exists(): shutil.copy2(OUT/name,checkpoint/name)
        runpy.run_path(str(ROOT/'export_clothing.py'))['run'](body_repair=report)
    else:
        bpy.ops.wm.save_as_mainfile(filepath=str(OUT/'underarm_candidate.blend'))
    print(json.dumps(report))
    return report


if __name__=='__main__': run('--publish' in sys.argv)
