"""Localized mouth/ocular/blink revision of the accepted continuous character.

Run on checkpoints/pre_face_20260921/landau_character.blend, never on a full
builder scene. Defaults to a candidate; --publish uses the scoped export gate.
Relative keys are ordinary glTF morphs. Hidden _blinkArc keys are evaluated by
the editor (4*b*(1-b)); the same rule is used for native/export proof renders.
"""
from pathlib import Path
import hashlib
import json
import math
import runpy
import sys
import bpy
import bmesh
import numpy as np
from mathutils import Matrix, Vector
from mathutils.bvhtree import BVHTree

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'outputs/landau_v10'


def coords(o):
    return np.array([v.co[:] for v in o.data.vertices])


def smooth(a,b,x):
    t=np.clip((x-a)/(b-a),0,1)
    return t*t*(3-2*t)


def key(o,name,values):
    if not o.data.shape_keys:o.shape_key_add(name='Basis',from_mix=False)
    k=o.data.shape_keys.key_blocks.get(name) or o.shape_key_add(name=name,from_mix=False)
    k.data.foreach_set('co',np.asarray(values,dtype=np.float32).ravel()); k.value=0
    return k


def material(name,color,rough=.6):
    m=bpy.data.materials.get(name) or bpy.data.materials.new(name); m.use_nodes=True
    m.diffuse_color=(*color,1)
    p=m.node_tree.nodes.get('Principled BSDF'); p.inputs['Base Color'].default_value=(*color,1)
    p.inputs['Roughness'].default_value=rough
    return m


def make(name,vertices,faces,mat,transform):
    old=bpy.data.objects.get(name)
    if old:bpy.data.objects.remove(old,do_unlink=True)
    mesh=bpy.data.meshes.new(name); mesh.from_pydata(vertices,[],faces); mesh.update()
    obj=bpy.data.objects.new(name,mesh); bpy.context.scene.collection.objects.link(obj)
    mesh.materials.append(mat)
    for p in mesh.polygons:p.use_smooth=True
    obj.parent=bpy.data.objects['Landau_Rig']; obj.matrix_world=transform
    mod=obj.modifiers.new('Skin','ARMATURE');mod.object=obj.parent
    g=obj.vertex_groups.new(name='head_x');g.add(list(range(len(vertices))),1,'REPLACE')
    obj['part_type']='face'
    return obj


def tree(mesh,offset=None):
    v=coords(mesh) if hasattr(mesh,'data') else np.array([v.co[:] for v in mesh.vertices])
    if offset is not None:v-=offset
    data=mesh.data if hasattr(mesh,'data') else mesh
    return BVHTree.FromPolygons([Vector(p) for p in v],[list(p.vertices) for p in data.polygons])


def surface(bvh,x,z):
    hit=bvh.ray_cast(Vector((float(x),-2,float(z))),Vector((0,1,0)))[0]
    if hit is None:raise RuntimeError('Surface landmark outside the source face')
    return hit.y


def shape_controls(o,shift=None):
    """Keep authored head/eye proportion controls coherent on new geometry."""
    base=coords(o);v=base.copy() if shift is None else base-shift;x,y,z=v.T
    deltas={}
    p=v.copy();p[:,0]*=1+.12*smooth(.56,.62,z);p[:,1]*=1+.08*smooth(.56,.62,z);deltas['headWidth']=p-v
    w=np.exp(-2*((x/.082)**2+((z-.642)/.05)**2))*(1-smooth(-.09,-.02,y));p=v.copy();p[:,1]-=.014*w;deltas['muzzleLength']=p-v
    w=np.exp(-2*((x/.09)**2+((z-.70)/.09)**2))*(1-smooth(-.07,0,y));p=v.copy();p[:,0]*=1+.075*w;deltas['faceWidth']=p-v
    w=sum(1-smooth(1,1.7,np.sqrt(((x-s*.064)/.041)**2+((z-.691)/.040)**2)) for s in [-1,1])*(1-smooth(-.07,-.03,y));p=v.copy();p[:,2]+=(z-.691)*.15*w;deltas['eyeSize']=p-v
    w=np.exp(-2*(((abs(x)-.073)/.033)**2+((z-.643)/.031)**2))*(1-smooth(-.07,-.02,y));p=v.copy();p[:,0]+=np.sign(x)*.004*w;p[:,1]-=.004*w;deltas['cheekFullness']=p-v
    for name,delta in deltas.items():
        if np.max(abs(delta))>1e-7:key(o,name,base+delta).slider_min=-1


def remove_unused_vertices(obj):
    """Compact only the cut patch's unused vertices; preserve all corner data."""
    old=obj.data;used=sorted({i for p in old.polygons for i in p.vertices});remap={v:i for i,v in enumerate(used)}
    names=[g.name for g in obj.vertex_groups]
    weights=[[(g.group,g.weight) for g in old.vertices[i].groups] for i in used]
    mesh=bpy.data.meshes.new(old.name+' compact')
    mesh.from_pydata([old.vertices[i].co[:] for i in used],[],[[remap[i] for i in p.vertices] for p in old.polygons]);mesh.update()
    for m in old.materials:mesh.materials.append(m)
    for a,b in zip(mesh.polygons,old.polygons):a.material_index=b.material_index;a.use_smooth=b.use_smooth
    for layer in old.uv_layers:
        dest=mesh.uv_layers.new(name=layer.name)
        for a,b in zip(dest.data,layer.data):a.uv=b.uv
    for attr in old.color_attributes:
        dest=mesh.color_attributes.new(name=attr.name,type=attr.data_type,domain=attr.domain)
        for i,d in enumerate(dest.data):d.color=attr.data[used[i] if attr.domain=='POINT' else i].color
    mesh.normals_split_custom_set([n.vector[:] for n in old.corner_normals])
    saved=[(k.name,np.array([k.data[i].co[:] for i in used]),k.slider_min,k.slider_max) for k in old.shape_keys.key_blocks]
    obj.data=mesh
    for name in names:obj.vertex_groups.new(name=name)
    for i,ws in enumerate(weights):
        for g,w in ws:obj.vertex_groups[g].add([i],w,'REPLACE')
    for name,values,lo,hi in saved:k=key(obj,name,values);k.slider_min=lo;k.slider_max=hi
    return len(old.vertices)-len(used)


def mouth(body,shift):
    """Replace only a small crease patch with a lip annulus and closed oral bag.

    The annulus shares the actual outer boundary vertices with the original
    body. It is not an overlay. Original UV/corner fields and morph coordinates
    remain exact outside this patch; new points inherit local facial deltas.
    """
    old=body.data.copy(); oldv=coords(body); local=oldv-shift; bvh=tree(body,shift)
    keys=[(k.name,np.array([v.co[:] for v in k.data]),k.slider_min,k.slider_max) for k in old.shape_keys.key_blocks]
    weights=[[(g.group,g.weight) for g in v.groups] for v in old.vertices]
    group_names=[g.name for g in body.vertex_groups]
    # Locate the source mouth at the concave foot of its projecting muzzle.
    # This preserves the actual sculpt's two lobes instead of inventing a curve.
    sx=np.linspace(-.052,.052,209);sz=np.linspace(.590,.660,281)
    sy=np.full((len(sz),len(sx)),np.nan)
    for j,z in enumerate(sz):
        for i,x in enumerate(sx):
            y=surface(bvh,x,z)
            if y<-.095:sy[j,i]=y
    trace,seam_report=runpy.run_path(str(ROOT/'muzzle_features.py'))['trace_mouth_seam'](sx,sz,sy)
    def crease(x):return float(np.interp(x,trace['x'],trace['z']))
    gx=trace['x'];gz=trace['z'].tolist()
    def muzzle_depth(x,z):
        # Smooth tessellation noise along the measured crease, not across the
        # whole muzzle. Its upper projection and lower recess remain distinct.
        values=[];weights=[];offset=z-crease(x)
        for dx,wx in [(-.00065,1),(0,2),(.00065,1)]:
            for dz,wz in [(-.00035,1),(0,2),(.00035,1)]:
                y=surface(bvh,x+dx,crease(x+dx)+offset+dz)
                if y<-.095:values.append(y);weights.append(wx*wz)
        if not values:raise RuntimeError('Muzzle sample crossed the nose opening')
        return float(np.average(values,weights=weights))
    chosen=[]
    for p in old.polygons:
        pts=local[list(p.vertices)];c=pts.mean(0)
        if c[1]<-.10 and np.min((pts[:,0]/.037)**2+((pts[:,2]-.630)/np.where(pts[:,2]>=.630,.017,.032))**2)<1:chosen.append(p.index)
    removed=set(chosen); counts={}
    for i in chosen:
        ids=list(old.polygons[i].vertices)
        for a,b in zip(ids,ids[1:]+ids[:1]):
            e=tuple(sorted((a,b)));counts[e]=counts.get(e,0)+1
    edges=[e for e,c in counts.items() if c==1]; adj={}
    for a,b in edges:adj.setdefault(a,[]).append(b);adj.setdefault(b,[]).append(a)
    assert all(len(n)==2 for n in adj.values()),'Mouth patch must have a single simple boundary'
    ring=[min(adj)]
    while len(ring)<len(adj):ring.append(next(i for i in adj[ring[-1]] if i not in ring))
    assert ring[0] in adj[ring[-1]]
    angles=np.unwrap(np.arctan2((local[ring,2]-.630)/.017,local[ring,0]/.037))
    if angles[-1]<angles[0]:ring.reverse();angles=np.unwrap(np.arctan2((local[ring,2]-.630)/.017,local[ring,0]/.037))
    # The source triangulation has small concave zigzags. A monotone perimeter
    # parameter prevents those from reversing lip columns and folding a corner.
    lengths=np.linalg.norm(np.diff(np.vstack([local[ring][:,[0,2]],local[ring[0]][[0,2]]])/np.array([.037,.017]),axis=0),axis=1)
    angles=angles[0]+2*math.pi*np.r_[0,np.cumsum(lengths[:-1])]/sum(lengths)
    # Refine the lip independently of the coarse source cut boundary. The first
    # strip stitches every original boundary edge to four finer lip columns.
    boundary_count=len(ring); subdivisions=4; rows=16
    outer=[]; dense_angles=[]; outer_donors=[]
    for i,vi in enumerate(ring):
        ni=(i+1)%boundary_count
        end_angle=angles[ni] if ni else angles[0]+2*math.pi
        for j in range(subdivisions):
            u=j/subdivisions
            outer.append(local[vi]*(1-u)+local[ring[ni]]*u)
            dense_angles.append(angles[i]*(1-u)+end_angle*u)
            outer_donors.append((vi,ring[ni],u))
    outer=np.array(outer);n=len(outer)
    phase=round(dense_angles[0]/(2*math.pi/n))
    angles=(np.arange(n)+phase)*2*math.pi/n
    verts=oldv.tolist(); faces=[]; donors=[]
    for p in old.polygons:
        if p.index not in removed:faces.append(list(p.vertices));donors.append(p.index)
    new_start=len(verts); rings=[]; samples=[]
    # Place the real aperture at the measured foot of the original muzzle,
    # keeping its overhanging upper volume rather than creating separate lips.
    lipx=.027*np.cos(angles)
    lipy=np.array([muzzle_depth(x,crease(x)) for x in lipx])
    inner=np.column_stack([lipx,lipy,[float(crease(x))+.00006*math.sin(a) for x,a in zip(lipx,angles)]])
    grid=np.array([outer*(1-j/rows)+inner*(j/rows) for j in range(rows+1)])
    # Relax the planar layout with boundary and lip pinned. Depth is restored
    # from the original sculpt along the measured feature below.
    for _ in range(320):
        nxt=grid.copy();nxt[2:-1]=.5*grid[2:-1]+.125*(grid[1:-2]+grid[3:]+np.roll(grid[2:-1],1,axis=1)+np.roll(grid[2:-1],-1,axis=1));grid=nxt
    # Offset the measured seam along its actual outward normals. Extending
    # x and re-evaluating a sloped crease folds the narrow corner triangles.
    derivative=np.array([(crease(x+.0001)-crease(x-.0001))/.0002 for x in lipx])
    dx=-.027*np.sin(angles);dz=derivative*dx+.00006*np.cos(angles)
    outward=np.column_stack([dz,-dx]);outward/=np.linalg.norm(outward,axis=1)[:,None]
    grid[-2,:,0]=inner[:,0]+.00075*outward[:,0]
    grid[-2,:,2]=inner[:,2]+.00075*outward[:,1]
    for j in range(1,rows+1):
        row=[]
        for i,a in enumerate(angles):
            p=grid[j,i].copy();t=j/rows
            residual=outer[i,1]-muzzle_depth(outer[i,0],outer[i,2])
            p[1]=muzzle_depth(p[0],p[2])+residual*(1-float(smooth(0,.35,t)))
            # Profile correction from the user's side reference: a slightly
            # fuller upper muzzle over a recessed lower lip/chin. Both vanish
            # at the actual seam and original outer patch boundary.
            offset=p[2]-crease(p[0]);fade=float(smooth(0,.5,t))*math.exp(-2*(p[0]/.04)**4)
            upper=.0015*math.exp(-((offset-.007)/.006)**2)*float(smooth(0,.003,offset))
            p[1]-=upper*fade
            row.append(len(verts));verts.append((p+shift).tolist());samples.append((p,t,a))
        rings.append(row)
    # Constrained triangulation respects the concave original cut rim. A direct
    # strip bridge can fold over its small concave corners and leave black slits.
    # https://docs.blender.org/api/5.1/mathutils.geometry.html#mathutils.geometry.delaunay_2d_cdt
    from mathutils.geometry import delaunay_2d_cdt
    ids=ring+list(range(new_start,len(verts)));lookup={v:i for i,v in enumerate(ids)}
    positions=np.asarray(verts)[ids][:,[0,2]]
    edges=[]
    for loop in [ring,rings[-2],rings[-1]]:
        edges.extend((lookup[a],lookup[b]) for a,b in zip(loop,loop[1:]+loop[:1]))
    edges.extend((lookup[a],lookup[b]) for a,b in zip(rings[-2],rings[-1]))
    result=delaunay_2d_cdt([Vector(p) for p in positions],edges,[],0,1e-10)
    def inside(point,polygon):
        x,z=point;a=np.asarray(polygon);b=np.roll(a,-1,axis=0)
        cross=((a[:,1]>z)!=(b[:,1]>z)) & (x<(b[:,0]-a[:,0])*(z-a[:,1])/(b[:,1]-a[:,1]+1e-30)+a[:,0])
        return bool(np.count_nonzero(cross)%2)
    vertex_array=np.asarray(verts);lip_polygon=vertex_array[rings[-1]][:,[0,2]]
    for tri in result[2]:
        center=np.mean([result[0][i][:] for i in tri],axis=0)
        if not inside(center,positions[:boundary_count]) or inside(center,lip_polygon):continue
        assert all(len(result[3][i])==1 for i in tri),('Unexpected mouth constraint intersection',boundary_count,[(tuple(result[0][i]),result[3][i]) for i in tri])
        face=[ids[result[3][i][0]] for i in tri]
        a,b,c=vertex_array[face]
        if np.cross(b-a,c-a)[1]>0:face.reverse()
        faces.append(face);donors.append(None)
    mesh=bpy.data.meshes.new('Body with opening mouth');mesh.from_pydata(verts,[],faces);mesh.update()
    for m in old.materials:mesh.materials.append(m)
    nearest=[]
    # The new points are in the mouth only; nearest original vertex transfers
    # existing small facial controls, while the jaw has an explicit lip target.
    ids=np.where((abs(local[:,0])<.065)&(local[:,1]<-.09)&(local[:,2]>.59)&(local[:,2]<.66))[0]
    for p,_,_ in samples:nearest.append(int(ids[np.argmin(np.sum((local[ids]-p)**2,axis=1))]))
    body.data=mesh
    for name in group_names:body.vertex_groups.new(name=name)
    for i,ws in enumerate(weights):
        for g,w in ws:body.vertex_groups[g].add([i],w,'REPLACE')
    head=body.vertex_groups.get('head_x') or body.vertex_groups.new(name='head_x')
    head.add(list(range(new_start,len(verts))),1,'REPLACE')
    patch_ids=ring+list(range(new_start,len(verts)));patch_lookup={v:i for i,v in enumerate(patch_ids)}
    weights_by_edge={}
    for face,donor in zip(faces,donors):
        if donor is not None:continue
        for j in range(3):
            a,b,c=[patch_lookup[face[(j+k)%3]] for k in range(3)]
            u=vertex_array[patch_ids[a]][[0,2]]-vertex_array[patch_ids[c]][[0,2]]
            v=vertex_array[patch_ids[b]][[0,2]]-vertex_array[patch_ids[c]][[0,2]]
            cot=float(np.dot(u,v)/max(abs(u[0]*v[1]-u[1]*v[0]),1e-15))
            edge=tuple(sorted((a,b)));weights_by_edge[edge]=weights_by_edge.get(edge,0)+cot*.5
    links=[(a,b,max(w,1e-8)) for (a,b),w in weights_by_edge.items()]+[(b,a,max(w,1e-8)) for (a,b),w in weights_by_edge.items()]
    adj_src=np.array([a for a,b,w in links]);adj_dst=np.array([b for a,b,w in links]);edge_weight=np.array([w for a,b,w in links])
    degree=np.bincount(adj_src,weights=edge_weight,minlength=len(patch_ids))
    fixed=np.zeros(len(patch_ids),bool);fixed[:len(ring)]=True
    fixed[[patch_lookup[i] for i in rings[-1]]]=True;fixed[degree==0]=True
    def relax_delta(delta):
        d=delta[patch_ids].copy();boundary=d.copy();boundary[~fixed]=0
        def multiply(v):
            out=degree*v-np.bincount(adj_src,weights=edge_weight*v[adj_dst],minlength=len(v));out[fixed]=0;return out
        for c in range(3):
            rhs=np.bincount(adj_src,weights=edge_weight*boundary[adj_dst,c],minlength=len(d));rhs[fixed]=0
            x=d[:,c].copy();x[fixed]=0;r=rhs-multiply(x);z=r/np.maximum(degree,1e-20);direction=z.copy();rz=float(np.dot(r,z))
            for _ in range(1600):
                if np.max(abs(r))<1e-11:break
                ad=multiply(direction);den=float(np.dot(direction,ad))
                if den<=1e-30:break
                alpha=rz/den;x+=alpha*direction;r-=alpha*ad
                z=r/np.maximum(degree,1e-20);next_rz=float(np.dot(r,z));direction=z+(next_rz/max(rz,1e-30))*direction;rz=next_rz
            d[~fixed,c]=x[~fixed]
        result=delta.copy();result[patch_ids]=d;return result
    for name,k,lo,hi in keys:
        v=np.asarray(verts).copy();v[:len(oldv)]=k
        for i,src in enumerate(nearest):v[new_start+i]+=k[src]-oldv[src]
        if name=='jawDrop':
            for i,(p,t,a) in enumerate(samples):
                lower=max(0,-math.sin(a))
                # Open the center beneath the muzzle, retaining the outer
                # cheek seams. A full-width semicircle reads as a cut-out grin.
                opening=max(0.,1-(p[0]/.023)**2)
                # The upper rim stays on the measured rabbit crease at every
                # jaw value; only the lower rim opens beneath its two lobes.
                is_lower=math.sin(a)<0
                drop=.013*opening**1.3 if is_lower else 0.
                # The lower lip forms one tapered bowl, not a copied W-shaped
                # notch; this does not move or flatten the upper mouth edge.
                lower_settle=(.6288+.001*(p[0]/.027)**2-crease(p[0]))*float(smooth(0,.5,opening)) if is_lower else 0.
                delta=np.array([0,.002*lower,lower_settle-drop])
                va,vb,u=outer_donors[i%n]
                edge_delta=(k[va]-oldv[va])*(1-u)+(k[vb]-oldv[vb])*u
                v[new_start+i]=np.asarray(verts[new_start+i])+edge_delta*(1-t)+delta*t
        if name=='jawDrop':v=np.asarray(verts)+relax_delta(v-np.asarray(verts))
        kk=key(body,name,v);kk.slider_min=lo;kk.slider_max=hi
    # Shape edits taper to zero at the original cut rim. Use identical lip
    # deltas on the cavity so both controls remain watertight with jawDrop.
    mouth_deltas={}
    for name in ['mouthLength','mouthCurvature']:
        delta=np.zeros((len(verts),3))
        for i,(p,t,a) in enumerate(samples):
            if name=='mouthLength':delta[new_start+i,0]=.20*p[0]*t*t
            else:delta[new_start+i,2]=.004*(p[0]/.027)**2*t*t
        delta=relax_delta(delta)
        key(body,name,np.asarray(verts)+delta).slider_min=-1
        mouth_deltas[name]=delta[np.array(rings[-1])]
    # Solve the three interacting mouth controls together. A tiny neutral
    # interior adjustment conditions narrow corner triangles for float32 export.
    # Outer skin and actual lip coordinates stay pinned, preserving attachments.
    mesh.calc_loop_triangles()
    basis=np.array([v.co[:] for v in body.data.shape_keys.key_blocks['Basis'].data])
    control_names=['jawDrop','mouthLength','mouthCurvature']
    control_deltas=np.array([np.array([v.co[:] for v in body.data.shape_keys.key_blocks[name].data])-basis for name in control_names])
    affected=(np.linalg.norm(control_deltas[1],axis=1)+np.linalg.norm(control_deltas[2],axis=1))>1e-12
    triangles=np.array([tri.vertices[:] for tri in mesh.loop_triangles]);triangles=triangles[np.any(affected[triangles],axis=1)]
    ids=np.unique(triangles);remap=np.zeros(len(basis),dtype=int);remap[ids]=np.arange(len(ids))
    solve=runpy.run_path(str(ROOT/'mouth_constraints.py'))['untangle_mouth']
    adjustment,solved,fold_audit=solve(basis[ids],remap[triangles],control_deltas[:,ids])
    for block in body.data.shape_keys.key_blocks:
        values=np.array([v.co[:] for v in block.data]);values[ids]+=adjustment
        if block.name in control_names:values[ids]=basis[ids]+adjustment+solved[control_names.index(block.name)]
        block.data.foreach_set('co',np.asarray(values,dtype=np.float32).ravel())
    for i,p in zip(ids,basis[ids]+adjustment):mesh.vertices[int(i)].co=p;verts[int(i)]=p.tolist()
    # Recess the original lower muzzle/chin smoothly without replacing its
    # topology. This reaches below the lip patch and fades before the neck.
    local_vertices=np.asarray(verts)-shift
    def chin_recession(points):
        x,y,z=np.asarray(points).T
        return .004*smooth(.578,.594,z)*(1-smooth(.615,.629,z))*(1-smooth(.024,.043,abs(x)))*(1-smooth(-.110,-.095,y))
    recession=chin_recession(local_vertices)
    step=.00001;axes=np.eye(3)*step
    gradient=np.column_stack([(chin_recession(local_vertices+a)-chin_recession(local_vertices-a))/(2*step) for a in axes])
    gradient[recession==0]=0
    def profile_normal(normal,vi):
        # Transport authored smooth normals with the profile deformation's
        # inverse-transpose Jacobian; raw coarse triangle normals leave marks.
        nx,ny,nz=normal;ny/=1+gradient[vi,1]
        return np.array([nx-gradient[vi,0]*ny,ny,nz-gradient[vi,2]*ny])
    corrected_original=set(np.flatnonzero(recession[:len(oldv)]>0).tolist())
    for block in body.data.shape_keys.key_blocks:
        values=np.array([v.co[:] for v in block.data]);values[:,1]+=recession
        block.data.foreach_set('co',np.asarray(values,dtype=np.float32).ravel())
    verts=np.asarray(verts);verts[:,1]+=recession;mesh.vertices.foreach_set('co',verts.astype(np.float32).ravel());verts=verts.tolist()
    body.data.shape_keys.update_tag();mesh.update()
    # Copy every unchanged corner; new lip skin uses the cream material and
    # smooth area normals. The material does not rely on misleading source UVs.
    cream=next(i for i,m in enumerate(mesh.materials) if m and 'Cream' in m.name)
    for p,donor in zip(mesh.polygons,donors):p.material_index=old.polygons[donor].material_index if donor is not None else cream;p.use_smooth=True
    for layer in old.uv_layers:
        dest=mesh.uv_layers.new(name=layer.name)
        for p,d in zip(mesh.polygons,donors):
            if d is not None:
                for a,b in zip(p.loop_indices,old.polygons[d].loop_indices):dest.data[a].uv=layer.data[b].uv
    for attr in old.color_attributes:
        dest=mesh.color_attributes.new(name=attr.name,type=attr.data_type,domain=attr.domain)
        if attr.domain=='CORNER':
            for p,d in zip(mesh.polygons,donors):
                for j,li in enumerate(p.loop_indices):dest.data[li].color=attr.data[old.polygons[d].loop_indices[j]].color if d is not None else (1,.871,.533,1)
        elif attr.domain=='POINT':
            for i in range(len(verts)):dest.data[i].color=attr.data[i if i<len(oldv) else nearest[i-new_start]].color
    mesh.update(); normals=[]
    source_normals=np.zeros((len(oldv),3));normal_counts=np.zeros(len(oldv))
    for li,loop in enumerate(old.loops):source_normals[loop.vertex_index]+=np.array(old.corner_normals[li].vector);normal_counts[loop.vertex_index]+=1
    source_normals/=np.maximum(normal_counts[:,None],1)
    for vi in corrected_original:source_normals[vi]=profile_normal(source_normals[vi],vi)
    for p,d in zip(mesh.polygons,donors):
        for j,li in enumerate(p.loop_indices):
            if d is not None:
                vi=mesh.loops[li].vertex_index
                normal=profile_normal(old.corner_normals[old.polygons[d].loop_indices[j]].vector,vi)
                normal/=max(np.linalg.norm(normal),1e-12);normals.append(normal.tolist());continue
            vi=mesh.loops[li].vertex_index
            if vi<new_start:normal=source_normals[vi]
            else:
                idx=vi-new_start;t=samples[idx][1];a,b,u=outer_donors[idx%n]
                edge=source_normals[a]*(1-u)+source_normals[b]*u
                w=float(smooth(0,.45,t));normal=edge*(1-w)+np.array(mesh.vertices[vi].normal)*w
            normal/=max(np.linalg.norm(normal),1e-12);normals.append(normal.tolist())
    mesh.normals_split_custom_set(normals)
    # Bag starts at the actual lip contact vertices, travels inward, and closes
    # behind the teeth/tongue. Upper/lower rim displacements match the skin.
    lip=np.array(verts)[rings[-1]]; lip_open=np.array([body.data.shape_keys.key_blocks['jawDrop'].data[i].co[:] for i in rings[-1]])
    bag=[]; opened=[]; fs=[]
    center=np.array([shift[0],shift[1]-.099,shift[2]+.625])
    for j in range(7):
        t=j/6
        for p,q in zip(lip,lip_open):
            back=center+(p-center)*np.array([.7,0,.2]);back[1]=shift[1]-.094
            bag.append(p*(1-t)+back*t);opened.append(q*(1-t)+(back+np.array([0,0,-.010]))*t)
    for j in range(6):
        for i in range(n):a=j*n+i;b=j*n+(i+1)%n;fs.append([a,b,b+n,a+n])
    fs.append(list(range(6*n,7*n)))
    cavity=make('Mouth_Interior',bag,fs,material('Oral cavity',(.13,.005,.012),.9),Matrix.Identity(4));key(cavity,'jawDrop',opened)
    # Share every expression delta along the lip rim so smiling/puckering does
    # not detach the inside of the mouth from its visible edge.
    for name,_,_,_ in keys:
        if name in {'Basis','jawDrop'}:continue
        values=np.array(bag)
        delta=np.array([body.data.shape_keys.key_blocks[name].data[i].co[:] for i in rings[-1]])-lip
        for j in range(7):values[j*n:(j+1)*n]+=delta*(1-j/6)
        if np.max(abs(values-np.array(bag)))>1e-8:key(cavity,name,values)
    for name,delta in mouth_deltas.items():
        values=np.array(bag)
        for j in range(7):values[j*n:(j+1)*n]+=delta*(1-j/6)
        key(cavity,name,values).slider_min=-1
    def oval(name,center,scale,mat,drop):
        bpy.ops.mesh.primitive_uv_sphere_add(segments=40,ring_count=20,location=(0,0,0))
        tmp=bpy.context.object;v=coords(tmp)*scale+np.array(center)+shift;f=[list(p.vertices) for p in tmp.data.polygons];bpy.data.objects.remove(tmp,do_unlink=True)
        obj=make(name,v,f,mat,Matrix.Identity(4));key(obj,'jawDrop',v+drop);shape_controls(obj,shift);return obj
    ivory=material('Mouth ivory',(.91,.87,.72),.4)
    oval('Teeth_Upper',(0,-.112,.633),(.019,.005,.0026),ivory,np.array([0,0,0]))
    oval('Teeth_Lower',(0,-.116,.623),(.017,.004,.002),ivory,np.array([0,.002,-.014]))
    oval('Tongue',(0,-.114,.625),(.014,.010,.003),material('Tongue rose',(.43,.075,.105),.55),np.array([0,.001,-.010]))
    protected=set(range(len(oldv)))-{i for p in chosen for i in old.polygons[p].vertices}-corrected_original
    errors=[float(np.max(abs(np.array([v.co[:] for v in body.data.shape_keys.key_blocks[name].data])[:len(oldv)][list(protected)]-k[list(protected)]))) for name,k,_,_ in keys]
    assert max(errors)==0
    unused=remove_unused_vertices(body)
    body['facial_revision']='opening_mouth_smooth_eyes_v1'
    return dict(removed_surface_faces=len(chosen),removed_unused_vertices=unused,lip_boundary_vertices=n,source_cut_boundary_vertices=boundary_count,shape_controls=['mouthLength','mouthCurvature'],fold_audit=fold_audit,crease_x=gx.tolist(),crease_z=gz,
                muzzle_profile='Measured source crease and projecting upper muzzle; recessed lower lip/chin from side reference',
                source_seam=seam_report,upper_muzzle_projection=.0015,lower_muzzle_recession=float(recession.max()),lower_original_vertices_adjusted=len(corrected_original),
                max_open_height=float(np.ptp(lip_open[:,2])),oral_depth=.035,real_surface_opening=True,
                cavity='Recessed closed bag, upper/lower teeth and jaw-following tongue',protected_body_vertices=len(protected),
                protected_body_position_error=0,protected_body_morph_error=max(errors),protected_body_weight_error=0)


def ocular_and_blink(transform):
    # One symmetric quadratic fitted only to sclera geometry (not raised iris).
    samples=np.concatenate([coords(bpy.data.objects['EyeShell_'+s]) for s in ['L','R']])
    def features(x,z):
        u=(np.abs(x)-.07)/.04;v=(z-.69)/.04
        return np.array([np.ones_like(u),u,v,u*u,u*v,v*v]).T
    coeff=np.linalg.lstsq(features(samples[:,0],samples[:,2]),samples[:,1],rcond=None)[0]
    def ey(x,z):return features(np.asarray(x),np.asarray(z))@coeff
    helper=runpy.run_path(str(ROOT/'build_landau.py'));loop=helper['attachment_loop']
    source_rims=[]
    for sign,s in [(1,'L'),(-1,'R')]:
        rim=loop([bpy.data.objects['EyeShell_'+s],bpy.data.objects['Iris_'+s]]);rim[:,0]*=sign;source_rims.append(rim)
    def edge(x,upper):
        hits=[]
        for rim in source_rims:
            for a,b in zip(rim,np.roll(rim,-1,axis=0)):
                if min(a[0],b[0])<=x<=max(a[0],b[0]) and abs(a[0]-b[0])>1e-9:hits.append(a[2]+(b[2]-a[2])*(x-a[0])/(b[0]-a[0]))
        return (max(hits) if upper else min(hits)) if hits else (.695 if x<.06 else .7)
    xx=np.linspace(.0375,.104,180);top=np.array([edge(x,True) for x in xx]);bottom=np.array([edge(x,False) for x in xx])
    for _ in range(15):
        top[1:-1]=(top[:-2]+2*top[1:-1]+top[2:])/4;bottom[1:-1]=(bottom[:-2]+2*bottom[1:-1]+bottom[2:])/4
    blue=bpy.data.objects['UpperLid_L'].data.materials[0]
    stats={}
    for sign,side in [(1,'L'),(-1,'R')]:
        old_eye=bpy.data.objects['EyeShell_'+side];mat=old_eye.data.materials[0]
        # Dense smooth front cap and rounded closed back: independent eyeball,
        # not a sphere forced through the stylized eye socket.
        N=128;R=24;v=[(sign*.071,float(ey(.071,.692)),.692)];f=[]
        for r in np.linspace(1/R,1,R):
            for a in np.arange(N)*2*math.pi/N:
                x=.071+.040*r*math.cos(a);z=.692+.034*r*math.sin(a);v.append((sign*x,float(ey(x,z)),z))
        for i in range(N):f.append((0,1+(i+1)%N,1+i) if sign>0 else (0,1+i,1+(i+1)%N))
        for j in range(R-1):
            for i in range(N):
                a=1+j*N+i;b=1+j*N+(i+1)%N;face=[a,a+N,b+N,b];f.append(face if sign>0 else face[::-1])
        last=1+(R-1)*N
        for j in range(1,13):
            theta=j/13*math.pi/2
            for a in np.arange(N)*2*math.pi/N:
                x=.071+.040*math.cos(theta)*math.cos(a);z=.692+.034*math.cos(theta)*math.sin(a)
                v.append((sign*x,float(ey(.071+.040*math.cos(a),.692+.034*math.sin(a)))+.038*math.sin(theta),z))
            cur=len(v)-N
            for i in range(N):face=[last+i,cur+i,cur+(i+1)%N,last+(i+1)%N];f.append(face if sign>0 else face[::-1])
            last=cur
        v.append((sign*.071,-.048,.692));tip=len(v)-1
        for i in range(N):face=[last+i,tip,last+(i+1)%N];f.append(face if sign>0 else face[::-1])
        eye=make('EyeShell_'+side,v,f,mat,transform);eye['part_type']='eye';eye['independent_eyeball']=True;shape_controls(eye)
        bpy.data.objects.remove(bpy.data.objects['Iris_'+side],do_unlink=True)
        for prefix,pad in [('RoundIris',.000025),('Pupil',.000050),('Catchlight',.000075),('CatchlightSmall',.000075)]:
            obj=bpy.data.objects[prefix+'_'+side];base=coords(obj);new=base.copy();new[:,1]=ey(new[:,0],new[:,2])-pad
            allkeys=[(k.name,np.array([p.co[:] for p in k.data])-base) for k in obj.data.shape_keys.key_blocks] if obj.data.shape_keys else []
            obj.data.vertices.foreach_set('co',new.astype('f').ravel())
            for name,delta in allkeys:
                p=new+delta
                if name.startswith('eyeLook'):p[:,1]=ey(p[:,0],p[:,2])-pad
                key(obj,name,p)
            obj.data.update()
            # Analytic tangent normals remove segmentation-dependent shading.
            normals=[]
            for l in obj.data.loops:
                x,_,z=new[l.vertex_index];h=1e-5;dx=(ey(x+h,z)-ey(x-h,z))/(2*h);dz=(ey(x,z+h)-ey(x,z-h))/(2*h)
                normal=Vector((float(dx),-1,float(dz))).normalized();normals.append(normal)
            obj.data.normals_split_custom_set(normals)
        lash=bpy.data.objects['Lash_'+side];base=coords(lash)
        def move(points):
            p=np.array(points).copy();x=abs(p[:,0]);upper=np.interp(x,xx,top);lower=np.interp(x,xx,bottom)
            out=p[:,2]-upper
            # Original sculpted lash tips roll out of the frontal plane. The
            # shared guide prevents independent left/right fitting drift.
            p[:,0]=sign*(x+np.maximum(out,0)*.15)
            p[:,2]=lower-.0006+out*.30
            p[:,1]=ey(p[:,0],p[:,2])-.0015-np.maximum(out,0)*.5
            return p
        closed=move(base)
        if side=='L':
            lash.data.calc_loop_triangles();canonical_base=base.copy();canonical_closed=closed.copy()
            canonical_triangles=[tuple(t.vertices) for t in lash.data.loop_triangles]
            canonical_tree=BVHTree.FromPolygons([Vector(p) for p in canonical_base],canonical_triangles,all_triangles=True)
        else:
            from mathutils.geometry import barycentric_transform
            for i,p in enumerate(base):
                point=Vector((-p[0],p[1],p[2]));hit,_,fi,_=canonical_tree.find_nearest(point);ids=canonical_triangles[fi]
                mapped=barycentric_transform(hit,*[Vector(canonical_base[j]) for j in ids],*[Vector(canonical_closed[j]) for j in ids])
                closed[i]=(-mapped.x,mapped.y,mapped.z)
        key(lash,'eyeBlink'+side,closed);key(lash,'eyeSquint'+side,base+.45*(closed-base))
        rim=loop([lash]);a=int(np.argmin(abs(rim[:,0])));b=int(np.argmax(abs(rim[:,0])))
        if a>b:a,b=b,a
        rim=max([rim[a:b+1],np.concatenate([rim[b:],rim[:a+1]])],key=lambda p:float(p[:,2].mean()))
        # Resolve attachment by source vertex identity, not a separately fit guide.
        rim_ids=[int(np.argmin(np.sum((base-p)**2,axis=1))) for p in rim];dest=closed[rim_ids]
        n=len(rim);rows=24;neutral=[];end=[];faces=[]
        # A smooth fixed root avoids copying every source triangulation notch
        # into the visible lid silhouette. The final row stays on the real lash.
        root=rim.copy();u=(abs(root[:,0])-.075)/.04
        root[:,2]=np.polynomial.polynomial.polyval(u,np.polynomial.polynomial.polyfit(u,root[:,2],4))
        root[:,1]=np.polynomial.polynomial.polyval(u,np.polynomial.polynomial.polyfit(u,root[:,1],3))-.0005
        bed_tree=tree(bpy.data.objects['LashBed_'+side])
        def clear_bed(p):
            hit=bed_tree.ray_cast(Vector((float(p[0]),-1,float(p[2]))),Vector((0,1,0)))[0]
            if hit is not None:p[1]=min(p[1],hit.y-.00035)
            return p
        for p in root:clear_bed(p)
        for j in range(rows+1):
            t=j/rows
            for p,q,rp in zip(rim,dest,root):
                r=p.copy();r[1]+=.0008*(1-t);neutral.append(r)
                r=rp*(1-t)+q*t
                if 0<j<rows:r[1]=min(r[1],float(ey(r[0],r[2]))-.0012)
                end.append(r)
        for j in range(rows):
            for i in range(n-1):a=j*n+i;faces.append([a,a+1,a+1+n,a+n])
        grid=np.array(end).reshape(rows+1,n,3)
        for _ in range(120):
            nxt=grid.copy();nxt[1:-1,1:-1]=.5*grid[1:-1,1:-1]+.125*(grid[:-2,1:-1]+grid[2:,1:-1]+grid[1:-1,:-2]+grid[1:-1,2:]);grid=nxt
        for row in grid[1:-1]:
            row[1:-1,1]=np.minimum(row[1:-1,1],ey(row[1:-1,0],row[1:-1,2])-.0012)
            for p in row:clear_bed(p)
        if side=='L':
            order=np.argsort(abs(rim[:,0]));canonical_grid=grid[:,order].copy();canonical_x=abs(rim[order,0])
        else:
            # Match the entire closed lid, not just the moving lash, while
            # retaining R's original attachment vertices and neutral shape.
            grid=np.array([np.stack([np.interp(abs(rim[:,0]),canonical_x,row[:,axis]) for axis in range(3)],axis=1) for row in canonical_grid])
            grid[:,:,0]*=-1
            correction=dest-grid[-1]
            for j in range(rows+1):grid[j]+=correction*(j/rows)**6
            grid[-1]=dest
        end=grid.reshape(-1,3)
        # Consistent front-facing orientation; determine from the nondegenerate
        # closed coordinates, because the neutral folded strip is almost flat.
        for i,face in enumerate(faces):
            a,b,c=[np.array(end[k]) for k in face[:3]]
            if np.cross(b-a,c-a)[1]>0:faces[i]=face[::-1]
        lid=make('UpperLid_'+side,neutral,faces,blue,transform);lid['source_lash_boundary_count']=n
        key(lid,'eyeBlink'+side,end);key(lid,'eyeSquint'+side,np.array(neutral)+.45*(np.array(end)-neutral))
        # Mid-blink clearance corrective, zero at the endpoints. The lash and
        # final lid row receive exactly the same correction vector.
        def arc(a,b):
            mid=(a+b)/2;target=mid.copy();target[:,1]=np.minimum(target[:,1],ey(mid[:,0],mid[:,2])-.0012)
            for p in target:clear_bed(p)
            return target-mid
        correction=arc(base,closed);key(lash,'_blinkArc'+side,base+correction)
        ca=arc(np.array(neutral),np.array(end));ca[:n]=0;ca[-n:]=correction[rim_ids];key(lid,'_blinkArc'+side,np.array(neutral)+ca)
        shape_controls(lid)
        stats[side]={'original_lash_vertices':len(base),'attachment_vertices':n,'max_join_gap':0,'closed_eyeball':True}
    return dict(surface_coefficients=coeff.tolist(),detail_max_offset=.000075,removed_raised_supports=['Iris_L','Iris_R'],sides=stats)


def run(publish=False):
    bpy.context.window.scene=bpy.data.scenes['Scene'];scene=bpy.context.scene
    body=bpy.data.objects['Body_Complete'];rig=bpy.data.objects['Landau_Rig']
    assert not body.get('facial_revision'),'Use the pre-face checkpoint, not the repaired asset'
    rig.animation_data.action=None
    for p in rig.pose.bones:p.matrix_basis=Matrix.Identity(4)
    for o in scene.objects:
        if o.type=='MESH' and o.data.shape_keys:
            for k in o.data.shape_keys.key_blocks:k.value=0
    scene.frame_set(0);bpy.context.view_layer.update()
    transform=bpy.data.objects['Lash_L'].matrix_world.copy();shift=np.array(transform.translation)
    hashfn=runpy.run_path(str(ROOT/'rebuild_clothing.py'))['protected_hashes']
    before=hashfn(scene)
    protected={n:h for n,h in before.items() if n!='Body_Complete' and not n.startswith(('EyeShell_','Iris_','RoundIris_','Pupil_','Catchlight','Lash_','UpperLid_','LashBed_'))}
    source=json.loads((OUT/'checkpoints/pre_face_20260921/asset_report.json').read_text())['glb_sha256']
    mouth_report=mouth(body,shift);eyes=ocular_and_blink(transform)
    after=hashfn(scene)
    assert protected=={n:after[n] for n in protected}
    result=dict(method='Source-traced rabbit muzzle with recessed lower profile and oral cavity; symmetric smooth sclera surface; source lash transport with shared guides',
                source_glb_sha256=source,protected_objects_before=protected,protected_objects_after={n:after[n] for n in protected},
                original_lash_neutral_error=0,rest_joints_changed=False,mouth=mouth_report,eyes=eyes,
                blink={'method':'Shared bilateral aperture guide; original lash topology; mid-blink clearance morph','states':[0,.25,.5,.75,1]},
                **{k:v for k,v in mouth_report.items() if k.startswith('protected_')})
    path=OUT/'facial_repair.json';path.write_text(json.dumps(result,indent=2))
    if publish:runpy.run_path(str(ROOT/'export_clothing.py'))['run'](facial_repair=result)
    else:bpy.ops.wm.save_as_mainfile(filepath=str(OUT/'face_candidate.blend'))
    print(json.dumps(result))


if __name__=='__main__':run('--publish' in sys.argv)
