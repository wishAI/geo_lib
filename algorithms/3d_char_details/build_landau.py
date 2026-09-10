"""Reproducible Landau character preparation in Blender; source inputs are immutable.

Segmentation uses reviewed Landau landmarks, not a general automatic classifier.
Hidden body geometry is reconstructed. Source facial surfaces are retained.
"""
from pathlib import Path
import bpy
import bmesh
import numpy as np
import json
import math
import hashlib
import runpy
from mathutils import Vector
from mathutils.bvhtree import BVHTree
from mathutils.kdtree import KDTree

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'outputs/landau_v10'
SOURCE=ROOT/'inputs/landau_v10/source.usdc'
OUT.mkdir(parents=True,exist_ok=True)
REPORT={'version':1,'source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        'status':'authored_preview','inferred_geometry':['Body_UnderClothes'],
        'limitations':['Single-image hidden body is inferred; not anatomical ground truth.',
            'Body skinning retains the source rig; extreme poses need art review.',
            'Facial controls are a custom subset, not a certified ARKit/VRM/FaceRig mapping.',
            'No engine-specific humanoid retargeting, LOD, collision or cloth simulation certification.']}

def smoothstep(a,b,x):
    t=np.clip((x-a)/(b-a),0,1);return t*t*(3-2*t)

def material(name,color,rough=.65):
    m=bpy.data.materials.new(name);m.diffuse_color=(*color,1);m.use_nodes=True
    n=m.node_tree.nodes.get('Principled BSDF');n.inputs['Base Color'].default_value=(*color,1)
    n.inputs['Roughness'].default_value=rough
    return m

def mesh_object(name,verts,faces,mat):
    d=bpy.data.meshes.new(name);d.from_pydata(verts,[],faces);d.update()
    bm=bmesh.new();bm.from_mesh(d)
    for f in bm.faces:
        if f.normal.y>0:f.normal_flip()
    bm.to_mesh(d);bm.free()
    o=bpy.data.objects.new(name,d);bpy.context.scene.collection.objects.link(o)
    d.materials.append(mat)
    for p in d.polygons:p.use_smooth=True
    return o

def bind(o,weights=None,bone='head_x'):
    o.parent=RIG;o.matrix_parent_inverse=RIG.matrix_world.inverted()
    mod=o.modifiers.new('Skin','ARMATURE');mod.object=RIG
    if weights is None:
        g=o.vertex_groups.new(name=bone);g.add(list(range(len(o.data.vertices))),1,'REPLACE')
    else:
        for i,ws in enumerate(weights):
            for name,w in ws:
                if w<1e-6:continue
                g=o.vertex_groups.get(name) or o.vertex_groups.new(name=name);g.add([i],float(w),'REPLACE')

def add_key(o,name,coords):
    base=np.array([v.co[:] for v in o.data.vertices])
    if np.max(np.abs(base-np.asarray(coords)))<1e-7:return
    if not o.data.shape_keys:o.shape_key_add(name='Basis')
    key=o.shape_key_add(name=name);key.data.foreach_set('co',np.asarray(coords,dtype=np.float32).reshape(-1))
    key.slider_min=-1 if name in {'headWidth','bodyWidth','earLength','muzzleLength','faceWidth','clothingEase','eyeSize','cheekFullness'} else 0;key.slider_max=1

def nearest_weights(p):
    _,idx,_=SOURCE_TREE.find(Vector(p));return WEIGHTS[idx]

def surface_y(x,z):
    hit=BVH.ray_cast(Vector((x,-1,z)),Vector((0,1,0)))[0]
    return hit.y if hit is not None else -.10

def subset(name,ids):
    # Copy corner UVs and authored split normals exactly. Recalculating normals at
    # garment boundaries makes seams visible even when the positions are unchanged.
    src=SOURCE_OBJ.data
    chosen=[src.polygons[i] for i in ids]
    source_ids=sorted({i for p in chosen for i in p.vertices})
    remap={old:new for new,old in enumerate(source_ids)}
    mesh=bpy.data.meshes.new(name)
    mesh.from_pydata([src.vertices[i].co[:] for i in source_ids],[],[[remap[i] for i in p.vertices] for p in chosen])
    for mat in src.materials:mesh.materials.append(mat)
    loops=[i for p in chosen for i in p.loop_indices]
    for layer in src.uv_layers:
        dest=mesh.uv_layers.new(name=layer.name)
        dest.data.foreach_set('uv',np.array([layer.data[i].uv[:] for i in loops],dtype=np.float32).ravel())
    for old,new in zip(chosen,mesh.polygons):
        new.use_smooth=old.use_smooth;new.material_index=old.material_index
    mesh.normals_split_custom_set([src.corner_normals[i].vector[:] for i in loops])
    obj=bpy.data.objects.new(name,mesh);bpy.context.scene.collection.objects.link(obj)
    obj.matrix_world=SOURCE_OBJ.matrix_world.copy()
    bind(obj,[WEIGHTS[i] for i in source_ids])
    obj['source_vertex_count']=len(source_ids)
    SOURCE_MAP[name]=source_ids
    return obj

def segment():
    groups={}
    for p in SOURCE_OBJ.data.polygons:
        x,y,z=p.center; ax=abs(x);side='L' if x>0 else 'R'
        if p.index in FACE_REGIONS:name=FACE_REGIONS[p.index]
        elif z>=.568:name='Head'
        elif y>.106 and .31<z<.42:name='Tail'
        elif ax>.177 and z<.377:name='Hand_'+side
        elif ax>.143 and .371<z<.424:name='Cuff_'+side
        elif ax>.073 and z>.388:name='Sleeve_'+side
        elif z<.119:name='Boot_'+side
        elif z<.307:name='Trousers'
        else:name='Vest'
        groups.setdefault(name,[]).append(p.index)
    parts=[subset(name,ids) for name,ids in groups.items()]
    REPORT['source_triangles']=len(SOURCE_OBJ.data.polygons)
    REPORT['separated_parts']={o.name:len(o.data.polygons) for o in parts}
    # Preserve the source shell; double-sided materials avoid fragile automatic thickening.
    for o in parts:
        if o.name.startswith(('Vest','Sleeve','Cuff','Trousers','Boot')):
            o['part_type']='clothing'
        elif o.name.startswith(('Iris_','EyeShell_')):o['part_type']='eye'
        elif o.name.startswith(('Face_','Brow_','Lash_','Nose')):o['part_type']='face'
        else:o['part_type']='body'
    return parts

def ellipsoid(name,center,scale):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=24,ring_count=16,location=center)
    o=bpy.context.object;o.name=name;o.scale=scale
    bpy.ops.object.transform_apply(location=False,rotation=False,scale=True)
    return o

def capsule(name,a,b,r1,r2):
    a,b=Vector(a),Vector(b);axis=b-a
    bpy.ops.mesh.primitive_cone_add(vertices=24,radius1=r1,radius2=r2,depth=axis.length,location=(a+b)/2)
    o=bpy.context.object;o.name=name;o.rotation_euler=axis.to_track_quat('Z','Y').to_euler()
    bpy.ops.object.transform_apply(location=False,rotation=True,scale=True)
    return o

def underbody():
    items=[ellipsoid('Torso',(0,.018,.435),(.058,.049,.125)),ellipsoid('Pelvis',(0,.015,.305),(.066,.045,.061)),
           ellipsoid('Neck',(0,.006,.563),(.024,.024,.036))]
    for s in [-1,1]:
        items.extend([capsule('UpperArm',(s*.055,.01,.520),(s*.118,.01,.449),.024,.019),
            capsule('Forearm',(s*.118,.01,.449),(s*.182,-.004,.375),.019,.015),
            ellipsoid('Elbow',(s*.118,.01,.449),(.020,.020,.021)),
            capsule('Thigh',(s*.049,.01,.311),(s*.061,-.014,.18),.030,.024),
            capsule('Shin',(s*.061,-.014,.18),(s*.055,-.010,.05),.019,.011),
            ellipsoid('Knee',(s*.061,-.014,.18),(.025,.025,.027)),
            ellipsoid('Foot',(s*.058,-.028,.034),(.020,.038,.019))])
    bpy.ops.object.select_all(action='DESELECT')
    for o in items:o.select_set(True)
    bpy.context.view_layer.objects.active=items[0];bpy.ops.object.join();o=bpy.context.object;o.name='Body_UnderClothes'
    bpy.ops.object.transform_apply(location=True,rotation=True,scale=True)
    o.data.remesh_voxel_size=.0045;bpy.ops.object.voxel_remesh()
    mod=o.modifiers.new('Smooth reconstructed skin','SMOOTH');mod.factor=.7;mod.iterations=6;bpy.ops.object.modifier_apply(modifier=mod.name)
    o.data.materials.clear();o.data.materials.append(FUR)
    for p in o.data.polygons:p.use_smooth=True
    bind(o,[nearest_weights(v.co) for v in o.data.vertices]);o['part_type']='inferred_body'
    o['provenance']='Reconstructed under clothing; source contains no hidden skin.'
    return o

def clean_face_material():
    def linear(hexcolor):
        values=[int(hexcolor[i:i+2],16)/255 for i in (1,3,5)]
        return tuple(v/12.92 if v<=.04045 else ((v+.055)/1.055)**2.4 for v in values)
    colors={'Head':('Fur blue','#4b9fb7'),'Face_Cream':('Cream face','#fff0c1'),
        'EyeShell':('Eye white','#fffbee'),'Iris':('Original iris support','#fffbee'),
        'Brow':('Navy brows','#182d5e'),'Lash':('Navy lashes','#182d5e'),
        'Nose':('Nose','#781729'),'InnerEar':('Inner ears','#e6a6b6')}
    mats={key:material(name,linear(color),.8) for key,(name,color) in colors.items()}
    for o in bpy.context.scene.objects:
        if o.type!='MESH':continue
        key=o.name if o.name in colors else o.name.split('_')[0]
        if key in mats:o.data.materials.clear();o.data.materials.append(mats[key])
    REPORT['material_changes']=['Actual sculpted face components separated at geometric creases before material assignment.',
        'Face, nose, sclera, iris, brows, lashes and inner ears use independent solid materials; no source color or normal atlas on these parts.']

def attachment_loop(objects):
    # Analyze welded positions, not split UV/normal vertex indices.
    points=[];lookup={};counts={}
    for obj in objects:
        remap={}
        for v in obj.data.vertices:
            key=tuple(round(float(t),6) for t in v.co)
            if key not in lookup:lookup[key]=len(points);points.append(tuple(v.co))
            remap[v.index]=lookup[key]
        for face in obj.data.polygons:
            ids=[remap[i] for i in face.vertices]
            for a,b in zip(ids,ids[1:]+ids[:1]):
                e=tuple(sorted((a,b)));counts[e]=counts.get(e,0)+1
    edges={e for e,n in counts.items() if n==1};loops=[]
    while edges:
        a,b=edges.pop();loop=[a,b]
        while loop[-1]!=loop[0]:
            candidate=next((e for e in edges if loop[-1] in e),None)
            if candidate is None:break
            edges.remove(candidate);loop.append(candidate[1] if candidate[0]==loop[-1] else candidate[0])
        if loop[-1]==loop[0]:loops.append(loop[:-1])
    return np.asarray(points)[max(loops,key=len)]

def lash_beds():
    from mathutils.geometry import tessellate_polygon
    for side in ['L','R']:
        source=bpy.data.objects['Lash_'+side];rim=attachment_loop([source])
        flat=[Vector((p[0],p[2],0)) for p in rim]
        lookup={tuple(v):i for i,v in enumerate(flat)}
        faces=[tuple(v if isinstance(v,int) else lookup[tuple(v)] for v in tri) for tri in tessellate_polygon([flat])]
        o=mesh_object('LashBed_'+side,rim,faces,bpy.data.materials['Cream face'])
        # Smooth depth over the real concave footprint; no convex hull and no
        # faceted harmonic interpolation across skinny triangulation diagonals.
        x=rim[:,0];z=rim[:,2]-.72
        matrix=np.stack([np.ones(len(x)),x,z,x*x,x*z,z*z],axis=1)
        coefficients=np.linalg.lstsq(matrix,rim[:,1],rcond=None)[0]
        bm=bmesh.new();bm.from_mesh(o.data)
        bmesh.ops.subdivide_edges(bm,edges=list(bm.edges),cuts=3,use_grid_fill=True)
        for v in bm.verts:
            x,y,z=v.co;z-=.72
            v.co.y=float(np.dot([1,x,z,x*x,x*z,z*z],coefficients))+.001
        bmesh.ops.recalc_face_normals(bm,faces=list(bm.faces));bm.to_mesh(o.data);bm.free()
        for face in o.data.polygons:face.use_smooth=True
        bind(o);o['part_type']='face';o['provenance']='Skin filling the true welded lash attachment footprint, fitted to its boundary depth.'
    REPORT['inferred_geometry'].append('Skin beneath lashes, bounded by actual welded attachment contours.')

def eye_details():
    ruby=material('Rounded ruby iris',(.38,.025,.052),.65)
    dark=material('Deep burgundy pupils',(.015,.0025,.006),.75)
    white=material('Eye catchlights',(.95,.95,.93),.8)
    def patch(name,cx,cz,rx,rz,mat,offset):
        def ocular_y(x,z):
            a=(abs(x)-.07)/.04;b=(z-.69)/.04
            return float(np.dot([1,a,b,a*a,a*b,b*b],ocular_coeff))-.002
        N=128 if rx>.02 else 96 if rx>.01 else 48; rings=18 if rx>.02 else 10 if rx>.01 else 4
        verts=[(cx,ocular_y(cx,cz)-offset,cz)];faces=[]
        for r in np.linspace(1/rings,1,rings):
            for i in range(N):
                a=i*2*math.pi/N;x=cx+rx*r*math.cos(a);z=cz+rz*r*math.sin(a)
                verts.append((x,ocular_y(x,z)-offset,z))
        for i in range(N):faces.append((0,1+i,1+(i+1)%N))
        for ring in range(rings-1):
            for i in range(N):
                a=1+ring*N+i;b=1+ring*N+(i+1)%N;faces.append((a,b,b+N,a+N))
        o=mesh_object(name,verts,faces,mat);bind(o);o['part_type']='eye'
        o['provenance']='Authored surface detail fitted to the separated source iris.'
    for sign,side in [(1,'L'),(-1,'R')]:
        samples=np.concatenate([np.array([v.co[:] for v in bpy.data.objects[n+'_'+side].data.vertices]) for n in ['EyeShell','Iris']])
        a=(abs(samples[:,0])-.07)/.04;b=(samples[:,2]-.69)/.04
        ocular_coeff=np.linalg.lstsq(np.stack([np.ones(len(a)),a,b,a*a,a*b,b*b],axis=1),samples[:,1],rcond=None)[0]
        cx=sign*.0615;cz=.693
        patch('RoundIris_'+side,cx,cz-.001,.0228,.025,ruby,.0014)
        patch('Pupil_'+side,cx,cz,.0135,.0175,dark,.0019)
        patch('Catchlight_'+side,cx-sign*.0055,cz+.009,.0042,.0056,white,.0023)
        patch('CatchlightSmall_'+side,cx+sign*.0035,cz+.013,.0016,.0021,white,.0023)
    REPORT['inferred_geometry'].append('Pupils and catchlights fitted to the retained non-spherical iris surfaces.')

def face_controls():
    for o in list(bpy.context.scene.objects):
        if o.type=='MESH' and (o.name=='Head' or o.get('part_type') in {'face','eye'}):face_controls_part(o)

def face_controls_part(o):
    v=np.array([p.co[:] for p in o.data.vertices]);x,y,z=v.T
    front=1-smoothstep(-.075,-.035,y)
    for sign,side in [(1,'L'),(-1,'R')]:
        cx=sign*.064;cz=.691
        u=(x-cx)/.039;h=(z-cz)/.036
        radius=np.sqrt(u*u+h*h)
        socket=smoothstep(.026,.041,abs(x))*(1-smoothstep(.110,.133,abs(x)))*smoothstep(.640,.663,z)*(1-smoothstep(.732,.752,z))*front
        socket*=smoothstep(0,.018,sign*x)
        # The ocular surface stays round through a blink. The upper lid and
        # original source lash are driven together in curved_lids().
        if o.name.startswith(('RoundIris_','Pupil_','Catchlight_','CatchlightSmall_')) and o.name.endswith('_'+side):
            for name,dx,dz in [('eyeLookIn',-sign*.004,0),('eyeLookOut',sign*.004,0),('eyeLookUp',0,.004),('eyeLookDown',0,-.004)]:
                vv=v.copy();vv[:,0]+=dx;vv[:,2]+=dz
                vv[:,1]+=np.array([surface_y(a+dx,b+dz)-surface_y(a,b) for a,b in zip(x,z)])
                add_key(o,name+side,vv)
    mouth=np.exp(-2*((x/.039)**2+((z-.630)/.024)**2))*front
    corners=np.minimum(1,(np.abs(x)/.022)**1.5)
    vv=v.copy();vv[:,2]+=.006*mouth*corners;vv[:,0]+=np.sign(x)*.003*mouth;add_key(o,'mouthSmile',vv)
    vv=v.copy();vv[:,2]-=.004*mouth*corners;add_key(o,'mouthFrown',vv)
    vv=v.copy();vv[:,0]*=1-.12*mouth;vv[:,1]-=.004*mouth;add_key(o,'mouthPucker',vv)
    lower=(1-smoothstep(.625,.642,z))*np.exp(-2*((x/.059)**2+((z-.62)/.040)**2))*front
    vv=v.copy();vv[:,2]-=.009*lower;vv[:,1]+=.002*lower;add_key(o,'jawDrop',vv)
    if o.name!='Head':return
    REPORT['limitations'].insert(0,'Jaw drop is a closed-mouth deformation; an oral cavity and production lip-sync topology are still required.')
    REPORT['limitations'].insert(1,'Gaze is a conservative curved-surface deformation. Extreme expression combinations still need art review.')

def curved_lids():
    blue=material('Blue teal eyelids',(.09,.36,.50),.85)
    cream=bpy.data.materials['Cream face']
    for sign,side in [(1,'L'),(-1,'R')]:
        lash=bpy.data.objects['Lash_'+side]
        source=np.array([v.co[:] for v in lash.data.vertices]);x,y,z=source.T
        eye_objects=[bpy.data.objects[n+'_'+side] for n in ['EyeShell','Iris']]
        ocular=np.concatenate([np.array([v.co[:] for v in o.data.vertices]) for o in eye_objects])
        xs=np.linspace(.035,.126,48);depths=[]
        for xx in xs:
            near=ocular[abs(abs(ocular[:,0])-xx)<.008]
            depths.append(float(near[:,1].min())-.0035 if len(near) else surface_y(sign*xx,.700)-.0035)
        depthfit=np.polynomial.Polynomial.fit(xs,depths,3)
        eye_rim=attachment_loop(eye_objects);eye_rim[:,0]*=sign
        xmin=float(eye_rim[:,0].min());xmax=float(eye_rim[:,0].max())
        def aperture(xx):
            xx=float(np.clip(xx,xmin+1e-6,xmax-1e-6));hits=[]
            for a,b in zip(eye_rim,np.roll(eye_rim,-1,axis=0)):
                if min(a[0],b[0])<=xx<=max(a[0],b[0]) and abs(a[0]-b[0])>1e-9:hits.append(float(a[2]+(b[2]-a[2])*(xx-a[0])/(b[0]-a[0])))
            return min(hits),max(hits)
        gx=np.linspace(xmin+1e-6,xmax-1e-6,160)
        gz=np.array([aperture(xx)[1] for xx in gx])
        # Smooth only the guide, not the source lash. Preserve its mesh and all
        # residual offsets in an eye-rim tangent frame instead of scaling Z.
        for _ in range(3):gz[1:-1]=(gz[:-2]+2*gz[1:-1]+gz[2:])/4
        start=np.stack([gx,gz],axis=1)
        u=np.linspace(0,1,len(gx));finish=np.stack([gx,np.array([aperture(xx)[0] for xx in gx])-.0008],axis=1)
        tangent=np.gradient(start,axis=0);tangent/=np.linalg.norm(tangent,axis=1)[:,None]
        target_tangent=np.gradient(finish,axis=0);target_tangent/=np.linalg.norm(target_tangent,axis=1)[:,None]
        normals=np.stack([-tangent[:,1],tangent[:,0]],axis=1)
        target_normals=np.stack([-target_tangent[:,1],target_tangent[:,0]],axis=1)
        def moving(points):
            result=np.array(points).copy()
            for i,p in enumerate(points):
                point=np.array([abs(p[0]),p[2]])
                k=int(np.argmin(np.sum((start-point)**2,axis=1)))
                residual=point-start[k];along=float(residual@tangent[k]);out=float(residual@normals[k])
                # Lash roll gives the sculpted tips depth as the rim closes.
                # Their original mesh is transported with the guide frame.
                angle=.90;xz=finish[k]+along*target_tangent[k]+out*math.cos(angle)*target_normals[k]
                xz[0]+=max(0,out)*.35*smoothstep(.35,.9,k/(len(gx)-1))
                result[i,0]=sign*xz[0];result[i,2]=xz[1]
                depth_offset=p[1]-surface_y(sign*start[k,0],start[k,1])
                result[i,1]=float(depthfit(xz[0]))+depth_offset*.65-out*math.sin(angle)
            return result
        target=moving(source)
        add_key(lash,'eyeBlink'+side,target);add_key(lash,'eyeSquint'+side,source*.55+target*.45)
        lash['eye_model']='Original source eyelash geometry; its own vertices move with blink. No replacement lashes.'
        rim=attachment_loop([lash]);a=int(np.argmin(abs(rim[:,0])));b=int(np.argmax(abs(rim[:,0])))
        if a>b:a,b=b,a
        paths=[rim[a:b+1],np.concatenate([rim[b:],rim[:a+1]])]
        rim=max(paths,key=lambda path:float(path[:,2].mean()))
        moved=moving(rim);N=len(rim);V=14
        # Sweep the ACTUAL source lash attachment edge. The final blue row is
        # position-identical to that moving edge, so the two surfaces cannot gap.
        base=[];closed=[];faces=[]
        for j in range(V+1):
            t=j/V
            for p,q in zip(rim,moved):
                a=p.copy();a[1]+=.0008*(1-t);base.append(a)
                closed.append(p*(1-t)+q*t)
        for j in range(V):
            for i in range(N-1):
                a=j*N+i;b=a+1;faces.append((a,b,b+N,a+N))
        # Relax the skin between its fixed attachment and moving lash rows.
        # The source rim has irregular triangles; copying each column literally
        # introduces vertical folds. Both attachment rows stay position-exact.
        grid=np.asarray(closed).reshape(V+1,N,3)
        for _ in range(80):
            relaxed=grid.copy()
            relaxed[1:-1,1:-1]=.5*grid[1:-1,1:-1]+.125*(grid[:-2,1:-1]+grid[2:,1:-1]+grid[1:-1,:-2]+grid[1:-1,2:])
            grid=relaxed
        eye_vertices=[];eye_faces=[]
        for prefix in ['EyeShell','Iris','RoundIris','Pupil','Catchlight','CatchlightSmall']:
            obj=bpy.data.objects[prefix+'_'+side];offset=len(eye_vertices)
            eye_vertices.extend(v.co.copy() for v in obj.data.vertices)
            eye_faces.extend([offset+i for i in face.vertices] for face in obj.data.polygons)
        eye_surface=BVHTree.FromPolygons(eye_vertices,eye_faces)
        for row in grid[1:-1]:
            for point in row[1:-1]:
                hit=eye_surface.ray_cast(Vector((point[0],-1,point[2])),Vector((0,1,0)))[0]
                if hit is not None:point[1]=min(point[1],hit.y-.002)
        closed=grid.reshape(-1,3)
        lid=mesh_object('UpperLid_'+side,closed,faces,blue)
        lid.data.vertices.foreach_set('co',np.asarray(base,dtype=np.float32).ravel());lid.data.update()
        bind(lid);lid['part_type']='face';lid['source_lash_boundary_count']=N
        add_key(lid,'eyeBlink'+side,closed);add_key(lid,'eyeSquint'+side,np.array(base)*.55+np.array(closed)*.45)
    REPORT['inferred_geometry'].append('Blue eyelid skin swept from the original moving lash attachment boundary; original lashes retained as the animated components.')

def improve_rig():
    bpy.ops.object.select_all(action='DESELECT');RIG.select_set(True);bpy.context.view_layer.objects.active=RIG
    bpy.ops.object.mode_set(mode='EDIT')
    for s,side in [(1,'l'),(-1,'r')]:
        b=RIG.data.edit_bones.new('ear_'+side);b.head=RIG.matrix_world.inverted()@Vector((s*.073,.022,.85));b.tail=RIG.matrix_world.inverted()@Vector((s*.096,.025,1.09));b.parent=RIG.data.edit_bones['head_x']
    b=RIG.data.edit_bones.new('tail_x');b.head=RIG.matrix_world.inverted()@Vector((0,.088,.365));b.tail=RIG.matrix_world.inverted()@Vector((0,.135,.365));b.parent=RIG.data.edit_bones['root_x']
    bpy.ops.object.mode_set(mode='OBJECT')
    corrected=0
    for o in [x for x in bpy.context.scene.objects if x.type=='MESH']:
        for v in o.data.vertices:
            x,y,z=v.co
            if (o.name=='Head' or o.name.startswith('InnerEar_')) and z>.82:
                w=float(smoothstep(.82,.90,z));ws=[('head_x',1-w),('ear_l' if x>0 else 'ear_r',w)];corrected+=1
            elif o.name=='Tail':ws=[('tail_x',1)]
            elif (o.name=='Head' or o.get('part_type') in {'face','eye'}) and z>.60:ws=[('head_x',1)]
            else:
                ws=[(o.vertex_groups[g.group].name,g.weight) for g in v.groups]
                ws=sorted(ws,key=lambda p:p[1],reverse=True)[:4];total=sum(w for n,w in ws)
                ws=[(n,w/total) for n,w in ws] if total else [('root_x',1)]
            for gi in [g.group for g in v.groups]:o.vertex_groups[gi].remove([v.index])
            for n,w in ws:
                if w<1e-6:continue
                g=o.vertex_groups.get(n) or o.vertex_groups.new(name=n);g.add([v.index],float(w),'REPLACE')
    REPORT['rig_improvements']={'ear_vertices_rebound':corrected,'new_bones':['ear_l','ear_r','tail_x'],'maximum_weights':4,'weight_normalization':True,'head_rigid_above_neck':True}

def proportions():
    for o in [x for x in bpy.context.scene.objects if x.type=='MESH']:
        v=np.array([p.co[:] for p in o.data.vertices]);x,y,z=v.T
        # Rest-space deformations are shared across skin, eyes and clothes.
        w=smoothstep(.56,.62,z);vv=v.copy();vv[:,0]*=1+.12*w;vv[:,1]*=1+.08*w;add_key(o,'headWidth',vv)
        w=(1-smoothstep(.43,.58,z))*smoothstep(.26,.38,z);vv=v.copy();vv[:,0]*=1+.12*w;vv[:,1]*=1+.12*w;add_key(o,'bodyWidth',vv)
        w=smoothstep(.83,.91,z);vv=v.copy();vv[:,2]+=.065*w;add_key(o,'earLength',vv)
        w=np.exp(-((x/.082)**2+((z-.642)/.05)**2)*2)*(1-smoothstep(-.09,-.02,y));vv=v.copy();vv[:,1]-=.014*w;add_key(o,'muzzleLength',vv)
        w=np.exp(-((x/.09)**2+((z-.70)/.09)**2)*2)*(1-smoothstep(-.07,.0,y));vv=v.copy();vv[:,0]*=1+.075*w;add_key(o,'faceWidth',vv)
        if o['part_type']=='clothing':
            normals=np.array([p.normal[:] for p in o.data.vertices]);add_key(o,'clothingEase',v+normals*.004)
        if o.name=='Head' or o.get('part_type') in {'face','eye'}:
            w=sum((1-smoothstep(1,1.7,np.sqrt(((x-s*.064)/.041)**2+((z-.691)/.040)**2))) for s in [-1,1])*(1-smoothstep(-.07,-.03,y))
            vv=v.copy();vv[:,2]+=(z-.691)*.15*w;add_key(o,'eyeSize',vv)
            w=np.exp(-2*(((abs(x)-.073)/.033)**2+((z-.643)/.031)**2))*(1-smoothstep(-.07,-.02,y))
            vv=v.copy();vv[:,0]+=np.sign(x)*.004*w;vv[:,1]-=.004*w;add_key(o,'cheekFullness',vv)
            for s,side in [(1,'L'),(-1,'R')]:
                w=np.exp(-(((x-s*.067)/.037)**2+((z-.754)/.022)**2)*2)*(1-smoothstep(-.06,0,y));vv=v.copy();vv[:,2]+=.009*w;add_key(o,'browUp'+side,vv)
                vv=v.copy();vv[:,2]-=.007*w;add_key(o,'browDown'+side,vv)

def validate_and_save():
    meshes=[o for o in bpy.context.scene.objects if o.type=='MESH'];bad=0;errors=[];key_counts={}
    for o in meshes:
        for v in o.data.vertices:
            total=sum(g.weight for g in v.groups)
            if abs(total-1)>1e-4 or len(v.groups)>4:bad+=1
        if o.data.shape_keys:
            for k in o.data.shape_keys.key_blocks:
                coords=np.array([v.co[:] for v in k.data]);
                if not np.isfinite(coords).all():errors.append(o.name+': '+k.name+' non-finite')
            key_counts[o.name]=len(o.data.shape_keys.key_blocks)-1
    REPORT['validation']={'invalid_skin_vertices':bad,'errors':errors,'triangles':sum(sum(len(p.vertices)-2 for p in o.data.polygons) for o in meshes),'mesh_count':len(meshes),'bones':len(RIG.data.bones),'shape_keys':key_counts}
    if bad or errors:raise RuntimeError('Asset validation failed: '+json.dumps(REPORT['validation']))
    for o in meshes:
        if o.data.shape_keys:
            for k in o.data.shape_keys.key_blocks:k.value=0
            o.data.shape_keys.update_tag()
    bpy.context.scene.frame_set(1);bpy.context.view_layer.update()
    # Never overwrite a content-addressed archived revision through its symlink.
    # New builds write fresh output files for the shared storage registrar.
    for name in ['landau_character.blend','landau_character.glb','asset_report.json']:
        path=OUT/name
        if path.is_symlink():path.unlink()
    # Preserve a compact editable master with embedded textures.
    bpy.ops.file.pack_all()
    bpy.ops.wm.save_as_mainfile(filepath=str(OUT/'landau_character.blend'))
    bpy.ops.object.select_all(action='DESELECT');RIG.select_set(True)
    for o in meshes:
        o.hide_set(False);o.select_set(True)
    bpy.ops.export_scene.gltf(filepath=str(OUT/'landau_character.glb'),export_format='GLB',use_selection=True,export_animations=False,export_morph=True,export_skins=True,export_extras=True,export_image_format='AUTO')
    for o in meshes:
        if o.get('default_hidden'):o.hide_set(True)
    REPORT['glb_sha256']=hashlib.sha256((OUT/'landau_character.glb').read_bytes()).hexdigest()
    REPORT['default_hidden']=[o.name for o in meshes if o.get('default_hidden')]
    REPORT['controls']=sorted({k.name for o in meshes if o.data.shape_keys for k in o.data.shape_keys.key_blocks if k.name!='Basis'})
    (OUT/'asset_report.json').write_text(json.dumps(REPORT,indent=2))
    print(json.dumps(REPORT['validation']))

def main():
    global SOURCE_OBJ,RIG,SOURCE_TREE,WEIGHTS,BVH,FUR,SOURCE_MAP,FACE_REGIONS
    for obj in list(bpy.context.scene.objects):bpy.data.objects.remove(obj,do_unlink=True)
    bpy.data.orphans_purge(do_recursive=True)
    bpy.ops.wm.usd_import(filepath=str(SOURCE))
    SOURCE_OBJ=next(o for o in bpy.context.scene.objects if o.type=='MESH')
    RIG=next(o for o in bpy.context.scene.objects if o.type=='ARMATURE');RIG.name='Landau_Rig'
    SOURCE_TREE=KDTree(len(SOURCE_OBJ.data.vertices))
    for v in SOURCE_OBJ.data.vertices:SOURCE_TREE.insert(v.co,v.index)
    SOURCE_TREE.balance()
    WEIGHTS=[[(SOURCE_OBJ.vertex_groups[g.group].name,g.weight) for g in v.groups] for v in SOURCE_OBJ.data.vertices]
    BVH=BVHTree.FromPolygons([v.co for v in SOURCE_OBJ.data.vertices],[list(p.vertices) for p in SOURCE_OBJ.data.polygons])
    SOURCE_MAP={}
    segmenter=runpy.run_path(str(ROOT/'segment_face_geometry.py'))
    FACE_REGIONS,geometry_report=segmenter['component_regions'](SOURCE_OBJ)
    REPORT['geometric_components']=geometry_report
    FUR=material('Reconstructed teal fur',(.16,.44,.48))
    segment()
    o=underbody();o.hide_render=True;o.hide_set(True);o['default_hidden']=True
    clean_face_material();lash_beds();eye_details();face_controls();curved_lids()
    REPORT['neutral_preservation']={'source_faces':len(SOURCE_OBJ.data.polygons),'retained_faces':sum(len(bpy.data.objects[n].data.polygons) for n in SOURCE_MAP),'max_position_error':max(float(np.max(np.abs(np.array([v.co[:] for v in bpy.data.objects[n].data.vertices])-np.array([SOURCE_OBJ.data.vertices[i].co[:] for i in ids])))) for n,ids in SOURCE_MAP.items()),'corner_uvs_preserved':True,'corner_normals_preserved':True}
    bpy.data.objects.remove(SOURCE_OBJ,do_unlink=True)
    improve_rig();proportions();validate_and_save()

if __name__=='__main__':main()
