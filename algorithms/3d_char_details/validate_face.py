"""Measure the repaired facial meshes in Blender, without changing the scene.

Usage: blender --background <candidate-or-master.blend> --python <this-file>
The source checkpoint supplies actual original lash geometry. The report covers
geometry and attachment; it does not certify likeness or production lip sync.
"""
from collections import Counter
from pathlib import Path
import hashlib
import json
import math
import bpy
import bmesh
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree
from mathutils.kdtree import KDTree

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'outputs/landau_v10'
STATES=(0,.25,.5,.75,1)


def points(obj, values=None, world=False):
    keys=obj.data.shape_keys
    basis=np.array([v.co[:] for v in (keys.key_blocks['Basis'].data if keys else obj.data.vertices)])
    result=basis.copy()
    for name,value in (values or {}).items():
        if keys and name in keys.key_blocks:
            key=keys.key_blocks[name]
            relative=np.array([v.co[:] for v in key.relative_key.data])
            result+=value*(np.array([v.co[:] for v in key.data])-relative)
    assert np.isfinite(result).all(), obj.name+' has non-finite evaluated coordinates'
    if world:
        matrix=np.array(obj.matrix_world)
        result=result@matrix[:3,:3].T+matrix[:3,3]
    return result


def blink_values(side,value):
    value=max(0,min(1,value))
    return {'eyeBlink'+side:value,'_blinkArc'+side:4*value*(1-value)}


def edge_counts(obj):
    counts=Counter()
    for face in obj.data.polygons:
        ids=list(face.vertices)
        for a,b in zip(ids,ids[1:]+ids[:1]):counts[tuple(sorted((a,b)))]+=1
    return counts


def boundary_loop(obj):
    adjacency={}
    for (a,b),count in edge_counts(obj).items():
        if count==1:
            adjacency.setdefault(a,[]).append(b);adjacency.setdefault(b,[]).append(a)
    assert adjacency and all(len(v)==2 for v in adjacency.values()), obj.name+' boundary is not a simple ring'
    loop=[min(adjacency)]
    while len(loop)<len(adjacency):
        nxt=[i for i in adjacency[loop[-1]] if i not in loop]
        assert nxt, obj.name+' has multiple boundary loops'
        loop.append(nxt[0])
    assert loop[0] in adjacency[loop[-1]]
    return loop


def nearest_indices(source,targets):
    kd=KDTree(len(source))
    for i,p in enumerate(source):kd.insert(Vector(p),i)
    kd.balance()
    return [(kd.find(Vector(p))[1],kd.find(Vector(p))[2]) for p in targets]


def mesh_tree(obj,vertices):
    return BVHTree.FromPolygons([Vector(p) for p in vertices],[list(f.vertices) for f in obj.data.polygons])


def run():
    scene=bpy.context.scene
    objects={o.name:o for o in scene.objects if o.type=='MESH'}
    evidence=json.loads((OUT/'facial_repair.json').read_text())
    coefficients=np.array(evidence['eyes']['surface_coefficients'])
    assert len(coefficients)==6 and np.isfinite(coefficients).all()
    def ey(x,z):
        u=(np.abs(x)-.07)/.04;v=(z-.69)/.04
        return np.array([np.ones_like(u),u,v,u*u,u*v,v*v]).T@coefficients
    eyes={};lashes={};joins={};mirror={}
    checkpoint=OUT/'checkpoints/pre_face_20260921/landau_character.blend'
    with bpy.data.libraries.load(str(checkpoint),link=False) as (source,target):
        assert {'Lash_L','Lash_R'}<=set(source.objects)
        target.objects=['Lash_L','Lash_R']
    originals=dict(zip(('L','R'),target.objects))
    try:
        for side in ('L','R'):
            shell=objects['EyeShell_'+side]
            bm=bmesh.new();bm.from_mesh(shell.data)
            topology={'vertices':len(bm.verts),'faces':len(bm.faces),
                      'boundary_edges':sum(e.is_boundary for e in bm.edges),
                      'nonmanifold_edges':sum(not e.is_manifold for e in bm.edges),
                      'wire_vertices':sum(not v.link_faces for v in bm.verts)}
            bm.free()
            assert topology['nonmanifold_edges']==topology['wire_vertices']==0, (side,topology)
            names=shell.data.shape_keys.key_blocks.keys() if shell.data.shape_keys else []
            assert not any(n.startswith(('eyeBlink','eyeSquint','_blinkArc')) for n in names)
            assert 'Iris_'+side not in objects, 'Raised original iris support remains'
            vertices=points(shell);bvh=mesh_tree(shell,vertices);details={};surface_errors=[]
            for prefix,pad in [('RoundIris',.000025),('Pupil',.000050),('Catchlight',.000075),('CatchlightSmall',.000075)]:
                obj=objects[prefix+'_'+side];v=points(obj)
                offset=ey(v[:,0],v[:,2])-v[:,1]
                error=float(np.max(abs(offset-pad)))
                assert error<2e-6, (obj.name,'analytic depth error',error)
                gaps=[]
                for p in v[::max(1,len(v)//160)]:
                    hit=bvh.ray_cast(Vector((p[0],-2,p[2])),Vector((0,1,0)))[0]
                    assert hit is not None, obj.name+' extends outside its eye'
                    gaps.append(float(hit.y-p[1]));surface_errors.append(abs(float(hit.y-ey(p[0],p[2]))))
                assert min(gaps)>-2e-5 and max(gaps)<.0002, (obj.name,'protrusion/penetration',min(gaps),max(gaps))
                details[prefix]={'analytic_offset_error':error,'surface_gap_min':min(gaps),'surface_gap_max':max(gaps)}
            assert max(surface_errors)<.00015, (side,'shell does not follow its shared surface')
            eyes[side]={'topology':topology,'stationary_during_blink':True,'details':details,
                        'max_sampled_shell_analytic_error':max(surface_errors)}
            lash=objects['Lash_'+side];old=originals[side]
            original=points(old);current=points(lash)
            assert original.shape==current.shape
            error=float(np.max(abs(original-current)))
            assert error==0, (side,'original lash neutral changed',error)
            faces=[list(p.vertices) for p in lash.data.polygons]
            assert faces==[list(p.vertices) for p in old.data.polygons], side+' lash topology changed'
            lashes[side]={'vertices':len(current),'polygons':len(faces),'neutral_position_error':error,'topology_exact':True}
            lid=objects['UpperLid_'+side];n=int(lid['source_lash_boundary_count'])
            mapping=nearest_indices(points(lash,world=True),points(lid,world=True)[-n:])
            assert max(d for _,d in mapping)<1e-6
            ids=[i for i,_ in mapping];samples=[]
            for value in STATES:
                settings=blink_values(side,value)
                a=points(lash,settings,True)[ids];b=points(lid,settings,True)[-n:]
                gap=float(np.max(np.linalg.norm(a-b,axis=1)))
                assert gap<1e-6, (side,value,'lash/lid gap',gap)
                samples.append({'blink':value,'arc_weight':4*value*(1-value),'max_join_gap':gap})
            joins[side]=samples
        for value in STATES:
            distances=[]
            for source_side,target_side in [('L','R'),('R','L')]:
                src=objects['Lash_'+source_side];dst=objects['Lash_'+target_side]
                mirrored=points(src,blink_values(source_side,value));mirrored[:,0]*=-1
                surface=mesh_tree(dst,points(dst,blink_values(target_side,value)))
                distances.extend(surface.find_nearest(Vector(p))[3] for p in mirrored)
            mirror[str(value)]={'symmetric_surface_distance_max':float(np.max(distances)),
                               'symmetric_surface_distance_p95':float(np.percentile(distances,95)),
                               'symmetric_surface_distance_mean':float(np.mean(distances))}
        distances=[]
        for source_side,target_side in [('L','R'),('R','L')]:
            mirrored=points(originals[source_side],{'eyeBlink'+source_side:1});mirrored[:,0]*=-1
            surface=mesh_tree(originals[target_side],points(originals[target_side],{'eyeBlink'+target_side:1}))
            distances.extend(surface.find_nearest(Vector(p))[3] for p in mirrored)
        previous={'max':float(np.max(distances)),'p95':float(np.percentile(distances,95)),'mean':float(np.mean(distances))}
    finally:
        for obj in originals.values():bpy.data.objects.remove(obj,do_unlink=True)
    body=objects['Body_Complete'];cavity=objects['Mouth_Interior'];rim=boundary_loop(cavity)
    base=points(cavity,world=True)[rim];mapping=nearest_indices(points(body,world=True),base)
    assert max(d for _,d in mapping)<1e-6, 'Mouth bag is detached from body lip'
    lip_ids=[i for i,_ in mapping];counts=edge_counts(body)
    assert len(set(lip_ids))==len(lip_ids), 'Mouth rim collapsed onto duplicate lip vertices'
    assert all(counts[tuple(sorted((a,b)))]==1 for a,b in zip(lip_ids,lip_ids[1:]+lip_ids[:1])), 'Body has no matching real mouth aperture'
    profile={};upper_rim_mask=None
    if evidence['mouth'].get('source_seam'):
        shift=np.array(objects['Lash_L'].matrix_world.translation)
        local_rim=base-shift
        expected=np.interp(local_rim[:,0],evidence['mouth']['crease_x'],evidence['mouth']['crease_z'])
        upper_rim_mask=local_rim[:,2]-expected>.000002
        trace_error=float(np.max(abs(local_rim[:,2]-expected)))
        assert trace_error<.00007, ('Mouth lost measured source crease',trace_error)
        current_tree=mesh_tree(body,points(body,world=True))
        with bpy.data.libraries.load(str(checkpoint),link=False) as (source,target):target.objects=['Body_Complete']
        original=target.objects[0]
        try:
            original_tree=mesh_tree(original,points(original,world=True))
            samples=[]
            for x in (0,.006,.012):
                row={'x':x}
                for label,z in [('upper',.636),('lower',.615)]:
                    start=Vector((x+shift[0],-2,z+shift[2]))
                    a=current_tree.ray_cast(start,Vector((0,1,0)))[0]
                    b=original_tree.ray_cast(start,Vector((0,1,0)))[0]
                    assert a is not None and b is not None
                    row[label+'_depth']=float(a.y);row[label+'_change_from_source']=float(a.y-b.y)
                assert row['lower_change_from_source']>.001, ('Lower muzzle did not recede',row)
                assert row['lower_depth']-row['upper_depth']>.007, ('Upper muzzle relief was flattened',row)
                samples.append(row)
            profile={'measured_seam_error':trace_error,'source_profile_samples':samples}
        finally:bpy.data.objects.remove(original,do_unlink=True)
    mouth=[]
    for value in (0,.5,1):
        values={'jawDrop':value};v=points(cavity,values,True);a=v[rim];b=points(body,values,True)[lip_ids]
        gap=float(np.max(np.linalg.norm(a-b,axis=1)))
        assert gap<1e-6, ('jaw',value,'cavity/lip gap',gap)
        if upper_rim_mask is not None:
            upper_drift=float(np.max(np.linalg.norm(a[upper_rim_mask]-base[upper_rim_mask],axis=1)))
            assert upper_drift<1e-7, ('Jaw opening flattened or moved the upper muzzle curve',value,upper_drift)
        area=abs(float(np.sum(a[:,0]*np.roll(a[:,2],-1)-np.roll(a[:,0],-1)*a[:,2])))/2
        assert math.isfinite(area) and area>1e-8, ('jaw',value,'invalid aperture',area)
        depth=float(v[:,1].max()-a[:,1].min())
        assert depth>.005, 'Mouth lacks recessed interior depth'
        mouth.append({'jawDrop':value,'projected_aperture_area':area,'aperture_height':float(np.ptp(a[:,2])),
                      'cavity_depth':depth,'max_lip_cavity_gap':gap,
                      'upper_rim_max_displacement':upper_drift if upper_rim_mask is not None else None})
    assert mouth[0]['projected_aperture_area']<mouth[1]['projected_aperture_area']<mouth[2]['projected_aperture_area']
    shape_states=[]
    body.data.calc_loop_triangles()
    rest=points(body,world=True)
    affected=np.linalg.norm(points(body,{'mouthLength':1},True)-rest,axis=1)>1e-10
    affected|=np.linalg.norm(points(body,{'mouthCurvature':1},True)-rest,axis=1)>1e-10
    triangles=np.array([t.vertices[:] for t in body.data.loop_triangles])
    triangles=triangles[np.any(affected[triangles],axis=1)]
    def projected_area(p):
        a=p[triangles[:,1]]-p[triangles[:,0]];b=p[triangles[:,2]]-p[triangles[:,0]]
        return a[:,0]*b[:,2]-a[:,2]*b[:,0]
    rest_area=projected_area(rest)
    assert np.all(rest_area>=-1e-12), 'Mouth patch folds in neutral'
    for jaw in (0,.5,1):
        for length in (-1,0,1):
            for curvature in (-1,0,1):
                values={'jawDrop':jaw,'mouthLength':length,'mouthCurvature':curvature}
                a=points(cavity,values,True)[rim];body_points=points(body,values,True);b=body_points[lip_ids]
                gap=float(np.max(np.linalg.norm(a-b,axis=1)))
                assert gap<1e-6, ('Mouth shape detached',values,gap)
                flips=int(np.count_nonzero(projected_area(body_points)<-1e-12))
                assert flips==0, ('Mouth shape folds',values,flips)
                shape_states.append(dict(values,max_lip_cavity_gap=gap,folded_triangles=flips,width=float(np.ptp(a[:,0]))))
    widths=[s['width'] for s in shape_states if s['jawDrop']==0 and s['mouthCurvature']==0]
    assert widths[0]<widths[1]<widths[2], ('Mouth length must visibly change width',widths)
    assert {'Teeth_Upper','Teeth_Lower','Tongue'}<=objects.keys()
    interior_motion={}
    for name in ('Teeth_Upper','Teeth_Lower','Tongue'):
        obj=objects[name];rest=points(obj,world=True)
        for value in (0,.5,1):points(obj,{'jawDrop':value},True)
        delta=points(obj,{'jawDrop':1},True)-rest
        interior_motion[name]={'max_displacement':float(np.max(np.linalg.norm(delta,axis=1))),
                               'mean_vertical_displacement':float(delta[:,2].mean())}
        if name=='Teeth_Upper':assert np.max(abs(delta))==0, 'Upper teeth move with the lower jaw'
        else:assert delta[:,2].mean()<-.001, name+' does not follow jaw opening'
    result={'status':'passed','asset':bpy.data.filepath,'source_checkpoint':str(checkpoint),
            'asset_sha256':hashlib.sha256(Path(bpy.data.filepath).read_bytes()).hexdigest(),
            'eyes':eyes,'original_lashes':lashes,'lid_lash_joins':joins,'bilateral_lash_residual':mirror,'previous_closed_lash_residual':previous,
            'mouth':{'boundary_vertices':len(rim),'real_body_aperture':True,'states':mouth,'shape_states':shape_states,'interior_motion':interior_motion,'profile':profile},
            'scope':'Measured mesh geometry and attachments. Nonzero mirror residuals retain source asymmetry; visual likeness and extreme expression combinations need art review.'}
    (OUT/'facial_validation.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2));return result


if __name__=='__main__':run()
