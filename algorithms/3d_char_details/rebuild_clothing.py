"""Segment original clothing without fitting or reshaping it.
Accepted skin/face data and original clothing design are preserved.
"""
from pathlib import Path
import hashlib
import json
import runpy
import bpy
import numpy as np
from mathutils import Matrix, Vector
from mathutils.kdtree import KDTree

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'outputs/landau_v10'
GARMENTS=('Vest','Sleeve_L','Sleeve_R','Cuff_L','Cuff_R','Trousers','Boot_L','Boot_R')
helpers=runpy.run_path(str(ROOT/'rebuild_body.py'))


def protected_hashes(scene):
    result={}
    for o in scene.objects:
        if o.type!='MESH' or o.name in GARMENTS:continue
        h=hashlib.sha256()
        for values in [[v.co[:] for v in o.data.vertices], [list(p.vertices) for p in o.data.polygons],
            [list(n.vector) for n in o.data.corner_normals],
            [[(g.group,g.weight) for g in v.groups] for v in o.data.vertices],
            [[k.name,[v.co[:] for v in k.data]] for k in o.data.shape_keys.key_blocks] if o.data.shape_keys else [],
            [[x.color[:] for x in a.data] for a in o.data.color_attributes],
            [m.name for m in o.data.materials]]:
            h.update(json.dumps(values).encode())
        result[o.name]=h.hexdigest()
    return result


def build(source, target, scene, templates):
    """Replace only semantic partitions/materials; do not fit or reshape cloth."""
    bpy.context.window.scene=scene
    before=protected_hashes(scene)
    placements=json.loads((OUT/'body_build.json').read_text())['garment_rigid_placement']
    segmenter=runpy.run_path(str(ROOT/'segment_clothing.py'))
    labels,components,report=segmenter['segment'](source)
    regions=segmenter['material_regions'](source,labels,components)
    vv=np.array([v.co[:] for v in source.data.vertices])
    source_weights=[{source.vertex_groups[g.group].name:g.weight for g in v.groups} for v in source.data.vertices]
    mats={n:helpers['palette']('Garment · '+n,c,r) for n,c,r in [
        ('jade fabric','#397f74',.78),('pale fabric','#a4c0c5',.83),
        ('navy leather','#243954',.5),('pale piping','#bed0d1',.67),('belt','#35515d',.62)]}
    objects={};source_ids={};audit={}
    for name in GARMENTS:
        old=templates[name]
        chosen=[p for p,label in zip(source.data.polygons,labels) if label==name]
        ids=sorted({i for p in chosen for i in p.vertices});remap={v:i for i,v in enumerate(ids)}
        mesh=bpy.data.meshes.new(name+' · sculpt seams')
        mesh.from_pydata(vv[ids].tolist(),[],[[remap[i] for i in p.vertices] for p in chosen]);mesh.update()
        loops=[i for p in chosen for i in p.loop_indices]
        for layer in source.data.uv_layers:
            dest=mesh.uv_layers.new(name=layer.name)
            dest.data.foreach_set('uv',np.array([layer.data[i].uv[:] for i in loops],dtype=np.float32).ravel())
        for m in mats.values():mesh.materials.append(m)
        mn=list(mats)
        for p,orig in zip(mesh.polygons,chosen):
            component=int(components[orig.index])
            if name=='Vest':
                label='jade fabric'
                if component in [36509,41266] or orig.index in regions['placket'] or orig.index in regions['collar']:label='pale piping'
                if component==38378:label='belt'
            elif name.startswith('Boot'):
                label='navy leather' if component in [9947,9360,9972,8728] else 'pale piping'
            elif name.startswith('Cuff'):label='navy leather'
            else:label='pale fabric'
            p.material_index=mn.index(label);p.use_smooth=orig.use_smooth
        mesh.normals_split_custom_set([source.data.corner_normals[i].vector[:] for i in loops])
        obj=bpy.data.objects.new(name+'__new',mesh);scene.collection.objects.link(obj)
        # Unlinked library objects have unevaluated matrix_world values. The
        # accepted integration report records their exact original translation.
        obj.matrix_world=Matrix.Translation(Vector(placements[name]))
        for k in old.keys():obj[k]=old[k]
        obj['segmentation']='Reviewed sculpt crease connectivity; not bone ownership'
        tree=KDTree(len(old.data.vertices))
        for v in old.data.vertices:tree.insert(v.co,v.index)
        tree.balance();near=[tree.find(Vector(vv[i]))[1] for i in ids]
        if old.data.shape_keys:
            for key in old.data.shape_keys.key_blocks:
                dest=obj.shape_key_add(name=key.name);dest.slider_min=key.slider_min;dest.slider_max=key.slider_max;dest.value=0
                for i,(si,oi) in enumerate(zip(ids,near)):
                    dest.data[i].co=Vector(vv[si])+key.data[oi].co-old.data.shape_keys.key_blocks[0].data[oi].co
        else:obj.shape_key_add(name='Basis')
        # Preserve the full source skin field after geometry-only segmentation.
        # Every garment may use multiple joints; no bone defines its ownership.
        weights=[source_weights[i] for i in ids]
        helpers['bind'](obj,target,weights)
        obj['default_hidden']=True;obj.hide_render=True;obj.hide_set(True)
        audit[name]=dict(source_faces=len(chosen),source_vertices=len(ids),
            influencing_bones=sorted({n for w in weights for n,v in w.items() if v>1e-5}),
            basis_max_source_error=max((v.co-Vector(vv[ids[v.index]])).length for v in mesh.vertices),
            original_rigid_placement=list(obj.location),
            legacy_material_names=[m.name for m in old.data.materials])
        current=bpy.data.objects.get(name)
        if current:bpy.data.objects.remove(current,do_unlink=True)
        obj.name=name;objects[name]=obj;source_ids[name]=ids
    reverse={}
    for name,ids in source_ids.items():
        for i,si in enumerate(ids):reverse.setdefault(tuple(np.round(vv[si],6)),[]).append((name,i))
    seams=[refs for refs in reverse.values() if len({n for n,i in refs})>1]
    # Correspondence is retained for inspection, without enforcing fit or welding.
    for name,o in objects.items():
        o['seam_vertices']=[dict(local=i,partners=[dict(part=n2,vertex=j) for n2,j in refs if n2!=name]) for refs in seams for n,i in refs if n==name]
    after=protected_hashes(scene)
    assert before==after,'Accepted skin or face changed'
    report['material_regions']={n:sorted(ids) for n,ids in regions.items()}
    report.update(garments=audit,shared_source_seam_samples=len(seams),
        protected_before=before,protected_after=after,
        fit_mode='Manual only. No automatic fit, reshaping, collision projection or cloth correction.',
        preserved_source='Original sculpt surfaces, corner UVs/normals, source skin weights and original rigid placements. Geometry-segmented clean palette.')
    (OUT/'clothing_rebuild.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({'garments':audit,'source_seams':len(seams)}))
    return objects,seams,report
