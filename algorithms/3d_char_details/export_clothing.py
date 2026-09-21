"""Publish the local segmented character and its supplied running action.

Writes fresh local files and never follows an archive symlink for output.
The original local 6446c39 snapshot and presets are not changed.
"""
from pathlib import Path
import hashlib
import json
import os
import struct
import runpy
import bpy
from mathutils import Matrix

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'outputs/landau_v10'


def preserve_facial_split_normals(path, mouth=None):
    """Keep authored jaw shading and transport smooth profile-control normals.

    Blender retains custom split normals on this sculpt when applying its jaw
    shape. Recomputed glTF morph normals otherwise introduce polygon creases at
    the original coarse muzzle boundary. Omit only this target's NORMAL delta
    on the body; profile targets use the smooth field Jacobians below.
    Missing morph NORMAL means a zero offset under the glTF specification.
    """
    data=path.read_bytes();size=struct.unpack_from('<I',data,12)[0]
    document=json.loads(data[20:20+size]);changed=0
    body_meshes={n['mesh'] for n in document['nodes'] if n.get('name')=='Body_Complete' and 'mesh' in n}
    for index,mesh in enumerate(document['meshes']):
        if index not in body_meshes:continue
        names=mesh.get('extras',{}).get('targetNames',[])
        if 'jawDrop' not in names:continue
        for primitive in mesh['primitives']:
            target=primitive['targets'][names.index('jawDrop')]
            if 'NORMAL' in target:del target['NORMAL'];changed+=1
    assert changed, 'Body jaw morph missing from exported asset'
    rest=data[20+size:]
    if mouth and mouth.get('surface_cleanup'):
        # Morph normals from the coarse source triangles reintroduce facets.
        # Transport the smooth authored normals by each profile field's
        # inverse-transpose Jacobian instead. This is a native glTF NORMAL
        # target, so the editor and exported files use the same smooth result.
        import numpy as np
        helper=runpy.run_path(str(ROOT/'repair_face.py'))
        binary=bytearray(rest[8:]);shift=np.array(bpy.data.objects['Lash_L'].matrix_world.translation)
        def read(index):
            a=document['accessors'][index];v=document['bufferViews'][a['bufferView']]
            assert a['componentType']==5126 and a['type']=='VEC3' and 'sparse' not in a
            offset=v.get('byteOffset',0)+a.get('byteOffset',0);stride=v.get('byteStride',12)
            return np.array([struct.unpack_from('<3f',binary,offset+i*stride) for i in range(a['count'])])
        def append(values):
            binary.extend(b'\0'*((-len(binary))%4));offset=len(binary)
            raw=np.asarray(values,dtype='<f4').tobytes();binary.extend(raw)
            view=len(document['bufferViews']);document['bufferViews'].append({'buffer':0,'byteOffset':offset,'byteLength':len(raw)})
            index=len(document['accessors']);document['accessors'].append({'bufferView':view,'componentType':5126,'count':len(values),'type':'VEC3'})
            return index
        def source(v):return np.column_stack([v[:,0],-v[:,2],v[:,1]])
        def gltf(v):return np.column_stack([v[:,0],v[:,2],-v[:,1]])
        def crease(x):return float(np.interp(x,mouth['crease_x'],mouth['crease_z']))
        fields={'muzzleLength':lambda p:-.014*helper['muzzle_influence'](p),
                'jawRecess':lambda p:.006*helper['jaw_influence'](p,crease)}
        for index in body_meshes:
            mesh=document['meshes'][index];names=mesh['extras']['targetNames']
            for primitive in mesh['primitives']:
                p=source(read(primitive['attributes']['POSITION']))-shift
                n=source(read(primitive['attributes']['NORMAL']))
                for name,field in fields.items():
                    target=primitive['targets'][names.index(name)]
                    step=.00001
                    grad=np.column_stack([(field(p+a)-field(p-a))/(2*step) for a in np.eye(3)*step])
                    ny=n[:,1]/(1+grad[:,1])
                    moved=np.column_stack([n[:,0]-grad[:,0]*ny,ny,n[:,2]-grad[:,2]*ny])
                    moved/=np.linalg.norm(moved,axis=1)[:,None]
                    target['NORMAL']=append(gltf(moved-n))
        document['buffers'][0]['byteLength']=len(binary)
        binary.extend(b'\0'*((-len(binary))%4))
        rest=struct.pack('<II',len(binary),0x004E4942)+bytes(binary)
    raw=json.dumps(document,separators=(',',':')).encode();raw+=b' '*((-len(raw))%4)
    path.write_bytes(struct.pack('<4sII',b'glTF',2,20+len(raw)+len(rest))+struct.pack('<II',len(raw),0x4E4F534A)+raw+rest)


def run(body_repair=None, facial_repair=None):
    scene=bpy.data.scenes['Scene'];bpy.context.window.scene=scene
    rig=bpy.data.objects['Landau_Rig']
    report=json.loads((OUT/'asset_report.json').read_text())
    clothing=json.loads((OUT/'clothing_rebuild.json').read_text())
    protected=runpy.run_path(str(ROOT/'rebuild_clothing.py'))['protected_hashes'](scene)
    baseline=clothing.get('protected_after_facial_repair',clothing.get('protected_after_underarm_repair',clothing['protected_before']))
    assert body_repair is None or facial_repair is None, 'Publish body and facial repairs separately'
    if facial_repair is not None:
        allowed={'Body_Complete','Mouth_Interior','Teeth_Upper','Teeth_Lower','Tongue'}|{prefix+'_'+side for prefix in (
            'EyeShell','Iris','RoundIris','Pupil','Catchlight','CatchlightSmall','Lash','UpperLid','LashBed') for side in ('L','R')}
        control_only=facial_repair.get('control_only_objects_before',{})
        if control_only:
            assert facial_repair.get('control_only_morphs')==['muzzleLength']
            assert set(control_only)<={'Nose','Brow_L','Brow_R','InnerEar_L','InnerEar_R'}
            current=runpy.run_path(str(ROOT/'rebuild_clothing.py'))['protected_hashes'](scene,excluded_shape_keys=('muzzleLength',))
            assert control_only==facial_repair['control_only_objects_after']=={n:current[n] for n in control_only}, 'Muzzle-only edit changed unrelated face data'
            allowed.update(control_only)
        unchanged={n:h for n,h in baseline.items() if n not in allowed}
        assert all(protected.get(n)==h for n,h in unchanged.items()), 'Facial repair changed an unrelated protected mesh'
        assert facial_repair['source_glb_sha256'] in {report['glb_sha256'],report.get('facial_repair',{}).get('source_glb_sha256')}, 'Facial repair uses a different source asset'
        evidence=facial_repair['protected_objects_before']
        assert evidence and evidence==facial_repair['protected_objects_after'], 'Unrelated object preservation failed'
        assert unchanged.keys()<=evidence.keys(), 'Incomplete unrelated-object evidence'
        assert facial_repair['protected_body_vertices']>0, 'Missing protected body vertices'
        for field in ('protected_body_position_error','protected_body_morph_error','protected_body_weight_error','original_lash_neutral_error'):
            assert facial_repair[field]==0, 'Facial repair preservation failed: '+field
        assert facial_repair['rest_joints_changed'] is False, 'Facial repair changed rest joints'
        assert isinstance(facial_repair['method'],str) and facial_repair['method'].strip()
        for field in ('eyes','mouth','blink'):
            assert isinstance(facial_repair[field],dict) and facial_repair[field], 'Missing facial evidence: '+field
        removed=(set(baseline)-set(protected))|set(report.get('facial_repair',{}).get('removed_objects',[]))
        assert removed<={'Iris_L','Iris_R'}, 'Unexpected facial component removal'
        facial_repair=dict(facial_repair,removed_objects=sorted(removed))
        report['facial_repair']=facial_repair
        report['neutral_preservation']['scope']='Historical body integration, before the separately recorded facial repair.'
        report['body_reconstruction']['facial_preservation_scope']='Historical body integration; current authorized facial changes are recorded in facial_repair.'
        clothing['protected_after_facial_repair']=protected
        report['limitations']=[s for s in report['limitations'] if not s.startswith('Jaw drop is a closed-mouth deformation')]
    elif body_repair is None:
        assert protected==baseline,'Accepted face/body/neck data changed'
    else:
        assert {n:h for n,h in protected.items() if n!='Body_Complete'}=={n:h for n,h in clothing['protected_before'].items() if n!='Body_Complete'},'Protected face changed'
        assert body_repair['protected_position_error']==0 and body_repair['protected_morph_error']==0
        assert body_repair['protected_weight_error']==0 and body_repair['after']['axilla_boundary']==0
        report['body_reconstruction']['underarm_repair']=body_repair
        clothing['protected_after_underarm_repair']=protected
    meshes=[o for o in scene.objects if o.type=='MESH']
    rig.animation_data.action=None
    for p in rig.pose.bones:p.matrix_basis=Matrix.Identity(4)
    for o in meshes:
        if o.data.shape_keys:
            for k in o.data.shape_keys.key_blocks:
                assert not k.name.startswith(('_motionFit','_run_')),'Automatic fitting is outside the requested scope'
                k.value=0
    scene.frame_set(0);bpy.context.view_layer.update()
    # Read material aliases from the accepted GLB, before changing any output.
    baseline=(OUT/'checkpoints/6446c39/landau_character.glb').read_bytes()
    size=struct.unpack_from('<I',baseline,12)[0];old=json.loads(baseline[20:20+size])
    aliases={}
    for node in old['nodes']:
        if node.get('name') not in clothing['garments'] or 'mesh' not in node:continue
        aliases[node['name']]=list({old['materials'][p['material']]['name'] for p in old['meshes'][node['mesh']]['primitives'] if 'material' in p})
    for n,values in aliases.items():clothing['garments'][n]['legacy_material_names']=values
    report['clothing_segmentation']=clothing
    report['running_animation']=json.loads((OUT/'running_retarget.json').read_text())
    report['preset_compatible_hashes']=list(dict.fromkeys(report.get('preset_compatible_hashes',[])+[report['glb_sha256']]))
    report['limitations']=[s for s in report['limitations'] if not s.startswith('Original clothes and boots')]
    report['limitations'].append('Original garment shapes and rigid placements are retained for manual fitting. Running preview does not automatically fit garments, simulate cloth, or guarantee clearance after manual edits.')
    report['limitations']=list(dict.fromkeys(report['limitations']))
    report['validation'].update(triangles=sum(sum(len(p.vertices)-2 for p in o.data.polygons) for o in meshes),
        mesh_count=len(meshes),bones=len(rig.data.bones),invalid_skin_vertices=sum(abs(sum(g.weight for g in v.groups)-1)>1e-4 or len(v.groups)>4 for o in meshes for v in o.data.vertices),errors=[])
    assert report['validation']['invalid_skin_vertices']==0
    report['validation']['shape_keys']={o.name:len(o.data.shape_keys.key_blocks)-1 for o in meshes if o.data.shape_keys}
    report['controls']=sorted({k.name for o in meshes if o.data.shape_keys for k in o.data.shape_keys.key_blocks if k.name!='Basis'})
    for n,entry in clothing['garments'].items():report['separated_parts'][n]=entry['source_faces']
    report['body_reconstruction']['clothing_segmentation']='Geometry-only source repartition; automatic fitting explicitly excluded.'
    report['body_reconstruction']['outfit_reset']='Original sculpt geometry with accepted rigid placement. Zero tailoring; manual fitting only.'
    bpy.ops.object.select_all(action='DESELECT');rig.select_set(True)
    for o in meshes:o.hide_set(False);o.select_set(True)
    rig.animation_data.action=bpy.data.actions['Running']
    scene.frame_start=1;scene.frame_end=33;scene.render.fps=60
    temporary=OUT/'landau_character.new.glb'
    bpy.ops.export_scene.gltf(filepath=str(temporary),export_format='GLB',use_selection=True,use_active_scene=True,
        export_animations=True,export_animation_mode='ACTIVE_ACTIONS',export_nla_strips_merged_animation_name='Running',
        export_frame_range=True,export_anim_slide_to_zero=True,export_morph_animation=False,
        export_morph=True,export_skins=True,export_extras=True,export_image_format='AUTO')
    if report.get('facial_repair'):
        preserve_facial_split_normals(temporary,report['facial_repair']['mouth'])
        report['facial_repair']['jaw_normal_mode']='Native authored split normals; zero body jawDrop normal delta'
        if report['facial_repair']['mouth'].get('surface_cleanup'):
            report['facial_repair']['profile_normal_mode']='Smooth authored normals transported by profile-field Jacobians'
    # os.replace replaces a symlink itself; it cannot overwrite its archive target.
    os.replace(temporary,OUT/'landau_character.glb')
    rig.animation_data.action=None
    for p in rig.pose.bones:p.matrix_basis=Matrix.Identity(4)
    scene.frame_set(0)
    for o in meshes:
        hidden=bool(o.get('default_hidden'));o.hide_set(hidden);o.hide_render=hidden
    # Preserve the action in the native master without posing the default model.
    bpy.data.actions['Running'].use_fake_user=True
    master=OUT/'landau_character.blend'
    if master.is_symlink():master.unlink()
    bpy.ops.wm.save_as_mainfile(filepath=str(master))
    report['glb_sha256']=hashlib.sha256((OUT/'landau_character.glb').read_bytes()).hexdigest()
    report['default_hidden']=[o.name for o in meshes if o.get('default_hidden')]
    for path,data in [(OUT/'asset_report.json',report),(OUT/'clothing_rebuild.json',clothing)]:
        temp=path.with_suffix('.new.json');temp.write_text(json.dumps(data,indent=2));os.replace(temp,path)
    print(json.dumps({'glb_sha256':report['glb_sha256'],'validation':report['validation']}))
    return report


if __name__=='__main__':run()
