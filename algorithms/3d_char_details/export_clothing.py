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


def run():
    scene=bpy.data.scenes['Scene'];bpy.context.window.scene=scene
    rig=bpy.data.objects['Landau_Rig']
    report=json.loads((OUT/'asset_report.json').read_text())
    clothing=json.loads((OUT/'clothing_rebuild.json').read_text())
    protected=runpy.run_path(str(ROOT/'rebuild_clothing.py'))['protected_hashes'](scene)
    assert protected==clothing['protected_before'],'Accepted face/body/neck data changed'
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
