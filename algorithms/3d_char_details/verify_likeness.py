"""Render neutral/half/closed and assert that blink does not change ocular shape."""
from pathlib import Path
import bpy
import json
import hashlib
import runpy
import sys

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'outputs/landau_v10'

def set_blink(value):
    for o in bpy.context.scene.objects:
        if o.type=='MESH' and o.data.shape_keys:
            for k in o.data.shape_keys.key_blocks:
                k.value=value if k.name in {'eyeBlinkL','eyeBlinkR'} else 4*value*(1-value) if k.name in {'_blinkArcL','_blinkArcR'} else 0
            o.data.shape_keys.update_tag();o.data.update()
    bpy.context.scene.frame_set(1);bpy.context.view_layer.update()

def main():
    reimport='--glb' in sys.argv
    if reimport:
        bpy.ops.wm.read_factory_settings(use_empty=True)
        bpy.ops.import_scene.gltf(filepath=str(OUT/'landau_character.glb'))
        if bpy.context.scene.world is None:bpy.context.scene.world=bpy.data.worlds.new('Proof world')
        from mathutils import Matrix
        for obj in bpy.context.scene.objects:
            if obj.type=='ARMATURE':
                if obj.animation_data:obj.animation_data.action=None
                for bone in obj.pose.bones:bone.matrix_basis=Matrix.Identity(4)
            if obj.get('default_hidden'):obj.hide_render=True
    body=bpy.data.objects.get('Body_Complete');lift=0.0
    if body:
        lift=body.get('head_rigid_lift',bpy.context.scene.get('head_rigid_lift'))
        if lift is None:
            report_path=OUT/'asset_report.json'
            report=json.loads(report_path.read_text()) if report_path.exists() else {}
            lift=report.get('body_reconstruction',{}).get('head_rigid_lift')
        if lift is None:raise RuntimeError('Complete body lacks head placement metadata for proof framing')
        lift=float(lift)
    assert not any(o.name.startswith('LowerLid') for o in bpy.context.scene.objects)
    independent=[]
    for o in bpy.context.scene.objects:
        if o.name.startswith(('EyeShell_','Iris_','RoundIris_','Pupil_','Catchlight')):
            names=set(o.data.shape_keys.key_blocks.keys()) if o.data.shape_keys else set()
            assert not any(n.startswith(('eyeBlink','eyeSquint','eyeWide')) for n in names),o.name+' still deforms with blink'
            independent.append(o.name)
    joins={}
    for side in ([] if reimport else ['L','R']):
        lash=bpy.data.objects['Lash_'+side];lid=bpy.data.objects['UpperLid_'+side]
        assert len(lid.data.materials)==1 and 'Blue' in lid.data.materials[0].name
        assert lash.data.materials[0].name=='Navy lashes'
        N=lid['source_lash_boundary_count'];maximum=0
        for value in [0,.5,1]:
            a=lash.data.shape_keys.key_blocks; b=lid.data.shape_keys.key_blocks
            points=[v.co.lerp(a['eyeBlink'+side].data[i].co,value) for i,v in enumerate(a['Basis'].data)]
            for i in range(len(b['Basis'].data)-N,len(b['Basis'].data)):
                point=b['Basis'].data[i].co.lerp(b['eyeBlink'+side].data[i].co,value)
                maximum=max(maximum,min((point-p).length for p in points))
        assert maximum<1e-6,(side,maximum)
        joins[side]={'original_lash_vertices':len(lash.data.vertices),'attachment_vertices':N,'max_join_gap':maximum}
    helpers=runpy.run_path(str(ROOT/'inspect_landau.py'));helpers['setup_render']()
    bpy.context.scene.world.use_nodes=True
    background=bpy.context.scene.world.node_tree.nodes.get('Background')
    background.inputs['Color'].default_value=(.035,.035,.035,1)
    background.inputs['Strength'].default_value=1
    # The accepted native master uses -1.25 EV; factory GLB imports start at 0.
    # Match both so color and shading comparisons are meaningful.
    bpy.context.scene.view_settings.exposure=-1.25
    bpy.context.scene.view_settings.gamma=1
    if reimport:
        render=helpers['render_view']
        helpers['render_view']=lambda name,*args,**kwargs:render('export_'+name,*args,**kwargs)
    for o in bpy.context.scene.objects:
        if o.type=='LIGHT':o.data.energy*=.7
    bpy.context.scene.cycles.samples=8;bpy.context.scene.render.resolution_percentage=70
    for value,name in [(0,'geometry_open'),(.5,'geometry_half'),(1,'geometry_closed')]:
        set_blink(value);helpers['render_view'](name,(0,-2,.7+lift),target=(0,0,.7+lift),scale=.36)
    set_blink(0)
    helpers['render_view']('geometry_threequarter',(.8,-2,.71+lift),target=(0,0,.7+lift),scale=.36)
    helpers['render_view']('geometry_full',(0,-3,.58+lift/2),target=(0,0,.58+lift/2),scale=1.28+lift)
    if bpy.data.objects.get('Mouth_Interior'):
        for value in [.5,1]:
            set_blink(0)
            for o in bpy.context.scene.objects:
                if o.type=='MESH' and o.data.shape_keys and 'jawDrop' in o.data.shape_keys.key_blocks:
                    o.data.shape_keys.key_blocks['jawDrop'].value=value;o.data.shape_keys.update_tag();o.data.update()
            bpy.context.view_layer.update()
            helpers['render_view']('mouth_'+('half' if value==.5 else 'open'),(0,-2,.66+lift),target=(0,0,.66+lift),scale=.25)
        set_blink(1)
        helpers['render_view']('blink_oblique',(.7,-2,.70+lift),target=(0,0,.70+lift),scale=.36)
        set_blink(0)
        for name,jaw,length,curve,side in [('neutral',0,0,0,0),('oblique',0,0,0,.6),('profile',0,0,0,2),('open',1,0,0,0),('wide',.5,1,1,0),('short',.5,-1,-1,0),('profile_inward',0,0,0,2)]:
            for o in bpy.context.scene.objects:
                if o.type=='MESH' and o.data.shape_keys:
                    for k in o.data.shape_keys.key_blocks:
                        k.value={'jawDrop':jaw,'mouthLength':length,'mouthCurvature':curve,'muzzleLength':.5 if name=='profile_inward' else 0,'jawRecess':.5 if name=='profile_inward' else 0}.get(k.name,0)
                    o.data.shape_keys.update_tag();o.data.update()
            bpy.context.view_layer.update()
            helpers['render_view']('mouth_refined_'+name,(side,-.08 if name.startswith('profile') else -2,.633+lift),target=(0,-.08 if name.startswith('profile') else 0,.633+lift),scale=.14 if name.startswith('profile') else .105)
        set_blink(0)
    result={'blink_static_ocular_parts':sorted(independent),'original_lash_attachment':joins,
        'states_rendered':[0,.5,1],'head_rigid_lift':lift,'lower_lid_geometry':False,'replacement_eyelashes':False}
    result['glb_reimport']=reimport
    asset=OUT/('landau_character.glb' if reimport else 'landau_character.blend')
    result['asset_sha256']=hashlib.sha256(asset.read_bytes()).hexdigest()
    if reimport:
        result['shading_scope']='Blender reimport checks geometry; its importer does not restore glTF morph NORMAL attributes. Check final shading in the sandbox WebGL renderer.'
    (OUT/('export_likeness_validation.json' if reimport else 'likeness_validation.json')).write_text(json.dumps(result,indent=2))
    print(json.dumps(result))

if __name__=='__main__':main()
