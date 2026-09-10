"""Render neutral/half/closed and assert that blink does not change ocular shape."""
from pathlib import Path
import bpy
import json
import runpy

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'outputs/landau_v10'

def set_blink(value):
    for o in bpy.context.scene.objects:
        if o.type=='MESH' and o.data.shape_keys:
            for k in o.data.shape_keys.key_blocks:k.value=value if k.name in {'eyeBlinkL','eyeBlinkR'} else 0
            o.data.shape_keys.update_tag();o.data.update()
    bpy.context.scene.frame_set(1);bpy.context.view_layer.update()

def main():
    assert not any(o.name.startswith('LowerLid') for o in bpy.context.scene.objects)
    independent=[]
    for o in bpy.context.scene.objects:
        if o.name.startswith(('EyeShell_','Iris_','RoundIris_','Pupil_','Catchlight')):
            names=set(o.data.shape_keys.key_blocks.keys()) if o.data.shape_keys else set()
            assert not any(n.startswith(('eyeBlink','eyeSquint','eyeWide')) for n in names),o.name+' still deforms with blink'
            independent.append(o.name)
    joins={}
    for side in ['L','R']:
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
    for o in bpy.context.scene.objects:
        if o.type=='LIGHT':o.data.energy*=.7
    bpy.context.scene.cycles.samples=8;bpy.context.scene.render.resolution_percentage=70
    for value,name in [(0,'geometry_open'),(.5,'geometry_half'),(1,'geometry_closed')]:
        set_blink(value);helpers['render_view'](name,(0,-2,.7),target=(0,0,.7),scale=.36)
    set_blink(0)
    helpers['render_view']('geometry_threequarter',(.8,-2,.71),target=(0,0,.7),scale=.36)
    helpers['render_view']('geometry_full',(0,-3,.58))
    result={'blink_static_ocular_parts':sorted(independent),'original_lash_attachment':joins,
        'states_rendered':[0,.5,1],'source_geometry_preserved':True,'lower_lid_geometry':False,'replacement_eyelashes':False}
    (OUT/'likeness_validation.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result))

if __name__=='__main__':main()
