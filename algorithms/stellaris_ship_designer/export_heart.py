"""Export the verified Heart blend to native PDX mesh/animation asset files.

Blender --background --python export_heart.py -- --blend ... --output ...
This exports assets, not a tested or installed gameplay mod.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import bpy



def enforce_native_joint_limit(rig, scene):
    """Remove six unweighted attachment bones and bake retained world poses.

    Browser/source skeleton stays intact. Never merge weighted same-name roots.
    The game rejects native meshes with more than 50 joints.
    """
    remove = {'hairFront_HairBack1', 'hairFront_HairBack2', 'horn_Head',
              'leftWing_Chest1', 'rightWing_Chest1', 'Tail1'}
    if len(rig.data.bones) <= 50:
        return {'joints': len(rig.data.bones), 'removedUnweightedBones': []}
    for obj in bpy.data.objects:
        if obj.type != 'MESH':
            continue
        for vertex in obj.data.vertices:
            for group in vertex.groups:
                if obj.vertex_groups[group.group].name in remove and group.weight > 1e-7:
                    raise ValueError('Native pruning would remove a weighted bone')
    for track in rig.animation_data.nla_tracks:
        track.mute = True
    rig.data.pose_position = 'POSE'
    names = [bone.name for bone in rig.data.bones if bone.name not in remove]
    snapshots = {}
    for clip in ['idle', 'moving', 'dash', 'planet_killer', 'moving_va', 'moving_genmaxx', 'moving_argodaemon']:
        action = bpy.data.actions[clip]
        rig.animation_data.action = action
        samples = []
        for frame in range(int(action.frame_range[0]), int(action.frame_range[1]) + 1):
            scene.frame_set(frame)
            bpy.context.view_layer.update()
            samples.append({name: rig.pose.bones[name].matrix.copy() for name in names})
        snapshots[clip] = samples
        action.name = clip + '_source_56j'
    rig.animation_data.action = None
    bpy.context.view_layer.objects.active = rig
    rig.select_set(True)
    bpy.ops.object.mode_set(mode='EDIT')
    for bone in rig.data.edit_bones:
        if bone.name in remove:
            continue
        parent = bone.parent
        while parent and parent.name in remove:
            parent = parent.parent
        if parent != bone.parent:
            matrix = bone.matrix.copy()
            bone.use_connect = False
            bone.parent = parent
            bone.matrix = matrix
    for name in remove:
        rig.data.edit_bones.remove(rig.data.edit_bones[name])
    bpy.ops.object.mode_set(mode='OBJECT')
    for clip, samples in snapshots.items():
        action = bpy.data.actions.new(clip)
        action.use_fake_user = True
        rig.animation_data.action = action
        for frame, poses in enumerate(samples, 1):
            scene.frame_set(frame)
            for bone in sorted(rig.pose.bones, key=lambda b: len(b.parent_recursive)):
                bone.rotation_mode = 'QUATERNION'
                bone.matrix = poses[bone.name]
                bpy.context.view_layer.update()
                for channel in ('location', 'rotation_quaternion', 'scale'):
                    bone.keyframe_insert(data_path=channel, frame=frame, group=bone.name)
    assert len(rig.data.bones) == 50
    return {'joints': 50, 'removedUnweightedBones': sorted(remove),
            'method': 'All retained world-space poses baked per frame; skin weights unchanged'}


def export(args):
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / 'helper_repos'))
    from io_pdx_mesh.pdx_blender.blender_import_export import export_meshfile, export_animfile
    from io_pdx_mesh import pdx_data

    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.open_mainfile(filepath=str(Path(args.blend).resolve()))
    design = json.loads((Path(__file__).parent / 'inputs/stellaris_heart.json').read_text())
    if design['ship'].get('hideMouthInterior', True):
        bpy.ops.object.select_all(action='DESELECT')
        for name in ('teeth', 'tongue'):
            obj = bpy.data.objects.get(name)
            if obj:
                obj.select_set(True)
        bpy.ops.object.delete(use_global=False)
    rig = bpy.data.objects['Stellaris_Heart_Rig']
    scene = bpy.context.scene
    joint_budget = enforce_native_joint_limit(rig, scene)
    scene.frame_set(1)
    rig.animation_data.action = None
    for track in rig.animation_data.nla_tracks:
        track.mute = True
    for bone in rig.pose.bones:
        bone.matrix_basis.identity()
    rig.data.pose_position = 'REST'
    textures = []
    for mat in bpy.data.materials:
        if not mat.use_nodes or 'shader' not in mat:
            continue
        bsdf = mat.node_tree.nodes.get('Principled BSDF')
        tex = next(n for n in mat.node_tree.nodes if n.type == 'TEX_IMAGE')
        source_image = Path(tex.image.filepath_from_user())
        if tex.image.packed_file:
            source_image = output / 'source_textures' / (mat.name + '.png')
            source_image.parent.mkdir(exist_ok=True)
            source_image.write_bytes(tex.image.packed_file.data)
        textures.append({'source': str(source_image), 'diff': str(output / (mat.name + '_diffuse.dds')), 'spec': str(output / (mat.name + '_specular.dds')), 'normal': str(output / (mat.name + '_normal.dds')), 'metallic': bsdf.inputs['Metallic'].default_value, 'roughness': bsdf.inputs['Roughness'].default_value, 'opacity': bsdf.inputs['Alpha'].default_value})
    # Format conversion preserves original diffuse RGB and multiplies existing
    # alpha by the configured material opacity. PDX uses its own
    # packed maps: normal G/A=XY, B=emissive; spec R=empire mask,
    # G=specularity, B=metalness, A=gloss. Clearcoat is approximated.
    manifest = output / 'texture_conversion.json'
    manifest.write_text(json.dumps(textures))
    subprocess.run([args.python, '-c', '''
import json, sys
from PIL import Image
for item in json.load(open(sys.argv[1])):
    image = Image.open(item['source']).convert('RGBA')
    if item['opacity'] < 1:
        alpha = image.getchannel('A').point(lambda value: round(value * item['opacity']))
        image.putalpha(alpha)
    image.save(item['diff'], pixel_format='DXT5')
    Image.new('RGBA', (4, 4), (128, 128, 0, 128)).save(item['normal'], pixel_format='DXT5')
    value = round((1-item['roughness'])*255)
    Image.new('RGBA', (4, 4), (0, 128, round(item['metallic']*255), value)).save(item['spec'], pixel_format='DXT5')
''', str(manifest)], check=True)
    for mat, item in zip([m for m in bpy.data.materials if m.use_nodes and 'shader' in m], textures):
        bsdf = mat.node_tree.nodes.get('Principled BSDF')
        tex = next(n for n in mat.node_tree.nodes if n.type == 'TEX_IMAGE')
        tex.image.filepath = item['diff']
        spec = mat.node_tree.nodes.new('ShaderNodeTexImage')
        spec.image = bpy.data.images.load(item['spec'])
        mat.node_tree.links.new(spec.outputs['Color'], bsdf.inputs['Roughness'])
        norm = mat.node_tree.nodes.new('ShaderNodeTexImage')
        norm.image = bpy.data.images.load(item['normal'])
        normal_map = mat.node_tree.nodes.new('ShaderNodeNormalMap')
        mat.node_tree.links.new(norm.outputs['Color'], normal_map.inputs['Color'])
        mat.node_tree.links.new(normal_map.outputs['Normal'], bsdf.inputs['Normal'])
    bpy.context.view_layer.objects.active = rig
    bpy.context.view_layer.update()
    mesh_path = output / 'stellaris_heart.mesh'
    export_meshfile(str(mesh_path))
    rig.data.pose_position = 'POSE'
    state_map = design['animation']['gameStateMap']
    for state in ('idle', 'moving', 'combat_moving', 'working', 'working_looping'):
        if state_map[state] not in bpy.data.actions:
            raise ValueError(f'Missing animation for {state}: {state_map[state]}')
    animation_info = []
    for name in ['idle', 'moving', 'dash', 'planet_killer', 'moving_va', 'moving_genmaxx', 'moving_argodaemon']:
        action = bpy.data.actions[name]
        rig.animation_data.action = action
        scene.frame_start, scene.frame_end = int(action.frame_range[0]), int(action.frame_range[1])
        bpy.context.view_layer.objects.active = rig
        path = output / ('heart_' + name + '.anim')
        export_animfile(str(path), scene.frame_start, scene.frame_end)
        decoded = pdx_data.read_meshfile(str(path))
        info = decoded.find('info')
        assert info.attrib['j'][0] == len(rig.data.bones)
        assert info.attrib['sa'][0] == scene.frame_end - scene.frame_start + 1
        animation_info.append({'name': name, 'frames': info.attrib['sa'][0], 'bones': info.attrib['j'][0], 'fps': info.attrib['fps'][0]})
    decoded = pdx_data.read_meshfile(str(mesh_path))
    meshes = decoded.find('object')
    locator = decoded.find('locator').find('horn_muzzle')
    assert locator is not None and locator.attrib['pa'] == ['horn_Horn']
    for mesh in meshes:
        for primitive in mesh.findall('mesh'):
            assert primitive.find('skin') is not None
    gfx = ['objectTypes = {', '  pdxmesh = {', '    name = "stellaris_heart_mesh"', '    file = "gfx/models/ships/stellaris_heart/stellaris_heart.mesh"']
    for item in animation_info:
        gfx.append('    animation = { id = "' + item['name'] + '" type = "heart_' + item['name'] + '" }')
    gfx += ['  }', '}']
    (output / 'stellaris_heart.gfx').write_text('\n'.join(gfx) + '\n')
    asset = ['# Native asset prototype; gameplay and DDS appearance require in-game validation.']
    for item in animation_info:
        asset.append('animation = { name = "heart_' + item['name'] + '" file = "heart_' + item['name'] + '.anim" }')
    asset += ['entity = {', '  name = "stellaris_heart_entity"', '  pdxmesh = "stellaris_heart_mesh"', '  default_state = "idle"', *[f'  state = {{ name = "{state}" animation = "{state_map[state]}" }}' for state in ('idle', 'moving', 'combat_moving')], f'  state = {{ name = "working" animation = "{state_map["working"]}" looping = no next_state = "working_looping" event = {{ time = 0 id = "beam_start" }} }}', f'  state = {{ name = "working_looping" animation = "{state_map["working_looping"]}" looping = yes event = {{ time = 0 id = "beam_start" }} }}', '  game_data = { size = 5.525 }', '}']
    (output / 'stellaris_heart.asset').write_text('\n'.join(asset) + '\n')
    report = {'nativeJointBudget': joint_budget, 'status': 'native_serialization_verified', 'gameRuntimeTested': False, 'meshObjects': len(meshes), 'animations': animation_info, 'gameStateMap': state_map, 'hornParent': locator.attrib['pa'][0], 'materials': 'Safe default uses opaque PdxMeshAdvanced passes. Per-group opacity can select alpha rendering; optional crystal groups add separate PdxMeshTerraAlphaBlend shells. Packed normal G/A=XY B=zero emission; spec R=zero empire mask G=specularity B=metalness A=gloss; clearcoat approximated.', 'hiddenMouthInterior': design['ship'].get('hideMouthInterior', True), 'files': [{'name': p.name, 'bytes': p.stat().st_size, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(output.iterdir()) if p.is_file() and p.suffix != '.json']}
    (output / 'native_validation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'files'}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--blend', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--python', default='/opt/homebrew/bin/python3')
    export(parser.parse_args(sys.argv[sys.argv.index('--') + 1:]))
