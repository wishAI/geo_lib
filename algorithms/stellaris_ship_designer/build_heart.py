"""Convert the user's PonyLumen SMD export into the designer's skinned GLB.

Run with Blender --background --python build_heart.py -- --source <folder>
--output <folder>. Original downloads are read only; artifacts are reproducible.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shlex
import sys
import zipfile

import bmesh
import bpy
from mathutils import Euler, Matrix, Vector
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pony_animation import parse as read_animation
from dmx_flight import Flight


def read_smd(text):
    nodes, poses, triangles = {}, {}, []
    lines = iter(text.splitlines())
    for line in lines:
        if line == 'nodes':
            for row in lines:
                if row == 'end':
                    break
                idx, name, parent = shlex.split(row)
                nodes[int(idx)] = (name, int(parent))
        elif line == 'skeleton':
            for row in lines:
                if row == 'end':
                    break
                if row.startswith('time '):
                    if row != 'time 0':
                        raise ValueError('Expected the creator single-frame export')
                    continue
                values = row.split()
                poses[int(values[0])] = Matrix.Translation(Vector(map(float, values[1:4]))) @ Euler(tuple(map(float, values[4:7])), 'XYZ').to_matrix().to_4x4()
        elif line == 'triangles':
            for material in lines:
                if material == 'end':
                    break
                tri = []
                for _ in range(3):
                    row = next(lines).split()
                    weights = {}
                    for j in range(int(row[9])):
                        bone, weight = int(row[10 + 2*j]), float(row[11 + 2*j])
                        weights[bone] = weights.get(bone, 0) + weight
                    # SMD's implicit remainder belongs to the parent (PonyLumen: 0).
                    remainder = max(0, 1 - sum(weights.values()))
                    weights[int(row[0])] = weights.get(int(row[0]), 0) + remainder
                    weights = {nodes[k][0]: v for k, v in weights.items() if v > 1e-7}
                    total = sum(weights.values())
                    tri.append((tuple(map(float, row[1:4])), tuple(map(float, row[4:7])), tuple(map(float, row[7:9])), {k: v / total for k, v in weights.items()}))
                triangles.append(tri)
    world = {}
    def global_pose(idx):
        if idx not in world:
            parent = nodes[idx][1]
            world[idx] = (global_pose(parent) if parent >= 0 else Matrix.Identity(4)) @ poses[idx]
        return world[idx]
    for idx in nodes:
        global_pose(idx)
    return nodes, world, triangles


def build(args):
    source, out = Path(args.source).expanduser(), Path(args.output).resolve()
    design = json.loads((Path(__file__).parent / 'inputs/stellaris_heart.json').read_text())
    material_groups = design['ship'].get('materialGroups', {})
    crystal_opacity = float(design['ship'].get('crystalOpacity', 1))
    crystal_shell_opacity = float(design['ship'].get('crystalShellOpacity', 0))
    hide_mouth_interior = bool(design['ship'].get('hideMouthInterior', True))
    defaults = {
        'crystal': False, 'opacity': 1.0, 'metallic': 0.0,
        'roughness': 0.4, 'clearcoat': 0.0, 'transmission': 0.0,
        'shellOpacity': 0.0, 'faceted': False,
    }
    def group_name(part):
        if part in {'leftWing', 'rightWing'}:
            return 'wings'
        if part in {'leftEye', 'rightEye'}:
            return 'eyes'
        if part in {'hairFront', 'hairBack', 'tail'}:
            return 'maneTail'
        return 'body' if part in {'pony', 'horn'} else None
    def group_config(group):
        config = dict(defaults)
        config.update(material_groups.get(group, {}))
        return config
    planet_killer_pitch = float(design['animation'].get('planetKillerPitch', -45))
    out.mkdir(parents=True, exist_ok=True)
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    archive = zipfile.ZipFile(source / 'pony.zip')
    smds = sorted((n for n in archive.namelist() if n.endswith('.smd')), key=lambda n: (n != 'pony.smd', n))
    parsed = {Path(n).stem: read_smd(archive.read(n).decode()) for n in smds}
    accessory_bones = dict(zip(range(5, 13), ['LeftLeg2', 'LeftForearm', 'RightLeg2', 'RightForearm', 'LeftFoot', 'LeftBall', 'RightFoot', 'RightBall']))
    animation_dir = Path(__file__).parent / 'inputs/heart_animation'
    file_map = json.loads((animation_dir / 'sources.json').read_text())['parts']
    anim_sources = {part: read_animation(animation_dir / Path(path).name) for part, path in file_map.items()}
    attachments = {'hairFront': 'Head', 'hairBack': 'Head', 'horn': 'Head', 'headgear': 'Head', 'collar': 'Neck', 'tail': 'Tail1', 'leftWing': 'Chest1', 'rightWing': 'Chest1'}
    attachments.update({f'accessory_{i}': bone for i, bone in accessory_bones.items()})
    def bone_name(part, name):
        return name if part == 'pony' else part + '_' + name
    # A part's original root bind transform MUST remain separate from the body
    # attachment. The creator replaces that root only after computing inverses.
    for part, data in anim_sources.items():
        nodes, _, triangles = parsed[part]
        if len(triangles) != len(data['faces']):
            raise ValueError(f'{part}: source topology differs from the exported pony')
        # Restore exact original weights: the SMD writer can truncate influences
        # when bone zero appears before another influence in its packed array.
        for tri, face in zip(triangles, data['faces']):
            for vertex, index in zip(tri, face):
                if (Vector(vertex[0]) - Vector(data['positions'][index])).length > 1e-4:
                    raise ValueError(f'{part}: customized geometry needs a separate retarget')
                raw = data['weights'][index] if data['weights'] else {data['root']: 1}
                total = sum(raw.values())
                vertex[3].clear()
                vertex[3].update({data['names'][k]: weight / total for k, weight in raw.items() if weight > 1e-7})
        for idx, (name, parent) in list(nodes.items()):
            nodes[idx] = (bone_name(part, name), parent)
        for tri in triangles:
            for vertex in tri:
                weights = vertex[3]
                for old in list(weights):
                    new = bone_name(part, old)
                    if new != old:
                        weights[new] = weights.pop(old)
    # Restore the custom cutie mark using the creator's seed-face projection,
    # adjacency region and saved UV controls (it is absent from the SMD texture).
    save = json.loads((source / 'ponysave.json').read_text())
    body = anim_sources['pony']
    adjacent = {}
    for face_id, face in enumerate(body['faces']):
        for vertex in face:
            adjacent.setdefault(body['original_indices'][vertex], set()).add(face_id)
    for side, seed in [('left', 7176), ('right', 3840)]:
        region = {seed}
        for _ in range(4):
            region |= {f for face_id in region for v in body['faces'][face_id] for f in adjacent[body['original_indices'][v]]}
        points = [Vector(body['positions'][v]) for v in body['faces'][seed]]
        center = sum(points, Vector()) / 3
        normal = (points[0] - points[1]).cross(points[0] - points[2]).normalized()
        right = Vector((0, 1, 0)).cross(normal).normalized()
        up = normal.cross(right)
        prefix = 'custom_' + side[0] + '_CM_'
        angle = save[prefix + 'angle']
        decal = []
        for face_id in sorted(region):
            triangle = []
            for pos, n, uv, weights in parsed['pony'][2][face_id]:
                delta = Vector(pos) - center
                x = 2 * delta.dot(right) * save[prefix + 'uscale'] * (-1 if save[prefix + 'reverse'] else 1)
                y = -2 * delta.dot(up) * save[prefix + 'vscale']
                u = x * math.cos(angle) - y * math.sin(angle) + save[prefix + 'ut']
                v = x * math.sin(angle) + y * math.cos(angle) + save[prefix + 'vt']
                # Account for the creator's 5-pixel transparent border around a
                # 502px image without altering the user's original bitmap.
                u, v = (u*512 - 5)/502, (v*512 - 5)/502
                triangle.append((tuple(Vector(pos) + Vector(n) * .012), n, (u, -(1-v)), dict(weights)))
            decal.append(triangle)
        parsed['cutie_mark_' + side] = (parsed['pony'][0], parsed['pony'][1], decal)
    def source_world(data, frame, root_override=None):
        world = {}
        def visit(idx):
            if idx not in world:
                values = data['local_matrices_column_major'][idx][min(frame, data['frame_count'] - 1)]
                local = Matrix([values[i:i+4] for i in range(0, 16, 4)]).transposed()
                parent = data['parents'][idx]
                world[idx] = (visit(parent) @ local if parent >= 0 else root_override.copy() if root_override is not None else local)
            return world[idx]
        return {name: visit(i) for i, name in enumerate(data['names'])}
    source_rest, source_parents = {}, {}
    for part, data in anim_sources.items():
        source_rest.update({bone_name(part, n): m for n, m in source_world(data, 0).items()})
        for idx, name in enumerate(data['names']):
            parent = data['parents'][idx]
            source_parents[bone_name(part, name)] = bone_name(part, data['names'][parent]) if parent >= 0 else attachments.get(part)
    # Source: Y up, +Z nose. Blender: Z up, -Y nose. GLB returns Y up/+Z.
    scale = args.length / (max(v[0][2] for tri in parsed['pony'][2] for v in tri) - min(v[0][2] for tri in parsed['pony'][2] for v in tri))
    basis = Matrix.Rotation(math.pi / 2, 4, 'X')
    def convert(mat):
        result = basis @ mat
        result.translation *= scale
        return result
    rig = bpy.data.objects.new('Stellaris_Heart_Rig', bpy.data.armatures.new('Heart_skeleton'))
    bpy.context.collection.objects.link(rig)
    bpy.context.view_layer.objects.active = rig
    rig.select_set(True)
    bpy.ops.object.mode_set(mode='EDIT')
    rest, parents = {}, {}
    for nodes, world, _ in parsed.values():
        for idx, (name, parent) in nodes.items():
            if name in rest:
                continue
            rest[name] = convert(source_rest.get(name, world[idx]))
            parents[name] = source_parents.get(name, nodes[parent][0] if parent >= 0 else None)
            bone = rig.data.edit_bones.new(name)
            bone.matrix = rest[name]
            bone.length = .12
    for name, parent in parents.items():
        if parent:
            rig.data.edit_bones[name].parent = rig.data.edit_bones[parent]
    bpy.ops.object.mode_set(mode='OBJECT')
    meshes = []
    textures = out / 'textures'
    textures.mkdir(exist_ok=True)
    for name, (_, _, triangles) in parsed.items():
        positions, normals, uvs, skin, faces, lookup = [], [], [], [], [], {}
        for triangle in triangles:
            face = []
            for pos, normal, uv, weights in triangle:
                key = (pos, normal, uv, tuple(sorted(weights.items())))
                if key not in lookup:
                    lookup[key] = len(positions)
                    positions.append(basis @ Vector(pos) * scale)
                    normals.append(basis.to_3x3() @ Vector(normal))
                    # Creator texture export is flipped relative to standard SMD UVs.
                    uvs.append((uv[0], -uv[1]))
                    skin.append(weights)
                face.append(lookup[key])
            if len(set(face)) == 3:
                faces.append(face)
        mesh = bpy.data.meshes.new(name)
        mesh.from_pydata(positions, [], faces)
        mesh.validate(clean_customdata=False)
        mesh.update()
        mesh.normals_split_custom_set_from_vertices(normals)
        uv_layer = mesh.uv_layers.new(name='UVMap')
        for loop in mesh.loops:
            uv_layer.data[loop.index].uv = uvs[loop.vertex_index]
        base_group = group_name(name)
        base_config = group_config(base_group) if base_group else defaults
        for polygon in mesh.polygons:
            polygon.use_smooth = not bool(base_config.get('faceted', False))
        obj = bpy.data.objects.new(name, mesh)
        bpy.context.collection.objects.link(obj)
        obj.parent = rig
        modifier = obj.modifiers.new('Original PonyLumen skin', 'ARMATURE')
        modifier.object = rig
        for bone in sorted({key for weights in skin for key in weights}):
            group = obj.vertex_groups.new(name=bone)
            for index, weights in enumerate(skin):
                if bone in weights:
                    group.add([index], weights[bone], 'REPLACE')
        path = textures / f'{name}.png'
        is_decal = name.startswith('cutie_mark_')
        path.write_bytes((source / 'cutie_mark.png').read_bytes() if is_decal else archive.read(f'{name}.png'))
        mat = bpy.data.materials.new(name)
        mat.use_nodes = True
        mat['shader'] = 'PdxMeshAdvanced'
        bsdf = mat.node_tree.nodes.get('Principled BSDF')
        tex = mat.node_tree.nodes.new('ShaderNodeTexImage')
        tex.image = bpy.data.images.load(str(path))
        mat.node_tree.links.new(tex.outputs['Color'], bsdf.inputs['Base Color'])
        if is_decal:
            tex.extension = 'CLIP'
            mat.node_tree.links.new(tex.outputs['Alpha'], bsdf.inputs['Alpha'])
            mat.surface_render_method = 'DITHERED'
            mat['shader'] = 'PdxMeshAlphaBlend'
        # Only yellow texels belong to the gold group. Non-yellow gemstones and
        # trim on jewelry objects retain their neutral material properties.
        config = base_config
        crystal = bool(config.get('crystal', False))
        opacity = float(config.get('opacity', crystal_opacity))
        bsdf.inputs['Metallic'].default_value = float(config.get('metallic', 0))
        bsdf.inputs['Roughness'].default_value = float(config.get('roughness', .4))
        bsdf.inputs['Coat Weight'].default_value = float(config.get('clearcoat', 0))
        bsdf.inputs['Coat Roughness'].default_value = .12
        bsdf.inputs['Alpha'].default_value = opacity
        if 'Transmission Weight' in bsdf.inputs:
            bsdf.inputs['Transmission Weight'].default_value = float(config.get('transmission', 0))
        if opacity < .999:
            mat.surface_render_method = 'DITHERED'
            mat['shader'] = 'PdxMeshTerraAlphaBlend' if crystal else 'PdxMeshAlphaBlend'
        mesh.materials.append(mat)
        # Material assignment follows the actual yellow texels, preserving blue
        # gemstones while making yellow feather/jewelry/detail faces metallic.
        from array import array
        pixels = array('f', [0.0]) * len(tex.image.pixels)
        tex.image.pixels.foreach_get(pixels)
        width, height = tex.image.size
        gold_faces = []
        for polygon in mesh.polygons:
            uv = sum((uv_layer.data[i].uv for i in polygon.loop_indices), Vector((0, 0))) / len(polygon.loop_indices)
            if is_decal and not (0 <= uv.x <= 1 and 0 <= uv.y <= 1):
                continue
            x, y = int((uv.x % 1) * width), int((uv.y % 1) * height)
            r, g, b, a = pixels[(y*width+x)*4:(y*width+x)*4+4]
            if a > .5 and r > .5 and g > .25 and r > g*1.12 and g > b*1.35:
                gold_faces.append(polygon)
        if gold_faces:
            gold_config = group_config('gold')
            gold_mat = mat.copy()
            gold_mat.name = name + '_gold'
            gold_bsdf = gold_mat.node_tree.nodes.get('Principled BSDF')
            gold_bsdf.inputs['Metallic'].default_value = float(gold_config.get('metallic', .85))
            gold_bsdf.inputs['Roughness'].default_value = float(gold_config.get('roughness', .24))
            gold_bsdf.inputs['Coat Weight'].default_value = float(gold_config.get('clearcoat', .2))
            gold_opacity = float(gold_config.get('opacity', 1))
            gold_bsdf.inputs['Alpha'].default_value = gold_opacity
            if 'Transmission Weight' in gold_bsdf.inputs:
                gold_bsdf.inputs['Transmission Weight'].default_value = float(gold_config.get('transmission', 0))
            gold_mat['shader'] = 'PdxMeshAdvanced'
            if gold_opacity < .999:
                gold_mat.surface_render_method = 'DITHERED'
                gold_mat['shader'] = 'PdxMeshTerraAlphaBlend' if gold_config.get('crystal') else 'PdxMeshAlphaBlend'
            mesh.materials.append(gold_mat)
            for polygon in gold_faces:
                polygon.material_index = 1
                polygon.use_smooth = not bool(gold_config.get('faceted', False))
        meshes.append(obj)
        def add_shell(group, source_material, keep_material_index):
            shell_config = group_config(group)
            shell_opacity = float(shell_config.get('shellOpacity', crystal_shell_opacity))
            if not shell_config.get('crystal') or shell_opacity <= 0:
                return
            shell = obj.copy()
            shell.name = name + '_' + group + '_crystal_shell'
            shell.data = obj.data.copy()
            shell.data.name = shell.name
            subset = bmesh.new()
            subset.from_mesh(shell.data)
            remove = [face for face in subset.faces if face.material_index != keep_material_index]
            bmesh.ops.delete(subset, geom=remove, context='FACES')
            if not subset.faces:
                subset.free()
                bpy.data.meshes.remove(shell.data)
                return
            subset.to_mesh(shell.data)
            subset.free()
            # Offset the shell just beyond the opaque surface.  This prevents
            # z-fighting while retaining the source skinning and animation.
            for vertex in shell.data.vertices:
                vertex.co += vertex.normal * .008
            shell_mat = source_material.copy()
            shell_mat.name = name + ('_gold' if group == 'gold' else '_' + group) + '_crystal_shell'
            shell_bsdf = shell_mat.node_tree.nodes.get('Principled BSDF')
            shell_bsdf.inputs['Alpha'].default_value = shell_opacity
            shell_bsdf.inputs['Metallic'].default_value = max(.15, float(shell_config.get('metallic', 0)))
            shell_bsdf.inputs['Roughness'].default_value = min(.12, float(shell_config.get('roughness', .4)))
            if 'Transmission Weight' in shell_bsdf.inputs:
                shell_bsdf.inputs['Transmission Weight'].default_value = .05
            shell_mat.surface_render_method = 'DITHERED'
            shell_mat['shader'] = 'PdxMeshTerraAlphaBlend'
            shell.data.materials.clear()
            shell.data.materials.append(shell_mat)
            for polygon in shell.data.polygons:
                polygon.material_index = 0
            bpy.context.collection.objects.link(shell)
            meshes.append(shell)
        if base_group:
            add_shell(base_group, mat, 0)
        if gold_faces:
            add_shell('gold', gold_mat, 1)
    # All offensive slots share one head-following horn tip locator.
    horn = bpy.data.objects['horn']
    tip = max((v.co for v in horn.data.vertices), key=lambda v: v.z).copy()
    locator = bpy.data.objects.new('horn_muzzle', None)
    bpy.context.collection.objects.link(locator)
    locator.parent = rig
    locator.parent_type = 'BONE'
    locator.parent_bone = 'horn_Horn'
    bpy.context.view_layer.update()
    locator.matrix_world = Matrix.Translation(tip) @ basis
    report = {'sourceSha256': hashlib.sha256((source / 'pony.zip').read_bytes()).hexdigest(), 'bones': [{'name': n, 'parent': p} for n, p in parents.items()], 'parts': list(parsed), 'triangles': sum(len(m.data.polygons) for m in meshes), 'vertices': sum(len(m.data.vertices) for m in meshes), 'scale': scale, 'bodyLength': args.length, 'hornTip': list(basis.inverted() @ tip), 'crystalOpacity': crystal_opacity, 'crystalShellOpacity': crystal_shell_opacity, 'materialGroups': material_groups, 'hideMouthInterior': hide_mouth_interior, 'planetKillerPitch': planet_killer_pitch, 'animations': [], 'nativeGameTested': False}
    scene = bpy.context.scene
    scene.render.fps = 24
    rig.animation_data_create()
    actions = []
    # The original fly cycle; moving is an explicitly labelled faster variant,
    # and dash is an authored streamlined pose using the same compatible rig.
    sync = design['animation'].get('fastFlightSync', {})
    community = {
        'moving_va': Flight(
            'moving_va',
            wing_cycles=sync.get('wingCycles', 7),
            wing_phase=sync.get('wingPhase', 0),
            tail_cycles=sync.get('tailCycles', 4),
            body_duration=sync.get('bodyDuration', 4),
            body_speed=sync.get('bodySpeed', 1),
            wing_speed=sync.get('wingSpeed', 1),
            tail_phase=sync.get('tailPhase', 0),
            lock_seamless=sync.get('lockSeamlessLoop', True),
        ),
        'moving_genmaxx': Flight('moving_genmaxx'),
        'moving_argodaemon': Flight('moving_argodaemon'),
    }
    clips = [('idle', 25, 'source'), ('moving', 19, 'adapted'), ('dash', 13, 'authored'),
             ('planet_killer', 25, 'source_pose_variant')]
    clips += [(name, flight.frames, 'retargeted') for name, flight in community.items()]
    for clip, frames, mode in clips:
        action = bpy.data.actions.new(clip)
        rig.animation_data.action = action
        scene.frame_start, scene.frame_end = 1, frames
        for f in range(frames):
            source_frame = 25 + int((f / (frames - 1) * 24) % 24)
            body_world = source_world(anim_sources['pony'], source_frame)
            world = dict(body_world)
            for part, data in anim_sources.items():
                if part == 'pony':
                    continue
                part_world = source_world(data, source_frame if 'Wing' in part else 0, body_world[attachments[part]])
                world.update({bone_name(part, n): m for n, m in part_world.items()})
            if clip in community:
                world = community[clip].pose(f / (frames - 1), anim_sources, source_world, attachments)
            scene.frame_set(f + 1)
            for bone in rig.pose.bones:
                bone.rotation_mode = 'QUATERNION'
                bone.matrix_basis = Matrix.Identity(4)
            for bone in sorted(rig.pose.bones, key=lambda b: len(b.parent_recursive)):
                if bone.name in world:
                    # Some creator bind matrices have a reflected basis. Blender
                    # edit bones retain only a proper rotation. Rebase the full
                    # source skinning transform onto Blender's actual rest bone;
                    # assigning the source matrix directly would twist the mesh.
                    target = convert(world[bone.name]) @ rest[bone.name].inverted() @ rig.data.bones[bone.name].matrix_local
                    if clip == 'dash':
                        # Forward lean is a whole skeleton pose, not mesh sliding.
                        target = Matrix.Rotation(math.radians(-16), 4, 'X') @ target
                    elif clip == 'planet_killer':
                        # Reuse the verified idle flight, pitched toward the
                        # target planet so the horn beam reads naturally.
                        # The editable value is expressed in the browser/game
                        # forward convention; Blender's converted X pitch has
                        # the opposite sign.
                        target = Matrix.Rotation(math.radians(-planet_killer_pitch), 4, 'X') @ target
                    bone.matrix = target
                    bpy.context.view_layer.update()
                for channel in ('location', 'rotation_quaternion', 'scale'):
                    bone.keyframe_insert(data_path=channel, frame=f + 1, group=bone.name)
        action.use_fake_user = True
        actions.append((action, frames))
        report['animations'].append({'name': clip, 'frames': frames, 'fps': 24, 'duration': (frames - 1) / 24, 'provenance': mode})
        if clip in community:
            report['animations'][-1]['retarget'] = community[clip].report
        rig.animation_data.action = None
    for action, frames in actions:
        track = rig.animation_data.nla_tracks.new()
        track.name = action.name
        track.strips.new(action.name, 1, action)
        track.mute = True
    rig.animation_data.action = actions[0][0]
    scene.frame_start, scene.frame_end = 1, 25
    max_skin_error = 0.0
    for frame in range(1, 26):
        scene.frame_set(frame)
        body_world = source_world(anim_sources['pony'], 25 + (frame - 1) % 24)
        expected = dict(body_world)
        for part, data in anim_sources.items():
            if part != 'pony':
                part_world = source_world(data, 25 + (frame - 1) % 24 if 'Wing' in part else 0, body_world[attachments[part]])
                expected.update({bone_name(part, n): m for n, m in part_world.items()})
        for name, matrix in expected.items():
            actual_skin = rig.pose.bones[name].matrix @ rig.data.bones[name].matrix_local.inverted()
            expected_skin = convert(matrix) @ rest[name].inverted()
            max_skin_error = max(max_skin_error, max(abs(actual_skin[i][j] - expected_skin[i][j]) for i in range(4) for j in range(4)))
    if max_skin_error > 0.0002:
        raise ValueError(f'Creator skinning reconstruction failed: {max_skin_error}')
    report['maxCreatorSkinMatrixError'] = max_skin_error
    report['verifiedFlyFrames'] = 25
    report['sourceGeometryAndWeightsVerified'] = True
    report['communityValidation'] = {}
    for name, flight in community.items():
        rig.animation_data.action = bpy.data.actions[name]
        matrix_error, seam_error, left_ear_error = 0.0, 0.0, 0.0
        root_anchor_error, wing_start_error, wing_motion = 0.0, 0.0, 0.0
        first = None
        left_ear_rest = rig.data.bones['Head'].matrix_local.inverted() @ rig.data.bones['LeftEar'].matrix_local
        pelvis_rest = rig.data.bones['vn_pony_reference'].matrix_local.inverted() @ rig.data.bones['Pelvis'].matrix_local
        wing_names = ['leftWing_LeftWingOpen', 'rightWing_RightWingOpen']
        wing_first = {}
        for frame in range(1, flight.frames + 1):
            scene.frame_set(frame)
            left_ear_local = rig.pose.bones['Head'].matrix.inverted() @ rig.pose.bones['LeftEar'].matrix
            left_ear_error = max(left_ear_error, max(abs(left_ear_local[i][j] - left_ear_rest[i][j]) for i in range(4) for j in range(4)))
            pelvis_local = rig.pose.bones['vn_pony_reference'].matrix.inverted() @ rig.pose.bones['Pelvis'].matrix
            root_anchor_error = max(root_anchor_error, (pelvis_local.translation-pelvis_rest.translation).length)
            for wing_name in wing_names:
                bone = rig.pose.bones[wing_name]
                local = bone.parent.matrix.inverted() @ bone.matrix
                rest_local = bone.parent.bone.matrix_local.inverted() @ bone.bone.matrix_local
                if frame == 1:
                    wing_first[wing_name] = local.copy()
                    wing_start_error = max(wing_start_error, max(abs(local[i][j]-rest_local[i][j]) for i in range(4) for j in range(4)))
                else:
                    wing_motion = max(wing_motion, max(abs(local[i][j]-wing_first[wing_name][i][j]) for i in range(4) for j in range(4)))
        for frame in sorted({1, flight.frames//4+1, flight.frames//2+1, 3*flight.frames//4+1, flight.frames}):
            scene.frame_set(frame)
            expected = flight.pose((frame-1)/(flight.frames-1), anim_sources, source_world, attachments)
            actual = {b.name: b.matrix.copy() for b in rig.pose.bones}
            for bone_name_, matrix_ in expected.items():
                skin = actual[bone_name_] @ rig.data.bones[bone_name_].matrix_local.inverted()
                target_skin = convert(matrix_) @ rest[bone_name_].inverted()
                matrix_error = max(matrix_error, max(abs(skin[i][j]-target_skin[i][j]) for i in range(4) for j in range(4)))
            if first is None:
                first = actual
            if frame == flight.frames:
                seam_error = max(abs(actual[b][i][j]-first[b][i][j]) for b in actual for i in range(4) for j in range(4))
        if matrix_error > .0002 or seam_error > .0002 or left_ear_error > .0002 or root_anchor_error > .0002 or wing_start_error > .0002 or wing_motion < .01:
            raise ValueError(f'{name}: retarget/loop/ear/root/wing check failed: {matrix_error}, {seam_error}, {left_ear_error}, {root_anchor_error}, {wing_start_error}, {wing_motion}')
        report['communityValidation'][name] = {'maxSkinMatrixError': matrix_error, 'maxLoopMatrixError': seam_error, 'leftEarRestLocalMatrixError': left_ear_error, 'rootAnchorTranslationError': root_anchor_error, 'wingStartRestMatrixError': wing_start_error, 'wingMotionMatrixRange': wing_motion, 'leftEarFramesChecked': flight.frames, 'sampledPoses': 5, 'passed': True}
    rig.animation_data.action = actions[0][0]
    scene.frame_set(1)
    bpy.ops.file.pack_all()
    bpy.ops.wm.save_as_mainfile(filepath=str(out / 'heart.blend'))
    bpy.ops.export_scene.gltf(filepath=str(out / 'stellaris_heart.glb'), export_format='GLB', export_animations=True, export_animation_mode='ACTIONS', export_yup=True)
    report['glbSha256'] = hashlib.sha256((out / 'stellaris_heart.glb').read_bytes()).hexdigest()
    (out / 'conversion.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--length', type=float, default=11.129903936386109)
    build(parser.parse_args(sys.argv[sys.argv.index('--') + 1:]))
