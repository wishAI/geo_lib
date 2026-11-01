"""Export the existing H1-2 comparison model and retargeting rules for the browser."""
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parent

def build(out, landau_joints):
    import bpy
    from mathutils import Euler
    from asset_paths import default_h1_2_urdf_path
    from avp_g1_pose import map_landau_pose_to_h1_2_pose, estimate_urdf_root_height
    source = default_h1_2_urdf_path()
    xml = ET.parse(source).getroot()
    bpy.ops.wm.read_factory_settings(use_empty=True)
    def vec(node, attr, default):
        return list(map(float, (node.get(attr, default) if node is not None else default).split()))
    def origin(node):
        return dict(xyz=vec(node,'xyz','0 0 0'), rpy=vec(node,'rpy','0 0 0'))
    hashes = {str(source.relative_to(ROOT.parents[1])): hashlib.sha256(source.read_bytes()).hexdigest()}
    links = []
    for link in xml.findall('link'):
        name = link.get('name')
        group = bpy.data.objects.new('unitree_' + name, None)
        bpy.context.collection.objects.link(group)
        links.append(name)
        for visual in link.findall('visual'):
            mesh = visual.find('geometry/mesh')
            if mesh is None:
                continue
            path = source.parent / mesh.get('filename')
            hashes[str(path.relative_to(ROOT.parents[1]))] = hashlib.sha256(path.read_bytes()).hexdigest()
            bpy.ops.wm.stl_import(filepath=str(path))
            obj = bpy.context.object
            obj.parent = group
            tr = origin(visual.find('origin'))
            obj.location = tr['xyz']
            obj.rotation_euler = Euler(tr['rpy'], 'XYZ')
            obj.scale = vec(mesh,'scale','1 1 1')
            if len(obj.data.polygons) > 800:
                mod = obj.modifiers.new('Browser detail', 'DECIMATE')
                mod.ratio = 800 / len(obj.data.polygons)
                bpy.ops.object.modifier_apply(modifier=mod.name)
            material = bpy.data.materials.new(name)
            material.diffuse_color = vec(visual.find('material/color'),'rgba','.65 .67 .7 1')
            material.use_nodes = True
            bsdf = material.node_tree.nodes.get('Principled BSDF')
            bsdf.inputs['Base Color'].default_value = material.diffuse_color
            bsdf.inputs['Roughness'].default_value = .6
            obj.data.materials.append(material)
    joints = []
    for j in xml.findall('joint'):
        limit, mimic = j.find('limit'), j.find('mimic')
        row = dict(name=j.get('name'), parent=j.find('parent').get('link'), child=j.find('child').get('link'), type=j.get('type'), axis=vec(j.find('axis'),'xyz','1 0 0'), **origin(j.find('origin')),
                   lower=float(limit.get('lower','-3.14159')) if limit is not None else 0,
                   upper=float(limit.get('upper','3.14159')) if limit is not None else 0)
        if mimic is not None:
            row['mimic'] = dict(joint=mimic.get('joint'), multiplier=float(mimic.get('multiplier','1')), offset=float(mimic.get('offset','0')))
        joints.append(row)
    baseline = map_landau_pose_to_h1_2_pose({})
    rules = {name: dict(bias=value, weights={}) for name,value in baseline.items()}
    # Export coefficients from the authoritative Python mapper, not a second set of constants.
    for j in landau_joints:
        key = j['child']
        sample = map_landau_pose_to_h1_2_pose({key: .0001})
        for name,value in sample.items():
            coefficient = (value - baseline[name]) / .0001
            if abs(coefficient) > 1e-9:
                rules[name]['weights'][key] = round(coefficient, 10)
    for name in rules:
        if name.endswith('elbow_pitch_joint'):
            rules[name]['minimum'] = .1
    bpy.ops.export_scene.gltf(filepath=str(out / 'unitree.glb'), export_format='GLB', export_yup=False, export_animations=False,
        export_draco_mesh_compression_enable=True, export_draco_mesh_compression_level=6,
        export_draco_position_quantization=16, export_draco_normal_quantization=10)
    hashes[str(Path(__file__).relative_to(ROOT.parents[1]))] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    mapper = ROOT / 'avp_g1_pose.py'
    hashes[str(mapper.relative_to(ROOT.parents[1]))] = hashlib.sha256(mapper.read_bytes()).hexdigest()
    payload = dict(name='Unitree H1-2', root='pelvis', rootHeight=estimate_urdf_root_height(source, baseline), links=links, joints=joints, rules=rules, sourceHashes=hashes, glbSha256=hashlib.sha256((out/'unitree.glb').read_bytes()).hexdigest())
    (out/'unitree.json').write_text(json.dumps(payload, separators=(',',':')))
    return payload
