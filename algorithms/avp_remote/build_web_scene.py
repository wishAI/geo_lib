"""Prepare the current file handoff and export USD skin/materials with Blender.
Run: blender --background --factory-startup --python algorithms/avp_remote/build_web_scene.py
"""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_assets():
    from asset_paths import default_landau_source_urdf, default_landau_source_usd
    path = ROOT / 'outputs/web_scene/scene.json'
    if not path.exists():
        return {'ready': False, 'reason': 'Browser scene has not been prepared'}
    source = json.loads(path.read_text())['source']
    urdf = default_landau_source_urdf()
    checks = [(urdf, source['urdfSha256']), (default_landau_source_usd(), source['usdSha256']),
              (ROOT / 'outputs/web_scene/character.glb', source['glbSha256'])]
    checks.extend((urdf.parent / name, sha) for name, sha in source['meshHashes'].items())
    checks.extend((ROOT.parents[1] / name, sha) for name, sha in source.get('dependencyHashes', {}).items())
    changed = [str(p) for p, sha in checks if not p.is_file() or digest(p) != sha]
    return {'ready': not changed, 'changed': changed, 'urdfSha256': source['urdfSha256']}


def build():
    import bpy
    from asset_setup import prepare_landau_inputs
    from asset_paths import default_landau_source_urdf
    from landau_pose import load_stl_mesh_arrays
    from landau_retarget import LandauUpperBodyRetargeter, AVP_TO_SCENE_OPTIONS
    from avp_tracking_schema import extract_tracking_frame
    import xml.etree.ElementTree as ET

    prepared = prepare_landau_inputs()
    out = ROOT / 'outputs/web_scene'
    out.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.wm.usd_import(filepath=str(prepared.usd_path), import_cameras=False, import_lights=False)
    rigs = [o for o in bpy.context.scene.objects if o.type == 'ARMATURE']
    if len(rigs) != 1:
        raise RuntimeError(f'Expected one USD skeleton, got {len(rigs)}')
    rig = rigs[0]
    rig.animation_data_clear()
    rig.data.pose_position = 'REST'
    for bone in rig.pose.bones:
        bone.matrix_basis.identity()
    meshes = [o for o in bpy.context.scene.objects if o.type == 'MESH']
    if not meshes or not any(o.vertex_groups for o in meshes):
        raise RuntimeError('USD import lost skinned geometry')
    # EXR is not a browser texture format. Convert in memory before glTF export.
    for image in bpy.data.images:
        if image.source == 'FILE' and image.size[0]:
            if max(image.size) > 1024:
                ratio = 1024 / max(image.size)
                image.scale(max(1, round(image.size[0] * ratio)), max(1, round(image.size[1] * ratio)))
            image.file_format = 'PNG'
            image.pack()
    bpy.ops.object.select_all(action='DESELECT')
    for o in [rig, *meshes]:
        o.select_set(True)
    bpy.ops.export_scene.gltf(filepath=str(out / 'character.glb'), export_format='GLB',
        use_selection=True, export_yup=False, export_animations=False, export_skins=True,
        export_image_format='AUTO', export_extras=True)

    xml = ET.parse(prepared.urdf_path).getroot()
    def vector(node, attr, default):
        return [float(v) for v in node.get(attr, default).split()] if node is not None else [float(v) for v in default.split()]
    def origin(node):
        return {'xyz': vector(node, 'xyz', '0 0 0'), 'rpy': vector(node, 'rpy', '0 0 0')}
    links, joints, mesh_data, hashes = [], [], {}, {}
    for link in xml.findall('link'):
        visuals = []
        for visual in link.findall('visual'):
            mesh = visual.find('geometry/mesh')
            if mesh is None:
                continue
            uri = mesh.get('filename')
            path = prepared.urdf_path.parent / uri
            if uri not in mesh_data:
                vertices, _, _ = load_stl_mesh_arrays(path)
                mesh_data[uri] = vertices.reshape(-1).round(8).tolist()
                hashes[uri] = digest(path)
            visuals.append({'mesh': uri, 'scale': vector(mesh, 'scale', '1 1 1'), **origin(visual.find('origin'))})
        links.append({'name': link.get('name'), 'visuals': visuals})
    for joint in xml.findall('joint'):
        limit = joint.find('limit')
        joints.append({'name': joint.get('name'), 'child': joint.find('child').get('link'),
            'parent': joint.find('parent').get('link'), 'type': joint.get('type'),
            'axis': vector(joint.find('axis'), 'xyz', '1 0 0'),
            'lower': float(limit.get('lower', '-3.14159')) if limit is not None else 0,
            'upper': float(limit.get('upper', '3.14159')) if limit is not None else 0,
            **origin(joint.find('origin'))})
    snapshot = json.loads((ROOT / 'avp_snapshot.json').read_text())
    retarget = LandauUpperBodyRetargeter(urdf_path=prepared.urdf_path,
        skeleton_json_path=prepared.skeleton_json_path, snapshot_path=ROOT / 'avp_snapshot.json', use_trac_ik=False)
    pose = retarget.retarget_frame(extract_tracking_frame(snapshot))
    from asset_paths import default_landau_source_skeleton_json, default_landau_source_texture_dir
    dependencies = [default_landau_source_skeleton_json(), ROOT / 'avp_snapshot.json',
        ROOT / 'landau_retarget.py', ROOT / 'landau_mapping_config.py', ROOT / 'avp_config.py',
        *[p for p in default_landau_source_texture_dir().rglob('*') if p.is_file()]]
    payload = {'version': 1, 'links': links, 'joints': joints, 'meshes': mesh_data,
        'snapshot': snapshot, 'snapshotPose': pose,
        'trackingTransform': AVP_TO_SCENE_OPTIONS.pretransform.tolist(),
        'source': {'urdf': str(default_landau_source_urdf().relative_to(ROOT.parents[1])),
            'urdfSha256': digest(prepared.urdf_path), 'usdSha256': digest(prepared.usd_path),
            'meshHashes': hashes, 'glbSha256': digest(out / 'character.glb'),
            'dependencyHashes': {str(p.relative_to(ROOT.parents[1])): digest(p) for p in dependencies},
            'bones': len(rig.data.bones), 'usdMeshes': len(meshes),
            'triangles': sum(len(v)//9 for v in mesh_data.values()),
            'textureMaxSize': 1024,
            'conversion': 'Blender USD import → glTF 2.0 skin + embedded PNG materials; no physics'}}
    (out / 'scene.json').write_text(json.dumps(payload, separators=(',', ':'), allow_nan=False))
    print(json.dumps(payload['source'], indent=2))

if __name__ == '__main__':
    if '--check' in sys.argv:
        print(json.dumps(check_assets()))
    else:
        build()
