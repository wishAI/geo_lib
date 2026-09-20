"""Canonical copied URDF, SI-unit FK and mesh geometry. No simulator dependency."""
import shutil
import xml.etree.ElementTree as ET
import hashlib
import numpy as np
from scipy.spatial.transform import Rotation
import trimesh
from state import ROOT, OUT, REPO, sha256, write_json

URDF = ROOT / 'inputs/landau_v10/landau_v10_parallel_mesh.urdf'
FINGERS = ('thumb', 'index', 'middle', 'ring', 'pinky')


def vector(element, key, default='0 0 0'):
    return np.fromstring(element.get(key, default) if element is not None else default, sep=' ')


def origin(element):
    t = np.eye(4)
    t[:3, :3] = Rotation.from_euler('xyz', vector(element, 'rpy')).as_matrix()
    t[:3, 3] = vector(element, 'xyz')
    return t


def copy_assets(source):
    source = __import__('pathlib').Path(source)
    URDF.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source / URDF.name, URDF)
    for p in sorted((source / 'mesh_collision_stl').rglob('*.stl')):
        dst = URDF.parent / p.relative_to(source)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, dst)  # dereference links; this sandbox owns real files
    return Robot().audit()


class Robot:
    def __init__(self, path=URDF):
        self.path = path
        self.xml = ET.parse(path).getroot()
        self.links = [e.get('name') for e in self.xml.findall('link')]
        raw = []
        for e in self.xml.findall('joint'):
            limit = e.find('limit')
            raw.append(dict(name=e.get('name'), type=e.get('type'),
                            parent=e.find('parent').get('link'), child=e.find('child').get('link'),
                            origin=origin(e.find('origin')), axis=vector(e.find('axis'), 'xyz', '1 0 0'),
                            lower=float(limit.get('lower', 0)) if limit is not None else 0.,
                            upper=float(limit.get('upper', 0)) if limit is not None else 0.,
                            velocity=float(limit.get('velocity', 0)) if limit is not None else 0.))
        self.joints, seen = [], {'base_link'}
        while raw:
            ready = [j for j in raw if j['parent'] in seen]
            if not ready:
                raise ValueError('Disconnected/cyclic URDF')
            for j in ready:
                self.joints.append(j); seen.add(j['child']); raw.remove(j)
        self.moving = [j for j in self.joints if j['type'] == 'revolute']
        self.names = [j['name'] for j in self.moving]
        self.active = [i for i, n in enumerate(self.names) if not any(f in n for f in FINGERS)]
        self.locked = [i for i in range(len(self.names)) if i not in self.active]
        self.lower = np.array([j['lower'] for j in self.moving])
        self.upper = np.array([j['upper'] for j in self.moving])
        self.speed = np.array([j['velocity'] for j in self.moving])
        self.index = {n: i for i, n in enumerate(self.names)}
        self.meshes = []
        for link in self.xml.findall('link'):
            for visual in link.findall('visual'):
                m = visual.find('geometry/mesh')
                if m is None:
                    continue
                mesh = trimesh.load_mesh(path.parent / m.get('filename'), process=False)
                verts = np.array(mesh.vertices) * vector(m, 'scale', '1 1 1')
                local = origin(visual.find('origin'))
                verts = verts @ local[:3, :3].T + local[:3, 3]
                self.meshes.append((link.get('name'), verts, np.array(mesh.faces), m.get('filename')))
        self.link_index = {n: i for i, n in enumerate(self.links)}

    def fk(self, q, base=None):
        transforms = {'base_link': np.eye(4) if base is None else base}
        for j in self.joints:
            t = j['origin'].copy()
            if j['type'] == 'revolute':
                axis = j['axis'] / np.linalg.norm(j['axis'])
                t[:3, :3] = t[:3, :3] @ Rotation.from_rotvec(axis * q[self.index[j['name']]]).as_matrix()
            transforms[j['child']] = transforms[j['parent']] @ t
        return transforms

    def vertices(self, transforms):
        return [(link, verts @ transforms[link][:3, :3].T + transforms[link][:3, 3], faces)
                for link, verts, faces, _ in self.meshes]

    def audit(self):
        import json
        pins = json.loads((ROOT / 'provenance.json').read_text())
        paths = sorted(set(self.path.parent / m.get('filename') for m in self.xml.findall('.//mesh')))
        h = hashlib.sha256()
        for p in paths:
            h.update(p.name.encode()); h.update(bytes.fromhex(sha256(p)))
        if sha256(self.path) != pins['urdf_sha256'] or h.hexdigest() != pins['mesh_tree_sha256'] or len(paths) != 68:
            raise ValueError('Canonical asset hash/count mismatch')
        rest = self.fk(np.zeros(len(self.names)))
        vertices = np.concatenate([v for _, v, _ in self.vertices(rest)])
        data = {'urdf_sha256': sha256(self.path), 'mesh_tree_sha256': h.hexdigest(), 'mesh_count':len(paths),
                'triangle_count': sum(len(f) for _, _, f, _ in self.meshes), 'link_count': len(self.links),
                'revolute_count':len(self.names), 'action_joints':[self.names[i] for i in self.active],
                'locked_joints':[self.names[i] for i in self.locked], 'lock_position_rad':0,
                'lock_reason':'Fingers held at zero. All other URDF revolutes, including shin twists for foot orientation, enabled for animation; independent of walking policy actions.',
                'units':{'xyz':'metres','angles':'radians','speed':'rad/s','mesh_scale':'URDF explicit scale or 1'},
                'world_axes':'+Z up; canonical mounted mesh/anatomical forward is native -Y; application forward +Y requires explicit retarget frame transport', 'root_x_mount':rest['root_x'].tolist(),
                'root_note':'root_x +90 degree roll preserved in FK. Base yaw operates around world Z; never treat root_x local Y as body heading.',
                'rest_bounds_m':[vertices.min(0).tolist(),vertices.max(0).tolist()],
                'joints':[{k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in j.items()} for j in self.joints],
                'rest_link_positions_m':{k:v[:3,3].tolist() for k,v in rest.items()},
                'files':[{'path':str(p.relative_to(REPO)), 'sha256':sha256(p),'size':p.stat().st_size} for p in paths]}
        write_json(OUT/'asset_audit.json', data)
        return data
