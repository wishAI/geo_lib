"""Small derived 3D recordings. Original simulation evidence is never rewritten."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

ALG = Path(__file__).resolve().parent
ANGLE_STEP = 1e-4


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pack_motion(q, times, joints):
    q, times = np.asarray(q), np.asarray(times)
    if (q.ndim != 2 or len(q) < 2 or q.shape[1] != 7+len(joints) or
            times.shape != (len(q),) or not np.isfinite(q).all() or
            not np.isfinite(times).all() or np.any(np.diff(times)<=0)):
        raise ValueError('Invalid complete motion trajectory')
    if not np.allclose(np.linalg.norm(q[:,3:7],axis=1),1.,atol=1e-4):
        raise ValueError('Invalid root quaternion')
    chunks, columns, size = [], [], 0
    restored = np.empty(q.shape)
    for i in range(q.shape[1]):
        values = q[:,i]
        lo, hi = float(values.min()), float(values.max())
        record = {'qpos':i}
        if i>=7 and hi-lo<=ANGLE_STEP:
            record['constant'] = (lo+hi)/2
            restored[:,i] = record['constant']
        else:
            if i<7:
                encoded = values.astype('<f4')
                record.update(dtype='f32',offset=0.,scale=1.)
            else:
                raw = np.rint((values-lo)/ANGLE_STEP)
                dtype = '<u2' if raw.max()<=65535 else '<u4'
                encoded = raw.astype(dtype)
                record.update(dtype='u16' if dtype=='<u2' else 'u32',offset=lo,scale=ANGLE_STEP)
            data = encoded.tobytes()
            record.update(byteOffset=size,byteLength=len(data))
            chunks.append(data);size+=len(data)
            restored[:,i] = encoded.astype(float)*record['scale']+record['offset']
        columns.append(record)
    dt = float(np.median(np.diff(times)))
    clock = {'start':float(times[0]),'step':dt}
    if not np.allclose(times,times[0]+np.arange(len(times))*dt,atol=1e-8,rtol=0):
        data = times.astype('<f8').tobytes()
        clock = {'byteOffset':size,'byteLength':len(data),'dtype':'f64'}
        chunks.append(data);size+=len(data)
    error = np.max(np.abs(restored-q),axis=0)
    if max(error[7:],default=0)>ANGLE_STEP/2+1e-12:
        raise ValueError('Joint compression exceeded declared error budget')
    metadata = dict(format='landau-walk-motion',version=1,frames=len(q),qposWidth=q.shape[1],
                    duration=float(times[-1]-times[0]),coordinates='Z-up,meters',
                    quaternion='wxyz',joints=joints,clock=clock,columns=columns,byteLength=size,
                    error=dict(maxJointRadians=float(max(error[7:],default=0)),
                               maxRootComponent=float(max(error[:7]))),
                    purpose='Derived interactive preview; original trajectory remains acceptance evidence')
    return metadata,b''.join(chunks)


def export_motion(directory):
    import mujoco
    directory = Path(directory).resolve()
    directory.relative_to((ALG/'outputs').resolve())
    result = json.loads((directory/'dynamics.json').read_text())
    if sha(directory/'trajectory.npz') != result['trajectory_sha256']:
        raise ValueError('Original trajectory hash mismatch')
    model = mujoco.MjModel.from_xml_path(str(directory/'model.xml'))
    joints = sorted([dict(name=model.joint(i).name,qpos=int(model.jnt_qposadr[i]))
                     for i in range(model.njnt) if model.jnt_type[i]!=mujoco.mjtJoint.mjJNT_FREE],
                    key=lambda j:j['qpos'])
    if [j['qpos'] for j in joints] != list(range(7,model.nq)):
        raise ValueError('Recording requires free root and scalar joints')
    with np.load(directory/'trajectory.npz') as trace:
        meta,raw = pack_motion(trace['qpos'],trace['time'],joints)
    profile = result['config'].get('model_profile', '')
    if not profile and result['config'].get('checkpoint'):
        training = json.loads((Path(result['config']['checkpoint']).parent/'training.json').read_text())
        if training['urdf_sha256'] != result['identity']['urdf_sha256']:
            raise ValueError('Recording checkpoint model mismatch')
        profile = training.get('model_profile', training.get('arguments',{}).get('profile',''))
    forward = result.get('semantic_forward_native', [0,-1,0] if profile in
                         ('anatomical_legs_v1','anatomical_legs_mass_v2') else [0,1,0])
    meta.update(semanticForwardNative=forward, modelProfile=profile or 'retained_v10')
    packed = gzip.compress(raw,compresslevel=9,mtime=0)
    payload = directory/'motion.bin.gz'
    payload.write_bytes(packed)
    meta.update(payload=payload.name,payloadSha256=sha(payload),compressedBytes=len(packed),
                originalTrajectoryBytes=(directory/'trajectory.npz').stat().st_size,
                source=dict(trajectorySha256=sha(directory/'trajectory.npz'),
                            modelXmlSha256=sha(directory/'model.xml'),
                            urdfSha256=result['identity']['urdf_sha256'],
                            meshTreeSha256=result['identity']['mesh_tree_sha256'],
                            checkpointSha256=result['config'].get('checkpoint_sha256')))
    (directory/'motion.json').write_text(json.dumps(meta,separators=(',',':'))+'\n')
    return meta


def export_scene(usd_source, *, urdf_path=None, output=None, specification=None):
    """File handoff from the prepared USD display asset; URDF meshes stay exact."""
    from . import model_spec
    urdf = Path(urdf_path or model_spec.URDF_PATH).resolve()
    spec = specification or model_spec.build_robot_spec(urdf)
    records = {j['name']: j for j in spec['joints']}
    tree = ET.parse(urdf).getroot()
    def vector(element,key,default):
        return [float(x) for x in element.get(key,default).split()]
    def origin(element):
        item = element.find('origin')
        if item is None:item=ET.Element('origin')
        return dict(xyz=vector(item,'xyz','0 0 0'),rpy=vector(item,'rpy','0 0 0'))
    joints=[]
    for j in tree.findall('joint'):
        axis=j.find('axis');limit=j.find('limit')
        joints.append(dict(name=j.get('name'),parent=j.find('parent').get('link'),
                           child=j.find('child').get('link'),type=j.get('type'),
                           axis=vector(axis,'xyz','1 0 0') if axis is not None else [1,0,0],
                           limits={k:float(v) for k,v in limit.attrib.items()} if limit is not None else {},
                           control=records.get(j.get('name'),{}).get('control','fixed'),
                           role=records.get(j.get('name'),{}).get('role','fixed'),**origin(j)))
    meshes={};links=[]
    for link in tree.findall('link'):
        visuals=[]
        for visual in link.findall('visual'):
            mesh=visual.find('geometry/mesh')
            if mesh is None:continue
            name=mesh.get('filename');path=(urdf.parent/name).resolve()
            path.relative_to((ALG/'inputs').resolve())
            if name not in meshes:
                data=path.read_bytes();count=int.from_bytes(data[80:84],'little')
                if len(data)!=84+50*count:raise ValueError('Expected binary STL')
                triangles=np.frombuffer(data,offset=84,dtype=np.dtype([('normal','<f4',3),('vertices','<f4',(3,3)),('attribute','<u2')]))
                meshes[name]=triangles['vertices'].reshape(-1).tolist()
            visuals.append(dict(mesh=name,scale=vector(mesh,'scale','1 1 1'),**origin(visual)))
        links.append(dict(name=link.get('name'),visuals=visuals))
    output=Path(output or ALG/'outputs/playback');output.mkdir(parents=True,exist_ok=True)
    target=ALG/'inputs/playback_character/character.glb';target.parent.mkdir(parents=True,exist_ok=True)
    usd_source=Path(usd_source).resolve()
    if target.exists() and sha(target)!=sha(usd_source):raise ValueError('Existing USD handoff differs')
    target.write_bytes(usd_source.read_bytes())
    scene=dict(format='landau-character-scene',version=1,joints=joints,links=links,meshes=meshes,
               poses={'nominal':spec['nominal_pose']['joint_positions_rad'],
                      'target':spec['nominal_pose']['joint_position_targets_rad']},
               model={'actionCount':len(spec['action_joints']), 'nominalLabel':'模型定义的初始姿态',
                      'profile':spec.get('model_profile','retained_v10'),
                      'forward':spec.get('variant',{}).get('semantic_forward_native',[0,1,0])},
               source=dict(urdfSha256=sha(urdf),usdGlbSha256=sha(target),usdHandoff=str(usd_source)))
    (output/'scene.json.gz').write_bytes(gzip.compress(json.dumps(scene,separators=(',',':')).encode(),9,mtime=0))
    return output/'scene.json.gz'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path)
    parser.add_argument('--usd-source',type=Path)
    args=parser.parse_args()
    if args.usd_source:print(export_scene(args.usd_source))
    if args.directory:
        meta=export_motion(args.directory)
        print(json.dumps({k:meta[k] for k in ('frames','duration','compressedBytes','originalTrajectoryBytes','error')}))
    if not args.usd_source and not args.directory:parser.error('Select recording directory or USD file handoff')


if __name__=='__main__':main()
