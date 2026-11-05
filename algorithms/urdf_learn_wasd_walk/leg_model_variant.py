"""Versioned anatomical leg experiment; original assets and checkpoints stay intact.

Topology reference: official Unitree G1 has hip 3 + knee 1 + ankle 2 DOFs.
https://github.com/unitreerobotics/unitree_ros/blob/master/robots/g1_description/g1_23dof.urdf
Landau limits and frames below derive from its own FK, not G1 dimensions.
"""
import argparse
import copy
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from . import model_spec as source

PROFILE = 'anatomical_legs_v1'
DIRECTORY = source.ALGORITHM_ROOT/'inputs'/PROFILE
URDF = DIRECTORY/'landau.urdf'
MASS_PROFILE = 'anatomical_legs_mass_v2'
TARGET_MASS_KG = 18.0  # Approximate small animation character, including large head.


def create_variant(profile=PROFILE):
    if profile not in (PROFILE,MASS_PROFILE):raise ValueError('Unknown anatomical profile')
    directory=source.ALGORITHM_ROOT/'inputs'/profile
    urdf=directory/'landau.urdf'
    tree=ET.parse(source.URDF_PATH);root=tree.getroot()
    if source._sha256(source.URDF_PATH)!=source.EXPECTED_URDF_SHA256:
        raise ValueError('Retained source identity changed')
    worlds,_=source._joint_world_transforms(root)
    joints={j.get('name'):j for j in root.findall('joint')}
    def axis_in_frame(j,world_axis):
        frame=source._compose(worlds[j.find('parent').get('link')],source._origin_transform(j.find('origin')))
        return np.asarray(frame[0]).T@world_axis
    def vector(v):return ' '.join(f'{float(x):.14g}' for x in v)
    changes=[]
    for side,suffix in (('left','l'),('right','r')):
        for role,axis in [('hip_roll',[0,1,0]),('hip_yaw',[0,0,1])]:
            j=joints[f'{side}_{role}_joint'];j.find('axis').set('xyz',vector(axis_in_frame(j,axis)))
        joints[f'{side}_knee_joint'].find('limit').set('lower','-0.25')
        joints[f'{side}_knee_joint'].find('limit').set('upper','2.0')
        joints[f'{side}_shin_roll_joint'].find('limit').set('lower','-0.15')
        joints[f'{side}_shin_roll_joint'].find('limit').set('upper','0.15')
        pitch=joints[f'{side}_ankle_pitch_joint'];foot=pitch.find('child').get('link')
        intermediate=f'{side}_ankle_pitch_link';pitch.find('child').set('link',intermediate)
        link=ET.SubElement(root,'link',name=intermediate)
        foot_inertial=next(l for l in root.findall('link') if l.get('name')==foot).find('inertial')
        inertial=copy.deepcopy(foot_inertial);link.append(inertial)
        # Identical frames and COM: partition 1% into the virtual articulation
        # body while preserving the original composite inertia at zero angle.
        for element,fraction in ((inertial,.01),(foot_inertial,.99)):
            mass=element.find('mass');mass.set('value',str(float(mass.get('value'))*fraction))
            tensor=element.find('inertia')
            for key,value in tensor.attrib.items():tensor.set(key,str(float(value)*fraction))
        roll=ET.SubElement(root,'joint',name=f'{side}_ankle_roll_joint',type='revolute')
        ET.SubElement(roll,'parent',link=intermediate);ET.SubElement(roll,'child',link=foot)
        ET.SubElement(roll,'origin',xyz='0 0 0',rpy='0 0 0')
        ET.SubElement(roll,'axis',xyz=vector(np.asarray(worlds[foot][0]).T@np.array([0,1,0])))
        ET.SubElement(roll,'limit',lower='-0.35',upper='0.35',effort='50',velocity='6')
        ET.SubElement(roll,'dynamics',damping='1',friction='0.05')
        changes.append({'side':side,'new_joint':roll.get('name'),'world_axis_zero':[0,1,0],
                        'knee_limits_rad':[-.25,2.0],'knee_anatomical_flexion_at_zero_rad':.25353})
    # Density scaling is explicit and scales inertia, passive friction and effort.
    # Preserve per-link distribution for this isolated articulation experiment.
    total=sum(float(m.get('value')) for m in root.findall('./link/inertial/mass'))
    scale=TARGET_MASS_KG/total
    def mass_group(name):
        if name.startswith(('thumb','index','middle','ring','pinky')):return 'fingers'
        if name in ('head_x','neck_x'):return 'head_neck'
        if name.startswith(('thigh','leg','foot','toes')) or 'ankle_' in name or 'hip_' in name:return 'legs'
        if name.startswith(('arm','forearm','hand','shoulder')):return 'arms'
        return 'torso'
    estimates={'head_neck':4.5,'torso':6.,'legs':5.4,'arms':1.8,'fingers':.3}
    group_mass={group:0. for group in estimates}
    for link in root.findall('link'):
        if link.find('inertial') is not None:group_mass[mass_group(link.get('name'))]+=float(link.find('inertial/mass').get('value'))
    for link in root.findall('link'):
        inertial=link.find('inertial')
        if inertial is None:continue
        link_scale=estimates[mass_group(link.get('name'))]/group_mass[mass_group(link.get('name'))] if profile==MASS_PROFILE else scale
        mass=inertial.find('mass');mass.set('value',str(float(mass.get('value'))*link_scale))
        inertia=inertial.find('inertia')
        for key,value in inertia.attrib.items():inertia.set(key,str(float(value)*link_scale))
    for joint in root.findall('joint'):
        limit=joint.find('limit')
        if limit is not None:limit.set('effort',str(float(limit.get('effort'))*scale))
        dynamics=joint.find('dynamics')
        if dynamics is not None:
            for key,value in dynamics.attrib.items():dynamics.set(key,str(float(value)*scale))
    for mesh in root.findall('.//mesh'):
        mesh.set('filename','../landau_v10/'+mesh.get('filename'))
    after,_=source._joint_world_transforms(root)
    error=max(float(np.max(np.abs(np.r_[np.asarray(after[n][0]).ravel(),after[n][1]]-np.r_[np.asarray(t[0]).ravel(),t[1]]))) for n,t in worlds.items())
    if error>1e-12:raise ValueError('Variant changed existing zero-pose bind frames')
    directory.mkdir(parents=True,exist_ok=True);ET.indent(tree);tree.write(urdf,encoding='utf-8',xml_declaration=True)
    audit={'profile':profile,'source_urdf_sha256':source._sha256(source.URDF_PATH),'urdf_sha256':source._sha256(urdf),
           'mass_kg':TARGET_MASS_KG,'density_scale':scale,'mass_groups_kg':estimates if profile==MASS_PROFILE else {k:v*scale for k,v in group_mass.items()},'mass_estimate':'Approximate animation character; not a measured biological mass',
           'semantic_forward_native':[0,-1,0],'legacy_semantic_forward_native':[0,1,0],
           'zero_bind_max_error':error,'changes':changes,'policy_actions':19,'inherited_milestone_passes':0,
           'source_reference':'https://github.com/unitreerobotics/unitree_ros/blob/master/robots/g1_description/g1_23dof.urdf'}
    (directory/'variant.json').write_text(json.dumps(audit,indent=2)+'\n');return audit


def specification(profile=PROFILE):
    if profile not in (PROFILE,MASS_PROFILE):raise ValueError('Unknown anatomical profile')
    directory=source.ALGORITHM_ROOT/'inputs'/profile
    urdf=directory/'landau.urdf'
    audit=json.loads((directory/'variant.json').read_text())
    if source._sha256(urdf)!=audit['urdf_sha256']:raise ValueError('Variant identity changed')
    spec=source.build_robot_spec(urdf)
    spec['model_profile']=profile;spec['variant']=audit;spec['hand_pose']='closed_fists_v1'
    spec['urdf_path']=str(urdf);spec['lineage']=profile+'_20261104'
    spec['source'].update(expected_urdf_sha256=audit['urdf_sha256'],derived_urdf_used=True)
    spec['frames'].update(body_forward_axis='-Y',body_strafe_axis='+X',
                          mapping={'linear_x':'strafe','linear_y':'-forward','angular_z':'yaw'})
    for side in ('left','right'):
        name=side+'_ankle_roll_joint';spec['action_joints'].append(name);spec['locked_joints'].remove(name)
    for r in spec['joints']:
        if r['name'].endswith('_ankle_roll_joint'):
            r.update(control='action',role=r['name'].removesuffix('_joint'),nominal_stiffness_nm_per_rad=12.,nominal_damping_nm_s_per_rad=.8)
        for k in ('nominal_stiffness_nm_per_rad','nominal_damping_nm_s_per_rad'):r[k]*=audit['density_scale']
    # Same authored non-finger seed; fixed fists and added ankle axes need fresh M1.
    pose={**spec['nominal_pose']['joint_positions_rad'],**source.fixed_hand_positions()}
    targets={**spec['nominal_pose']['joint_position_targets_rad'],**source.fixed_hand_positions()}
    spec['nominal_pose'].update(joint_positions_rad=pose,joint_position_targets_rad=targets,
                              geometry=source.analyze_pose_geometry(pose,urdf))
    for r in spec['joints']:
        r['nominal_position_rad']=pose[r['name']];r['nominal_target_rad']=targets[r['name']]
    return spec


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--export-scene',action='store_true')
    p.add_argument('--passive-name');p.add_argument('--profile',choices=(PROFILE,MASS_PROFILE),default=PROFILE);args=p.parse_args()
    if args.passive_name:
        from types import SimpleNamespace
        from . import mujoco_backend as backend
        spec=specification(args.profile);original=backend.build_model
        backend.build_model=lambda **kw:original(urdf_path=spec['urdf_path'],specification=spec,**kw)
        backend.run(SimpleNamespace(name=args.passive_name,backend='mujoco_warp_cuda',seconds=30.,
            dt=.002,seed=42,pose='geometric',gain_scale=1.,noslip_iterations=0,contact_timeconst=.004,
            assistance=0.,checkpoint=None,forward=0.,render=True,support_profile='animation_approximate_v1',model_profile=args.profile))
        from .motion_recording import export_motion
        export_motion(backend.OUTPUT/args.passive_name)
        return
    audit=create_variant(args.profile);spec=specification(args.profile)
    if args.export_scene:
        from .motion_recording import export_scene
        export_scene(source.ALGORITHM_ROOT/'inputs/playback_character/character.glb',urdf_path=spec['urdf_path'],
                     output=source.ALGORITHM_ROOT/'outputs/playback'/args.profile,specification=spec)
    print(json.dumps(audit,indent=2))

if __name__=='__main__':main()
