"""Controlled contact and body-inertia comparisons, never milestone certification.

A rigid sole is an explicit massless collision proxy spanning heel and toe at
nominal stance; articulated toe mass and motors remain. Contact mechanics:
https://mujoco.readthedocs.io/en/3.5.0/modeling.html#contact-parameters
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import xml.etree.ElementTree as ET


def head_inertia(model, xml, *, centroid_only=False):
    """Declared uniform-solid head estimate; preserve every body's mass.

    Other visual partitions are not reliable physical volumes. In particular
    the tiny neck fragment does not justify moving its existing mass.
    https://trimesh.org/trimesh.base.html#trimesh.base.Trimesh.mass_properties
    """
    import mujoco
    import numpy as np
    import trimesh
    from .mujoco_backend import numbers, digest
    root=ET.fromstring(xml);body=root.find(".//body[@name='head_x']")
    geom=body.find('geom');asset=root.find("./asset/mesh[@name='"+geom.get('mesh')+"']")
    if (not np.allclose(np.fromstring(geom.get('pos','0 0 0'),sep=' '),0) or
        not np.allclose(np.fromstring(geom.get('quat','1 0 0 0'),sep=' '),[1,0,0,0])):
        raise ValueError('Head estimate requires mesh aligned with link coordinates')
    path=Path(asset.get('file'));mesh=trimesh.load_mesh(path,process=True)
    mesh.apply_scale(np.fromstring(asset.get('scale','1 1 1'),sep=' '))
    if not mesh.is_watertight or not mesh.is_winding_consistent or mesh.volume<=0:
        raise ValueError('Head mesh does not define a positive closed volume')
    mass=float(model.body('head_x').mass[0]);tensor=mesh.moment_inertia*(mass/mesh.volume)
    if np.linalg.eigvalsh(tensor).min()<=0:raise ValueError('Invalid central inertia')
    inertial=body.find('inertial');inertial.set('pos',numbers(mesh.center_mass))
    if not centroid_only:
        inertial.attrib.pop('diaginertia',None);inertial.attrib.pop('quat',None)
        inertial.set('fullinertia',numbers([*np.diag(tensor),tensor[0,1],tensor[0,2],tensor[1,2]]))
    source=ET.tostring(root,encoding='unicode');changed=mujoco.MjModel.from_xml_string(source)
    if not np.array_equal(changed.body_mass,model.body_mass):raise ValueError('Head estimate changed body masses')
    return changed,source,{'assumption':'uniform solid head only; unchanged authored mass allocation',
        'centroid_only':centroid_only,'mesh_sha256':digest(path),'volume_m3':float(mesh.volume),
        'center_m':mesh.center_mass.tolist(),'estimated_uniform_central_inertia_kg_m2':tensor.tolist(),
        'applied_principal_inertia_kg_m2':changed.body('head_x').inertia.tolist(),
        'applied_inertial_quaternion_wxyz':changed.body('head_x').iquat.tolist()}


def continuous_sole(model, spec, xml, *, flat=True):
    import mujoco
    import numpy as np
    from .mujoco_backend import initialize, numbers
    data=initialize(model,spec,pose='geometric')
    root=ET.fromstring(xml); assets=root.find('asset'); bounds={}
    for suffix in ('l','r'):
        points=[]
        for body_name in ('foot_'+suffix,'toes_01_'+suffix):
            b=model.body(body_name).id
            for g in range(model.body_geomadr[b],model.body_geomadr[b]+model.body_geomnum[b]):
                mesh=model.geom_dataid[g];start=model.mesh_vertadr[mesh];count=model.mesh_vertnum[mesh]
                points.extend(model.mesh_vert[start:start+count]@data.geom_xmat[g].reshape(3,3).T+data.geom_xpos[g])
                root.find(".//geom[@name='"+model.geom(g).name+"']").set('contype','0')
                root.find(".//geom[@name='"+model.geom(g).name+"']").set('conaffinity','0')
        points=np.asarray(points);low=points.min(0);high=points.max(0)
        # Compare a flat sole with the convex envelope of the actual geometry.
        # Both are massless rigid contact proxies; toe mass/motors remain.
        if flat:
            high[2]=low[2]+.008
            points=np.array(list(itertools.product(*zip(low,high))))
        b=model.body('foot_'+suffix).id
        local=(points-data.xpos[b])@data.xmat[b].reshape(3,3)
        mesh_name='continuous_sole_'+suffix
        ET.SubElement(assets,'mesh',name=mesh_name,vertex=numbers(local.ravel()))
        body=root.find(".//body[@name='foot_"+suffix+"']")
        ET.SubElement(body,'geom',name=mesh_name,type='mesh',mesh=mesh_name,contype='2',conaffinity='1',
                      friction='0.9 0.005 0.0001',solref='0.004 1',rgba='0.2 0.5 0.7 0.7')
        bounds[suffix]={'world_low_m':low.tolist(),'world_high_m':high.tolist(),'local_vertices':local.tolist(),
                        'method':'flat_box' if flat else 'convex_envelope_of_foot_and_toe'}
    result=ET.tostring(root,encoding='unicode'); changed=mujoco.MjModel.from_xml_string(result)
    if not np.array_equal(changed.body_mass,model.body_mass) or not np.allclose(changed.body_inertia,model.body_inertia,rtol=0,atol=1e-12):
        raise ValueError('Contact proxy changed authored mass/inertia')
    return changed,result,bounds


def run(args):
    import mujoco
    import numpy as np
    from . import mujoco_backend as backend
    from .leg_model_variant import specification
    spec=specification('anatomical_legs_mass_v2')
    spec['nominal_pose_profile']=args.nominal_pose_profile
    checkpoint=Path(args.gait_checkpoint).resolve() if args.gait_checkpoint else None
    training=json.loads((checkpoint.parent/'training.json').read_text()) if checkpoint else None
    residual=bool(training and training.get('policy_family')=='anatomical_feedback_residual_ppo')
    if training and training.get('whole_body'):
        from .model_spec import FINGER_JOINTS
        extra=[r['name'] for r in spec['joints'] if r['name'] not in FINGER_JOINTS and r['name'] not in spec['action_joints']]
        spec['action_joints']+=extra
        spec['locked_joints']=[n for n in spec['locked_joints'] if n not in extra]
        for record in spec['joints']:
            if record['name'] in extra:record.update(control='action',role=record['name'].removesuffix('_joint'))
        spec['whole_body']=True
    model,spec,xml=backend.build_model(urdf_path=spec['urdf_path'],specification=spec,contact_timeconst=.004)
    folder=backend.OUTPUT/args.name;folder.mkdir(parents=True,exist_ok=False)
    (folder/'probe_source.py').write_text(Path(__file__).read_text())
    backend.write_json(folder/'arguments.json',vars(args))
    gait=parameters=actor=implementation=None
    if args.gait_checkpoint:
        import torch
        from .landau_forward_control import load_gait_source, PERIOD
        from .mujoco_policy import observe
        recorded_xml=(checkpoint.parent/'model.xml').read_text()
        def asset_bound_xml(source):
            tree=ET.fromstring(source)
            for mesh in tree.findall('./asset/mesh'):
                if mesh.get('file'):mesh.set('file','sha256:'+backend.digest(Path(mesh.get('file'))))
            return ET.tostring(tree)
        if (training.get('policy_family') not in ('anatomical_periodic_feedback_cem','anatomical_feedback_residual_ppo') or training.get('graft') or
            training['checkpoints'][checkpoint.name]!=backend.digest(checkpoint) or
            training['model_xml_sha256']!=hashlib.sha256(recorded_xml.encode()).hexdigest() or
            asset_bound_xml(xml)!=asset_bound_xml(recorded_xml) or
            any(training[key]!=spec['source'][key] for key in ('urdf_sha256','mesh_tree_sha256')) or
            training.get('nominal_pose_profile','geometric')!=args.nominal_pose_profile or
            training['action_joints']!=spec['action_joints']):
            raise ValueError('Stepping comparison requires the exact anatomical gait and nominal model')
        xml=recorded_xml;model=mujoco.MjModel.from_xml_string(xml)
        gait,gait_source=load_gait_source(checkpoint,training)
        if residual:
            import importlib.util
            parent=Path(training['parent_checkpoint'])
            if (backend.digest(parent)!=training['parent_checkpoint_sha256'] or
                backend.digest(parent.parent/'training.json')!=training['parent_training_sha256']):raise ValueError('Frozen gait parent changed')
            parameters=torch.load(parent,map_location='cpu',weights_only=False)['parameters']
            if parameters.tolist()!=training['gait_parameters']:raise ValueError('Frozen gait parameters changed')
            residual_source=checkpoint.parent/'residual_source.py'
            if backend.digest(residual_source)!=training['residual_source_sha256']:raise ValueError('Residual source changed')
            module=importlib.util.spec_from_file_location('probe_residual_'+training['residual_source_sha256'][:16],residual_source)
            implementation=importlib.util.module_from_spec(module);module.loader.exec_module(implementation)
            conditioned=training.get('gait_conditioning')=='curriculum_level_v1'
            actor=implementation.make_actor(num_actions=len(training['action_joints']),**({'conditioned':True} if conditioned else {}))
            actor.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=False)['actor_state_dict'],strict=True);actor.eval()
            actor.gait_conditioned=conditioned;actor.residual_hold_steps=training.get('residual_hold_steps',1)
            actor.residual_clock_period=training.get('residual_clock_period',float(parameters[0]))
            actor.residual_ramp_s=training.get('residual_ramp_s',float(parameters[15]))
            (folder/'residual_source.py').write_text(residual_source.read_text())
        else:parameters=torch.load(checkpoint,map_location='cpu',weights_only=False)['parameters']
        (folder/'gait_source.py').write_text(gait_source.read_text());torch.set_num_threads(1)
    rows=[]
    cases=([('head_original',.9,None),('head_centroid',.9,None),('head_uniform',.9,None)] if args.head_inertia else
        [('articulated',.9,angle) for angle in (-.35,-.2,-.1,0.)] if args.posture else
        [(mode,mu,None) for mode,mu in [('articulated',.9),('continuous_sole',.9),('continuous_sole',.6),('continuous_sole',1.2),('continuous_hull',.9)]])
    if args.foot_models:cases=[case for case in cases if case[0] in args.foot_models]
    for mode,friction,waist in cases:
        candidate,source,bounds=continuous_sole(model,spec,xml,flat=mode=='continuous_sole') if mode.startswith('continuous_') else (model,xml,{})
        if mode in ('head_centroid','head_uniform'):
            candidate,source,bounds=head_inertia(model,xml,centroid_only=mode=='head_centroid')
        root=ET.fromstring(source)
        for geom in root.findall('.//geom'):geom.set('friction',f'{friction} 0.005 0.0001')
        source=ET.tostring(root,encoding='unicode');candidate=mujoco.MjModel.from_xml_string(source)
        data=backend.initialize(candidate,spec,pose='geometric');pelvis=candidate.body('root_x').id;base=candidate.body('base_link').id
        if waist is not None:
            data.qpos[candidate.joint('waist_pitch_joint').qposadr[0]]=waist
            data.ctrl[candidate.actuator('waist_pitch_joint').id]=waist
            mujoco.mj_forward(candidate,data)
        nominal_q=data.qpos.copy();nominal_ctrl=data.ctrl.copy()
        names=spec['action_joints'];joints=[candidate.joint(n).id for n in names]
        actuators=[candidate.actuator(n).id for n in names];previous=np.zeros(len(names),dtype=np.float32)
        if gait:
            if (not np.allclose(nominal_q,training['nominal_q'],rtol=0,atol=1e-12) or
                not np.allclose(nominal_ctrl,training['nominal_ctrl'],rtol=0,atol=1e-12)):
                raise ValueError('Stepping nominal pose or control changed')
            limits=np.asarray(training['action_scale'])/.08
        if actor:
            for attribute in ('_residual_bucket','_residual_latent'):
                if hasattr(actor,attribute):delattr(actor,attribute)
        start=data.xpos[pelvis].copy();initial_com=data.subtree_com[base].copy()
        qpos=[data.qpos.copy()];times=[float(data.time)];max_tilt=0.;max_drift=0.;force_peak=0.;nonfoot=set();contact_count=[data.ncon]
        fallen=False
        for step in range(round(args.seconds/candidate.opt.timestep)):
            if gait and step%10==0:
                rotation=data.xmat[base].reshape(3,3);phase=2*np.pi*data.time/PERIOD
                drift=data.xpos[pelvis,:2]-np.asarray(training['reference_xy']) if training.get('walking_position_observation')=='native_world_xy_m_v1' else np.zeros(2)
                value=np.r_[observe(candidate,data,nominal_q,joints,previous),drift,1-data.time/30.,
                            -.2,0.,0.,np.sin(phase),np.cos(phase),rotation[1,0],rotation[0,0]-1.]
                if residual:
                    from .mujoco_policy import foot_state
                    feet=foot_state(candidate,data,clearance=True,diagnostics=True);weight=float(candidate.body_mass.sum())*9.81
                    value=np.r_[value,feet['left_normal_force_N']/weight,feet['right_normal_force_N']/weight,
                        feet['left_clearance_m'],feet['right_clearance_m'],data.xpos[pelvis,2]-training['reference_height']]
                with torch.no_grad():
                    observation=torch.tensor(value.astype(np.float32))[None]
                    action=(implementation.evaluate_mean(actor,None,gait,parameters,observation,names,
                        torch.tensor(training['base_action_scale'])/.08,base_names=training['gait_action_joints'],
                        standing_names=training['standing_action_joints']) if residual else gait.gait_action(observation,parameters,names))
                    previous=action[0].numpy().clip(-limits,limits)
                data.ctrl[:]=nominal_ctrl;data.ctrl[actuators]+=.08*previous
            mujoco.mj_step(candidate,data);mujoco.mj_forward(candidate,data)
            tilt=float(np.arccos(np.clip(data.xmat[base].reshape(3,3)[2,2],-1,1)));max_tilt=max(max_tilt,tilt)
            max_drift=max(max_drift,float(np.linalg.norm(data.xpos[pelvis,:2]-start[:2])))
            total=0.
            for c in range(data.ncon):
                contact=data.contact[c]
                if 0 not in contact.geom:continue
                other=max(contact.geom);name=candidate.geom(other).name
                force=np.zeros(6);mujoco.mj_contactForce(candidate,data,c,force)
                total+=float((contact.frame.reshape(3,3).T@force[:3])[2])
                if not name.startswith(('foot_','toes_','continuous_sole')) and force[0]>.01:nonfoot.add(name)
            force_peak=max(force_peak,total/(candidate.body_mass.sum()*9.81))
            fallen=not np.isfinite(data.qpos).all() or tilt>np.pi/6 or data.xpos[pelvis,2]<start[2]-.08
            if (step+1)%10==0 or fallen:
                qpos.append(data.qpos.copy());times.append(float(data.time));contact_count.append(data.ncon)
            if fallen:break
        name=mode+'_'+str(friction)+('_waist_'+str(waist) if waist is not None else '');(folder/(name+'.xml')).write_text(source)
        np.savez_compressed(folder/(name+'.npz'),qpos=np.asarray(qpos),time_s=np.asarray(times),contacts=np.asarray(contact_count),dt=.02)
        rows.append({'foot_model':mode,'friction':friction,'waist_target_rad':waist,'initial_com_m':initial_com.tolist(),'duration_s':float(data.time),
            'survived_duration':not fallen and step+1==round(args.seconds/candidate.opt.timestep),'max_tilt_rad':max_tilt,
            'max_drift_m':max_drift,'peak_support_bw':force_peak,'nonfoot_contacts':sorted(nonfoot),
            'semantic_forward_m':float(-(data.xpos[pelvis,1]-start[1])),
            'final_com_m':data.subtree_com[base].tolist(),'mass_kg':float(candidate.body_mass.sum()),
            'sole_bounds':bounds,'model_xml_sha256':backend.digest(folder/(name+'.xml'))})
    backend.write_json(folder/('head_inertia_comparison.json' if args.head_inertia else 'posture_comparison.json' if args.posture else 'contact_comparison.json'),{'diagnostic_only':True,'backend':'mujoco_native_cpu',
        'mujoco_version':mujoco.__version__,'seconds':args.seconds,'inherited_milestone_passes':0,
        'nominal_pose_profile':args.nominal_pose_profile,
        'policy_family':training.get('policy_family') if training else None,
        'action_joints':training.get('action_joints') if training else [],
        'walking_position_observation':training.get('walking_position_observation') if training else None,
        'residual_source_sha256':training.get('residual_source_sha256') if residual else None,
        'gait_checkpoint_sha256':backend.digest(args.gait_checkpoint) if gait else None,
        'source_sha256':backend.digest(__file__),'urdf_sha256':spec['source']['urdf_sha256'],'results':rows})
    print(json.dumps([{k:v for k,v in row.items() if k!='sole_bounds'} for row in rows]),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--name',required=True);p.add_argument('--seconds',type=float,default=30.)
    p.add_argument('--nominal-pose-profile',choices=('geometric','upright_waist_v1'),default='geometric')
    p.add_argument('--gait-checkpoint',help='CPU stepping comparison with frozen anatomical gait or whole-body policy; never a GPU milestone claim')
    p.add_argument('--foot-models',nargs='+',choices=('articulated','continuous_sole','continuous_hull'),help='Limit foot-contact comparisons')
    group=p.add_mutually_exclusive_group()
    group.add_argument('--posture',action='store_true',help='Compare upright waist targets with unchanged feet and masses')
    group.add_argument('--head-inertia',action='store_true',help='Compare head centroid and uniform-solid inertia estimates at unchanged masses')
    args=p.parse_args()
    if not 0<args.seconds<=30 or '/' in args.name:raise ValueError('Invalid diagnostic bounds')
    if args.gait_checkpoint and args.posture:raise ValueError('Stepping comparison must retain the checkpoint nominal posture')
    if args.foot_models and (args.posture or args.head_inertia):raise ValueError('Foot selection cannot filter posture or head diagnostics')
    run(args)

if __name__=='__main__':main()
