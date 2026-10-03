"""Exact-asset CPU MuJoCo diagnostics; independent of Isaac evidence and objects.

Mesh collision uses MuJoCo's convex hull, matching the existing Isaac hull
contract. All URDF inertias, joints, limits and collision transforms are retained.
Non-action joints remain compliant PD holds, never welded. No gate promotion here.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import sys
import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import psutil
from scipy.spatial.transform import Rotation

from algorithms.urdf_learn_wasd_walk import model_spec
from algorithms.urdf_learn_wasd_walk.passive_stand import evaluate_gate, evaluate_free_root_support

OUTPUT = model_spec.ALGORITHM_ROOT / 'outputs' / 'mujoco'


def numbers(values):
    return ' '.join(format(float(v), '.17g') for v in values)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_physics_health(counts, capacities, arrays, *, clock_delta=None, dt=.002):
    """Fail closed on device overflows/nonfinite dynamics, independent of warnings."""
    for name,count in counts.items():
        if count < 0 or count >= capacities[name]:
            raise RuntimeError(f'{name} capacity exhausted: {count}/{capacities[name]}')
    for name,values in arrays.items():
        if not np.isfinite(values).all():
            raise RuntimeError(f'Nonfinite physics array: {name}')
    if clock_delta is not None and not np.isclose(clock_delta,dt,rtol=.005,atol=1e-7):
        raise RuntimeError(f'Unexpected GPU simulation clock increment {clock_delta}; possible reset')


def write_json(path, value):
    path=Path(path)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def origin(element):
    if element is None:
        return {'pos': '0 0 0', 'quat': '1 0 0 0'}
    q = Rotation.from_euler('xyz', [float(x) for x in element.get('rpy', '0 0 0').split()]).as_quat()
    return {'pos': element.get('xyz', '0 0 0'), 'quat': numbers(q[[3, 0, 1, 2]])}


def build_model(*, dt=0.002, gain_scale=1.0, noslip_iterations=0, contact_timeconst=None):
    build_start=time.perf_counter()
    spec = model_spec.build_robot_spec()
    spec_s=time.perf_counter()-build_start
    if spec['source']['urdf_sha256'] != model_spec.EXPECTED_URDF_SHA256:
        raise ValueError('URDF identity mismatch')
    urdf = ET.parse(model_spec.URDF_PATH).getroot()
    root = ET.Element('mujoco', model='landau_rabbit_ear')
    ET.SubElement(root, 'compiler', angle='radian', inertiafromgeom='false', fusestatic='false')
    opt = ET.SubElement(root, 'option', timestep=str(dt), gravity='0 0 -9.81', integrator='implicitfast',
                        solver='Newton', iterations='50', tolerance='1e-10', cone='elliptic', noslip_iterations=str(noslip_iterations))
    ET.SubElement(opt, 'flag', autoreset='disable')
    visual = ET.SubElement(root, 'visual')
    ET.SubElement(visual, 'global', offwidth='640', offheight='480')
    assets = ET.SubElement(root, 'asset')
    mesh_names = {}
    for mesh in urdf.findall('.//collision/geometry/mesh'):
        key = (mesh.get('filename'), mesh.get('scale', '1 1 1'))
        if key not in mesh_names:
            name = f'mesh_{len(mesh_names)}'
            mesh_names[key] = name
            ET.SubElement(assets, 'mesh', name=name, file=str(model_spec.URDF_PATH.parent / key[0]), scale=key[1])
    world = ET.SubElement(root, 'worldbody')
    ET.SubElement(world, 'light', pos='1 -1 3', dir='-0.3 0.3 -1', diffuse='0.8 0.8 0.8')
    ET.SubElement(world, 'geom', name='floor', type='plane', size='4 4 .1', contype='1', conaffinity='2',
                  friction='0.9 0.005 0.0001', rgba='0.23 0.26 0.29 1')
    links = {x.get('name'): x for x in urdf.findall('link')}
    children = {}
    for joint in urdf.findall('joint'):
        children.setdefault(joint.find('parent').get('link'), []).append(joint)
    records = {r['name']: r for r in spec['joints']}
    actuator = ET.SubElement(root, 'actuator')

    def add_body(parent, name, incoming=None):
        body = ET.SubElement(parent, 'body', name=name, **origin(None if incoming is None else incoming.find('origin')))
        inertial = links[name].find('inertial')
        if inertial is not None:
            inertia = inertial.find('inertia')
            # Rotate the complete URDF tensor into the link frame before diagonalization.
            rot = Rotation.from_quat(np.array([float(x) for x in origin(inertial.find('origin'))['quat'].split()])[[1, 2, 3, 0]]).as_matrix()
            a = {k: float(inertia.get(k, '0')) for k in ('ixx', 'iyy', 'izz', 'ixy', 'ixz', 'iyz')}
            tensor = rot @ np.array([[a['ixx'], a['ixy'], a['ixz']], [a['ixy'], a['iyy'], a['iyz']], [a['ixz'], a['iyz'], a['izz']]]) @ rot.T
            ET.SubElement(body, 'inertial', pos=origin(inertial.find('origin'))['pos'], mass=inertial.find('mass').get('value'),
                          fullinertia=numbers([tensor[0, 0], tensor[1, 1], tensor[2, 2], tensor[0, 1], tensor[0, 2], tensor[1, 2]]))
        if incoming is None:
            ET.SubElement(body, 'freejoint', name='floating_base')
        elif incoming.get('type') != 'fixed':
            r = records[incoming.get('name')]
            ET.SubElement(body, 'joint', name=r['name'], type='hinge', axis=numbers(r['axis_joint_frame']),
                          range=numbers(r['limits_rad']), limited='true', damping='0')
            ET.SubElement(actuator, 'position', name=r['name'], joint=r['name'],
                          kp=str(gain_scale * r['nominal_stiffness_nm_per_rad']),
                          kv=str(math.sqrt(gain_scale) * r['nominal_damping_nm_s_per_rad']),
                          forcelimited='true', forcerange=numbers([-r['effort_limit'], r['effort_limit']]),
                          ctrllimited='true', ctrlrange=numbers(r['limits_rad']))
        for i, collision in enumerate(links[name].findall('collision')):
            mesh = collision.find('geometry/mesh')
            if mesh is None:
                raise ValueError('Unimplemented non-mesh collision; refusing geometry loss')
            ET.SubElement(body, 'geom', name=f'{name}_collision_{i}', type='mesh',
                          mesh=mesh_names[(mesh.get('filename'), mesh.get('scale', '1 1 1'))],
                          **origin(collision.find('origin')), contype='2', conaffinity='1',
                          friction='0.9 0.005 0.0001', rgba='0.75 0.58 0.36 1')
        for joint in children.get(name, []):
            add_body(body, joint.find('child').get('link'), joint)
    add_body(world, 'base_link')
    if contact_timeconst is not None:
        if not 2*dt <= contact_timeconst <= .1:
            raise ValueError('Contact time constant must lie between 2*dt and 0.1 seconds')
        for geom in root.findall('.//geom'):
            geom.set('solref', numbers([contact_timeconst, 1.]))
    xml = ET.tostring(root, encoding='unicode')
    compile_start=time.perf_counter()
    model = mujoco.MjModel.from_xml_string(xml)
    spec['backend_build_timing']={'asset_audit_s':spec_s,'xml_build_s':compile_start-build_start-spec_s,'mujoco_compile_s':time.perf_counter()-compile_start}
    return model, spec, xml


def audit_model(model, spec):
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    tree = ET.parse(model_spec.URDF_PATH).getroot()
    transforms, _ = model_spec._joint_world_transforms(tree)
    position_errors, rotation_errors, inertia_errors, mass_errors, axes = [], [], [], [], []
    for link in tree.findall('link'):
        name = link.get('name')
        b = model.body(name).id
        rotation, position = transforms[name]
        position_errors.append(float(np.max(np.abs(data.xpos[b] - position))))
        rotation_errors.append(float(np.max(np.abs(data.xmat[b].reshape(3, 3) - rotation))))
        inertial = link.find('inertial')
        if inertial is not None:
            mass_errors.append(abs(model.body_mass[b] - float(inertial.find('mass').get('value'))))
            i = inertial.find('inertia')
            a = {k: float(i.get(k, '0')) for k in ('ixx','iyy','izz','ixy','ixz','iyz')}
            r = np.array(model_spec._origin_transform(inertial.find('origin'))[0])
            expected = r @ np.array([[a['ixx'],a['ixy'],a['ixz']],[a['ixy'],a['iyy'],a['iyz']],[a['ixz'],a['iyz'],a['izz']]]) @ r.T
            r = Rotation.from_quat(model.body_iquat[b][[1,2,3,0]]).as_matrix()
            actual = r @ np.diag(model.body_inertia[b]) @ r.T
            inertia_errors.append(float(np.max(np.abs(expected-actual))))
    for r in spec['joints']:
        j = model.joint(r['name']).id
        axes.append(float(np.dot(data.xaxis[j], r['axis_world_zero_pose'])))
        if not np.allclose(model.jnt_range[j], r['limits_rad'], atol=1e-12):
            raise ValueError('Compiled joint limit mismatch')
    audit = {'source': spec['source'], 'total_mass_kg': float(model.body_mass.sum()),
             'links': model.nbody-1, 'movable_joints': model.njnt-1, 'policy_actions': len(spec['action_joints']),
             'pd_hold_joints': len(spec['locked_joints']), 'collision_meshes': model.nmesh,
             'collision_geoms': model.ngeom-1, 'max_fk_position_error_m': max(position_errors),
             'max_fk_rotation_error': max(rotation_errors), 'minimum_directed_axis_cosine': min(axes),
             'max_mass_error_kg': max(mass_errors), 'max_inertia_tensor_error': max(inertia_errors),
             'collision_semantics': 'all source collision meshes; per-mesh convex hull; self collision disabled as in Isaac',
             'velocity_limit_semantics': 'URDF 4 rad/s monitored and fails validation; no hard state clamp',
             'action_joints': spec['action_joints'], 'pd_hold_names': spec['locked_joints']}
    if max(position_errors+rotation_errors+mass_errors+inertia_errors) > 1e-8 or min(axes) < .999999:
        raise ValueError(f'Compiled model mismatch: {audit}')
    if model.nbody-1 != spec['structure']['link_count'] or model.nmesh != 68 or model.nu != 69:
        raise ValueError('Model structure changed')
    return audit


@dataclass
class Assistance:
    """Training-only vertical pelvis spring and orientation PD; never propels XY."""
    coefficient: float = 0.0
    force_limit_n: float = 9.0
    torque_limit_nm: float = 1.0
    successes: int = 0

    def update(self, success_rate, *, window_episodes):
        if window_episodes < 20:
            return self.coefficient
        if success_rate >= .9:
            self.successes += 1
            if self.successes >= 3:
                self.coefficient = max(0.0, round(self.coefficient - .1, 10))
                self.successes = 0
        else:
            self.successes = 0
            if success_rate < .6:
                self.coefficient = min(1.0, round(self.coefficient + .1, 10))
        return self.coefficient

    def wrench(self, height_error, vertical_velocity, orientation_error, angular_velocity):
        if not 0 <= self.coefficient <= 1:
            raise ValueError('Assistance coefficient outside [0,1]')
        force = np.array([0., 0., np.clip(80*height_error-8*vertical_velocity, -self.force_limit_n, self.force_limit_n)])
        torque = 3*np.asarray(orientation_error)-.3*np.asarray(angular_velocity)
        torque *= min(1., self.torque_limit_nm/max(np.linalg.norm(torque), 1e-12))
        return self.coefficient*np.concatenate([force, torque])


def initialize(model, spec, *, pose='source', gain_scale=1.):
    data = mujoco.MjData(model)
    geometry = model_spec.derive_static_pose() if pose == 'geometric' else spec['nominal_pose']['geometry']
    positions = geometry['joint_positions_rad']
    for r in spec['joints']:
        j, a = model.joint(r['name']), model.actuator(r['name'])
        q = positions.get(r['name'], 0.)
        data.qpos[j.qposadr[0]] = q
        data.ctrl[a.id] = q if pose == 'geometric' else r['nominal_target_rad']
    data.qpos[2] = geometry['ground_aligned_base_z_m']
    mujoco.mj_forward(model, data)
    return data


def run(args):
    if args.forward and not args.checkpoint:
        raise ValueError('Forward evaluation requires a command-conditioned checkpoint')
    if args.seconds <= 0 or args.seconds > 60 or args.dt <= 0 or args.gain_scale <= 0:
        raise ValueError('Diagnostics require 0 < duration <= 60, positive dt and gains')
    out = (OUTPUT / args.name).resolve()
    out.relative_to(OUTPUT.resolve())
    out.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    model, spec, xml = build_model(dt=args.dt, gain_scale=args.gain_scale,
        noslip_iterations=args.noslip_iterations, contact_timeconst=getattr(args,'contact_timeconst',None))
    compile_s = time.perf_counter()-start
    audit = audit_model(model, spec)
    write_json(out/'model_audit.json', audit)
    (out/'backend_source.py').write_text(Path(__file__).read_text())
    (out/'model.xml').write_text(xml)
    data = initialize(model, spec, pose=args.pose, gain_scale=args.gain_scale)
    initial_q = data.qpos.copy()
    pelvis = model.body('root_x').id
    initial_pelvis = data.xpos[pelvis].copy()
    initial_rotation = data.xmat[pelvis].reshape(3,3).copy()
    initial_com = data.subtree_com[model.body('base_link').id].copy()
    pose_geometry=model_spec.analyze_pose_geometry({r['name']:float(data.qpos[model.joint(r['name']).qposadr[0]]) for r in spec['joints']})
    support_hull=pose_geometry['support_hull_xy_m']
    minimum_margin=float('inf'); first_support_exit=None
    control_identity=hashlib.sha256(json.dumps({'initial_qpos':data.qpos.tolist(),'initial_ctrl':data.ctrl.tolist()},sort_keys=True).encode()).hexdigest()
    assist = Assistance(args.assistance)
    policy = None
    previous_action = np.zeros(17, dtype=np.float32)
    policy_calls = 0
    max_action = 0.
    if args.checkpoint:
        if args.assistance != 0:
            raise ValueError('Policy evaluation requires exactly zero auxiliary assistance')
        import torch
        from algorithms.urdf_learn_wasd_walk.mujoco_policy import load_actor, observe
        torch.set_num_threads(1)
        policy, checkpoint_data = load_actor(args.checkpoint, model, spec)
        if checkpoint_data['backend']!=getattr(args,'backend','mujoco_cpu') or checkpoint_data['mujoco_version']!=mujoco.__version__:
            raise ValueError('Policy backend/version differs from evaluation; no inherited pass across physics backends')
        if args.forward and checkpoint_data.get('observation_dim',60) != 65:
            raise ValueError('Forward command requires a 65-observation policy')
        if checkpoint_data['model_xml_sha256'] != hashlib.sha256(xml.encode()).hexdigest():
            raise ValueError('Policy physics configuration differs from evaluation model')
        if not np.allclose(initial_q, checkpoint_data['nominal_q'], atol=1e-12):
            raise ValueError('Policy nominal pose differs')
        action_jids = [model.joint(name).id for name in spec['action_joints']]
        action_aids = [model.actuator(name).id for name in spec['action_joints']]
        nominal_ctrl = np.array(checkpoint_data['nominal_ctrl'])
    frame_states, rows = [data.qpos.copy()], [{'time_s': 0., 'assistance_coefficient': assist.coefficient, 'external_wrench_world': [0.]*6}]
    foot_names = {'foot_l', 'foot_r', 'toes_01_l', 'toes_01_r'}
    force = np.zeros(6)
    max_tilt = max_drop = max_drift = max_speed = max_force = max_assist = 0.
    support_ratios, bad_contacts = [], set()
    limits = np.array([r['velocity_limit'] for r in spec['joints']])
    joint_dofs = np.array([model.joint(r['name']).dofadr[0] for r in spec['joints']])
    process = psutil.Process()
    warp_runtime = None
    if getattr(args,'backend','mujoco_cpu') == 'mujoco_warp_cuda':
        from algorithms.urdf_learn_wasd_walk.mujoco_warp_runtime import WarpEvaluation
        warp_runtime = WarpEvaluation(model,data,out)
    cpu_start = process.cpu_times()
    loop_start = time.perf_counter()
    failure = None
    steps = round(args.seconds/args.dt)
    foot_samples=[]; liftoffs={'left':0,'right':0}; air_runs={'left':0,'right':0}; touched={'left':False,'right':False}
    qmin=data.qpos.copy(); qmax=data.qpos.copy()
    if args.forward:
        from algorithms.urdf_learn_wasd_walk.mujoco_policy import foot_state
        from algorithms.urdf_learn_wasd_walk.forward_walk_contract import evaluate_forward_gate
    for step in range(steps):
        if policy is not None and step % round(.02/args.dt) == 0:
            obs = observe(model, data, initial_q, action_jids, previous_action)
            if checkpoint_data.get('observation_dim',60) == 65:
                phase = 2*np.pi*data.time
                extra = [args.forward,0,0,np.sin(phase),np.cos(phase)] if args.forward else [0.]*5
                obs = np.concatenate([obs,extra]).astype(np.float32)
            with torch.no_grad():
                action = policy.actor(torch.from_numpy(obs)).numpy().clip(-1,1)
            data.ctrl[:] = nominal_ctrl
            scale = checkpoint_data['action_scale'] if args.forward else checkpoint_data.get('stand_action_scale',checkpoint_data['action_scale'])
            data.ctrl[action_aids] += scale*action
            previous_action = action
            policy_calls += 1
            max_action = max(max_action, float(np.abs(action).max()))
        rot = data.xmat[pelvis].reshape(3,3)
        orientation_error = Rotation.from_matrix(initial_rotation @ rot.T).as_rotvec()
        vel = np.zeros(6)
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, pelvis, vel, 0)
        wrench = assist.wrench(initial_pelvis[2]-data.xpos[pelvis,2], vel[5], orientation_error, vel[:3])
        data.xfrc_applied[:] = 0
        data.xfrc_applied[pelvis] = wrench
        if warp_runtime is None:
            mujoco.mj_step(model, data)
            # Refresh derived positions to the post-integration state for exact frame timestamps.
            mujoco.mj_forward(model, data)
        else:
            try:
                warp_runtime.step(data)
            except Exception as error:
                failure=f'GPU physics validation error: {error}'
                write_json(out/'physics_failure.json',{'error':repr(error),'completed_physics_steps':warp_runtime.step_count})
                np.savez_compressed(out/'failure_state.npz',qpos=data.qpos,qvel=data.qvel,qacc=data.qacc)
                break
        if not args.forward:
            margin=model_spec.support_polygon_margin(data.subtree_com[model.body('base_link').id,:2],support_hull)
            minimum_margin=min(minimum_margin,margin)
            if margin<=0 and first_support_exit is None: first_support_exit=float(data.time)
        qmin=np.minimum(qmin,data.qpos); qmax=np.maximum(qmax,data.qpos)
        if args.forward:
            feet=foot_state(model,data,clearance=True)
            foot_samples.append(feet)
            for side in ('left','right'):
                if feet[side+'_contact']:
                    touched[side]=True; air_runs[side]=0
                elif touched[side] and feet[side+'_clearance_m']>.002:
                    air_runs[side]+=1
                    if air_runs[side]==round(.04/args.dt): liftoffs[side]+=1
                else: air_runs[side]=0
        tilt = math.acos(float(np.clip(data.xmat[model.body('base_link').id].reshape(3,3)[2,2], -1, 1)))
        max_tilt = max(max_tilt, tilt)
        max_drop = max(max_drop, initial_pelvis[2]-data.xpos[pelvis,2])
        max_drift = max(max_drift, float(np.linalg.norm(data.xpos[pelvis,:2]-initial_pelvis[:2])))
        speed = np.abs(data.qvel[joint_dofs])
        max_speed = max(max_speed, float(speed.max()))
        max_force = max(max_force, float(np.abs(data.actuator_force).max()))
        max_assist = max(max_assist, float(np.linalg.norm(wrench)))
        support = 0.
        for c in range(data.ncon):
            contact = data.contact[c]
            ids = [int(contact.geom[0]), int(contact.geom[1])]
            if 0 not in ids:
                continue
            b = model.geom_bodyid[max(ids)]
            name = model.body(b).name
            mujoco.mj_contactForce(model, data, c, force)
            if name in foot_names:
                support += float((contact.frame.reshape(3,3).T @ force[:3])[2])
            elif abs(force[0]) > .01:
                bad_contacts.add(name)
        support_ratios.append(support/(model.body_mass.sum()*9.81))
        if (step+1) % max(1, round(.02/args.dt)) == 0 or step == steps-1:
            frame_states.append(data.qpos.copy())
            rows.append({'time_s': float(data.time), 'pelvis_position_m': data.xpos[pelvis].tolist(),
                         'com_m': data.subtree_com[model.body('base_link').id].tolist(), 'tilt_rad': tilt,
                         'support_body_weight_ratio': support_ratios[-1], 'max_joint_speed': float(speed.max()),
                         'assistance_coefficient': assist.coefficient, 'external_wrench_world': wrench.tolist()})
        if not np.isfinite(data.qpos).all() or any(w.number for w in data.warning):
            failure = 'nonfinite state or MuJoCo numerical warning'
            break
        if tilt > math.pi/6 or initial_pelvis[2]-data.xpos[pelvis,2] > .08:
            failure = 'fall'
            break
    wall_s = time.perf_counter()-loop_start
    warp_info = warp_runtime.close() if warp_runtime is not None else None
    cpu = process.cpu_times()
    metrics = {'duration_s': float(data.time), 'reset_count': 0, 'done_count': int(failure is not None),
               'fall_count': int(failure == 'fall'), 'max_reference_tilt_rad': max_tilt,
               'root_height_drop_m': max_drop, 'horizontal_drift_m': max_drift,
               'max_abs_action': max_action, 'max_abs_command': abs(args.forward), 'policy_calls': policy_calls, 'max_joint_speed_rad_s': max_speed,
               'max_motor_torque_nm': max_force, 'peak_auxiliary_wrench_norm': max_assist,
               'mean_support_body_weight_ratio': float(np.mean(support_ratios)) if support_ratios else 0.,
               'peak_support_body_weight_ratio': float(max(support_ratios)) if support_ratios else 0., 'nonfoot_contacts': sorted(bad_contacts)}
    if not args.forward:
        if not math.isfinite(minimum_margin):minimum_margin=0.
        metrics.update(minimum_support_polygon_margin_m=minimum_margin,first_support_exit_time_s=first_support_exit,
                       mean_support_force_body_weight_ratio=metrics['mean_support_body_weight_ratio'],
                       peak_support_force_body_weight_ratio=metrics['peak_support_body_weight_ratio'])
    gate_metrics = dict(metrics)
    if policy is not None:
        gate_metrics['max_abs_action'] = 0.  # Passive-only predicate; actions are audited above.
    if args.forward:
        metrics.update(semantic_forward_displacement_m=float(data.xpos[pelvis,1]-initial_pelvis[1]),
                       semantic_strafe_displacement_m=float(data.xpos[pelvis,0]-initial_pelvis[0]),
                       left_foot_liftoff_count=liftoffs['left'],right_foot_liftoff_count=liftoffs['right'],
                       mean_contact_foot_slip_mps=float(np.mean([f['mean_slip_mps'] for f in foot_samples])),
                       simultaneous_air_fraction=float(np.mean([not f['left_contact'] and not f['right_contact'] for f in foot_samples])),
                       leg_joint_excursion_rad={n:float((qmax-qmin)[model.joint(n).qposadr[0]]) for n in spec['action_joints']},
                       policy_inference_steps=policy_calls,control_steps=math.ceil((step+1)/round(.02/args.dt)))
        failures=evaluate_forward_gate(metrics)
    else:
        passed, failures = evaluate_gate(gate_metrics)
        failures.extend(evaluate_free_root_support(metrics))
    if failure:
        failures.append(failure)
    if max_speed > limits.min()+1e-6:
        failures.append('joint velocity exceeded URDF limit')
    if not .5 <= metrics['mean_support_body_weight_ratio'] <= 1.5:
        failures.append('mean ground support outside [0.5, 1.5] body weight')
    if metrics['peak_support_body_weight_ratio'] > 3:
        failures.append('peak support exceeded 3 body weights')
    if bad_contacts:
        failures.append('nonfoot ground contact')
    if args.assistance != 0 or max_assist != 0:
        failures.append('training assistance enabled; never milestone evidence')
    performance = {'model_build_compile_s': compile_s, **spec['backend_build_timing'], 'simulation_wall_s': wall_s,
                   'physics_transitions_per_s_including_diagnostics': (step+1)/wall_s,
                   'control_transitions_per_s_equivalent': (step+1)*args.dt/.02/wall_s,
                   'process_cpu_percent_one_core_100': 100*(cpu.user+cpu.system-cpu_start.user-cpu_start.system)/wall_s,
                   'process_rss_bytes': process.memory_info().rss, 'host_ram_used_bytes': psutil.virtual_memory().used,
                   'gpu_utilization': None, 'vram_bytes': None, 'gpu_unavailable_reason': 'GPU not used for native CPU physics',
                   'ppo_iteration_s': None}
    if warp_info is not None:
        performance.update(gpu_unavailable_reason=None,gpu_resources_file=str(out/'resources.jsonl'),
                           warp_setup=warp_info,gpu_utilization=warp_info['gpu_peak_utilization_percent'],
                           vram_bytes=warp_info['gpu_peak_vram_mib']*1024**2)
    config = vars(args)
    if args.checkpoint:
        config['checkpoint_sha256'] = digest(args.checkpoint)
    identity = {'backend': getattr(args,'backend','mujoco_cpu'), 'mujoco_version': mujoco.__version__, 'seed': args.seed,
                'urdf_sha256': spec['source']['urdf_sha256'], 'mesh_tree_sha256': spec['source']['mesh_tree_sha256'],
                'model_xml_sha256': hashlib.sha256(xml.encode()).hexdigest(), 'source_sha256': digest(__file__), 'initial_control_sha256':control_identity,
                'config_sha256': hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()}
    if warp_info is not None:
        identity.update(warp_version=warp_info['versions']['warp-lang'],mujoco_warp_version=warp_info['versions']['mujoco-warp'],
                        warp_reset_semantics_sha256=warp_info['warp_forward_source_sha256'],
                        warp_runtime_source_sha256=warp_info['source_sha256'],warp_io_source_sha256=warp_info['warp_io_source_sha256'])
    result = {'status': 'dynamics_passed_proof_pending' if not failures else 'failed',
              'gate_passed': False, 'canonical_milestones_modified': False,
              'milestone': 'gate_5m_no_reset' if args.forward else 'stand_30s_no_reset' if policy is not None else 'stand_zero_signal_30s_no_reset', 'identity': identity, 'config': config,
              'metrics': metrics, 'failures': failures, 'performance': performance,
              'reproduce': shlex.join(['env', *[f'{k}={os.environ[k]}' for k in ('MUJOCO_GL','LIBGL_ALWAYS_SOFTWARE') if k in os.environ],sys.executable, '-m', 'algorithms.urdf_learn_wasd_walk.mujoco_backend', *sys.argv[1:]]),
              'created_at': datetime.now(timezone.utc).isoformat(), 'visual_review': 'pending'}
    write_json(out/'dynamics.json', result)
    write_json(out/'trace.json', rows)
    np.savez_compressed(out/'trajectory.npz', qpos=np.array(frame_states), time=np.array([r['time_s'] for r in rows]))
    result['trajectory_sha256']=digest(out/'trajectory.npz')
    write_json(out/'dynamics.json',result)
    if args.render:
        try:
            import subprocess
            render_env=dict(os.environ)
            render_env.setdefault('MUJOCO_GL','egl')
            subprocess.run([sys.executable, '-m', 'algorithms.urdf_learn_wasd_walk.mujoco_render', str(out)],
                           check=True, timeout=180,env=render_env)
            result['proof'] = json.loads((out/'proof_metadata.json').read_text())
        except Exception as error:
            result['proof_error'] = repr(error)
        write_json(out/'dynamics.json', result)
    progress_path = model_spec.ALGORITHM_ROOT/'outputs/backend_progress.json'
    progress = json.loads(progress_path.read_text()) if progress_path.exists() else {}
    pass_key='gpu_backend_passes' if warp_info is not None else 'backend_passes'
    current_gate=next((gate for gate in ('stand_zero_signal_30s_no_reset','stand_30s_no_reset') if progress.get(pass_key,{}).get(gate,{}).get('status')!='passed'),'gate_5m_no_reset')
    progress.update(updated_at=datetime.now(timezone.utc).isoformat(), current_gate=current_gate,evaluation_gate=result['milestone'],
                    simulator=identity['backend'], variant=f'passive_{args.pose}_gain{args.gain_scale}', assistance_coefficient=args.assistance,
                    active_process=None, throughput=performance, validation=metrics, failure_metrics=failures,
                    next_step='diagnose earliest failed passive criterion; require full proof and visual review before promotion',
                    artifact_paths=[str(out)], iteration=None, checkpoint=None)
    write_json(progress_path, progress)
    print(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', required=True)
    parser.add_argument('--backend',choices=('mujoco_cpu','mujoco_warp_cuda'),default='mujoco_cpu')
    parser.add_argument('--seconds', type=float, default=30.)
    parser.add_argument('--dt', type=float, default=.002)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--pose', choices=['source', 'geometric'], default='geometric')
    parser.add_argument('--gain-scale', type=float, default=1.)
    parser.add_argument('--noslip-iterations', type=int, default=0)
    parser.add_argument('--contact-timeconst', type=float)
    parser.add_argument('--assistance', type=float, default=0.)
    parser.add_argument('--checkpoint')
    parser.add_argument('--forward', type=float, default=0.)
    parser.add_argument('--render', action='store_true')
    run(parser.parse_args())


if __name__ == '__main__':
    main()
