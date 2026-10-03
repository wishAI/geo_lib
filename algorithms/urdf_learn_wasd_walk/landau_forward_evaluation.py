"""Exact no-reset evaluation of the 70-feature Landau walking controller.

Uses the existing independent single-world Warp physics bridge. Records every
physics-step foot observation and counts completed swings, not mere liftoffs.
"""
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import sys
import time
import mujoco
import numpy as np
import psutil
from scipy.spatial.transform import Rotation
from algorithms.urdf_learn_wasd_walk import model_spec, mujoco_backend as backend
from algorithms.urdf_learn_wasd_walk.passive_stand import evaluate_gate, evaluate_free_root_support

OUTPUT = backend.OUTPUT
build_model = backend.build_model
audit_model = backend.audit_model
initialize = backend.initialize
Assistance = backend.Assistance
write_json = backend.write_json
digest = backend.digest

def run(args):
    direction=getattr(args,'direction',None)
    if direction and (getattr(args,'turn',False) or getattr(args,'teleop',False) or not args.forward):
        raise ValueError('Direction requires a separate moving-command evaluation')
    if direction:
        from algorithms.urdf_learn_wasd_walk.landau_direction_contract import DIRECTIONS, GATE_WIDTH_M
        direction_axis=np.asarray(DIRECTIONS[direction])
        previous_along=previous_cross=0.
        direction_reached=False
        direction_horizon=args.seconds
    turn_mode=bool(getattr(args,'turn',False))
    teleop_mode=bool(getattr(args,'teleop',False))
    if teleop_mode and (turn_mode or not args.forward or args.seconds!=60.):
        raise ValueError('Teleop requires a separate60s moving-command evaluation')
    if turn_mode and (not args.forward or args.seconds<24):
        raise ValueError('Turn evaluation requires a moving profile and at least 24 seconds')
    if turn_mode and args.seconds<args.turn_hold_start+5.:
        raise ValueError('Turn evaluation must include five seconds of zero-command hold')
    if args.forward and not args.checkpoint:
        raise ValueError('Forward evaluation requires a command-conditioned checkpoint')
    if args.seconds <= 0 or args.seconds > (240 if direction else 120) or args.dt <= 0 or args.gain_scale <= 0:
        raise ValueError('Diagnostics require a positive bounded duration (M7: 240 s; otherwise: 120 s), dt and gains')
    target_distance=float(getattr(args,'target_distance_m',5.))
    if target_distance not in (5.,10.):raise ValueError('Unsupported distance gate')
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
    raw_min=np.full(17,np.inf); raw_max=np.full(17,-np.inf)
    target_min=np.full(17,np.inf); target_max=np.full(17,-np.inf)
    if args.checkpoint:
        if args.assistance != 0:
            raise ValueError('Policy evaluation requires exactly zero auxiliary assistance')
        import torch
        from algorithms.urdf_learn_wasd_walk.mujoco_policy import load_actor, observe
        torch.set_num_threads(1)
        policy, checkpoint_data = load_actor(args.checkpoint, model, spec)
        if checkpoint_data['backend']!=getattr(args,'backend','mujoco_cpu') or checkpoint_data['mujoco_version']!=mujoco.__version__:
            raise ValueError('Policy backend/version differs from evaluation; no inherited pass across physics backends')
        if checkpoint_data.get('observation_dim') != 70:
            raise ValueError('This evaluator requires the 70-feature Landau policy')
        if checkpoint_data['model_xml_sha256'] != hashlib.sha256(xml.encode()).hexdigest():
            raise ValueError('Policy physics configuration differs from evaluation model')
        if not np.allclose(initial_q, checkpoint_data['nominal_q'], atol=1e-12):
            raise ValueError('Policy nominal pose differs')
        action_jids = [model.joint(name).id for name in spec['action_joints']]
        action_aids = [model.actuator(name).id for name in spec['action_joints']]
        nominal_ctrl = np.array(checkpoint_data['nominal_ctrl'])
    frame_states, rows = [data.qpos.copy()], [{'time_s': 0., 'assistance_coefficient': assist.coefficient, 'external_wrench_world': [0.]*6}]
    frame_velocities=[data.qvel.copy()]
    policy_observations=[];policy_raw_actions=[];policy_applied_actions=[];policy_times=[]
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
    roll_qadr={side:model.joint(side+'_hip_roll_joint').qposadr[0] for side in ('left','right')}
    roll_aids={side:model.actuator(side+'_hip_roll_joint').id for side in ('left','right')}
    qmin=data.qpos.copy(); qmax=data.qpos.copy()
    swing={s:{'air_s':0.,'peak_m':0.,'touched':False,'completed':0} for s in ('left','right')}
    flight_s=max_flight_s=0.
    max_heading=0.
    hold_anchor=None; hold_drift=hold_error=hold_speed=hold_yaw_speed=0.
    hold_first=None; hold_last=None; hold_command=0.
    start_rotation=data.xmat[model.body('base_link').id].reshape(3,3)
    previous_heading=math.atan2(start_rotation[1,0],start_rotation[0,0]);turn_heading=0.
    if args.forward:
        from algorithms.urdf_learn_wasd_walk.mujoco_policy import foot_state
        from algorithms.urdf_learn_wasd_walk.forward_walk_contract import evaluate_forward_gate
    for step in range(steps):
        if policy is not None and step % round(.02/args.dt) == 0:
            obs = observe(model, data, initial_q, action_jids, previous_action)
            with torch.no_grad():
                raw_action = policy.actor(torch.from_numpy(obs)).numpy()
            raw_min=np.minimum(raw_min,raw_action);raw_max=np.maximum(raw_max,raw_action)
            data.ctrl[:] = nominal_ctrl
            scale = checkpoint_data['action_scale'] if args.forward else checkpoint_data.get('stand_action_scale',checkpoint_data['action_scale'])
            bound = np.asarray(scale)/.08 if checkpoint_data.get('action_mapping')=='legacy_radians_v2' else 1.
            action = np.clip(raw_action,-bound,bound)
            policy_observations.append(obs.copy());policy_raw_actions.append(raw_action.copy())
            policy_applied_actions.append(action.copy());policy_times.append(float(data.time))
            if checkpoint_data.get('action_mapping')=='legacy_radians_v2': scale=.08
            data.ctrl[action_aids] += np.asarray(scale)*action
            offsets=data.ctrl[action_aids]-nominal_ctrl[action_aids]
            target_min=np.minimum(target_min,offsets);target_max=np.maximum(target_max,offsets)
            previous_action = action
            policy_calls += 1
            max_action = max(max_action, float(np.abs(action/bound).max()))
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
            feet=foot_state(model,data,clearance=True,diagnostics=True)
            contact_velocity=np.zeros(6)
            mujoco.mj_objectVelocity(model,data,mujoco.mjtObj.mjOBJ_BODY,pelvis,contact_velocity,0)
            feet['root_vertical_velocity_mps']=float(contact_velocity[5])
            for side in roll_qadr:
                feet[side+'_hip_roll_actual_rad']=float(data.qpos[roll_qadr[side]])
                feet[side+'_hip_roll_target_rad']=float(data.ctrl[roll_aids[side]])
            foot_samples.append({'time_s':float(data.time),**feet})
            flight_s = flight_s+args.dt if not feet['left_contact'] and not feet['right_contact'] else 0.
            max_flight_s=max(max_flight_s,flight_s)
            for side,state in swing.items():
                if feet[side+'_contact']:
                    if state['touched'] and state['air_s']>=.06 and state['peak_m']>=.015:
                        state['completed']+=1
                    state.update(touched=True,air_s=0.,peak_m=0.)
                elif state['touched']:
                    state['air_s']+=args.dt
                    state['peak_m']=max(state['peak_m'],feet[side+'_clearance_m'])
            for side in ('left','right'):
                if feet[side+'_contact']:
                    touched[side]=True; air_runs[side]=0
                elif touched[side] and feet[side+'_clearance_m']>.002:
                    air_runs[side]+=1
                    if air_runs[side]==round(.04/args.dt): liftoffs[side]+=1
                else: air_runs[side]=0
        tilt = math.acos(float(np.clip(data.xmat[model.body('base_link').id].reshape(3,3)[2,2], -1, 1)))
        base_rotation=data.xmat[model.body('base_link').id].reshape(3,3)
        heading=math.atan2(base_rotation[1,0],base_rotation[0,0])
        turn_heading+=math.atan2(math.sin(heading-previous_heading),math.cos(heading-previous_heading))
        previous_heading=heading
        max_heading=max(max_heading,abs(heading))
        if turn_mode:
            if data.time+1e-8>=args.turn_hold_start:
                if hold_anchor is None:hold_anchor=data.xpos[pelvis,:2].copy()
                hold_drift=max(hold_drift,float(np.linalg.norm(data.xpos[pelvis,:2]-hold_anchor)))
            if data.time+1e-8>=args.seconds-5.:
                if hold_first is None:hold_first=float(data.time)
                hold_last=float(data.time)
                error=abs(turn_heading-math.pi/2)
                hold_error=max(hold_error,error)
                velocity=np.zeros(6)
                mujoco.mj_objectVelocity(model,data,mujoco.mjtObj.mjOBJ_BODY,pelvis,velocity,0)
                hold_speed=max(hold_speed,float(np.linalg.norm(velocity[3:5])))
                hold_yaw_speed=max(hold_yaw_speed,abs(float(velocity[2])))
                if policy_times[-1]+1e-8>=args.seconds-5.:
                    hold_command=max(hold_command,float(np.abs(policy_observations[-1][63:66]).max()))
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
        if args.forward:
            # Retain the brief impact peaks that a 50 Hz video trace can miss.
            foot_samples[-1].update(support_body_weight_ratio=support_ratios[-1],
                                    max_joint_speed_rad_s=float(speed.max()))
        if (step+1) % max(1, round(.02/args.dt)) == 0 or step == steps-1:
            frame_states.append(data.qpos.copy())
            frame_velocities.append(data.qvel.copy())
            rows.append({'time_s': float(data.time), 'pelvis_position_m': data.xpos[pelvis].tolist(),
                         'heading_rad':heading,
                         'com_m': data.subtree_com[model.body('base_link').id].tolist(), 'tilt_rad': tilt,
                         'support_body_weight_ratio': support_ratios[-1], 'max_joint_speed': float(speed.max()),
                         'assistance_coefficient': assist.coefficient, 'external_wrench_world': wrench.tolist()})
            if direction:
                delta=data.xpos[pelvis,:2]-initial_pelvis[:2]
                along=float(delta@direction_axis)
                cross=float(delta[0]*direction_axis[1]-delta[1]*direction_axis[0])
                if previous_along<10.<=along:
                    fraction=(10.-previous_along)/(along-previous_along)
                    direction_reached=abs(previous_cross+fraction*(cross-previous_cross))<=GATE_WIDTH_M
                previous_along,previous_cross=along,cross
        if not np.isfinite(data.qpos).all() or any(w.number for w in data.warning):
            failure = 'nonfinite state or MuJoCo numerical warning'
            break
        if tilt > math.pi/6 or initial_pelvis[2]-data.xpos[pelvis,2] > .08:
            failure = 'fall'
            break
        if direction and direction_reached:
            break
    wall_s = time.perf_counter()-loop_start
    warp_info = warp_runtime.close() if warp_runtime is not None else None
    cpu = process.cpu_times()
    if direction:
        # A world gate ends at its first valid crossing on a control boundary.
        # The declared search horizon remains recorded for exact reproduction.
        args.direction_horizon_s=direction_horizon
        if direction_reached:args.seconds=round(float(data.time)/.02)*.02
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
    if policy is not None:
        metrics['action_target_range_rad']={name:[float(lo),float(hi)] for name,lo,hi in zip(spec['action_joints'],target_min,target_max)}
        metrics['raw_actor_output_range']={name:[float(lo),float(hi)] for name,lo,hi in zip(spec['action_joints'],raw_min,raw_max)}
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
        metrics.update(left_completed_swings=swing['left']['completed'],right_completed_swings=swing['right']['completed'],
                       max_simultaneous_flight_s=max_flight_s,max_heading_deviation_rad=max_heading,
                       left_peak_clearance_m=max(f['left_clearance_m'] for f in foot_samples),
                       right_peak_clearance_m=max(f['right_clearance_m'] for f in foot_samples))
        if turn_mode:
            from algorithms.urdf_learn_wasd_walk.landau_turn_control import evaluate_turn_gate
            metrics.update(hold_max_heading_error_rad=hold_error,hold_max_drift_m=hold_drift,
                hold_max_horizontal_speed_mps=hold_speed,hold_max_yaw_speed_rad_s=hold_yaw_speed,
                hold_duration_s=hold_last-hold_first if hold_first is not None else 0.,
                hold_max_abs_command=hold_command,final_heading_rad=turn_heading)
            failures=evaluate_turn_gate(metrics,args.seconds)
        elif teleop_mode:
            from algorithms.urdf_learn_wasd_walk.landau_teleop_contract import response_metrics,evaluate_gate as evaluate_teleop
            responses=response_metrics([r['time_s'] for r in rows],
                [initial_pelvis if i==0 else r['pelvis_position_m'] for i,r in enumerate(rows)],
                [0. if i==0 else r['heading_rad'] for i,r in enumerate(rows)],foot_samples)
            metrics['teleop_response']=responses
            failures=evaluate_teleop(metrics,responses)
        elif direction:
            from algorithms.urdf_learn_wasd_walk.landau_direction_contract import gate_metrics,evaluate_gate as evaluate_direction
            metrics.update(gate_metrics(direction,[r['time_s'] for r in rows],
                [initial_pelvis if i==0 else r['pelvis_position_m'] for i,r in enumerate(rows)]))
            failures=evaluate_direction(metrics,args.seconds)
        else:failures=evaluate_forward_gate(metrics,required_distance_m=target_distance)
        for side in ('left','right'):
            if swing[side]['completed']<3:failures.append(f'{side} has fewer than 3 completed swings with 15 mm clearance and 60 ms airtime')
        if max_flight_s>.12:failures.append('continuous simultaneous flight exceeded 120 ms')
        if metrics['mean_contact_foot_slip_mps']>.1:failures.append('mean contact foot slip exceeded 0.1 m/s')
        if not direction and not turn_mode and not teleop_mode and max_heading>math.radians(30):failures.append('forward heading deviated more than 30 degrees')
        write_json(out/'foot_trace.json',foot_samples)
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
              'milestone': 'gate_10m_four_directions_no_reset' if direction else 'teleop_60s_forward_turn' if teleop_mode else 'yaw_turn_90deg_hold' if turn_mode else f'gate_{target_distance:g}m_no_reset' if args.forward else 'stand_30s_no_reset' if policy is not None else 'stand_zero_signal_30s_no_reset', 'identity': identity, 'config': config,
              'metrics': metrics, 'failures': failures, 'performance': performance,
              'reproduce': shlex.join(['env', *[f'{k}={os.environ[k]}' for k in ('MUJOCO_GL','LIBGL_ALWAYS_SOFTWARE') if k in os.environ],sys.executable, '-m', 'algorithms.urdf_learn_wasd_walk.landau_forward_control', *sys.argv[1:]]),
              'created_at': datetime.now(timezone.utc).isoformat(), 'visual_review': 'pending'}
    result['controller_source_sha256']=digest(Path(__file__).with_name('landau_forward_control.py'))
    if args.forward:
        result['foot_trace_sha256']=digest(out/'foot_trace.json')
        result['walking_acceptance']={'minimum_completed_swings_per_foot':3,
            'minimum_swing_airtime_s':.06,'minimum_swing_clearance_m':.015,
            'maximum_continuous_flight_s':.12,'maximum_mean_contact_slip_mps':.1,
            'maximum_heading_deviation_degrees':30.}
        if direction:
            result['walking_acceptance'].pop('maximum_heading_deviation_degrees')
            protocol=Path(__file__).with_name('landau_direction_contract.py')
            (out/'direction_protocol_source.py').write_text(protocol.read_text())
            result['direction_protocol_source_sha256']=digest(protocol)
            result['direction_interpretation']='turn and walk to world gate from unchanged nominal start pose'
        if teleop_mode:
            result['walking_acceptance'].pop('maximum_heading_deviation_degrees')
            contract=Path(__file__).with_name('landau_teleop_contract.py')
            (out/'teleop_validator_source.py').write_text(contract.read_text())
            result['teleop_validator_source_sha256']=digest(contract)
            result['teleop_protocol']='scripted joystick replay; two yaw signs, stop/restart, second stop; no human-input claim'
        if turn_mode:
            result['walking_acceptance'].pop('maximum_heading_deviation_degrees')
            result['turn_acceptance']={'target_heading_degrees':90.,'maximum_hold_heading_error_degrees':5.,
                'minimum_hold_duration_s':5.,'maximum_hold_drift_m':.03,
                'maximum_hold_horizontal_speed_mps':.05,'maximum_hold_yaw_speed_rad_s':.1}
            turn_contract=Path(__file__).with_name('landau_turn_control.py')
            result['turn_validator_source_sha256']=digest(turn_contract)
            (out/'turn_validator_source.py').write_text(turn_contract.read_text())
    (out/'controller_source.py').write_text(Path(__file__).with_name('landau_forward_control.py').read_text())
    write_json(out/'dynamics.json', result)
    write_json(out/'trace.json', rows)
    np.savez_compressed(out/'trajectory.npz', qpos=np.array(frame_states), qvel=np.array(frame_velocities), time=np.array([r['time_s'] for r in rows]))
    if policy_observations:
        np.savez_compressed(out/'policy_trace.npz',observation=np.array(policy_observations),
            raw_action=np.array(policy_raw_actions),applied_action=np.array(policy_applied_actions),time=np.array(policy_times))
        result['policy_trace_sha256']=digest(out/'policy_trace.npz')
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
    ledger_path=model_spec.ALGORITHM_ROOT/'milestones.json'
    ledger=json.loads(ledger_path.read_text()) if ledger_path.is_file() else {}
    current_gate=next((gate['id'] for gate in ledger.get('milestones',[]) if gate['status']!='passed'),result['milestone'])
    progress.update(updated_at=datetime.now(timezone.utc).isoformat(), current_gate=current_gate,evaluation_gate=result['milestone'],
                    simulator=identity['backend'], variant=f'passive_{args.pose}_gain{args.gain_scale}', assistance_coefficient=args.assistance,
                    active_process=None, throughput=performance, validation=metrics, failure_metrics=failures,
                    next_step='diagnose earliest failed passive criterion; require full proof and visual review before promotion',
                    artifact_paths=[str(out)], iteration=None, checkpoint=None)
    write_json(progress_path, progress)
    print(json.dumps(result, indent=2))
    return result
