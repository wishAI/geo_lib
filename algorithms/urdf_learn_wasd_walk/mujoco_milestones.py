"""Certify explicit MuJoCo/model-specific milestones from immutable evidence.

The historical Isaac ledger is archived, never translated into MuJoCo passes.
Standing certificates require full trajectory/video evidence and visual review.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import shlex

ROOT = Path(__file__).resolve().parents[2]
ALG = Path(__file__).resolve().parent
LEDGER = ALG / 'milestones.json'
LINEAGE = 'landau_balanced_hands_mujoco_20260922'
STANDING = ('stand_zero_signal_30s_no_reset', 'stand_30s_no_reset')


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def artifact(path, kind):
    path = Path(path).resolve()
    return {'kind': kind, 'path': str(path.relative_to(ROOT)), 'sha256': digest(path)}


def initialize(folder):
    folder = Path(folder).resolve()
    folder.relative_to((ALG / 'outputs').resolve())
    previous = read(LEDGER)
    if previous['lineage'] == LINEAGE:
        return previous
    record = read(folder / 'dynamics.json')
    require(record['identity']['backend'] == 'mujoco_warp_cuda', 'Expected validated MuJoCo backend')
    variant = read(folder / 'mass_variant.json')
    require(variant['profile'] == 'balanced_hands_v1', 'Unexpected mass profile')
    require(not variant['geometry_changed'] and not variant['inertial_origins_changed'], 'Unauthorized geometry/inertial-origin change')
    require(abs(variant['total_mass_before_kg'] - variant['total_mass_after_kg']) < 1e-9, 'Mass not conserved')
    archive = ALG / 'outputs/milestone_lineages' / previous['lineage']
    archive.mkdir(parents=True, exist_ok=True)
    old = archive / 'milestones.json'
    require(not old.exists() or old.read_bytes() == LEDGER.read_bytes(), 'Archive collision')
    old.write_bytes(LEDGER.read_bytes())
    evolution = ALG / 'outputs/evolution.json'
    if evolution.is_file() and not (archive / 'evolution.json').exists():
        (archive / 'evolution.json').write_bytes(evolution.read_bytes())
    ledger = {'version': 2, 'lineage': LINEAGE, 'backend': 'mujoco_warp_cuda',
        'implementationStatus': 'milestone_1_in_progress', 'historyCarriedForward': False,
        'assetContract': {**previous['assetContract'], 'modelXmlSha256': record['identity']['model_xml_sha256'],
                         'massProfile': 'balanced_hands_v1', 'massVariant': variant,
                         'physicsIdentity': {k: record['identity'][k] for k in
                           ('mujoco_version', 'warp_version', 'mujoco_warp_version')},
                         'initialControlSha256': record['identity']['initial_control_sha256']},
        'previousBackendLedger': artifact(old, 'archived_isaac_milestones'),
        'invalidatedLineages': previous.get('invalidatedLineages', []),
        'acceptance': {'standing_max_drift_m': .03, 'standing_max_heading_change_deg': 5.,
                       'standing_duration_s': 30., 'assistance': 0.,
                       'same_checkpoint_cumulative_validation': True},
        'milestones': [{**{k: item[k] for k in ('order', 'id', 'stage')},
                        'status': 'in_progress' if item['order'] == 1 else 'not_started'}
                       for item in previous['milestones']]}
    check_standing(folder, ledger)
    write(LEDGER, ledger)
    sync_gui(ledger)
    return ledger


def check_standing(folder, ledger, checkpoint=None):
    import numpy as np
    from algorithms.urdf_learn_wasd_walk.passive_stand import evaluate_gate, evaluate_free_root_support
    import xml.etree.ElementTree as ET
    folder = Path(folder).resolve()
    folder.relative_to((ALG / 'outputs').resolve())
    result = read(folder / 'dynamics.json')
    proof = read(folder / 'proof_metadata.json')
    review = read(folder / 'visual_review.json')
    identity, metrics, cfg = result['identity'], result['metrics'], result['config']
    contract = ledger['assetContract']
    require(identity['config_sha256'] == hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest(), 'Configuration hash mismatch')
    require(result['performance']['warp_setup']['physics_steps'] == 15000, 'Incomplete physics trace')
    expected = STANDING[int(checkpoint is not None)]
    require(result['milestone'] == expected, 'Wrong milestone evidence')
    require(result['status'] == 'dynamics_passed_proof_pending' and not result['failures'], 'Dynamics failed')
    require(identity['backend'] == ledger['backend'] == 'mujoco_warp_cuda', 'Physics backend mismatch')
    for key, wanted in [('urdf_sha256', contract['urdfSha256']), ('mesh_tree_sha256', contract['meshTreeSha256']),
                        ('model_xml_sha256', contract['modelXmlSha256']),
                        ('initial_control_sha256', contract['initialControlSha256']),
                        *contract['physicsIdentity'].items()]:
        require(identity.get(key) == wanted, f'Identity mismatch: {key}')
    require(digest(folder / 'model.xml') == identity['model_xml_sha256'], 'Replay model differs from simulated model')
    original_metrics = dict(metrics)
    if checkpoint is not None:
        original_metrics['max_abs_action'] = 0.  # Passive-only action predicate.
    valid, failures = evaluate_gate(original_metrics)
    require(valid and not evaluate_free_root_support(metrics), 'Standing physical acceptance failed: ' + str(failures))
    urdf = ALG / 'inputs/landau_v10/landau_v10_parallel_mesh.urdf'
    require(digest(urdf) == contract['urdfSha256'], 'Source URDF changed')
    source = ET.parse(urdf).getroot()
    tree = hashlib.sha256()
    for path in sorted({(urdf.parent / mesh.get('filename')).resolve() for mesh in source.findall('.//mesh')}):
        tree.update(path.name.encode())
        tree.update(bytes.fromhex(digest(path)))
    require(tree.hexdigest() == contract['meshTreeSha256'], 'Source mesh tree changed')
    limits = [float(j.find('limit').get('velocity')) for j in ET.parse(urdf).getroot().findall('joint')
              if j.get('type') != 'fixed' and j.find('limit') is not None]
    require(metrics['max_joint_speed_rad_s'] <= min(limits) + 1e-6, 'URDF joint speed exceeded')
    for key in ('reset_count', 'done_count', 'fall_count', 'peak_auxiliary_wrench_norm', 'max_abs_command'):
        require(metrics[key] == 0, f'Nonzero {key}')
    require(cfg['assistance'] == 0 and cfg['forward'] == 0 and cfg['seconds'] == 30., 'Wrong standing protocol')
    require(cfg['dt'] == .002 and cfg['contact_timeconst'] == .004 and cfg['gain_scale'] == 1., 'Changed physics parameters')
    require(metrics['duration_s'] >= 30. - 1e-6, 'Incomplete standing duration')
    require(metrics['horizontal_drift_m'] <= ledger['acceptance']['standing_max_drift_m'], 'Standing drift exceeded 3 cm')
    require(not metrics['nonfoot_contacts'] and metrics['minimum_support_polygon_margin_m'] > 0, 'Invalid ground support')
    require(.5 <= metrics['mean_support_body_weight_ratio'] <= 1.5 and metrics['peak_support_body_weight_ratio'] <= 3., 'Invalid support forces')
    require(all(math.isfinite(v) for v in metrics.values() if isinstance(v, (int, float))), 'Nonfinite metric')
    if checkpoint is None:
        require(cfg['checkpoint'] is None and metrics['max_abs_action'] == metrics['policy_calls'] == 0, 'Passive gate used policy')
    else:
        checkpoint = Path(checkpoint).resolve()
        require(Path(cfg['checkpoint']).resolve() == checkpoint and cfg['checkpoint_sha256'] == digest(checkpoint), 'Checkpoint mismatch')
        training = read(checkpoint.parent / 'training.json')
        require(not str(training.get('status','')).startswith('invalidated_'), 'Training evidence was invalidated')
        if training.get('policy_family')=='periodic_feedback_cem':
            check_gait_source(folder,result,training)
        require(training['checkpoints'][checkpoint.name] == digest(checkpoint), 'Checkpoint not from completed training')
        for key in ('model_xml_sha256', 'urdf_sha256', 'mesh_tree_sha256', 'backend', 'mujoco_version'):
            require(training[key] == identity[key], f'Checkpoint identity mismatch: {key}')
        controls = {'initial_qpos': training['nominal_q'], 'initial_ctrl': training['nominal_ctrl']}
        require(hashlib.sha256(json.dumps(controls, sort_keys=True).encode()).hexdigest() == identity['initial_control_sha256'], 'Checkpoint initial controls mismatch')
        require(checkpoint.name != 'model_initial.pt' and metrics['policy_calls'] == 1500, 'No trained policy control')
    for key, name in [('video_sha256', 'proof.mp4'), ('trajectory_sha256', 'trajectory.npz'),
                      ('dynamics_sha256', 'dynamics.json'), ('model_xml_sha256', 'model.xml')]:
        require(proof[key] == digest(folder / name), f'Proof hash mismatch: {name}')
    require(result['trajectory_sha256'] == digest(folder / 'trajectory.npz'), 'Dynamics trajectory mismatch')
    require(identity['source_sha256'] == digest(folder / 'backend_source.py'), 'Dynamics source mismatch')
    require(proof['renderer_source_sha256'] == digest(folder / 'renderer_source.py'), 'Renderer source mismatch')
    require(proof['kind'] == 'state_replay_of_exact_dynamics_trajectory' and proof['character_visible_every_frame'], 'Incomplete visible replay')
    require(review.get('decision') == 'accepted' and review['video_sha256'] == proof['video_sha256'] and review['trajectory_sha256'] == proof['trajectory_sha256'], 'Missing matching visual acceptance')
    trace = np.load(folder / 'trajectory.npz')
    q, times = trace['qpos'], trace['time']
    require(np.isfinite(q).all() and np.isfinite(times).all(), 'Nonfinite trajectory')
    require(len(times) == len(q) == proof['frames'] and len(times) >= 1501, 'Incomplete trajectory/video frames')
    require(abs(times[0]) < 1e-8 and times[-1] >= 30.-1e-6 and np.allclose(np.diff(times), .02, atol=1e-6), 'Reset or gap in trajectory time')
    w, x, y, z = q[:, 3:7].T
    heading = np.unwrap(np.arctan2(2*(w*z+x*y), 1-2*(y*y+z*z)))
    heading_change = float(np.rad2deg(np.max(np.abs(heading-heading[0]))))
    require(heading_change <= ledger['acceptance']['standing_max_heading_change_deg'], 'Uncommanded heading drift')
    media = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=nb_frames,duration', '-of', 'json', str(folder / 'proof.mp4')]))['streams'][0]
    require(int(media['nb_frames']) == len(q) and float(media['duration']) >= 30., 'Encoded video incomplete')
    return result, {**metrics, 'max_heading_change_degrees': heading_change}


def check_gait_source(folder,result,training):
    expected=[value for path,value in training['source_sha256'].items() if Path(path).name=='landau_gait_search.py']
    require(len(expected)==1 and result.get('gait_source_sha256')==expected[0], 'Gait source not bound to training')
    require(digest(folder/'gait_source.py')==expected[0], 'Gait source snapshot changed')
    require(result['controller_source_sha256']==digest(folder/'controller_source.py'), 'Gait evaluation adapter changed')
    if training.get('command_extension')=='yaw_v1':
        require(digest(folder/'turn_source.py')==training['turn_source_sha256'], 'Cumulative controller omits trained turn source')
        require(result.get('turn_source_sha256')==training['turn_source_sha256'], 'Dynamics omits trained turn source')
        require(result.get('controller_memory_sha256')==digest(folder/'controller_memory.json'), 'Controller memory evidence changed')
        memory=read(folder/'controller_memory.json')
        require(memory.get('dispatch')=='command_driven_yaw_extension', 'Cumulative evaluation bypasses turn controller')
        require(memory['turn_source_sha256']==training['turn_source_sha256'], 'Cumulative turn controller source mismatch')
        if not result['config'].get('turn') and not result['config'].get('teleop') and not result['config'].get('direction'):
            require(memory['turned'] is False and memory['integrated_reference_rad']==0., 'Zero-yaw cumulative evaluation entered turn mode')


def check_walking(folder, ledger, checkpoint, distance=5., turn=False, teleop=False, direction=None):
    """Recheck the full distance run; only standing has a 30 s contract."""
    import numpy as np
    folder=Path(folder).resolve(); checkpoint=Path(checkpoint).resolve()
    folder.relative_to((ALG/'outputs').resolve())
    result=read(folder/'dynamics.json'); metrics=result['metrics']; cfg=result['config']
    identity=result['identity']; contract=ledger['assetContract']
    require(distance in (5.,10.), 'Unsupported walking gate')
    require(sum((bool(turn),bool(teleop),bool(direction)))<=1,'Choose one walking protocol')
    if direction:
        from algorithms.urdf_learn_wasd_walk import landau_direction_contract as dc
        require(direction in dc.DIRECTIONS and distance==10.,'Wrong directional gate')
    expected='gate_10m_four_directions_no_reset' if direction else 'teleop_60s_forward_turn' if teleop else 'yaw_turn_90deg_hold' if turn else f'gate_{distance:g}m_no_reset'
    require(result['milestone']==expected and not result['failures'], 'Walking dynamics failed')
    require(cfg.get('direction')==direction and bool(cfg.get('turn',False))==turn and bool(cfg.get('teleop',False))==teleop, 'Command protocol mismatch')
    require(result['status']=='dynamics_passed_proof_pending', 'Wrong walking status')
    require(identity['backend']==ledger['backend']=='mujoco_warp_cuda', 'Walking backend mismatch')
    for key,wanted in [('urdf_sha256',contract['urdfSha256']),('mesh_tree_sha256',contract['meshTreeSha256']),
                       ('model_xml_sha256',contract['modelXmlSha256']),('initial_control_sha256',contract['initialControlSha256']),
                       *contract['physicsIdentity'].items()]:
        require(identity.get(key)==wanted, 'Walking identity mismatch: '+key)
    require(digest(folder/'model.xml')==contract['modelXmlSha256'], 'Walking replay model mismatch')
    require(identity['config_sha256']==hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest(), 'Walking config mismatch')
    # TRAINING_RULES assigns 30 s to standing, not a walking deadline.
    duration=float(cfg['seconds'])
    require(math.isfinite(duration) and (24. if turn else 30.)<=duration<=(240. if direction else 120.), 'Walking diagnostic duration outside supported bounds')
    physics_steps=round(duration/.002); control_steps=round(duration/.02)
    require(abs(control_steps*.02-duration)<1e-8, 'Walking duration not aligned to control steps')
    require(cfg['assistance']==0 and cfg['dt']==.002 and 0<cfg['forward']<=.4, 'Wrong walking protocol')
    if not turn and not teleop:require(cfg.get('target_distance_m',5.)==distance, 'Walking distance protocol mismatch')
    require(cfg['gain_scale']==1. and cfg['contact_timeconst']==.004, 'Walking physics changed')
    require(Path(cfg['checkpoint']).resolve()==checkpoint and cfg['checkpoint_sha256']==digest(checkpoint), 'Walking checkpoint mismatch')
    training=read(checkpoint.parent/'training.json')
    require(not str(training.get('status','')).startswith('invalidated_'), 'Training evidence was invalidated')
    if training.get('policy_family')=='periodic_feedback_cem':
        check_gait_source(folder,result,training)
    require(checkpoint.name!='model_initial.pt' and training['checkpoints'][checkpoint.name]==digest(checkpoint), 'Walking checkpoint not trained')
    for key in ('model_xml_sha256','urdf_sha256','mesh_tree_sha256','backend','mujoco_version'):
        require(training[key]==identity[key], 'Walking training mismatch: '+key)
    controls={'initial_qpos':training['nominal_q'],'initial_ctrl':training['nominal_ctrl']}
    require(hashlib.sha256(json.dumps(controls,sort_keys=True).encode()).hexdigest()==identity['initial_control_sha256'], 'Walking initial controls mismatch')
    require(result['performance']['warp_setup']['physics_steps']==physics_steps and metrics['policy_calls']==control_steps, 'Incomplete walking control')
    for key in ('reset_count','done_count','fall_count','peak_auxiliary_wrench_norm'):
        require(metrics[key]==0, 'Walking '+key+' must be zero')
    require(all(math.isfinite(v) for v in metrics.values() if isinstance(v,(int,float))), 'Nonfinite walking metric')
    require(abs(metrics['duration_s']-duration)<1e-6, 'Walking did not complete the recorded duration')
    if turn:
        from algorithms.urdf_learn_wasd_walk.landau_turn_control import evaluate_turn_gate
        require(not evaluate_turn_gate(metrics,duration), 'Turn or settled hold failed')
        require(training.get('policy_family')=='periodic_feedback_cem', 'Unsupported turn policy family')
        require(training.get('command_extension')=='yaw_v1', 'Checkpoint lacks trained yaw control')
        require(training['turn_source_sha256']==digest(checkpoint.parent/'turn_source.py')==digest(folder/'turn_source.py'), 'Turn source mismatch')
        require(result['turn_validator_source_sha256']==digest(folder/'turn_validator_source.py'), 'Turn validator source mismatch')
    elif not teleop and not direction:
        require(metrics['semantic_forward_displacement_m']>=distance, f'Walking did not reach {distance:g} m')
        require(abs(metrics['semantic_strafe_displacement_m'])<=.75, 'Walking lateral drift exceeded .75 m')
    require(metrics['max_reference_tilt_rad']<=math.pi/6 and metrics['root_height_drop_m']<=.08, 'Walking collapse')
    require(metrics['max_joint_speed_rad_s']<=4.+1e-6 and metrics['max_abs_action']<=1.+1e-6, 'Walking exceeded joint/action limits')
    require(not metrics['nonfoot_contacts'] and .5<=metrics['mean_support_body_weight_ratio']<=1.5 and metrics['peak_support_body_weight_ratio']<=3., 'Walking invalid ground support')
    for name in ('left_hip_pitch_joint','right_hip_pitch_joint','left_knee_joint','right_knee_joint'):
        require(metrics['leg_joint_excursion_rad'][name]>=.05, 'Frozen walking leg: '+name)
    require(result['foot_trace_sha256']==digest(folder/'foot_trace.json'), 'Walking foot trace mismatch')
    feet=read(folder/'foot_trace.json'); require(len(feet)==physics_steps, 'Incomplete foot trace')
    times=np.array([row['time_s'] for row in feet])
    require(np.isfinite(times).all() and abs(times[0]-.002)<1e-6 and abs(times[-1]-duration)<1e-6 and np.allclose(np.diff(times),.002,atol=1e-6), 'Foot trace time gap/reset')
    states={side:dict(touched=False,air=0.,peak=0.,count=0,turn_count=0) for side in ('left','right')}
    flight=peak_flight=0.; air_count=0
    for row in feet:
        require(all(math.isfinite(v) for v in row.values() if isinstance(v,(int,float))), 'Nonfinite foot trace')
        airborne=not row['left_contact'] and not row['right_contact']
        flight=flight+.002 if airborne else 0.; peak_flight=max(peak_flight,flight); air_count+=int(airborne)
        for side,state in states.items():
            if row[side+'_contact']:
                if state['touched'] and state['air']>=.06 and state['peak']>=.015:
                    state['count']+=1
                    if turn and training['command_profile']['turn_start_s']<=row['time_s']<=training['command_profile']['turn_end_s']:
                        state['turn_count']+=1
                state.update(touched=True,air=0.,peak=0.)
            elif state['touched']:
                state['air']+=.002;state['peak']=max(state['peak'],row[side+'_clearance_m'])
    require(peak_flight<=.12 and air_count/len(feet)<=.05, 'Hopping rather than walking')
    require(np.mean([row['mean_slip_mps'] for row in feet])<=.1, 'Sliding rather than walking')
    for side,state in states.items():
        require(state['count']>=3 and state['count']==metrics[side+'_completed_swings'], 'Incomplete or mismatched '+side+' swings')
        if turn:require(state['turn_count']>=3, 'No repeated '+side+' swings during yaw commands')
    proof=read(folder/'proof_metadata.json'); review=read(folder/'visual_review.json')
    for key,name in [('video_sha256','proof.mp4'),('trajectory_sha256','trajectory.npz'),('dynamics_sha256','dynamics.json'),('model_xml_sha256','model.xml')]:
        require(proof[key]==digest(folder/name), 'Walking proof mismatch: '+name)
    require(result['trajectory_sha256']==digest(folder/'trajectory.npz') and identity['source_sha256']==digest(folder/'backend_source.py'), 'Walking dynamics source/trajectory mismatch')
    require(result['controller_source_sha256']==digest(folder/'controller_source.py') and proof['renderer_source_sha256']==digest(folder/'renderer_source.py'), 'Walking controller/renderer mismatch')
    require(proof['kind']=='state_replay_of_exact_dynamics_trajectory' and proof['character_visible_every_frame'], 'Walking replay is incomplete')
    require(review.get('decision')=='accepted' and review['video_sha256']==proof['video_sha256'] and review['trajectory_sha256']==proof['trajectory_sha256'], 'Walking visual review missing')
    trace=np.load(folder/'trajectory.npz'); q,t=trace['qpos'],trace['time']
    require(np.isfinite(q).all() and np.isfinite(t).all() and len(q)==len(t)==proof['frames']==control_steps+1, 'Incomplete walking trajectory')
    require(abs(t[0])<1e-8 and abs(t[-1]-duration)<1e-6 and np.allclose(np.diff(t),.02,atol=1e-6), 'Walking trajectory time gap/reset')
    w,x,y,z=q[:,3:7].T; heading=np.unwrap(np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z)))
    if turn:
        hold_start=float(cfg['turn_hold_start'])
        require(hold_start==training['command_profile']['hold_start_s'] and duration>=hold_start+5., 'Turn hold protocol differs')
        hold=t>=duration-5.-1e-8
        error=heading[hold]-heading[0]-math.pi/2
        require(np.max(np.abs(error))<=math.radians(5), 'Trajectory does not corroborate 90 degree hold')
        stopped=q[t>=hold_start-1e-8,:2]
        require(np.linalg.norm(stopped-stopped[0],axis=1).max()<=.035, 'Trajectory does not corroborate stationary hold')
        require(result['policy_trace_sha256']==digest(folder/'policy_trace.npz'), 'Turn policy trace mismatch')
        policy=np.load(folder/'policy_trace.npz');obs=policy['observation'];pt=policy['time']
        require(obs.shape==(control_steps,70) and np.isfinite(obs).all() and len(pt)==control_steps, 'Incomplete turn command trace')
        require(np.allclose(pt,t[:-1],atol=1e-6), 'Turn policy clock mismatch')
        require(abs(float(obs[:,65].sum())*.02-math.pi/2)<.001, 'Yaw commands do not integrate to 90 degrees')
        require(np.max(np.abs(obs[pt>=hold_start-1e-8,63:66]))==0., 'Stopped hold has nonzero commands')
        memory=read(folder/'controller_memory.json')
        require(memory['simulation_reset'] is False and abs(memory['anchor_time']-hold_start)<.021, 'Standing handoff changed reset semantics')
        require(memory['turn_source_sha256']==training['turn_source_sha256'], 'Standing handoff source mismatch')
    elif not teleop and not direction:
        require(np.max(np.abs(heading-heading[0]))<=math.pi/6, 'Walking heading exceeded 30 degrees')
        require(q[-1,1]-q[0,1]>=distance-.05, 'Trajectory does not corroborate forward displacement')
    if teleop:
        from algorithms.urdf_learn_wasd_walk import landau_teleop_contract as tc
        import mujoco
        require(duration==60. and training.get('memory_version')==2, 'Wrong teleop controller or duration')
        require(training.get('command_extension')=='yaw_v1', 'Teleop controller missing yaw support')
        require(result['teleop_validator_source_sha256']==digest(folder/'teleop_validator_source.py'), 'Teleop validator snapshot changed')
        require(result['command_protocol_source_sha256']==training['teleop_contract_sha256']==digest(folder/'command_protocol_source.py')==digest(checkpoint.parent/'teleop_contract.py'), 'Teleop command source mismatch')
        require(result['policy_trace_sha256']==digest(folder/'policy_trace.npz'), 'Teleop command trace mismatch')
        policy=np.load(folder/'policy_trace.npz');obs=policy['observation'];pt=policy['time']
        require(obs.shape==(3000,70) and np.isfinite(obs).all() and np.allclose(pt,t[:-1],atol=1e-6), 'Incomplete teleop commands')
        expected_commands=np.array([tc.command_profile(round(float(v)/.02)*.02) for v in pt])
        require(np.allclose(obs[:,63:66],expected_commands,atol=1e-7,rtol=0), 'Teleop did not receive the declared joystick commands')
        model=mujoco.MjModel.from_xml_path(str(folder/'model.xml'));data=mujoco.MjData(model)
        positions=[];headings=[]
        for frame in q:
            data.qpos[:]=frame;mujoco.mj_forward(model,data)
            positions.append(data.xpos[model.body('root_x').id].copy())
            rot=data.xmat[model.body('base_link').id].reshape(3,3)
            headings.append(math.atan2(rot[1,0],rot[0,0]))
        responses=tc.response_metrics(t,np.array(positions),np.array(headings),feet)
        require(not tc.evaluate_gate(metrics,responses), 'Teleop reconstructed response failed: '+str(tc.evaluate_gate(metrics,responses)))
        reported=metrics['teleop_response']
        require(not tc.evaluate_gate(metrics,reported), 'Reported teleop response failed')
        for actual,claimed in zip(responses['blocks'],reported['blocks']):
            for key in ('forward_progress_m','heading_change_rad','max_heading_excursion_rad'):
                require(abs(actual[key]-claimed[key])<1e-5, 'Teleop response differs from trajectory: '+key)
            require(actual['completed_swings']==claimed['completed_swings'], 'Teleop swing accounting differs')
        for actual,claimed in zip(responses['holds'],reported['holds']):
            for key in ('drift_m','settled_speed_mps','heading_drift_rad'):
                # GPU float32 positions and CPU FK differ by sub-micrometres;
                # a20ms finite difference amplifies this in speed (observed1.2e-5m/s).
                # Both reconstructed and reported metrics must still pass the full gate.
                tolerance=1e-4 if key=='settled_speed_mps' else 1e-5
                require(abs(actual[key]-claimed[key])<tolerance, 'Teleop hold differs from trajectory: '+key)
        memory=read(folder/'controller_memory.json')
        require(memory['simulation_reset'] is False and memory.get('memory_version')==2, 'Wrong teleop memory semantics')
        stops=[e['time_s'] for e in memory['events'] if e['kind']=='stop_anchor']
        restarts=[e['time_s'] for e in memory['events'] if e['kind']=='restart']
        require(len(stops)==2 and np.allclose(stops,[20.,50.],atol=.021) and len(restarts)==1 and abs(restarts[0]-25.02)<.021, 'Missing repeated-stop/restart memory evidence')
        require(abs(memory['integrated_reference_rad'])<1e-5, 'Teleop heading reference did not follow both yaw signs')
    if direction:
        import mujoco
        require(training.get('command_extension')=='yaw_v1' and training.get('memory_version')==2, 'Direction controller missing command memory')
        require(result['direction_protocol_source_sha256']==digest(folder/'direction_protocol_source.py')==digest(dc.__file__), 'Direction command source mismatch')
        require(result['policy_trace_sha256']==digest(folder/'policy_trace.npz'), 'Direction command trace mismatch')
        with np.load(folder/'policy_trace.npz') as policy:
            obs=policy['observation'];pt=policy['time']
        require(obs.shape==(control_steps,70) and np.isfinite(obs).all() and np.allclose(pt,t[:-1],atol=1e-6), 'Incomplete direction commands')
        require(np.allclose(q[0],training['nominal_q'],atol=1e-7,rtol=0), 'Direction start pose changed')
        model=mujoco.MjModel.from_xml_path(str(folder/'model.xml'));data=mujoco.MjData(model)
        positions=[];headings=[]
        for frame in q:
            data.qpos[:]=frame;mujoco.mj_forward(model,data)
            positions.append(data.xpos[model.body('root_x').id].copy())
            rot=data.xmat[model.body('base_link').id].reshape(3,3)
            headings.append(math.atan2(rot[1,0],rot[0,0]))
        positions=np.array(positions)
        measured=dc.gate_metrics(direction,t,positions)
        for key,value in measured.items():
            if isinstance(value,(float,int)):
                require(abs(value-metrics[key])<1e-4, 'Direction metric differs from trajectory: '+key)
            else:require(value==metrics[key], 'Direction crossing differs from trajectory')
        require(not dc.evaluate_gate({**metrics,**measured},duration), 'Reconstructed directional gate failed')
        commands=np.array([dc.command(direction,xy[:2]-positions[0,:2],heading,float(age),cfg['forward'])
                           for xy,heading,age in zip(positions[:-1],headings[:-1],pt)])
        require(np.allclose(obs[:,63:66],commands,atol=1e-6,rtol=0), 'Direction did not receive declared joystick commands')
        memory=read(folder/'controller_memory.json')
        require(memory['simulation_reset'] is False and memory.get('memory_version')==2, 'Wrong direction memory semantics')
        require(abs(memory['integrated_reference_rad']-float(obs[:-1,65].sum())*.02)<1e-4, 'Direction heading reference differs from commands')
    media=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=nb_frames,duration','-of','json',str(folder/'proof.mp4')]))['streams'][0]
    require(int(media['nb_frames'])==control_steps+1 and float(media['duration'])>=duration, 'Walking video incomplete')
    return result,metrics


def sync_gui(ledger):
    path = ALG / 'gui/manifest.json'
    gui = read(path)
    states = {m['id']: m['status'].replace('_', ' ') for m in ledger['milestones']}
    for item in gui['milestones']:
        item['status'] = states[item['id']]
    gui['runtimeLabel'] = 'TK2 · MuJoCo · balanced_hands_v1'
    write(path, gui)


def reproduction_job(result, checkpoint):
    cfg=result['config']
    if checkpoint:
        training=read(Path(checkpoint).parent/'training.json')
        walking=training['observation_dim']==70
        module='landau_forward_control' if walking else 'landau_rsl_control'
        args=['--mode','evaluate','--name',cfg['name']+'__replay','--checkpoint',str(Path(checkpoint).resolve())]
        if walking:args+=['--forward',str(cfg['forward']),'--seconds',str(cfg['seconds']),'--target-distance',str(cfg.get('target_distance_m',5.))]
        if cfg.get('turn'):args+=['--turn']
        if cfg.get('teleop'):args+=['--teleop']
        if cfg.get('direction'):args+=['--direction',cfg['direction']]
    else:
        # Historical dynamics accidentally named the underlying backend module;
        # retain its exact adapter arguments and record the correct entry point.
        original=shlex.split(result['reproduce'])
        args=original[original.index('-m')+2:]
        require('--baseline' in args and '--name' in args, 'Passive reproduction arguments unavailable')
        args[args.index('--name')+1]=cfg['name']+'__replay'
        module='continuation'
    return {'kind':'module','module':'algorithms.urdf_learn_wasd_walk.'+module,'args':args,'timeout_s':1200 if cfg.get('direction') else 240}


def certify(folder, checkpoint=None, passive_folder=None, standing_folder=None, five_metre_folder=None, ten_metre_folder=None, turn_folder=None):
    ledger = read(LEDGER)
    require(ledger['lineage'] == LINEAGE, 'Initialize the explicit model/backend lineage first')
    index = 5 if turn_folder is not None else 4 if ten_metre_folder is not None else 3 if five_metre_folder is not None else 2 if standing_folder is not None else int(checkpoint is not None)
    target = ledger['milestones'][index]
    require(all(item['status'] == 'passed' for item in ledger['milestones'][:index]), 'Prior milestone unresolved')
    require(target['status'] == 'in_progress', 'Only the first unresolved milestone can be certified')
    result, metrics = check_walking(folder, ledger, checkpoint, 10. if index==3 else 5., turn=index==4,teleop=index==5) if index >= 2 else check_standing(folder, ledger, checkpoint)
    components = []
    if checkpoint:
        require(passive_folder is not None, 'Fresh cumulative passive component required')
        passive_result, _ = check_standing(passive_folder, ledger)
        require(datetime.fromisoformat(passive_result['created_at']).timestamp() >= Path(checkpoint).stat().st_mtime, 'Passive replay predates candidate checkpoint')
        components.append(artifact(Path(passive_folder) / 'dynamics.json', 'cumulative_passive_dynamics'))
        components.append(artifact(Path(passive_folder) / 'proof_metadata.json', 'cumulative_passive_proof'))
    if index >= 2:
        standing_result, _ = check_standing(standing_folder, ledger, checkpoint)
        require(datetime.fromisoformat(standing_result['created_at']).timestamp() >= Path(checkpoint).stat().st_mtime, 'Standing replay predates candidate checkpoint')
        components.extend([artifact(Path(standing_folder)/'dynamics.json','cumulative_policy_stand_dynamics'),
                           artifact(Path(standing_folder)/'proof_metadata.json','cumulative_policy_stand_proof')])
    if index >= 3:
        require(standing_folder is not None, 'Fresh cumulative policy standing required')
        five_result, _ = check_walking(five_metre_folder, ledger, checkpoint, 5.)
        require(datetime.fromisoformat(five_result['created_at']).timestamp() >= Path(checkpoint).stat().st_mtime, '5 m replay predates candidate checkpoint')
        components.extend([artifact(Path(five_metre_folder)/'dynamics.json','cumulative_5m_dynamics'),
                           artifact(Path(five_metre_folder)/'proof_metadata.json','cumulative_5m_proof')])
    if index >= 4:
        ten_result,_=check_walking(ten_metre_folder,ledger,checkpoint,10.)
        require(datetime.fromisoformat(ten_result['created_at']).timestamp()>=Path(checkpoint).stat().st_mtime, '10 m replay predates candidate checkpoint')
        components.extend([artifact(Path(ten_metre_folder)/'dynamics.json','cumulative_10m_dynamics'),
                           artifact(Path(ten_metre_folder)/'proof_metadata.json','cumulative_10m_proof')])
    if index >= 5:
        turn_result,_=check_walking(turn_folder,ledger,checkpoint,turn=True)
        require(datetime.fromisoformat(turn_result['created_at']).timestamp()>=Path(checkpoint).stat().st_mtime, 'Turn replay predates candidate checkpoint')
        components.extend([artifact(Path(turn_folder)/'dynamics.json','cumulative_turn_dynamics'),
                           artifact(Path(turn_folder)/'proof_metadata.json','cumulative_turn_proof')])
    folder = Path(folder).resolve()
    evidence = [artifact(folder / name, kind) for name, kind in [('dynamics.json', 'dynamics_validation'),
        ('proof_metadata.json', 'proof_validation'), ('visual_review.json', 'visual_review'),
        ('proof.mp4', 'video'), ('trajectory.npz', 'trajectory'), ('model.xml', 'model')]]
    if index >= 2:
        evidence.append(artifact(folder/'foot_trace.json','walking_foot_trace'))
        evidence.append(artifact(folder/'controller_source.py','controller_source'))
    policy_kind=read(Path(checkpoint).parent/'training.json').get('policy_family','rsl_rl_ppo') if checkpoint else None
    if checkpoint:
        evidence.append(artifact(Path(checkpoint).parent/'training.json','checkpoint_training_metadata'))
    if policy_kind=='periodic_feedback_cem':
        evidence.append(artifact(folder/'gait_source.py','trained_gait_source'))
    if index==4:
        evidence.extend(artifact(folder/name,kind) for name,kind in [('turn_source.py','trained_turn_source'),
            ('turn_validator_source.py','turn_validator_source'),('policy_trace.npz','command_and_policy_trace'),
            ('controller_memory.json','standing_handoff_memory')])
    if index==5:
        evidence.extend(artifact(folder/name,kind) for name,kind in [('turn_source.py','trained_turn_source'),
            ('teleop_validator_source.py','teleop_validator_source'),('policy_trace.npz','command_and_policy_trace'),
            ('command_protocol_source.py','joystick_protocol'),('controller_memory.json','command_memory')])
    ckpt = {'kind': policy_kind, 'path': str(Path(checkpoint).resolve().relative_to(ROOT)), 'sha256': digest(checkpoint)} if checkpoint else {'kind': 'passive_pd_configuration', 'identity': ledger['assetContract']['modelXmlSha256']}
    final = {'status': 'passed', 'lineage': LINEAGE, 'backend': ledger['backend'], 'milestone': target['id'],
             'assembled_at': datetime.now(timezone.utc).isoformat(), 'checkpoint': ckpt,
             'identity': result['identity'], 'metrics': metrics, 'evidence': evidence,
             'cumulative_components': components, 'candidate_checkpoint_sha256': digest(checkpoint) if checkpoint else None, 'scope': 'MuJoCo balanced_hands_v1 only; no Isaac or real-world claim',
             'reproduction_job':reproduction_job(result,checkpoint),
             'finalizer_source_sha256': digest(__file__)}
    (folder/'milestone_finalizer_source.py').write_text(Path(__file__).read_text())
    write(folder / 'milestone_validation.json', final)
    target.update(status='passed', passedAt=final['assembled_at'], checkpoint=ckpt, metrics=metrics,
                  evidence=[artifact(folder / 'milestone_validation.json', 'validation'), *evidence])
    ledger['milestones'][index+1]['status'] = 'in_progress'
    ledger['implementationStatus'] = f'milestone_{index+1}_passed_milestone_{index+2}_in_progress'
    write(LEDGER, ledger)
    sync_gui(ledger)
    return final


def certify_directions(folder, checkpoint, direction_folders, cumulative_folders):
    """Promote M7 only with four independent world gates and exact M1–M6 rechecks."""
    from algorithms.urdf_learn_wasd_walk.landau_direction_contract import DIRECTIONS
    ledger=read(LEDGER);checkpoint=Path(checkpoint).resolve();folder=Path(folder).resolve()
    folder.relative_to((ALG/'outputs').resolve())
    require(ledger['lineage']==LINEAGE, 'Wrong direction lineage')
    require(all(m['status']=='passed' for m in ledger['milestones'][:6]), 'Prior milestone unresolved')
    target=ledger['milestones'][6]
    require(target['id']=='gate_10m_four_directions_no_reset' and target['status']=='in_progress', 'Only unresolved M7 can be certified')
    require(len(direction_folders)==4 and len({Path(p).resolve() for p in direction_folders})==4, 'Four independent direction folders required')
    require(len(cumulative_folders)==6 and all(p is not None for p in cumulative_folders), 'Exact M1–M6 components required')
    require(not folder.exists(), 'Certificate folder already exists')
    directions={};evidence=[];components=[]
    candidate_time=checkpoint.stat().st_mtime
    for path in direction_folders:
        path=Path(path);direction=read(path/'dynamics.json')['config'].get('direction')
        require(direction in DIRECTIONS and direction not in directions, 'Missing or duplicate direction')
        result,metrics=check_walking(path,ledger,checkpoint,10.,direction=direction)
        require(datetime.fromisoformat(result['created_at']).timestamp()>=candidate_time, 'Direction replay predates checkpoint')
        directions[direction]={'metrics':metrics,'reproduction_job':reproduction_job(result,checkpoint)}
        for name in ('dynamics.json','proof_metadata.json','visual_review.json','proof.mp4','trajectory.npz',
                     'policy_trace.npz','foot_trace.json','direction_protocol_source.py','controller_memory.json',
                     'turn_source.py','gait_source.py','controller_source.py','backend_source.py','model.xml'):
            evidence.append(artifact(path/name,direction+'_'+name))
    require(set(directions)==set(DIRECTIONS), 'All world directions required')
    for index,path in enumerate(cumulative_folders):
        path=Path(path)
        if index<2:result,_=check_standing(path,ledger,checkpoint if index else None)
        else:result,_=check_walking(path,ledger,checkpoint,10. if index==3 else 5.,turn=index==4,teleop=index==5)
        require(datetime.fromisoformat(result['created_at']).timestamp()>=candidate_time, 'Cumulative replay predates checkpoint')
        components.extend(artifact(path/name,f'cumulative_m{index+1}_{name}')
                          for name in ('dynamics.json','proof_metadata.json','visual_review.json','proof.mp4'))
    ckpt={'kind':read(checkpoint.parent/'training.json')['policy_family'],
          'path':str(checkpoint.relative_to(ROOT)),'sha256':digest(checkpoint)}
    evidence.append(artifact(checkpoint.parent/'training.json','checkpoint_training_metadata'))
    metrics={'directions':{k:v['metrics'] for k,v in directions.items()}}
    final={'status':'passed','lineage':LINEAGE,'backend':ledger['backend'],'milestone':target['id'],
           'assembled_at':datetime.now(timezone.utc).isoformat(),'checkpoint':ckpt,'metrics':metrics,
           'evidence':evidence,'cumulative_components':components,'directions':directions,
           'scope':'Turn and walk through four world gates; unchanged nominal start; no body-relative strafe claim',
           'finalizer_source_sha256':digest(__file__)}
    folder.mkdir()
    (folder/'milestone_finalizer_source.py').write_text(Path(__file__).read_text())
    write(folder/'milestone_validation.json',final)
    target.update(status='passed',passedAt=final['assembled_at'],checkpoint=ckpt,metrics=metrics,
                  evidence=[artifact(folder/'milestone_validation.json','validation'),*evidence])
    ledger['milestones'][7]['status']='in_progress'
    ledger['implementationStatus']='milestone_7_passed_milestone_8_in_progress'
    write(LEDGER,ledger);sync_gui(ledger)
    return final


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=('init', 'certify', 'certify-directions'))
    p.add_argument('--directory', required=True, type=Path)
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--passive-directory', type=Path)
    p.add_argument('--standing-directory', type=Path)
    p.add_argument('--five-metre-directory', type=Path)
    p.add_argument('--ten-metre-directory', type=Path)
    p.add_argument('--turn-directory',type=Path)
    p.add_argument('--teleop-directory',type=Path)
    p.add_argument('--direction-directory',type=Path,action='append',default=[])
    args = p.parse_args()
    if args.mode=='certify-directions':
        if args.checkpoint is None:p.error('Direction certification requires --checkpoint')
        result=certify_directions(args.directory,args.checkpoint,args.direction_directory,
            [args.passive_directory,args.standing_directory,args.five_metre_directory,args.ten_metre_directory,args.turn_directory,args.teleop_directory])
    else:
        result = initialize(args.directory) if args.mode == 'init' else certify(args.directory, args.checkpoint, args.passive_directory, args.standing_directory, args.five_metre_directory,args.ten_metre_directory,args.turn_directory)
    print(json.dumps({'lineage': result['lineage'], 'status': result.get('status', result.get('implementationStatus'))}))


if __name__ == '__main__':
    main()
