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
        if training.get('policy_family') in ('periodic_feedback_cem','periodic_feedback_residual_ppo'):
            check_gait_source(folder,result,training)
        check_residual_source(folder,result,training,checkpoint)
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
        if not result['config'].get('turn') and not result['config'].get('teleop') and not result['config'].get('direction') and not result['config'].get('triangle'):
            require(memory['turned'] is False and memory['integrated_reference_rad']==0., 'Zero-yaw cumulative evaluation entered turn mode')



def check_yaw_hold_calibration(folder,result,training,checkpoint):
    """Independently verify bounded parameter overrides and recurse through donor lineage."""
    import torch
    checkpoint=Path(checkpoint);folder=Path(folder)
    donor=Path(training['calibration_donor_checkpoint']);old=read(donor.parent/'training.json')
    require(digest(donor)==training['calibration_donor_checkpoint_sha256'] and
        digest(donor.parent/'training.json')==training['calibration_donor_training_sha256'] and
        old['checkpoints'][donor.name]==digest(donor) and not str(old.get('status','')).startswith('invalidated_'), 'Calibration donor changed')
    require(old.get('residual_variant')=='right_balance_rebased_dual_residual_v1', 'Wrong calibration donor variant')
    candidate=torch.load(checkpoint,map_location='cpu',weights_only=False)
    original=torch.load(donor,map_location='cpu',weights_only=False)
    require(candidate.get('residual_variant')==training['residual_variant']==result.get('residual_variant'), 'Calibration variant mismatch')
    mutable={'residual_variant','arguments','checkpoints','created_at','status','canonical_milestone_pass','policy_composition'}
    for key,value in old.items():
        if key not in mutable:require(training.get(key)==value, 'Calibration metadata changed: '+key)
    require({k:v for k,v in training['arguments'].items() if k!='name'}==
        {k:v for k,v in old['arguments'].items() if k!='name'}, 'Calibration arguments changed')
    p=candidate['turn_parameters'];fixed=[i for i in range(20) if i not in (3,17)]
    require(tuple(p.shape)==(20,) and bool(torch.isfinite(p).all()) and
        torch.equal(p[fixed],original['turn_parameters'][fixed]), 'Only calibration parameters3/17 may change')
    require(-.12<=float(p[3])<=.12 and .5<=float(p[17])<=2., 'Calibration bounds changed')
    require(torch.equal(candidate['parameters'],original['parameters']), 'Calibration gait changed')
    for key in ('standing_actor_state_dict','residual_actor_state_dict','active_residual_actor_state_dict'):
        left,right=candidate[key],original[key]
        require(left.keys()==right.keys() and all(torch.equal(left[k],right[k]) for k in left), 'Calibration frozen actor changed')
    for filename in ('control_source.py','turn_source.py','teleop_contract.py','model.xml','right_rebase_source.py'):
        require(digest(checkpoint.parent/filename)==digest(donor.parent/filename), 'Calibration deployment changed: '+filename)
    for source in ('residual_source','active_residual_source'):
        require(training[source+'_sha256']==digest(checkpoint.parent/(source+'.py'))==
            digest(folder/(source+'.py'))==result.get(source+'_sha256'), 'Calibration residual source mismatch')
    require(result.get('residual_dispatch')=='deterministic_cruise_plus_active_left_hip_roll_v1', 'Calibration dispatch mismatch')
    calibration=training['yaw_hold_calibration'];index=calibration['candidate_index'];screens=calibration['screens']
    require(calibration['source_sha256']==digest(checkpoint.parent/'yaw_hold_rebase_source.py'), 'Calibration composition source changed')
    require(calibration['changed_indices']==[3,17] and isinstance(index,int) and index>=0 and len(screens)>=2 and
        len({str(Path(r['path']).resolve()) for r in screens})==len(screens) and
        len({r['sha256'] for r in screens})==len(screens), 'Calibration repeated screen contract changed')
    for evidence in screens:
        path=Path(evidence['path']);screen=read(path)
        require(digest(path)==evidence['sha256'] and screen['checkpoint_sha256']==digest(donor) and
            screen['kind']=='nominal_common_start_yaw_hold_calibration' and screen['changed_indices']==[3,17] and
            digest(path.parent/'screen_source.py')==screen['source_sha256'], 'Calibration screen changed')
        require(screen.get('training_sha256')==digest(donor.parent/'training.json'), 'Calibration screen donor metadata changed')
        for key in ('turn_source_sha256','residual_source_sha256','active_residual_source_sha256'):
            require(screen.get(key)==old[key], 'Calibration screen controller source changed')
        for filename,key in (('audit_source.py','audit_source_sha256'),('teleop_source.py','teleop_source_sha256'),
                             ('direction_training_source.py','common_start_source_sha256')):
            require(digest(path.parent/filename)==screen.get(key), 'Calibration screen helper changed')
        require(torch.equal(torch.tensor(screen['candidate_parameters'][index],dtype=p.dtype),p.cpu()), 'Calibration screen parameters changed')
        for mode in ('m5','m6'):
            row=screen['results'][mode]['candidate_records'][index]
            require(row['index']==index and len(row['eligible'])==4 and all(row['eligible']), 'Calibration screen qualification missing')
    donor_result={'residual_variant':old['residual_variant'],
        'residual_dispatch':'deterministic_cruise_plus_active_left_hip_roll_v1',
        **{key:old[key] for key in ('residual_source_sha256','active_residual_source_sha256')}}
    check_residual_source(donor.parent,donor_result,old,donor)


def check_right_controller_calibration(folder,result,training,checkpoint):
    """Independent four-gain composition audit, followed by the complete donor audit."""
    import torch
    from algorithms.urdf_learn_wasd_walk.landau_right_rebase import deployment_ast
    checkpoint=Path(checkpoint);folder=Path(folder)
    donor=Path(training['right_controller_donor_checkpoint']);old=read(donor.parent/'training.json')
    require(digest(donor)==training['right_controller_donor_checkpoint_sha256'] and
        digest(donor.parent/'training.json')==training['right_controller_donor_training_sha256'] and
        old['checkpoints'][donor.name]==digest(donor) and not str(old.get('status','')).startswith('invalidated_'), 'Right donor changed')
    require(old.get('residual_variant')=='yaw_hold_calibrated_dual_residual_v1', 'Wrong right controller donor variant')
    original=torch.load(donor,map_location='cpu',weights_only=False)
    candidate=torch.load(checkpoint,map_location='cpu',weights_only=False)
    require(candidate.get('residual_variant')==training['residual_variant']==result.get('residual_variant'), 'Right calibration variant mismatch')
    mutable={'residual_variant','arguments','checkpoints','created_at','status','canonical_milestone_pass','policy_composition'}
    for key,value in old.items():
        if key not in mutable:require(training.get(key)==value, 'Right donor metadata changed: '+key)
    require({k:v for k,v in training['arguments'].items() if k!='name'}==
        {k:v for k,v in old['arguments'].items() if k!='name'}, 'Right calibration arguments changed')
    indices=[7,8,12,13];fixed=[i for i in range(20) if i not in indices];p=candidate['turn_parameters']
    require(tuple(p.shape)==(20,) and bool(torch.isfinite(p).all()) and
        torch.equal(p[fixed],original['turn_parameters'][fixed]), 'Only right calibration parameters7/8/12/13 may change')
    require(all(lo<=float(p[i])<=hi for i,lo,hi in ((7,0.,3.),(8,-.12,.12),(12,-.06,.03),(13,-.25,.25))), 'Right calibration bounds changed')
    require(torch.equal(candidate['parameters'],original['parameters']), 'Right calibration gait changed')
    for key in ('standing_actor_state_dict','residual_actor_state_dict','active_residual_actor_state_dict'):
        a,b=candidate[key],original[key]
        require(a.keys()==b.keys() and all(torch.equal(a[k],b[k]) for k in a), 'Right calibration actor changed')
    for filename in ('control_source.py','turn_source.py','teleop_contract.py','model.xml','right_rebase_source.py','yaw_hold_rebase_source.py'):
        require(digest(checkpoint.parent/filename)==digest(donor.parent/filename), 'Right calibration deployment changed')
    for source in ('residual_source','active_residual_source'):
        require(training[source+'_sha256']==digest(checkpoint.parent/(source+'.py'))==
            digest(folder/(source+'.py'))==result.get(source+'_sha256'), 'Right calibration residual source mismatch')
    require(result.get('residual_dispatch')=='deterministic_cruise_plus_active_left_hip_roll_v1', 'Right calibration dispatch mismatch')
    evidence=training['right_controller_calibration'];search=Path(evidence['search_checkpoint']);sm=read(search.parent/'training.json')
    require(digest(checkpoint.parent/'right_controller_rebase_source.py')==evidence['source_sha256'], 'Right composition helper changed')
    require(digest(search)==evidence['search_checkpoint_sha256']==sm['checkpoints'][search.name] and
        digest(search.parent/'training.json')==evidence['search_training_sha256'] and
        sm['status']=='completed' and sm['policy_family']=='periodic_feedback_cem', 'Right search changed')
    args=sm['arguments'];contract=sm['right_controller_search']
    require(all(args[k] for k in ('right_controller_search','right_only','teleop','direction_train','common_starts')) and
        args['closed_loop_direction']=='right' and args['seconds']>=100. and args['turn_duration']==old['arguments']['turn_duration'] and
        contract['searched_indices']==evidence['changed_indices']==indices and contract['training_force_target_bw']==2.9 and
        contract['hard_force_limit_bw']==3., 'Right search contract changed')
    for key in ('model_xml_sha256','urdf_sha256','mesh_tree_sha256','action_joints','action_scale','nominal_q','nominal_ctrl'):
        require(sm[key]==old[key], 'Right search model changed')
    require(digest(search.parent/'turn_source.py')==sm['turn_source_sha256'] and
        deployment_ast(search.parent/'turn_source.py')==deployment_ast(donor.parent/'turn_source.py') and
        digest(search.parent/'control_source.py')==digest(donor.parent/'control_source.py'), 'Right search deployment changed')
    for filename,key in (('direction_training_source.py','direction_training_source_sha256'),
                         ('direction_training_protocol.py','direction_training_protocol_sha256'),('teleop_contract.py','teleop_contract_sha256')):
        require(digest(search.parent/filename)==sm[key], 'Right search helper changed')
    seed=Path(evidence['seed_checkpoint']);seed_meta=read(seed.parent/'training.json')
    require(seed==(search.parent.parent/Path(args['seed_checkpoint']).parent.name/Path(args['seed_checkpoint']).name).resolve() and
        digest(seed)==evidence['seed_checkpoint_sha256']==seed_meta['checkpoints'][seed.name] and
        digest(seed.parent/'training.json')==evidence['seed_training_sha256'], 'Right search seed changed')
    require(seed==Path(old['parent_checkpoint']) and digest(seed)==old['parent_checkpoint_sha256'] and
        digest(seed.parent/'training.json')==old['parent_training_sha256'], 'Right search seed differs from donor parent')
    seed_blob=torch.load(seed,map_location='cpu',weights_only=False)
    search_blob=torch.load(search,map_location='cpu',weights_only=False)
    for base_blob in (seed_blob,search_blob):
        a,b=base_blob['standing_actor_state_dict'],original['standing_actor_state_dict']
        require(torch.equal(base_blob['parameters'],original['parameters']) and a.keys()==b.keys() and
            all(torch.equal(a[k],b[k]) for k in a), 'Right search gait or standing differs from donor')
    require(tuple(seed_blob['turn_parameters'].shape)==(20,), 'Right search seed schema changed')
    records=evidence['records']
    require(len(records)>=2 and len({r['table'] for r in records})==len(records), 'Right search repetition missing')
    for item in records:
        filename=item['table']
        require(Path(filename).name==filename and filename in sm['generation_candidate_tables'], 'Right search generation not bound')
        path=search.parent/filename
        require(digest(path)==item['sha256']==sm['generation_candidate_tables'][filename], 'Right search table changed')
        index=item['candidate'];row=read(path)[index];values=torch.tensor(row['parameters'],dtype=p.dtype)
        require(row['candidate']==index and row['eligible'] and row['world_gates']['completed']==[True]*4 and
            len(row['failure_bits'])==4 and not any(row['failure_bits']) and len(row['peak_support_bw'])==4 and
            all(math.isfinite(x) and 0<=x<=2.9 for x in row['peak_support_bw']) and tuple(values.shape)==(20,) and
            torch.equal(values[indices],p[indices].cpu()) and torch.equal(values[fixed],seed_blob['turn_parameters'][fixed]),
            'Right repeated candidate qualification changed')
    donor_result={'residual_variant':old['residual_variant'],'residual_dispatch':'deterministic_cruise_plus_active_left_hip_roll_v1',
        **{key:old[key] for key in ('residual_source_sha256','active_residual_source_sha256')}}
    check_residual_source(donor.parent,donor_result,old,donor)


def check_residual_source(folder,result,training,checkpoint):
    """Independently bind the hybrid controller and its unchanged frozen parent."""
    import torch
    if training.get('residual_variant')=='right_controller_calibrated_dual_residual_v1':
        check_right_controller_calibration(folder,result,training,checkpoint)
        return
    require(not any(k.startswith('right_controller_') for k in training), 'Unexpected right controller calibration metadata')
    if training.get('residual_variant')=='yaw_hold_calibrated_dual_residual_v1':
        check_yaw_hold_calibration(folder,result,training,checkpoint)
        return
    require(not any(k.startswith(('calibration_donor_','yaw_hold_calibration')) for k in training), 'Unexpected yaw/hold calibration metadata')
    adapted=training.get('residual_variant')=='adapted_cruise_frozen_active_v1'
    rebased=training.get('residual_variant')=='right_balance_rebased_dual_residual_v1'
    require(rebased or not any(k.startswith('frozen_dual_') for k in training), 'Unexpected right rebase metadata')
    active=adapted or rebased or training.get('residual_variant')=='cruise_plus_active_left_v1'
    candidate=torch.load(checkpoint,map_location='cpu',weights_only=False)
    require(candidate.get('residual_variant')==training.get('residual_variant'), 'Active residual variant mismatch')
    if not active:
        require(training.get('residual_variant') is None and 'active_residual_actor_state_dict' not in candidate and
            not any(k.startswith(('active_residual_','frozen_cruise_','frozen_active_','cruise_initialization_','frozen_dual_')) for k in training) and
            not any(k.startswith('active_residual_') for k in result) and result.get('residual_variant') is None,
            'Unexpected active residual weights or metadata')
    else:
        require(training.get('policy_family')=='periodic_feedback_residual_ppo', 'Active residual family mismatch')
    if training.get('policy_family')!='periodic_feedback_residual_ppo':
        candidate=torch.load(checkpoint,map_location='cpu',weights_only=False)
        require('residual_actor_state_dict' not in candidate, 'Residual weights mislabeled as another policy family')
        require('residual_source_sha256' not in training and 'residual_source_sha256' not in result,
            'Residual evidence mislabeled as another policy family')
        return
    checkpoint=Path(checkpoint);folder=Path(folder)
    expected=training['residual_source_sha256']
    require(expected==digest(checkpoint.parent/'residual_source.py')==digest(folder/'residual_source.py')==result.get('residual_source_sha256'), 'Residual source mismatch')
    require(result.get('residual_dispatch')==('deterministic_cruise_plus_active_left_hip_roll_v1' if active else 'deterministic_cruise_hip_roll_v1'), 'Residual inference dispatch mismatch')
    require(training['residual_bound_rad']==.02 and training['residual_feature_dim']==69, 'Residual bounds/schema changed')
    require(training['residual_action_joints']==['left_hip_roll_joint','right_hip_roll_joint'], 'Residual joints changed')
    require(training['residual_gate']=='left_cruise_blend * clamp(abs(forward)/0.2,0,1)', 'Residual gate changed')
    parent=Path(training['parent_checkpoint'])
    require(digest(parent)==training['parent_checkpoint_sha256'], 'Frozen parent changed')
    require(digest(parent.parent/'training.json')==training['parent_training_sha256'], 'Frozen parent metadata changed')
    parent_meta=read(parent.parent/'training.json')
    require(parent_meta['checkpoints'][parent.name]==digest(parent), 'Frozen parent provenance mismatch')
    require(parent_meta.get('policy_family')=='periodic_feedback_cem' and not str(parent_meta.get('status','')).startswith('invalidated_'), 'Invalid frozen parent')
    require(parent_meta['turn_source_sha256']==training['turn_source_sha256'], 'Frozen turn implementation changed')
    require(digest(parent.parent/'control_source.py')==digest(checkpoint.parent/'control_source.py'), 'Frozen gait implementation changed')
    original=torch.load(parent,map_location='cpu',weights_only=False)
    candidate=torch.load(checkpoint,map_location='cpu',weights_only=False)
    require('residual_actor_state_dict' in candidate, 'Missing residual actor')
    if active:
        require(result.get('residual_variant')==training['residual_variant'], 'Active evidence variant mismatch')
        require('active_residual_actor_state_dict' in candidate, 'Missing active residual actor')
        require(training.get('active_residual_bound_rad')==.02 and training.get('active_residual_feature_dim')==69 and
            training.get('active_residual_action_joints')==['left_hip_roll_joint','right_hip_roll_joint'] and
            training.get('active_residual_gate')=='(1-left_cruise_blend) * left_feedback_blend * clamp(abs(forward)/0.2,0,1)', 'Active residual contract mismatch')
        require(training.get('active_residual_source_sha256')==digest(checkpoint.parent/'active_residual_source.py')==
            digest(folder/'active_residual_source.py')==result.get('active_residual_source_sha256'), 'Active residual source mismatch')
        cruise_comparison=candidate
        lineage=training
        if rebased:
            require(not any(k.startswith(('frozen_cruise_','frozen_active_','cruise_initialization_')) for k in training), 'Right rebase has misleading frozen lineage')
            require(training.get('right_rebase',{}).get('source_sha256')==digest(checkpoint.parent/'right_rebase_source.py'), 'Right composition source changed')
            donor=Path(training['frozen_dual_checkpoint']);lineage=read(donor.parent/'training.json')
            require(digest(donor)==training['frozen_dual_checkpoint_sha256'] and digest(donor.parent/'training.json')==training['frozen_dual_training_sha256'], 'Frozen dual input changed')
            require(lineage.get('policy_family')=='periodic_feedback_residual_ppo' and lineage.get('residual_variant')=='cruise_plus_active_left_v1' and
                not str(lineage.get('status','')).startswith('invalidated_') and lineage['checkpoints'][donor.name]==digest(donor), 'Invalid frozen dual donor')
            duration=training.get('arguments',{}).get('turn_duration')
            require(isinstance(duration,(int,float)) and 14.<=duration<=90. and duration==lineage.get('arguments',{}).get('turn_duration') and
                duration==parent_meta.get('arguments',{}).get('turn_duration'), 'Frozen turn duration changed')
            cruise_comparison=torch.load(donor,map_location='cpu',weights_only=False)
            require(cruise_comparison.get('residual_variant')=='cruise_plus_active_left_v1', 'Frozen dual donor mislabeled')
            for key in ('residual_source_sha256','active_residual_source_sha256','residual_bound_rad','residual_feature_dim','residual_action_joints','residual_gate',
                        'active_residual_bound_rad','active_residual_feature_dim','active_residual_action_joints','active_residual_gate'):
                require(lineage.get(key)==training[key], 'Frozen dual contract changed')
            for source in ('residual_source','active_residual_source','turn_source'):
                require(digest(donor.parent/(source+'.py'))==lineage[source+'_sha256'], 'Frozen dual source changed')
            fixed=[i for i in range(20) if i not in (12,13)]
            require(tuple(candidate['turn_parameters'].shape)==(20,) and
                torch.equal(candidate['turn_parameters'][fixed],cruise_comparison['turn_parameters'][fixed]), 'Only right sway parameters12/13 may change')
            p=candidate['turn_parameters']
            require(bool(torch.isfinite(p).all()) and -.06<=float(p[12])<=.03 and -.25<=float(p[13])<=.25, 'Right sway outside bounded calibration')
            require(torch.equal(candidate['parameters'],cruise_comparison['parameters']), 'Frozen dual gait changed')
            for key in ('standing_actor_state_dict','residual_actor_state_dict','active_residual_actor_state_dict'):
                left,right=cruise_comparison[key],candidate[key]
                require(left.keys()==right.keys() and all(torch.equal(left[k],right[k]) for k in left), 'Frozen dual weights changed')
            import ast
            def deployment(path):
                names={'CommandMemory','commanded_action','configure_profile','command_profile','evaluate_turn_gate'}
                tree=ast.parse(path.read_text())
                nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
                constants=[n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in {'TURN_START','TURN_END','HOLD_START','TURN_RATE'} for t in n.targets)]
                require({n.name for n in nodes}==names and len(constants)==4, 'Incomplete saved turn implementation')
                class FixedArity(ast.NodeTransformer):
                    def visit_If(self,node):
                        test=node.test
                        first=test.values[0] if isinstance(test,ast.BoolOp) and isinstance(test.op,ast.And) else test
                        if (isinstance(first,ast.Compare) and ast.dump(first.left)==ast.dump(ast.parse('p.shape[1]',mode='eval').body) and
                            len(first.ops)==1 and isinstance(first.ops[0],ast.GtE) and isinstance(first.comparators[0],ast.Constant) and
                            isinstance(first.comparators[0].value,int) and first.comparators[0].value>20):
                            return [self.visit(n) for n in node.orelse]
                        return self.generic_visit(node)
                return ast.dump(FixedArity().visit(ast.Module(body=constants+nodes,type_ignores=[])),include_attributes=False)
            require(deployment(checkpoint.parent/'turn_source.py')==deployment(donor.parent/'turn_source.py'), 'Right rebase deployment changed')
        if adapted:
            require(not any(k.startswith('frozen_cruise_') for k in training), 'Adapted cruise falsely marked frozen')
            donor=Path(training['frozen_active_checkpoint']);lineage=read(donor.parent/'training.json')
            require(digest(donor)==training['frozen_active_checkpoint_sha256'] and digest(donor.parent/'training.json')==training['frozen_active_training_sha256'], 'Frozen active input changed')
            require(lineage.get('policy_family')=='periodic_feedback_residual_ppo' and lineage.get('residual_variant')=='cruise_plus_active_left_v1' and
                not str(lineage.get('status','')).startswith('invalidated_') and lineage['checkpoints'][donor.name]==digest(donor), 'Invalid frozen active donor')
            cruise_comparison=torch.load(donor,map_location='cpu',weights_only=False)
            require(cruise_comparison.get('residual_variant')=='cruise_plus_active_left_v1', 'Frozen active donor mislabeled')
            for key in ('parent_checkpoint_sha256','parent_training_sha256','residual_source_sha256','turn_source_sha256','active_residual_source_sha256',
                        'active_residual_bound_rad','active_residual_feature_dim','active_residual_action_joints','active_residual_gate',
                        'residual_bound_rad','residual_feature_dim','residual_action_joints','residual_gate'):
                require(lineage.get(key)==training[key], 'Frozen active contract changed')
            for source in ('residual_source','active_residual_source','turn_source'):
                require(digest(donor.parent/(source+'.py'))==training[source+'_sha256'], 'Frozen active source changed')
            for key in ('parameters','turn_parameters'):
                require(torch.equal(cruise_comparison[key],candidate[key]), 'Frozen active controller changed')
            for key in ('standing_actor_state_dict','active_residual_actor_state_dict'):
                left,right=cruise_comparison[key],candidate[key]
                require(left.keys()==right.keys() and all(torch.equal(left[k],right[k]) for k in left), 'Frozen active weights changed')
            require(Path(training['cruise_initialization_checkpoint']).resolve()==Path(lineage['frozen_cruise_checkpoint']).resolve() and
                training['cruise_initialization_checkpoint_sha256']==lineage['frozen_cruise_checkpoint_sha256'] and
                training['cruise_initialization_training_sha256']==lineage['frozen_cruise_training_sha256'], 'Cruise initialization changed')
        elif not rebased:
            require(not any(k.startswith(('frozen_active_','cruise_initialization_','frozen_dual_')) for k in training), 'Unexpected adapted cruise metadata')
        cruise=Path(lineage['frozen_cruise_checkpoint']);cruise_meta=read(cruise.parent/'training.json')
        require(digest(cruise)==lineage['frozen_cruise_checkpoint_sha256'] and digest(cruise.parent/'training.json')==lineage['frozen_cruise_training_sha256'], 'Frozen cruise input changed')
        require(cruise_meta.get('policy_family')=='periodic_feedback_residual_ppo' and cruise_meta.get('residual_variant') is None and
            not str(cruise_meta.get('status','')).startswith('invalidated_') and cruise_meta['checkpoints'][cruise.name]==digest(cruise), 'Invalid frozen cruise input')
        for key in ('parent_checkpoint_sha256','parent_training_sha256','residual_source_sha256','turn_source_sha256'):
            require(cruise_meta[key]==lineage[key], 'Frozen cruise lineage changed')
        require(digest(cruise.parent/'residual_source.py')==expected, 'Frozen cruise source changed')
        cruise_blob=torch.load(cruise,map_location='cpu',weights_only=False)
        require(cruise_blob.get('residual_variant') is None and 'active_residual_actor_state_dict' not in cruise_blob, 'Invalid cruise-only weights')
        for key in ('parameters','turn_parameters'):
            require(torch.equal(cruise_blob[key],cruise_comparison[key]), 'Frozen cruise controller changed')
        for key in ('standing_actor_state_dict','residual_actor_state_dict'):
            left,right=cruise_blob[key],cruise_comparison[key]
            require(left.keys()==right.keys() and all(torch.equal(left[k],right[k]) for k in left), 'Frozen cruise weights changed')
    if training.get('composition'):
        composition=training['composition'];learned=Path(composition['learned_residual_checkpoint'])
        require(digest(learned)==composition['learned_residual_checkpoint_sha256'], 'Learned residual input changed')
        learned_meta=read(learned.parent/'training.json')
        require(learned_meta.get('policy_family')=='periodic_feedback_residual_ppo' and
            learned_meta['checkpoints'][learned.name]==digest(learned), 'Learned residual provenance mismatch')
        require(learned_meta['residual_source_sha256']==expected, 'Rebased residual implementation changed')
        learned_blob=torch.load(learned,map_location='cpu',weights_only=False)
        before=learned_blob['residual_actor_state_dict'];after=candidate['residual_actor_state_dict']
        require(before.keys()==after.keys() and all(torch.equal(before[k],after[k]) for k in before), 'Rebased residual weights changed')
    require(tuple(original['turn_parameters'].shape)==(20,), 'Frozen parent requires20turnparameters')
    for key in ('parameters','turn_parameters'):
        require(torch.equal(original[key],candidate[key]), f'Frozen {key} changed')
    left=original['standing_actor_state_dict'];right=candidate['standing_actor_state_dict']
    require(left.keys()==right.keys() and all(torch.equal(left[k],right[k]) for k in left), 'Frozen standing policy changed')


def check_walking(folder, ledger, checkpoint, distance=5., turn=False, teleop=False, direction=None, triangle=False):
    """Recheck the full distance run; only standing has a 30 s contract."""
    import numpy as np
    folder=Path(folder).resolve(); checkpoint=Path(checkpoint).resolve()
    folder.relative_to((ALG/'outputs').resolve())
    result=read(folder/'dynamics.json'); metrics=result['metrics']; cfg=result['config']
    identity=result['identity']; contract=ledger['assetContract']
    require(distance in (5.,10.), 'Unsupported walking gate')
    require(sum((bool(turn),bool(teleop),bool(direction),bool(triangle)))<=1,'Choose one walking protocol')
    if direction:
        from algorithms.urdf_learn_wasd_walk import landau_direction_contract as dc
        require(direction in dc.DIRECTIONS and distance==10.,'Wrong directional gate')
    expected='triangle_path_follow_no_reset' if triangle else 'gate_10m_four_directions_no_reset' if direction else 'teleop_60s_forward_turn' if teleop else 'yaw_turn_90deg_hold' if turn else f'gate_{distance:g}m_no_reset'
    require(result['milestone']==expected and not result['failures'], 'Walking dynamics failed')
    require(bool(cfg.get('triangle',False))==triangle, 'Triangle protocol mismatch')
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
    require(math.isfinite(duration) and (24. if turn else 30.)<=duration<=(600. if triangle else 240. if direction else 120.), 'Walking diagnostic duration outside supported bounds')
    physics_steps=round(duration/.002); control_steps=round(duration/.02)
    require(abs(control_steps*.02-duration)<1e-8, 'Walking duration not aligned to control steps')
    require(cfg['assistance']==0 and cfg['dt']==.002 and 0<cfg['forward']<=.4, 'Wrong walking protocol')
    if not triangle and not turn and not teleop:require(cfg.get('target_distance_m',5.)==distance, 'Walking distance protocol mismatch')
    require(cfg['gain_scale']==1. and cfg['contact_timeconst']==.004, 'Walking physics changed')
    require(Path(cfg['checkpoint']).resolve()==checkpoint and cfg['checkpoint_sha256']==digest(checkpoint), 'Walking checkpoint mismatch')
    training=read(checkpoint.parent/'training.json')
    require(not str(training.get('status','')).startswith('invalidated_'), 'Training evidence was invalidated')
    if training.get('policy_family') in ('periodic_feedback_cem','periodic_feedback_residual_ppo'):
        check_gait_source(folder,result,training)
    check_residual_source(folder,result,training,checkpoint)
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
        require(training.get('policy_family') in ('periodic_feedback_cem','periodic_feedback_residual_ppo'), 'Unsupported turn policy family')
        require(training.get('command_extension')=='yaw_v1', 'Checkpoint lacks trained yaw control')
        require(training['turn_source_sha256']==digest(checkpoint.parent/'turn_source.py')==digest(folder/'turn_source.py'), 'Turn source mismatch')
        require(result['turn_validator_source_sha256']==digest(folder/'turn_validator_source.py'), 'Turn validator source mismatch')
    elif not triangle and not teleop and not direction:
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
    elif not triangle and not teleop and not direction:
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
        navigation=cfg.get('direction_navigation','legacy')
        require(navigation in ('legacy','signed_v2','filtered_signed_v3','reference_v4','anti_windup_v5','hysteretic_v6'), 'Unknown directional navigation')
        navigator=None
        if navigation!='legacy':
            from algorithms.urdf_learn_wasd_walk import landau_signed_navigation as signed
            require(result.get('signed_navigation_source_sha256')==digest(folder/'signed_navigation_source.py')==digest(signed.__file__),
                    'Signed navigation source mismatch')
            navigator=signed.make_navigator(navigation,direction)
        else:require('signed_navigation_source_sha256' not in result, 'Legacy direction mislabeled with signed navigation')
        commands=np.array([(navigator.command(xy[:2]-positions[0,:2],heading,float(age),cfg['forward']) if navigator else
                           dc.command(direction,xy[:2]-positions[0,:2],heading,float(age),cfg['forward']))
                           for xy,heading,age in zip(positions[:-1],headings[:-1],pt)])
        require(np.allclose(obs[:,63:66],commands,atol=1e-6,rtol=0), 'Direction did not receive declared joystick commands')
        memory=read(folder/'controller_memory.json')
        require(memory['simulation_reset'] is False and memory.get('memory_version')==2, 'Wrong direction memory semantics')
        require(abs(memory['integrated_reference_rad']-float(obs[:-1,65].sum())*.02)<1e-4, 'Direction heading reference differs from commands')
        if navigation in ('reference_v4','anti_windup_v5','hysteretic_v6'):
            state=navigator.record();reported=result.get('direction_navigation_state',{})
            require(state.keys()==reported.keys(), 'Missing navigation reference state')
            for key,value in state.items():
                if isinstance(value,float):require(abs(value-reported[key])<1e-6, 'Navigation state differs: '+key)
                else:require(value==reported[key], 'Navigation state differs: '+key)
            require(abs(state['shadow_reference_rad']-memory['integrated_reference_rad'])<1e-7, 'Navigator shadow reference differs from controller')
        else:require('direction_navigation_state' not in result, 'Unexpected reference navigation state')
    if triangle:check_triangle_trajectory(folder,result,training,q,t,control_steps)
    media=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=nb_frames,duration','-of','json',str(folder/'proof.mp4')]))['streams'][0]
    require(int(media['nb_frames'])==control_steps+1 and float(media['duration'])>=duration, 'Walking video incomplete')
    return result,metrics


def check_triangle_trajectory(folder, result, training, q, t, control_steps):
    """Independently reconstruct geometry, then replay joystick decisions only."""
    import numpy as np
    import mujoco
    from algorithms.urdf_learn_wasd_walk import landau_triangle_navigation as tc, landau_signed_navigation as signed, landau_direction_contract as dc
    cfg,metrics=result['config'],result['metrics']
    require(cfg['forward']==.2 and cfg.get('triangle_horizon_s')==600., 'Triangle prospective protocol changed')
    require(result.get('triangle_contract')==tc.contract(), 'Triangle geometry/acceptance changed')
    require(training.get('command_extension')=='yaw_v1' and training.get('memory_version')==2, 'Triangle requires joystick memory')
    require(np.allclose(q[0],training['nominal_q'],atol=1e-7,rtol=0), 'Triangle start changed')
    for name,source in [('triangle_navigation',tc.__file__),('signed_navigation',signed.__file__),('direction_protocol',dc.__file__)]:
        require(result.get(name+'_source_sha256')==digest(folder/(name+'_source.py'))==digest(source), 'Triangle source mismatch: '+name)
    require(result['policy_trace_sha256']==digest(folder/'policy_trace.npz'), 'Triangle command trace changed')
    with np.load(folder/'policy_trace.npz') as trace:obs,pt=trace['observation'],trace['time']
    require(obs.shape==(control_steps,70) and np.isfinite(obs).all() and np.allclose(pt,t[:-1],atol=1e-6), 'Incomplete triangle command trace')
    model=mujoco.MjModel.from_xml_path(str(folder/'model.xml'));data=mujoco.MjData(model)
    positions=[];headings=[]
    for frame in q:
        data.qpos[:]=frame;mujoco.mj_forward(model,data)
        positions.append(data.xpos[model.body('root_x').id,:2].copy())
        rotation=data.xmat[model.body('base_link').id].reshape(3,3)
        headings.append(math.atan2(rotation[1,0],rotation[0,0]))
    xy=np.array(positions)-positions[0]
    vertices=np.array(tc.VERTICES);distances=[];targets=[]
    for edge in range(3):
        a,b=vertices[edge:edge+2];delta=b-a
        projection=np.clip((xy-a)@delta/(delta@delta),0.,1.)
        distances.append(np.linalg.norm(xy-a-projection[:,None]*delta,axis=1))
        targets.extend([((a+b)/2.,.75,f'edge_{edge+1}_midpoint'),
                        (b,3.5 if edge<2 else .75,f'corner_{edge+1}' if edge<2 else 'closure')])
    max_corridor=float(np.min(distances,axis=0).max())
    require(max_corridor<=2., 'Reconstructed triangle exceeded corridor')
    visits=[];index=0
    for target,radius,label in targets:
        choices=np.flatnonzero(np.linalg.norm(xy[index:]-target,axis=1)<=radius)
        require(len(choices)>0,'Triangle missing ordered target: '+label)
        index+=int(choices[0]);visits.append(dict(target=label,time_s=float(t[index]),distance_m=float(np.linalg.norm(xy[index]-target))))
        if label!='closure':index+=1
    require(index==len(t)-1, 'Triangle must end at first complete circuit')
    require(metrics.get('triangle_complete') is True and len(metrics.get('triangle_visits',[]))==6, 'Triangle completion evidence missing')
    for actual,reported in zip(visits,metrics['triangle_visits']):
        require(actual['target']==reported['target'] and abs(actual['time_s']-reported['time_s'])<1e-6 and abs(actual['distance_m']-reported['distance_m'])<1e-5, 'Triangle waypoint evidence differs')
    for key,actual in [('triangle_max_corridor_m',max_corridor),('triangle_path_length_m',float(np.linalg.norm(np.diff(xy,axis=0),axis=1).sum())),('triangle_closure_distance_m',float(np.linalg.norm(xy[-1])))]:
        require(abs(metrics[key]-actual)<1e-4, 'Triangle geometry differs: '+key)
    require(not tc.evaluate_gate(metrics,cfg['seconds']), 'Triangle path/quality gate failed')
    nav=tc.Navigator()
    commands=np.array([nav.command(point,heading,float(age),.2) for point,heading,age in zip(xy[:-1],headings[:-1],pt)])
    require(np.allclose(obs[:,63:66],commands,atol=1e-6,rtol=0), 'Triangle commands differ from pose-driven joystick navigation')
    state=nav.record();reported=result.get('triangle_navigation_state',{})
    require(state.keys()==reported.keys(), 'Missing triangle navigator memory')
    for key,value in state.items():
        if isinstance(value,float):require(abs(value-reported[key])<1e-6, 'Triangle navigator differs: '+key)
        else:require(value==reported[key], 'Triangle navigator differs: '+key)
    memory=read(folder/'controller_memory.json')
    require(memory['simulation_reset'] is False and memory.get('memory_version')==2, 'Triangle memory reset or wrong version')
    require(memory['turn_source_sha256']==training['turn_source_sha256']==digest(folder/'turn_source.py'), 'Triangle controller source differs')
    require(abs(memory['integrated_reference_rad']-float(obs[:-1,65].sum())*.02)<1e-4, 'Triangle reference does not follow joystick commands')
    require(abs(memory['integrated_reference_rad']-state['shadow_reference_rad'])<1e-7, 'Triangle shadow reference mismatch')


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
        if cfg.get('triangle'):
            args+=['--triangle']
            args[args.index('--seconds')+1]=str(cfg['triangle_horizon_s'])
        if cfg.get('turn'):args+=['--turn']
        if cfg.get('teleop'):args+=['--teleop']
        if cfg.get('direction'):
            args+=['--direction',cfg['direction']]
            if cfg.get('direction_navigation','legacy')!='legacy':
                args+=['--direction-navigation',cfg['direction_navigation']]
    else:
        # Historical dynamics accidentally named the underlying backend module;
        # retain its exact adapter arguments and record the correct entry point.
        original=shlex.split(result['reproduce'])
        args=original[original.index('-m')+2:]
        require('--baseline' in args and '--name' in args, 'Passive reproduction arguments unavailable')
        args[args.index('--name')+1]=cfg['name']+'__replay'
        module='continuation'
    return {'kind':'module','module':'algorithms.urdf_learn_wasd_walk.'+module,'args':args,'timeout_s':3600 if cfg.get('triangle') else 1200 if cfg.get('direction') else 240}


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
    if policy_kind in ('periodic_feedback_cem','periodic_feedback_residual_ppo'):
        evidence.append(artifact(folder/'gait_source.py','trained_gait_source'))
    if policy_kind=='periodic_feedback_residual_ppo':
        evidence.append(artifact(folder/'residual_source.py','trained_residual_source'))
        if read(checkpoint.parent/'training.json').get('residual_variant'):
            evidence.append(artifact(folder/'active_residual_source.py','trained_active_residual_source'))
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
        directions[direction]['navigation']=result['config'].get('direction_navigation','legacy')
        if directions[direction]['navigation']!='legacy':
            evidence.append(artifact(path/'signed_navigation_source.py',direction+'_signed_navigation_source.py'))
        for name in ('dynamics.json','proof_metadata.json','visual_review.json','proof.mp4','trajectory.npz',
                     'policy_trace.npz','foot_trace.json','direction_protocol_source.py','controller_memory.json',
                     'turn_source.py','gait_source.py','controller_source.py','backend_source.py','model.xml'):
            evidence.append(artifact(path/name,direction+'_'+name))
        if read(checkpoint.parent/'training.json').get('policy_family')=='periodic_feedback_residual_ppo':
            evidence.append(artifact(path/'residual_source.py',direction+'_residual_source.py'))
            if read(checkpoint.parent/'training.json').get('residual_variant'):
                evidence.append(artifact(path/'active_residual_source.py',direction+'_active_residual_source.py'))
    require(set(directions)==set(DIRECTIONS), 'All world directions required')
    require(len({v['navigation'] for v in directions.values()})==1, 'All directions must use the same navigator version')
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
    if read(checkpoint.parent/'training.json').get('residual_variant')=='right_balance_rebased_dual_residual_v1':
        evidence.append(artifact(checkpoint.parent/'right_rebase_source.py','checkpoint_right_composition_source'))
    if read(checkpoint.parent/'training.json').get('residual_variant')=='right_controller_calibrated_dual_residual_v1':
        evidence.append(artifact(checkpoint.parent/'right_controller_rebase_source.py','checkpoint_right_controller_composition_source'))
    if read(checkpoint.parent/'training.json').get('residual_variant')=='yaw_hold_calibrated_dual_residual_v1':
        evidence.append(artifact(checkpoint.parent/'yaw_hold_rebase_source.py','checkpoint_yaw_hold_composition_source'))
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


def certify_triangle(folder, checkpoint, triangle_folder):
    """M8 reuses only revalidated M7 components of the identical checkpoint."""
    ledger=read(LEDGER);checkpoint=Path(checkpoint).resolve();folder=Path(folder).resolve()
    triangle_folder=Path(triangle_folder).resolve();folder.relative_to((ALG/'outputs').resolve())
    require(ledger['lineage']==LINEAGE and all(m['status']=='passed' for m in ledger['milestones'][:7]), 'Prior triangle milestone unresolved')
    target=ledger['milestones'][7]
    require(target['id']=='triangle_path_follow_no_reset' and target['status']=='in_progress', 'Only unresolved M8 can be certified')
    require(not folder.exists(), 'Triangle certificate folder exists')
    previous=ledger['milestones'][6]
    certificates=[a for a in previous['evidence'] if a['kind']=='validation']
    require(len(certificates)==1, 'M7 certificate identity missing')
    prior_artifact=certificates[0];prior_path=(ROOT/prior_artifact['path']).resolve()
    prior_path.relative_to((ALG/'outputs').resolve())
    require(digest(prior_path)==prior_artifact['sha256'], 'M7 certificate changed')
    prior=read(prior_path)
    require(prior['status']=='passed' and prior['milestone']==previous['id'] and prior['lineage']==LINEAGE and prior['backend']==ledger['backend'], 'Invalid M7 certificate')
    require(prior['checkpoint']==previous['checkpoint'] and prior['checkpoint']['sha256']==digest(checkpoint) and (ROOT/prior['checkpoint']['path']).resolve()==checkpoint, 'M7 used another checkpoint')
    artifacts={}
    for item in prior['evidence']+prior['cumulative_components']:
        path=(ROOT/item['path']).resolve();path.relative_to((ALG/'outputs').resolve())
        require(digest(path)==item['sha256'], 'M7 artifact changed: '+item['kind'])
        require(item['kind'] not in artifacts, 'Duplicate M7 evidence kind')
        artifacts[item['kind']]=path
    candidate_time=checkpoint.stat().st_mtime
    navigation=set()
    for direction in ('forward','left','right','backward'):
        path=artifacts[direction+'_dynamics.json'].parent
        result,_=check_walking(path,ledger,checkpoint,10.,direction=direction)
        require(datetime.fromisoformat(result['created_at']).timestamp()>=candidate_time, 'M7 direction predates checkpoint')
        navigation.add(result['config'].get('direction_navigation','legacy'))
    require(len(navigation)==1, 'M7 navigator versions differ')
    for index in range(6):
        path=artifacts[f'cumulative_m{index+1}_dynamics.json'].parent
        if index<2:result,_=check_standing(path,ledger,checkpoint if index else None)
        else:result,_=check_walking(path,ledger,checkpoint,10. if index==3 else 5.,turn=index==4,teleop=index==5)
        require(datetime.fromisoformat(result['created_at']).timestamp()>=candidate_time, 'Cumulative replay predates checkpoint')
    result,metrics=check_walking(triangle_folder,ledger,checkpoint,triangle=True)
    require(datetime.fromisoformat(result['created_at']).timestamp()>=candidate_time, 'Triangle replay predates checkpoint')
    evidence=[artifact(prior_path,'revalidated_m7_certificate')]
    for name in ('dynamics.json','proof_metadata.json','visual_review.json','proof.mp4','trajectory.npz',
                 'policy_trace.npz','foot_trace.json','triangle_navigation_source.py','signed_navigation_source.py','direction_protocol_source.py',
                 'controller_memory.json','turn_source.py','gait_source.py','residual_source.py','active_residual_source.py',
                 'controller_source.py','backend_source.py','renderer_source.py','model.xml'):
        evidence.append(artifact(triangle_folder/name,name))
    evidence.append(artifact(checkpoint.parent/'training.json','checkpoint_training_metadata'))
    final=dict(status='passed',lineage=LINEAGE,backend=ledger['backend'],milestone=target['id'],
               assembled_at=datetime.now(timezone.utc).isoformat(),checkpoint=prior['checkpoint'],metrics=metrics,
               evidence=evidence,prior_components_revalidated=True,reproduction_job=reproduction_job(result,checkpoint),
               scope='One counterclockwise18m triangle; joystick-only pursuit, ordered side/corner visits,2m corridor,.75m closure; no reset',
               finalizer_source_sha256=digest(__file__))
    folder.mkdir();(folder/'milestone_finalizer_source.py').write_text(Path(__file__).read_text())
    write(folder/'milestone_validation.json',final)
    target.update(status='passed',passedAt=final['assembled_at'],checkpoint=prior['checkpoint'],metrics=metrics,
                  evidence=[artifact(folder/'milestone_validation.json','validation'),*evidence])
    ledger['milestones'][8]['status']='in_progress';ledger['implementationStatus']='milestone_8_passed_milestone_9_in_progress'
    ledger['sessionStatus'].update(certifiedMilestones=8,nextMilestone=ledger['milestones'][8]['id'],nextWork='Closed square path following')
    write(LEDGER,ledger);sync_gui(ledger)
    return final


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=('init', 'certify', 'certify-directions', 'certify-triangle'))
    p.add_argument('--directory', required=True, type=Path)
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--passive-directory', type=Path)
    p.add_argument('--standing-directory', type=Path)
    p.add_argument('--five-metre-directory', type=Path)
    p.add_argument('--ten-metre-directory', type=Path)
    p.add_argument('--turn-directory',type=Path)
    p.add_argument('--teleop-directory',type=Path)
    p.add_argument('--triangle-directory',type=Path)
    p.add_argument('--direction-directory',type=Path,action='append',default=[])
    args = p.parse_args()
    if args.mode=='certify-triangle':
        if args.checkpoint is None or args.triangle_directory is None:p.error('Triangle certification requires checkpoint and triangle directory')
        result=certify_triangle(args.directory,args.checkpoint,args.triangle_directory)
    elif args.mode=='certify-directions':
        if args.checkpoint is None:p.error('Direction certification requires --checkpoint')
        result=certify_directions(args.directory,args.checkpoint,args.direction_directory,
            [args.passive_directory,args.standing_directory,args.five_metre_directory,args.ten_metre_directory,args.turn_directory,args.teleop_directory])
    else:
        result = initialize(args.directory) if args.mode == 'init' else certify(args.directory, args.checkpoint, args.passive_directory, args.standing_directory, args.five_metre_directory,args.ten_metre_directory,args.turn_directory)
    print(json.dumps({'lineage': result['lineage'], 'status': result.get('status', result.get('implementationStatus'))}))


if __name__ == '__main__':
    main()
