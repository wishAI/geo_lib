"""Finalize only separate MuJoCo standing evidence after explicit visual review."""
from pathlib import Path
import json
import numpy as np
import imageio.v2 as imageio
from algorithms.urdf_learn_wasd_walk.mujoco_backend import digest, write_json
from algorithms.urdf_learn_wasd_walk.passive_stand import evaluate_gate, evaluate_free_root_support
from algorithms.urdf_learn_wasd_walk.forward_walk_contract import evaluate_forward_gate


def finalize(runs, review, output, *, previous=None):
    if len({Path(p).resolve() for p in runs}) != len(runs):
        raise ValueError('Independent runs must have distinct paths')
    if len(runs) < 2:
        raise ValueError('Two independent complete dynamics runs required')
    evidence = [json.loads((Path(p)/'dynamics.json').read_text()) for p in runs]
    first = evidence[0]
    if first['milestone'] not in {'stand_zero_signal_30s_no_reset','stand_30s_no_reset','gate_5m_no_reset'}:
        raise ValueError('Unsupported gate')
    physics_keys = ('backend','mujoco_version','urdf_sha256','mesh_tree_sha256','model_xml_sha256','initial_control_sha256',
                    'warp_version','mujoco_warp_version','warp_reset_semantics_sha256','warp_runtime_source_sha256','warp_io_source_sha256')
    for directory, result in zip(runs,evidence):
        path = Path(directory)
        metrics=dict(result['metrics'])
        if result['identity'].get('backend')=='mujoco_warp_cuda':
            if digest(path/'warp_runtime_source.py')!=result['identity'].get('warp_runtime_source_sha256'):
                raise ValueError('GPU runtime snapshot changed or is not bound to evidence')
            if not result['identity'].get('warp_io_source_sha256'):
                raise ValueError('GPU force-transfer implementation is not bound to evidence')
        if result['milestone']=='gate_5m_no_reset':
            failures=evaluate_forward_gate(metrics)
        else:
            if result['milestone']=='stand_30s_no_reset': metrics['max_abs_action']=0.
            _,failures=evaluate_gate(metrics)
            failures.extend(evaluate_free_root_support(metrics))
        if failures:
            raise ValueError(f'Recomputed gate failed: {failures}')
        if metrics['max_joint_speed_rad_s']>4.000001 or metrics['nonfoot_contacts']:
            raise ValueError('Velocity or nonfoot-contact contract failed')
        if digest(path/'trajectory.npz') != result['trajectory_sha256']:
            raise ValueError('Dynamics trajectory changed')
        trajectory=np.load(path/'trajectory.npz')
        if trajectory['time'][0]!=0 or trajectory['time'][-1]<30-1e-6 or np.max(np.diff(trajectory['time']))>.020001:
            raise ValueError('Incomplete dynamics trajectory')
        if result['config'].get('checkpoint') and digest(result['config']['checkpoint']) != result['config']['checkpoint_sha256']:
            raise ValueError('Checkpoint artifact changed')
        if result['failures'] or result['status'] != 'dynamics_passed_proof_pending':
            raise ValueError('Dynamics failed')
        if result['milestone'] != first['milestone'] or any(result['identity'].get(k) != first['identity'].get(k) for k in physics_keys):
            raise ValueError('Run identity mismatch')
        if result['config'].get('checkpoint_sha256') != first['config'].get('checkpoint_sha256'):
            raise ValueError('Checkpoint mismatch')
        if result['config']['assistance'] != 0 or result['metrics']['peak_auxiliary_wrench_norm'] != 0:
            raise ValueError('Assisted evidence is ineligible')
        if digest(path/'model.xml') != result['identity']['model_xml_sha256']:
            raise ValueError('Model artifact changed')
    proof_run = Path(runs[-1]); last = evidence[-1]
    review_data = json.loads(Path(review).read_text())
    video = proof_run/'proof.mp4'
    if review_data.get('accepted') is not True or not review_data.get('reviewer'):
        raise ValueError('Explicit visual review required')
    if digest(video) != review_data['video_sha256'] or digest(video) != last['proof']['video_sha256']:
        raise ValueError('Reviewed video changed')
    if digest(proof_run/'trajectory.npz') != last['proof']['trajectory_sha256']:
        raise ValueError('Trajectory changed')
    states = np.load(proof_run/'trajectory.npz')
    if states['time'][0] != 0 or states['time'][-1] < 30-1e-6 or np.max(np.diff(states['time'])) > .020001:
        raise ValueError('Incomplete full-duration trajectory')
    reader = imageio.get_reader(video)
    frames = reader.count_frames(); metadata = reader.get_meta_data()
    visibility=[]
    for frame in reader:
        rgb=frame.astype(float)
        character=(rgb[:,:,0]>1.1*rgb[:,:,1]) & (rgb[:,:,1]>1.1*rgb[:,:,2]) & (rgb[:,:,0]>25)
        yy,xx=np.nonzero(character)
        visibility.append(bool(len(xx)>100 and xx.min()>2 and xx.max()<frame.shape[1]-3 and yy.min()>2 and yy.max()<frame.shape[0]-3))
    reader.close()
    if len(visibility)!=frames or not all(visibility):
        raise ValueError('Character missing or cropped in one or more proof frames')
    if frames != len(states['time']) or frames/metadata['fps'] < 30:
        raise ValueError('Incomplete proof video')
    if first['milestone'] in {'stand_30s_no_reset','gate_5m_no_reset'}:
        if previous is None:
            raise ValueError('Passive predecessor proof required')
        predecessor = json.loads(Path(previous).read_text())
        expected = 'stand_30s_no_reset' if first['milestone']=='gate_5m_no_reset' else 'stand_zero_signal_30s_no_reset'
        if not predecessor['passed'] or predecessor['milestone'] != expected:
            raise ValueError('Wrong predecessor')
        if any(predecessor['identity'].get(k) != first['identity'].get(k) for k in physics_keys):
            raise ValueError('Predecessor physics mismatch')
        if first['milestone']=='gate_5m_no_reset' and predecessor['checkpoint_sha256'] != first['config'].get('checkpoint_sha256'):
            raise ValueError('Walking checkpoint must itself re-pass standing')
        if any(r['metrics']['policy_calls'] < 1500 for r in evidence):
            raise ValueError('Incomplete policy control')
    result={'passed':True,'milestone':first['milestone'],'identity':first['identity'],
            'checkpoint_sha256':first['config'].get('checkpoint_sha256'),
            'namespace':first['identity']['backend']+'_rabbit_ear_20260918','canonical_milestones_modified':False,
            'runs':[{'path':str(p),'dynamics_sha256':digest(Path(p)/'dynamics.json')} for p in runs],
            'visual_review':review_data,'character_visible_every_frame':True,'validator_source_sha256':digest(__file__),'video_frames':frames,'video_duration_s':frames/metadata['fps'],
            'previous_evidence':None if previous is None else {'path':str(previous),'sha256':digest(previous)},
            'metrics':last['metrics']}
    write_json(output,result)
    return result
