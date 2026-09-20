"""Validate repeated slow zero-assistance development runs, never a milestone."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from algorithms.urdf_learn_wasd_walk.mujoco_backend import OUTPUT, digest, write_json


def zero_failures(record, wrenches):
    """Reject auxiliary guidance even when summary gait metrics look successful."""
    errors=[];m=record['metrics'];cfg=record['config']
    numeric=[m.get(k,float('nan')) for k in ['duration_s','forward_m','max_joint_speed_rad_s',
        'max_effort_fraction','mean_contact_slip_mps','both_feet_unloaded_physics_fraction',
        'maximum_both_feet_unloaded_duration_s','mean_support_body_weight_ratio','peak_support_body_weight_ratio']]
    if not np.isfinite(numeric).all():errors.append('nonfinite or missing dynamics metrics')
    if record.get('teacher_targets_evaluated',True) or record.get('total_teacher_guidance_coefficient',1)!=0:
        errors.append('teacher guidance enabled')
    if record.get('teacher_blend_coefficient',1)!=0 or record.get('balance_assistance_coefficient',1)!=0:
        errors.append('nonzero assistance coefficient')
    if (cfg.get('motor_pose_sequence') or cfg.get('teacher_tracking_integral',0)!=0 or
            cfg.get('motor_balance_gain',0)!=0 or record.get('teacher_blend_schedule') or
            record.get('diagnostic_oracle_student') or record.get('diagnostic_observation_replay_sha256')):
        errors.append('reference assistance configured')
    if wrenches.shape!=(60000,6) or not np.isfinite(wrenches).all() or np.any(wrenches!=0):
        errors.append('external wrench trace is incomplete or nonzero')
    rows=record.get('samples',[])
    if len(rows)!=6000:errors.append('incomplete 50Hz dynamics trace')
    for row in rows:
        values=[row.get('teacher_coefficient',1),*row.get('wrench',[1]),
            *row.get('teacher_target_contribution_rad',[1]),*row.get('teacher_tracking_integral_offset_rad',[1]),
            *row.get('motor_balance_requested_offset_rad',[1])]
        if not np.isfinite(values).all() or np.any(np.asarray(values)!=0) or row.get('motor_reference_positions_rad') is not None:
            errors.append('logged teacher contribution or reference remains');break
    if m['duration_s']<120-1e-6 or m['fall'] or m['reset_count'] or m['done_count'] or m['nonfoot_contacts']:
        errors.append('duration/fall/reset/done/nonfoot failure')
    if m['forward_m']<=.2 or min(m['liftoffs'].values())<4:
        errors.append('insufficient slow-development stepping or travel')
    if m['max_joint_speed_rad_s']>4+1e-6 or m['max_effort_fraction']>1+1e-6 or m['mean_contact_slip_mps']>=.05:
        errors.append('motor limit or slip failure')
    if (m.get('ground_load_sampling_hz')!=500 or m.get('both_feet_unloaded_physics_fraction',1)>=.02 or
            m.get('maximum_both_feet_unloaded_duration_s',1)>.02 or
            not .5<=m.get('mean_support_body_weight_ratio',0)<=1.5 or m.get('peak_support_body_weight_ratio',4)>3):
        errors.append('500Hz ground support failure or missing evidence')
    return errors


def completed_placements(rows):
    """Count actual landed swings, not clearance dips during one flight."""
    events=[]
    for side in ('left','right'):
        air=[];active=None;contact_run=0
        for row in rows:
            contact=row['feet'][f'{side}_contact'];height=row['feet'][f'{side}_clearance_m']
            if active is None:
                air=air+[row] if not contact and height>.002 else []
                if len(air)==3:
                    active={'foot':side,'start_s':air[0]['time_s'],'start_forward_m':air[0]['foot_positions_m'][side][1],
                        'peak_clearance_m':max(r['feet'][f'{side}_clearance_m'] for r in air)}
            else:
                active['peak_clearance_m']=max(active['peak_clearance_m'],height)
                contact_run=contact_run+1 if contact else 0
                if contact_run>=2:
                    active.update(end_s=row['time_s'],forward_placement_m=row['foot_positions_m'][side][1]-active['start_forward_m'])
                    events.append(active);active=None;air=[];contact_run=0
    return sorted(events,key=lambda e:e['end_s'])


def validate(directories, proof, output):
    if len(directories)<3:raise ValueError('Three independent complete evaluations required')
    folders=[Path(p).resolve() for p in directories];proof=Path(proof).resolve();output=Path(output).resolve()
    for p in folders+[proof,output]:p.relative_to(OUTPUT.resolve())
    if len(set(folders))!=len(folders):raise ValueError('Duplicate evaluation directory')
    if proof not in folders:raise ValueError('Proof must belong to an evaluated trajectory')
    errors=[];runs=[];identities=[]
    for folder in folders:
        record=json.loads((folder/'result.json').read_text())
        wrench=np.load(folder/'auxiliary_wrench.npz')['wrench']
        failed=zero_failures(record,wrench)
        if hashlib.sha256(json.dumps(record['config'],sort_keys=True).encode()).hexdigest()!=record['config_sha256']:
            failed.append('configuration identity mismatch')
        ck=Path(record['config']['student']).resolve();ck.relative_to(OUTPUT.resolve())
        if digest(ck)!=record['student_checkpoint_sha256']:failed.append('checkpoint identity mismatch')
        training=json.loads((ck.parent/'training.json').read_text())
        demonstration=Path(training['config']['demonstration']).resolve();demonstration.relative_to(OUTPUT.resolve())
        teacher_identity=json.loads((demonstration.parent/'result.json').read_text())
        # Offline provenance only: evaluation did not read reference actions.
        if (training['checkpoint_sha256']!=digest(ck) or training['demonstration_sha256']!=digest(demonstration) or
                training['source_sha256']!=digest(ck.parent/'student_source.py') or
                training['review_sha256']!=digest(demonstration.parent/'visual_review.json')):
            failed.append('training provenance identity mismatch')
        if (teacher_identity['model_xml_sha256']!=record['model_xml_sha256'] or
                teacher_identity['audit']['action_joints']!=record['audit']['action_joints']):
            failed.append('training/evaluation model or action mapping differs')
        if digest(folder/'model.xml')!=record['model_xml_sha256']:failed.append('model identity mismatch')
        if digest(folder/'teacher_source.py')!=record['source_sha256'] or digest(folder/'warp_runtime_source.py')!=record['backend']['source_sha256']:
            failed.append('recorded source identity mismatch')
        placements=completed_placements(record['samples'])
        if any(sum(e['foot']==side and e['forward_placement_m']>.01 and e['peak_clearance_m']>.01 for e in placements)<4 for side in ('left','right')):
            failed.append('insufficient landed forward foot placement')
        if any(a['foot']==b['foot'] for a,b in zip(placements,placements[1:])):failed.append('foot placements do not alternate')
        asset=record['audit']['source']
        if asset['urdf_sha256']!='859d3c29930822f77750f6dcc0940e1c7e84393817cdefdcdc36c0025ddb46ca' or asset['mesh_tree_sha256']!='a34be1b4f2732de526c23fd1bc53e945b9e647110432fe466521fb7e73676f73':
            failed.append('canonical asset identity mismatch')
        identities.append((record['student_checkpoint_sha256'],record['model_xml_sha256'],record['backend']['source_sha256'],json.dumps(record['backend']['versions'],sort_keys=True)))
        errors.extend(f'{folder.name}: {e}' for e in failed)
        runs.append({'directory':str(folder),'metrics':record['metrics'],'failures':failed,
            'completed_placements':placements,
            'result_sha256':digest(folder/'result.json'),'trajectory_sha256':digest(folder/'trajectory.npz'),
            'auxiliary_wrench_sha256':digest(folder/'auxiliary_wrench.npz'),'source_sha256':record['source_sha256'],
            'config_sha256':record['config_sha256'],'checkpoint_sha256':record['student_checkpoint_sha256']})
    if len(set(identities))!=1:errors.append('checkpoint/model/runtime identity differs across repeats')
    metadata=json.loads((proof/'proof_metadata.json').read_text());review=json.loads((proof/'visual_review.json').read_text())
    for key,file in [('trajectory_sha256','trajectory.npz'),('video_sha256','proof.mp4')]:
        if metadata[key]!=digest(proof/file) or review[key]!=digest(proof/file):errors.append(f'proof {key} mismatch')
    if metadata['frames']!=6000 or metadata['fps']!=50 or not metadata['character_visible_every_frame']:
        errors.append('incomplete proof video')
    if not review.get('accepted_as_zero_assistance_development') or review.get('result_sha256')!=digest(proof/'result.json'):
        errors.append('missing exact zero-assistance visual review')
    report={'created_at':datetime.now(timezone.utc).isoformat(),'status':'zero_assistance_slow_steps_demonstrated' if not errors else 'failed',
        'milestone_pass':False,'scope':'Three120s slow-stepping development trials only. NOT a policy-standing or5m gate. Allthree groundloads/externalwrenches/fall/limits audited500Hz.',
        'failures':errors,'runs':runs,'proof_directory':str(proof),'proof_metadata_sha256':digest(proof/'proof_metadata.json'),
        'visual_review_sha256':digest(proof/'visual_review.json'),'validator_source_sha256':digest(__file__),
        'remaining':['Exact student must independently pass policy-controlled standing with zero command.','5m within30s and cumulative proof remain unmet.','Phase/startup gait generator has no learned proprioceptive recovery or command-generalization evidence.']}
    write_json(output,report)
    print(json.dumps({'status':report['status'],'failures':errors,'milestone_pass':False}))
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--runs',nargs='+',required=True);p.add_argument('--proof',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();validate(a.runs,a.proof,a.output)
