"""Sequential, measured assistance reductions for the fresh ragdoll student.

Each stage needs three complete physical trials. The first regression stops the
worker cycle and rolls back the recorded coefficient; no milestone is promoted.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import torch

from algorithms.urdf_learn_wasd_walk.mujoco_backend import OUTPUT, digest, write_json
from algorithms.urdf_learn_wasd_walk.mujoco_ragdoll_teacher import LINEAGE


def failures(result):
    m=result['metrics'];bad=[]
    if result.get('teacher_blend_schedule'):bad.append('diagnostic rescue schedule is not curriculum evidence')
    if result.get('diagnostic_oracle_student'):bad.append('oracle teacher guidance is not student curriculum evidence')
    if result.get('diagnostic_observation_replay_sha256'):bad.append('reference observations are not student curriculum evidence')
    if result.get('teacher_blend_coefficient',result['config']['coefficient'])!=result['config']['coefficient']:bad.append('diagnostic teacher blend differs from curriculum coefficient')
    if m['duration_s']<30-1e-6:bad.append('duration below30s')
    if m['forward_m']<1.:bad.append('forward below1m')
    if min(m['liftoffs'].values())<10:bad.append('fewer than10 sustainedlifts each')
    if m['mean_contact_slip_mps']>=.05:bad.append('contact slip at least.05m/s')
    if m['max_joint_speed_rad_s']>4.:bad.append('source velocity limit exceeded')
    if m['max_effort_fraction']>1.00001:bad.append('source effort limit exceeded')
    if m['both_air_fraction']>=.02:bad.append('excessive flight')
    if m['fall'] or m['reset_count'] or m['done_count'] or m['nonfoot_contacts']:bad.append('fall/reset/done/nonfootcontact')
    if result['config']['coefficient']==0 and (result['teacher_targets_evaluated'] or max(m['max_abs_wrench_components'])!=0):bad.append('zero-stage assistance notoff')
    return bad


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--name',required=True);p.add_argument('--student',required=True);p.add_argument('--first-trial')
    args=p.parse_args();root=OUTPUT/'ragdoll'/LINEAGE;out=root/args.name
    out.resolve().relative_to(root.resolve());out.mkdir(parents=True,exist_ok=False)
    student=Path(args.student).resolve();student.relative_to(root.resolve());sha=digest(student)
    teacher_config=torch.load(student,map_location='cpu',weights_only=False).get('teacher_config',{})
    if teacher_config.get('motor_pose_sequence') and digest(teacher_config['motor_pose_sequence'])!=teacher_config['motor_pose_sequence_sha256']:
        raise ValueError('Motor reference changed since training')
    parameter_names=('phase_delay','period','stride','clearance','height_gain','orientation_damping','waist_amplitude','hip_roll_amplitude','lateral_assistance_scale','foot_placement_gain','root_lean_amplitude','waist_transfer_sharpness','motor_balance_gain','motor_balance_scale','waist_phase_lead','motor_pose_sequence','teacher_tracking_integral')
    teacher_args=[arg for key in parameter_names if key in teacher_config and teacher_config[key] is not None for arg in ('--'+key.replace('_','-'),str(teacher_config[key]))]
    state_path=root/'curriculum.json';state=json.loads(state_path.read_text())
    if not state['assisted_walking_demonstrated']:raise ValueError('Reviewed teacher required')
    state.update(student=str(student),student_sha256=sha,active_cycle=str(out),source_sha256=digest(__file__))
    (out/'curriculum_source.py').write_text(Path(__file__).read_text())
    progress_path=OUTPUT.parent/'backend_progress.json';progress=json.loads(progress_path.read_text())
    # Preserve the worker identity supplied by the parent before child diagnostics.
    active=progress.get('active_process');accepted=1.;history=[];cycle_error=None
    try:
        for coefficient in (.8,.6,.4,.2,.1,.05,0.):
            for trial in range(1,4):
                name=f'{args.name}_c{round(coefficient*100):02}_trial{trial}'
                if coefficient==.8 and trial==1 and args.first_trial:
                    result_path=Path(args.first_trial).resolve();result_path.relative_to(root.resolve())
                    result=json.loads(result_path.read_text())
                else:
                    command=[sys.executable,'-m','algorithms.urdf_learn_wasd_walk.mujoco_ragdoll_teacher','--name',name,'--seconds','30','--coefficient',str(coefficient),'--student',str(student)]+teacher_args
                    progress.update(updated_at=datetime.now(timezone.utc).isoformat(),active_process=active,assistance_coefficient=coefficient,teacher_blend_coefficient=coefficient,student_progress=f'physicalcurriculum coefficient{coefficient} trial{trial}/3',next_step='Complete measured stage; rollback and stop on first regression.',resume_procedure=f'Inspect {out}/cycle.json and worker result; do not duplicate activecycle.')
                    write_json(progress_path,progress)
                    with (out/f'{name}.log').open('w') as log:
                        child=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=180)
                    if child.returncode:raise RuntimeError(f'Child failed {child.returncode}: {name}')
                    result_path=root/name/'result.json';result=json.loads(result_path.read_text())
                if result['student_checkpoint_sha256']!=sha or result['config']['coefficient']!=coefficient:raise ValueError('Wrong evaluated checkpoint/coefficient')
                bad=failures(result);entry={'coefficient':coefficient,'trial':trial,'result':str(result_path),'sha256':digest(result_path),'failures':bad,'metrics':result['metrics']};history.append(entry)
                state['history'].append(entry);state['coefficient']=coefficient
                progress.update(updated_at=datetime.now(timezone.utc).isoformat(),validation=result['metrics'],failure_metrics=bad,artifact_paths=[str(out),str(result_path)])
                write_json(out/'cycle.json',{'state':'running','student_sha256':sha,'history':history,'last_accepted_coefficient':accepted,'milestone_pass':False})
                if bad:
                    state.update(coefficient=accepted,status='regression_rolled_back',regressed_coefficient=coefficient,next_step='Diagnose measured failure; retain accepted assisted teacher/student stage. No further reduction until competency restored.')
                    progress.update(assistance_coefficient=accepted,next_step=state['next_step'],blocker=f'Regression at coefficient{coefficient}: '+', '.join(bad))
                    write_json(state_path,state)
                    write_json(out/'cycle.json',{'state':'regression_rolled_back','history':history,'last_accepted_coefficient':accepted,'milestone_pass':False})
                    return
                write_json(state_path,state);write_json(progress_path,progress)
            accepted=coefficient;state['last_accepted_coefficient']=accepted
            progress.update(accepted_assistance_coefficient=accepted,accepted_student={'checkpoint':str(student),'coefficient':accepted})
            write_json(state_path,state);write_json(progress_path,progress)
        state.update(status='zero_assistance_dynamics_requires_cumulative_proof',next_step='Independently revalidate stand and5m with this exact student, fullproof and visualreview; no automaticpromotion')
        progress.update(next_step=state['next_step'],blocker=None)
        write_json(state_path,state);write_json(out/'cycle.json',{'state':state['status'],'history':history,'last_accepted_coefficient':accepted,'milestone_pass':False})
    except Exception as error:
        cycle_error=repr(error)
        state.update(status='runtime_failure',coefficient=accepted)
        write_json(state_path,state)
        progress.update(blocker=cycle_error,next_step='Inspect cycle failure.json before restart; do not claim stage completion.')
        write_json(out/'failure.json',{'error':repr(error),'history':history,'milestone_pass':False});raise
    finally:
        progress.update(updated_at=datetime.now(timezone.utc).isoformat(),active_process=None,job={'status':'failed' if cycle_error else 'completed','cycle':str(out)},assistance_coefficient=state['coefficient'],teacher_blend_coefficient=state['coefficient'])
        write_json(progress_path,progress)

if __name__=='__main__':main()
