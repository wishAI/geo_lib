"""Bounded physical ankle-authority diagnostic, never walking gate evidence.

Compare the existing ankle residual cap with a retargeted static-FK solution.
Only ordinary joint motor targets change; the floating root is never prescribed.
Cases with bounded physical pelvis assistance are explicitly separate controls.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from algorithms.urdf_learn_wasd_walk.mujoco_backend import (
    OUTPUT, Assistance, audit_model, build_model, digest, initialize, write_json,
)
from algorithms.urdf_learn_wasd_walk.mujoco_policy import foot_state
from algorithms.urdf_learn_wasd_walk.mujoco_warp_runtime import WarpEvaluation


def run_case(name, ankle_cap, coefficient, shift='none', study=False):
    out=OUTPUT/name
    out.resolve().relative_to(OUTPUT.resolve());out.mkdir(parents=True,exist_ok=False)
    model,spec,xml=build_model(noslip_iterations=0,contact_timeconst=.004)
    audit=audit_model(model,spec);data=initialize(model,spec,pose='geometric')
    (out/'model.xml').write_text(xml);(out/'probe_source.py').write_text(Path(__file__).read_text())
    pelvis=model.body('root_x').id;base=model.body('base_link').id
    nominal=data.ctrl.copy();initial=data.xpos[pelvis].copy()
    reference_rotation=data.xmat[pelvis].reshape(3,3).copy()
    joints=['left_hip_pitch_joint','left_knee_joint','left_ankle_pitch_joint']
    aids=[model.actuator(n).id for n in joints]
    qadr=[int(model.joint(n).qposadr[0]) for n in joints]
    delta=np.array([.117542,.175312,-min(.292854,ankle_cap)])
    shift_targets={'none':{},'waist_positive':{'waist_roll_joint':.24},
                   'waist_negative':{'waist_roll_joint':-.24},
                   'hips_positive':{'left_hip_roll_joint':.12,'right_hip_roll_joint':.12},
                   'hips_negative':{'left_hip_roll_joint':-.12,'right_hip_roll_joint':-.12}}[shift]
    shift_aids=[model.actuator(n).id for n in shift_targets]
    shift_qadr=[int(model.joint(n).qposadr[0]) for n in shift_targets]
    def smooth(x):
        x=float(np.clip(x,0,1));return x*x*(3-2*x)
    assistance=Assistance(coefficient);runtime=WarpEvaluation(model,data,out)
    rows=[];forces=[];states=[];fall=False;start=time.perf_counter()
    try:
        for step in range(2000 if study else 1000):
            t=step*.002
            envelope=(smooth((t-1.8)/.3)*smooth((2.9-t)/.3) if study else
                      float(np.clip((t-.5)/.3,0,1)*np.clip((1.5-t)/.3,0,1)))
            data.ctrl[:]=nominal;data.ctrl[aids]+=envelope*delta
            shift_envelope=smooth((t-.5)/1.)*smooth((4.-t)/.8) if study else 0.
            data.ctrl[shift_aids]+=shift_envelope*np.array(list(shift_targets.values()))
            velocity=np.zeros(6)
            mujoco.mj_objectVelocity(model,data,mujoco.mjtObj.mjOBJ_BODY,pelvis,velocity,0)
            error=Rotation.from_matrix(reference_rotation@data.xmat[pelvis].reshape(3,3).T).as_rotvec()
            wrench=assistance.wrench(initial[2]-data.xpos[pelvis,2],velocity[5],error,velocity[:3])
            data.xfrc_applied[:]=0;data.xfrc_applied[pelvis]=wrench
            runtime.step(data)
            tilt=float(np.arccos(np.clip(data.xmat[base].reshape(3,3)[2,2],-1,1)))
            fall=bool(tilt>np.pi/6 or data.xpos[pelvis,2]<initial[2]-.08)
            forces.append([data.time,*wrench.tolist()])
            if step%10==0 or fall:
                feet=foot_state(model,data,clearance=True)
                normal_forces={'left':0.,'right':0.};contact_points=[];contact_force=np.zeros(6)
                for index in range(data.ncon):
                    ids=list(data.contact[index].geom)
                    if 0 not in ids:continue
                    body=model.body(model.geom_bodyid[max(ids)]).name
                    side='left' if body in ('foot_l','toes_01_l') else 'right' if body in ('foot_r','toes_01_r') else None
                    if side:
                        mujoco.mj_contactForce(model,data,index,contact_force)
                        normal_forces[side]+=max(float(contact_force[0]),0.)
                        contact_points.append({'side':side,'position_m':data.contact[index].pos.tolist(),'normal_force_n':float(contact_force[0])})
                rows.append({'time_s':data.time,'envelope':envelope,'feet':feet,
                    'pelvis_position_m':data.xpos[pelvis].tolist(),'tilt_rad':tilt,
                    'com_world_m':data.subtree_com[base].tolist(),'foot_normal_forces_n':normal_forces,
                    'foot_contact_points':contact_points,'shift_envelope':shift_envelope,
                    'shift_actual_residual_rad':(data.qpos[shift_qadr]-nominal[shift_aids]).tolist(),
                    'requested_residual_rad':(envelope*delta).tolist(),
                    'actual_residual_rad':(data.qpos[qadr]-nominal[aids]).tolist(),
                    'motor_torques_nm':data.actuator_force[aids].tolist(),
                    'max_joint_speed_rad_s':float(np.abs(data.qvel[6:]).max()),
                    'auxiliary_wrench':wrench.tolist(),'fall':fall})
                states.append(data.qpos.copy())
            if fall:break
    finally:
        runtime_info=runtime.close()
    hold=(2.1,2.6) if study else (.8,1.2)
    valid=[r for r in rows if hold[0]<=r['time_s']<=hold[1] and not r['fall'] and r['feet']['right_contact']]
    shifted=[r for r in rows if 1.5<=r['time_s']<1.8 and not r['fall']] if study else []
    np.savez_compressed(out/'trajectory.npz',qpos=np.array(states),time=np.array([r['time_s'] for r in rows]))
    np.savez_compressed(out/'auxiliary_wrench.npz',trace=np.array(forces),coefficient=coefficient)
    result={'created_at':datetime.now(timezone.utc).isoformat(),'milestone_pass':False,
        'purpose':'Physical motor/contact diagnostic with explicit assistance controls; not policy or gate evidence',
        'config':{'ankle_residual_cap_rad':ankle_cap,'assistance_coefficient':coefficient,
                  'duration_bound_s':4. if study else 2.,'seed':42,'commanded_joints':joints,
                  'weight_shift_targets_rad':shift_targets,'study':study},
        'source_sha256':digest(__file__),'model_xml_sha256':hashlib.sha256(xml.encode()).hexdigest(),
        'audit':audit,'backend':runtime_info,'duration_s':data.time,'fall':fall,
        'physics_and_validation_wall_s':time.perf_counter()-start,
        'maximum_hold_left_clearance_with_right_contact_m':max((r['feet']['left_clearance_m'] for r in valid),default=None),
        'hold_samples_left_airborne_right_contact':sum(not r['feet']['left_contact'] for r in valid),
        'mean_left_load_fraction_before_swing':float(np.mean([r['foot_normal_forces_n']['left']/max(sum(r['foot_normal_forces_n'].values()),1e-9) for r in shifted])) if shifted else None,
        'maximum_applied_force_n':float(np.linalg.norm(np.array(forces)[:,1:4],axis=1).max()),
        'maximum_applied_torque_nm':float(np.linalg.norm(np.array(forces)[:,4:7],axis=1).max()),
        'samples':rows}
    write_json(out/'result.json',result)
    return {'directory':str(out),**{k:result[k] for k in ('duration_s','fall','maximum_hold_left_clearance_with_right_contact_m','hold_samples_left_airborne_right_contact','mean_left_load_fraction_before_swing','maximum_applied_force_n','maximum_applied_torque_nm')}}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--name',required=True)
    parser.add_argument('--weight-shift-study',action='store_true')
    args=parser.parse_args();report={'milestone_pass':False,'cases':[]}
    path=OUTPUT/f'motor_probe_{args.name}.json'
    if path.exists():raise FileExistsError(path)
    if args.weight_shift_study:
        for shift in ('none','waist_positive','waist_negative','hips_positive','hips_negative'):
            report['cases'].append(run_case(f'motor_probe_{args.name}_{shift}',.35,0.,shift,True));write_json(path,report)
    else:
        for coefficient in (0.,1.):
            for cap in (.24,.35):
                name=f'motor_probe_{args.name}_assist{int(coefficient)}_cap{int(cap*100):02}'
                report['cases'].append(run_case(name,cap,coefficient));write_json(path,report)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
