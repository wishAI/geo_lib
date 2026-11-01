"""Actuator-only diagnostic of FK single-support witnesses; never a learned gate.

Starts from certified nominal free-root state. Only PD targets are interpolated;
o root transform, auxiliary wrench, velocity overwrite, or reset is applied.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    import mujoco
    import numpy as np
    from algorithms.urdf_learn_wasd_walk.landau_rsl_control import configure_model
    from algorithms.urdf_learn_wasd_walk.mujoco_policy import foot_state
    from algorithms.urdf_learn_wasd_walk.mujoco_warp_runtime import WarpEvaluation
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--poses',required=True);p.add_argument('--name',required=True)
    p.add_argument('--ramp-seconds',type=float,default=6.)
    p.add_argument('--mode',choices=('witness','hip_roll_preload','shortening_pulse'),default='witness')
    p.add_argument('--preload-amplitude',type=float,default=.1)
    args=p.parse_args()
    if not 2<=args.ramp_seconds<=10:raise ValueError('Bounded diagnostic ramp required')
    if args.preload_amplitude not in (.1,.25,.35):raise ValueError('Bounded hip-roll diagnostic amplitude required')
    backend,_,_=configure_model('balanced_hands_v1')
    pose_path=Path(args.poses).resolve();pose_path.relative_to(backend.OUTPUT.resolve())
    poses=json.loads(pose_path.read_text())
    if args.mode=='hip_roll_preload':
        poses['witnesses']=[{'side':side,'hip_roll_delta_rad':sign*args.preload_amplitude} for side,sign in (('l',1),('r',-1))]
    elif args.mode=='shortening_pulse':
        poses['witnesses']=[{'side':side,'swing_side':swing,'joint_deltas_rad':{'hip_pitch':-.2,'knee':.4,'ankle_pitch':-.2}} for side,swing in (('l','right'),('r','left'))]
    for witness in poses['witnesses']:
        side=witness['side'];swing='right' if side=='l' else 'left'
        folder=(backend.OUTPUT/(args.name+'_'+side)).resolve();folder.relative_to(backend.OUTPUT.resolve())
        folder.mkdir(exist_ok=False)
        model,spec,xml=backend.build_model(dt=.002,gain_scale=1.,noslip_iterations=0,contact_timeconst=.004)
        if hashlib.sha256(xml.encode()).hexdigest()!=poses['model_xml_sha256']:raise ValueError('Witness model mismatch')
        data=backend.initialize(model,spec,pose='geometric',gain_scale=1.)
        if not np.array_equal(data.qpos,poses['nominal_qpos']) or not np.array_equal(data.ctrl,poses['nominal_ctrl']):raise ValueError('Witness initial state mismatch')
        nominal=data.ctrl.copy()
        target=np.array(witness['full_pd_targets_rad']) if args.mode=='witness' else nominal.copy()
        if args.mode=='hip_roll_preload':
            for name in spec['action_joints']:
                if 'hip_roll' in name:target[model.actuator(name).id]+=witness['hip_roll_delta_rad']
        elif args.mode=='shortening_pulse':
            for name in spec['action_joints']:
                for joint,delta in witness['joint_deltas_rad'].items():
                    if name==swing+'_'+joint+'_joint':target[model.actuator(name).id]+=delta
        active=[model.actuator(n).id for n in spec['action_joints']]
        passive=[i for i in range(model.nu) if i not in active]
        if not np.array_equal(target[passive],nominal[passive]):raise ValueError('Witness changes non-action targets')
        if np.max(np.abs(target-nominal))>.51:raise ValueError('Witness target exceeds bounded probe')
        (folder/'model.xml').write_text(xml);(folder/'backend_source.py').write_text(Path(__file__).read_text())
        (folder/'witness.json').write_text(json.dumps(witness,indent=2))
        states=[data.qpos.copy()];times=[0.];rows=[];failure=None
        base,pelvis=model.body('base_link').id,model.body('root_x').id
        height=float(data.xpos[pelvis,2]); initial=data.xpos[pelvis,:2].copy()
        initial_com=data.subtree_com[base].copy()
        foot_rotations={s:data.xmat[model.body('foot_'+s).id].reshape(3,3).copy() for s in ('l','r')}
        max_tilt=max_drop=max_speed=max_clear=max_air=air=0.;hold_clear=[]
        dofs=[model.joint(r['name']).dofadr[0] for r in spec['joints']]
        sim=WarpEvaluation(model,data,folder);began=time.perf_counter()
        duration=1.+args.ramp_seconds+4.+(args.ramp_seconds+2. if args.mode=='hip_roll_preload' else 0.)
        if args.mode=='shortening_pulse':duration=3.6
        try:
            for step in range(round(duration/.002)):
                if step%10==0:
                    phase=np.clip((data.time-1.)/args.ramp_seconds,0.,1.)
                    if args.mode=='hip_roll_preload' and data.time>1.+args.ramp_seconds+4.:
                        phase=1.-np.clip((data.time-1.-args.ramp_seconds-4.)/args.ramp_seconds,0.,1.)
                    blend=phase*phase*(3.-2.*phase)
                    if args.mode=='shortening_pulse':
                        phase=np.clip((data.time-1.)/.25,0.,1.) if data.time<1.35 else 1.-np.clip((data.time-1.35)/.25,0.,1.)
                        blend=phase**3*(10.-15.*phase+6.*phase**2)
                    data.ctrl[:]=nominal+blend*(target-nominal)
                data.xfrc_applied[:]=0.;data.qfrc_applied[:]=0.
                sim.step(data)
                tilt=float(np.arccos(np.clip(data.xmat[base].reshape(3,3)[2,2],-1,1)))
                drop=height-float(data.xpos[pelvis,2]);speed=float(np.abs(data.qvel[dofs]).max())
                max_tilt=max(max_tilt,tilt);max_drop=max(max_drop,drop);max_speed=max(max_speed,speed)
                feet=foot_state(model,data,clearance=True)
                clear=feet[swing+'_clearance_m'];max_clear=max(max_clear,clear)
                air=air+.002 if not feet[swing+'_contact'] and clear>=.015 else 0.;max_air=max(max_air,air)
                if data.time>=1.+args.ramp_seconds:hold_clear.append(clear)
                if (step+1)%10==0:
                    states.append(data.qpos.copy());times.append(float(data.time))
                    forces={'l':0.,'r':0.};force=np.zeros(6)
                    for index in range(data.ncon):
                        contact=data.contact[index]
                        if 0 not in contact.geom:continue
                        body=model.body(model.geom_bodyid[max(contact.geom)]).name
                        suffix=body[-1:]
                        if body not in ('foot_l','foot_r','toes_01_l','toes_01_r'):continue
                        mujoco.mj_contactForce(model,data,index,force)
                        forces[suffix]+=max(float((contact.frame.reshape(3,3).T@force[:3])[2]),0.)
                    rotations={s:float(np.arccos(np.clip((np.trace(foot_rotations[s].T@data.xmat[model.body('foot_'+s).id].reshape(3,3))-1.)/2.,-1,1))) for s in ('l','r')}
                    rows.append({'time_s':float(data.time),'target_blend':float(blend),'tilt_rad':tilt,
                                 'com_displacement_m':(data.subtree_com[base]-initial_com).tolist(),
                                 'vertical_load_n':forces,'left_load_fraction':forces['l']/max(sum(forces.values()),1e-9),
                                 'foot_rotation_from_initial_rad':rotations,
                                 'max_joint_speed_rad_s':speed,'max_target_error_rad':float(np.abs(data.ctrl[active]-data.qpos[[model.joint(n).qposadr[0] for n in spec['action_joints']]]).max()),**feet})
                if tilt>np.pi/6 or drop>.08:failure='fall';break
        finally:physics=sim.close()
        np.savez_compressed(folder/'trajectory.npz',qpos=np.array(states),time=np.array(times))
        backend.write_json(folder/'trace.json',rows)
        metrics={'duration_s':float(data.time),'reset_count':0,'done_count':int(failure is not None),'fall_count':int(failure=='fall'),
                 'horizontal_drift_m':float(np.linalg.norm(data.xpos[pelvis,:2]-initial)),
                 'max_reference_tilt_rad':max_tilt,'root_height_drop_m':max_drop,'max_joint_speed_rad_s':max_speed,
                 'swing_peak_clearance_m':max_clear,'sustained_15mm_swing_s':max_air,
                 'minimum_hold_clearance_m':min(hold_clear) if hold_clear else None,'peak_auxiliary_wrench_norm':0.}
        result={'created_at':datetime.now(timezone.utc).isoformat(),'status':'diagnostic_only','gate_passed':False,
                'milestone':'weight_transfer_diagnostic','controller_kind':'scripted_joint_targets',
                'config':vars(args),'metrics':metrics,'failures':([failure] if failure else [])+['Scripted geometry diagnostic; not trained-policy milestone evidence'],
                'identity':{'backend':'mujoco_warp_cuda','model_xml_sha256':hashlib.sha256(xml.encode()).hexdigest(),
                    'source_sha256':backend.digest(__file__),'urdf_sha256':spec['source']['urdf_sha256'],'mesh_tree_sha256':spec['source']['mesh_tree_sha256']},
                'pose_file_sha256':backend.digest(pose_path),'trajectory_sha256':backend.digest(folder/'trajectory.npz'),
                'performance':{'simulation_wall_s':time.perf_counter()-began,'warp_setup':physics},
                'root_state_interventions':0,'reproduce_job':{'kind':'module','module':'algorithms.urdf_learn_wasd_walk.landau_support_probe','args':sys.argv[1:]}}
        backend.write_json(folder/'dynamics.json',result)
        subprocess.run([sys.executable,'-m','algorithms.urdf_learn_wasd_walk.continuation','--mode','render','--render-directory',str(folder)],check=True,timeout=120)
        print(json.dumps({'side':side,'metrics':metrics}),flush=True)


if __name__=='__main__':main()
