"""Physical walking teacher for the fresh ragdoll-first curriculum.

IK is evaluated only on scratch data to obtain ordinary motor position targets.
The simulated root is never written after initialization. Balance assistance has
zero forward force: only bounded lateral/vertical forces and orientation torque.
All joint control/effort limits remain those of the canonical URDF.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from algorithms.urdf_learn_wasd_walk.mujoco_backend import OUTPUT, build_model, initialize, audit_model, digest, write_json
from algorithms.urdf_learn_wasd_walk.mujoco_policy import foot_state, observe
from algorithms.urdf_learn_wasd_walk.mujoco_warp_runtime import WarpEvaluation

LINEAGE = 'ragdoll_walk_first_20260918'


def smooth(x):
    x = np.clip(x, 0., 1.)
    return x*x*(3.-2.*x)


def diagnostic_teacher_blend(t, initial, rescue_time=None):
    """Explicit quarter-second handback, never a curriculum or gate pass."""
    if rescue_time is None:return initial
    return float(initial+(1-initial)*smooth((t-rescue_time)/.25))


def sustained_liftoffs(samples):
    """Count each flight once, rearming only after two loaded 50 Hz samples."""
    counts={'left':0,'right':0}
    state={side:[False,0,0] for side in counts}
    for sample in samples:
        for side in counts:
            feet=sample['feet'];armed,ground,air=state[side]
            if feet[side+'_contact']:
                ground+=1;air=0
                if ground>=2:armed=True
            else:
                ground=0
                air=air+1 if feet[side+'_clearance_m']>.002 else 0
                if armed and air>=3:
                    counts[side]+=1;armed=False
            state[side]=[armed,ground,air]
    return counts


class MotorPoseSequence:
    """Interpolate audited motor-only references; scratch root poses are never used."""
    def __init__(self, model, spec, data, path, model_sha256):
        source=Path(path).resolve();source.relative_to(OUTPUT.resolve())
        record=json.loads(source.read_text())
        if record.get('schema')!=1 or record.get('model_xml_sha256')!=model_sha256:
            raise ValueError('Motor reference model identity mismatch')
        if record['action_joints']!=spec['action_joints']:
            raise ValueError('Motor reference joint order mismatch')
        self.nominal=data.ctrl.copy();self.ik_errors=None
        self.aids=np.array([model.actuator(n).id for n in spec['action_joints']])
        self.times=np.asarray(record['times_s'],dtype=float)
        self.positions=np.asarray(record['positions_rad'],dtype=float)
        self.loop=bool(record.get('loop',False))
        self.loop_start=float(record.get('loop_start_s',0.))
        if (self.times.ndim!=1 or len(self.times)<2 or self.times[0]!=0
                or not np.all(np.diff(self.times)>0) or not np.isfinite(self.times).all()
                or self.positions.shape!=(len(self.times),len(self.aids))
                or not np.isfinite(self.positions).all()):
            raise ValueError('Invalid motor reference dimensions/times')
        if (np.any(self.positions<model.actuator_ctrlrange[self.aids,0])
                or np.any(self.positions>model.actuator_ctrlrange[self.aids,1])):
            raise ValueError('Motor reference violates original position limits')
        starts=np.flatnonzero(np.isclose(self.times,self.loop_start,rtol=0.,atol=1e-9))
        if (not np.isfinite(self.loop_start) or self.loop_start<0
                or self.loop_start>=self.times[-1] or len(starts)!=1):
            raise ValueError('Loop start must identify a reference node before the end')
        self.loop_duration=self.times[-1]-self.loop_start
        if self.loop and not np.allclose(self.positions[starts[0]],self.positions[-1],rtol=0.,atol=1e-9):
            raise ValueError('Looped motor reference must close continuously')

    def targets(self,t,lateral_velocity=0.):
        if self.loop and t>=self.times[-1]:
            t=self.loop_start+(t-self.loop_start)%self.loop_duration
        index=min(max(np.searchsorted(self.times,t,side='right')-1,0),len(self.times)-2)
        fraction=smooth((t-self.times[index])/(self.times[index+1]-self.times[index]))
        target=self.nominal.copy()
        target[self.aids]=(1-fraction)*self.positions[index]+fraction*self.positions[index+1]
        return target


class WalkingTeacher:
    def __init__(self, model, spec, data, *, period=2., stride=.06, clearance=.025, waist_amplitude=.18, hip_roll_amplitude=0., foot_placement_gain=0., waist_transfer_sharpness=0., waist_phase_lead=0.):
        self.model = model
        self.nominal = data.ctrl.copy()
        self.period = period
        self.stride = stride
        self.clearance = clearance
        self.waist_amplitude = waist_amplitude
        self.waist_transfer_sharpness = waist_transfer_sharpness
        self.waist_phase_lead = waist_phase_lead
        self.hip_roll_amplitude = hip_roll_amplitude
        self.foot_placement_gain = foot_placement_gain
        self.tables = {}
        self.ik_errors = {}
        scratch = mujoco.MjData(model)
        scratch.qpos[:] = data.qpos
        for side, body in [('left','foot_l'), ('right','foot_r')]:
            names = [f'{side}_{j}_joint' for j in ('hip_pitch','knee','ankle_pitch')]
            aids = np.array([model.actuator(n).id for n in names])
            qids = [int(model.joint(n).qposadr[0]) for n in names]
            bid = model.body(body).id
            mujoco.mj_kinematics(model,scratch)
            pos = scratch.xpos[bid].copy()
            rot = scratch.xmat[bid].reshape(3,3).copy()
            original = scratch.qpos[qids].copy()
            rows=[]; errors=[]
            for phase in np.linspace(0,1,201):
                # 60% stance; 40% swing. The stance sole moves backward relative
                # to the body, producing forward propulsion through real contact.
                if phase < .6:
                    y = stride*(.5-phase/.6)
                    z = 0.
                else:
                    u=(phase-.6)/.4
                    y=stride*(smooth(u)-.5)
                    z=clearance*np.sin(np.pi*u)**2
                target=pos+np.array([0,y,z])
                def residual(q):
                    scratch.qpos[qids]=q
                    mujoco.mj_kinematics(model,scratch)
                    dr=Rotation.from_matrix(rot@scratch.xmat[bid].reshape(3,3).T).as_rotvec()
                    return np.r_[(scratch.xpos[bid]-target)[1:], .08*dr]
                result=least_squares(residual,original,bounds=(model.actuator_ctrlrange[aids,0]+1e-6,model.actuator_ctrlrange[aids,1]-1e-6),max_nfev=80,xtol=1e-10,gtol=1e-10,ftol=1e-10)
                rows.append(result.x);errors.append(float(np.linalg.norm(residual(result.x))))
            scratch.qpos[qids]=original
            self.tables[side]=(aids,np.array(rows))
            self.ik_errors[side]=max(errors)

    def targets(self,t, lateral_velocity=0.):
        target=self.nominal.copy()
        ramp=smooth((t-.5)/1.)
        for side,offset in [('left',.6),('right',.1)]:
            phase=((max(t-.5,0.)/self.period)+offset)%1.
            aids,table=self.tables[side]
            q=np.array([np.interp(phase,np.linspace(0,1,201),table[:,i]) for i in range(3)])
            target[aids]+=ramp*(q-self.nominal[aids])
            # A laterally moving body places the swing foot in that direction.
            # Canonical hip-roll FK has negative foot-X derivative. This is an
            # ordinary motor target correction, never an external root force.
            if phase>=.6:
                envelope=np.sin(np.pi*(phase-.6)/.4)**2
                hip=self.model.actuator(f'{side}_hip_roll_joint').id
                target[hip]-=ramp*envelope*np.clip(self.foot_placement_gain*lateral_velocity,-.12,.12)
        # Explicit upper-body weight transfer; normal motor tracking, not a root move.
        aid=self.model.actuator('waist_roll_joint').id
        transfer=np.sin(2*np.pi*max(t-.5,0)/self.period+self.waist_phase_lead)
        if self.waist_transfer_sharpness>0:
            transfer=np.tanh(self.waist_transfer_sharpness*transfer)/np.tanh(self.waist_transfer_sharpness)
        target[aid]+=self.waist_amplitude*ramp*transfer
        for side in ('left','right'):
            hip=self.model.actuator(f'{side}_hip_roll_joint').id
            target[hip]-=self.hip_roll_amplitude*ramp*np.sin(2*np.pi*max(t-.5,0)/self.period)
        return np.clip(target,self.model.actuator_ctrlrange[:,0],self.model.actuator_ctrlrange[:,1])


def run(args):
    blend=args.coefficient if args.teacher_blend is None else args.teacher_blend
    initial_blend=blend
    rescue_time=args.teacher_rescue_time
    oracle=args.diagnostic_oracle_student
    replay=None;replay_sha=None
    if args.diagnostic_observation_replay:
        if not args.student or args.coefficient!=0 or not 0<blend<1 or rescue_time is not None or oracle:
            raise ValueError('Observation replay requires a fixed blended learned student and external support0')
        replay_path=Path(args.diagnostic_observation_replay).resolve();replay_path.relative_to(OUTPUT.resolve())
        replay=np.load(replay_path)['observations'];replay_sha=digest(replay_path)
        if replay.ndim!=2 or replay.shape[1]!=63 or len(replay)<int(np.ceil(args.seconds/.02)) or not np.isfinite(replay).all():
            raise ValueError('Incomplete or invalid reference observations')
    if oracle and (args.student or rescue_time is not None or not 0<blend<1 or args.coefficient!=0):
        raise ValueError('Oracle requires no learned student, fixed partial teacher blend and external support0')
    if rescue_time is not None and (not 0<initial_blend<1 or not args.student or args.coefficient!=0 or not 0<=rescue_time<args.seconds-.25):
        raise ValueError('Rescue diagnostic requires a blended student, external support0, and a bounded rescue time')
    out=OUTPUT/'ragdoll'/LINEAGE/args.name
    out.resolve().relative_to(OUTPUT.resolve());out.mkdir(parents=True,exist_ok=False)
    start=time.perf_counter()
    model,spec,xml=build_model(noslip_iterations=0,contact_timeconst=.004)
    audit=audit_model(model,spec);data=initialize(model,spec,pose='geometric')
    (out/'model.xml').write_text(xml);(out/'teacher_source.py').write_text(Path(__file__).read_text())
    reference_sha=None
    if args.motor_pose_sequence:
        reference=Path(args.motor_pose_sequence).resolve();reference.relative_to(OUTPUT.resolve())
        reference_sha=digest(reference);(out/'motor_pose_sequence.json').write_text(reference.read_text())
    if blend>0 and args.motor_pose_sequence:
        teacher=MotorPoseSequence(model,spec,data,args.motor_pose_sequence,hashlib.sha256(xml.encode()).hexdigest())
        if teacher.loop and not np.isclose(args.period,teacher.loop_duration):
            raise ValueError('Loop duration must match student phase period')
    elif blend>0:
        teacher=WalkingTeacher(model,spec,data,period=args.period,stride=args.stride,clearance=args.clearance,waist_amplitude=args.waist_amplitude,hip_roll_amplitude=args.hip_roll_amplitude,foot_placement_gain=args.foot_placement_gain,waist_transfer_sharpness=args.waist_transfer_sharpness,waist_phase_lead=args.waist_phase_lead)
    else:
        from types import SimpleNamespace
        teacher=SimpleNamespace(nominal=data.ctrl.copy(),ik_errors=None)
    student=None
    if args.student:
        import torch
        from algorithms.urdf_learn_wasd_walk.mujoco_ragdoll_student import load_student
        student=load_student(args.student)
    student_target=teacher.nominal.copy()
    pelvis=model.body('root_x').id;base=model.body('base_link').id
    initial=data.xpos[pelvis].copy();rotation=data.xmat[pelvis].reshape(3,3).copy()
    previous_target=data.ctrl.copy();q_nominal=data.qpos.copy();previous_action=np.zeros(17);demonstrations=[]
    action_jids=[model.joint(n).id for n in spec['action_joints']]
    runtime=WarpEvaluation(model,data,out)
    startup=time.perf_counter()-start;begin=time.perf_counter()
    rows=[];states=[];wrenches=[];fall=False;maxspeed=0.;maxtorque=0.;nonfeet=set()
    liftoffs={'left':0,'right':0};landings={'left':0,'right':0};air={'left':0,'right':0};last={'left':True,'right':True}
    maxclear={'left':0.,'right':0.};bothair=0;slip=[];peak_tracking=0.
    action_aids=[model.actuator(n).id for n in spec['action_joints']]
    action_qids=[int(model.joint(n).qposadr[0]) for n in spec['action_joints']]
    filtered_load=np.zeros(2);tracking_integral=np.zeros(model.nu)
    body_weight=float(model.body_mass.sum()*np.linalg.norm(model.opt.gravity))
    ground_force=np.zeros(6);unsupported_steps=0;unsupported_run=0;max_unsupported_run=0
    support_sum=0.;support_peak=0.
    for step in range(round(args.seconds/.002)):
        t=step*.002
        blend=diagnostic_teacher_blend(t,initial_blend,rescue_time)
        velocity=np.zeros(6);mujoco.mj_objectVelocity(model,data,mujoco.mjtObj.mjOBJ_BODY,pelvis,velocity,0)
        lean=-args.root_lean_amplitude*smooth((t-.5)/1.)*np.sin(2*np.pi*max(t-.5,0)/args.period) if args.coefficient>0 or blend>0 else 0.
        orientation_reference=Rotation.from_rotvec([0.,lean,0.]).as_matrix()@rotation
        error=Rotation.from_matrix(orientation_reference@data.xmat[pelvis].reshape(3,3).T).as_rotvec()
        requested=teacher.targets(t,lateral_velocity=velocity[3]) if blend>0 else teacher.nominal.copy()
        motor_balance_offset=np.zeros(model.nu);load_fraction=np.zeros(2);motor_reaction=0.
        if blend>0 and args.motor_balance_gain!=0:
            contact_load=np.zeros(2);contact_force=np.zeros(6)
            for ci in range(data.ncon):
                gids=list(data.contact[ci].geom)
                if 0 not in gids:continue
                body=model.body(model.geom_bodyid[max(gids)]).name
                side=0 if body in ('foot_l','toes_01_l') else 1 if body in ('foot_r','toes_01_r') else None
                if side is not None:
                    mujoco.mj_contactForce(model,data,ci,contact_force)
                    contact_load[side]+=max(float(contact_force[0]),0.)
            filtered_load+=(1-np.exp(-.002/.04))*(contact_load-filtered_load)
            if contact_load.sum()>=2 and filtered_load.sum()>=2:
                load_fraction=filtered_load/filtered_load.sum()
                motor_reaction=args.motor_balance_scale*float(np.clip(args.motor_balance_gain*error[1]-np.sign(args.motor_balance_gain)*.15*velocity[1],-.25,.25))
                for index,side in enumerate(('left','right')):
                    name=f'{side}_hip_roll_joint';aid=model.actuator(name).id;jid=model.joint(name).id
                    axis=(data.xmat[model.jnt_bodyid[jid]].reshape(3,3)@model.jnt_axis[jid])[1]
                    if abs(axis)<.2:raise RuntimeError('Hip roll axis outside feedback diagnostic domain')
                    # Equal/opposite motor reaction on pelvis, gated by actual load.
                    # This is a bounded target offset, never an external force.
                    motor_balance_offset[aid]=np.clip(-load_fraction[index]*motor_reaction/(model.actuator_gainprm[aid,0]*axis),-.02*args.motor_balance_scale,.02*args.motor_balance_scale)
                requested=np.clip(requested+motor_balance_offset,model.actuator_ctrlrange[:,0],model.actuator_ctrlrange[:,1])
        motor_reference=requested[action_aids].copy() if blend>0 else None
        if blend>0 and args.teacher_tracking_integral>0:
            tracking_integral[action_aids]+=args.teacher_tracking_integral*.002*(requested[action_aids]-data.qpos[action_qids])
            tracking_integral=np.clip(tracking_integral,-.08,.08)
            # Anti-windup retains original motor target limits; only17 action
            # motors integrate. All52 held joints retain nominal PD targets.
            tracking_integral=np.clip(tracking_integral,model.actuator_ctrlrange[:,0]-requested,model.actuator_ctrlrange[:,1]-requested)
            requested=requested+tracking_integral
        if step%10==0:
            phase=2*np.pi*max(t-args.phase_delay,0.)/args.period
            obs=np.r_[observe(model,data,q_nominal,action_jids,previous_action), args.stride/(.6*args.period),np.sin(phase),np.cos(phase)].astype(np.float32)
            if replay is not None:
                if args.diagnostic_replay_block=='joint_position':obs[9:26]=replay[step//10,9:26]
                else:obs=replay[step//10].copy()
            if student is not None:
                if student.startup_clock_s>0:obs=np.r_[obs,np.float32(np.clip(t/student.startup_clock_s,0.,1.))]
                with torch.no_grad():student_action=student(torch.as_tensor(obs,device='cuda:0')).cpu().numpy()
                student_target[action_aids]=teacher.nominal[action_aids]+student.action_scale*student_action
                student_target=np.clip(student_target,model.actuator_ctrlrange[:,0],model.actuator_ctrlrange[:,1])
        # Reference target rate is bounded below source velocity limits; no state clamp.
        target=previous_target+np.clip(requested-previous_target,-3.*.002,3.*.002)
        previous_target=target.copy()
        if oracle and step%10==0:
            # Timing counterfactual only: this branch is also entirely teacher
            # assistance, sampled at exactly the learned actor's50Hz cadence.
            student_target=target.copy()
        data.ctrl[:]=blend*target+(1-blend)*student_target
        if step%10==0:
            previous_action=(data.ctrl[action_aids]-teacher.nominal[action_aids])/.5
            demonstrations.append((obs.copy(),previous_action.copy()))
        velocity=np.zeros(6);mujoco.mj_objectVelocity(model,data,mujoco.mjtObj.mjOBJ_BODY,pelvis,velocity,0)
        wrench=np.zeros(6)
        wrench[0]=args.lateral_assistance_scale*np.clip(100*(initial[0]-data.xpos[pelvis,0])-10*velocity[3],-4,4)
        wrench[2]=np.clip(args.height_gain*(initial[2]-data.xpos[pelvis,2])-30*velocity[5],-15,15)
        wrench[3:]=30*error-args.orientation_damping*velocity[:3]
        norm=np.linalg.norm(wrench[3:])
        if norm>5:wrench[3:]*=5/norm
        wrench*=args.coefficient
        data.xfrc_applied[:]=0;data.xfrc_applied[pelvis]=wrench
        runtime.step(data)
        wrenches.append(wrench.copy())
        tilt=float(np.arccos(np.clip(data.xmat[base].reshape(3,3)[2,2],-1,1)))
        maxspeed=max(maxspeed,float(np.abs(data.qvel[6:]).max()))
        maxtorque=max(maxtorque,float(np.max(np.abs(data.actuator_force)/np.maximum(model.actuator_forcerange[:,1],1e-9))))
        fall=bool(tilt>np.pi/6 or data.xpos[pelvis,2]<initial[2]-.08)
        force={'left':0.,'right':0.}
        for i in range(data.ncon):
            ids=list(data.contact[i].geom)
            if 0 in ids:
                body=model.body(model.geom_bodyid[max(ids)]).name
                if body not in ('foot_l','foot_r','toes_01_l','toes_01_r'):nonfeet.add(body)
                side='left' if body in ('foot_l','toes_01_l') else 'right' if body in ('foot_r','toes_01_r') else None
                if side:
                    mujoco.mj_contactForce(model,data,i,ground_force)
                    force[side]+=max(float(ground_force[0]),0.)
        supported=force['left']>.1 or force['right']>.1
        unsupported_steps+=int(not supported)
        unsupported_run=0 if supported else unsupported_run+1
        max_unsupported_run=max(max_unsupported_run,unsupported_run)
        support_ratio=sum(force.values())/body_weight
        support_sum+=support_ratio;support_peak=max(support_peak,support_ratio)
        if step%10==0 or fall:
            feet=foot_state(model,data,clearance=True)
            for side in ('left','right'):
                contact=feet[f'{side}_contact'];clear=feet[f'{side}_clearance_m']
                maxclear[side]=max(maxclear[side],clear)
                air[side]=air[side]+1 if not contact and clear>.002 else 0
                if air[side]==3:liftoffs[side]+=1
                if contact and not last[side]:landings[side]+=1
                last[side]=contact
            bothair+=int(not feet['left_contact'] and not feet['right_contact']);slip.append(feet['mean_slip_mps'])
            peak_tracking=max(peak_tracking,float(np.max(abs(data.qpos[action_qids]-data.ctrl[action_aids]))))
            rows.append({'time_s':data.time,'feet':feet,'foot_force_n':force,'pelvis_position_m':data.xpos[pelvis].tolist(),'com_world_m':data.subtree_com[base].tolist(),'tilt_rad':tilt,'root_velocity':velocity.tolist(),'orientation_reference_world_y_rad':float(lean),'foot_positions_m':{side:data.xpos[model.body(body).id].tolist() for side,body in [('left','foot_l'),('right','foot_r')]},'wrench':wrench.tolist(),'teacher_coefficient':blend,'teacher_target_contribution_rad':(blend*(target-teacher.nominal)[action_aids]).tolist(),'student_target_contribution_rad':((1-blend)*(student_target-teacher.nominal)[action_aids]).tolist(),'motor_balance_requested_offset_rad':motor_balance_offset[action_aids].tolist(),'motor_balance_load_fraction':load_fraction.tolist(),'motor_balance_requested_reaction_nm':motor_reaction,'teacher_tracking_integral_offset_rad':tracking_integral[action_aids].tolist(),'motor_reference_positions_rad':motor_reference.tolist() if motor_reference is not None else None,'motor_targets':data.ctrl[action_aids].tolist(),'motor_actual':data.qpos[action_qids].tolist(),'motor_torques_nm':data.actuator_force[action_aids].tolist()})
            states.append(data.qpos.copy())
        if fall:break
    info=runtime.close();wall=time.perf_counter()-begin
    clearance_runs=liftoffs
    liftoffs=sustained_liftoffs(rows)
    np.savez_compressed(out/'trajectory.npz',qpos=np.array(states),time=np.array([r['time_s'] for r in rows]))
    np.savez_compressed(out/'demonstrations.npz',observations=np.array([x[0] for x in demonstrations]),actions=np.array([x[1] for x in demonstrations]),action_scale=.5)
    np.savez_compressed(out/'auxiliary_wrench.npz',wrench=np.array(wrenches),dt=.002,coefficient=args.coefficient)
    metrics={'duration_s':data.time,'forward_m':float(data.xpos[pelvis,1]-initial[1]),'strafe_m':float(data.xpos[pelvis,0]-initial[0]),'fall':fall,'reset_count':0,'done_count':0,'liftoffs':liftoffs,'clearance_run_counts_before_contact_latch':clearance_runs,'landings':landings,'maximum_clearance_m':maxclear,'both_air_fraction':bothair/max(len(rows),1),'mean_contact_slip_mps':float(np.mean(slip)),'max_joint_speed_rad_s':maxspeed,'max_effort_fraction':maxtorque,'peak_motor_tracking_error_rad':peak_tracking,'nonfoot_contacts':sorted(nonfeet),'max_abs_wrench_components':np.max(np.abs(wrenches),axis=0).tolist()}
    result={'lineage':LINEAGE,'created_at':datetime.now(timezone.utc).isoformat(),'milestone_pass':False,'scope':'assisted teacher development, never zero-assistance gate evidence','reproduction_argv':__import__('sys').argv,'config_sha256':hashlib.sha256(json.dumps(dict(vars(args),motor_pose_sequence_sha256=reference_sha),sort_keys=True).encode()).hexdigest(),'config':dict(vars(args),motor_pose_sequence_sha256=reference_sha),'student_checkpoint_sha256':digest(args.student) if args.student else None,'teacher_blend_coefficient':blend,'balance_assistance_coefficient':args.coefficient,'teacher_targets_evaluated':blend>0,'source_sha256':digest(__file__),'model_xml_sha256':hashlib.sha256(xml.encode()).hexdigest(),'audit':audit,'backend':info,'startup_including_ik_s':startup,'simulation_and_diagnostics_wall_s':wall,'physics_steps_per_wall_s':len(wrenches)/wall,'ik_max_residual':teacher.ik_errors,'metrics':metrics,'samples':rows,'assisted_dynamics_candidate':bool(not fall and not nonfeet and metrics['forward_m']>.2 and min(liftoffs.values())>=3 and maxspeed<=4 and metrics['both_air_fraction']<.02 and metrics['mean_contact_slip_mps']<.05)}
    result['teacher_blend_schedule']=None if rescue_time is None else {'kind':'diagnostic_handback','initial':initial_blend,'final':1.,'start_s':rescue_time,'ramp_s':.25}
    metrics.update(ground_load_sampling_hz=500,both_feet_unloaded_physics_fraction=unsupported_steps/len(wrenches),
        maximum_both_feet_unloaded_duration_s=max_unsupported_run*.002,
        mean_support_body_weight_ratio=support_sum/len(wrenches),peak_support_body_weight_ratio=support_peak)
    if unsupported_steps/len(wrenches)>=.02 or not .5<=metrics['mean_support_body_weight_ratio']<=1.5 or support_peak>3:
        result['assisted_dynamics_candidate']=False
    result['student_startup_clock_s']=student.startup_clock_s if student is not None else 0.
    result['diagnostic_oracle_student']=oracle
    result['diagnostic_observation_replay_sha256']=replay_sha
    result['student_observation_source']='saved_teacher_reference' if replay is not None else 'actual_simulated_state'
    result['diagnostic_replay_block']=args.diagnostic_replay_block if replay is not None else None
    result['total_teacher_guidance_coefficient']=1. if oracle or replay is not None else blend
    if rescue_time is not None or oracle or replay is not None:result['assisted_dynamics_candidate']=False
    write_json(out/'result.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ('samples','audit','backend')},indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--name',required=True)
    p.add_argument('--phase-delay',type=float,default=.5,help='Student phase-clock origin; set to the motor-reference loop start for a one-time startup prefix')
    p.add_argument('--teacher-tracking-integral',type=float,default=0.,help='Training teacher joint-error integral gain, bounded0.08rad and original target limits; disabled with teacher blend0')
    p.add_argument('--motor-pose-sequence',help='Audited motor-only pose sequence JSON; never applies scratch root poses')
    p.add_argument('--teacher-blend',type=float,help='Diagnostic override only; guardedcurriculum uses coupled coefficient')
    p.add_argument('--teacher-rescue-time',type=float,help='Diagnostic only: smoothly return full motor teacher control at this time; never eligible as a curriculum or demonstration pass')
    p.add_argument('--diagnostic-oracle-student',action='store_true',help='Timing diagnostic only: replace student branch with current teacher target sampled and held at50Hz; total teacher guidance remains1')
    p.add_argument('--diagnostic-observation-replay',help='Diagnostic only: actor sees saved teacher observations; explicit reference forcing, never curriculum or distillation evidence')
    p.add_argument('--diagnostic-replay-block',choices=['all','joint_position'],default='all',help='Isolate joint-position feedback while keeping other observations live')
    p.add_argument('--student');p.add_argument('--seconds',type=float,default=12.);p.add_argument('--coefficient',type=float,default=1.)
    p.add_argument('--period',type=float,default=2.);p.add_argument('--stride',type=float,default=.06)
    p.add_argument('--root-lean-amplitude',type=float,default=0.)
    p.add_argument('--foot-placement-gain',type=float,default=0.)
    p.add_argument('--lateral-assistance-scale',type=float,default=1.)
    p.add_argument('--hip-roll-amplitude',type=float,default=0.)
    p.add_argument('--waist-amplitude',type=float,default=.18)
    p.add_argument('--motor-balance-scale',type=float,default=1.,help='Diagnostic motor feedback strength; scales0.25Nm reaction cap and0.02rad offset cap, original actuator limits retained')
    p.add_argument('--motor-balance-gain',type=float,default=0.,help='Contact-gated hip motor balance: signed Nm/rad (negative tests reverse correction), bounded0.25Nm requested reaction and0.02rad target offset')
    p.add_argument('--waist-phase-lead',type=float,default=0.,help='Waist motor transfer phase advance in radians; foot timing unchanged')
    p.add_argument('--waist-transfer-sharpness',type=float,default=0.,help='0 gives sine; positive values sustain waist transfer during single support')
    p.add_argument('--orientation-damping',type=float,default=.3)
    p.add_argument('--clearance',type=float,default=.025);p.add_argument('--height-gain',type=float,default=1000.)
    args=p.parse_args()
    if not 0<=args.phase_delay<=120:raise ValueError('Invalid phase delay')
    if not 0<=args.teacher_tracking_integral<=2 or not -np.pi<=args.waist_phase_lead<=np.pi or not 0<=args.motor_balance_scale<=4 or not -12<=args.motor_balance_gain<=12 or not 0<=args.waist_transfer_sharpness<=3 or (args.teacher_blend is not None and not 0<=args.teacher_blend<=1) or not 0<=args.coefficient<=1 or not 0<=args.lateral_assistance_scale<=1 or not 0<args.seconds<=120:raise ValueError('Invalid bounds')
    run(args)

if __name__=='__main__':main()
