"""Diagnostic motor-only replay of a hash-bound offline gait retarget.

The optimized root poses supply feedback references only. No root position,
velocity, wrench, contact, mass or simulator state is assigned by this controller.
This is not a trained policy or a certifiable checkpoint family.
"""
import json
from pathlib import Path


class PhaseRetarget:
    def __init__(self, path, metadata, period=1.8, ramp=3., feedback=.5, phase_start=0., startup_hold=False, velocity_feedforward=False):
        import numpy as np
        from scipy.interpolate import PchipInterpolator
        from scipy.spatial.transform import Rotation
        from .mujoco_backend import digest
        self.path=Path(path).resolve()
        report=json.loads(self.path.read_text())
        names=metadata['gait_action_joints']
        if (report.get('kind')!='offline_m7_phase_retarget_feasibility' or
            report.get('certifiable') is not False or report['action_joints']!=names or
            report['target_model_xml_sha256']!=metadata['model_xml_sha256'] or
            not report.get('interpolated_within_action_and_joint_bounds')):
            raise ValueError('Retarget model, joints or feasibility contract differs')
        if not 1.2<=period<=4. or not 1.<=ramp<=8. or not 0<=feedback<=1. or not 0<=phase_start<1:
            raise ValueError('Retarget diagnostic settings outside bounds')
        donor=Path(report['source_checkpoint'])
        if digest(donor)!=report['source_checkpoint_sha256']:
            raise ValueError('Retarget donor changed')
        reference=Path(report['target_checkpoint'])
        target=json.loads((reference.parent/'training.json').read_text())
        if (digest(reference)!=report['target_checkpoint_sha256'] or
            digest(reference.parent/'training.json')!=report['target_training_sha256'] or
            any(target[key]!=metadata[key] for key in ('nominal_q','nominal_ctrl','model_xml_sha256','gait_action_joints'))):
            raise ValueError('Retarget nominal pose or target checkpoint changed')
        phase=np.asarray(report['phase']);offsets=np.asarray(report['joint_offsets_rad'])
        root=Rotation.from_rotvec(np.asarray(report['root_rotvec'])).as_matrix()
        references=np.stack((np.arctan2(root[:,2,1],root[:,2,2]),
                             np.arctan2(root[:,2,0],root[:,2,2])),axis=1)
        values=np.concatenate((offsets,references),axis=1)
        if (phase.shape!=(32,) or offsets.shape!=(32,19) or not np.isfinite(values).all() or
            not np.allclose(phase,(np.arange(32)+.5)/32)):
            raise ValueError('Invalid retarget samples')
        curve=PchipInterpolator(np.r_[phase-1,phase,phase+1],np.tile(values,(3,1)))
        self.knots=curve.x;self.coefficients=curve.c;self.names=names
        self.period=period;self.ramp=ramp;self.feedback=feedback
        self.phase_start=phase_start;self.startup_hold=startup_hold
        self.velocity_gain=None
        if velocity_feedforward:
            import mujoco
            model_path=Path(report['target_model_xml'])
            if digest(model_path)!=report['target_model_xml_sha256']:
                raise ValueError('Retarget actuator model changed')
            model=mujoco.MjModel.from_xml_path(str(model_path))
            gains=[]
            for name in names:
                matches=np.flatnonzero(model.actuator_trnid[:,0]==model.joint(name).id)
                if len(matches)!=1:raise ValueError('Expected one position actuator per joint')
                i=int(matches[0]);kp=float(model.actuator_gainprm[i,0]);kv=-float(model.actuator_biasprm[i,2])
                if (kp<=0 or kv<0 or model.actuator_gaintype[i]!=0 or model.actuator_biastype[i]!=1 or
                    not np.allclose(model.actuator_gear[i],[1,0,0,0,0,0]) or
                    not np.allclose(model.actuator_biasprm[i,:2],[0,-kp])):
                    raise ValueError('Unsupported motor law for velocity feedforward')
                gains.append(kv/kp)
            self.velocity_gain=gains
        self.contract=dict(template_sha256=digest(self.path),source_sha256=digest(__file__),
            donor_checkpoint=str(donor),donor_sha256=digest(donor),period_s=period,
            target_checkpoint=str(reference),target_checkpoint_sha256=digest(reference),
            target_training_sha256=report['target_training_sha256'],
            velocity_feedforward=velocity_feedforward,velocity_gain_s=self.velocity_gain,
            startup_ramp_s=ramp,feedback_scale=feedback,phase_start=phase_start,startup_hold=startup_hold,diagnostic_only=True,
            root_control='none; root orientation is a joint-feedback reference only',
            inherited_milestone_passes=0)

    def sample(self, phase, derivative=False):
        import torch
        knots=torch.as_tensor(self.knots,device=phase.device,dtype=phase.dtype)
        coefficients=torch.as_tensor(self.coefficients,device=phase.device,dtype=phase.dtype)
        phase=phase.remainder(1)
        index=torch.searchsorted(knots,phase,right=True)-1
        delta=(phase-knots[index])[:,None]
        if derivative:return (3*coefficients[0,index]*delta+2*coefficients[1,index])*delta+coefficients[2,index]
        return ((coefficients[0,index]*delta+coefficients[1,index])*delta+coefficients[2,index])*delta+coefficients[3,index]

    def gait_action(self, observation, parameters, names):
        import torch
        if names!=self.names or observation.shape[1]!=76:
            raise ValueError('Retarget observation layout differs')
        age=(1-observation[:,68])*30
        phase=(age-(1+self.ramp if self.startup_hold else 2)).clamp_min(0)/self.period+self.phase_start
        values=self.sample(phase)
        ramp_fraction=((age-1)/self.ramp).clamp(0,1)
        activation=ramp_fraction.square()*(3-2*ramp_fraction)
        offsets=values[:,:19]*activation[:,None]
        if self.velocity_gain is not None:
            ramp_velocity=6*ramp_fraction*(1-ramp_fraction)/self.ramp
            phase_velocity=(age>(1+self.ramp if self.startup_hold else 2)).to(age.dtype)/self.period
            desired_velocity=values[:,:19]*ramp_velocity[:,None]+self.sample(phase,derivative=True)[:,:19]*(activation*phase_velocity)[:,None]
            # Current MuJoCo position motors produce kp*(ctrl-q)-kv*qdot.
            # Adding kv/kp*v_des yields ordinary position+velocity PD tracking.
            # https://mujoco.readthedocs.io/en/3.5.0/XMLreference.html#actuator-position
            offsets+=desired_velocity*offsets.new_tensor(self.velocity_gain)
        pitch=torch.atan2(-observation[:,7],-observation[:,8])
        roll=torch.atan2(-observation[:,6],-observation[:,8])
        pitch_error=pitch-values[:,19]*activation
        roll_error=roll-values[:,20]*activation
        p=parameters.expand(len(observation),-1)
        ankle=p[:,5]*pitch_error+p[:,6]*observation[:,3]
        lateral=-p[:,16]*roll_error+p[:,17]*observation[:,4]
        heading=torch.atan2(observation[:,74],observation[:,75]+1)
        yaw=p[:,12]*heading+p[:,13]*observation[:,5]
        for side in ('left','right'):
            for joint,value in (('ankle_pitch',ankle),('ankle_roll',lateral),('hip_yaw',yaw)):
                offsets[:,names.index(side+'_'+joint+'_joint')]+=self.feedback*activation*value
        return offsets/.08
