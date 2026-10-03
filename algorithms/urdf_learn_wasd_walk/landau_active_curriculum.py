"""Mixed recorded-state active-turn curriculum; never nominal-start evidence.

Reference-state initialization and heading preservation follow the general
imitation/task combination in https://arxiv.org/abs/1804.02717, without pose replay.
"""
import json
import math
from pathlib import Path
import numpy as np
import torch
from algorithms.urdf_learn_wasd_walk.landau_cruise_residual import CruiseMemory


class ActiveMemory(CruiseMemory):
    """Vector memory for left-turn/first-stop episodes, before any restart."""
    def __init__(self,n,device):
        super().__init__(n,device)
        self.previous_moving=torch.ones(n,device=device,dtype=torch.bool)
        self.stop_xy=torch.zeros((n,2),device=device)
        self.stop_time=torch.full((n,),float('inf'),device=device,dtype=torch.float64)
        self.hold_heading=torch.zeros(n,device=device)

    def advance_command(self,observation,positions,age,elapsed):
        command=observation[:,63:66]
        moving=command.abs().amax(1)>0
        stopped=~moving & self.previous_moving
        if bool((moving & ~self.previous_moving & torch.isfinite(self.stop_time)).any()):
            raise ValueError('Active curriculum must end before restart')
        self.stop_xy[stopped]=positions[stopped,:2]
        self.stop_time[stopped]=age[stopped]
        self.hold_heading[stopped]=torch.atan2(observation[stopped,68],observation[stopped,69]+1.)
        self._reference+=self.previous_yaw*elapsed
        # All bank rows have already turned left and no restart. Match the full
        # scalar predicate; zero yaw during deceleration/holding is not cruise.
        cruising=(command[:,2]==0.) & ((command[:,0]-.2).abs()<1e-7) & (command[:,1]==0.)
        self.left_cruise_start=torch.where(cruising,torch.minimum(self.left_cruise_start,age),float('inf'))
        target=(cruising & (age-self.left_cruise_start>=.2-1e-9)).double()
        self._blend+=(target-self._blend).clamp(-elapsed,elapsed)
        self._feedback+=((command[:,2]>0.).double()-self._feedback).clamp(-elapsed,elapsed)
        self.previous_yaw=command[:,2].double();self.previous_moving=moving

    def prior(self,observation,positions,rotations,age):
        result=observation[:,:63].clone()
        held=~self.previous_moving & torch.isfinite(self.stop_time)
        delta=torch.zeros_like(positions);delta[:,:2]=positions[:,:2]-self.stop_xy
        local=torch.bmm(rotations.transpose(1,2),delta[:,:,None])[:,:2,0]/.03
        result[held,60:62]=local[held]
        result[held,62]=(1.-(age[held]-self.stop_time[held])/30.).float()
        return result


def commands(mode,age,displacement,heading,turn_duration):
    from algorithms.urdf_learn_wasd_walk.landau_direction_training import commands as pursuit
    clock=torch.round(age/.02)*.02
    command=pursuit('backward',displacement,heading,100.)
    end=3.+turn_duration
    fade=((clock-end)/2.).clamp(0,1);fade=fade.square()*(3.-2.*fade)
    m5=torch.stack((.2*(1.-fade),torch.zeros_like(age),
        ((clock>=3.)&(clock<end)).double()*math.pi/(2.*turn_duration)),1)
    fade=((clock-18.)/2.).clamp(0,1);fade=fade.square()*(3.-2.*fade)
    forward=.2*(1.-fade)
    restart=((clock-25.)/2.).clamp(0,1);restart=restart.square()*(3.-2.*restart)
    forward=torch.where(clock>=25.,.2*restart,forward)
    last=((clock-48.)/2.).clamp(0,1);last=last.square()*(3.-2.*last)
    forward=torch.where(clock>=48.,.2*(1.-last),forward)
    m6=torch.stack((forward,torch.zeros_like(age),
        (((clock>=6.)&(clock<18.)).double()-((clock>=32.)&(clock<44.)).double())*math.pi/88.),1)
    command=torch.where((mode==1)[:,None],m5,command)
    return torch.where((mode==2)[:,None],m6,command)


def build_bank(folders,parent,cruise,turn,output):
    from algorithms.urdf_learn_wasd_walk import mujoco_backend as backend
    metadata=json.loads((Path(parent).parent/'training.json').read_text())
    rows=[];sources=[];teachers=[]
    # Fixed 0.2 s sampling and balanced protocol sampling during resets. The
    # two adjacent partitions are correlated and never called independent tests.
    for mode,(folder,start,upper,checkpoint) in enumerate(zip(folders,(3.,3.,6.),(84.8,49.8,18.8),(parent,cruise,cruise))):
        folder=Path(folder).resolve();result=json.loads((folder/'dynamics.json').read_text());cfg=result['config']
        if cfg['checkpoint_sha256']!=backend.digest(checkpoint):raise ValueError('Active bank checkpoint mismatch')
        if (cfg['forward']!=.2 or cfg['assistance']!=0. or cfg['dt']!=.002 or
            cfg['contact_timeconst']!=.004 or cfg['gain_scale']!=1.):raise ValueError('Active bank physics mismatch')
        if (mode==0 and cfg.get('direction')!='backward') or (mode==1 and not cfg.get('turn')) or (mode==2 and not cfg.get('teleop')):
            raise ValueError('Active bank protocol mismatch')
        for key in ('backend','mujoco_version','model_xml_sha256','urdf_sha256','mesh_tree_sha256'):
            if result['identity'][key]!=metadata[key]:raise ValueError('Active bank model mismatch: '+key)
        if any(result['metrics'][key]!=0 for key in ('fall_count','reset_count','done_count')):raise ValueError('Fallen active bank source')
        for filename,key in (('foot_trace.json','foot_trace_sha256'),('policy_trace.npz','policy_trace_sha256'),('trajectory.npz','trajectory_sha256')):
            if backend.digest(folder/filename)!=result[key]:raise ValueError('Active bank source changed')
        feet=json.loads((folder/'foot_trace.json').read_text())
        first_bad=min((r['time_s'] for r in feet if r['support_body_weight_ratio']>3. or r['max_joint_speed_rad_s']>4.+1e-6),default=float('inf'))
        if first_bad<=upper+.02:raise ValueError('Unsafe active curriculum initialization')
        trajectory=np.load(folder/'trajectory.npz');trace=np.load(folder/'policy_trace.npz')
        if not np.allclose(trajectory['time'][:-1],trace['time'],rtol=0,atol=1e-6):raise ValueError('Active bank clocks differ')
        positions={round(r['time_s']/.02):r['pelvis_position_m'] for r in json.loads((folder/'trace.json').read_text()) if 'pelvis_position_m' in r}
        heading=np.unwrap(np.arctan2(trace['observation'][:,68],trace['observation'][:,69]+1.))
        teachers.append(heading)
        memory=turn.CommandMemory()
        for i,(age,obs) in enumerate(zip(trace['time'],trace['observation'])):
            xyz=torch.tensor([positions.get(i,[0.,0.,0.])],dtype=torch.float32)
            memory.observe(torch.from_numpy(obs)[None],xyz,torch.eye(3)[None],float(age))
            if start-1e-6<=age<=upper+1e-6 and i%10==0:
                if memory.last_yaw_sign!=1 or memory.restart_time is not None:raise ValueError('Unsupported bank memory')
                rows.append(dict(qpos=trajectory['qpos'][i],qvel=trajectory['qvel'][i],previous=obs[43:60],age=age,
                    reference=memory.reference,blend=memory.left_cruise_blend,feedback=memory.left_feedback_blend,
                    cruise_start=memory.left_cruise_start if memory.left_cruise_start is not None else float('inf'),
                    previous_yaw=float(obs[65]),mode=mode,previous_moving=memory.previous_moving,
                    stop_xy=memory.anchor_xy[0].numpy() if memory.anchor_xy is not None else np.zeros(2),
                    stop_time=memory.anchor_time if memory.anchor_time is not None else float('inf'),
                    hold_heading=heading[round(memory.anchor_time/.02)] if memory.anchor_time is not None else 0.,
                    source_index=i))
        sources.append(dict(directory=str(folder),mode=mode,count=sum(r['mode']==mode for r in rows),start_s=start,upper_s=upper,
            first_unsafe_force_or_speed_s=first_bad if math.isfinite(first_bad) else None,
            hashes={n:backend.digest(folder/n) for n in ('trajectory.npz','policy_trace.npz','foot_trace.json','trace.json','dynamics.json')}))
    bank={key:np.array([r[key] for r in rows]) for key in rows[0]}
    length=max(map(len,teachers));bank['teacher_heading']=np.array([np.pad(h,(0,length-len(h)),mode='edge') for h in teachers])
    np.savez_compressed(output,**bank)
    return dict(sources=sources,count=len(rows),protocol_sampling='uniform among three; even within-protocol indices train, odd reserved correlated screen',
        limitations='Recorded-state training resets with cleared solver warmstart; no nominal-start or exact continuation claim.')
