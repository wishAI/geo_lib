"""Batched CUDA Landau training physics; evaluations use WarpEvaluation instead.

Source geometry and 69 physical PD motors are unchanged. Seventeen residual
actions and all PPO semantics match the native backend. Training-only pelvis
assistance is updated at 50 Hz and held as a bounded wrench for ten steps.
"""
from collections import deque
import os
import time

import mujoco
import mujoco_warp as mjwarp
import numpy as np
import torch
import warp as wp

from algorithms.urdf_learn_wasd_walk.mujoco_backend import (
    Assistance, OUTPUT, audit_model, build_model, initialize,
)


@wp.kernel
def record_occupancy(pairs:wp.array(dtype=int),contacts:wp.array(dtype=int),
                     constraints:wp.array(dtype=int),maximum:wp.array(dtype=int)):
    world=wp.tid()
    if world==0:
        wp.atomic_max(maximum,0,pairs[0])
        wp.atomic_max(maximum,1,contacts[0])
    wp.atomic_max(maximum,2,constraints[world])


class WarpBatch:
    def __init__(self,n,seed,coefficient,stage='stand',gait_reward='single',contact_timeconst=.004,forward_speed=.4,forward_tracking_variance=.25):
        self.stage=stage; self.gait_reward=gait_reward; self.n=n
        self.forward_tracking_variance=forward_tracking_variance
        self.assistance=Assistance(coefficient); self.completed=deque(maxlen=20)
        self.completed_since_update=0; self.wrench_trace=[]
        self.max_force=self.max_torque=self.force_impulse=self.torque_impulse=0.
        self.total_resets=0; self.gait_active_steps=0; self.walking_steps=0
        self.maximum_clearance=0.; self.maximum_joint_speed=0.
        self.model,self.spec,self.xml=build_model(noslip_iterations=0,contact_timeconst=contact_timeconst)
        self.audit=audit_model(self.model,self.spec)
        template=initialize(self.model,self.spec,pose='geometric')
        self.nominal_q=template.qpos.copy(); self.nominal_ctrl=template.ctrl.copy()
        self.jids=[self.model.joint(k).id for k in self.spec['action_joints']]
        self.aids=[self.model.actuator(k).id for k in self.spec['action_joints']]
        self.base=self.model.body('base_link').id; self.pelvis=self.model.body('root_x').id
        os.environ['TMPDIR']='/tmp'
        wp.config.kernel_cache_dir=str(OUTPUT/'warp_training_cache')
        wp.init(); wp.set_device('cuda:0')
        if not wp.get_device().is_cuda or not torch.cuda.is_available():
            raise RuntimeError('CUDA physics and CUDA policy tensors are required')
        flags=int(self.model.opt.disableflags)
        try:
            self.model.opt.disableflags=flags & ~int(mujoco.mjtDisableBit.mjDSBL_AUTORESET)
            self.wm=mjwarp.put_model(self.model)
        finally: self.model.opt.disableflags=flags
        t=time.perf_counter()
        warm=mjwarp.put_data(self.model,template,nworld=n,nconmax=128,njmax=512)
        mjwarp.step(self.wm,warm); mjwarp.forward(self.wm,warm); wp.synchronize()
        del warm
        self.wd=mjwarp.put_data(self.model,template,nworld=n,nconmax=128,njmax=512)
        self.maximum_occupancy=wp.zeros(3,dtype=int,device='cuda:0')
        occupancy_args=[self.wd.ncollision,self.wd.nacon,self.wd.nefc,self.maximum_occupancy]
        wp.launch(record_occupancy,dim=n,inputs=occupancy_args)
        self.compile_s=time.perf_counter()-t
        with wp.ScopedCapture() as capture:
            for _ in range(10):
                mjwarp.step(self.wm,self.wd)
                wp.launch(record_occupancy,dim=n,inputs=occupancy_args)
            mjwarp.forward(self.wm,self.wd)
            wp.launch(record_occupancy,dim=n,inputs=occupancy_args)
        self.graph=capture.graph
        with wp.ScopedCapture() as capture: mjwarp.forward(self.wm,self.wd)
        self.forward_graph=capture.graph
        self.q=wp.to_torch(self.wd.qpos); self.v=wp.to_torch(self.wd.qvel)
        self.ctrl=wp.to_torch(self.wd.ctrl); self.wrench=wp.to_torch(self.wd.xfrc_applied)
        self.pos=wp.to_torch(self.wd.xpos); self.rot=wp.to_torch(self.wd.xmat)
        self.quat=wp.to_torch(self.wd.xquat); self.cvel=wp.to_torch(self.wd.cvel)
        self.com=wp.to_torch(self.wd.subtree_com)
        self.jq=torch.as_tensor(self.model.jnt_qposadr[self.jids],device='cuda',dtype=torch.long)
        self.jv=torch.as_tensor(self.model.jnt_dofadr[self.jids],device='cuda',dtype=torch.long)
        self.nominal=torch.tensor(self.nominal_q,device='cuda',dtype=torch.float32)
        self.targets=torch.tensor(self.nominal_ctrl,device='cuda',dtype=torch.float32)
        self.reference_xy=torch.tensor(template.xpos[self.pelvis,:2],device='cuda',dtype=torch.float32)
        self.reference_height=float(template.xpos[self.pelvis,2])
        self.reference_quat=torch.tensor(template.xquat[self.pelvis],device='cuda',dtype=torch.float32)
        self.previous=torch.zeros((n,17),device='cuda'); self.episode_steps=torch.zeros(n,device='cuda',dtype=torch.long)
        self.air_steps=torch.zeros((n,2),device='cuda',dtype=torch.long)
        self.touched=torch.zeros((n,2),device='cuda',dtype=torch.bool)
        self.total_liftoffs=torch.zeros(2,device='cuda',dtype=torch.long)
        self.commands=torch.tensor([0. if stage=='stand' or i%4==0 else forward_speed for i in range(n)],device='cuda')
        self.scale=torch.where(self.commands!=0,.24,.08)[:,None]
        self.foot_geoms=[]
        for side,suffix in ((0,'l'),(1,'r')):
            for body in (f'foot_{suffix}',f'toes_01_{suffix}'):
                b=self.model.body(body).id
                for g in range(self.model.body_geomadr[b],self.model.body_geomadr[b]+self.model.body_geomnum[b]):
                    mesh=self.model.geom_dataid[g]; begin=self.model.mesh_vertadr[mesh]; count=self.model.mesh_vertnum[mesh]
                    vertices=torch.tensor(self.model.mesh_vert[begin:begin+count],device='cuda')
                    self.foot_geoms.append((side,g,vertices))
        self.geom_body=torch.tensor(self.model.geom_bodyid,device='cuda',dtype=torch.long)
        self.contact_slots=torch.arange(self.wd.naconmax,device='cuda')
        self.contact_sides=torch.full((self.model.ngeom,),-1,device='cuda',dtype=torch.long)
        for side,g,_ in self.foot_geoms:self.contact_sides[g]=side
        self.reset(torch.ones(n,device='cuda',dtype=torch.bool))

    def reset(self,mask):
        count=int(mask.sum())
        if not count:return
        self.q[mask]=self.nominal; self.v[mask]=0
        wp.to_torch(self.wd.qacc_warmstart)[mask]=0
        wp.to_torch(self.wd.time)[mask]=0
        self.wrench[mask]=0; wp.to_torch(self.wd.qfrc_applied)[mask]=0
        noise=torch.zeros((count,self.model.nq),device='cuda')
        noise[:,self.jq]=torch.rand((count,17),device='cuda')*.004-.002
        self.q[mask]+=noise
        low=torch.tensor(self.model.jnt_range[self.jids,0],device='cuda',dtype=torch.float32)
        high=torch.tensor(self.model.jnt_range[self.jids,1],device='cuda',dtype=torch.float32)
        selected=self.q[mask]; selected[:,self.jq]=selected[:,self.jq].clamp(low,high);self.q[mask]=selected
        self.ctrl[mask]=self.targets; self.previous[mask]=0; self.episode_steps[mask]=0
        self.air_steps[mask]=0;self.touched[mask]=False
        torch.cuda.synchronize()
        wp.capture_launch(self.forward_graph)
        wp.synchronize()

    def local_velocity(self):
        r=self.rot[:,self.base]
        linear=torch.bmm(r.transpose(1,2),self.v[:,:3,None]).squeeze(-1)
        return linear,self.v[:,3:6]

    def observations(self):
        linear,angular=self.local_velocity()
        obs=torch.cat((linear,angular,-self.rot[:,self.base,2,:],
            self.q[:,self.jq]-self.nominal[self.jq],.1*self.v[:,self.jv],self.previous),dim=1)
        if self.stage=='forward':
            phase=2*torch.pi*self.episode_steps*.02
            extra=torch.stack((self.commands,torch.zeros_like(phase),torch.zeros_like(phase),phase.sin(),phase.cos()),dim=1)
            extra[self.commands==0]=0;obs=torch.cat((obs,extra),dim=1)
        return obs

    def assistance_wrench(self):
        self.wrench.zero_()
        if not self.assistance.coefficient:return
        angular=torch.bmm(self.rot[:,self.base],self.v[:,3:6,None]).squeeze(-1)
        velocity=self.v[:,:3]+torch.cross(angular,self.pos[:,self.pelvis]-self.q[:,:3],dim=1)
        current=self.quat[:,self.pelvis];target=self.reference_quat
        # target * conjugate(current), shortest world-frame rotation vector.
        w=target[0]*current[:,0]+(target[1:]*current[:,1:]).sum(-1)
        xyz=-target[0]*current[:,1:]+current[:,0,None]*target[1:]-torch.cross(target[1:].expand(self.n,3),current[:,1:],dim=1)
        xyz*=torch.where(w<0,-1.,1.)[:,None]
        length=xyz.norm(dim=1)
        error=xyz*(2*torch.atan2(length,w.abs())/length.clamp_min(1e-8))[:,None]
        torque=3*error-.3*angular
        torque*=torch.minimum(torch.ones_like(length),1/torque.norm(dim=1).clamp_min(1e-8))[:,None]
        self.wrench[:,self.pelvis,2]=(80*(self.reference_height-self.pos[:,self.pelvis,2])-8*velocity[:,2]).clamp(-9,9)
        self.wrench[:,self.pelvis,3:]=torque
        self.wrench*=self.assistance.coefficient
        actual=self.wrench[:,self.pelvis]
        force=actual[:,:3].norm(dim=1); torque=actual[:,3:].norm(dim=1)
        self.max_force=max(self.max_force,float(force.max()));self.max_torque=max(self.max_torque,float(torque.max()))
        self.force_impulse+=float(force.sum())*.02;self.torque_impulse+=float(torque.sum())*.02
        row=actual[0].tolist(); base=float(self.episode_steps[0])*.02
        for k in range(10):self.wrench_trace.append([base+k*.002,self.assistance.coefficient,*row])

    def feet(self,with_forces=False):
        con=self.wd.contact; geom=wp.to_torch(con.geom).long(); wid=wp.to_torch(con.worldid).long()
        address=wp.to_torch(con.efc_address)[:,0].long()
        g=geom.max(dim=1).values.clamp(0,self.model.ngeom-1)
        side=self.contact_sides[g]
        valid=(self.contact_slots<wp.to_torch(self.wd.nacon)[0]) & (geom.min(dim=1).values==0) & (side>=0) & (address>=0)
        safe_world=wid.clamp(0,self.n-1);safe_address=address.clamp(0,self.wd.njmax-1)
        normal=torch.where(valid,wp.to_torch(self.wd.efc.force)[safe_world,safe_address].clamp_min(0),0.)
        forces=torch.zeros(self.n*2,device='cuda')
        forces.scatter_add_(0,safe_world*2+side.clamp_min(0),normal)
        contacts=forces.reshape(self.n,2)>.1
        body=self.geom_body[g];spatial=self.cvel[safe_world,body]
        point=wp.to_torch(con.pos)-self.com[safe_world,self.base]
        speed=spatial[:,3:]+torch.cross(spatial[:,:3],point,dim=1)
        usable=valid & (normal>.05)
        sums=torch.zeros(self.n,device='cuda');counts=torch.zeros_like(sums)
        sums.scatter_add_(0,safe_world,torch.where(usable,speed[:,:2].norm(dim=1),0.))
        counts.scatter_add_(0,safe_world,usable.float())
        slip=sums/counts.clamp_min(1)
        clearance=torch.full((self.n,2),float('inf'),device='cuda')
        if self.gait_reward in ('phase','load'):
            grot=wp.to_torch(self.wd.geom_xmat);gpos=wp.to_torch(self.wd.geom_xpos)
            for side,g,vertices in self.foot_geoms:
                height=(vertices@grot[:,g,2,:].T).T.min(dim=1).values+gpos[:,g,2]
                clearance[:,side]=torch.minimum(clearance[:,side],height)
        result=(contacts,slip,clearance)
        return (*result,forces.reshape(self.n,2)) if with_forces else result

    def step(self,actions):
        actions=actions.clamp(-1,1)
        self.ctrl[:]=self.targets;self.ctrl[:,self.aids]+=self.scale*actions
        self.assistance_wrench()
        # Warp owns a capturable stream. Explicit boundaries order Torch's
        # default stream writes and physics; do not capture the null stream.
        torch.cuda.synchronize()
        wp.capture_launch(self.graph)
        wp.synchronize()
        self.episode_steps+=1
        pairs,contacts,constraints=self.maximum_occupancy.numpy()
        if pairs>=self.wd.naconmax or contacts>=self.wd.naconmax or constraints>=self.wd.njmax:
            raise RuntimeError('Training contact/constraint capacity exceeded')
        tilt=self.rot[:,self.base,2,2].clamp(-1,1).acos()
        drift=(self.pos[:,self.pelvis,:2]-self.reference_xy).norm(dim=1)
        fallen=(tilt>torch.pi/6) | (self.pos[:,self.pelvis,2]<self.reference_height-.08)
        invalid=~torch.isfinite(self.q).all(dim=1) | ~torch.isfinite(self.v).all(dim=1)
        timeout=self.episode_steps>=1500;done=fallen | invalid | timeout
        reward=1-4*tilt.square()-8*drift.square()-.02*self.v.square().mean(-1)-.02*actions.square().mean(-1)-.01*(actions-self.previous).square().mean(-1)
        if self.stage=='forward':
            linear,angular=self.local_velocity();contacts,slip,clearance,forces=self.feet(with_forces=True)
            track=(-(linear[:,1]-self.commands).square()/self.forward_tracking_variance-linear[:,0].square()/.25).exp()
            gait=(contacts[:,0]!=contacts[:,1]).float()
            if self.gait_reward in ('phase','load'):
                phase=self.episode_steps.float()*.02%1
                left_stance=phase>=.5
                correct=(contacts[:,0]==left_stance)&(contacts[:,1]!=left_stance)
                swing_clear=torch.where(left_stance,clearance[:,1],clearance[:,0])
                target=.012*(2*torch.pi*phase).sin().square()
                gait=correct*((-((swing_clear-target)/.008).square()).exp())
                if self.gait_reward=='load':
                    from algorithms.urdf_learn_wasd_walk.mujoco_policy import phase_load_score
                    supported_weight=(float(self.model.body_mass.sum())*9.81-self.wrench[:,self.pelvis,2]).clamp_min(1.)
                    gait=phase_load_score(phase,forces,clearance,supported_weight)
                self.maximum_clearance=max(self.maximum_clearance,float(clearance.max()))
                self.touched|=contacts
                airborne=(~contacts)&(clearance>.002)&self.touched
                self.air_steps=torch.where(airborne,self.air_steps+1,0)
                self.total_liftoffs+=((self.air_steps==2)&(self.commands!=0)[:,None]).sum(dim=0)
            walk=1+2*track+linear[:,1]-.25*angular[:,2].square()-2*tilt.square()+.25*gait-.15*slip-.02*actions.square().mean(-1)-.01*(actions-self.previous).square().mean(-1)
            reward=torch.where(self.commands!=0,walk,reward)
            moving=self.commands!=0
            self.last_diagnostics={'moving_forward_speed_mps':linear[moving,1].mean(),
                'moving_lateral_speed_abs_mps':linear[moving,0].abs().mean(),
                'moving_tracking_score':track[moving].mean(),'moving_gait_score':gait[moving].mean(),
                'moving_tilt_rad':tilt[moving].mean(),'moving_fall_fraction':fallen[moving].float().mean()} if self.n>1 else {}
            self.gait_active_steps+=int(((gait>0)&(self.commands!=0)).sum());self.walking_steps+=int((self.commands!=0).sum())
        reward=torch.where(fallen|invalid,-5.,reward)
        self.maximum_joint_speed=max(self.maximum_joint_speed,float(self.v[:,self.jv].abs().max()))
        self.previous=actions.clone()
        if done.any():
            success=timeout & ~fallen & ~invalid
            displacement=self.pos[:,self.pelvis,1]-self.reference_xy[1]
            success&=torch.where(self.commands!=0,displacement>=5,drift<.03)
            self.completed.extend(success[done].tolist());count=int(done.sum())
            self.completed_since_update+=count;self.total_resets+=count
            self.reset(done)
        return self.observations(),reward,done.float(),(timeout&~fallen&~invalid).float()
