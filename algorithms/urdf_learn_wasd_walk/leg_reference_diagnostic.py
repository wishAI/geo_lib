"""CPU/Warp comparison of an IK support-transfer diagnostic.

This is a mechanics experiment, not a trained policy or milestone validator.
The live state is never assigned after initialization. Scratch IK sets only
19 position-motor targets; fingers remain at the saved fist targets.
"""
from pathlib import Path
import argparse
import hashlib
import json
import time
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation


def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--model-dir',required=True,type=Path)
 p.add_argument('--output',required=True,type=Path)
 p.add_argument('--backend',choices=('native','warp'),default='warp')
 p.add_argument('--duration',type=float,default=80)
 p.add_argument('--step',type=float,default=.02)
 p.add_argument('--ff',type=float,default=1)
 p.add_argument('--yaw-frequency',type=float,choices=(0.,5.),default=0.)
 p.add_argument('--heading-ik',action='store_true',help='Correct measured heading through the whole leg IK chain')
 p.add_argument('--dynamic-com',action='store_true',help='Bounded COM preview and supported inverse dynamics after the entry hold')
 p.add_argument('--double-support',type=float,default=.2)
 p.add_argument('--entry-snapshot',type=Path,help='Initialize a diagnostic at a recorded40s entry; never a no-reset certificate')
 p.add_argument('--normal-force-feedback',action='store_true',help='Bounded smooth foot-height correction from measured contact load')
 p.add_argument('--cycle',type=float,default=36.,help='Seconds per repeated footstep after the initial support transfer')
 p.add_argument('--crouch',type=float,default=.012,help='COM lowering during the first swing, metres')
 a=p.parse_args()
 if not 12<=a.duration<=120 or not .01<=a.step<=.06 or a.ff!=1:raise ValueError('Bounded diagnostic limits exceeded')
 if not .5<=a.cycle<=36 or not .012<=a.crouch<=.035:raise ValueError('Bounded timing/crouch limits exceeded')
 if a.heading_ik and a.yaw_frequency:raise ValueError('Compare heading controllers separately')
 if a.dynamic_com and (a.step>.04 or a.cycle>9):raise ValueError('Outside researched dynamic geometry/timing')
 if not .2<=a.double_support<=.5:raise ValueError('Double-support fraction outside diagnostic limits')
 if a.normal_force_feedback and not a.dynamic_com:raise ValueError('Normal-force comparison requires the dynamic reference')
 if 'outputs' not in a.output.resolve().parts:raise ValueError('Diagnostic must stay in outputs')
 a.output.mkdir(parents=True,exist_ok=False)
 xml=(a.model_dir/'model.xml').read_bytes()
 if hashlib.sha256(xml).hexdigest()!='83e755398195838c68003613b151c5be2bfbeb65eadaeec1598e508ec43edca6':raise ValueError('Exact current model required')
 meta=json.loads((a.model_dir/'metadata.json').read_text())
 m=mujoco.MjModel.from_xml_path(str(a.model_dir/'model.xml'))
 names=meta['action_joints'];jids=[m.joint(n).id for n in names]
 qa=m.jnt_qposadr[jids];va=m.jnt_dofadr[jids];aid=[m.actuator(n).id for n in names]
 nominal=np.array(meta['nominal_q']);ctrl=np.array(meta['nominal_ctrl'])
 base=m.body('base_link').id;pelvis=m.body('root_x').id
 (a.output/'source.py').write_text(Path(__file__).read_text())
 (a.output/'model.xml').write_bytes(xml)
 feet=[m.body('foot_'+s).id for s in ['l','r']];geoms=[];sides={}
 for si,s in enumerate(['l','r']):
  for bn in ['foot_'+s,'toes_01_'+s]:
   b=m.body(bn).id
   for g in range(m.body_geomadr[b],m.body_geomadr[b]+m.body_geomnum[b]):
    mesh=m.geom_dataid[g];begin=m.mesh_vertadr[mesh];verts=m.mesh_vert[begin:begin+m.mesh_vertnum[mesh]].copy();geoms.append((si,g,verts));sides[g]=si

 contact_rows=[];time_offset=0.
 def points(d):
  out=[[],[]]
  for si,g,v in geoms:out[si].append(v@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g])
  return [np.concatenate(x) for x in out]
 def clearance(d):return np.array([x[:,2].min() for x in points(d)])
 def forces(d):
  normal=np.zeros(2);cop=np.zeros((2,3));f=np.zeros(6)
  for i in range(d.ncon):
   c=d.contact[i]
   if min(c.geom)!=0:continue
   g=max(c.geom)
   if g not in sides:continue
   mujoco.mj_contactForce(m,d,i,f);w=max(f[0],0);si=sides[g];normal[si]+=w;cop[si]+=w*c.pos
   # Actual solver contact frame and wrench; never reconstructed by CPU dynamics.
   contact_rows.append(np.r_[time_offset+d.time,si,c.geom,c.pos,c.frame,f,c.dist])
  return normal,cop/np.maximum(normal[:,None],1e-12)
 def init():
  d=mujoco.MjData(m);d.qpos[:]=nominal;d.ctrl[:]=ctrl;mujoco.mj_forward(m,d)
  return d

 sim=None
 def advance(data):
  if sim is not None:sim.step(data)
  else:mujoco.mj_step(m,data);mujoco.mj_forward(m,data)
 def smooth(t,x,y):
  s=np.clip((t-x)/(y-x),0,1);return s*s*(3-2*s)
 d=init();s=mujoco.MjData(m);b=mujoco.MjData(m)
 com0=d.subtree_com[base].copy();f0=d.xpos[feet].copy();flat0=d.xmat[feet].reshape(2,3,3).copy()
 legnames=[side+'_'+j+'_joint' for side in ['left','right'] for j in ['hip_pitch','hip_roll','hip_yaw','knee','ankle_pitch','ankle_roll','shin_roll']];ids=np.r_[np.arange(6),m.jnt_dofadr[[m.joint(n).id for n in legnames]]];kp=m.actuator_gainprm[aid,0]
 qhold=None
 full=[];metrics=[];planner=[];air=np.zeros(2);peak=np.zeros(2);swings=np.zeros(2,int);maxclear=np.zeros(2);lastfeet=None;slip=0.;slipcount=0;maxerr=0
 dynamic=False;dynamic_plan=None;last_reference=None;last_velocity=np.zeros(m.nv);dynamic_cache={};wrench_trace=[]
 last_com=None;last_com_velocity=np.zeros(3);last_com_acceleration=np.zeros(3)
 heading_position=0.;heading_velocity=0.
 force_sum=np.zeros(2);force_samples=0;normal_position=np.zeros(2);normal_velocity=np.zeros(2);normal_trace=[]
 kd=-m.actuator_biasprm[aid,2]
 entry_keys=('fhold','chold','Rhold','rhold','qhold','qik','com0','f0','flat0',
             'last_reference','last_velocity','last_com','last_com_velocity','last_com_acceleration')
 entry_signature={name:getattr(a,name) for name in ('step','crouch','heading_ik','dynamic_com','yaw_frequency')}
 if a.entry_snapshot:
  entry_meta=json.loads(a.entry_snapshot.with_suffix('.json').read_text())
  if entry_meta['signature']!=entry_signature or entry_meta['model_sha256']!=hashlib.sha256(xml).hexdigest():raise ValueError('Entry model/controller initialization mismatch')
  if entry_meta['snapshot_sha256']!=hashlib.sha256(a.entry_snapshot.read_bytes()).hexdigest():raise ValueError('Entry snapshot hash mismatch')
  with np.load(a.entry_snapshot,allow_pickle=False) as snap:
   fhold,chold,Rhold,rhold,qhold,qik,com0,f0,flat0,last_reference,last_velocity,last_com,last_com_velocity,last_com_acceleration=(snap[name].copy() for name in entry_keys)
   d.qpos[:]=snap['qpos'];d.qvel[:]=snap['qvel'];d.ctrl[:]=snap['ctrl'];mujoco.mj_forward(m,d);d.qacc_warmstart[:]=snap['qacc_warmstart']
   heading_position=float(snap['heading_position']);heading_velocity=float(snap['heading_velocity'])
   if 'force_sum' in snap:
    force_sum=snap['force_sum'].copy();force_samples=int(snap['force_samples'])
   elif a.normal_force_feedback:raise ValueError('Force-feedback comparison requires recorded force history')
  time_offset=float(entry_meta['time'])
  if time_offset!=40. or a.duration<=time_offset:raise ValueError('Only40s diagnostic entry snapshots supported')
  (a.output/'entry_initialization.json').write_text(json.dumps(entry_meta,indent=2)+'\n')
 entry_saved=bool(a.entry_snapshot)

 def gettau(data):
  if dynamic:
   from .leg_com_reference import contact_wrench_map
   velocity=np.zeros(m.nv)
   mujoco.mj_differentiatePos(m,velocity,.02,last_reference,data.qpos)
   data.qvel[:]=velocity;mujoco.mj_forward(m,data)
   data.qacc[:]=(velocity-last_velocity)/.02
   inverse=np.zeros(m.nv);mujoco.mj_rne(m,data,1,inverse)
   jacp=[];jacr=[]
   for si,body in enumerate(feet):
    jp=np.zeros((3,m.nv));jr=jp.copy();mujoco.mj_jac(m,data,jp,jr,planned_cops[si],body);jacp.append(jp);jacr.append(jr)
   mapping,force_map,moment_map=contact_wrench_map(data.qpos[:3],data.xmat[base].reshape(3,3),planned_cops,planned_weights,jacp,jacr)
   tau=inverse-mapping@inverse[:6]
   mass_matrix=np.zeros((m.nv,m.nv));mujoco.mj_fullM(m,mass_matrix,data.qM)
   dynamic_cache.update(velocity=velocity,mass=mass_matrix-mapping@mass_matrix[:6],root_residual=tau[:6].copy(),
                        force=force_map@inverse[:6],moment=moment_map@inverse[:6])
   return tau
  tau=data.qfrc_bias.copy();com=data.subtree_com[base];f=data.xpos[feet];w=np.clip((com[0]-f[1,0])/(f[0,0]-f[1,0]),0,1)
  for si,body in enumerate(feet):
   cop=np.array([f[si,0],f[si,1]+com[1]-(w*f[0,1]+(1-w)*f[1,1]),0]);jp=np.zeros((3,m.nv));mujoco.mj_jac(m,data,jp,None,cop,body);tau-=jp.T@np.array([0,0,18*9.81*([w,1-w][si])])
  return tau
 if a.backend=='warp':
  from .mujoco_warp_runtime import WarpEvaluation
  sim=WarpEvaluation(m,d,a.output)
 try:
  for i in range(round((a.duration-time_offset)/.002)):
   t=time_offset+i*.002;tr=t;phase=max(tr-2,0);half=int(phase//16);local=phase%16;sign=1 if half%2==0 else -1;end=.14 if sign>0 else -.12;start=0 if half==0 else -.12 if sign>0 else .14;shift=start+(end-start)*smooth(local,0,6);lift=.65*smooth(local,8,10)*(1-smooth(local,12,14)) if tr>2 else 0
   if i%10==0:
    if t>=40 and not entry_saved:
     values=(fhold,chold,Rhold,rhold,qhold,qik,com0,f0,flat0,last_reference,last_velocity,last_com,last_com_velocity,last_com_acceleration)
     entry_path=a.output/'entry_state.npz'
     np.savez_compressed(entry_path,**dict(zip(entry_keys,values)),qpos=d.qpos,qvel=d.qvel,ctrl=d.ctrl,qacc_warmstart=d.qacc_warmstart,heading_position=heading_position,heading_velocity=heading_velocity,force_sum=force_sum,force_samples=force_samples)
     entry_meta=dict(time=40.,signature=entry_signature,model_sha256=hashlib.sha256(xml).hexdigest(),snapshot_sha256=hashlib.sha256(entry_path.read_bytes()).hexdigest(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),diagnostic_only=True)
     entry_path.with_suffix('.json').write_text(json.dumps(entry_meta,indent=2)+'\n');entry_saved=True
    averaged_force=force_sum/max(force_samples,1);force_sum=np.zeros(2);force_samples=0
    if t<12:
     R=d.xmat[base].reshape(3,3);roll=np.arctan2(R[2,0],R[2,2]);pitch=np.arctan2(R[2,1],R[2,2]);f=smooth(t,1,2);d.ctrl[:]=ctrl
     for side in ['left','right']:
      d.ctrl[m.actuator(side+'_ankle_roll_joint').id]+=shift-f*(roll-.0035-shift)+.0125*f*d.qvel[4]
      d.ctrl[m.actuator(side+'_ankle_pitch_joint').id]+=.3*f*(pitch+.0411)+.03*f*d.qvel[3]
     for joint,value in [('hip_pitch',-.5*lift),('knee',lift),('ankle_pitch',-.5*lift)]:d.ctrl[m.actuator('left_'+joint+'_joint').id]+=value
     com=d.subtree_com[base].copy();fp=d.xpos[feet].copy()
    else:
     if qhold is None:
      fhold=d.xpos[feet].copy();chold=d.subtree_com[base].copy();clearhold=clearance(d);Rhold=d.xmat[base].reshape(3,3).copy();rhold=d.xmat[feet].reshape(2,3,3).copy();qhold=d.qpos.copy();qik=qhold.copy();b.qpos[:]=qhold;b.qvel[:]=0;mujoco.mj_forward(m,b);tau=gettau(b);target=qhold[qa]+tau[va]/kp;print('switch',d.time,'dtarget',np.round(target-d.ctrl[aid],4),flush=True)
     fp=fhold.copy();com=chold.copy();br=Rhold.copy()
     flatten=smooth(t,14,18);fp[0,1]-=a.step*flatten
     yawhold=Rotation.from_matrix(rhold[0]).as_euler('xyz')[2];yawflat=Rotation.from_matrix(flat0[0]).as_euler('xyz')[2];flatrot=Rotation.from_rotvec([0,0,yawhold-yawflat]).as_matrix()@flat0[0]
     footrots=rhold.copy();footrots[0]=Rotation.from_rotvec(flatten*Rotation.from_matrix(flatrot@rhold[0].T).as_rotvec()).as_matrix()@rhold[0]
     fp[0,2]+=(f0[0,2]+.032-fhold[0,2])*flatten-(.032+.0001)*smooth(t,18,22)
     com[2]-=a.crouch*smooth(t,14,18)
     transfer=smooth(t,28,40);com[0]+=(fhold[:,0].mean()-chold[0])*transfer;com[1]+=.5*(fp[0,1]-fp[1,1])*transfer
     br=Rotation.from_rotvec([0,.14*transfer,0]).as_matrix()@Rhold
     if t>=40:
      copoffset=chold-fhold[1];fp=fhold.copy();fp[0,1]-=a.step;fp[0,2]=f0[0,2]-.0001
      footrots=rhold.copy();footrots[0]=flatrot
      cycle=int((t-40)//a.cycle);u=(t-40-cycle*a.cycle)*(36/a.cycle)
      for completed in range(cycle):
       swing=(completed+1)%2;stance=1-swing;fp[swing,1]=fp[stance,1]-a.step
      swing=(cycle+1)%2;stance=1-swing
      footstart=fp.copy();com[:2]=[fp[:,0].mean(),fp[:,1].mean()+copoffset[1]]
      supportcom=fp[stance,:2]+copoffset[:2]
      shiftfull=smooth(u,0,12);com[:2]+=(supportcom-com[:2])*shiftfull
      angle=.14+(.14 if stance==0 else -.14)*shiftfull
      liftnew=.03*smooth(u,12,16)*(1-smooth(u,20,24));fp[swing,2]+=liftnew
      fp[swing,1]+=(fp[stance,1]-a.step-fp[swing,1])*smooth(u,16,20)
      shiftnext=smooth(u,24,36);endcom=np.array([fp[:,0].mean(),fp[:,1].mean()+copoffset[1]])
      com[:2]+=(endcom-com[:2])*shiftnext;angle+=(.14-angle)*shiftnext
      br=Rotation.from_rotvec([0,angle,0]).as_matrix()@Rhold
     if a.dynamic_com and t>=40:
      from .leg_com_reference import ComReference
      if dynamic_plan is None:
       dynamic_plan=ComReference(initial_feet=fp,initial_com=com,cop_offset=copoffset,
                                 step=a.step,cycle=a.cycle,horizon=max(a.duration-40+4,8),
                                 initial_velocity=last_com_velocity,initial_acceleration=last_com_acceleration,
                                 double_support=a.double_support)
      fp,planned_weights,com,com_velocity,com_acceleration=dynamic_plan.sample(t-40)
      if a.normal_force_feedback:
       from .leg_com_reference import normal_force_reference
       normal_position,normal_velocity=normal_force_reference(normal_position,normal_velocity,averaged_force,planned_weights*m.body_mass.sum()*9.81,body_weight=m.body_mass.sum()*9.81)
       fp[:,2]+=normal_position
       normal_trace.append(np.r_[t,normal_position,averaged_force,planned_weights])
      planned_cops=np.c_[fp[:,:2]+copoffset[:2],np.zeros(2)]
      br=Rotation.from_rotvec([0,.14,0]).as_matrix()@Rhold
      dynamic=True
     br_ik=br
     if a.heading_ik:
      # Keep br as the nominal world heading. Correct through all leg axes,
      # including passive shin compliance, rather than applying a hip-only
      # torque that omits the opposing shin component of a ground yaw wrench.
      actual_rotation=d.xmat[base].reshape(3,3)
      heading_error=Rotation.from_matrix(actual_rotation@br.T).as_rotvec()[2]
      heading_correction=-heading_error
      if a.dynamic_com:
       from .leg_com_reference import filter_heading
       heading_position,heading_velocity=filter_heading(heading_position,heading_velocity,heading_correction)
       heading_correction=heading_position
      br_ik=Rotation.from_rotvec([0.,0.,heading_correction]).as_matrix()@br
     s.qpos[:]=qik;s.qvel[:]=0
     for it in range(20):
      mujoco.mj_forward(m,s);es=[];js=[]
      for si,body in enumerate(feet):
       jp=np.zeros((3,m.nv));jr=jp.copy();mujoco.mj_jacBody(m,s,jp,jr,body);es.extend([fp[si]-s.xpos[body],.2*Rotation.from_matrix(footrots[si]@s.xmat[body].reshape(3,3).T).as_rotvec()]);js.extend([jp[:,ids],.2*jr[:,ids]])
      jc=np.zeros((3,m.nv));mujoco.mj_jacSubtreeCom(m,s,jc,base);es.append(com-s.subtree_com[base]);js.append(jc[:,ids]);jp=np.zeros((3,m.nv));jr=jp.copy();mujoco.mj_jacBody(m,s,jp,jr,base);es.append(.2*Rotation.from_matrix(br_ik@s.xmat[base].reshape(3,3).T).as_rotvec());js.append(.2*jr[:,ids]);
      taustatic=gettau(s)
      for side in ['left','right']:
       ji=m.joint(side+'_shin_roll_joint').id;vi=m.jnt_dofadr[ji];qi=m.jnt_qposadr[ji];ac=m.actuator(side+'_shin_roll_joint').id
       damping=-m.actuator_biasprm[ac,2] if dynamic else 0.
       es.append(np.array([ctrl[ac]-(taustatic[vi]+damping*s.qvel[vi])/m.actuator_gainprm[ac,0]-s.qpos[qi]]));row=np.zeros((1,len(ids)));row[0,np.where(ids==vi)[0][0]]=1
       if dynamic:
        row[0]+=dynamic_cache['mass'][vi,ids]/(.02**2*m.actuator_gainprm[ac,0])
        row[0,np.where(ids==vi)[0][0]]+=damping/(.02*m.actuator_gainprm[ac,0])
       js.append(row)
      e=np.concatenate(es);J=np.vstack(js)
      if abs(e).max()<1e-7:break
      v=np.zeros(m.nv);v[ids]=np.clip(np.linalg.lstsq(J,e,rcond=1e-6)[0],-.1,.1);mujoco.mj_integratePos(m,s.qpos,v,1)
     maxerr=max(maxerr,abs(e).max());qik=s.qpos.copy();mujoco.mj_forward(m,s);target=qik[qa]+gettau(s)[va]/kp
     reference_velocity=np.zeros(m.nv)
     if last_reference is not None:mujoco.mj_differentiatePos(m,reference_velocity,.02,last_reference,qik)
     if dynamic:
      target+=kd*reference_velocity[va]/kp
      wrench_trace.append(np.r_[t,planned_weights,dynamic_cache['force'],dynamic_cache['moment'],dynamic_cache['root_residual'],planned_cops.ravel()])
     last_reference=qik.copy();last_velocity=reference_velocity
     current_com_velocity=np.zeros(3) if last_com is None else (com-last_com)/.02
     last_com_acceleration=(current_com_velocity-last_com_velocity)/.02
     last_com=com.copy();last_com_velocity=current_com_velocity
     planner.append(np.r_[t,target,br.ravel(),com,fp.ravel(),br_ik.ravel()])
     d.ctrl[aid]=target
     R=d.xmat[base].reshape(3,3);roll=np.arctan2(R[2,0],R[2,2]);pitch=np.arctan2(R[2,1],R[2,2]);roll0=np.arctan2(br[2,0],br[2,2]);pitch0=np.arctan2(br[2,1],br[2,2])
     for side in ['left','right']:
      d.ctrl[m.actuator(side+'_ankle_roll_joint').id]+=-(roll-roll0)+.0125*d.qvel[4]
      d.ctrl[m.actuator(side+'_ankle_pitch_joint').id]+=.3*(pitch-pitch0)+.03*d.qvel[3]
     if a.yaw_frequency:
      # A world-yaw error is invisible to projected-gravity roll/pitch feedback.
      # Convert a critically damped centroidal yaw correction to hip PD targets.
      mass_matrix=np.zeros((m.nv,m.nv));mujoco.mj_fullM(m,mass_matrix,s.qM)
      vertical_local=s.xmat[base].reshape(3,3).T@np.array([0.,0.,1.])
      delta=s.subtree_com[base]-s.qpos[:3]
      inertia=float(vertical_local@mass_matrix[3:6,3:6]@vertical_local-m.body_mass.sum()*np.dot(delta[:2],delta[:2]))
      if not np.isfinite(inertia) or inertia<=0:raise ValueError('Invalid centroidal yaw inertia')
      yaw_error=Rotation.from_matrix(R@br.T).as_rotvec()[2]
      yaw_rate=(R@d.qvel[3:6])[2]
      moment=inertia*(a.yaw_frequency**2*yaw_error+2*a.yaw_frequency*yaw_rate)
      left_weight=np.clip((com[0]-fp[1,0])/(fp[0,0]-fp[1,0]),0,1)
      for si,side in enumerate(['left','right']):
       ji=m.joint(side+'_hip_yaw_joint').id;ac=m.actuator(side+'_hip_yaw_joint').id
       d.ctrl[ac]+=[left_weight,1-left_weight][si]*moment*s.xaxis[ji,2]/m.actuator_gainprm[ac,0]
   advance(d);force,cop=forces(d);force_sum+=force;force_samples+=1;clear=clearance(d);maxclear=np.maximum(maxclear,clear);contact=force>.1
   if i==1000 and not time_offset:com0=d.subtree_com[base].copy();f0=d.xpos[feet].copy();flat0=d.xmat[feet].reshape(2,3,3).copy()
   for si in range(2):
    if contact[si]:
     if air[si]>=.06 and peak[si]>=.015:swings[si]+=1
     air[si]=0;peak[si]=0
     if lastfeet is not None:slip+=np.linalg.norm((d.xpos[feet[si]]-lastfeet[si])[:2]);slipcount+=1
    else:air[si]+=.002;peak[si]=max(peak[si],clear[si])
   lastfeet=d.xpos[feet].copy();tilt=np.arccos(np.clip(d.xmat[base].reshape(3,3)[2,2],-1,1));full.append(np.r_[time_offset+d.time,d.qpos,d.qvel,d.ctrl]);metrics.append(np.r_[time_offset+d.time,force,clear,d.subtree_com[base],d.xpos[feet].ravel(),com,fp.ravel(),cop.ravel()])
   if i%1000==0:print('trace',round(time_offset+d.time,2),np.round(d.subtree_com[base],4),np.round(force,2),np.round(clear,4),round(tilt,3),flush=True)
   if tilt>.5236 or d.xpos[pelvis,2]<.25:break
 finally:
  warp_info=sim.close() if sim is not None else None

 states=np.array(full);quaternions=states[:,4:8]
 headings=np.unwrap(np.arctan2(2*(quaternions[:,0]*quaternions[:,3]+quaternions[:,1]*quaternions[:,2]),1-2*(quaternions[:,2]**2+quaternions[:,3]**2)))
 motors=states[:,-m.nu:];speeds=states[:,1+m.nq+6:1+m.nq+m.nv]
 result=dict(diagnostic_only=True,certifiable=False,backend=a.backend,seconds=time_offset+d.time,
  initialization='recorded_entry_state' if a.entry_snapshot else 'nominal_standing',start_time_s=time_offset,simulated_seconds=d.time,
  planner_clock='integer physics-step count; live simulator clock remains independently audited',
  normal_force_feedback=a.normal_force_feedback,
  yaw_frequency_rad_s=a.yaw_frequency,heading_ik=a.heading_ik,dynamic_com=a.dynamic_com,cycle_s=a.cycle,double_support_fraction=a.double_support,crouch_m=a.crouch,max_heading_deviation_rad=float(abs(headings-headings[0]).max()),
  forward_m=float(com0[1]-d.subtree_com[base,1]),foot_forward_m=(f0[:,1]-d.xpos[feet,1]).tolist(),
  completed_swings=swings.tolist(),peak_clearance_m=maxclear.tolist(),final_tilt_rad=float(tilt),
  max_ik_residual=float(maxerr),mean_contact_foot_center_slip_mps=float(slip/max(slipcount*.002,1e-9)),
  max_joint_speed_rad_s=float(abs(speeds).max()),
  max_control_limit_violation_rad=float(np.maximum(0,np.maximum(m.actuator_ctrlrange[:,0]-motors,motors-m.actuator_ctrlrange[:,1])).max()),
  root_state_interventions=0,external_force_interventions=0,control_hz=50,physics_hz=500,
  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),model_sha256=hashlib.sha256(xml).hexdigest(),
  metadata_sha256=hashlib.sha256((a.model_dir/'metadata.json').read_bytes()).hexdigest(),
  nq=m.nq,nv=m.nv,nu=m.nu,warp=warp_info)
 if a.dynamic_com:
  from . import leg_com_reference
  result['com_reference_source_sha256']=hashlib.sha256(Path(leg_com_reference.__file__).read_bytes()).hexdigest()
  (a.output/'com_reference_source.py').write_text(Path(leg_com_reference.__file__).read_text())
  result['planned_wrench_feasibility_review_required']=True
  result['heading_reference_frequency_rad_s']=2. if a.heading_ik else None
 result['contact_columns']=['time','side','geom0','geom1','x','y','z']+[f'frame_{i}' for i in range(9)]+['force_normal','force_tangent1','force_tangent2','torque_normal','torque_tangent1','torque_tangent2','distance']
 result['metrics_cop_columns']=[23,24,25,26,27,28]
 np.savez_compressed(a.output/'trajectory.npz',full=states,metrics=np.array(metrics),planner=np.array(planner),planned_wrenches=np.array(wrench_trace),contacts=np.array(contact_rows),normal_feedback=np.array(normal_trace),nq=m.nq,nv=m.nv,nu=m.nu)
 result['trajectory_sha256']=hashlib.sha256((a.output/'trajectory.npz').read_bytes()).hexdigest()
 (a.output/'diagnostic.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps(result),flush=True)


if __name__=='__main__':main()
