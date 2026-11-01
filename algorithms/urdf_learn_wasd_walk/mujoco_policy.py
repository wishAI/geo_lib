"""Bounded Landau stand PPO and training-only pelvis assistance experiments.

Structure reference: official unitree_rl_mjlab G1 PPO (1425b15f), 24-step
rollouts, clipped PPO, GAE(0.99,0.95), five epochs, four minibatches, ELU MLP.
This small CPU adaptation is not the official G1 task or a locomotion result.
"""
from __future__ import annotations
import argparse
from collections import deque
from datetime import datetime, timezone
import hashlib
import json
import os
import psutil
from pathlib import Path
import shlex
import sys
import threading
import time

import mujoco
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal
from scipy.spatial.transform import Rotation

from algorithms.urdf_learn_wasd_walk import model_spec
from algorithms.urdf_learn_wasd_walk.mujoco_backend import Assistance, OUTPUT, audit_model, build_model, digest, initialize, write_json

ACTION_SCALE = .08


class ActorCritic(nn.Module):
    def __init__(self, observation_dim=60, forward_exploration_multiplier=1., mean_activation='linear'):
        super().__init__()
        self.forward_exploration_multiplier=forward_exploration_multiplier
        self.observation_dim=observation_dim
        def mlp(out):
            return nn.Sequential(nn.Linear(observation_dim,128), nn.ELU(), nn.Linear(128,128), nn.ELU(), nn.Linear(128,out))
        self.actor, self.critic = mlp(17), mlp(1)
        self.log_std = nn.Parameter(torch.full((17,), -2.3))
        nn.init.orthogonal_(self.actor[-1].weight, gain=.01)
        nn.init.zeros_(self.actor[-1].bias)
        if mean_activation not in ('linear','tanh'):
            raise ValueError('Unknown policy mean activation')
        self.mean_activation=mean_activation
        if mean_activation=='tanh':
            # Bound the Gaussian mean, retaining its correctly scored sampling
            # distribution and the existing physical action limits.
            self.actor.append(nn.Tanh())

    def distribution(self, obs):
        sigma=self.log_std.clamp(-5,0).exp()
        if self.observation_dim==65:
            multiplier=torch.where(obs[...,60]!=0,self.forward_exploration_multiplier,1.)
            sigma=sigma*multiplier[...,None]
        return Normal(self.actor(obs),sigma)


def observe(model, data, q_nominal, action_joint_ids, previous_action):
    jids = np.asarray(action_joint_ids)
    qadr, dadr = model.jnt_qposadr[jids], model.jnt_dofadr[jids]
    base = model.body('base_link').id
    rot = data.xmat[base].reshape(3,3)
    vel = np.zeros(6)
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, base, vel, 1)
    return np.concatenate([vel[3:], vel[:3], rot.T @ [0,0,-1], data.qpos[qadr]-q_nominal[qadr],
                           .1*data.qvel[dadr], previous_action]).astype(np.float32)


def foot_state(model, data, *, clearance=False, diagnostics=False):
    forces = {'left':0.,'right':0.}; slips=[]
    weighted_velocity={side:0. for side in forces};weights={side:0. for side in forces}
    weighted_position={side:np.zeros(3) for side in forces}
    force=np.zeros(6)
    for index in range(data.ncon):
        contact=data.contact[index]
        ids=list(contact.geom)
        if 0 not in ids: continue
        geom=max(ids); name=model.body(model.geom_bodyid[geom]).name
        side='left' if name in {'foot_l','toes_01_l'} else 'right' if name in {'foot_r','toes_01_r'} else None
        if side is None: continue
        mujoco.mj_contactForce(model,data,index,force)
        forces[side]+=max(float(force[0]),0.)
        if force[0]>.05:
            velocity=np.zeros(6)
            mujoco.mj_objectVelocity(model,data,mujoco.mjtObj.mjOBJ_GEOM,int(geom),velocity,0)
            point_velocity=velocity[3:]+np.cross(velocity[:3],contact.pos-data.geom_xpos[geom])
            slips.append(float(np.linalg.norm(point_velocity[:2])))
            if diagnostics:
                weights[side]+=float(force[0])
                weighted_velocity[side]+=float(force[0]*point_velocity[2])
                weighted_position[side]+=float(force[0])*contact.pos
    result = {'left_contact':forces['left']>.1,'right_contact':forces['right']>.1,
              'mean_slip_mps':float(np.mean(slips)) if slips else 0.}
    if diagnostics:
        for side in forces:
            result[side+'_normal_force_N']=forces[side]
            result[side+'_contact_vertical_velocity_mps']=weighted_velocity[side]/weights[side] if weights[side] else None
            result[side+'_contact_centroid_world_m']=(weighted_position[side]/weights[side]).tolist() if weights[side] else None
    if clearance:
        for side,suffix in (('left','l'),('right','r')):
            heights=[]
            for body in (f'foot_{suffix}',f'toes_01_{suffix}'):
                b=model.body(body).id
                for geom in range(model.body_geomadr[b],model.body_geomadr[b]+model.body_geomnum[b]):
                    mesh=model.geom_dataid[geom]
                    begin=model.mesh_vertadr[mesh]; count=model.mesh_vertnum[mesh]
                    vertices=model.mesh_vert[begin:begin+count]
                    heights.append(float(np.min(vertices @ data.geom_xmat[geom].reshape(3,3)[2])+data.geom_xpos[geom,2]))
            result[side+'_clearance_m']=min(heights)
    return result


class StandBatch:
    def __init__(self, n, seed, coefficient, stage="stand", gait_reward="single", forward_speed=.4, forward_tracking_variance=.25):
        self.gait_reward=gait_reward
        self.forward_tracking_variance=forward_tracking_variance
        self.stage = stage
        self.commands = np.array([0. if i % 4 == 0 or stage == "stand" else forward_speed for i in range(n)])
        self.wrench_trace = []
        self.model, self.spec, self.xml = build_model(noslip_iterations=20)
        self.audit = audit_model(self.model, self.spec)
        template = initialize(self.model, self.spec, pose='geometric')
        self.nominal_q = template.qpos.copy()
        self.nominal_ctrl = template.ctrl.copy()
        self.jids = [self.model.joint(name).id for name in self.spec['action_joints']]
        self.aids = [self.model.actuator(name).id for name in self.spec['action_joints']]
        self.data = [mujoco.MjData(self.model) for _ in range(n)]
        self.rng = np.random.default_rng(seed)
        self.previous = np.zeros((n,17), dtype=np.float32)
        self.episode_steps = np.zeros(n,dtype=int)
        self.assistance = Assistance(coefficient)
        self.pelvis = self.model.body('root_x').id
        self.base = self.model.body('base_link').id
        self.reference_rotation = template.xmat[self.pelvis].reshape(3,3).copy()
        self.reference_height = template.xpos[self.pelvis,2]
        self.reference_xy = template.xpos[self.pelvis,:2].copy()
        self.completed = deque(maxlen=20)
        self.completed_since_update = 0
        self.max_force = self.max_torque = 0.
        self.force_impulse = self.torque_impulse = 0.
        for i in range(n): self.reset(i)

    def reset(self,i):
        d = self.data[i]
        mujoco.mj_resetData(self.model,d)
        d.qpos[:] = self.nominal_q
        # Training randomization only; physical joint perturbations at episode initialization.
        for jid in self.jids:
            adr = self.model.jnt_qposadr[jid]
            low, high = self.model.jnt_range[jid]
            d.qpos[adr] = np.clip(d.qpos[adr]+self.rng.uniform(-.002,.002),low,high)
        d.ctrl[:] = self.nominal_ctrl
        self.previous[i] = 0
        self.episode_steps[i] = 0
        mujoco.mj_forward(self.model,d)

    def observations(self):
        rows = []
        for i,d in enumerate(self.data):
            observation = observe(self.model,d,self.nominal_q,self.jids,self.previous[i])
            if self.stage == 'forward':
                phase = 2*np.pi*d.time
                extra = [self.commands[i], 0, 0, np.sin(phase), np.cos(phase)] if self.commands[i] else [0.]*5
                observation = np.concatenate([observation, extra]).astype(np.float32)
            rows.append(observation)
        return np.stack(rows)

    def step(self, actions):
        rewards, dones, timeouts = [], [], []
        for i,d in enumerate(self.data):
            action = np.clip(actions[i],-1,1)
            d.ctrl[:] = self.nominal_ctrl
            d.ctrl[self.aids] += (.24 if self.stage=='forward' and self.commands[i] else ACTION_SCALE)*action
            for _ in range(10):
                d.xfrc_applied[:] = 0
                wrench = np.zeros(6)
                if self.assistance.coefficient:
                    vel = np.zeros(6)
                    mujoco.mj_objectVelocity(self.model,d,mujoco.mjtObj.mjOBJ_BODY,self.pelvis,vel,0)
                    err = Rotation.from_matrix(self.reference_rotation @ d.xmat[self.pelvis].reshape(3,3).T).as_rotvec()
                    wrench = self.assistance.wrench(self.reference_height-d.xpos[self.pelvis,2],vel[5],err,vel[:3])
                    d.xfrc_applied[self.pelvis] = wrench
                    f,t = np.linalg.norm(wrench[:3]),np.linalg.norm(wrench[3:])
                    self.max_force=max(self.max_force,float(f)); self.max_torque=max(self.max_torque,float(t))
                    self.force_impulse += f*.002; self.torque_impulse += t*.002
                if self.assistance.coefficient and i == 0:
                    self.wrench_trace.append([float(d.time), self.assistance.coefficient, *wrench.tolist()])
                mujoco.mj_step(self.model,d)
            mujoco.mj_forward(self.model,d)
            tilt = np.arccos(np.clip(d.xmat[self.base].reshape(3,3)[2,2],-1,1))
            drift = np.linalg.norm(d.xpos[self.pelvis,:2]-self.reference_xy)
            fallen = tilt > np.pi/6 or d.xpos[self.pelvis,2] < self.reference_height-.08
            invalid = not np.isfinite(d.qpos).all() or any(w.number for w in d.warning)
            self.episode_steps[i] += 1
            timeout = self.episode_steps[i] >= 1500
            done = fallen or invalid or timeout
            reward = 1.-4*tilt**2-8*drift**2-.02*np.mean(d.qvel**2)-.02*np.mean(action**2)-.01*np.mean((action-self.previous[i])**2)
            if self.stage == 'forward' and self.commands[i]:
                velocity = np.zeros(6)
                mujoco.mj_objectVelocity(self.model,d,mujoco.mjtObj.mjOBJ_BODY,self.base,velocity,1)
                contact = foot_state(self.model, d, clearance=self.gait_reward=='phase')
                track = np.exp(-(velocity[4]-self.commands[i])**2/self.forward_tracking_variance-velocity[3]**2/.25)
                single_support = int(contact['left_contact'] != contact['right_contact'])
                # Reward stepping through contacts; no prescribed targets or external propulsion.
                reward = 1.+2*track+velocity[4]-.25*velocity[2]**2-2*tilt**2
                gait=phase_gait_score(d.time,contact) if self.gait_reward=='phase' else single_support
                reward += .25*gait-.15*contact['mean_slip_mps']-.02*np.mean(action**2)
                reward -= .01*np.mean((action-self.previous[i])**2)
            rewards.append(-5. if fallen or invalid else float(reward))
            dones.append(done); timeouts.append(timeout and not fallen and not invalid)
            self.previous[i] = action
            if done:
                success = timeout and not fallen and not invalid
                if self.stage == 'forward' and self.commands[i]:
                    success = success and d.xpos[self.pelvis,1]-self.reference_xy[1] >= 5.
                else:
                    success = success and drift < .03
                self.completed.append(bool(success))
                self.completed_since_update += 1
                self.reset(i)
        return self.observations(), np.array(rewards,np.float32), np.array(dones,np.float32), np.array(timeouts,np.float32)


def phase_gait_score(time_s, feet):
    """Bounded alternating contact/clearance reward; no reference force or target."""
    phase=time_s % 1.
    left_stance=phase>=.5
    if feet['left_contact'] != left_stance or feet['right_contact'] == left_stance:
        return 0.
    swing='right' if left_stance else 'left'
    clearance_target=.012*np.sin(2*np.pi*phase)**2
    return float(np.exp(-((feet[swing+'_clearance_m']-clearance_target)/.008)**2))


def phase_load_score(phase, forces, clearance, supported_weight):
    """Continuous bounded phase score from actual foot forces and mesh clearance.

    Supports partial weight transfer before liftoff; both-air states have low
    loading score. This reward applies no reference forces or position targets.
    """
    sine=(2*torch.pi*phase).sin()
    left_load=.5+.5*(-2*sine).clamp(-1,1)
    target_load=torch.stack((left_load,1-left_load),dim=-1)
    target_clearance=.012*torch.stack((sine.clamp_min(0).square(),(-sine).clamp_min(0).square()),dim=-1)
    normalized_load=forces/supported_weight.clamp_min(1e-6)[...,None]
    load_score=(-((normalized_load-target_load).square().sum(-1))/.25).exp()
    clear_score=(-((clearance-target_clearance).square().sum(-1))/(.008**2)).exp()
    return .75*load_score+.25*clear_score


def restore_curriculum(batch, checkpoint):
    state=checkpoint.get('curriculum_state',{'coefficient':checkpoint.get('assistance_coefficient',0.)})
    coefficient=float(state['coefficient'])
    if not 0<=coefficient<=1: raise ValueError('Invalid saved assistance coefficient')
    batch.assistance.coefficient=coefficient
    batch.assistance.successes=int(state.get('successes',0))
    batch.completed=deque(state.get('completed',[]),maxlen=20)
    batch.completed_since_update=int(state.get('completed_since_update',0))


def load_actor(checkpoint, model, spec):
    blob = torch.load(checkpoint,map_location='cpu',weights_only=True)
    if blob['urdf_sha256'] != spec['source']['urdf_sha256'] or blob['mesh_tree_sha256'] != spec['source']['mesh_tree_sha256']:
        raise ValueError('Policy asset identity mismatch')
    if blob['backend'] not in ('mujoco_cpu','mujoco_warp_cuda') or blob['action_joints'] != spec['action_joints']:
        raise ValueError('Policy backend/joint identity mismatch')
    policy = ActorCritic(blob.get('observation_dim',60),blob.get('forward_exploration_multiplier',1.),blob.get('mean_activation','linear')); policy.load_state_dict(blob['model']); policy.eval()
    return policy, blob


def train(args):
    torch.set_num_threads(2)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    out=(OUTPUT/'training'/args.name).resolve(); out.relative_to(OUTPUT.resolve()); out.mkdir(parents=True,exist_ok=False)
    command=shlex.join([sys.executable,'-m','algorithms.urdf_learn_wasd_walk.mujoco_policy',*sys.argv[1:]])
    device=torch.device('cuda:0' if args.backend=='mujoco_warp_cuda' else 'cpu')
    resource_stop=resource_sampler=None
    if device.type=='cuda':
        from algorithms.urdf_learn_wasd_walk.mujoco_g1_benchmark import sample_resources
        resource_stop=threading.Event()
        resource_sampler=threading.Thread(target=sample_resources,args=(os.getpid(),out,resource_stop),daemon=True)
        resource_sampler.start()
    start=time.perf_counter()
    if args.backend=='mujoco_warp_cuda':
        from algorithms.urdf_learn_wasd_walk.mujoco_warp_batch import WarpBatch
        batch=WarpBatch(args.num_envs,args.seed,args.assistance,args.stage,args.gait_reward,args.contact_timeconst,args.forward_speed,args.forward_tracking_variance)
    else:
        batch=StandBatch(args.num_envs,args.seed,args.assistance,args.stage,args.gait_reward,args.forward_speed,args.forward_tracking_variance)
    build_s=time.perf_counter()-start
    policy=ActorCritic(65 if args.stage=='forward' else 60,args.forward_exploration_multiplier,args.mean_activation)
    if args.stand_checkpoint:
        parent, parent_blob = load_actor(args.stand_checkpoint,batch.model,batch.spec)
        if parent_blob['backend']!=args.backend or parent_blob['model_xml_sha256']!=hashlib.sha256(batch.xml.encode()).hexdigest():
            raise ValueError('Stand transfer requires matching backend and physics configuration')
        if args.stage != 'forward' or parent_blob.get('observation_dim',60) != 60:
            raise ValueError('Transfer requires a 60-observation stand parent')
        transferred = policy.state_dict()
        for name,value in parent.state_dict().items():
            if name.endswith('0.weight'):
                transferred[name].zero_(); transferred[name][:,:60] = value
            else: transferred[name] = value
        policy.load_state_dict(transferred)
    policy.to(device)
    optimizer=torch.optim.Adam(policy.parameters(),lr=args.learning_rate)
    actor_parameters=[policy.log_std,*policy.actor.parameters()]
    critic_parameters=list(policy.critic.parameters())
    if args.resume:
        resumed, blob=load_actor(args.resume,batch.model,batch.spec)
        if blob['backend']!=args.backend or blob['mujoco_version']!=mujoco.__version__ or blob['model_xml_sha256'] != hashlib.sha256(batch.xml.encode()).hexdigest():
            raise ValueError('Resume model configuration mismatch')
        if resumed.mean_activation!=policy.mean_activation:
            raise ValueError('Resume must preserve the checkpoint mean activation')
        policy.load_state_dict(resumed.state_dict()); optimizer.load_state_dict(blob['optimizer'])
        policy.forward_exploration_multiplier=resumed.forward_exploration_multiplier
        restore_curriculum(batch,blob)
    curriculum_enabled=bool(args.assistance>0 or (args.resume and blob.get('curriculum_enabled',blob.get('assistance_coefficient',0)>0)))
    obs=torch.as_tensor(batch.observations(),device=device)
    (out/'source.py').write_text(Path(__file__).read_text())
    (out/'backend_source.py').write_text(Path(sys.modules[build_model.__module__].__file__).read_text())
    write_json(out/'config.json',vars(args)); write_json(out/'model_audit.json',batch.audit)
    (out/'model.xml').write_text(batch.xml)
    if args.backend=='mujoco_warp_cuda':
        (out/'warp_batch_source.py').write_text(Path(sys.modules[batch.__class__.__module__].__file__).read_text())
    progress_path=model_spec.ALGORITHM_ROOT/'outputs/backend_progress.json'
    history=[]; start_training=time.perf_counter(); process=psutil.Process()
    for iteration in range(args.iterations):
        iteration_start=time.perf_counter(); rollout=[]; cpu_start=process.cpu_times()
        for step in range(24):
            with torch.no_grad():
                dist=policy.distribution(obs); action=dist.sample(); value=policy.critic(obs).squeeze(-1)
                logp=dist.log_prob(action).sum(-1)
            next_obs,reward,done,timeout=batch.step(action if device.type=='cuda' else action.numpy())
            # Time-limit episodes form a finite 30-second stand task, no bootstrap across reset.
            rollout.append((obs,action,logp,value,torch.as_tensor(reward,device=device),torch.as_tensor(done,device=device)))
            obs=torch.as_tensor(next_obs,device=device)
        if device.type=='cuda':torch.cuda.synchronize()
        rollout_s=time.perf_counter()-iteration_start
        with torch.no_grad(): next_value=policy.critic(obs).squeeze(-1)
        advantage=torch.zeros(args.num_envs,device=device); advantages=[]; returns=[]
        for o,a,lp,v,r,d in reversed(rollout):
            delta=r+.99*next_value*(1-d)-v
            advantage=delta+.99*.95*(1-d)*advantage
            advantages.append(advantage); returns.append(advantage+v); next_value=v
        O,A,LP,V,R,D=[torch.cat([row[k] for row in rollout]) for k in range(6)]
        ADV=torch.cat(list(reversed(advantages))); RET=torch.cat(list(reversed(returns)))
        ADV=(ADV-ADV.mean())/(ADV.std()+1e-8)
        learn_start=time.perf_counter()
        actor_before=[p.detach().clone() for p in actor_parameters]
        critic_before=[p.detach().clone() for p in critic_parameters]
        gradient_rows=[]
        stop_update=False; optimizer_steps=0
        for epoch in range(5):
            for ids in torch.randperm(len(O),device=device).chunk(4):
                dist=policy.distribution(O[ids]); logp=dist.log_prob(A[ids]).sum(-1)
                logratio=logp-LP[ids]
                ratio=logratio.exp()
                approximate_kl=float(((ratio-1)-logratio).mean().detach())
                if args.target_kl and approximate_kl > 1.5*args.target_kl:
                    stop_update=True
                    break
                surrogate=torch.minimum(ratio*ADV[ids],ratio.clamp(.8,1.2)*ADV[ids])
                value=policy.critic(O[ids]).squeeze(-1)
                value_clipped=V[ids]+(value-V[ids]).clamp(-.2,.2)
                value_loss=torch.maximum((value-RET[ids]).square(),(value_clipped-RET[ids]).square()).mean()
                loss=-surrogate.mean()+value_loss-.01*dist.entropy().sum(-1).mean()
                optimizer.zero_grad(); loss.backward()
                actor_norm=torch.stack([p.grad.detach().square().sum() for p in actor_parameters]).sum().sqrt()
                critic_norm=torch.stack([p.grad.detach().square().sum() for p in critic_parameters]).sum().sqrt()
                gradient_rows.append(torch.stack((actor_norm,critic_norm)))
                if args.gradient_clipping=='separate':
                    # Match upstream rsl-rl: two clipping groups, one Adam.
                    nn.utils.clip_grad_norm_(actor_parameters,1.)
                    nn.utils.clip_grad_norm_(critic_parameters,1.)
                else:
                    nn.utils.clip_grad_norm_(policy.parameters(),1.)
                optimizer.step(); optimizer_steps+=1
            if stop_update: break
        if batch.wrench_trace:
            # Actual applied vectors for env 0 at every physics step; all-env maxima/impulses below.
            with (out/'assistance_env0.jsonl').open('a') as f:
                for row in batch.wrench_trace: f.write(json.dumps(row)+'\n')
            batch.wrench_trace.clear()
        if curriculum_enabled and batch.completed_since_update >= 20:
            batch.assistance.update(float(np.mean(batch.completed)),window_episodes=len(batch.completed))
            batch.completed_since_update=0
        with torch.no_grad():
            gradient_mean=torch.stack(gradient_rows).mean(0).tolist() if gradient_rows else [None,None]
            actor_delta=float(torch.stack([(p-old).square().sum() for p,old in zip(actor_parameters,actor_before)]).sum().sqrt())
            critic_delta=float(torch.stack([(p-old).square().sum() for p,old in zip(critic_parameters,critic_before)]).sum().sqrt())
            target_variance=float(RET.var())
            optimization={'gradient_clipping':args.gradient_clipping,
                'actor_preclip_gradient_norm_mean':gradient_mean[0],
                'critic_preclip_gradient_norm_mean':gradient_mean[1],
                'actor_parameter_update_norm':actor_delta,'critic_parameter_update_norm':critic_delta,
                'rollout_value_mean':float(V.mean()),'return_target_mean':float(RET.mean()),
                'rollout_value_rmse':float((V-RET).square().mean().sqrt()),
                'rollout_value_explained_variance':1-float((RET-V).var())/target_variance if target_variance>1e-12 else None}
        if device.type=='cuda':torch.cuda.synchronize()
        metrics={'iteration':iteration+1,'rollout_s':rollout_s,'learning_s':time.perf_counter()-learn_start,
                 'iteration_s':time.perf_counter()-iteration_start,'rss_bytes':process.memory_info().rss,
                 'cpu_percent_one_core_100':100*(sum(process.cpu_times()[:2])-sum(cpu_start[:2]))/(time.perf_counter()-iteration_start),
                 'approximate_kl':approximate_kl,'optimizer_steps':optimizer_steps,'kl_early_stop':stop_update,'control_transitions_per_s':args.num_envs*24/rollout_s,
                 'physics_transitions_per_s':args.num_envs*240/rollout_s,'mean_reward':float(R.mean()),
                 'recent_30s_success_rate':float(np.mean(batch.completed)) if batch.completed else None,
                 'assistance_coefficient':batch.assistance.coefficient,'max_applied_force_n':batch.max_force,
                 'max_applied_torque_nm':batch.max_torque,'force_impulse_norm_ns':batch.force_impulse,
                 'torque_impulse_norm_nms':batch.torque_impulse,'elapsed_training_s':time.perf_counter()-start_training}
        metrics.update(optimization)
        if hasattr(batch,'last_diagnostics'):
            metrics['last_control_step_diagnostics']={k:float(v) for k,v in batch.last_diagnostics.items()}
        if device.type=='cuda':
            metrics.update(gait_reward_active_fraction=batch.gait_active_steps/max(1,batch.walking_steps),
                maximum_foot_clearance_m=batch.maximum_clearance,maximum_joint_speed_rad_s=batch.maximum_joint_speed,
                training_liftoffs_per_side=batch.total_liftoffs.tolist(),
                total_training_resets=batch.total_resets,tensor_device=str(device),torch_peak_vram_bytes=torch.cuda.max_memory_allocated())
        history.append(metrics)
        with (out/'iterations.jsonl').open('a') as f: f.write(json.dumps(metrics)+'\n')
        if (iteration+1)%10==0 or iteration==args.iterations-1 or time.perf_counter()-start_training>args.budget_s:
            blob={'model':policy.state_dict(),'optimizer':optimizer.state_dict(),'iteration':iteration+1,
                  'observation_dim':65 if args.stage=='forward' else 60,'stage':args.stage,
                  'backend':args.backend,'mujoco_version':mujoco.__version__,'action_joints':batch.spec['action_joints'],
                  'forward_exploration_multiplier':policy.forward_exploration_multiplier,
                  'mean_activation':policy.mean_activation,
                  'urdf_sha256':batch.spec['source']['urdf_sha256'],'mesh_tree_sha256':batch.spec['source']['mesh_tree_sha256'],
                  'nominal_q':batch.nominal_q.tolist(),'nominal_ctrl':batch.nominal_ctrl.tolist(),
                  'action_scale':.24 if args.stage=='forward' else ACTION_SCALE,'stand_action_scale':ACTION_SCALE,'assistance_coefficient':batch.assistance.coefficient,
                  'source_sha256':digest(__file__),'model_xml_sha256':hashlib.sha256(batch.xml.encode()).hexdigest(),
                  'seed':args.seed,'config':vars(args),'gate_passed':False,'curriculum_enabled':curriculum_enabled,
                  'curriculum_state':{'coefficient':batch.assistance.coefficient,'successes':batch.assistance.successes,
                                      'completed':list(batch.completed),'completed_since_update':batch.completed_since_update},
                  'resume_semantics':'network/optimizer/curriculum restored; randomized training episodes restart',
                  'parent_checkpoint_sha256':digest(args.resume or args.stand_checkpoint) if args.resume or args.stand_checkpoint else None}
            checkpoint=out/f'model_{iteration+1}.pt'; torch.save(blob,checkpoint)
            progress=json.loads(progress_path.read_text())
            progress.update(updated_at=datetime.now(timezone.utc).isoformat(),current_gate='gate_5m_no_reset' if args.stage=='forward' else 'stand_30s_no_reset',simulator=args.backend,
                            variant=args.name,assistance_coefficient=batch.assistance.coefficient,active_process={**(progress.get('active_process') or {}),'pid':os.getpid(),'pid_namespace':'worker_private' if device.type=='cuda' else 'sandbox','host_pid':None,'command':command,'cwd':str(Path.cwd())},
                            throughput=metrics,iteration=iteration+1,checkpoint=str(checkpoint),validation={'training_is_not_gate_evidence':True},
                            next_step='evaluate exact checkpoint for 30s with auxiliary forces disabled; retain failures',artifact_paths=[str(out)])
            write_json(progress_path,progress)
            print(json.dumps(metrics),flush=True)
        if time.perf_counter()-start_training>args.budget_s: break
    summary={'status':'training_completed_unvalidated','command':command,'model_build_s':build_s,
             'gpu_warmup_compile_and_step_s':getattr(batch,'compile_s',None),'wall_s':time.perf_counter()-start,
             'checkpoint':str(checkpoint),'checkpoint_sha256':digest(checkpoint),'metrics':history[-1], 'config':vars(args)}
    if resource_stop is not None:
        resource_stop.set();resource_sampler.join(timeout=3)
        summary.update(resource_file=str(out/'resources.jsonl'),gpu_physics_initial_compilation_s=batch.compile_s,
                       assistance_update_hz=50,assistance_wrench_hold_physics_steps=10)
    write_json(out/'training.json',summary)
    progress=json.loads(progress_path.read_text()); progress['active_process']=None; write_json(progress_path,progress)
    print(json.dumps(summary),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--name',required=True); p.add_argument('--seed',type=int,default=42)
    p.add_argument('--backend',choices=('mujoco_cpu','mujoco_warp_cuda'),default='mujoco_cpu')
    p.add_argument('--contact-timeconst',type=float,default=.004)
    p.add_argument('--forward-exploration-multiplier',type=float,default=1.)
    p.add_argument('--mean-activation',choices=['linear','tanh'],default='linear')
    p.add_argument('--num-envs',type=int,default=16); p.add_argument('--iterations',type=int,default=200)
    p.add_argument('--budget-s',type=float,default=300); p.add_argument('--assistance',type=float,default=0.)
    p.add_argument('--resume')
    p.add_argument('--stage',choices=['stand','forward'],default='stand')
    p.add_argument('--stand-checkpoint')
    p.add_argument('--forward-speed',type=float,default=.4,help='Training forward command in m/s; does not change gate distance')
    p.add_argument('--forward-tracking-variance',type=float,default=.25)
    p.add_argument('--learning-rate',type=float,default=1e-3)
    p.add_argument('--target-kl',type=float,default=0.)
    p.add_argument('--gradient-clipping',choices=['joint','separate'],default='joint')
    p.add_argument('--gait-reward',choices=['single','phase','load'],default='single')
    args=p.parse_args()
    if args.num_envs<1 or args.iterations<1 or args.budget_s<=0 or not 0<=args.assistance<=1 or not 0<args.forward_exploration_multiplier<=10: p.error('invalid experiment bounds')
    if not 0 < args.forward_speed <= 1.:p.error('forward speed must be in (0, 1] m/s')
    if not 0 < args.forward_tracking_variance <= 1.:p.error('forward tracking variance must be in (0, 1]')
    if args.gait_reward=='load' and args.backend!='mujoco_warp_cuda':p.error('The continuous load experiment currently requires the audited GPU force adapter')
    train(args)


if __name__=='__main__': main()
