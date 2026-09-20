"""Fresh supervised student of reviewed physical ragdoll demonstrations.

No old Landau or G1 weights are loaded. Evaluation and assistance reduction are
separate physical experiments; low imitation error is never a milestone pass.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from algorithms.urdf_learn_wasd_walk.mujoco_backend import OUTPUT, digest, write_json


class Student(nn.Module):
    def __init__(self, observation_dim=63, residual_step_rad=0., action_scale=.5, phase_clock_only=False, temporal_harmonics=0):
        super().__init__()
        if observation_dim not in (63,64):raise ValueError('Unsupported student observation dimension')
        self.observation_dim=observation_dim
        if not 0<=residual_step_rad<=.1 or not 0<action_scale<=2:raise ValueError('Invalid residual action bounds')
        self.residual_step_rad=residual_step_rad
        self.action_scale=action_scale
        self.phase_clock_only=phase_clock_only
        if not isinstance(temporal_harmonics,int) or not 0<=temporal_harmonics<=16 or (temporal_harmonics and not phase_clock_only):raise ValueError('Temporal harmonics require clock-only generator and bounded integer count')
        self.temporal_harmonics=temporal_harmonics
        self.register_buffer('frequencies',torch.arange(1,temporal_harmonics+1,dtype=torch.float32),persistent=False)
        if phase_clock_only and (observation_dim!=64 or residual_step_rad):raise ValueError('Clock-only generator requires startup clock and absolute targets')
        self.actor=nn.Sequential(nn.Linear(observation_dim+4*temporal_harmonics,128),nn.ELU(),nn.Linear(128,128),nn.ELU(),nn.Linear(128,17),nn.Tanh())
        if residual_step_rad:
            nn.init.zeros_(self.actor[-2].weight);nn.init.zeros_(self.actor[-2].bias)

    def forward(self,x):
        if self.phase_clock_only:x=torch.cat([torch.zeros_like(x[...,:60]),x[...,60:]],dim=-1)
        if self.temporal_harmonics:
            phase=torch.atan2(x[...,61],x[...,62])[...,None]*self.frequencies
            startup=2*torch.pi*x[...,63,None]*self.frequencies
            x=torch.cat([x,phase.sin(),phase.cos(),startup.sin(),startup.cos()],dim=-1)
        change=self.actor(x)
        if self.residual_step_rad:
            return torch.clamp((.5*x[...,43:60]+self.residual_step_rad*change)/self.action_scale,-1.,1.)
        return change


def observation_noise_bounds(scale, mode='all_proprio'):
    """Training-only uniform perturbations in the 63 observation units.

    Small local label-preserving augmentation is a hypothesis, not a physics
    expert at perturbed states. Commands and phase remain exact. Joint velocity
    observations are already scaled by 0.1; previous actions by 1/0.5 rad.
    """
    if not np.isfinite(scale) or not 0 <= scale <= 2:
        raise ValueError('Invalid observation noise scale')
    if mode not in ('all_proprio','previous_action'):raise ValueError('Invalid observation noise mode')
    bounds=scale*np.array([.02]*3+[.02]*3+[.005]*3+[.005]*17+
                         [.01]*17+[.02]*17+[0.]*3,dtype=np.float32)
    if mode=='previous_action':bounds[:43]=0
    return bounds


def load_student(path):
    checkpoint=torch.load(path,map_location='cuda:0',weights_only=False)
    if checkpoint['lineage']!='ragdoll_walk_first_20260918':raise ValueError('Wrong student lineage')
    model=Student(checkpoint.get('observation_dim',63),checkpoint.get('residual_step_rad',0.),float(checkpoint['action_scale']),checkpoint.get('phase_clock_only',False),checkpoint.get('temporal_harmonics',0)).to('cuda:0');model.load_state_dict(checkpoint['state_dict']);model.eval()
    model.action_scale=float(checkpoint['action_scale'])
    model.startup_clock_s=float(checkpoint.get('startup_clock_s',0.))
    if model.observation_dim!=63+int(model.startup_clock_s>0):raise ValueError('Student clock/dimension mismatch')
    return model


def prepare_observations(observations, startup_clock_s=0.):
    """Expose startup progress without loading teacher motion during inference."""
    x=np.asarray(observations)
    if not np.isfinite(startup_clock_s) or not 0<=startup_clock_s<=120:raise ValueError('Invalid startup clock')
    if startup_clock_s==0:
        if x.ndim!=2 or x.shape[1]!=63:raise ValueError('Legacy student requires63 observations')
        return x
    clock=np.clip(np.arange(len(x))*.02/startup_clock_s,0.,1.).astype(np.float32)
    if x.ndim!=2 or x.shape[1] not in (63,64):raise ValueError('Invalid observation dimensions')
    if x.shape[1]==64:
        if not np.allclose(x[:,-1],clock,rtol=0,atol=1e-6):raise ValueError('Recorded startup clock mismatch')
        return x
    return np.c_[x,clock]



def partition_trajectories(xs, ys, *, interleaved=False):
    """Keep training and held-out samples from every visited trajectory."""
    if len(xs)!=len(ys) or not xs:raise ValueError('Unpaired trajectories')
    train_x=[];train_y=[];valid_x=[];valid_y=[]
    for x,y in zip(xs,ys):
        if len(x)!=len(y) or len(x)<5:raise ValueError('Unaligned or short trajectory')
        if interleaved:
            heldout=np.arange(len(x))%5==4
            train_x.append(x[~heldout]);train_y.append(y[~heldout])
            valid_x.append(x[heldout]);valid_y.append(y[heldout])
        else:
            split=int(.8*len(x))
            train_x.append(x[:split]);train_y.append(y[:split])
            valid_x.append(x[split:]);valid_y.append(y[split:])
    return tuple(np.concatenate(v) for v in (train_x,train_y,valid_x,valid_y))


def aligned_expert_samples(samples, observation_count):
    """Match pre-step 50 Hz actor calls to their post-step 2 ms physics records.

    A fall between actor calls adds a terminal physics sample, not an extra
    observation. Never silently truncate or pair that sample with another call.
    """
    times=np.array([row['time_s'] for row in samples])
    expected=np.arange(observation_count)*.02+.002
    indices=np.searchsorted(times,expected-1e-7)
    if np.any(indices>=len(times)) or not np.allclose(times[indices],expected,rtol=0.,atol=1e-7):
        raise ValueError('Expert/observation timestamp alignment mismatch')
    return [samples[int(i)] for i in indices]


def train(args):
    demonstration=Path(args.demonstration).resolve();demonstration.relative_to(OUTPUT.resolve())
    review=json.loads((demonstration.parent/'visual_review.json').read_text())
    if not review.get('accepted_as_assisted_teacher'):raise ValueError('Reviewed assisted teacher required')
    metrics=json.loads((demonstration.parent/'result.json').read_text())
    if not metrics.get('assisted_dynamics_candidate'):raise ValueError('Demonstration failed assisted dynamics')
    if review.get('trajectory_sha256')!=digest(demonstration.parent/'trajectory.npz'):raise ValueError('Review identity mismatch')
    out=OUTPUT/'ragdoll'/'ragdoll_walk_first_20260918'/args.name
    out.resolve().relative_to(OUTPUT.resolve());out.mkdir(parents=True,exist_ok=False)
    torch.manual_seed(42);np.random.seed(42);torch.set_num_threads(2)
    if not torch.cuda.is_available():raise RuntimeError('GPU worker required')
    start=time.perf_counter();dataset=np.load(demonstration)
    xs=[dataset['observations']];ys=[dataset['actions']*float(dataset['action_scale'])/args.action_scale];aggregate_records=[]
    for directory in args.aggregate:
        folder=Path(directory).resolve();folder.relative_to(OUTPUT.resolve())
        recorded=json.loads((folder/'result.json').read_text());coefficient=recorded.get('teacher_blend_coefficient',recorded['config']['coefficient'])
        if recorded.get('diagnostic_oracle_student'):raise ValueError('Oracle timing diagnostic is not a student aggregation trajectory')
        if recorded.get('diagnostic_observation_replay_sha256'):raise ValueError('Reference observations are not visited student states')
        if recorded.get('teacher_blend_schedule'):raise ValueError('Scheduled rescue is diagnostic only; fixed-blend expert labels required')
        if recorded['lineage']!='ragdoll_walk_first_20260918' or coefficient<=0:raise ValueError('Expert labels unavailable')
        motor_defaults={'phase_delay':.5,'period':2.,'stride':.06,'clearance':.025,'waist_amplitude':.18,'hip_roll_amplitude':0.,'foot_placement_gain':0.,'waist_transfer_sharpness':0.,'motor_balance_gain':0.,'motor_balance_scale':1.,'waist_phase_lead':0.,'teacher_tracking_integral':0.}
        for key,default in motor_defaults.items():
            if recorded['config'].get(key,default)!=metrics['config'].get(key,default):raise ValueError(f'Incompatible motor reference: {key}')
        if recorded['config'].get('motor_pose_sequence_sha256')!=metrics['config'].get('motor_pose_sequence_sha256'):
            raise ValueError('Incompatible motor pose reference')
        visited=np.load(folder/'demonstrations.npz')
        # Relabel states visited by the blended student using the actual logged
        # unblended expert targets, not the student's own mixed action.
        aligned=aligned_expert_samples(recorded['samples'],len(visited['observations']))
        labels=np.array([r['teacher_target_contribution_rad'] for r in aligned])/(coefficient*args.action_scale)
        if len(labels)!=len(visited['observations']):raise ValueError('Expert/observation alignment mismatch')
        xs.append(visited['observations']);ys.append(labels)
        aggregate_records.append({'path':str(folder),'trajectory_sha256':digest(folder/'trajectory.npz'),'result_sha256':digest(folder/'result.json'),'coefficient':coefficient})
    xs=[prepare_observations(v,args.startup_clock_s) for v in xs]
    train_x,train_y,valid_x,valid_y=partition_trajectories(xs,ys,interleaved=args.interleaved_split)
    x=torch.tensor(np.concatenate([train_x,valid_x]),dtype=torch.float32,device='cuda:0')
    y=torch.tensor(np.concatenate([train_y,valid_y]),dtype=torch.float32,device='cuda:0')
    split=len(train_x);model=Student(x.shape[1],args.residual_step_rad,args.action_scale,args.phase_clock_only,args.temporal_harmonics).to('cuda:0');optimizer=torch.optim.Adam(model.parameters(),lr=1e-3)
    lengths=torch.tensor([len(a)-len(a)//5 if args.interleaved_split else int(.8*len(a)) for a in xs],device='cuda:0')
    starts=torch.cat([torch.zeros(1,device='cuda:0',dtype=torch.long),lengths.cumsum(0)[:-1]])
    log=[]
    noise_bounds=torch.tensor(np.r_[observation_noise_bounds(args.observation_noise_scale,args.observation_noise_mode),np.zeros(x.shape[1]-63)],dtype=torch.float32,device='cuda:0')
    for iteration in range(args.updates):
        if args.balanced_trajectories:
            # A short failure must not disappear among thousands of steady
            # gait samples. Held-out data remain separate and unrepeated.
            trajectory=torch.randint(len(lengths),(256,),device='cuda:0')
            ids=starts[trajectory]+(torch.rand(256,device='cuda:0')*lengths[trajectory]).long()
        else:
            ids=torch.randint(split,(256,),device='cuda:0')
        inputs=x[ids]
        if args.observation_noise_scale:
            inputs=inputs+(2*torch.rand_like(inputs)-1)*noise_bounds
        loss=(model(inputs)-y[ids]).square().mean()
        optimizer.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
        if iteration%100==0 or iteration==args.updates-1:
            with torch.no_grad():validation=(model(x[split:])-y[split:]).square().mean()
            log.append({'iteration':iteration,'training_mse':float(loss),'heldout_mse':float(validation)})
    torch.cuda.synchronize()
    checkpoint=out/'student.pt';torch.save({'lineage':'ragdoll_walk_first_20260918','state_dict':model.state_dict(),'teacher_config':metrics['config'],'source_sha256':digest(__file__),'demonstration_sha256':digest(demonstration),'action_scale':args.action_scale,'observation_dim':model.observation_dim,'residual_step_rad':args.residual_step_rad,'phase_clock_only':args.phase_clock_only,'temporal_harmonics':args.temporal_harmonics,'startup_clock_s':args.startup_clock_s,'seed':42},checkpoint)
    write_json(out/'training.json',{'milestone_pass':False,'fresh_initialization':True,'dataset_split':('everyfifthsample heldout pertrajectory; temporallycorrelated interpolation, notindependentgeneralization' if args.interleaved_split else 'first80percent training andlast20percent heldout independently pertrajectory'),'seed':42,'device':torch.cuda.get_device_name(),'wall_s':time.perf_counter()-start,'config':vars(args),'checkpoint_sha256':digest(checkpoint),'demonstration_sha256':digest(demonstration),'dataset_aggregation_reference':'https://proceedings.mlr.press/v15/ross11a.html','aggregate_records':aggregate_records,'review_sha256':digest(demonstration.parent/'visual_review.json'),'source_sha256':digest(__file__),'iterations':log,'next_step':'Physical student/teacher blending evaluation; reduce coefficient only after sustained gait, rollback on regression'})
    (out/'student_source.py').write_text(Path(__file__).read_text());print(json.dumps(log[-1]))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--demonstration',required=True);p.add_argument('--name',required=True);p.add_argument('--updates',type=int,default=2000);p.add_argument('--aggregate',action='append',default=[])
    p.add_argument('--action-scale',type=float,default=.5,help='Student output normalization only; physical motor limits remain unchanged')
    p.add_argument('--balanced-trajectories',action='store_true',help='Sample each demonstration/failure trajectory equally before sampling a training state')
    p.add_argument('--interleaved-split',action='store_true',help='Hold out each fifth sample so late failure states remain in training; this is temporally correlated interpolation, not independent validation')
    p.add_argument('--observation-noise-scale',type=float,default=0.,help='Training-only local uniform observation augmentation; commands and gait phase unchanged')
    p.add_argument('--observation-noise-mode',choices=['all_proprio','previous_action'],default='all_proprio',help='Limit augmentation to previous actions when testing the joint-tracking precision tradeoff')
    p.add_argument('--startup-clock-s',type=float,default=0.,help='Optional explicit startup-progress observation; model-owned clock only, no teacher reference lookup')
    p.add_argument('--residual-step-rad',type=float,default=0.,help='Optional bounded learned correction to previous applied motor target; no teacher lookup at inference')
    p.add_argument('--phase-clock-only',action='store_true',help='Learned gait generator from command/phase/startup only; no trajectory lookup, proprioceptive policy feedback, or previous-action path')
    p.add_argument('--temporal-harmonics',type=int,default=0,help='Clock-only Fourier time features,1..16; no motion reference is evaluated at inference')
    args=p.parse_args()
    if not 0<=args.residual_step_rad<=.1:raise ValueError('Invalid residual action bound')
    if not 0<args.action_scale<=2:raise ValueError('Invalid output scale')
    observation_noise_bounds(args.observation_noise_scale,args.observation_noise_mode)
    if not np.isfinite(args.startup_clock_s) or not 0<=args.startup_clock_s<=120:raise ValueError('Invalid startup clock')
    train(args)

if __name__=='__main__':main()
