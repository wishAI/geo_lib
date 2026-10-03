"""Per-world M7 steering and command memory for training, never certification.

The serial evaluator remains the acceptance path. This module mirrors its
full-forward, single-turn-direction protocol without replaying another pose.
"""
import math


def common_start_offsets(nominal, joint_indices, replicas, seed):
    """One nominal and shared bounded joint perturbations, independent of CEM RNG."""
    import torch
    generator=torch.Generator(device='cpu').manual_seed(seed)
    noise=torch.rand((replicas,len(joint_indices)),generator=generator)*.004-.002
    noise[0]=0.
    offsets=torch.zeros((replicas,len(nominal)),dtype=nominal.dtype,device=nominal.device)
    offsets[:,joint_indices]=noise.to(device=nominal.device,dtype=nominal.dtype)
    return offsets


def balance_grid(center, candidates, radius):
    """Fixed sway grid with two unchanged controls; repeat exactly across generations."""
    import torch
    side=math.isqrt(candidates)
    if side*side!=candidates or side<3:raise ValueError('Balance grid requires a square of at least9candidates')
    result=center.repeat(candidates,1)
    sweep=torch.linspace(-radius,radius,side,device=center.device,dtype=center.dtype)
    result[:,5]=(center[5]+sweep.repeat_interleave(side)).clamp(-1,1)
    result[:,6]=(center[6]+sweep.repeat(side)).clamp(-1,1)
    result[:2]=center
    return result


def cruise_balance_parameters(normalized, lows, highs, seed):
    """Search only post-left-turn sway; retain all other seed bits exactly."""
    result=seed.expand(len(normalized),-1).clone()
    result[:,15:17]=lows[15:17]+(normalized[:,15:17]+1)*.5*(highs[15:17]-lows[15:17])
    center=2*(seed-lows)/(highs-lows)-1.
    result[(normalized[:,15:17]==center[15:17]).all(1)]=seed
    return result


def cruise_rate_grid(seed):
    """Duplicate incumbents and six bounded rate offsets; freeze first20 bits."""
    result=seed.repeat(8,1)
    result[:,20]=seed.new_tensor([float(seed[20]),float(seed[20]),-.01,.005,.01,.015,.02,.03])
    return result


def cruise_balance_grid(seed, candidates, lows, highs):
    """Wide physical-unit sway grid, repeated controls and active-turn settings."""
    import math
    import torch
    side=math.isqrt(candidates)
    if side*side!=candidates or side<3:raise ValueError('Cruise grid requires a square of at least9candidates')
    result=seed.repeat(candidates,1)
    sweep=torch.linspace(-1.,1.,side,device=seed.device,dtype=seed.dtype)
    result[:,15]=(seed[15]+.018*sweep.repeat_interleave(side)).clamp(lows[15],highs[15])
    result[:,16]=(seed[16]+.15*sweep.repeat(side)).clamp(lows[16],highs[16])
    result[:2]=seed
    result[2,15:17]=seed[5:7]
    return result


def commands(direction, displacement, heading, seconds):
    import torch
    from algorithms.urdf_learn_wasd_walk import landau_direction_contract as contract
    xy=displacement.to(torch.float64);angle=heading.to(torch.float64)
    result=torch.zeros((len(xy),3),dtype=torch.float64,device=xy.device)
    result[:,0]=.2
    if direction=='forward' or seconds<3.:return result
    axis=xy.new_tensor(contract.DIRECTIONS[direction])
    progress=(xy*axis).sum(1)
    ahead=torch.maximum(torch.full_like(progress,10.),progress+2.)
    target=ahead[:,None]*axis-xy
    bearing=torch.atan2(-target[:,0],target[:,1])
    error=torch.atan2((bearing-angle).sin(),(bearing-angle).cos())
    if direction=='backward':error=torch.where(error < -math.pi/2,error+2.*math.pi,error)
    yaw=(.3*error).clamp(-contract.MAX_YAW_RATE,contract.MAX_YAW_RATE)
    result[:,2]=yaw.clamp_max(0.) if direction=='right' else yaw.clamp_min(0.)
    return result


class DirectionMemory:
    """Independent references/dwells; shared side and onset for one direction.

    No stop, restart or simulator-state writes. Double-precision accumulation
    matches the scalar command memory, with float32 values for actor arithmetic.
    """
    def __init__(self, count, device, direction):
        import torch
        self.direction=direction
        self.last_yaw_sign=-1 if direction=='right' else 1 if direction!='forward' else 0
        self._reference=torch.zeros(count,dtype=torch.float64,device=device)
        self._blend=torch.zeros_like(self._reference)
        self._feedback_blend=torch.zeros_like(self._reference)
        self.previous_yaw=torch.zeros_like(self._reference)
        self.left_cruise_start=torch.full_like(self._reference,float('inf'))
        self.previous_time=None
        self.turned=False
        self.anchor_xy=None;self.anchor_time=None;self.restart_time=None
        self.reference=self._reference.float()
        self.left_cruise_blend=self._blend.float()
        self.left_feedback_blend=self._feedback_blend.float()

    def observe(self, observation, positions, rotations, seconds, active=None):
        import torch
        yaw=observation[:,65].to(torch.float64)
        live=torch.ones_like(yaw,dtype=torch.bool) if active is None else active
        elapsed=0. if self.previous_time is None else seconds-self.previous_time
        self._reference+=torch.where(live,self.previous_yaw*elapsed,0.)
        if not self.turned and self.last_yaw_sign and seconds>=3.:
            if not bool(((yaw*self.last_yaw_sign>0.)|~live).all()):
                raise ValueError('Direction worlds did not share the declared initial yaw onset')
            self.turned=True
        cruising=(yaw==0.) & self.turned & (self.last_yaw_sign>0)
        self.left_cruise_start=torch.where(cruising,
            torch.minimum(self.left_cruise_start,torch.full_like(yaw,seconds)),float('inf'))
        target=(cruising & (seconds-self.left_cruise_start>=.2-1e-9)).to(torch.float64)
        self._blend+=torch.where(live,(target-self._blend).clamp(-elapsed,elapsed),0.)
        self._feedback_blend+=torch.where(live,((yaw>0.).to(torch.float64)-self._feedback_blend).clamp(-elapsed,elapsed),0.)
        self.reference=self._reference.to(observation.dtype)
        self.left_cruise_blend=self._blend.to(observation.dtype)
        self.left_feedback_blend=self._feedback_blend.to(observation.dtype)
        self.previous_yaw=torch.where(live,yaw,0.);self.previous_time=seconds
        return observation.clone(),observation[:,:63].clone()

    def record(self):
        return {'anchor_xy':[],'anchor_time':None,'simulation_reset':False,
            'dispatch':'per_world_direction_commands','turned':self.turned,
            'integrated_reference_rad':self._reference.cpu().tolist(),
            'left_cruise_blend':self._blend.cpu().tolist(),
            'left_feedback_blend':self._feedback_blend.cpu().tolist(),
            'restart_time':None,'gait_clock_global':True,'memory_version':2}


class GateTracker:
    """First valid interpolated crossing; freeze completed episode metrics."""
    def __init__(self, direction, origins):
        import torch
        from algorithms.urdf_learn_wasd_walk.landau_direction_contract import DIRECTIONS
        self.origins=origins[:,:2].clone().to(torch.float64)
        self.axis=self.origins.new_tensor(DIRECTIONS[direction])
        self.previous=torch.zeros_like(self.origins)
        self.completed=torch.zeros(len(origins),device=origins.device,dtype=torch.bool)
        self.crossing_time=torch.full((len(origins),),float('inf'),device=origins.device,dtype=torch.float64)
        self.crossing_lateral=torch.full_like(self.crossing_time,float('nan'))
        self.progress=torch.zeros_like(self.crossing_time)
        self.lateral=torch.zeros_like(self.crossing_time)

    def update(self, positions, seconds, active):
        import torch
        from algorithms.urdf_learn_wasd_walk.landau_direction_contract import GATE_WIDTH_M
        xy=positions[:,:2].to(torch.float64)-self.origins
        progress=(xy*self.axis).sum(1);old=(self.previous*self.axis).sum(1)
        lateral=xy[:,0]*self.axis[1]-xy[:,1]*self.axis[0]
        old_lateral=self.previous[:,0]*self.axis[1]-self.previous[:,1]*self.axis[0]
        fraction=(10.-old)/(progress-old).clamp_min(1e-12)
        crossing=old_lateral+fraction*(lateral-old_lateral)
        fresh=active & ~self.completed & (old<10.) & (progress>=10.) & (crossing.abs()<=GATE_WIDTH_M)
        self.crossing_time=torch.where(fresh,seconds-.02+fraction*.02,self.crossing_time)
        self.crossing_lateral=torch.where(fresh,crossing,self.crossing_lateral)
        self.progress=torch.where(active,progress,self.progress)
        self.lateral=torch.where(active,lateral,self.lateral)
        self.completed|=fresh
        self.previous=xy
        return fresh

    def record(self, members):
        def finite(values):return [v if math.isfinite(v) else None for v in values.detach().cpu().tolist()]
        return {'completed':self.completed[members].cpu().tolist(),
            'crossing_time_s':finite(self.crossing_time[members]),
            'cross_track_m':finite(self.crossing_lateral[members]),
            'progress_m':finite(self.progress[members]),'final_lateral_m':finite(self.lateral[members])}
