"""Train a bounded yaw-command extension around the certified walking controller.

The base gait and standing network remain frozen. Reference heading integrates
the semantic yaw command; stop anchors are controller memory, never state resets.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time

TURN_PARAMETERS = {
    'yaw_stride_gain': (0., .4),
    'yaw_stride_phase': (-.6, .1),
    'heading_feedback_delta': (0., .12),
    'hold_heading_feedback': (-.12, .12),
    'differential_stride_gain': (-.12, 0.),
    'turn_roll_amplitude_offset': (-.06, .03),
    'turn_roll_phase_offset': (-.25, .25),
}
TURN_START = 3.
TURN_END = 17.
HOLD_START = 19.
TURN_RATE = math.pi / (2 * (TURN_END - TURN_START))


def configure_profile(turn_duration):
    global TURN_END,HOLD_START,TURN_RATE
    if not 14.<=turn_duration<=90.:raise ValueError('Turn duration outside bounded range')
    TURN_END=TURN_START+turn_duration
    HOLD_START=TURN_END+2.
    TURN_RATE=math.pi/(2.*turn_duration)


def candidate_table_reference(checkpoint, metadata):
    """Bind selected search rows to the checkpoint's generation, never a newer table."""
    checkpoint=Path(checkpoint)
    generation_name='candidates_'+checkpoint.stem.removeprefix('model_')+'.json'
    saved_tables=metadata.get('generation_candidate_tables',{})
    if saved_tables:
        if generation_name not in saved_tables:raise ValueError('No candidate table bound to this checkpoint generation')
        return checkpoint.parent/generation_name,saved_tables[generation_name]
    if checkpoint.stem!='model_'+str(metadata['generations_completed']-1):
        raise ValueError('Legacy candidate table only describes the final generation')
    return checkpoint.parent/'candidates.json',metadata.get('candidate_table_sha256')


def command_profile(seconds):
    """A joystick test: walk, turn left, slow down, then hold for >=5 seconds."""
    yaw = TURN_RATE if TURN_START <= seconds < TURN_END else 0.
    reference = TURN_RATE * min(max(seconds - TURN_START, 0.), TURN_END - TURN_START)
    fade = min(max((seconds - TURN_END) / (HOLD_START - TURN_END), 0.), 1.)
    forward = .2 * (1. - fade * fade * (3. - 2. * fade))
    return forward, yaw, reference


def evaluate_turn_gate(metrics, required_duration):
    """Independent M5 acceptance: turn, stop and stay settled for five seconds."""
    failures = []
    for key in ('reset_count','done_count','fall_count'):
        if metrics.get(key) != 0: failures.append(f'{key} is not zero')
    bounds = {'max_reference_tilt_rad':math.pi/6,'root_height_drop_m':.08,
        'max_abs_action':1.+1e-6,'simultaneous_air_fraction':.05,
        'hold_max_heading_error_rad':math.radians(5),'hold_max_drift_m':.03,
        'hold_max_horizontal_speed_mps':.05,'hold_max_yaw_speed_rad_s':.1}
    for key, maximum in bounds.items():
        value = float(metrics.get(key,math.inf))
        if not math.isfinite(value) or value > maximum: failures.append(f'{key} exceeded {maximum:g}')
    duration=float(metrics.get('duration_s',math.nan))
    hold_duration=float(metrics.get('hold_duration_s',math.nan))
    if not math.isfinite(duration) or duration+1e-6 < required_duration:
        failures.append('turn episode ended before its declared duration')
    if not math.isfinite(hold_duration) or hold_duration+1e-6 < 5.:
        failures.append('settled hold is shorter than five seconds')
    if metrics.get('hold_max_abs_command',math.inf) != 0:
        failures.append('hold received a nonzero command')
    if metrics.get('policy_inference_steps',-1) != metrics.get('control_steps',-2):
        failures.append('checkpoint did not produce every action')
    for name in ('left_hip_pitch_joint','right_hip_pitch_joint','left_knee_joint','right_knee_joint'):
        if metrics.get('leg_joint_excursion_rad',{}).get(name,0) < .05:
            failures.append(f'{name} excursion is too small for walking')
    return failures


class CommandMemory:
    """Controller state driven by actual commands; no simulator state writes.

    The same helper runs on CPU evaluation and batched CUDA training. All worlds
    share the command stream, while stop positions are stored per world.
    """
    def __init__(self):
        self.reference = 0.
        self.previous_time = None
        self.previous_yaw = 0.
        self.previous_moving = False
        self.last_yaw_sign = 0
        self.turned = False
        self.first_turn_time = None
        self.yaw_onset = None
        self.anchor_xy = None
        self.anchor_time = None
        self.restart_time = None
        self.events = []
        self.age = 0.
        self.left_cruise_start = None
        self.left_cruise_blend = 0.
        self.left_feedback_blend = 0.

    def observe(self, observation, positions, rotations, seconds):
        import torch
        command = observation[0, 63:66].detach().cpu().tolist()
        moving = max(abs(v) for v in command) > 0.
        yaw = command[2]
        if self.previous_time is not None:
            self.reference += self.previous_yaw * (seconds - self.previous_time)
        if yaw != 0. and (self.previous_yaw == 0. or yaw * self.previous_yaw < 0.):
            self.yaw_onset = seconds
            self.last_yaw_sign = 1 if yaw>0 else -1
            self.events.append({'kind':'yaw_onset','time_s':seconds,'yaw':yaw})
        if yaw != 0. and not self.turned:
            self.turned = True
            self.first_turn_time = seconds
        if not moving and self.previous_moving:
            self.anchor_xy = positions[:, :2].clone()
            self.anchor_time = seconds
            self.events.append({'kind':'stop_anchor','time_s':seconds})
        if moving and not self.previous_moving and self.anchor_xy is not None:
            self.restart_time = seconds
            self.events.append({'kind':'restart','time_s':seconds})
        obs = observation.clone()
        cruising_left = (self.last_yaw_sign > 0 and yaw == 0.
                         and abs(command[0] - .2) < 1e-7 and command[1] == 0.
                         and (self.restart_time is None or self.yaw_onset > self.restart_time))
        if cruising_left:
            if self.left_cruise_start is None: self.left_cruise_start = seconds
        else: self.left_cruise_start = None
        # Bound changes in the correction when pursuit emits intermittent yaw.
        # A renewed yaw command must not remove learned balance in one action.
        target_blend = float(cruising_left and seconds-self.left_cruise_start >= .2-1e-9)
        elapsed = max(0., seconds-self.previous_time) if self.previous_time is not None else 0.
        self.left_cruise_blend += max(-elapsed, min(elapsed, target_blend-self.left_cruise_blend))
        self.left_feedback_blend += max(-elapsed,min(elapsed,float(yaw>0.)-self.left_feedback_blend))
        self.age = seconds
        prior = observation[:, :63].clone()
        if self.anchor_xy is not None and not moving:
            # Preserve the certified moving prior exactly; anchors serve zero-command holds only.
            delta = torch.zeros_like(positions)
            delta[:, :2] = positions[:, :2] - self.anchor_xy
            prior[:, 60:62] = torch.bmm(rotations.transpose(1,2),delta[:,:,None])[:, :2, 0] / .03
            prior[:, 62] = 1. - (seconds - self.anchor_time) / 30.
        self.previous_time, self.previous_yaw = seconds, yaw
        self.previous_moving = moving
        return obs, prior

    def ramp(self, onset, period):
        value = min(max((self.previous_time - onset) / period, 0.), 1.) if onset is not None else 0.
        return value * value * (3. - 2. * value)

    def record(self):
        return {'anchor_xy':self.anchor_xy.detach().cpu().tolist() if self.anchor_xy is not None else [],
            'anchor_time':self.anchor_time,'simulation_reset':False,
            'dispatch':'command_driven_yaw_extension','turned':self.turned,
            'integrated_reference_rad':self.reference,'events':self.events,
            'restart_time':self.restart_time,'gait_clock_global':True,'memory_version':2,
            'left_feedback_blend':self.left_feedback_blend}


def commanded_action(base, walking, standing_actor, observation, prior_observation,
                     parameters, names, reference, turned, memory=None):
    import torch
    from tensordict import TensorDict
    from types import SimpleNamespace
    if not turned and (memory is None or memory.anchor_xy is None):
        # Exact no-turn behavior, including the original standing-age encoding.
        return base.evaluate_mean(SimpleNamespace(parameters=walking,
            standing_prior=standing_actor), observation, names)
    p = parameters.expand(observation.shape[0], -1)
    obs = observation.clone()
    heading = torch.atan2(obs[:, 68], obs[:, 69] + 1.)
    error = torch.atan2(torch.sin(heading-reference), torch.cos(heading-reference))
    obs[:, 68], obs[:, 69] = error.sin(), error.cos()-1.
    # The original damping now sees commanded-rate error.
    obs[:, 5] -= obs[:, 65]
    restart_envelope=1.
    gait_parameters=walking
    if p.shape[1]>9 and memory is not None and memory.restart_time is not None:
        elapsed=memory.previous_time-memory.restart_time
        restart_envelope=torch.where(p[:,9]<=0.,1.,(elapsed/p[:,9].clamp_min(1e-6)).clamp(0,1))
        restart_envelope=restart_envelope.square()*(3.-2.*restart_envelope)
        # Learn a phase for resuming the oscillator, without changing simulator time.
        obs[:,62]-=p[:,10]*walking[0]/(2.*torch.pi*30.)
        gait_parameters=walking.expand(len(obs),-1).clone()
        gait_parameters[:,1:4]*=restart_envelope[:,None]
    moving = base.gait_action(obs, gait_parameters, names)
    age = (1.-obs[:, 62])*30.
    phase = 2*torch.pi*(age-1.).clamp_min(0)/walking[0]
    ramp=((age-TURN_START)/walking[0]).clamp(0,1)
    ramp=ramp*ramp*(3.-2.*ramp)
    yaw_ramp=ramp
    yaw_scale=1.
    if p.shape[1]>17 and memory is not None and memory.last_yaw_sign>0:
        yaw_scale=p[:,17]
    heading_gain=p[:,2]
    if p.shape[1]>14 and memory is not None and (torch.is_tensor(memory.left_cruise_blend) or memory.left_cruise_blend > 0.):
        blend=memory.left_cruise_blend
        heading_gain=heading_gain+blend*(p[:,14]-heading_gain)
    if p.shape[1]>7 and memory is not None and memory.last_yaw_sign<0:
        yaw_scale=p[:,7]
        heading_gain=p[:,8]
    index = {name: i for i, name in enumerate(names)}
    for side, sign in (('left', 1.), ('right', -1.)):
        moving[:, index[side+'_hip_yaw_joint']] += (
            -sign*p[:, 0]*yaw_scale*obs[:, 65]*(phase+p[:, 1]).cos()*yaw_ramp+heading_gain*error*ramp)/.08
        if p.shape[1]>4:
            # Signed differential sweep; the coupled contact dynamics determine yaw sign.
            sagittal=-p[:,4]*yaw_scale*obs[:,65]*yaw_ramp*(phase+walking[14]).cos()/.08
            moving[:,index[side+'_hip_pitch_joint']]+=sagittal
            moving[:,index[side+'_ankle_pitch_joint']]-=sagittal
        if p.shape[1]>5:
            nominal=-walking[3]*(phase+walking[4]).sin()
            roll_amplitude,roll_phase=p[:,5],p[:,6]
            if p.shape[1]>13 and memory is not None and memory.last_yaw_sign<0:
                roll_amplitude,roll_phase=p[:,12],p[:,13]
            if p.shape[1]>16 and memory is not None and (torch.is_tensor(memory.left_cruise_blend) or memory.left_cruise_blend > 0.):
                blend=memory.left_cruise_blend
                roll_amplitude=roll_amplitude+blend*(p[:,15]-roll_amplitude)
                roll_phase=roll_phase+blend*(p[:,16]-roll_phase)
            adjusted=-(walking[3]+roll_amplitude*ramp)*(phase+walking[4]+roll_phase*ramp).sin()
            moving[:,index[side+'_hip_roll_joint']]+=(adjusted-nominal)*restart_envelope/.08
            if p.shape[1]>=20 and memory is not None:
                roll=torch.atan2(-obs[:,6],-obs[:,8])
                correction=(p[:,18]*roll+p[:,19]*obs[:,4]).clamp(-.03,.03)
                moving[:,index[side+'_hip_roll_joint']]+=memory.left_feedback_blend*correction/.08
    prior = standing_actor(TensorDict({'actor': prior_observation},
                                    batch_size=[len(prior_observation)]))
    hold_gain=p[:,3]
    if p.shape[1]>11 and memory is not None and memory.last_yaw_sign<0:hold_gain=p[:,11]
    for side in ('left', 'right'):
        prior[:, index[side+'_hip_yaw_joint']] += hold_gain*error/.08
    blend = torch.maximum(observation[:, 63].abs()/.2,
                          observation[:, 65].abs()/TURN_RATE).clamp(0, 1)
    return blend[:, None]*moving+(1.-blend[:, None])*prior


def heading_increment(current, previous):
    """Local yaw increment, preserving turns beyond pi when accumulated."""
    import torch
    delta=current-previous
    return torch.atan2(delta.sin(),delta.cos())


def absolute_hold_fitness(duration, final_error, hold_error, hold_drift):
    """One absolute-heading objective, including episodes failing before the hold."""
    import torch
    return 20.*duration-500.*torch.maximum(final_error,hold_error)-500.*hold_drift


def install_physics_audit(batch, batches, count):
    """Use the same 2 ms physical guards as the certified straight-gait search."""
    import torch
    import warp as wp
    from algorithms.urdf_learn_wasd_walk import landau_gait_search as audit
    dofs = wp.array([int(batch.model.joint(row['name']).dofadr[0])
                    for row in batch.spec['joints']], dtype=wp.int32, device='cuda:0')
    speed = wp.zeros(count, dtype=wp.float32, device='cuda:0')
    support = wp.zeros_like(speed); peak = wp.zeros_like(speed)
    nonfoot = wp.zeros(count, dtype=wp.int32, device='cuda:0')
    normal = wp.zeros((count, 2), dtype=wp.float32, device='cuda:0')
    current = wp.zeros_like(nonfoot); flight = wp.zeros_like(nonfoot); total = wp.zeros_like(nonfoot)
    feet = wp.from_torch(batch.contact_sides.to(torch.int32)); con = batch.wd.contact
    speed_inputs = [batch.wd.qvel, dofs, len(batch.spec['joints']), speed]
    support_inputs = [batch.wd.nacon, con.geom, con.worldid, con.efc_address,
                      con.frame, batch.wd.efc.force, feet, support, nonfoot, normal]
    occupancy = [batch.wd.ncollision, batch.wd.nacon, batch.wd.nefc, batch.maximum_occupancy]
    def audit_step():
        wp.launch(audit.record_joint_speeds, dim=count, inputs=speed_inputs)
        wp.launch(audit.clear_support, dim=count, inputs=[support, normal])
        wp.launch(audit.sum_support, dim=batch.wd.naconmax, inputs=support_inputs)
        wp.launch(audit.record_support, dim=count, inputs=[support, peak])
        wp.launch(audit.record_airborne, dim=count, inputs=[normal, current, flight, total])
        wp.launch(batches.record_occupancy, dim=count, inputs=occupancy)
    audit_step()
    with wp.ScopedCapture() as capture:
        for _ in range(10):
            batches.mjwarp.step(batch.wm, batch.wd)
            batches.mjwarp.forward(batch.wm, batch.wd)
            audit_step()
    batch.graph = capture.graph
    return {key: wp.to_torch(value) for key, value in dict(speed=speed, force=peak,
        nonfoot=nonfoot, flight=flight, flight_total=total, flight_current=current).items()}


def train(args):
    configure_profile(args.turn_duration)
    import torch
    import warp as wp
    from algorithms.urdf_learn_wasd_walk.landau_rsl_control import configure_model
    from algorithms.urdf_learn_wasd_walk.landau_forward_control import load_gait_source, make_actor, action_scales
    backend, batches, _ = configure_model('balanced_hands_v1')
    parent = Path(args.checkpoint).resolve()
    parent.relative_to((backend.OUTPUT/'training').resolve())
    parent_meta = json.loads((parent.parent/'training.json').read_text())
    if parent_meta['checkpoints'][parent.name] != backend.digest(parent):
        raise ValueError('Parent checkpoint changed')
    base, source = load_gait_source(parent, parent_meta)
    blob = torch.load(parent, map_location='cpu', weights_only=False)
    if len(blob['parameters']) != 16:
        raise ValueError('Require the certified 16-parameter walking controller')
    folder = backend.OUTPUT/'training'/args.name
    folder.mkdir(exist_ok=False)
    torch.set_num_threads(4); torch.manual_seed(42)
    batch = batches.WarpBatch(args.num_envs, 42, 0., 'stand', 'load', .004, .2, .0225)
    if hashlib.sha256(batch.xml.encode()).hexdigest() != parent_meta['model_xml_sha256']:
        raise ValueError('Physical model differs from certified parent')
    audit = install_physics_audit(batch, batches, args.num_envs)
    names = batch.spec['action_joints']; n = args.num_envs; replicas = 4
    limits = torch.tensor(action_scales(names, .4), device='cuda')/.08
    walking = blob['parameters'].to('cuda')
    prior = make_actor(63, 'cuda'); prior.load_state_dict(blob['standing_actor_state_dict'])
    prior.eval(); prior.requires_grad_(False)
    sb = None
    if args.seed_checkpoint:
        seed = Path(args.seed_checkpoint).resolve(); sm = json.loads((seed.parent/'training.json').read_text())
        if sm['checkpoints'][seed.name] != backend.digest(seed) or sm['parent_checkpoint_sha256'] != backend.digest(parent):
            raise ValueError('Turn seed provenance mismatch')
        sb = torch.load(seed, map_location='cpu', weights_only=False)
    include_feedback=args.left_feedback_only or (sb is not None and len(sb['turn_parameters'])==20)
    include_left_yaw = args.left_yaw_only or include_feedback or (sb is not None and len(sb['turn_parameters']) >= 18)
    if include_left_yaw and not args.left_cruise_only:
        raise ValueError('18-parameter seed requires the left-cruise curriculum')
    teleop=bool(getattr(args,'teleop',False))
    if teleop:
        TURN_PARAMETERS.update(right_yaw_scale=(0.,3.),right_heading_feedback_delta=(-.12,.12),
            restart_ramp_s=(0.,2.5),restart_phase_rad=(-math.pi,math.pi),right_hold_feedback=(-.12,.12))
    if getattr(args,'direction_train',False):
        TURN_PARAMETERS.update(right_roll_amplitude_offset=(-.06,.03),right_roll_phase_offset=(-.25,.25))
    if getattr(args,'left_cruise_only',False):
        TURN_PARAMETERS.update(left_cruise_heading_feedback=(-.12,.12),
            left_cruise_roll_amplitude=(-.06,.03),left_cruise_roll_phase=(-.25,.25))
    if include_left_yaw:TURN_PARAMETERS.update(left_yaw_scale=(.5,2.))
    if include_feedback:TURN_PARAMETERS.update(left_roll_feedback_delta=(-.05,.05),left_roll_rate_feedback_delta=(-.015,.015))
    lows = torch.tensor([v[0] for v in TURN_PARAMETERS.values()], device='cuda')
    highs = torch.tensor([v[1] for v in TURN_PARAMETERS.values()], device='cuda')
    rate_scale=(math.pi/28)/TURN_RATE
    initial = torch.tensor([.02*rate_scale, -.3, .0514, -.04, -.0343*rate_scale,0.,0.], device='cuda')
    if teleop:initial=torch.cat((initial,initial.new_tensor([1.,.0514,0.,0.,-.04])))
    if getattr(args,'direction_train',False):initial=torch.cat((initial,initial.new_zeros(2)))
    if getattr(args,'left_cruise_only',False):initial=torch.cat((initial,initial[2:3],initial[5:7]))
    if include_left_yaw:initial=torch.cat((initial,initial.new_ones(1)))
    if include_feedback:initial=torch.cat((initial,initial.new_zeros(2)))
    mean = 2*(initial-lows)/(highs-lows)-1.; std = torch.full_like(mean, args.search_std)
    if args.seed_checkpoint:
        seed_parameters=sb['turn_parameters'].to('cuda')
        if args.seed_candidate is not None:
            candidate_path,expected_table=candidate_table_reference(seed,sm)
            if expected_table and backend.digest(candidate_path)!=expected_table:
                raise ValueError('Candidate table changed')
            candidate_table=json.loads(candidate_path.read_text())
            selected=next(row for row in candidate_table if row['candidate']==args.seed_candidate)
            seed_parameters=torch.tensor(selected['parameters'],device='cuda')
        if len(seed_parameters)<len(TURN_PARAMETERS):
            seed_parameters=torch.cat((seed_parameters,seed_parameters.new_zeros(len(TURN_PARAMETERS)-len(seed_parameters))))
        if teleop and len(sb['turn_parameters'])==7 and args.seed_candidate is None:
            seed_parameters[7]=1.;seed_parameters[8]=seed_parameters[2]
        if teleop and len(sb['turn_parameters'])<12:seed_parameters[11]=seed_parameters[3]
        if getattr(args,'direction_train',False) and len(sb['turn_parameters'])<14:
            seed_parameters[12:14]=seed_parameters[5:7]
        if getattr(args,'left_cruise_only',False) and len(sb['turn_parameters'])<15:
            seed_parameters[14]=seed_parameters[2]
        if getattr(args,'left_cruise_only',False) and len(sb['turn_parameters'])<17:
            seed_parameters[15:17]=seed_parameters[5:7]
        if include_left_yaw and len(sb['turn_parameters'])<18:seed_parameters[17]=1.
        if args.left_checkpoint:
            left_checkpoint=Path(args.left_checkpoint).resolve()
            left_meta=json.loads((left_checkpoint.parent/'training.json').read_text())
            if left_meta['checkpoints'][left_checkpoint.name]!=backend.digest(left_checkpoint) or left_meta['parent_checkpoint_sha256']!=backend.digest(parent):
                raise ValueError('Left checkpoint provenance mismatch')
            left_blob=torch.load(left_checkpoint,map_location='cpu',weights_only=False)
            seed_parameters[:7]=left_blob['turn_parameters'][:7].to('cuda')
        if args.preserve_yaw_amplitude:
            seed_parameters[[0,4]]*=sm['command_profile']['yaw_rate_rad_s']/TURN_RATE
        if ((seed_parameters<lows)|(seed_parameters>highs)).any():raise ValueError('Seed parameters outside search bounds')
        mean = 2*(seed_parameters-lows)/(highs-lows)-1.
    teleop=bool(getattr(args,'teleop',False))
    if teleop:
        from algorithms.urdf_learn_wasd_walk import landau_teleop_contract as teleop_contract
    direction_train=bool(getattr(args,'direction_train',False))
    closed_direction=args.closed_loop_direction
    if direction_train:
        from algorithms.urdf_learn_wasd_walk import landau_direction_contract as direction_contract
        if closed_direction:
            from types import SimpleNamespace
            from algorithms.urdf_learn_wasd_walk import landau_direction_training as direction_training
            training_protocol=SimpleNamespace(BLOCKS=(),HOLDS=(),YAW_RATE=direction_contract.MAX_YAW_RATE)
        else:training_protocol=direction_contract.TurnTrainingProtocol(args.seconds, 'left' if args.left_cruise_only else 'right')
        if args.left_yaw_only or args.turn_hold_training:training_protocol=direction_contract.TurnHoldTrainingProtocol(args.seconds,args.turn_duration)
        if args.direction_command_trace:
            import numpy as np
            command_trace=Path(args.direction_command_trace).resolve()
            command_trace.relative_to(backend.OUTPUT.resolve())
            with np.load(command_trace,allow_pickle=False) as recorded:
                times=recorded['time'];commands=recorded['observation'][:,63:66].copy()
            if len(times)!=len(commands) or not np.allclose(times,np.arange(len(times))*.02,rtol=0,atol=1e-6):
                raise ValueError('Recorded command clock is incomplete or reset')
            training_protocol=direction_contract.RecordedLeftTrainingProtocol(commands,args.seconds)
            np.savez_compressed(folder/'direction_commands.npz',time=times,command=commands)
    else:training_protocol=teleop_contract if teleop else None
    if args.common_starts:
        from algorithms.urdf_learn_wasd_walk import landau_direction_training as direction_training
    common_offsets=direction_training.common_start_offsets(batch.nominal,batch.jq,replicas,args.start_seed) if args.common_starts else None
    grid_center=mean.clone()
    body_weight = float(batch.model.body_mass.sum())*9.81
    meta = dict(parent_meta)
    for key in ('forward_fitness_contract','lateral_acceptance','initial_search_parameters','metrics','wall_s',
                'generations_completed','search_seed_checkpoint'):
        meta.pop(key,None)
    meta.update(created_at=datetime.now(timezone.utc).isoformat(), arguments={**vars(args),'hip_roll_range':.4},
        status='running', stop_reason=None,
        command_extension='yaw_v1', policy_composition='Frozen trained gait and standing MLP; CEM-trained yaw command and hold feedback',
        parent_checkpoint=str(parent), parent_checkpoint_sha256=backend.digest(parent),
        source_sha256={**parent_meta['source_sha256'], str(Path(__file__)): backend.digest(__file__)},
        turn_source_sha256=backend.digest(__file__), turn_parameter_ranges=TURN_PARAMETERS,
        command_profile={'turn_start_s':TURN_START,'turn_end_s':TURN_END,'hold_start_s':HOLD_START,
                         'yaw_rate_rad_s':TURN_RATE,'target_heading_rad':math.pi/2},
        controller_memory=f'Latch standing XY and age once at {HOLD_START:g}s; never reset simulation state or clock.',
        research_context={'reference':'https://arxiv.org/abs/2407.17683',
            'application':'Motivates adapting foot placement for turning and whole-body balance. This implementation searches bounded joint-space corrections; it is not the paper MPC/RL algorithm.'},
        objective='Turn 90 degrees and settle; zero falls, resets, nonfoot contacts and physical-limit violations.',
        failure_bitmask={'tilt':1,'height_drop':2,'joint_speed':4,'support_force':8,'nonfoot':16,'flight':32,'nonfinite':64},
        turn_seed_checkpoint=args.seed_checkpoint,
        canonical_milestone_pass=False, checkpoints={})
    if args.left_checkpoint:
        meta['frozen_left_checkpoint']={'path':str(left_checkpoint),'sha256':backend.digest(left_checkpoint)}
    if args.seed_candidate is not None:
        meta['seed_candidate_record']={'path':str(candidate_path),'sha256':backend.digest(candidate_path),'candidate':args.seed_candidate}
    if teleop:
        meta.update(memory_version=2,objective='60s scripted forward/left/stop/restart/right/stop joystick response',
            controller_memory='Command-integrated heading; re-anchor each all-zero stop; preserve certified moving-prior and global gait/ramp behavior; no simulator reset.')
        meta['teleop_contract_sha256']=backend.digest(teleop_contract.__file__)
        (folder/'teleop_contract.py').write_text(Path(teleop_contract.__file__).read_text())
    if direction_train:
        meta.update(objective='M7 sustained right90degree turn then forward walking; frozen M1-M5 parameters',direction_training_protocol_sha256=backend.digest(direction_contract.__file__))
        if args.left_cruise_only: meta['objective']='M7 left turn then straight: search left cruising heading and lateral sway feedback, preserving prior 14 parameters'
        if args.left_turn_refine:
            meta['objective']='M7 left active-turn and cruising balance refinement; exact M5 recheck required'
            meta['left_turn_refinement']={'searched_indices':[2,5,6,14,15,16],
                'post_turn_force_target_bw':2.85,'full_run_hard_force_limit_bw':3.,
                'force_margin_penalty_per_bw':1000.,
                'reference':'https://github.com/leggedrobotics/legged_gym/blob/master/legged_gym/envs/base/legged_robot.py',
                'application':'Bounded excess-contact-force penalty adapted to worst-replica post-turn peak, keeping the full-run hard guard unchanged.'}
        if args.left_yaw_only:
            meta['objective']='Calibrate independent left yaw scale on exact M5 commands; freeze previous17turn parameters and gait/standing weights'
        if args.left_heading_grid:
            meta['objective']='Diagnostic cruise heading gain grid; freeze all other turn, gait and standing parameters'
            meta['heading_gain_note']='Full cruise net hip-yaw heading gain is walking[12]+turn[14]; diagnose sign response without assuming contact dynamics.'
        if args.left_balance_only:
            meta['objective']='Refine active-left sway amplitude/phase with force margin; preserve yaw calibration and cruise feedback; exact M5 recheck required'
            meta['left_turn_refinement']['searched_indices']=[5,6]
        if args.left_cruise_balance_only:
            meta['left_turn_refinement']['searched_indices']=[15,16]
            meta['cruise_balance_scope']='Search post-left-turn sway amplitude/phase only; all other seed parameters restored bit-exactly after denormalization.'
        if closed_direction:
            meta['objective']='Per-world closed-loop world gate, unchanged physics limits, force margin and completion time; separate serial proof required'
            meta['closed_loop_direction']=closed_direction
            meta['direction_training_source_sha256']=backend.digest(direction_training.__file__)
            meta['training_termination']='Freeze metrics at valid gate crossing; reset terminal worlds only after recording outcome; never reactivate them.'
            (folder/'direction_training_source.py').write_text(Path(direction_training.__file__).read_text())
        if args.common_starts:
            meta['common_start_offsets_qpos']=common_offsets.cpu().tolist()
            meta['direction_training_source_sha256']=backend.digest(direction_training.__file__)
            (folder/'direction_training_source.py').write_text(Path(direction_training.__file__).read_text())
            meta['common_starts_reference']='https://pubsonline.informs.org/doi/10.1287/mnsc.45.11.1570'
            meta['common_starts_scope']='Same nominal plus3joint perturbations for every candidate/generation; does not claim deterministic physics.'
        if args.left_balance_grid:
            meta['balance_grid']={'center_normalized':grid_center.cpu().tolist(),'radius_normalized':args.search_std,
                'unchanged_control_candidates':[0,1],'repeat_same_grid_each_generation':True}
        if include_feedback:
            meta['left_roll_feedback']={'searched_indices':[18,19] if args.left_feedback_only else [],
                'maximum_residual_rad':.03,'command_envelope_rate_per_s':1.,
                'references':['https://arxiv.org/abs/2103.15309','https://arxiv.org/abs/1812.03201'],
                'scope':'Add bounded torso roll/rate correction during left commands, smoothly decay afterward; freeze gait/standing and all turn parameters outside the declared search indices. This changes approach/load transfer at50Hz, not impact forces at500Hz.'}
            if args.left_feedback_only:meta['left_turn_refinement']['searched_indices']=[18,19]
        if args.left_damping_grid:
            meta['damping_grid']={'unchanged_control_candidates':[0,1],'repeat_same_grid_each_generation':True}
            meta['left_roll_feedback']['searched_indices']=[19]
            meta['left_turn_refinement']['searched_indices']=[19]
        if args.turn_hold_training:
            meta['objective']='Shared-start exact90degree turn/hold screening for left feedback with force margin; preserve previous18parameters; later M7 gate proof required'
            meta['turn_hold_fitness']='20*duration -500*max(final_absolute_heading_error,settled_absolute_heading_error) -500*hold_drift; retain force-margin penalty and physics/eligibility bonuses. No relative turn-onset yaw reward.'
        if args.feedback_yaw_refine:
            meta['objective']='Jointly calibrate left yaw scale and torso feedback on exact90degree turn/hold with force margin; preserve first17parameters and gait; later M7 proof required'
            meta['left_turn_refinement']['searched_indices']=[17,18,19]
        if args.direction_command_trace:
            meta['direction_commands']={'source_path':str(command_trace),'source_sha256':backend.digest(command_trace),
                'commands_sha256':backend.digest(folder/'direction_commands.npz'),
                'scope':'Semantic commands only; no pose/action/state replay; padded with forward .2 and zero yaw',
                'response_blocks':training_protocol.BLOCKS}
        (folder/'direction_training_protocol.py').write_text(Path(direction_contract.__file__).read_text())
    (folder/'control_source.py').write_text(source.read_text())
    (folder/'turn_source.py').write_text(Path(__file__).read_text())
    (folder/'model.xml').write_text(batch.xml)
    backend.write_json(folder/'metadata.json', meta)
    def progress(state, generation):
        path=backend.OUTPUT.parent/'backend_progress.json'
        record=json.loads(path.read_text()) if path.exists() else {}
        record.update(updated_at=datetime.now(timezone.utc).isoformat(),current_gate='gate_10m_four_directions_no_reset' if direction_train else 'teleop_60s_forward_turn' if teleop else 'yaw_turn_90deg_hold',
            simulator='mujoco_warp_cuda',variant='balanced_hands_v1',iteration=generation,
            active_process=args.name if state=='running' else None,artifact_paths=[str(folder)],
            next_step='Train bounded yaw feedback, then independently evaluate turn and hold with video.',
            assistance_coefficient=0.)
        backend.write_json(path,record)
    progress('running',0)
    began = time.perf_counter(); best_z = mean.clone()
    for generation in range(args.generations):
        candidates = n//replicas
        z = (mean+std*torch.randn(candidates, len(TURN_PARAMETERS), device='cuda')).clamp(-1, 1); z[0] = best_z
        if getattr(args,'right_only',False):
            z[:,:7]=mean[:7];z[:,9:12]=mean[9:12]
        if getattr(args,'left_cruise_only',False):
            fixed=[i for i in range(14) if not args.left_turn_refine or i not in (2,5,6)]
            z[:,fixed]=mean[fixed]
            if include_left_yaw and not args.left_yaw_only and not args.feedback_yaw_refine:z[:,17]=mean[17]
        if args.left_balance_only:
            fixed=[i for i in range(len(TURN_PARAMETERS)) if i not in (5,6)]
            z[:,fixed]=mean[fixed]
        if args.left_cruise_balance_only:
            fixed=[i for i in range(len(TURN_PARAMETERS)) if i not in (15,16)]
            z[:,fixed]=grid_center[fixed]
        if args.left_balance_grid:
            z=direction_training.balance_grid(grid_center,candidates,args.search_std)
        if include_feedback and not args.left_feedback_only:z[:,18:20]=mean[18:20]
        if args.left_feedback_only:
            frozen_count=17 if args.feedback_yaw_refine else 18
            z[:,:frozen_count]=grid_center[:frozen_count]
        if args.left_damping_grid:
            if candidates!=8:raise ValueError('Damping grid requires8candidates with4replicas')
            z[:]=grid_center
            values=z.new_tensor([float(initial[19]),float(initial[19]),-.015,-.010,-.005,.005,.010,.015])
            if args.seed_checkpoint:values[:2]=seed_parameters[19]
            z[:,19]=2*(values-lows[19])/(highs[19]-lows[19])-1.
        if args.left_heading_grid:
            if candidates!=8:raise ValueError('Heading grid requires8candidates with4replicas')
            z[:]=mean
            values=z.new_tensor([float(seed_parameters[14]),.039,.03662172,.035,.03,.02,0.,-.02])
            z[:,14]=2*(values-lows[14])/(highs[14]-lows[14])-1.
        if args.left_yaw_only:
            z[:,:17]=mean[:17]
            if generation==0:
                z[:,17]=torch.linspace(-1.,1.,candidates,device='cuda');z[0]=mean
        if getattr(args,'restart_only',False):z[:,:9]=mean[:9]
        if getattr(args,'teleop_refine',False):
            fixed=list(range(7));z[:,fixed]=mean[fixed]
        if getattr(args,'right_grid',False):
            side=math.isqrt(candidates)
            if side*side!=candidates:raise ValueError('Right grid requires a square candidate count')
            z[:,7]=torch.linspace(-1.,1.,side,device='cuda').repeat_interleave(side)
            z[:,8]=torch.linspace(-1.,1.,side,device='cuda').repeat(side)
        if getattr(args,'restart_grid',False):
            side=math.isqrt(candidates)
            if side*side!=candidates:raise ValueError('Restart grid requires a square candidate count')
            z[:,9]=torch.linspace(-1.,1.,side,device='cuda').repeat_interleave(side)
            z[:,10]=torch.linspace(-1.,1.,side,device='cuda').repeat(side)
        if getattr(args,'right_grid',False) or getattr(args,'restart_grid',False):z[0]=mean
        params = (lows+(z+1)*.5*(highs-lows)).repeat_interleave(replicas, dim=0)
        if args.left_cruise_balance_only:
            params=direction_training.cruise_balance_parameters(z,lows,highs,seed_parameters).repeat_interleave(replicas,dim=0)
        if args.left_feedback_only:params[:,:frozen_count]=seed_parameters[:frozen_count]
        if args.left_damping_grid:
            params[:,18]=seed_parameters[18]
            params[:2*replicas,18:20]=seed_parameters[18:20]
        if args.diagnostic_sweep:
            diagnostic=torch.zeros((candidates,len(TURN_PARAMETERS)),device='cuda')
            if args.diagnostic_grid:
                side=math.isqrt(candidates)
                if side*side!=candidates:raise ValueError('Grid requires a square candidate count')
                diagnostic[:,0]=torch.linspace(0.,.08,side,device='cuda').repeat_interleave(side)
                diagnostic[:,2]=torch.linspace(float(lows[2]),float(highs[2]),side,device='cuda').repeat(side)
                if args.stride_grid:
                    diagnostic[:,0]=.02
                    diagnostic[:,2]=torch.linspace(0.,.12,side,device='cuda').repeat(side)
                    diagnostic[:,4]=torch.linspace(-.12,-.02,side,device='cuda').repeat_interleave(side)
            else:diagnostic[:,0]=torch.linspace(0.,.3,candidates,device='cuda')
            diagnostic[:,1]=-.3;diagnostic[:,3]=-.04
            z=2*(diagnostic-lows)/(highs-lows)-1.
            params=diagnostic.repeat_interleave(replicas,dim=0)
        batch.reset(torch.ones(n, device='cuda', dtype=torch.bool)); batch.q[::replicas] = batch.nominal
        if common_offsets is not None:
            batch.q[:]=batch.nominal+common_offsets.repeat(candidates,1)
        torch.cuda.synchronize(); wp.capture_launch(batch.forward_graph); wp.synchronize()
        for value in audit.values(): value.zero_()
        alive = torch.ones(n, device='cuda', dtype=torch.bool)
        failure_bits=torch.zeros(n,device='cuda',dtype=torch.int32)
        duration = torch.zeros(n, device='cuda'); yaw_error = torch.full_like(duration, math.pi/2)
        hold_drift = torch.zeros_like(duration); hold_error = torch.zeros_like(duration)
        force_max = torch.zeros_like(duration); speed_max = torch.zeros_like(duration)
        post_turn_force_max = torch.zeros_like(duration)
        heading_final = torch.zeros_like(duration); anchor = None
        heading_start=torch.zeros_like(duration);heading_window=torch.zeros_like(duration)
        heading_turn_end=torch.zeros_like(duration);heading_turn_window=torch.zeros_like(duration)
        reached_turn_end=torch.zeros(n,device='cuda',dtype=torch.bool)
        counts = torch.zeros((n,2), device='cuda', dtype=torch.long)
        air = torch.zeros((n,2), device='cuda'); peak = torch.zeros_like(air)
        touched = torch.zeros_like(air, dtype=torch.bool)
        memory=direction_training.DirectionMemory(n,'cuda',closed_direction) if closed_direction else CommandMemory() if teleop else None
        gate=direction_training.GateTracker(closed_direction,batch.pos[:,batch.pelvis]) if closed_direction else None
        terminal_flight=torch.zeros_like(duration);slip_total=torch.zeros_like(duration)
        failure_time=torch.full_like(duration,float('nan'))
        block_records=[];block_start={};block_progress={};block_rotation={}
        hold_heading=None
        for step in range(round(args.seconds/.02)):
            age = step*.02; forward, yaw, reference = command_profile(age)
            active=alive & ~gate.completed if closed_direction else alive
            if args.left_turn_refine and step==round(TURN_START/.02):
                # force_max retains the complete episode; audit now accumulates
                # a separate post-onset peak for the stricter training margin.
                audit['force'].zero_()
            if closed_direction:
                rotation=batch.rot[:,batch.base].to(torch.float64)
                current_heading=torch.atan2(rotation[:,1,0],rotation[:,0,0])
                world_commands=direction_training.commands(closed_direction,
                    batch.pos[:,batch.pelvis,:2].to(torch.float64)-gate.origins,current_heading,age)
                forward=world_commands[:,0].float();yaw=world_commands[:,2].float()
            elif teleop:forward,_,yaw=training_protocol.command_profile(age)
            if step==round(TURN_START/.02):
                heading_start=torch.atan2(batch.rot[:,batch.base,1,0],batch.rot[:,batch.base,0,0]).clone()
            if step==round((args.seconds-2.)/.02):
                heading_window=torch.atan2(batch.rot[:,batch.base,1,0],batch.rot[:,batch.base,0,0]).clone()
            if step==round((TURN_END-2.)/.02):
                heading_turn_window=torch.atan2(batch.rot[:,batch.base,1,0],batch.rot[:,batch.base,0,0]).clone()
            if step==round(TURN_END/.02):
                heading_turn_end=torch.atan2(batch.rot[:,batch.base,1,0],batch.rot[:,batch.base,0,0]).clone()
                reached_turn_end=alive.clone()
            extra = torch.zeros((n,10), device='cuda')
            extra[:,2] = 1.-age/30.; extra[:,3] = forward; extra[:,5] = yaw
            extra[:,6] = math.sin(2*math.pi*age/.6); extra[:,7] = math.cos(2*math.pi*age/.6)
            extra[:,8] = batch.rot[:,batch.base,1,0]; extra[:,9] = batch.rot[:,batch.base,0,0]-1.
            obs = torch.cat((batch.observations(),extra),dim=1)
            prior_obs = obs[:,:63].clone(); prior_obs[:,60:62] = 0.
            if not teleop and age >= HOLD_START:
                if anchor is None: anchor = batch.pos[:,batch.pelvis,:2].clone()
                delta = torch.zeros((n,3),device='cuda'); delta[:,:2] = batch.pos[:,batch.pelvis,:2]-anchor
                prior_obs[:,60:62] = torch.bmm(batch.rot[:,batch.base].transpose(1,2),delta[:,:,None])[:,:2,0]/.03
                prior_obs[:,62] = 1.-(age-HOLD_START)/30.
            if teleop:
                if closed_direction:
                    obs,prior_obs=memory.observe(obs,batch.pos[:,batch.pelvis],batch.rot[:,batch.base],age,active)
                else:obs,prior_obs=memory.observe(obs,batch.pos[:,batch.pelvis],batch.rot[:,batch.base],age)
                reference=memory.reference
                anchor=memory.anchor_xy
                previous_xy=batch.pos[:,batch.pelvis,:2].clone()
                previous_heading=torch.atan2(batch.rot[:,batch.base,1,0],batch.rot[:,batch.base,0,0]).clone()
                for label,start,end,sign in training_protocol.BLOCKS:
                    if step==round(start/.02):
                        block_start[label]=(previous_heading.clone(),counts.clone())
                        block_progress[label]=torch.zeros_like(duration)
                        block_rotation[label]=torch.zeros_like(duration)
            with torch.no_grad():
                action = commanded_action(base,walking,prior,obs,prior_obs,params,names,reference,memory.turned if teleop else age>=TURN_START,memory=memory).clamp(-limits,limits)
            batch.ctrl[:] = batch.targets; batch.ctrl[:,batch.aids] += .08*torch.where(active[:,None],action,0.)
            batch.wrench.zero_(); torch.cuda.synchronize(); wp.capture_launch(batch.graph); wp.synchronize()
            occupancy = batch.maximum_occupancy.numpy()
            if max(occupancy[:2])>=batch.wd.naconmax or occupancy[2]>=batch.wd.njmax:
                raise RuntimeError('Physics contact capacity exceeded')
            tilt = batch.rot[:,batch.base,2,2].clamp(-1,1).acos()
            failed = (tilt>math.pi/6)|(batch.pos[:,batch.pelvis,2]<batch.reference_height-.08)
            failed |= (~torch.isfinite(batch.q).all(1))|(~torch.isfinite(batch.v).all(1))
            failed |= (audit['speed']>4.+1e-6)|(audit['force']>3.*body_weight)|(audit['nonfoot']!=0)|(audit['flight']>60)
            force_max = torch.maximum(force_max,torch.where(active,audit['force']/body_weight,0.))
            if args.left_turn_refine and age>=TURN_START:
                post_turn_force_max=torch.maximum(post_turn_force_max,torch.where(active,audit['force']/body_weight,0.))
            speed_max = torch.maximum(speed_max,torch.where(active,audit['speed'],0.))
            newly = active & failed; alive &= ~newly
            duration += (active & ~newly).float()*.02
            failure_time=torch.where(newly,age+.02,failure_time)
            if closed_direction:terminal_flight=torch.where(active,audit['flight_total'].float(),terminal_flight)
            bits=(tilt>math.pi/6).int()+2*(batch.pos[:,batch.pelvis,2]<batch.reference_height-.08).int()
            bits+=4*(audit['speed']>4.+1e-6).int()+8*(audit['force']>3.*body_weight).int()
            bits+=16*(audit['nonfoot']!=0).int()+32*(audit['flight']>60).int()
            bits+=64*((~torch.isfinite(batch.q).all(1))|(~torch.isfinite(batch.v).all(1))).int()
            failure_bits=torch.where(newly,bits,failure_bits)
            heading = torch.atan2(batch.rot[:,batch.base,1,0],batch.rot[:,batch.base,0,0])
            target=reference if teleop else math.pi/2
            error = torch.atan2(torch.sin(heading-target),torch.cos(heading-target)).abs()
            measuring=active & alive
            yaw_error = torch.where(measuring,error,yaw_error); heading_final = torch.where(measuring,heading,heading_final)
            contacts,slip,clearance,_ = batch.feet(with_forces=True)
            counts += (contacts&touched&(air>=.06)&(peak>=.015)&measuring[:,None]).long()
            slip_total+=torch.where(measuring,slip,0.)*.02
            air = torch.where(contacts,0.,air+.02); peak = torch.where(contacts,0.,torch.maximum(peak,clearance)); touched |= contacts
            if teleop:
                delta=batch.pos[:,batch.pelvis,:2]-previous_xy
                forward_delta=-delta[:,0]*previous_heading.sin()+delta[:,1]*previous_heading.cos()
                for label,start,end,sign in training_protocol.BLOCKS:
                    if start<=age<end:
                        block_progress[label]+=torch.where(alive,forward_delta,0.)
                        block_rotation[label]+=torch.where(alive,heading_increment(heading,previous_heading),0.)
                    if step+1==round(end/.02):
                        _,count_start=block_start[label]
                        change=block_rotation[label].clone()
                        block_records.append((label,block_progress[label].clone(),change,counts-count_start,sign,end-start))
                for start,end in training_protocol.HOLDS:
                    if step==round(start/.02):hold_heading=previous_heading.clone()
                    if start<=age<end:
                        drift=(batch.pos[:,batch.pelvis,:2]-anchor).norm(dim=1)
                        herror=torch.atan2((heading-hold_heading).sin(),(heading-hold_heading).cos()).abs()
                        if args.left_yaw_only or args.turn_hold_training:
                            herror=torch.atan2((heading-math.pi/2).sin(),(heading-math.pi/2).cos()).abs()
                        hold_drift=torch.maximum(hold_drift,torch.where(alive,drift,0.))
                        hold_error=torch.maximum(hold_error,torch.where(alive,herror,0.))
            elif age >= HOLD_START:
                drift = (batch.pos[:,batch.pelvis,:2]-anchor).norm(dim=1)
                hold_drift = torch.maximum(hold_drift,torch.where(alive,drift,0.))
            if not teleop and age >= args.seconds-5.:
                hold_error = torch.maximum(hold_error,torch.where(alive,error,0.))
            batch.previous = action
            finished=gate.update(batch.pos[:,batch.pelvis],age+.02,measuring) if closed_direction else torch.zeros_like(newly)
            if (newly|finished).any(): batch.reset(newly|finished)
            if not (alive & ~gate.completed if closed_direction else alive).any():break
        physics = alive & (terminal_flight<=duration/.002*.05) if closed_direction else alive & (audit['flight_total']<=round(args.seconds/.002)*.05)
        eligible_world = physics & (hold_error<=math.radians(5)) & (hold_drift<=.03) & (counts.min(1).values>=3)
        block_reward=torch.zeros_like(duration)
        if teleop and not closed_direction:
            if len(block_records)!=len(training_protocol.BLOCKS):eligible_world[:]=False
            for label,progress_value,change,swings,sign,length in block_records:
                eligible_world &= (progress_value>=.04*length)&(swings.min(1).values>=2)
                block_reward+=50.*progress_value.clamp(0.,.2*length)
                if sign:
                    target=training_protocol.YAW_RATE*length
                    response=sign*change
                    eligible_world &= (response>=.5*target)&(response<=1.5*target)
                    block_reward-=500.*(response-target).abs()
                elif direction_train and args.left_cruise_only:
                    eligible_world &= change.abs() <= math.radians(15)
                    block_reward -= 500.*change.abs()
            if direction_train and args.left_cruise_only:
                eligible_world &= yaw_error <= math.radians(15)
        fitness = 20.*duration-100.*yaw_error-500.*hold_drift-100.*hold_error+block_reward
        if args.turn_hold_training:
            # M5 requires an absolute final heading. Relative yaw from turn onset
            # conflicts with that goal when pre-turn drift or stopping changes yaw.
            fitness=absolute_hold_fitness(duration,yaw_error,hold_error,hold_drift)
        if closed_direction:
            eligible_world=physics & gate.completed & (counts.min(1).values>=3) & (slip_total/duration.clamp_min(.02)<=.1)
            fitness=20.*duration+100.*gate.progress.float().clamp(-10.,10.)-20.*gate.lateral.float().abs()
            fitness+=10000.*gate.completed.float()-21.*torch.where(gate.completed,duration,0.)
        if args.left_turn_refine:
            eligible_world &= post_turn_force_max<=2.85
            fitness-=1000.*(post_turn_force_max-2.85).clamp_min(0.)
        fitness += 1000.*physics.float()+10000.*eligible_world.float()
        score = fitness.reshape(candidates,replicas).min(1).values
        eligible = eligible_world.reshape(candidates,replicas).all(1)
        elite = score.topk(max(2,candidates//8)).indices; best = int(elite[0]); best_z = z[best].clone()
        mean = .25*mean+.75*z[elite].mean(0); std = (.25*std+.75*z[elite].std(0,unbiased=False)).clamp_min(.035)
        members = slice(best*replicas,(best+1)*replicas)
        metrics = {'generation':generation,'wall_s':time.perf_counter()-began,'eligible':bool(eligible[best]),
            'eligible_candidate_count':int(eligible.sum()),'survival_s':duration[members].tolist(),
            'heading_rad':heading_final[members].tolist(),'final_heading_error_rad':yaw_error[members].tolist(),
            'hold_max_heading_error_rad':hold_error[members].tolist(),'hold_max_drift_m':hold_drift[members].tolist(),
            'max_joint_speed_rad_s':float(speed_max[members].max()),'peak_support_body_weight_ratio':float(force_max[members].max()),
            'completed_swings':counts[members].tolist(),'parameters':dict(zip(TURN_PARAMETERS,params[best*replicas].tolist()))}
        metrics['failure_bits']=failure_bits[members].tolist()
        if closed_direction:
            metrics['world_gates']=gate.record(members)
            metrics['failure_time_s']=[v if math.isfinite(v) else None for v in failure_time[members].tolist()]
        if args.left_turn_refine:metrics['post_turn_peak_support_bw']=post_turn_force_max[members].tolist()
        if teleop:
            metrics['command_blocks']=[{'label':label,'forward_m':progress_value[members].tolist(),
                'heading_change_rad':change[members].tolist(),'swings':swings[members].tolist()}
                for label,progress_value,change,swings,sign,length in block_records]
            metrics['controller_memory']=memory.record()
            metrics['controller_memory']['anchor_xy']=metrics['controller_memory']['anchor_xy'][members]
            if closed_direction:
                for key in ('integrated_reference_rad','left_cruise_blend','left_feedback_blend'):
                    metrics['controller_memory'][key]=metrics['controller_memory'][key][members]
        yaw_change=torch.atan2((heading_final-heading_start).sin(),(heading_final-heading_start).cos())
        late_rate=torch.atan2((heading_final-heading_window).sin(),(heading_final-heading_window).cos())/2.
        turn_rate=torch.atan2((heading_turn_end-heading_turn_window).sin(),(heading_turn_end-heading_turn_window).cos())/2.
        def observed(values,valid):return [v if ok else None for v,ok in zip(values.tolist(),valid.tolist())]
        metrics.update(turn_heading_change_rad=yaw_change[members].tolist(),
            last_two_seconds_mean_yaw_rate=observed(late_rate[members],alive[members]),
            turn_end_heading_rad=observed(heading_turn_end[members],reached_turn_end[members]),
            last_two_turn_seconds_mean_yaw_rate=observed(turn_rate[members],reached_turn_end[members]))
        if closed_direction:
            for key in ('last_two_seconds_mean_yaw_rate','turn_end_heading_rad','last_two_turn_seconds_mean_yaw_rate'):
                metrics.pop(key,None)
        with (folder/'generations.jsonl').open('a') as stream: stream.write(json.dumps(metrics)+'\n')
        if teleop and (getattr(args,'right_grid',False) or getattr(args,'restart_grid',False) or getattr(args,'left_cruise_only',False)):
            backend.write_json(folder/'candidates.json',[{'candidate':i,
                'parameters':params[i*replicas].tolist(),'survival_s':duration[i*replicas:(i+1)*replicas].tolist(),
                'failure_bits':failure_bits[i*replicas:(i+1)*replicas].tolist(),
                'failure_time_s':[v if math.isfinite(v) else None for v in failure_time[i*replicas:(i+1)*replicas].tolist()],
                'world_gates':gate.record(slice(i*replicas,(i+1)*replicas)) if closed_direction else None,
                'eligible':bool(eligible[i]),'score':float(score[i]),
                'heading_rad':heading_final[i*replicas:(i+1)*replicas].tolist(),
                'final_heading_error_rad':yaw_error[i*replicas:(i+1)*replicas].tolist(),
                'hold_drift_m':hold_drift[i*replicas:(i+1)*replicas].tolist(),
                'hold_max_heading_error_rad':hold_error[i*replicas:(i+1)*replicas].tolist(),
                'peak_support_bw':force_max[i*replicas:(i+1)*replicas].tolist(),
                'post_turn_peak_support_bw':post_turn_force_max[i*replicas:(i+1)*replicas].tolist() if args.left_turn_refine else None,
                'blocks':[{'label':label,'forward_m':progress_value[i*replicas:(i+1)*replicas].tolist(),
                    'heading_change_rad':change[i*replicas:(i+1)*replicas].tolist(),
                    'swings':swings[i*replicas:(i+1)*replicas].tolist()}
                    for label,progress_value,change,swings,sign,length in block_records]} for i in range(candidates)])
        if args.diagnostic_sweep:
            backend.write_json(folder/'yaw_sweep.json',[
                {'parameters':dict(zip(TURN_PARAMETERS,params[i*replicas].tolist())),
                 'survival_s':duration[i*replicas:(i+1)*replicas].tolist(),
                 'heading_rad':heading_final[i*replicas:(i+1)*replicas].tolist(),
                 'physics_pass':physics[i*replicas:(i+1)*replicas].tolist(),
                 'failure_bits':failure_bits[i*replicas:(i+1)*replicas].tolist(),
                 'turn_heading_change_rad':yaw_change[i*replicas:(i+1)*replicas].tolist(),
                 'last_two_seconds_mean_yaw_rate':observed(late_rate[i*replicas:(i+1)*replicas],alive[i*replicas:(i+1)*replicas]),
                 'peak_support_bw':force_max[i*replicas:(i+1)*replicas].tolist(),
                 'max_joint_speed':speed_max[i*replicas:(i+1)*replicas].tolist()}
                for i in range(candidates)])
        if (folder/'candidates.json').exists():
            (folder/f'candidates_{generation}.json').write_bytes((folder/'candidates.json').read_bytes())
        torch.save({'parameters':blob['parameters'],'turn_parameters':params[best*replicas].detach().cpu(),
                    'standing_actor_state_dict':blob['standing_actor_state_dict']},folder/f'model_{generation}.pt')
        print(json.dumps(metrics),flush=True)
        progress('running',generation+1)
    meta.update(metrics=metrics,wall_s=time.perf_counter()-began,generations_completed=args.generations,
                status='completed',stop_reason='generation_budget_exhausted',
                checkpoints={p.name:backend.digest(p) for p in folder.glob('model_*.pt')})
    if (folder/'candidates.json').exists():meta['candidate_table_sha256']=backend.digest(folder/'candidates.json')
    meta['generation_candidate_tables']={p.name:backend.digest(p) for p in folder.glob('candidates_*.json')}
    backend.write_json(folder/'training.json',meta)
    progress('completed',args.generations)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name',required=True); parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--num-envs',type=int,default=256); parser.add_argument('--generations',type=int,default=8)
    parser.add_argument('--seconds',type=float,default=26.); parser.add_argument('--seed-checkpoint')
    parser.add_argument('--turn-duration',type=float,default=14.)
    parser.add_argument('--search-std',type=float,default=.3)
    parser.add_argument('--preserve-yaw-amplitude',action='store_true',help='Rescale seeded feedforward gains when changing yaw-command duration')
    parser.add_argument('--left-checkpoint',help='Preserve the seven certified left-turn/hold parameters from this checkpoint')
    parser.add_argument('--seed-candidate',type=int,help='Resume an explicitly recorded candidate from the seed run grid')
    parser.add_argument('--teleop-refine',action='store_true',help='Refine hold, right-turn and restart parameters; preserve left walking corrections')
    parser.add_argument('--restart-only',action='store_true',help='Search only restart ramp and phase')
    parser.add_argument('--restart-grid',action='store_true',help='One-generation restart ramp/phase grid')
    parser.add_argument('--right-only',action='store_true',help='Freeze the certified seven left/hold parameters; search right-specific feedback')
    parser.add_argument('--right-grid',action='store_true',help='One-generation right amplitude/heading grid')
    parser.add_argument('--direction-train',action='store_true',help='M7 sustained right-turn curriculum; requires teleop memory and frozen left parameters')
    parser.add_argument('--left-cruise-only',action='store_true',help='M7 left-turn curriculum; search straight-after-left heading and lateral sway feedback')
    parser.add_argument('--direction-command-trace',help='Replay semantic commands from a recorded left gate, then continue straight; training only')
    parser.add_argument('--closed-loop-direction',choices=('forward','left','right','backward'),help='Compute independent per-world commands from pose and optimize actual gate crossing')
    parser.add_argument('--left-turn-refine',action='store_true',help='Also refine left active heading/sway with post-onset force margin; requires M5 recheck')
    parser.add_argument('--left-yaw-only',action='store_true',help='Calibrate a separate left yaw scale on M5; freeze17balance parameters')
    parser.add_argument('--left-heading-grid',action='store_true',help='Diagnose8cruise heading gains with every other parameter frozen')
    parser.add_argument('--left-balance-only',action='store_true',help='Refine only active-left sway amplitude/phase on recorded steering with force margin')
    parser.add_argument('--common-starts',action='store_true',help='Compare closed-loop candidates on identical bounded starting perturbations')
    parser.add_argument('--start-seed',type=int,default=4242)
    parser.add_argument('--left-balance-grid',action='store_true',help='Repeat fixed sway grid with two unchanged controls and common starts')
    parser.add_argument('--left-cruise-balance-only',action='store_true',help='Search only post-left-turn sway amplitude/phase; preserve all other seed parameters exactly')
    parser.add_argument('--left-feedback-only',action='store_true',help='Learn two bounded left-turn torso roll/rate feedback corrections; freeze prior18parameters')
    parser.add_argument('--left-damping-grid',action='store_true',help='Repeat8left roll-rate gains, including duplicate unchanged controls')
    parser.add_argument('--turn-hold-training',action='store_true',help='Screen left feedback on exactM5commands and hold with the same force margin')
    parser.add_argument('--feedback-yaw-refine',action='store_true',help='Also recalibrate existing left yaw scale during exact turn/hold feedback screening')
    parser.add_argument('--teleop',action='store_true',help='Train the60s scripted joystick protocol with shared command memory')
    parser.add_argument('--diagnostic-sweep',action='store_true',help='Short null-to-small yaw amplitude comparison; never milestone evidence')
    parser.add_argument('--diagnostic-grid',action='store_true',help='Sweep small differential yaw and signed heading correction')
    parser.add_argument('--stride-grid',action='store_true',help='Sweep differential sagittal stride and heading correction')
    args = parser.parse_args()
    if args.common_starts and not (args.closed_loop_direction or args.turn_hold_training):raise ValueError('Common starts require closed-loop direction or exact turn/hold training')
    if args.left_cruise_balance_only and (not args.left_turn_refine or not args.common_starts or args.closed_loop_direction not in ('left','backward') or args.left_balance_only or args.left_feedback_only or args.left_yaw_only or args.left_heading_grid):
        raise ValueError('Cruise balance search requires a separate shared-start closed-loop left curriculum with force margin')
    if args.left_balance_grid and not (args.common_starts and args.left_balance_only):raise ValueError('Balance grid requires common starts and balance-only refinement')
    if args.left_feedback_only and (not args.left_turn_refine or not args.common_starts or not (args.closed_loop_direction in ('left','backward') or args.turn_hold_training) or args.left_balance_only or args.left_yaw_only or args.left_heading_grid):
        raise ValueError('Left feedback search requires its own shared-start closed-loop left curriculum with force margin')
    if args.turn_hold_training and (not args.left_feedback_only or args.closed_loop_direction or args.direction_command_trace or args.seconds<3.+args.turn_duration+2.+7.):
        raise ValueError('Turn/hold screening requires independent left feedback training and a complete five-second hold')
    if args.feedback_yaw_refine and (not args.turn_hold_training or args.left_damping_grid):raise ValueError('Joint yaw/feedback search requires exact turn/hold screening without the damping grid')
    if args.left_damping_grid and (not args.left_feedback_only or args.num_envs!=32):raise ValueError('Damping grid requires32worlds and left feedback-only search')
    if args.left_checkpoint and (not args.teleop or not args.seed_checkpoint):raise ValueError('Left preservation requires seeded teleop training')
    if args.seed_candidate is not None and not args.seed_checkpoint:raise ValueError('Candidate selection requires a seed run')
    if args.teleop_refine and (not args.teleop or args.right_only or args.restart_only):raise ValueError('Choose one teleop search subset')
    if args.restart_only and (not args.teleop or args.right_only):raise ValueError('Restart-only search requires separate teleop search')
    if args.restart_grid and (not args.restart_only or args.generations!=1):raise ValueError('Restart grid requires one generation')
    if args.right_only and not args.teleop:raise ValueError('Right-only search requires teleop')
    if args.right_grid and (not args.right_only or args.generations!=1):raise ValueError('Right grid requires one right-only generation')
    if args.teleop and not args.direction_train and (args.seconds!=60. or args.diagnostic_sweep):raise ValueError('Teleop training requires full60s episodes')
    minimum_seconds=6. if args.diagnostic_sweep or args.closed_loop_direction else 24.
    maximum_seconds=240. if args.left_heading_grid or args.left_balance_only or args.closed_loop_direction else 120.
    if not 32<=args.num_envs<=1024 or args.num_envs%4 or not 1<=args.generations<=20 or not minimum_seconds<=args.seconds<=maximum_seconds:
        raise ValueError('Turn training exceeds bounded budget')
    if not args.diagnostic_sweep and not args.direction_train and args.seconds<3.+args.turn_duration+2.+5.:
        raise ValueError('Full training requires at least five seconds of zero-command hold')
    if args.diagnostic_sweep and args.generations!=1:raise ValueError('Diagnostic sweep uses one generation')
    if args.diagnostic_grid and not args.diagnostic_sweep:raise ValueError('Grid requires diagnostic sweep')
    if args.stride_grid and not args.diagnostic_grid:raise ValueError('Stride grid requires diagnostic grid')
    if not .03<=args.search_std<=.4:raise ValueError('Search spread outside bounded range')
    if args.left_cruise_only and (not args.direction_train or args.right_only or not args.seed_checkpoint):
        raise ValueError('Left cruising search requires separate seeded direction curriculum')
    if args.direction_command_trace and not args.left_cruise_only:
        raise ValueError('Recorded steering curriculum requires left-cruise-only training')
    if args.left_turn_refine and not args.left_cruise_only:
        raise ValueError('Left-turn refinement requires the seeded left cruising curriculum')
    if args.left_yaw_only and (not args.left_cruise_only or args.left_turn_refine or args.direction_command_trace):
        raise ValueError('Left yaw calibration requires a separate seeded left curriculum')
    if args.left_heading_grid and (not args.left_cruise_only or args.left_turn_refine or args.left_yaw_only or not args.direction_command_trace or args.num_envs!=32 or args.generations!=1):
        raise ValueError('Heading grid requires recorded left steering,32worlds and1generation')
    if args.left_balance_only and (not args.left_turn_refine or args.left_heading_grid or args.left_yaw_only or not (args.direction_command_trace or args.closed_loop_direction)):
        raise ValueError('Balance-only search requires recorded steering and force-margin refinement')
    if args.closed_loop_direction and (not args.direction_train or args.direction_command_trace or args.left_yaw_only or args.left_heading_grid):
        raise ValueError('Closed-loop direction training cannot mix recorded or fixed turn protocols')
    if args.direction_train and (not args.teleop or not (args.right_only or args.left_cruise_only) or args.seconds<(6. if args.closed_loop_direction else 55.)):
        raise ValueError('Direction curriculum requires teleop memory, a frozen search subset and >=55 s')
    train(args)


if __name__=='__main__':
    main()
