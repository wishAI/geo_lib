"""Learn bounded periodic motor targets and proprioceptive feedback with CEM.

No pelvis forces, root-state guidance, inverse-kinematic state assignments, or
checkpoint promotion. The controller family is motivated by feedback gait
policies (https://arxiv.org/abs/2103.15309), not copied from that implementation.
Independent full-duration evaluation remains the acceptance authority.
"""
import argparse
import faulthandler
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import time
faulthandler.enable()
import warp as wp


@wp.kernel
def record_joint_speeds(velocity: wp.array2d(dtype=wp.float32), dofs: wp.array(dtype=wp.int32), count: int, maximum: wp.array(dtype=wp.float32)):
    world=wp.tid()
    speed=float(0.)
    for index in range(count):
        speed=wp.max(speed,wp.abs(velocity[world,dofs[index]]))
    maximum[world]=wp.max(maximum[world],speed)


@wp.kernel
def clear_support(support: wp.array(dtype=wp.float32), normal: wp.array2d(dtype=wp.float32)):
    world=wp.tid()
    support[world]=0.
    normal[world,0]=0.
    normal[world,1]=0.


@wp.kernel
def sum_support(ncon: wp.array(dtype=wp.int32), geom: wp.array(dtype=wp.vec2i),
                world: wp.array(dtype=wp.int32), address: wp.array2d(dtype=wp.int32),
                frame: wp.array(dtype=wp.mat33), force: wp.array2d(dtype=wp.float32),
                feet: wp.array(dtype=wp.int32), support: wp.array(dtype=wp.float32),
                nonfoot: wp.array(dtype=wp.int32), normal: wp.array2d(dtype=wp.float32)):
    contact=wp.tid()
    if contact<ncon[0]:
        pair=geom[contact]
        index=address[contact,0]
        if wp.min(pair[0],pair[1])==0 and index>=0:
            other=wp.max(pair[0],pair[1])
            w=world[contact]
            if feet[other]>=0:
                wp.atomic_add(normal,w,feet[other],wp.max(force[w,index],0.))
                axes=frame[contact]
                vertical=float(0.)
                for axis in range(3):
                    row=address[contact,axis]
                    if row>=0 and row<force.shape[1]:
                        vertical+=axes[axis,2]*force[w,row]
                wp.atomic_add(support,w,vertical)
            elif wp.abs(force[w,index])>.01:
                wp.atomic_max(nonfoot,w,1)


@wp.kernel
def record_support(support: wp.array(dtype=wp.float32), maximum: wp.array(dtype=wp.float32)):
    world=wp.tid()
    maximum[world]=wp.max(maximum[world],support[world])


@wp.kernel
def record_airborne(normal: wp.array2d(dtype=wp.float32), current: wp.array(dtype=wp.int32),
                    maximum: wp.array(dtype=wp.int32), total: wp.array(dtype=wp.int32)):
    world=wp.tid()
    if normal[world,0]<=.1 and normal[world,1]<=.1:
        current[world]+=1
        total[world]+=1
        maximum[world]=wp.max(maximum[world],current[world])
    else:
        current[world]=0


PARAMETERS = {
    'period_s': (.7, 1.4),
    'knee_flexion_rad': (.25, .5),
    'fore_aft_hip_rad': (-.2, .2),
    'hip_roll_rad': (.05, .35),
    'roll_phase_rad': (-3.141592653589793, 3.141592653589793),
    'ankle_pitch_feedback': (-1.2, 1.2),
    'ankle_pitch_damping': (-.2, .2),
    'hip_roll_feedback': (-1.2, 1.2),
    'hip_roll_damping': (-.2, .2),
    'forward_velocity_feedback': (-.4, .4),
    'hip_pitch_feedback': (-.8, .8),
    'waist_pitch_feedback': (-.5, .5),
    'heading_feedback': (-.3, .3),
    'heading_damping': (-.15, .15),
    'stride_phase_rad': (-.45, .45),
    'startup_ramp_s': (.5, 1.8),
}


def gait_action(observation, parameters, names):
    """70 raw observable features -> 17 legacy-unit motor offsets, batched."""
    import torch
    p=parameters.expand(observation.shape[0],-1)
    age=(1.-observation[:,62])*30.
    phase=2*torch.pi*(age-1.).clamp_min(0)/p[:,0]
    startup=((age-1.)/p[:,15]).clamp(0,1)
    startup=startup.square()*(3.-2.*startup)
    sine=phase.sin()
    pitch=torch.atan2(-observation[:,7],-observation[:,8])
    roll=torch.atan2(-observation[:,6],-observation[:,8])
    heading=torch.atan2(observation[:,68],observation[:,69]+1.)
    ankle=p[:,5]*pitch+p[:,6]*observation[:,3]+p[:,9]*(observation[:,1]-observation[:,63])
    lateral=-p[:,3]*(phase+p[:,4]).sin()*startup+p[:,7]*roll+p[:,8]*observation[:,4]
    yaw=p[:,12]*heading+p[:,13]*observation[:,5]
    offsets=torch.zeros((observation.shape[0],len(names)),device=observation.device,dtype=observation.dtype)
    index={name:i for i,name in enumerate(names)}
    for side,sign in (('left',1.),('right',-1.)):
        lift=(sign*sine).clamp_min(0).square()*p[:,1]*startup
        stride=sign*p[:,2]*(phase+p[:,14]).cos()*startup
        offsets[:,index[side+'_knee_joint']]=lift
        offsets[:,index[side+'_hip_pitch_joint']]=-.5*lift+stride+p[:,10]*pitch
        offsets[:,index[side+'_ankle_pitch_joint']]=-.5*lift-stride+ankle
        offsets[:,index[side+'_hip_roll_joint']]=lateral
        offsets[:,index[side+'_hip_yaw_joint']]=yaw
    offsets[:,index['waist_pitch_joint']]=p[:,11]*pitch
    return offsets/.08


def evaluate_mean(policy, observation, names):
    """The saved trained standing network owns zero-command inference."""
    from tensordict import TensorDict
    result=gait_action(observation,policy.parameters,names)
    standing=observation[:,63]==0
    if standing.any():
        selected=observation[standing,:63]
        result[standing]=policy.standing_prior(TensorDict({'actor':selected},batch_size=[len(selected)]))
    return result


def main():
    import mujoco
    import torch
    from algorithms.urdf_learn_wasd_walk.landau_rsl_control import configure_model
    from algorithms.urdf_learn_wasd_walk.landau_forward_control import action_scales, PERIOD
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--name',required=True);p.add_argument('--checkpoint',required=True)
    p.add_argument('--num-envs',type=int,default=512);p.add_argument('--generations',type=int,default=20)
    p.add_argument('--seconds',type=float,default=6.)
    p.add_argument('--seed-checkpoint')
    p.add_argument('--target-distance',type=float,choices=(5.,10.),default=5.)
    p.add_argument('--objective',choices=('discovery','forward'),default='discovery')
    p.add_argument('--replicas',type=int,choices=(1,4),default=4)
    p.add_argument('--search-std',type=float,default=.35)
    p.add_argument('--minimum-search-std',type=float,default=.08,help='Normalized search-deviation floor; smaller values permit fine balance refinement')
    p.add_argument('--stop-forward-m',type=float,help='Finish the search for independent validation after an eligible candidate reaches this worst-replica distance')
    p.add_argument('--search-mode',choices=('cem','stride_sweep'),default='cem')
    p.add_argument('--sweep-primary',choices=tuple(PARAMETERS),default='fore_aft_hip_rad')
    p.add_argument('--sweep-min',type=float,default=.025)
    p.add_argument('--sweep-max',type=float,default=.12)
    p.add_argument('--sweep-secondary',choices=tuple(k for k in PARAMETERS if k!='fore_aft_hip_rad'))
    p.add_argument('--secondary-min',type=float)
    p.add_argument('--secondary-max',type=float)
    p.add_argument('--exclude-seed',action='store_true',help='Grid experiments only: evaluate the requested region without reinserting the parent')
    p.add_argument('--initial-parameters',type=json.loads,default={},help='Explicit bounded moving-controller search initialization; never modifies the standing branch')
    p.add_argument('--stride-sign',choices=('any','positive','negative'),default='any')
    p.add_argument('--trace-seed',action='store_true',help='Record first-generation nominal seed observations and states for evaluator parity diagnosis')
    args=p.parse_args()
    if not 32<=args.num_envs<=2048 or not 1<=args.generations<=40 or not 3<=args.seconds<=120:
        raise ValueError('Bounded search limits exceeded')
    if args.num_envs%args.replicas:raise ValueError('Replicas must divide the world count')
    if not .05<=args.search_std<=.65:raise ValueError('Search deviation outside bounded range')
    if not .02<=args.minimum_search_std<=.15:raise ValueError('Search deviation floor outside bounded range')
    if args.stop_forward_m is not None and (args.objective!='forward' or args.seconds<30. or not args.target_distance<=args.stop_forward_m<=args.target_distance+.5):
        raise ValueError('Candidate stop distance requires at least30seconds and a threshold within0.5m above the distance gate')
    if args.exclude_seed and args.search_mode!='stride_sweep':raise ValueError('Seed exclusion is only supported for grid experiments')
    if args.stride_sign!='any' and args.search_mode!='cem':raise ValueError('Stride-sign restriction applies to CEM discovery')
    if not isinstance(args.initial_parameters,dict):raise ValueError('Initial parameters must be a JSON object')
    if args.search_mode=='stride_sweep' and (args.generations!=1 or not args.seed_checkpoint or not PARAMETERS[args.sweep_primary][0]<=args.sweep_min<args.sweep_max<=PARAMETERS[args.sweep_primary][1]):
        raise ValueError('Parameter sweep requires one generation, a seed, and declared parameter bounds')
    if args.sweep_secondary:
        if args.sweep_secondary==args.sweep_primary:raise ValueError('Sweep axes must differ')
        bounds=PARAMETERS[args.sweep_secondary]
        if args.search_mode!='stride_sweep' or args.secondary_min is None or args.secondary_max is None or not bounds[0]<=args.secondary_min<args.secondary_max<=bounds[1]:
            raise ValueError('Secondary sweep must stay inside the existing parameter bounds')
    backend,batches,variant=configure_model('balanced_hands_v1')
    parent=Path(args.checkpoint).resolve();parent.relative_to((backend.OUTPUT/'training').resolve())
    parent_meta=json.loads((parent.parent/'training.json').read_text())
    if parent_meta['observation_dim']!=63 or parent_meta['checkpoints'][parent.name]!=backend.digest(parent):
        raise ValueError('Require the audited standing checkpoint')
    folder=(backend.OUTPUT/'training'/args.name).resolve();folder.relative_to((backend.OUTPUT/'training').resolve())
    folder.mkdir(exist_ok=False)
    torch.set_num_threads(4);torch.manual_seed(42)
    began=time.perf_counter()
    batch=batches.WarpBatch(args.num_envs,42,0.,'stand','load',.004,.2,.0225)
    # The 4 rad/s acceptance limit applies to all physical joints at every
    # physics step, including short contact transients between policy updates.
    speed_dofs=wp.array([int(batch.model.joint(row['name']).dofadr[0]) for row in batch.spec['joints']],dtype=wp.int32,device='cuda:0')
    speed_history=wp.zeros(args.num_envs,dtype=wp.float32,device='cuda:0')
    speed_view=wp.to_torch(speed_history)
    speed_inputs=[batch.wd.qvel,speed_dofs,len(batch.spec['joints']),speed_history]
    wp.launch(record_joint_speeds,dim=args.num_envs,inputs=speed_inputs)
    if batch.model.opt.cone!=mujoco.mjtCone.mjCONE_ELLIPTIC:
        raise ValueError('Per-physics contact force audit requires elliptic contact coordinates')
    support=wp.zeros(args.num_envs,dtype=wp.float32,device='cuda:0')
    support_history=wp.zeros_like(support)
    nonfoot_history=wp.zeros(args.num_envs,dtype=wp.int32,device='cuda:0')
    nonfoot_view=wp.to_torch(nonfoot_history)
    normal=wp.zeros((args.num_envs,2),dtype=wp.float32,device='cuda:0')
    flight_current=wp.zeros(args.num_envs,dtype=wp.int32,device='cuda:0')
    flight_maximum=wp.zeros_like(flight_current);flight_total=wp.zeros_like(flight_current)
    flight_max_view=wp.to_torch(flight_maximum);flight_total_view=wp.to_torch(flight_total)
    support_view=wp.to_torch(support_history)
    feet=wp.from_torch(batch.contact_sides.to(torch.int32))
    con=batch.wd.contact
    support_inputs=[batch.wd.nacon,con.geom,con.worldid,con.efc_address,con.frame,batch.wd.efc.force,feet,support,nonfoot_history,normal]
    def audit_support():
        wp.launch(clear_support,dim=args.num_envs,inputs=[support,normal])
        wp.launch(sum_support,dim=batch.wd.naconmax,inputs=support_inputs)
        wp.launch(record_support,dim=args.num_envs,inputs=[support,support_history])
    audit_support()
    body_weight=float(batch.model.body_mass.sum())*9.81
    minimum_swings=max(1,math.ceil(max(0.,args.seconds-2.)*.5))
    occupancy_inputs=[batch.wd.ncollision,batch.wd.nacon,batch.wd.nefc,batch.maximum_occupancy]
    with wp.ScopedCapture() as capture:
        for _ in range(10):
            batches.mjwarp.step(batch.wm,batch.wd)
            # Match independent validation's post-integration refresh cadence.
            # Contact/solver ordering can otherwise diverge in marginal gaits.
            batches.mjwarp.forward(batch.wm,batch.wd)
            wp.launch(record_joint_speeds,dim=args.num_envs,inputs=speed_inputs)
            audit_support()
            wp.launch(record_airborne,dim=args.num_envs,inputs=[normal,flight_current,flight_maximum,flight_total])
            wp.launch(batches.record_occupancy,dim=args.num_envs,inputs=occupancy_inputs)
    batch.graph=capture.graph
    model_hash=hashlib.sha256(batch.xml.encode()).hexdigest()
    if model_hash!=parent_meta['model_xml_sha256']:raise ValueError('Standing model mismatch')
    batch.commands[:]=.2
    names=batch.spec['action_joints'];n=args.num_envs
    limits=torch.tensor(action_scales(names,.4),device='cuda')/.08
    lows=torch.tensor([v[0] for v in PARAMETERS.values()],device='cuda')
    highs=torch.tensor([v[1] for v in PARAMETERS.values()],device='cuda')
    mean=torch.zeros(len(PARAMETERS),device='cuda');std=torch.full_like(mean,.65)
    seed_checkpoint=None
    if args.seed_checkpoint:
        import inspect
        from algorithms.urdf_learn_wasd_walk.landau_forward_control import load_gait_source
        seed_checkpoint=Path(args.seed_checkpoint).resolve();seed_checkpoint.relative_to((backend.OUTPUT/'training').resolve())
        seed_meta=json.loads((seed_checkpoint.parent/'training.json').read_text())
        if seed_meta.get('policy_family')!='periodic_feedback_cem' or seed_meta['model_xml_sha256']!=model_hash or seed_meta['checkpoints'][seed_checkpoint.name]!=backend.digest(seed_checkpoint):
            raise ValueError('Search seed provenance mismatch')
        seed_module,_=load_gait_source(seed_checkpoint,seed_meta)
        seed_ranges=dict(seed_module.PARAMETERS)
        if seed_ranges.get('fore_aft_hip_rad')==(0.,.2):
            seed_ranges['fore_aft_hip_rad']=PARAMETERS['fore_aft_hip_rad']
        if seed_ranges.get('forward_velocity_feedback') in ((-.4,.4),(-.8,.4)):
            seed_ranges['forward_velocity_feedback']=PARAMETERS['forward_velocity_feedback']
        if seed_ranges.get('stride_phase_rad')==(-.45,0.):
            seed_ranges['stride_phase_rad']=PARAMETERS['stride_phase_rad']
        seed_source=inspect.getsource(seed_module.gait_action)
        current_source=inspect.getsource(gait_action)
        fixed_startup_source=current_source.replace('/p[:,15]','/.8')
        legacy_startup=(seed_ranges=={k:v for k,v in PARAMETERS.items() if k!='startup_ramp_s'}
                        and seed_source==fixed_startup_source)
        legacy_phase=(seed_ranges=={k:v for k,v in PARAMETERS.items() if k not in ('stride_phase_rad','startup_ramp_s')}
                      and seed_source==fixed_startup_source.replace('(phase+p[:,14])','phase'))
        if not (legacy_phase or legacy_startup) and (seed_source!=current_source or seed_ranges!=PARAMETERS):
            raise ValueError('Search seed controller family changed')
        seed_parameters=torch.load(seed_checkpoint,map_location='cuda:0',weights_only=False)['parameters']
        if legacy_phase:
            # Exactly preserve the parent's motor targets before optimizing the
            # added touchdown-timing parameter. Saved parents retain own source.
            seed_parameters=torch.cat((seed_parameters,torch.zeros(1,device='cuda:0')))
        if legacy_phase or legacy_startup:
            seed_parameters=torch.cat((seed_parameters,torch.tensor([.8],device='cuda:0')))
        for key,value in args.initial_parameters.items():
            if key not in PARAMETERS or not PARAMETERS[key][0]<=value<=PARAMETERS[key][1]:
                raise ValueError('Initial search parameter outside declared bounds')
            seed_parameters[list(PARAMETERS).index(key)]=value
        if seed_parameters.shape!=lows.shape or not torch.isfinite(seed_parameters).all() or ((seed_parameters<lows-1e-6)|(seed_parameters>highs+1e-6)).any():
            raise ValueError('Search seed parameters outside bounds')
        mean=((seed_parameters-lows)/(highs-lows)*2.-1.).clamp(-1,1);std.fill_(args.search_std)
    if args.initial_parameters and not seed_checkpoint:raise ValueError('Explicit initialization needs a provenance-bound seed checkpoint')
    if args.stride_sign=='negative' and mean[2]>0 or args.stride_sign=='positive' and mean[2]<0:
        raise ValueError('Seed stride conflicts with the requested search region')
    standing=torch.load(parent,map_location='cpu',weights_only=False)['actor_state_dict']
    metadata={'created_at':datetime.now(timezone.utc).isoformat(),'arguments':{**vars(args),'hip_roll_range':.4},
        'policy_family':'periodic_feedback_cem','policy_composition':('Grid-selected' if args.search_mode=='stride_sweep' else 'CEM-trained')+' periodic proprioceptive feedback; frozen standing MLP at zero command',
        'backend':'mujoco_warp_cuda','mujoco_version':mujoco.__version__,'observation_dim':70,'period_s':PERIOD,
        'observation_clock_period_s':PERIOD,'gait_period_source':'saved checkpoint parameters[0], independently optimized in [0.7, 1.4] seconds',
        'packages':{name:importlib.metadata.version(name) for name in ('torch','mujoco','mujoco-warp','warp-lang','rsl-rl-lib')},
        'action_joints':names,'action_scale':action_scales(names,.4),'stand_action_scale':.08,
        'action_mapping':'legacy_radians_v2','nominal_q':batch.nominal_q.tolist(),'nominal_ctrl':batch.nominal_ctrl.tolist(),
        'reference_xy':batch.reference_xy.tolist(),'model_xml_sha256':model_hash,
        'urdf_sha256':batch.spec['source']['urdf_sha256'],'mesh_tree_sha256':batch.spec['source']['mesh_tree_sha256'],
        'parent_checkpoint':str(seed_checkpoint or parent),'parent_checkpoint_sha256':backend.digest(seed_checkpoint or parent),'mass_variant':variant,
        'standing_checkpoint':str(parent),'standing_checkpoint_sha256':backend.digest(parent),
        'search_seed_checkpoint':str(seed_checkpoint) if seed_checkpoint else None,
        'objective':args.objective,
        'forward_fitness_contract':f'After survival and sustained-swing constraints, prioritize worst-replica distance up to {args.target_distance+.5:g} m for validation margin; clearance score only breaks near ties. Incomplete candidates retain discovery shaping.',
        'joint_speed_acceptance':'all 69 physical joints, every 0.002 s physics step; above 4 rad/s makes the candidate ineligible',
        'support_force_acceptance':'Actual foot-contact vertical forces every 0.002 s; above 3 body weights makes the candidate ineligible',
        'physics_refresh_contract':'mjwarp.step then mjwarp.forward every 0.002 s, matching independent validation cadence; policy control every 0.02 s',
        'lateral_acceptance':'Final semantic lateral displacement at most0.75m for every start, matching independent gate; excess makes candidate ineligible',
        'nonfoot_contact_acceptance':'Any non-foot ground contact above 0.01 N makes the candidate ineligible',
        'flight_acceptance':'Both foot normal forces <=0.1 N, measured every 0.002 s: at most 0.12 s continuous and 5% total flight, matching independent validation',
        'completed_swing_contract':'At least 0.06 s air and 0.015 m clearance before touchdown; no phase-aliased opposite-foot-contact exclusion',
        'stride_phase_seed_migration':'14-parameter parents append exactly zero stride phase; existing motor-target terms unchanged; signed stride expansion recorded separately',
        'signed_stride_seed_migration':'Physical parent parameters preserved unless explicitly overridden in arguments.initial_parameters; search coordinates re-encoded under [-.2, .2] rad',
        'startup_seed_migration':'Older parents append their fixed 0.8 s startup ramp; all steady-state motor-target terms remain unchanged',
        'velocity_feedback_seed_migration':'Physical gains preserved within the current [-0.4, 0.4] bounds; wider-range parents are usable only when their actual gain lies inside these bounds',
        'signed_phase_seed_migration':'Existing physical stride phase preserved while extending the search from [-0.45, 0] to [-0.45, 0.45] rad',
        'initial_search_parameters':dict(zip(PARAMETERS,(lows+(mean+1)*.5*(highs-lows)).tolist())),
        'candidate_replicas':args.replicas,'replica_selection':'worst fitness; first replica starts exactly nominal, remaining replicas use bounded joint noise',
        'minimum_training_swings_per_foot':minimum_swings,
        'failure_bitmask':{'tilt':1,'height':2,'nonfinite':4,'joint_speed':8,'support_force':16,'heading':32,'nonfoot_ground_contact':64,'continuous_flight':128,'flight_fraction':256,'lateral_drift':512},
        'parameter_ranges':PARAMETERS,'assistance':0.,'canonical_milestone_pass':False,
        'training_reset_semantics':'Reset only between independent candidate episodes; never validation evidence',
        'source_sha256':{str(x):backend.digest(x) for x in (Path(__file__),Path(batches.__file__),Path(backend.__file__))}}
    backend.write_json(folder/'metadata.json',metadata)
    (folder/'model.xml').write_text(batch.xml);(folder/'control_source.py').write_text(Path(__file__).read_text())
    def record_progress(state,generation):
        path=backend.OUTPUT.parent/'backend_progress.json'
        record=json.loads(path.read_text()) if path.exists() else {}
        record.update(updated_at=datetime.now(timezone.utc).isoformat(),current_gate=f'gate_{args.target_distance:g}m_no_reset',
            simulator='mujoco_warp_cuda',variant='balanced_hands_v1',iteration=generation,
            active_process=args.name if state=='running' else None,assistance_coefficient=0.,artifact_paths=[str(folder)],
            next_step='Search coordinated gait and proprioceptive balance feedback; no milestone claim.' if state=='running' else
                      f'Evaluate the saved checkpoint for{args.seconds:g}seconds against{args.target_distance:g}m and record debug video.')
        backend.write_json(path,record)
    record_progress('running',0)
    best_z=mean.clone() if seed_checkpoint else None
    for generation in range(args.generations):
        candidates=n//args.replicas
        z=(mean+std*torch.randn(candidates,len(mean),device='cuda')).clamp(-1,1)
        if args.stride_sign=='negative':z[:,2].clamp_(max=0.)
        if args.stride_sign=='positive':z[:,2].clamp_(min=0.)
        if args.search_mode=='stride_sweep':
            z=mean.expand(candidates,-1).clone()
            amplitudes=torch.linspace(args.sweep_min,args.sweep_max,candidates,device='cuda')
            if args.sweep_secondary:
                columns=4 if candidates<64 else 8
                rows=math.ceil(candidates/columns)
                indices=torch.arange(candidates,device='cuda')
                amplitudes=torch.linspace(args.sweep_min,args.sweep_max,rows,device='cuda')[indices//columns]
                second=list(PARAMETERS).index(args.sweep_secondary)
                values=torch.linspace(args.secondary_min,args.secondary_max,columns,device='cuda')[indices%columns]
                z[:,second]=2.*(values-lows[second])/(highs[second]-lows[second])-1.
            primary=list(PARAMETERS).index(args.sweep_primary)
            z[:,primary]=2.*(amplitudes-lows[primary])/(highs[primary]-lows[primary])-1.
        if best_z is not None and not args.exclude_seed:z[0]=best_z
        params=(lows+(z+1)*.5*(highs-lows)).repeat_interleave(args.replicas,dim=0)
        batch.reset(torch.ones(n,device='cuda',dtype=torch.bool))
        batch.q[::args.replicas]=batch.nominal
        torch.cuda.synchronize()
        wp.capture_launch(batch.forward_graph);wp.synchronize()
        speed_view.zero_()
        support_view.zero_()
        nonfoot_view.zero_()
        flight_current.zero_();flight_maximum.zero_();flight_total.zero_()
        alive=torch.ones(n,device='cuda',dtype=torch.bool)
        duration=torch.zeros(n,device='cuda');score=torch.zeros_like(duration)
        completed=torch.zeros((n,2),device='cuda',dtype=torch.long)
        air=torch.zeros((n,2),device='cuda');peak=torch.zeros_like(air);touched=torch.zeros_like(air,dtype=torch.bool)
        peak_clear=torch.zeros_like(air);max_speed=torch.zeros_like(duration)
        max_support=torch.zeros_like(duration)
        failure_reasons=torch.zeros(n,device='cuda',dtype=torch.int32)
        failure_gravity=torch.zeros((n,3),device='cuda')
        failure_velocity=torch.zeros((n,6),device='cuda')
        max_heading=torch.zeros_like(duration);max_tilt=torch.zeros_like(duration)
        last_progress=torch.zeros_like(duration)
        last_lateral=torch.zeros_like(duration)
        seed_trace=[]
        for step in range(round(args.seconds/.02)):
            phase=2*torch.pi*(step*.02)/PERIOD
            extra=torch.zeros((n,10),device='cuda')
            extra[:,2]=1.-step*.02/30.;extra[:,3]=.2
            extra[:,6]=torch.sin(torch.tensor(phase,device='cuda'));extra[:,7]=torch.cos(torch.tensor(phase,device='cuda'))
            extra[:,8]=batch.rot[:,batch.base,1,0];extra[:,9]=batch.rot[:,batch.base,0,0]-1.
            obs=torch.cat((batch.observations(),extra),dim=1)
            action=gait_action(obs,params,names).clamp(-limits,limits)
            if args.trace_seed and generation==0:
                seed_trace.append((obs[0].cpu().numpy().copy(),action[0].cpu().numpy().copy(),
                                   batch.q[0].cpu().numpy().copy(),batch.v[0].cpu().numpy().copy()))
            batch.ctrl[:]=batch.targets;batch.ctrl[:,batch.aids]+=.08*torch.where(alive[:,None],action,0.)
            batch.wrench.zero_();torch.cuda.synchronize()
            batches.wp.capture_launch(batch.graph);batches.wp.synchronize()
            occupancy=batch.maximum_occupancy.numpy()
            if max(occupancy[:2])>=batch.wd.naconmax or occupancy[2]>=batch.wd.njmax:
                raise RuntimeError('Candidate search exceeded physics contact capacity')
            tilt=batch.rot[:,batch.base,2,2].clamp(-1,1).acos()
            finite=torch.isfinite(batch.q).all(1)&torch.isfinite(batch.v).all(1)
            max_speed=torch.maximum(max_speed,torch.where(alive,speed_view,0.))
            max_support=torch.maximum(max_support,torch.where(alive,support_view/body_weight,0.))
            heading=torch.atan2(batch.rot[:,batch.base,1,0],batch.rot[:,batch.base,0,0])
            max_heading=torch.maximum(max_heading,torch.where(alive,heading.abs(),0.))
            max_tilt=torch.maximum(max_tilt,torch.where(alive,tilt,0.))
            reason=(tilt>torch.pi/6).int()+2*(batch.pos[:,batch.pelvis,2]<batch.reference_height-.08).int()+4*(~finite).int()+8*(speed_view>4.+1e-6).int()+16*(support_view>3.*body_weight).int()+32*(heading.abs()>torch.pi/6).int()+64*(nonfoot_view!=0).int()+128*(flight_max_view>60).int()
            failed=reason!=0
            newly_failed=alive&failed
            failure_reasons=torch.where(newly_failed,reason,failure_reasons)
            failure_gravity=torch.where(newly_failed[:,None],-batch.rot[:,batch.base,2,:],failure_gravity)
            failure_velocity=torch.where(newly_failed[:,None],batch.v[:,:6],failure_velocity)
            alive &= ~failed
            contacts,slip,clearance,_=batch.feet(with_forces=True)
            landed=contacts&touched&(air>=.06)&(peak>=.015)&alive[:,None]
            completed+=landed.long()
            air=torch.where(contacts,0.,air+.02);peak=torch.where(contacts,0.,torch.maximum(peak,clearance))
            touched|=contacts
            peak_clear=torch.maximum(peak_clear,torch.where(alive[:,None],clearance,0.))
            speed=speed_view
            heading=torch.atan2(batch.rot[:,batch.base,1,0],batch.rot[:,batch.base,0,0])
            progress=batch.pos[:,batch.pelvis,1]-batch.reference_xy[1]
            last_progress=torch.where(alive,progress,last_progress)
            last_lateral=torch.where(alive,batch.pos[:,batch.pelvis,0]-batch.reference_xy[0],last_lateral)
            duration+=alive.float()*.02
            # Whole-episode discovery objective; observed support/clearance, not FK targets.
            discovery=(clearance.clamp(0,.018)*contacts.flip(1)).sum(1)/.018
            rate=1.+2.*discovery-.5*tilt.square()-2.*slip-2.*heading.square()-.2*(speed-4.).clamp_min(0).square()-6.*(~contacts.any(1)).float()
            score+=torch.where(alive,rate,0.).nan_to_num(0.)*.02
            batch.previous=action
            if newly_failed.any():batch.reset(newly_failed)
        excess_flight=flight_total_view>round(args.seconds/.002)*.05
        failure_reasons=torch.where(alive&excess_flight,256,failure_reasons)
        alive &= ~excess_flight
        excess_lateral=last_lateral.abs()>.75
        failure_reasons=torch.where(alive&excess_lateral,512,failure_reasons)
        alive &= ~excess_lateral
        if seed_trace:
            import numpy as np
            np.savez_compressed(folder/'seed_trace.npz',
                **{key:np.array([row[i] for row in seed_trace]) for i,key in enumerate(('observation','action','qpos','qvel'))})
        if args.objective=='forward':
            bouts=completed.clamp_max(minimum_swings)
            has_steps=(completed.min(1).values>=minimum_swings).float()
            fitness=score+5.*bouts.min(1).values+bouts.sum(1)+50.*last_progress.clamp(-.5,args.target_distance)*has_steps-20.*(~alive).float()
            # Once the walking constraints hold, extra lift must not outweigh
            # useful distance. Hard physics/gait eligibility is applied below.
            fitness=torch.where(alive & has_steps.bool(),
                                1000.*last_progress.clamp(-.5,args.target_distance+.5)+.01*score,fitness)
        else:
            fitness=score+5.*completed.min(1).values+completed.sum(1)+10.*last_progress.clamp(-.5,1.)-10.*(~alive).float()
        # Give failed straightness candidates a useful gradient toward the corridor.
        fitness-=100.*last_lateral.abs()
        fitness=torch.where(torch.isfinite(fitness),fitness,-1e6)
        candidate_fitness=fitness.reshape(candidates,args.replicas).min(1).values
        physics_eligible=alive.reshape(candidates,args.replicas).all(1)
        gait_eligible=(completed.min(1).values>=minimum_swings).reshape(candidates,args.replicas).all(1)
        eligible=physics_eligible&gait_eligible
        if args.search_mode=='stride_sweep':
            summary=[]
            for index in range(candidates):
                group=slice(index*args.replicas,(index+1)*args.replicas)
                summary.append({'candidate':index,'stride_amplitude_rad':float(params[index*args.replicas,2]),
                    'primary_parameter':args.sweep_primary,
                    'primary_value':float(params[index*args.replicas,list(PARAMETERS).index(args.sweep_primary)]),
                    'lateral_m':last_lateral[group].tolist(),
                    'secondary_parameter':args.sweep_secondary,
                    'secondary_value':float(params[index*args.replicas,list(PARAMETERS).index(args.sweep_secondary)]) if args.sweep_secondary else None,
                    'eligible':bool(eligible[index]),'forward_m':last_progress[group].tolist(),
                    'survival_s':duration[group].tolist(),'completed_swings':completed[group].tolist(),
                    'max_joint_speed_rad_s':float(max_speed[group].max()),
                    'peak_support_body_weight_ratio':float(max_support[group].max()),
                    'failure_reasons':failure_reasons[group].tolist(),
                    'failure_projected_gravity':failure_gravity[group].tolist(),
                    'failure_root_velocity':failure_velocity[group].tolist(),
                    'max_heading_rad':float(max_heading[group].max()),'max_tilt_rad':float(max_tilt[group].max()),
                    'max_continuous_flight_s':float(flight_max_view[group].max())*.002,
                    'simultaneous_air_fraction':float(flight_total_view[group].max())/round(args.seconds/.002),
                    'worst_fitness':float(candidate_fitness[index])})
            backend.write_json(folder/'stride_sweep.json',summary)
        # All surviving candidates outrank incomplete candidates. If discovery
        # has no eligible candidate yet, retain a clearly labeled diagnostic.
        candidate_fitness=torch.where(physics_eligible,candidate_fitness,candidate_fitness-1e6)
        candidate_fitness=torch.where(gait_eligible,candidate_fitness,candidate_fitness-1e5)
        elite_count=max(2,candidates//8)
        if eligible.any():elite_count=min(elite_count,int(eligible.sum()))
        elite=candidate_fitness.topk(elite_count).indices
        candidate=int(elite[0]);best_z=z[candidate].clone()
        members=slice(candidate*args.replicas,(candidate+1)*args.replicas)
        best=candidate*args.replicas+int(fitness[members].argmin())
        mean=.25*mean+.75*z[elite].mean(0)
        std=(.25*std+.75*z[elite].std(0,unbiased=False)).clamp_min(args.minimum_search_std)
        metrics={'generation':generation,'wall_s':time.perf_counter()-began,'fitness':float(fitness[best]),
            'eligible':bool(eligible[candidate]),'eligible_candidate_count':int(eligible.sum()),
            'physics_eligible_candidate_count':int(physics_eligible.sum()),'minimum_swings_per_foot':minimum_swings,
            'replica_failure_reasons':failure_reasons[members].tolist(),
            'survival_s':float(duration[members].min()),'forward_m':float(last_progress[members].min()),
            'completed_swings':completed[members].min(0).values.tolist(),'peak_clearance_m':peak_clear[members].min(0).values.tolist(),
            'max_action_joint_speed_rad_s':float(max_speed[members].max()),'max_joint_speed_rad_s':float(max_speed[members].max()),'population_survival_fraction':float(alive.float().mean()),
            'peak_support_body_weight_ratio':float(max_support[members].max()),
            'max_continuous_flight_s':float(flight_max_view[members].max())*.002,
            'simultaneous_air_fraction':float(flight_total_view[members].max())/round(args.seconds/.002),
            'replica_lateral_m':last_lateral[members].tolist(),
            'replica_forward_m':last_progress[members].tolist(),'replica_survival_s':duration[members].tolist(),
            'population_max_completed_swings':completed.max(0).values.tolist(),
            'incumbent_probe':{'eligible':bool(eligible[0]),'forward_m':float(last_progress[:args.replicas].min()),
                'completed_swings':completed[:args.replicas].min(0).values.tolist(),
                'failure_reasons':failure_reasons[:args.replicas].tolist(),
                'max_continuous_flight_s':float(flight_max_view[:args.replicas].max())*.002,
                'simultaneous_air_fraction':float(flight_total_view[:args.replicas].max())/round(args.seconds/.002)},
            'parameters':dict(zip(PARAMETERS,params[best].tolist()))}
        with (folder/'generations.jsonl').open('a') as stream:stream.write(json.dumps(metrics)+'\n')
        torch.save({'parameters':params[best].detach().cpu(),'standing_actor_state_dict':standing,
                    'search_mean':mean.detach().cpu(),'search_std':std.detach().cpu()},folder/f'model_{generation}.pt')
        print(json.dumps(metrics),flush=True)
        record_progress('running',generation+1)
        if args.stop_forward_m is not None and metrics['eligible'] and metrics['forward_m']>=args.stop_forward_m:
            break
    metadata.update(wall_s=time.perf_counter()-began,metrics=metrics,
        generations_completed=generation+1,
        stop_reason='candidate_distance_target_reached' if args.stop_forward_m is not None and metrics['eligible'] and metrics['forward_m']>=args.stop_forward_m else 'generation_budget_exhausted',
        checkpoints={x.name:backend.digest(x) for x in folder.glob('model_*.pt')})
    backend.write_json(folder/'training.json',metadata)
    record_progress('completed',generation+1)


if __name__=='__main__':main()
