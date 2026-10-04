"""Frozen-dual hold/yaw grids; training screens are never milestone acceptance.

Both saved learned corrections run during deceleration and restart. Only the
existing left hold gain varies in paired-state mode. Nominal mode also varies
left yaw scale; model, gait, actors and all other turn gains stay frozen.
The separate active-scale mode is attribution only, with recorded joystick
commands and evolving simulated observations; it never exports a policy.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path


def parameter_grid(seed):
    import torch
    if tuple(seed.shape) != (20,) or not torch.isfinite(seed).all():
        raise ValueError('Hold calibration requires finite CEM20 parameters')
    gains = torch.tensor([float(seed[3]), float(seed[3]), -.12, -.09, -.06,
                          -.03, 0., .03, .06, .09, .12], device=seed.device)
    rows = seed.repeat(len(gains), 1)
    rows[:, 3] = gains
    return rows


def nominal_grid(seed):
    import torch
    parameter_grid(seed)  # Shape/finite validation shared with the hold-only grid.
    pairs = [(float(seed[3]),float(seed[17]))]*2
    pairs += [(hold,float(yaw)) for hold in (-.06,-.03,float(seed[3]))
              for yaw in torch.linspace(1.45,1.65,11)]
    rows = seed.repeat(len(pairs),1)
    rows[:,[3,17]] = torch.tensor(pairs,device=seed.device)
    return rows


def feedback_grid(seed):
    """Small existing roll/rate gain grid with two exact incumbent controls."""
    import torch
    parameter_grid(seed)
    rows=seed.repeat(27,1)
    offsets=seed.new_tensor([(a,b) for a in (-.005,-.0025,0.,.0025,.005)
                            for b in (-.0015,-.00075,0.,.00075,.0015)])
    rows[2:,18:20]+=offsets
    if bool((rows[:,18].abs()>.05).any() or (rows[:,19].abs()>.015).any()):
        raise ValueError('Feedback grid exceeds existing controller bounds')
    return rows


def scale_active(baseline, complete, scales):
    """Bounded attribution only; preserve the exact incumbent at scale one."""
    import torch
    if scales.ndim != 1 or len(scales) != len(baseline) or not torch.isfinite(scales).all() or bool(((scales<0)|(scales>1)).any()):
        raise ValueError('Active diagnostic scale must be in0..1 per world')
    return torch.where((scales==1)[:,None],complete,baseline+scales[:,None]*(complete-baseline))


def load_starts(folders, checkpoint, meta, turn, mode, age, digest):
    import numpy as np
    import torch
    allowed = {digest(checkpoint), meta['frozen_dual_checkpoint_sha256']}
    states, histories, sources, command_histories = [], [], [], []
    index = round(age/.02)
    for folder in folders:
        folder = Path(folder).resolve()
        result = json.loads((folder/'dynamics.json').read_text())
        cfg = result['config']
        protocol_ok = (cfg.get('direction') in ('left','backward') and cfg['checkpoint_sha256']==digest(checkpoint)) if mode=='cruise_retention' else (bool(cfg.get('triangle')) and cfg['checkpoint_sha256']==digest(checkpoint)) if mode in ('cruise_attribution','triangle_active') else (cfg.get('direction')=='backward' and cfg['checkpoint_sha256']==digest(checkpoint)) if mode=='active' else cfg.get('turn' if mode=='m5' else 'teleop')
        if cfg['checkpoint_sha256'] not in allowed or not protocol_ok:
            raise ValueError('Warm start checkpoint or protocol mismatch')
        if mode=='triangle_active':
            from algorithms.urdf_learn_wasd_walk import mujoco_milestones as gates
            if (meta.get('residual_variant')!='half_cruise_frozen_dual_v1' or
                any(cfg.get(k) for k in ('turn','teleop','direction')) or
                any(result['metrics'].get(k,1)!=0 for k in ('reset_count','done_count','fall_count')) or
                result['metrics'].get('nonfoot_contacts',True)):
                raise ValueError('Triangle-active attribution requires exact half-cruise, no falls/resets/nonfoot contact')
            gates.check_residual_source(folder,result,meta,checkpoint)
        if any(cfg[key] != value for key,value in dict(forward=.2, assistance=0., dt=.002,
                contact_timeconst=.004, gain_scale=1., noslip_iterations=0).items()):
            raise ValueError('Warm start physics changed')
        for key in ('model_xml_sha256','urdf_sha256','mesh_tree_sha256','backend','mujoco_version'):
            if result['identity'][key] != meta[key]:raise ValueError('Warm start model changed')
        if cfg['checkpoint_sha256']==digest(checkpoint):
            for key in ('turn_source_sha256','residual_source_sha256','active_residual_source_sha256'):
                if result.get(key)!=meta[key]:raise ValueError('Warm start controller source changed')
        hashes = {}
        for filename,key in (('trajectory.npz','trajectory_sha256'),('policy_trace.npz','policy_trace_sha256'),
                             ('foot_trace.json','foot_trace_sha256')):
            hashes[filename] = digest(folder/filename)
            if hashes[filename] != result[key]:raise ValueError('Warm start trace changed')
        feet = json.loads((folder/'foot_trace.json').read_text())
        foot_times=np.array([r['time_s'] for r in feet])
        if (len(foot_times)<round(age/.002) or not np.isfinite(foot_times).all() or
            not np.allclose(foot_times,np.arange(1,len(feet)+1)*.002,atol=1e-6,rtol=0) or
            any(not math.isfinite(v) for r in feet if r['time_s']<=age
                for v in r.values() if isinstance(v,(int,float)))):
            raise ValueError('Warm start foot trace has invalid values or clock')
        # Attribution may inspect a later viable state after an earlier isolated
        # force failure, but requires a clean10s lead-in. This is never training
        # qualification or milestone evidence; every prior violation is retained.
        prior_violations=[dict(time_s=r['time_s'],support_bw=r['support_body_weight_ratio'],speed_rad_s=r['max_joint_speed_rad_s'])
                          for r in feet if r['time_s']<=age and (r['support_body_weight_ratio']>3. or r['max_joint_speed_rad_s']>4.+1e-6)]
        if any(r['time_s'] >= (age-10. if mode=='cruise_attribution' else 0.) for r in prior_violations):
            raise ValueError('Unsafe warm start prefix')
        if result['metrics']['reset_count'] != 0:raise ValueError('Warm start source reset')
        trace = np.load(folder/'policy_trace.npz')
        trajectory = np.load(folder/'trajectory.npz')
        if (len(trace['time']) <= index or abs(trace['time'][index]-age) > 1e-6 or
            len(trajectory['time'])<=index or
            not np.allclose(trace['time'][:index+1],np.arange(index+1)*.02,atol=1e-6,rtol=0) or
            not np.allclose(trace['time'][:index+1],trajectory['time'][:index+1],atol=1e-6,rtol=0)):
            raise ValueError('Warm start clocks differ')
        if (not all(np.isfinite(trajectory[key][:index+1]).all() for key in ('qpos','qvel')) or
            not np.isfinite(trace['observation'][:index+1]).all() or
            np.any(-trace['observation'][:index+1,8]<math.cos(math.pi/6))):
            raise ValueError('Unhealthy warm start state prefix')
        command_histories.append((trace['time'][:index+1],trace['observation'][:index+1,63:66]))
        memory = turn.CommandMemory()
        for t,obs in zip(trace['time'][:index],trace['observation'][:index]):
            if not np.isfinite(obs).all() or -float(obs[8]) < math.cos(math.pi/6):
                raise ValueError('Unhealthy warm start prefix')
            memory.observe(torch.from_numpy(obs)[None],torch.zeros((1,3)),torch.eye(3)[None],float(t))
        if memory.anchor_xy is not None or memory.restart_time is not None or memory.last_yaw_sign != 1:
            raise ValueError('Warm starts must precede the first stop after left turning')
        if mode in ('cruise_attribution','cruise_retention') and (memory.left_cruise_blend!=1. or memory.left_feedback_blend!=0.):
            raise ValueError('Cruise attribution must start in fully settled cruise')
        histories.append(memory)
        states.append(dict(qpos=trajectory['qpos'][index],qvel=trajectory['qvel'][index],
                           previous=trace['observation'][index,43:60]))
        sources.append(dict(directory=str(folder),age_s=age,checkpoint_sha256=cfg['checkpoint_sha256'],
                            prior_physical_violations=prior_violations,source_failures=result['failures'],
                            hashes={**hashes,'dynamics.json':digest(folder/'dynamics.json')}))
    # Scalar memory is valid only when every restored world shares exact command history.
    if any(not all(np.array_equal(a,b) for a,b in zip(item,command_histories[0]))
           for item in command_histories[1:]):
        raise ValueError('Paired warm starts have different command histories')
    if any(vars(item) != vars(histories[0]) for item in histories[1:]):
        raise ValueError('Paired warm starts have different command memories')
    return states, histories[0], sources


def run(args):
    import numpy as np
    import torch
    import warp as wp
    from algorithms.urdf_learn_wasd_walk.landau_rsl_control import configure_model
    from algorithms.urdf_learn_wasd_walk.landau_forward_control import load_gait_source, make_actor, action_scales
    from algorithms.urdf_learn_wasd_walk.landau_cruise_residual import load_source
    from algorithms.urdf_learn_wasd_walk.landau_right_rebase import check_contract
    from algorithms.urdf_learn_wasd_walk.landau_turn_control import install_physics_audit
    from algorithms.urdf_learn_wasd_walk import landau_teleop_contract as teleop
    nominal_mode=args.nominal_grid or args.feedback_nominal_grid
    cruise_diagnostic=getattr(args,'cruise_scale_diagnostic',False) or getattr(args,'cruise_retention_diagnostic',False)
    triangle_active=getattr(args,'triangle_active_scale_diagnostic',False)
    active_diagnostic=args.active_scale_diagnostic or triangle_active
    scale_diagnostic=active_diagnostic or cruise_diagnostic
    backend,batches,_ = configure_model('balanced_hands_v1')
    checkpoint = Path(args.checkpoint).resolve()
    meta = json.loads((checkpoint.parent/'training.json').read_text())
    if meta['checkpoints'][checkpoint.name] != backend.digest(checkpoint):raise ValueError('Checkpoint changed')
    if meta['action_scale'] != action_scales(meta['action_joints'],.4):raise ValueError('Calibration action scales changed')
    blob = torch.load(checkpoint,map_location='cpu',weights_only=False)
    half_cruise=meta.get('residual_variant')=='half_cruise_frozen_dual_v1'
    if triangle_active != half_cruise:raise ValueError('Half-cruise calibration is supported only by triangle-active attribution')
    if half_cruise:
        from algorithms.urdf_learn_wasd_walk import landau_cruise_scale as half
        check_contract=half.check_contract
        half.check_sources(checkpoint.parent,meta,backend.digest)
    elif meta.get('residual_variant')=='right_controller_calibrated_dual_residual_v1':
        from algorithms.urdf_learn_wasd_walk.landau_right_controller_rebase import check_contract
    elif meta.get('residual_variant')=='yaw_hold_calibrated_dual_residual_v1':
        from algorithms.urdf_learn_wasd_walk.landau_yaw_hold_rebase import check_contract
    check_contract(meta,blob,backend.digest)
    turn = load_source(checkpoint.parent/'turn_source.py',meta['turn_source_sha256'],'hold_saved_turn')
    if meta['arguments']['turn_duration'] != 44.:raise ValueError('Calibration requires the current44s turn profile')
    turn.configure_profile(meta['arguments']['turn_duration'])
    base,_ = load_gait_source(checkpoint,meta)
    residual = load_source(checkpoint.parent/'residual_source.py',meta['residual_source_sha256'],'hold_saved_cruise')
    active = load_source(checkpoint.parent/'active_residual_source.py',meta['active_residual_source_sha256'],'hold_saved_active')
    if triangle_active:
        if Path(args.triangle_directory).resolve()==Path(args.triangle_repeat_directory).resolve():
            raise ValueError('Triangle active attribution requires two distinct source runs')
        prepared=[('triangle_active',*load_starts([source],checkpoint,meta,turn,'triangle_active',age,backend.digest),age,end)
                  for source,age,end in [(args.triangle_directory,age,339.) for age in (332.,333.,334.)]+
                                        [(args.triangle_repeat_directory,age,85.) for age in (78.,80.)]]
    elif getattr(args,'cruise_retention_diagnostic',False):
        prepared=[('cruise_retention',*load_starts([source],checkpoint,meta,turn,'cruise_retention',age,backend.digest),age,end)
                  for source,age,end in ((args.left_directory,60.,72.),(args.left_directory,84.,96.),
                                         (args.backward_directory,96.,108.),(args.backward_directory,140.,152.))]
    elif cruise_diagnostic:
        prepared=[('cruise_attribution',*load_starts([args.triangle_directory],checkpoint,meta,turn,'cruise_attribution',age,backend.digest),age,204.6)
                  for age in (198.,200.,202.)]
    elif args.feedback_grid:
        prepared = [('active',*load_starts([args.backward_directory],checkpoint,meta,turn,'active',age,backend.digest),age,end)
                    for age,end in ((137.,145.),(138.,145.),(139.,145.),(111.,119.),(126.,134.),(4.,15.))]
    elif args.active_scale_diagnostic:
        prepared = [('active',*load_starts([args.backward_directory],checkpoint,meta,turn,'active',age,backend.digest),age,145.)
                    for age in (137.,138.,139.)]
    elif nominal_mode:
        prepared = [(mode,[],turn.CommandMemory(),[],0.,end) for mode,end in [('m5',56.),('m6',60.)]]
    else:
        prepared = [(mode,*load_starts(folders,checkpoint,meta,turn,mode,age,backend.digest),age,end)
                    for mode,folders,age,end in [('m5',args.m5_directory,46.8,56.),('m6',args.m6_directory,18.,60.)]]
    folder = backend.OUTPUT/args.name
    folder.mkdir(exist_ok=False)
    (folder/'screen_source.py').write_bytes(Path(__file__).read_bytes())
    if half_cruise:(folder/'cruise_scale_source.py').write_bytes((checkpoint.parent/'cruise_scale_source.py').read_bytes())
    (folder/'audit_source.py').write_bytes(Path(install_physics_audit.__code__.co_filename).read_bytes())
    if backend.digest(teleop.__file__) != backend.digest(checkpoint.parent/'teleop_contract.py'):
        raise ValueError('Calibration teleop protocol differs from saved controller')
    (folder/'teleop_source.py').write_bytes(Path(teleop.__file__).read_bytes())
    torch.set_num_threads(4);torch.manual_seed(42)
    grid = (blob['turn_parameters'].repeat(3 if cruise_diagnostic else 5,1) if scale_diagnostic else
            (feedback_grid if args.feedback_grid or args.feedback_nominal_grid else nominal_grid if nominal_mode else parameter_grid)(blob['turn_parameters'])).to('cuda')
    prior = make_actor(63,'cuda');prior.load_state_dict(blob['standing_actor_state_dict']);prior.eval()
    cruise_actor = residual.make_actor('cuda');cruise_actor.load_state_dict(blob['residual_actor_state_dict']);cruise_actor.eval()
    active_actor = residual.make_actor('cuda');active_actor.load_state_dict(blob['active_residual_actor_state_dict']);active_actor.eval()
    walking = blob['parameters'].to('cuda')
    results = {}
    for mode,states,memory,sources,start,end in prepared:
        width = 4 if nominal_mode else 2*len(states)
        n = len(grid)*width
        batch = batches.WarpBatch(n,42,0.,'stand','load',.004,.2,.0225)
        if hashlib.sha256(batch.xml.encode()).hexdigest() != meta['model_xml_sha256']:raise ValueError('Calibration model changed')
        audit = install_physics_audit(batch,batches,n)
        if nominal_mode:
            from algorithms.urdf_learn_wasd_walk.landau_direction_training import common_start_offsets
            (folder/'direction_training_source.py').write_bytes(Path(common_start_offsets.__code__.co_filename).read_bytes())
            offsets = common_start_offsets(batch.nominal,batch.jq,width,42)
            batch.q[:] = batch.nominal+offsets.repeat(len(grid),1)
            batch.v.zero_();batch.previous.zero_()
            sources = [{'nominal_start_offsets_qpos':offsets.cpu().tolist(),'seed':42}]
        else:
            for target,key in ((batch.q,'qpos'),(batch.v,'qvel'),(batch.previous,'previous')):
                target[:] = torch.tensor(np.array([s[key] for s in states]),device='cuda').repeat(len(grid)*2,1)
        batch.ctrl[:] = batch.targets
        wp.to_torch(batch.wd.qacc_warmstart).zero_();wp.to_torch(batch.wd.qfrc_applied).zero_();batch.wrench.zero_()
        wp.to_torch(batch.wd.time).fill_(start)
        wp.capture_launch(batch.forward_graph);wp.synchronize()
        for value in audit.values():value.zero_()
        params = grid.repeat_interleave(width,dim=0)
        scales = torch.tensor([0.,.5,1.] if cruise_diagnostic else [0.,.25,.5,.75,1.],device='cuda').repeat_interleave(width) if scale_diagnostic else None
        replay=None
        if mode in ('active','cruise_attribution','cruise_retention','triangle_active'):
            with np.load(Path(sources[0]['directory'])/'policy_trace.npz') as saved:
                replay={key:saved[key] for key in ('time','observation')}
        if replay is not None:
            stop = round(end/.02)+1
            if (len(replay['time']) < stop or not np.allclose(replay['time'][:stop],np.arange(stop)*.02,atol=1e-6,rtol=0) or
                not np.isfinite(replay['observation'][:stop]).all()):
                raise ValueError('Active replay has a missing or invalid future command suffix')
        names = batch.spec['action_joints']
        limits = torch.tensor(action_scales(names,.4),device='cuda')/.08
        weight = float(batch.model.body_mass.sum())*9.81
        live = torch.ones(n,dtype=torch.bool,device='cuda')
        bits = torch.zeros(n,dtype=torch.int,device='cuda')
        peak = torch.zeros(n,device='cuda')
        foot_peak = torch.zeros((n,2),device='cuda')
        first_failure = torch.full((n,),float('nan'),device='cuda')
        hold_error = torch.zeros(n,device='cuda');hold_drift = torch.zeros_like(hold_error)
        hold_speed = torch.zeros_like(hold_error);hold_yaw_speed = torch.zeros_like(hold_error)
        holds = () if mode in ('active','cruise_attribution','cruise_retention','triangle_active') else ((51.,56.),) if mode == 'm5' else teleop.HOLDS
        max_heading_deviation = torch.zeros(n,device='cuda')
        body_progress = torch.zeros(n,device='cuda')
        end_heading = torch.zeros(n,device='cuda')
        origin = batch.pos[:,batch.pelvis,:2].clone()
        terminal_xy = origin.clone()
        hold_origins = {};hold_headings = {}
        durations = torch.zeros(n,device='cuda')
        last_xy = batch.pos[:,batch.pelvis,:2].clone()
        for step in range(round((end-start)/.02)):
            age = round((start+step*.02)/.02)*.02
            if step%500 == 0:print(json.dumps(dict(mode=mode,age_s=age,live=int(live.sum()))),flush=True)
            if mode == 'm5':forward,yaw,_ = turn.command_profile(age);command = (forward,0.,yaw)
            elif mode in ('active','cruise_attribution','cruise_retention','triangle_active'):command = replay['observation'][round(age/.02),63:66]
            else:command = teleop.command_profile(age)
            rot = batch.rot[:,batch.base]
            extra = torch.zeros((n,10),device='cuda')
            extra[:,2] = 1.-age/30.;extra[:,3:6] = torch.tensor(command,device='cuda')
            extra[:,6] = math.sin(2*math.pi*age/.6);extra[:,7] = math.cos(2*math.pi*age/.6)
            extra[:,8] = rot[:,1,0];extra[:,9] = rot[:,0,0]-1.
            obs = torch.cat((batch.observations(),extra),dim=1)
            with torch.no_grad():
                obs,prior_obs = memory.observe(obs,batch.pos[:,batch.pelvis],rot,age)
                action = turn.commanded_action(base,walking,prior,obs,prior_obs,params,names,
                                               memory.reference,memory.turned,memory=memory)
                baseline=action
                action = residual.apply(cruise_actor,action,obs,walking,memory,names)
                if half_cruise:action=half.apply(baseline,action)
                if cruise_diagnostic:action=scale_active(baseline,action,scales)
                complete = active.apply(active_actor,action,obs,walking,memory,names,residual)
                action = (scale_active(action,complete,scales) if active_diagnostic else complete).clamp(-limits,limits)
            heading = torch.atan2(rot[:,1,0],rot[:,0,0])
            before_heading = heading.clone()
            for a,b in holds:
                # M5 drift begins at49s, but heading acceptance begins at51s.
                anchor_age = a-2. if mode == 'm5' else a
                if abs(age-anchor_age) < 1e-8:
                    hold_origins[a] = batch.pos[:,batch.pelvis,:2].clone()
                    hold_headings[a] = heading.clone()
            batch.ctrl[:] = batch.targets
            batch.ctrl[:,batch.aids] += .08*torch.where(live[:,None],action,0.)
            batch.wrench.zero_();wp.capture_launch(batch.graph);wp.synchronize()
            occupancy = batch.maximum_occupancy.numpy()
            if max(occupancy[:2]) >= batch.wd.naconmax or occupancy[2] >= batch.wd.njmax:raise RuntimeError('Calibration capacity exceeded')
            batch.previous[:] = action
            xy = batch.pos[:,batch.pelvis,:2]
            heading = torch.atan2(batch.rot[:,batch.base,1,0],batch.rot[:,batch.base,0,0])
            if mode in ('active','cruise_attribution','cruise_retention','triangle_active'):
                teacher = replay['observation'][round((age+.02)/.02)]
                target = math.atan2(float(teacher[68]),float(teacher[69]+1.))
                error = torch.atan2((heading-target).sin(),(heading-target).cos()).abs()
                max_heading_deviation = torch.maximum(max_heading_deviation,torch.where(live,error,0.))
                delta = xy-last_xy
                body_progress += torch.where(live,-before_heading.sin()*delta[:,0]+before_heading.cos()*delta[:,1],0.)
                terminal_xy = torch.where(live[:,None],xy,terminal_xy)
                end_heading = torch.where(live,heading,end_heading)
            tilt = batch.rot[:,batch.base,2,2].clamp(-1,1).acos()
            failure = (tilt>math.pi/6).int()+2*(batch.pos[:,batch.pelvis,2]<batch.reference_height-.08).int()
            failure += 4*(audit['speed']>4.+1e-6).int()+8*(audit['force']>3.*weight).int()
            failure += 16*(audit['nonfoot']!=0).int()+32*(audit['flight']>60).int()
            failure += 64*(~torch.isfinite(batch.q).all(1)|~torch.isfinite(batch.v).all(1)).int()
            peak = torch.maximum(peak,torch.where(live,audit['force']/weight,0.))
            foot_peak = torch.maximum(foot_peak,torch.where(live[:,None],audit['foot_peak']/weight,0.))
            first_failure = torch.where(live & (failure!=0),age+.02,first_failure)
            angular_world = torch.bmm(batch.rot[:,batch.base],batch.v[:,3:6,None]).squeeze(-1)
            velocity = batch.v[:,:3]+torch.cross(angular_world,batch.pos[:,batch.pelvis]-batch.q[:,:3],dim=1)
            for a,b in holds:
                anchor_age = a-2. if mode == 'm5' else a
                if anchor_age-1e-8 <= age+.02 <= b+1e-8 and a in hold_origins:
                    drift = (xy-hold_origins[a]).norm(dim=1)
                    hold_drift = torch.maximum(hold_drift,torch.where(live,drift,0.))
                if a-1e-8 <= age+.02 <= b+1e-8 and a in hold_origins:
                    target = math.pi/2 if mode == 'm5' else hold_headings[a]
                    error = torch.atan2((heading-target).sin(),(heading-target).cos()).abs()
                    hold_error = torch.maximum(hold_error,torch.where(live,error,0.))
                    if mode == 'm5':
                        hold_speed = torch.maximum(hold_speed,torch.where(live,velocity[:,:2].norm(dim=1),0.))
                        hold_yaw_speed = torch.maximum(hold_yaw_speed,torch.where(live,angular_world[:,2].abs(),0.))
                    elif age+.02 >= a+2.-1e-8:
                        hold_speed = torch.maximum(hold_speed,torch.where(live,(xy-last_xy).norm(dim=1)/.02,0.))
            bits = torch.where(live & (failure!=0),failure,bits)
            durations += live.float()*.02
            live &= failure == 0
            last_xy = xy.clone()
        if set(hold_origins) != {a for a,b in holds}:raise ValueError('Incomplete calibration hold windows')
        air_fraction = audit['flight_total'].float()*.002/(end-start)
        eligible = (air_fraction<=.05)&(bits==0)&(hold_error<=math.pi/36)&(hold_drift<=.03)&(hold_speed<=.05)&(hold_yaw_speed<=.1)
        result_key = mode+'_'+str(start) if mode in ('active','cruise_attribution','cruise_retention','triangle_active') else mode
        results[result_key] = dict(sources=sources,start_s=start,end_s=end,
            candidate_records=[dict(index=i,gain=float(grid[i,3]),left_yaw_scale=float(grid[i,17]),
                active_scale=float(scales[i*width]) if active_diagnostic else 1.,
                cruise_scale=float(scales[i*width]) if cruise_diagnostic else .5 if half_cruise else 1.,
                first_failure_control_time_s=[v if math.isfinite(v) else None for v in first_failure[i*width:(i+1)*width].tolist()],
                per_foot_peak_bw=foot_peak[i*width:(i+1)*width].tolist(),
                max_heading_deviation_rad=max_heading_deviation[i*width:(i+1)*width].tolist(),
                body_forward_progress_m=body_progress[i*width:(i+1)*width].tolist(),
                terminal_xy_m=terminal_xy[i*width:(i+1)*width].tolist(),
                final_heading_rad=end_heading[i*width:(i+1)*width].tolist(),eligible=eligible[i*width:(i+1)*width].tolist(),
                failure_bits=bits[i*width:(i+1)*width].tolist(),air_fraction=air_fraction[i*width:(i+1)*width].tolist(),peak_support_bw=peak[i*width:(i+1)*width].tolist(),
                hold_error_rad=hold_error[i*width:(i+1)*width].tolist(),hold_drift_m=hold_drift[i*width:(i+1)*width].tolist(),
                hold_speed_mps=hold_speed[i*width:(i+1)*width].tolist(),hold_yaw_speed_rad_s=hold_yaw_speed[i*width:(i+1)*width].tolist(),duration_s=durations[i*width:(i+1)*width].tolist())
                for i in range(len(grid))])
        del batch,audit
    backend.write_json(folder/'screen.json',dict(kind='recorded_state_triangle_active_scale_attribution' if triangle_active else 'recorded_state_cruise_retention' if getattr(args,'cruise_retention_diagnostic',False) else 'recorded_state_cruise_scale_attribution' if cruise_diagnostic else 'nominal_common_start_feedback_calibration' if args.feedback_nominal_grid else 'recorded_state_feedback_calibration' if args.feedback_grid else 'recorded_state_active_scale_attribution' if args.active_scale_diagnostic else 'nominal_common_start_yaw_hold_calibration' if nominal_mode else 'paired_recorded_state_hold_calibration',canonical_milestone_pass=False,
        checkpoint=str(checkpoint),checkpoint_sha256=backend.digest(checkpoint),training_sha256=backend.digest(checkpoint.parent/'training.json'),
        source_sha256=backend.digest(__file__),turn_source_sha256=meta['turn_source_sha256'],
        frozen_cruise_output_scale=.5 if half_cruise else 1.,
        cruise_scale_source_sha256=meta['cruise_scale_source_sha256'] if half_cruise else None,
        residual_source_sha256=meta['residual_source_sha256'],active_residual_source_sha256=meta['active_residual_source_sha256'],
        audit_source_sha256=backend.digest(Path(install_physics_audit.__code__.co_filename)),
        teleop_source_sha256=backend.digest(folder/'teleop_source.py'),
        common_start_source_sha256=backend.digest(folder/'direction_training_source.py') if nominal_mode else None,
        candidate_parameters=grid.cpu().tolist(),changed_indices=[18,19] if args.feedback_grid or args.feedback_nominal_grid else [] if scale_diagnostic else [3,17] if nominal_mode else [3],results=results,
        screen_criteria=('Physical survival and force margin across late, healthy renewed-turn and early-turn windows; independent nominal proofs required.' if args.feedback_grid else 'Physical survival/forces only; residual-scale attribution is not training or qualification.' if scale_diagnostic else
            'Physical survival, force, joint speed, contacts, flight and hold metrics only; M6 block response/qualified swings still require nominal validation.'),
        limitations=('Diagnostic-only restored states after earlier source force failure; clean10s lead-in; source violations retained; NO export/qualification. ' if args.cruise_scale_diagnostic else '')+('Recorded pre-action states and full command memory; cleared solver warmstart; recorded joystick replay, not nominal navigation or acceptance. ' if scale_diagnostic or args.feedback_grid else 'Common nominal/perturbed starts with simultaneous batched duplicate controls; no independent serial proof. ' if nominal_mode else 'Cleared solver warmstart; correlated states and batched copies, not exact continuation or nominal acceptance. M6 prefix before18s is not simulated. ')+'Hold metrics sampled20ms; physical force/speed/contact guards sampled2ms.'))
    print(json.dumps({mode:[sum(r['eligible']) for r in result['candidate_records']] for mode,result in results.items()}),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name',required=True);parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--nominal-grid',action='store_true')
    parser.add_argument('--active-scale-diagnostic',action='store_true')
    parser.add_argument('--triangle-active-scale-diagnostic',action='store_true')
    parser.add_argument('--triangle-repeat-directory')
    parser.add_argument('--cruise-scale-diagnostic',action='store_true')
    parser.add_argument('--cruise-retention-diagnostic',action='store_true')
    parser.add_argument('--left-directory')
    parser.add_argument('--triangle-directory')
    parser.add_argument('--feedback-grid',action='store_true')
    parser.add_argument('--feedback-nominal-grid',action='store_true')
    parser.add_argument('--backward-directory')
    parser.add_argument('--m5-directory',action='append',default=[])
    parser.add_argument('--m6-directory',action='append',default=[])
    args = parser.parse_args()
    if sum((args.nominal_grid,args.feedback_grid,args.active_scale_diagnostic,args.feedback_nominal_grid,args.cruise_scale_diagnostic,args.cruise_retention_diagnostic,args.triangle_active_scale_diagnostic))>1:
        raise ValueError('Choose exactly one calibration mode')
    if args.triangle_active_scale_diagnostic:
        if (not args.triangle_directory or not args.triangle_repeat_directory or args.left_directory or
            args.backward_directory or args.m5_directory or args.m6_directory):
            raise ValueError('Triangle active attribution requires exactly two declared triangle source directories')
    elif args.triangle_repeat_directory:raise ValueError('Triangle repeat directory requires triangle active attribution')
    elif args.cruise_retention_diagnostic:
        if not args.left_directory or not args.backward_directory or args.triangle_directory or args.m5_directory or args.m6_directory:raise ValueError('Cruise retention requires left and backward source directories only')
    elif args.left_directory:raise ValueError('Left directory requires cruise retention')
    elif args.cruise_scale_diagnostic:
        if not args.triangle_directory or args.backward_directory or args.m5_directory or args.m6_directory:raise ValueError('Cruise attribution requires only a triangle source directory')
    elif args.triangle_directory:raise ValueError('Triangle directory requires cruise attribution')
    elif args.active_scale_diagnostic or args.feedback_grid:
        if (args.active_scale_diagnostic and args.feedback_grid) or args.nominal_grid or not args.backward_directory or args.m5_directory or args.m6_directory:
            raise ValueError('Recorded-state modes require exactly one mode and one backward directory')
    elif args.backward_directory:raise ValueError('Backward directory requires a recorded-state mode')
    elif args.nominal_grid or args.feedback_nominal_grid:
        if args.m5_directory or args.m6_directory:raise ValueError('Nominal grid cannot use warm starts')
    elif len(args.m5_directory)!=2 or len(args.m6_directory)!=2:
        raise ValueError('Require paired passing and failed M5/M6 states')
    run(args)


if __name__ == '__main__':main()
