"""Paired warm-state residual diagnostics; never milestone certification."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def project_active_delta(action,baseline,beta,names):
    """Diagnostic convex projection; beta1 preserves the exact original action."""
    import torch
    left,right=[names.index(side+'_hip_roll_joint') for side in ('left','right')]
    dl=action[:,left]-baseline[:,left];dr=action[:,right]-baseline[:,right]
    common=(dl+dr)*.5;difference=(dl-dr)*.5
    result=action.clone()
    result[:,left]=torch.where(beta==1.,action[:,left],baseline[:,left]+common+beta*difference)
    result[:,right]=torch.where(beta==1.,action[:,right],baseline[:,right]+common-beta*difference)
    return result


def build_late_diagnostic_bank(folder,checkpoint,meta,turn,output,digest):
    """Healthy pre-failure states for attribution only; never acceptance evidence."""
    import numpy as np
    import torch
    folder=Path(folder).resolve();result=json.loads((folder/'dynamics.json').read_text());cfg=result['config']
    if cfg['checkpoint_sha256']!=digest(checkpoint) or cfg.get('direction')!='backward':raise ValueError('Late diagnostic checkpoint/protocol mismatch')
    if (cfg['forward']!=.2 or cfg['assistance']!=0. or cfg['dt']!=.002 or
        cfg['contact_timeconst']!=.004 or cfg['gain_scale']!=1.):raise ValueError('Late diagnostic physics mismatch')
    for key in ('backend','mujoco_version','model_xml_sha256','urdf_sha256','mesh_tree_sha256'):
        if result['identity'][key]!=meta[key]:raise ValueError('Late diagnostic model mismatch')
    for filename,key in (('foot_trace.json','foot_trace_sha256'),('policy_trace.npz','policy_trace_sha256'),('trajectory.npz','trajectory_sha256')):
        if digest(folder/filename)!=result[key]:raise ValueError('Late diagnostic source changed')
    if result['metrics']['reset_count']!=0 or result['metrics']['duration_s']<=146.:raise ValueError('Late diagnostic prefix missing or reset')
    feet=json.loads((folder/'foot_trace.json').read_text())
    first_bad=min((r['time_s'] for r in feet if r['support_body_weight_ratio']>3. or r['max_joint_speed_rad_s']>4.+1e-6),default=float('inf'))
    if first_bad<=146.:raise ValueError('Unsafe late diagnostic prefix')
    trajectory=np.load(folder/'trajectory.npz');trace=np.load(folder/'policy_trace.npz')
    count=len(trace['time'])
    # A fall partway through a control interval has no final control-time frame.
    if len(trajectory['time']) not in (count,count+1) or not np.allclose(trajectory['time'][:count],trace['time'],rtol=0,atol=1e-6):raise ValueError('Late diagnostic clocks differ')
    memory=turn.CommandMemory();rows=[]
    heading=np.unwrap(np.arctan2(trace['observation'][:,68],trace['observation'][:,69]+1.))
    for i,(age,obs) in enumerate(zip(trace['time'],trace['observation'])):
        if age>145.8+1e-6:break
        if not np.isfinite(obs).all() or -float(obs[8])<math.cos(math.pi/6):raise ValueError('Unhealthy late diagnostic prefix tilt/state')
        memory.observe(torch.from_numpy(obs)[None],torch.zeros(1,3),torch.eye(3)[None],float(age))
        if 140.-1e-6<=age and i%10==0:
            if memory.last_yaw_sign!=1 or memory.restart_time is not None or not memory.previous_moving:raise ValueError('Unsupported late memory')
            rows.append(dict(qpos=trajectory['qpos'][i],qvel=trajectory['qvel'][i],previous=obs[43:60],age=age,
                reference=memory.reference,blend=memory.left_cruise_blend,feedback=memory.left_feedback_blend,
                cruise_start=memory.left_cruise_start if memory.left_cruise_start is not None else float('inf'),
                previous_yaw=float(obs[65]),mode=0,previous_moving=True,stop_xy=np.zeros(2),stop_time=float('inf'),hold_heading=0.,source_index=i))
    if len(rows)!=30:raise ValueError('Incomplete late diagnostic states')
    bank={key:np.array([r[key] for r in rows]) for key in rows[0]}
    # Teacher errors are diagnostic only and clamped beyond the recorded trace.
    bank['teacher_heading']=np.repeat(np.pad(heading,(0,max(0,9001-len(heading))),mode='edge')[None],3,axis=0)
    np.savez_compressed(output,**bank)
    return {'source_directory':str(folder),'source_hashes':{name:digest(folder/name) for name in ('dynamics.json','trajectory.npz','policy_trace.npz','foot_trace.json')},
        'first_unsafe_force_or_speed_s':first_bad if math.isfinite(first_bad) else None,'start_range_s':[140.,145.8],
        'scope':'Healthy prefix of failed nominal run; diagnostic resets only, cleared solver warmstart. Teacher heading clamped beyond recorded end; no acceptance claim.'}


def run(args):
    import numpy as np
    import torch
    import warp as wp
    from algorithms.urdf_learn_wasd_walk.landau_rsl_control import configure_model
    from algorithms.urdf_learn_wasd_walk.landau_forward_control import load_gait_source,make_actor,action_scales
    from algorithms.urdf_learn_wasd_walk.landau_cruise_residual import load_source,CruiseMemory,FAMILY
    from algorithms.urdf_learn_wasd_walk.landau_turn_control import install_physics_audit
    from algorithms.urdf_learn_wasd_walk.landau_direction_training import commands
    backend,batches,_=configure_model('balanced_hands_v1')
    checkpoint=Path(args.checkpoint).resolve();meta=json.loads((checkpoint.parent/'training.json').read_text())
    if meta['policy_family']!=FAMILY or str(meta.get('status','')).startswith('invalidated_') or meta['checkpoints'][checkpoint.name]!=backend.digest(checkpoint):raise ValueError('Residual checkpoint changed')
    folder=backend.OUTPUT/args.name;folder.mkdir(exist_ok=False)
    blob=torch.load(checkpoint,map_location='cpu',weights_only=False)
    from algorithms.urdf_learn_wasd_walk.landau_active_residual import check_contract
    adapt_mode=meta.get('residual_variant')=='adapted_cruise_frozen_active_v1'
    if adapt_mode:
        from algorithms.urdf_learn_wasd_walk.landau_cruise_adaptation import check_contract
        if args.seconds!=16. or any((args.reference_checkpoint,args.late_backward_directory,args.complete_stops,args.early_turn_screen,args.active_projection_grid)):
            raise ValueError('Adapted cruise screen requires a separate16s paired comparison')
    active_mode=check_contract(meta,blob,backend.digest)
    mixed_mode=active_mode and not adapt_mode
    reference_blob=None;reference_identity=None
    if args.reference_checkpoint:
        reference_path=Path(args.reference_checkpoint).resolve()
        reference_meta=json.loads((reference_path.parent/'training.json').read_text())
        if reference_meta.get('policy_family')!=FAMILY or str(reference_meta.get('status','')).startswith('invalidated_') or reference_meta['checkpoints'][reference_path.name]!=backend.digest(reference_path):
            raise ValueError('Reference checkpoint changed')
        reference_blob=torch.load(reference_path,map_location='cpu',weights_only=False)
        if not active_mode or not check_contract(reference_meta,reference_blob,backend.digest):raise ValueError('Reference comparison requires two active residuals')
        for key in ('frozen_cruise_checkpoint_sha256','active_residual_source_sha256','residual_source_sha256','turn_source_sha256','source_sha256','model_xml_sha256'):
            if reference_meta[key]!=meta[key]:raise ValueError('Reference composition mismatch: '+key)
        for source in ('active_residual_source','residual_source','turn_source'):
            if backend.digest(reference_path.parent/(source+'.py'))!=reference_meta[source+'_sha256']:raise ValueError('Reference source changed')
        if reference_meta['curriculum']['bank_sha256']!=meta['curriculum']['bank_sha256'] or reference_meta['arguments']['turn_duration']!=meta['arguments']['turn_duration']:
            raise ValueError('Reference curriculum or command profile differs')
        reference_identity={'checkpoint':str(reference_path),'checkpoint_sha256':backend.digest(reference_path),
            'training_sha256':backend.digest(reference_path.parent/'training.json')}
    if active_mode:
        if mixed_mode and args.seconds!=6. and not args.complete_stops and not args.late_backward_directory:raise ValueError('Mixed active screen requires6s or complete-stop episodes')
        if args.complete_stops and not meta['arguments'].get('active_long'):raise ValueError('Complete-stop screen requires long-episode training')
        from algorithms.urdf_learn_wasd_walk import landau_active_curriculum as active_bank,landau_direction_training as active_direction
        for module in ((active_direction,) if adapt_mode else (active_bank,active_direction)):
            if backend.digest(module.__file__)!=meta['training_source_sha256'][Path(module.__file__).name]:raise ValueError('Active screen training helper changed')
        active_head=load_source(checkpoint.parent/'active_residual_source.py',meta['active_residual_source_sha256'],'screen_saved_active')
    residual=load_source(checkpoint.parent/'residual_source.py',meta['residual_source_sha256'],'screen_saved_residual')
    if args.active_projection_grid and not active_mode:raise ValueError('Projection grid requires active residual')
    if args.early_turn_screen and not active_mode:raise ValueError('Early-turn screen requires active residual')
    if args.complete_stops and not active_mode:raise ValueError('Complete-stop screen requires active residual')
    if args.late_backward_directory and (not active_mode or reference_blob is None):raise ValueError('Late diagnostic requires candidate and reference active heads')
    base,_=load_gait_source(checkpoint,meta)
    turn=load_source(checkpoint.parent/'turn_source.py',meta['turn_source_sha256'],'screen_saved_turn')
    turn.configure_profile(meta['arguments']['turn_duration'])
    bank_path=checkpoint.parent/'warm_states.npz'
    if backend.digest(bank_path)!=meta['curriculum']['bank_sha256']:raise ValueError('Curriculum bank changed')
    late_identity=None
    if args.late_backward_directory:
        bank_path=folder/'late_states.npz'
        late_identity=build_late_diagnostic_bank(args.late_backward_directory,checkpoint,meta,turn,bank_path,backend.digest)
    torch.set_num_threads(4);torch.manual_seed(42)
    groups=6 if args.late_backward_directory else 8 if args.active_projection_grid else 4
    width=30 if args.late_backward_directory or adapt_mode else 48 if active_mode else 32;n=width*groups;batch=batches.WarpBatch(n,42,0.,'stand','load',.004,.2,.0225)
    if hashlib.sha256(batch.xml.encode()).hexdigest()!=meta['model_xml_sha256']:raise ValueError('Screen model changed')
    audit=install_physics_audit(batch,batches,n)
    bank={k:torch.tensor(v,device='cuda') for k,v in np.load(bank_path).items()}
    # Two copies of each controller share the same held-out states.
    if adapt_mode:
        pool=torch.arange(1,len(bank['age']),2,device='cuda')
        if len(pool)!=15:raise ValueError('Adaptation screen requires15 reserved late states')
        indices=pool.repeat(2*groups)
    elif args.late_backward_directory:
        indices=torch.arange(width,device='cuda').repeat(groups);mode=bank['mode'][indices]
    elif active_mode:
        pools=[torch.where(bank['mode']==i)[0][1::2] for i in range(3)]
        if args.early_turn_screen:
            pools=[pool[(bank['age'][pool]>=3.) & (bank['age'][pool]<=7.)] for pool in pools]
            if any(not len(pool) for pool in pools):raise ValueError('Missing reserved early-turn states')
        indices=torch.cat([pool[torch.arange(16,device='cuda')%len(pool)
            if args.early_turn_screen and len(pool)<16 else torch.linspace(0,len(pool)-1,16,device='cuda').long()]
            for pool in pools]).repeat(groups)
        mode=bank['mode'][indices]
    else:indices=(torch.linspace(0,len(bank['age'])//2-1,32,device='cuda').long()*2+1).repeat(4)
    batch.q[:]=bank['qpos'][indices].float();batch.v[:]=bank['qvel'][indices].float()
    batch.previous[:]=bank['previous'][indices].float();batch.ctrl[:]=batch.targets
    wp.to_torch(batch.wd.qacc_warmstart).zero_();wp.to_torch(batch.wd.qfrc_applied).zero_();batch.wrench.zero_()
    wp.to_torch(batch.wd.time)[:]=bank['age'][indices].float()
    age=bank['age'][indices].clone();start_age=age.clone()
    ends=age+args.seconds
    if args.complete_stops:
        saved_trainer=load_source(checkpoint.parent/'training_sources/landau_cruise_residual.py',
            meta['training_source_sha256']['landau_cruise_residual.py'],'screen_saved_trainer')
        ends=saved_trainer.active_episode_end(mode,age)
    horizons=ends-start_age
    steps=round(float(horizons.max())/.02)
    memory=active_bank.ActiveMemory(n,'cuda') if mixed_mode else residual.CruiseMemory(n,'cuda') if adapt_mode else CruiseMemory(n,'cuda')
    if mixed_mode:
        for name in ('previous_moving','stop_xy','stop_time','hold_heading'):getattr(memory,name)[:]=bank[name][indices].to(getattr(memory,name).dtype)
    memory._reference[:]=bank['reference'][indices];memory._blend[:]=bank['blend'][indices]
    memory._feedback[:]=bank['feedback'][indices];memory.left_cruise_start[:]=bank['cruise_start'][indices]
    memory.previous_yaw[:]=bank['previous_yaw'][indices].double()
    for value in audit.values():value.zero_()
    torch.cuda.synchronize();wp.capture_launch(batch.forward_graph);wp.synchronize()
    names=batch.spec['action_joints'];limits=torch.tensor(action_scales(names,.4),device='cuda')/.08
    prior=make_actor(63,'cuda');prior.load_state_dict(blob['standing_actor_state_dict']);prior.eval()
    actor=residual.make_actor('cuda');actor.load_state_dict(blob['residual_actor_state_dict']);actor.eval()
    if active_mode:
        active_actor=residual.make_actor('cuda');active_actor.load_state_dict(blob['active_residual_actor_state_dict']);active_actor.eval()
        if adapt_mode:
            original_path=Path(meta['frozen_active_checkpoint'])
            original_blob=torch.load(original_path,map_location='cpu',weights_only=False)
            original_cruise=residual.make_actor('cuda');original_cruise.load_state_dict(original_blob['residual_actor_state_dict']);original_cruise.eval()
            reference_identity={'checkpoint':str(original_path),'checkpoint_sha256':meta['frozen_active_checkpoint_sha256'],
                'training_sha256':meta['frozen_active_training_sha256']}
        if reference_blob is not None:
            reference_actor=residual.make_actor('cuda');reference_actor.load_state_dict(reference_blob['active_residual_actor_state_dict']);reference_actor.eval()
    max_heading_error=torch.zeros(n,device='cuda');max_hold_drift=torch.zeros_like(max_heading_error)
    max_hold_heading=torch.zeros_like(max_heading_error);body_progress=torch.zeros_like(max_heading_error)
    walking=blob['parameters'].to('cuda');turn_parameters=blob['turn_parameters'].to('cuda')
    live=torch.ones(n,device='cuda',dtype=torch.bool);peak=torch.zeros(n,device='cuda');duration=torch.zeros_like(peak)
    bits=torch.zeros(n,device='cuda',dtype=torch.int);origin=batch.pos[:,batch.pelvis,:2].clone();terminal=origin.clone()
    slip_total=torch.zeros_like(peak);air_total=torch.zeros_like(peak);swings=torch.zeros((n,2),device='cuda',dtype=torch.long)
    air=torch.zeros((n,2),device='cuda');height=torch.zeros_like(air);touched=torch.zeros_like(air,dtype=torch.bool)
    projection_beta=torch.tensor([1.,1.,0.,0.,.5,.5,1.,1.],device='cuda').repeat_interleave(width) if args.active_projection_grid else None
    event_fields=['age_s','heading_rad','command_reference_rad','signed_target_error_rad','signed_teacher_error_rad',
        'local_forward_mps','local_strafe_mps','roll_rad','roll_rate_rad_s','root_height_m']
    events={key:torch.full((n,len(event_fields)),float('nan'),device='cuda') for key in ('yaw_off','residual_off','stop','settled')} if args.active_projection_grid else {}
    event_seen={key:torch.zeros(n,device='cuda',dtype=torch.bool) for key in events}
    max_correction=torch.zeros_like(peak);weight=float(batch.model.body_mass.sum())*9.81
    def observe():
        rotation=batch.rot[:,batch.base].double();heading=torch.atan2(rotation[:,1,0],rotation[:,0,0])
        command=commands('backward',batch.pos[:,batch.pelvis,:2].double()-batch.reference_xy.double(),heading,100.)
        if mixed_mode:command=active_bank.commands(mode,age,batch.pos[:,batch.pelvis,:2].double()-batch.reference_xy.double(),heading,meta['arguments']['turn_duration'])
        extra=torch.zeros((n,10),device='cuda');extra[:,2]=(1.-age/30.).float();extra[:,3:6]=command.float()
        phase=2*torch.pi*age/.6;extra[:,6]=phase.sin().float();extra[:,7]=phase.cos().float()
        extra[:,8]=rotation[:,1,0].float();extra[:,9]=(rotation[:,0,0]-1).float()
        return torch.cat((batch.observations(),extra),dim=1)
    obs=observe()
    for step in range(steps):
        if step%250==0:print(json.dumps({'step':step,'live':int(live.sum())}),flush=True)
        with torch.no_grad():
            prior_obs=memory.prior(obs,batch.pos[:,batch.pelvis],batch.rot[:,batch.base],age) if mixed_mode else obs[:,:63]
            baseline=turn.commanded_action(base,walking,prior,obs,prior_obs,turn_parameters,names,memory.reference,True,memory=memory)
            if adapt_mode:
                comparison=residual.apply(original_cruise,baseline,obs,walking,memory,names)
                comparison=active_head.apply(active_actor,comparison,obs,walking,memory,names,residual)
            if active_mode:
                baseline=residual.apply(actor,baseline,obs,walking,memory,names)
                action=active_head.apply(active_actor,baseline,obs,walking,memory,names,residual)
                if args.active_projection_grid:action=project_active_delta(action,baseline,projection_beta,names)
            else:action=residual.apply(actor,baseline,obs,walking,memory,names)
            if not adapt_mode:comparison=active_head.apply(reference_actor,baseline,obs,walking,memory,names,residual) if reference_blob is not None else baseline
            if args.late_backward_directory:
                action[:width*2]=baseline[:width*2]
                action[width*2:width*4]=comparison[width*2:width*4]
            else:action[:width*2]=comparison[:width*2]
            correction=(action-(comparison if adapt_mode else baseline)).abs().amax(1)*.08
            max_correction=torch.maximum(max_correction,torch.where(live,correction,0.))
            action=action.clamp(-limits,limits)
        measuring=live.clone()
        if args.active_projection_grid:
            before_command=obs[:,63:66].clone();before_envelope=active_head.envelope(memory,obs).clone()
        before_xy=batch.pos[:,batch.pelvis,:2].clone()
        before_heading=torch.atan2(batch.rot[:,batch.base,1,0],batch.rot[:,batch.base,0,0])
        batch.ctrl[:]=batch.targets;batch.ctrl[:,batch.aids]+=.08*torch.where(live[:,None],action,0.);batch.wrench.zero_()
        torch.cuda.synchronize();wp.capture_launch(batch.graph);wp.synchronize()
        occupancy=batch.maximum_occupancy.numpy()
        if max(occupancy[:2])>=batch.wd.naconmax or occupancy[2]>=batch.wd.njmax:raise RuntimeError('Screen capacity exceeded')
        batch.previous[:]=action;age+=live.double()*.02
        tilt=batch.rot[:,batch.base,2,2].clamp(-1,1).acos()
        failure=(tilt>torch.pi/6).int()+2*(batch.pos[:,batch.pelvis,2]<batch.reference_height-.08).int()
        failure+=4*(audit['speed']>4.+1e-6).int()+8*(audit['force']>3.*weight).int()
        failure+=16*(audit['nonfoot']!=0).int()+32*(audit['flight']>60).int()
        failure+=64*(~torch.isfinite(batch.q).all(1)|~torch.isfinite(batch.v).all(1)).int()
        peak=torch.maximum(peak,torch.where(live,audit['force']/weight,0.));duration+=live.float()*.02
        terminal=torch.where(live[:,None],batch.pos[:,batch.pelvis,:2],terminal)
        delta=batch.pos[:,batch.pelvis,:2]-before_xy
        body_progress+=torch.where(live,-before_heading.sin()*delta[:,0]+before_heading.cos()*delta[:,1],0.)
        contacts,slip,clearance,_=batch.feet(with_forces=True)
        swings+=(contacts&touched&(air>=.06)&(height>=.015)&live[:,None]).long()
        slip_total+=torch.where(live,slip,0.)*.02;air_total+=torch.where(live,(~contacts.any(1)).float(),0.)*.02
        air=torch.where(contacts,0.,air+.02);height=torch.where(contacts,0.,torch.maximum(height,clearance));touched|=contacts
        if mixed_mode:
            heading=torch.atan2(batch.rot[:,batch.base,1,0],batch.rot[:,batch.base,0,0])
            teacher=bank['teacher_heading'][mode,torch.round(age/.02).long()]
            error=torch.atan2((heading-teacher).sin(),(heading-teacher).cos()).abs().float()
            max_heading_error=torch.maximum(max_heading_error,torch.where(live,error,0.))
            drift=(batch.pos[:,batch.pelvis,:2]-memory.stop_xy).norm(dim=1)
            held=torch.isfinite(memory.stop_time)&~memory.previous_moving
            max_hold_drift=torch.maximum(max_hold_drift,torch.where(live&held,drift,0.))
            target=torch.where(mode==1,torch.pi/2,memory.hold_heading)
            error=torch.atan2((heading-target).sin(),(heading-target).cos()).abs()
            max_hold_heading=torch.maximum(max_hold_heading,torch.where(live&held,error,0.))
            if meta['arguments'].get('active_long'):
                settled=held & (age-memory.stop_time>=2.-1e-8)
                angular_world=torch.bmm(batch.rot[:,batch.base],batch.v[:,3:6,None]).squeeze(-1)
                velocity=batch.v[:,:3]+torch.cross(angular_world,batch.pos[:,batch.pelvis]-batch.q[:,:3],dim=1)
                failure+=128*(held & (drift>.03)).int()+256*(settled & (error>torch.pi/36)).int()
                failure+=512*(settled & (velocity[:,:2].norm(dim=1)>.05)).int()+1024*(settled & (angular_world[:,2].abs()>.1)).int()
        bits=torch.where(live&(failure!=0),failure,bits);live&=(failure==0)&(age<ends-1e-8)
        obs=observe()
        if mixed_mode:memory.advance_command(obs,batch.pos[:,batch.pelvis],age,.02)
        else:memory.advance(obs[:,65].double(),age,.02)
        if args.active_projection_grid:
            heading=torch.atan2(obs[:,68],obs[:,69]+1.)
            teacher=bank['teacher_heading'][mode,torch.round(age/.02).long()]
            target=torch.where(mode==1,torch.pi/2,torch.where(torch.isfinite(memory.stop_time),memory.hold_heading,memory.reference))
            signed=torch.atan2((heading-target).sin(),(heading-target).cos())
            teacher_error=torch.atan2((heading-teacher).sin(),(heading-teacher).cos())
            velocity,_=batch.local_velocity()
            values=torch.stack((age,heading,memory.reference,signed,teacher_error,velocity[:,1],velocity[:,0],
                torch.atan2(-obs[:,6],-obs[:,8]),obs[:,4],batch.pos[:,batch.pelvis,2]),1).float()
            masks={'yaw_off':(before_command[:,2]>0)&(obs[:,65]==0),
                'residual_off':(before_envelope>0)&(active_head.envelope(memory,obs)==0),
                'stop':(before_command.abs().amax(1)>0)&(obs[:,63:66].abs().amax(1)==0),
                'settled':torch.isfinite(memory.stop_time)&(age-memory.stop_time>=2.-1e-8)}
            for key,mask in masks.items():
                take=mask & measuring & ~event_seen[key]
                events[key]=torch.where(take[:,None],values,events[key]);event_seen[key]|=take
    records=[]
    for group in range(groups):
        selection=slice(group*width,(group+1)*width)
        label=('frozen' if group<2 else 'reference' if group<4 else 'residual') if args.late_backward_directory else ('reference' if reference_blob is not None or adapt_mode else 'frozen') if group<2 else 'residual'
        records.append({'controller':label,'copy':group%2,
            'projection_beta':None if group<2 or not args.active_projection_grid else float(projection_beta[group*width]),
            'events':{key:{'seen':event_seen[key][selection].tolist(),'values':[[value if math.isfinite(value) else None for value in row] for row in events[key][selection].tolist()]} for key in events},
            'survived_count':int(((bits==0)&(age>=ends-1e-8))[selection].sum()),'start_ages_s':start_age[selection].tolist(),
            'terminal_age_s':age[selection].tolist(),'required_duration_s':horizons[selection].tolist(),'duration_s':duration[selection].tolist(),'failure_bits':bits[selection].tolist(),
            'body_forward_progress_m':body_progress[selection].tolist(),
            'peak_support_bw':peak[selection].tolist(),'backward_progress_m':(-(terminal-origin)[:,1])[selection].tolist(),
            'mean_slip_mps':(slip_total/duration.clamp_min(.02))[selection].tolist(),
            'air_fraction':(air_total/duration.clamp_min(.02))[selection].tolist(),'qualified_swings':swings[selection].tolist(),
            'max_correction_rad':max_correction[selection].tolist(),
            **({'modes':mode[selection].tolist(),'max_teacher_heading_error_rad':max_heading_error[selection].tolist(),
                'max_hold_drift_m':max_hold_drift[selection].tolist(),'max_hold_heading_error_rad':max_hold_heading[selection].tolist()} if mixed_mode else {})})
    backend.write_json(folder/'screen.json',{'kind':'paired_training_state_diagnostic','canonical_milestone_pass':False,
        'checkpoint':str(checkpoint),'checkpoint_sha256':backend.digest(checkpoint),'seconds':None if args.complete_stops else args.seconds,'complete_stops':args.complete_stops,'max_screen_seconds':steps*.02,
        'source_sha256':backend.digest(__file__),'residual_source_sha256':meta['residual_source_sha256'],
        'audit_source_sha256':backend.digest(Path(install_physics_audit.__code__.co_filename)),
        'curriculum_bank_sha256':backend.digest(bank_path),'records':records,
        'residual_variant':meta.get('residual_variant'),'active_residual_source_sha256':meta.get('active_residual_source_sha256'),
        'memory_source_sha256':backend.digest(active_bank.__file__) if mixed_mode else meta['residual_source_sha256'] if adapt_mode else backend.digest(Path(__file__).with_name('landau_cruise_residual.py')),
        'active_projection_grid':args.active_projection_grid,'event_fields':event_fields if events else [],
        'early_turn_screen':args.early_turn_screen,
        'reference_identity':reference_identity,
        'late_diagnostic_identity':late_identity,
        'direction_source_sha256':backend.digest(active_direction.__file__) if active_mode else None,
        'comparison':'original versus adapted cruise with identical frozen active correction' if adapt_mode else 'frozen cruise alone versus previous and current active heads on identical late states' if late_identity else 'reference active head versus candidate active head' if reference_blob is not None else 'frozen cruise versus frozen cruise plus active head' if active_mode else 'frozen CEM versus cruise residual',
        'limitations':'Adjacent held-out states share one trajectory; repeats are simultaneousbatch copies, not independent nominal-start proof.'})
    print(json.dumps([{'controller':r['controller'],'copy':r['copy'],'projection_beta':r['projection_beta'],'survived':r['survived_count'],'worst_peak_bw':max(r['peak_support_bw'])} for r in records]),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name',required=True);parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--seconds',type=float,default=20.)
    parser.add_argument('--complete-stops',action='store_true')
    parser.add_argument('--active-projection-grid',action='store_true')
    parser.add_argument('--early-turn-screen',action='store_true')
    parser.add_argument('--reference-checkpoint')
    parser.add_argument('--late-backward-directory')
    args=parser.parse_args()
    if not 6.<=args.seconds<=30.:raise ValueError('Screen horizon must be6..30s')
    if args.early_turn_screen and (args.complete_stops or args.active_projection_grid):raise ValueError('Early-turn screen is a separate short diagnostic')
    if args.reference_checkpoint and args.active_projection_grid:raise ValueError('Reference comparison and projection grid are separate diagnostics')
    if args.late_backward_directory and (args.complete_stops or args.early_turn_screen or args.active_projection_grid):raise ValueError('Late attribution is a separate diagnostic')
    run(args)


if __name__=='__main__':main()
