"""Paired warm-state residual diagnostics; never milestone certification."""
import argparse
import hashlib
import json
from pathlib import Path


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
    if meta['policy_family']!=FAMILY or meta['checkpoints'][checkpoint.name]!=backend.digest(checkpoint):raise ValueError('Residual checkpoint changed')
    folder=backend.OUTPUT/args.name;folder.mkdir(exist_ok=False)
    blob=torch.load(checkpoint,map_location='cpu',weights_only=False)
    from algorithms.urdf_learn_wasd_walk.landau_active_residual import check_contract
    active_mode=check_contract(meta,blob,backend.digest)
    if active_mode:
        if args.seconds!=6. and not args.complete_stops:raise ValueError('Mixed active screen requires6s or complete-stop episodes')
        if args.complete_stops and not meta['arguments'].get('active_long'):raise ValueError('Complete-stop screen requires long-episode training')
        from algorithms.urdf_learn_wasd_walk import landau_active_curriculum as active_bank,landau_direction_training as active_direction
        for module in (active_bank,active_direction):
            if backend.digest(module.__file__)!=meta['training_source_sha256'][Path(module.__file__).name]:raise ValueError('Active screen training helper changed')
        active_head=load_source(checkpoint.parent/'active_residual_source.py',meta['active_residual_source_sha256'],'screen_saved_active')
    residual=load_source(checkpoint.parent/'residual_source.py',meta['residual_source_sha256'],'screen_saved_residual')
    if args.complete_stops and not active_mode:raise ValueError('Complete-stop screen requires active residual')
    base,_=load_gait_source(checkpoint,meta)
    turn=load_source(checkpoint.parent/'turn_source.py',meta['turn_source_sha256'],'screen_saved_turn')
    turn.configure_profile(meta['arguments']['turn_duration'])
    bank_path=checkpoint.parent/'warm_states.npz'
    if backend.digest(bank_path)!=meta['curriculum']['bank_sha256']:raise ValueError('Curriculum bank changed')
    torch.set_num_threads(4);torch.manual_seed(42)
    width=48 if active_mode else 32;n=width*4;batch=batches.WarpBatch(n,42,0.,'stand','load',.004,.2,.0225)
    if hashlib.sha256(batch.xml.encode()).hexdigest()!=meta['model_xml_sha256']:raise ValueError('Screen model changed')
    audit=install_physics_audit(batch,batches,n)
    bank={k:torch.tensor(v,device='cuda') for k,v in np.load(bank_path).items()}
    # Two copies of each controller share the same held-out states.
    if active_mode:
        pools=[torch.where(bank['mode']==i)[0][1::2] for i in range(3)]
        indices=torch.cat([pool[torch.linspace(0,len(pool)-1,16,device='cuda').long()] for pool in pools]).repeat(4)
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
    memory=active_bank.ActiveMemory(n,'cuda') if active_mode else CruiseMemory(n,'cuda')
    if active_mode:
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
    max_heading_error=torch.zeros(n,device='cuda');max_hold_drift=torch.zeros_like(max_heading_error)
    max_hold_heading=torch.zeros_like(max_heading_error);body_progress=torch.zeros_like(max_heading_error)
    walking=blob['parameters'].to('cuda');turn_parameters=blob['turn_parameters'].to('cuda')
    live=torch.ones(n,device='cuda',dtype=torch.bool);peak=torch.zeros(n,device='cuda');duration=torch.zeros_like(peak)
    bits=torch.zeros(n,device='cuda',dtype=torch.int);origin=batch.pos[:,batch.pelvis,:2].clone();terminal=origin.clone()
    slip_total=torch.zeros_like(peak);air_total=torch.zeros_like(peak);swings=torch.zeros((n,2),device='cuda',dtype=torch.long)
    air=torch.zeros((n,2),device='cuda');height=torch.zeros_like(air);touched=torch.zeros_like(air,dtype=torch.bool)
    max_correction=torch.zeros_like(peak);weight=float(batch.model.body_mass.sum())*9.81
    def observe():
        rotation=batch.rot[:,batch.base].double();heading=torch.atan2(rotation[:,1,0],rotation[:,0,0])
        command=commands('backward',batch.pos[:,batch.pelvis,:2].double()-batch.reference_xy.double(),heading,100.)
        if active_mode:command=active_bank.commands(mode,age,batch.pos[:,batch.pelvis,:2].double()-batch.reference_xy.double(),heading,meta['arguments']['turn_duration'])
        extra=torch.zeros((n,10),device='cuda');extra[:,2]=(1.-age/30.).float();extra[:,3:6]=command.float()
        phase=2*torch.pi*age/.6;extra[:,6]=phase.sin().float();extra[:,7]=phase.cos().float()
        extra[:,8]=rotation[:,1,0].float();extra[:,9]=(rotation[:,0,0]-1).float()
        return torch.cat((batch.observations(),extra),dim=1)
    obs=observe()
    for step in range(steps):
        if step%250==0:print(json.dumps({'step':step,'live':int(live.sum())}),flush=True)
        with torch.no_grad():
            prior_obs=memory.prior(obs,batch.pos[:,batch.pelvis],batch.rot[:,batch.base],age) if active_mode else obs[:,:63]
            baseline=turn.commanded_action(base,walking,prior,obs,prior_obs,turn_parameters,names,memory.reference,True,memory=memory)
            if active_mode:
                baseline=residual.apply(actor,baseline,obs,walking,memory,names)
                action=active_head.apply(active_actor,baseline,obs,walking,memory,names,residual)
            else:action=residual.apply(actor,baseline,obs,walking,memory,names)
            action[:width*2]=baseline[:width*2]
            correction=(action-baseline).abs().amax(1)*.08
            max_correction=torch.maximum(max_correction,torch.where(live,correction,0.))
            action=action.clamp(-limits,limits)
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
        if active_mode:
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
        if active_mode:memory.advance_command(obs,batch.pos[:,batch.pelvis],age,.02)
        else:memory.advance(obs[:,65].double(),age,.02)
    records=[]
    for group in range(4):
        selection=slice(group*width,(group+1)*width)
        records.append({'controller':'frozen' if group<2 else 'residual','copy':group%2,
            'survived_count':int(((bits==0)&(age>=ends-1e-8))[selection].sum()),'start_ages_s':start_age[selection].tolist(),
            'terminal_age_s':age[selection].tolist(),'required_duration_s':horizons[selection].tolist(),'duration_s':duration[selection].tolist(),'failure_bits':bits[selection].tolist(),
            'body_forward_progress_m':body_progress[selection].tolist(),
            'peak_support_bw':peak[selection].tolist(),'backward_progress_m':(-(terminal-origin)[:,1])[selection].tolist(),
            'mean_slip_mps':(slip_total/duration.clamp_min(.02))[selection].tolist(),
            'air_fraction':(air_total/duration.clamp_min(.02))[selection].tolist(),'qualified_swings':swings[selection].tolist(),
            'max_correction_rad':max_correction[selection].tolist(),
            **({'modes':mode[selection].tolist(),'max_teacher_heading_error_rad':max_heading_error[selection].tolist(),
                'max_hold_drift_m':max_hold_drift[selection].tolist(),'max_hold_heading_error_rad':max_hold_heading[selection].tolist()} if active_mode else {})})
    backend.write_json(folder/'screen.json',{'kind':'paired_training_state_diagnostic','canonical_milestone_pass':False,
        'checkpoint':str(checkpoint),'checkpoint_sha256':backend.digest(checkpoint),'seconds':None if args.complete_stops else args.seconds,'complete_stops':args.complete_stops,'max_screen_seconds':steps*.02,
        'source_sha256':backend.digest(__file__),'residual_source_sha256':meta['residual_source_sha256'],
        'audit_source_sha256':backend.digest(Path(install_physics_audit.__code__.co_filename)),
        'curriculum_bank_sha256':backend.digest(bank_path),'records':records,
        'residual_variant':meta.get('residual_variant'),'active_residual_source_sha256':meta.get('active_residual_source_sha256'),
        'memory_source_sha256':backend.digest(active_bank.__file__) if active_mode else backend.digest(Path(__file__).with_name('landau_cruise_residual.py')),
        'direction_source_sha256':backend.digest(active_direction.__file__) if active_mode else None,
        'comparison':'frozen cruise versus frozen cruise plus active head' if active_mode else 'frozen CEM versus cruise residual',
        'limitations':'Adjacent held-out states share one trajectory; repeats are simultaneousbatch copies, not independent nominal-start proof.'})
    print(json.dumps([{'controller':r['controller'],'copy':r['copy'],'survived':r['survived_count'],'worst_peak_bw':max(r['peak_support_bw'])} for r in records]),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name',required=True);parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--seconds',type=float,default=20.)
    parser.add_argument('--complete-stops',action='store_true')
    args=parser.parse_args()
    if not 6.<=args.seconds<=30.:raise ValueError('Screen horizon must be6..30s')
    run(args)


if __name__=='__main__':main()
