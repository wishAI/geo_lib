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
    residual=load_source(checkpoint.parent/'residual_source.py',meta['residual_source_sha256'],'screen_saved_residual')
    base,_=load_gait_source(checkpoint,meta)
    turn=load_source(checkpoint.parent/'turn_source.py',meta['turn_source_sha256'],'screen_saved_turn')
    turn.configure_profile(meta['arguments']['turn_duration'])
    bank_path=checkpoint.parent/'warm_states.npz'
    if backend.digest(bank_path)!=meta['curriculum']['bank_sha256']:raise ValueError('Curriculum bank changed')
    torch.set_num_threads(4);torch.manual_seed(42)
    n=128;batch=batches.WarpBatch(n,42,0.,'stand','load',.004,.2,.0225)
    if hashlib.sha256(batch.xml.encode()).hexdigest()!=meta['model_xml_sha256']:raise ValueError('Screen model changed')
    audit=install_physics_audit(batch,batches,n)
    bank={k:torch.tensor(v,device='cuda') for k,v in np.load(bank_path).items()}
    #32held-out states, two copies of the frozen and learned controller each.
    indices=(torch.linspace(0,len(bank['age'])//2-1,32,device='cuda').long()*2+1).repeat(4)
    batch.q[:]=bank['qpos'][indices].float();batch.v[:]=bank['qvel'][indices].float()
    batch.previous[:]=bank['previous'][indices].float();batch.ctrl[:]=batch.targets
    wp.to_torch(batch.wd.qacc_warmstart).zero_();wp.to_torch(batch.wd.qfrc_applied).zero_();batch.wrench.zero_()
    wp.to_torch(batch.wd.time)[:]=bank['age'][indices].float()
    age=bank['age'][indices].clone();start_age=age.clone()
    memory=CruiseMemory(n,'cuda')
    memory._reference[:]=bank['reference'][indices];memory._blend[:]=bank['blend'][indices]
    memory._feedback[:]=bank['feedback'][indices];memory.left_cruise_start[:]=bank['cruise_start'][indices]
    memory.previous_yaw[:]=bank['previous_yaw'][indices].double()
    for value in audit.values():value.zero_()
    torch.cuda.synchronize();wp.capture_launch(batch.forward_graph);wp.synchronize()
    names=batch.spec['action_joints'];limits=torch.tensor(action_scales(names,.4),device='cuda')/.08
    prior=make_actor(63,'cuda');prior.load_state_dict(blob['standing_actor_state_dict']);prior.eval()
    actor=residual.make_actor('cuda');actor.load_state_dict(blob['residual_actor_state_dict']);actor.eval()
    walking=blob['parameters'].to('cuda');turn_parameters=blob['turn_parameters'].to('cuda')
    live=torch.ones(n,device='cuda',dtype=torch.bool);peak=torch.zeros(n,device='cuda');duration=torch.zeros_like(peak)
    bits=torch.zeros(n,device='cuda',dtype=torch.int);origin=batch.pos[:,batch.pelvis,:2].clone();terminal=origin.clone()
    slip_total=torch.zeros_like(peak);air_total=torch.zeros_like(peak);swings=torch.zeros((n,2),device='cuda',dtype=torch.long)
    air=torch.zeros((n,2),device='cuda');height=torch.zeros_like(air);touched=torch.zeros_like(air,dtype=torch.bool)
    max_correction=torch.zeros_like(peak);weight=float(batch.model.body_mass.sum())*9.81
    def observe():
        rotation=batch.rot[:,batch.base].double();heading=torch.atan2(rotation[:,1,0],rotation[:,0,0])
        command=commands('backward',batch.pos[:,batch.pelvis,:2].double()-batch.reference_xy.double(),heading,100.)
        extra=torch.zeros((n,10),device='cuda');extra[:,2]=(1.-age/30.).float();extra[:,3:6]=command.float()
        phase=2*torch.pi*age/.6;extra[:,6]=phase.sin().float();extra[:,7]=phase.cos().float()
        extra[:,8]=rotation[:,1,0].float();extra[:,9]=(rotation[:,0,0]-1).float()
        return torch.cat((batch.observations(),extra),dim=1)
    obs=observe()
    for step in range(round(args.seconds/.02)):
        if step%250==0:print(json.dumps({'step':step,'live':int(live.sum())}),flush=True)
        with torch.no_grad():
            baseline=turn.commanded_action(base,walking,prior,obs,obs[:,:63],turn_parameters,names,memory.reference,True,memory=memory)
            action=residual.apply(actor,baseline,obs,walking,memory,names)
            action[:64]=baseline[:64]
            correction=(action-baseline).abs().amax(1)*.08
            max_correction=torch.maximum(max_correction,torch.where(live,correction,0.))
            action=action.clamp(-limits,limits)
        batch.ctrl[:]=batch.targets;batch.ctrl[:,batch.aids]+=.08*torch.where(live[:,None],action,0.);batch.wrench.zero_()
        torch.cuda.synchronize();wp.capture_launch(batch.graph);wp.synchronize()
        occupancy=batch.maximum_occupancy.numpy()
        if max(occupancy[:2])>=batch.wd.naconmax or occupancy[2]>=batch.wd.njmax:raise RuntimeError('Screen capacity exceeded')
        batch.previous[:]=action;age+=.02
        tilt=batch.rot[:,batch.base,2,2].clamp(-1,1).acos()
        failure=(tilt>torch.pi/6).int()+2*(batch.pos[:,batch.pelvis,2]<batch.reference_height-.08).int()
        failure+=4*(audit['speed']>4.+1e-6).int()+8*(audit['force']>3.*weight).int()
        failure+=16*(audit['nonfoot']!=0).int()+32*(audit['flight']>60).int()
        failure+=64*(~torch.isfinite(batch.q).all(1)|~torch.isfinite(batch.v).all(1)).int()
        peak=torch.maximum(peak,torch.where(live,audit['force']/weight,0.));duration+=live.float()*.02
        terminal=torch.where(live[:,None],batch.pos[:,batch.pelvis,:2],terminal)
        contacts,slip,clearance,_=batch.feet(with_forces=True)
        swings+=(contacts&touched&(air>=.06)&(height>=.015)&live[:,None]).long()
        slip_total+=torch.where(live,slip,0.)*.02;air_total+=torch.where(live,(~contacts.any(1)).float(),0.)*.02
        air=torch.where(contacts,0.,air+.02);height=torch.where(contacts,0.,torch.maximum(height,clearance));touched|=contacts
        bits=torch.where(live&(failure!=0),failure,bits);live&=failure==0
        obs=observe();memory.advance(obs[:,65].double(),age,.02)
    records=[]
    for group in range(4):
        selection=slice(group*32,(group+1)*32)
        records.append({'controller':'frozen' if group<2 else 'residual','copy':group%2,
            'survived_count':int(live[selection].sum()),'start_ages_s':start_age[selection].tolist(),
            'duration_s':duration[selection].tolist(),'failure_bits':bits[selection].tolist(),
            'peak_support_bw':peak[selection].tolist(),'backward_progress_m':(-(terminal-origin)[:,1])[selection].tolist(),
            'mean_slip_mps':(slip_total/duration.clamp_min(.02))[selection].tolist(),
            'air_fraction':(air_total/duration.clamp_min(.02))[selection].tolist(),'qualified_swings':swings[selection].tolist(),
            'max_correction_rad':max_correction[selection].tolist()})
    backend.write_json(folder/'screen.json',{'kind':'paired_training_state_diagnostic','canonical_milestone_pass':False,
        'checkpoint':str(checkpoint),'checkpoint_sha256':backend.digest(checkpoint),'seconds':args.seconds,
        'source_sha256':backend.digest(__file__),'residual_source_sha256':meta['residual_source_sha256'],
        'audit_source_sha256':backend.digest(Path(install_physics_audit.__code__.co_filename)),
        'curriculum_bank_sha256':backend.digest(bank_path),'records':records,
        'limitations':'Adjacent held-out states share one trajectory; repeats are simultaneousbatch copies, not independent nominal-start proof.'})
    print(json.dumps([{'controller':r['controller'],'copy':r['copy'],'survived':r['survived_count'],'worst_peak_bw':max(r['peak_support_bw'])} for r in records]),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name',required=True);parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--seconds',type=float,default=20.)
    args=parser.parse_args()
    if not 6.<=args.seconds<=30.:raise ValueError('Screen horizon must be6..30s')
    run(args)


if __name__=='__main__':main()
