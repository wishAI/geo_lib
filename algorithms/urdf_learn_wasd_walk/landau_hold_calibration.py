"""Frozen-dual hold/yaw grids; training screens are never milestone acceptance.

Both saved learned corrections run during deceleration and restart. Only the
existing left hold gain varies in paired-state mode. Nominal mode also varies
left yaw scale; model, gait, actors and all other turn gains stay frozen.
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
        if cfg['checkpoint_sha256'] not in allowed or not cfg.get('turn' if mode == 'm5' else 'teleop'):
            raise ValueError('Warm start checkpoint or protocol mismatch')
        if any(cfg[key] != value for key,value in dict(forward=.2, assistance=0., dt=.002,
                contact_timeconst=.004, gain_scale=1.).items()):
            raise ValueError('Warm start physics changed')
        for key in ('model_xml_sha256','urdf_sha256','mesh_tree_sha256','backend','mujoco_version'):
            if result['identity'][key] != meta[key]:raise ValueError('Warm start model changed')
        hashes = {}
        for filename,key in (('trajectory.npz','trajectory_sha256'),('policy_trace.npz','policy_trace_sha256'),
                             ('foot_trace.json','foot_trace_sha256')):
            hashes[filename] = digest(folder/filename)
            if hashes[filename] != result[key]:raise ValueError('Warm start trace changed')
        feet = json.loads((folder/'foot_trace.json').read_text())
        if any(r['time_s'] <= age and (r['support_body_weight_ratio'] > 3. or
               r['max_joint_speed_rad_s'] > 4.+1e-6) for r in feet):
            raise ValueError('Unsafe warm start prefix')
        if result['metrics']['reset_count'] != 0:raise ValueError('Warm start source reset')
        trace = np.load(folder/'policy_trace.npz')
        trajectory = np.load(folder/'trajectory.npz')
        if (len(trace['time']) <= index or abs(trace['time'][index]-age) > 1e-6 or
            not np.allclose(trace['time'][:index+1],trajectory['time'][:index+1],atol=1e-6,rtol=0)):
            raise ValueError('Warm start clocks differ')
        command_histories.append((trace['time'][:index+1],trace['observation'][:index+1,63:66]))
        memory = turn.CommandMemory()
        for t,obs in zip(trace['time'][:index],trace['observation'][:index]):
            if not np.isfinite(obs).all() or -float(obs[8]) < math.cos(math.pi/6):
                raise ValueError('Unhealthy warm start prefix')
            memory.observe(torch.from_numpy(obs)[None],torch.zeros((1,3)),torch.eye(3)[None],float(t))
        if memory.anchor_xy is not None or memory.restart_time is not None or memory.last_yaw_sign != 1:
            raise ValueError('Warm starts must precede the first stop after left turning')
        histories.append(memory)
        states.append(dict(qpos=trajectory['qpos'][index],qvel=trajectory['qvel'][index],
                           previous=trace['observation'][index,43:60]))
        sources.append(dict(directory=str(folder),age_s=age,checkpoint_sha256=cfg['checkpoint_sha256'],
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
    backend,batches,_ = configure_model('balanced_hands_v1')
    checkpoint = Path(args.checkpoint).resolve()
    meta = json.loads((checkpoint.parent/'training.json').read_text())
    if meta['checkpoints'][checkpoint.name] != backend.digest(checkpoint):raise ValueError('Checkpoint changed')
    if meta['action_scale'] != action_scales(meta['action_joints'],.4):raise ValueError('Calibration action scales changed')
    blob = torch.load(checkpoint,map_location='cpu',weights_only=False)
    check_contract(meta,blob,backend.digest)
    turn = load_source(checkpoint.parent/'turn_source.py',meta['turn_source_sha256'],'hold_saved_turn')
    if meta['arguments']['turn_duration'] != 44.:raise ValueError('Calibration requires the current44s turn profile')
    turn.configure_profile(meta['arguments']['turn_duration'])
    base,_ = load_gait_source(checkpoint,meta)
    residual = load_source(checkpoint.parent/'residual_source.py',meta['residual_source_sha256'],'hold_saved_cruise')
    active = load_source(checkpoint.parent/'active_residual_source.py',meta['active_residual_source_sha256'],'hold_saved_active')
    if args.nominal_grid:
        prepared = [(mode,[],turn.CommandMemory(),[],0.,end) for mode,end in [('m5',56.),('m6',60.)]]
    else:
        prepared = [(mode,*load_starts(folders,checkpoint,meta,turn,mode,age,backend.digest),age,end)
                    for mode,folders,age,end in [('m5',args.m5_directory,46.8,56.),('m6',args.m6_directory,18.,60.)]]
    folder = backend.OUTPUT/args.name
    folder.mkdir(exist_ok=False)
    (folder/'screen_source.py').write_bytes(Path(__file__).read_bytes())
    (folder/'audit_source.py').write_bytes(Path(install_physics_audit.__code__.co_filename).read_bytes())
    if backend.digest(teleop.__file__) != backend.digest(checkpoint.parent/'teleop_contract.py'):
        raise ValueError('Calibration teleop protocol differs from saved controller')
    (folder/'teleop_source.py').write_bytes(Path(teleop.__file__).read_bytes())
    torch.set_num_threads(4);torch.manual_seed(42)
    grid = (nominal_grid if args.nominal_grid else parameter_grid)(blob['turn_parameters']).to('cuda')
    prior = make_actor(63,'cuda');prior.load_state_dict(blob['standing_actor_state_dict']);prior.eval()
    cruise_actor = residual.make_actor('cuda');cruise_actor.load_state_dict(blob['residual_actor_state_dict']);cruise_actor.eval()
    active_actor = residual.make_actor('cuda');active_actor.load_state_dict(blob['active_residual_actor_state_dict']);active_actor.eval()
    walking = blob['parameters'].to('cuda')
    results = {}
    for mode,states,memory,sources,start,end in prepared:
        width = 4 if args.nominal_grid else 2*len(states)
        n = len(grid)*width
        batch = batches.WarpBatch(n,42,0.,'stand','load',.004,.2,.0225)
        if hashlib.sha256(batch.xml.encode()).hexdigest() != meta['model_xml_sha256']:raise ValueError('Calibration model changed')
        audit = install_physics_audit(batch,batches,n)
        if args.nominal_grid:
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
        names = batch.spec['action_joints']
        limits = torch.tensor(action_scales(names,.4),device='cuda')/.08
        weight = float(batch.model.body_mass.sum())*9.81
        live = torch.ones(n,dtype=torch.bool,device='cuda')
        bits = torch.zeros(n,dtype=torch.int,device='cuda')
        peak = torch.zeros(n,device='cuda')
        hold_error = torch.zeros(n,device='cuda');hold_drift = torch.zeros_like(hold_error)
        hold_speed = torch.zeros_like(hold_error);hold_yaw_speed = torch.zeros_like(hold_error)
        holds = ((51.,56.),) if mode == 'm5' else teleop.HOLDS
        hold_origins = {};hold_headings = {}
        durations = torch.zeros(n,device='cuda')
        last_xy = batch.pos[:,batch.pelvis,:2].clone()
        for step in range(round((end-start)/.02)):
            age = round((start+step*.02)/.02)*.02
            if step%500 == 0:print(json.dumps(dict(mode=mode,age_s=age,live=int(live.sum()))),flush=True)
            if mode == 'm5':forward,yaw,_ = turn.command_profile(age);command = (forward,0.,yaw)
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
                action = residual.apply(cruise_actor,action,obs,walking,memory,names)
                action = active.apply(active_actor,action,obs,walking,memory,names,residual).clamp(-limits,limits)
            heading = torch.atan2(rot[:,1,0],rot[:,0,0])
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
            tilt = batch.rot[:,batch.base,2,2].clamp(-1,1).acos()
            failure = (tilt>math.pi/6).int()+2*(batch.pos[:,batch.pelvis,2]<batch.reference_height-.08).int()
            failure += 4*(audit['speed']>4.+1e-6).int()+8*(audit['force']>3.*weight).int()
            failure += 16*(audit['nonfoot']!=0).int()+32*(audit['flight']>60).int()
            failure += 64*(~torch.isfinite(batch.q).all(1)|~torch.isfinite(batch.v).all(1)).int()
            peak = torch.maximum(peak,torch.where(live,audit['force']/weight,0.))
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
        results[mode] = dict(sources=sources,start_s=start,end_s=end,
            candidate_records=[dict(index=i,gain=float(grid[i,3]),left_yaw_scale=float(grid[i,17]),eligible=eligible[i*width:(i+1)*width].tolist(),
                failure_bits=bits[i*width:(i+1)*width].tolist(),air_fraction=air_fraction[i*width:(i+1)*width].tolist(),peak_support_bw=peak[i*width:(i+1)*width].tolist(),
                hold_error_rad=hold_error[i*width:(i+1)*width].tolist(),hold_drift_m=hold_drift[i*width:(i+1)*width].tolist(),
                hold_speed_mps=hold_speed[i*width:(i+1)*width].tolist(),hold_yaw_speed_rad_s=hold_yaw_speed[i*width:(i+1)*width].tolist(),duration_s=durations[i*width:(i+1)*width].tolist())
                for i in range(len(grid))])
        del batch,audit
    backend.write_json(folder/'screen.json',dict(kind='nominal_common_start_yaw_hold_calibration' if args.nominal_grid else 'paired_recorded_state_hold_calibration',canonical_milestone_pass=False,
        checkpoint=str(checkpoint),checkpoint_sha256=backend.digest(checkpoint),training_sha256=backend.digest(checkpoint.parent/'training.json'),
        source_sha256=backend.digest(__file__),turn_source_sha256=meta['turn_source_sha256'],
        residual_source_sha256=meta['residual_source_sha256'],active_residual_source_sha256=meta['active_residual_source_sha256'],
        audit_source_sha256=backend.digest(Path(install_physics_audit.__code__.co_filename)),
        teleop_source_sha256=backend.digest(folder/'teleop_source.py'),
        common_start_source_sha256=backend.digest(folder/'direction_training_source.py') if args.nominal_grid else None,
        candidate_parameters=grid.cpu().tolist(),changed_indices=[3,17] if args.nominal_grid else [3],results=results,
        screen_criteria='Physical survival, force, joint speed, contacts, flight and hold metrics only; M6 block response/qualified swings still require nominal validation.',
        limitations=('Common nominal/perturbed starts with simultaneous batched duplicate controls; no independent serial proof. ' if args.nominal_grid else 'Cleared solver warmstart; correlated states and batched copies, not exact continuation or nominal acceptance. M6 prefix before18s is not simulated. ')+'Hold metrics sampled20ms; physical force/speed/contact guards sampled2ms.'))
    print(json.dumps({mode:[sum(r['eligible']) for r in result['candidate_records']] for mode,result in results.items()}),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name',required=True);parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--nominal-grid',action='store_true')
    parser.add_argument('--m5-directory',action='append',default=[])
    parser.add_argument('--m6-directory',action='append',default=[])
    args = parser.parse_args()
    if args.nominal_grid:
        if args.m5_directory or args.m6_directory:raise ValueError('Nominal grid cannot use warm starts')
    elif len(args.m5_directory)!=2 or len(args.m6_directory)!=2:
        raise ValueError('Require paired passing and failed M5/M6 states')
    run(args)


if __name__ == '__main__':main()
