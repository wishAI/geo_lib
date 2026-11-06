"""Sustained physical forward speed, never command magnitude or peak speed."""
import math

TARGETS=(.1,.4,.8,1.2)
WARMUP_S=10.
MEASUREMENT_S=30.
BLOCK_S=5.
DURATION_S=WARMUP_S+MEASUREMENT_S
PROTOCOL='sustained_forward_speed_v1'


def gate_id(target):
    if target not in TARGETS:raise ValueError('Unsupported walking speed target')
    return 'walk_speed_'+f'{target:.1f}'.replace('.','p')+'_mps_30s'


def pelvis_positions(model,qpos):
    """Reconstruct the skeleton root, including its rotated free-root offset."""
    import mujoco
    import numpy as np
    data=mujoco.MjData(model);pelvis=model.body('root_x').id;positions=[]
    for frame in qpos:
        data.qpos[:]=frame;mujoco.mj_kinematics(model,data)
        positions.append(data.xpos[pelvis].copy())
    return np.asarray(positions)


def metrics(times,positions,feet,forward_sign=-1):
    import numpy as np
    t=np.asarray(times);q=np.asarray(positions)
    if (t.ndim!=1 or q.ndim!=2 or q.shape[1]!=3 or len(q)!=len(t) or len(t)<2 or
        not np.isfinite(t).all() or not np.isfinite(q).all() or
        abs(t[0])>1e-8 or abs(t[-1]-DURATION_S)>1e-6 or
        not np.allclose(np.diff(t),.02,atol=1e-6,rtol=0) or forward_sign not in (-1,1)):
        raise ValueError('Speed measurement requires the complete fixed 40 s trajectory')
    events={s:[] for s in ('left','right')};states={s:dict(touched=False,air=0.,peak=0.,supported=True) for s in events}
    ft=np.array([row['time_s'] for row in feet])
    if len(ft)!=round(DURATION_S/.002) or not np.allclose(ft,np.arange(1,len(ft)+1)*.002,atol=1e-6,rtol=0):
        raise ValueError('Speed measurement requires the complete physics foot trace')
    for row in feet:
        for side,state in states.items():
            opposite='right' if side=='left' else 'left'
            if row[side+'_contact']:
                if state['touched'] and state['supported'] and row[opposite+'_contact'] and state['air']>=.06 and state['peak']>=.015:
                    events[side].append(row['time_s'])
                state.update(touched=True,air=0.,peak=0.,supported=True)
            elif state['touched']:
                state['air']+=.002;state['peak']=max(state['peak'],row[side+'_clearance_m'])
                state['supported']=state['supported'] and bool(row[opposite+'_contact'])
    blocks=[]
    for start in np.arange(WARMUP_S,DURATION_S,BLOCK_S):
        end=start+BLOCK_S;a=round(start/.02);b=round(end/.02)
        samples=[row for row in feet if start<row['time_s']<=end+1e-8]
        blocks.append({'start_s':float(start),'end_s':float(end),
            'forward_speed_mps':float(forward_sign*(q[b,1]-q[a,1])/BLOCK_S),
            'mean_contact_slip_mps':float(np.mean([row['mean_slip_mps'] for row in samples])),
            'completed_swings':{s:int(sum(start<v<=end+1e-8 for v in values)) for s,values in events.items()}})
    return {'protocol':PROTOCOL,'warmup_s':WARMUP_S,'measurement_s':MEASUREMENT_S,'blocks':blocks,
        'mean_forward_speed_mps':sum(b['forward_speed_mps'] for b in blocks)/len(blocks),
        'sustained_forward_speed_mps':min(b['forward_speed_mps'] for b in blocks)}


def failures(response,target):
    gate_id(target);issues=[]
    if response.get('protocol')!=PROTOCOL or response.get('warmup_s')!=WARMUP_S or response.get('measurement_s')!=MEASUREMENT_S:
        return ['Speed protocol changed']
    blocks=response.get('blocks',[])
    if len(blocks)!=6:return ['Six consecutive 5 s speed blocks required']
    for i,block in enumerate(blocks):
        if block['start_s']!=WARMUP_S+i*BLOCK_S or block['end_s']!=WARMUP_S+(i+1)*BLOCK_S:issues.append('Speed block boundaries changed')
        speed=block['forward_speed_mps'];slip=block['mean_contact_slip_mps']
        if not math.isfinite(speed) or speed<target:issues.append(f'Block {i+1} forward speed below {target:g} m/s')
        if not math.isfinite(slip) or slip>min(.1,.2*target):issues.append(f'Block {i+1} contact slip exceeds walking limit')
        if any(block['completed_swings'].get(s,0)<1 for s in ('left','right')):issues.append(f'Block {i+1} lacks alternating steps')
    return issues
