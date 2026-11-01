"""Repeatable M6 joystick replay and per-block response checks.

The schedule supplies semantic commands only. It never supplies poses/actions.
"""
import math

YAW_RATE = math.pi / 88.
BLOCKS = (
    ('forward', 0., 6., 0), ('left', 6., 18., 1),
    ('restart_forward', 27., 32., 0), ('right', 32., 44., -1),
    ('straight_after_turn', 44., 48., 0),
)
HOLDS = ((20., 25.), (50., 60.))


def smooth(value):
    value = min(max(value, 0.), 1.)
    return value * value * (3. - 2. * value)


def command_profile(seconds):
    forward = .2
    if 18. <= seconds < 20.: forward *= 1. - smooth((seconds - 18.) / 2.)
    elif 20. <= seconds < 25.: forward = 0.
    elif 25. <= seconds < 27.: forward *= smooth((seconds - 25.) / 2.)
    elif 48. <= seconds < 50.: forward *= 1. - smooth((seconds - 48.) / 2.)
    elif seconds >= 50.: forward = 0.
    yaw = YAW_RATE if 6. <= seconds < 18. else -YAW_RATE if 32. <= seconds < 44. else 0.
    return forward, 0., yaw


def response_metrics(times, positions, headings, feet):
    """Measure response from the recorded state and full-rate contact trace."""
    import numpy as np
    times = np.asarray(times); positions = np.asarray(positions); headings = np.unwrap(headings)
    counts = {side: [] for side in ('left', 'right')}
    states = {side: dict(touched=False, air=0., peak=0.) for side in counts}
    for row in feet:
        for side, state in states.items():
            if row[side + '_contact']:
                if state['touched'] and state['air'] >= .06 and state['peak'] >= .015:
                    counts[side].append(row['time_s'])
                state.update(touched=True, air=0., peak=0.)
            elif state['touched']:
                state['air'] += .002
                state['peak'] = max(state['peak'], row[side + '_clearance_m'])
    blocks = []
    for label, start, end, sign in BLOCKS:
        mask = (times >= start - 1e-6) & (times <= end + 1e-6)
        t, xy, yaw = times[mask], positions[mask, :2], headings[mask]
        valid = len(t) > 1 and t[0] <= start + .021 and t[-1] >= end - .021
        blocks.append(dict(label=label, start_s=start, end_s=end, yaw_sign=sign, complete=bool(valid),
            forward_progress_m=float(np.sum(np.diff(xy, axis=0) * np.c_[-np.sin(yaw[:-1]), np.cos(yaw[:-1])])) if len(t)>1 else 0.,
            heading_change_rad=float(yaw[-1]-yaw[0]) if len(t)>1 else 0.,
            max_heading_excursion_rad=float(np.abs(yaw-yaw[0]).max()) if len(t) else 0.,
            completed_swings={side: sum(start <= t <= end for t in ts) for side, ts in counts.items()}))
    holds = []
    for start, end in HOLDS:
        mask = (times >= start - 1e-6) & (times <= end + 1e-6)
        t, xy, yaw = times[mask], positions[mask, :2], headings[mask]
        valid = len(t)>1 and t[0]<=start+.021 and t[-1]>=end-.021
        settled = t >= start + 2.
        speed = np.linalg.norm(np.diff(xy,axis=0),axis=1)/np.diff(t) if len(t)>1 else np.array([])
        holds.append(dict(start_s=start,end_s=end,complete=bool(valid),
            drift_m=float(np.linalg.norm(xy-xy[0],axis=1).max()) if len(t) else None,
            settled_speed_mps=float(speed[settled[1:]].max()) if settled.sum()>1 else None,
            heading_drift_rad=float(np.abs(yaw-yaw[0]).max()) if len(t) else None))
    return {'blocks':blocks, 'holds':holds}


def evaluate_gate(metrics, responses):
    failures = []
    if not isinstance(responses,dict):return ['missing response records']
    expected_blocks=[(label,start,end,sign) for label,start,end,sign in BLOCKS]
    actual_blocks=[(r.get('label'),r.get('start_s'),r.get('end_s'),r.get('yaw_sign')) for r in responses.get('blocks',[])]
    actual_holds=[(r.get('start_s'),r.get('end_s')) for r in responses.get('holds',[])]
    if actual_blocks!=expected_blocks or actual_holds!=list(HOLDS):return ['wrong or missing command blocks/holds']
    for key in ('reset_count','done_count','fall_count'):
        if metrics.get(key) != 0: failures.append(f'{key} is not zero')
    for key, bound in {'max_reference_tilt_rad':math.pi/6,'root_height_drop_m':.08,
                       'max_abs_action':1.+1e-6,'simultaneous_air_fraction':.05}.items():
        value = metrics.get(key, math.inf)
        if not math.isfinite(value) or value > bound: failures.append(f'{key} exceeded {bound:g}')
    if not math.isfinite(metrics.get('duration_s',math.nan)) or abs(metrics.get('duration_s',0)-60.) > 1e-6: failures.append('teleop did not complete60s')
    if metrics.get('policy_inference_steps') != 3000: failures.append('teleop requires3000policy actions')
    for block in responses['blocks']:
        if not block['complete']: failures.append(block['label']+' incomplete')
        if not math.isfinite(block['forward_progress_m']) or block['forward_progress_m'] < .04*(block['end_s']-block['start_s']):
            failures.append(block['label']+' insufficient forward response')
        if min(block['completed_swings'].values()) < 2: failures.append(block['label']+' no repeated walking swings')
        if not block['yaw_sign']:
            excursion=block.get('max_heading_excursion_rad',math.inf)
            if not math.isfinite(excursion) or excursion>math.pi/6:failures.append(block['label']+' uncommanded turn')
        if block['yaw_sign']:
            delta=block['yaw_sign']*block['heading_change_rad']
            target=YAW_RATE*(block['end_s']-block['start_s'])
            if not .5*target <= delta <= 1.5*target: failures.append(block['label']+' wrong yaw response')
    for hold in responses['holds']:
        if not hold['complete']: failures.append(f"hold{hold['start_s']:g}s incomplete")
        for key,bound in [('drift_m',.03),('settled_speed_mps',.05),('heading_drift_rad',math.radians(5))]:
            value=hold.get(key)
            if value is None or not math.isfinite(value) or value>bound:failures.append(f"hold{hold['start_s']:g}s {key} failed")
    return failures
