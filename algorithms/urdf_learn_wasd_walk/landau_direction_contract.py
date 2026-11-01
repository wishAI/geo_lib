"""M7 world gates reached by forward/yaw joystick commands from one start pose.

Directions describe world displacement, not body-relative sideways gait. The
adapter reads pose and emits commands only; it never changes simulator state.
"""
import math

DIRECTIONS = {'forward': (0., 1.), 'left': (-1., 0.),
              'right': (1., 0.), 'backward': (0., -1.)}
MAX_YAW_RATE = math.pi / 88.
MAX_DURATION_S = 240.
GATE_WIDTH_M = .75


class TurnTrainingProtocol:
    """Sustained right turn, then straight walking; M7 diagnostic curriculum."""
    YAW_RATE = MAX_YAW_RATE
    HOLDS = ()

    def __init__(self, seconds, side='right'):
        if seconds < 55.: raise ValueError('Sustained turn training requires at least 55 s')
        if side not in ('left', 'right'): raise ValueError('Unknown turn side')
        self.sign = 1 if side == 'left' else -1
        self.BLOCKS = ((side, 3., 47., self.sign), ('straight_after_turn', 47., seconds, 0))

    def command_profile(self, seconds):
        return .2, 0., self.sign * MAX_YAW_RATE if 3. <= seconds < 47. else 0.


class RecordedLeftTrainingProtocol:
    """Replay semantic steering only, then extend straight walking for validation.

    No pose, action, or simulator state is replayed. Subsequent gate evaluations
    recompute their own commands from the new simulated poses.
    """
    YAW_RATE = MAX_YAW_RATE
    HOLDS = ()

    def __init__(self, samples, seconds):
        self.samples = [tuple(float(v) for v in row) for row in samples]
        if (not self.samples or any(len(row) != 3 or not all(math.isfinite(v) for v in row)
                or abs(row[0]-.2)>1e-7 or row[1]!=0. or not 0.<=row[2]<=MAX_YAW_RATE+1e-7
                for row in self.samples)):
            raise ValueError('Expected finite full-forward/left semantic commands only')
        active = [i for i,row in enumerate(self.samples) if row[2]>0.]
        if not active or self.samples[-1][2]!=0.:
            raise ValueError('Recorded left turn must end in straight walking')
        first_zero = next(i for i in range(active[0]+1,len(self.samples)) if self.samples[i][2]==0.)
        straight_start = (active[-1]+1)*.02
        if seconds < max(len(self.samples)*.02,straight_start+20.):
            raise ValueError('Replay needs at least20s of final straight walking')
        self.BLOCKS=(('left',active[0]*.02,first_zero*.02,1),
                     ('straight_after_turn',straight_start,seconds,0))

    def command_profile(self, seconds):
        index=round(seconds/.02)
        return self.samples[index] if index<len(self.samples) else (.2,0.,0.)


class TurnHoldTrainingProtocol:
    """Exact M5 commands for calibrating left yaw without changing balance."""
    def __init__(self, seconds, turn_duration):
        if not 14.<=turn_duration<=90.: raise ValueError('Turn duration outside bounded range')
        self.turn_end=3.+turn_duration
        self.hold_start=self.turn_end+2.
        if seconds < self.hold_start+7.: raise ValueError('M5 requires a full settled hold')
        self.YAW_RATE=math.pi/(2.*turn_duration)
        self.BLOCKS=(('left',3.,self.turn_end,1),)
        self.HOLDS=((self.hold_start+2.,seconds),)

    def command_profile(self, seconds):
        fade=min(max((seconds-self.turn_end)/2.,0.),1.)
        return .2*(1.-fade*fade*(3.-2.*fade)),0.,self.YAW_RATE if 3.<=seconds<self.turn_end else 0.


def command(direction, displacement, heading, seconds, forward=.2):
    """Steer to a fixed 10 m world gate with bounded semantic commands."""
    axis = DIRECTIONS[direction]
    if direction == 'forward': return forward, 0., 0.
    dx, dy = displacement
    progress = dx * axis[0] + dy * axis[1]
    # Beyond the gate keep walking in its direction, without commanding a U-turn.
    ahead = max(10., progress + 2.)
    tx, ty = ahead * axis[0] - dx, ahead * axis[1] - dy
    bearing = math.atan2(-tx, ty)
    error = math.atan2(math.sin(bearing - heading), math.cos(bearing - heading))
    # A backward target needs the chosen left arc even when initial body sway
    # makes the shortest arc slightly rightward. Keep small overshoot at zero.
    if direction == 'backward' and error < -math.pi/2: error += 2.*math.pi
    yaw = max(-MAX_YAW_RATE, min(MAX_YAW_RATE, .3 * error))
    # Each gate uses one turn direction. Small pose oscillations must not swap
    # left/right balance settings; overshoot is corrected by straight progress.
    yaw = min(0., yaw) if direction == 'right' else max(0., yaw)
    return forward, 0., yaw if seconds >= 3. else 0.


def gate_metrics(direction, times, positions):
    """Interpolate the first crossing of the bounded gate, not path length."""
    axis = DIRECTIONS[direction]
    if len(times) != len(positions) or len(times) < 2:
        raise ValueError('Direction evaluation requires aligned trajectory samples')
    if not all(math.isfinite(float(v)) for v in times):
        raise ValueError('Nonfinite time')
    if any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError('Trajectory clock reset or gap')
    if any(len(p) < 2 or not all(math.isfinite(float(v)) for v in p[:2]) for p in positions):
        raise ValueError('Invalid position')
    origin = positions[0]
    along, cross = [], []
    for p in positions:
        x, y = p[0] - origin[0], p[1] - origin[1]
        along.append(x * axis[0] + y * axis[1])
        cross.append(x * axis[1] - y * axis[0])
    crossing = None
    for i in range(1, len(times)):
        if along[i-1] < 10. <= along[i]:
            fraction = (10. - along[i-1]) / (along[i] - along[i-1])
            lateral = cross[i-1] + fraction * (cross[i] - cross[i-1])
            if abs(lateral) <= GATE_WIDTH_M:
                crossing = float(times[i-1] + fraction * (times[i] - times[i-1]))
                break
    return {'direction': direction, 'gate_crossing_time_s': crossing,
            'gate_cross_track_m': float(lateral) if crossing is not None else None,
            'direction_progress_m': float(along[-1]),
            'direction_cross_track_m': float(cross[-1]),
            'average_gate_speed_mps': 10. / crossing if crossing else None}


def evaluate_gate(metrics, duration):
    """Retain forward walking quality requirements, substituting world crossing."""
    from algorithms.urdf_learn_wasd_walk.forward_walk_contract import evaluate_forward_gate
    projected = dict(metrics, semantic_forward_displacement_m=metrics['direction_progress_m'],
                     semantic_strafe_displacement_m=metrics.get('gate_cross_track_m')
                     if metrics.get('gate_crossing_time_s') is not None else metrics['direction_cross_track_m'])
    failures = evaluate_forward_gate(projected, required_distance_m=10.)
    if metrics.get('gate_crossing_time_s') is None:
        failures.append('no crossing of the 10 m directional gate within 0.75 m')
    if metrics.get('duration_s', 0.) + 1e-6 < duration:
        failures.append('direction run ended before declared duration')
    return failures
