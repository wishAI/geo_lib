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

    def __init__(self, seconds):
        if seconds < 55.: raise ValueError('Sustained turn training requires at least 55 s')
        self.BLOCKS = (('right', 3., 47., -1), ('straight_after_turn', 47., seconds, 0))

    @staticmethod
    def command_profile(seconds):
        return .2, 0., -MAX_YAW_RATE if 3. <= seconds < 47. else 0.


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
    # Resolve the exactly-behind ambiguity consistently as a left turn.
    if direction == 'backward' and abs(abs(error) - math.pi) < 1e-8: error = math.pi
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
