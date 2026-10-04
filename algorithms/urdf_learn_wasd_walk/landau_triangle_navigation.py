"""M8 joystick-only triangle pursuit and prospective geometric acceptance.

Lookahead along a polyline follows the geometric pursuit principle described in
https://www.ri.cmu.edu/pub_files/pub3/coulter_r_craig_1992_1/coulter_r_craig_1992_1.pdf
The yaw servo is the existing V6 body-heading controller, not a wheel model.
No controller reference, action, or simulator state is assigned here.
"""
import math
import struct

from algorithms.urdf_learn_wasd_walk.landau_signed_navigation import HystereticNavigator

VERSION = 'triangle_pursuit_v1'
SIDE_M = 18.
VERTICES = ((0., 0.), (0., SIDE_M), (-SIDE_M*math.sqrt(3.)/2., SIDE_M/2.), (0., 0.))
LOOKAHEAD_M = 4.
CORNER_RADIUS_M = 3.5
CORRIDOR_M = 2.
MIDPOINT_RADIUS_M = .75
CLOSURE_RADIUS_M = .75
MAX_DURATION_S = 600.
MILESTONE = 'triangle_path_follow_no_reset'


def contract():
    return dict(version=VERSION, vertices_m=[list(v) for v in VERTICES],
                lookahead_m=LOOKAHEAD_M, corner_radius_m=CORNER_RADIUS_M,
                corridor_m=CORRIDOR_M, midpoint_radius_m=MIDPOINT_RADIUS_M,
                closure_radius_m=CLOSURE_RADIUS_M, horizon_s=MAX_DURATION_S,
                forward_command=.2, orientation='counterclockwise')


def projection(point, a, b):
    dx, dy = b[0]-a[0], b[1]-a[1]
    fraction = max(0., min(1., ((point[0]-a[0])*dx+(point[1]-a[1])*dy)/(dx*dx+dy*dy)))
    nearest = (a[0]+fraction*dx, a[1]+fraction*dy)
    return fraction, math.dist(point[:2], nearest)


def point_at(progress):
    edge = min(2, int(max(0., progress)/SIDE_M))
    fraction = min(1., max(0., progress/SIDE_M-edge))
    a, b = VERTICES[edge:edge+2]
    return (a[0]+fraction*(b[0]-a[0]), a[1]+fraction*(b[1]-a[1]))


class Navigator(HystereticNavigator):
    def __init__(self):
        super().__init__('left')
        self.progress = 0.

    def command(self, displacement, heading, seconds, forward=.2):
        if abs(forward-.2)>1e-9:
            raise ValueError('Triangle v1 requires validated full-forward command')
        if self.reference_time is not None:
            elapsed = seconds-self.reference_time
            if elapsed <= 0.:raise ValueError('Navigation clock must increase')
            self.reference += self.previous_yaw*elapsed
        # Only the current and following edge can advance pursuit. In particular,
        # the closing edge at the start cannot skip the two other sides.
        edge = min(2, int(self.progress/SIDE_M))
        candidates = []
        for i in range(edge, min(3, edge+2)):
            fraction, distance = projection(displacement, *VERTICES[i:i+2])
            candidates.append((distance, (i+fraction)*SIDE_M))
        _, progress = min(candidates)
        self.progress = max(self.progress, progress)
        target = point_at(self.progress+LOOKAHEAD_M)
        bearing = math.atan2(-(target[0]-displacement[0]), target[1]-displacement[1])
        error = math.atan2(math.sin(bearing-heading), math.cos(bearing-heading))
        yaw = self.steering_yaw(self.correction_error(error, seconds)) if seconds >= 3. else 0.
        gap = math.atan2(math.sin(self.reference-heading), math.cos(self.reference-heading))
        if self.blocked_sign and gap*self.blocked_sign <= self.TRACKING_RELEASE_RAD:self.blocked_sign = 0
        if abs(gap)>self.TRACKING_LIMIT_RAD and gap*yaw>0.:self.blocked_sign = 1 if yaw>0. else -1
        if yaw*self.blocked_sign>0.:yaw = 0.
        self.previous_yaw = struct.unpack('f', struct.pack('f', yaw))[0]
        self.reference_time = seconds
        return forward, 0., self.previous_yaw

    def record(self):
        return {**super().record(), 'progress_m':self.progress, 'version':VERSION}


class Tracker:
    """Ordered midpoint/corner visits prevent shortcuts and early closure."""
    def __init__(self):
        self.targets = []
        for edge in range(3):
            self.targets.append((point_at((edge+.5)*SIDE_M), MIDPOINT_RADIUS_M, f'edge_{edge+1}_midpoint'))
            self.targets.append((VERTICES[edge+1], CORNER_RADIUS_M if edge<2 else CLOSURE_RADIUS_M,
                                 f'corner_{edge+1}' if edge<2 else 'closure'))
        self.visits = []
        self.max_corridor = 0.
        self.length = 0.
        self.previous = None
        self.previous_time = None
        self.final_distance = 0.

    @property
    def complete(self):return len(self.visits)==len(self.targets)

    def update(self, point, seconds):
        point = tuple(float(v) for v in point[:2])
        if not all(math.isfinite(v) for v in (*point, seconds)) or (self.previous_time is not None and seconds<=self.previous_time):
            raise ValueError('Invalid triangle trajectory')
        self.max_corridor = max(self.max_corridor, min(projection(point,*VERTICES[i:i+2])[1] for i in range(3)))
        if self.previous is not None:self.length += math.dist(point, self.previous)
        self.previous, self.previous_time = point, seconds
        self.final_distance = math.hypot(*point)
        if not self.complete:
            target, radius, label = self.targets[len(self.visits)]
            distance = math.dist(point, target)
            if distance<=radius:self.visits.append(dict(target=label,time_s=float(seconds),distance_m=distance))

    def metrics(self):
        return dict(triangle_complete=self.complete, triangle_visits=self.visits,
                    triangle_max_corridor_m=self.max_corridor, triangle_path_length_m=self.length,
                    triangle_closure_distance_m=self.final_distance)


def evaluate_gate(metrics, duration):
    from algorithms.urdf_learn_wasd_walk.forward_walk_contract import evaluate_forward_gate
    # Substitute explicit path geometry for the forward-only displacement test.
    quality = dict(metrics, semantic_forward_displacement_m=0., semantic_strafe_displacement_m=0.)
    failures = evaluate_forward_gate(quality, required_distance_m=0.)
    if not metrics.get('triangle_complete'):failures.append('ordered triangle circuit incomplete')
    if metrics.get('triangle_max_corridor_m', math.inf)>CORRIDOR_M:failures.append('triangle corridor exceeded 2 m')
    if metrics.get('triangle_closure_distance_m', math.inf)>CLOSURE_RADIUS_M:failures.append('triangle closure exceeded 0.75 m')
    if not 0.<duration<=MAX_DURATION_S or metrics.get('duration_s',0.)+1e-6<duration:failures.append('triangle duration invalid')
    return failures
