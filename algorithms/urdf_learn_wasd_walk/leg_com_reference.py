"""Bounded flat-ground COM reference for mechanics diagnostics.

The LIPM relation is p = c - h/g * c_ddot (Kajita et al., ICRA 2003,
https://people.csail.mit.edu/katiebyl/kb/DW2008/papers_of_tangential_interest/kajita03.pdf).
This solves the finite trajectory as a boundary-value problem. It does not
forward-integrate the unstable pendulum, apply forces, or certify locomotion.
"""
import numpy as np
from scipy.linalg import solve_banded


def filter_heading(position, velocity, target, *, dt=.02, frequency=2.):
    """Exact critically damped update for a held heading target.

    Measurement-dependent IK corrections must not be differentiated directly
    into inertial feedforward. This gives that reference finite acceleration.
    """
    error = position - target
    slope = velocity + frequency * error
    decay = np.exp(-frequency * dt)
    return target + (error + slope * dt) * decay, (velocity - frequency * slope * dt) * decay


def normal_force_reference(position, velocity, measured, desired, *, body_weight, dt=.02):
    """Small landing compliance: a load deficit lowers the foot reference.

    Force is averaged over a control interval by the caller. The critically
    damped reference limits derivative amplification of solver contact chatter.
    Offsets are in metres; this only changes joint-motor reference geometry.
    """
    desired = np.asarray(desired)
    target = np.clip(.003 * (np.asarray(measured) - desired) / body_weight, -.001, .001)
    # An early touch by the unloaded swing foot must not command it upward.
    target = np.where(desired > .01 * body_weight, target, 0.)
    return filter_heading(position, velocity, target, dt=dt, frequency=8.)


def contact_wrench_map(root_position, root_rotation, points, weights, jacp, jacr):
    """Map the six floating-root inverse forces into supported joint forces.

    MuJoCo free-joint translation is world-frame and rotation is local-frame.
    Each foot receives its support fraction of total force and of the remaining
    world moment. Physical COP/friction feasibility must be checked separately.
    The returned map is a calculation only; it never actuates the free root.
    """
    weights = np.asarray(weights)
    if np.any(weights < 0) or not np.isclose(weights.sum(), 1):
        raise ValueError('Support weights must form a convex combination')
    arm = weights @ points - root_position
    cross = np.array([[0., -arm[2], arm[1]],
                      [arm[2], 0., -arm[0]], [-arm[1], arm[0], 0.]])
    force_map = np.c_[np.eye(3), np.zeros((3, 3))]
    moment_map = np.c_[-cross, root_rotation]
    jp = np.einsum('i,ijk->jk', weights, jacp)
    jr = np.einsum('i,ijk->jk', weights, jacr)
    mapping = jp.T @ force_map + jr.T @ moment_map
    if not np.allclose(mapping[:6], np.eye(6), atol=1e-9):
        raise ValueError('Contact wrench does not close the floating-root balance')
    return mapping, force_map, moment_map


def bounded_com(zmp, *, height, dt, initial, terminal):
    """Solve the centered discrete LIPM with fixed endpoint COM positions."""
    zmp = np.asarray(zmp, dtype=float)
    if zmp.ndim != 2 or zmp.shape[1] != 2 or len(zmp) < 3:
        raise ValueError('Expected at least three horizontal ZMP samples')
    if height <= 0 or dt <= 0 or not np.isfinite(zmp).all():
        raise ValueError('Invalid LIPM input')
    ratio = height / (9.81 * dt * dt)
    count = len(zmp) - 2
    band = np.zeros((3, count))
    band[0, 1:] = -ratio
    band[1] = 1 + 2 * ratio
    band[2, :-1] = -ratio
    rhs = zmp[1:-1].copy()
    rhs[0] += ratio * np.asarray(initial)
    rhs[-1] += ratio * np.asarray(terminal)
    com = np.empty_like(zmp)
    com[0], com[-1] = initial, terminal
    com[1:-1] = solve_banded((1, 1), band, rhs)
    velocity = np.gradient(com, dt, axis=0, edge_order=2)
    acceleration = 9.81 / height * (com - zmp)
    return com, velocity, acceleration


def ease(value):
    value = np.clip(value, 0., 1.)
    return value**3 * (10 + value * (-15 + 6 * value))


def footsteps(time, *, initial_feet, step, cycle, lift, lead=4., double_support=.2):
    """Left support first, with continuous double-support weights.

    Time zero begins a double-support entry hold. The first right swing retains
    its actual initial position; later steps use the regular alternating span.
    """
    feet = np.array(initial_feet, dtype=float, copy=True)
    if not .2 <= double_support <= .5:
        raise ValueError('Double support must occupy 20–50% of each cycle')
    if time <= lead:
        return feet, np.array([.5, .5])
    phase = (time - lead) / cycle
    index = int(np.floor(phase))
    fraction = phase - index
    stance, swing = index % 2, 1 - index % 2
    forward = initial_feet[0, 1] - index * step
    feet[stance, 1] = forward
    start = initial_feet[1, 1] if index == 0 else forward + step
    half_support = double_support / 2
    swing_phase = np.clip((fraction - half_support) / (1 - double_support), 0., 1.)
    feet[swing, 1] = start + (forward - step - start) * ease(swing_phase)
    # Zero velocity AND acceleration at takeoff/landing. A sin² height has
    # nonzero endpoint acceleration and introduces an inverse-dynamics jump.
    feet[swing, 2] += lift * 64 * swing_phase**3 * (1 - swing_phase)**3
    weights = np.zeros(2)
    weights[stance] = 1.
    if fraction < half_support:
        weights[stance] = ease((fraction + half_support) / double_support)
    elif fraction > 1 - half_support:
        weights[stance] = 1 - ease((fraction - (1 - half_support)) / double_support)
    weights[swing] = 1 - weights[stance]
    return feet, weights


class ComReference:
    """Precompute a bounded COM plan, including a four-second entry hold."""

    def __init__(self, *, initial_feet, initial_com, cop_offset, step, cycle,
                 lift=.02, horizon=84., dt=.002, initial_velocity=None,
                 initial_acceleration=None, double_support=.2):
        if not .5 <= cycle <= 9 or not .01 <= step <= .04:
            raise ValueError('Outside researched diagnostic geometry/timing')
        self.dt = dt
        self.initial_feet = np.array(initial_feet, copy=True)
        self.step, self.cycle, self.lift = step, cycle, lift
        self.double_support = double_support
        self.times = np.arange(round(horizon / dt) + 1) * dt
        self.cop_offset = np.asarray(cop_offset)[:2]
        self.zmp = np.array([
            weight @ feet[:, :2] + self.cop_offset
            for feet, weight in (self.feet_at(t) for t in self.times)
        ])
        # Keep the initial position exactly continuous with the preceding hold.
        self.com, self.velocity, self.acceleration = bounded_com(
            self.zmp, height=initial_com[2], dt=dt,
            initial=initial_com[:2], terminal=self.zmp[-1])
        self.height = initial_com[2]
        # A position-only pendulum boundary can have a nonzero entry velocity.
        # Bridge the entire double-support entry with matching c, v and a.
        # Its implied ZMP, rather than the original constant ZMP, sets support.
        end = round(4. / dt)
        velocity = np.zeros(2) if initial_velocity is None else np.asarray(initial_velocity)[:2]
        acceleration = np.zeros(2) if initial_acceleration is None else np.asarray(initial_acceleration)[:2]
        endpoint = (self.com[end].copy(), self.velocity[end].copy(), self.acceleration[end].copy())
        def bridge(start, finish, first, last):
            duration = finish - start
            coefficients = np.zeros((6, 2))
            coefficients[:3] = [first[0], duration * first[1], duration**2 * first[2] / 2]
            residual = np.array([
                last[0] - coefficients[:3].sum(axis=0),
                duration * last[1] - coefficients[1] - 2 * coefficients[2],
                duration**2 * last[2] - 2 * coefficients[2],
            ])
            coefficients[3:] = np.linalg.solve([[1., 1., 1.], [3., 4., 5.], [6., 12., 20.]], residual)
            selected = (self.times >= start) & (self.times <= finish)
            u = (self.times[selected, None] - start) / duration
            self.com[selected] = sum(u**j * coefficients[j] for j in range(6))
            self.velocity[selected] = sum(j * u**(j - 1) * coefficients[j] / duration for j in range(1, 6))
            self.acceleration[selected] = sum(j * (j - 1) * u**(j - 2) * coefficients[j] / duration**2 for j in range(2, 6))
        hold = (np.asarray(initial_com)[:2], np.zeros(2), np.zeros(2))
        self.com[:end + 1], self.velocity[:end + 1], self.acceleration[:end + 1] = hold
        bridge(0., 1., (hold[0], velocity, acceleration), hold)
        bridge(3., 4., hold, endpoint)
        self.zmp[:end + 1] = self.com[:end + 1] - self.height / 9.81 * self.acceleration[:end + 1]

    def feet_at(self, time):
        return footsteps(time, initial_feet=self.initial_feet, step=self.step,
                         cycle=self.cycle, lift=self.lift, double_support=self.double_support)

    def sample(self, time):
        if not 0 <= time <= self.times[-1]:
            raise ValueError('Reference horizon exhausted')
        index = min(round(time / self.dt), len(self.times) - 1)
        feet, weights = self.feet_at(time)
        if time <= 4.:
            cops = feet[:, :2] + self.cop_offset
            left = (self.zmp[index, 0] - cops[1, 0]) / (cops[0, 0] - cops[1, 0])
            if not 0 <= left <= 1:
                raise ValueError('Entry bridge ZMP leaves the conservative double-support interval')
            weights = np.array([left, 1 - left])
        return (feet, weights, np.r_[self.com[index], self.height],
                np.r_[self.velocity[index], 0.], np.r_[self.acceleration[index], 0.])
