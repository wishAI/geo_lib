"""Opt-in M7 joystick navigation; the learned controller remains unchanged.

Legacy one-sided training and evidence retain their original command generator.
This serial navigator is replayed from the full pose trace by the validator.
"""
import math

from algorithms.urdf_learn_wasd_walk import landau_direction_contract as contract

VERSION = 'signed_v2'
FILTERED_VERSION = 'filtered_signed_v3'
VERSIONS = (VERSION, FILTERED_VERSION)
HEADING_TOLERANCE_RAD = math.radians(3.)


class Navigator:
    def __init__(self, direction):
        if direction not in contract.DIRECTIONS:
            raise ValueError('Unknown world gate')
        self.direction = direction
        self.backward_aligned = False

    def command(self, displacement, heading, seconds, forward=.2):
        if self.direction == 'forward' or seconds < 3.:
            return forward, 0., 0.
        # Choose the initial left arc once. Never re-arm it after an overshoot.
        if self.direction == 'backward' and not self.backward_aligned:
            self.backward_aligned = math.cos(heading) < 0.
            if not self.backward_aligned:
                return contract.command(self.direction, displacement, heading, seconds, forward)
        axis = contract.DIRECTIONS[self.direction]
        dx, dy = displacement
        progress = dx * axis[0] + dy * axis[1]
        ahead = max(10., progress + 2.)
        bearing = math.atan2(-(ahead * axis[0] - dx), ahead * axis[1] - dy)
        error = math.atan2(math.sin(bearing-heading), math.cos(bearing-heading))
        error = self.correction_error(error, seconds)
        # Exact zero permits the controller's straight-walking dwell. Outside
        # tolerance, steer continuously from zero in either direction.
        correction = math.copysign(max(0., abs(error)-HEADING_TOLERANCE_RAD), error)
        yaw = max(-contract.MAX_YAW_RATE, min(contract.MAX_YAW_RATE, .3*correction))
        return forward, 0., yaw

    def correction_error(self, error, seconds):
        return error


class FilteredNavigator(Navigator):
    """Suppress gait-cycle heading sway before selecting a steering direction."""
    TIME_CONSTANT_S = .5

    def __init__(self, direction):
        super().__init__(direction)
        self.filtered_error = None
        self.previous_time = None

    def correction_error(self, error, seconds):
        if self.previous_time is not None:
            elapsed = seconds-self.previous_time
            if elapsed <= 0.:
                raise ValueError('Navigation clock must increase')
            delta = math.atan2(math.sin(error-self.filtered_error), math.cos(error-self.filtered_error))
            filtered = self.filtered_error + (-math.expm1(-elapsed/self.TIME_CONSTANT_S))*delta
            self.filtered_error = math.atan2(math.sin(filtered), math.cos(filtered))
        else:
            self.filtered_error = error
        self.previous_time = seconds
        return self.filtered_error


def make_navigator(version, direction):
    if version == VERSION:return Navigator(direction)
    if version == FILTERED_VERSION:return FilteredNavigator(direction)
    raise ValueError('Unknown directional navigation')
