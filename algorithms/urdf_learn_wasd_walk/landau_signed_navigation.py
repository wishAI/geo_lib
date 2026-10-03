"""Opt-in M7 joystick navigation; the learned controller remains unchanged.

Legacy one-sided training and evidence retain their original command generator.
This serial navigator is replayed from the full pose trace by the validator.
"""
import math

from algorithms.urdf_learn_wasd_walk import landau_direction_contract as contract

VERSION = 'signed_v2'
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
        # Exact zero permits the controller's straight-walking dwell. Outside
        # tolerance, steer continuously from zero in either direction.
        correction = math.copysign(max(0., abs(error)-HEADING_TOLERANCE_RAD), error)
        yaw = max(-contract.MAX_YAW_RATE, min(contract.MAX_YAW_RATE, .3*correction))
        return forward, 0., yaw
