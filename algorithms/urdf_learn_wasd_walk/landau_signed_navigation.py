"""Opt-in M7 joystick navigation; the learned controller remains unchanged.

Legacy one-sided training and evidence retain their original command generator.
This serial navigator is replayed from the full pose trace by the validator.
"""
import math
import struct

from algorithms.urdf_learn_wasd_walk import landau_direction_contract as contract

VERSION = 'signed_v2'
FILTERED_VERSION = 'filtered_signed_v3'
REFERENCE_VERSION = 'reference_v4'
ANTI_WINDUP_VERSION = 'anti_windup_v5'
HYSTERETIC_VERSION = 'hysteretic_v6'
VERSIONS = (VERSION, FILTERED_VERSION, REFERENCE_VERSION, ANTI_WINDUP_VERSION, HYSTERETIC_VERSION)
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
        return forward, 0., self.steering_yaw(error)

    def steering_yaw(self, error):
        # Exact zero permits the controller's straight-walking dwell. Outside
        # tolerance, steer continuously from zero in either direction.
        correction = math.copysign(max(0., abs(error)-HEADING_TOLERANCE_RAD), error)
        yaw = max(-contract.MAX_YAW_RATE, min(contract.MAX_YAW_RATE, .3*correction))
        return yaw

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


class ReferenceNavigator(Navigator):
    """Servo the frozen controller's integrated reference using joystick yaw.

    The shadow reference integrates delivered float32 commands exactly; it never
    assigns controller or simulation state. Bearing is filtered, with 3 degree
    start / 1 degree stop hysteresis so straight-walking dwell remains possible.
    Related multiloop tracking principle (this adapter still requires validation):
    https://www.mathworks.com/help/simulink/slref/prevent-block-windup-in-multiloop-control.html
    """
    STOP_TOLERANCE_RAD = math.radians(1.)

    def __init__(self, direction):
        super().__init__(direction)
        self.bearing_filter = FilteredNavigator(direction)
        self.reference = 0.
        self.previous_yaw = 0.
        self.previous_time = None
        self.steering_sign = 0

    def command(self, displacement, heading, seconds, forward=.2):
        if self.previous_time is not None:
            elapsed = seconds-self.previous_time
            if elapsed <= 0.:
                raise ValueError('Navigation clock must increase')
            self.reference += self.previous_yaw*elapsed
        yaw = 0.
        if self.direction != 'forward' and seconds >= 3.:
            axis = contract.DIRECTIONS[self.direction]
            dx, dy = displacement
            ahead = max(10., dx*axis[0]+dy*axis[1]+2.)
            bearing = math.atan2(-(ahead*axis[0]-dx), ahead*axis[1]-dy)
            bearing = self.bearing_filter.correction_error(bearing, seconds)
            if self.direction == 'backward' and not self.backward_aligned:
                self.backward_aligned = math.cos(heading) < 0.
            if self.direction == 'backward' and not self.backward_aligned:
                yaw = contract.command(self.direction, displacement, heading, seconds, forward)[2]
                self.steering_sign = 1 if yaw > 0. else 0
            else:
                error = math.atan2(math.sin(bearing-self.reference), math.cos(bearing-self.reference))
                if self.steering_sign and error*self.steering_sign <= self.STOP_TOLERANCE_RAD:
                    self.steering_sign = 0
                if not self.steering_sign and abs(error) > HEADING_TOLERANCE_RAD:
                    self.steering_sign = 1 if error > 0. else -1
                if self.steering_sign:
                    yaw = max(-contract.MAX_YAW_RATE, min(contract.MAX_YAW_RATE, .3*error))
        self.previous_yaw = struct.unpack('f', struct.pack('f', yaw))[0]
        self.previous_time = seconds
        return forward, 0., self.previous_yaw

    def record(self):
        return {'shadow_reference_rad': self.reference, 'previous_yaw': self.previous_yaw,
                'previous_time_s': self.previous_time, 'steering_sign': self.steering_sign,
                'backward_aligned': self.backward_aligned,
                'filtered_bearing_rad': self.bearing_filter.filtered_error}


class AntiWindupNavigator(FilteredNavigator):
    """Keep body-heading pursuit, but stop winding a lagging yaw reference away.

    Only the delivered joystick command is limited. The learned controller,
    simulation state, physical guards and forward speed remain unchanged.
    A ten-degree hysteresis avoids gait-cycle crossings of the tracking limit.
    Conditional integration principle (our heading-based limit is experimental):
    https://www.mathworks.com/help/simulink/slref/anti-windup-control-using-a-pid-controller.html
    """
    TRACKING_LIMIT_RAD = math.radians(30.)
    TRACKING_RELEASE_RAD = math.radians(20.)

    def __init__(self, direction):
        super().__init__(direction)
        self.reference = 0.
        self.reference_time = None
        self.previous_yaw = 0.
        self.blocked_sign = 0

    def command(self, displacement, heading, seconds, forward=.2):
        if self.reference_time is not None:
            elapsed = seconds-self.reference_time
            if elapsed <= 0.:
                raise ValueError('Navigation clock must increase')
            self.reference += self.previous_yaw*elapsed
        command = super().command(displacement, heading, seconds, forward)
        yaw = command[2]
        gap = math.atan2(math.sin(self.reference-heading), math.cos(self.reference-heading))
        if self.blocked_sign and gap*self.blocked_sign <= self.TRACKING_RELEASE_RAD:
            self.blocked_sign = 0
        if abs(gap) > self.TRACKING_LIMIT_RAD and gap*yaw > 0.:
            self.blocked_sign = 1 if yaw > 0. else -1
        if yaw*self.blocked_sign > 0.:
            yaw = 0.
        self.previous_yaw = struct.unpack('f', struct.pack('f', yaw))[0]
        self.reference_time = seconds
        return forward, 0., self.previous_yaw

    def record(self):
        return {'shadow_reference_rad': self.reference, 'previous_yaw': self.previous_yaw,
                'previous_time_s': self.reference_time, 'blocked_sign': self.blocked_sign,
                'backward_aligned': self.backward_aligned, 'filtered_error_rad': self.filtered_error}


class HystereticNavigator(AntiWindupNavigator):
    """Body-heading pursuit with longer zero-yaw dwell and reference anti-windup."""
    START_TOLERANCE_RAD = math.radians(5.)
    STOP_TOLERANCE_RAD = math.radians(2.)

    def __init__(self, direction):
        super().__init__(direction)
        self.steering_sign = 0

    def steering_yaw(self, error):
        if self.steering_sign and error*self.steering_sign <= self.STOP_TOLERANCE_RAD:
            self.steering_sign = 0
        if not self.steering_sign and abs(error) > self.START_TOLERANCE_RAD:
            self.steering_sign = 1 if error > 0. else -1
        if not self.steering_sign:
            return 0.
        return max(-contract.MAX_YAW_RATE, min(contract.MAX_YAW_RATE, .3*error))

    def record(self):
        return {**super().record(), 'steering_sign': self.steering_sign}


def make_navigator(version, direction):
    if version == VERSION:return Navigator(direction)
    if version == FILTERED_VERSION:return FilteredNavigator(direction)
    if version == REFERENCE_VERSION:return ReferenceNavigator(direction)
    if version == ANTI_WINDUP_VERSION:return AntiWindupNavigator(direction)
    if version == HYSTERETIC_VERSION:return HystereticNavigator(direction)
    raise ValueError('Unknown directional navigation')
