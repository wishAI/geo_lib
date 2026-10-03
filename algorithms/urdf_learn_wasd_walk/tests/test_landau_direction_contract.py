import math
import unittest

from algorithms.urdf_learn_wasd_walk import landau_direction_contract as contract


class DirectionContractTests(unittest.TestCase):
    def test_commands_turn_toward_all_world_gates(self):
        for direction, sign in [('forward', 0), ('left', 1), ('right', -1), ('backward', 1)]:
            command = contract.command(direction, (0., 0.), 0., 3.)
            self.assertEqual(command[:2], (.2, 0.))
            self.assertEqual(command[2], sign * contract.MAX_YAW_RATE)
        self.assertEqual(contract.command('right', (0., 0.), 0., 2.), (.2, 0., 0.))
        self.assertAlmostEqual(contract.command('left', (0., 0.), math.pi/2, 3.)[2], 0.)
        for heading in (-.02,.02):
            self.assertEqual(contract.command('forward',(0.,0.),heading,10.)[2],0.)
        for heading in (-math.pi, -.5, 0., .5, math.pi):
            self.assertLessEqual(contract.command('right',(8.,2.),heading,80.)[2],0.)
            self.assertGreaterEqual(contract.command('left',(-8.,2.),heading,80.)[2],0.)

    def test_crossing_requires_correct_direction_and_gate_width(self):
        for direction, (x, y) in contract.DIRECTIONS.items():
            result = contract.gate_metrics(direction, [0., 10., 12.],
                [(4., 3.), (4.+9*x, 3.+9*y), (4.+11*x, 3.+11*y)])
            self.assertEqual(result['gate_crossing_time_s'], 11.)
            self.assertEqual(result['direction_progress_m'], 11.)
        self.assertIsNone(contract.gate_metrics('forward', [0., 10.], [(0., 0.), (1., 11.)])['gate_crossing_time_s'])
        self.assertIsNone(contract.gate_metrics('backward', [0., 10.], [(0., 0.), (0., 11.)])['gate_crossing_time_s'])

    def test_backward_keeps_left_arc_under_initial_heading_sway(self):
        for heading in (-.2,0.,.2):
            self.assertEqual(contract.command('backward',(.02,.2),heading,3.)[2],contract.MAX_YAW_RATE)
        self.assertEqual(contract.command('backward',(0.,0.),-math.pi+.1,100.)[2],0.)

    def test_rejects_reset_and_nonfinite_trace(self):
        for times, positions in [([0., 0.], [(0., 0.), (0., 11.)]),
                                 ([0., 1.], [(0., 0.), (math.nan, 11.)])]:
            with self.assertRaises(ValueError):
                contract.gate_metrics('forward', times, positions)

    def test_training_replay_keeps_steering_pulses_and_extends_straight_only(self):
        rows=[(.2,0.,0.)]*150+[(.2,0.,contract.MAX_YAW_RATE)]*2200
        rows += [(.2,0.,0.)]*25+[(.2,0.,.001)]*5+[(.2,0.,0.)]*50
        protocol=contract.RecordedLeftTrainingProtocol(rows,80.)
        self.assertEqual(protocol.command_profile(47.5),(.2,0.,.001))
        self.assertEqual(protocol.command_profile(79.),(.2,0.,0.))
        self.assertEqual(protocol.BLOCKS[-1],('straight_after_turn',47.6,80.,0))
        with self.assertRaises(ValueError):contract.RecordedLeftTrainingProtocol(rows,50.)
        with self.assertRaises(ValueError):contract.RecordedLeftTrainingProtocol([(.2,0.,-.01)],80.)

    def test_yaw_calibration_uses_exact_m5_commands(self):
        from algorithms.urdf_learn_wasd_walk import landau_turn_control as turn
        protocol=contract.TurnHoldTrainingProtocol(56.,44.)
        turn.configure_profile(44.)
        try:
            for step in range(2800):
                t=step*.02;forward,yaw,_=turn.command_profile(t)
                self.assertEqual(protocol.command_profile(t),(forward,0.,yaw))
            self.assertEqual(protocol.HOLDS,((51.,56.),))
        finally:turn.configure_profile(14.)


if __name__ == '__main__':
    unittest.main()
