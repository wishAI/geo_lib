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

    def test_rejects_reset_and_nonfinite_trace(self):
        for times, positions in [([0., 0.], [(0., 0.), (0., 11.)]),
                                 ([0., 1.], [(0., 0.), (math.nan, 11.)])]:
            with self.assertRaises(ValueError):
                contract.gate_metrics('forward', times, positions)


if __name__ == '__main__':
    unittest.main()
