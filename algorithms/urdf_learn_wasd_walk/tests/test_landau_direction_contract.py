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


class SignedNavigationTests(unittest.TestCase):
    def test_backward_initial_arc_and_permanent_alignment_latch(self):
        from algorithms.urdf_learn_wasd_walk.landau_signed_navigation import Navigator
        for heading in (-.2,0.,.2):
            nav=Navigator('backward')
            self.assertEqual(nav.command((.02,.2),heading,3.),contract.command('backward',(.02,.2),heading,3.))
            self.assertFalse(nav.backward_aligned)
            nav.command((0.,0.),2.,60.)
            self.assertTrue(nav.backward_aligned)
            self.assertLess(nav.command((0.,0.),-.2,200.)[2],0.)
            self.assertTrue(nav.backward_aligned)

    def test_signed_corrections_and_zero_band_at_each_gate(self):
        from algorithms.urdf_learn_wasd_walk.landau_signed_navigation import Navigator,HEADING_TOLERANCE_RAD
        for direction,bearing in [('left',math.pi/2),('right',-math.pi/2),('backward',math.pi)]:
            nav=Navigator(direction)
            for offset in (-.8,.8):
                yaw=nav.command((0.,0.),bearing+offset,100.)[2]
                self.assertLess(yaw*offset,0.)
                self.assertLessEqual(abs(yaw),contract.MAX_YAW_RATE)
            for offset in (-.99,0.,.99):
                self.assertEqual(nav.command((0.,0.),bearing+offset*HEADING_TOLERANCE_RAD,100.)[2],0.)
            yaw=nav.command((0.,0.),bearing+HEADING_TOLERANCE_RAD+1e-5,100.)[2]
            self.assertAlmostEqual(yaw,-3e-6,places=10)

    def test_forward_and_initial_delay_preserve_full_speed(self):
        from algorithms.urdf_learn_wasd_walk.landau_signed_navigation import Navigator
        for direction in contract.DIRECTIONS:
            self.assertEqual(Navigator(direction).command((1.,2.),.8,2.),(.2,0.,0.))
        self.assertEqual(Navigator('forward').command((1.,2.),.8,100.),(.2,0.,0.))
        with self.assertRaises(ValueError):Navigator('unknown')

    def test_signed_navigation_cannot_run_with_legacy_training_memory(self):
        from unittest.mock import patch
        from algorithms.urdf_learn_wasd_walk import landau_forward_control as control
        with patch('sys.argv',['control','--name','unused','--checkpoint','unused','--direction-navigation','signed_v2']),patch.object(control,'train') as train:
            with self.assertRaisesRegex(ValueError,'serial evaluation only'):control.main()
            train.assert_not_called()

    def test_filter_removes_cycle_sway_but_tracks_sustained_error(self):
        from algorithms.urdf_learn_wasd_walk.landau_signed_navigation import FilteredNavigator
        nav=FilteredNavigator('left')
        yaws=[]
        for step in range(501):
            age=3.+step*.02
            heading=math.pi/2+.08*math.sin(2*math.pi*(age-3.)/.9)
            yaws.append(nav.command((0.,0.),heading,age)[2])
        self.assertEqual(max(abs(y) for y in yaws[-100:]),0.)
        for step in range(1,51):
            yaw=nav.command((0.,0.),math.pi/2+.2,13.+step*.02)[2]
        self.assertLess(yaw,-.03)
        with self.assertRaisesRegex(ValueError,'clock must increase'):
            nav.command((0.,0.),math.pi/2+.2,14.)

    def test_filter_takes_short_arc_across_angle_wrap(self):
        from algorithms.urdf_learn_wasd_walk.landau_signed_navigation import FilteredNavigator,make_navigator,Navigator
        nav=FilteredNavigator('left')
        self.assertEqual(nav.correction_error(3.1,3.),3.1)
        self.assertGreater(abs(nav.correction_error(-3.1,3.02)),3.)
        self.assertIsInstance(make_navigator('signed_v2','left'),Navigator)
        self.assertIsInstance(make_navigator('filtered_signed_v3','left'),FilteredNavigator)
        with self.assertRaises(ValueError):make_navigator('unknown','left')


class ReferenceNavigationTests(unittest.TestCase):
    def test_shadow_matches_real_controller_for_delivered_commands(self):
        import torch
        from algorithms.urdf_learn_wasd_walk.landau_turn_control import CommandMemory
        from algorithms.urdf_learn_wasd_walk.landau_signed_navigation import ReferenceNavigator
        for direction in contract.DIRECTIONS:
            nav, memory = ReferenceNavigator(direction), CommandMemory()
            for step in range(7501):
                age = step*.02
                obs = torch.zeros((1, 70))
                obs[0, 63:66] = torch.tensor(nav.command((.2*math.sin(age), -.03*age),
                    .04*age, age))
                memory.observe(obs, torch.zeros((1, 3)), torch.eye(3)[None], age)
                self.assertEqual(nav.reference, memory.reference)
            self.assertAlmostEqual(nav.record()['shadow_reference_rad'], memory.record()['integrated_reference_rad'])
            with self.assertRaisesRegex(ValueError, 'clock must increase'):
                nav.command((0., 0.), 0., age)

    def test_reference_reaches_finite_zero_dwell_without_heading_chatter(self):
        from algorithms.urdf_learn_wasd_walk.landau_signed_navigation import ReferenceNavigator
        for direction, bearing in [('left', math.pi/2), ('right', -math.pi/2)]:
            nav = ReferenceNavigator(direction)
            yaws = [nav.command((0., 0.), bearing+.1*math.sin(step*.14), step*.02)[2]
                    for step in range(6000)]
            self.assertTrue(any(yaws))
            self.assertEqual(set(yaws[-1000:]), {0.})
            self.assertLess(abs(nav.reference-bearing), math.radians(1.))
            onsets = sum(y != 0. and p == 0. for p,y in zip(yaws, yaws[1:]))
            self.assertEqual(onsets, 1)

    def test_hysteresis_continues_inside_start_band_and_stops_inside_one_degree(self):
        from algorithms.urdf_learn_wasd_walk.landau_signed_navigation import ReferenceNavigator
        nav = ReferenceNavigator('left')
        nav.reference = math.pi/2-math.radians(2.)
        self.assertEqual(nav.command((0.,0.),0.,3.)[2], 0.)
        nav.steering_sign = 1
        self.assertGreater(nav.command((0.,0.),0.,3.02)[2], 0.)
        nav.reference = math.pi/2-math.radians(.9)
        self.assertEqual(nav.command((0.,0.),0.,3.04)[2], 0.)
        self.assertEqual(nav.steering_sign, 0)

    def test_backward_initial_sway_latch_and_wrapped_reference(self):
        from algorithms.urdf_learn_wasd_walk.landau_signed_navigation import ReferenceNavigator
        for heading in (-.2, 0., .2):
            nav = ReferenceNavigator('backward')
            self.assertGreater(nav.command((.02,.2),heading,3.)[2], 0.)
            self.assertFalse(nav.backward_aligned)
            nav.command((0.,0.),2.,60.)
            self.assertTrue(nav.backward_aligned)
            nav.reference = math.pi+.2
            self.assertLess(nav.command((0.,0.),-.2,60.02)[2], 0.)
            self.assertTrue(nav.backward_aligned)


class AntiWindupNavigationTests(unittest.TestCase):
    def test_only_commands_worsening_reference_tracking_are_limited(self):
        from algorithms.urdf_learn_wasd_walk.landau_signed_navigation import AntiWindupNavigator
        nav = AntiWindupNavigator('backward')
        nav.reference = math.radians(170.)
        # Facing beyond the gate, with the controller reference already lagging.
        self.assertEqual(nav.command((0.,0.),math.radians(215.),100.)[2], 0.)
        self.assertEqual(nav.blocked_sign, -1)
        # Remain blocked between the start and release thresholds.
        self.assertEqual(nav.command((0.,0.),math.radians(195.),101.)[2], 0.)
        self.assertEqual(nav.blocked_sign, -1)
        nav.command((0.,0.),math.radians(189.),102.)
        self.assertEqual(nav.blocked_sign, 0)
        # A command that reduces the same reference lag remains available.
        nav = AntiWindupNavigator('backward')
        nav.reference = math.radians(130.)
        self.assertGreater(nav.command((0.,0.),math.radians(175.),100.)[2], 0.)

    def test_delivered_commands_match_memory_including_blocked_intervals(self):
        import torch
        from algorithms.urdf_learn_wasd_walk.landau_turn_control import CommandMemory
        from algorithms.urdf_learn_wasd_walk.landau_signed_navigation import AntiWindupNavigator
        nav, memory = AntiWindupNavigator('backward'), CommandMemory()
        blocked = 0
        for step in range(10001):
            age = step*.02
            heading = min(.047*age,math.radians(220.))
            obs = torch.zeros((1,70))
            obs[0,63:66] = torch.tensor(nav.command((0.,0.),heading,age))
            memory.observe(obs,torch.zeros((1,3)),torch.eye(3)[None],age)
            self.assertEqual(nav.reference,memory.reference)
            blocked += int(bool(nav.blocked_sign))
        self.assertGreater(blocked,100)


class HystereticNavigationTests(unittest.TestCase):
    def test_body_error_hysteresis_preserves_steering_between_thresholds(self):
        from algorithms.urdf_learn_wasd_walk.landau_signed_navigation import HystereticNavigator
        for sign in (-1,1):
            nav = HystereticNavigator('left')
            self.assertEqual(nav.steering_yaw(sign*math.radians(4.)),0.)
            self.assertGreater(sign*nav.steering_yaw(sign*math.radians(6.)),0.)
            self.assertGreater(sign*nav.steering_yaw(sign*math.radians(3.)),0.)
            self.assertEqual(nav.steering_yaw(sign*math.radians(1.9)),0.)
            self.assertEqual(nav.steering_yaw(sign*math.radians(4.9)),0.)

    def test_filtered_heading_sway_retains_zero_command_dwell(self):
        from algorithms.urdf_learn_wasd_walk.landau_signed_navigation import HystereticNavigator
        for direction,bearing in [('left',math.pi/2),('right',-math.pi/2)]:
            nav = HystereticNavigator(direction)
            for step in range(501):
                age = 3.+step*.02
                yaw = nav.command((0.,0.),bearing+.07*math.sin(step*.14),age)[2]
                self.assertEqual(yaw,0.)
            self.assertEqual(nav.record()['steering_sign'],0)


if __name__ == '__main__':
    unittest.main()
