import math
import unittest

from algorithms.urdf_learn_wasd_walk.landau_turn_control import command_profile, configure_profile, evaluate_turn_gate


class TurnContractTests(unittest.TestCase):
    def test_slower_profile_preserves_angle_and_stop(self):
        try:
            configure_profile(20.)
            dt=.002
            angle=sum(command_profile(i*dt)[1]*dt for i in range(16000))
            self.assertAlmostEqual(angle,math.pi/2,places=10)
            self.assertEqual(command_profile(25),(0.,0.,math.pi/2))
        finally:configure_profile(14.)

    def test_command_integrates_to_quarter_turn_and_stops(self):
        dt=.002
        angle=sum(command_profile(i*dt)[1]*dt for i in range(13000))
        self.assertAlmostEqual(angle,math.pi/2,places=10)
        self.assertEqual(command_profile(26),(0.,0.,math.pi/2))

    def test_hold_needs_complete_stationary_evidence(self):
        metrics=dict(reset_count=0,done_count=0,fall_count=0,duration_s=26.,
            max_reference_tilt_rad=.2,root_height_drop_m=.01,max_abs_action=1.,
            simultaneous_air_fraction=.01,hold_max_heading_error_rad=.02,
            hold_max_drift_m=.01,hold_max_horizontal_speed_mps=.01,
            hold_max_yaw_speed_rad_s=.02,hold_duration_s=5.,hold_max_abs_command=0.,
            policy_inference_steps=1300,control_steps=1300,
            leg_joint_excursion_rad={n:.2 for n in ('left_hip_pitch_joint','right_hip_pitch_joint','left_knee_joint','right_knee_joint')})
        self.assertEqual(evaluate_turn_gate(metrics,26.),[])
        for key,value in [('hold_duration_s',4.99),('hold_max_heading_error_rad',.1),
                          ('hold_max_horizontal_speed_mps',.06),('hold_max_drift_m',.04),
                          ('hold_max_abs_command',.01),('reset_count',1)]:
            self.assertTrue(evaluate_turn_gate({**metrics,key:value},26.),key)
        self.assertTrue(evaluate_turn_gate({**metrics,'hold_max_drift_m':float('nan')},26.))


if __name__=='__main__':unittest.main()
