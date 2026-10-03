import unittest
import torch
from algorithms.urdf_learn_wasd_walk.landau_turn_control import CommandMemory


class CommandMemoryTests(unittest.TestCase):
    def setUp(self):
        self.m=CommandMemory()
        self.obs=torch.zeros(2,70);self.obs[:,62]=1.
        self.pos=torch.zeros(2,3);self.rot=torch.eye(3).expand(2,-1,-1)

    def step(self,t,forward,yaw=0.):
        self.obs[:,63]=forward;self.obs[:,65]=yaw;self.obs[:,62]=1.-t/30.
        return self.m.observe(self.obs,self.pos,self.rot,t)

    def test_each_stop_gets_a_fresh_anchor_without_state_writes(self):
        self.step(0,.2);self.pos[:,1]=1.
        self.step(20,0);self.assertEqual(self.m.anchor_time,20)
        self.step(25,0);self.step(25.02,.0001)
        self.pos[:,1]=2.;before=self.pos.clone();self.step(50,0)
        self.assertEqual(self.m.anchor_time,50)
        self.assertTrue(torch.equal(self.m.anchor_xy,self.pos[:,:2]))
        self.assertTrue(torch.equal(before,self.pos))
        self.assertEqual([e['kind'] for e in self.m.events],['stop_anchor','restart','stop_anchor'])

    def test_restart_preserves_certified_moving_prior(self):
        self.step(0,.2);self.step(20,0);self.pos[:,1]=.001
        _,prior_before=self.step(25,0)
        adjusted,prior_after=self.step(25.02,.0001)
        self.assertTrue(torch.equal(prior_after,self.obs[:,:63]))
        self.assertAlmostEqual(float(adjusted[0,62]),1.-25.02/30.,places=6)
        self.assertAlmostEqual(self.m.restart_time,25.02)

    def test_second_braking_does_not_use_old_stop_location(self):
        self.step(0,.2);self.step(20,0);self.step(25.02,.0001)
        self.step(27,.2);self.pos[:,1]=2.
        _,prior=self.step(49,.1)
        self.assertTrue(torch.equal(prior[:,60:62],self.obs[:,60:62]))
        _,prior=self.step(50,0)
        self.assertTrue(torch.equal(prior[:,60:62],torch.zeros(2,2)))

    def test_yaw_integrates_commands_and_ramps_at_actual_onset(self):
        self.step(0,.2);self.step(6,.2,.04)
        self.assertEqual(self.m.reference,0.)
        self.assertEqual(self.m.ramp(self.m.yaw_onset,1.),0.)
        self.step(7,.2,.04);self.assertAlmostEqual(self.m.reference,.04)
        self.step(18,.2,0.);self.assertAlmostEqual(self.m.reference,.48)
        self.step(32,.2,-.04);self.assertAlmostEqual(self.m.reference,.48)
        self.assertEqual(self.m.ramp(self.m.yaw_onset,1.),0.)
        self.step(44,.2,0.);self.assertAlmostEqual(self.m.reference,0.)

    def test_initial_zero_command_preserves_standing_age_and_drift(self):
        self.obs[:,60]=.2
        obs,prior=self.step(20,0)
        self.assertTrue(torch.equal(obs,self.obs))
        self.assertTrue(torch.equal(prior,self.obs[:,:63]))
        self.assertIsNone(self.m.anchor_time)

    def test_cruising_gain_does_not_activate_in_certified_command_streams(self):
        from algorithms.urdf_learn_wasd_walk import landau_turn_control as turn, landau_teleop_contract as teleop
        turn.configure_profile(44.)
        try:
            for mode,seconds in [('turn',56),('teleop',60)]:
                self.m=CommandMemory()
                for step in range(seconds*50):
                    age=step*.02
                    if mode=='turn': forward,yaw,_=turn.command_profile(age)
                    else: forward,_,yaw=teleop.command_profile(age)
                    self.step(age,forward,yaw)
                    self.assertEqual(self.m.left_cruise_blend,0.)
                    if self.m.left_cruise_start is not None:
                        self.assertEqual(self.m.ramp(self.m.left_cruise_start+.2,1.),0.)
        finally: turn.configure_profile(14.)

    def test_cruising_gain_requires_sustained_straight_full_forward(self):
        self.step(3,.2,.0357);self.step(47,.2)
        self.step(47.1,.2);self.assertEqual(self.m.ramp(self.m.left_cruise_start+.2,1.),0.)
        self.step(48.2,.2);self.assertAlmostEqual(self.m.ramp(self.m.left_cruise_start+.2,1.),1.)
        self.assertAlmostEqual(self.m.left_cruise_blend,1.)
        self.step(48.3,.2,.001);self.assertIsNone(self.m.left_cruise_start)
        self.assertAlmostEqual(self.m.left_cruise_blend,.9)
        self.step(49,.199);self.assertIsNone(self.m.left_cruise_start)
        self.assertAlmostEqual(self.m.left_cruise_blend,.2)
        self.step(49.2,.199);self.assertAlmostEqual(self.m.left_cruise_blend,0.)


if __name__=='__main__':unittest.main()
