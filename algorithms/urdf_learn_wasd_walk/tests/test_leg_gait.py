import unittest
from types import SimpleNamespace
import torch
from algorithms.urdf_learn_wasd_walk import leg_gait_search as gait
from algorithms.urdf_learn_wasd_walk.model_spec import ACTION_JOINTS


class LegGaitTests(unittest.TestCase):
    def setUp(self):
        self.names=[*ACTION_JOINTS,'left_ankle_roll_joint','right_ankle_roll_joint']
        self.obs=torch.zeros(2,76);self.obs[:,8]=-1;self.obs[:,68]=1-2/30;self.obs[:,69]=-.2
        self.parameters=torch.tensor([(a+b)/2 for a,b in gait.PARAMETERS.values()])

    def test_ankle_feedback_has_independent_authority(self):
        self.obs[:,6]=.1
        before=gait.gait_action(self.obs,self.parameters,self.names)
        self.parameters[16]=.5
        after=gait.gait_action(self.obs,self.parameters,self.names)
        torch.testing.assert_close(before[:,:17],after[:,:17])
        self.assertTrue(torch.all((after[:,17:]-before[:,17:]).abs()>.1))
        self.assertEqual(after.shape,(2,19))
        self.assertTrue(torch.isfinite(after).all())

    def test_zero_command_keeps_the_exact_saved_standing_actor(self):
        self.obs[0,69]=0
        calls=[]
        def prior(td):
            calls.append(td['actor'].clone())
            return torch.full((len(td['actor']),19),.0123)
        policy=SimpleNamespace(parameters=self.parameters,standing_prior=prior)
        expected=gait.gait_action(self.obs,self.parameters,self.names)
        result=gait.evaluate_mean(policy,self.obs,self.names)
        torch.testing.assert_close(result[0],torch.full((19,),.0123))
        torch.testing.assert_close(result[1],expected[1])
        torch.testing.assert_close(calls[0],self.obs[:1,:69])

    def test_rejects_legacy_shape(self):
        with self.assertRaises(ValueError):gait.gait_action(torch.zeros(1,70),self.parameters,list(ACTION_JOINTS))

    def test_diagnostic_reference_transfers_support_before_flexing(self):
        parameters=torch.tensor(gait.REFERENCE_PARAMETERS)
        # Independent timing from the successful 50 Hz single-step diagnostic:
        # transfer2-5s, lift6-7s, hold7-8s, lower8-9s. The later mirrored
        # segment is a proposed search reference, not a certified gait.
        for seconds,left,right in ((4.,0.,0.),(6.5,.325,0.),(7.5,.65,0.),
                                   (9.5,0.,0.),(12.,0.,0.),(15.5,0.,.65)):
            self.obs[:,68]=1-seconds/30
            action=gait.gait_action(self.obs,parameters,self.names)*.08
            for side,expected in (('left',left),('right',right)):
                self.assertAlmostEqual(float(action[0,self.names.index(side+'_knee_joint')]),expected,places=5)
                self.assertAlmostEqual(float(action[0,self.names.index(side+'_hip_pitch_joint')]),-.5*expected,places=5)
                self.assertEqual(float(action[0,self.names.index(side+'_hip_roll_joint')]),0.)

    def test_velocity_command_has_no_startup_step(self):
        self.obs[:,68]=1
        commanded=gait.gait_action(self.obs,self.parameters,self.names)
        self.obs[:,69]=0
        stopped=gait.gait_action(self.obs,self.parameters,self.names)
        torch.testing.assert_close(commanded,stopped)
        self.obs[:,68]=1-3/30
        self.obs[:,69]=-.2
        moving=gait.gait_action(self.obs,self.parameters,self.names)
        self.obs[:,69]=0
        idle=gait.gait_action(self.obs,self.parameters,self.names)
        ankle=self.names.index('left_ankle_pitch_joint')
        self.assertLess(float(moving[0,ankle]-idle[0,ankle]),0.)

    def test_first_stride_waits_for_the_first_swing(self):
        parameters=torch.tensor(gait.REFERENCE_PARAMETERS)
        parameters[2]=.1
        for seconds in (0.,2.,4.,5.9):
            self.obs[:,68]=1-seconds/30
            action=gait.gait_action(self.obs,parameters,self.names)*.08
            for side in ('left','right'):
                self.assertEqual(float(action[0,self.names.index(side+'_hip_pitch_joint')]),0.)
        # First touchdown and the following side switch remain continuous.
        self.obs[:,68]=1-9.5/30
        before=gait.gait_action(self.obs,parameters,self.names)*.08
        self.obs[:,68]=1-10.1/30
        after=gait.gait_action(self.obs,parameters,self.names)*.08
        for side in ('left','right'):
            i=self.names.index(side+'_hip_pitch_joint')
            torch.testing.assert_close(before[:,i],after[:,i])

if __name__=='__main__':unittest.main()
