import math
import unittest
from algorithms.urdf_learn_wasd_walk import landau_teleop_contract as c


class TeleopContractTests(unittest.TestCase):
    def test_commands_contain_both_turns_and_two_stops(self):
        self.assertGreater(c.command_profile(6)[2],0)
        self.assertLess(c.command_profile(32)[2],0)
        self.assertEqual(c.command_profile(20),(0.,0.,0.))
        self.assertEqual(c.command_profile(50),(0.,0.,0.))
        self.assertEqual(c.command_profile(25),(0.,0.,0.))
        self.assertAlmostEqual(c.command_profile(26)[0],.1)
        self.assertEqual(c.command_profile(27),(.2,0.,0.))
        self.assertAlmostEqual(sum(c.command_profile(i*.02)[2]*.02 for i in range(3000)),0.)

    def fixture(self):
        m=dict(duration_s=60.,policy_inference_steps=3000,reset_count=0,done_count=0,fall_count=0,
               max_reference_tilt_rad=.1,root_height_drop_m=.01,max_abs_action=.8,simultaneous_air_fraction=0.)
        r=dict(blocks=[dict(label=n,start_s=a,end_s=b,yaw_sign=y,complete=True,
                           forward_progress_m=.1*(b-a),max_heading_excursion_rad=0.,heading_change_rad=y*c.YAW_RATE*(b-a),
                           completed_swings={'left':3,'right':3}) for n,a,b,y in c.BLOCKS],
               holds=[dict(start_s=a,end_s=b,complete=True,drift_m=.01,settled_speed_mps=.01,
                           heading_drift_rad=.01) for a,b in c.HOLDS])
        return m,r

    def test_each_turn_and_restart_is_required(self):
        m,r=self.fixture();self.assertEqual(c.evaluate_gate(m,r),[])
        r['blocks'][3]['heading_change_rad']*= -1
        self.assertIn('right wrong yaw response',c.evaluate_gate(m,r))
        m,r=self.fixture();r['blocks'][2]['completed_swings']['left']=0
        self.assertIn('restart_forward no repeated walking swings',c.evaluate_gate(m,r))

    def test_each_hold_must_be_complete_finite_and_settled(self):
        for index in (0,1):
            m,r=self.fixture();r['holds'][index]['drift_m']=.04
            self.assertTrue(c.evaluate_gate(m,r))
            m,r=self.fixture();r['holds'][index]['settled_speed_mps']=math.nan
            self.assertTrue(c.evaluate_gate(m,r))
            m,r=self.fixture();r['holds'][index]['complete']=False
            self.assertTrue(c.evaluate_gate(m,r))

    def test_missing_nonfinite_and_uncommanded_turn_fail(self):
        m,r=self.fixture();self.assertTrue(c.evaluate_gate(m,{'blocks':[],'holds':[]}))
        m,r=self.fixture();m['duration_s']=math.nan;self.assertTrue(c.evaluate_gate(m,r))
        m,r=self.fixture();r['blocks'][0]['forward_progress_m']=math.nan;self.assertTrue(c.evaluate_gate(m,r))
        m,r=self.fixture();r['blocks'][-1]['max_heading_excursion_rad']=2*math.pi
        self.assertIn('straight_after_turn uncommanded turn',c.evaluate_gate(m,r))

    def test_no_padding_short_or_fallen_episode(self):
        for key,value in [('duration_s',59.),('policy_inference_steps',2999),('fall_count',1)]:
            m,r=self.fixture();m[key]=value;self.assertTrue(c.evaluate_gate(m,r))


if __name__=='__main__':unittest.main()
