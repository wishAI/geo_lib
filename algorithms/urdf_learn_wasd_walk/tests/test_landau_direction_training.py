import math
import unittest
import torch

from algorithms.urdf_learn_wasd_walk import landau_direction_contract as contract
from algorithms.urdf_learn_wasd_walk.landau_direction_training import commands,DirectionMemory,GateTracker,common_start_offsets,balance_grid,cruise_balance_parameters,cruise_rate_grid,cruise_balance_grid
from algorithms.urdf_learn_wasd_walk.landau_turn_control import CommandMemory,commanded_action,candidate_table_reference,absolute_hold_fitness


class DirectionTrainingTests(unittest.TestCase):
    def test_wide_cruise_grid_covers_support_timing_and_preserves_controls(self):
        seed=torch.zeros(20);seed[5:7]=torch.tensor([-.013,.018]);seed[15:17]=torch.tensor([.004,-.006])
        lows=torch.full((20,),-.25);highs=-lows
        result=cruise_balance_grid(seed,64,lows,highs)
        fixed=[i for i in range(20) if i not in (15,16)]
        self.assertTrue(torch.equal(result[:,fixed],seed[fixed].repeat(64,1)))
        self.assertTrue(torch.equal(result[0],seed));self.assertTrue(torch.equal(result[1],seed))
        self.assertTrue(torch.equal(result[2,15:17],seed[5:7]))
        self.assertAlmostEqual(float(result[:,16].max()-result[:,16].min()),.30,places=6)
        self.assertAlmostEqual(float(result[:,15].max()-result[:,15].min()),.036,places=6)
        self.assertTrue(torch.equal(result,cruise_balance_grid(seed,64,lows,highs)))

    def test_cruise_rate_grid_preserves_seed_and_duplicate_controls(self):
        seed=torch.linspace(-.1,.1,21);seed[20]=.012
        result=cruise_rate_grid(seed)
        self.assertTrue(torch.equal(result[:,:20],seed[:20].repeat(8,1)))
        self.assertTrue(torch.equal(result[0],seed));self.assertTrue(torch.equal(result[1],seed))
        self.assertTrue(torch.equal(result[:,20],torch.tensor([.012,.012,-.01,.005,.01,.015,.02,.03])))

    def test_cruise_search_preserves_calibrated_yaw_and_feedback_exactly(self):
        seed=torch.linspace(-.1,.1,20)
        lows=torch.full((20,),-.25);highs=-lows
        normalized=torch.linspace(-1.,1.,80).reshape(4,20)
        normalized[0]=2*(seed-lows)/(highs-lows)-1.
        result=cruise_balance_parameters(normalized,lows,highs,seed)
        fixed=[i for i in range(20) if i not in (15,16)]
        self.assertTrue(torch.equal(result[:,fixed],seed[fixed].repeat(4,1)))
        self.assertTrue(torch.equal(result[0],seed))
        self.assertFalse(torch.equal(result[0,15:17],result[-1,15:17]))
        self.assertTrue(bool(((result[:,15:17]>=lows[15:17])&(result[:,15:17]<=highs[15:17])).all()))

    def test_hold_score_prefers_absolute_target_and_does_not_reward_missing_hold(self):
        # A90degree relative turn can still stop20degrees short of the absolute target.
        score=absolute_hold_fitness(torch.tensor([56.,56.,40.]),
            torch.tensor([math.radians(20),math.radians(4),math.radians(30)]),
            torch.tensor([math.radians(21),math.radians(5),0.]),torch.tensor([.02,.02,.0]))
        self.assertGreater(float(score[1]),float(score[0]))
        self.assertLess(float(score[2]),float(score[0]))

    def test_candidate_selection_cannot_silently_read_a_later_generation(self):
        tables={'generation_candidate_tables':{'candidates_0.json':'first','candidates_1.json':'second'}}
        path,digest=candidate_table_reference('/tmp/run/model_0.pt',tables)
        self.assertEqual(path.name,'candidates_0.json');self.assertEqual(digest,'first')
        with self.assertRaises(ValueError):candidate_table_reference('/tmp/run/model_2.pt',tables)
        legacy={'generations_completed':2,'candidate_table_sha256':'final'}
        with self.assertRaises(ValueError):candidate_table_reference('/tmp/run/model_0.pt',legacy)
        path,digest=candidate_table_reference('/tmp/run/model_1.pt',legacy)
        self.assertEqual(path.name,'candidates.json');self.assertEqual(digest,'final')

    def test_common_starts_preserve_nominal_and_do_not_consume_search_rng(self):
        nominal=torch.arange(12,dtype=torch.float32);joints=torch.tensor([7,9,11])
        before=torch.random.get_rng_state().clone()
        offsets=common_start_offsets(nominal,joints,4,4242)
        self.assertTrue(torch.equal(before,torch.random.get_rng_state()))
        self.assertTrue(torch.equal(offsets,common_start_offsets(nominal,joints,4,4242)))
        self.assertTrue(torch.equal(offsets[0],torch.zeros_like(nominal)))
        self.assertTrue(torch.equal(offsets[:,:7],torch.zeros(4,7)))
        self.assertLessEqual(float(offsets.abs().max()),.002)
        self.assertFalse(torch.equal(offsets,common_start_offsets(nominal,joints,4,4243)))
        starts=nominal+offsets.repeat(16,1)
        for group in range(16):self.assertTrue(torch.equal(starts[:4],starts[group*4:group*4+4]))

    def test_balance_grid_keeps_frozen_parameters_and_duplicate_controls(self):
        center=torch.linspace(-.4,.4,18)
        grid=balance_grid(center,64,.16)
        self.assertTrue(torch.equal(grid[0],center));self.assertTrue(torch.equal(grid[1],center))
        fixed=[i for i in range(18) if i not in (5,6)]
        self.assertTrue(torch.equal(grid[:,fixed],center[fixed].repeat(64,1)))
        self.assertLessEqual(float((grid[:,5:7]-center[5:7]).abs().max()),.1600001)
        self.assertTrue(torch.equal(grid,balance_grid(center,64,.16)))
        with self.assertRaises(ValueError):balance_grid(center,8,.16)

    def test_batched_commands_match_scalar_at_angle_boundaries(self):
        xy=torch.tensor([[0.,0.],[.02,.2],[-4.,2.],[-10.,-.74],[0.,-11.]],dtype=torch.float64)
        headings=torch.tensor([0.,-.2,math.pi-1e-8,-math.pi+1e-8,-math.pi+.1],dtype=torch.float64)
        for direction in contract.DIRECTIONS:
            for age in (0.,2.98,3.,95.):
                expected=torch.tensor([contract.command(direction,p.tolist(),float(h),age) for p,h in zip(xy,headings)],dtype=torch.float64)
                self.assertTrue(torch.allclose(commands(direction,xy,headings,age),expected,rtol=0.,atol=1e-12))

    def test_independent_memory_and_actions_match_scalar_controllers(self):
        from algorithms.urdf_learn_wasd_walk import landau_gait_search as base
        from algorithms.urdf_learn_wasd_walk.landau_forward_control import make_actor
        names=[s+'_'+j+'_joint' for s in ('left','right') for j in ('hip_pitch','hip_yaw','hip_roll','knee','ankle_pitch','toe')]+['waist_yaw_joint','waist_roll_joint','waist_pitch_joint','left_shoulder_pitch_joint','right_shoulder_pitch_joint']
        walking=torch.tensor([(lo+hi)/2 for lo,hi in base.PARAMETERS.values()])
        params=torch.tensor([.1,-.3,.04,.01,-.01,-.01,-.02,1.,.04,0.,0.,.07,-.01,-.02,.02,.005,-.02,1.4])
        zero_feedback=torch.cat((params,torch.zeros(2)))
        active_feedback=torch.cat((params,torch.tensor([.05,.015])))
        zero_cruise=torch.cat((active_feedback,torch.zeros(1)))
        active_cruise=torch.cat((active_feedback,torch.tensor([.03])))
        unaffected=[i for i,n in enumerate(names) if 'hip_roll' not in n]
        hips=[i for i,n in enumerate(names) if 'hip_roll' in n]
        prior=make_actor(63);prior.eval()
        for direction,sign in [('left',1),('right',-1)]:
            memory=DirectionMemory(3,'cpu',direction);serial=[CommandMemory() for _ in range(3)]
            pos=torch.zeros(3,3);rot=torch.eye(3).repeat(3,1,1)
            for step in range(351):
                t=step*.02;obs=torch.zeros(3,70);obs[:,8]=-1.;obs[:,62]=1.-t/30.;obs[:,63]=.2
                obs[:,6]=.1;obs[:,4]=torch.tensor([-3.,0.,3.])
                for i in range(3):
                    active=t>=3. and (t<4.+.3*i or 5.+.1*i<=t<5.14+.1*i)
                    obs[i,65]=sign*.03 if active else 0.
                    obs[i,68]=math.sin(.2*i);obs[i,69]=math.cos(.2*i)-1.
                batch_obs,batch_prior=memory.observe(obs,pos,rot,t)
                for i,scalar in enumerate(serial):
                    single,standing=scalar.observe(obs[i:i+1],pos[i:i+1],rot[i:i+1],t)
                    self.assertAlmostEqual(float(memory._reference[i]),scalar.reference,places=12)
                    self.assertAlmostEqual(float(memory._blend[i]),scalar.left_cruise_blend,places=12)
                    self.assertAlmostEqual(float(memory._feedback_blend[i]),scalar.left_feedback_blend,places=12)
                    if step in (0,149,150,219,269,300,350):
                        with torch.no_grad():
                            expected=commanded_action(base,walking,prior,single,standing,params,names,scalar.reference,scalar.turned,scalar)
                            actual=commanded_action(base,walking,prior,batch_obs,batch_prior,params,names,memory.reference,memory.turned,memory)[i:i+1]
                        self.assertTrue(torch.equal(actual,expected),(direction,step,i,float((actual-expected).abs().max())))
                        with torch.no_grad():
                            zero=commanded_action(base,walking,prior,single,standing,zero_feedback,names,scalar.reference,scalar.turned,scalar)
                            feedback=commanded_action(base,walking,prior,single,standing,active_feedback,names,scalar.reference,scalar.turned,scalar)
                            batched_feedback=commanded_action(base,walking,prior,batch_obs,batch_prior,active_feedback,names,memory.reference,memory.turned,memory)[i:i+1]
                            zero_rate=commanded_action(base,walking,prior,single,standing,zero_cruise,names,scalar.reference,scalar.turned,scalar)
                            cruise=commanded_action(base,walking,prior,single,standing,active_cruise,names,scalar.reference,scalar.turned,scalar)
                            batched_cruise=commanded_action(base,walking,prior,batch_obs,batch_prior,active_cruise,names,memory.reference,memory.turned,memory)[i:i+1]
                        self.assertTrue(torch.equal(zero,expected),'Zero extension changed baseline actions')
                        self.assertTrue(torch.equal(feedback,batched_feedback),'Residual scalar/batch behavior differs')
                        self.assertTrue(torch.equal(feedback[:,unaffected],expected[:,unaffected]))
                        self.assertLessEqual(float((feedback-expected).abs().max()),.03/.08+1e-6)
                        if direction=='right':self.assertTrue(torch.equal(feedback,expected))
                        self.assertTrue(torch.equal(zero_rate,feedback),'Zero cruise extension changed actions')
                        self.assertTrue(torch.equal(cruise,batched_cruise),'Cruise scalar/batch behavior differs')
                        self.assertTrue(torch.equal(cruise[:,unaffected],feedback[:,unaffected]))
                        expected_delta=scalar.left_cruise_blend*(.03*single[:,4]).clamp(-.05,.05)/.08
                        self.assertTrue(torch.allclose((cruise-feedback)[:,hips],expected_delta[:,None].expand(-1,2),atol=1e-6,rtol=0.))

    def test_gate_requires_crossing_width_and_keeps_terminal_result(self):
        gate=GateTracker('left',torch.zeros(3,3))
        active=torch.ones(3,dtype=torch.bool)
        gate.update(torch.tensor([[-9.,.5,0.],[-9.,1.,0.],[-9.,0.,0.]]),1.,active)
        active[2]=False
        finished=gate.update(torch.tensor([[-11.,.5,0.],[-11.,1.,0.],[-11.,0.,0.]]),1.02,active)
        self.assertEqual(finished.tolist(),[True,False,False])
        self.assertAlmostEqual(float(gate.crossing_time[0]),1.01)
        self.assertAlmostEqual(float(gate.crossing_lateral[0]),.5)
        gate.update(torch.zeros(3,3),1.04,active & ~gate.completed)
        self.assertEqual(float(gate.progress[0]),11.)
        self.assertAlmostEqual(float(gate.crossing_time[0]),1.01)

    def test_terminal_memory_does_not_accumulate_further_commands(self):
        memory=DirectionMemory(2,'cpu','left');obs=torch.zeros(2,70);obs[:,63]=.2;obs[:,65]=.03
        pos=torch.zeros(2,3);rot=torch.eye(3).repeat(2,1,1)
        memory.observe(obs,pos,rot,3.)
        memory.observe(obs,pos,rot,3.02,torch.tensor([False,True]))
        self.assertEqual(float(memory._reference[0]),0.)
        self.assertGreater(float(memory._reference[1]),0.)
        self.assertEqual(float(memory._feedback_blend[0]),0.)
        self.assertAlmostEqual(float(memory._feedback_blend[1]),.02)


if __name__=='__main__':unittest.main()
