import copy
import json
import unittest
import numpy as np
from algorithms.urdf_learn_wasd_walk import walking_speed_contract as speed
from algorithms.urdf_learn_wasd_walk.milestone_schedule import ordered_milestones,advance_schedule,require_current_gate


class SpeedTests(unittest.TestCase):
    def setUp(self):
        self.t=np.arange(2001)*.02;self.q=np.zeros((2001,3));self.q[:,1]=-.101*self.t
        self.feet=[]
        for t in np.arange(1,20001)*.002:
            phase=t%1
            self.feet.append({'time_s':t,'left_contact':not .1<phase<.4,'right_contact':not .6<phase<.9,
                'left_clearance_m':.03,'right_clearance_m':.03,'mean_slip_mps':.005})

    def test_fixed_full_trace_accepts_alternating_sustained_walk(self):
        response=speed.metrics(self.t,self.q,self.feet)
        self.assertEqual(json.loads(json.dumps(response)),response)
        self.assertFalse(speed.failures(response,.1))
        self.assertAlmostEqual(response['sustained_forward_speed_mps'],.101)
        self.assertTrue(speed.failures(response,.4))

    def test_sprint_then_stop_and_wrong_direction_fail(self):
        self.q[:,1]=-.4*np.minimum(self.t,20)
        response=speed.metrics(self.t,self.q,self.feet)
        self.assertGreater(response['mean_forward_speed_mps'],.1)
        self.assertTrue(speed.failures(response,.1))
        self.q[:,1]=.2*self.t
        self.assertTrue(speed.failures(speed.metrics(self.t,self.q,self.feet),.1))
        self.q[:,0]=self.t;self.q[:,1]=0
        self.assertTrue(speed.failures(speed.metrics(self.t,self.q,self.feet),.1))

    def test_sliding_frozen_feet_missing_time_and_resets_fail(self):
        for row in self.feet:row['mean_slip_mps']=.03
        self.assertTrue(speed.failures(speed.metrics(self.t,self.q,self.feet),.1))
        for row in self.feet:row.update(mean_slip_mps=0.,left_contact=True,right_contact=True)
        self.assertTrue(speed.failures(speed.metrics(self.t,self.q,self.feet),.1))
        with self.assertRaises(ValueError):speed.metrics(self.t[:-1],self.q[:-1],self.feet)
        self.t[100:]-=1
        with self.assertRaises(ValueError):speed.metrics(self.t,self.q,self.feet)

    def test_speed_priority_preserves_legacy_order(self):
        legacy=[{'id':'m'+str(i),'status':'passed' if i<=3 else 'not_started'} for i in range(1,13)]
        ledger={'milestones':copy.deepcopy(legacy)};advance_schedule(ledger)
        require_current_gate(ledger,'m4')
        ledger['speedMilestones']=[{'id':speed.gate_id(v),'status':'not_started'} for v in speed.TARGETS]
        ledger['executionOrder']=[x['id'] for x in legacy[:3]+ledger['speedMilestones']+legacy[3:]]
        advance_schedule(ledger);require_current_gate(ledger,speed.gate_id(.1))
        with self.assertRaises(ValueError):require_current_gate(ledger,'m4')
        self.assertEqual([x['id'] for x in ledger['milestones']],[x['id'] for x in legacy])
        self.assertEqual(len(ordered_milestones(ledger)),16)
        self.assertEqual(speed.TARGETS[-1],1.2)
        ledger['executionOrder'].append('m1')
        with self.assertRaises(ValueError):ordered_milestones(ledger)

    def test_hops_do_not_count_as_supported_steps(self):
        for row in self.feet:
            row['right_contact']=row['left_contact']
        response=speed.metrics(self.t,self.q,self.feet)
        self.assertTrue(speed.failures(response,.1))
        self.assertTrue(all(block['completed_swings']=={'left':0,'right':0} for block in response['blocks']))

    def test_rotated_pelvis_offset_is_part_of_displacement(self):
        import mujoco
        model=mujoco.MjModel.from_xml_string('<mujoco><worldbody><body><freejoint/><geom size=".1"/><body name="root_x" pos="0 0 .3"/></body></worldbody></mujoco>')
        q=np.tile(model.qpos0,(2,1));q[1,3:7]=[np.cos(.1),np.sin(.1),0,0]
        positions=speed.pelvis_positions(model,q)
        self.assertAlmostEqual(positions[1,1],-.3*np.sin(.2))
        self.assertEqual(q[1,1],0.)


if __name__=='__main__':unittest.main()
