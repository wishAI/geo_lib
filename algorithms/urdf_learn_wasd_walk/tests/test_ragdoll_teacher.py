"""Contract checks for scratch IK and preservation of physical model/state."""
import unittest
try:
    import numpy as np
    import mujoco
    from algorithms.urdf_learn_wasd_walk.mujoco_backend import build_model, initialize
    from algorithms.urdf_learn_wasd_walk.mujoco_ragdoll_teacher import WalkingTeacher
except ImportError:
    mujoco = None


@unittest.skipIf(mujoco is None, 'Task-local MuJoCo/Warp dependencies required')
class RagdollTeacherContract(unittest.TestCase):
    def test_reference_generation_cannot_move_simulated_root_or_change_limits(self):
        model,spec,_=build_model(noslip_iterations=0,contact_timeconst=.004)
        data=initialize(model,spec,pose='geometric')
        qpos=data.qpos.copy();qvel=data.qvel.copy();ctrl=data.ctrl.copy()
        limits=model.actuator_ctrlrange.copy();effort=model.actuator_forcerange.copy()
        masses=model.body_mass.copy();inertias=model.body_inertia.copy()
        teacher=WalkingTeacher(model,spec,data,waist_amplitude=.3,hip_roll_amplitude=-.12,foot_placement_gain=.5,waist_transfer_sharpness=2.,waist_phase_lead=np.pi/2)
        for t in np.linspace(0,10,501):
            target=teacher.targets(float(t),lateral_velocity=.5*np.sin(t))
            self.assertTrue(np.all(target>=limits[:,0]))
            self.assertTrue(np.all(target<=limits[:,1]))
        np.testing.assert_array_equal(data.qpos,qpos)
        np.testing.assert_array_equal(data.qvel,qvel)
        np.testing.assert_array_equal(data.ctrl,ctrl)
        np.testing.assert_array_equal(model.actuator_ctrlrange,limits)
        np.testing.assert_array_equal(model.actuator_forcerange,effort)
        np.testing.assert_array_equal(model.body_mass,masses)
        np.testing.assert_array_equal(model.body_inertia,inertias)
        self.assertLess(max(teacher.ik_errors.values()),1e-5)
        # Both sides receive distinct, continuous cyclic foot-placement targets.
        for side in ('left','right'):
            aids,table=teacher.tables[side]
            np.testing.assert_allclose(table[0],table[-1],atol=1e-5)
            self.assertGreater(float(np.ptp(table[:,1])),.1)


    def test_motor_pose_sequence_ignores_scratch_root_and_preserves_held_joints(self):
        import hashlib,json,tempfile
        from pathlib import Path
        from algorithms.urdf_learn_wasd_walk.mujoco_backend import OUTPUT
        from algorithms.urdf_learn_wasd_walk.mujoco_ragdoll_teacher import MotorPoseSequence
        model,spec,xml=build_model(noslip_iterations=0,contact_timeconst=.004)
        data=initialize(model,spec,pose='geometric');before=data.qpos.copy()
        aids=[model.actuator(n).id for n in spec['action_joints']]
        positions=np.tile(data.ctrl[aids],(3,1));positions[1,13]=.6
        record={'schema':1,'model_xml_sha256':hashlib.sha256(xml.encode()).hexdigest(),
                'action_joints':spec['action_joints'],'times_s':[0.,3.,6.],
                'positions_rad':positions.tolist(),'loop':True,'scratch_root_pose':[100,100,100]}
        with tempfile.TemporaryDirectory(dir=OUTPUT) as temp:
            path=Path(temp)/'reference.json';path.write_text(json.dumps(record))
            teacher=MotorPoseSequence(model,spec,data,path,record['model_xml_sha256'])
            held=np.array([i for i in range(model.nu) if i not in aids])
            np.testing.assert_array_equal(teacher.targets(3)[held],data.ctrl[held])
            self.assertAlmostEqual(teacher.targets(3)[aids[13]],.6)
            np.testing.assert_array_equal(data.qpos,before)
            record['positions_rad'][1][13]=.71;path.write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError,'original position limits'):
                MotorPoseSequence(model,spec,data,path,record['model_xml_sha256'])

    def test_loop_repeats_gait_without_replaying_startup_transfer(self):
        import hashlib,json,tempfile
        from pathlib import Path
        from algorithms.urdf_learn_wasd_walk.mujoco_backend import OUTPUT
        from algorithms.urdf_learn_wasd_walk.mujoco_ragdoll_teacher import MotorPoseSequence
        model,spec,xml=build_model(noslip_iterations=0,contact_timeconst=.004)
        data=initialize(model,spec,pose='geometric')
        aids=[model.actuator(n).id for n in spec['action_joints']]
        positions=np.tile(data.ctrl[aids],(4,1));positions[:,13]=[0.,.5,-.5,.5]
        record={'schema':1,'model_xml_sha256':hashlib.sha256(xml.encode()).hexdigest(),
                'action_joints':spec['action_joints'],'times_s':[0.,3.,6.,9.],
                'positions_rad':positions.tolist(),'loop':True,'loop_start_s':3.}
        with tempfile.TemporaryDirectory(dir=OUTPUT) as temp:
            path=Path(temp)/'reference.json';path.write_text(json.dumps(record))
            teacher=MotorPoseSequence(model,spec,data,path,record['model_xml_sha256'])
            self.assertEqual(teacher.loop_duration,6.)
            np.testing.assert_allclose(teacher.targets(10),teacher.targets(4))
            np.testing.assert_allclose(teacher.targets(15),teacher.targets(3))
            self.assertNotEqual(teacher.targets(10)[aids[13]],teacher.targets(1)[aids[13]])
            record['positions_rad'][-1][13]=.4;path.write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError,'close continuously'):
                MotorPoseSequence(model,spec,data,path,record['model_xml_sha256'])


@unittest.skipIf(mujoco is None, 'Task-local MuJoCo/Warp dependencies required')
class CurriculumAcceptanceContract(unittest.TestCase):
    def test_diagnostic_rescue_is_bounded_and_zero_stage_unchanged(self):
        from algorithms.urdf_learn_wasd_walk.mujoco_ragdoll_teacher import diagnostic_teacher_blend
        self.assertEqual(diagnostic_teacher_blend(100,0),0)
        self.assertEqual(diagnostic_teacher_blend(51,.6,52),.6)
        self.assertAlmostEqual(diagnostic_teacher_blend(52.125,.6,52),.8)
        self.assertEqual(diagnostic_teacher_blend(52.25,.6,52),1)
        values=[diagnostic_teacher_blend(t,.6,52) for t in np.linspace(50,55,100)]
        self.assertTrue(np.all(np.diff(values)>=0))
        self.assertTrue(all(.6<=v<=1 for v in values))

    def test_clearance_dips_cannot_count_one_flight_twice(self):
        from algorithms.urdf_learn_wasd_walk.mujoco_ragdoll_teacher import sustained_liftoffs
        def samples(values):
            return [{'feet':{f'{side}_{key}':value for side in ('left','right')
                    for key,value in [('contact',contact),('clearance_m',height)]}}
                    for contact,height in values]
        ground=[(True,0.)]*2;air=[(False,.01)]*3
        one_flight=ground+air+[(False,.001)]+air
        self.assertEqual(sustained_liftoffs(samples(one_flight)),{'left':1,'right':1})
        self.assertEqual(sustained_liftoffs(samples(one_flight+[(True,0.)]+air)),{'left':1,'right':1})
        self.assertEqual(sustained_liftoffs(samples(one_flight+ground+air)),{'left':2,'right':2})

    def test_zero_stage_rejects_reference_or_force_even_when_gait_metrics_pass(self):
        from algorithms.urdf_learn_wasd_walk.mujoco_ragdoll_curriculum import failures
        result={'config':{'coefficient':0.},'teacher_targets_evaluated':False,'metrics':{
            'duration_s':30.,'forward_m':1.1,'liftoffs':{'left':15,'right':15},
            'mean_contact_slip_mps':.01,'max_joint_speed_rad_s':3.,'max_effort_fraction':.1,
            'both_air_fraction':0.,'fall':False,'reset_count':0,'done_count':0,
            'nonfoot_contacts':[],'max_abs_wrench_components':[0.]*6}}
        self.assertEqual(failures(result),[])
        result['diagnostic_oracle_student']=True
        self.assertIn('oracle teacher guidance is not student curriculum evidence',failures(result))
        result.pop('diagnostic_oracle_student')
        result['diagnostic_observation_replay_sha256']='diagnostic'
        self.assertIn('reference observations are not student curriculum evidence',failures(result))
        result.pop('diagnostic_observation_replay_sha256')
        result['teacher_blend_schedule']={'kind':'diagnostic_handback'}
        self.assertIn('diagnostic rescue schedule is not curriculum evidence',failures(result))
        result.pop('teacher_blend_schedule')
        result['teacher_blend_coefficient']=1.
        self.assertIn('diagnostic teacher blend differs from curriculum coefficient',failures(result))
        result['teacher_blend_coefficient']=0.
        result['teacher_targets_evaluated']=True
        self.assertIn('zero-stage assistance notoff',failures(result))
        result['teacher_targets_evaluated']=False
        result['metrics']['max_abs_wrench_components'][2]=1e-9
        self.assertIn('zero-stage assistance notoff',failures(result))
        result['metrics']['max_abs_wrench_components'][2]=0.
        result['metrics']['done_count']=1
        self.assertIn('fall/reset/done/nonfootcontact',failures(result))


@unittest.skipIf(mujoco is None, 'Task-local dependencies required')
class DatasetAggregationContract(unittest.TestCase):
    def test_zero_development_rejects_hidden_reference_force_and_missing_ground_loads(self):
        from copy import deepcopy
        from algorithms.urdf_learn_wasd_walk.mujoco_ragdoll_zero_evidence import zero_failures
        row={'teacher_coefficient':0,'wrench':[0]*6,'teacher_target_contribution_rad':[0]*17,
             'teacher_tracking_integral_offset_rad':[0]*17,'motor_balance_requested_offset_rad':[0]*17}
        record={'teacher_targets_evaluated':False,'total_teacher_guidance_coefficient':0,
            'teacher_blend_coefficient':0,'balance_assistance_coefficient':0,'config':{},'samples':[row]*6000,
            'metrics':{'duration_s':120,'fall':False,'reset_count':0,'done_count':0,'nonfoot_contacts':[],
                'forward_m':.22,'liftoffs':{'left':5,'right':4},'max_joint_speed_rad_s':1.3,
                'max_effort_fraction':.04,'mean_contact_slip_mps':.003,'ground_load_sampling_hz':500,
                'both_feet_unloaded_physics_fraction':0,'maximum_both_feet_unloaded_duration_s':0,
                'mean_support_body_weight_ratio':1,'peak_support_body_weight_ratio':1.3}}
        wrench=np.zeros((60000,6));self.assertEqual(zero_failures(record,wrench),[])
        altered=deepcopy(record);altered['config']['motor_pose_sequence']='reference.json'
        self.assertIn('reference assistance configured',zero_failures(altered,wrench))
        wrench[123,2]=1e-12
        self.assertIn('external wrench trace is incomplete or nonzero',zero_failures(record,wrench))
        wrench[123,2]=0
        altered=deepcopy(record);altered['metrics'].pop('ground_load_sampling_hz')
        self.assertIn('500Hz ground support failure or missing evidence',zero_failures(altered,wrench))
        altered=deepcopy(record);altered['metrics']['forward_m']=float('nan')
        self.assertIn('nonfinite or missing dynamics metrics',zero_failures(altered,wrench))
        altered=deepcopy(record);altered['samples'][0]['teacher_target_contribution_rad'][0]=1e-12
        self.assertIn('logged teacher contribution or reference remains',zero_failures(altered,wrench))

    def test_clock_only_generator_cannot_read_joint_state_or_action_history(self):
        import torch
        from algorithms.urdf_learn_wasd_walk.mujoco_ragdoll_student import Student
        model=Student(64,phase_clock_only=True,temporal_harmonics=16)
        a=torch.randn(64);b=a.clone();b[:60]=torch.randn(60)*100
        torch.testing.assert_close(model(a),model(b))
        a.requires_grad_();model(a).sum().backward()
        self.assertTrue(torch.all(a.grad[:60]==0))
        self.assertGreater(float(a.grad[60:].abs().sum()),0)
        with self.assertRaises(ValueError):Student(64,residual_step_rad=.05,phase_clock_only=True)
        with self.assertRaises(ValueError):Student(64,temporal_harmonics=16)

    def test_residual_student_uses_only_previous_action_and_bounded_correction(self):
        import torch
        from algorithms.urdf_learn_wasd_walk.mujoco_ragdoll_student import Student
        model=Student(64,residual_step_rad=.05,action_scale=1.8)
        x=torch.zeros((2,64));x[:,43:60]=.2
        torch.testing.assert_close(1.8*model(x),torch.full((2,17),.1))
        with torch.no_grad():model.actor[-2].bias.fill_(100.)
        torch.testing.assert_close(1.8*model(x),torch.full((2,17),.15))
        x[:,43:60]=10.
        self.assertTrue(torch.all(model(x)==1.))
        with self.assertRaises(ValueError):Student(residual_step_rad=.2)

    def test_startup_clock_is_explicit_bounded_and_not_appended_twice(self):
        from algorithms.urdf_learn_wasd_walk.mujoco_ragdoll_student import prepare_observations
        raw=np.zeros((3001,63),dtype=np.float32)
        expanded=prepare_observations(raw,55.5)
        self.assertEqual(expanded.shape,(3001,64))
        self.assertEqual(expanded[0,-1],0.)
        self.assertEqual(expanded[-1,-1],1.)
        self.assertGreater(expanded[1650,-1],0.)
        np.testing.assert_array_equal(prepare_observations(expanded,55.5),expanded)
        with self.assertRaisesRegex(ValueError,'clock mismatch'):prepare_observations(expanded,33.)
        with self.assertRaisesRegex(ValueError,'Legacy'):prepare_observations(expanded,0.)

    def test_noise_preserves_command_phase_and_rejects_invalid_bounds(self):
        from algorithms.urdf_learn_wasd_walk.mujoco_ragdoll_student import observation_noise_bounds
        bounds=observation_noise_bounds(1.)
        self.assertEqual(bounds.shape,(63,))
        np.testing.assert_array_equal(bounds[60:],0.)
        np.testing.assert_array_equal(observation_noise_bounds(0.),0.)
        self.assertTrue(np.all(bounds[:60]>0))
        previous=observation_noise_bounds(1.,'previous_action')
        np.testing.assert_array_equal(previous[:43],0.)
        np.testing.assert_array_equal(previous[43:60],bounds[43:60])
        np.testing.assert_array_equal(previous[60:],0.)
        for scale in (-1.,float('nan'),float('inf'),3.):
            with self.assertRaises(ValueError):observation_noise_bounds(scale)

    def test_interleaved_split_keeps_startup_and_terminal_failure_states(self):
        from algorithms.urdf_learn_wasd_walk.mujoco_ragdoll_student import partition_trajectories
        x=np.arange(117).reshape(-1,1);y=x+1000
        tx,ty,vx,vy=partition_trajectories([x],[y],interleaved=True)
        self.assertTrue({0,1,2,115,116}.issubset(set(tx[:,0])))
        self.assertFalse(set(tx[:,0])&set(vx[:,0]))
        self.assertEqual(len(tx)+len(vx),117)
        np.testing.assert_array_equal(ty,tx+1000)
        np.testing.assert_array_equal(vy,vx+1000)

    def test_terminal_fall_sample_does_not_shift_expert_labels(self):
        from algorithms.urdf_learn_wasd_walk.mujoco_ragdoll_student import aligned_expert_samples
        rows=[{'time_s':.002,'label':'first'},{'time_s':.022,'label':'second'},
              {'time_s':.034,'label':'terminal'}]
        self.assertEqual([r['label'] for r in aligned_expert_samples(rows,2)],['first','second'])
        with self.assertRaisesRegex(ValueError,'timestamp alignment'):
            aligned_expert_samples([rows[0],rows[2]],2)

    def test_latest_failure_trajectory_is_not_entirely_held_out(self):
        from algorithms.urdf_learn_wasd_walk.mujoco_ragdoll_student import partition_trajectories
        trajectories=[np.full((10,2),i) for i in range(5)]
        targets=[np.full((10,1),i+10) for i in range(5)]
        tx,ty,vx,vy=partition_trajectories(trajectories,targets)
        self.assertEqual(set(tx[:,0]),set(range(5)))
        self.assertEqual(set(vx[:,0]),set(range(5)))
        np.testing.assert_array_equal(ty[:,0],tx[:,0]+10)
        np.testing.assert_array_equal(vy[:,0],vx[:,0]+10)
        self.assertEqual(len(tx),40);self.assertEqual(len(vx),10)

if __name__=='__main__':unittest.main()
