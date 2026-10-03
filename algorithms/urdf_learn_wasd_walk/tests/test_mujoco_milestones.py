import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from algorithms.urdf_learn_wasd_walk import mujoco_milestones as gates


class StandingCertificateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.folder = self.root / 'outputs/run'
        self.folder.mkdir(parents=True)
        urdf = self.root / 'inputs/landau_v10/landau_v10_parallel_mesh.urdf'
        urdf.parent.mkdir(parents=True)
        urdf.write_text('<robot><joint type="revolute"><limit velocity="4"/></joint></robot>')
        for name in ('model.xml', 'backend_source.py', 'renderer_source.py', 'proof.mp4'):
            (self.folder / name).write_text(name)
        q = np.zeros((1501, 7)); q[:, 3] = 1
        np.savez(self.folder / 'trajectory.npz', qpos=q, time=np.arange(1501)*.02)
        self.cfg = dict(assistance=0, forward=0, seconds=30., dt=.002, contact_timeconst=.004,
                        gain_scale=1., checkpoint=None)
        self.metrics = dict(duration_s=30., reset_count=0, done_count=0, fall_count=0,
            peak_auxiliary_wrench_norm=0., max_abs_command=0., max_abs_action=0., policy_calls=0,
            max_reference_tilt_rad=.01, root_height_drop_m=.001, horizontal_drift_m=.002,
            nonfoot_contacts=[], minimum_support_polygon_margin_m=.02, first_support_exit_time_s=None,
            mean_support_body_weight_ratio=1., peak_support_body_weight_ratio=1.5,
            mean_support_force_body_weight_ratio=1., peak_support_force_body_weight_ratio=1.5,
            max_joint_speed_rad_s=1.)
        identity = dict(backend='mujoco_warp_cuda', model_xml_sha256=gates.digest(self.folder/'model.xml'),
            urdf_sha256=gates.digest(urdf), mesh_tree_sha256=hashlib.sha256().hexdigest(),
            initial_control_sha256='controls', source_sha256=gates.digest(self.folder/'backend_source.py'),
            config_sha256=hashlib.sha256(json.dumps(self.cfg, sort_keys=True).encode()).hexdigest())
        self.record = dict(milestone=gates.STANDING[0], status='dynamics_passed_proof_pending', failures=[],
            identity=identity, config=self.cfg, metrics=self.metrics,
            trajectory_sha256=gates.digest(self.folder/'trajectory.npz'),
            performance={'warp_setup': {'physics_steps':15000}})
        self.ledger = dict(backend='mujoco_warp_cuda', assetContract=dict(
            urdfSha256=identity['urdf_sha256'], meshTreeSha256=identity['mesh_tree_sha256'],
            modelXmlSha256=identity['model_xml_sha256'], initialControlSha256='controls', physicsIdentity={}),
            acceptance={'standing_max_drift_m':.03, 'standing_max_heading_change_deg':5.})
        self.addCleanup(patch.stopall)
        patch.object(gates, 'ALG', self.root).start()
        patch.object(gates.subprocess, 'check_output', return_value=b'{"streams":[{"nb_frames":"1501","duration":"30.02"}]}').start()
        self.save()

    def save(self):
        gates.write(self.folder/'dynamics.json', self.record)
        proof = {key:gates.digest(self.folder/name) for key,name in (
            ('model_xml_sha256','model.xml'),('dynamics_sha256','dynamics.json'),
            ('trajectory_sha256','trajectory.npz'),('video_sha256','proof.mp4'),
            ('renderer_source_sha256','renderer_source.py'))}
        proof.update(kind='state_replay_of_exact_dynamics_trajectory', character_visible_every_frame=True, frames=1501)
        gates.write(self.folder/'proof_metadata.json', proof)
        gates.write(self.folder/'visual_review.json', dict(decision='accepted',
            video_sha256=proof['video_sha256'], trajectory_sha256=proof['trajectory_sha256']))

    def test_complete_passive_certificate(self):
        gates.check_standing(self.folder, self.ledger)

    def test_different_replay_model_cannot_use_claimed_identity(self):
        (self.folder/'model.xml').write_text('different model')
        self.save()
        with self.assertRaisesRegex(ValueError, 'Replay model'):
            gates.check_standing(self.folder, self.ledger)

    def test_empty_failure_list_cannot_hide_fall_or_tilt(self):
        for field,value in [('fall_count',1),('max_reference_tilt_rad',1.),('first_support_exit_time_s',2.)]:
            original = self.metrics[field]; self.metrics[field] = value; self.save()
            with self.subTest(field=field), self.assertRaises(ValueError):
                gates.check_standing(self.folder, self.ledger)
            self.metrics[field] = original

    def test_assistance_is_rejected(self):
        self.metrics['peak_auxiliary_wrench_norm'] = 1.; self.save()
        with self.assertRaisesRegex(ValueError, 'peak_auxiliary'):
            gates.check_standing(self.folder, self.ledger)

    def test_corrupt_video_is_rejected(self):
        (self.folder/'proof.mp4').write_text('changed')
        with self.assertRaisesRegex(ValueError, 'Proof hash'):
            gates.check_standing(self.folder, self.ledger)

    def test_clock_reset_is_rejected(self):
        p=self.folder/'trajectory.npz'; trace=np.load(p); q=trace['qpos']; times=trace['time']; times[700]=0
        np.savez(p,qpos=q,time=times); self.record['trajectory_sha256']=gates.digest(p); self.save()
        with self.assertRaisesRegex(ValueError,'Reset or gap'):
            gates.check_standing(self.folder,self.ledger)


class WalkingCertificateTests(StandingCertificateTests):
    def direction_fixture(self):
        from algorithms.urdf_learn_wasd_walk import landau_direction_contract as dc
        cp=self.teleop_fixture()
        self.cfg.update(teleop=False,direction='forward',target_distance_m=10.)
        self.record['milestone']='gate_10m_four_directions_no_reset'
        self.record['identity']['config_sha256']=hashlib.sha256(json.dumps(self.cfg,sort_keys=True).encode()).hexdigest()
        t=np.arange(3001)*.02;q=np.zeros((3001,7));q[:,3]=1.;q[:,1]=np.linspace(0.,11.,3001)
        np.savez(self.folder/'trajectory.npz',qpos=q,time=t)
        self.record['trajectory_sha256']=gates.digest(self.folder/'trajectory.npz')
        obs=np.zeros((3000,70));obs[:,63]=.2
        np.savez(self.folder/'policy_trace.npz',observation=obs,time=t[:-1])
        self.record['policy_trace_sha256']=gates.digest(self.folder/'policy_trace.npz')
        (self.folder/'direction_protocol_source.py').write_text(Path(dc.__file__).read_text())
        self.record['direction_protocol_source_sha256']=gates.digest(dc.__file__)
        training=gates.read(self.folder/'training.json');training['nominal_q']=q[0].tolist()
        gates.write(self.folder/'training.json',training)
        control_hash=hashlib.sha256(json.dumps({'initial_qpos':training['nominal_q'],'initial_ctrl':training['nominal_ctrl']},sort_keys=True).encode()).hexdigest()
        self.ledger['assetContract']['initialControlSha256']=control_hash
        self.record['identity']['initial_control_sha256']=control_hash
        self.metrics.update(dc.gate_metrics('forward',t,q[:,:3]),control_steps=3000,
            left_foot_liftoff_count=100,right_foot_liftoff_count=100,mean_contact_foot_slip_mps=.01)
        self.teleop_save()
        return cp

    def test_direction_reconstructs_gate_and_commands(self):
        cp=self.direction_fixture()
        gates.check_walking(self.folder,self.ledger,cp,10.,direction='forward')
        with np.load(self.folder/'policy_trace.npz') as trace:obs=trace['observation'];t=trace['time']
        obs[:,64]=.2
        np.savez(self.folder/'policy_trace.npz',observation=obs,time=t)
        self.record['policy_trace_sha256']=gates.digest(self.folder/'policy_trace.npz');self.teleop_save()
        with self.assertRaisesRegex(ValueError,'declared joystick commands'):
            gates.check_walking(self.folder,self.ledger,cp,10.,direction='forward')

    def test_direction_claim_cannot_hide_short_trajectory(self):
        cp=self.direction_fixture()
        with np.load(self.folder/'trajectory.npz') as trace:q=trace['qpos'];t=trace['time']
        q[:,1]*=.5
        np.savez(self.folder/'trajectory.npz',qpos=q,time=t)
        self.record['trajectory_sha256']=gates.digest(self.folder/'trajectory.npz');self.teleop_save()
        with self.assertRaisesRegex(ValueError,'crossing differs|metric differs'):
            gates.check_walking(self.folder,self.ledger,cp,10.,direction='forward')

    def test_direction_promotion_requires_four_distinct_runs_and_six_predecessors(self):
        checkpoint=self.folder/'candidate.pt';checkpoint.write_bytes(b'candidate')
        ledger=dict(lineage=gates.LINEAGE,milestones=[{'status':'passed'} for _ in range(6)]+
                    [{'id':'gate_10m_four_directions_no_reset','status':'in_progress'}])
        path=self.root/'milestones.json';gates.write(path,ledger)
        with patch.object(gates,'LEDGER',path):
            with self.assertRaisesRegex(ValueError,'Four independent'):
                gates.certify_directions(self.root/'outputs/certificate',checkpoint,[self.folder]*4,[self.folder]*6)
            with self.assertRaisesRegex(ValueError,'M1–M6 components'):
                gates.certify_directions(self.root/'outputs/certificate',checkpoint,
                    [self.root/f'outputs/{i}' for i in range(4)],[self.folder]*5)
            ledger['milestones'][5]['status']='in_progress';gates.write(path,ledger)
            with self.assertRaisesRegex(ValueError,'Prior milestone'):
                gates.certify_directions(self.root/'outputs/certificate',checkpoint,[],[])
        self.assertFalse((self.root/'outputs/certificate').exists())

    def turn_fixture(self):
        cp,_=self.walking_fixture(26.)
        self.cfg.update(turn=True,turn_hold_start=19.)
        self.record['milestone']='yaw_turn_90deg_hold'
        self.record['identity']['config_sha256']=hashlib.sha256(json.dumps(self.cfg,sort_keys=True).encode()).hexdigest()
        self.metrics.update(hold_max_heading_error_rad=.01,hold_max_drift_m=.01,
            hold_max_horizontal_speed_mps=.01,hold_max_yaw_speed_rad_s=.01,hold_duration_s=5.,
            hold_max_abs_command=0.,simultaneous_air_fraction=0.,policy_inference_steps=1300,control_steps=1300)
        training=gates.read(self.folder/'training.json')
        for name in ('turn_source.py','turn_validator_source.py'):(self.folder/name).write_text('saved turn source')
        training.update(command_extension='yaw_v1',turn_source_sha256=gates.digest(self.folder/'turn_source.py'),
            command_profile={'turn_start_s':3.,'turn_end_s':17.,'hold_start_s':19.},policy_family='periodic_feedback_cem')
        (self.folder/'gait_source.py').write_text('frozen gait')
        self.record['gait_source_sha256']=gates.digest(self.folder/'gait_source.py')
        training['source_sha256']={'landau_gait_search.py':self.record['gait_source_sha256']}
        gates.write(self.folder/'training.json',training)
        self.record['turn_validator_source_sha256']=gates.digest(self.folder/'turn_validator_source.py')
        t=np.arange(1301)*.02;heading=np.minimum(t/17.,1.)*np.pi/2
        q=np.zeros((1301,7));q[:,3]=np.cos(heading/2);q[:,6]=np.sin(heading/2)
        q[:,1]=np.minimum(t/19.,1.)*2.
        np.savez(self.folder/'trajectory.npz',qpos=q,time=t)
        self.record['trajectory_sha256']=gates.digest(self.folder/'trajectory.npz')
        obs=np.zeros((1300,70));obs[(t[:-1]>=3)&(t[:-1]<17),65]=np.pi/28
        np.savez(self.folder/'policy_trace.npz',observation=obs,time=t[:-1])
        self.record['policy_trace_sha256']=gates.digest(self.folder/'policy_trace.npz')
        gates.write(self.folder/'controller_memory.json',dict(simulation_reset=False,anchor_time=19.,
            turn_source_sha256=training['turn_source_sha256'],dispatch='command_driven_yaw_extension',
            turned=True,integrated_reference_rad=np.pi/2))
        self.record.update(turn_source_sha256=training['turn_source_sha256'],
            controller_memory_sha256=gates.digest(self.folder/'controller_memory.json'))
        self.save();proof=gates.read(self.folder/'proof_metadata.json');proof['frames']=1301
        gates.write(self.folder/'proof_metadata.json',proof)
        return cp

    def test_turn_requires_actual_trajectory_rotation(self):
        cp=self.turn_fixture();gates.check_walking(self.folder,self.ledger,cp,turn=True)
        path=self.folder/'trajectory.npz';trace=np.load(path);q=trace['qpos'];t=trace['time']
        q[:,3]=1.;q[:,6]=0.
        np.savez(path,qpos=q,time=t);self.record['trajectory_sha256']=gates.digest(path)
        self.save();proof=gates.read(self.folder/'proof_metadata.json');proof['frames']=1301
        gates.write(self.folder/'proof_metadata.json',proof)
        with self.assertRaisesRegex(ValueError,'90 degree hold'):
            gates.check_walking(self.folder,self.ledger,cp,turn=True)

    def test_turn_command_trace_and_saved_source_are_required(self):
        cp=self.turn_fixture()
        path=self.folder/'policy_trace.npz';trace=np.load(path);obs=trace['observation'];t=trace['time'];obs[:,65]=0.
        np.savez(path,observation=obs,time=t);self.record['policy_trace_sha256']=gates.digest(path)
        self.save();proof=gates.read(self.folder/'proof_metadata.json');proof['frames']=1301
        gates.write(self.folder/'proof_metadata.json',proof)
        with self.assertRaisesRegex(ValueError,'integrate to 90'):
            gates.check_walking(self.folder,self.ledger,cp,turn=True)

    def test_extra_full_rotation_cannot_pass_quarter_turn(self):
        cp=self.turn_fixture()
        path=self.folder/'trajectory.npz';trace=np.load(path);q=trace['qpos'];t=trace['time']
        heading=np.minimum(t/17.,1.)*2.5*np.pi
        q[:,3]=np.cos(heading/2);q[:,6]=np.sin(heading/2)
        np.savez(path,qpos=q,time=t);self.record['trajectory_sha256']=gates.digest(path)
        self.save();proof=gates.read(self.folder/'proof_metadata.json');proof['frames']=1301
        gates.write(self.folder/'proof_metadata.json',proof)
        with self.assertRaisesRegex(ValueError,'90 degree hold'):
            gates.check_walking(self.folder,self.ledger,cp,turn=True)
        (self.folder/'turn_source.py').write_text('changed')
        with self.assertRaisesRegex(ValueError,'turn source'):
            gates.check_walking(self.folder,self.ledger,cp,turn=True)

    def teleop_fixture(self):
        from algorithms.urdf_learn_wasd_walk import landau_teleop_contract as tc
        cp,feet=self.walking_fixture(60.)
        xml='<mujoco><worldbody><body name="base_link"><freejoint/><geom type="sphere" size=".1" mass="1"/><body name="root_x"/></body></worldbody></mujoco>'
        (self.folder/'model.xml').write_text(xml)
        self.record['identity']['model_xml_sha256']=gates.digest(self.folder/'model.xml')
        self.ledger['assetContract']['modelXmlSha256']=gates.digest(self.folder/'model.xml')
        self.cfg.update(teleop=True,turn=False)
        self.record['milestone']='teleop_60s_forward_turn'
        self.record['identity']['config_sha256']=hashlib.sha256(json.dumps(self.cfg,sort_keys=True).encode()).hexdigest()
        training=gates.read(self.folder/'training.json')
        training.update(model_xml_sha256=gates.digest(self.folder/'model.xml'),command_extension='yaw_v1',memory_version=2,policy_family='periodic_feedback_cem')
        for name in ('turn_source.py','teleop_validator_source.py','command_protocol_source.py','gait_source.py'):
            (self.folder/name).write_text('snapshot '+name)
        (self.folder/'teleop_contract.py').write_text((self.folder/'command_protocol_source.py').read_text())
        training.update(turn_source_sha256=gates.digest(self.folder/'turn_source.py'),teleop_contract_sha256=gates.digest(self.folder/'teleop_contract.py'),source_sha256={'landau_gait_search.py':gates.digest(self.folder/'gait_source.py')})
        gates.write(self.folder/'training.json',training)
        t=np.arange(3001)*.02;commands=np.array([tc.command_profile(v) for v in t[:-1]])
        yaw=np.r_[0.,np.cumsum(commands[:,2]*.02)]
        xy=np.zeros((3001,2));xy[1:]=np.cumsum(np.c_[-np.sin(yaw[:-1]),np.cos(yaw[:-1])]*commands[:,0,None]*.01,axis=0)
        q=np.zeros((3001,7));q[:,:2]=xy;q[:,3]=np.cos(yaw/2);q[:,6]=np.sin(yaw/2)
        np.savez(self.folder/'trajectory.npz',qpos=q,time=t)
        obs=np.zeros((3000,70));obs[:,63:66]=commands
        np.savez(self.folder/'policy_trace.npz',observation=obs,time=t[:-1])
        self.record.update(trajectory_sha256=gates.digest(self.folder/'trajectory.npz'),policy_trace_sha256=gates.digest(self.folder/'policy_trace.npz'),
            gait_source_sha256=gates.digest(self.folder/'gait_source.py'),turn_source_sha256=training['turn_source_sha256'],
            command_protocol_source_sha256=training['teleop_contract_sha256'],teleop_validator_source_sha256=gates.digest(self.folder/'teleop_validator_source.py'))
        gates.write(self.folder/'controller_memory.json',dict(simulation_reset=False,memory_version=2,turned=True,
            integrated_reference_rad=0.,dispatch='command_driven_yaw_extension',turn_source_sha256=training['turn_source_sha256'],
            events=[dict(kind='stop_anchor',time_s=20.),dict(kind='restart',time_s=25.02),dict(kind='stop_anchor',time_s=50.)]))
        self.record['controller_memory_sha256']=gates.digest(self.folder/'controller_memory.json')
        self.metrics.update(policy_inference_steps=3000,simultaneous_air_fraction=0.,
            teleop_response=tc.response_metrics(t,q[:,:3],yaw,feet))
        self.teleop_save()
        return cp

    def teleop_save(self):
        self.save();proof=gates.read(self.folder/'proof_metadata.json');proof['frames']=3001
        gates.write(self.folder/'proof_metadata.json',proof)

    def test_teleop_requires_each_command_and_reconstructed_response(self):
        cp=self.teleop_fixture();gates.check_walking(self.folder,self.ledger,cp,teleop=True)
        path=self.folder/'policy_trace.npz';trace=np.load(path);obs=trace['observation'];t=trace['time']
        obs[obs[:,65]<0,65]=0.;np.savez(path,observation=obs,time=t)
        self.record['policy_trace_sha256']=gates.digest(path);self.teleop_save()
        with self.assertRaisesRegex(ValueError,'declared joystick commands'):
            gates.check_walking(self.folder,self.ledger,cp,teleop=True)

    def test_teleop_forged_success_metrics_do_not_hide_missing_turn(self):
        cp=self.teleop_fixture();path=self.folder/'trajectory.npz';trace=np.load(path);q=trace['qpos'];t=trace['time']
        q[:,3]=1.;q[:,6]=0.;np.savez(path,qpos=q,time=t)
        self.record['trajectory_sha256']=gates.digest(path);self.teleop_save()
        with self.assertRaisesRegex(ValueError,'reconstructed response'):
            gates.check_walking(self.folder,self.ledger,cp,teleop=True)

    def test_teleop_fk_speed_quantization_is_bounded(self):
        cp=self.teleop_fixture()
        hold=self.metrics['teleop_response']['holds'][1]
        hold['settled_speed_mps']+=5e-5;self.teleop_save()
        gates.check_walking(self.folder,self.ledger,cp,teleop=True)
        hold['settled_speed_mps']+=.001;self.teleop_save()
        with self.assertRaisesRegex(ValueError,'hold differs from trajectory'):
            gates.check_walking(self.folder,self.ledger,cp,teleop=True)

    def test_teleop_requires_second_stop_memory(self):
        cp=self.teleop_fixture();path=self.folder/'controller_memory.json';memory=gates.read(path)
        memory['events'].pop();gates.write(path,memory)
        self.record['controller_memory_sha256']=gates.digest(path);self.teleop_save()
        with self.assertRaisesRegex(ValueError,'repeated-stop/restart'):
            gates.check_walking(self.folder,self.ledger,cp,teleop=True)

    def walking_fixture(self, seconds=30., distance=5.):
        steps=round(seconds/.002); frames=round(seconds/.02)+1
        self.cfg['seconds']=seconds
        self.metrics['duration_s']=seconds
        self.record['performance']['warp_setup']['physics_steps']=steps
        checkpoint=self.folder/'model_100.pt';checkpoint.write_bytes(b'trained')
        controls={'initial_qpos':[0.]*7,'initial_ctrl':[0.]}
        control_hash=hashlib.sha256(json.dumps(controls,sort_keys=True).encode()).hexdigest()
        self.ledger['assetContract']['initialControlSha256']=control_hash
        self.record['identity']['initial_control_sha256']=control_hash
        self.record['milestone']=f'gate_{distance:g}m_no_reset'
        self.cfg['target_distance_m']=distance
        self.cfg.update(checkpoint=str(checkpoint),checkpoint_sha256=gates.digest(checkpoint),forward=.2)
        self.record['identity']['config_sha256']=hashlib.sha256(json.dumps(self.cfg,sort_keys=True).encode()).hexdigest()
        self.metrics.update(semantic_forward_displacement_m=distance+.25,semantic_strafe_displacement_m=.01,
            policy_calls=round(seconds/.02),max_abs_command=.2,max_abs_action=.5,
            leg_joint_excursion_rad={k:.2 for k in ('left_hip_pitch_joint','right_hip_pitch_joint','left_knee_joint','right_knee_joint')})
        (self.folder/'controller_source.py').write_text('controller')
        self.record['controller_source_sha256']=gates.digest(self.folder/'controller_source.py')
        training={k:self.record['identity'][k] for k in ('model_xml_sha256','urdf_sha256','mesh_tree_sha256','backend')}
        self.record['identity']['mujoco_version']='test';training['mujoco_version']='test'
        training.update(nominal_q=controls['initial_qpos'],nominal_ctrl=controls['initial_ctrl'],checkpoints={checkpoint.name:gates.digest(checkpoint)})
        gates.write(self.folder/'training.json',training)
        feet=[]; counts={'left':0,'right':0}; was_air={'left':False,'right':False}
        for i in range(steps):
            phase=i%300; left_air=25<=phase<125;right_air=175<=phase<275
            for side,air in [('left',left_air),('right',right_air)]:
                counts[side]+=int(was_air[side] and not air);was_air[side]=air
            feet.append(dict(time_s=(i+1)*.002,left_contact=not left_air,right_contact=not right_air,
                left_clearance_m=.02 if left_air else 0.,right_clearance_m=.02 if right_air else 0.,mean_slip_mps=.01))
        self.metrics.update({k+'_completed_swings':v for k,v in counts.items()})
        gates.write(self.folder/'foot_trace.json',feet)
        self.record['foot_trace_sha256']=gates.digest(self.folder/'foot_trace.json')
        q=np.zeros((frames,7));q[:,3]=1;q[:,1]=np.linspace(0,distance+.25,frames)
        np.savez(self.folder/'trajectory.npz',qpos=q,time=np.arange(frames)*.02)
        self.record['trajectory_sha256']=gates.digest(self.folder/'trajectory.npz');self.save()
        proof=gates.read(self.folder/'proof_metadata.json');proof['frames']=frames
        gates.write(self.folder/'proof_metadata.json',proof)
        patch.object(gates.subprocess,'check_output',return_value=json.dumps({'streams':[{'nb_frames':str(frames),'duration':str(seconds+.02)}]}).encode()).start()
        return checkpoint,feet

    def test_completed_walking_protocol(self):
        cp,_=self.walking_fixture();gates.check_walking(self.folder,self.ledger,cp)

    def test_longer_walking_run_preserves_full_trace_requirement(self):
        cp,feet=self.walking_fixture(45.)
        gates.check_walking(self.folder,self.ledger,cp)
        gates.write(self.folder/'foot_trace.json',feet[:15000])
        self.record['foot_trace_sha256']=gates.digest(self.folder/'foot_trace.json');self.save()
        with self.assertRaisesRegex(ValueError,'Incomplete foot trace'):
            gates.check_walking(self.folder,self.ledger,cp)

    def test_ten_metre_gate_requires_full_distance(self):
        cp,_=self.walking_fixture(80.,10.)
        gates.check_walking(self.folder,self.ledger,cp,10.)
        self.metrics['semantic_forward_displacement_m']=6.5;self.save()
        with self.assertRaisesRegex(ValueError,'reach 10 m'):
            gates.check_walking(self.folder,self.ledger,cp,10.)

    def test_claimed_swings_cannot_replace_foot_trace(self):
        cp,feet=self.walking_fixture()
        for row in feet:row.update(left_contact=True,left_clearance_m=0.)
        gates.write(self.folder/'foot_trace.json',feet)
        self.record['foot_trace_sha256']=gates.digest(self.folder/'foot_trace.json');self.save()
        with self.assertRaisesRegex(ValueError,'left swings'):
            gates.check_walking(self.folder,self.ledger,cp)

    def test_sliding_does_not_pass_by_distance(self):
        cp,feet=self.walking_fixture()
        for row in feet:row['mean_slip_mps']=.2
        gates.write(self.folder/'foot_trace.json',feet)
        self.record['foot_trace_sha256']=gates.digest(self.folder/'foot_trace.json');self.save()
        with self.assertRaisesRegex(ValueError,'Sliding'):
            gates.check_walking(self.folder,self.ledger,cp)

    def test_corrupted_foot_trace_is_rejected(self):
        cp,_=self.walking_fixture();(self.folder/'foot_trace.json').write_text('[]')
        with self.assertRaisesRegex(ValueError,'foot trace mismatch'):
            gates.check_walking(self.folder,self.ledger,cp)


if __name__ == '__main__':
    unittest.main()
