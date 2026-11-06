import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch
import torch
from algorithms.urdf_learn_wasd_walk import leg_residual_training as residual
from algorithms.urdf_learn_wasd_walk.model_spec import ACTION_JOINTS


class ResidualContractTests(unittest.TestCase):
    def test_phase_retarget_cubic_matches_periodic_reference(self):
        import numpy as np
        from scipy.interpolate import PchipInterpolator
        from algorithms.urdf_learn_wasd_walk.phase_retarget_probe import PhaseRetarget
        phase=(np.arange(32)+.5)/32
        values=np.stack((np.sin(2*np.pi*phase),np.cos(2*np.pi*phase)),axis=1)
        curve=PchipInterpolator(np.r_[phase-1,phase,phase+1],np.tile(values,(3,1)))
        probe=PhaseRetarget.__new__(PhaseRetarget)
        probe.knots=curve.x;probe.coefficients=curve.c
        points=torch.linspace(-1,2,3001,dtype=torch.float64)
        np.testing.assert_allclose(probe.sample(points).numpy(),curve(points.numpy()%1),atol=1e-12)
        np.testing.assert_allclose(probe.sample(points,derivative=True).numpy(),curve(points.numpy()%1,1),atol=1e-12)
        from algorithms.urdf_learn_wasd_walk import mujoco_milestones as m
        for result in ({'diagnostic_phase_template':{}},{'config':{'diagnostic_phase_template':{}}}):
            with self.assertRaisesRegex(ValueError,'cannot be certified'):m.check_certifiable_configuration(result)

    def test_touchdown_quality_preserves_slow_landings_without_rewarding_hops(self):
        velocity=torch.tensor([-.022901,-.046302,-.05,-.185928,-.3,0.,.1])
        quality=residual.touchdown_quality(velocity)
        torch.testing.assert_close(quality[:3],torch.ones(3))
        self.assertAlmostEqual(float(quality[3]),.119182,places=5)
        self.assertLess(float(quality[4]),float(quality[3]))
        contact=torch.tensor([[True,True],[True,False]])
        qualified,_=residual.supported_swing_landing(contact,torch.ones_like(contact),torch.zeros_like(contact))
        self.assertEqual(float((qualified*residual.touchdown_quality(torch.zeros(2,2))).sum()),0.)

    def test_intermediate_curriculum_inspection_cannot_certify(self):
        from algorithms.urdf_learn_wasd_walk import mujoco_milestones as m
        m.check_certifiable_configuration({'config':{}})
        for level in (0.,.2,1.):
            for result in ({'diagnostic_curriculum_level':level},{'config':{'diagnostic_curriculum_level':level}}):
                with self.assertRaisesRegex(ValueError,'cannot be certified'):m.check_certifiable_configuration(result)

    def test_curriculum_provenance_rejects_changed_start_and_target(self):
        from copy import deepcopy
        from algorithms.urdf_learn_wasd_walk import mujoco_milestones as m, leg_gait_search as gait
        with tempfile.TemporaryDirectory() as directory, patch.object(m,'ALG',Path(directory)):
            start=Path(directory)/'outputs'/'start'/'model.pt';start.parent.mkdir(parents=True)
            end=start.parent.parent/'target.pt'
            torch.save({'parameters':torch.arange(22,dtype=torch.float32)},start)
            torch.save({'parameters':torch.arange(23,dtype=torch.float32)},end)
            source=start.parent/'control_source.py';source.write_text('# test snapshot\n')
            identity={k:[] for k in ('model_xml_sha256','action_joints','nominal_q','nominal_ctrl','support_profile','standing_checkpoint_sha256')}
            metadata={**identity,'policy_family':'anatomical_periodic_feedback_cem','checkpoints':{'model.pt':m.digest(start)}}
            m.write(start.parent/'training.json',metadata)
            curriculum=dict(checkpoint=str(start),sha256=m.digest(start),training_sha256=m.digest(start.parent/'training.json'),
                source_sha256=m.digest(source),start_parameters=list(range(22))+[0],target_parameters=list(range(23)),
                start_speed_mps=.006,target_speed_mps=.1,inherited_milestone_passes=0)
            training=dict(whole_body=True,gait_conditioning='curriculum_level_v1',timing_contract='anatomical_schedule_v2',
                parent_checkpoint=str(end),gait_parameters=list(range(23)),gait_curriculum=curriculum,
                arguments=dict(curriculum_start_checkpoint=str(start),curriculum_start_speed=.006,target_speed=.1))
            with patch('algorithms.urdf_learn_wasd_walk.landau_forward_control.load_gait_source',return_value=(gait,source)):
                m.check_gait_curriculum(training,identity)
                for field,value in [('sha256','changed'),('source_sha256','changed'),('training_sha256','changed'),
                                    ('start_parameters',list(range(23))),('target_parameters',[0]*23),('inherited_milestone_passes',1)]:
                    changed=deepcopy(training);changed['gait_curriculum'][field]=value
                    with self.subTest(field=field),self.assertRaises(ValueError):m.check_gait_curriculum(changed,identity)
                changed=deepcopy(training);changed['whole_body']=False
                with self.assertRaisesRegex(ValueError,'semantics'):m.check_gait_curriculum(changed,identity)

    def test_per_candidate_period_matches_independent_actor_features(self):
        obs=torch.zeros(3,123);obs[:,110]=1-torch.tensor([6.,10.,20.])/30
        periods=torch.tensor([2.,6.,14.])
        batched=residual.features(obs,periods)
        separate=torch.cat([residual.features(obs[i:i+1],periods[i]) for i in range(3)])
        torch.testing.assert_close(batched,separate,rtol=0,atol=0)
        self.assertFalse(torch.equal(batched[0],residual.features(obs[:1],14.)[0]))

    def test_long_episode_clock_uses_exact_float32_encoding(self):
        import numpy as np
        from algorithms.urdf_learn_wasd_walk import mujoco_milestones as m
        times=np.arange(120000)*.02
        values=(1-times/30).astype(np.float32)
        self.assertGreater(np.max(abs(values-(1-times/30))),1e-6)
        m.check_float32_policy_clock(values,times)
        values[-1]=np.nextafter(values[-1],np.float32(0))
        with self.assertRaisesRegex(ValueError,'clock changed'):
            m.check_float32_policy_clock(values,times)

    def test_initial_export_requires_exact_trained_ancestry(self):
        from algorithms.urdf_learn_wasd_walk import mujoco_milestones as m
        with tempfile.TemporaryDirectory() as directory, patch.object(m,'ALG',Path(directory)):
            root=Path(directory)/'outputs';root.mkdir()
            donor=root/'donor';donor.mkdir();child=root/'child';child.mkdir()
            weights={'actor_state_dict':{'weight':torch.tensor([1.])},'critic_state_dict':{'weight':torch.tensor([2.])}}
            torch.save(weights,donor/'model_1.pt');torch.save(weights,child/'model_initial.pt')
            (donor/'residual_source.py').write_text('# frozen source\n')
            metadata={'policy_family':'anatomical_feedback_residual_ppo','status':'completed',
                      **{k:[] for k in ('timing_contract','action_joints','model_xml_sha256','nominal_q','nominal_ctrl','residual_feature_dim')},
                      'residual_source_sha256':m.digest(donor/'residual_source.py'),
                      'checkpoints':{'model_1.pt':m.digest(donor/'model_1.pt')}}
            m.write(donor/'training.json',metadata)
            child_meta={**metadata,'checkpoints':{'model_initial.pt':m.digest(child/'model_initial.pt')}}
            with self.assertRaisesRegex(ValueError,'no trained weight ancestry'):
                m.check_trained_checkpoint(child/'model_initial.pt',child_meta)
            child_meta['residual_weight_parent']={'checkpoint':str(donor/'model_1.pt'),'sha256':m.digest(donor/'model_1.pt'),
                'training_sha256':m.digest(donor/'training.json'),'source_sha256':metadata['residual_source_sha256']}
            m.check_trained_checkpoint(child/'model_initial.pt',child_meta)
            weights['actor_state_dict']['weight']+=1
            torch.save(weights,child/'model_initial.pt');child_meta['checkpoints']['model_initial.pt']=m.digest(child/'model_initial.pt')
            with self.assertRaisesRegex(ValueError,'weights differ'):
                m.check_trained_checkpoint(child/'model_initial.pt',child_meta)
            child_meta['residual_weight_parent']['training_sha256']='changed'
            with self.assertRaisesRegex(ValueError,'hash mismatch'):
                m.check_trained_checkpoint(child/'model_initial.pt',child_meta)

    def setUp(self):
        self.names=[*ACTION_JOINTS,'left_ankle_roll_joint','right_ankle_roll_joint']
        self.obs=torch.zeros(2,81);self.obs[:,8]=-1;self.obs[:,68]=1-4/30;self.obs[:,69]=-.2
        self.base=torch.linspace(-5,5,19)[None].expand(2,-1)

    def test_speed_curriculum_preserves_slow_reward_and_has_high_speed_signal(self):
        velocity=torch.tensor([[0.,-.01,0.],[.02,-.03,0.]])
        ramp=torch.tensor([.5,1.])
        expected=(-(velocity[:,1]+.03*ramp).square()/.0025-velocity[:,0].square()/.0025).exp()
        torch.testing.assert_close(residual.velocity_tracking_score(velocity,.03,ramp),expected)
        for target in (.1,.4,.8,1.2):
            velocity=torch.tensor([[0.,0.,0.],[0.,-target/2,0.],[0.,-target,0.],[target,-target,0.]],requires_grad=True)
            score=residual.velocity_tracking_score(velocity,target,torch.ones(4))
            self.assertTrue(0<float(score[0])<float(score[1])<float(score[2]))
            self.assertLess(float(score[3]),float(score[2]))
            score[0].backward();self.assertLess(float(velocity.grad[0,1]),0.)

    def test_per_world_curriculum_preserves_endpoints_and_schedule(self):
        from algorithms.urdf_learn_wasd_walk.leg_gait_search import REFERENCE_PARAMETERS
        start=torch.tensor(REFERENCE_PARAMETERS);target=start.clone();target[0]=3.5;target[2]=-.22;target[15]=1.5;target[22]=1.
        levels=torch.tensor([0.,.4,1.]);p=residual.curriculum_parameters(start,target,levels)
        torch.testing.assert_close(p[0],start,rtol=0,atol=0)
        torch.testing.assert_close(p[-1],target,atol=1e-6,rtol=0)
        age=torch.tensor([3.,6.,8.])
        batched=residual.reference_targets(age,p)
        for i in range(3):
            expected=residual.reference_targets(age[i:i+1],p[i])
            for actual,wanted in zip(batched,expected):torch.testing.assert_close(actual[i:i+1],wanted,rtol=0,atol=0)
        velocity=torch.zeros(3,3);targets=torch.tensor([.006,.04,.1]);ramp=torch.ones(3)
        batched=residual.velocity_tracking_score(velocity,targets,ramp)
        for i in range(3):torch.testing.assert_close(batched[i:i+1],residual.velocity_tracking_score(velocity[i:i+1],targets[i],ramp[i:i+1]))

    def test_curriculum_requires_survival_both_supported_feet_and_progress(self):
        self.assertAlmostEqual(residual.next_curriculum_level(.2,False,[2,2],.1),.25)
        for fallen,swings,forward in [(True,[2,2],.1),(False,[2,1],.1),(False,[2,2],-.1)]:
            self.assertAlmostEqual(residual.next_curriculum_level(.2,fallen,swings,forward),.175)
        self.assertEqual(residual.next_curriculum_level(0.,True,[0,0],0.),0.)
        self.assertEqual(residual.next_curriculum_level(1.,False,[2,2],.1),1.)

    def test_observed_curriculum_input_preserves_donor_and_rejects_modified_graft(self):
        from tensordict import TensorDict
        from algorithms.urdf_learn_wasd_walk import mujoco_milestones as m
        old=residual.make_actor(num_actions=33);new=residual.make_actor(num_actions=33,conditioned=True)
        residual.load_transferred_weights(new,old.state_dict(),True)
        inputs=torch.randn(8,122)
        with torch.no_grad():
            expected=old(TensorDict({'actor':inputs},batch_size=[8]))
            actual=new(TensorDict({'actor':torch.cat((inputs,torch.rand(8,1)),dim=1)},batch_size=[8]))
        torch.testing.assert_close(actual,expected,atol=1e-6,rtol=0)
        prior={key:old.state_dict() for key in ('actor_state_dict','critic_state_dict')}
        expanded={key:{k:v.clone() for k,v in new.state_dict().items()} for key in prior}
        m.check_transferred_weights(expanded,prior,True,'bad graft')
        column=next(k for k,v in expanded['actor_state_dict'].items() if v.ndim==2 and v.shape[1]==123)
        expanded['actor_state_dict'][column][0,-1]=.01
        with self.assertRaisesRegex(ValueError,'bad graft'):m.check_transferred_weights(expanded,prior,True,'bad graft')
        raw=torch.zeros(2,123);value=residual.conditioned_features(raw,10.,torch.tensor([0.,1.]))
        self.assertEqual(value.shape,(2,123));self.assertEqual(value[:, -1].tolist(),[0.,1.])

    def test_episode_metrics_count_falls_despite_successful_foot_swings(self):
        records=[{'fallen':True,'swings':[2,2],'age_s':12.,'forward_m':-.1},
                 {'fallen':False,'swings':[2,1],'age_s':30.,'forward_m':.2}]
        metrics=residual.recent_episode_metrics(records)
        self.assertEqual(metrics['recent_episode_survival_fraction'],.5)
        self.assertEqual(metrics['recent_episode_both_feet_fraction'],1.)
        self.assertEqual(metrics['recent_episode_mean_duration_s'],21.)
        self.assertEqual(residual.recent_episode_metrics(records,window=1)['recent_episode_survival_fraction'],1.)
        self.assertEqual(residual.recent_episode_metrics([]),{'recent_episode_count':0})

    def test_zero_residual_is_exact_and_zero_command_ignores_bad_latent(self):
        actual=residual.compose(self.base,torch.zeros_like(self.base),self.obs,self.names,1.)
        self.assertTrue(torch.equal(actual,self.base))
        self.obs[:,69]=0
        actual=residual.compose(self.base,torch.full_like(self.base,float('nan')),self.obs,self.names,1.)
        self.assertTrue(torch.equal(actual,self.base))

    def test_features_use_gait_period_and_omit_unbounded_age(self):
        first=residual.features(self.obs,1.2)
        self.obs[:,68]-=1.2/30
        self.obs[:,72:74]=123  # Legacy phase fields are not the gait's clock.
        second=residual.features(self.obs,1.2)
        self.assertEqual(first.shape,(2,80))
        torch.testing.assert_close(first[:,:78],second[:,:78],atol=2e-5,rtol=0)
        self.obs[:,76]=1
        changed=residual.features(self.obs,1.2)
        self.assertTrue(torch.equal(changed[:,:73],second[:,:73]))
        self.assertTrue(torch.all(changed[:,73]==1))

    def test_cross_track_extension_preserves_actor_inputs_and_reward_schedule(self):
        from algorithms.urdf_learn_wasd_walk.leg_gait_search import REFERENCE_PARAMETERS
        parameters=torch.tensor(REFERENCE_PARAMETERS);parameters[20]=2.
        age=torch.linspace(0,90,4501)
        for old,new in zip(residual.reference_targets(age,parameters[:20]),residual.reference_targets(age,parameters)):
            torch.testing.assert_close(old,new,rtol=0,atol=0)
        for n in (19,33):
            obs=torch.zeros(2,24+3*n);before=residual.features(obs,14.)
            obs[:,9+3*n:11+3*n]=torch.tensor([[.5,-5.],[-1.,-10.]])
            torch.testing.assert_close(residual.features(obs,14.),before,rtol=0,atol=0)

    def test_one_lift_channel_coordinates_hip_knee_and_ankle(self):
        latent=torch.zeros_like(self.base);latent[:,self.names.index('left_knee_joint')]=1.
        out=residual.compose(self.base,latent,self.obs,self.names,1.)
        correction=(out-self.base)*.08
        knee=correction[:,self.names.index('left_knee_joint')]
        for joint in ('hip_pitch','ankle_pitch'):
            torch.testing.assert_close(correction[:,self.names.index('left_'+joint+'_joint')],-.5*knee,atol=1e-6,rtol=0)
        self.assertTrue(torch.all(correction[:,self.names.index('right_knee_joint')]==0))

    def test_motor_residual_is_bounded_and_toes_unchanged(self):
        out=residual.compose(self.base,torch.full_like(self.base,100),self.obs,self.names,1.)
        limits=torch.tensor(residual.residual_bounds(self.names))
        self.assertTrue(torch.all(((out-self.base)*.08).abs()<=limits+1e-6))
        for i,name in enumerate(self.names):
            if 'toe' in name:self.assertTrue(torch.equal(out[:,i],self.base[:,i]))

    def test_residual_hold_uses_physical_clock_and_restarts_after_reset(self):
        class Actor:
            residual_hold_steps=5
            calls=0
            def __call__(self,value):
                self.calls+=1
                return value['actor'][:,73:74].expand(-1,19).clone()
        actor=Actor();obs=self.obs[:1].clone()
        outputs=[]
        for step in (0,1,4,5,6):
            obs[:,68]=1-step/1500;obs[:,76]=step
            outputs.append(residual.held_actor_mean(actor,obs,1.2).clone())
        self.assertEqual(actor.calls,2)
        self.assertEqual([float(x[0,0]) for x in outputs],[0.,0.,0.,5.,5.])
        obs[:,68]=1;obs[:,76]=.7
        out=residual.held_actor_mean(actor,obs,1.2)
        self.assertEqual(actor.calls,3)
        self.assertAlmostEqual(float(out[0,0]),.7,places=6)

    def test_supported_landing_rejects_hops_and_support_loss_at_touchdown(self):
        history=torch.ones((3,2),dtype=torch.bool)
        # Normal left swing; both feet hop; left swing with later support loss.
        contact=torch.tensor([[False,True],[False,False],[False,True]])
        _,history=residual.supported_swing_landing(contact,torch.zeros_like(contact),history)
        contact=torch.tensor([[True,True],[True,True],[True,False]])
        landed=torch.tensor([[True,False],[True,True],[True,False]])
        rewarded,_=residual.supported_swing_landing(contact,landed,history)
        self.assertEqual(rewarded.tolist(),[[True,False],[False,False],[False,False]])

    def test_reference_never_reloads_swing_foot_and_covers_both_sides(self):
        parameters=torch.tensor([16.,.66,-.0057,.134,.397,.287,.028,0.,.0076,0.,0.,-.02,0.,0.,.503,2.97,.964,.0126,.890,.123])
        age=torch.linspace(0,34,1701)
        load,clearance=residual.reference_targets(age,parameters)
        torch.testing.assert_close(load.sum(1),torch.ones(len(age)))
        self.assertTrue(torch.all(load[clearance>.001]<1e-5))
        torch.testing.assert_close(load[0],torch.tensor([.5,.5]))
        for time,side in ((7,0),(8,0),(15,1),(16,1)):
            row=round(time/.02)
            self.assertGreater(float(clearance[row,side]),.03)
            self.assertEqual(float(clearance[row,1-side]),0.)
            # A supported 35 mm swing must beat grounded feet at this phase.
            actual=clearance[row].clone();actual[side]=.035
            self.assertLess(float((actual-clearance[row]).square().sum()),float(clearance[row].square().sum()))

    def test_first_transfer_is_observable_at_same_periodic_phase(self):
        self.obs[:,68]=1-torch.tensor([4.,20.])/30
        values=residual.features(self.obs,16.)
        torch.testing.assert_close(values[0,:78],values[1,:78],atol=1e-5,rtol=0)
        self.assertNotEqual(float(values[0,-1]),float(values[1,-1]))

class WholeBodyTests(unittest.TestCase):
    def test_named_projection_and_new_output_independence(self):
        from algorithms.urdf_learn_wasd_walk.leg_model_variant import specification
        from algorithms.urdf_learn_wasd_walk.model_spec import FINGER_JOINTS
        spec=specification('anatomical_legs_mass_v2'); old=spec['action_joints']
        names=old+[r['name'] for r in spec['joints'] if r['name'] not in FINGER_JOINTS and r['name'] not in old]
        self.assertEqual(len(names),33)
        raw=torch.arange(123.).reshape(1,-1)
        projected=residual.project_observation(raw,names,old)
        self.assertEqual(projected.shape,(1,81))
        self.assertTrue(torch.equal(projected[:,66:],raw[:,108:]))
        original=torch.arange(19.).reshape(1,-1)
        expanded=residual.scatter_action(original,names,old)
        self.assertTrue(torch.equal(expanded[:,:19],original))
        self.assertTrue(torch.equal(expanded[:,19:],torch.zeros(1,14)))
        self.assertEqual(residual.features(raw,.9).shape,(1,122))
        self.assertTrue(all(v>0 for v in residual.residual_bounds(names)))
        self.assertFalse(set(names)&set(FINGER_JOINTS))

class CampaignTests(unittest.TestCase):
    def test_speed_campaign_separates_training_target_and_fixed_proof(self):
        from types import SimpleNamespace
        from algorithms.urdf_learn_wasd_walk.whole_body_campaign import commands
        args=SimpleNamespace(name='anatomical_speed',parent=Path('/tmp/parent.pt'),residual_parent=Path('/tmp/donor.pt'),
            iterations=2,training_speed=.03,speed_target=.1,evaluate_initial=True,with_passive=True)
        checkpoint,stages=commands(args,Path('/tmp/output'))
        self.assertEqual(checkpoint.name,'model_initial.pt')
        self.assertEqual([x[0] for x in stages],['train','passive','stand','walk'])
        train=stages[0][2];walk=stages[-1][2]
        self.assertEqual(train[train.index('--target-speed')+1],'0.03')
        self.assertEqual(walk[walk.index('--speed-target')+1],'0.1')
        self.assertNotIn('--seconds',walk)
        self.assertNotIn('--stop-at-gate',walk)

    def test_failed_standing_stops_serial_campaign_before_walking(self):
        from algorithms.urdf_learn_wasd_walk import whole_body_campaign as campaign, mujoco_backend as backend
        with tempfile.TemporaryDirectory() as temporary:
            output=Path(temporary);parent=output/'training/parent/model.pt';parent.parent.mkdir(parents=True);parent.write_bytes(b'parent')
            name='anatomical_test_campaign';calls=[]
            def run(command,**kwargs):
                calls.append(command)
                if command[2].endswith('leg_forward_control'):
                    folder=output/(name+'_stand');folder.mkdir()
                    (folder/'dynamics.json').write_text(json.dumps({'failures':['fall'],'metrics':{'fall_count':1}}))
            with patch.object(backend,'OUTPUT',output),patch('sys.argv',['campaign','--name',name,'--parent',str(parent),'--iterations','2']),patch.object(campaign.subprocess,'run',side_effect=run):
                campaign.main()
            record=json.loads((output/(name+'_campaign')/'campaign.json').read_text())
            self.assertEqual(len(calls),2)
            self.assertEqual(record['status'],'numerical_failure')
            self.assertFalse(record['canonical_milestone_pass'])
            self.assertEqual([row['stage'] for row in record['evaluations']],['stand'])

class ResidualProvenanceTests(unittest.TestCase):
    def test_parent_gait_snapshot_binding_and_anatomical_reproduction(self):
        from algorithms.urdf_learn_wasd_walk.landau_forward_control import load_gait_source
        from algorithms.urdf_learn_wasd_walk import mujoco_milestones as gates
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);folder=root/'outputs/run';folder.mkdir(parents=True)
            (folder/'control_source.py').write_text('SENTINEL=7\n')
            sha=gates.digest(folder/'control_source.py')
            metadata={'policy_family':residual.FAMILY,'model_profile':'anatomical_legs_mass_v2',
                'source_sha256':{'backend.py':'backend'},'parent_source_sha256':{'leg_gait_search.py':sha}}
            checkpoint=folder/'model_50.pt'
            (folder/'training.json').write_text(json.dumps(metadata))
            module,_=load_gait_source(checkpoint,metadata)
            self.assertEqual(module.SENTINEL,7)
            job=gates.reproduction_job({'config':{'name':'residual','forward':.2,'seconds':45.,'target_distance_m':5.}},checkpoint)
            self.assertTrue(job['module'].endswith('.leg_forward_control'))
            self.assertIn('45.0',job['args'])
            job=gates.reproduction_job({'config':{'name':'long_residual','forward':.2,'seconds':550.,
                'forward_horizon_s':900.,'stop_at_gate':True,'target_distance_m':5.}},checkpoint)
            self.assertIn('900.0',job['args'])
            self.assertIn('--stop-at-gate',job['args'])
            self.assertGreaterEqual(job['timeout_s'],3900)
            (folder/'control_source.py').write_text('SENTINEL=8\n')
            with self.assertRaisesRegex(ValueError,'source changed'):load_gait_source(checkpoint,metadata)

    def test_certificate_rejects_changed_parent_bounds_or_snapshot(self):
        from algorithms.urdf_learn_wasd_walk import mujoco_milestones as gates
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);parent=root/'outputs/parent/model_23.pt';candidate=root/'outputs/residual/model_50.pt';folder=root/'outputs/eval'
            for path in (parent.parent,candidate.parent,folder):path.mkdir(parents=True)
            names=[*ACTION_JOINTS,'left_ankle_roll_joint','right_ankle_roll_joint']
            torch.save({'parameters':torch.ones(20)},parent)
            parent_bytes=parent.read_bytes()
            for path in (parent.parent/'control_source.py',candidate.parent/'control_source.py',folder/'gait_source.py'):path.write_text('gait')
            old={k:k for k in ('model_profile','model_xml_sha256','urdf_sha256','mesh_tree_sha256','hand_pose','support_profile','standing_checkpoint','standing_checkpoint_sha256')}
            old.update(action_joints=names,nominal_q=[0],nominal_ctrl=[0],action_scale=[.1]*19,
                policy_family='anatomical_periodic_feedback_cem',checkpoints={parent.name:gates.digest(parent)},source_sha256={'leg_gait_search.py':gates.digest(folder/'gait_source.py')})
            (parent.parent/'training.json').write_text(json.dumps(old))
            for path in (candidate.parent/'residual_source.py',folder/'residual_source.py'):path.write_text(Path(residual.__file__).read_text())
            training={**old,'policy_family':residual.FAMILY,'parent_checkpoint':str(parent),
                'parent_checkpoint_sha256':gates.digest(parent),'parent_training_sha256':gates.digest(parent.parent/'training.json'),
                'parent_source_sha256':old['source_sha256'],'residual_source_sha256':gates.digest(folder/'residual_source.py'),
                'observation_dim':81,'residual_feature_dim':80,'base_action_scale':old['action_scale'],
                'timing_contract':residual.TIMING_CONTRACT,'discount_horizon_s':20.,'gae_horizon_s':2.,
                'reward_schedule_parameters':[1.]*20,
                'action_scale':residual.action_scales(names),'residual_bounds_rad':residual.residual_bounds(names)}
            result={'policy_family':residual.FAMILY,'residual_source_sha256':training['residual_source_sha256']}
            torch.save({'actor_state_dict':residual.make_actor().state_dict()},candidate)
            with patch.object(gates,'ALG',root):
                gates.check_residual_source(folder,result,training,candidate)
                training['residual_bounds_rad'][0]+=1
                with self.assertRaisesRegex(ValueError,'bounds changed'):gates.check_residual_source(folder,result,training,candidate)
                training['residual_bounds_rad']=residual.residual_bounds(names)
                parent.write_bytes(b'changed gait')
                with self.assertRaisesRegex(ValueError,'parent changed'):gates.check_residual_source(folder,result,training,candidate)
                parent.write_bytes(parent_bytes)
                (folder/'residual_source.py').write_text('changed residual')
                with self.assertRaisesRegex(ValueError,'source mismatch'):gates.check_residual_source(folder,result,training,candidate)


if __name__=='__main__':unittest.main()
