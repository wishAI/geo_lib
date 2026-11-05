import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch
import torch
from algorithms.urdf_learn_wasd_walk import leg_residual_training as residual
from algorithms.urdf_learn_wasd_walk.model_spec import ACTION_JOINTS


class ResidualContractTests(unittest.TestCase):
    def setUp(self):
        self.names=[*ACTION_JOINTS,'left_ankle_roll_joint','right_ankle_roll_joint']
        self.obs=torch.zeros(2,81);self.obs[:,8]=-1;self.obs[:,68]=1-4/30;self.obs[:,69]=-.2
        self.base=torch.linspace(-5,5,19)[None].expand(2,-1)

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
        self.assertEqual(first.shape,(2,78))
        torch.testing.assert_close(first,second,atol=2e-5,rtol=0)
        self.obs[:,76]=1
        changed=residual.features(self.obs,1.2)
        self.assertTrue(torch.equal(changed[:,:73],second[:,:73]))
        self.assertTrue(torch.all(changed[:,73]==1))

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
            (folder/'control_source.py').write_text('SENTINEL=8\n')
            with self.assertRaisesRegex(ValueError,'source changed'):load_gait_source(checkpoint,metadata)

    def test_certificate_rejects_changed_parent_bounds_or_snapshot(self):
        from algorithms.urdf_learn_wasd_walk import mujoco_milestones as gates
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);parent=root/'outputs/parent/model_23.pt';candidate=root/'outputs/residual/model_50.pt';folder=root/'outputs/eval'
            for path in (parent.parent,candidate.parent,folder):path.mkdir(parents=True)
            names=[*ACTION_JOINTS,'left_ankle_roll_joint','right_ankle_roll_joint']
            parent.write_bytes(b'frozen gait')
            for path in (parent.parent/'control_source.py',candidate.parent/'control_source.py',folder/'gait_source.py'):path.write_text('gait')
            old={k:k for k in ('model_profile','model_xml_sha256','urdf_sha256','mesh_tree_sha256','hand_pose','support_profile','standing_checkpoint','standing_checkpoint_sha256')}
            old.update(action_joints=names,nominal_q=[0],nominal_ctrl=[0],action_scale=[.1]*19,
                policy_family='anatomical_periodic_feedback_cem',checkpoints={parent.name:gates.digest(parent)},source_sha256={'leg_gait_search.py':gates.digest(folder/'gait_source.py')})
            (parent.parent/'training.json').write_text(json.dumps(old))
            for path in (candidate.parent/'residual_source.py',folder/'residual_source.py'):path.write_text('residual')
            training={**old,'policy_family':residual.FAMILY,'parent_checkpoint':str(parent),
                'parent_checkpoint_sha256':gates.digest(parent),'parent_training_sha256':gates.digest(parent.parent/'training.json'),
                'parent_source_sha256':old['source_sha256'],'residual_source_sha256':gates.digest(folder/'residual_source.py'),
                'observation_dim':81,'residual_feature_dim':78,'base_action_scale':old['action_scale'],
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
                parent.write_bytes(b'frozen gait')
                (folder/'residual_source.py').write_text('changed residual')
                with self.assertRaisesRegex(ValueError,'source mismatch'):gates.check_residual_source(folder,result,training,candidate)


if __name__=='__main__':unittest.main()
