import math
import unittest
from types import SimpleNamespace
import torch
from algorithms.urdf_learn_wasd_walk import landau_cruise_residual as residual
from algorithms.urdf_learn_wasd_walk.landau_turn_control import CommandMemory


class ResidualTests(unittest.TestCase):
    def setUp(self):
        self.names=['left_hip_roll_joint','other','right_hip_roll_joint']
        self.obs=torch.zeros(4,70);self.obs[:,63]=.2;self.obs[:,8]=-1.
        self.memory=SimpleNamespace(reference=0.,left_cruise_blend=torch.tensor([0.,.25,.5,1.]),left_feedback_blend=0.)
    def test_zero_and_inactive_exact(self):
        baseline=torch.randn(4,3)
        self.assertTrue(torch.equal(baseline,residual.compose(baseline,torch.zeros(4,2),self.memory,self.names,self.obs)))
        changed=residual.compose(baseline,torch.ones(4,2),self.memory,self.names,self.obs)
        self.assertTrue(torch.equal(changed[0],baseline[0]))
        self.assertTrue(torch.equal(changed[:,1],baseline[:,1]))
        self.assertLessEqual(float(((changed-baseline)*.08).abs().max()),.020001)
    def test_stop_bypasses_decay(self):
        self.obs[:,63]=0.
        baseline=torch.randn(4,3)
        self.assertTrue(torch.equal(baseline,residual.compose(baseline,torch.full((4,2),float('nan')),self.memory,self.names,self.obs)))
    def test_feature_age_is_phase_only(self):
        self.obs[:,62]=1.-101./30.
        f=residual.features(self.obs,torch.tensor([.8]),self.memory)
        self.assertEqual(f.shape,(4,69));self.assertTrue(torch.isfinite(f).all())
        self.obs[:,62]-=.8/30.
        g=residual.features(self.obs,torch.tensor([.8]),self.memory)
        self.assertTrue(torch.allclose(f,g,atol=1e-4))
    def test_zero_head_deployment_exact(self):
        actor=residual.make_actor()
        with torch.no_grad():actor.mlp[-1].weight.zero_();actor.mlp[-1].bias.zero_()
        actor.eval();baseline=torch.randn(4,3)
        result=residual.apply(actor,baseline,self.obs,torch.tensor([.893]),self.memory,self.names)
        self.assertTrue(torch.equal(result,baseline))
    def test_batched_memory_matches_scalar(self):
        scalar=CommandMemory();batch=residual.CruiseMemory(1,'cpu')
        batch.turned=False
        obs=torch.zeros(1,70);obs[:,63]=.2
        # Starts with a left turn, then cruising, renewed yaw, then cruising.
        for step in range(300):
            age=step*.02;obs[:,65]=.035 if step<30 or 90<=step<120 else 0.
            scalar.observe(obs,torch.zeros(1,3),torch.eye(3)[None],age)
            batch.advance(obs[:,65].double(),torch.tensor([age],dtype=torch.float64),.02 if step else 0.)
            self.assertAlmostEqual(float(batch.reference[0]),scalar.reference,places=6)
            self.assertAlmostEqual(float(batch.left_cruise_blend[0]),scalar.left_cruise_blend,places=6)
            self.assertAlmostEqual(float(batch.left_feedback_blend[0]),scalar.left_feedback_blend,places=6)


class ResidualBindingTests(unittest.TestCase):
    def setUp(self):
        import tempfile
        from pathlib import Path
        from algorithms.urdf_learn_wasd_walk import mujoco_milestones as gates
        self.gates=gates;self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.parent_dir=self.root/'parent';self.parent_dir.mkdir()
        self.folder=self.root/'evaluation';self.folder.mkdir();self.candidate_dir=self.root/'candidate';self.candidate_dir.mkdir()
        self.parent=self.parent_dir/'model_1.pt';self.checkpoint=self.candidate_dir/'model_2.pt'
        self.original={'parameters':torch.arange(16.),'turn_parameters':torch.arange(20.),'standing_actor_state_dict':{'normalizer':torch.ones(3)}}
        torch.save(self.original,self.parent)
        for folder in (self.parent_dir,self.candidate_dir):
            (folder/'control_source.py').write_text('frozen gait')
        for folder in (self.candidate_dir,self.folder):(folder/'residual_source.py').write_text('residual')
        self.parent_meta={'policy_family':'periodic_feedback_cem','status':'complete','checkpoints':{self.parent.name:gates.digest(self.parent)},'turn_source_sha256':'turn'}
        gates.write(self.parent_dir/'training.json',self.parent_meta)
        self.blob={**self.original,'residual_actor_state_dict':{'weight':torch.zeros(2,2)}};torch.save(self.blob,self.checkpoint)
        self.meta={'policy_family':residual.FAMILY,'residual_source_sha256':gates.digest(self.folder/'residual_source.py'),
            'residual_bound_rad':.02,'residual_feature_dim':69,'residual_action_joints':['left_hip_roll_joint','right_hip_roll_joint'],
            'residual_gate':'left_cruise_blend * clamp(abs(forward)/0.2,0,1)',
            'parent_checkpoint':str(self.parent),'parent_checkpoint_sha256':gates.digest(self.parent),
            'parent_training_sha256':gates.digest(self.parent_dir/'training.json'),'turn_source_sha256':'turn'}
        self.result={'residual_source_sha256':self.meta['residual_source_sha256'],'residual_dispatch':'deterministic_cruise_hip_roll_v1'}
    def check(self):self.gates.check_residual_source(self.folder,self.result,self.meta,self.checkpoint)
    def test_valid_frozen_composition(self):self.check()
    def test_relabeled_weights_rejected(self):
        self.meta={'policy_family':'periodic_feedback_cem'};self.result={}
        with self.assertRaisesRegex(ValueError,'weights mislabeled'):self.check()
    def test_changed_bound_rejected(self):
        self.meta['residual_bound_rad']=.03
        with self.assertRaisesRegex(ValueError,'bounds/schema'):self.check()
    def test_changed_frozen_tensor_rejected(self):
        self.blob['turn_parameters']=self.blob['turn_parameters']+.001;torch.save(self.blob,self.checkpoint)
        with self.assertRaisesRegex(ValueError,'turn_parameters changed'):self.check()
    def test_invalidated_parent_rejected(self):
        self.parent_meta['status']='invalidated_test';self.gates.write(self.parent_dir/'training.json',self.parent_meta)
        self.meta['parent_training_sha256']=self.gates.digest(self.parent_dir/'training.json')
        with self.assertRaisesRegex(ValueError,'Invalid frozen parent'):self.check()
    def test_changed_source_rejected(self):
        (self.folder/'residual_source.py').write_text('changed')
        with self.assertRaisesRegex(ValueError,'source mismatch'):self.check()


if __name__=='__main__':unittest.main()
