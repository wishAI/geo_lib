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
    def test_left_yaw_grid_freezes_sway_and_cruise(self):
        from algorithms.urdf_learn_wasd_walk.landau_direction_training import left_yaw_grid
        seed=torch.arange(20.)*.03;seed[17]=1.6034596
        grid=left_yaw_grid(seed)
        self.assertTrue(torch.equal(grid[:2],seed.expand(2,-1)))
        fixed=[i for i in range(20) if i!=17]
        self.assertTrue(torch.equal(grid[:,fixed],seed[fixed].expand(16,-1)))
        self.assertAlmostEqual(float(grid[2,17]),1.65,places=6)
        self.assertAlmostEqual(float(grid[-1,17]),2.30,places=6)

    def test_active_sway_can_screen_exact_turn_hold(self):
        from unittest.mock import patch
        from algorithms.urdf_learn_wasd_walk import landau_turn_control as turn
        argv=['turn','--name','test','--checkpoint','parent.pt','--seed-checkpoint','seed.pt',
            '--num-envs','256','--generations','2','--seconds','56','--turn-duration','44',
            '--teleop','--direction-train','--left-cruise-only','--left-turn-refine',
            '--left-balance-only','--left-balance-grid','--common-starts','--turn-hold-training']
        with patch('sys.argv',argv),patch.object(turn,'train') as train:
            turn.main();train.assert_called_once()
        yaw_argv=[v for v in argv if v not in ('--left-balance-only','--left-balance-grid')]
        yaw_argv[yaw_argv.index('256')]='64';yaw_argv.append('--left-yaw-grid')
        with patch('sys.argv',yaw_argv),patch.object(turn,'train') as train:
            turn.main();train.assert_called_once()

    def test_active_balance_preserves_learned_cruise(self):
        from algorithms.urdf_learn_wasd_walk.landau_direction_training import active_balance_parameters
        torch.manual_seed(9)
        seed=torch.rand(20);lows=torch.full_like(seed,-1.);highs=torch.ones_like(seed)
        center=2*(seed-lows)/(highs-lows)-1.;z=center.repeat(4,1);z[2:,5:7]+=.01
        result=active_balance_parameters(z,lows,highs,seed)
        self.assertTrue(torch.equal(result[:2],seed.expand(2,-1)))
        fixed=[i for i in range(20) if i not in (5,6)]
        self.assertTrue(torch.equal(result[:,fixed],seed[fixed].expand(4,-1)))
        self.assertFalse(torch.equal(result[2,5:7],seed[5:7]))

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


class ResidualRebaseTests(unittest.TestCase):
    def setUp(self):
        import tempfile,json
        from pathlib import Path
        from unittest.mock import patch
        from algorithms.urdf_learn_wasd_walk import mujoco_backend as backend
        self.backend=backend;self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.new=self.root/'training/new';self.old=self.root/'training/old'
        self.new.mkdir(parents=True);self.old.mkdir(parents=True)
        self.parent=self.new/'model_1.pt';self.previous=self.old/'model_299.pt'
        self.original={'parameters':torch.arange(16.),'turn_parameters':torch.arange(20.),
            'standing_actor_state_dict':{'normalizer':torch.ones(3)},
            'residual_actor_state_dict':{'weight':torch.randn(2,2)},'actor_state_dict':{'weight':torch.ones(2,2)}}
        self.fresh={k:v for k,v in self.original.items() if k in ('parameters','turn_parameters','standing_actor_state_dict')}
        self.fresh['turn_parameters']=self.fresh['turn_parameters'].clone();self.fresh['turn_parameters'][5:7]+=.001
        torch.save(self.original,self.previous);torch.save(self.fresh,self.parent)
        for folder in (self.old,self.new):
            for name in ('control_source.py','turn_source.py','teleop_contract.py','model.xml'):(folder/name).write_text(name)
        (self.old/'residual_source.py').write_text('residual');(self.old/'warm_states.npz').write_bytes(b'bank')
        shared=dict(status='complete',model_xml_sha256='model',urdf_sha256='urdf',mesh_tree_sha256='meshes',
            action_joints=['left_hip_roll_joint'],action_scale=[.4],nominal_q=[0.],nominal_ctrl=[0.],
            arguments={'turn_duration':44.},source_sha256={'gait':'hash'},turn_source_sha256=backend.digest(self.new/'turn_source.py'))
        self.new_meta={**shared,'policy_family':'periodic_feedback_cem','checkpoints':{self.parent.name:backend.digest(self.parent)}}
        self.old_meta={**shared,'policy_family':residual.FAMILY,'checkpoints':{self.previous.name:backend.digest(self.previous)},
            'residual_source_sha256':backend.digest(self.old/'residual_source.py')}
        backend.write_json(self.new/'training.json',self.new_meta);backend.write_json(self.old/'training.json',self.old_meta)
        self.addCleanup(patch.stopall);patch.object(backend,'OUTPUT',self.root).start()
        self.args=SimpleNamespace(name='composed',checkpoint=str(self.parent),residual_checkpoint=str(self.previous))
    def test_export_uses_bound_generation_row(self):
        import contextlib,io,json
        from algorithms.urdf_learn_wasd_walk.landau_turn_control import export_candidate
        table=self.new/'candidates_1.json'
        rows=[{'candidate':0,'parameters':self.original['turn_parameters'].tolist()},
              {'candidate':1,'parameters':self.fresh['turn_parameters'].tolist()}]
        self.backend.write_json(table,rows)
        self.new_meta['generation_candidate_tables']={table.name:self.backend.digest(table)}
        self.backend.write_json(self.new/'training.json',self.new_meta)
        with contextlib.redirect_stdout(io.StringIO()):export_candidate(self.parent,0,'selected')
        folder=self.root/'training/selected';blob=torch.load(folder/'model_0.pt',weights_only=False)
        self.assertTrue(torch.equal(blob['turn_parameters'],self.original['turn_parameters']))
        meta=json.loads((folder/'training.json').read_text())
        self.assertEqual(meta['candidate_selection']['candidate'],0);self.assertFalse(meta['canonical_milestone_pass'])
        table.write_text('changed')
        with self.assertRaisesRegex(ValueError,'table changed'):export_candidate(self.parent,1,'bad')
        self.assertFalse((self.root/'training/bad').exists())

    def test_rebase_keeps_learned_weights_and_frozen_cruise(self):
        import contextlib,io,json
        with contextlib.redirect_stdout(io.StringIO()):residual.rebase(self.args)
        folder=self.root/'training/composed';result=torch.load(folder/'model_0.pt',weights_only=False)
        self.assertTrue(torch.equal(result['residual_actor_state_dict']['weight'],self.original['residual_actor_state_dict']['weight']))
        self.assertTrue(torch.equal(result['turn_parameters'],self.fresh['turn_parameters']))
        meta=json.loads((folder/'training.json').read_text())
        self.assertFalse(meta['canonical_milestone_pass']);self.assertEqual(meta['composition']['changed_turn_indices'],[5,6])
    def test_rebase_rejects_active_variant(self):
        self.old_meta['residual_variant']='cruise_plus_active_left_v1'
        self.backend.write_json(self.old/'training.json',self.old_meta)
        with self.assertRaisesRegex(ValueError,'cruise-only'):residual.rebase(self.args)
        self.assertFalse((self.root/'training/composed').exists())

    def test_rebase_rejects_changed_cruise(self):
        self.fresh['turn_parameters'][15]+=.001;torch.save(self.fresh,self.parent)
        self.new_meta['checkpoints'][self.parent.name]=self.backend.digest(self.parent)
        self.backend.write_json(self.new/'training.json',self.new_meta)
        with self.assertRaisesRegex(ValueError,'Only active sway'):residual.rebase(self.args)
        self.assertFalse((self.root/'training/composed').exists())


if __name__=='__main__':unittest.main()
