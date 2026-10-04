import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from algorithms.urdf_learn_wasd_walk import landau_right_rebase as rebase
from algorithms.urdf_learn_wasd_walk import landau_active_residual as active
from algorithms.urdf_learn_wasd_walk import mujoco_milestones as gates
from algorithms.urdf_learn_wasd_walk import mujoco_backend as backend
from algorithms.urdf_learn_wasd_walk import landau_turn_control as turn


class RightRebaseTests(unittest.TestCase):
    def test_feedback_grid_preserves_actors_parameter_scope(self):
        from algorithms.urdf_learn_wasd_walk.landau_hold_calibration import feedback_grid
        seed=torch.linspace(-.001,.001,20);seed[18]=-.00996226;seed[19]=-.00627491
        rows=feedback_grid(seed)
        self.assertEqual(rows.shape,(27,20))
        self.assertTrue(torch.equal(rows[:2],seed.repeat(2,1)))
        self.assertTrue(torch.equal(rows[:,:18],seed[:18].repeat(27,1)))
        self.assertTrue(torch.allclose(rows[:,18].min(),seed[18]-.005))
        self.assertTrue(torch.allclose(rows[:,19].max(),seed[19]+.0015))
        seed[19]=.015
        with self.assertRaisesRegex(ValueError,'bounds'):feedback_grid(seed)

    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.training=self.root/'training';self.training.mkdir()
        self.old_parent=self.training/'old';self.cruise=self.training/'cruise';self.dual=self.training/'dual';self.parent=self.training/'new'
        for folder in (self.old_parent,self.cruise,self.dual,self.parent):
            folder.mkdir()
            for name in ('control_source.py','residual_source.py','active_residual_source.py','teleop_contract.py','model.xml'):
                (folder/name).write_text(name)
            (folder/'turn_source.py').write_bytes(Path(turn.__file__).read_bytes())
        (self.parent/'turn_source.py').write_text((self.parent/'turn_source.py').read_text()+'\n# training-only change\n')
        self.old_blob={'parameters':torch.arange(16.),'turn_parameters':torch.linspace(-.02,.02,20),
            'standing_actor_state_dict':{'weight':torch.ones(2,2)}}
        common=dict(status='complete',policy_family='periodic_feedback_cem',arguments={'turn_duration':44.},
            model_xml_sha256='model',urdf_sha256='urdf',mesh_tree_sha256='meshes',action_joints=['left_hip_roll_joint','right_hip_roll_joint'],
            action_scale=[.4,.4],nominal_q=[0.],nominal_ctrl=[0.],source_sha256={'gait':'hash'})
        self.save(self.old_parent,self.old_blob,common)
        cruise_blob={**self.old_blob,'residual_actor_state_dict':{'weight':torch.ones(2,2)}}
        cruise_meta={**common,**self.parent_fields(self.old_parent), 'policy_family':'periodic_feedback_residual_ppo',
            'residual_source_sha256':gates.digest(self.cruise/'residual_source.py'),'residual_bound_rad':.02,'residual_feature_dim':69,
            'residual_action_joints':['left_hip_roll_joint','right_hip_roll_joint'],
            'residual_gate':'left_cruise_blend * clamp(abs(forward)/0.2,0,1)'}
        self.save(self.cruise,cruise_blob,cruise_meta)
        dual_blob={**cruise_blob,'residual_variant':active.VARIANT,'active_residual_actor_state_dict':{'weight':torch.zeros(2,2)}}
        dual_meta={**cruise_meta,'residual_variant':active.VARIANT,'active_residual_bound_rad':.02,'active_residual_feature_dim':69,
            'active_residual_gate':active.GATE,'active_residual_action_joints':['left_hip_roll_joint','right_hip_roll_joint'],
            'active_residual_source_sha256':gates.digest(self.dual/'active_residual_source.py'),
            'frozen_cruise_checkpoint':str(self.cruise/'model.pt'),'frozen_cruise_checkpoint_sha256':gates.digest(self.cruise/'model.pt'),
            'frozen_cruise_training_sha256':gates.digest(self.cruise/'training.json')}
        self.save(self.dual,dual_blob,dual_meta)
        fresh=copy.deepcopy(self.old_blob);fresh['turn_parameters'][12:14]+=.001
        self.save(self.parent,fresh,common)
        self.addCleanup(patch.stopall);patch.object(backend,'OUTPUT',self.root).start()
        self.cp=rebase.compose(self.parent/'model.pt',self.dual/'model.pt','composed')
        self.meta=gates.read(self.cp.parent/'training.json');self.blob=torch.load(self.cp,weights_only=False)
        self.result=dict(residual_variant=rebase.VARIANT,residual_dispatch=active.DISPATCH,
            residual_source_sha256=self.meta['residual_source_sha256'],active_residual_source_sha256=self.meta['active_residual_source_sha256'])

    def save(self,folder,blob,meta):
        torch.save(blob,folder/'model.pt')
        gates.write(folder/'training.json',{**meta,'turn_source_sha256':gates.digest(folder/'turn_source.py'),
            'checkpoints':{'model.pt':gates.digest(folder/'model.pt')}})

    def parent_fields(self,folder):
        return dict(parent_checkpoint=str(folder/'model.pt'),parent_checkpoint_sha256=gates.digest(folder/'model.pt'),
                    parent_training_sha256=gates.digest(folder/'training.json'))

    def check_both(self):
        torch.save(self.blob,self.cp)
        self.assertTrue(rebase.check_contract(self.meta,self.blob,gates.digest))
        gates.check_residual_source(self.cp.parent,self.result,self.meta,self.cp)

    def test_only_right_sway_changes_and_old_sources_remain_bound(self):
        self.check_both()
        self.assertNotEqual(self.meta['turn_source_sha256'],gates.read(self.dual/'training.json')['turn_source_sha256'])
        self.assertEqual(self.meta['right_rebase']['changed_turn_indices'],[12,13])
        self.assertFalse(any(k.startswith('frozen_cruise_') for k in self.meta))

    def test_changed_learned_head_rejected_independently(self):
        for key in ('residual_actor_state_dict','active_residual_actor_state_dict'):
            self.blob[key]['weight']+=.01;torch.save(self.blob,self.cp)
            with self.assertRaisesRegex(ValueError,'Frozen dual weights'):rebase.check_contract(self.meta,self.blob,gates.digest)
            with self.assertRaisesRegex(ValueError,'Frozen dual weights'):gates.check_residual_source(self.cp.parent,self.result,self.meta,self.cp)
            self.blob[key]['weight']-=.01

    def test_left_parameter_change_rejected_independently(self):
        self.blob['turn_parameters'][5]+=.001;torch.save(self.blob,self.cp)
        with self.assertRaisesRegex(ValueError,'Only right sway'):rebase.check_contract(self.meta,self.blob,gates.digest)
        with self.assertRaisesRegex(ValueError,'Only right sway'):gates.check_residual_source(self.cp.parent,self.result,self.meta,self.cp)

    def test_donor_tampering_rejected(self):
        with (self.dual/'model.pt').open('ab') as stream:stream.write(b'changed')
        with self.assertRaisesRegex(ValueError,'Frozen dual input'):self.check_both()

    def test_turn_duration_cannot_change_through_metadata(self):
        self.meta['arguments']['turn_duration']=60.
        with self.assertRaisesRegex(ValueError,'Frozen turn duration'):rebase.check_contract(self.meta,self.blob,gates.digest)
        with self.assertRaisesRegex(ValueError,'Frozen turn duration'):gates.check_residual_source(self.cp.parent,self.result,self.meta,self.cp)

    def test_changed_deployment_code_rejected_before_composition(self):
        source=self.parent/'turn_source.py';source.write_text(source.read_text().replace('TURN_START = 3.','TURN_START = 4.'))
        pm=gates.read(self.parent/'training.json');pm['turn_source_sha256']=gates.digest(source);gates.write(self.parent/'training.json',pm)
        with self.assertRaisesRegex(ValueError,'preserve deployment code'):
            rebase.compose(self.parent/'model.pt',self.dual/'model.pt','bad')
        self.assertFalse((self.training/'bad').exists())

    def test_arity_comparison_ignores_only_unreachable_extra_parameters(self):
        original=self.parent/'turn_source.py';other=self.root/'turn_copy.py'
        other.write_text(original.read_text().replace('clamp(-.05,.05)','clamp(-.04,.04)'))
        self.assertEqual(rebase.deployment_ast(original),rebase.deployment_ast(other))
        other.write_text(other.read_text().replace('p.shape[1]>=21','p.shape[1]>=20'))
        self.assertNotEqual(rebase.deployment_ast(original),rebase.deployment_ast(other))



class HoldCalibrationTests(unittest.TestCase):
    def test_gain_grid_preserves_every_other_parameter_and_duplicates_controls(self):
        from algorithms.urdf_learn_wasd_walk.landau_hold_calibration import parameter_grid
        seed = torch.linspace(-.02,.03,20)
        rows = parameter_grid(seed)
        self.assertEqual(tuple(rows.shape),(11,20))
        self.assertTrue(torch.equal(rows[0],seed))
        self.assertTrue(torch.equal(rows[1],seed))
        mask = torch.arange(20)!=3
        self.assertTrue(torch.equal(rows[:,mask],seed[mask].repeat(11,1)))
        self.assertGreater(float(rows[:,3].max()),.11)
        self.assertLess(float(rows[:,3].min()),-.11)
        for invalid in (torch.zeros(19),torch.full((20,),float('nan'))):
            with self.assertRaises(ValueError):parameter_grid(invalid)



class NominalHoldCalibrationTests(unittest.TestCase):
    def test_yaw_hold_grid_preserves_the_other_eighteen_parameters(self):
        from algorithms.urdf_learn_wasd_walk.landau_hold_calibration import nominal_grid
        seed = torch.linspace(-.02,.03,20);seed[17] = 1.603
        rows = nominal_grid(seed)
        self.assertEqual(tuple(rows.shape),(35,20))
        self.assertTrue(torch.equal(rows[0],seed))
        self.assertTrue(torch.equal(rows[1],seed))
        fixed = [i for i in range(20) if i not in (3,17)]
        self.assertTrue(torch.equal(rows[:,fixed],seed[fixed].repeat(35,1)))
        self.assertTrue(bool(((rows[:,17]>=1.45)&(rows[:,17]<=1.65)).all()))


class YawHoldRebaseTests(unittest.TestCase):
    def setUp(self):
        from algorithms.urdf_learn_wasd_walk import landau_yaw_hold_rebase as calibration
        self.calibration = calibration
        self.fixture = RightRebaseTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        f = self.fixture
        params = f.blob['turn_parameters'].clone();params[3] = -.03;params[17] = 1.49
        screens = []
        for i in range(2):
            folder = f.root/('screen'+str(i));folder.mkdir()
            source = folder/'screen_source.py';source.write_text('test bounded screen')
            screen = dict(test_run=i,kind='nominal_common_start_yaw_hold_calibration',changed_indices=[3,17],
                checkpoint_sha256=gates.digest(f.cp),source_sha256=gates.digest(source),
                candidate_parameters=[params.tolist()],results={mode:{'candidate_records':[
                    {'index':0,'eligible':[True]*4}]} for mode in ('m5','m6')})
            screen['training_sha256'] = gates.digest(f.cp.parent/'training.json')
            for key in ('turn_source_sha256','residual_source_sha256','active_residual_source_sha256'):
                screen[key] = f.meta[key]
            for filename,key in (('audit_source.py','audit_source_sha256'),('teleop_source.py','teleop_source_sha256'),
                                 ('direction_training_source.py','common_start_source_sha256')):
                helper = folder/filename;helper.write_text(filename);screen[key] = gates.digest(helper)
            path = folder/'screen.json';gates.write(path,screen);screens.append(path)
        self.cp = calibration.compose(f.cp,screens,0,'yaw_hold')
        self.meta = gates.read(self.cp.parent/'training.json')
        self.blob = torch.load(self.cp,weights_only=False)
        self.result = {**f.result,'residual_variant':calibration.VARIANT}

    def check(self):
        torch.save(self.blob,self.cp)
        self.assertTrue(self.calibration.check_contract(self.meta,self.blob,gates.digest))
        gates.check_residual_source(self.cp.parent,self.result,self.meta,self.cp)

    def test_composition_preserves_parent_and_both_learned_heads(self):
        self.check()
        self.assertEqual(self.meta['parent_checkpoint'],self.fixture.meta['parent_checkpoint'])
        self.assertEqual(self.meta['calibration_donor_checkpoint'],str(self.fixture.cp))

    def test_fixed_controller_and_actors_rejected_independently(self):
        self.blob['turn_parameters'][12] += .001;torch.save(self.blob,self.cp)
        for check in (lambda:self.calibration.check_contract(self.meta,self.blob,gates.digest),
                      lambda:gates.check_residual_source(self.cp.parent,self.result,self.meta,self.cp)):
            with self.assertRaisesRegex(ValueError,'Only .*parameters3/17'):check()
        self.blob['turn_parameters'][12] -= .001
        self.blob['active_residual_actor_state_dict']['weight'] += .01
        torch.save(self.blob,self.cp)
        with self.assertRaisesRegex(ValueError,'actor'):self.calibration.check_contract(self.meta,self.blob,gates.digest)
        with self.assertRaisesRegex(ValueError,'actor'):gates.check_residual_source(self.cp.parent,self.result,self.meta,self.cp)

    def test_same_screen_cannot_claim_two_repeats(self):
        self.meta['yaw_hold_calibration']['screens'][1] = self.meta['yaw_hold_calibration']['screens'][0]
        with self.assertRaisesRegex(ValueError,'distinct'):self.calibration.check_contract(self.meta,self.blob,gates.digest)
        with self.assertRaisesRegex(ValueError,'repeated screen'):gates.check_residual_source(self.cp.parent,self.result,self.meta,self.cp)

    def test_selected_row_and_turn_duration_are_bound(self):
        self.blob['turn_parameters'][17] += .01;torch.save(self.blob,self.cp)
        with self.assertRaisesRegex(ValueError,'selected screen row'):self.calibration.check_contract(self.meta,self.blob,gates.digest)
        with self.assertRaisesRegex(ValueError,'screen parameters'):gates.check_residual_source(self.cp.parent,self.result,self.meta,self.cp)
        self.meta['arguments']['turn_duration'] = 45.
        with self.assertRaisesRegex(ValueError,'arguments'):self.calibration.check_contract(self.meta,self.blob,gates.digest)
        with self.assertRaisesRegex(ValueError,'arguments'):gates.check_residual_source(self.cp.parent,self.result,self.meta,self.cp)

    def test_screen_helper_changes_rejected_independently(self):
        screen = Path(self.meta['yaw_hold_calibration']['screens'][0]['path'])
        (screen.parent/'direction_training_source.py').write_text('changed')
        with self.assertRaisesRegex(ValueError,'screen helper'):self.calibration.check_contract(self.meta,self.blob,gates.digest)
        with self.assertRaisesRegex(ValueError,'screen helper'):gates.check_residual_source(self.cp.parent,self.result,self.meta,self.cp)


class ActiveAttributionTests(unittest.TestCase):
    def test_scales_preserve_exact_incumbent_and_baseline_endpoints(self):
        from algorithms.urdf_learn_wasd_walk.landau_hold_calibration import scale_active
        base = torch.tensor([[1000.,-3.],[.001,2.],[.7,-.1]])
        full = base+torch.tensor([[.01,.02],[-.03,.01],[.2,-.1]])
        out = scale_active(base,full,torch.tensor([1.,0.,.5]))
        self.assertTrue(torch.equal(out[0],full[0]))
        self.assertTrue(torch.equal(out[1],base[1]))
        self.assertTrue(torch.equal(out[2],base[2]+.5*(full[2]-base[2])))
        for scales in (torch.tensor([1.,0.,1.1]),torch.tensor([1.,0.,float('nan')]),torch.ones(2)):
            with self.assertRaises(ValueError):scale_active(base,full,scales)


class RightControllerRebaseTests(unittest.TestCase):
    def setUp(self):
        from algorithms.urdf_learn_wasd_walk import landau_right_controller_rebase as calibration
        self.calibration=calibration;self.fixture=YawHoldRebaseTests();self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        f=self.fixture;root=f.fixture.root;seed=Path(f.meta['parent_checkpoint'])
        self.search=root/'training'/'right_search';self.search.mkdir()
        for filename in ('turn_source.py','control_source.py','teleop_contract.py'):
            (self.search/filename).write_bytes((f.cp.parent/filename).read_bytes())
        sb=copy.deepcopy(torch.load(seed,weights_only=False));sb['turn_parameters'][[7,8,12,13]]=torch.tensor([1.55,.07,-.006,.04])
        torch.save(sb,self.search/'model_1.pt')
        sm=copy.deepcopy(gates.read(seed.parent/'training.json'))
        sm.update(status='completed',arguments=dict(turn_duration=44.,right_controller_search=True,right_only=True,
            teleop=True,direction_train=True,common_starts=True,closed_loop_direction='right',seconds=120.,seed_checkpoint=str(seed)),
            checkpoints={'model_1.pt':gates.digest(self.search/'model_1.pt')},
            right_controller_search={'searched_indices':[7,8,12,13],'training_force_target_bw':2.9,'hard_force_limit_bw':3.})
        for filename,key in (('direction_training_source.py','direction_training_source_sha256'),
                             ('direction_training_protocol.py','direction_training_protocol_sha256'),('teleop_contract.py','teleop_contract_sha256')):
            if not (self.search/filename).exists():(self.search/filename).write_text(filename)
            sm[key]=gates.digest(self.search/filename)
        sm['generation_candidate_tables']={}
        for generation in (0,1):
            filename=f'candidates_{generation}.json'
            gates.write(self.search/filename,[{'candidate':0,'parameters':sb['turn_parameters'].tolist(),'eligible':True,
                'failure_bits':[0]*4,'peak_support_bw':[2.8]*4,'world_gates':{'completed':[True]*4},'generation':generation}])
            sm['generation_candidate_tables'][filename]=gates.digest(self.search/filename)
        gates.write(self.search/'training.json',sm)
        self.cp=calibration.compose(f.cp,self.search/'model_1.pt',['0:0','1:0'],'right_controller')
        self.meta=gates.read(self.cp.parent/'training.json');self.blob=torch.load(self.cp,weights_only=False)
        self.result={**f.result,'residual_variant':calibration.VARIANT}

    def checks(self):
        torch.save(self.blob,self.cp)
        return (lambda:self.calibration.check_contract(self.meta,self.blob,gates.digest),
                lambda:gates.check_residual_source(self.cp.parent,self.result,self.meta,self.cp))

    def test_preserves_calibrated_left_and_recursive_parent(self):
        for check in self.checks():check()
        self.assertTrue(torch.equal(self.blob['turn_parameters'][[3,17]],self.fixture.blob['turn_parameters'][[3,17]]))

    def test_left_and_actor_changes_rejected_independently(self):
        original=copy.deepcopy(self.blob)
        for index in (3,17,18):
            self.blob=copy.deepcopy(original);self.blob['turn_parameters'][index]+=.001
            for check in self.checks():
                with self.assertRaisesRegex(ValueError,'Only right'):check()
        self.blob=copy.deepcopy(original);self.blob['active_residual_actor_state_dict']['weight']+=.01
        for check in self.checks():
            with self.assertRaisesRegex(ValueError,'actor'):check()

    def test_repeated_table_and_source_tampering_rejected(self):
        records=self.meta['right_controller_calibration']['records']
        records[1]=copy.deepcopy(records[0])
        for check in self.checks():
            with self.assertRaisesRegex(ValueError,'repet|repeated'):check()
        self.meta=gates.read(self.cp.parent/'training.json')
        (self.search/'direction_training_source.py').write_text('changed')
        for check in self.checks():
            with self.assertRaisesRegex(ValueError,'helper'):check()

    def test_search_standing_change_rejected_even_with_updated_hashes(self):
        search=self.search/'model_1.pt';sb=torch.load(search,weights_only=False)
        sb['standing_actor_state_dict']['weight']+=.1;torch.save(sb,search)
        sm=gates.read(self.search/'training.json');sm['checkpoints'][search.name]=gates.digest(search)
        gates.write(self.search/'training.json',sm)
        self.meta['right_controller_calibration'].update(search_checkpoint_sha256=gates.digest(search),
            search_training_sha256=gates.digest(self.search/'training.json'))
        for check in self.checks():
            with self.assertRaisesRegex(ValueError,'standing'):check()


if __name__=='__main__':unittest.main()
