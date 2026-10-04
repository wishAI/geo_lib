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


if __name__=='__main__':unittest.main()
