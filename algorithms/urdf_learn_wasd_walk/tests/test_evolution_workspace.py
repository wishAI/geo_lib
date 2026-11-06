"""Model isolation and checkpoint provenance for the walking workspace."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from algorithms.urdf_learn_wasd_walk import evolution


class WorkspaceTests(unittest.TestCase):
    def test_curriculum_links_both_gait_endpoints_and_weight_donor(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);algorithm=root/'algorithms/walk';output=algorithm/'outputs'
            base=output/'tk2-backend-20260918/source/algorithms/urdf_learn_wasd_walk/outputs/mujoco/training'
            for name in ('start','target','weights','child'):
                folder=base/('anatomical_'+name);folder.mkdir(parents=True)
                record={'status':'completed','checkpoints':{'model_1.pt':name}}
                if name=='child':
                    record.update(policy_family='anatomical_feedback_residual_ppo',training_target_forward_mps=.1,
                        parent_checkpoint=str(base/'anatomical_target/model_1.pt'),parent_checkpoint_sha256='target',
                        residual_weight_parent={'checkpoint':str(base/'anatomical_weights/model_1.pt'),'sha256':'weights'},
                        gait_curriculum={'checkpoint':str(base/'anatomical_start/model_1.pt'),'sha256':'start'})
                (folder/'training.json').write_text(json.dumps(record))
            ledger=algorithm/'milestones.json';ledger.write_text('{}')
            with patch.object(evolution,'REPO_ROOT',root),patch.object(evolution,'ALGORITHM_ROOT',algorithm):
                tree=evolution.enrich_workspace({'nodes':[],'summary':{}},output,ledger)
            child=next(n for n in tree['nodes'] if n['id']=='run:anatomical_child')
            self.assertEqual(set(child['parentIds']),{'run:anatomical_start','run:anatomical_target','run:anatomical_weights'})
            self.assertEqual(child['milestoneId'],'walk_speed_0p1_mps_30s')

    def test_identical_warm_weights_resolve_declared_run_without_cycles(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);algorithm=root/'algorithms/walk';output=algorithm/'outputs'
            base=output/'tk2-backend-20260918/source/algorithms/urdf_learn_wasd_walk/outputs/mujoco'
            donor=base/'training/anatomical_donor';child=base/'training/anatomical_child'
            for folder,last in ((donor,'donor-last'),(child,'child-last')):
                folder.mkdir(parents=True)
                record={'status':'completed','checkpoints':{'model_initial.pt':'shared','model_1.pt':last}}
                if folder==child:record['residual_weight_parent']={'checkpoint':str(donor/'model_initial.pt'),'sha256':'shared'}
                (folder/'training.json').write_text(json.dumps(record))
            evaluation=base/'anatomical_eval';evaluation.mkdir()
            (evaluation/'progress.json').write_text(json.dumps({'status':'running','checkpoint':str(child/'model_initial.pt'),'checkpoint_sha256':'shared'}))
            ledger=algorithm/'milestones.json';ledger.write_text('{}')
            with patch.object(evolution,'REPO_ROOT',root),patch.object(evolution,'ALGORITHM_ROOT',algorithm):
                tree=evolution.enrich_workspace({'nodes':[],'summary':{}},output,ledger)
            nodes={n['id']:n for n in tree['nodes']}
            self.assertEqual(nodes['run:anatomical_child']['parentIds'],['run:anatomical_donor'])
            self.assertEqual(nodes['run:anatomical_eval']['parentIds'],['run:anatomical_child'])
            self.assertTrue(all(n['id'] not in n['parentIds'] for n in nodes.values()))
            missing={'status':'running','checkpoint':str(base/'unindexed/model_initial.pt'),'checkpoint_sha256':'shared'}
            (evaluation/'progress.json').write_text(json.dumps(missing))
            with patch.object(evolution,'REPO_ROOT',root),patch.object(evolution,'ALGORITHM_ROOT',algorithm):
                tree=evolution.enrich_workspace({'nodes':[],'summary':{}},output,ledger)
            node=next(n for n in tree['nodes'] if n['id']=='run:anatomical_eval')
            self.assertEqual(node['parentIds'],[])

    def test_initial_checkpoint_parent_resolves_even_when_run_has_later_updates(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);algorithm=root/'algorithms/walk';output=algorithm/'outputs'
            base=output/'tk2-backend-20260918/source/algorithms/urdf_learn_wasd_walk/outputs/mujoco/training'
            for name,record in [('anatomical_parent',{'checkpoints':{'model_initial.pt':'initial-sha','model_1.pt':'last-sha'}}),
                                ('anatomical_child',{'parent_checkpoint_sha256':'initial-sha'})]:
                folder=base/name;folder.mkdir(parents=True)
                (folder/'training.json').write_text(json.dumps({'status':'completed',**record}))
            ledger=algorithm/'milestones.json';ledger.write_text('{}')
            with patch.object(evolution,'REPO_ROOT',root),patch.object(evolution,'ALGORITHM_ROOT',algorithm):
                tree=evolution.enrich_workspace({'nodes':[],'summary':{}},output,ledger)
            child=next(n for n in tree['nodes'] if n['id']=='run:anatomical_child')
            self.assertIn('run:anatomical_parent',child['parentIds'])

    def test_live_evaluation_resolves_to_final_evidence_without_changing_node(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);algorithm=root/'algorithms/walk';output=algorithm/'outputs'
            run=output/'tk2-backend-20260918/source/algorithms/urdf_learn_wasd_walk/outputs/mujoco/anatomical_long_walk'
            run.mkdir(parents=True);progress=run/'progress.json'
            progress.write_text(json.dumps({'status':'running','checkpoint':'candidate.pt','checkpoint_sha256':'candidate-sha',
                'metrics':{'duration_s':300.,'semantic_forward_displacement_m':1.2}}))
            ledger=algorithm/'milestones.json';ledger.write_text(json.dumps({'sessionStatus':{'currentExperiment':{'evidence':str(progress)}}}))
            with patch.object(evolution,'REPO_ROOT',root),patch.object(evolution,'ALGORITHM_ROOT',algorithm):
                tree=evolution.enrich_workspace({'nodes':[],'summary':{}},output,ledger)
                node=next(n for n in tree['nodes'] if n['id']=='run:anatomical_long_walk')
                self.assertEqual(node['status'],'running')
                self.assertEqual(node['metrics']['semantic_forward_displacement_m'],1.2)
                (run/'dynamics.json').write_text(json.dumps({'failures':['distance too short'],'config':{'checkpoint':'candidate.pt'}}))
                tree=evolution.enrich_workspace({'nodes':[],'summary':{}},output,ledger)
                node=next(n for n in tree['nodes'] if n['id']=='run:anatomical_long_walk')
                self.assertEqual(node['status'],'failed')
                self.assertEqual(tree['currentNodeId'],node['id'])

    def test_active_training_does_not_present_input_parent_as_produced_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);algorithm=root/'algorithms/walk';output=algorithm/'outputs'
            run=output/'tk2-backend-20260918/source/algorithms/urdf_learn_wasd_walk/outputs/mujoco/training/anatomical_active'
            run.mkdir(parents=True)
            (run/'metadata.json').write_text(json.dumps({'status':'running','policy_family':'anatomical_feedback_residual_ppo',
                'arguments':{'checkpoint':'parent.pt','iterations':60,'num_envs':256},'parent_checkpoint_sha256':'parent-sha',
                'timing_contract':'anatomical_schedule_v2','runner_config':{'num_steps_per_env':64}}))
            (run/'iterations.jsonl').write_text(json.dumps({'iteration':5})+'\n')
            previous=run.parent.parent/'anatomical_previous'
            previous.mkdir();(previous/'dynamics.json').write_text(json.dumps({'failures':['fall']}))
            ledger=algorithm/'milestones.json';ledger.write_text(json.dumps({'sessionStatus':{'currentExperiment':{
                'evidence':str(previous/'dynamics.json'),
                'activeTraining':{'name':'anatomical_active','state':'running'}}}}))
            with patch.object(evolution,'REPO_ROOT',root),patch.object(evolution,'ALGORITHM_ROOT',algorithm):
                tree=evolution.enrich_workspace({'nodes':[],'summary':{}},output,ledger)
            node=next(n for n in tree['nodes'] if n['id']=='run:anatomical_active')
            self.assertEqual(node['status'],'running')
            self.assertIsNone(node.get('checkpointPath'))
            self.assertEqual(node['parentCheckpointSha256'],'parent-sha')
            self.assertEqual(node['trainingProgress']['completedIterations'],5)
            self.assertEqual(node['trainingProgress']['requestedIterations'],60)
            self.assertEqual(node['trainingProgress']['sampleCount'],81920)
            self.assertEqual(tree['currentNodeId'],node['id'])
            # A stale/completed active pointer must not hide the final proof.
            (run/'training.json').write_text(json.dumps({'status':'completed'}))
            with patch.object(evolution,'REPO_ROOT',root),patch.object(evolution,'ALGORITHM_ROOT',algorithm):
                tree=evolution.enrich_workspace({'nodes':[],'summary':{}},output,ledger)
            self.assertEqual(tree['currentNodeId'],'run:anatomical_previous')

    def test_exact_parent_checkpoint_model_scope_and_motion_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            algorithm = root / 'algorithms/walk'
            output = algorithm / 'outputs'
            (algorithm / 'gui').mkdir(parents=True)
            urdf = algorithm / 'robot.urdf'
            urdf.write_text('<robot name="v2"/>')
            sha = hashlib.sha256(urdf.read_bytes()).hexdigest()
            def write(path, value):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(value))
            write(algorithm / 'gui/manifest.json', {'viewer': {'urdfCandidates': [
                {'label': 'v2', 'path': str(urdf.relative_to(root))}]}})
            ledger = algorithm / 'milestones.json'
            write(ledger, {'lineage': 'v2', 'assetContract': {'urdfSha256': sha}, 'milestones': []})
            runtime = output / 'tk2-backend-20260918/source/algorithms/urdf_learn_wasd_walk/outputs/mujoco'
            train = runtime / 'training/anatomical_train/training.json'
            write(train, {'urdf_sha256': sha, 'action_joints': ['hip'],
                'arguments': {'checkpoint': 'parent.pt'}, 'parent_checkpoint_sha256': 'parent-sha',
                'checkpoints': {'model_0.pt': 'child-sha'}, 'hand_pose': 'closed_fists_v1',
                'nominal_q': [0,0,0,1,0,0,0,.2,.3], 'nominal_ctrl': [.4,-1.1]})
            train.with_name('model.xml').write_text('<mujoco><worldbody><body><freejoint/><joint name="hip"/><joint name="finger"/></body></worldbody><actuator><position joint="hip"/><position joint="finger"/></actuator></mujoco>')
            train.with_name('model_0.pt').write_bytes(b'child checkpoint')
            run = runtime / 'anatomical_eval/dynamics.json'
            write(run, {'identity': {'urdf_sha256': sha}, 'milestone': 'gate_5m_no_reset',
                'config': {'checkpoint': str(train.with_name('model_0.pt')), 'checkpoint_sha256': 'child-sha'},
                'trajectory_sha256': 'trajectory', 'failures': ['distance too short']})
            write(run.with_name('motion.json'), {'source': {'urdfSha256': sha, 'trajectorySha256': 'wrong'}})
            payload = {'nodes': [{'id': 'parent', 'kind': 'milestone', 'checkpointSha256': 'parent-sha',
                'lineage': 'v2', 'checkpointPath': 'parent.pt', 'artifacts': [], 'parentIds': []}], 'summary': {}}
            with patch.object(evolution, 'REPO_ROOT', root), patch.object(evolution, 'ALGORITHM_ROOT', algorithm):
                tree = evolution.enrich_workspace(payload, output, ledger)
            nodes = {n['id']: n for n in tree['nodes']}
            training = nodes['run:anatomical_train']
            evaluation = nodes['run:anatomical_eval']
            self.assertEqual(training['checkpointPath'], str(train.with_name('model_0.pt')))
            self.assertEqual(training['parentIds'], ['parent'])
            self.assertEqual(nodes['parent']['checkpointSha256'], 'parent-sha')
            self.assertEqual(nodes['parent']['checkpointPath'], 'parent.pt')
            self.assertEqual(evaluation['parentIds'], [training['id']])
            self.assertEqual(training['milestoneId'], 'gate_5m_no_reset')
            self.assertEqual(evaluation['actionJoints'], ['hip'])
            self.assertEqual(evaluation['defaultJointPositions'], {'hip': .2, 'finger': -1.1})
            self.assertEqual(training['defaultJointPositions'], evaluation['defaultJointPositions'])
            self.assertEqual(evaluation['bodySettings']['hands'], 'closed_fists_v1')
            self.assertEqual(evaluation['status'], 'failed')
            self.assertNotIn('playback', evaluation)
            self.assertEqual(evaluation['modelId'], sha)
            self.assertEqual(tree['summary']['checkpointBytes'], len(b'child checkpoint'))
            self.assertEqual(tree['models'][0]['milestones'], [])

    def test_recorded_defaults_require_matching_model_and_preserve_control_semantics(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            model = folder / 'model.xml'
            model.write_text('<mujoco><worldbody><body><freejoint/><joint name="hip"/><joint name="finger"/><joint name="passive"/></body></worldbody><actuator><position joint="hip"/><position joint="finger"/><motor joint="passive"/></actuator></mujoco>')
            record = dict(model_xml_sha256=evolution._digest(model), action_joints=['hip'],
                          nominal_q=[0,0,0,1,0,0,0,.2,.3,.4], nominal_ctrl=[.8,-1.1,9])
            self.assertEqual(evolution.recorded_joint_defaults(folder, record), {'hip':.2, 'finger':-1.1, 'passive':.4})
            record['model_xml_sha256'] = 'wrong'
            self.assertEqual(evolution.recorded_joint_defaults(folder, record), {})
            record = dict(nominal_q=[0,0], action_joints=[])
            self.assertEqual(evolution.recorded_joint_defaults(folder, record), {})
            model.unlink()
            self.assertEqual(evolution.recorded_joint_defaults(folder, record), {})

    def test_unknown_model_does_not_inherit_active_certificates(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ledger = root / 'milestones.json'
            ledger.write_text('{}')
            payload = {'nodes': [{'id': 'old', 'lineage': 'old', 'kind': 'experiment',
                'checkpointSha256': {'sha256': 'abc'}, 'artifacts': []}], 'summary': {}}
            with patch.object(evolution, 'ALGORITHM_ROOT', root):
                tree = evolution.enrich_workspace(payload, root / 'outputs', ledger)
            self.assertEqual(tree['nodes'][0]['modelId'], 'unidentified:old')
            self.assertEqual(tree['models'][0]['milestones'], [])
            self.assertEqual(tree['nodes'][0]['checkpointSha256'], 'abc')
