"""Model isolation and checkpoint provenance for the walking workspace."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from algorithms.urdf_learn_wasd_walk import evolution


class WorkspaceTests(unittest.TestCase):
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
                'checkpoints': {'model_0.pt': 'child-sha'}})
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
            self.assertEqual(evaluation['status'], 'failed')
            self.assertNotIn('playback', evaluation)
            self.assertEqual(evaluation['modelId'], sha)
            self.assertEqual(tree['summary']['checkpointBytes'], len(b'child checkpoint'))
            self.assertEqual(tree['models'][0]['milestones'], [])

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
