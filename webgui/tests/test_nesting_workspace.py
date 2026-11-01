import tempfile
import json
import unittest
from pathlib import Path
from unittest.mock import patch

from webgui import server, storage
from types import SimpleNamespace


class NestingWorkspaceTests(unittest.TestCase):
    def test_history_exposes_known_artifacts_without_symlink_or_log_reads(self):
        relative = 'algorithms/widget_nesting_2d/outputs/gui_runs'
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / 'abcd'
            run.mkdir()
            (run / 'problem.json').write_text('{}')
            (run / 'solution.json').write_text('{}')
            (run / 'private.log').write_text('not an artifact')
            (run / 'run.json').symlink_to(root / 'outside.json')
            (root / 'outside.json').write_text('{}')
            manifest = {'workspace': {'outputRoots': [relative]}}
            with patch.object(server, 'manifest_map', return_value={'widget_nesting_2d': manifest}), patch.object(server, 'artifact_candidates', return_value=[('repo', root)]):
                files = server.nesting_workspace_files()
            self.assertEqual(set(files), {relative + '/abcd/problem.json', relative + '/abcd/solution.json'})

    def test_gui_jobs_get_distinct_output_folders_and_artifact_paths(self):
        manager = server.JobManager()
        with patch.object(server.threading.Thread, 'start'):
            first = manager.start_example('widget_nesting_2d', 'complex_board', 'local', {})
            # Complete it before starting the second job: shared nesting resource.
            manager.jobs[first['id']].status = 'succeeded'
            second = manager.start_example('widget_nesting_2d', 'complex_board', 'tk2', {'case': 'constrained_single_board', 'seed': 42})
        self.assertNotEqual(first['id'], second['id'])
        for job in (first, second):
            destination = 'algorithms/widget_nesting_2d/outputs/gui_runs/' + job['id']
            self.assertEqual(job['command'][job['command'].index('--output') + 1], destination)
            self.assertTrue(all(a['path'].startswith(destination + '/') for a in job['artifacts']))
        self.assertIn('algorithms/widget_nesting_2d/inputs/constrained_single_board.json', second['command'])
        with self.assertRaisesRegex(ValueError, 'already running'):
            manager.start_example('widget_nesting_2d', 'complex_board', 'local', {})

    def test_failed_launch_remains_in_persistent_history(self):
        manager = server.JobManager()
        with patch.object(server.threading.Thread, 'start'):
            public = manager.start_example('widget_nesting_2d', 'complex_board', 'tk2', {'seed': 42})
        job = manager.jobs[public['id']]
        job.status = 'failed'
        job.log = 'Python executable unavailable'
        with tempfile.TemporaryDirectory() as directory, patch.object(server, 'REPO_ROOT', Path(directory)), patch.object(server, 'resolve_artifact', return_value=None):
            manager._persist_nesting_job(job)
            path = Path(directory) / 'algorithms/widget_nesting_2d/outputs/gui_runs' / job.id / 'run.json'
            record = json.loads(path.read_text())
        self.assertEqual(record['status'], 'failed')
        self.assertEqual(record['parameters']['seed'], 42)
        self.assertEqual(record['caseId'], 'complex_dual_board')
        self.assertIn('unavailable', record['error'])

    def test_source_only_refresh_does_not_hydrate_unrelated_assets(self):
        with patch.object(storage.subprocess, 'run', return_value=SimpleNamespace(returncode=0, stdout='', stderr='')) as run:
            result = storage.sync_source_tk2(hydrate_assets=False)
        self.assertTrue(result['ok'])
        self.assertEqual(run.call_count, 2)
        self.assertFalse(any('hydrate' in str(call.args) for call in run.call_args_list))

    def test_unknown_case_and_out_of_range_budget_are_rejected(self):
        manager = server.JobManager()
        for values in ({'case': '../secret'}, {'budget': 121}, {'seed': -1}):
            with self.assertRaises(ValueError):
                manager.start_example('widget_nesting_2d', 'complex_board', 'local', values)

    def test_inventory_preserves_one_level_allowlist_and_skips_symlink_directories(self):
        relative = 'algorithms/widget_nesting_2d/outputs/captured'
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            child = root / 'case'
            child.mkdir()
            deep = child / 'deep'
            deep.mkdir()
            for path in (root/'index.json', root/'analysis.json', root/'overview.png',
                         child/'problem.json', child/'nfp_42.solution.json', child/'result.png',
                         child/'run.json', child/'private.log', child/'index.json', deep/'problem.json'):
                path.write_text('{}')
            outside = root / 'outside'
            outside.mkdir()
            (outside / 'secret.solution.json').write_text('{}')
            # Keep the outside directory outside the root being scanned.
            actual = root / 'actual'
            actual.mkdir()
            (actual/'linked').symlink_to(outside, target_is_directory=True)
            (actual/'problem.json').symlink_to(child/'problem.json')
            self.assertEqual(list(server._nesting_artifact_entries(actual)), [])
            names = {name for name, _ in server._nesting_artifact_entries(root)}
            self.assertIn('case/nfp_42.solution.json', names)
            self.assertIn('overview.png', names)
            self.assertNotIn('case/index.json', names)
            self.assertNotIn('case/private.log', names)
            self.assertNotIn('case/deep/problem.json', names)

    def test_history_bundle_prefers_analysis_and_includes_legacy_run_metrics(self):
        relative = 'algorithms/widget_nesting_2d/outputs/captured'
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            case = root / 'gui'
            case.mkdir()
            (root/'index.json').write_text('{"results": [{"engine": "old"}]}')
            (root/'analysis.json').write_text('{"results": [{"engine": "nfp"}]}')
            (case/'run.json').write_text('{"caseId": "legacy"}')
            (case/'solution.json').write_text('{"placements": [], "score": {"placed_area": 0}}')
            (case/'nfp_42.solution.json').write_text('{"placements": ["large geometry"]}')
            manifest = {'workspace': {'outputRoots': [relative]}}
            def candidates(path):
                return [('repo', root / path.removeprefix(relative).lstrip('/'))]
            with patch.object(server, 'manifest_map', return_value={'widget_nesting_2d': manifest}), patch.object(server, 'artifact_candidates', side_effect=candidates):
                bundle = server.nesting_history(include_reports=True)
            documents = bundle['documents']
            self.assertIn(relative+'/analysis.json', documents)
            self.assertNotIn(relative+'/index.json', documents)
            self.assertEqual(documents[relative+'/gui/solution.json']['score']['placed_area'], 0)
            self.assertNotIn(relative+'/gui/nfp_42.solution.json', documents)
            self.assertFalse(bundle['errors'])

    def test_history_bundle_keeps_other_reports_when_one_capture_is_corrupt(self):
        relative = 'algorithms/widget_nesting_2d/outputs/captured'
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'index.json').write_text('{"results": []}')
            (root/'run.json').write_text('incomplete JSON')
            manifest = {'workspace': {'outputRoots': [relative]}}
            def candidates(path):
                return [('repo', root / path.removeprefix(relative).lstrip('/'))]
            with patch.object(server, 'manifest_map', return_value={'widget_nesting_2d': manifest}), patch.object(server, 'artifact_candidates', side_effect=candidates):
                bundle = server.nesting_history(include_reports=True)
            self.assertEqual(bundle['documents'][relative+'/index.json'], {'results': []})
            self.assertEqual(bundle['errors'][0]['path'], relative+'/run.json')

    def test_declared_source_artifacts_need_no_history_scan(self):
        artifact = 'algorithms/widget_nesting_2d/gui/workspace.js'
        manifest = {'artifacts': [{'path': artifact}]}
        with patch.object(server, 'discover_manifests', return_value=[manifest]), patch.object(server, 'nesting_workspace_files', side_effect=AssertionError('unnecessary history scan')):
            self.assertEqual(server.declared_artifact_paths(include_nesting=False), {artifact})

    def test_history_refresh_picks_up_new_results_and_newer_copies(self):
        import os
        relative = 'algorithms/widget_nesting_2d/outputs/captured'
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo, cloud = root/'repo', root/'cloud'
            repo.mkdir(); cloud.mkdir()
            (repo/'index.json').write_text('{}')
            (cloud/'index.json').write_text('{"newer": true}')
            os.utime(repo/'index.json', (1,1)); os.utime(cloud/'index.json', (2,2))
            manifest = {'workspace': {'outputRoots': [relative]}}
            with patch.object(server, 'manifest_map', return_value={'widget_nesting_2d': manifest}), patch.object(server, 'artifact_candidates', return_value=[('repo',repo),('nextcloud',cloud)]):
                first = server.nesting_workspace_files()
                (repo/'run.json').write_text('{}')
                second = server.nesting_workspace_files()
            self.assertEqual(first[relative+'/index.json']['source'], 'nextcloud')
            self.assertNotIn(relative+'/run.json', first)
            self.assertIn(relative+'/run.json', second)


if __name__ == '__main__':
    unittest.main()
