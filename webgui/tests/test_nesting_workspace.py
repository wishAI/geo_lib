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


if __name__ == '__main__':
    unittest.main()
