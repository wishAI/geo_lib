"""Sync must preserve unrelated staging and both versions of conflicting data."""
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('nesting_sync', Path(__file__).parents[1] / 'sync.py')
sync = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sync)


class OutputSyncTests(unittest.TestCase):
    def test_union_and_changes_on_either_side(self):
        def record(value):
            return {'sha256': value, 'mtime_ns': 1}
        left = {'local': record('a'), 'from_mac': record('new'), 'from_tk2': record('old')}
        right = {'remote': record('b'), 'from_mac': record('old'), 'from_tk2': record('new')}
        push, pull, backups = sync.plan_outputs(left, right, {'from_mac': 'old', 'from_tk2': 'old'})
        self.assertEqual(set(push), {'local', 'from_mac'})
        self.assertEqual(set(pull), {'remote', 'from_tk2'})
        self.assertEqual(backups, [])

    def test_concurrent_result_edit_preserves_both_versions(self):
        left = {'x': {'sha256': 'mac', 'mtime_ns': 1}}
        right = {'x': {'sha256': 'tk2', 'mtime_ns': 2}}
        push, pull, backups = sync.plan_outputs(left, right, {'x': 'old'})
        self.assertEqual(push, [])
        self.assertEqual(pull, ['x'])
        self.assertEqual(backups, [('x', 'mac', 'mac'), ('x', 'tk2', 'tk2')])

    def test_conflict_archive_rejects_changed_file(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            p = root / sync.SCOPE / 'outputs/result.json'
            p.parent.mkdir(parents=True)
            p.write_text('original')
            old = sync.digest(p)
            sync.preserve(root, [('result.json', 'mac', old)])
            backup = p.parent / '.sync/conflicts/mac' / ('result.json.' + old)
            self.assertEqual(backup.read_text(), 'original')
            p.write_text('changed')
            with self.assertRaisesRegex(RuntimeError, 'changed during sync'):
                sync.preserve(root, [('result.json', 'mac', old)])

    def test_nested_replay_keeps_geometry_scores_and_distinct_ids(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            dataset = root / sync.SCOPE / 'outputs/replays'
            rows = []
            for mode in ['restricted', 'gui']:
                captured = dataset / mode / 'same_case'
                captured.mkdir(parents=True)
                (captured / 'problem.json').write_text('{"boards":[],"widgets":[]}')
                (captured / 'nfp_7.solution.json').write_text('{"placements":[]}')
                rows.append({'id': mode + '/same_case', 'case_id': 'same_case', 'status': 'valid',
                             'effective_config': {'seed': 7}, 'score': {'placed_area': 123.4}})
            (dataset / 'coverage.json').write_text(json.dumps({'trials': rows, 'source_sha256': {'solver.py': 'saved'}}))
            self.assertEqual(sync.prepare_history_views(root)['replay_trials'], 2)
            report = dataset / 'index.json'
            mtime = report.stat().st_mtime_ns
            data = json.loads(report.read_text())
            self.assertEqual(len({r['id'] for r in data['results']}), 2)
            for row in data['results']:
                self.assertEqual(row['score']['placed_area'], 123.4)
                self.assertEqual((root / row['solutionPath']).read_text(), '{"placements":[]}')
                self.assertEqual((root / row['problemPath']).read_text(), '{"boards":[],"widgets":[]}')
            sync.prepare_history_views(root)
            self.assertEqual(report.stat().st_mtime_ns, mtime)



class SourceSyncTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        sync.git(self.root, 'init', '-q')
        sync.git(self.root, 'config', 'user.name', 'Sync test')
        sync.git(self.root, 'config', 'user.email', 'sync@example.invalid')
        self.scope = self.root / sync.SCOPE
        self.scope.mkdir(parents=True)
        (self.scope / 'solver.py').write_text('base\n')
        (self.root / 'unrelated.txt').write_text('base\n')
        sync.git(self.root, 'add', '.')
        sync.git(self.root, 'commit', '-qm', 'base')
        self.base = sync.git(self.root, 'rev-parse', 'HEAD').stdout.strip().decode()

    def tearDown(self):
        self.temp.cleanup()

    def test_snapshot_excludes_outputs_and_preserves_real_index(self):
        (self.root / 'unrelated.txt').write_text('staged unrelated\n')
        sync.git(self.root, 'add', 'unrelated.txt')
        index = (self.root / '.git/index').read_bytes()
        (self.scope / 'solver.py').write_text('edited\n')
        output = self.scope / 'outputs/result.json'
        output.parent.mkdir()
        output.write_text('{}')
        result = sync.snapshot(self.root, self.base)
        self.assertEqual((self.root / '.git/index').read_bytes(), index)
        self.assertEqual(sync.git(self.root, 'show', result['commit'] + ':unrelated.txt').stdout, b'base\n')
        self.assertNotIn(b'outputs/result.json', sync.git(self.root, 'ls-tree', '-r', result['commit']).stdout)

    def test_source_conflict_stops_without_editing_worktree(self):
        (self.scope / 'solver.py').write_text('mac\n')
        left = sync.snapshot(self.root, self.base)
        (self.scope / 'solver.py').write_text('tk2\n')
        right = sync.snapshot(self.root, self.base)
        result = sync.git(self.root, 'merge-tree', '--write-tree', left['commit'], right['commit'], check=False)
        self.assertEqual(result.returncode, 1)
        self.assertEqual((self.scope / 'solver.py').read_text(), 'tk2\n')

    def test_disjoint_source_edits_merge(self):
        (self.scope / 'solver.py').write_text('mac\n')
        left = sync.snapshot(self.root, self.base)
        (self.scope / 'solver.py').write_text('base\n')
        (self.scope / 'new.py').write_text('tk2\n')
        right = sync.snapshot(self.root, self.base)
        result = sync.git(self.root, 'merge-tree', '--write-tree', left['commit'], right['commit'])
        tree = result.stdout.splitlines()[0].decode()
        self.assertEqual(sync.git(self.root, 'show', tree + ':' + sync.SCOPE + '/solver.py').stdout, b'mac\n')
        self.assertEqual(sync.git(self.root, 'show', tree + ':' + sync.SCOPE + '/new.py').stdout, b'tk2\n')

    def test_apply_rejects_edits_since_snapshot(self):
        original = sync.snapshot(self.root, self.base)
        (self.scope / 'solver.py').write_text('concurrent\n')
        with self.assertRaisesRegex(RuntimeError, 'changed during sync'):
            sync.apply_source(self.root, original['commit'], original['commit'], self.base)
        self.assertEqual((self.scope / 'solver.py').read_text(), 'concurrent\n')


if __name__ == '__main__':
    unittest.main()
