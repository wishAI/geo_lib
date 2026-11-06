import gzip
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from webgui import server


class LineageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'lineage.json'
        self.tree = {'generatedAt': 'unchanged-time', 'models': [{'id': 'robot'}],
                     'currentNodeId': 'child', 'nodes': [
                         {'id': 'parent', 'modelId': 'robot', 'parentIds': [], 'kind': 'milestone'},
                         {'id': 'child', 'modelId': 'robot', 'parentIds': ['parent'],
                          'metrics': {'trace': list(range(10000))}, 'defaultJointPositions': {'hip': .2},
                          'playback': {'motionPath': 'motion.json'}}]}
        self.path.write_text(json.dumps(self.tree))
        manifest = {'walk': {'inspector': {'type': 'evolutionTree', 'path': 'declared.json'}}}
        for target, kwargs in [('manifest_map', {'return_value': manifest}),
                               ('resolve_artifact', {'return_value': ('repo', self.path)}),
                               ('declared_artifact_paths', {'side_effect': AssertionError('Global artifact scan')})]:
            patcher = patch.object(server, target, **kwargs)
            patcher.start()
            self.addCleanup(patcher.stop)
        server.lineage_snapshot.cache_clear()

    def test_small_index_and_exact_on_demand_record(self):
        summary = server.lineage_payload('walk')
        self.assertLess(len(json.dumps(summary)), len(json.dumps(self.tree)) / 20)
        self.assertEqual(summary['nodes'][1]['parentIds'], ['parent'])
        self.assertTrue(summary['nodes'][1]['hasPlayback'])
        self.assertNotIn('metrics', summary['nodes'][1])
        self.assertEqual(server.lineage_payload('walk', 'child')['node'], self.tree['nodes'][1])
        with self.assertRaises(ValueError):
            server.lineage_payload('unknown')
        with self.assertRaises(ValueError):
            server.lineage_payload('walk', '../unknown')

    def test_cache_and_invalidation_do_not_depend_on_generated_at(self):
        first = server.lineage_payload('walk')
        with patch.object(Path, 'read_text', side_effect=AssertionError('Reread ledger')):
            self.assertIs(server.lineage_payload('walk'), first)
            server.lineage_payload('walk', 'child')
        self.tree['nodes'][1]['label'] = 'Updated record'
        self.path.write_text(json.dumps(self.tree))
        second = server.lineage_payload('walk')
        self.assertNotEqual(first['revision'], second['revision'])
        self.assertEqual(second['nodes'][1]['label'], 'Updated record')

    def test_compressed_progress_length_and_conditional_response(self):
        handler = server.Handler.__new__(server.Handler)
        handler.headers = {'Accept-Encoding': 'gzip'}
        sent = {}
        handler.send_response = lambda status: sent.update(status=status)
        handler.send_header = lambda name, value: sent.update({name: value})
        handler.end_headers = lambda: None
        handler.wfile = io.BytesIO()
        handler._json(self.tree, revalidate=True)
        body = gzip.decompress(handler.wfile.getvalue())
        self.assertEqual(len(body), int(sent['X-Uncompressed-Content-Length']))
        self.assertEqual(json.loads(body), self.tree)
        handler.headers['If-None-Match'] = sent['ETag']
        handler.wfile = io.BytesIO()
        handler._json(self.tree, revalidate=True)
        self.assertEqual(sent['status'], 304)
        self.assertEqual(handler.wfile.getvalue(), b'')
