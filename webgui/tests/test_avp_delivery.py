"""LAN delivery must keep the AVP session independent of cloud asset inspection."""
import gzip
import http.client
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch
from webgui import server


class AvpDeliveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.content = b'const tracking = "head and hands";\n' * 300
        (self.root / 'index.html').write_bytes(self.content)
        self.static_patch = patch.object(server, 'STATIC_ROOT', self.root)
        self.static_patch.start()
        self.http = server.GeoGuiServer(('127.0.0.1', 0), server.Handler)
        self.thread = threading.Thread(target=self.http.serve_forever, daemon=True)
        self.thread.start()
        self.host = '127.0.0.1:' + str(self.http.server_port)

    def tearDown(self):
        self.http.shutdown()
        self.http.server_close()
        self.thread.join()
        self.static_patch.stop()
        self.tmp.cleanup()
        server.compressed_file.cache_clear()

    def request(self, method='GET', path='/', headers=None, body=None):
        connection = http.client.HTTPConnection(self.host, timeout=5)
        connection.request(method, path, body=body, headers=headers or {})
        response = connection.getresponse()
        result = response.status, dict(response.getheaders()), response.read()
        connection.close()
        return result

    def test_compression_revalidation_and_rebuild_invalidation(self):
        status, headers, data = self.request(headers={'Accept-Encoding': 'gzip'})
        self.assertEqual(status, 200)
        self.assertEqual(gzip.decompress(data), self.content)
        self.assertLess(len(data), len(self.content) // 10)
        status, _, data = self.request(headers={'If-None-Match': headers['ETag'], 'Accept-Encoding': 'gzip'})
        self.assertEqual((status, data), (304, b''))
        (self.root / 'index.html').write_bytes(self.content + b'changed')
        status, new_headers, data = self.request(headers={'If-None-Match': headers['ETag']})
        self.assertEqual(status, 200)
        self.assertNotEqual(new_headers['ETag'], headers['ETag'])
        self.assertEqual(data, self.content + b'changed')
        _, no_gzip, _ = self.request(headers={'Accept-Encoding': 'gzip;q=0'})
        self.assertNotIn('Content-Encoding', no_gzip)

    def test_local_avp_delivery_does_not_scan_cloud_or_other_sandboxes(self):
        relative = 'algorithms/avp_remote/gui/viewer.js'
        target = self.root / relative
        target.parent.mkdir(parents=True)
        target.write_bytes(self.content)
        with patch.object(server, 'REPO_ROOT', self.root), patch.object(server, 'avp_artifact_paths', return_value={relative}), patch.object(server, 'resolve_artifact', side_effect=AssertionError('cloud scan')), patch.object(server, 'declared_artifact_paths', side_effect=AssertionError('global scan')):
            status, _, data = self.request(path='/api/artifact?path=' + relative)
            self.assertEqual((status, data), (200, self.content))
            status, _, _ = self.request(path='/api/artifact?path=algorithms/avp_remote/private.json')
            self.assertEqual(status, 403)

    def test_local_wasm_decoder_is_compressed_and_revalidated(self):
        payload = b'\x00asm' + bytes(range(256)) * 1024
        (self.root / 'decoder.wasm').write_bytes(payload)
        status, headers, body = self.request(path='/decoder.wasm', headers={'Accept-Encoding': 'gzip'})
        self.assertEqual(status, 200)
        self.assertEqual(headers['Content-Type'], 'application/wasm')
        self.assertEqual(gzip.decompress(body), payload)
        status, _, body = self.request(path='/decoder.wasm', headers={'If-None-Match': headers['ETag']})
        self.assertEqual((status, body), (304, b''))

    def test_other_declared_assets_keep_lazy_workspace_resolution(self):
        relative = 'algorithms/example/result.json'
        target = self.root / 'example.json'
        target.write_bytes(b'{"ok":true}')
        with patch.object(server, 'declared_artifact_paths', return_value={relative}), patch.object(server, 'resolve_artifact', return_value=('repo', target)), patch.object(server, 'nesting_workspace_files', side_effect=AssertionError('unnecessary scan')):
            status, _, data = self.request(path='/api/artifact?path=' + relative)
            self.assertEqual((status, data), (200, b'{"ok":true}'))

    def test_mutations_allow_https_same_origin_and_reject_foreign_origin(self):
        with patch.object(server, 'avp_pose', return_value={'pose': {}}) as pose:
            status, _, _ = self.request('POST', '/api/avp/pose', {'Origin': 'https://' + self.host, 'Content-Type': 'application/json'}, '{}')
            self.assertEqual(status, 200)
            pose.assert_called_once()
            status, _, _ = self.request('POST', '/api/jobs', {'Origin': 'https://foreign.invalid', 'Content-Type': 'application/json'}, '{}')
            self.assertEqual(status, 400)

    def test_explicit_lan_binding_keeps_localhost_default(self):
        self.assertEqual(server.HOST, '127.0.0.1')
        for host in ['0.0.0.0', '192.168.2.31', '100.114.67.32', '127.0.0.1']:
            self.assertEqual(server.gui_bind_host(host), host)
        with self.assertRaises(ValueError):
            server.gui_bind_host('8.8.8.8')


if __name__ == '__main__':
    unittest.main()
