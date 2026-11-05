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

    def test_noncurrent_walk_node_can_load_declared_compressed_motion(self):
        import json
        prefix = 'algorithms/urdf_learn_wasd_walk/outputs/'
        tree_path, motion_path = prefix + 'evolution.json', prefix + 'older_run/motion.bin.gz'
        tree = self.root / tree_path
        samples = self.root / motion_path
        samples.parent.mkdir(parents=True)
        samples.write_bytes(gzip.compress(b'recorded motion'))
        tree.write_text(json.dumps({'nodes': [{'id': 'older', 'artifacts': [
            {'path': motion_path, 'kind': 'binary'},
            {'path': prefix + 'older_run/model.pt', 'kind': 'binary'},
            {'path': prefix + '../private/motion.bin.gz', 'kind': 'binary'}]}]}))
        manifest = {'id': 'urdf_learn_wasd_walk', 'inspector': {'type': 'evolutionTree', 'path': tree_path}}
        def resolve(relative):
            path = self.root / relative
            return ('repo', path) if path.is_file() else None
        with patch.object(server, 'REPO_ROOT', self.root), patch.object(server.storage, 'cloud_root', return_value=self.root / 'cloud'), patch.object(server, 'discover_manifests', return_value=[manifest]), patch.object(server, 'resolve_artifact', side_effect=resolve), patch.object(server, 'nesting_workspace_files', return_value={}):
            status, _, data = self.request(path='/api/artifact?path=' + motion_path)
            self.assertEqual((status, data), (200, samples.read_bytes()))
            for private in (prefix + 'older_run/model.pt', prefix + '../private/motion.bin.gz', prefix + 'unlisted/motion.bin.gz'):
                status, _, _ = self.request(path='/api/artifact?path=' + private)
                self.assertEqual(status, 403)

    def test_shared_asset_identity_and_version_validation(self):
        import json
        from urllib.parse import urlencode
        left, right = self.root / 'one.stl', self.root / 'two.stl'
        left.write_bytes(b'identical mesh');right.write_bytes(left.read_bytes())
        urdf = '<robot name="shared"><link name="body"><visual><geometry><mesh filename="package://body.stl"/></geometry></visual></link></robot>'
        with patch.object(server, '_resolve_robot_asset', side_effect=lambda model, uri: left if model == 'one.urdf' else right):
            one = server.robot_asset_identities('one.urdf', urdf)
            two = server.robot_asset_identities('two.urdf', urdf)
            self.assertEqual(one, two)
        with patch.object(server, '_allowed_robot_path', return_value=True), patch.object(server, '_resolve_robot_asset', return_value=left):
            path = '/api/robot/asset?' + urlencode({'urdf': 'one.urdf', 'uri': 'package://body.stl'})
            status, _, body = self.request(path=path+'&metadata=1')
            self.assertEqual(status, 200)
            identity = json.loads(body)
            self.assertEqual(identity['byteLength'], len(b'identical mesh'))
            status, headers, body = self.request(path=path+'&sha256='+identity['sha256'])
            self.assertEqual((status, body), (200, b'identical mesh'))
            self.assertIn('immutable', headers['Cache-Control'])
            left.write_bytes(b'changed mesh')
            status, _, _ = self.request(path=path+'&sha256='+identity['sha256'])
            self.assertEqual(status, 409)
            status, _, body = self.request(path=path+'&metadata=1')
            self.assertNotEqual(json.loads(body)['sha256'], identity['sha256'])
        with patch.object(server, '_allowed_robot_path', return_value=False):
            status, _, _ = self.request(path=path+'&metadata=1')
            self.assertEqual(status, 403)

    def test_mesh_workbench_supports_shared_file_identity(self):
        import json
        asset = self.root / 'part.stl';asset.write_bytes(b'workbench mesh')
        with patch.object(server, '_mesh_asset_relative', return_value=('part.stl', None)), patch.object(server, 'resolve_artifact', return_value=('repo', asset)):
            status, _, body = self.request(path='/api/mesh/asset?sandbox=model&part=leg&variant=source&metadata=1')
            self.assertEqual(status, 200)
            self.assertEqual(json.loads(body), server.asset_identity(asset))

    def test_artifact_metadata_uses_original_allowlist(self):
        import json
        relative = 'algorithms/example/shared.glb'
        artifact = self.root / 'shared.glb';artifact.write_bytes(b'character')
        with patch.object(server, 'declared_artifact_paths', return_value={relative}), patch.object(server, 'nesting_workspace_files', return_value=set()), patch.object(server, 'resolve_artifact', return_value=('repo', artifact)):
            status, _, body = self.request(path='/api/artifact?path='+relative+'&metadata=1')
            self.assertEqual(status, 200)
            self.assertEqual(json.loads(body), server.asset_identity(artifact))
            status, _, _ = self.request(path='/api/artifact?path=private.glb&metadata=1')
            self.assertEqual(status, 403)

    def test_robot_json_is_cached_compressed_and_invalidated(self):
        urdf = self.root / 'robot.urdf'
        urdf.write_text('<robot name="test">' + '<!-- mesh -->' * 500 + '</robot>')
        with patch.object(server, '_allowed_robot_path', return_value=True), patch.object(server, 'resolve_artifact', return_value=('repo', urdf)):
            status, headers, body = self.request(path='/api/robot/urdf?path=robot.urdf', headers={'Accept-Encoding': 'gzip'})
            self.assertEqual(status, 200)
            self.assertIn('no-cache', headers['Cache-Control'])
            self.assertNotIn('no-store', headers['Cache-Control'])
            self.assertIn(b'<robot', gzip.decompress(body))
            status, _, body = self.request(path='/api/robot/urdf?path=robot.urdf', headers={'If-None-Match': headers['ETag']})
            self.assertEqual((status, body), (304, b''))
            urdf.write_text('<robot name="changed"/>')
            status, new_headers, _ = self.request(path='/api/robot/urdf?path=robot.urdf', headers={'If-None-Match': headers['ETag']})
            self.assertEqual(status, 200)
            self.assertNotEqual(headers['ETag'], new_headers['ETag'])

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

    def test_avp_cloud_link_requires_registered_target(self):
        relative = 'algorithms/avp_remote/outputs/web_scene/scene.json'
        repo, cloud = self.root / 'repo', self.root / 'cloud'
        cloud.mkdir()
        target = cloud / 'scene.json'
        target.write_bytes(self.content)
        link = repo / relative
        link.parent.mkdir(parents=True)
        link.symlink_to(target)
        with patch.object(server, 'REPO_ROOT', repo), patch.object(server.storage, 'cloud_root', return_value=cloud), patch.object(server, 'avp_artifact_paths', return_value={relative}):
            with patch.object(server.storage, 'managed_entries', return_value=[{'cloudPath': 'scene.json'}]):
                status, _, data = self.request(path='/api/artifact?path=' + relative)
                self.assertEqual((status, data), (200, self.content))
            with patch.object(server.storage, 'managed_entries', return_value=[]):
                status, _, _ = self.request(path='/api/artifact?path=' + relative)
                self.assertEqual(status, 400)

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
