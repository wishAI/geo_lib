import hashlib
import http.client
from pathlib import Path
import plistlib
import ssl
import tempfile
import threading
import unittest
from unittest.mock import patch

from webgui import lan_https, server


class LanHttpsTests(unittest.TestCase):
    def test_certificate_profile_renewal_and_trusted_https(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = lan_https.prepare('127.0.0.1', root)
            ca_before = (root/'root.crt').read_bytes()
            leaf_before = (root/'server.crt').read_bytes()
            profile = plistlib.loads(Path(result['profile']).read_bytes())
            self.assertEqual(len(profile['PayloadContent']), 1)
            self.assertEqual(profile['PayloadContent'][0]['PayloadType'], 'com.apple.security.root')
            self.assertEqual(profile['PayloadContent'][0]['PayloadContent'], lan_https.openssl('x509', '-in', root/'root.crt', '-outform', 'DER'))
            self.assertEqual((root/'server.key').stat().st_mode & 0o777, 0o600)
            lan_https.prepare('127.0.0.1', root)
            self.assertEqual(leaf_before, (root/'server.crt').read_bytes())
            # Leaf renewal for a changed LAN IP must preserve the installed CA.
            lan_https.prepare('192.168.2.32', root)
            self.assertEqual(ca_before, (root/'root.crt').read_bytes())
            self.assertNotEqual(leaf_before, (root/'server.crt').read_bytes())
            lan_https.prepare('127.0.0.1', root)
            context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            context.load_cert_chain(root/'server.crt', root/'server.key')
            listener = server.GeoGuiServer(('127.0.0.1', 0), server.Handler)
            listener.tls_profile = root/'geo-library.mobileconfig'
            listener.lan_https_url = 'https://127.0.0.1:' + str(listener.server_port)
            listener.socket = context.wrap_socket(listener.socket, server_side=True)
            thread = threading.Thread(target=listener.serve_forever, daemon=True)
            thread.start()
            try:
                trusted = ssl.create_default_context(cafile=str(root/'root.crt'))
                connection = http.client.HTTPSConnection('127.0.0.1', listener.server_port, context=trusted)
                connection.request('GET', '/setup/vision-pro.mobileconfig')
                response = connection.getresponse()
                self.assertEqual(response.status, 200)
                self.assertEqual(response.getheader('Content-Type'), 'application/x-apple-aspen-config')
                downloaded = response.read()
                self.assertEqual(hashlib.sha256(downloaded).digest(), hashlib.sha256(listener.tls_profile.read_bytes()).digest())
                connection.close()
                connection = http.client.HTTPSConnection('127.0.0.1', listener.server_port, context=trusted)
                connection.request('GET', '/setup/server.key')
                response = connection.getresponse()
                self.assertEqual(response.status, 404)
                response.read()
                connection.close()
                with self.assertRaises(ssl.SSLCertVerificationError):
                    connection = http.client.HTTPSConnection('127.0.0.1', listener.server_port)
                    connection.request('GET', '/api/connection')
            finally:
                listener.shutdown()
                listener.server_close()
                thread.join()


if __name__ == '__main__':
    unittest.main()
