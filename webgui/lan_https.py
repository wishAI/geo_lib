"""Create a local HTTPS identity and an Apple certificate-only setup profile.
Private keys stay outside the repository; renewing a leaf reuses the trusted CA.
"""
from __future__ import annotations

import argparse
import ipaddress
import json
import os
from pathlib import Path
import plistlib
import subprocess
import uuid


def openssl(*arguments: str) -> bytes:
    result = subprocess.run(['openssl', *map(str, arguments)], capture_output=True, check=True)
    return result.stdout


def prepare(host: str, directory: Path) -> dict:
    address = ipaddress.ip_address(host)
    if address.version != 4 or not address.is_private or address.is_unspecified:
        raise ValueError('Use a specific private LAN IPv4 address')
    directory = directory.expanduser().resolve()
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(directory, 0o700)
    ca_key, ca_cert = directory/'root.key', directory/'root.crt'
    key, cert = directory/'server.key', directory/'server.crt'
    profile = directory/'geo-library.mobileconfig'
    if ca_key.exists() != ca_cert.exists():
        raise ValueError('Incomplete local CA; preserve the existing key/certificate pair')
    old_umask = os.umask(0o077)
    try:
        if not ca_cert.exists():
            openssl('req', '-x509', '-newkey', 'rsa:3072', '-nodes', '-sha256', '-days', '1825',
                    '-keyout', ca_key, '-out', ca_cert, '-subj', '/CN=Geo Library LAN Root',
                    '-addext', 'basicConstraints=critical,CA:TRUE,pathlen:0',
                    '-addext', 'keyUsage=critical,keyCertSign,cRLSign')
        renew = not (key.is_file() and cert.is_file())
        if not renew:
            try:
                openssl('x509', '-in', cert, '-checkend', '2592000', '-noout')
                openssl('x509', '-in', cert, '-checkip', host, '-noout')
                openssl('verify', '-CAfile', ca_cert, '-verify_ip', host, cert)
            except subprocess.CalledProcessError:
                renew = True
        if renew:
            csr, extensions = directory/'server.csr', directory/'server.ext'
            extensions.write_text(f'basicConstraints=critical,CA:FALSE\nkeyUsage=critical,digitalSignature,keyEncipherment\nextendedKeyUsage=serverAuth\nsubjectAltName=IP:{host}\n')
            openssl('req', '-new', '-newkey', 'rsa:2048', '-nodes', '-keyout', key, '-out', csr,
                    '-subj', '/CN=Geo Library LAN')
            openssl('x509', '-req', '-in', csr, '-CA', ca_cert, '-CAkey', ca_key,
                    '-set_serial', str(uuid.uuid4().int), '-out', cert, '-days', '365', '-sha256', '-extfile', extensions)
            csr.unlink()
            extensions.unlink()
        certificate = openssl('x509', '-in', ca_cert, '-outform', 'DER')
        root_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, 'geo-library.lan.root'))
        payload = {
            'PayloadType': 'Configuration', 'PayloadVersion': 1,
            'PayloadIdentifier': 'local.geo-library.lan', 'PayloadUUID': str(uuid.uuid5(uuid.NAMESPACE_DNS, 'geo-library.lan.profile')),
            'PayloadDisplayName': 'Geo Library 局域网 HTTPS',
            'PayloadDescription': '仅安装 Geo Library 局域网根证书，用于 Vision Pro Safari 动作录制。安装后请在证书信任设置中启用 Geo Library LAN Root。',
            'PayloadOrganization': 'Geo Library', 'PayloadRemovalDisallowed': False,
            'PayloadContent': [{
                'PayloadType': 'com.apple.security.root', 'PayloadVersion': 1,
                'PayloadIdentifier': 'local.geo-library.lan.root', 'PayloadUUID': root_id,
                'PayloadDisplayName': 'Geo Library LAN Root',
                'PayloadCertificateFileName': 'Geo-Library-LAN-Root.cer', 'PayloadContent': certificate,
            }],
        }
        profile.write_bytes(plistlib.dumps(payload))
        for private in (ca_key, key):
            os.chmod(private, 0o600)
    finally:
        os.umask(old_umask)
    return {'host': host, 'certificate': str(cert), 'key': str(key), 'profile': str(profile),
            'rootFingerprint': openssl('x509', '-in', ca_cert, '-noout', '-fingerprint', '-sha256').decode().strip()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', required=True)
    parser.add_argument('--output-dir', type=Path, default=Path.home()/'.local/state/geo-lib/tls')
    args = parser.parse_args()
    print(json.dumps(prepare(args.host, args.output_dir), ensure_ascii=False))


if __name__ == '__main__':
    main()
