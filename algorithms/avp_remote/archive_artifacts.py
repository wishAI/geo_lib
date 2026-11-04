"""Archive AVP evidence using the repository's Nextcloud manifest/link contract.

Dry run by default. --apply uploads through the configured rclone remote, checks
remote bytes, then registers links. Live recording storage is never modified.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

REPO = Path(__file__).resolve().parents[2]
OUTPUTS = Path('algorithms/avp_remote/outputs')
GROUPS = ('web_scene', 'hand_mapping_review', 'unitree_hand_review', 'head_mapping_review')


def inventory(recordings):
    sources = []
    for group in GROUPS:
        sources.extend((path, path.relative_to(REPO))
                       for path in sorted((REPO / OUTPUTS / group).rglob('*')) if path.is_file())
    for pattern in ('*.json.gz', '*.meta.json'):
        sources.extend((path, OUTPUTS / 'recordings' / path.name)
                       for path in sorted(recordings.glob(pattern)) if path.is_file())
    return sources


def archive(recordings, remote, apply=False):
    manifest_path = REPO / 'large_files.json'
    manifest = json.loads(manifest_path.read_text())
    cloud = Path(os.environ.get('GEO_CLOUD_ROOT', manifest['cloudRoot'])).expanduser()
    entries = []
    with tempfile.TemporaryDirectory(prefix='avp-archive-') as temporary:
        stage = Path(temporary)
        for source, relative in inventory(recordings):
            # Snapshot before hashing/uploading, so an ongoing rebuild cannot
            # change what is registered in the manifest.
            snapshot = stage / 'snapshot'
            shutil.copyfile(source, snapshot)
            sha = hashlib.sha256(snapshot.read_bytes()).hexdigest()
            cloud_path = Path('assets/avp_remote') / sha / source.name
            target = stage / cloud_path
            target.parent.mkdir(parents=True, exist_ok=True)
            snapshot.replace(target)
            entries.append(dict(repoPath=relative.as_posix(), cloudPath=cloud_path.as_posix(),
                                size=target.stat().st_size, sha256=sha))
        if not apply or not entries:
            return dict(applied=False, files=entries)
        subprocess.run(['rclone', 'copy', str(stage), remote, '--immutable', '--no-traverse'], check=True)
        # WebDAV has no trustworthy native SHA-256: download and compare bytes.
        subtree = 'assets/avp_remote'
        subprocess.run(['rclone', 'check', str(stage / subtree), remote.rstrip('/') + '/' + subtree,
                        '--one-way', '--download'], check=True)
        for item in entries:
            path = REPO / item['repoPath']
            if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() != item['sha256']:
                raise RuntimeError(f'File changed during archive; keeping local revision: {path}')
        by_path = {item['repoPath']: item for item in manifest['files']}
        by_path.update({item['repoPath']: item for item in entries})
        manifest['files'] = list(by_path.values())
        # Register first; a crash leaves recoverable links via storage hydrate.
        staged_manifest = manifest_path.with_suffix('.json.tmp')
        staged_manifest.write_text(json.dumps(manifest, indent=2) + '\n')
        staged_manifest.replace(manifest_path)
        for item in entries:
            path = REPO / item['repoPath']
            path.parent.mkdir(parents=True, exist_ok=True)
            link = path.with_name(path.name + '.archive-link')
            link.symlink_to(cloud / item['cloudPath'])
            link.replace(path)
    return dict(applied=True, files=entries)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recordings', type=Path,
                        default=Path.home() / '.local/state/geo-lib/avp-recordings')
    parser.add_argument('--remote', default='ugreen-nextcloud:Projects/geo_lib',
                        help='rclone destination corresponding to GEO_CLOUD_ROOT')
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    print(json.dumps(archive(args.recordings.expanduser(), args.remote, args.apply), indent=2))


if __name__ == '__main__':
    main()
