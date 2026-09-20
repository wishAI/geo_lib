"""Fetch focused backend commits and declared evidence without overwriting edits."""
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SSH = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=12']
SOURCES = {
    'urdf_learn_wasd_walk': ('/home/wishai/vscode/geo_lib/algorithms/urdf_learn_wasd_walk/outputs/tk2-backend-20260918/source', 'codex/landau-mujoco-assistance', '6302172'),
    'motion_anim_generate': ('/home/wishai/.cache/geo-lib-motion-20260920/source', 'codex/motion-anim-generate', '3af07eae5660d3d28fe69952c257da3f873b9bd5'),
}


def run(args, **kwargs):
    return subprocess.check_output(args, cwd=ROOT, text=True, timeout=180, **kwargs).strip()


def sync_source(name, remote, branch, base):
    if run(['git', 'diff', '--cached', '--name-only']):
        raise RuntimeError('Staged changes exist; leave the index intact and retry after review.')
    ref = f'refs/remotes/tk2-sandbox/{name}'
    run(['git', '-c', 'core.sshCommand=' + shlex.join(SSH), 'fetch', '--no-tags', f'tk2:{remote}', f'{branch}:{ref}'])
    changes = run(['git', 'cherry', 'HEAD', ref, base]).splitlines()
    applied = []
    for line in changes:
        sign, commit = line.split()
        if sign == '-':
            continue
        paths = run(['git', 'diff-tree', '--no-commit-id', '--name-only', '-r', commit]).splitlines()
        if not paths or any(not p.startswith(f'algorithms/{name}/') for p in paths):
            raise RuntimeError(f'Commit {commit} changes paths outside {name}; review manually.')
        if any('/outputs/' in p or Path(p).suffix in {'.pt', '.ckpt', '.safetensors'} for p in paths):
            raise RuntimeError(f'Commit {commit} contains generated evidence or weights; review manually.')
        # Git refuses overlapping dirty changes and leaves unrelated work untouched.
        run(['git', 'cherry-pick', commit])
        applied.append(commit)
    return applied


def sync_artifacts(name, remote):
    manifest = json.loads((ROOT / f'algorithms/{name}/gui/manifest.json').read_text())
    paths = {item['path'] for item in manifest.get('artifacts', [])}
    for example in manifest.get('examples', []):
        paths.update(item['path'] for item in example.get('artifacts', []))
    if name == 'motion_anim_generate' and manifest.get('inspector', {}).get('path'):
        paths.add(manifest['inspector']['path'])
    prefix = f'algorithms/{name}/outputs/'
    paths = sorted(p for p in paths if p.startswith(prefix) and '..' not in Path(p).parts)
    code = 'import os,json; root=' + repr(remote) + '; paths=' + repr(paths) + '; print(json.dumps([p for p in paths if os.path.isfile(os.path.join(root,p))]))'
    available = json.loads(run([*SSH, 'tk2', 'python3 -c ' + shlex.quote(code)]))
    destination = Path(os.environ.get('GEO_CLOUD_ROOT', str(Path.home() / 'Nextcloud/Projects/geo_lib'))) / 'remote_outputs'
    for relative in available:
        # Backend JSON evolves atomically; rsync writes a temporary destination before rename.
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        run(['rsync', '-az', '--checksum', '-e', shlex.join(SSH), f'tk2:{remote}/{relative}', str(target)])
        # The walk builder consumes the compact progress file locally. Videos stay in cloud storage.
        if relative.endswith('/backend_progress.json'):
            local = ROOT / relative
            local.parent.mkdir(parents=True, exist_ok=True)
            temporary = local.with_suffix('.json.sync-tmp')
            temporary.write_bytes(target.read_bytes())
            temporary.replace(local)
    return {'synced': available, 'pending': sorted(set(paths) - set(available))}


def main():
    report = {}
    for name, (remote, branch, base) in SOURCES.items():
        report[name] = {'commits': sync_source(name, remote, branch, base), 'artifacts': sync_artifacts(name, remote)}
    run([sys.executable, 'algorithms/motion_anim_generate/make_preview.py'])
    run([sys.executable, 'algorithms/urdf_learn_wasd_walk/evolution.py'])
    run([sys.executable, 'geo', 'storage', 'audit'])
    out = ROOT / 'algorithms/motion_anim_generate/outputs/last_sync.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
