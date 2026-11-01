"""Explicit bidirectional nesting sync: Git source snapshots and saved output files.

Run on the Mac: python3 algorithms/widget_nesting_2d/sync.py
The initial --initialize run requires the TK2 source edits to have been merged here.
No checkout branch, real Git index, unrelated algorithm, or output deletion is used.
"""
from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import uuid

SCOPE = 'algorithms/widget_nesting_2d'
BASE = 'refs/geo-sync/widget_nesting_2d/base'
SNAPSHOT = 'refs/geo-sync/widget_nesting_2d/snapshot'
SSH = ['-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8']


def git(root, *args, env=None, data=None, check=True):
    result = subprocess.run(['git', '-C', str(root), *args], input=data,
                            capture_output=True, env=env)
    if check and result.returncode:
        raise RuntimeError(result.stderr.decode(errors='replace') or result.stdout.decode(errors='replace'))
    return result


def snapshot(root, base):
    # A separate index prevents unrelated staged work from entering a sync commit.
    with tempfile.TemporaryDirectory(prefix='geo-nesting-index-') as folder:
        env = dict(os.environ, GIT_INDEX_FILE=str(Path(folder) / 'index'))
        git(root, 'read-tree', base, env=env)
        names = git(root, 'ls-files', '--cached', '--others', '--exclude-standard', '-z', '--', SCOPE, env=env).stdout.split(b'\0')
        names = [name for name in names if name and not name.startswith((SCOPE + '/outputs/').encode()) and not name.startswith((SCOPE + '/logs/').encode())]
        if names:
            git(root, '--literal-pathspecs', 'add', '-A', '--pathspec-from-file=-', '--pathspec-file-nul', env=env, data=b'\0'.join(names) + b'\0')
        tree = git(root, 'write-tree', env=env).stdout.strip().decode()
    for line in git(root, 'ls-tree', '-rl', tree, '--', SCOPE).stdout.decode().splitlines():
        if int(line.split()[3]) > 5242880:
            raise RuntimeError('Source exceeds 5 MiB: ' + line.split('\t', 1)[1])
    if tree == git(root, 'rev-parse', base + '^{tree}').stdout.strip().decode():
        commit = base
    else:
        commit = git(root, 'commit-tree', tree, '-p', base,
                     data=b'Snapshot nesting source for two-way sync\n').stdout.strip().decode()
    git(root, 'update-ref', SNAPSHOT, commit)
    return {'commit': commit, 'tree': tree}


def apply_source(root, before, after, base):
    if snapshot(root, base)['tree'] != git(root, 'rev-parse', before + '^{tree}').stdout.strip().decode():
        raise RuntimeError('Nesting source changed during sync; retry after the current edit finishes.')
    patch = git(root, 'diff', '--binary', before, after, '--', SCOPE).stdout
    if patch:
        git(root, 'apply', '--check', '-', data=patch)
        git(root, 'apply', '-', data=patch)
    git(root, 'update-ref', BASE, after)
    return {'commit': after}


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def inventory(root):
    output = Path(root) / SCOPE / 'outputs'
    output.mkdir(parents=True, exist_ok=True)
    result = {}
    for p in sorted(output.rglob('*')):
        relative = p.relative_to(output)
        if '.sync' in relative.parts or p.is_symlink() or not p.is_file():
            continue
        stat = p.stat()
        result[str(relative)] = {'sha256': digest(p), 'mtime_ns': stat.st_mtime_ns, 'bytes': stat.st_size}
    return result


def plan_outputs(local, remote, previous):
    push, pull, backups = [], [], []
    for path in sorted(local.keys() | remote.keys()):
        left, right = local.get(path), remote.get(path)
        if not right:
            push.append(path)
        elif not left:
            pull.append(path)
        elif left['sha256'] != right['sha256']:
            old = previous.get(path)
            if old == left['sha256']:
                pull.append(path)
            elif old == right['sha256']:
                push.append(path)
            else:
                # Both versions are retained before selecting the newest result.
                backups.extend([(path, 'mac', left['sha256']), (path, 'tk2', right['sha256'])])
                (push if left['mtime_ns'] >= right['mtime_ns'] else pull).append(path)
    return push, pull, backups


def preserve(root, files):
    output = Path(root) / SCOPE / 'outputs'
    count = 0
    for name, host, expected in files:
        p = output / name
        if p.resolve().is_relative_to(output.resolve()) and digest(p) == expected:
            dest = output / '.sync' / 'conflicts' / host / (name + '.' + expected)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dest)
            count += 1
        else:
            raise RuntimeError('Result changed during sync: ' + name)
    return {'preserved': count}


def prepare_history_views(root):
    """Expose captured constraint replays through the existing GUI artifact contract."""
    root = Path(root)
    output = root / SCOPE / 'outputs'
    prepared = 0
    for coverage in sorted(output.glob('*/coverage.json')):
        data = json.loads(coverage.read_text())
        if not isinstance(data.get('trials'), list):
            continue
        target = coverage.parent / 'index.json'
        if target.exists() and json.loads(target.read_text()).get('generated_by') != 'nesting_sync_history_views':
            continue
        rows = []
        for trial in data['trials']:
            identity = Path(trial['id'])
            if identity.is_absolute() or '..' in identity.parts:
                raise RuntimeError('Unsafe captured trial id: ' + str(identity))
            captured = coverage.parent / identity
            view = coverage.parent / ('view_' + hashlib.sha256(str(identity).encode()).hexdigest()[:16])
            view.mkdir(exist_ok=True)
            seed = (trial.get('effective_config') or trial.get('config_overrides') or {}).get('seed', 42)
            for name in ['problem.json', f'nfp_{seed}.solution.json']:
                src, dst = captured / name, view / name
                if src.is_file() and (not dst.exists() or digest(src) != digest(dst)):
                    shutil.copy2(src, dst)
            rows.append({**trial, 'engine': 'nfp', 'seed': seed,
                         'problemPath': str((view / 'problem.json').relative_to(root)),
                         'solutionPath': str((view / f'nfp_{seed}.solution.json').relative_to(root)),
                         'provenance': {'capturedTrial': trial, 'sourceSha256': data.get('source_sha256'),
                                        'coverageReport': str(coverage.relative_to(root))}})
        target = coverage.parent / 'index.json'
        document = json.dumps({'generated_by': 'nesting_sync_history_views', 'results': rows, 'source_sha256': data.get('source_sha256'),
                               'contract': {'note': 'Captured historical constraint replays; each trial retains its budget, effective settings, and original baseline.'}}, indent=2) + '\n'
        if not target.exists() or target.read_text() != document:
            target.write_text(document)
        prepared += len(rows)
    return {'replay_trials': prepared}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--peer', default='tk2')
    parser.add_argument('--peer-root', default='/home/wishai/vscode/geo_lib')
    parser.add_argument('--initialize', action='store_true')
    parser.add_argument('--worker', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        task = json.loads(base64.b64decode(args.worker))
        root = Path.cwd()
        op = task.pop('op')
        print(json.dumps({'snapshot': snapshot, 'apply': apply_source, 'inventory': inventory,
                          'preserve': preserve, 'prepare': prepare_history_views}[op](root, **task)))
        return
    root = Path(__file__).resolve().parents[2]
    source = Path(__file__).read_bytes()

    def peer(op, **kwargs):
        encoded = base64.b64encode(json.dumps(dict(op=op, **kwargs)).encode()).decode()
        command = 'cd ' + shlex.quote(args.peer_root) + ' && python3 - --worker ' + shlex.quote(encoded)
        completed = subprocess.run(['ssh', *SSH, args.peer, command], input=source, capture_output=True)
        if completed.returncode:
            raise RuntimeError(completed.stderr.decode(errors='replace'))
        return json.loads(completed.stdout)

    base_result = git(root, 'rev-parse', '--verify', BASE, check=False)
    if base_result.returncode:
        if not args.initialize:
            raise RuntimeError('No shared source baseline. Merge the TK2 edits here, then run --initialize once.')
        base = git(root, 'rev-parse', 'HEAD').stdout.strip().decode()
        git(root, 'push', f'{args.peer}:{args.peer_root}', f'{base}:{BASE}')
        git(root, 'update-ref', BASE, base)
    else:
        base = base_result.stdout.strip().decode()
    local = snapshot(root, base)
    remote = peer('snapshot', base=base)
    git(root, 'fetch', '--no-tags', f'{args.peer}:{args.peer_root}', f'+{SNAPSHOT}:refs/geo-sync/widget_nesting_2d/remote')
    if args.initialize:
        merged = local['commit']
    elif local['tree'] == remote['tree']:
        merged = local['commit']
    else:
        result = git(root, 'merge-tree', '--write-tree', local['commit'], remote['commit'], check=False)
        if result.returncode:
            raise RuntimeError('Source conflicts; snapshots retained in ' + SNAPSHOT + ' and refs/geo-sync/widget_nesting_2d/remote:\n' + result.stdout.decode())
        tree = result.stdout.splitlines()[0].decode()
        merged = git(root, 'commit-tree', tree, '-p', local['commit'], '-p', remote['commit'],
                     data=b'Merge Mac and TK2 nesting source\n').stdout.strip().decode()
    git(root, 'push', f'{args.peer}:{args.peer_root}', f'{merged}:refs/geo-sync/widget_nesting_2d/pending')
    peer('apply', before=remote['commit'], after=merged, base=base)
    apply_source(root, local['commit'], merged, base)
    print('Git source synchronized: ' + merged, flush=True)

    prepare_history_views(root)
    peer('prepare')
    state = root / SCOPE / 'outputs/.sync/results.json'
    previous = json.loads(state.read_text()) if state.exists() else {}
    left, right = inventory(root), peer('inventory')
    push, pull, backups = plan_outputs(left, right, previous)
    preserve(root, [b for b in backups if b[1] == 'mac'])
    peer('preserve', files=[b for b in backups if b[1] == 'tk2'])
    print(f'Result transfer: {len(push)} files to TK2, {len(pull)} files to Mac', flush=True)
    output = root / SCOPE / 'outputs'
    transfer_backup = '.sync/transfer_backups/' + uuid.uuid4().hex
    remote_output = args.peer + ':' + args.peer_root + '/' + SCOPE + '/outputs/'
    for names, start, end, entries in [(push, str(output) + '/', remote_output, left), (pull, remote_output, str(output) + '/', right)]:
        if not names:
            continue
        buckets = [[] for _ in range(min(4, len(names)))]
        sizes = [0] * len(buckets)
        for name in sorted(names, key=lambda p: entries[p]['bytes'], reverse=True):
            index = min(range(len(buckets)), key=lambda i: sizes[i])
            buckets[index].append(name)
            sizes[index] += entries[name]['bytes']
        with tempfile.TemporaryDirectory(prefix='geo-nesting-files-') as folder:
            def transfer(pair):
                index, batch = pair
                listing = Path(folder) / str(index)
                listing.write_bytes(b'\0'.join(name.encode() for name in batch) + b'\0')
                subprocess.run(['rsync', '-az', '--checksum', '--timeout=90', '--backup', '--backup-dir=' + transfer_backup,
                                '--from0', '--files-from=' + str(listing), '-e', 'ssh ' + ' '.join(SSH), start, end], check=True)
            with ThreadPoolExecutor(max_workers=len(buckets)) as pool:
                list(pool.map(transfer, enumerate(buckets)))
    left, right = inventory(root), peer('inventory')
    differences = [p for p in left.keys() | right.keys() if left.get(p, {}).get('sha256') != right.get(p, {}).get('sha256')]
    if differences:
        raise RuntimeError('Outputs changed during transfer; rerun sync: ' + ', '.join(differences[:10]))
    state.parent.mkdir(parents=True, exist_ok=True)
    state.write_text(json.dumps({p: v['sha256'] for p, v in left.items()}, indent=2) + '\n')
    print(json.dumps({'source_commit': merged, 'pushed_files': len(push), 'pulled_files': len(pull),
                      'preserved_versions': len(backups), 'verified_result_files': len(left)}, indent=2))


if __name__ == '__main__':
    try:
        main()
    except (RuntimeError, subprocess.CalledProcessError) as error:
        raise SystemExit(str(error))
