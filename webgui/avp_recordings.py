"""Durable, immutable LAN recording storage; no algorithm imports or browser state."""
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
import threading

ROOT = Path.home() / '.local/state/geo-lib/avp-recordings'
LOCK = threading.Lock()
MAX_UPLOAD = 96 * 1024 * 1024


def _matrix(value):
    return (isinstance(value, list) and len(value) == 4
            and all(isinstance(row, list) and len(row) == 4 and all(_number(v) for v in row) for row in value)
            and all(abs(v - (1 if i == 3 else 0)) < 1e-4 for i, v in enumerate(value[3])))


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def validate(clip):
    if not isinstance(clip, dict) or clip.get('format') != 'avp-motion-clip' or clip.get('version') != 1:
        raise ValueError('Unsupported motion recording')
    for key, limit in [('name', 200), ('createdAt', 64), ('urdfSha256', 64)]:
        if not isinstance(clip.get(key), str) or not 0 < len(clip[key]) <= limit:
            raise ValueError('Invalid recording metadata: ' + key)
    if not re.fullmatch('[0-9a-f]{64}', clip['urdfSha256']):
        raise ValueError('Invalid model hash')
    frames = clip.get('frames')
    if not isinstance(frames, list) or not 1 <= len(frames) <= 1800:
        raise ValueError('Invalid recording frame count')
    last = -1
    for f in frames:
        if not isinstance(f, dict) or not _number(f.get('t')) or not last < f['t'] <= 60:
            raise ValueError('Invalid recording timestamps')
        last = f['t']
        pose = f.get('pose')
        if not isinstance(pose, dict) or not 1 <= len(pose) <= 256 or not all(_number(v) for v in pose.values()):
            raise ValueError('Invalid joint pose')
        for key in ('tracking', 'rawTracking'):
            tracking = f.get(key)
            if tracking is None:
                continue
            if not isinstance(tracking, dict):
                raise ValueError('Invalid tracking')
            for name, value in tracking.items():
                valid = _matrix(value) if name in ('head', 'left_wrist', 'right_wrist') else (
                    name in ('left_arm', 'right_arm') and isinstance(value, list)
                    and len(value) in (25, 27) and all(_matrix(m) for m in value))
                if not valid:
                    raise ValueError('Invalid tracking matrix')
        if f.get('alignment') is not None and not _matrix(f['alignment']):
            raise ValueError('Invalid alignment')
    return clip


def _atomic(path, content):
    with tempfile.NamedTemporaryFile(dir=ROOT, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def save(clip):
    validate(clip)
    content = json.dumps(clip, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    identifier = hashlib.sha256(content).hexdigest()
    summary = {key: clip[key] for key in ('name', 'createdAt', 'urdfSha256')}
    summary.update(id=identifier, key='server:' + identifier, frames=len(clip['frames']), duration=clip['frames'][-1]['t'])
    with LOCK:
        ROOT.mkdir(parents=True, exist_ok=True, mode=0o700)
        path = ROOT / (identifier + '.json.gz')
        if not path.exists():
            _atomic(path, gzip.compress(content, compresslevel=3, mtime=0))
        _atomic(ROOT / (identifier + '.meta.json'), json.dumps(summary, ensure_ascii=False).encode())
    return summary


def summaries():
    result = []
    for path in ROOT.glob('*.meta.json'):
        try:
            item = json.loads(path.read_text())
            if (ROOT / (item['id'] + '.json.gz')).is_file():
                result.append(item)
        except (ValueError, OSError, KeyError):
            continue
    return sorted(result, key=lambda item: item['createdAt'], reverse=True)


def read(identifier):
    if not re.fullmatch('[0-9a-f]{64}', identifier):
        raise ValueError('Invalid recording identifier')
    path = ROOT / (identifier + '.json.gz')
    if not path.is_file():
        raise KeyError('Recording not found')
    return json.loads(gzip.decompress(path.read_bytes()))
