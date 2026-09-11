"""Local character preset history. Snapshots are immutable; names/archive flags aren't."""
from __future__ import annotations
import json
import math
import os
import re
import tempfile
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent / 'outputs' / 'presets'
_LOCK = threading.RLock()


def validate_settings(value):
    if not isinstance(value, dict) or value.get('version') != 1:
        raise ValueError('Unsupported preset format')
    if not re.fullmatch(r'[a-f0-9]{64}', str(value.get('assetHash', ''))):
        raise ValueError('Missing asset revision')
    def visit(item, depth=0):
        if depth > 8:
            raise ValueError('Preset nesting is too deep')
        if isinstance(item, dict):
            for key, val in item.items():
                if key in {'__proto__', 'constructor', 'prototype'} or len(key) > 160:
                    raise ValueError('Invalid property name')
                visit(val, depth+1)
        elif isinstance(item, list):
            for val in item: visit(val, depth+1)
        elif isinstance(item, float) and not math.isfinite(item):
            raise ValueError('Preset contains a non-finite value')
    visit(value)
    for key in ('morphs', 'bones', 'parts', 'outfit', 'links', 'bodyFrame'):
        if key in value and not isinstance(value[key], dict):
            raise ValueError('Invalid preset field: '+key)
    if len(json.dumps(value, allow_nan=False).encode()) > 200_000:
        raise ValueError('Preset is too large')
    return value


def clean_name(value):
    if not isinstance(value, str) or not value.strip() or len(value.strip()) > 80:
        raise ValueError('Use a preset name of 1–80 characters')
    return value.strip()


class PresetStore:
    def __init__(self, root=ROOT):
        self.root = Path(root)
        self.path = self.root / 'history.json'

    def read(self):
        with _LOCK:
            return json.loads(self.path.read_text()) if self.path.exists() else {'version': 1, 'entries': []}

    def write(self, data):
        self.root.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(prefix='.history-', dir=self.root)
        try:
            with os.fdopen(fd, 'w') as stream:
                json.dump(data, stream, ensure_ascii=False, allow_nan=False, indent=2)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(name, self.path)
        finally:
            if os.path.exists(name): os.unlink(name)

    def list(self):
        return {'entries': [{k: v for k, v in e.items() if k != 'settings'} for e in reversed(self.read()['entries'])]}

    def get(self, entry_id):
        return next((e for e in self.read()['entries'] if e['id'] == entry_id), None)

    def update(self, request):
        if not isinstance(request, dict): raise ValueError('Expected a preset request')
        action = request.get('action')
        with _LOCK:
            data = self.read()
            if action == 'save':
                settings = validate_settings(request.get('settings'))
                now = datetime.now(timezone.utc).isoformat()
                entry = {'id': uuid.uuid4().hex, 'name': clean_name(request.get('name')), 'createdAt': now,
                         'updatedAt': now, 'archived': False, 'assetHash': settings['assetHash'], 'settings': settings}
                data['entries'].append(entry)
            elif action in {'rename', 'archive'}:
                entry = next((e for e in data['entries'] if e['id'] == request.get('id')), None)
                if entry is None: raise KeyError('Preset not found')
                if action == 'rename': entry['name'] = clean_name(request.get('name'))
                else:
                    if not isinstance(request.get('archived'), bool): raise ValueError('Expected archive flag')
                    entry['archived'] = request['archived']
                entry['updatedAt'] = datetime.now(timezone.utc).isoformat()
            else: raise ValueError('Unknown preset action')
            self.write(data)
            return entry


def handle(method, query=None, body=None):
    store = PresetStore()
    if method == 'POST': return store.update(body)
    entry_id = (query or {}).get('id', [''])[0]
    if not entry_id: return store.list()
    entry = store.get(entry_id)
    if entry is None: raise KeyError('Preset not found')
    return entry
