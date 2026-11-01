"""Validated six-face crop edits, stored separately from immutable world data."""
from __future__ import annotations
import json
import os
from pathlib import Path
import threading

KEYS = ('x0', 'y0', 'z0', 'x1', 'y1', 'z1')


def validate_bounds(value, town):
    if not isinstance(value, dict) or set(value) != set(KEYS):
        raise ValueError('Provide all six bounds: x0, y0, z0, x1, y1, z1')
    if any(type(value[k]) is not int for k in KEYS):
        raise ValueError('Bounds must be whole block coordinates')
    q = {k: value[k] for k in KEYS}
    for axis in 'xyz':
        if q[axis+'0'] >= q[axis+'1']:
            raise ValueError(f'{axis.upper()} minimum must be below maximum')
    if not 0 <= q['y0'] < q['y1'] <= 256:
        raise ValueError('Bilicraft Y bounds must lie between 0 and 256')
    for axis in 'xz':
        if q[axis+'0'] < town[axis+'0'] or q[axis+'1'] > town[axis+'1']:
            raise ValueError('Bounds must stay inside the extracted Luna Town region')
    if (q['x1']-q['x0'])*(q['z1']-q['z0']) > 131072:
        raise ValueError('A building box is limited to 131,072 columns; reduce X or Z extent')
    return q


class BoundsStore:
    def __init__(self, root, report):
        self.path = Path(root)/'box_overrides.json'
        self.report = report
        self.ids = {b['id'] for b in report['buildings']}
        self.lock = threading.RLock()

    def read(self):
        with self.lock:
            if not self.path.exists():
                return {'schema':'geo.minecraft-building-bounds/v1','buildings':{}}
            result = json.loads(self.path.read_text())
            if result.get('schema') != 'geo.minecraft-building-bounds/v1':
                raise ValueError('Unsupported bounds index')
            for key, value in result['buildings'].items():
                if key not in self.ids: raise ValueError('Unknown building in bounds index')
                validate_bounds(value, self.report['bounds'])
            return result

    def save(self, payload):
        if not isinstance(payload, dict) or payload.get('id') not in self.ids:
            raise ValueError('Unknown building')
        key = payload['id']
        with self.lock:
            result = self.read()
            if payload.get('reset') is True:
                result['buildings'].pop(key, None)
            else:
                result['buildings'][key] = validate_bounds(payload.get('bounds'), self.report['bounds'])
            temporary = self.path.with_suffix(f'.{os.getpid()}.tmp')
            temporary.write_text(json.dumps(result, indent=2)+'\n')
            temporary.replace(self.path)
            return result
