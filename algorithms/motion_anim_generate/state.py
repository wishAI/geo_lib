"""Atomic task-local progress and GUI evolution records; no external integration."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'outputs'
REPO = ROOT.parents[1]


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def progress(status, **fields):
    path = OUT / 'backend_progress.json'
    old = json.loads(path.read_text()) if path.exists() else {}
    old.update(fields)
    old.update(status=status, updated_at=datetime.now(timezone.utc).isoformat(),
               source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
               process={'pid_namespace': os.getpid(), 'host_pid': 'unavailable in Codex namespace'},
               pins=json.loads((ROOT / 'provenance.json').read_text()))
    write_json(path, old)


def artifact(path, kind='json'):
    return {'path': str(Path(path).relative_to(REPO)), 'kind': kind, 'label': Path(path).name}


def node(node_id, parent_ids, status, **fields):
    path = OUT / 'evolution.json'
    data = json.loads(path.read_text()) if path.exists() else {
        'schemaVersion': 1, 'type': 'evolutionTree', 'lineage': 'kimodo_landau_v10',
        'primaryMetric': 'retarget_rmse_m',
        'milestones': [], 'nodes': [], 'visibleNodeBudget': 40}
    value = {'id': node_id, 'parentIds': parent_ids, 'status': status,
             'kind': 'experiment', 'label': node_id, 'metrics': {}, 'artifacts': [], **fields}
    data['nodes'] = [n for n in data['nodes'] if n['id'] != node_id] + [value]
    data.update(generatedAt=datetime.now(timezone.utc).isoformat(), currentNodeId=node_id,
                defaultVisibleNodeIds=[n['id'] for n in data['nodes']], overviewNodes=data['nodes'],
                summary={'nodeCount': len(data['nodes']), 'failedCount': sum(n['status'] == 'failed' for n in data['nodes'])})
    write_json(path, data)
