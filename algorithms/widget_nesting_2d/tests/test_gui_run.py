import hashlib
import json
import subprocess
from types import SimpleNamespace

import pytest

from algorithms.widget_nesting_2d.gui import run


def arguments(tmp_path):
    source = tmp_path / 'input.json'
    source.write_text(json.dumps({'boards': [], 'widgets': [], 'config': {'seed': 9}}))
    return SimpleNamespace(input=str(source), output=str(tmp_path / 'captured'), seed=42,
                           beam_width=3, population_size=4, generations=2, rotation_step_degrees=15,
                           rotation_mode='free', candidate_mode=None, time_limit_seconds=2,
                           algorithms=['nfp', 'opennest', 'spyrrow'])


def test_comparison_captures_one_input_and_retains_all_engine_outcomes(tmp_path, monkeypatch):
    args = arguments(tmp_path)
    monkeypatch.setattr(run, 'parse_args', lambda: args)
    inputs = []
    def engine(args, output, engine, seed):
        inputs.append((output / 'problem.json').read_bytes())
        if engine == 'nfp':
            (output / 'nfp_42.solution.json').write_text('{"placements": []}')
        return {'engine': engine, 'seed': seed, 'status': 'valid' if engine == 'nfp' else 'unsupported'}
    monkeypatch.setattr(run, 'execute_engine', engine)
    run.main()
    output = tmp_path / 'captured'
    record = json.loads((output / 'run.json').read_text())
    assert len(set(inputs)) == 1
    assert record['inputSha256'] == hashlib.sha256(inputs[0]).hexdigest()
    assert [r['engine'] for r in record['results']] == args.algorithms
    assert record['results'][1]['status'] == 'unsupported'
    assert record['startedAt'] < record['finishedAt']
    assert record['parameters']['population_size'] == 4
    assert record['parameters']['seed'] == 42
    assert record['validation'] == 'see_algorithm_results'
    assert not (output / 'opennest_42.solution.json').exists()


def test_cancelled_comparison_retains_completed_results_and_marks_remaining(tmp_path, monkeypatch):
    args = arguments(tmp_path)
    monkeypatch.setattr(run, 'parse_args', lambda: args)
    def engine(args, output, engine, seed):
        if engine != 'nfp':
            raise KeyboardInterrupt()
        (output / 'nfp_42.solution.json').write_text('{}')
        return {'engine': engine, 'seed': seed, 'status': 'valid'}
    monkeypatch.setattr(run, 'execute_engine', engine)
    with pytest.raises(KeyboardInterrupt):
        run.main()
    record = json.loads((tmp_path / 'captured' / 'run.json').read_text())
    assert record['status'] == 'cancelled'
    assert [r['status'] for r in record['results']] == ['valid', 'cancelled', 'cancelled']
    assert record['finishedAt']


def test_timeout_kills_worker_and_records_no_fabricated_score(tmp_path, monkeypatch):
    args = arguments(tmp_path)
    class Process:
        killed = False
        def communicate(self, timeout=None):
            if timeout:
                raise subprocess.TimeoutExpired('worker', timeout)
            return '', ''
        def kill(self):
            self.killed = True
    process = Process()
    monkeypatch.setattr(run.subprocess, 'Popen', lambda *a, **kw: process)
    row = run.execute_engine(args, tmp_path, 'opennest', 42)
    assert process.killed
    assert row['status'] == 'timeout'
    assert row['seed'] == 42
    assert 'score' not in row
