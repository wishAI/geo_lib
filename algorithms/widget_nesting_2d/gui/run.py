"""Capture one immutable input and every requested algorithm result for a GUI run."""
import argparse
import hashlib
import json
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from .. import solver


def parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--algorithms', choices=['all', 'nfp'], default='nfp')
    gui, remaining = parser.parse_known_args()
    original = sys.argv
    try:
        sys.argv = [original[0], *remaining]
        args = solver._parse_args()
    finally:
        sys.argv = original
    args.algorithms = ['nfp', 'opennest', 'spyrrow'] if gui.algorithms == 'all' else ['nfp']
    return args


def execute_engine(args, output, engine, seed):
    """Bound each adapter in its own process; cancellation cannot orphan it."""
    result = output / f'{engine}_{seed}.json'
    command = [sys.executable, '-m', 'algorithms.widget_nesting_2d.benchmark', '--worker',
               '--problem', str(output / 'problem.json'), '--result', str(result),
               '--engine', engine, '--seed', str(seed), '--matched-seconds', str(args.time_limit_seconds),
               '--sdk-seconds', str(args.time_limit_seconds), '--preserve-search-config', '--render']
    started = datetime.now(timezone.utc).isoformat()
    clock = time.perf_counter()
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    timeout = max(15., args.time_limit_seconds + 10.)
    try:
        _, stderr = process.communicate(timeout=timeout)
        if process.returncode or not result.exists():
            row = {'status': 'error', 'reason': stderr[-2000:] or f'Worker exited {process.returncode} without a result'}
        else:
            row = json.loads(result.read_text())
    except subprocess.TimeoutExpired:
        process.kill()
        process.communicate()
        row = {'status': 'timeout', 'reason': f'Exceeded {timeout:g}s hard process limit'}
    except BaseException:
        process.kill()
        process.communicate()
        raise
    row.update(engine=engine, seed=seed, startedAt=started,
               finishedAt=datetime.now(timezone.utc).isoformat(), wall_seconds=time.perf_counter()-clock)
    return row


def main():
    args = parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    raw = json.loads(Path(args.input).read_text())
    overrides = {key: getattr(args, key) for key in ('seed', 'beam_width', 'population_size', 'generations', 'rotation_step_degrees', 'rotation_mode', 'candidate_mode') if getattr(args, key) is not None}
    raw.setdefault('config', {}).update(overrides)
    # Every adapter receives this exact snapshot, including original widget angle constraints.
    (output / 'problem.json').write_text(json.dumps(raw, indent=2) + '\n')
    seed = raw['config'].get('seed', 7)
    revision = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True)
    run = {'id': output.name, 'caseId': Path(args.input).stem, 'startedAt': datetime.now(timezone.utc).isoformat(),
           'sourceRevision': revision.stdout.strip() or None, 'parameters': raw['config'],
           'inputSha256': hashlib.sha256((output / 'problem.json').read_bytes()).hexdigest(),
           'sourceSha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(solver.__file__).parent.glob('*.py'))},
           'timeLimitSeconds': args.time_limit_seconds, 'status': 'running', 'validation': 'pending',
           'algorithms': args.algorithms, 'results': [],
           'contract': {'matched_seconds': args.time_limit_seconds, 'seeds': [seed],
                        'budget_note': 'Same requested time budget and captured input; local solver retains the selected population and generation limits. Inspect actual timings.',
                        'rotation_note': 'Original per-widget restrictions preserved. OpenNest approximates unconstrained free rotation in 5 degree steps.',
                        'spyrrow_contract': 'All-items strip feasibility probe; an oversized strip is inconclusive, with no partial-area score.'}}
    def save():
        temporary = output / 'run.json.tmp'
        temporary.write_text(json.dumps(run, indent=2) + '\n')
        temporary.replace(output / 'run.json')
    save()
    def interrupted(signum, frame):
        raise KeyboardInterrupt('Run cancelled')
    previous_handler = signal.signal(signal.SIGTERM, interrupted)
    try:
        for engine in args.algorithms:
            print(f'Running {engine} on captured input, seed {seed}…', flush=True)
            row = execute_engine(args, output, engine, seed)
            run['results'].append(row)
            save()
            print(json.dumps(row), flush=True)
            if engine == 'nfp' and row['status'] == 'valid':
                shutil.copyfile(output / f'nfp_{seed}.solution.json', output / 'solution.json')
                image = output / f'nfp_{seed}.png'
                if image.exists():
                    shutil.copyfile(image, output / 'nesting_layout.png')
        run.update(status='succeeded', validation='passed' if all(r['status'] == 'valid' for r in run['results']) else 'see_algorithm_results')
    except BaseException as error:
        status = 'cancelled' if isinstance(error, KeyboardInterrupt) else 'failed'
        run.update(status=status, validation='not_passed', error=str(error))
        completed = {r['engine'] for r in run['results']}
        for engine in args.algorithms:
            if engine not in completed:
                run['results'].append({'engine': engine, 'seed': seed, 'status': status,
                                       'reason': 'Comparison interrupted before this algorithm completed'})
        raise
    finally:
        signal.signal(signal.SIGTERM, previous_handler)
        run['finishedAt'] = datetime.now(timezone.utc).isoformat()
        save()


if __name__ == '__main__':
    main()
