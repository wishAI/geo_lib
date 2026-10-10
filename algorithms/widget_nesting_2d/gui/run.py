"""Capture immutable GUI run inputs, parameters, revision, and validation outcome."""
import json
import hashlib
import subprocess
import sys
import signal
from datetime import datetime, timezone
from pathlib import Path

from .. import solver


def main():
    args = solver._parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    raw = json.loads(Path(args.input).read_text())
    overrides = {key: getattr(args, key) for key in ('seed', 'beam_width', 'population_size', 'generations', 'rotation_step_degrees', 'candidate_mode') if getattr(args, key) is not None}
    raw.setdefault('config', {}).update(overrides)
    (output / 'problem.json').write_text(json.dumps(raw, indent=2) + '\n')
    revision = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True)
    run = {'id': output.name, 'caseId': Path(args.input).stem, 'startedAt': datetime.now(timezone.utc).isoformat(),
           'sourceRevision': revision.stdout.strip() or None, 'parameters': raw['config'],
           'inputSha256': hashlib.sha256((output / 'problem.json').read_bytes()).hexdigest(),
           'sourceSha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(solver.__file__).parent.glob('*.py'))},
           'timeLimitSeconds': args.time_limit_seconds, 'status': 'running', 'validation': 'pending'}
    def save():
        (output / 'run.json').write_text(json.dumps(run, indent=2) + '\n')
    save()
    # The solver emits and independently validates the exact placement polygons.
    original = sys.argv
    sys.argv = [original[0], *original[1:]]
    sys.argv[sys.argv.index('--input') + 1] = str(output / 'problem.json')
    def interrupted(signum, frame):
        raise KeyboardInterrupt("Run cancelled")
    signal.signal(signal.SIGTERM, interrupted)
    try:
        solver.main()
        run.update(status='succeeded', validation='passed')
    except BaseException as error:
        run.update(status='cancelled' if isinstance(error, KeyboardInterrupt) else 'failed', validation='not_passed', error=str(error))
        raise
    finally:
        sys.argv = original
        run['finishedAt'] = datetime.now(timezone.utc).isoformat()
        save()


if __name__ == '__main__':
    main()
