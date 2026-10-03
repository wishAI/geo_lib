"""Isolated, timeout-bounded fixed-sheet comparisons with optional native SDKs."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import importlib.metadata
import hashlib
import importlib.util
import json
import math
import platform
import subprocess
import sys
import time
from pathlib import Path

from shapely.affinity import rotate, translate
from shapely.geometry import box

from .case_library import build_case_library
from .problem import ProblemSpec, save_solution
from .solver import (SolverConfig, SolutionResult, Placement, _build_runtime, _initial_state,
                     _apply_placement, _board_metrics, solve_problem, solution_to_dict, validate_solution)


class Unsupported(ValueError):
    pass


class StripExceedsBoard(ValueError):
    def __init__(self, metrics):
        self.metrics = metrics
        super().__init__("Validated all-items strip exceeds the requested sheet width; fixed-sheet feasibility is inconclusive")


def spyrrow_solution(problem, config, seconds):
    import spyrrow
    if len(problem.boards) != 1:
        raise Unsupported("Spyrrow probe requires one rectangular sheet")
    board = problem.boards[0].polygon.to_polygon(name="board")
    if not board.equals(box(*board.bounds)) or any(w.polygon.holes for w in problem.widgets):
        raise Unsupported("Spyrrow does not support holes or nonrectangular sheets")
    if seconds < 1 or not float(seconds).is_integer():
        raise Unsupported("Spyrrow accepts integer second budgets")
    _, items = _build_runtime(problem, config)
    x0, y0, x1, y1 = board.bounds
    if any(all(v.bounds[3]-v.bounds[1] > y1-y0 for v in item.rotation_variants) for item in items.values()):
        raise Unsupported("No permitted orientation fits the strip height for at least one item")
    parts = [spyrrow.Item(i.instance_id, list(i.polygon_local.exterior.coords[:-1]), 1,
                         [v.angle_degrees for v in i.rotation_variants]) for i in items.values()]
    result = spyrrow.StripPackingInstance("geo_lib", y1-y0, parts).solve(
        spyrrow.StripPackingConfig(total_computation_time=int(seconds), num_workers=1,
                                  seed=config.seed, early_termination=False))
    if not math.isfinite(result.width) or result.width <= 0:
        raise AssertionError("SDK returned an invalid strip width")
    placements = []
    for raw in result.placed_items:
        item = items[raw.id]
        polygon = translate(rotate(item.polygon_local, raw.rotation, origin=(0,0)),
                            raw.translation[0]+x0, raw.translation[1]+y0)
        placements.append(Placement(item.instance_id, item.widget_id, problem.boards[0].board_id,
                                    0, raw.rotation, polygon, polygon.centroid.x, polygon.centroid.y))
    strip_raw = problem.to_json()
    strip_raw['boards'][0]['polygon'] = {'shell': [[x0,y0],[x0+result.width,y0],
                                                 [x0+result.width,y1],[x0,y1]]}
    strip_problem = ProblemSpec.from_json(strip_raw)
    strip_solution = _result_from_placements(strip_problem, config, placements, 'spyrrow', seconds)
    validate_solution(strip_problem, strip_solution, tolerance=config.placement_tolerance, config=config)
    if strip_solution.skipped_item_ids:
        raise AssertionError("Spyrrow omitted requested items from its strip solution")
    metrics = {'required_strip_width': result.width, 'sheet_width': x1-x0,
               'width_ratio': result.width/(x1-x0), 'strip_density': result.density,
               'strip_validated': True, 'fixed_sheet_fit': result.width <= x1-x0+config.placement_tolerance}
    if not metrics['fixed_sheet_fit']:
        raise StripExceedsBoard(metrics)
    solution = _result_from_placements(problem, config, placements, 'spyrrow', seconds)
    solution.search_stats.update(metrics)
    return solution


def _result_from_placements(problem, config, placements, engine, seconds):
    boards, items = _build_runtime(problem, config)
    state = _initial_state(boards)
    for p in placements:
        state = _apply_placement(state, p)
    placed_ids = {p.item_instance_id for p in placements}
    skipped = tuple(i for i in items if i not in placed_ids)
    metrics = tuple(_board_metrics(b, bs) for b, bs in zip(boards, state.board_states))
    return SolutionResult(tuple(items), tuple(placements), skipped, sum(p.area for p in placements),
                          sum(items[i].area for i in skipped),
                          max(m['max_rest_rectangle_area'] for m in metrics),
                          sum(m['sum_rest_rectangle_area'] for m in metrics), metrics,
                          {'engine': engine, 'time_budget_seconds': seconds})


def sdk_solution(problem, config, engine, seconds):
    if engine == 'spyrrow':
        return spyrrow_solution(problem, config, seconds)
    boards, items = _build_runtime(problem, config)
    item_list = list(items.values())
    placements = []
    if engine == 'opennest':
        from compas.geometry import Polyline
        from compas_nest import nest_geo, nest_sheets, opennest
        def line(coords):
            pts = [[float(x), float(y), 0] for x, y in coords]
            if pts[0] != pts[-1]:
                pts.append(pts[0])
            return Polyline(pts)
        parts, sheets = nest_geo(), nest_sheets()
        for item in item_list:
            angles = sorted(v.angle_degrees for v in item.rotation_variants)
            n = len(angles)
            if any(abs(a-i*360/n) > 1e-5 for i, a in enumerate(angles)):
                raise Unsupported('OpenNest accepts uniform rotation counts, not this angle list')
            p = item.polygon_local
            parts.add_part(line(p.exterior.coords), holes=[line(r.coords) for r in p.interiors], rotations=n)
        for b in boards:
            sheets.add_sheet(line(b.polygon.exterior.coords), holes=[line(r.coords) for r in b.polygon.interiors])
        result = opennest(generations=100000, rotations=4, spacing=0, seed=config.seed,
                          num_seeds=1, use_parallel=False, exact_nfp=True, use_holes=True,
                          time_budget_secs=seconds, verbose=False).solve(parts, sheets)
        for raw in result.placements:
            if int(raw["sheet_id"]) == -1:
                continue
            i, board_index = int(raw['part_index']), int(raw['sheet_id'])
            if not 0 <= i < len(item_list) or not 0 <= board_index < len(boards):
                raise AssertionError('SDK used an unrequested item or sheet')
            item = item_list[i]
            angle = math.degrees(raw['angle']) % 360
            polygon = translate(rotate(item.polygon_local, angle, origin=(0,0)), raw['tx'] + result.sheet_origins[board_index][0],
                                raw['ty'] + result.sheet_origins[board_index][1])
            placements.append(Placement(item.instance_id, item.widget_id, boards[board_index].board_id,
                                        board_index, angle, polygon, polygon.centroid.x, polygon.centroid.y))
    else:
        raise Unsupported(engine)
    return _result_from_placements(problem, config, placements, engine, seconds)


def run_worker(args):
    raw = json.loads(Path(args.problem).read_text())
    problem = ProblemSpec.from_json(raw)
    config = SolverConfig.from_problem(problem, {'seed': args.seed, 'placement_tolerance': 1e-5,
                                                 'candidate_mode': 'contact' if args.engine == 'contact' else 'nfp'})
    if args.matched_seconds is not None and args.engine in ('nfp', 'contact'):
        from dataclasses import replace
        config = replace(config, population_size=12, generations=100000)
    started = time.perf_counter()
    row = {'engine': args.engine, 'seed': args.seed}
    try:
        if args.engine == 'legacy':
            spec = importlib.util.spec_from_file_location('algorithms.widget_nesting_2d.legacy', args.legacy_solver)
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            legacy_config = module.SolverConfig.from_problem(problem, {'seed': args.seed, 'placement_tolerance': 1e-5})
            solution = module.solve_problem(problem, legacy_config)
        elif args.engine in ('nfp', 'contact'):
            from .nfp import clear_caches, cache_info
            clear_caches()
            solution = solve_problem(problem, config, time_limit_seconds=args.matched_seconds)
            row['nfp_cache'] = cache_info()
        else:
            solution = sdk_solution(problem, config, args.engine, args.sdk_seconds)
        row['solve_seconds'] = time.perf_counter()-started
        validation_start = time.perf_counter()
        validate_solution(problem, solution, tolerance=1e-5, config=config)
        row['validation_seconds'] = time.perf_counter()-validation_start
        if args.engine == 'spyrrow':
            row['strip_metrics'] = {k:v for k,v in solution.search_stats.items() if k not in ('engine', 'time_budget_seconds')}
        row.update(status='valid', score=solution_to_dict(problem, solution)['score'],
                   placed_count=len(solution.placements), requested_count=sum(w.quantity for w in problem.widgets))
        save_solution(Path(args.result).with_suffix('.solution.json'), solution_to_dict(problem, solution))
        from .render import render_solution_image
        # Timing excludes rendering, while parent hard timeout includes startup/render.
        row['solve_validate_seconds'] = time.perf_counter()-started
        if args.render:
            render_solution_image(problem, solution, Path(args.result).with_suffix('.png'))
    except StripExceedsBoard as exc:
        row.update(status='strip_exceeds_board', reason=str(exc), strip_metrics=exc.metrics)
    except Unsupported as exc:
        row.update(status='unsupported', reason=str(exc))
    except Exception as exc:
        row.update(status='error', reason=f'{type(exc).__name__}: {exc}')
    row.setdefault('solve_validate_seconds', time.perf_counter()-started)
    save_solution(args.result, row)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', default='algorithms/widget_nesting_2d/outputs/benchmark')
    parser.add_argument('--case', action='append', default=[])
    parser.add_argument('--large', action='store_true', help='Include opt-in scenes with 100–1000 widgets')
    parser.add_argument('--diverse', action='store_true', help='Include mixed scenes with 66–256 distinct outlines')
    parser.add_argument('--engines', nargs='+', default=['contact', 'nfp', 'opennest'])
    parser.add_argument('--seeds', nargs='+', type=int, default=[7, 42])
    parser.add_argument('--timeout', type=float, default=30)
    parser.add_argument('--matched-seconds', type=float, default=None, help='Soft search budget for local and SDK engines')
    parser.add_argument('--sdk-seconds', type=float, default=2)
    parser.add_argument('--legacy-solver')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--jobs', type=int, default=1)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--problem')
    parser.add_argument('--result')
    parser.add_argument('--engine')
    parser.add_argument('--seed', type=int, default=7)
    args = parser.parse_args()
    if args.matched_seconds is not None:
        if not math.isfinite(args.matched_seconds) or args.matched_seconds <= 0:
            parser.error("--matched-seconds must be finite and positive")
        args.sdk_seconds = args.matched_seconds
    if args.worker:
        run_worker(args)
        return
    library = build_case_library(include_large=args.large, include_diverse=args.diverse)
    root = Path(args.output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    versions = {}
    for name in ['shapely', 'compas_nest', 'spyrrow']:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    source_hashes = {name: hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
                     for name in ('solver.py', 'nfp.py', 'constructive.py', 'raster_start.py',
                                  'problem.py', 'benchmark.py', 'case_library.py', 'diverse_cases.py')}
    if args.legacy_solver:
        source_hashes['legacy_solver.py'] = hashlib.sha256(Path(args.legacy_solver).read_bytes()).hexdigest()
    report = {'source_sha256': source_hashes, 'platform': platform.platform(), 'python': sys.version, 'versions': versions,
              'contract': {'objective': 'fixed sheets: maximize placed area, then preserved edge rectangle',
                           'tolerance': 1e-5, 'seeds': args.seeds, 'cold_cache': True,
                           'hard_timeout_seconds': args.timeout, 'sdk_budget_seconds': args.sdk_seconds,
                           'matched_seconds': args.matched_seconds,
                           'spyrrow_contract': 'Separate all-items strip feasibility probe; exceeding width is inconclusive, never a partial-area comparison.',
                           'budget_note': ('Shared soft search budget; inspect actual solve time and outer timeouts. Local population 12, generation cap 100000.' if args.matched_seconds is not None else 'Local engines use fixed search counts; SDK uses wall-time. Not equal-time quality claims.'),
                           'legacy_solver': args.legacy_solver}, 'results': []}
    report['contract']['concurrent_jobs'] = args.jobs
    if args.resume and (root / 'index.json').exists():
        previous = json.loads((root / 'index.json').read_text())
        if previous.get('source_sha256') != report['source_sha256'] or previous.get('contract') != report['contract']:
            raise ValueError('Resume requires identical source hashes and benchmark contract; use a new output root')
        report['results'] = previous['results']
    done = {(r['case_id'], r['seed'], r['engine']) for r in report['results'] if r['status'] != 'error'}
    report['results'] = [r for r in report['results'] if r['status'] != 'error']
    jobs = []
    for case_id in args.case or library:
        case = library[case_id]
        directory = root / case_id
        directory.mkdir(exist_ok=True)
        problem_path = directory / 'problem.json'
        if args.resume and problem_path.exists():
            from .problem import _round_value
            if json.loads(problem_path.read_text()) != _round_value(case.problem):
                raise ValueError(f'Resume problem changed: {case_id}')
        save_solution(problem_path, case.problem)
        if case.witness:
            save_solution(directory/'feasibility_witness.json', case.witness)
        for seed in args.seeds:
            for engine in args.engines:
                if (case_id, seed, engine) in done:
                    continue
                result = directory / f'{engine}_{seed}.json'
                cmd = [sys.executable, '-X', 'faulthandler', '-m', 'algorithms.widget_nesting_2d.benchmark', '--worker',
                       '--problem', str(problem_path), '--result', str(result), '--engine', engine,
                       '--seed', str(seed), '--sdk-seconds', str(args.sdk_seconds)]
                if args.matched_seconds is not None:
                    cmd.extend(['--matched-seconds', str(args.matched_seconds)])
                if args.render:
                    cmd.append('--render')
                if args.legacy_solver:
                    cmd.extend(['--legacy-solver', args.legacy_solver])
                jobs.append((cmd, result, case_id, seed, engine))
    def execute(job):
        cmd, result, case_id, seed, engine = job
        result.unlink(missing_ok=True)
        start = time.perf_counter()
        try:
            completed = subprocess.run(cmd, capture_output=True, text=True, timeout=args.timeout)
            if completed.returncode or not result.exists():
                row = {'status': 'error', 'returncode': completed.returncode,
                       'reason': completed.stderr[-2000:] or f'Worker exited with code {completed.returncode} without a result'}
            else:
                row = json.loads(result.read_text())
        except subprocess.TimeoutExpired:
            row = {'status': 'timeout', 'reason': f'exceeded {args.timeout}s hard process limit'}
        row.update(case_id=case_id, seed=seed, engine=engine, wall_seconds=time.perf_counter()-start)
        return row
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        for future in as_completed([pool.submit(execute, job) for job in jobs]):
            row = future.result()
            report['results'].append(row)
            save_solution(root / 'index.json', report)
            print(json.dumps(row), flush=True)



if __name__ == '__main__':
    main()
