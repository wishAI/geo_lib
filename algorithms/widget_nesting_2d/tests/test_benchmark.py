from types import SimpleNamespace

import pytest

from algorithms.widget_nesting_2d.benchmark import StripExceedsBoard, spyrrow_solution
from algorithms.widget_nesting_2d.problem import ProblemSpec
from algorithms.widget_nesting_2d.solver import SolverConfig, validate_solution


def test_spyrrow_probe_distinguishes_fit_from_inconclusive_width(monkeypatch):
    problem = ProblemSpec.from_json({
        'boards': [{'id': 'offset', 'polygon': {'shell': [[-5,-5],[5,-5],[5,5],[-5,5]]}}],
        'widgets': [{'id': 'square', 'quantity': 1, 'allowed_angles_degrees': [0],
                     'polygon': {'shell': [[3,7],[5,7],[5,9],[3,9]]}}]})
    result = SimpleNamespace(width=2., density=.2, placed_items=[
        SimpleNamespace(id='square#1', rotation=0., translation=(1.,1.))])
    fake = SimpleNamespace(Item=lambda *a: a, StripPackingConfig=lambda **kw: kw,
                           StripPackingInstance=lambda *a: SimpleNamespace(solve=lambda config: result))
    monkeypatch.setitem(__import__('sys').modules, 'spyrrow', fake)
    config = SolverConfig.from_problem(problem)
    solution = spyrrow_solution(problem, config, 2)
    validate_solution(problem, solution, config=config)
    assert solution.search_stats['fixed_sheet_fit']
    result.width = 12.
    result.placed_items[0].translation = (10.,1.)
    with pytest.raises(StripExceedsBoard) as caught:
        spyrrow_solution(problem, config, 2)
    assert caught.value.metrics['strip_validated']
    assert not caught.value.metrics['fixed_sheet_fit']
    assert caught.value.metrics['width_ratio'] == 1.2
    assert 'placed_area' not in caught.value.metrics


def test_spyrrow_enables_continuous_rotation_and_validates_actual_angle(monkeypatch):
    from shapely.affinity import rotate
    from shapely.geometry import box
    tilted = rotate(box(-5, -1, 5, 1), 17.321, origin=(0, 0))
    problem = ProblemSpec.from_json({
        'boards': [{'id': 'b', 'polygon': {'shell': [[0,0],[11,0],[11,2.001],[0,2.001]]}}],
        'widgets': [{'id': 'bar', 'quantity': 1,
                     'polygon': {'shell': [list(p) for p in tilted.exterior.coords[:-1]]}}]})
    angles = []
    def item(*args):
        angles.append(args[3])
        return args
    result = SimpleNamespace(width=10.001, density=.9, placed_items=[
        SimpleNamespace(id='bar#1', rotation=342.679, translation=(5.,1.))])
    fake = SimpleNamespace(Item=item, StripPackingConfig=lambda **kw: kw,
                           StripPackingInstance=lambda *a: SimpleNamespace(solve=lambda config: result))
    monkeypatch.setitem(__import__('sys').modules, 'spyrrow', fake)
    config = SolverConfig.from_problem(problem)
    solution = spyrrow_solution(problem, config, 2)
    assert angles == [None]
    validate_solution(problem, solution, config=config)


def test_opennest_uses_dense_angles_only_for_free_widgets(monkeypatch):
    from algorithms.widget_nesting_2d.benchmark import sdk_solution
    counts = []; options = {}
    fake = SimpleNamespace(
        nest_geo=lambda: SimpleNamespace(add_part=lambda *a, **kw: counts.append(kw['rotations'])),
        nest_sheets=lambda: SimpleNamespace(add_sheet=lambda *a, **kw: None))
    def engine(**kw):
        options.update(kw)
        return SimpleNamespace(solve=lambda *a: SimpleNamespace(placements=[]))
    fake.opennest = engine
    monkeypatch.setitem(__import__('sys').modules, 'compas_nest', fake)
    monkeypatch.setitem(__import__('sys').modules, 'compas.geometry', SimpleNamespace(Polyline=lambda p:p))
    polygon = {'shell': [[0,0],[2,0],[2,2],[0,2]]}
    problem = ProblemSpec.from_json({
        'boards': [{'id': 'b', 'polygon': polygon}],
        'widgets': [{'id':'free', 'quantity':1, 'polygon':polygon},
                    {'id':'fixed', 'quantity':1, 'polygon':polygon, 'allowed_angles_degrees':[0,180]}]})
    sdk_solution(problem, SolverConfig.from_problem(problem), 'opennest', 2)
    assert counts == [72, 2]
    assert options['rotations'] == 72


def test_free_benchmark_variants_release_constraints_without_changing_source():
    from algorithms.widget_nesting_2d.benchmark import benchmark_problem
    raw = {'widgets': [{'allowed_angles_degrees': [0], 'rotation_step_degrees': 90}]}
    free = benchmark_problem(raw, 'free')
    assert 'allowed_angles_degrees' not in free['widgets'][0]
    assert 'rotation_step_degrees' not in free['widgets'][0]
    assert raw['widgets'][0]['allowed_angles_degrees'] == [0]
    assert benchmark_problem(raw, 'discrete')['widgets'] == raw['widgets']


def test_spyrrow_rejects_impossible_free_height_before_native_call(monkeypatch):
    from algorithms.widget_nesting_2d.benchmark import Unsupported
    problem = ProblemSpec.from_json({
        'boards': [{'id': 'b', 'polygon': {'shell': [[0,0],[10,0],[10,1],[0,1]]}}],
        'widgets': [{'id': 'w', 'quantity': 1, 'polygon': {'shell': [[0,0],[2,0],[2,2],[0,2]]}}]})
    # No native methods are available; failure must happen before calling them.
    monkeypatch.setitem(__import__('sys').modules, 'spyrrow', SimpleNamespace())
    with pytest.raises(Unsupported, match='strip height'):
        spyrrow_solution(problem, SolverConfig.from_problem(problem), 2)
