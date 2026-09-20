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
