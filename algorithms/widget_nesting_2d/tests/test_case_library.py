from __future__ import annotations

from algorithms.widget_nesting_2d.case_library import build_case_library
from algorithms.widget_nesting_2d.problem import ProblemSpec


def test_case_library_contains_realistic_benchmark_cases() -> None:
    cases = build_case_library()
    assert "shirts_combo" in cases
    assert "trousers_shortage" in cases
    assert "swim_curved_mix" in cases

    for case_id in ["shirts_combo", "trousers_shortage", "swim_curved_mix"]:
        problem = ProblemSpec.from_json(cases[case_id].problem)
        assert len(problem.widgets) >= 3
        assert len(problem.boards) >= 1
