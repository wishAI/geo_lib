import random

from shapely.affinity import translate

from algorithms.widget_nesting_2d.case_library import build_case_library
from algorithms.widget_nesting_2d.problem import ProblemSpec
from algorithms.widget_nesting_2d.solver import SolverConfig, solve_problem, validate_solution, _build_runtime, _build_placement
from algorithms.widget_nesting_2d.benchmark import _result_from_placements
from algorithms.widget_nesting_2d.raster_start import raster_layout, _spread


def test_bitset_collision_spreading_matches_exhaustive_shifts():
    rng=random.Random(91)
    for _ in range(100):
        value=rng.getrandbits(128)
        count=rng.randrange(1,65)
        expected=0
        for shift in range(count):
            expected |= value >> shift
        assert _spread(value,count)==expected


def test_envelope_constructor_respects_offset_board_and_rotation_restriction():
    p=ProblemSpec.from_json({'boards':[{'id':'b','polygon':{'shell':[[-5,20],[55,20],[55,120],[-5,120]]}}],
        'widgets':[{'id':'tile','quantity':100,'allowed_angles_degrees':[90],
                    'polygon':{'shell':[[0,0],[10,0],[10,6],[0,6]]}}]})
    solution=solve_problem(p,time_limit_seconds=10)
    validate_solution(p,solution)
    assert len(solution.placements)==100
    assert all(p.rotation_degrees==90 for p in solution.placements)
    assert solution.search_stats['termination_reason']=='constructive_full_fit'


def test_raster_constructor_retains_and_reuses_a_real_hole():
    p=ProblemSpec.from_json({'boards':[{'id':'b','polygon':{'shell':[[0,0],[8,0],[8,8],[0,8]]}}],
        'widgets':[{'id':'frame','quantity':1,'allowed_angles_degrees':[0],
                    'polygon':{'shell':[[0,0],[8,0],[8,8],[0,8]],'holes':[[[2,2],[6,2],[6,6],[2,6]]]}},
                   {'id':'plug','quantity':1,'allowed_angles_degrees':[0],
                    'polygon':{'shell':[[0,0],[3,0],[3,3],[0,3]]}}]})
    cfg=SolverConfig.from_problem(p); boards,items=_build_runtime(p,cfg)
    raw=raster_layout(boards,items,None,resolution=128)
    placements=[_build_placement(items[k],boards[bi],bi,v.angle_degrees,translate(v.polygon,xoff=x,yoff=y))
                for k,bi,v,x,y in raw]
    s=_result_from_placements(p,cfg,placements,'raster',1)
    validate_solution(p,s)
    assert len(s.placements)==2
    assert len(next(p for p in s.placements if p.widget_id=='frame').polygon.interiors)==1


def test_large_constructor_deadline_preserves_partial_incumbent(monkeypatch):
    from algorithms.widget_nesting_2d import solver, constructive
    case=build_case_library(include_large=True)['large_tiles_100']
    p=ProblemSpec.from_json(case.problem)
    clock=[0.]
    def partial_then_expire(boards,items,deadline,tolerance):
        key=next(iter(items));v=items[key].rotation_variants[0]
        clock[0]=2.
        yield [(key,0,v,boards[0].bounds[0]-v.bounds[0],boards[0].bounds[1]-v.bounds[1])]
    monkeypatch.setattr(constructive,'envelope_layouts',partial_then_expire)
    monkeypatch.setattr(solver.time,'perf_counter',lambda:clock[0])
    s=solve_problem(p,time_limit_seconds=1)
    validate_solution(p,s)
    assert len(s.placements)==1
    assert len(s.skipped_item_ids)==99
    assert s.search_stats['budget_exhausted']


def test_mixed_catalog_has_real_diversity_and_valid_independent_witness():
    from shapely import normalize
    lib=build_case_library(include_diverse=True)
    case=lib['diverse_256']
    p=ProblemSpec.from_json(case.problem)
    assert len(p.widgets)==256
    assert len(case.problem['source']['families'])==17
    assert len({normalize(w.polygon.to_polygon(name=w.widget_id)).wkb for w in p.widgets})==256
    cfg=SolverConfig.from_problem(p);boards,items=_build_runtime(p,cfg)
    placements=[]
    for r in case.witness['placements']:
        item=items[r['item']];v=item.rotation_variants[0]
        placements.append(_build_placement(item,boards[0],0,0,
            translate(v.polygon,xoff=r['x']-v.bounds[0],yoff=r['y']-v.bounds[1])))
    s=_result_from_placements(p,cfg,placements,'independent_witness',0)
    validate_solution(p,s)
    assert len(s.placements)==256
    assert case.problem['source']['known_feasible']
    assert 'witness' not in case.problem
    assert not lib['source_mix_66_dense'].problem['source']['known_feasible']
