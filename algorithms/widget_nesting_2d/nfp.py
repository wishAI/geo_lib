"""Cached configuration-space obstacles from convex Minkowski sums.

Constrained triangulation preserves concavities and holes. These are floating
point polygonal NFPs, not an exact-arithmetic or continuous-rotation solver.
"""
from functools import lru_cache

from shapely import constrained_delaunay_triangles, from_wkb
from shapely.affinity import translate
from shapely.geometry import MultiPoint, box
from shapely.ops import unary_union


@lru_cache(maxsize=512)
def _convex_parts(wkb):
    polygon = from_wkb(wkb)
    if polygon.equals(polygon.convex_hull):
        return (tuple(polygon.exterior.coords[:-1]),)
    triangles = constrained_delaunay_triangles(polygon)
    return tuple(tuple(t.exterior.coords[:-1]) for t in triangles.geoms)


@lru_cache(maxsize=2048)
def _obstacle(fixed_wkb, moving_wkb):
    # A (+) -B is the set of translations where B intersects A.
    hulls = [MultiPoint([(ax-bx, ay-by) for ax, ay in a for bx, by in b]).convex_hull
             for a in _convex_parts(fixed_wkb) for b in _convex_parts(moving_wkb)]
    return unary_union(hulls)


def _points(geometry):
    if geometry.is_empty:
        return
    if geometry.geom_type == 'Polygon':
        yield from geometry.exterior.coords
        for ring in geometry.interiors:
            yield from ring.coords
    elif hasattr(geometry, 'geoms'):
        for part in geometry.geoms:
            yield from _points(part)
    else:
        yield from geometry.coords


@lru_cache(maxsize=256)
def _board_obstacle(board_wkb, moving_wkb):
    board, moving = from_wkb(board_wkb), from_wkb(moving_wkb)
    x0, y0, x1, y1 = board.bounds
    a, b, c, d = moving.bounds
    margin = max(c-a, d-b, 1.0) * 2
    outside = box(x0-margin, y0-margin, x1+margin, y1+margin).difference(board)
    return _obstacle(outside.wkb, moving_wkb)


def translation_candidates(board, moving, placements, tolerance):
    """Return feasible-region vertices, including near-degenerate contact fits.

    A tiny inward obstacle offset exposes zero-clearance contacts. Every returned
    translation must still pass the solver's independent containment/overlap test.
    """
    x0, y0, x1, y1 = board.bounds
    a, b, c, d = moving.bounds
    lo_x, lo_y, hi_x, hi_y = x0-a, y0-b, x1-c, y1-d
    if hi_x < lo_x-tolerance or hi_y < lo_y-tolerance:
        return []
    eps = min(tolerance / (100 * max(x1-x0, y1-y0, 1.0)), 1e-9)
    domain = box(lo_x-eps, lo_y-eps, max(lo_x, hi_x)+eps, max(lo_y, hi_y)+eps)
    exact_domain = box(lo_x, lo_y, max(lo_x, hi_x), max(lo_y, hi_y))
    points = [(lo_x, lo_y), (lo_x, hi_y), (hi_x, lo_y), (hi_x, hi_y)]
    obstacles = []
    if not board.equals(box(*board.bounds)):
        obstacles.append(_board_obstacle(board.buffer(eps, join_style=2).wkb, moving.wkb))
    for fixed, dx, dy in placements:
        obstacles.append(translate(_obstacle(fixed.wkb, moving.wkb), xoff=dx, yoff=dy))
    if obstacles:
        # Offset individually: unioning closed obstacles first can erase a
        # feasible zero-width contact corridor between two different pieces.
        exact_forbidden = unary_union(obstacles)
        points.extend(_points(exact_forbidden.boundary.intersection(exact_domain)))
        forbidden = unary_union([ob.buffer(-eps, join_style=2) for ob in obstacles])
        domain = domain.difference(forbidden)
    points.extend(_points(domain))
    return points


def cache_info():
    info = _obstacle.cache_info()
    return {'hits': info.hits, 'misses': info.misses, 'size': info.currsize}


def clear_caches():
    _obstacle.cache_clear()
    _board_obstacle.cache_clear()
    _convex_parts.cache_clear()
