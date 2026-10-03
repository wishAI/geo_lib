"""Conservative bitset polygon construction; output retains original vertices.

Cells intersecting a polygon are reserved, so raster search cannot introduce
positive-area overlap. This pass trades a little clearance for fast full layouts.
"""
import math
import time
import numpy as np
import shapely
from shapely.geometry import box


def _spread(value, count):
    """OR value >> k for k in [0, count), in logarithmic work."""
    result = value
    covered = 1
    while covered * 2 <= count:
        result |= result >> covered
        covered *= 2
    if covered < count:
        result |= result >> (count-covered)
    return result


def _mask(polygon, step):
    x0, y0, x1, y1 = polygon.bounds
    width, height = math.ceil((x1-x0)/step), math.ceil((y1-y0)/step)
    xx, yy = np.meshgrid(np.arange(width), np.arange(height))
    cells = shapely.box(x0+xx*step, y0+yy*step, x0+(xx+1)*step, y0+(yy+1)*step)
    filled = shapely.intersects(polygon, cells)
    rows, spans = [], []
    for row in filled:
        indices = np.flatnonzero(row)
        bits = 0
        runs = []
        for index in indices:
            index = int(index)
            bits |= 1 << index
            if runs and runs[-1][1] == index:
                runs[-1] = (runs[-1][0], index+1)
            else:
                runs.append((index, index+1))
        rows.append(bits)
        spans.append(runs)
    # Most occupied rows reject candidate positions first.
    scan = sorted(range(height), key=lambda i: rows[i].bit_count(), reverse=True)
    return width, height, rows, spans, scan


def raster_layout(boards, items, deadline, *, resolution=512, order_mode='area'):
    valid = [i for i,b in enumerate(boards) if b.polygon.equals(box(*b.bounds))]
    if not valid:
        return []
    step = max(max(boards[i].bounds[2]-boards[i].bounds[0], boards[i].bounds[3]-boards[i].bounds[1])
               for i in valid) / resolution
    grids = {i: (math.floor((boards[i].bounds[2]-boards[i].bounds[0])/step),
                 [0] * math.floor((boards[i].bounds[3]-boards[i].bounds[1])/step)) for i in valid}
    masks = {}
    placed = []
    def order_key(key):
        item = items[key]
        a,b,c,d = item.rotation_variants[0].bounds
        if order_mode == 'long_side':
            return (max(c-a,d-b), item.area)
        if order_mode == 'bbox':
            return (item.bbox_area, item.area)
        if order_mode == 'height':
            return (d-b, c-a)
        return (item.area, item.bbox_area)
    order = sorted(items, key=order_key, reverse=True)
    for key in order:
        if deadline is not None and time.perf_counter() >= deadline:
            break
        best = None
        for variant in items[key].rotation_variants:
            if deadline is not None and time.perf_counter() >= deadline:
                break
            a, b, c, d = variant.bounds
            if not any(c-a <= boards[bi].bounds[2]-boards[bi].bounds[0] and
                       d-b <= boards[bi].bounds[3]-boards[bi].bounds[1] for bi in valid):
                continue
            cache_key = variant.polygon.wkb
            if cache_key not in masks:
                masks[cache_key] = _mask(variant.polygon, step)
            width, height, rows, spans, scan = masks[cache_key]
            for bi in valid:
                nx, occupied = grids[bi]
                if width > nx or height > len(occupied):
                    continue
                allowed = (1 << (nx-width+1)) - 1
                for y in range(len(occupied)-height+1):
                    if best is not None and y+height > best[0][0]:
                        break
                    if y % 16 == 0 and deadline is not None and time.perf_counter() >= deadline:
                        break
                    available = allowed
                    for r in scan:
                        occupied_row = occupied[y+r]
                        if not occupied_row:
                            continue
                        for a, b in spans[r]:
                            available &= ~_spread(occupied_row >> a, b-a)
                        if not available:
                            break
                    if available:
                        x = (available & -available).bit_length()-1
                        rank = (y+height, x, bi)
                        if best is None or rank < best[0]:
                            best = (rank, bi, variant, x, y, rows)
                        break
        if best is None:
            continue
        _, bi, v, x, y, rows = best
        occupied = grids[bi][1]
        for r, bits in enumerate(rows):
            occupied[y+r] |= bits << x
        b = boards[bi].bounds
        placed.append((key, bi, v, b[0]+x*step-v.bounds[0], b[1]+y*step-v.bounds[1]))
    return placed
