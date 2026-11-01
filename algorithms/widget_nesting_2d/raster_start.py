"""Conservative bitset polygon construction; output retains original vertices.

Cells intersecting a polygon are reserved, so raster search cannot introduce
positive-area overlap. This pass trades a little clearance for fast full layouts.
"""
import math
import random
import time
from dataclasses import replace
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
    if width*height <= 1024:
        # For many tiny outlines, vectorized cell predicates beat per-strip
        # geometry traversal. Keep this fast path bounded in memory.
        xx, yy = np.meshgrid(np.arange(width), np.arange(height))
        cells = shapely.box(x0+xx*step, y0+yy*step, x0+(xx+1)*step, y0+(yy+1)*step)
        filled = shapely.intersects(polygon, cells)
        rows, spans = [], []
        for row in filled:
            bits, runs = 0, []
            for index in np.flatnonzero(row):
                index = int(index)
                bits |= 1 << index
                if runs and runs[-1][1] == index:
                    runs[-1] = (runs[-1][0], index+1)
                else:
                    runs.append((index, index+1))
            rows.append(bits)
            spans.append(runs)
        scan = sorted(range(height), key=lambda i: rows[i].bit_count(), reverse=True)
        return width, height, rows, spans, scan
    # Intersect horizontal strips rather than allocating width*height cell
    # polygons. Each connected strip component reserves its entire x interval;
    # this remains conservative, including holes and disconnected row spans.
    yy = np.arange(height)
    strips = shapely.box(x0, y0+yy*step, x0+width*step, y0+(yy+1)*step)
    sections = shapely.intersection(polygon, strips)
    rows, spans = [], []
    def components(geometry):
        if geometry.is_empty:
            return
        if hasattr(geometry, 'geoms'):
            for part in geometry.geoms:
                yield from components(part)
        else:
            yield geometry
    for section in sections:
        bits = 0
        intervals = sorted((max(0, math.floor((part.bounds[0]-x0)/step)),
                            min(width, math.floor((part.bounds[2]-x0)/step)+1))
                           for part in components(section))
        runs = []
        for a, b in intervals:
            if runs and a <= runs[-1][1]:
                runs[-1] = (runs[-1][0], max(b, runs[-1][1]))
            else:
                runs.append((a, b))
        for a, b in runs:
            bits |= ((1 << (b-a))-1) << a
        rows.append(bits)
        spans.append(runs)
    # Most occupied rows reject candidate positions first.
    scan = sorted(range(height), key=lambda i: rows[i].bit_count(), reverse=True)
    return width, height, rows, spans, scan


def raster_layout(boards, items, deadline, *, resolution=512, order_mode='area',
                  order=None, mask_cache=None):
    valid = [i for i,b in enumerate(boards) if b.polygon.equals(box(*b.bounds))]
    if not valid:
        return []
    step = max(max(boards[i].bounds[2]-boards[i].bounds[0], boards[i].bounds[3]-boards[i].bounds[1])
               for i in valid) / resolution
    grids = {i: (math.floor((boards[i].bounds[2]-boards[i].bounds[0])/step),
                 [0] * math.floor((boards[i].bounds[3]-boards[i].bounds[1])/step)) for i in valid}
    masks = {} if mask_cache is None else mask_cache
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
    order = sorted(items, key=order_key, reverse=True) if order is None else order
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
            cache_key = (variant.polygon.wkb, step)
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


def raster_search(boards, items, deadline, *, seed, resolution=192):
    """Bounded joint search over order and permitted angular variants.

    Greedily choosing each part's lowest orientation can block later parts.
    Small chromosomes fix one angle per part, while shared strip masks make
    hundreds of mutations inexpensive. Returned geometry is still vector data.
    """
    if not any(b.polygon.equals(box(*b.bounds)) for b in boards):
        return
    rng = random.Random(seed)
    masks, seen, elite = {}, set(), []
    keys = list(items)
    best = None
    best_stock = None
    for _ in range(4000):
        if deadline is not None and time.perf_counter() >= deadline:
            return
        if elite and rng.random() < .8:
            _, parent_order, parent_angles = rng.choice(elite[:8])
            order, angles = list(parent_order), dict(parent_angles)
            if len(order) > 1 and rng.random() < .5:
                a, b = rng.sample(range(len(order)), 2)
                order[a], order[b] = order[b], order[a]
            for key in rng.sample(order, rng.randint(1, min(2, len(order)))):
                angles[key] = rng.randrange(len(items[key].rotation_variants))
        else:
            order = keys[:]
            rng.shuffle(order)
            angles = {k: rng.randrange(len(items[k].rotation_variants)) for k in keys}
        signature = tuple((k, angles[k]) for k in order)
        if signature in seen:
            continue
        seen.add(signature)
        pool = {k: replace(i, rotation_variants=(i.rotation_variants[angles[k]],))
                for k, i in items.items()}
        raw = raster_layout(boards, pool, deadline, resolution=resolution,
                            order=order, mask_cache=masks)
        area = sum(items[k].area for k, _, _, _, _ in raw)
        # Preserve order-search diversity even when pieces cannot all fit.
        height = max((v.bounds[3]+y for _, _, v, _, y in raw), default=0.)
        rank = (round(area, 6), -height)
        elite.append((rank, order, angles))
        elite.sort(key=lambda entry: entry[0], reverse=True)
        del elite[16:]
        # A taller but narrower layout can leave a larger right-hand strip.
        # Keep the fast height-based elite search, but never filter out an
        # improvement of the actual per-board remainder objective.
        stock, bbox_area = [], 0.
        for bi, board in enumerate(boards):
            bounds = [(v.bounds[0]+x, v.bounds[1]+y, v.bounds[2]+x, v.bounds[3]+y)
                      for _, index, v, x, y in raw if index == bi]
            if not bounds:
                stock.append(board.area if board.polygon.equals(box(*board.bounds)) else 0.)
                continue
            a, b = min(r[0] for r in bounds), min(r[1] for r in bounds)
            c, d = max(r[2] for r in bounds), max(r[3] for r in bounds)
            x0, y0, x1, y1 = board.bounds
            stock.extend(((a-x0)*(y1-y0), (x1-c)*(y1-y0),
                          (b-y0)*(x1-x0), (y1-d)*(x1-x0)))
            bbox_area += (c-a)*(d-b)
        stock_rank = (round(area, 6), round(max(stock, default=0.), 6), -bbox_area)
        if best is None or rank > best or best_stock is None or stock_rank > best_stock:
            best = max(rank, best) if best is not None else rank
            best_stock = max(stock_rank, best_stock) if best_stock is not None else stock_rank
            yield raw
