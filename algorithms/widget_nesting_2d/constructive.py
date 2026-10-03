"""Conservative bounding-envelope construction for a fast feasible incumbent.

Maximal free rectangle splitting follows the MaxRects family described by
Jylanki: https://github.com/juj/RectangleBinPack . Polygon geometry is unchanged;
envelopes only reserve space. Irregular boards still use the polygon solver.
"""
import time


def _subtract(free, used, eps):
    x, y, r, t = used
    split = []
    for a, b, c, d in free:
        if r <= a + eps or x >= c - eps or t <= b + eps or y >= d - eps:
            split.append((a, b, c, d))
            continue
        if x > a + eps:
            split.append((a, b, x, d))
        if r < c - eps:
            split.append((r, b, c, d))
        if y > b + eps:
            split.append((a, b, c, y))
        if t < d - eps:
            split.append((a, t, c, d))
    # Overlapping free rectangles are intentional; remove contained ones.
    unique = list(dict.fromkeys(split))
    return [p for i, p in enumerate(unique) if not any(
        i != j and q[0] <= p[0] and q[1] <= p[1] and q[2] >= p[2] and q[3] >= p[3]
        for j, q in enumerate(unique))]


def envelope_layouts(boards, items, deadline, tolerance):
    """Yield complete/partial attempts as (instance, board, variant, dx, dy).

    Only actual permitted rotations are considered. No fixture metadata or
    known-feasible witness is available to this constructor.
    """
    from shapely.geometry import box
    valid_boards = [i for i, b in enumerate(boards) if b.polygon.equals(box(*b.bounds))]
    if not valid_boards:
        return
    eps = min(tolerance / 100, 1e-10)
    variants = {}
    for key, item in items.items():
        seen = set()
        variants[key] = []
        for v in item.rotation_variants:
            a, b, c, d = v.bounds
            # Equal envelopes need only one orientation in this conservative pass.
            size = (round(c-a, 10), round(d-b, 10))
            if size not in seen:
                seen.add(size)
                variants[key].append((v, c-a, d-b))
    orders = []
    for metric in (lambda i: max(i.rotation_variants[0].bounds[2]-i.rotation_variants[0].bounds[0],
                                 i.rotation_variants[0].bounds[3]-i.rotation_variants[0].bounds[1]),
                   lambda i: i.bbox_area, lambda i: i.area):
        order = tuple(sorted(items, key=lambda k: metric(items[k]), reverse=True))
        if order not in orders:
            orders.append(order)
    for mode in ('short_side', 'area', 'bottom_left'):
        for order in orders:
            free = {i: [boards[i].bounds] for i in valid_boards}
            placed = []
            for key in order:
                if deadline is not None and time.perf_counter() >= deadline:
                    yield placed
                    return
                best = None
                for board_index in valid_boards:
                    for x, y, r, t in free[board_index]:
                        for v, w, h in variants[key]:
                            dw, dh = r-x-w, t-y-h
                            if dw < -eps or dh < -eps:
                                continue
                            if mode == 'short_side':
                                score = (min(dw,dh), max(dw,dh), y+h, x)
                            elif mode == 'area':
                                score = ((r-x)*(t-y)-w*h, min(dw,dh), y+h, x)
                            else:
                                score = (y+h, x, min(dw,dh), max(dw,dh))
                            if best is None or score < best[0]:
                                best = (score, board_index, v, x, y, w, h)
                if best is None:
                    continue
                _, bi, v, x, y, w, h = best
                placed.append((key, bi, v, x-v.bounds[0], y-v.bounds[1]))
                free[bi] = _subtract(free[bi], (x,y,x+w,y+h), eps)
            yield placed
            if len(placed) == len(items):
                return
