"""Create a self-contained review page from validated benchmark artifacts."""
import argparse
import html
import json
import statistics
from collections import Counter
from pathlib import Path

from shapely.geometry import Polygon
from shapely.ops import unary_union

from .problem import ProblemSpec, save_solution
from .solver import SolverConfig, _build_runtime, _board_rest_rectangles


def normalized_score(problem, payload):
    """Recompute leftover rectangles with one scorer, including legacy output."""
    boards, _ = _build_runtime(problem, SolverConfig.from_problem(problem))
    maxima, sums = [], []
    for board in boards:
        polygons = [Polygon(p['polygon']['shell'], p['polygon'].get('holes', []))
                    for p in payload['placements'] if p['board_id'] == board.board_id]
        rects = _board_rest_rectangles(board, unary_union(polygons))
        maxima.append(max((r['area'] for r in rects), default=0))
        sums.append(sum(r['area'] for r in rects))
    return dict(payload['score'], max_rest_rectangle_area=max(maxima), sum_rest_rectangle_area=sum(sums))


def diagram(problem, payload):
    colors = ['#0b7963', '#b85836', '#407ec5', '#995ca1', '#b18420', '#39828f']
    parts = []
    def path(spec, color):
        rings = [spec['shell'], *spec.get('holes', [])]
        d = ' '.join('M ' + ' L '.join(f'{x:g},{y:g}' for x, y in r) + ' Z' for r in rings)
        return f'<path d="{d}" fill="{color}" fill-rule="evenodd" stroke="#253647" stroke-width="0.5" vector-effect="non-scaling-stroke"/>'
    for board in problem.boards:
        x0,y0,x1,y1 = board.polygon.to_polygon(name=board.board_id).bounds
        w,h=x1-x0,y1-y0
        parts.append(f'<svg viewBox="{x0-w*.04} {y0-h*.04} {w*1.08} {h*1.08}" role="img" aria-label="{html.escape(board.board_id)}">')
        parts.append(path(board.polygon.to_json(), '#f4eddf'))
        placements = sorted([p for p in payload['placements'] if p['board_id']==board.board_id],
                            key=lambda p: Polygon(p['polygon']['shell']).area, reverse=True)
        for i,p in enumerate(placements):
            parts.append(path(p['polygon'], colors[i%len(colors)]))
        parts.append('</svg>')
    return ''.join(parts)


def render_case_comparisons(root, report, problems, solutions):
    """Render same-seed layouts as image artifacts supported by the sandbox GUI."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from shapely.plotting import patch_from_polygon

    engines = list(dict.fromkeys(r['engine'] for r in report['results']))
    seed = 42 if 42 in report['contract']['seeds'] else report['contract']['seeds'][0]
    palette = ['#0b7963', '#b85836', '#407ec5', '#995ca1', '#b18420', '#39828f']
    names = {'nfp': 'Our hybrid solver' if 'constructive.py' in report.get('source_sha256', {}) else 'Our NFP + GA',
             'before': 'Previous NFP + GA', 'opennest': 'OpenNest', 'spyrrow': 'Sparrow'}
    for case, problem in problems.items():
        fig, axes = plt.subplots(len(problem.boards), len(engines), squeeze=False,
                                 figsize=(max(12, 5 * len(engines)),
                                          (8.5 if len(engines)==1 else 5.4) * len(problem.boards)), dpi=160)
        fig.set_facecolor('#f8fafb')
        rows = {r['engine']: r for r in report['results'] if r['case_id'] == case and r['seed'] == seed}
        ids = sorted({p['item_instance_id'] for (c, _, s), payload in solutions.items()
                      if c == case and s == seed for p in payload['placements']})
        colors = {item: palette[i % len(palette)] for i, item in enumerate(ids)}
        boards, _ = _build_runtime(problem, SolverConfig.from_problem(problem))
        for col, engine in enumerate(engines):
            row = rows.get(engine, {'status': 'not run'})
            payload = solutions.get((case, engine, seed))
            for board_index, (spec, board) in enumerate(zip(problem.boards, boards)):
                ax = axes[board_index, col]
                ax.set_facecolor('#f8fafb')
                ax.set_aspect('equal')
                ax.axis('off')
                title = names.get(engine, engine)
                if row['status'] == 'valid':
                    score = row['score']
                    title += (f"\n{row['placed_count']}/{row['requested_count']} parts · "
                              f"{score['placed_ratio_vs_requested']:.1%} requested area"
                              f"\nLargest scored remainder: {score['max_rest_rectangle_area']:.1f}"
                              f"\nSolve + adapter: {row.get('solve_seconds', row['solve_validate_seconds']):.2f}s")
                ax.set_title(title, fontsize=11, pad=18)
                if not payload:
                    message = row['status'].replace('_', ' ').capitalize()
                    if row['status'] == 'strip_exceeds_board':
                        m = row['strip_metrics']
                        message = (f"All-items strip needs width {m['required_strip_width']:.2f}"
                                   f"\nSheet width: {m['sheet_width']:.2f}"
                                   '\n\nFixed-sheet fit is inconclusive')
                    elif row['status'] == 'unsupported':
                        message += '\n' + row.get('reason', '')
                        message = message.replace(' or nonrectangular sheets', '\nor nonrectangular sheets')
                    ax.text(.5, .5, message, transform=ax.transAxes, ha='center', va='center',
                            fontsize=10, color='#526577', wrap=True)
                    continue
                polygon = spec.polygon.to_polygon(name=spec.board_id)
                ax.add_patch(patch_from_polygon(polygon, facecolor='#eee6d5', edgecolor='#253647', linewidth=1.4))
                placed = [p for p in payload['placements'] if p['board_id'] == spec.board_id]
                polygons = []
                for p in placed:
                    poly = Polygon(p['polygon']['shell'], p['polygon'].get('holes', []))
                    polygons.append(poly)
                    ax.add_patch(patch_from_polygon(poly, facecolor=colors[p['item_instance_id']],
                                                    edgecolor='#253647', linewidth=.5))
                rects = _board_rest_rectangles(board, unary_union(polygons))
                if rects:
                    rect = max(rects, key=lambda r: r['area'])
                    x0, y0, x1, y1 = rect['bounds']
                    ax.add_patch(Rectangle((x0, y0), x1-x0, y1-y0, facecolor='#95bc72',
                                           alpha=.3, edgecolor='#365c22', linestyle='--', linewidth=1.5))
                x0, y0, x1, y1 = polygon.bounds
                pad = max(x1-x0, y1-y0) * .035
                ax.set_xlim(x0-pad, x1+pad)
                ax.set_ylim(y0-pad, y1+pad)
        budget = report['contract'].get('matched_seconds')
        fig.suptitle(f"{case.replace('_', ' ').title()} · seed {seed}"
                     + (f" · {budget:g}s soft search budget" if budget else ''), fontsize=16, y=.97)
        fig.text(.5, .025, 'Same input shapes, sheet, rotations and seed. Colors identify the same part; holes remain transparent.\n'
                 'Green dashed overlay: scored remaining rectangle. This single seed does not show run-to-run reliability.',
                 ha='center', fontsize=9, color='#526577')
        fig.tight_layout(rect=(0, .10, 1, .80))
        fig.savefig(root / f'{case}_comparison.png', facecolor=fig.get_facecolor())
        plt.close(fig)


def render_nfp_budget_comparison(root, other_root):
    """Compare completed NFP trials at two budgets without mixing engine scores."""
    import matplotlib.pyplot as plt
    first = json.loads((root/'index.json').read_text())
    second = json.loads((other_root/'index.json').read_text())
    seed = second['contract']['seeds'][0]
    lookup = {(r['case_id'], r['seed']): r for r in first['results']
              if r['engine'] == 'nfp' and r['status'] == 'valid'}
    trials = [r for r in second['results'] if r['engine'] == 'nfp'
              and r['status'] == 'valid' and r['seed'] == seed and (r['case_id'], seed) in lookup]
    if not trials:
        raise ValueError('No matching valid NFP trials for the budget comparison')
    for row in trials:
        name = row['case_id']
        if json.loads((root/name/'problem.json').read_text()) != json.loads((other_root/name/'problem.json').read_text()):
            raise ValueError(f'Budget comparison requires identical input geometry and config: {name}')
    fig, ax = plt.subplots(figsize=(11, 4.5), dpi=160)
    for offset, report, color in ((-.18, first, '#407ec5'), (.18, second, '#0b7963')):
        for i, later in enumerate(trials):
            row = lookup[later['case_id'], seed] if report is first else later
            value = 100 * row['placed_count'] / row['requested_count']
            ax.barh(i+offset, value, height=.32, color=color,
                    label=f"{report['contract']['matched_seconds']:g}s budget" if i == 0 else None)
            ax.text(value+1, i+offset, f"{row['placed_count']} / {row['requested_count']}", va='center', fontsize=10)
    ax.set_yticks(range(len(trials)), [r['case_id'].replace('large_', '').replace('_', ' ') for r in trials])
    ax.invert_yaxis()
    ax.set_xlim(0, 110)
    ax.set_xlabel('Placed widgets / requested widgets (%)')
    ax.set_title(f'NFP large-scene budget sensitivity · seed {seed}')
    ax.axvline(100, color='#526577', linestyle='--', linewidth=1)
    ax.grid(axis='x', alpha=.15)
    ax.legend(loc='lower right')
    fig.tight_layout()
    fig.savefig(other_root/'budget_comparison.png')
    plt.close(fig)


def render_revision_comparisons(root, previous_root):
    current=json.loads((root/'index.json').read_text())
    previous=json.loads((previous_root/'index.json').read_text())
    rows=[]; solutions={}; problems={}
    for run, directory, label in ((previous,previous_root,'before'),(current,root,'nfp')):
        for row in run['results']:
            case=row['case_id']
            if row['engine']!='nfp' or not (root/case/'problem.json').exists():
                continue
            if json.loads((root/case/'problem.json').read_text()) != json.loads((directory/case/'problem.json').read_text()):
                raise ValueError(f'Before/after input mismatch: {case}')
            problems[case]=ProblemSpec.from_json(json.loads((root/case/'problem.json').read_text()))
            copy=dict(row,engine=label);rows.append(copy)
            if row['status']=='valid':
                payload=json.loads((directory/case/f"nfp_{row['seed']}.solution.json").read_text())
                copy['score']=normalized_score(problems[case],payload)
                solutions[case,label,row['seed']]=payload
    render_case_comparisons(root,dict(current,results=rows),problems,solutions)


def render_input_catalog(problem, path):
    import math
    import matplotlib.pyplot as plt
    from shapely.plotting import patch_from_polygon
    count=len(problem.widgets);columns=math.ceil(math.sqrt(count));rows=math.ceil(count/columns)
    fig,axes=plt.subplots(rows,columns,figsize=(columns*1.1,rows*1.2),squeeze=False,dpi=160)
    colors=['#0b7963','#407ec5','#b85836','#995ca1','#b18420','#39828f']
    for ax in axes.flat:
        ax.axis('off')
    for i,(ax,w) in enumerate(zip(axes.flat,problem.widgets)):
        p=w.polygon.to_polygon(name=w.widget_id)
        ax.add_patch(patch_from_polygon(p,facecolor=colors[i%len(colors)],edgecolor='#253647',linewidth=.5))
        x0,y0,x1,y1=p.bounds;pad=max(x1-x0,y1-y0)*.05
        ax.set_xlim(x0-pad,x1+pad);ax.set_ylim(y0-pad,y1+pad);ax.set_aspect('equal')
        ax.set_title(w.widget_id.replace('generated_',''),fontsize=5)
    fig.suptitle(f'{count} distinct input outlines · holes preserved\nEach preview scaled to its cell; solver uses the recorded dimensions',fontsize=14)
    fig.tight_layout(rect=(0,0,1,.95))
    fig.savefig(path);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--nfp-budget-run', type=Path, help='Also plot NFP counts against another completed budget run')
    parser.add_argument('--previous-run', type=Path, help='Create before/after layout images with identical inputs')
    args=parser.parse_args()
    report=json.loads((args.root/'index.json').read_text())
    rows=report['results']
    engines=list(dict.fromkeys(r['engine'] for r in rows))
    cases=list(dict.fromkeys(r['case_id'] for r in rows))
    solutions={}
    problems={}
    for case in cases:
        problems[case]=ProblemSpec.from_json(json.loads((args.root/case/'problem.json').read_text()))
    for row in rows:
        if row['status']=='valid':
            path=args.root/row['case_id']/f"{row['engine']}_{row['seed']}.solution.json"
            payload=json.loads(path.read_text())
            row['native_score']=row['score']
            row['score']=normalized_score(problems[row['case_id']], payload)
            solutions[row['case_id'],row['engine'],row['seed']]=payload
    report['score_note']='Rest rectangles recomputed with the same scorer for every engine; their overlapping sum is only a heuristic.'
    save_solution(args.root/'analysis.json',report)
    render_case_comparisons(args.root, report, problems, solutions)
    if args.previous_run:
        render_revision_comparisons(args.root,args.previous_run)
    catalog_counts=set()
    for case,problem in problems.items():
        count=len(problem.widgets)
        if 64 <= count <= 256 and count not in catalog_counts:
            render_input_catalog(problem,args.root/f'catalog_{count}.png')
            catalog_counts.add(count)
    if args.nfp_budget_run:
        render_nfp_budget_comparison(args.root, args.nfp_budget_run)
    contract = report['contract']
    budget_text = (f"Each engine receives a {contract['matched_seconds']:g}-second soft search budget. Inspect actual solve times: geometry operations and SDK calls can overshoot."
                   if contract.get('matched_seconds') is not None else
                   f"Local solvers use the case search counts; OpenNest gets a {contract['sdk_budget_seconds']:g}-second native budget. This is not an equal-time ranking.")
    text=['<!doctype html><meta charset="utf-8"><title>Nesting benchmark</title>',
          '<style>body{font:15px system-ui;color:#243746;background:#f8fafb;margin:32px auto;max-width:1400px;padding:0 20px}h1{font-size:30px}table{border-collapse:collapse;width:100%;background:white}th,td{text-align:left;padding:10px;border-bottom:1px solid #dde3e8}th{background:#e8eef1}small{color:#526577}details{margin:16px 0;padding:12px;background:white;border:1px solid #dde3e8;border-radius:8px}summary{cursor:pointer;font-weight:650}.layouts{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:18px}svg{display:block;width:100%;height:190px;margin:12px 0}.note{max-width:1000px;line-height:1.6}a{color:#236b91}pre{white-space:pre-wrap}</style>',
          '<h1>2D nesting: geometry, search, and SDK comparison</h1>',
          f'<p>{len(cases)} cases · {len(rows)} trials · seeds {html.escape(str(report["contract"]["seeds"]))} · TK2 Linux</p>',
          f'<p class="note">Fixed sheets, unchanged shapes and allowed rotations, zero requested clearance. Primary score: placed area. Secondary score: largest proven edge rectangle. {budget_text} The outer process timeout is {contract["hard_timeout_seconds"]:g} seconds, including startup. Concurrent trials: {contract.get("concurrent_jobs",1)}. Unsupported constraints and timeouts have no quality score.</p>',
          '<table><tr><th>Engine</th><th>Valid</th><th>Timeout</th><th>Unsupported</th><th>Strip exceeds board</th><th>Error</th></tr>']
    for engine in engines:
        counts=Counter(r['status'] for r in rows if r['engine']==engine)
        text.append('<tr><td>'+html.escape(engine)+'</td>'+''.join(f'<td>{counts[s]}</td>' for s in ('valid','timeout','unsupported','strip_exceeds_board','error'))+'</tr>')
    text.append('</table><p>Spyrrow uses a separate all-items strip objective. A valid result certifies that every item fits this sheet; an over-wide strip is inconclusive about fixed-sheet feasibility and has no partial-area score. Hole and nonrectangular cases remain unsupported.</p>')
    text.append('<table><tr><th>Scene</th><th>Requested widgets</th><th>Distinct shapes</th><th>Requested area / board area</th></tr>')
    for case, problem in problems.items():
        requested = sum(w.quantity for w in problem.widgets)
        area = sum(w.quantity * w.polygon.to_polygon(name=w.widget_id).area for w in problem.widgets)
        stock = sum(b.polygon.to_polygon(name=b.board_id).area for b in problem.boards)
        text.append(f'<tr><td>{html.escape(case)}</td><td>{requested}</td><td>{len(problem.widgets)}</td><td>{area/stock:.1%}</td></tr>')
    text.append('</table>')
    text.append('<table><tr><th>Case</th><th>Engine</th><th>Median placed/requested area</th><th>Range across valid seeds</th></tr>')
    for case in cases:
        for engine in engines:
            ratios=[r['score']['placed_ratio_vs_requested'] for r in rows if r['case_id']==case and r['engine']==engine and r['status']=='valid']
            if ratios:
                text.append(f'<tr><td>{html.escape(case)}</td><td>{html.escape(engine)}</td><td>{statistics.median(ratios):.1%}</td><td>{min(ratios):.1%}–{max(ratios):.1%}</td></tr>')
    text.append('</table>')
    text.append('<table><tr><th>Engine</th><th>Median solve + adapter time</th><th>Median final validation</th><th>Maximum process wall time</th></tr>')
    for engine in engines:
        trials = [r for r in rows if r['engine'] == engine]
        solve_times = [r['solve_seconds'] for r in trials if 'solve_seconds' in r]
        validation_times = [r['validation_seconds'] for r in trials if 'validation_seconds' in r]
        def seconds(values):
            return f'{statistics.median(values):.3f}s' if values else '—'
        text.append(f'<tr><td>{html.escape(engine)}</td><td>{seconds(solve_times)}</td><td>{seconds(validation_times)}</td><td>{max(r["wall_seconds"] for r in trials):.3f}s</td></tr>')
    text.append('</table><p>Adapter time includes input conversion and result assembly; Sparrow also validates its intermediate strip inside the adapter. Timings are not pure native search times. Unsupported trials have no solve time.</p>')
    text.append('<p>Expand a case to inspect the best validated seed for each engine. Coordinates are shown in the source frame; holes reveal the board beneath.</p>')
    for case in cases:
        text.append(f'<details><summary>{html.escape(case)}</summary><div class="layouts">')
        for engine in engines:
            trials=[r for r in rows if r['case_id']==case and r['engine']==engine]
            valid=[r for r in trials if r['status']=='valid']
            text.append(f'<section><h3>{html.escape(engine)}</h3>')
            if valid:
                best=max(valid,key=lambda r:(r['score']['placed_area'],r['score']['max_rest_rectangle_area']))
                text.append(f'<p>{best["placed_count"]}/{best["requested_count"]} parts · {best["score"]["placed_ratio_vs_requested"]:.1%} requested area<br>Rest rectangle {best["score"]["max_rest_rectangle_area"]:.3f} · seed {best["seed"]}<br><small>Solve {best.get("solve_seconds",best["solve_validate_seconds"]):.3f}s</small></p>')
                text.append(diagram(problems[case],solutions[case,engine,best['seed']]))
            for r in trials:
                if 'strip_metrics' in r:
                    m=r['strip_metrics']
                    text.append(f'<small>Seed {r["seed"]}: strip width {m["required_strip_width"]:.4f} / sheet width {m["sheet_width"]:.4f}</small><br>')
                text.append(f'<small>Seed {r["seed"]}: {html.escape(r["status"])} {html.escape(r.get("reason", ""))}</small><br>')
            text.append('</section>')
        text.append('</div></details>')
    text.append('<h2>Reproducibility</h2><pre>'+html.escape(json.dumps({k:v for k,v in report.items() if k!='results'},indent=2))+'</pre>')
    # A scientific comparison figure is also usable without a browser.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, max(3, len(cases)*0.38+1.6)), dpi=150)
    for offset, engine, color in [(-.17, 'nfp', '#0b7963'), (.17, 'opennest', '#407ec5')]:
        for i, case in enumerate(cases):
            values=[100*r['score']['placed_ratio_vs_requested'] for r in rows
                    if r['case_id']==case and r['engine']==engine and r['status']=='valid']
            if values:
                med=statistics.median(values)
                ax.barh(i+offset, med, height=.3, color=color, label=engine if i==0 else None)
                ax.errorbar(med,i+offset,xerr=[[max(0,med-min(values))],[max(0,max(values)-med)]],
                            color='#253647',capsize=2,linewidth=1)
    ax.set_yticks(range(len(cases)), cases)
    ax.invert_yaxis()
    ax.set_xlim(0,105)
    ax.set_xlabel('Placed / requested area (%) · median and range across valid seeds')
    title_budget = f"{contract['matched_seconds']:g}-second soft budget" if contract.get('matched_seconds') else 'fixed generation budgets'
    ax.set_title(f"NFP vs OpenNest · {title_budget}")
    ax.grid(axis='x',alpha=.2)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color='#0b7963',label='NFP'), Patch(color='#407ec5',label='OpenNest')], loc='lower right')
    fig.tight_layout()
    fig.savefig(args.root/'comparison.png')
    plt.close(fig)
    (args.root/'report.html').write_text('\n'.join(text))
    print(args.root/'report.html')


if __name__=='__main__':
    main()
