# 2D Widget Nesting

This module solves a hole-aware 2D nesting problem on one or more boards.

Supported behavior:

- multiple boards
- multiple widget types with quantities
- polygon widgets with optional holes
- free rotation via angle sampling
- no-overlap placement
- placing smaller widgets inside holes of larger widgets
- lexicographic objective:
  - maximize placed widget area first
  - if all requested widgets fit, prefer layouts that preserve a large edge-aligned empty rectangle

## Approach

For batches of 32 or more instances, default `nfp` mode first builds a feasible full-scene incumbent with two inexpensive constructors. A [MaxRects-style](https://github.com/juj/RectangleBinPack) pass packs conservative bounding envelopes using only permitted rotations. A polygon-aware bitset raster pass handles concavities and holes when envelopes waste too much space. Raster cells are collision reservations only: returned placements retain the original vector outlines, and the independent validator checks those exact polygons. No external nesting SDK is called by our solver.

The large-batch path returns the first full constructive layout. If construction is partial, NFP repair and the existing evolutionary search retain the best feasible incumbent. This prioritizes fitting every widget; it does **not** spend the remaining budget optimizing reusable stock after a full fit. Set `config.constructive_start` to `false` for the polygon-search ablation. Smaller scenes keep the original NFP/beam behavior.

The default `nfp` mode builds cached configuration-space obstacles from convex Minkowski sums. Shapely 2.1 constrained triangulation retains concave boundaries and holes. Inner-fit candidates come from subtracting obstacles from translation space; sampled contacts remain a fallback for degenerate fits. Every placement is independently checked against the original shapes.

An evolutionary search optimizes item order, while a beam retains alternative placements and sampled rotations. Repeated copies share canonical order chromosomes. Population construction is bounded, and elites leave room for offspring even with small populations. `--candidate-mode contact` keeps the repaired original contact heuristic available for ablation.

This is floating-point polygonal NFP geometry, not an exact-arithmetic or globally optimal solver. Zero-width fit regions can disappear during polygon unions; contact fallbacks and dedicated exact-fit tests reduce that risk but do not establish completeness. Rotation remains discrete. Triangle-pair Minkowski construction can be expensive for detailed outlines. There is currently no kerf/clearance model or continuous-rotation optimization.

The largest-rest-rectangle metric considers proven strips outside occupied bounds, not every possible empty rectangle. `sum_rest_rectangle_area` sums overlapping candidates and is only a ranking heuristic, not remaining material area.

## Hybrid scalability and genuinely mixed cases

The final solver verification covered **17 large/diverse scenes**: 16 full fits and one valid partial fit, with no invalid layouts or timeouts. All six earlier large scenes now fully fit, on both tested seeds. Representative TK2 results (seed 42):

| Scene | Previous 10-second result | Improved result | Improved solve time |
|---|---:|---:|---:|
| 1,000 rectangles | 124 / 1,000 | 1,000 / 1,000 | 0.11s |
| 300 shirts | 26 / 300 | 300 / 300 | 0.05s |
| 260 letters | 20 / 260 | 260 / 260 | 0.87s |
| 1,000 widgets / 256 distinct outlines | Not previously tested | 1,000 / 1,000 | 0.89s |
| 1,000 widgets / 1,000 distinct outlines | Not previously tested | 1,000 / 1,000 | 1.10s |
| Dense 256-outline mix, 60% requested area load | Not previously tested | 256 / 256 | 1.19s |

Solve time excludes the final independent validator, process startup and artifact writes. For 1,000 distinct outlines, final validation took **1.73s** and the complete benchmark worker took **3.20s**. Stopping at a full fit is a different use of the time budget from optimizing until the deadline; these times are not a pure NFP-kernel speedup claim.

The diverse catalogue starts with **66 source outlines**: all 35 vendored clothing pieces (8 shirts, 17 trousers, 10 swim), all 26 letters, and five existing widget shapes. Twelve additional procedural families vary geometry and topology: convex polygons, arrows, L brackets, T tabs, U channels, zigzags, combs, stars, crescents, dog-bones, notched plates, and frames with one to three holes. Hole-free companion suites replace frames with stairs and exclude source shapes with holes. The largest scene has **1,000 distinct outlines across 17 families**, one instance each; these are not simply rotated or scaled copies. An audit canonicalized translation, uniform scale and quarter-turn rotation at normalized precision 1e-8 and still found 1,000 unique shapes.

Known-feasible capacity scenes have separate grid witnesses; the solver never reads those witnesses. They are intentionally spacious (roughly 14–22% material-area load). Separate dense mixed cases use 55–60% load and do not assume feasibility until a full layout is validated. This separates capacity/throughput testing from difficult nesting quality. The 256-outline catalogue preview and all comparison images are available in the sandbox Results panel.

The dense 66-source-shape case remains a hard case: our solver and OpenNest each placed **62 / 66**, covering **95.63%** and **94.57%** of requested area respectively. This case has no full-fit certificate. Do not claim that every arbitrary input can now be fully packed. On the dense 256-shape mix, both solvers fit all pieces; OpenNest preserved a larger scored remainder (**6841.59** versus **4800.38**), so compaction remains an improvement target. OpenNest exceeded the 45-second hard limit on the 512-/1,000-widget diverse capacity scenes. Sparrow fitted all 1,000 hole-free mixed widgets in about 27.88s including adapter work; hole cases remain unsupported.

Evidence folders:

- `outputs/hybrid_large_10s`: 12 original-large trials, plus before/after images.
- `outputs/diverse_comparison_10s`: 24 trials across eight mixed scenes; source catalogues and three-engine images.
- `outputs/diverse_simple_10s`: six additional hole-free mixed trials.
- `outputs/unique_1000_10s`: 1,000-distinct-outline solve and `diversity_audit.json`.
- `outputs/hybrid_final_verification`: 17 trials using the final solver source.

This update includes **60 benchmark trials** in total, counting the final verification separately. Regression checks: **23 tests pass on Mac and TK2**, including restricted rotations on an offset board, hole reuse in raster construction, bitset collision equivalence, independent diverse-case witnesses and deadline preservation of a partial incumbent.

Use `--diverse` to opt into these cases, for example:

```bash
python -m algorithms.widget_nesting_2d.benchmark --diverse \
  --case diverse_256_dense --case simple_mix_256_dense --case unique_1000 \
  --engines nfp --seeds 42 --matched-seconds 10 --timeout 30 --jobs 1 \
  --output-root algorithms/widget_nesting_2d/outputs/my_diverse_run
python -m algorithms.widget_nesting_2d.benchmark_report \
  algorithms/widget_nesting_2d/outputs/my_diverse_run
```

## Input JSON

```json
{
  "units": "mm",
  "boards": [
    {
      "id": "board_a",
      "polygon": {
        "shell": [[0, 0], [320, 0], [320, 220], [0, 220]]
      }
    }
  ],
  "widgets": [
    {
      "id": "frame_large",
      "quantity": 2,
      "polygon": {
        "shell": [[-60, -42], [60, -42], [60, 42], [-60, 42]],
        "holes": [[[-28, -16], [28, -16], [28, 16], [-28, 16]]]
      }
    }
  ],
  "config": {
    "rotation_step_degrees": 15,
    "beam_width": 6,
    "population_size": 12,
    "generations": 6
  }
}
```

## Install

```bash
pyenv activate ptenv
python -m pip install "shapely>=2.1" "numpy>=1.26" matplotlib pytest
```

## Run

Single case:

```bash
pyenv activate ptenv
python -m algorithms.widget_nesting_2d.solver \
  --input algorithms/widget_nesting_2d/inputs/complex_dual_board.json \
  --output algorithms/widget_nesting_2d/outputs/complex_dual_board
```

Benchmark-backed suite:

```bash
pyenv activate ptenv
python -m algorithms.widget_nesting_2d.run_case_suite \
  --output-root algorithms/widget_nesting_2d/outputs
```

Outputs:

- per-case `problem.json`
- per-case `solution.json`
- per-case `nesting_layout.png`
- `outputs/index.json` listing all generated cases and scores

Generated outputs are ignored by git on purpose.

The suite has 24 cases: original widgets, full-fill and shortage cases, exact hole fits, multiple holes, slanted contacts, board cutouts, impossible orientations, all A–Z DejaVu Sans outlines in seven groups, three public clothing subsets, and their tight-board variants. Public subsets and generated stress variants are explicitly labeled; they are not the full original published instances.

Native SDK comparison (optional dependencies in an isolated environment):

```bash
python -m pip install "shapely>=2.1" matplotlib pytest compas_nest spyrrow
python -m algorithms.widget_nesting_2d.benchmark \
  --output-root algorithms/widget_nesting_2d/outputs/benchmark_20260920 \
  --engines contact nfp opennest --seeds 7 42 --timeout 15 --sdk-seconds 2 --jobs 4
python -m algorithms.widget_nesting_2d.benchmark_report \
  algorithms/widget_nesting_2d/outputs/benchmark_20260920
```

`--legacy-solver /path/to/untouched_solver.py --engines legacy ...` also compares a source snapshot taken before edits. `--resume` retains completed trials and retries errors. Each worker starts with cold NFP caches; timeouts include Python startup. SDK time budgets and local generation budgets differ, so this is a bounded diagnostic, not an equal-time ranking. Reports preserve errors/timeouts/unsupported constraints and validate identity, shape, rotations, board containment, overlap and area accounting at an explicit 1e-5 tolerance. The report recomputes leftover metrics using the same scorer for all engines.

OpenNest's per-part rotations must be uniform around 360 degrees; other angle lists are reported unsupported. Its `sheet_id=-1` entries are unplaced pieces. Spyrrow is a supplementary all-items strip feasibility probe. It uses the same item outlines and allowed angles at the original sheet height. A fully validated strip fitting the sheet certifies a complete fit. An over-wide strip is reported as `strip_exceeds_board`, which is inconclusive about fixed-sheet feasibility and receives no partial-area score. Multiple/irregular boards and item holes remain unsupported.

TK2 environment used for this comparison: `/home/wishai/.cache/geo-nesting-bench/venv/bin/python`; OpenNest `compas_nest==0.1.1.post5`, `spyrrow==0.9.0`, Shapely `2.1.2`. Commercial SDKs were not activated or benchmarked.

Optional MuJoCo debug render:

```bash
pyenv activate ptenv
python -m pip install mujoco numpy pillow
python -m algorithms.widget_nesting_2d.solver \
  --input algorithms/widget_nesting_2d/inputs/complex_dual_board.json \
  --output algorithms/widget_nesting_2d/outputs/complex_dual_board \
  --mujoco-debug
```

## Tests

```bash
pyenv activate ptenv
pytest -q algorithms/widget_nesting_2d/tests/test_widget_nesting.py
```

## References

- Clipper2 overview: https://angusj.com/clipper2/Docs/Overview.htm
- libnest2d: https://github.com/tamasmeszaros/libnest2d
- DeepNest: https://github.com/deepnest-next/deepnest
- jagua-rs benchmark assets: https://github.com/JeroenGar/jagua-rs/tree/main/assets
- Burke et al., no-fit polygon / irregular nesting heuristic background: https://www.graham-kendall.com/papers/bhkw2007.pdf


## Open-source comparison first

The active references are **OpenNest** (`compas_nest` 0.1.1.post5) for fixed-sheet NFP + genetic nesting, and **Sparrow** (`spyrrow` 0.9.0) for an independently validated strip-feasibility probe. Commercial SDK acquisition is deferred.

Use a shared soft budget instead of equating generation counts with native search time:

```bash
python -m algorithms.widget_nesting_2d.benchmark \
  --output-root algorithms/widget_nesting_2d/outputs/open_source_2s \
  --engines nfp opennest --seeds 7 42 --matched-seconds 2 --timeout 20 --jobs 2
python -m algorithms.widget_nesting_2d.benchmark \
  --output-root algorithms/widget_nesting_2d/outputs/open_source_10s \
  --engines nfp opennest spyrrow --seeds 7 42 123 \
  --case shirts_combo_tight --case trousers_shortage_tight \
  --case swim_curved_mix_tight --case alphabet_efgh \
  --case alphabet_uvwxyz --case alphabet_abcdefghijklmnopqrstuvwxyz \
  --matched-seconds 10 --timeout 35 --jobs 1
python -m algorithms.widget_nesting_2d.benchmark_report \
  algorithms/widget_nesting_2d/outputs/open_source_10s
```

The first run covers all 24 cases; the second uses six difficult cases, three seeds and serial trials. Each native engine uses one worker. Local timed search uses population 12 and a high generation ceiling, preserving each case's beam and rotation settings. Report actual `solve_seconds` separately from validation and startup: individual geometry/native operations can overrun soft deadlines. Reports show seed ranges as well as best layouts. Resume rejects changed source hashes, budgets or input geometry rather than silently mixing experiments.

The solver API also accepts `solve_problem(problem, config, time_limit_seconds=2)` and the CLI accepts `--time-limit-seconds 2`. Exhaustion returns the best feasible incumbent, with all unplaced instances accounted for. A process timeout is still needed to enforce an absolute upper bound.

The 2-second sweep completed **96 trials**: NFP returned 48 valid, fully accounted layouts; OpenNest returned 38 valid layouts and 10 unsupported rotation-list outcomes. Neither timed out or produced invalid geometry. “Valid” includes partial or empty layouts; it is not a claim that every part fitted. The full alphabet exposes a substantial short-budget gap: NFP placed **26.31%** of requested area while OpenNest placed **100%**. Tight swim improved to **86.09%** in both NFP seeds, matching OpenNest's placed area but retaining less useful stock.

The serial 10-second sweep completed **54 trials**. NFP returned 17 valid layouts and one unexplained worker exit; OpenNest returned 18 valid layouts; Sparrow returned nine validated full-sheet fits, six validated strips wider than the requested sheet, and three unsupported cases with holes. There were no hard timeouts. These are soft search budgets: actual calls can overrun, so use the recorded timings rather than assuming precisely equal elapsed time.

| Case | NFP placed/requested area | OpenNest placed/requested area | Sparrow all-items strip probe |
|---|---:|---:|---|
| Tight shirts | 88.82%, 88.82%, 100% | 88.82% in all seeds | All eight fit in all seeds; width about 24.53 / 26 |
| Tight trousers | 62.61% in all seeds | 61.50% in all seeds | Width about 112.00 / 65; inconclusive for fixed-sheet feasibility |
| Tight swim | 86.09% in all seeds | 86.09% in all seeds | Width about 254.94 / 234; inconclusive for fixed-sheet feasibility |
| EFGH | 100% in two valid seeds; one worker exit | 100% in all seeds | All fit in all seeds |
| UVWXYZ | 100% in all seeds | 100% in all seeds | All fit in all seeds |
| Full alphabet | 39.69% (8 / 26 parts) in all seeds | 100% (26 / 26 parts) in all seeds | Unsupported: holes |

Equal placed area can hide a material stock-quality gap: on tight swim, OpenNest preserved a largest scored edge rectangle of **1970.88**, versus **542.29** for NFP. The rectangle scorer is identical for all engines; it is a heuristic for useful remaining stock, not an exact maximum-inscribed-rectangle solver.

The EFGH NFP seed-123 exit had no captured stderr, so its cause is unknown. The original failed row is retained in `outputs/open_source_10s`. Separate diagnostic reruns passed with all four letters on TK2 (`outputs/open_source_recheck/index.json`) and the Mac (`outputs/local_recheck/index.json`). Future worker failures include exit codes and Python fault diagnostics. The successful reruns do not erase the original failure or establish that it is fixed. All **18 tests pass on both machines**.

Use `outputs/open_source_2s/report.html` and `outputs/open_source_10s/report.html` for layouts and per-seed status; each folder also contains normalized `analysis.json` and `comparison.png`. The main comparison totals **150 trials**, excluding smoke tests and diagnostic reruns. Source hashes, package versions, input geometry and actual timings are recorded with each run.

In the sandbox GUI, open **2D Widget Nesting → Results** for six side-by-side layout images. These compare seed 42 at the 10-second soft budget, with consistent part colors, transparent holes, placement counts and a green overlay for the scored remaining rectangle. Sparrow panels explicitly show unsupported or over-wide outcomes when no fixed-sheet layout is available. Regenerating `benchmark_report` also regenerates these `*_comparison.png` artifacts.

A separate local profile of the full alphabet found most time in placement generation, including thousands of Python-level geometry translations and convex Minkowski hull constructions. The next speed improvement should avoid materializing translated fallback polygons just to recover their anchor offsets, followed by reducing convex decomposition costs. This profile is diagnostic, not a cross-machine timing comparison.

Keep NFP + genetic ordering as the baseline, but prioritize placement throughput and an explicit compaction/local-repair stage before increasing population sizes. Sparrow's consistent shirt fits show why a different search family is useful as a reference. Its [open-source nesting paper](https://arxiv.org/html/2509.13329v3) studies construction and local-search approaches; the present six-case sample does not establish a universal winner.

## Before hybrid construction: large-scene scaling tests (2026-09-20)

The original suite topped out at 26 requested widgets. The opt-in large suite adds six scenes with **100–1,000 instances**, using one repeated rectangle, four original shirt outlines, or 26 original letter outlines. Shapes are repeated without removing holes or simplifying vertices. These are quantity-scaling tests, not tests of 1,000 distinct complex shapes.

```bash
python -m algorithms.widget_nesting_2d.benchmark --large \
  --case large_tiles_100 --case large_tiles_500 --case large_tiles_1000 \
  --case large_shirts_100 --case large_shirts_300 --case large_alphabet_260 \
  --engines nfp opennest spyrrow --seeds 7 42 \
  --matched-seconds 10 --timeout 60 --jobs 1 \
  --output-root algorithms/widget_nesting_2d/outputs/large_scene_10s
python -m algorithms.widget_nesting_2d.benchmark_report \
  algorithms/widget_nesting_2d/outputs/large_scene_10s
```

All **36 trials completed**: 12 valid NFP layouts, 12 valid OpenNest layouts, 10 valid Sparrow layouts and two unsupported Sparrow hole cases. No errors, invalid layouts or hard timeouts were recorded. Counts below span seeds 7 and 42:

| Scene | Our NFP + GA: placed parts | OpenNest: placed parts | Sparrow: placed parts |
|---|---:|---:|---:|
| 100 rectangles | 100 / 100 | 100 / 100 | 100 / 100 |
| 500 rectangles | 133–134 / 500 | 500 / 500 | 500 / 500 |
| 1,000 rectangles | 124 / 1,000 | 1,000 / 1,000 | 1,000 / 1,000 |
| 100 shirts | 21 / 100 | 100 / 100 | 100 / 100 |
| 300 shirts | 26 / 300 | 300 / 300 | 300 / 300 |
| 260 letters | 20 / 260 | 260 / 260 | Unsupported: holes |

The requested budget is **10 seconds, not an exact equal-elapsed-time cutoff**. NFP calls took about 10 seconds; OpenNest took about 10–12.4 seconds for rectangles/shirts and **37.1–38.8 seconds for 260 letters**. Sparrow's valid calls took about 10.2–17.9 seconds. These timings include adapter setup/result assembly, and Sparrow's intermediate-strip validation. Final common validation and process startup are separately recorded. The images include measured solve-plus-adapter times so native overruns and wrapper costs remain visible.

All three rectangle scenes have independently validated full grid witnesses under `outputs/large_scene_witness/`. The construction uses columns of 10 × 6 rectangles with 10% extra space on each sheet axis; it proves that partial NFP output reflects search throughput, not insufficient stock. Shirt sheets have nominal requested-area load 65%; letter sheets 40%. The native full fits also establish feasibility for these particular inputs.

NFP exhausted the 10-second budget within its first placement order on every large scene except the 100-rectangle case. A separate Mac profile of 100 shirts spent approximately 9.88 / 10.03 seconds in candidate generation, with around 188,700 geometry translations. Profiling overhead changes throughput, so that run is a diagnosis rather than an SDK timing comparison. Prioritize cheaper candidate generation, reuse of identical geometry, and fast construction of a complete first layout before tuning the genetic population.

The sandbox **Results** panel exposes six new large-scene layout comparisons, a summary chart and the 1,000-rectangle grid witness. Full per-seed evidence is in `outputs/large_scene_10s/analysis.json`, with the HTML report alongside it.

Three additional **60-second NFP diagnostics**, using seed 42 and identical input/configuration, also returned valid layouts:

| Scene | NFP at 10 seconds | NFP at 60 seconds | OpenNest full-fit solve + adapter time, seed 42 |
|---|---:|---:|---:|
| 1,000 rectangles | 124 / 1,000 | 420 / 1,000 | 12.42 seconds |
| 300 shirts | 26 / 300 | 49 / 300 | 11.33 seconds |
| 260 letters | 20 / 260 | 37 / 260 | 37.05 seconds |

The longer NFP calls took 60.01–60.14 seconds, excluding final validation/startup. This confirms a substantial implementation-throughput gap even when NFP receives more elapsed time. These three diagnostic runs use one seed and are separate from the 36-trial comparison; the combined large-scene total is **39 trials**, plus three constructed feasibility witnesses.

```bash
python -m algorithms.widget_nesting_2d.benchmark --large \
  --case large_tiles_1000 --case large_shirts_300 --case large_alphabet_260 \
  --engines nfp --seeds 42 --matched-seconds 60 --timeout 90 --jobs 1 \
  --output-root algorithms/widget_nesting_2d/outputs/large_scene_nfp_60s
python -m algorithms.widget_nesting_2d.benchmark_report \
  algorithms/widget_nesting_2d/outputs/large_scene_10s \
  --nfp-budget-run algorithms/widget_nesting_2d/outputs/large_scene_nfp_60s
```

The resulting budget-sensitivity chart and raw 60-second trial data are also available in the sandbox Results panel.

## Earlier fixed-generation TK2 results (2026-09-20)

The completed sweep has **192 trials: 24 cases × 2 seeds × 4 engines**. Every completed layout was independently validated; no supported completed trial had a validation error.

| Engine | Valid | 15-second timeouts | Unsupported angle sets |
|---|---:|---:|---:|
| Untouched original | 46 | 2 | 0 |
| Repaired contact heuristic | 20 | 28 | 0 |
| Cached NFP + GA | 45 | 3 | 0 |
| OpenNest | 38 | 0 | 10 |

These completion counts include intentionally impossible cases returning an empty, valid layout; they do not mean all requested parts fit. Exact zero-clearance cases expose different numerical behavior in OpenNest and should not be generalized to ordinary cutting jobs.

- Tight shirt subset: NFP increased placed/requested area from **78.39% to 88.82%** versus the untouched solver, on both seeds.
- Slanted three-edge fit: the old and repaired contact modes missed the piece; NFP and OpenNest placed it.
- Tight swim subset: NFP placed **71.69%** of requested area versus OpenNest's **86.09%**, a remaining **14.40 percentage-point** gap.
- Loose shirt subset: all pieces fit, but best-seed remaining rectangle was **208.00** for the bounded NFP run, **220.00** for the original, and **278.18** for OpenNest. Thus NFP does not dominate the original on every secondary score.
- Increasing NFP to beam 6, population 12 and 6 generations matched **278.18** on TK2 (seed 7; 43 unique orders). The CLI took **26.02 seconds including validation/rendering**, versus OpenNest's 2-second native search budget. This establishes a search-efficiency gap, not an equal-phase timing ratio.
- NFP timed out on the full A–Z case for both seeds and on QRST for seed 42. All other NFP trials completed under the outer limit.

The report is `outputs/benchmark_20260920/report.html`; `analysis.json` keeps every trial, including non-success outcomes, and recomputes leftover scores consistently. The original snapshot is retained on TK2 at `/home/wishai/.cache/geo-nesting-bench/legacy_solver.py`. Final regression checks: **15 passed on Mac and TK2**; repository 5 MiB audit passed.

## Commercial Linux SDK candidates

- **HCL NestLib** is the strongest verified commercial candidate for a vendor-supported integration. Its [official system requirements](https://nestlib.geometricglobal.com/support/system-requirements/) explicitly list Linux and embeddable APIs. A current Ubuntu-compatible binary, evaluation license and redistribution terms still need to be obtained from HCL; this work did not install or measure it.
- **NestProfessor** lists a Linux C++ GCC/.NET Core SDK for Debian, CentOS and Ubuntu on its [official download page](https://nestprofessor.com/en/?page_id=1285). SDK evaluation requires a temporary activation license. Its public download route returned HTTP 403/406 from TK2, so no package or license was acquired. The GUI's advertised evaluation period should not be assumed to cover the Linux SDK.
- **Alma PowerNest** has a documented [token-authenticated REST API](https://doc.powernestlib.com/getting-started/). That is callable from Linux, but does not by itself verify an installable native Linux SDK. Do not equate a cloud service with on-premise Linux deployment.

OpenNest and Sparrow are useful open-source reference engines, not evidence of parity with commercial products such as NestLib. No vendor was contacted and no paid license was purchased.

## Research and next improvements

- [Burke et al. (2007), robust NFP generation](https://doi.org/10.1016/j.ejor.2006.03.011): geometry must account for holes, interlocking concavities and contact degeneracies. This motivated genuine configuration-space obstacles and tight-fit regression tests.
- [Rocha (2019), no-fit polygon generation](https://arxiv.org/abs/1903.11139): perfect-fit regions require special handling. Floating-point unions can discard isolated points/lines; the current implementation remains a heuristic at those degeneracies.
- [Sparrow paper](https://arxiv.org/html/2509.13329v3): shrinking and overlap-resolution local search offer a useful direction beyond constructive ordering. NFP + genetic search is a sound architecture, but its quality also depends on placement scoring, diversity, caching and refinement.

Next measured priorities are orientation-aware genetic chromosomes, geometric beam diversity, a compaction phase that can rearrange groups of pieces, and faster convex decomposition/NFP construction. These are recommendations, not claims that those features already exist.
