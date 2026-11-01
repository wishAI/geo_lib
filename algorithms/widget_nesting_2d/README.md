# 2D Widget Nesting

This module solves a hole-aware 2D nesting problem on one or more boards.

Supported behavior:

- multiple boards
- multiple widget types with quantities
- polygon widgets with optional holes
- free rotation using geometry-derived real-valued angles and bounded local angular refinement
- no-overlap placement
- placing smaller widgets inside holes of larger widgets
- lexicographic objective:
  - maximize placed widget area first
  - if all requested widgets fit, prefer layouts that preserve a large edge-aligned empty rectangle

## Approach

For free-rotation scenes and batches of 32 or more instances, default `nfp` mode first builds a feasible full-scene incumbent with two inexpensive constructors. A [MaxRects-style](https://github.com/juj/RectangleBinPack) pass packs conservative bounding envelopes using only permitted rotations. A polygon-aware bitset raster pass handles concavities and holes when envelopes waste too much space. Raster cells are collision reservations only: returned placements retain the original vector outlines, and the independent validator checks those exact polygons. No external nesting SDK is called by our solver.

Free-angle envelope restarts continue after the first cardinal full fit, and the best layout survives. Small free-rotation scenes (up to 32 instances) also try narrower working sheets and a bounded joint search over part order and angular variants. Shared conservative strip masks make these mutations inexpensive. Full layouts can improve reusable stock; partial layouts prioritize placed area. Scenes missing just one of many parts preserve time for insertion repair. Boundary reinsertion then uses up to 0.5 seconds of the remaining budget for angular refinement, including the incumbent angle and useful interior moves. Failed or expired replacements retain the incumbent. Large scenes retain the bounded construction path. Set `config.constructive_start` to `false` for the polygon-search ablation. Smaller fully restricted scenes keep the original NFP/beam behavior.

The default `nfp` mode builds cached configuration-space obstacles from convex Minkowski sums. Shapely 2.1 constrained triangulation retains concave boundaries and holes. Inner-fit candidates come from subtracting obstacles from translation space; sampled contacts remain a fallback for degenerate fits. Adjacent triangles are merged into convex pieces under a fixed comparison cap, reducing Minkowski products without unbounded preprocessing. Rectangular-sheet candidates are ranked from bounds before expensive translation and collision checks. Every placement is independently checked against the original shapes.

An evolutionary search optimizes item order, while a beam retains alternative placements and sampled rotations. Repeated copies share canonical order chromosomes. Population construction is bounded, and elites leave room for offspring even with small populations. `--candidate-mode contact` keeps the repaired original contact heuristic available for ablation.

This is floating-point polygonal NFP geometry, not an exact-arithmetic or globally optimal solver. Zero-width fit regions can disappear during polygon unions; contact fallbacks and dedicated exact-fit tests reduce that risk but do not establish completeness. Free rotation is a bounded heuristic, not a complete search over all angles: coarse seeds include 30-degree samples and arbitrary edge-alignment angles, tight empty-sheet fits receive continuous bounding-box refinement, then local search halves angular steps from 7.5 to 0.46875 degrees while re-solving translations. Triangle-pair Minkowski construction can be expensive for detailed outlines. There is currently no kerf/clearance model.

The largest-rest-rectangle metric considers proven strips outside occupied bounds, not every possible empty rectangle. `sum_rest_rectangle_area` sums overlapping candidates and is only a ranking heuristic, not remaining material area.

## Free rotation and short runtime

`rotation_mode` now defaults to `free`. Unrestricted widgets can use arbitrary angles; an explicit widget `allowed_angles_degrees` list or widget `rotation_step_degrees` still restricts that widget. Use `--rotation-mode discrete --rotation-step-degrees 90` for the previous four-direction search. The global angle step applies only in discrete mode.

The CLI and GUI default to a **2-second soft search budget**, adjustable with `--time-limit-seconds`. The API still accepts an explicit `time_limit_seconds`. Geometry operations, initialization and validation can exceed a soft deadline; benchmark workers enforce a separate hard timeout. Search shares variants across repeated instances, computes anchors only when needed, and compares cardinal construction with compact-angle and full-angle restarts. Small-scene angular repair improves the existing layout instead of restarting on an empty board. The validator uses a spatial index to avoid testing every pair of distant parts, while checking the original vector polygons and actual returned angles.

```bash
python -m algorithms.widget_nesting_2d.solver \
  --input algorithms/widget_nesting_2d/inputs/complex_dual_board.json \
  --output algorithms/widget_nesting_2d/outputs/gui_example \
  --rotation-mode free --time-limit-seconds 2
python -m algorithms.widget_nesting_2d.benchmark --diverse --large \
  --case free_oblique_bar --case shirts_combo_tight --case alphabet_efgh \
  --case diverse_256_dense --case unique_1000 --case large_tiles_1000 \
  --engines nfp opennest spyrrow --seeds 7 42 --rotation-mode free \
  --matched-seconds 2 --timeout 15 --jobs 1 \
  --output-root algorithms/widget_nesting_2d/outputs/free_rotation_2s
```

Benchmark `--rotation-mode free` creates explicitly labelled copies with **all** original widget angle restrictions relaxed for **every** engine. Source cases stay unchanged. This is a different problem from the original restricted benchmarks below. Use benchmark `--rotation-mode discrete` to retain their original constraints.

[Sparrow's API](https://spyrrow.readthedocs.io/en/latest/api.html) receives `allowed_orientations=None` for continuous rotation. [OpenNest's API](https://petrasvestartas.github.io/compas_nest/api/compas_nest.nfp/) supports rotation counts, so free widgets receive **72 orientations (5-degree approximation)**. Reports label this limitation; it is not equivalent to continuous search. Sparrow remains a strip-feasibility probe and does not support holes. A minimum-width precheck rejects impossible strip heights before native calls; native failures on otherwise feasible tight shapes are retained as errors.

The new `free_oblique_bar` case requires an angle outside both quarter turns and the 5-degree grid. Tests also cover exact 17.321-degree alignment, arbitrary-angle validation, preservation of explicit restrictions, local angular refinement, free SDK configuration and impossible strip heights.

The active search now continues beyond a cardinal full fit: it compares compact-angle and full-angle construction, then removes and reinserts boundary pieces under a shared 0.5-second cap. Incomplete layouts retain their full placement-search deadline; rotation polishing cannot consume the time needed to fill an exact hole. A replacement is accepted only when the complete layout score improves. Arbitrary angles are shown on small comparison images, and every panel reports the non-cardinal placement count.

## Current joint order/rotation verification

The final suite covers **45 inputs**: all 44 distinct recovered historical inputs/configurations plus the arbitrary-angle regression. **37 tests pass.** It contains 135 fresh local trials (seeds 7, 42, 123), 90 fresh SDK trials (seed 42), and 50 separate current-solver historical replays preserving original angle restrictions and recorded budgets/seeds. All 185 local/replay layouts validated, along with three saved grid witnesses. The 135 free trials produced **120 full fits and 15 partial fits**; median solve time was **0.934s**, maximum **2.007s**. Fifteen paired trials increased placed area against the previous free solver, with no placed-area regressions across the original 42 cases and three seeds.

| Case | Previous free result | Current result across three seeds | Current solve time |
|---|---|---|---|
| Tight shirts | 7/8 | 8/8 | 1.65–1.66s |
| Tight swim | 3/4 | 4/4 | 1.47–1.48s |
| Board cutout | 7/8 | 8/8 | 0.51s |
| Roomy swim | 4/4; reusable stock 12,448.58 | 4/4; reusable stock 29,344.21 | 1.45s |
| 1,000 distinct outlines | 1,000/1,000 | 1,000/1,000 | 1.13–1.14s |
| 1,000 rectangles | 1,000/1,000 | 1,000/1,000 | 0.31–0.33s |

Paired seed-42 quality comparisons rank placed area first, then largest reusable stock **only when both engines fully fit**. Against OpenNest: **20 wins, 5 ties, 12 losses** over 37 valid comparisons. Against Sparrow: **4 wins, 11 losses** over 15 valid comparisons; our solver fully fits all 15, with losses in leftover stock. Our solver does not dominate either reference on every case. The remaining local partial cases are trousers shortage, tight trousers, concave interlock, knapsack, and dense source mix 66. Some are over capacity; a partial layout alone is not an infeasibility proof.

OpenNest also had eight hard timeouts. Sparrow had 23 unsupported inputs, five over-wide strips, one native error and one timeout. These are excluded from quality wins. Both libraries receive the same free-angle input copies: Sparrow continuous rotation, OpenNest its 5° approximation. Native runs can exceed the requested 2s soft budget; the report retains actual timings and the 15s worker cutoff. For example, 1,000 rectangles took our solver 0.314s at seed 42, OpenNest 12.341s (914/1,000 parts), and Sparrow 14.014s (all parts, smaller reusable stock).

The final local optimization uses vectorized cell masks for footprints of at most 1,024 cells, and conservative horizontal-strip intersections for larger footprints. This retains throughput on thousands of tiny outlines while speeding searches with a few large parts. NFP convex merging is capped at 2,048 pair checks and bypassed above 96 triangles. All actual output polygons, holes, and permitted rotations are independently validated.

Current artifacts, also linked first in the GUI Results panel:

- `outputs/joint_rotation_final_comparison/gallery.html`: all 45 three-engine images, seven overview sheets, and five before/after comparisons.
- `outputs/joint_rotation_final_comparison/report.html`: every fresh comparison score, timing, unsupported result, error and timeout.
- `outputs/joint_rotation_final_comparison/summary.json`: pairwise wins/ties/losses and all paired area changes.
- `outputs/joint_rotation_final_history/report.html`: 50 current restricted/original-budget replays and all 44 historical input/configuration images. No restricted catalogue replay lost placed area against its selected saved historical baseline. All three recovered GUI configurations fully fit under their original 0.2s budget when deliberately switched to free rotation; exact restricted runs remain separately labelled.
- `outputs/joint_rotation_final_comparison/run_suite.py` and `build_report.py`: suite orchestration and report assembly. SDK runs were freshly measured in this task immediately before the final tiny-footprint mask optimization; only `raster_start.py` differs in source hashes, and neither SDK adapter calls it. Per-row source hashes and identical inputs are verified and retained.

The repository storage audit still reports three pre-existing oversized AVP texture files outside this algorithm; no nesting source file exceeds its limit.

## Previous free-rotation verification

The preceding revision’s verification covered **all 42 catalogue cases**, including the earlier 24 small cases, the oblique-angle regression, six large batches and eleven mixed suites. **34 tests pass.** On the final solver, **126 serial trials** (seeds 7, 42 and 123) all validated: **102 full fits and 24 partial fits**, with no local errors or hard timeouts. Median solve time was **0.525s**, maximum **2.012s** with a 2-second soft budget. At seed 42, none of the cases lost placed area versus the preceding full sweep. Eight cases remained partial: trousers shortage, concave interlock, board cutout, knapsack, tight shirts, tight trousers, tight swim and dense source mix 66. Some are over-capacity instances; partial placement is not proof of infeasibility.

The shirt subset now preserves stock **227.65** versus the previous **180**, with one part at **285.945°**, while retaining all eight parts. The swim subset fits four parts at **192.67°, 12.67°, 194.31° and 328.30°**. Both solve in about **0.52s**. The 300-shirt scene fits all pieces in about **0.23s**, with **82 non-cardinal orientations** in seed 42. Rectangles can still select cardinal angles when those pack better; rotations are not forced for appearance.

Images and evidence:

- `outputs/active_free_rotation_comparison/gallery.html`: all 42 cases in six overview sheets and 42 three-engine comparison images, including partial results.
- `outputs/active_free_rotation_comparison/report.html`: **210 measured trials**, comprising the final 126 local trials plus 84 SDK trials from the preceding full sweep. The SDK adapters and inputs are unchanged; each row retains its original source run and complete source hashes. SDKs use seed 42; our solver uses three seeds. This composite is deliberately labelled and is not a resumed single-source run.
- `outputs/active_free_rotation_all_cases_v2_2s`: exact final-solver inputs, hashes, timing, independently validated layouts, and per-case angle counts.
- `outputs/free_rotation_all_cases_2s`: the complete preceding 126-trial three-engine sweep, retained as the comparison baseline.

Across the 84 SDK trials, OpenNest returned 34 valid layouts and eight timeouts; Sparrow returned 15 validated fixed-sheet fits, five over-wide strips, 20 unsupported cases, one native error and one timeout. OpenNest remains a 5-degree approximation; Sparrow uses continuous angles but its strip objective and hole restrictions remain different. These outcomes are visible in the gallery and are not treated as packing scores.

The recovered history was checked separately in `outputs/history_coverage_recheck_20261010/report.html`: 109 saved problem files reduce to 44 distinct inputs/configurations, comprising the 41 original catalogue inputs and three additional GUI configurations. All 1,003 distinct handoff files passed SHA-256 checks. **50 fresh current-solver trials validated**, plus the three saved 100/500/1,000-tile grid witnesses. The 41 catalogue replays preserve explicit angle restrictions and holes, use recorded seed 42 and 2s/10s budgets, and have no placed-area losses against the selected historical baselines. The remaining nine trials compare each saved GUI configuration with its original restrictions, a deliberate free-angle override at its original 0.2s budget, and that override at 2s. The dual-board variants fit 12/12 in about 0.20s with free rotation; the constrained variant fits 9/10 at 0.20s and 10/10 at 0.76s. The report includes all 44 layout images, original/current source provenance, and separate solve/validation/wall timings. This covers every distinct recovered input with representative recorded contracts, not every historical trial or budget/seed combination; historical timing differences are not controlled speedup measurements.

Earlier six-case validation: **31 tests passed**. The 36 comparison trials are in `outputs/free_rotation_2s`; after adding the narrow-interval angle fallback, all 12 local trials were repeated with the final solver in `outputs/free_rotation_final_check`. All local layouts validated: ten full fits and two 7/8 tight-shirt fits, with no local errors or timeouts. Final timings across seeds 7 and 42:

| Scene | Placed | Solve time | Independent validation |
|---|---:|---:|---:|
| Tight oblique bar | 1/1 | 0.25s | <0.001s |
| EFGH | 4/4 | 0.25–0.26s | 0.002s |
| Dense mixed outlines | 256/256 | 1.20s | 0.10s |
| Distinct mixed outlines | 1,000/1,000 | 1.12–1.14s | 0.40–0.41s |
| Rectangles | 1,000/1,000 | 0.10–0.12s | 0.09s |
| Tight shirts | 7/8 | 2.00s | 0.002s |

The 1,000-distinct-outline worker took 1.89–1.91s including startup and adapter/report preparation. Construction still prioritizes full fit over maximum reusable stock. Sparrow fitted all eight shirts in both seeds; our solver and OpenNest fitted seven. OpenNest's 5-degree approximation could not fit the tight oblique bar; Sparrow 0.9.0 panicked during native initialization on that feasible tight case, retained as two errors. OpenNest hit the 15-second hard cutoff on both dense-256 and unique-1,000 trials; Sparrow does not support their holes. The native 1,000-rectangle calls exceeded the requested soft budget (roughly 11.5–13.9s), so the report records actual time rather than claiming equal elapsed time. These failures and limits are not scored as packing wins.


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
    "rotation_mode": "free",
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

The original small suite had 24 cases (the current library adds the oblique-angle regression and opt-in large/mixed cases): original widgets, full-fill and shortage cases, exact hole fits, multiple holes, slanted contacts, board cutouts, impossible orientations, all A–Z DejaVu Sans outlines in seven groups, three public clothing subsets, and their tight-board variants. Public subsets and generated stress variants are explicitly labeled; they are not the full original published instances.

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

### Mac ↔ TK2 sync

Run `python3 algorithms/widget_nesting_2d/sync.py` from the Mac to merge nesting source changes in both directions through Git and transfer all saved outputs in both directions. The command uses private `refs/geo-sync/widget_nesting_2d/` snapshots, preserves both checkout branches and unrelated staging, and stops on source conflicts. Result files are never deleted; differing result versions are archived under `outputs/.sync/conflicts/` before a winner is copied, with the last synchronized hash used to detect which side changed. Every resulting output file is checked by SHA256. This is an explicit sync command, not a background process. The GUI-owned TK2 runtime remains separate and is refreshed from the merged Mac source before a run.
