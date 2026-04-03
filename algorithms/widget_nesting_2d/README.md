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

The implementation is a practical hybrid:

- `shapely` handles polygon containment, rotation, boolean difference, and hole geometry.
- placement candidates are generated from free-space boundary contacts, inspired by no-fit-polygon style contact placement
- each candidate is compacted toward preferred board sides so the remaining stock stays as contiguous as possible
- an evolutionary search optimizes item order, while a beam keeps several partial layouts alive per order
- rotation variants are precomputed once per widget instance to reduce repeated geometry work during the search

This is not a full exact NFP solver. It is a robust repo-fit heuristic intended to work well on irregular polygons and hole reuse without pulling in a large external C++ stack.

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
python -m pip install "shapely>=2.0" matplotlib pytest
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

Suite cases currently include:

- synthetic single-widget full-fill
- synthetic hole reuse
- `shirts` benchmark subset
- `trousers` benchmark subset
- `swim` benchmark subset

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
