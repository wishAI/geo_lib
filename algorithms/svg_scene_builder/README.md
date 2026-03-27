# SVG Scene Builder

This stage converts an SVG floor plan into a self-contained MuJoCo scene package.

Outputs:
- `scene.xml`
- `semantic_layout.npz`
- `layout_preview.png`
- `start_pose.json`
- `scene_summary.json`
- `scene_package.json`
- `source_svg.svg`

The package includes the robot start pose and all metadata required by downstream consumers.

## Run

```bash
pyenv activate ptenv
python -m algorithms.svg_scene_builder.builder \
  --svg algorithms/svg_scene_builder/svg_room_map.svg \
  --output algorithms/svg_scene_builder/outputs/sample_scene \
  --map-resolution 0.02
```

## Test

```bash
pyenv activate ptenv
pytest algorithms/svg_scene_builder/tests -q
```
