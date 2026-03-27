# SVG Auto SLAM Mapping

This module converts `algorithms/svg_auto_slam_mapping/svg_room_map.svg` into:

- a simplified MuJoCo scene built from box geoms only
- a planar robot with 2D lidar
- a timed autonomous exploration run
- a ROS2-compatible occupancy map as `map.pgm` and `map.yaml`

## Environment

```bash
pyenv activate ptenv
python -m pip install mujoco numpy pillow pytest
```

ROS2 is optional for execution. The generated map format matches the standard ROS2 `map_server` layout.

## Run

```bash
pyenv activate ptenv
python -m algorithms.svg_auto_slam_mapping.pipeline \
  --svg algorithms/svg_auto_slam_mapping/svg_room_map.svg \
  --output algorithms/svg_auto_slam_mapping/outputs/sample_run \
  --timeout 5.0
```

Main outputs:

- `scene.xml`
- `layout_preview.png`
- `trajectory.json`
- `scene_summary.json`
- `mapping_summary.json`
- `map.pgm`
- `map.yaml`

## Tests

```bash
pyenv activate ptenv
pytest algorithms/svg_auto_slam_mapping/tests -q
```

The tests validate:

- scene bbox and geometry counts
- MuJoCo headless loading
- robot motion and lidar ranges
- a 5 second mapping run
- ROS2 grayscale map export and pixel ratios
