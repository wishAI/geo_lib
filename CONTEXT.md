# Project Context

## Relevant Algorithms

- `algorithms/svg_scene_builder`
  - Owns SVG parsing, semantic layout extraction, start-pose selection, and MuJoCo scene generation.
  - Produces a self-contained scene package.
- `algorithms/simple_auto_slam_mapping`
  - Owns route planning, robot motion, lidar simulation, occupancy mapping, and ROS2-style map export.
  - Consumes only a copied scene package.

## Stage Handoff Rule

Sandbox rule for future agents:
- Do not call, import, or depend on Python APIs from one algorithm folder to another.
- Handoff between stages must be file-based only.
- The output directory of `algorithms/svg_scene_builder` must be copied into an input directory for `algorithms/simple_auto_slam_mapping` before mapping runs.
- For `algorithms/simple_auto_slam_mapping`, input paths must be inside an `inputs/` directory and output paths must be inside an `outputs/` directory.
- If a combined workflow is needed, orchestrate it by copying files, not by cross-folder imports.

## Scene Builder Outputs

Expected files from `algorithms/svg_scene_builder`:
- `scene.xml`
- `semantic_layout.npz`
- `layout_preview.png`
- `start_pose.json`
- `scene_summary.json`
- `scene_package.json`
- `source_svg.svg`

These outputs include:
- MuJoCo scene geometry
- start pose for the robot
- layout resolution and dimensions
- semantic occupancy grids needed by the mapping stage
- bbox and geom-count metadata

## Mapping Stage Inputs And Outputs

Expected inputs for `algorithms/simple_auto_slam_mapping`:
- copied scene-builder package files listed above
- path must be under an `inputs/` ancestor

Expected outputs:
- `map.pgm`
- `map.yaml`
- `trajectory.json`
- `route.json`
- `mapping_summary.json`
- `input_package_snapshot.json`
- `snapshots/map_010s.pgm`, `snapshots/map_020s.pgm`, ... at the configured period
- path must be under an `outputs/` ancestor

## Environment Notes

- Python environment: `pyenv activate ptenv`
- MuJoCo is used in headless mode with `MUJOCO_GL=egl`
- ROS2 is installed on this machine under `/opt/ros/humble`, but the CLI may not be on `PATH` unless the ROS2 setup script is sourced

## Validation Expectations

- MuJoCo scene should load headless without GUI
- Scene geoms should stay simple, using box geoms only
- Mapping tests should run on copied input only
- Mapping snapshots should be saved at the configured interval
- ROS2 map output must remain grayscale `pgm + yaml`
