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

## Web GUI And Remote Runtime

- Start the local control plane with `./geo gui`; it binds only to `127.0.0.1:8767`.
- Every algorithm owns its GUI contract in `algorithms/<name>/gui/manifest.json`.
- The root web host may discover manifests, start allowlisted commands, and copy declared result files, but it must not import algorithm code.
- TK2 commands run through the existing `tk2` SSH profile in the GUI-owned `/home/wishai/.cache/geo-lib-webgui/current` workspace. The Mac source is refreshed there before every remote run; the developer checkout is never rewritten.
- Isaac examples declare the shared `isaac` resource; the GUI refuses to run two Isaac jobs simultaneously.

## Large File And Nextcloud Contract

- The repository threshold is 5 MiB (`5242880` bytes).
- Source and input files at or below the threshold belong in Git unless another ignore rule already classifies them as generated output.
- Files above the threshold live under `~/Nextcloud/Projects/geo_lib` and are declared in root `large_files.json` with repository path, cloud path, size, and SHA-256.
- `./geo storage hydrate` restores repo-path symlinks from the manifest. Do not hard-code the macOS File Provider folder name.
- TK2 has a separate `/home/wishai/Nextcloud` folder without an always-running sync client. Use the GUI's explicit code refresh and cloud push/pull actions (or `./geo storage sync-code-tk2`, `push-tk2`, and `pull-tk2`) and never assume automatic two-way sync.
- Remote GUI results are copied to `~/Nextcloud/Projects/geo_lib/remote_outputs/<repo-relative-path>` so the Mac can preview them without adding generated artifacts to Git.
- Run `./geo storage audit` before handing off changes; tracked files over 5 MiB are a failure.

## Validation Expectations

- MuJoCo scene should load headless without GUI
- Scene geoms should stay simple, using box geoms only
- Mapping tests should run on copied input only
- Mapping snapshots should be saved at the configured interval
- ROS2 map output must remain grayscale `pgm + yaml`
