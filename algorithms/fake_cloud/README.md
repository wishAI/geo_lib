# Fake Cloud Dataset (MuJoCo, Headless)

This module generates a synthetic ship-part style point-cloud dataset using MuJoCo offscreen rendering only (no GUI window).

## Key behavior

- 2 to 3 vertical boards only.
- Vertical boards are orthogonal or parallel to each other (yaw in {0, 90}).
- 2 to 3 depth views (default 3).
- Camera views are sampled on a ring above the bottom plane.
- Camera azimuths are generated as: first_view_deg + i * view_step_deg.
- Cameras always look at the bottom-plane center.
- Per-view output is noisy point cloud directly (no clean/ and noisy/ folders).
- `cam_info.json` stores all per-view camera transforms in one file.
- Camera transforms in `cam_info.json` are the pose-error transforms used by `merged_with_pose_error.ply`.
- Two merged world-frame clouds are exported:
  - one without camera pose error
  - one with small camera pose error

## Install

pyenv activate ptenv
python -m pip install mujoco numpy pillow pytest

## Generate one scene

python algorithms/fake_cloud/generate_dataset.py --output algorithms/fake_cloud/outputs/sample_scene --seed 0

## Config

python algorithms/fake_cloud/generate_dataset.py --config algorithms/fake_cloud/example_config.json --output algorithms/fake_cloud/outputs/sample_scene --seed 0

## Output layout

outputs/sample_scene/
  scene.json
  config_used.json
  resolved_config.json
  structure_preview.png
  cam_info.json
  view1.ply
  view2.ply
  ...
  merged_without_pose_error.ply
  merged_with_pose_error.ply

## Tests

pyenv activate ptenv
python algorithms/fake_cloud/generate_dataset.py --output algorithms/fake_cloud/outputs/sample_scene --seed 0
