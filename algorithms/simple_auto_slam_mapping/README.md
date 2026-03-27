# Simple Auto SLAM Mapping

This stage consumes a copied scene package from `algorithms/svg_scene_builder` and produces a ROS2-compatible occupancy map.

Sandbox rule:
- do not import or call code from `algorithms/svg_scene_builder`
- copy the scene-builder output files into this folder's input directory first
- mapping inputs must live under an `inputs/` directory
- mapping outputs must live under an `outputs/` directory
- run mapping only against the copied input directory

Required input files:
- `scene.xml`
- `semantic_layout.npz`
- `layout_preview.png`
- `start_pose.json`
- `scene_summary.json`
- `scene_package.json`

Outputs:
- `map.pgm`
- `map.yaml`
- `trajectory.json`
- `route.json`
- `mapping_summary.json`
- `input_package_snapshot.json`
- `snapshots/map_010s.pgm`, `snapshots/map_020s.pgm`, ... every 10 seconds by default

## Copy Input And Run

```bash
pyenv activate ptenv
mkdir -p algorithms/simple_auto_slam_mapping/inputs/sample_scene
cp algorithms/svg_scene_builder/outputs/sample_scene/scene.xml algorithms/simple_auto_slam_mapping/inputs/sample_scene/
cp algorithms/svg_scene_builder/outputs/sample_scene/semantic_layout.npz algorithms/simple_auto_slam_mapping/inputs/sample_scene/
cp algorithms/svg_scene_builder/outputs/sample_scene/layout_preview.png algorithms/simple_auto_slam_mapping/inputs/sample_scene/
cp algorithms/svg_scene_builder/outputs/sample_scene/start_pose.json algorithms/simple_auto_slam_mapping/inputs/sample_scene/
cp algorithms/svg_scene_builder/outputs/sample_scene/scene_summary.json algorithms/simple_auto_slam_mapping/inputs/sample_scene/
cp algorithms/svg_scene_builder/outputs/sample_scene/scene_package.json algorithms/simple_auto_slam_mapping/inputs/sample_scene/
python -m algorithms.simple_auto_slam_mapping.mapping \
  --input algorithms/simple_auto_slam_mapping/inputs/sample_scene \
  --output algorithms/simple_auto_slam_mapping/outputs/sample_run \
  --timeout 60.0 \
  --snapshot-period 10.0
```

You can also let the mapping CLI perform the copy step for convenience. It still copies into `inputs/` before loading:

```bash
pyenv activate ptenv
python -m algorithms.simple_auto_slam_mapping.mapping \
  --copy-from algorithms/svg_scene_builder/outputs/sample_scene \
  --input algorithms/simple_auto_slam_mapping/inputs/sample_scene \
  --output algorithms/simple_auto_slam_mapping/outputs/sample_run \
  --timeout 60.0 \
  --snapshot-period 10.0
```

## Test

```bash
pyenv activate ptenv
pytest algorithms/simple_auto_slam_mapping/tests -q
```
