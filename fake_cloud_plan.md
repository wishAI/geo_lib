Build a small Python project that generates synthetic point cloud test data for a “ship part” style scene using MuJoCo in headless/offscreen mode only, no GUI window.
Write this project under the algorithms folder. 

Goal
- Generate a simple synthetic dataset for testing point-cloud algorithms.
- The scene should look like:
  - one large bottom board
  - multiple vertical boards mounted on it
  - some vertical boards should intersect each other
- Capture depth from multiple camera angles
- Convert depth to point clouds
- Export per-view PLY files
- Export a JSON file for each view containing the world camera pose
- Export one merged cloud from all views
- Add configurable small camera pose errors so the merged cloud shows slight “double vision”
- Export one colored visualization image that shows each plane/board in a different color for quick inspection of scene structure

Camera baseline
Use a depth-camera configuration roughly based on Orbbec Astra / Astra Pro class specs:
- depth resolution: 640x480
- depth FPS target: 30
- depth FOV: about H=58.4 deg, V=45.5 deg
- usable depth range model around 0.6m to 8m
- precision reference around ±3 mm at 1 m
Do not try to exactly emulate firmware behavior. Just use these as reasonable defaults for synthetic generation.

Important references for defaults:
- Orbbec Astra series official page lists 640x480@30fps, H58.4° V45.5°, range about 0.6–8 m, precision ±3 mm @ 1 m.
- Astra Mini Pro page shows the same 640x480@30fps and H58.4° V45.5° class values.
Use these values as configurable defaults in a config file, not hardcoded everywhere.

Technical requirements
1. Use Python.
2. Use MuJoCo for scene definition and offscreen depth rendering.
3. Must work without GUI.
4. Use a clean project structure.
5. Use deterministic random seed support.
6. Keep the implementation simple and readable.

Suggested project structure
- generate_dataset.py
- scene_builder.py
- renderer.py
- camera_sampler.py
- pointcloud.py
- noise_model.py
- visualize.py
- config.py
- tests/
- outputs/

Scene requirements
- Bottom board:
  - one large flat rectangular board
  - example size around 1.2m x 0.8m x 0.03m
- Vertical boards:
  - create 3 to 6 thin upright boards
  - each board is rectangular
  - some can be parallel, some can intersect
  - attach them on top of the bottom board
  - randomize positions and lengths within a controlled range
- Keep the scene mostly planar and simple
- Store exact ground-truth board transforms and dimensions in a scene JSON

Rendering requirements
- Create multiple fixed camera views around the object
- Example: 8 views around a ring or partial ring, looking at the scene center
- Also allow camera elevation differences
- Render depth offscreen
- Convert depth into 3D points in camera frame, then transform to world frame
- Save:
  - one PLY per camera view
  - one JSON per camera view containing camera pose in world coordinates
  - one merged PLY from all views

Camera pose JSON format
For each view, save something like:
{
  "camera_name": "...",
  "position_world": [x, y, z],
  "rotation_world_from_camera": [[...],[...],[...]],
  "transform_world_from_camera": [[4x4]],
  "fov_y_deg": ...,
  "width": 640,
  "height": 480
}

Noise/error model
Implement two separate configurable effects:

A. Depth noise
- Add small range-dependent depth noise
- Keep it simple, e.g. sigma(z) = a + b * z^2
- Default should be mild, roughly in the spirit of a few mm at around 1 m

B. Camera pose error
- Before converting each view cloud into world frame, perturb the camera pose slightly
- Small translation noise and small rotation noise
- This should be configurable
- The goal is that the merged cloud shows a slight double-edge / double-vision effect
- Also allow disabling this for clean output

Visualization image requirement
Generate an image that visualizes the basic structure of the ship-part scene:
- each plane/board should have a different solid color
- this image is for human inspection
- simplest acceptable version:
  - render a top-down or angled scene preview with each board colored differently
- save as PNG
- file name example: structure_preview.png

Outputs per scene
For one generated scene, output:
- scene.json
- structure_preview.png
- clean/
  - view_000.ply
  - view_000_camera.json
  - ...
  - merged_clean.ply
- noisy/
  - view_000.ply
  - view_000_camera.json
  - ...
  - merged_noisy.ply

Tests
Write tests. Keep them practical and automatic.

1. Scene generation test
- verify bottom board exists
- verify at least 3 vertical boards exist
- verify some boards are vertical within tolerance
- verify scene metadata JSON is valid

2. Point count range test
- after generating a scene with default settings, each per-view point cloud should have point count in a reasonable range
- the merged cloud should also have point count in a reasonable range
- choose sensible lower/upper bounds based on image resolution and expected visibility
- do not use overly brittle exact numbers

3. 3D bounding box range test
- for the merged cloud, compute axis-aligned bounding box
- verify bbox dimensions are within expected ranges for the generated ship-part scene
- example idea:
  - x extent should be within a range consistent with board lengths
  - y extent within scene width range
  - z extent should be roughly bottom thickness + vertical board height range
- again use tolerance ranges, not exact values

4. Camera pose JSON test
- verify each output view has a matching camera JSON
- verify transform shape is 4x4
- verify rotation matrix is close to orthonormal

5. Pose-noise effect test
- generate one clean merged cloud and one noisy merged cloud with pose perturbation enabled
- confirm they are not identical
- confirm noisy merged cloud remains within reasonable bbox bounds
- optionally measure average nearest-neighbor difference > threshold

6. Visualization image test
- verify PNG exists
- verify image dimensions are nonzero
- verify there are multiple unique colors present

Implementation guidance
- Prefer simple thin boxes for boards instead of trying to use infinite planes
- Keep board dimensions and positions configurable
- Make a default config file for:
  - board sizes
  - number of views
  - camera intrinsics/FOV
  - noise levels
  - output paths
  - random seed
- Write a simple CLI:
  - python generate_dataset.py --output outputs/sample_scene --seed 0
- Make it easy to generate one scene first

Dependencies
Use only commonly available packages if possible:
- mujoco
- numpy
- scipy if needed
- trimesh or open3d or plyfile for saving PLY
- pillow or matplotlib for visualization image
- pytest for tests

Acceptance criteria
The task is done when:
1. I can run one command to generate one sample scene
2. It runs headless without opening GUI
3. It outputs multiple per-view PLY files
4. It outputs matching camera-pose JSON files
5. It outputs a merged cloud
6. It outputs a colored structure preview PNG
7. Tests pass
8. Clean and noisy outputs are both supported
9. Noisy merged cloud shows slight multi-view misalignment from camera pose perturbation

Please also include:
- a short README with setup and run commands
- one example config file
- one example output folder layout
- comments in code where depth is converted to points and where pose noise is injected
