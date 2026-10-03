# AVP Remote Tracking Demo

## Browser and Vision Pro

Open **AVP Remote** in `./geo gui`. The default Snapshot scene needs neither
Isaac Sim nor a headset. Drag to orbit, scroll to zoom, choose USD/URDF/both,
and edit joint sliders. Capture a pose, save/load its JSON, or save a PNG.
Legacy native tracking JSON can also be loaded; it is retargeted locally.

Tracking markers are on by default in both robot views and in WebXR. Gold marks
the head, cyan the left hand, and pink the right hand. Finger connections and
head/wrist orientation axes can be toggled independently. Choose **Tracking
markers only** for a close view, or click a point on desktop to inspect its input
and mapped coordinates. Markers use the exact retargeting pretransform, before
arm IK. Snapshot mode includes the recorded markers without a headset; pose-only
JSON has no tracking markers. Live points disappear when tracking is missing.
WebXR supplies 25 joints per hand; native 27-joint snapshots also show the two
forearm points. No elbows, knees or other missing measurements are invented.

As checked in September 2026, public Vision Pro APIs expose head and hand poses,
but no raw facial blendshape coefficients or wearer full-body skeleton. Apple
explicitly confirms that visionOS has no face-tracking API; matted Persona video
is rendered media, not access to facial rig values
([WWDC26 Group Lab, 27:02](https://developer.apple.com/videos/play/wwdc2026/8004/?time=1622)).
Apple's [ARKit migration guide](https://developer.apple.com/documentation/visionos/bringing-your-arkit-app-to-visionos)
also excludes full-body tracking. Face/body capture would require a separate
source, such as iPhone ARKit or external trackers. Head/hand-driven body IK would
be an estimate, not measured body joints. The UI distinguishes tracking input
from estimated arm IK; legs remain at explicit defaults.

The current parallel URDF package is copied from
`usd_parallel_urdf/outputs/urdf_packages/landau_v10/`. **Asset provenance →
Refresh latest assets** updates that handoff and exports the original USD as a
skinned GLB using Blender. All 68 bones, three PBR textures, and materials are
retained; browser textures are resized to at most 1024 px. Source USD files are
unchanged. Asset hashes detect changed URDF, meshes, USD, textures and mapping
code. Generated files stay in `outputs/web_scene/`.

For Vision Pro, open the same GUI through a **trusted HTTPS** endpoint and press
**Open in Vision Pro**. The GUI itself still binds only to loopback; provision a
private HTTPS reverse proxy separately. Enable **Follow my head & hands**, allow
hand tracking, and use Recenter while holding a comfortable reference pose.
Inside the immersive scene, gaze/pinch the Live/pause, Snapshot, Recenter, and
Exit buttons. Snapshot remains usable if hand tracking is declined. This uses
[`immersive-vr`, supported by Safari on visionOS](https://webkit.org/blog/15865/webkit-features-in-safari-18-0/),
with [natural gaze/pinch input](https://webkit.org/blog/15162/introducing-natural-input-for-webxr-in-apple-vision-pro/).
Real headset presentation still needs device validation.

Live/imported tracking uses the existing head/finger mapping and joint-limited
CCD in an isolated NumPy worker, without Isaac or TRAC-IK. Rendering continues
at the browser's frame rate; pose requests are bounded to one in flight and
interpolated between updates. Set `AVP_WEB_PYTHON` to a Python with NumPy if
needed; the Mac host can also use Blender's bundled Python. WebXR's 25 hand
joints map to the first 25 native joints; no forearm tracking is fabricated.
Missing hands return to the default pose. This is a pose viewer, not physics or
walking control.

Prepare from the command line:

```sh
blender --background --factory-startup --python algorithms/avp_remote/build_web_scene.py
python3 algorithms/avp_remote/build_web_scene.py --check
node --test algorithms/avp_remote/tests/test_browser_math.mjs
node --test algorithms/avp_remote/tests/test_tracking_markers.mjs
```

The Isaac workflows below remain available as legacy diagnostics.


This project streams Apple Vision Pro tracking data (head + hand joints) into a local Python pipeline, retargets the upper body onto the copied Landau URDF, mirrors the solved joint state onto the matching USD character in Isaac Sim, and can optionally show a Unitree H1_2 URDF baseline beside them for pose debugging.

## What it does

- Receives tracking frames from Vision Pro.
- Bridges tracking data over UDP/ZMQ.
- Copies the Landau URDF/USD/skeleton/STL handoff into `algorithms/avp_remote/inputs/landau_v10/`.
- Solves left/right arm IK with the local `helper_repos/pytracik` TRAC-IK helper.
- Maps solved URDF joints onto the USD skeleton so the colored USD model follows the same pose.
- Optionally imports a Unitree H1_2 URDF baseline and drives it from the same solved AVP pose for side-by-side comparison.
- Keeps the legacy wrist/joint/head marker viewer for debugging.
- Includes unit tests for config, schema, transform math, asset setup, and snapshot retargeting.

## Main files

- `avp_config.py`: central runtime settings (with environment overrides).
- `asset_setup.py`: copies the URDF/USD/skeleton/STL inputs from `algorithms/usd_parallel_urdf/`.
- `avp_bridge.py`: sends tracking frames to local bridge transport.
- `run_avp_landau_session.py`: headless-safe Isaac Sim launcher for the AVP Landau session.
- `avp_landau_session.py`: Isaac Sim session that retargets AVP motion onto the Landau URDF joint space and mirrors the pose onto the USD character.
- `avp_wrist_marker.py`: Isaac Sim app that renders markers from incoming tracking data or snapshot file.
- `landau_retarget.py`: AVP head/hand to Landau joint retargeting logic.
- `landau_mapping_config.py`: editable AVP-to-Landau pose mapping rules, per-joint sign/scale, and explicit defaults for untracked legs.
- `landau_pose.py`: apply Landau joint positions onto the copied USD skeleton.
- `avp_tracking_schema.py`: frame extraction/parsing utilities.
- `avp_transform_utils.py`: transform composition and conversion helpers.
- `avp_snapshot_io.py`: shared snapshot JSON read/write helpers.

## Configuration (environment variables)

These values are read from `avp_config.py`:

- `AVP_IP` (default: `192.168.2.206`)
- `BRIDGE_HOST` (default: `127.0.0.1`)
- `BRIDGE_PORT` (default: `45800`)
- `USE_ZMQ` (default: `true`)
- `SEND_HZ` (default: `60`)
- `ISAAC_SIM_SH_PATH` (default: `/home/wishai/vscode/IsaacLab/_isaac_sim/isaac-sim.sh`)
- `AVP_USD_PATH` (default: `inputs/landau_v10/landau_v10.usdc`)
- `AVP_URDF_PATH` (default: `inputs/landau_v10/landau_v10_parallel_mesh.urdf`)
- `AVP_SKELETON_JSON_PATH` (default: `inputs/landau_v10/landau_v10_skeleton.json`)
- `AVP_MESH_ROOT` (default: `inputs/landau_v10/mesh_collision_stl`)
- `AVP_ROBOT_XML_PATH` (default: `google_robot/robot.xml`)
- `AVP_SNAPSHOT_PATH` (default: `avp_snapshot.json`)
- `AVP_H1_2_URDF_PATH` (default: `../../helper_repos/xr_teleoperate_shallow/assets/h1_2/h1_2.urdf`)

Example:

```bash
AVP_IP=192.168.2.90 BRIDGE_PORT=45900 python3 avp_bridge.py
```

Capture snapshot while bridging (press `k` in terminal):

```bash
python3 avp_bridge.py --snapshot-path ./avp_snapshot.json --snapshot-key k
```

Run the full Landau retarget session from a saved snapshot:

```bash
/home/wishai/vscode/IsaacLab/_isaac_sim/python.sh \
  /home/wishai/vscode/geo_lib/algorithms/avp_remote/run_avp_landau_session.py \
  --headless \
  --tracking-source snapshot \
  --snapshot-path /home/wishai/vscode/geo_lib/algorithms/avp_remote/avp_snapshot.json
```

Run the same session with the Isaac Sim GUI:

```bash
/home/wishai/vscode/IsaacLab/_isaac_sim/python.sh \
  /home/wishai/vscode/geo_lib/algorithms/avp_remote/run_avp_landau_session.py \
  --experience base \
  --tracking-source snapshot \
  --snapshot-path /home/wishai/vscode/geo_lib/algorithms/avp_remote/avp_snapshot.json
```

The default snapshot and bridge sessions show two compare views:

- raw Landau URDF meshes on the left
- solved USD Landau character in the center

Enable the third H1_2 compare baseline only if you need it:

```bash
/home/wishai/vscode/IsaacLab/_isaac_sim/python.sh \
  /home/wishai/vscode/geo_lib/algorithms/avp_remote/run_avp_landau_session.py \
  --experience base \
  --tracking-source snapshot \
  --snapshot-path /home/wishai/vscode/geo_lib/algorithms/avp_remote/avp_snapshot.json \
  --baseline
```

For a short GUI smoke test that exits on its own:

```bash
/home/wishai/vscode/IsaacLab/_isaac_sim/python.sh \
  /home/wishai/vscode/geo_lib/algorithms/avp_remote/run_avp_landau_session.py \
  --experience base \
  --tracking-source snapshot \
  --snapshot-path /home/wishai/vscode/geo_lib/algorithms/avp_remote/avp_snapshot.json \
  --max-frames 5
```

Run the full Landau retarget session from live AVP bridge data:

```bash
python3 /home/wishai/vscode/geo_lib/algorithms/avp_remote/avp_bridge.py

/home/wishai/vscode/IsaacLab/_isaac_sim/python.sh \
  /home/wishai/vscode/geo_lib/algorithms/avp_remote/run_avp_landau_session.py \
  --headless \
  --tracking-source bridge \
  --snapshot-path /home/wishai/vscode/geo_lib/algorithms/avp_remote/avp_snapshot.json
```

Optional: attempt a live URDF articulation import in headless mode too:

```bash
/home/wishai/vscode/IsaacLab/_isaac_sim/python.sh \
  /home/wishai/vscode/geo_lib/algorithms/avp_remote/run_avp_landau_session.py \
  --headless \
  --import-stage-urdf \
  --tracking-source snapshot \
  --snapshot-path /home/wishai/vscode/geo_lib/algorithms/avp_remote/avp_snapshot.json
```

Run the live bridge session with the Isaac Sim GUI:

```bash
python3 /home/wishai/vscode/geo_lib/algorithms/avp_remote/avp_bridge.py

/home/wishai/vscode/IsaacLab/_isaac_sim/python.sh \
  /home/wishai/vscode/geo_lib/algorithms/avp_remote/run_avp_landau_session.py \
  --experience base \
  --tracking-source bridge \
  --snapshot-path /home/wishai/vscode/geo_lib/algorithms/avp_remote/avp_snapshot.json
```

Run the legacy marker viewer in snapshot mode:

```bash
/home/wishai/vscode/IsaacLab/_isaac_sim/isaac-sim.sh \
  --exec "/home/wishai/vscode/geo_lib/algorithms/avp_remote/avp_wrist_marker.py \
    --tracking-source snapshot \
    --snapshot-path /home/wishai/vscode/geo_lib/algorithms/avp_remote/avp_snapshot.json"
```

## Run tests

From repo root:

```bash
python3 -m unittest discover -s algorithms/avp_remote/tests -p 'test_*.py'
```

## Notes

- Use `run_avp_landau_session.py` with Isaac Sim's `python.sh` for both headless and GUI launches in this project.
- When using Isaac Sim's `--exec`, pass the script path and its flags as one quoted string. Otherwise Kit consumes flags like `--tracking-source` itself and the marker app falls back to bridge mode.
- `avp_landau_session.py`, `avp_wrist_marker.py`, and `load_usd.py` automatically populate `inputs/landau_v10/` from `algorithms/usd_parallel_urdf/` on startup.
- The runtime retargeter always uses the copied URDF offline, but stage-side URDF import is skipped by default because that importer is unstable here. Pass `--import-stage-urdf` to try it explicitly.
- `avp_landau_session.py` keeps the H1_2 compare baseline disabled by default because the Isaac Sim URDF importer can hard-crash on that asset. Pass `--baseline` to opt in, `--no-g1` remains accepted as a compatibility alias, and `--g1-urdf-path` / `--h1-2-urdf-path` both override the baseline URDF path.
- In `snapshot` mode, the pose is applied once and the GUI stays open until you close Isaac Sim. Add `--max-frames 5` if you want an auto-exiting smoke test.
- The runtime retargeter uses the local `helper_repos/pytracik` build for arm IK, so no extra pip install is required in this repo.
- AVP tracking currently retargets head, arms, and hands only. Leg pose keys are held at explicit zero defaults in `landau_mapping_config.py`.
- The Landau asset's zero-angle URDF/skeleton pose is not a straight neutral stance, because several leg joints already carry non-identity `origin rpy` / bind transforms in the source asset.
- If you want to flip a joint direction or reduce a joint response, edit `landau_mapping_config.py` and change that pose key's `scale`. A negative `scale` flips the sign.
- Unit tests are lightweight and can run without Isaac Sim.
