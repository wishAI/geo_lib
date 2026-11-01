# AVP Remote Tracking Demo

## Motion studio and LAN access

The redesigned studio includes a bottom timeline, play/pause, seek, 0.25–2×
speed, looping, JSON import/export and automatic IndexedDB storage of every
take on the same browser/origin. The take selector restores previous recordings.
Stopping leaves live tracking active; **录制新一段** also resumes tracking from
playback and begins once a fresh head sample arrives. In Vision Pro Safari, enter immersive mode,
allow hand tracking, and pinch **开始录制** / **结束** in the scene. Each take is
limited to 60 seconds / 1,800 samples, at a target 30 Hz. Tracking samples, missing
tracking, unaligned Z-up input, alignment and the displayed robot pose are saved;
playback requires no headset or pose solver. Displayed IK is currently solved
locally in every XR frame with bounded, warm-started CCD and short pose smoothing; recorded robot joints are estimates, not measured
body motion. Session end or loss of visibility stops recording and saves the take.
Export JSON for a durable backup; browser storage is not shared between URLs.

AVP is part of the main Geo Library service, on its normal port 8767:

```sh
./geo gui --host 0.0.0.0 --port 8767 --no-browser
```

Open the Geo Library homepage and select **AVP Remote**, or use
`/#/sandbox/avp_remote`. Localhost remains the default when `--host` is omitted.
On this TK2 the LAN homepage is `http://192.168.2.31:8767/`.
For Vision Pro capture without Tailscale, first open
`http://192.168.2.31:8767/setup/vision-pro` in the headset's Safari. Install the
certificate-only profile and enable full trust for **Geo Library LAN Root** under
Settings → General → About → Certificate Trust Settings. Then open
`https://192.168.2.31:8443/#/sandbox/avp_remote`, enter immersive mode, allow hand
tracking, and pinch **开始录制** / **结束**. HTTP permits preview/playback only;
Safari requires a trusted secure context for WebXR even on private Wi-Fi.

Both LAN listeners run in the same Geo GUI process and bind only to
`192.168.2.31`. No public tunnel or headset VPN is needed. The user service
`geo-web-gui.service` keeps the GUI running independently of a terminal session.
Host TLS identity files live outside the repository in
`~/.local/state/geo-lib/tls/`; private keys are never served. To provision another
LAN address, run `python3 -m webgui.lan_https --host <LAN-IP>` and pass the generated
`--tls-cert`, `--tls-key`, `--tls-profile`, `--https-port 8443`, and `--https-url`
to `./geo gui`. Changing the host IP requires updating the service's host and URL;
the provisioner renews the server certificate while preserving the installed CA.

The server compresses and revalidates browser assets, and serves AVP's declared
local package without scanning cloud copies or other sandboxes. The base URDF and
capture controls become usable before USD and Unitree finish downloading; model
loads run concurrently with visible progress and a 45-second network timeout.
Browser-only PBR textures are capped at 512 px; original USD and textures are unchanged.
Both GLBs use Draco geometry compression (16-bit positions). The matching Three.js
r168 WASM decoder is served locally, gzip-compressed and cached with ETags; no
external CDN is needed. GLBs are also stored in the browser by SHA-256; changed
versions replace old entries, corrupted entries are downloaded again, and blocked
storage falls back to the network. The original skin, joint hierarchy and approved head mesh
remain intact. See [Three.js DRACOLoader](https://threejs.org/docs/pages/DRACOLoader.html).
A Chrome measurement on 2026-11-01 with a simulated 2 Mbps / 60 ms connection
reduced cold full-model loading from 16.57 s to 10.00 s; a second visit took
3.19 s. The two GLBs shrank from 6.07 MB to 1.39 MB (gzip transfer: 3.07 MB
to 1.29 MB). These are controlled browser measurements, not headset Wi-Fi timings.
Background resource/job refreshes preserve the mounted recording session. Live head, arm and finger retargeting runs in the browser and does not call
`/api/avp/pose`; that endpoint remains available for imported single frames.
Mapping rules are exported from `landau_mapping_config.py`, and all URDF limits
remain enforced. The immersive HUD reports the actual update Hz. A desktop Chrome
simulation with the pose endpoint blocked measured 58 updates in 2 seconds and
0.2 ms median / 0.3 ms p95 solve time (software rendering, not a headset FPS claim).

The old producer package still contains a 540-face alpha-shape `head_x.stl`.
`inputs/landau_v10/mesh_overrides.json` pins the corrected 1,200-face rabbit-ear
head, handed off as a file from the retained walking input. Asset preparation
copies this verified mesh into AVP's own inputs after synchronizing the package.
It preserves a valid local copy when the producer is unavailable and fails if a
replacement has a different hash; it never silently falls back to the old head.
Browser asset checks include the override manifest, copied head hash and builder
code. Changing the approved version requires explicitly updating the manifest.

### Browser versus Tracking Streamer

The existing native bridge uses
[Improbable-AI/VisionProTeleop](https://github.com/Improbable-AI/VisionProTeleop).
Its native ARKit code includes 27 joints per hand (two additional forearm joints)
and configurable predicted hand poses. WebXR provides 25 hand joints and a viewer
pose after permission in an immersive session. Gaze/pinch interactions do not
expose a continuous eye-tracking stream. Native RealityKit mixed-reality rendering,
UVC recording, marker tracking and simulation streaming are not reproduced by this
Three.js `immersive-vr` viewer. Headset calibration, performance and actual hand
orientation still require device validation; browser QA uses explicitly synthetic
XR inputs and does not establish native/web tracking equivalence.

## Browser and Vision Pro

The immersive renderer uses a 0.65 framebuffer scale, maximum available fixed
foveation, and no MSAA or preserved drawing buffer. Each native hand joint is
read once per XR frame and shared by the first-person hands and retargeter.
Unavailable or invalid transforms are omitted. Update exceptions cannot stop
frame submission; repeated update failures pause capture, while repeated render
failures end the session with an error. Settings → 沉浸运行状态 retains bounded
local diagnostics (frame counters, CPU time, draw calls and errors, no tracking
poses); 导出运行诊断 exports the report after a failure or browser reload.

`tests/browser_xr_runtime.py` exercises the actual Three.js XR manager and stereo
render path with a simulated device/framebuffer, including injected joint and
solver errors, pinch recording twice, playback hands, visibility and reentry.
It requires Playwright and the running Geo GUI (`AVP_GUI_URL` overrides the LAN
URL). This test cannot certify Safari/visionOS GPU performance or headset stability.

AVP fills the available browser viewport, with capture controls above the scene
and the transport fixed below it. **设置** opens an internal settings sidebar;
**← Geo Library** returns to the normal homepage. The document itself does not
scroll, including narrow and short windows.

Inside immersive mode, first-person hands show measured joints and bone lines
at their unscaled WebXR reference-space positions (left cyan, right pink). They
keep updating while robot capture is paused or a take is playing, and clear when
tracking, session visibility or the session is lost. Missing joints are omitted;
no missing bone connections are invented. **沉浸模式：显示自己的手** toggles them
independently of the comparison-model markers.

The top capture guide explicitly separates preview, immersive connection, live
head/left/right-hand tracking, and recording. Press **开始头手追踪**, allow both
system permissions, hold hands in front of the chest, then use **开始录制** in the
scene. Colored status indicators use fresh WebXR samples only, never snapshot or
playback data. Permission failures and HTTP-only access show actionable guidance.
The scene controls sit 35 cm above the initial eye height, 1.55 m forward,
scaled to 68%, above the models. They include a Chinese tracking-status panel. Head data is required to start a take; Stop remains
available if tracking is lost. Recording and playback remain separate actions.

Immersive buttons use a visible head-directed reticle, animated hover borders and
press/click feedback. visionOS does not expose continuous eye gaze: the actual
eye-directed transient ray is supplied at pinch start. The button target is
latched on `selectstart` and activated once on `select`, even if the release ray
has moved or disappeared. A cancelled gesture does not activate anything.
See [WebKit natural input](https://webkit.org/blog/15162/introducing-natural-input-for-webxr-in-apple-vision-pro/).


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
retained; browser textures are resized to at most 512 px. Source USD files are
unchanged. Asset hashes detect changed URDF, meshes, USD, textures and mapping
code. Generated files stay in `outputs/web_scene/`.

For Vision Pro, open the same GUI through the trusted LAN HTTPS endpoint and
press **开始头手追踪**, then allow hand tracking. Use **重新校准** while holding
a comfortable reference pose. Inside the immersive scene, use the head reticle
or the eye-directed pinch-start ray to select Record, Play, Live/pause, Recenter,
and Exit. This uses
[`immersive-vr`, supported by Safari on visionOS](https://webkit.org/blog/15865/webkit-features-in-safari-18-0/).
Real headset presentation and input still need device validation.

Live tracking uses the browser-local head/finger mapper and warm-started CCD on
each XR frame. Imported single-frame tracking can still use the isolated NumPy
worker, without Isaac or TRAC-IK; set `AVP_WEB_PYTHON` to a Python with NumPy for
that endpoint. WebXR's 25 hand
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

### Comparison view

The four views, left to right, show Unitree **H1-2**, Landau USD, corrected
Landau URDF, and head/hand input. Unitree faces the same display direction as Landau. Unitree uses the repository's existing heuristic Landau→H1-2
mapping (not direct full-body tracking or a command to hardware), including joint
limits and URDF finger mimic joints. Its display scale is 0.52×; coordinates and
recorded data remain in meters. H1-2 assets come from
`helper_repos/xr_teleoperate_shallow/assets/h1_2`; the browser build exports reduced
visual meshes and records hashes of the URDF, meshes and mapping implementation.

The input view uses colored measured head/hand markers and finger links. Gray
dashed head/neck/shoulder/torso/wrist connections are **schematic**, not measured
body joints. Missing head data removes the schematic; missing hands remove their
connections. These lines are never saved as tracking observations. All four views
follow the same live or playback frame. Select an individual view for inspection.
