# AVP Remote Tracking Demo

This project streams Apple Vision Pro tracking data (head + hand joints) into a local Python pipeline, retargets the upper body onto the copied Landau URDF, mirrors the solved joint state onto the matching USD character in Isaac Sim, and can show a Unitree G1 URDF baseline beside them for pose debugging.

## What it does

- Receives tracking frames from Vision Pro.
- Bridges tracking data over UDP/ZMQ.
- Copies the Landau URDF/USD/skeleton/STL handoff into `algorithms/avp_remote/inputs/landau_v10/`.
- Solves left/right arm IK with the local `helper_repos/pytracik` TRAC-IK helper.
- Maps solved URDF joints onto the USD skeleton so the colored USD model follows the same pose.
- Imports a Unitree G1 URDF baseline and drives it from the same solved AVP pose for side-by-side comparison.
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
- `AVP_G1_URDF_PATH` (default: `../../helper_repos/unitree_ros/robots/g1_description/g1_29dof_with_hand_rev_1_0.urdf`)

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

The snapshot and bridge sessions now show three baselines by default:

- raw Landau URDF meshes on the left
- solved USD Landau character in the center
- Unitree G1 URDF on the right

Disable the G1 baseline if needed:

```bash
/home/wishai/vscode/IsaacLab/_isaac_sim/python.sh \
  /home/wishai/vscode/geo_lib/algorithms/avp_remote/run_avp_landau_session.py \
  --experience base \
  --tracking-source snapshot \
  --snapshot-path /home/wishai/vscode/geo_lib/algorithms/avp_remote/avp_snapshot.json \
  --no-g1
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
- `avp_landau_session.py` imports the G1 baseline by default when `AVP_G1_URDF_PATH` exists. Use `--no-g1` to disable it or `--g1-urdf-path` to point at a different G1 URDF.
- In `snapshot` mode, the pose is applied once and the GUI stays open until you close Isaac Sim. Add `--max-frames 5` if you want an auto-exiting smoke test.
- The runtime retargeter uses the local `helper_repos/pytracik` build for arm IK, so no extra pip install is required in this repo.
- Unit tests are lightweight and can run without Isaac Sim.
