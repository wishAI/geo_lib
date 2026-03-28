# AVP Remote Tracking Demo

This project streams Apple Vision Pro tracking data (head + hand joints) into a local Python pipeline and visualizes markers in Isaac Sim on top of a USD character.

## What it does

- Receives tracking frames from Vision Pro.
- Bridges tracking data over UDP/ZMQ.
- Converts tracking matrices into USD/world transforms.
- Visualizes wrist/joint/head markers in Isaac Sim.
- Includes unit tests for schema and transform math.

## Main files

- `config.py`: central runtime settings (with environment overrides).
- `avp_bridge.py`: sends tracking frames to local bridge transport.
- `avp_wrist_marker.py`: Isaac Sim app that renders markers from incoming tracking data or snapshot file.
- `avp_tracking_schema.py`: frame extraction/parsing utilities.
- `avp_transform_utils.py`: transform composition and conversion helpers.
- `avp_snapshot_io.py`: shared snapshot JSON read/write helpers.

## Configuration (environment variables)

These values are read from `config.py`:

- `AVP_IP` (default: `192.168.2.206`)
- `BRIDGE_HOST` (default: `127.0.0.1`)
- `BRIDGE_PORT` (default: `45800`)
- `USE_ZMQ` (default: `true`)
- `SEND_HZ` (default: `60`)
- `AVP_USD_PATH` (default: `landau_v8.usdc` for marker app, `landau_v10.usdc` for USD loader)
- `AVP_ROBOT_XML_PATH` (default: `google_robot/robot.xml`)
- `AVP_SNAPSHOT_PATH` (default: `avp_snapshot.json`)

Example:

```bash
AVP_IP=192.168.2.90 BRIDGE_PORT=45900 python3 avp_bridge.py
```

Capture snapshot while bridging (press `k` in terminal):

```bash
python3 avp_bridge.py --snapshot-path ./avp_snapshot.json --snapshot-key k
```

Run marker in snapshot replay mode:

```bash
python3 avp_wrist_marker.py --tracking-source snapshot --snapshot-path ./avp_snapshot.json
```

## Run tests

From repo root:

```bash
python3 -m unittest discover -s tests -p 'test_*.py'
```

## Notes

- Running visualization scripts requires an Isaac Sim environment with required dependencies.
- Unit tests are lightweight and can run without Isaac Sim.
