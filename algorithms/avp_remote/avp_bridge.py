import argparse
import importlib
import json
import os
import select
import socket
import sys
import time
from pathlib import Path

from avp_snapshot_io import save_snapshot_payload
from avp_tracking_schema import extract_tracking_frame, frame_to_payload
from avp_config import (
    AVP_IP,
    AVP_SNAPSHOT_PATH,
    BRIDGE_HOST,
    BRIDGE_PORT,
    SEND_HZ,
    USE_ZMQ,
    ZMQ_BIND,
    ZMQ_BIND_ENDPOINT,
    ZMQ_CONNECT_ENDPOINT,
    ZMQ_TOPIC_TRACKING,
)

try:
    import zmq
    HAS_ZMQ = True
except Exception:
    zmq = None
    HAS_ZMQ = False


def _load_vision_pro_streamer():
    try:
        module = importlib.import_module("avp_stream")
    except Exception as exc:
        raise RuntimeError(
            "Live AVP bridge requires the external 'avp_stream' module. "
            "Run this command from the environment where that module is installed."
        ) from exc

    streamer_cls = getattr(module, "VisionProStreamer", None)
    if streamer_cls is None:
        raise RuntimeError("The imported 'avp_stream' module does not expose VisionProStreamer.")
    return streamer_cls


class SnapshotKeyListener:
    def __init__(self, snapshot_key):
        self.snapshot_key = snapshot_key
        self.enabled = False
        self._fd = None
        self._termios = None
        self._tty = None
        self._old_attrs = None

        if not sys.stdin.isatty():
            print("[AVP] Snapshot key listener disabled because stdin is not a TTY.")
            return

        try:
            import termios
            import tty

            self._termios = termios
            self._tty = tty
            self._fd = sys.stdin.fileno()
            self._old_attrs = termios.tcgetattr(self._fd)
            tty.setcbreak(self._fd)
            self.enabled = True
        except Exception as exc:
            print(f"[AVP] Snapshot key listener disabled: {type(exc).__name__}: {exc}")
            self.enabled = False

    def poll_pressed(self):
        if not self.enabled:
            return False

        pressed = False
        while True:
            ready, _, _ = select.select([sys.stdin], [], [], 0)
            if not ready:
                break

            try:
                ch = os.read(self._fd, 1)
            except Exception:
                break

            if not ch:
                break
            if ch.decode("utf-8", errors="ignore") == self.snapshot_key:
                pressed = True
        return pressed

    def close(self):
        if not self.enabled or self._termios is None or self._old_attrs is None:
            return
        try:
            self._termios.tcsetattr(self._fd, self._termios.TCSADRAIN, self._old_attrs)
        except Exception:
            pass


def _parse_args():
    parser = argparse.ArgumentParser(description="Bridge AVP tracking frames and support one-key snapshot capture.")
    parser.add_argument(
        "--snapshot-path",
        default=str(AVP_SNAPSHOT_PATH),
        help="Path to overwrite with the latest tracking payload when snapshot key is pressed.",
    )
    parser.add_argument(
        "--snapshot-key",
        default="k",
        help="Single character key that saves the latest frame payload to --snapshot-path.",
    )
    args = parser.parse_args()

    if len(args.snapshot_key) != 1:
        parser.error("--snapshot-key must be a single character")

    return args


def _make_bridge_sender():
    if USE_ZMQ and HAS_ZMQ:
        ctx = zmq.Context.instance()
        sock = ctx.socket(zmq.PUB)
        if ZMQ_BIND:
            sock.bind(ZMQ_BIND_ENDPOINT)
        else:
            sock.connect(ZMQ_CONNECT_ENDPOINT)
        return ("zmq", sock)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    return ("udp", sock)


def _send_bridge(bridge, payload):
    kind, sock = bridge
    raw = json.dumps(payload).encode("utf-8")
    if kind == "zmq":
        sock.send_multipart([ZMQ_TOPIC_TRACKING, raw])
    else:
        sock.sendto(raw, (BRIDGE_HOST, BRIDGE_PORT))


def main():
    args = _parse_args()
    snapshot_path = Path(args.snapshot_path).expanduser()

    vision_pro_streamer = _load_vision_pro_streamer()
    streamer = vision_pro_streamer(ip=AVP_IP)
    streamer.start_webrtc()

    bridge = _make_bridge_sender()
    key_listener = SnapshotKeyListener(args.snapshot_key)
    if key_listener.enabled:
        print(f"[AVP] Press '{args.snapshot_key}' to overwrite snapshot: {snapshot_path}")

    latest_payload = None
    period = 1.0 / float(SEND_HZ)
    try:
        while True:
            tracking = streamer.get_latest()
            if tracking:
                frame = extract_tracking_frame(tracking)
                payload = frame_to_payload(frame)
                latest_payload = payload
                _send_bridge(bridge, payload)

            if key_listener.poll_pressed():
                if latest_payload is None:
                    print("[AVP] Snapshot requested, but no frame has been received yet.")
                else:
                    try:
                        save_snapshot_payload(latest_payload, snapshot_path)
                        print(f"[AVP] Snapshot saved to {snapshot_path}")
                    except Exception as exc:
                        print(f"[AVP] Snapshot save failed: {type(exc).__name__}: {exc}")

            time.sleep(period)
    finally:
        key_listener.close()


if __name__ == "__main__":
    main()
