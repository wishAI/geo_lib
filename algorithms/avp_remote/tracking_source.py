from __future__ import annotations

import importlib
import json
import socket
from pathlib import Path

from avp_snapshot_io import SnapshotIOError, load_snapshot_payload
from avp_tracking_schema import extract_tracking_frame
from avp_config import (
    AVP_IP,
    AVP_SNAPSHOT_PATH,
    BRIDGE_HOST,
    BRIDGE_PORT,
    USE_ZMQ,
    ZMQ_CONNECT_ENDPOINT,
    ZMQ_SUB_TOPICS,
)


def _print_error(context, err) -> None:
    print(f"[AVP] {context}: {type(err).__name__}: {err}")


class TrackingSourceError(Exception):
    pass


try:
    import zmq

    HAS_ZMQ = True
except Exception as exc:
    zmq = None
    HAS_ZMQ = False
    _print_error("zmq import failed", exc)


USE_DIRECT_AVP_STREAM = False


def _load_vision_pro_streamer():
    try:
        module = importlib.import_module("avp_stream")
    except Exception as exc:
        raise TrackingSourceError(
            "Live AVP streaming requires the external 'avp_stream' module, "
            "but it could not be imported in this environment."
        ) from exc

    streamer_cls = getattr(module, "VisionProStreamer", None)
    if streamer_cls is None:
        raise TrackingSourceError(
            "The imported 'avp_stream' module does not expose VisionProStreamer."
        )
    return streamer_cls


class TrackingStream:
    def __init__(self, tracking_source: str = "bridge", snapshot_path: Path | str = AVP_SNAPSHOT_PATH):
        if tracking_source not in ("bridge", "snapshot"):
            raise TrackingSourceError(f"Unsupported tracking source: {tracking_source}")

        self.tracking_source = tracking_source
        self.snapshot_path = Path(snapshot_path).expanduser()
        self.streamer = None
        self.bridge_sock = None
        self.snapshot_frame = None
        self._setup()

    def _make_bridge_udp_socket(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind((BRIDGE_HOST, BRIDGE_PORT))
        sock.setblocking(False)
        return ("udp", sock)

    def _make_bridge_zmq_socket(self):
        ctx = zmq.Context.instance()
        sock = ctx.socket(zmq.SUB)
        sock.connect(ZMQ_CONNECT_ENDPOINT)
        for topic in ZMQ_SUB_TOPICS:
            sock.setsockopt(zmq.SUBSCRIBE, topic)
        return ("zmq", sock)

    def _setup(self):
        if self.tracking_source == "snapshot":
            try:
                payload = load_snapshot_payload(self.snapshot_path)
            except SnapshotIOError as exc:
                raise TrackingSourceError(f"Snapshot mode failed: {exc}") from exc
            self.snapshot_frame = extract_tracking_frame(payload)
            print(f"[AVP] Using snapshot tracking from {self.snapshot_path}")
            return

        if USE_DIRECT_AVP_STREAM and AVP_IP:
            vision_pro_streamer = _load_vision_pro_streamer()
            self.streamer = vision_pro_streamer(ip=AVP_IP)
            self.streamer.start_webrtc()
            print(f"[AVP] Using direct avp_stream to Vision Pro at {AVP_IP}")
            return

        if USE_ZMQ and HAS_ZMQ:
            self.bridge_sock = self._make_bridge_zmq_socket()
            print(f"[AVP] Listening bridge via ZMQ on {ZMQ_CONNECT_ENDPOINT}")
            return

        self.bridge_sock = self._make_bridge_udp_socket()
        print(f"[AVP] Listening bridge via UDP on {BRIDGE_HOST}:{BRIDGE_PORT}")

    def _recv_bridge(self):
        kind, sock = self.bridge_sock
        try:
            if kind == "zmq":
                parts = sock.recv_multipart(flags=zmq.NOBLOCK)
                if len(parts) < 2:
                    return None
                payload = parts[1]
            else:
                payload, _ = sock.recvfrom(65535)
        except BlockingIOError:
            return None
        except Exception:
            return None

        if not payload:
            return None

        try:
            return json.loads(payload.decode("utf-8"))
        except Exception as exc:
            _print_error("bridge json decode failed", exc)
            return None

    def get_tracking_frame(self):
        if self.tracking_source == "snapshot":
            return self.snapshot_frame

        raw = self.streamer.get_latest() if self.streamer is not None else self._recv_bridge()
        if raw is None:
            return None
        return extract_tracking_frame(raw)
