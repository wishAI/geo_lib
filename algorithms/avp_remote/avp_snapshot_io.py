import json
import os
import tempfile
from pathlib import Path


class SnapshotIOError(Exception):
    """Base class for snapshot read/write errors."""


class SnapshotNotFoundError(SnapshotIOError):
    """Raised when the snapshot file does not exist."""


class SnapshotDecodeError(SnapshotIOError):
    """Raised when the snapshot file cannot be decoded as valid payload JSON."""


def save_snapshot_payload(payload, snapshot_path):
    path = Path(snapshot_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_path = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=True)
            handle.write("\n")
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def load_snapshot_payload(snapshot_path):
    path = Path(snapshot_path).expanduser()
    if not path.exists():
        raise SnapshotNotFoundError(f"Snapshot file does not exist: {path}")

    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except json.JSONDecodeError as exc:
        raise SnapshotDecodeError(f"Snapshot file is not valid JSON: {path}") from exc
    except OSError as exc:
        raise SnapshotIOError(f"Failed to read snapshot file: {path}") from exc

    if not isinstance(payload, dict):
        raise SnapshotDecodeError(f"Snapshot payload must be a JSON object: {path}")
    return payload
