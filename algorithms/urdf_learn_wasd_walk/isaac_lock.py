from __future__ import annotations

import atexit
import fcntl
import json
import os
import sys
from argparse import Namespace
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .asset_paths import outputs_dir


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_ready(value: Any) -> Any:
    if isinstance(value, Namespace):
        return {key: _json_ready(val) for key, val in vars(value).items()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def isaac_lock_dir() -> Path:
    path = outputs_dir() / "locks"
    path.mkdir(parents=True, exist_ok=True)
    return path


def isaac_lock_path() -> Path:
    return isaac_lock_dir() / "isaac_sim.lock"


def isaac_lock_metadata_path() -> Path:
    return isaac_lock_dir() / "isaac_sim.lock.json"


def legacy_isaac_lock_path() -> Path:
    return outputs_dir() / "history" / "isaac_sim.lock"


def legacy_isaac_lock_metadata_path() -> Path:
    return outputs_dir() / "history" / "isaac_sim.lock.json"


def _read_lock_metadata(metadata_path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _can_acquire_lock(lock_path: Path) -> bool:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as lock_file:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
    return True


def assert_no_active_isaac_lock() -> None:
    for lock_path, metadata_path in (
        (isaac_lock_path(), isaac_lock_metadata_path()),
        (legacy_isaac_lock_path(), legacy_isaac_lock_metadata_path()),
    ):
        if _can_acquire_lock(lock_path):
            continue
        metadata = _read_lock_metadata(metadata_path)
        message = (
            "Another Isaac workflow is already running. Refusing to continue because multiple Isaac processes "
            "can crash this project."
        )
        if isinstance(metadata, dict):
            message += (
                f" Active holder: script={metadata.get('script_name')} pid={metadata.get('pid')} "
                f"started_at={metadata.get('started_at')} cwd={metadata.get('cwd')}"
            )
        raise RuntimeError(message)


@dataclass
class IsaacLockHandle:
    script_name: str
    lock_file: Any
    metadata_path: Path
    released: bool = False

    def release(self) -> None:
        if self.released:
            return
        self.released = True
        try:
            self.metadata_path.unlink(missing_ok=True)
        except Exception:
            pass
        try:
            fcntl.flock(self.lock_file.fileno(), fcntl.LOCK_UN)
        except Exception:
            pass
        try:
            self.lock_file.close()
        except Exception:
            pass


def acquire_isaac_lock(script_name: str, args: Namespace | None = None) -> IsaacLockHandle:
    if legacy_isaac_lock_path().exists() and not _can_acquire_lock(legacy_isaac_lock_path()):
        metadata = _read_lock_metadata(legacy_isaac_lock_metadata_path())
        message = (
            f"Another Isaac workflow is already running. Refusing to start '{script_name}' to avoid "
            "the multi-Isaac crash path."
        )
        if isinstance(metadata, dict):
            message += (
                f" Active holder: script={metadata.get('script_name')} pid={metadata.get('pid')} "
                f"started_at={metadata.get('started_at')} cwd={metadata.get('cwd')}"
            )
        raise RuntimeError(message)
    lock_file = isaac_lock_path().open("a+", encoding="utf-8")
    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        metadata = _read_lock_metadata(isaac_lock_metadata_path())
        message = (
            f"Another Isaac workflow is already running. Refusing to start '{script_name}' to avoid "
            "the multi-Isaac crash path."
        )
        if isinstance(metadata, dict):
            message += (
                f" Active holder: script={metadata.get('script_name')} pid={metadata.get('pid')} "
                f"started_at={metadata.get('started_at')} cwd={metadata.get('cwd')}"
            )
        try:
            lock_file.close()
        except Exception:
            pass
        raise RuntimeError(message) from exc

    metadata = {
        "script_name": script_name,
        "pid": os.getpid(),
        "started_at": _timestamp(),
        "cwd": str(Path.cwd()),
        "argv": list(sys.argv),
        "cli_args": _json_ready(args) if args is not None else None,
    }
    isaac_lock_metadata_path().write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    handle = IsaacLockHandle(script_name=script_name, lock_file=lock_file, metadata_path=isaac_lock_metadata_path())
    atexit.register(handle.release)
    return handle
