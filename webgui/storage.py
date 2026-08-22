from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "large_files.json"
DEFAULT_CLOUD_ROOT = Path.home() / "Nextcloud" / "Projects" / "geo_lib"
DEFAULT_REMOTE_CLOUD_ROOT = "/home/wishai/Nextcloud/Projects/geo_lib"
DEFAULT_REMOTE_WORKSPACE = "/home/wishai/.cache/geo-lib-webgui/current"
LARGE_FILE_BYTES = 5 * 1024 * 1024
SSH_ARGS = (
    "-o", "BatchMode=yes",
    "-o", "ConnectTimeout=8",
    "-o", "ServerAliveInterval=15",
    "-o", "ServerAliveCountMax=2",
)


def cloud_root() -> Path:
    return Path(os.environ.get("GEO_CLOUD_ROOT", DEFAULT_CLOUD_ROOT)).expanduser()


def load_manifest(path: Path = MANIFEST_PATH) -> dict:
    if not path.exists():
        return {"version": 1, "thresholdBytes": LARGE_FILE_BYTES, "files": []}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload.get("files"), list):
        raise ValueError("large_files.json must contain a files list")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_repo_path(relative: str) -> Path:
    relative_path = Path(relative)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError(f"Unsafe repository path: {relative}")
    candidate = REPO_ROOT / relative_path
    candidate.parent.resolve(strict=False).relative_to(REPO_ROOT.resolve())
    return candidate


def _safe_cloud_path(relative: str) -> Path:
    root = cloud_root().expanduser()
    candidate = root / relative
    candidate.resolve(strict=False).relative_to(root.resolve(strict=False))
    return candidate


def managed_entries() -> list[dict]:
    return list(load_manifest().get("files", []))


def status() -> dict:
    root = cloud_root()
    entries = []
    for item in managed_entries():
        repo_path = _safe_repo_path(item["repoPath"])
        source = _safe_cloud_path(item["cloudPath"])
        source_exists = source.is_file()
        hydrated = repo_path.is_symlink() and repo_path.resolve(strict=False) == source.resolve(strict=False)
        stable_link = hydrated and Path(os.readlink(repo_path)) == source
        valid = source_exists and (
            not item.get("sha256") or _sha256(source) == item["sha256"]
        )
        entries.append({
            "repoPath": item["repoPath"],
            "cloudPath": item["cloudPath"],
            "sourceExists": source_exists,
            "hydrated": hydrated,
            "stableLink": stable_link,
            "valid": valid,
            "size": source.stat().st_size if source_exists else None,
        })
    return {
        "root": str(root),
        "available": root.is_dir(),
        "thresholdBytes": int(load_manifest().get("thresholdBytes", LARGE_FILE_BYTES)),
        "entries": entries,
        "summary": {
            "total": len(entries),
            "available": sum(1 for entry in entries if entry["sourceExists"]),
            "hydrated": sum(1 for entry in entries if entry["hydrated"]),
            "valid": sum(1 for entry in entries if entry["valid"]),
        },
    }


def hydrate(*, copy_files: bool = False) -> dict:
    linked: list[str] = []
    missing: list[str] = []
    invalid: list[str] = []
    for item in managed_entries():
        repo_path = _safe_repo_path(item["repoPath"])
        source = _safe_cloud_path(item["cloudPath"])
        if not source.is_file():
            missing.append(item["cloudPath"])
            continue
        if item.get("sha256") and _sha256(source) != item["sha256"]:
            invalid.append(item["cloudPath"])
            continue
        repo_path.parent.mkdir(parents=True, exist_ok=True)
        if repo_path.exists() or repo_path.is_symlink():
            if repo_path.is_symlink() and Path(os.readlink(repo_path)) == source:
                linked.append(item["repoPath"])
                continue
            if repo_path.is_dir():
                raise RuntimeError(f"Refusing to replace directory: {repo_path}")
            repo_path.unlink()
        if copy_files:
            shutil.copy2(source, repo_path)
        else:
            repo_path.symlink_to(source)
        linked.append(item["repoPath"])
    return {"hydrated": linked, "missing": missing, "invalid": invalid, "copyMode": copy_files}


def audit_tracked_files() -> dict:
    completed = subprocess.run(
        ["git", "ls-files", "-z"], cwd=REPO_ROOT, check=True, capture_output=True
    )
    oversized = []
    for raw in completed.stdout.split(b"\0"):
        if not raw:
            continue
        relative = raw.decode("utf-8", errors="surrogateescape")
        path = REPO_ROOT / relative
        try:
            size = path.stat().st_size
        except OSError:
            continue
        if size > LARGE_FILE_BYTES:
            oversized.append({"path": relative, "size": size})
    return {"thresholdBytes": LARGE_FILE_BYTES, "oversizedTrackedFiles": oversized, "ok": not oversized}


def _run_rsync(source: str, destination: str) -> dict:
    command = [
        "rsync", "-azP", "--timeout=90",
        "-e", "ssh " + " ".join(SSH_ARGS),
        source, destination,
    ]
    completed = subprocess.run(
        command, cwd=REPO_ROOT, text=True, capture_output=True, timeout=180, check=False
    )
    return {
        "ok": completed.returncode == 0,
        "returncode": completed.returncode,
        "command": command,
        "output": (completed.stdout + completed.stderr)[-12000:],
    }


def sync_tk2(direction: str, *, remote: str = "tk2") -> dict:
    local = cloud_root()
    local.mkdir(parents=True, exist_ok=True)
    if direction == "push":
        subprocess.run(
            ["ssh", *SSH_ARGS, remote, "mkdir", "-p", DEFAULT_REMOTE_CLOUD_ROOT],
            check=True, timeout=30,
        )
        return _run_rsync(f"{local}/", f"{remote}:{DEFAULT_REMOTE_CLOUD_ROOT}/")
    if direction == "pull":
        return _run_rsync(f"{remote}:{DEFAULT_REMOTE_CLOUD_ROOT}/", f"{local}/")
    raise ValueError("direction must be push or pull")


def sync_source_tk2(*, remote: str = "tk2") -> dict:
    """Refresh the GUI-owned TK2 workspace without touching a developer checkout."""
    subprocess.run(
        ["ssh", *SSH_ARGS, remote, "mkdir", "-p", DEFAULT_REMOTE_WORKSPACE],
        check=True, timeout=30,
    )
    command = [
        "rsync", "-az", "--delete", "--timeout=90",
        "--exclude=.git/", "--exclude=.DS_Store", "--exclude=__pycache__/",
        "--exclude=*.pyc", "--exclude=logs/", "--exclude=.kit_portable/",
        "--exclude=algorithms/*/outputs/",
        "-e", "ssh " + " ".join(SSH_ARGS),
        f"{REPO_ROOT}/", f"{remote}:{DEFAULT_REMOTE_WORKSPACE}/",
    ]
    synced = subprocess.run(
        command, cwd=REPO_ROOT, text=True, capture_output=True, timeout=180, check=False,
    )
    output = synced.stdout + synced.stderr
    if synced.returncode == 0:
        hydrate_command = f"cd {DEFAULT_REMOTE_WORKSPACE} && python3 -m webgui.storage hydrate"
        hydrated = subprocess.run(
            ["ssh", *SSH_ARGS, remote, hydrate_command],
            text=True, capture_output=True, timeout=180, check=False,
        )
        output += hydrated.stdout + hydrated.stderr
        returncode = hydrated.returncode
    else:
        returncode = synced.returncode
    return {
        "ok": returncode == 0,
        "returncode": returncode,
        "command": command,
        "workspace": DEFAULT_REMOTE_WORKSPACE,
        "output": output[-12000:],
    }


def pull_remote_artifacts(sandbox: str, relative_paths: Iterable[str], *, remote: str = "tk2") -> list[dict]:
    results: list[dict] = []
    destination_root = cloud_root() / "remote_outputs"
    for relative in relative_paths:
        source_path = f"{DEFAULT_REMOTE_WORKSPACE.rstrip('/')}/{relative.lstrip('/')}"
        destination = destination_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        result = _run_rsync(f"{remote}:{source_path}", str(destination))
        result["path"] = relative
        results.append(result)
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Geo Library Nextcloud-backed large-file manager")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("status")
    subparsers.add_parser("audit")
    hydrate_parser = subparsers.add_parser("hydrate")
    hydrate_parser.add_argument("--copy", action="store_true", help="Copy instead of creating local symlinks")
    subparsers.add_parser("push-tk2")
    subparsers.add_parser("pull-tk2")
    subparsers.add_parser("sync-code-tk2")
    args = parser.parse_args(argv)
    if args.command == "status":
        payload = status()
    elif args.command == "audit":
        payload = audit_tracked_files()
    elif args.command == "hydrate":
        payload = hydrate(copy_files=args.copy)
    elif args.command == "push-tk2":
        payload = sync_tk2("push")
    elif args.command == "sync-code-tk2":
        payload = sync_source_tk2()
    else:
        payload = sync_tk2("pull")
    print(json.dumps(payload, indent=2))
    return 0 if payload.get("ok", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
