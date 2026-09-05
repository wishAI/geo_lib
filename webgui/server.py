from __future__ import annotations

import argparse
import io
import json
import mimetypes
import os
import re
import shlex
import signal
import subprocess
import threading
import time
import uuid
import webbrowser
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from . import storage


REPO_ROOT = Path(__file__).resolve().parents[1]
STATIC_ROOT = Path(__file__).resolve().parent / "static"
HOST = "127.0.0.1"
DEFAULT_PORT = 8767
REMOTE_HOST = "tk2"
REMOTE_ROOT = storage.DEFAULT_REMOTE_WORKSPACE
SSH_ARGS = [
    "-o", "BatchMode=yes", "-o", "ConnectTimeout=8",
    "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=2",
]
MAX_LOG_CHARS = 180_000
PARAMETER_NAME = re.compile(r"^[a-z][a-z0-9_]*$")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def discover_manifests() -> list[dict]:
    manifests = []
    for path in sorted((REPO_ROOT / "algorithms").glob("*/gui/manifest.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        sandbox = path.parents[1].name
        if payload.get("id") != sandbox:
            raise ValueError(f"Manifest id {payload.get('id')!r} does not match {sandbox!r}")
        payload["manifestPath"] = str(path.relative_to(REPO_ROOT))
        manifests.append(payload)
    return manifests


def manifest_map() -> dict[str, dict]:
    return {item["id"]: item for item in discover_manifests()}


def _safe_under(root: Path, relative: str) -> Path:
    if not relative or Path(relative).is_absolute():
        raise ValueError("Path must be a non-empty repository-relative path")
    root = root.resolve(strict=False)
    candidate = (root / relative).resolve(strict=False)
    candidate.relative_to(root)
    return candidate


def artifact_candidates(relative: str) -> list[tuple[str, Path]]:
    return [
        ("repo", _safe_under(REPO_ROOT, relative)),
        ("nextcloud", _safe_under(storage.cloud_root() / "remote_outputs", relative)),
        ("nextcloud", _safe_under(storage.cloud_root(), relative)),
    ]


def resolve_artifact(relative: str) -> tuple[str, Path] | None:
    available = [(source, path) for source, path in artifact_candidates(relative) if path.is_file()]
    if not available:
        return None
    return max(available, key=lambda item: item[1].stat().st_mtime)


def declared_artifact_paths() -> set[str]:
    paths: set[str] = set()
    for manifest in discover_manifests():
        for example in manifest.get("examples", []):
            for artifact in example.get("artifacts", []):
                paths.add(artifact["path"])
        for candidate in manifest.get("viewer", {}).get("urdfCandidates", []):
            paths.add(candidate["path"])
        inspector = manifest.get("inspector", {})
        if inspector.get("path"):
            paths.add(inspector["path"])
    return paths


def _parameter_values(example: dict, supplied: dict) -> dict[str, str]:
    definitions = {item["id"]: item for item in example.get("parameters", [])}
    unknown = set(supplied) - set(definitions)
    if unknown:
        raise ValueError(f"Unknown parameters: {', '.join(sorted(unknown))}")
    values: dict[str, str] = {}
    for name, definition in definitions.items():
        if not PARAMETER_NAME.fullmatch(name):
            raise ValueError(f"Invalid parameter id: {name}")
        value = supplied.get(name, definition.get("default"))
        if value is None:
            raise ValueError(f"Missing parameter: {name}")
        kind = definition.get("type", "text")
        if kind in {"number", "integer"}:
            number = float(value)
            if kind == "integer" and not number.is_integer():
                raise ValueError(f"{name} must be an integer")
            if "min" in definition and number < float(definition["min"]):
                raise ValueError(f"{name} is below the minimum")
            if "max" in definition and number > float(definition["max"]):
                raise ValueError(f"{name} is above the maximum")
            value = str(int(number)) if kind == "integer" else str(number)
        elif kind == "select":
            choices = [str(choice) for choice in definition.get("choices", [])]
            if str(value) not in choices:
                raise ValueError(f"{name} must be one of {', '.join(choices)}")
        else:
            value = str(value)
            pattern = definition.get("pattern")
            if pattern and not re.fullmatch(pattern, value):
                raise ValueError(f"{name} has an invalid value")
        values[name] = str(value)
    return values


def build_example_command(manifest: dict, example: dict, supplied: dict, target: str | None = None) -> list[str]:
    values = _parameter_values(example, supplied)
    template = example.get("commands", {}).get(target, example.get("command", []))
    command = []
    for token in template:
        expanded = str(token)
        for name, value in values.items():
            expanded = expanded.replace("{" + name + "}", value)
        if re.search(r"\{[a-z][a-z0-9_]*\}", expanded):
            raise ValueError(f"Unresolved command parameter in {expanded!r}")
        command.append(expanded)
    if not command:
        raise ValueError("Example has no command")
    return command


@dataclass
class Job:
    id: str
    sandbox: str
    example: str
    target: str
    command: list[str]
    resource: str | None
    artifacts: list[dict]
    status: str = "queued"
    createdAt: str = field(default_factory=utc_now)
    startedAt: str | None = None
    finishedAt: str | None = None
    returncode: int | None = None
    log: str = ""
    sync: list[dict] = field(default_factory=list)
    process: subprocess.Popen[str] | None = field(default=None, repr=False)

    def public(self) -> dict:
        return {
            "id": self.id,
            "sandbox": self.sandbox,
            "example": self.example,
            "target": self.target,
            "command": list(self.command),
            "resource": self.resource,
            "artifacts": list(self.artifacts),
            "status": self.status,
            "createdAt": self.createdAt,
            "startedAt": self.startedAt,
            "finishedAt": self.finishedAt,
            "returncode": self.returncode,
            "log": self.log,
            "sync": list(self.sync),
        }


class JobManager:
    def __init__(self) -> None:
        self.jobs: dict[str, Job] = {}
        self.lock = threading.RLock()

    def list(self) -> list[dict]:
        with self.lock:
            return [job.public() for job in sorted(self.jobs.values(), key=lambda item: item.createdAt, reverse=True)[:30]]

    def get(self, job_id: str) -> Job:
        with self.lock:
            if job_id not in self.jobs:
                raise KeyError(job_id)
            return self.jobs[job_id]

    def start_example(self, sandbox: str, example_id: str, target: str | None, parameters: dict) -> dict:
        manifests = manifest_map()
        manifest = manifests.get(sandbox)
        if not manifest:
            raise ValueError("Unknown sandbox")
        example = next((item for item in manifest.get("examples", []) if item.get("id") == example_id), None)
        if not example:
            raise ValueError("Unknown example")
        allowed = example.get("targets", [example.get("target", "local")])
        chosen_target = target or example.get("target") or allowed[0]
        if chosen_target not in allowed or chosen_target not in {"local", "tk2"}:
            raise ValueError("Execution target is not allowed for this example")
        command = build_example_command(manifest, example, parameters, chosen_target)
        resource = example.get("resource")
        with self.lock:
            if resource and any(job.status in {"queued", "running"} and job.resource == resource for job in self.jobs.values()):
                raise ValueError(f"A {resource} job is already running")
            job = Job(
                id=uuid.uuid4().hex[:12], sandbox=sandbox, example=example_id,
                target=chosen_target, command=command, resource=resource,
                artifacts=list(example.get("artifacts", [])),
            )
            self.jobs[job.id] = job
        threading.Thread(target=self._run, args=(job,), daemon=True).start()
        return job.public()

    def start_storage(self, action: str) -> dict:
        commands = {
            "hydrate": [os.fspath(Path(os.sys.executable)), "-m", "webgui.storage", "hydrate"],
            "audit": [os.fspath(Path(os.sys.executable)), "-m", "webgui.storage", "audit"],
            "push-tk2": [os.fspath(Path(os.sys.executable)), "-m", "webgui.storage", "push-tk2"],
            "pull-tk2": [os.fspath(Path(os.sys.executable)), "-m", "webgui.storage", "pull-tk2"],
            "sync-code-tk2": [os.fspath(Path(os.sys.executable)), "-m", "webgui.storage", "sync-code-tk2"],
        }
        if action not in commands:
            raise ValueError("Unknown storage action")
        with self.lock:
            job = Job(
                id=uuid.uuid4().hex[:12], sandbox="storage", example=action,
                target="local", command=commands[action], resource="storage", artifacts=[],
            )
            if any(item.status in {"queued", "running"} and item.resource == "storage" for item in self.jobs.values()):
                raise ValueError("A storage operation is already running")
            self.jobs[job.id] = job
        threading.Thread(target=self._run, args=(job,), daemon=True).start()
        return job.public()

    def cancel(self, job_id: str) -> dict:
        job = self.get(job_id)
        with self.lock:
            process = job.process
            if job.status not in {"queued", "running"}:
                return job.public()
            job.status = "cancelling"
        if process and process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                process.terminate()
        return job.public()

    def _append(self, job: Job, text: str) -> None:
        with self.lock:
            job.log = (job.log + text)[-MAX_LOG_CHARS:]

    def _run(self, job: Job) -> None:
        with self.lock:
            job.status = "running"
            job.startedAt = utc_now()
        if job.target == "tk2":
            self._append(job, f"$ sync Mac source → {REMOTE_HOST}:{REMOTE_ROOT}\n")
            try:
                synced = storage.sync_source_tk2(remote=REMOTE_HOST)
            except Exception as error:  # noqa: BLE001
                self._append(job, f"Source sync failed: {error}\n")
                with self.lock:
                    job.status = "failed"
                    job.returncode = -1
                    job.finishedAt = utc_now()
                return
            self._append(job, synced.get("output", ""))
            if not synced.get("ok"):
                self._append(job, "\nSource sync failed; remote command was not started.\n")
                with self.lock:
                    job.status = "failed"
                    job.returncode = int(synced.get("returncode", -1))
                    job.finishedAt = utc_now()
                return
            remote = f"cd {shlex.quote(REMOTE_ROOT)} && {shlex.join(job.command)}"
            command = ["ssh", *SSH_ARGS, REMOTE_HOST, remote]
        else:
            command = job.command
        self._append(job, f"$ {shlex.join(command)}\n\n")
        try:
            process = subprocess.Popen(
                command, cwd=REPO_ROOT, text=True, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, bufsize=1, start_new_session=True,
            )
            with self.lock:
                job.process = process
            assert process.stdout is not None
            for line in process.stdout:
                self._append(job, line)
            returncode = process.wait()
            with self.lock:
                job.returncode = returncode
                final_status = "succeeded" if returncode == 0 else ("cancelled" if job.status == "cancelling" else "failed")
            if returncode == 0 and job.target == "tk2" and job.artifacts:
                paths = [
                    item["path"].rstrip("/") + "/" if item.get("kind") == "directory" else item["path"]
                    for item in job.artifacts
                ]
                try:
                    job.sync = storage.pull_remote_artifacts(job.sandbox, paths)
                    for result in job.sync:
                        if result.get("ok"):
                            self._append(job, f"\nSynced artifact to Nextcloud: {result['path']}\n")
                        else:
                            self._append(job, f"\nArtifact not synced: {result['path']}\n{result.get('output', '')}\n")
                except Exception as error:  # noqa: BLE001
                    self._append(job, f"\nArtifact sync failed: {error}\n")
            with self.lock:
                job.status = final_status
        except Exception as error:  # noqa: BLE001
            self._append(job, f"\nLaunch failed: {error}\n")
            with self.lock:
                job.status = "failed"
                job.returncode = -1
        finally:
            with self.lock:
                job.finishedAt = utc_now()
                job.process = None


JOBS = JobManager()


def remote_status() -> dict:
    command = ["ssh", *SSH_ARGS, REMOTE_HOST, "printf ready"]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=10, check=False)
        return {"online": completed.returncode == 0 and completed.stdout == "ready", "detail": completed.stderr.strip()[:240]}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"online": False, "detail": str(error)}


def artifact_inventory(sandbox: str) -> list[dict]:
    manifest = manifest_map().get(sandbox)
    if not manifest:
        raise ValueError("Unknown sandbox")
    declared: dict[str, dict] = {}
    for example in manifest.get("examples", []):
        for artifact in example.get("artifacts", []):
            if artifact.get("syncOnly"):
                continue
            declared[artifact["path"]] = artifact
    inventory = []
    for path, artifact in declared.items():
        resolved = resolve_artifact(path)
        inventory.append({
            **artifact,
            "exists": bool(resolved),
            "source": resolved[0] if resolved else None,
            "size": resolved[1].stat().st_size if resolved else None,
            "modifiedAt": datetime.fromtimestamp(resolved[1].stat().st_mtime, timezone.utc).isoformat() if resolved else None,
            "url": f"/api/artifact?path={path}" if resolved else None,
        })
    return inventory


def robot_candidates() -> list[dict]:
    candidates = []
    allowed = declared_artifact_paths()
    for manifest in discover_manifests():
        for item in manifest.get("viewer", {}).get("urdfCandidates", []):
            path = item["path"]
            if path not in allowed:
                continue
            resolved = resolve_artifact(path)
            candidates.append({**item, "sandbox": manifest["id"], "exists": bool(resolved), "source": resolved[0] if resolved else None})
    return candidates


JOINT_GROUP_LABELS = {
    "left_arm": "Left arm",
    "right_arm": "Right arm",
    "left_leg": "Left leg",
    "right_leg": "Right leg",
    "body": "Body",
}


def _side_in_name(name: str, side: str) -> bool:
    tokens = re.split(r"[^a-z0-9]+", name.lower())
    short = "l" if side == "left" else "r"
    return side in tokens or short in tokens or name.lower().startswith(f"{side}_") or name.lower().endswith(f"_{short}")


def _joint_group(name: str) -> str:
    lowered = name.lower()
    leg_words = ("hip", "thigh", "knee", "shin", "ankle", "foot", "toe", "leg")
    for side in ("left", "right"):
        if not _side_in_name(lowered, side):
            continue
        return f"{side}_leg" if any(word in lowered for word in leg_words) else f"{side}_arm"
    return "body"


def _robot_joint_info(urdf_text: str) -> list[dict]:
    root = ET.fromstring(urdf_text)
    joints = []
    for node in root.findall("joint"):
        kind = node.attrib.get("type", "fixed")
        if kind == "fixed":
            continue
        limit = node.find("limit")
        lower = -3.141592653589793 if kind == "continuous" else float(limit.attrib.get("lower", -3.141592653589793)) if limit is not None else -3.141592653589793
        upper = 3.141592653589793 if kind == "continuous" else float(limit.attrib.get("upper", 3.141592653589793)) if limit is not None else 3.141592653589793
        name = node.attrib.get("name", "")
        group = _joint_group(name)
        joints.append({
            "name": name,
            "type": kind,
            "lower": lower,
            "upper": upper,
            "value": 0.0,
            "group": group,
            "groupLabel": JOINT_GROUP_LABELS[group],
        })
    return joints


def _mesh_workbench(sandbox: str) -> tuple[dict, dict]:
    manifest = manifest_map().get(sandbox)
    if not manifest or not manifest.get("meshWorkbench"):
        raise ValueError("Mesh workbench is not declared for this sandbox")
    return manifest, manifest["meshWorkbench"]


def _mesh_apply_example(manifest: dict, config: dict) -> dict:
    example_id = config.get("applyExample")
    example = next((item for item in manifest.get("examples", []) if item.get("id") == example_id), None)
    if not example:
        raise ValueError("Mesh workbench apply example is missing")
    return example


def _mesh_part_names(sandbox: str) -> list[str]:
    _, config = _mesh_workbench(sandbox)
    summary = resolve_artifact(config["summaryPath"])
    names: list[str] = []
    if summary:
        payload = json.loads(summary[1].read_text(encoding="utf-8"))
        links = payload.get("links", {})
        if isinstance(links, dict):
            names = [str(name) for name in links]
    if not names:
        stl_dir = config["stlDir"]
        for _, root in artifact_candidates(stl_dir):
            if root.is_dir():
                names.extend(path.stem for path in root.glob("*.stl"))
                break
    return sorted(set(names), key=lambda name: (_joint_group(name), name))


def _mesh_asset_relative(sandbox: str, part: str, variant: str, *, validate: bool = True) -> tuple[str, bool]:
    _, config = _mesh_workbench(sandbox)
    if validate and part not in _mesh_part_names(sandbox):
        raise ValueError("Unknown mesh body part")
    if variant not in {"source", "stl"}:
        raise ValueError("Mesh variant must be source or stl")
    directory = config["sourceMeshDir"] if variant == "source" else config["stlDir"]
    relative = f"{directory.rstrip('/')}/{part}.stl"
    return relative, False


def _mesh_catalog(sandbox: str) -> dict:
    manifest, config = _mesh_workbench(sandbox)
    example = _mesh_apply_example(manifest, config)
    parts = []
    for name in _mesh_part_names(sandbox):
        source_relative, source_fallback = _mesh_asset_relative(sandbox, name, "source", validate=False)
        stl_relative, _ = _mesh_asset_relative(sandbox, name, "stl", validate=False)
        source_resolved = resolve_artifact(source_relative)
        stl_resolved = resolve_artifact(stl_relative)
        group = _joint_group(name)
        parts.append({
            "name": name,
            "group": group,
            "groupLabel": JOINT_GROUP_LABELS[group],
            "sourceExists": bool(source_resolved) and not source_fallback,
            "stlExists": bool(stl_resolved),
            "sourceVersion": str(source_resolved[1].stat().st_mtime_ns) if source_resolved else "",
            "stlVersion": str(stl_resolved[1].stat().st_mtime_ns) if stl_resolved else "",
        })
    current = {}
    summary = resolve_artifact(config["summaryPath"])
    if summary:
        payload = json.loads(summary[1].read_text(encoding="utf-8"))
        lowpoly = payload.get("config", {}).get("lowpoly_default", {})
        current = {
            "method": payload.get("mesh_simplify_mode"),
            "target_face_ratio": lowpoly.get("target_face_ratio"),
            "max_faces": lowpoly.get("max_faces"),
            "max_hull_faces": payload.get("max_hull_faces"),
            "target_hull_points": payload.get("target_hull_points"),
            "min_thickness": payload.get("config", {}).get("min_thickness"),
        }
    return {
        "sandbox": sandbox,
        "parts": parts,
        "applyExample": example["id"],
        "target": example.get("target", "tk2"),
        "targets": example.get("targets", [example.get("target", "tk2")]),
        "parameters": example.get("parameters", []),
        "current": {key: value for key, value in current.items() if value is not None},
    }


def _mesh_part_urdf(sandbox: str, part: str, variant: str) -> dict:
    relative, fallback = _mesh_asset_relative(sandbox, part, variant)
    if not resolve_artifact(relative):
        raise ValueError("Selected mesh asset is not available")
    robot = ET.Element("robot", {"name": f"{part}_{variant}_preview"})
    link = ET.SubElement(robot, "link", {"name": part})
    visual = ET.SubElement(link, "visual")
    geometry = ET.SubElement(visual, "geometry")
    ET.SubElement(geometry, "mesh", {"filename": f"{part}.stl"})
    return {
        "urdf": ET.tostring(robot, encoding="unicode"),
        "path": relative,
        "joints": [],
        "part": part,
        "variant": variant,
        "sourceFallback": fallback,
    }


def _allowed_robot_path(relative: str) -> bool:
    return any(item.get("path") == relative for item in robot_candidates())


def _resolve_robot_asset(urdf_relative: str, uri: str) -> Path | None:
    resolved = resolve_artifact(urdf_relative)
    if not resolved:
        return None
    _, urdf_path = resolved
    cleaned = uri.replace("\\", "/")
    variants = [cleaned]
    if cleaned.startswith("package://"):
        variants.append(cleaned.removeprefix("package://"))
        variants.append("/".join(cleaned.removeprefix("package://").split("/")[1:]))
    if cleaned.startswith("file://"):
        variants.append(cleaned.removeprefix("file://"))
    roots = [urdf_path.parent, REPO_ROOT, storage.cloud_root(), storage.cloud_root() / "remote_outputs"]
    for root in roots:
        for variant in variants:
            candidate = (root / variant).resolve(strict=False) if not Path(variant).is_absolute() else Path(variant).resolve(strict=False)
            allowed_roots = [REPO_ROOT.resolve(), storage.cloud_root().resolve(strict=False)]
            if not any(_is_relative_to(candidate, allowed) for allowed in allowed_roots):
                continue
            if candidate.is_file():
                return candidate
    return None


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


class GeoGuiServer(ThreadingHTTPServer):
    allow_reuse_address = True


class Handler(BaseHTTPRequestHandler):
    server_version = "GeoLab/1.0"

    def log_message(self, format: str, *args) -> None:
        print(f"[geo-gui] {format % args}")

    def _json(self, payload: object, status: HTTPStatus = HTTPStatus.OK) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _error(self, message: str, status: HTTPStatus = HTTPStatus.BAD_REQUEST) -> None:
        self._json({"error": message}, status)

    def _body(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        if length > 1_000_000:
            raise ValueError("Request body is too large")
        return json.loads(self.rfile.read(length) or b"{}")

    def _file(self, path: Path, *, cache: str = "no-cache") -> None:
        if not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        content = path.read_bytes()
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        if path.suffix.lower() == ".pgm":
            try:
                from PIL import Image

                converted = io.BytesIO()
                with Image.open(io.BytesIO(content)) as image:
                    image.save(converted, format="PNG")
                content = converted.getvalue()
                content_type = "image/png"
            except (ImportError, OSError):
                pass
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", cache)
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(content)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        try:
            if parsed.path == "/api/health":
                self._json({"status": "ok", "service": "geo-web-gui", "sandboxes": len(discover_manifests())})
                return
            if parsed.path == "/api/catalog":
                self._json({"sandboxes": discover_manifests(), "checkedAt": utc_now()})
                return
            if parsed.path == "/api/status":
                cloud = storage.status()
                self._json({"mac": {"online": True}, "tk2": remote_status(), "nextcloud": cloud, "checkedAt": utc_now()})
                return
            if parsed.path == "/api/jobs":
                self._json({"jobs": JOBS.list()})
                return
            if parsed.path.startswith("/api/jobs/"):
                self._json(JOBS.get(parsed.path.rsplit("/", 1)[-1]).public())
                return
            if parsed.path.startswith("/api/artifacts/"):
                self._json({"artifacts": artifact_inventory(parsed.path.rsplit("/", 1)[-1])})
                return
            if parsed.path == "/api/artifact":
                relative = query.get("path", [""])[0]
                if relative not in declared_artifact_paths():
                    self._error("Artifact is not declared", HTTPStatus.FORBIDDEN)
                    return
                resolved = resolve_artifact(relative)
                if not resolved:
                    self._error("Artifact is not available", HTTPStatus.NOT_FOUND)
                    return
                self._file(resolved[1])
                return
            if parsed.path == "/api/storage":
                self._json({"storage": storage.status(), "audit": storage.audit_tracked_files()})
                return
            if parsed.path == "/api/robot/catalog":
                self._json({"robots": robot_candidates()})
                return
            if parsed.path == "/api/robot/urdf":
                relative = query.get("path", [""])[0]
                if not _allowed_robot_path(relative):
                    self._error("URDF is not declared", HTTPStatus.FORBIDDEN)
                    return
                resolved = resolve_artifact(relative)
                if not resolved:
                    self._error("URDF is not available", HTTPStatus.NOT_FOUND)
                    return
                text = resolved[1].read_text(encoding="utf-8")
                self._json({"urdf": text, "path": relative, "joints": _robot_joint_info(text)})
                return
            if parsed.path == "/api/robot/asset":
                urdf = query.get("urdf", [""])[0]
                uri = query.get("uri", [""])[0]
                if not _allowed_robot_path(urdf):
                    self._error("URDF is not declared", HTTPStatus.FORBIDDEN)
                    return
                asset = _resolve_robot_asset(urdf, uri)
                if not asset:
                    self._error("Robot asset is not available", HTTPStatus.NOT_FOUND)
                    return
                self._file(asset, cache="public, max-age=300")
                return
            if parsed.path == "/api/mesh/catalog":
                sandbox = query.get("sandbox", [""])[0]
                self._json(_mesh_catalog(sandbox))
                return
            if parsed.path == "/api/mesh/urdf":
                sandbox = query.get("sandbox", [""])[0]
                part = query.get("part", [""])[0]
                variant = query.get("variant", [""])[0]
                self._json(_mesh_part_urdf(sandbox, part, variant))
                return
            if parsed.path == "/api/mesh/asset":
                sandbox = query.get("sandbox", [""])[0]
                part = query.get("part", [""])[0]
                variant = query.get("variant", [""])[0]
                relative, _ = _mesh_asset_relative(sandbox, part, variant)
                resolved = resolve_artifact(relative)
                if not resolved:
                    self._error("Mesh asset is not available", HTTPStatus.NOT_FOUND)
                    return
                self._file(resolved[1], cache="public, max-age=60")
                return
            if parsed.path in {"/", "/index.html"}:
                self._file(STATIC_ROOT / "index.html")
                return
            relative = parsed.path.lstrip("/")
            static = _safe_under(STATIC_ROOT, relative)
            self._file(static, cache="public, max-age=60")
        except KeyError:
            self._error("Not found", HTTPStatus.NOT_FOUND)
        except (ValueError, OSError, ET.ParseError) as error:
            self._error(str(error))

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        try:
            body = self._body()
            if parsed.path == "/api/jobs":
                job = JOBS.start_example(
                    str(body.get("sandbox", "")), str(body.get("example", "")),
                    body.get("target"), body.get("parameters", {}),
                )
                self._json(job, HTTPStatus.ACCEPTED)
                return
            if parsed.path.endswith("/cancel") and parsed.path.startswith("/api/jobs/"):
                job_id = parsed.path.split("/")[-2]
                self._json(JOBS.cancel(job_id))
                return
            if parsed.path.startswith("/api/storage/"):
                action = parsed.path.rsplit("/", 1)[-1]
                self._json(JOBS.start_storage(action), HTTPStatus.ACCEPTED)
                return
            self._error("Not found", HTTPStatus.NOT_FOUND)
        except KeyError:
            self._error("Not found", HTTPStatus.NOT_FOUND)
        except (ValueError, json.JSONDecodeError, OSError) as error:
            self._error(str(error))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the local-only Geo Library web GUI")
    parser.add_argument("--host", default=HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args(argv)
    if args.host not in {"127.0.0.1", "localhost"}:
        raise SystemExit("Geo Web GUI must bind to 127.0.0.1/localhost")
    server = GeoGuiServer((HOST, args.port), Handler)
    url = f"http://{HOST}:{args.port}/"
    print(f"Geo Web GUI: {url}", flush=True)
    if not args.no_browser:
        threading.Timer(0.35, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever(poll_interval=0.3)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
