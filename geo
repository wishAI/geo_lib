#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
USD_ROOT = REPO_ROOT / "algorithms" / "usd_parallel_urdf"
USD_OUTPUTS = USD_ROOT / "outputs"
USD_INPUTS = USD_ROOT / "inputs"
AVP_ROOT = REPO_ROOT / "algorithms" / "avp_remote"

PTENV_PYTHON = Path.home() / ".pyenv" / "versions" / "ptenv" / "bin" / "python"
ISAACLAB_SH = Path("/home/wishai/vscode/IsaacLab/isaaclab.sh")
ISAACSIM_PYTHON = Path("/home/wishai/vscode/IsaacLab/_isaac_sim/python.sh")
ISAACSIM_SH = Path("/home/wishai/vscode/IsaacLab/_isaac_sim/isaac-sim.sh")
DEFAULT_REMOTE_HOST = "tk2"
DEFAULT_REMOTE_ROOT = "/home/wishai/vscode/geo_lib"
DEFAULT_RSYNC_SSH = "ssh -o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=15 -o ServerAliveCountMax=2"


@dataclass(frozen=True)
class LaunchSpec:
    runner: str
    argv: list[str]
    env: dict[str, str] | None = None
    sidecars: tuple["LaunchSpec", ...] = ()
    success_artifact: Path | None = None
    failure_artifact: Path | None = None
    console_log: Path | None = None
    required_artifact_status: str | None = None


def _default_usd_path() -> Path:
    candidate = USD_INPUTS / "landau_v10.usdc"
    legacy = AVP_ROOT / "landau_v10.usdc"
    if candidate.exists():
        return candidate
    if legacy.exists():
        return legacy
    return candidate


def _default_avp_snapshot_path() -> Path:
    repo_candidate = REPO_ROOT / "avp_snapshot.json"
    if repo_candidate.exists():
        return repo_candidate
    return AVP_ROOT / "avp_snapshot.json"


def _extract_option_value(args: list[str], flag: str) -> str | None:
    for index, item in enumerate(args):
        if item == flag and index + 1 < len(args):
            return args[index + 1]
        if item.startswith(f"{flag}="):
            return item.split("=", 1)[1]
    return None


def _asset_tag(usd_path: Path) -> str:
    stem = usd_path.stem.strip() or "asset"
    return re.sub(r"[^A-Za-z0-9._-]+", "_", stem)


def _resolve_usd_asset_paths(extra_args: list[str]) -> dict[str, Path]:
    usd_path_value = _extract_option_value(extra_args, "--usd-path")
    output_dir_value = _extract_option_value(extra_args, "--output-dir")
    usd_path = Path(usd_path_value).expanduser() if usd_path_value else _default_usd_path()
    output_dir = Path(output_dir_value).expanduser() if output_dir_value else USD_OUTPUTS
    tag = _asset_tag(usd_path)
    primitive_name = f"{tag}_parallel"
    mesh_name = f"{primitive_name}_mesh"
    return {
        "usd_path": usd_path,
        "output_dir": output_dir,
        "primitive_urdf": output_dir / f"{primitive_name}.urdf",
        "mesh_urdf": output_dir / f"{mesh_name}.urdf",
        "primitive_validation_dir": output_dir / f"validation_{tag}",
        "mesh_validation_dir": output_dir / f"validation_mesh_{tag}",
    }


def _repo_arg(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise SystemExit(f"{label} not found: {path}")


def _require_executable(name: str) -> None:
    if shutil.which(name) is None:
        raise SystemExit(f"Required executable not found on PATH: {name}")


def _quote_remote_path(path: str) -> str:
    return shlex.quote(path)


def _resolve_algorithm_output_paths(project: str, remote_root: str) -> tuple[Path, str]:
    raw_project = project.strip()
    if Path(raw_project).is_absolute():
        raise SystemExit(f"Project must be relative to algorithms/: {project}")
    normalized = raw_project.strip("/")
    if normalized.startswith("algorithms/"):
        normalized = normalized.removeprefix("algorithms/")
    if normalized.endswith("/outputs"):
        normalized = normalized.removesuffix("/outputs")
    if not normalized or normalized in {".", ".."}:
        raise SystemExit("Project must be an algorithm name such as `usd_parallel_urdf`.")
    if normalized.startswith("../") or "/../" in normalized or normalized.endswith("/.."):
        raise SystemExit(f"Unsafe project path: {project}")
    local_output = REPO_ROOT / "algorithms" / normalized / "outputs"
    algorithm_root = local_output.parent
    if not algorithm_root.exists():
        raise SystemExit(f"Unknown local algorithm: algorithms/{normalized}")

    remote_output = f"{remote_root.rstrip('/')}/algorithms/{normalized}/outputs/"
    return local_output, remote_output


def _extract_global_flags(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    dry_run = False
    verbose = False
    filtered: list[str] = []
    for item in argv:
        if item == "--dry-run":
            dry_run = True
            continue
        if item == "--verbose":
            verbose = True
            continue
        filtered.append(item)
    return argparse.Namespace(dry_run=dry_run, verbose=verbose), filtered


def _add_gui_flags(parser: argparse.ArgumentParser, *, default_headless: bool) -> None:
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--headless", dest="headless", action="store_true", help="Launch headless.")
    group.add_argument("--gui", dest="headless", action="store_false", help="Launch with GUI.")
    parser.set_defaults(headless=default_headless)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="geo",
        description=(
            "Repo launcher for geo_lib presets and env passthroughs.\n"
            "Global flags: --dry-run prints the resolved command and exits; "
            "--verbose prints it before execution."
        ),
        epilog=(
            "Examples:\n"
            "  ./geo gui\n"
            "  ./geo storage status\n"
            "  ./geo --dry-run usd animate\n"
            "  ./geo usd animate\n"
            "  ./geo usd animate --camera-view hands --cycle-count 1\n"
            "  ./geo walk milestones\n"
            "  ./geo pull-output usd_parallel_urdf\n"
            "  ./geo avp session --gui --baseline\n"
            "  ./geo pt -m pytest algorithms/usd_parallel_urdf/tests -q\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="group", required=True)

    gui_parser = subparsers.add_parser("gui", help="Open the local-only Geo Web GUI.")
    gui_parser.add_argument("--host", default="127.0.0.1")
    gui_parser.add_argument("--port", type=int, default=8767)
    gui_parser.add_argument("--no-browser", action="store_true")

    storage_parser = subparsers.add_parser("storage", help="Manage Nextcloud-backed large files and TK2 sync.")
    storage_subparsers = storage_parser.add_subparsers(dest="storage_cmd", required=True)
    storage_subparsers.add_parser("status", help="Show cloud asset and hydration status.")
    storage_subparsers.add_parser("audit", help="Fail when a tracked file exceeds 5 MiB.")
    storage_hydrate = storage_subparsers.add_parser("hydrate", help="Restore declared repo-path links from Nextcloud.")
    storage_hydrate.add_argument("--copy", action="store_true", help="Copy files instead of making symlinks.")
    storage_subparsers.add_parser("push-tk2", help="Push Mac shared files to TK2 without deleting remote files.")
    storage_subparsers.add_parser("pull-tk2", help="Pull TK2 shared files to the Mac without deleting local files.")
    storage_subparsers.add_parser("sync-code-tk2", help="Refresh the GUI-owned TK2 source workspace.")

    subparsers.add_parser("pt", help="Run ptenv Python directly.")
    subparsers.add_parser("isaac", help="Pass through to Isaac Lab's `isaaclab.sh -p`.")
    subparsers.add_parser("simpy", help="Pass through to Isaac Sim's `python.sh`.")
    subparsers.add_parser("sim", help="Pass through to Isaac Sim's `isaac-sim.sh`.")

    pull_output = subparsers.add_parser(
        "pull-output",
        help="Pull an algorithm's ignored outputs/ folder from TK2 via rsync.",
    )
    pull_output.add_argument("project", help="Algorithm name, for example `usd_parallel_urdf`.")
    pull_output.add_argument(
        "--remote",
        default=DEFAULT_REMOTE_HOST,
        help=f"SSH host to pull from. Defaults to {DEFAULT_REMOTE_HOST}.",
    )
    pull_output.add_argument(
        "--remote-root",
        default=DEFAULT_REMOTE_ROOT,
        help=f"Remote geo_lib root. Defaults to {DEFAULT_REMOTE_ROOT}.",
    )
    pull_output.add_argument(
        "--no-delete",
        dest="delete",
        action="store_false",
        help="Keep local output files that no longer exist on the remote.",
    )
    pull_output.set_defaults(delete=True)

    usd_parser = subparsers.add_parser("usd", help="USD Parallel URDF presets.")
    usd_subparsers = usd_parser.add_subparsers(dest="usd_cmd", required=True)

    usd_inspect = usd_subparsers.add_parser("inspect", help="Inspect the source USD skeleton.")
    _add_gui_flags(usd_inspect, default_headless=True)

    usd_subparsers.add_parser("build", help="Build primitive + mesh URDF outputs.")
    usd_subparsers.add_parser("build-mesh", help="Build only the mesh-backed URDF outputs.")

    usd_validate = usd_subparsers.add_parser("validate", help="Validate the generated URDF in Isaac.")
    _add_gui_flags(usd_validate, default_headless=True)
    usd_validate.add_argument("--mesh", action="store_true", help="Use the default mesh-backed URDF/output paths.")

    usd_animate = usd_subparsers.add_parser("animate", help="Play synchronized USD + URDF animation.")
    _add_gui_flags(usd_animate, default_headless=False)

    usd_render = usd_subparsers.add_parser("render", help="Render a posed overview image.")
    _add_gui_flags(usd_render, default_headless=True)
    usd_render.add_argument("--mesh", action="store_true", help="Use the default mesh-backed URDF path.")

    usd_subparsers.add_parser("compare", help="Run offline FK comparison in ptenv.")
    usd_subparsers.add_parser("test", help="Run usd_parallel_urdf unit tests in ptenv.")

    walk_parser = subparsers.add_parser("walk", help="Build and validate Landau locomotion one milestone at a time.")
    walk_subparsers = walk_parser.add_subparsers(dest="walk_cmd", required=True)
    walk_subparsers.add_parser("milestones", help="Print the clean machine-readable milestone ladder.")
    walk_subparsers.add_parser("evolution", help="Rebuild the real checkpoint and experiment evolution tree.")
    walk_subparsers.add_parser("inspect", help="Audit the retained URDF and print the robot control contract.")
    walk_subparsers.add_parser(
        "validate-passive", help="Run camera-free dynamics, viewport proof, and final assembly sequentially."
    )
    walk_subparsers.add_parser(
        "validate-passive-dynamics", help="Run only the camera-free passive dynamics component in Isaac Lab."
    )
    walk_subparsers.add_parser(
        "render-passive-proof", help="Run only the separate viewport proof component in Isaac Lab."
    )
    walk_subparsers.add_parser(
        "finalize-passive", help="Assemble final passive evidence from two already-passed components."
    )
    walk_subparsers.add_parser(
        "train-policy-stand", help="Train the milestone-2 manager-based RSL-RL PPO standing policy."
    )
    walk_subparsers.add_parser(
        "validate-policy-stand", help="Run policy dynamics, viewport proof, and final assembly sequentially."
    )
    walk_subparsers.add_parser(
        "validate-policy-stand-dynamics", help="Run camera-free checkpoint inference for policy standing."
    )
    walk_subparsers.add_parser(
        "render-policy-stand-proof", help="Render the passed policy checkpoint through the active viewport."
    )
    walk_subparsers.add_parser(
        "finalize-policy-stand", help="Assemble policy stand evidence from already-passed components."
    )
    walk_subparsers.add_parser(
        "train-forward-walk", help="Fine-tune the passed stand checkpoint for the flat 5 m +Y gate."
    )
    walk_subparsers.add_parser(
        "validate-forward-walk", help="Sequentially re-pass stand, validate 5 m, render proof, and finalize."
    )
    walk_subparsers.add_parser(
        "validate-forward-walk-stand", help="Re-pass the 30 s policy stand with the walking checkpoint."
    )
    walk_subparsers.add_parser(
        "validate-forward-walk-dynamics", help="Run camera-free 5 m forward checkpoint inference."
    )
    walk_subparsers.add_parser(
        "render-forward-walk-proof", help="Render the passed 5 m forward checkpoint behavior."
    )
    walk_subparsers.add_parser(
        "finalize-forward-walk", help="Assemble existing cumulative 5 m component evidence."
    )
    walk_subparsers.add_parser(
        "probe-forward-reference",
        help="Run one camera-free open-loop v3 phase-reference experiment.",
    )
    walk_subparsers.add_parser("test", help="Run pure-Python walk contract tests.")

    avp_parser = subparsers.add_parser("avp", help="AVP presets.")
    avp_subparsers = avp_parser.add_subparsers(dest="avp_cmd", required=True)

    avp_bridge = avp_subparsers.add_parser("bridge", help="Run the AVP bridge in ptenv.")
    avp_bridge.add_argument("--avp-ip", type=str, default=None, help="Override AVP_IP for the bridge.")
    avp_bridge.add_argument("--bridge-host", type=str, default=None, help="Override BRIDGE_HOST.")
    avp_bridge.add_argument("--bridge-port", type=int, default=None, help="Override BRIDGE_PORT.")
    avp_bridge.add_argument("--send-hz", type=int, default=None, help="Override SEND_HZ.")
    avp_bridge.add_argument("--snapshot-path", type=str, default=None, help="Snapshot file path for bridge capture.")
    avp_bridge.add_argument(
        "--transport",
        choices=("udp", "zmq"),
        default="udp",
        help="Local bridge transport. Defaults to UDP so the bridge and Isaac runtimes stay compatible even when only one env has pyzmq installed.",
    )

    avp_session = avp_subparsers.add_parser("session", help="Run the AVP Landau session.")
    _add_gui_flags(avp_session, default_headless=False)
    tracking_group = avp_session.add_mutually_exclusive_group()
    tracking_group.add_argument(
        "--snapshot",
        dest="tracking_source",
        action="store_const",
        const="snapshot",
        help="Use a saved snapshot payload.",
    )
    tracking_group.add_argument(
        "--bridge",
        dest="tracking_source",
        action="store_const",
        const="bridge",
        help="Use live bridge tracking.",
    )
    avp_session.set_defaults(tracking_source="snapshot")
    avp_session.add_argument(
        "--with-bridge",
        action="store_true",
        help="When using --bridge, auto-start the local AVP bridge sidecar too.",
    )
    avp_session.add_argument("--avp-ip", type=str, default=None, help="Override AVP_IP for the bridge sidecar.")
    avp_session.add_argument("--bridge-host", type=str, default=None, help="Override BRIDGE_HOST.")
    avp_session.add_argument("--bridge-port", type=int, default=None, help="Override BRIDGE_PORT.")
    avp_session.add_argument("--send-hz", type=int, default=None, help="Override SEND_HZ for the bridge sidecar.")
    avp_session.add_argument("--snapshot-path", type=str, default=None, help="Snapshot file path.")
    avp_session.add_argument(
        "--transport",
        choices=("udp", "zmq"),
        default="udp",
        help="Local bridge transport. Defaults to UDP so the bridge and Isaac runtimes stay compatible even when only one env has pyzmq installed.",
    )

    avp_marker = avp_subparsers.add_parser("marker", help="Run the AVP wrist marker viewer.")
    _add_gui_flags(avp_marker, default_headless=False)
    marker_tracking_group = avp_marker.add_mutually_exclusive_group()
    marker_tracking_group.add_argument(
        "--snapshot",
        dest="tracking_source",
        action="store_const",
        const="snapshot",
        help="Use a saved snapshot payload.",
    )
    marker_tracking_group.add_argument(
        "--bridge",
        dest="tracking_source",
        action="store_const",
        const="bridge",
        help="Use live bridge tracking.",
    )
    avp_marker.set_defaults(tracking_source="snapshot")
    avp_marker.add_argument("--snapshot-path", type=str, default=None, help="Snapshot file path.")
    avp_marker.add_argument("--bridge-host", type=str, default=None, help="Override BRIDGE_HOST.")
    avp_marker.add_argument("--bridge-port", type=int, default=None, help="Override BRIDGE_PORT.")
    avp_marker.add_argument(
        "--transport",
        choices=("udp", "zmq"),
        default="udp",
        help="Local bridge transport. Defaults to UDP so the marker viewer matches the bridge sidecar transport.",
    )

    avp_subparsers.add_parser("test", help="Run AVP unit tests in ptenv.")

    return parser


def _resolved_command(spec: LaunchSpec) -> tuple[list[str], dict[str, str]]:
    if spec.runner == "pt":
        _require_file(PTENV_PYTHON, "ptenv python")
        cmd = [str(PTENV_PYTHON), *spec.argv]
        env = os.environ.copy()
        env.setdefault("MUJOCO_GL", "egl")
    elif spec.runner == "isaac":
        _require_file(ISAACLAB_SH, "Isaac Lab launcher")
        cmd = [str(ISAACLAB_SH), "-p", *spec.argv]
        env = os.environ.copy()
    elif spec.runner == "simpy":
        _require_file(ISAACSIM_PYTHON, "Isaac Sim python")
        cmd = [str(ISAACSIM_PYTHON), *spec.argv]
        env = os.environ.copy()
    elif spec.runner == "sim":
        _require_file(ISAACSIM_SH, "Isaac Sim shell")
        cmd = [str(ISAACSIM_SH), *spec.argv]
        env = os.environ.copy()
    elif spec.runner == "direct":
        cmd = list(spec.argv)
        env = os.environ.copy()
    else:
        raise SystemExit(f"Unsupported runner: {spec.runner}")

    if spec.env:
        env.update(spec.env)
    return cmd, env


def _display_command(spec: LaunchSpec, cmd: list[str]) -> str:
    env_prefix = ""
    if spec.env:
        env_prefix = " ".join(f"{key}={shlex.quote(value)}" for key, value in sorted(spec.env.items())) + " "
    return env_prefix + shlex.join(cmd)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _run_logged(cmd: list[str], env: dict[str, str], console_log: Path | None) -> int:
    if console_log is None:
        return int(subprocess.run(cmd, cwd=REPO_ROOT, env=env, check=False).returncode)
    console_log.parent.mkdir(parents=True, exist_ok=True)
    with console_log.open("w", encoding="utf-8") as stream:
        process = subprocess.Popen(
            cmd,
            cwd=REPO_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        try:
            for line in process.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                stream.write(line)
                stream.flush()
            return int(process.wait())
        except KeyboardInterrupt:
            process.terminate()
            try:
                process.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=3.0)
            raise
        finally:
            process.stdout.close()


def _write_launcher_failure(
    spec: LaunchSpec, cmd: list[str], returncode: int, reason: str
) -> None:
    if spec.failure_artifact is None:
        return
    spec.failure_artifact.parent.mkdir(parents=True, exist_ok=True)
    launcher = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "reason": reason,
        "returncode": returncode,
        "command": cmd,
        "expected_success_artifact": (
            str(spec.success_artifact) if spec.success_artifact is not None else None
        ),
        "console_log": str(spec.console_log) if spec.console_log is not None else None,
        "console_log_sha256": (
            _sha256(spec.console_log)
            if spec.console_log is not None and spec.console_log.is_file() else None
        ),
    }
    if spec.failure_artifact.exists():
        try:
            evidence = json.loads(spec.failure_artifact.read_text(encoding="utf-8"))
            evidence["launcher"] = launcher
        except (OSError, ValueError):
            evidence = {"schema_version": 1, "status": "launcher_child_failed", **launcher}
    else:
        evidence = {"schema_version": 1, "status": "launcher_child_failed", **launcher}
    spec.failure_artifact.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _run_with_runner(spec: LaunchSpec, *, dry_run: bool, verbose: bool) -> int:
    cmd, env = _resolved_command(spec)
    sidecars: list[tuple[list[str], dict[str, str]]] = [_resolved_command(sidecar) for sidecar in spec.sidecars]

    if dry_run or verbose:
        if sidecars:
            for index, (sidecar_spec, sidecar_data) in enumerate(zip(spec.sidecars, sidecars), start=1):
                sidecar_cmd, _ = sidecar_data
                print(f"# sidecar {index}: {_display_command(sidecar_spec, sidecar_cmd)}", flush=True)
            print(f"# main: {_display_command(spec, cmd)}", flush=True)
        else:
            print(_display_command(spec, cmd), flush=True)
    if dry_run:
        return 0

    if spec.success_artifact is not None and spec.success_artifact.exists():
        spec.success_artifact.unlink()
    if spec.failure_artifact is not None and spec.failure_artifact.exists():
        spec.failure_artifact.unlink()
    if spec.console_log is not None and spec.console_log.exists():
        spec.console_log.unlink()

    sidecar_processes: list[subprocess.Popen[str]] = []
    try:
        for sidecar_cmd, sidecar_env in sidecars:
            sidecar_processes.append(
                subprocess.Popen(
                    sidecar_cmd,
                    cwd=REPO_ROOT,
                    env=sidecar_env,
                    stdin=subprocess.DEVNULL,
                )
            )
        if sidecar_processes:
            time.sleep(1.0)
        returncode = _run_logged(cmd, env, spec.console_log)
        if returncode != 0:
            # A validator may intentionally return non-zero after writing a
            # complete failed-gate artifact. Reserve launcher failure evidence
            # for exits that produced no component artifact.
            if spec.success_artifact is None or not spec.success_artifact.is_file():
                _write_launcher_failure(spec, cmd, returncode, "child process returned non-zero")
            return returncode
        if returncode == 0 and spec.success_artifact is not None and not spec.success_artifact.is_file():
            print(
                f"Expected success artifact was not produced: {spec.success_artifact}",
                file=sys.stderr,
                flush=True,
            )
            _write_launcher_failure(spec, cmd, returncode, "success artifact was not produced")
            return 1
        if returncode == 0 and spec.required_artifact_status is not None:
            try:
                artifact = json.loads(spec.success_artifact.read_text(encoding="utf-8"))
                status = artifact.get("status")
            except (AttributeError, OSError, ValueError) as error:
                print(f"Unable to audit component artifact status: {error}", file=sys.stderr)
                _write_launcher_failure(spec, cmd, returncode, "component artifact status unreadable")
                return 1
            if status != spec.required_artifact_status:
                print(
                    f"Component artifact status is {status!r}, expected "
                    f"{spec.required_artifact_status!r}: {spec.success_artifact}",
                    file=sys.stderr,
                    flush=True,
                )
                return 1
        return returncode
    except KeyboardInterrupt:
        return 130
    finally:
        for process in sidecar_processes:
            if process.poll() is None:
                process.terminate()
        for process in sidecar_processes:
            try:
                process.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=3.0)


def _env_override_map(
    *,
    avp_ip: str | None = None,
    bridge_host: str | None = None,
    bridge_port: int | None = None,
    send_hz: int | None = None,
    snapshot_path: str | None = None,
    use_zmq: bool | None = None,
) -> dict[str, str]:
    env: dict[str, str] = {}
    if avp_ip is not None:
        env["AVP_IP"] = avp_ip
    if bridge_host is not None:
        env["BRIDGE_HOST"] = bridge_host
    if bridge_port is not None:
        env["BRIDGE_PORT"] = str(bridge_port)
    if send_hz is not None:
        env["SEND_HZ"] = str(send_hz)
    if snapshot_path is not None:
        env["AVP_SNAPSHOT_PATH"] = snapshot_path
    if use_zmq is not None:
        env["USE_ZMQ"] = "1" if use_zmq else "0"
    return env


def _build_spec(args: argparse.Namespace, extra_args: list[str]) -> LaunchSpec:
    if args.group == "gui":
        if extra_args:
            raise SystemExit(f"Unexpected GUI arguments: {shlex.join(extra_args)}")
        argv = [sys.executable, "-m", "webgui.server", "--host", args.host, "--port", str(args.port)]
        if args.no_browser:
            argv.append("--no-browser")
        return LaunchSpec("direct", argv)

    if args.group == "storage":
        if extra_args:
            raise SystemExit(f"Unexpected storage arguments: {shlex.join(extra_args)}")
        argv = [sys.executable, "-m", "webgui.storage", args.storage_cmd]
        if args.storage_cmd == "hydrate" and args.copy:
            argv.append("--copy")
        return LaunchSpec("direct", argv)

    if args.group == "pull-output":
        if extra_args:
            raise SystemExit(f"Unexpected pull-output arguments: {shlex.join(extra_args)}")
        _require_executable("rsync")
        local_output, remote_output = _resolve_algorithm_output_paths(args.project, args.remote_root)
        rsync_args = ["rsync", "-azP", "--timeout=60", "-e", DEFAULT_RSYNC_SSH]
        if args.delete:
            rsync_args.append("--delete")
        rsync_args.extend([f"{args.remote}:{_quote_remote_path(remote_output)}", f"{local_output}/"])
        return LaunchSpec("direct", rsync_args)

    if args.group == "pt":
        if not extra_args:
            raise SystemExit("`geo pt` expects Python arguments, for example: ./geo pt -m pytest ...")
        return LaunchSpec("pt", extra_args)

    if args.group == "isaac":
        if not extra_args:
            raise SystemExit("`geo isaac` expects script or module arguments, for example: ./geo isaac -m ...")
        return LaunchSpec("isaac", extra_args)

    if args.group == "simpy":
        if not extra_args:
            raise SystemExit("`geo simpy` expects script arguments, for example: ./geo simpy algorithms/avp_remote/run_avp_landau_session.py")
        return LaunchSpec("simpy", extra_args)

    if args.group == "sim":
        if not extra_args:
            raise SystemExit("`geo sim` expects Isaac Sim shell arguments, for example: ./geo sim --exec ...")
        return LaunchSpec("sim", extra_args)

    if args.group == "usd":
        asset_paths = _resolve_usd_asset_paths(extra_args)

        if args.usd_cmd == "inspect":
            argv = ["algorithms/usd_parallel_urdf/inspect_usd_skeleton.py"]
            if args.headless:
                argv.append("--headless")
            argv.extend(extra_args)
            return LaunchSpec("isaac", argv)

        if args.usd_cmd == "build":
            return LaunchSpec("isaac", ["algorithms/usd_parallel_urdf/build_parallel_urdf.py", *extra_args])

        if args.usd_cmd == "build-mesh":
            return LaunchSpec(
                "isaac",
                [
                    "algorithms/usd_parallel_urdf/build_parallel_urdf.py",
                    "--geometry-mode",
                    "mesh",
                    *extra_args,
                ],
            )

        if args.usd_cmd == "validate":
            argv = ["algorithms/usd_parallel_urdf/validate_parallel_scene.py"]
            if args.headless:
                argv.append("--headless")
            if args.mesh:
                argv.extend(
                    [
                        "--urdf-path",
                        _repo_arg(asset_paths["mesh_urdf"]),
                        "--output-dir",
                        _repo_arg(asset_paths["mesh_validation_dir"]),
                    ]
                )
            argv.extend(extra_args)
            return LaunchSpec("isaac", argv)

        if args.usd_cmd == "animate":
            argv = [
                "algorithms/usd_parallel_urdf/play_parallel_animation.py",
                "--urdf-path",
                _repo_arg(asset_paths["mesh_urdf"]),
                "--animation-clip",
                "walk_cycle",
                "--camera-view",
                "walk_side",
            ]
            if args.headless:
                argv.append("--headless")
            argv.extend(extra_args)
            return LaunchSpec("isaac", argv)

        if args.usd_cmd == "render":
            default_output = (
                asset_paths["mesh_validation_dir"] / "scene_pose.png"
                if args.mesh
                else asset_paths["primitive_validation_dir"] / "scene_pose.png"
            )
            argv = [
                "algorithms/usd_parallel_urdf/render_parallel_scene.py",
                "--posed",
                "--view",
                "overview",
                "--output-path",
                _repo_arg(default_output),
            ]
            if args.mesh:
                argv.extend(["--urdf-path", _repo_arg(asset_paths["mesh_urdf"])])
            if args.headless:
                argv.append("--headless")
            argv.extend(extra_args)
            return LaunchSpec("isaac", argv)

        if args.usd_cmd == "compare":
            return LaunchSpec("pt", ["algorithms/usd_parallel_urdf/compare_urdf_pose_offline.py", *extra_args])

        if args.usd_cmd == "test":
            return LaunchSpec(
                "pt",
                ["-m", "unittest", "discover", "-s", "algorithms/usd_parallel_urdf/tests", *extra_args],
            )

    if args.group == "walk":
        if args.walk_cmd == "milestones":
            if extra_args:
                raise SystemExit(f"Unexpected milestone arguments: {shlex.join(extra_args)}")
            return LaunchSpec(
                "direct",
                [sys.executable, "-m", "json.tool", "algorithms/urdf_learn_wasd_walk/milestones.json"],
            )
        if args.walk_cmd == "evolution":
            return LaunchSpec(
                "direct",
                [sys.executable, "algorithms/urdf_learn_wasd_walk/evolution.py", *extra_args],
                success_artifact=REPO_ROOT / "algorithms" / "urdf_learn_wasd_walk" / "outputs" / "evolution.json",
            )
        if args.walk_cmd == "inspect":
            return LaunchSpec(
                "direct",
                [sys.executable, "algorithms/urdf_learn_wasd_walk/model_spec.py", *extra_args],
            )
        if args.walk_cmd in {
            "validate-passive", "validate-passive-dynamics", "render-passive-proof", "finalize-passive"
        }:
            output_value = _extract_option_value(extra_args, "--output-dir")
            output_dir = Path(output_value).expanduser() if output_value else REPO_ROOT / "algorithms" / "urdf_learn_wasd_walk" / "outputs" / "stand_zero_signal_30s_no_reset"
            if not output_dir.is_absolute():
                output_dir = REPO_ROOT / output_dir
            smoke = "--smoke" in extra_args
            if args.walk_cmd == "validate-passive":
                evidence_name = "smoke_validation.json" if smoke else "validation.json"
                return LaunchSpec(
                    "direct",
                    [sys.executable, "algorithms/urdf_learn_wasd_walk/passive_pipeline.py", *extra_args],
                    env={"TERM": "xterm"},
                    success_artifact=output_dir.resolve() / evidence_name,
                )
            if args.walk_cmd == "validate-passive-dynamics":
                stem = "dynamics_smoke" if smoke else "dynamics"
                evidence_name = f"{stem}_validation.json"
                return LaunchSpec(
                    "isaac",
                    ["algorithms/urdf_learn_wasd_walk/passive_stand.py", "--phase", "dynamics", *extra_args],
                    env={"TERM": "xterm"},
                    success_artifact=output_dir.resolve() / evidence_name,
                    failure_artifact=output_dir.resolve() / f"{stem}_failure.json",
                    console_log=output_dir.resolve() / f"{stem}_console.log",
                )
            if args.walk_cmd == "render-passive-proof":
                evidence_name = "proof_smoke_validation.json" if smoke else "proof_validation.json"
                return LaunchSpec(
                    "isaac",
                    ["algorithms/urdf_learn_wasd_walk/passive_stand.py", "--phase", "proof", *extra_args],
                    env={"TERM": "xterm"},
                    success_artifact=output_dir.resolve() / evidence_name,
                )
            evidence_name = "smoke_validation.json" if smoke else "validation.json"
            return LaunchSpec(
                "direct",
                [
                    sys.executable,
                    "algorithms/urdf_learn_wasd_walk/passive_pipeline.py",
                    "--finalize-only",
                    *extra_args,
                ],
                success_artifact=output_dir.resolve() / evidence_name,
            )
        if args.walk_cmd in {
            "train-policy-stand", "validate-policy-stand", "validate-policy-stand-dynamics",
            "render-policy-stand-proof", "finalize-policy-stand",
        }:
            output_value = _extract_option_value(extra_args, "--output-dir")
            output_dir = (
                Path(output_value).expanduser()
                if output_value
                else REPO_ROOT / "algorithms" / "urdf_learn_wasd_walk" / "outputs" / "stand_30s_no_reset"
            )
            if not output_dir.is_absolute():
                output_dir = REPO_ROOT / output_dir
            smoke = "--smoke" in extra_args
            if args.walk_cmd == "train-policy-stand":
                return LaunchSpec(
                    "isaac",
                    ["algorithms/urdf_learn_wasd_walk/policy_stand.py", "--mode", "train", *extra_args],
                    env={"TERM": "xterm"},
                    success_artifact=output_dir.resolve() / "training.json",
                    failure_artifact=output_dir.resolve() / "train_failure.json",
                    console_log=output_dir.resolve() / "train_console.log",
                )
            if args.walk_cmd == "validate-policy-stand":
                evidence_name = "smoke_validation.json" if smoke else "validation.json"
                return LaunchSpec(
                    "direct",
                    [sys.executable, "algorithms/urdf_learn_wasd_walk/policy_stand_pipeline.py", *extra_args],
                    env={"TERM": "xterm"},
                    success_artifact=output_dir.resolve() / evidence_name,
                )
            if args.walk_cmd == "validate-policy-stand-dynamics":
                stem = "dynamics_smoke" if smoke else "dynamics"
                return LaunchSpec(
                    "isaac",
                    ["algorithms/urdf_learn_wasd_walk/policy_stand.py", "--mode", "dynamics", *extra_args],
                    env={"TERM": "xterm"},
                    success_artifact=output_dir.resolve() / f"{stem}_validation.json",
                    failure_artifact=output_dir.resolve() / f"{stem}_failure.json",
                    console_log=output_dir.resolve() / f"{stem}_console.log",
                )
            if args.walk_cmd == "render-policy-stand-proof":
                evidence_name = "proof_smoke_validation.json" if smoke else "proof_validation.json"
                return LaunchSpec(
                    "isaac",
                    ["algorithms/urdf_learn_wasd_walk/policy_stand.py", "--mode", "proof", *extra_args],
                    env={"TERM": "xterm"},
                    success_artifact=output_dir.resolve() / evidence_name,
                )
            evidence_name = "smoke_validation.json" if smoke else "validation.json"
            return LaunchSpec(
                "direct",
                [
                    sys.executable, "algorithms/urdf_learn_wasd_walk/policy_stand_pipeline.py",
                    "--finalize-only", *extra_args,
                ],
                success_artifact=output_dir.resolve() / evidence_name,
            )
        if args.walk_cmd == "probe-forward-reference":
            output_value = _extract_option_value(extra_args, "--output-dir")
            output_dir = (
                Path(output_value).expanduser()
                if output_value
                else REPO_ROOT / "algorithms" / "urdf_learn_wasd_walk" / "outputs"
                / "gate_5m_no_reset" / "reference_probe_v3" / "baseline"
            )
            if not output_dir.is_absolute():
                output_dir = REPO_ROOT / output_dir
            return LaunchSpec(
                "isaac",
                ["algorithms/urdf_learn_wasd_walk/forward_reference_probe.py", *extra_args],
                env={"TERM": "xterm"},
                success_artifact=output_dir.resolve() / "reference_probe.json",
                failure_artifact=output_dir.resolve() / "reference_probe_failure.json",
                console_log=output_dir.resolve() / "reference_probe_console.log",
                required_artifact_status="passed",
            )
        if args.walk_cmd in {
            "train-forward-walk", "validate-forward-walk", "validate-forward-walk-stand",
            "validate-forward-walk-dynamics", "render-forward-walk-proof", "finalize-forward-walk",
        }:
            output_value = _extract_option_value(extra_args, "--output-dir")
            output_dir = (
                Path(output_value).expanduser()
                if output_value
                else REPO_ROOT / "algorithms" / "urdf_learn_wasd_walk" / "outputs" / "gate_5m_no_reset"
            )
            if not output_dir.is_absolute():
                output_dir = REPO_ROOT / output_dir
            smoke = "--smoke" in extra_args
            if args.walk_cmd == "train-forward-walk":
                return LaunchSpec(
                    "isaac",
                    ["algorithms/urdf_learn_wasd_walk/forward_walk.py", "--mode", "train", *extra_args],
                    env={"TERM": "xterm"},
                    success_artifact=output_dir.resolve() / "training.json",
                    failure_artifact=output_dir.resolve() / "train_failure.json",
                    console_log=output_dir.resolve() / "train_console.log",
                )
            if args.walk_cmd == "validate-forward-walk":
                evidence_name = "smoke_validation.json" if smoke else "validation.json"
                return LaunchSpec(
                    "direct",
                    [sys.executable, "algorithms/urdf_learn_wasd_walk/forward_walk_pipeline.py", *extra_args],
                    env={"TERM": "xterm"},
                    success_artifact=output_dir.resolve() / evidence_name,
                )
            modes = {
                "validate-forward-walk-stand": "stand",
                "validate-forward-walk-dynamics": "forward",
                "render-forward-walk-proof": "proof",
            }
            if args.walk_cmd in modes:
                mode = modes[args.walk_cmd]
                evidence_name = {
                    "stand": "stand_dynamics",
                    "forward": "forward_dynamics",
                    "proof": "proof",
                }[mode] + ("_smoke_validation.json" if smoke else "_validation.json")
                stem = evidence_name.removesuffix("_validation.json")
                return LaunchSpec(
                    "isaac",
                    ["algorithms/urdf_learn_wasd_walk/forward_walk.py", "--mode", mode, *extra_args],
                    env={"TERM": "xterm"},
                    success_artifact=output_dir.resolve() / evidence_name,
                    failure_artifact=output_dir.resolve() / f"{mode}{'_smoke' if smoke else ''}_failure.json",
                    console_log=output_dir.resolve() / f"{stem}_console.log",
                )
            evidence_name = "smoke_validation.json" if smoke else "validation.json"
            return LaunchSpec(
                "direct",
                [
                    sys.executable, "algorithms/urdf_learn_wasd_walk/forward_walk_pipeline.py",
                    "--finalize-only", *extra_args,
                ],
                success_artifact=output_dir.resolve() / evidence_name,
            )
        if args.walk_cmd == "test":
            return LaunchSpec(
                "direct",
                [
                    sys.executable,
                    "-m",
                    "unittest",
                    "discover",
                    "-s",
                    "algorithms/urdf_learn_wasd_walk/tests",
                    *extra_args,
                ],
            )

    if args.group == "avp":
        snapshot_path = getattr(args, "snapshot_path", None) or _repo_arg(_default_avp_snapshot_path())
        bridge_env = _env_override_map(
            avp_ip=getattr(args, "avp_ip", None),
            bridge_host=getattr(args, "bridge_host", None),
            bridge_port=getattr(args, "bridge_port", None),
            send_hz=getattr(args, "send_hz", None),
            snapshot_path=snapshot_path,
            use_zmq=(getattr(args, "transport", "udp") == "zmq"),
        )

        if args.avp_cmd == "bridge":
            argv = ["algorithms/avp_remote/avp_bridge.py"]
            if snapshot_path is not None:
                argv.extend(["--snapshot-path", snapshot_path])
            argv.extend(extra_args)
            return LaunchSpec("pt", argv, env=bridge_env)

        if args.avp_cmd == "session":
            effective_tracking_source = args.tracking_source
            effective_with_bridge = args.with_bridge
            if args.avp_ip:
                effective_tracking_source = "bridge"
                effective_with_bridge = True

            if effective_with_bridge and effective_tracking_source != "bridge":
                raise SystemExit("`--with-bridge` requires `--bridge`.")
            argv = ["algorithms/avp_remote/run_avp_landau_session.py"]
            if args.headless:
                argv.append("--headless")
            else:
                argv.extend(["--experience", "base"])
            argv.extend(["--tracking-source", effective_tracking_source, "--snapshot-path", snapshot_path])
            argv.extend(extra_args)
            sidecars: tuple[LaunchSpec, ...] = ()
            if effective_with_bridge:
                sidecars = (
                    LaunchSpec(
                        "pt",
                        ["algorithms/avp_remote/avp_bridge.py", "--snapshot-path", snapshot_path],
                        env=bridge_env,
                    ),
                )
            return LaunchSpec("simpy", argv, env=bridge_env, sidecars=sidecars)

        if args.avp_cmd == "marker":
            exec_argv = [
                "algorithms/avp_remote/avp_wrist_marker.py",
                "--tracking-source",
                args.tracking_source,
                "--snapshot-path",
                snapshot_path,
                *extra_args,
            ]
            sim_argv: list[str] = []
            if args.headless:
                sim_argv.append("--headless")
            sim_argv.extend(["--exec", shlex.join(exec_argv)])
            return LaunchSpec("sim", sim_argv, env=bridge_env)

        if args.avp_cmd == "test":
            return LaunchSpec(
                "pt",
                [
                    "-m",
                    "unittest",
                    "discover",
                    "-s",
                    "algorithms/avp_remote/tests",
                    "-p",
                    "test_*.py",
                    *extra_args,
                ],
            )

    raise SystemExit("Unsupported command.")


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    global_flags, filtered_argv = _extract_global_flags(argv)
    parser = _build_parser()
    args, extra_args = parser.parse_known_args(filtered_argv)
    spec = _build_spec(args, extra_args)
    return _run_with_runner(spec, dry_run=global_flags.dry_run, verbose=global_flags.verbose)


if __name__ == "__main__":
    raise SystemExit(main())
