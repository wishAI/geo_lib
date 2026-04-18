#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
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


@dataclass(frozen=True)
class LaunchSpec:
    runner: str
    argv: list[str]
    env: dict[str, str] | None = None
    sidecars: tuple["LaunchSpec", ...] = ()


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
            "  ./geo --dry-run usd animate\n"
            "  ./geo usd animate\n"
            "  ./geo usd animate --camera-view hands --cycle-count 1\n"
            "  ./geo walk train --max_iterations 2\n"
            "  ./geo walk play --gui --visual-mode usd\n"
            "  ./geo avp session --gui --baseline\n"
            "  ./geo pt -m pytest algorithms/usd_parallel_urdf/tests -q\n"
            "  ./geo isaac -m algorithms.urdf_learn_wasd_walk.scripts.train --robot landau --stage fwd_only --headless\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="group", required=True)

    subparsers.add_parser("pt", help="Run ptenv Python directly.")
    subparsers.add_parser("isaac", help="Pass through to Isaac Lab's `isaaclab.sh -p`.")
    subparsers.add_parser("simpy", help="Pass through to Isaac Sim's `python.sh`.")
    subparsers.add_parser("sim", help="Pass through to Isaac Sim's `isaac-sim.sh`.")

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

    walk_parser = subparsers.add_parser("walk", help="URDF locomotion presets.")
    walk_subparsers = walk_parser.add_subparsers(dest="walk_cmd", required=True)

    walk_smoke = walk_subparsers.add_parser("smoke", help="Run the headless walk smoke test.")
    _add_gui_flags(walk_smoke, default_headless=True)

    walk_train = walk_subparsers.add_parser("train", help="Train the default Landau Stage A task.")
    _add_gui_flags(walk_train, default_headless=True)

    walk_validate = walk_subparsers.add_parser("validate", help="Validate a walk checkpoint.")
    _add_gui_flags(walk_validate, default_headless=True)

    walk_play = walk_subparsers.add_parser("play", help="Play back a walk checkpoint.")
    _add_gui_flags(walk_play, default_headless=True)

    walk_subparsers.add_parser("teleop", help="Teleoperate the default Landau Stage A checkpoint.")
    walk_subparsers.add_parser("test", help="Run pure-Python walk tests in ptenv.")

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
        completed = subprocess.run(cmd, cwd=REPO_ROOT, env=env, check=False)
        return int(completed.returncode)
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
        base_defaults = ["--robot", "landau", "--stage", "fwd_only"]

        if args.walk_cmd == "smoke":
            argv = [
                "-m",
                "algorithms.urdf_learn_wasd_walk.scripts.smoke_test",
                *base_defaults,
                "--steps",
                "32",
            ]
            if args.headless:
                argv.append("--headless")
            argv.extend(extra_args)
            return LaunchSpec("isaac", argv)

        if args.walk_cmd == "train":
            argv = ["-m", "algorithms.urdf_learn_wasd_walk.scripts.train", *base_defaults]
            if args.headless:
                argv.append("--headless")
            argv.extend(extra_args)
            return LaunchSpec("isaac", argv)

        if args.walk_cmd == "validate":
            argv = ["-m", "algorithms.urdf_learn_wasd_walk.scripts.validate_walk", *base_defaults]
            if args.headless:
                argv.append("--headless")
            argv.extend(extra_args)
            return LaunchSpec("isaac", argv)

        if args.walk_cmd == "play":
            argv = ["-m", "algorithms.urdf_learn_wasd_walk.scripts.play", *base_defaults]
            if args.headless:
                argv.append("--headless")
            argv.extend(extra_args)
            return LaunchSpec("isaac", argv)

        if args.walk_cmd == "teleop":
            argv = ["-m", "algorithms.urdf_learn_wasd_walk.scripts.teleop", *base_defaults, *extra_args]
            return LaunchSpec("isaac", argv)

        if args.walk_cmd == "test":
            return LaunchSpec("pt", ["-m", "pytest", "algorithms/urdf_learn_wasd_walk/tests", "-q", *extra_args])

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
