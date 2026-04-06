from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


MODULE_ROOT = Path(__file__).resolve().parent
MODULE_ROOT_STR = str(MODULE_ROOT)
if MODULE_ROOT_STR in sys.path:
    sys.path.remove(MODULE_ROOT_STR)
sys.path.insert(0, MODULE_ROOT_STR)

ISAAC_ROOT = Path("/home/wishai/vscode/IsaacLab/_isaac_sim")
EXPERIENCE_BY_NAME = {
    "avp_headless": MODULE_ROOT / "isaacsim.avp_headless.kit",
    "base": ISAAC_ROOT / "apps" / "isaacsim.exp.base.kit",
    "full": ISAAC_ROOT / "apps" / "isaacsim.exp.full.kit",
}


def _log(message: str) -> None:
    print(f"[AVP-LAUNCH] {message}", flush=True)


def _parse_launcher_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        add_help=False,
        description="Bootstrap the AVP Landau session inside Isaac Sim's Python runtime.",
    )
    parser.add_argument(
        "--portable-root",
        type=Path,
        default=MODULE_ROOT / ".kit_portable" / "avp_landau_session",
        help="Portable Kit data root used for headless/debug runs.",
    )
    parser.add_argument(
        "--experience",
        choices=tuple(EXPERIENCE_BY_NAME),
        default="avp_headless",
        help="Isaac Sim experience used to launch the session.",
    )
    return parser.parse_known_args(argv)


def _prepare_portable_root(portable_root: Path) -> Path:
    home_root = portable_root / "home"
    documents_root = home_root / "Documents"
    screenshot_root = portable_root / "documents" / "Kit" / "shared" / "screenshots"
    documents_root.mkdir(parents=True, exist_ok=True)
    screenshot_root.mkdir(parents=True, exist_ok=True)
    os.environ["HOME"] = str(home_root)
    return screenshot_root


def main() -> None:
    launcher_args, session_argv = _parse_launcher_args(sys.argv[1:])
    portable_root = launcher_args.portable_root.resolve()
    screenshot_root = _prepare_portable_root(portable_root)
    sys.argv = [sys.argv[0], *session_argv]
    headless = "--headless" in session_argv
    _log(f"Starting SimulationApp with experience {EXPERIENCE_BY_NAME[launcher_args.experience]}")

    from isaacsim import SimulationApp

    # This environment can stall indefinitely while waiting for the viewport to
    # report ready. We skip that wait for both headless and GUI launches and let
    # the session drive the app loop explicitly.
    SimulationApp._wait_for_viewport = lambda self: None  # type: ignore[attr-defined]

    app = SimulationApp(
        {
            "headless": headless,
            "extra_args": [
                "--portable-root",
                str(portable_root),
                f"--/app/captureFrame/path={screenshot_root}",
                f"--/persistent/app/captureFrame/path={screenshot_root}",
            ],
        },
        experience=str(EXPERIENCE_BY_NAME[launcher_args.experience]),
    )
    _log("SimulationApp initialized")

    try:
        _log("Importing avp_landau_session")
        import avp_landau_session as session

        _log("Imported avp_landau_session")
        session.main()
        _log("Session main returned")
    finally:
        try:
            import omni.timeline

            omni.timeline.get_timeline_interface().stop()
        except Exception:
            pass
        app.close()


if __name__ == "__main__":
    main()
