from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Drive the copied Landau URDF from AVP tracking and mirror the solved pose onto the USD character.",
    )
    parser.add_argument(
        "--tracking-source",
        choices=("bridge", "snapshot"),
        default="bridge",
        help="Use live bridge tracking or a saved snapshot payload.",
    )
    parser.add_argument(
        "--snapshot-path",
        default=None,
        help="Snapshot file used for --tracking-source snapshot and calibration scaling.",
    )
    parser.add_argument(
        "--refresh-inputs",
        action="store_true",
        help="Re-copy URDF/USD/STL inputs from algorithms/usd_parallel_urdf before launch.",
    )
    parser.add_argument(
        "--show-urdf",
        action="store_true",
        help="Leave the imported URDF articulation visible instead of showing only the USD character. Requires --import-stage-urdf.",
    )
    parser.add_argument(
        "--show-solved-markers",
        action="store_true",
        help="Render the solved hand marker overlay. Disabled by default because these debug markers can destabilize Isaac Sim in this scene.",
    )
    parser.add_argument(
        "--import-stage-urdf",
        action="store_true",
        help="Attempt to import the URDF articulation into the live Isaac stage. Disabled by default because that importer is unstable here.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run Isaac Sim headless.",
    )
    parser.add_argument(
        "--baseline-urdf-path",
        "--h1-2-urdf-path",
        "--g1-urdf-path",
        dest="baseline_urdf_path",
        default=None,
        help="Optional third-baseline URDF. Defaults to Unitree H1_2 from the cloned xr_teleoperate helper repo.",
    )
    parser.add_argument(
        "--baseline",
        "--with-baseline",
        "--with-h1-2",
        "--with-g1",
        dest="enable_baseline",
        action="store_true",
        help="Import the third articulated baseline in the compare scene. Disabled by default because the Isaac Sim importer can hard-crash on this asset.",
    )
    parser.add_argument(
        "--no-baseline",
        "--no-h1-2",
        "--no-g1",
        dest="enable_baseline",
        action="store_false",
        help="Disable the third articulated baseline in the compare scene.",
    )
    parser.set_defaults(enable_baseline=False)
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Optional frame limit for smoke tests. Zero means run until the app closes.",
    )
    parser.add_argument(
        "--dex-hand-rate-hz",
        type=float,
        default=20.0,
        help=(
            "Maximum live update rate for the dex hand-retarget helper. "
            "Lower values reduce main-thread stalls from subprocess IPC; "
            "zero or negative means retarget every frame."
        ),
    )
    parser.add_argument(
        "--arm-rate-hz",
        type=float,
        default=6.0,
        help=(
            "Maximum live update rate for arm retargeting in bridge mode. "
            "Lower values improve GUI smoothness by reusing the last arm solve between updates; "
            "zero or negative means solve arms every frame."
        ),
    )
    parser.add_argument(
        "--marker-rate-hz",
        type=float,
        default=15.0,
        help=(
            "Maximum live update rate for raw and solved marker visualization in bridge mode. "
            "Lower values improve GUI smoothness by decoupling debug marker updates from the render loop; "
            "zero or negative means update markers every frame."
        ),
    )
    parser.add_argument(
        "--arm-solver",
        choices=("auto", "fast", "accurate"),
        default="auto",
        help=(
            "Arm IK mode. 'fast' accepts the TracIK result directly; "
            "'accurate' falls back to slower CCD refinement when the IK target error is too large; "
            "'auto' uses fast for live bridge mode and accurate for snapshot mode."
        ),
    )
    dex_hand_group = parser.add_mutually_exclusive_group()
    dex_hand_group.add_argument(
        "--dex-hands",
        dest="dex_hands",
        action="store_true",
        help="Enable the external dex hand retarget helper.",
    )
    dex_hand_group.add_argument(
        "--no-dex-hands",
        dest="dex_hands",
        action="store_false",
        help="Disable the external dex hand retarget helper and use heuristic finger mapping.",
    )
    parser.set_defaults(dex_hands=None)
    parser.add_argument(
        "--profile-loop",
        action="store_true",
        help="Print aggregate timing for the main AVP session loop.",
    )
    parser.add_argument(
        "--profile-log-interval",
        type=int,
        default=120,
        help="How many frames to accumulate between loop timing logs when --profile-loop is enabled.",
    )
    return parser


def parse_args(argv: list[str]):
    return build_parser().parse_args(argv)
