from __future__ import annotations

import argparse
import json

from algorithms.urdf_learn_wasd_walk.training_lineage import reset_landau_training_state


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Archive the current Landau state and initialize a fresh lineage.")
    parser.add_argument("--lineage-name", type=str, default=None)
    parser.add_argument("--archive-tag", type=str, default=None)
    parser.add_argument("--note", type=str, default=None)
    parser.add_argument("--archive-logs", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    payload = reset_landau_training_state(
        lineage_name=args.lineage_name,
        archive_tag=args.archive_tag,
        note=args.note,
        archive_logs=args.archive_logs,
    )
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
