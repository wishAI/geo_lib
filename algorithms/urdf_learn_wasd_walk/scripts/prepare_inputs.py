from __future__ import annotations

import argparse

from algorithms.urdf_learn_wasd_walk.asset_setup import prepare_landau_inputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Copy the custom Landau URDF and synced USD visual handoff into local inputs/.")
    parser.add_argument("--refresh", action="store_true", help="Replace the local copied handoff from the source folder.")
    args = parser.parse_args()

    prepared = prepare_landau_inputs(refresh=args.refresh)
    print(f"[INPUTS] urdf: {prepared.urdf_path}")
    print(f"[INPUTS] mesh_root: {prepared.mesh_root}")
    print(f"[INPUTS] usd: {prepared.usd_path}")
    print(f"[INPUTS] skeleton_json: {prepared.skeleton_json_path}")
    print(f"[INPUTS] textures: {prepared.texture_dir}")


if __name__ == "__main__":
    main()
