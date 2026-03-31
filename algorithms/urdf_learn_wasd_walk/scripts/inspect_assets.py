from __future__ import annotations

import argparse

from algorithms.urdf_learn_wasd_walk.asset_setup import prepare_landau_inputs
from algorithms.urdf_learn_wasd_walk.robot_specs import load_landau_robot_spec


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect the copied custom URDF asset.")
    parser.add_argument("--refresh", action="store_true", help="Refresh the copied inputs before inspection.")
    args = parser.parse_args()

    prepared = prepare_landau_inputs(refresh=args.refresh)
    spec = load_landau_robot_spec()

    print(f"[ASSET] robot: {spec.display_name}")
    print(f"[ASSET] urdf: {spec.urdf_path}")
    print(f"[ASSET] usd: {prepared.usd_path}")
    print(f"[ASSET] skeleton_json: {prepared.skeleton_json_path}")
    print(f"[ASSET] textures: {prepared.texture_dir}")
    print(f"[ASSET] root link: {spec.root_link_name}")
    print(f"[ASSET] primary feet: {spec.primary_foot_links}")
    print(f"[ASSET] support links: {spec.support_link_names}")
    print(f"[ASSET] termination links: {spec.termination_link_names}")
    print(f"[ASSET] init root height: {spec.init_root_height:.4f}")
    print(f"[ASSET] leg joints: {len(spec.joint_groups.leg_joints)}")
    print(f"[ASSET] foot joints: {len(spec.joint_groups.foot_joints)}")
    print(f"[ASSET] arm joints: {len(spec.joint_groups.arm_joints)}")
    print(f"[ASSET] finger joints: {len(spec.joint_groups.finger_joints)}")
    print(f"[ASSET] missing meshes: {len(spec.missing_meshes)}")


if __name__ == "__main__":
    main()
