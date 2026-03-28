# 1. Start the Simulator (Must be done before other imports)
from isaacsim import SimulationApp
# Launch with GUI (headless=False)
simulation_app = SimulationApp({"headless": False})

# 2. Import Utils (Now safe to import)
from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage
from omni.isaac.core import World

from config import ASSET_PRIM, LOAD_USD_PATH
from usd_utils import (
    apply_gray_override,
    apply_rest_pose,
    find_first_skeleton,
    print_skeleton_info,
)
from usd_skeleton_schema import (
    LEFT_FOOT_JOINT,
    LEFT_HAND_ROOT_JOINT,
    RIGHT_HAND_ROOT_JOINT,
    get_missing_expected_joints,
    get_unexpected_joints,
)

LEFT_HAND_JOINT = LEFT_HAND_ROOT_JOINT
RIGHT_HAND_JOINT = RIGHT_HAND_ROOT_JOINT

def main():
    import math

    # Initialize the World (handles physics, timeline, etc.)
    world = World(stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()

    # 3. Load your USD file
    # "prim_path" is where it will appear in the stage tree (e.g., /World/MyRobot)
    add_reference_to_stage(usd_path=str(LOAD_USD_PATH), prim_path=ASSET_PRIM)

    stage = get_current_stage()
    # clear_skel_animation(stage, "/World/MyAsset")
    print_skeleton_info(stage, ASSET_PRIM)
    apply_gray_override(stage, ASSET_PRIM)
    skel = find_first_skeleton(stage, ASSET_PRIM)
    # Prepare a simple looping animation on representative joints from the current rig.
    if skel:
        joints = skel.GetJointsAttr().Get() or []
        joint_names = [str(j) for j in joints]

        missing = get_missing_expected_joints(joint_names)
        unexpected = get_unexpected_joints(joint_names)
        if missing:
            print(f"[USD] Missing expected model joints: {len(missing)}")
        if unexpected:
            print(f"[USD] Unexpected joints in skeleton: {len(unexpected)}")

        if LEFT_HAND_JOINT not in joint_names:
            print(f"[USD] Joint not found for animation: {LEFT_HAND_JOINT}")
        if RIGHT_HAND_JOINT not in joint_names:
            print(f"[USD] Joint not found for animation: {RIGHT_HAND_JOINT}")
        if LEFT_FOOT_JOINT not in joint_names:
            print(f"[USD] Joint not found for animation: {LEFT_FOOT_JOINT}")

    # Reset the world to make sure physics are ready
    world.reset()

    # 4. Simulation Loop
    t = 0.0
    dt = world.get_physics_dt() if hasattr(world, "get_physics_dt") else 1.0 / 60.0
    while simulation_app.is_running():
        if skel:
            left_angle = 20.0 * math.sin(2.0 * math.pi * 0.5 * t)
            apply_rest_pose(
                skel,
                joint_rotations={
                    RIGHT_HAND_JOINT: ((1.0, 0.0, 0.0), left_angle),
                    LEFT_FOOT_JOINT: ((1.0, 0.0, 0.0), left_angle),
                },
            )

        world.step(render=True)  # Steps physics and rendering
        t += dt
    
    simulation_app.close()

if __name__ == "__main__":
    main()
