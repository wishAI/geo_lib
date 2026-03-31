Project Overview
Train a bipedal humanoid to walk via velocity commands (WASD / gamepad) in NVIDIA Isaac Lab using PPO (RSL-RL).
Two robots: 
Unitree G1 in the assets/ folder (work mainly baseline to verify algorithm works)
Custom URDF: use usd_parallel_urdf/outputs/landau_v10_parallel_mesh.urdf, and corresponding mesh_collision_stl folder, copy this into the inputs of this algorithm folder

The final deliverable is a trained policy that accepts (v_x, v_y, yaw_rate) commands and outputs joint position targets.


The phases in brief, each with tests that gate the next phase:
Phase 0 —  + RSL-RL
Phase 1 — Load both URDFs (custom + G1), validate joints/meshes, spawn in viewer
Phase 2 — Define the velocity-tracking task (observations, commands, rewards, terminations) as an Isaac Lab ManagerBasedRLEnvCfg
Phase 3 — Validate every reward term individually (no NaNs, correct signs, correct response to perfect/bad tracking)
Phase 4 — Wire up PPO training via RSL-RL, smoke test with 2 iterations, then real training (~4hrs per robot)
Phase 5-6 — Inference scripts with WASD/gamepad input mapper, integration tests that verify the policy actually walks


A few things to note:
The reward weights in Phase 2 are starting values from established locomotion work — they will need tuning, especially for your custom robot
The plan references Isaac Lab's existing locomotion examples as the ground truth for code patterns — the agent should read those before writing new code
Phase 4's "real" training is a manual long-running step (hours on GPU), so the automated test only checks the pipeline doesn't crash
The troubleshooting table at the bottom covers the most common failure modes


If there's a long training task, you can wake up every 15 - 20min to check current performance.
There's a 4080S on this machine you can use.

--- 

Follow the sandbox rule in CONTEXT.md, read it about this project
You can find the isaac sim python env in /home/wishai/vscode/IsaacLab/isaaclab.sh, run it in headless mode if you need.
If no need to use isaac env, you can use  pyenv activate ptenv
