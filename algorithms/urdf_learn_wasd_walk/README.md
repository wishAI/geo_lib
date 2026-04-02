# URDF Learn WASD Walk

Train a humanoid locomotion policy in Isaac Lab that tracks `(v_x, v_y, yaw_rate)` commands and outputs joint position targets.

This folder is self-contained within the repository rules:

- The custom robot handoff is copied into this folder's `inputs/`.
- The implementation does not import Python code from other algorithm folders.
- Long-running Isaac workflows are isolated to CLI scripts under `scripts/`.

## Robots

- `g1`
  - Baseline using Isaac Lab's Unitree G1 task patterns.
  - Wrapped locally so the command interface is direct yaw-rate control instead of heading control.
- `landau`
  - Custom URDF loaded from `inputs/landau_v10/landau_v10_parallel_mesh.urdf`.
  - Mesh references are copied under `inputs/landau_v10/mesh_collision_stl/`.
  - The original colored USD asset, skeleton pose map, and textures are also copied under `inputs/landau_v10/` for GUI playback.

## Layout

- `inputs/`
  - copied custom URDF handoff
- `outputs/`
  - run artifacts you choose to place here
- `agents/`
  - local PPO configs
- `scripts/`
  - train, play, teleop, validation, smoke-test entry points
- `tests/`
  - fast pure-Python validation

## Fast Validation

Pure Python checks:

```bash
python3 -m pytest algorithms/urdf_learn_wasd_walk/tests -q
python3 -m algorithms.urdf_learn_wasd_walk.scripts.inspect_assets
python3 -m algorithms.urdf_learn_wasd_walk.scripts.validate_rewards
```

Headless Isaac environment smoke test:

```bash
/home/wishai/vscode/IsaacLab/isaaclab.sh -p -m algorithms.urdf_learn_wasd_walk.scripts.smoke_test \
  --robot landau --stage fwd_only --headless --steps 32
```

Two-iteration PPO smoke test:

```bash
/home/wishai/vscode/IsaacLab/isaaclab.sh -p -m algorithms.urdf_learn_wasd_walk.scripts.train \
  --robot g1 --headless --max_iterations 2
```

## Train

Baseline G1:

```bash
/home/wishai/vscode/IsaacLab/isaaclab.sh -p -m algorithms.urdf_learn_wasd_walk.scripts.train \
  --robot g1 --headless
```

Custom URDF Stage A forward walk:

```bash
/home/wishai/vscode/IsaacLab/isaaclab.sh -p -m algorithms.urdf_learn_wasd_walk.scripts.train \
  --robot landau --stage fwd_only --headless
```

The intended long run is hours, not minutes. Use `--max_iterations 2` first to confirm the pipeline.
For Landau, prefer explicit `--stage` values such as `fwd_only` or `fwd_yaw` instead of relying on the full-task default.

## Playback

Replay a known-good Stage A checkpoint and optionally force a fixed command:

```bash
/home/wishai/vscode/IsaacLab/isaaclab.sh -p -m algorithms.urdf_learn_wasd_walk.scripts.play \
  --robot landau --stage fwd_only --headless \
  --load_run 2026-04-02_17-20-34_staged_fix_fast_forward_contact_recover \
  --checkpoint model_3050.pt \
  --steps 500 --command-vx 0.5 --command-vy 0.0 --command-yaw 0.0
```

The script exports `policy.pt` and `policy.onnx` next to the loaded checkpoint under an `exported/` folder.
`play` requires a trained checkpoint under `logs/rsl_rl/<experiment>/...`, or explicit `--experiment_name` / `--load_run` / `--checkpoint` overrides.

For `landau`, you can also launch GUI playback and mirror the live URDF articulation onto the copied colored USD model:

```bash
/home/wishai/vscode/IsaacLab/isaaclab.sh -p -m algorithms.urdf_learn_wasd_walk.scripts.play \
  --robot landau --stage fwd_only --visual-mode usd \
  --load_run 2026-04-02_17-20-34_staged_fix_fast_forward_contact_recover \
  --checkpoint model_3050.pt \
  --steps 500
```

Use `--visual-mode both` to show the colored USD model and the URDF together. Synced USD visual mode uses a single displayed environment.

## Teleop

Keyboard teleop uses `W/S` for forward/backward, `A/D` for left/right strafe, and `Q/E` for yaw.

```bash
/home/wishai/vscode/IsaacLab/isaaclab.sh -p -m algorithms.urdf_learn_wasd_walk.scripts.teleop \
  --robot landau --stage fwd_only \
  --load_run 2026-04-02_17-20-34_staged_fix_fast_forward_contact_recover \
  --checkpoint model_3050.pt
```

For `landau`, GUI teleop defaults to the synced colored `.usdc` model. Pass `--visual-mode urdf` to see only the imported URDF visuals, or `--visual-mode both` to overlay both.
`teleop` requires GUI mode, so do not pass `--headless`.
`teleop` also requires a trained checkpoint for the selected robot and stage.

Gamepad teleop:

```bash
/home/wishai/vscode/IsaacLab/isaaclab.sh -p -m algorithms.urdf_learn_wasd_walk.scripts.teleop \
  --robot g1 --input-device gamepad
```

### Checkpoint compatibility

- The current Landau staged setup uses action dim `29` and observation dim `99`.
- Older Landau checkpoints from before the staged/full-body change used smaller policy shapes such as action dim `12` and observation dim `48`.
- If you launch `play` or `teleop` against one of those older checkpoints, RSL-RL will fail with `size mismatch for ActorCritic`.
- For Landau, always pass a matching `--stage`, and when in doubt also pass explicit `--load_run` and `--checkpoint`.

## Notes

- The custom reward weights are starting values, not tuned final values.
- The `landau` task is flat-ground only in this first implementation.
- `play` and `teleop` use local play-task variants with randomized command resampling disabled.
