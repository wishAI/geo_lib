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
- `train_history.md`
  - chronological handoff log for what was tried, what regressed, and which checkpoints are still worth loading
- `internet_walk_reward_notes.md`
  - internet research notes for how this repo currently defines proper walking versus cheating motion
- `README.txt`
  - short operator summary of the current best-known checkpoints and strict-validator status

## Current Landau Status

- There is still no Landau Stage A checkpoint that cleanly passes the latest strict proper-walk validator.
- The best current forward-walk reference is `2026-04-05_18-31-42_phase_clock_v6_posture_narrowing/model_400.pt`.
- That checkpoint is strong on forward displacement, single-support ratio, non-support contact cleanliness, and stability, but it still fails the strict width gate:
  - mean support width `0.3591`
  - strict threshold `0.3214`
- The later `v7` twist-clamp experiment did not fix that remaining failure mode.
- Use [train_history.md](/home/wishai/vscode/geo_lib/algorithms/urdf_learn_wasd_walk/train_history.md) as the source of truth before promoting a checkpoint.

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

Strict Stage A walk validation:

```bash
/home/wishai/vscode/IsaacLab/isaaclab.sh -p -m algorithms.urdf_learn_wasd_walk.scripts.validate_walk \
  --robot landau --stage fwd_only --headless \
  --experiment_name geo_landau_fwd_only \
  --load_run <RUN_NAME> \
  --checkpoint model_<N>.pt
```

`validate_walk.py` is no longer just a displacement smoke test. For Landau Stage A it now checks, by default:

- forward displacement in the initial forward frame
- single-support ratio and low flight ratio
- max double-support ratio
- mean support width
- mean primary-foot force share
- touchdown step length and touchdown root straddle
- yaw-rate error for forward-only commands
- control-root height
- non-support contact count

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

The train CLI also supports gentle fine-tune overrides that were added for checkpoint rescue runs:

```bash
/home/wishai/vscode/IsaacLab/isaaclab.sh -p -m algorithms.urdf_learn_wasd_walk.scripts.train \
  --robot landau --stage fwd_only --headless \
  --resume True \
  --experiment_name geo_landau_fwd_only \
  --load_run <RUN_NAME> \
  --checkpoint model_<N>.pt \
  --max_iterations 80 \
  --learning_rate 0.0001 \
  --entropy_coef 0.003 \
  --desired_kl 0.003 \
  --num_learning_epochs 4 \
  --num_mini_batches 4 \
  --reset_optimizer
```

## Playback

Replay the current best Stage A reference checkpoint headless and optionally force a fixed command:

```bash
/home/wishai/vscode/IsaacLab/isaaclab.sh -p -m algorithms.urdf_learn_wasd_walk.scripts.play \
  --robot landau --stage fwd_only --headless \
  --load_run 2026-04-05_18-31-42_phase_clock_v6_posture_narrowing \
  --checkpoint model_400.pt \
  --steps 500 --command-vx 0.5 --command-vy 0.0 --command-yaw 0.0
```

Current caveat:

- `2026-04-05_18-31-42_phase_clock_v6_posture_narrowing/model_400.pt` is the best current playback reference for the new phase-clock Stage A setup
- it is **not** a clean strict-validator pass under the latest proper-walk criteria because the mean support width is still too large
- the short follow-up run `2026-04-05_18-38-24_phase_clock_v7_twist_clamp/model_449.pt` also failed on the same width metric
- `model_3348.pt` is now best treated as a legacy anti-crawl comparison checkpoint, not the main recommended Stage A playback target
- use [train_history.md](/home/wishai/vscode/geo_lib/algorithms/urdf_learn_wasd_walk/train_history.md) for the current truth before promoting a checkpoint

The script exports `policy.pt` and `policy.onnx` next to the loaded checkpoint under an `exported/` folder.
`play` requires a trained checkpoint under `logs/rsl_rl/<experiment>/...`, or explicit `--experiment_name` / `--load_run` / `--checkpoint` overrides.
For Landau checkpoints, `play` now restores the saved `params/env.yaml` action scales and command ranges before constructing the environment so older checkpoints are not replayed through newer Stage A control mappings.

For `landau`, this is the explicit GUI playback command. It mirrors the live URDF articulation onto the copied colored USD model:

```bash
/home/wishai/vscode/IsaacLab/isaaclab.sh -p -m algorithms.urdf_learn_wasd_walk.scripts.play \
  --robot landau --stage fwd_only --visual-mode usd \
  --load_run 2026-04-05_18-31-42_phase_clock_v6_posture_narrowing \
  --checkpoint model_400.pt \
  --steps 500 --command-vx 0.5 --command-vy 0.0 --command-yaw 0.0
```

Use `--visual-mode both` to show the colored USD model and the URDF together. Synced USD visual mode uses a single displayed environment.
For a pure URDF GUI view, use `--visual-mode urdf` instead of `usd`.

## Teleop

Keyboard teleop uses `W/S` for forward/backward, `A/D` for left/right strafe, and `Q/E` for yaw.

```bash
/home/wishai/vscode/IsaacLab/isaaclab.sh -p -m algorithms.urdf_learn_wasd_walk.scripts.teleop \
  --robot landau --stage fwd_only \
  --load_run 2026-04-05_18-31-42_phase_clock_v6_posture_narrowing \
  --checkpoint model_400.pt
```

For `landau`, GUI teleop defaults to the synced colored `.usdc` model. Pass `--visual-mode urdf` to see only the imported URDF visuals, or `--visual-mode both` to overlay both.
`teleop` requires GUI mode, so do not pass `--headless`.
`teleop` also requires a trained checkpoint for the selected robot and stage.
`teleop` keyboard `W` maps to semantic forward `0.8`. The latest phase-clock checkpoints were strongest around semantic forward `0.5-0.7`, so fixed-command `play --command-vx 0.5` is still the better apples-to-apples inspection path for the strict Stage A benchmark.
For `landau fwd_only`, semantic zero now stays a true zero command instead of snapping to the minimum forward stage speed. Teleop now defaults to latched trim commands plus `--idle-action-mode policy`; press `L` to zero the command, or pass `--no-latch-command` if you want hold-to-command behavior.

Gamepad teleop:

```bash
/home/wishai/vscode/IsaacLab/isaaclab.sh -p -m algorithms.urdf_learn_wasd_walk.scripts.teleop \
  --robot g1 --input-device gamepad
```

### Checkpoint compatibility

- The current Landau staged setup uses action dim `29` and observation dim `111`.
- Older Landau checkpoints from before the staged/full-body change used smaller policy shapes such as action dim `12` and observation dim `48`.
- If you launch `play` or `teleop` against one of those older checkpoints, RSL-RL will fail with `size mismatch for ActorCritic`.
- For Landau, always pass a matching `--stage`, and when in doubt also pass explicit `--load_run` and `--checkpoint`.

## Notes

- The custom reward weights are starting values, not tuned final values.
- The `landau` task is flat-ground only in this first implementation.
- `play` and `teleop` use local play-task variants with randomized command resampling disabled.
- `play` and `validate_walk` now refresh observations after forcing commands, so fixed-command playback and validation are no longer one step behind the requested command.
- In `play`, if any of `--command-vx/--command-vy/--command-yaw` is provided, omitted axes default to `0.0`.
- Older fast-run checkpoints such as `model_3050.pt` and `model_3248.pt` are no longer recommended for GUI use; the stricter anti-crawl validator found they could move by dragging non-support links on the ground.
- `model_3348.pt` is no longer the main GUI recommendation in this README. Keep it only as a legacy anti-crawl comparison point.
- The repo now defines "proper walking" more strictly than earlier runs did. A checkpoint can move forward and still fail if it does so with excessive flight, excessive double-support shuffling, wide stance, low root height, or bad touchdown geometry.
