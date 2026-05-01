# URDF Learn WASD Walk

Train and operate a Landau locomotion policy in Isaac Lab using the same semantic command space everywhere:

- `forward`
- `strafe`
- `yaw`

That is the command space behind `play`, `teleop`, path evaluation, and the goal ladder. Manual control stays joystick/WASD-first; waypoint tests now convert paths into the same joystick-style signals instead of using a separate control path.

## Inputs

- Robot asset: `inputs/landau_v10/landau_v10_parallel_mesh.urdf`
- Support files: `inputs/landau_v10/mesh_collision_stl/`, `landau_v10_skeleton.json`, textures, and `landau_v10.usdc`

## Main Commands

Use the repo launcher. It keeps the Isaac environment and defaults consistent.

```bash
./geo walk play --stage game
./geo walk teleop --stage game
./geo walk train --stage stand --headless --run_name stand_trial
./geo walk diagnose --stage stand --headless --action-mode zero
./geo walk diagnose --stage stand --headless --action-mode policy
./geo walk validate --stage fwd_only --headless
./geo walk eval --stage fwd_yaw --headless --path-preset gate --gate-direction forward --path-distance 10
./geo walk refs
```

## Manual Control

Interactive playback:

```bash
./geo walk play --stage game
```

Interactive teleop with keyboard:

```bash
./geo walk teleop --stage game
```

Keyboard defaults:

- `W` / `S`: forward / backward
- `A` / `D`: turn left / right in game mode
- `Q` / `E`: strafe left / right in game mode
- `L`: zero the command

The simple operator goal is: use the same `forward, strafe, yaw` inputs to move the character around the map, then reuse that command space for automated milestone tests.

## Training Workflow

The active source of truth is:

- `TRAINING_RULES.md`
- `outputs/history/active_lineage.json`
- `outputs/history/refs/index.json`
- `outputs/history/refs/milestones.json`

Normal loop:

1. Refresh refs and inspect the earliest unresolved milestone.
2. Smoke test or diagnose if the failure is reset/idle related.
3. Train only the stage needed for that earliest milestone.
4. Re-run the milestone suite on the exact candidate checkpoint.
5. Record the milestone only after the checkpoint passes the current test and every earlier ladder test again.

Useful commands:

```bash
./geo walk test
./geo walk smoke --stage stand --headless
./geo walk diagnose --stage stand --headless --action-mode zero --steps 600 --max-done-count 0
./geo walk train --stage stand --headless --run_name stand_v1
./geo walk diagnose --stage stand --headless --load_run <run> --checkpoint model_<n>.pt --action-mode policy --steps 600 --max-done-count 0
./geo walk milestone --milestone-id stand_zero_signal_30s_no_reset --stage stand --load_run <run> --checkpoint model_<n>.pt
./geo walk milestone --milestone-id stand_30s_no_reset --stage stand --load_run <run> --checkpoint model_<n>.pt
```

## Path And Gate Evaluation

`eval` now supports path presets that emit joystick-style commands from waypoints.

Forward 10 m gate:

```bash
./geo walk eval --stage fwd_only --headless --path-preset gate --gate-direction forward --path-distance 10
```

10 m gate in the left direction:

```bash
./geo walk eval --stage fwd_yaw --headless --path-preset gate --gate-direction left --path-distance 10
```

Closed triangle path:

```bash
./geo walk eval --stage fwd_yaw --headless --path-preset triangle --path-edge-length 3.5 --path-arrival-radius 0.35
```

Closed square path:

```bash
./geo walk eval --stage fwd_yaw --headless --path-preset square --path-edge-length 3.0 --path-arrival-radius 0.35
```

Explicit waypoint file:

```bash
./geo walk eval --stage game --headless --path-file /abs/path/to/waypoints.json
```

## Goal Ladder

The milestone order is strict:

1. `stand_zero_signal_30s_no_reset`
2. `stand_30s_no_reset`
3. `gate_5m_no_reset`
4. `gate_10m_no_reset`
5. `yaw_turn_90deg_hold`
6. `teleop_60s_forward_turn`
7. `gate_10m_four_directions_no_reset`
8. `triangle_path_follow_no_reset`
9. `square_path_follow_no_reset`
10. `terrain_5m_no_reset`
11. `obstacle_stop_before_collision`
12. `game_10m_no_reset`

Rules:

- Any hard reset, fall, or done event auto-fails the test.
- A later milestone only counts when the same checkpoint also re-passes every earlier milestone.
- Moving checkpoints must still look like walking. Sliding, collapsing, hopping, or frozen-joint motion does not count.

## History And Promotion

Keep the machine-readable history current:

```bash
./geo walk refs
./geo walk gate --stage game
./geo walk milestone --milestone-id <id> --stage <stage> --load_run <run> --checkpoint model_<n>.pt
```

Only promote a checkpoint after the relevant gate suite passes on that exact file.
