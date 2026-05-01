# Landau Training Rules

Train and validate the earliest unresolved milestone first.

The ladder is cumulative:

1. A checkpoint only reaches milestone `N` after that same checkpoint re-passes milestones `1..N-1`.
2. Any hard reset, fall, or `done` event during the test auto-fails the run.
3. Do not hand off or promote a later-stage checkpoint while an earlier ladder test is broken.

## Hard Rules

- `stand_zero_signal_30s_no_reset` is the first hard gate.
- `stand_zero_signal_30s_no_reset` is a passive URDF + PD gate, not a policy milestone. Record it with `--load_run passive --checkpoint none` after the diagnostic ledger shows a passing zero-action run.
- Every passed milestone must store the exact checkpoint path, stage, run name, evidence kind, and a short note.
- Moving checkpoints only count when the pose still looks like walking. Sliding, collapsing, hopping, or frozen-joint motion do not pass.
- Upper-body balance matters. Do not mark a moving milestone as passed if the arms are unnaturally frozen.
- Use the same semantic command space everywhere: `forward`, `strafe`, `yaw`.
- Path tests must use joystick-style commands derived from the waypoint path. Do not bypass the command interface with a separate direct controller.
- Never run two Isaac processes at the same time.

## Goal Ladder

1. `stand_zero_signal_30s_no_reset`
   Stage: `stand`
   Pass when: the robot stays upright for 30 seconds with zero actions, zero commanded motion, zero falls, zero hard resets, and zero `done` events.
   Record with: `diagnostic`

2. `stand_30s_no_reset`
   Stage: `stand`
   Pass when: the robot stays upright for 30 seconds under the stand policy with zero falls, zero hard resets, and zero `done` events.
   Record with: `diagnostic`

3. `gate_5m_no_reset`
   Stage: `fwd_only`
   Pass when: the robot reaches the 5 m flat forward gate with zero falls, zero hard resets, zero `done` events, and the pose still looks like walking.
   Record with: `evaluation`

4. `gate_10m_no_reset`
   Stage: `fwd_only`
   Pass when: the robot reaches the 10 m flat forward gate with zero falls, zero hard resets, zero `done` events, and the pose still looks like walking.
   Record with: `evaluation`

5. `yaw_turn_90deg_hold`
   Stage: `fwd_yaw`
   Pass when: the robot responds to a yaw command, turns, settles, and holds without a fall or hard reset.
   Record with: `validation`

6. `teleop_60s_forward_turn`
   Stage: `fwd_yaw`
   Pass when: the robot responds to teleop forward and turn input for 60 seconds without reset and without frozen joints.
   Record with: `manual` plus GUI review

7. `gate_10m_four_directions_no_reset`
   Stage: `fwd_yaw`
   Pass when: the same checkpoint clears 10 m gates in `forward`, `left`, `right`, and `backward` directions with zero falls, zero hard resets, and zero `done` events.
   Record with: `evaluation`

8. `triangle_path_follow_no_reset`
   Stage: `fwd_yaw`
   Pass when: the robot follows a closed triangle path using joystick-style forward/yaw commands generated from the path, staying within the waypoint tolerance and avoiding all hard resets.
   Record with: `evaluation`

9. `square_path_follow_no_reset`
   Stage: `fwd_yaw`
   Pass when: the robot follows a closed square path using joystick-style forward/yaw commands generated from the path, staying within the waypoint tolerance and avoiding all hard resets.
   Record with: `evaluation`

10. `terrain_5m_no_reset`
   Stage: `game`
   Pass when: the robot crosses 5 m of the small rough-terrain map with zero falls, zero hard resets, and zero `done` events.
   Record with: `evaluation`

11. `obstacle_stop_before_collision`
    Stage: `game`
    Pass when: the robot brakes and stops before collision on the game map without a hard reset.
    Record with: `evaluation`

12. `game_10m_no_reset`
    Stage: `game`
    Pass when: the robot clears the mixed 10 m game gate with terrain and obstacles enabled, without a hard reset and while preserving a walking pose.
    Record with: `evaluation`

## Evaluation Commands

Stand gate:

```bash
./geo walk diagnose --stage stand --headless --action-mode zero --steps 600 --max-done-count 0 --min-control-root-height 0.17
./geo walk diagnose --stage stand --headless --action-mode policy --steps 600 --max-done-count 0 --min-control-root-height 0.17
```

Forward gates:

```bash
./geo walk eval --stage fwd_only --headless --path-preset gate --gate-direction forward --path-distance 5
./geo walk eval --stage fwd_only --headless --path-preset gate --gate-direction forward --path-distance 10
```

Four-direction gate suite:

```bash
./geo walk eval --stage fwd_yaw --headless --path-preset gate --gate-direction left --path-distance 10
./geo walk eval --stage fwd_yaw --headless --path-preset gate --gate-direction right --path-distance 10
./geo walk eval --stage fwd_yaw --headless --path-preset gate --gate-direction backward --path-distance 10
```

Polygon path tests:

```bash
./geo walk eval --stage fwd_yaw --headless --path-preset triangle --path-edge-length 3.5 --path-arrival-radius 0.35
./geo walk eval --stage fwd_yaw --headless --path-preset square --path-edge-length 3.0 --path-arrival-radius 0.35
```

## Recording Rules

- The source of truth is the active lineage plus the refs under `outputs/history/`.
- Record the first checkpoint that passes each milestone.
- Do not replace a milestone entry with prose only.
- When recording a later milestone, note that the same checkpoint re-passed every earlier test in the ladder.
- Add a GUI note for moving milestones describing whether the pose still looked like walking and whether the arms were helping balance.
- If no ledger record exists yet, do not pretend the milestone passed. Run the relevant diagnose, validate, or eval command first.

## Common Commands

- Refresh refs:
  `./geo walk refs`
- Record a zero-signal standing checkpoint:
  `./geo walk milestone --milestone-id stand_zero_signal_30s_no_reset --stage stand --load_run <run> --checkpoint model_<n>.pt --manual-review "upright for 30 s with zero signal"`
- Record a standing checkpoint:
  `./geo walk milestone --milestone-id stand_30s_no_reset --stage stand --load_run <run> --checkpoint model_<n>.pt --manual-review "upright for 30 s"`
- Record a forward gate checkpoint:
  `./geo walk milestone --milestone-id gate_10m_no_reset --stage fwd_only --load_run <run> --checkpoint model_<n>.pt --manual-review "walking pose preserved; arms swing visible"`
- Reset to a clean lineage:
  `./geo walk reset --lineage-name <name>`

## Restart Workflow

1. Read this file.
2. Read `outputs/history/active_lineage.json`.
3. Read `outputs/history/refs/index.json`.
4. Read `outputs/history/refs/milestones.json`.
5. Read `outputs/history/checkpoint_registry.json`.
6. Tail the JSONL ledgers under `outputs/history/`.
7. Identify the earliest unresolved milestone.
8. Train only for that milestone first.
9. Re-run every earlier ladder test on the candidate checkpoint.
10. Record the milestone only after the checkpoint passes the full cumulative suite.
