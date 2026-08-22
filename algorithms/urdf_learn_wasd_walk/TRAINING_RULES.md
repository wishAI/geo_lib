# Landau Milestones — Clean Restart Contract

This file is the only retained behavioral contract from the old locomotion implementation. No milestone is currently passed and no earlier checkpoint or prose claim carries forward.

## Cumulative Rules

1. Work on the earliest unresolved milestone only.
2. A checkpoint reaches milestone `N` only after that exact checkpoint re-passes `1..N-1`.
3. Any fall, hard reset, or `done` event fails the run.
4. Motion passes only when it still looks like walking; sliding, collapsing, hopping, and frozen-joint motion fail.
5. Upper-body balance must remain natural enough for visual review.
6. Use one semantic command space everywhere: `forward`, `strafe`, `yaw`.
7. Path following must generate joystick-style commands rather than bypassing the policy interface.
8. Never run two Isaac processes at once.
9. Record evidence only after the implementation provides a repeatable validator and a short proof video.

## Milestone Ladder

1. `stand_zero_signal_30s_no_reset` (`stand`): passive URDF + PD control; zero action, zero command, zero fall, zero reset, zero `done` for 30 seconds.
2. `stand_30s_no_reset` (`stand`): policy-controlled stand for 30 seconds with zero fall, reset, or `done`.
3. `gate_5m_no_reset` (`fwd_only`): reach a 5 m flat forward gate with a walking pose and no reset.
4. `gate_10m_no_reset` (`fwd_only`): reach a 10 m flat forward gate and re-pass milestones 1–3.
5. `yaw_turn_90deg_hold` (`fwd_yaw`): turn 90 degrees, settle, and hold without reset.
6. `teleop_60s_forward_turn` (`fwd_yaw`): respond to forward and turn input for 60 seconds without reset or frozen joints.
7. `gate_10m_four_directions_no_reset` (`fwd_yaw`): the same checkpoint clears forward, left, right, and backward 10 m gates.
8. `triangle_path_follow_no_reset` (`fwd_yaw`): follow a closed triangle with joystick-style waypoint commands and no reset.
9. `square_path_follow_no_reset` (`fwd_yaw`): follow a closed square within tolerance and with no reset.
10. `terrain_5m_no_reset` (`game`): cross 5 m of the small rough-terrain map with no reset.
11. `obstacle_stop_before_collision` (`game`): brake before collision without reset.
12. `game_10m_no_reset` (`game`): clear 10 m of mixed terrain and obstacles while preserving a walking pose.

`milestones.json` is the machine-readable source for status. It starts with every milestone set to `not_started`.
