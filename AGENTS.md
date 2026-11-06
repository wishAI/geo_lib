# AGENTS

This is the durable restart file for coding agents working in `geo_lib`.

If the repository is heavily restructured, or most code files are deleted and rebuilt, start here.

## Read Order

1. `AGENTS.md`
2. `agent_guide.md`
3. `CONTEXT.md`
4. `skills/geo-walk-training/SKILL.md`
5. `algorithms/urdf_learn_wasd_walk/TRAINING_RULES.md`
6. `algorithms/urdf_learn_wasd_walk/milestones.json`
7. `algorithms/urdf_learn_wasd_walk/gui/manifest.json`

## Task-Specific Project Skills

- For AVP head/neck or hand/finger mapping, palm orientation, arm skin deformation, URDF-to-USD pose transfer, or their playback regressions, read [`skills/avp-hand-arm-retargeting/SKILL.md`](skills/avp-hand-arm-retargeting/SKILL.md). It covers both `avp_remote` and `usd_parallel_urdf`; this work does not advance the locomotion milestone ladder.

## Long-Running Training and Host Stability

For unattended runs, cloud-backed GUI changes, or SSH/host instability, follow [the walking skill's runtime guidance](skills/geo-walk-training/references/runtime-stability.md). Keep runs checkpointable and bounded, preserve cached/streamed asset delivery, and record interrupted workers truthfully after a reboot. Git cleanup must preserve ignored training outputs and checkpoint lineage.

## What To Preserve If You Wipe And Rebuild

Keep these files and folders even if most code is deleted:

- `AGENTS.md`
- `agent_guide.md`
- `CONTEXT.md`
- `skills/geo-walk-training/SKILL.md`
- `skills/avp-hand-arm-retargeting/`
- `algorithms/urdf_learn_wasd_walk/TRAINING_RULES.md`
- `algorithms/urdf_learn_wasd_walk/milestones.json`
- `algorithms/urdf_learn_wasd_walk/inputs/landau_v10/`

The previous implementation, run history, and checkpoint lineage were intentionally removed. If these clean-room files are preserved, another coding agent can rebuild from the acceptance contract without inheriting stale experiments.

## Landau Continuation Rule

- First unresolved milestone wins.
- Standing for 30 seconds with zero signal and no reset is the first hard gate.
- Then standing for 30 seconds under policy control without reset is the next gate.
- Then record the first 5 m no-reset gate checkpoint.
- Then pass sustained walking speed gates at 0.1, 0.4, 0.8 and the user-confirmed 1.2 m/s target, following the explicit execution order in `milestones.json`.
- Then record the first 10 m no-reset gate checkpoint.
- Do not skip forward to later polish while an earlier gate is still broken.

## Commands

- Open the local sandbox GUI: `./geo gui`
- Inspect clean milestones: `./geo walk milestones`
- Check shared assets: `./geo storage status`
- Audit the 5 MiB repository limit: `./geo storage audit`

## If The Codebase Is Rebuilt From Scratch

Create the new code around the preserved rules and history, not around memory.

- Do not recover the deleted implementation from Git history unless the user explicitly asks.
- Recreate the minimal launcher commands first.
- Recreate the stand diagnostic first.
- Recreate the 5 m evaluation next.
- Recreate machine-readable milestone recording before long training.
- Only then rebuild play, teleop, terrain, obstacle, and path-follow flows.
