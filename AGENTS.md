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

## What To Preserve If You Wipe And Rebuild

Keep these files and folders even if most code is deleted:

- `AGENTS.md`
- `agent_guide.md`
- `CONTEXT.md`
- `skills/geo-walk-training/SKILL.md`
- `algorithms/urdf_learn_wasd_walk/TRAINING_RULES.md`
- `algorithms/urdf_learn_wasd_walk/milestones.json`
- `algorithms/urdf_learn_wasd_walk/inputs/landau_v10/`

The previous implementation, run history, and checkpoint lineage were intentionally removed. If these clean-room files are preserved, another coding agent can rebuild from the acceptance contract without inheriting stale experiments.

## Landau Continuation Rule

- First unresolved milestone wins.
- Standing for 30 seconds with zero signal and no reset is the first hard gate.
- Then standing for 30 seconds under policy control without reset is the next gate.
- Then record the first 5 m no-reset gate checkpoint.
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
