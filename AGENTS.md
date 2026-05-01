# AGENTS

This is the durable restart file for coding agents working in `geo_lib`.

If the repository is heavily restructured, or most code files are deleted and rebuilt, start here.

## Read Order

1. `AGENTS.md`
2. `agent_guide.md`
3. `CONTEXT.md`
4. `skills/geo-walk-training/SKILL.md`
5. `algorithms/urdf_learn_wasd_walk/TRAINING_RULES.md`
6. `algorithms/urdf_learn_wasd_walk/outputs/history/active_lineage.json`
7. `algorithms/urdf_learn_wasd_walk/outputs/history/refs/index.json`
8. `algorithms/urdf_learn_wasd_walk/outputs/history/refs/milestones.json`
9. `algorithms/urdf_learn_wasd_walk/outputs/history/checkpoint_registry.json`

## What To Preserve If You Wipe And Rebuild

Keep these files and folders even if most code is deleted:

- `AGENTS.md`
- `agent_guide.md`
- `CONTEXT.md`
- `skills/geo-walk-training/SKILL.md`
- `algorithms/urdf_learn_wasd_walk/TRAINING_RULES.md`
- `algorithms/urdf_learn_wasd_walk/outputs/history/`
- `algorithms/urdf_learn_wasd_walk/inputs/landau_v10/`
- any promoted or milestone checkpoints you still care about under `logs/rsl_rl/`

If those are preserved, another coding agent can reconstruct the project state and continue training without the old chat.

## Landau Continuation Rule

- First unresolved milestone wins.
- Standing for 30 seconds with zero signal and no reset is the first hard gate.
- Then standing for 30 seconds under policy control without reset is the next gate.
- Then record the first 5 m no-reset gate checkpoint.
- Then record the first 10 m no-reset gate checkpoint.
- Do not skip forward to later polish while an earlier gate is still broken.

## Commands

- Refresh machine-readable state:
  - `./geo walk refs`
- Record a milestone:
  - `./geo walk milestone --milestone-id <id> --stage <stage> --load_run <run> --checkpoint model_<n>.pt`
- Reset into a clean lineage:
  - `./geo walk reset --lineage-name <name>`

## If The Codebase Is Rebuilt From Scratch

Create the new code around the preserved rules and history, not around memory.

- Recreate the minimal launcher commands first.
- Recreate the stand diagnostic first.
- Recreate the 5 m evaluation next.
- Recreate milestone recording before long training.
- Only then rebuild play, teleop, terrain, obstacle, and path-follow flows.
