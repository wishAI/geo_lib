---
name: geo-walk-training
description: Use when working on Isaac Lab locomotion training, playback, teleop, validation, reproducibility, or restart planning for /home/wishai/vscode/geo_lib/algorithms/urdf_learn_wasd_walk. Prefer ./geo commands, read TRAINING_RULES.md and outputs/history refs first, keep Isaac single-process, and record milestone checkpoints before promoting later stages.
---

# Geo Walk Training

Use this local repo skill for `algorithms/urdf_learn_wasd_walk`.

## First Reads

1. `AGENTS.md`
2. `algorithms/urdf_learn_wasd_walk/TRAINING_RULES.md`
3. `algorithms/urdf_learn_wasd_walk/README.md`
4. `algorithms/urdf_learn_wasd_walk/outputs/history/active_lineage.json`
5. `algorithms/urdf_learn_wasd_walk/outputs/history/refs/index.json`
6. `algorithms/urdf_learn_wasd_walk/outputs/history/refs/milestones.json`
7. `algorithms/urdf_learn_wasd_walk/outputs/history/checkpoint_registry.json`
8. Tail the JSONL ledgers under `algorithms/urdf_learn_wasd_walk/outputs/history/`

## Non-Negotiables

- Never run two Isaac processes at the same time.
- Prefer `./geo` over raw Isaac commands.
- Train only for the earliest unresolved milestone in `TRAINING_RULES.md`.
- Record the first checkpoint that passes each milestone with `./geo walk milestone`.
- Do not call a moving policy "walking" unless the pose still looks like walking and the arms help balance.
- Do not treat backward or strafe polish as a blocker before the forward gates are stable.
- Start game-stage terrain work on the small map first.

## Commands

- Smoke:
  - `./geo walk smoke --stage game --headless`
- Train:
  - `./geo walk train --stage <stage> --headless --num_envs 64 --run_name <run_name>`
- Diagnose stand stability:
  - `./geo walk diagnose --stage stand --headless --action-mode policy`
- Evaluate forward gate:
  - `./geo walk eval --stage fwd_only --headless`
- Playback gate:
  - `./geo walk gate --stage game`
- Record milestone:
  - `./geo walk milestone --milestone-id <id> --stage <stage> --load_run <run> --checkpoint model_<n>.pt`
- Refresh refs:
  - `./geo walk refs`
- Clean restart:
  - `./geo walk reset --lineage-name <name>`

## Handoff Rule

If the code is restructured or mostly deleted, preserve these first:

1. `AGENTS.md`
2. `algorithms/urdf_learn_wasd_walk/TRAINING_RULES.md`
3. `algorithms/urdf_learn_wasd_walk/outputs/history/`
4. `algorithms/urdf_learn_wasd_walk/inputs/landau_v10/`
5. `skills/geo-walk-training/SKILL.md`

Another agent can restart from those files without needing the old conversation.
