---
name: geo-walk-training
description: Use when rebuilding Isaac Lab locomotion training, validation, playback, or teleop in the intentionally empty urdf_learn_wasd_walk sandbox. Read the clean milestone ladder first, implement only the earliest unresolved milestone, keep Isaac single-process, and do not revive old history or code.
---

# Geo Walk Training — Clean-Room Rebuild

The old Landau training implementation and its experimental history were intentionally removed on 2026-08-22. Do not search Git history, old branches, or remote checkpoints to reconstruct it unless the user explicitly asks. The clean reset is a product decision, not missing context.

## First Reads

1. `AGENTS.md`
2. `algorithms/urdf_learn_wasd_walk/README.md`
3. `algorithms/urdf_learn_wasd_walk/TRAINING_RULES.md`
4. `algorithms/urdf_learn_wasd_walk/milestones.json`
5. `algorithms/urdf_learn_wasd_walk/gui/manifest.json`

## Rebuild Rules

- Work on the first `not_started` milestone only.
- Recreate the smallest launcher, environment, validator, and tests needed for that milestone.
- Never run two Isaac processes simultaneously.
- Use TK2 for Isaac work and keep commands compatible with the root Geo Web GUI.
- Keep generated runs and checkpoints out of Git. Files over 5 MiB follow root `large_files.json` and the Nextcloud storage contract.
- A passing implementation needs machine-readable validation plus a short proof video before the milestone status changes.
- Later motion, teleop, terrain, obstacle, and path logic must not be added while an earlier gate is unresolved.

## Current State

There is no train/play/validation implementation. `./geo walk milestones` is the only supported walk command and every milestone is `not_started`.
