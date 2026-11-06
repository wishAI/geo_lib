---
name: geo-walk-training
description: Train, validate, and inspect Landau locomotion in urdf_learn_wasd_walk on TK2. Follow the current milestone execution order, preserve clean lineage and proof requirements, and bound resource use and cloud-asset I/O during long runs.
---

# Geo Walk Training

The old implementation and experimental history were intentionally removed on 2026-08-22. Do not reconstruct that deleted implementation from Git history or old checkpoints unless the user requests it. Preserve the new implementation and its explicit lineage.

## First Reads

1. `AGENTS.md`
2. `algorithms/urdf_learn_wasd_walk/README.md`
3. `algorithms/urdf_learn_wasd_walk/TRAINING_RULES.md`
4. `algorithms/urdf_learn_wasd_walk/milestones.json`
5. `algorithms/urdf_learn_wasd_walk/gui/manifest.json`

For unattended training, large cloud-backed assets, or GUI/SSH instability, also read [runtime stability](references/runtime-stability.md).

## Training and Acceptance

- Work on the earliest unresolved gate in the ledger's explicit `executionOrder`, including an already `in_progress` gate. Do not skip it because later code exists.
- Use the current code and CLI as the implementation inventory. Historical clean-room notes do not establish that training commands are still absent.
- Keep Isaac single-process on TK2; use the backend and interpreter declared by the current launcher. Do not run a second training job against the same lineage or outputs.
- Keep algorithm imports isolated and hand off through declared files.
- Keep generated runs and checkpoints out of Git. Durable files over 5 MiB follow root `large_files.json` and the Nextcloud contract.
- A pass requires machine-readable validation, checkpoint/source hashes, the exact acceptance protocol, and a short proof video with the required visual review. Keep diagnostic or review-pending candidates distinct from certified gates.
- Preserve the donor checkpoint, parent lineage, and cumulative proof needed by later gates. Do not inherit old certificates merely because weights were reused.

## Long Runs

- Choose bounded batches, CPU/thread concurrency, checkpoint cadence, log retention, and a local storage budget before launching an unattended campaign. Start conservatively and increase only while the host remains responsive.
- Keep the latest resumable checkpoint and the best checkpoint/proof per milestone; publish selected durable artifacts instead of every temporary rollout. Propose any pruning of existing runs before deletion.
- Preserve streamed large-file delivery and the bounded TK2 Nextcloud read cache. Avoid eager reads, duplicate cloud downloads, or repeated hashing of large files for GUI refreshes.
- Keep GUI status and console summaries small; use bounded log tails and refresh rates instead of loading complete training histories on every update.
- After a reboot or confirmed worker loss, mark the run interrupted and preserve its checkpoint/evidence. Never leave a dashboard claiming that an absent process is still training, or promote a milestone because a run ended.

## Current State

Read the live ledger and running-process state. Certified gates, current speed target, checkpoint identity, and interrupted/pending work must come from those records, not a hard-coded status in this skill.
