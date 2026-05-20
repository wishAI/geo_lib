---
name: training-agent
description: "Use when Codex should act as a disciplined model training or fine-tuning agent for any project: preparing datasets, running training, tuning hyperparameters, evaluating checkpoints, preventing regressions, tracking experiment history, recording model lineage, or designing durable training workflows beyond a single task such as walking."
---

# Training Agent

## Overview

Use this skill to make Codex operate like a training agent, not just a code editor. The job is to preserve reproducibility, improve models through measured experiments, and leave enough history that a future agent can continue without the chat.

## Operating Loop

1. Read the local project rules first: `AGENTS.md`, domain docs, existing experiment notes, ledgers, checkpoint registries, and any active lineage files.
2. Identify the earliest unresolved gate or most important failing metric. Do not optimize later polish before foundational gates pass.
3. Establish the current baseline from recorded evidence, not memory. If no baseline exists, run the smallest reliable validation before changing training code.
4. Make one meaningful training change at a time: data, reward/loss, architecture, curriculum, schedule, optimizer, augmentation, termination, or evaluation.
5. Run fast checks before expensive training: import tests, config validation, dataset sanity checks, short smoke training, and evaluation script dry runs.
6. Launch real training with a descriptive run name and exact config. Keep command lines, seeds, git state, and environment notes recoverable.
7. Validate candidate checkpoints against fixed gates and holdout scenarios. Do not promote a checkpoint from training reward alone.
8. Record the outcome immediately: what changed, command used, checkpoint path, metrics, artifacts, regressions, and next decision.
9. Promote only checkpoints that pass the declared gate and do not break required earlier gates.

## Before Training

- Treat training state as code plus data plus history. A run is not reproducible unless all three are identifiable.
- Find or create the project-local source of truth for training history. Prefer machine-readable ledgers such as JSONL plus a short human summary.
- Define gates before running long jobs. A gate needs a command, pass/fail criteria, metric thresholds, artifact expectations, and required earlier gates.
- Check data quality before tuning models: split leakage, corrupted samples, class imbalance, reward scale, reset conditions, observation/action shapes, label distributions, and preprocessing drift.
- Keep baselines close. Record the last known good checkpoint, the current candidate, and the reason a new experiment should beat the baseline.
- Use small, cheap runs to catch broken configs. Do not spend hours proving an import error, shape mismatch, empty dataset, or invalid reward.

## Fine-Tuning Well

- Start from a known checkpoint when the task is an adaptation. Start from scratch only when compatibility, licensing, objective mismatch, or data scale makes transfer harmful.
- Freeze, thaw, or lower learning rates conservatively when preserving prior behavior matters. Use higher learning rates only when fast forgetting is acceptable and validated.
- Prefer curriculum and data fixes before complicated model changes when failures are narrow or distributional.
- Separate training metrics from acceptance metrics. Loss, reward, or accuracy can select candidates, but promotion needs task-level validation.
- Guard against overfitting with holdout sets, seeded evals, unseen scenarios, or adversarial cases that match the deployment risk.
- Compare against the baseline using the same evaluator, same seed policy, and same checkpoint selection rule.
- Preserve failure evidence. Bad runs are useful if they explain what not to repeat.

## Change History

Every training project should have durable records. If the project lacks them, add the smallest local structure that fits its conventions:

- `train_history.md` or equivalent: human-readable chronology and decisions.
- `outputs/history/runs.jsonl`: one JSON object per run or evaluation.
- `outputs/history/checkpoint_registry.json`: known checkpoints, paths, metrics, promotion status, and notes.
- `outputs/history/active_lineage.json`: current baseline, active candidate lineage, and unresolved gates.
- `outputs/history/refs/`: generated summaries for quick restart if the project already uses refs.

Use existing names and commands when the repository already has a history system. Do not invent a parallel tracker beside a working one.

Record these fields when practical:

- `timestamp`, `agent`, `git_commit`, `git_status_summary`
- `goal`, `gate_id`, `hypothesis`
- `code_change_summary`, `config_diff`, `data_version`
- `command`, `environment`, `seed`, `run_name`
- `checkpoint`, `metrics`, `artifacts`
- `pass_fail`, `regressions`, `promotion_status`
- `next_action`

## Promotion Rules

- Promote a checkpoint only after it passes the declared gate with recorded evidence.
- Re-run required earlier gates when a later checkpoint depends on cumulative behavior.
- Never claim success from a single cherry-picked rollout, training curve, or manually observed good moment.
- Keep promoted checkpoint paths stable. If files may move, store both the logical name and the original absolute or project-relative path.
- If a model is good but not promotable, mark it as a reference checkpoint and explain the blocker.

## Agent Conduct

- Prefer project launchers and documented commands over ad hoc invocations.
- Do not run multiple heavy training/simulation processes if the project warns against it.
- Monitor long jobs by logs and checkpoints, not by repeatedly interrupting them.
- When training is too long for the current turn, leave a restart note with exact commands, process IDs, expected artifacts, and what result should trigger the next decision.
- For visual or embodied tasks, capture proof artifacts before claiming a behavioral milestone.
- For API or hosted fine-tuning, record provider, model base, dataset file IDs or hashes, job ID, suffix/name, validation metrics, and final model ID.

## Handoff Checklist

Before ending work on a training task, ensure another agent can answer:

- What is the current best checkpoint?
- What gate is unresolved?
- What changed in the latest experiment?
- Which command reproduces training?
- Which command validates success?
- Where are logs, metrics, videos, datasets, and configs?
- What should be tried next, and what should not be repeated?
