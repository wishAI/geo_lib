# Landau Training Restart Plan

Last updated: 2026-04-19

Scope: redesign `algorithms/urdf_learn_wasd_walk` so training can climb the
`TRAINING_RULES.md` ladder from a clean base. The milestone system stays.
Everything else is on the table.

---

## 1. Why the current project is stuck

### 1.1 Evidence from the active lineage

Lineage `restart_20260419_clean_zero_signal` currently has:

- `current_target_milestone = stand_zero_signal_30s_no_reset`
- zero rows in `training_runs.jsonl`, `diagnostic_runs.jsonl`,
  `validation_runs.jsonl`, `evaluation_runs.jsonl`
- nine archived lineages in `outputs/archives/` between 2026-04-18 and
  2026-04-19 — the ladder has been reset repeatedly without passing the
  very first gate

Last archived diagnostic (previous lineage, `stand_stage0_armbalance_150`,
`model_149.pt`, policy action mode, 600 steps):

- `status = failed`
- `failure = done count failed: value=6 threshold=0`
- `first_done_step = 92`
- `min_control_root_height = 0.072`

That means even after 150 iterations of pure `stand` training the robot
falls in under 2 s of simulated time and drops to ~7 cm root height
(floor threshold is `0.15` m). No later milestone can begin.

### 1.2 Root-cause ranking

1. **Unphysical robot dynamics.**
   `inputs/landau_v10/landau_v10_parallel_mesh.urdf` has:
   - 71 mass-bearing links
   - total mass ~ `1.83 kg`
   - largest link mass `0.08 kg`
   - body forward axis is `+Y`, which forces a second remap layer on top
     of Isaac Lab's velocity command
   A ~2 kg biped cannot produce realistic ankle torques, ground reaction
   forces, or inertia. Every past "wide-stance shuffle", "seated crawl",
   and "hop" failure mode is consistent with the policy exploiting the
   fact that staying upright costs almost nothing and toppling costs
   almost nothing. The cheat space is larger than the real-walking
   region. This is the single largest reason reward tuning has saturated.

2. **Reward surface has exploded.**
   - `landau_env_cfg.py` = 1357 lines
   - `custom_rewards.py` = 966 lines
   - `LandauStandEnvCfg` alone touches ~30 reward weights (`alive`,
     `upright`, `flat_orientation_l2`, 6 foot-phase rewards, 5
     width/stance rewards, 5 joint-deviation rewards, 4 action/dof
     rewards, multiple non-support penalties, etc.). Most are weighted
     zero for stand but are still evaluated.
   The policy optimizes a moving target that trades these terms against
   each other even when the only active goal is "don't fall with zero
   command". This violates the rule set by `train_history.md`:
   "simple reward-weight increases are not enough" — we kept adding
   weights instead of cutting them.

3. **Asymmetric stand init pose.**
   `build_landau_balance_init_joint_pos` injects:
   - `left_shoulder_lift = 0.12`, `right_shoulder_lift = -0.12`
   - `left_elbow = 0.22`, `right_elbow = 0.22`
   - `left_upper_arm_roll = 0.05`, `right_upper_arm_roll = 0.05`
   The zero-signal gate requires surviving a pure PD hold with **no
   action**. The init pose itself pushes the arms into a non-equilibrium
   configuration that the PD controller then has to fight, biasing the
   torso yaw / roll immediately.

4. **Policy shape is oversized for the first milestone.**
   Stand uses `LANDAU_BALANCE_CONTROLLED_JOINTS` (lower body + torso +
   arms + hands), yielding ~29 action dims and ~300 observation dims for
   a task that only needs the robot to not fall from a fixed pose under
   a zero command. The search space is bigger than the problem.

5. **Milestone/curriculum plumbing is correct, but drowning.**
   `TRAINING_RULES.md`, `active_lineage.json`, and `milestones.json` are
   sound and should be kept. The lineage reset / archive logic in
   `training_lineage.py` and the history ledgers work. The code that
   needs to be rewritten is the *env and reward definition layer* — not
   the bookkeeping.

### 1.3 Short summary

The robot is physically too light, the reward function tries to shape
behavior the asset cannot produce, and the first milestone is being
attacked with a full-body policy from an asymmetric pose. The ladder
bookkeeping is not the problem — it correctly refuses to promote any
checkpoint because none actually stand.

---

## 2. What we keep

Do **not** rebuild these. They are the project's durable spine.

- `TRAINING_RULES.md` — goal ladder, hard rules, restart workflow
- `outputs/history/` — lineage + refs + JSONL ledgers
- `training_lineage.py`, `run_history.py` — lineage / ledger code
- `task_registry.py` — gym registration shape
- `scripts/train.py`, `scripts/play.py`, `scripts/validate_walk.py`,
  `scripts/teleop.py`, `scripts/smoke_test.py`
- `isaac_lock.py` — single-Isaac-process guard
- `./geo walk` launcher and its subcommands
- `inputs/landau_v10/landau_v10_skeleton.json` and mesh/USD assets
  (the geometry is fine; only the inertial numbers are wrong)

---

## 3. What we redesign

### 3.1 URDF mass pass (blocking fix)

Create `inputs/landau_v10/landau_v10_parallel_mesh.fixed.urdf` (or patch
the existing file via a preprocessing script) so that:

- total mass is in the range `30–45 kg`
- torso/pelvis mass dominates (combined ≥ 40 % of total)
- thigh, shin, foot, arm segments carry realistic per-segment mass
  (use the ratios in any standard humanoid URDF — H1, G1, or Atlas as a
  reference)
- inertia tensors scale consistently (either recompute from mesh volume
  * uniform density, or mirror H1 block inertias scaled by segment
  length)

Add `scripts/rescale_landau_mass.py` that reads the current URDF, applies
the scaling table, and writes the fixed URDF. The script must be
deterministic and idempotent. Commit both the script and the fixed URDF.

Acceptance:

- a 600-step, action-zero, command-zero diagnostic run on
  `LandauStandEnvCfg_PLAY` with the fixed URDF records
  `first_done_step = -1` (never fell) and
  `min_control_root_height ≥ 0.17`
- this is the only test this milestone needs; no training yet

### 3.2 Collapse `landau_env_cfg.py`

Split the 1357-line file into:

- `landau_common.py` — URDF loading, joint groups, actuator defaults,
  init-pose builders, terrain scene, command space helpers
- `landau_stand_cfg.py` — `LandauStandEnvCfg` + `_PLAY`
- `landau_walk_cfg.py` — `LandauFwdOnlyEnvCfg`, `LandauFwdYawEnvCfg`
  (share a `_WalkBase` that only adds tracking + alive)
- `landau_game_cfg.py` — `LandauGameEnvCfg`, terrain presets

Each stage config is allowed at most **8 non-zero reward terms**.
If a term's weight becomes zero, delete the term — do not leave it as a
`weight = 0.0` line. If a new reward is needed later, add it
intentionally, not by default.

### 3.3 Replace `custom_rewards.py`

Delete `custom_rewards.py`. Introduce `landau_rewards.py` with only the
rewards each active stage actually uses:

Stand:
- `upright_posture_bonus` (stock)
- `control_root_height_floor` (custom, hard floor)
- `non_support_contacts_count` (count only, no force variant)
- `action_rate_l2` (stock)
- `dof_pos_limits` (stock)

Fwd-only adds:
- `track_lin_vel_xy_exp` (stock)
- `feet_air_time_positive_biped` (stock, primary feet)

Fwd-yaw adds:
- `track_ang_vel_z_exp` (stock)

Game adds:
- `terrain_levels_vel` curriculum (stock)
- `obstacle_brake_penalty` (minimal custom)

No phase clocks, no support-width deviation, no
`touchdown_support_width_excess`, no `joint_deviation_leg_twist`, etc.
If the simplified stack does not converge on a passing checkpoint at a
given milestone, *that* is the moment to design one new term — backed
by a diagnosed failure, not by a shopping list.

### 3.4 Stand-stage specifics

- Drop the stand action space to lower body + torso core only
  (`LANDAU_CORE_CONTROLLED_JOINTS`). Arms and hands stay at their default
  pose — they do not move during the first two milestones.
- Use the raw URDF default pose for init. Delete the asymmetric shoulder
  and elbow offsets from `build_landau_balance_init_joint_pos`.
- `commands.base_velocity.ranges` all stay `(0, 0)` and
  `rel_standing_envs = 1.0`, as today.
- PD gains: re-tune `configure_landau_stand_actuators` against the
  **fixed** URDF mass. Target damping ratio ≈ 1.0 per joint under the
  new inertias; do not keep the current `stiffness=190, damping=16`
  legs — those were tuned against a 1.8 kg robot.
- Add a dedicated `validate_stand.py` script that runs the
  `stand_zero_signal_30s_no_reset` scenario on a single env, returns
  JSON, and writes to `diagnostic_runs.jsonl` via `run_history.py`.

### 3.5 Observation slimming

The current policy exposes gait clock, foot contact state, foot mode
time, and foot positions in root frame even during stand. Disable all
of those in `LandauStandEnvCfg` and add them back per-stage in
`landau_walk_cfg.py`. The stand observation should be within
`base_ang_vel + projected_gravity + joint_pos + joint_vel + last_action`,
i.e. exactly the stock `LocomotionVelocityRoughEnvCfg` default minus
the command terms we know are zero.

### 3.6 Dead-file cleanup

Before starting, remove files already deleted in the working tree that
still have stale imports:

- `scripts/inspect_assets.py`, `scripts/prepare_inputs.py`,
  `scripts/validate_rewards.py`, `reward_probe.py`,
  `tests/test_reward_probe.py` — already deleted; prune any remaining
  references under `tests/` and `scripts/`.

Keep `tests/` green after each change. The `./geo walk test` command is
the gate.

---

## 4. Restart workflow (keeps the milestone system)

Run in order. Do not skip a step that failed — fix it.

### Step 0 — snapshot current state

```bash
./geo walk reset --lineage-name restart_20260419_mass_fix
```

This archives the empty `restart_20260419_clean_zero_signal` lineage and
creates a fresh one whose first target is still
`stand_zero_signal_30s_no_reset`.

### Step 1 — land the URDF mass fix

1. Implement `scripts/rescale_landau_mass.py` and write the fixed URDF.
2. Update `asset_setup.py` / `asset_paths.py` to load the fixed URDF.
3. Update `tests/test_asset_setup.py` and `tests/test_urdf_utils.py`
   to assert the new total mass.
4. Run `./geo walk test`. Must stay green.

### Step 2 — simplify configs and rewards

1. Introduce `landau_common.py`, `landau_stand_cfg.py`,
   `landau_walk_cfg.py`, `landau_game_cfg.py` as in §3.2.
2. Move `landau_rewards.py` in and delete `custom_rewards.py`.
3. Update `task_registry.py` entry points to match the new module paths.
4. Update every test under `tests/` that imported the old modules.
5. `./geo walk test`. Must stay green.

### Step 3 — diagnose zero-signal stand (no training)

```bash
./geo walk diagnose --stage stand --headless \
  --action-mode zero --steps 900 \
  --max-done-count 0 --min-control-root-height 0.17
```

Pass condition: `first_done_step = -1` and
`min_control_root_height ≥ 0.17`.

If it fails, the fix lives in the URDF / PD gains, **not** in rewards.
Do not add reward terms to make this pass — adjust mass distribution,
joint damping, or init pose. This is a passive-stability test.

When it passes, record the milestone:

```bash
./geo walk milestone \
  --milestone-id stand_zero_signal_30s_no_reset \
  --stage stand \
  --load_run passive \
  --checkpoint none \
  --manual-review "URDF + PD hold survives 30 s with zero action"
```

(The lineage code should already accept `checkpoint none` for passive
milestones; if it does not, extend `training_lineage.py` to mark a
milestone as `passive` and skip the checkpoint re-pass requirement for
stand_zero_signal only.)

### Step 4 — stand policy for milestone 2

Train short. Do not overshoot.

```bash
./geo walk train --stage stand --headless \
  --run_name stand_policy_v1 --max_iterations 200
```

Diagnose every 50 iterations with `--action-mode policy`. The first
checkpoint that passes `stand_30s_no_reset` is the one we promote.
Record with `./geo walk milestone --milestone-id stand_30s_no_reset ...`.

### Step 5 — forward-only ladder

Only after milestone 2 is recorded.

```bash
./geo walk train --stage fwd_only --headless \
  --resume True --load_run <stand_run> --checkpoint model_<n>.pt \
  --run_name fwd_only_v1 --max_iterations 1500
./geo walk eval --stage fwd_only --headless \
  --path-preset gate --gate-direction forward --path-distance 5
./geo walk eval --stage fwd_only --headless \
  --path-preset gate --gate-direction forward --path-distance 10
```

Record `gate_5m_no_reset`, then `gate_10m_no_reset`, on the exact
checkpoint. Re-run the stand diagnostics on that checkpoint before
recording — the ladder re-pass rule stays.

### Step 6 — fwd_yaw, path, and game stages

Follow the ladder as written in `TRAINING_RULES.md`. Each new stage
inherits the config only by explicit subclassing in
`landau_walk_cfg.py` / `landau_game_cfg.py`. No cross-stage reward
weight bleed-through.

---

## 5. Hard rules for the redesign

These carry forward the spirit of `TRAINING_RULES.md` into the code
layer:

- **Fix the physics before the reward.** No new reward term until the
  URDF mass pass lands and the passive stand diagnostic goes green.
- **Budget per stage: 8 reward terms max.** More terms require a
  documented diagnosis in `train_history.md` and an entry in
  `internet_walk_reward_notes.md`.
- **Never add a reward with weight 0.** Add the term only when it has
  non-zero weight.
- **One env config per stage, no cross-stage inheritance depth >1.**
  Stand does not inherit from Flat. Walk does not inherit from Stand.
  Each stage subclasses `LocomotionVelocityRoughEnvCfg` (or a shared
  `_LandauBase` that only wires the robot and common observations).
- **Zero-signal stand must pass passively.** If we need a trained
  policy to stand still with zero command, the asset is wrong.
- **Every milestone record names an exact file.** Lineage reset is the
  only way to retry; no silent overwrites.
- **Isaac lock stays on.** Single process at a time.

---

## 6. Files this plan touches

New:
- `scripts/rescale_landau_mass.py`
- `inputs/landau_v10/landau_v10_parallel_mesh.fixed.urdf`
- `landau_common.py`
- `landau_stand_cfg.py`
- `landau_walk_cfg.py`
- `landau_game_cfg.py`
- `landau_rewards.py`
- `scripts/validate_stand.py`

Modified:
- `asset_setup.py`, `asset_paths.py` — point at fixed URDF
- `task_registry.py` — updated entry points
- `tests/*` — update imports and asset assertions
- `TRAINING_RULES.md` — add one-line note that
  `stand_zero_signal_30s_no_reset` is a passive (URDF + PD) test, not a
  trained policy test
- `train_history.md` — append a new "2026-04-19 mass-fix restart"
  section as runs happen

Removed:
- `custom_rewards.py`
- the 1357-line monolithic `landau_env_cfg.py`
- stale `generated_apps/`, `reward_probe.py` artifacts already pending
  deletion in `git status`

---

## 7. Definition of done for this restart

The restart is complete when, on a single lineage:

1. `stand_zero_signal_30s_no_reset` is recorded against the fixed URDF,
   passive mode.
2. `stand_30s_no_reset` is recorded against a short stand policy.
3. `gate_5m_no_reset` and `gate_10m_no_reset` are recorded against the
   same forward-only checkpoint.
4. `train_history.md` has a new section documenting the mass fix, the
   config collapse, and the passing run names.
5. `./geo walk test` is green.

Beyond that, continue the ladder per `TRAINING_RULES.md`. No shortcut.
