## Landau Training History

Last updated: 2026-04-02

### Goal

Make `algorithms/urdf_learn_wasd_walk` Stage A (`--stage fwd_only`) produce a checkpoint that passes:

- `validate_walk.py --robot landau --stage fwd_only --headless`
- expected acceptance:
  - planar displacement `>= 0.3 m` over `500` steps
  - mean forward speed `>= 0.15 m/s`
  - low orthogonal drift

### Key implementation fixes already landed

These code changes are on branch `task/fix_urdf_learn`:

- staged task/config selection for `fwd_only`, `fwd_yaw`, `full`
- explicit semantic forward-axis mapping via `forward_body_axis`
- teleop held/repeat-key handling fix
- `validate_walk.py` semantic command default fix
  - Stage A/B defaults must stay in semantic command space before remapping
- Stage A/B train-time command bias
  - `LandauFwdOnlyEnvCfg` and `LandauFwdYawEnvCfg` now sample `lin_vel_y` in `(0.35, 0.5)` instead of `(0.0, 0.5)`

### Training commands already run

#### 1. Initial Stage A run

Command:

```bash
/home/wishai/vscode/IsaacLab/isaaclab.sh -p -m algorithms.urdf_learn_wasd_walk.scripts.train \
  --robot landau --stage fwd_only --headless --run_name staged_fix
```

Useful checkpoint:

- `logs/rsl_rl/geo_landau_fwd_only/2026-04-02_13-19-56_staged_fix/model_300.pt`

Validation command:

```bash
/home/wishai/vscode/IsaacLab/isaaclab.sh -p -m algorithms.urdf_learn_wasd_walk.scripts.validate_walk \
  --robot landau --stage fwd_only --headless \
  --load_run 2026-04-02_13-19-56_staged_fix \
  --checkpoint model_300.pt
```

Result:

- failed
- planar displacement: `0.0899 m`
- mean forward speed: `0.0327 m/s`

#### 2. Resume from model_300

Command:

```bash
/home/wishai/vscode/IsaacLab/isaaclab.sh -p -m algorithms.urdf_learn_wasd_walk.scripts.train \
  --robot landau --stage fwd_only --headless \
  --resume True \
  --load_run 2026-04-02_13-19-56_staged_fix \
  --checkpoint model_300.pt \
  --run_name staged_fix_resume
```

Useful checkpoint:

- `logs/rsl_rl/geo_landau_fwd_only/2026-04-02_13-25-59_staged_fix_resume/model_700.pt`

Validation command:

```bash
/home/wishai/vscode/IsaacLab/isaaclab.sh -p -m algorithms.urdf_learn_wasd_walk.scripts.validate_walk \
  --robot landau --stage fwd_only --headless \
  --experiment_name geo_landau_fwd_only \
  --load_run 2026-04-02_13-25-59_staged_fix_resume \
  --checkpoint model_700.pt
```

Result:

- failed
- planar displacement: `0.1219 m`
- mean forward speed: `0.0354 m/s`

#### 3. Resume from model_700 after biasing train-time forward commands

Code change before this run:

- `LandauFwdOnlyEnvCfg.lin_vel_y = (0.35, 0.5)`

Command:

```bash
/home/wishai/vscode/IsaacLab/isaaclab.sh -p -m algorithms.urdf_learn_wasd_walk.scripts.train \
  --robot landau --stage fwd_only --headless \
  --resume True \
  --experiment_name geo_landau_fwd_only \
  --load_run 2026-04-02_13-25-59_staged_fix_resume \
  --checkpoint model_700.pt \
  --max_iterations 900 \
  --run_name staged_fix_bias_forward
```

Useful checkpoint:

- `logs/rsl_rl/geo_landau_fwd_only/2026-04-02_13-33-09_staged_fix_bias_forward/model_900.pt`

Validation command:

```bash
/home/wishai/vscode/IsaacLab/isaaclab.sh -p -m algorithms.urdf_learn_wasd_walk.scripts.validate_walk \
  --robot landau --stage fwd_only --headless \
  --experiment_name geo_landau_fwd_only \
  --load_run 2026-04-02_13-33-09_staged_fix_bias_forward \
  --checkpoint model_900.pt
```

Result:

- failed
- planar displacement: `0.1496 m`
- mean forward speed: `0.0506 m/s`

#### 4. Fresh Stage A run with stricter tracking reward and larger lower-body action scale

Code changes before this run:

- `LandauFwdOnlyEnvCfg.track_lin_vel_xy_exp.weight = 3.0`
- `LandauFwdOnlyEnvCfg.track_lin_vel_xy_exp.std = 0.2`
- `LandauFwdOnlyEnvCfg.track_ang_vel_z_exp.weight = 0.0`
- `LandauFwdOnlyEnvCfg.feet_air_time.weight = 1.0`
- `LandauFwdOnlyEnvCfg.action_rate_l2.weight = -0.0025`
- `LandauFwdOnlyEnvCfg.actions.joint_pos.scale = 0.35`

Command:

```bash
/home/wishai/vscode/IsaacLab/isaaclab.sh -p -m algorithms.urdf_learn_wasd_walk.scripts.train \
  --robot landau --stage fwd_only --headless \
  --run_name staged_fix_strict_reward
```

Useful checkpoint:

- `logs/rsl_rl/geo_landau_fwd_only/2026-04-02_15-42-13_staged_fix_strict_reward/model_800.pt`

Validation command:

```bash
/home/wishai/vscode/IsaacLab/isaaclab.sh -p -m algorithms.urdf_learn_wasd_walk.scripts.validate_walk \
  --robot landau --stage fwd_only --headless \
  --experiment_name geo_landau_fwd_only \
  --load_run 2026-04-02_15-42-13_staged_fix_strict_reward \
  --checkpoint model_800.pt
```

Result:

- still failed Stage A acceptance, but this was the first run to clear displacement
- planar displacement: `0.4173 m`
- mean forward speed: `0.0973 m/s`
- interpretation:
  - stricter reward shaping helped a lot
  - reward-only tuning on the 12-joint lower-body policy is still not enough

#### 5. Fresh Stage A run with full-body non-finger control and stock primary-foot biped reward

Code changes before this run:

- controlled joints expanded from lower body only to:
  - legs
  - feet/toes
  - torso
  - arms/hands
- fingers remain uncontrolled
- observation space expanded to match controlled joints
- Stage A action dim became `29`
- Stage A observation dim became `99`
- Stage A action scale moved to a per-joint dict with larger lower-body authority
- `feet_air_time` switched to stock `feet_air_time_positive_biped` on primary feet only
- `feet_slide` switched to primary feet only
- Stage A default `max_iterations` increased to `2500`

Command:

```bash
/home/wishai/vscode/IsaacLab/isaaclab.sh -p -m algorithms.urdf_learn_wasd_walk.scripts.train \
  --robot landau --stage fwd_only --headless \
  --run_name staged_fix_fullbody_stage_a
```

Useful checkpoint:

- `logs/rsl_rl/geo_landau_fwd_only/2026-04-02_15-59-09_staged_fix_fullbody_stage_a/model_500.pt`

Validation command:

```bash
/home/wishai/vscode/IsaacLab/isaaclab.sh -p -m algorithms.urdf_learn_wasd_walk.scripts.validate_walk \
  --robot landau --stage fwd_only --headless \
  --experiment_name geo_landau_fwd_only \
  --load_run 2026-04-02_15-59-09_staged_fix_fullbody_stage_a \
  --checkpoint model_500.pt
```

Result:

- passed Stage A acceptance
- planar displacement: `0.3956 m`
- mean forward speed: `0.2875 m/s`
- orthogonal drift speed: `0.0261 m/s`
- notes:
  - this is the first checkpoint that actually passes the Stage A plan requirements
  - mean yaw rate during validation was still high (`1.33 rad/s`), so Stage B / heading stabilization is still the logical next improvement, but Stage A forward teleop should now work

### Important validation bug that was found

Earlier `validate_walk.py` Stage A defaults incorrectly rewrote the semantic command to:

- semantic `(0.0, 0.5, 0.0)`

This then got remapped again for Landau and effectively became:

- env command `(0.0, 0.0, 0.0)`

That bug is fixed. Stage A defaults must remain:

- semantic `(0.5, 0.0, 0.0)`
- env `(0.0, 0.5, 0.0)` after Landau remap

### Current diagnosis

The main Stage A blocker is resolved.

The successful recipe was:

1. keep stricter Stage A forward-tracking reward
2. stop training only the lower body
3. switch the gait reward to stock primary-foot biped logic
4. give the lower body more action authority
5. train on the broader policy interface long enough to reach a passing checkpoint

### Official/upstream comparison that matters

These findings came from local IsaacLab source inspection plus official public IsaacLab discussions:

1. Official H1/G1 velocity tasks use full-joint position control by default.
   - Base `ActionsCfg` uses `joint_names=[".*"]` with `scale=0.5`
   - Base observations also include all joint positions and velocities
2. Official humanoid biped rewards use primary foot bodies only.
   - `feet_air_time_positive_biped` is applied on one foot body per side, not grouped `foot+toe` contact bundles
3. Official G1/H1 runner defaults are longer than our Stage A default.
   - rough tasks use `max_iterations = 3000`
   - our Stage A had been capped at `1500`
4. A relevant official IsaacLab discussion for custom G1 assets reported that the default `1500` iterations did not converge and that a retuned setup needed much longer training.
5. Landau URDF dynamics are probably unrealistic.
   - total URDF mass is only about `1.63 kg`
   - this is a real risk, but it is a larger model-fidelity issue than the training-config fixes below

### Recommended next experiments

Highest-leverage next changes:

1. If you only need forward/back teleop, use the passing Stage A checkpoint.
2. If you want cleaner heading behavior, continue to Stage B (`fwd_yaw`) from the successful Stage A run.
3. Keep validating with the fixed semantic command:
   - semantic `(0.5, 0.0, 0.0)`
   - env `(0.0, 0.5, 0.0)` for Landau
4. If future regressions appear, compare against `model_500.pt` from `staged_fix_fullbody_stage_a` first.

### Current training method for future agent

This is the working method that produced the first passing Stage A checkpoint:

1. Use the patched Stage A config that now boots with:
   - action dim `29`
   - observation dim `99`
   - controlled joints = lower body + torso + arms + hands
2. Train with:

```bash
/home/wishai/vscode/IsaacLab/isaaclab.sh -p -m algorithms.urdf_learn_wasd_walk.scripts.train \
  --robot landau --stage fwd_only --headless \
  --run_name staged_fix_fullbody_stage_a
```

3. Poll roughly every 10-20 minutes.
4. Validate checkpoints with:

```bash
/home/wishai/vscode/IsaacLab/isaaclab.sh -p -m algorithms.urdf_learn_wasd_walk.scripts.validate_walk \
  --robot landau --stage fwd_only --headless \
  --experiment_name geo_landau_fwd_only \
  --load_run <RUN_NAME> \
  --checkpoint model_<N>.pt
```

5. Recommended checkpoint cadence to inspect first:
   - `model_400.pt`
   - `model_800.pt`
   - `model_1200.pt`
   - `model_1600.pt`
   - `model_2000.pt`
   - final checkpoint

### Operational notes for future agent

- There were no stray IsaacLab processes at the last check.
- Best current checkpoint is `model_500.pt` from `staged_fix_fullbody_stage_a`, and it passes Stage A validation.
- `model_800.pt` from `staged_fix_strict_reward` was the first checkpoint to pass displacement but still failed speed (`0.097 m/s`).
- During long training, polling every 10-20 minutes is reasonable.
- Do not assume PPO reward means the walk is fixed. Always rerun `validate_walk.py`.

## 2026-04-02 Fast-Run Optimization Follow-Up

This section records the later Stage A speed-optimization work after the first pass.

### Goal

- Keep the passing Stage A walk.
- Increase top forward speed.
- Keep actual left/right stepping quality instead of accepting a one-sided hop.

### Config evolution

Fast-run tuning was applied in `landau_env_cfg.py`:

1. Raise Stage A/FwdYaw command ceiling from `0.8` to `1.0`.
2. Keep stronger forward tracking and add stronger yaw/slip pressure.
3. Reduce upper-body action authority slightly so the policy cannot buy speed mostly through spin torque.

Then a second fix was needed:

1. The stock upstream `feet_air_time_positive_biped` reward can still score a pathological one-sided single-stance gait.
2. To correct that, a grouped side-landing reward was added in `custom_rewards.py`:
   - `grouped_support_first_contact_biped`
3. Stage A was then retuned to use:
   - `feet_air_time.weight = 0.5`
   - `feet_step_contact.weight = 1.0`

### Runs and results

#### 1. Fast-forward baseline resume

Run:
- `2026-04-02_16-18-45_staged_fix_fast_forward`

Method:
- resumed from the first passing Stage A checkpoint
- widened Stage A command band toward faster forward motion

Useful checkpoints:
- `model_1500.pt` at semantic command `(1.0, 0.0, 0.0)` was not available yet because play was still capped below `1.0`
- `model_1750.pt` under the widened play cap showed:
  - at semantic command `(1.0, 0.0, 0.0)`
    - planar displacement `0.5098 m`
    - mean forward speed `0.5050 m/s`
    - mean yaw rate `1.6505 rad/s`
    - validator pass, but still very spin-heavy
  - at semantic command `(0.5, 0.0, 0.0)`
    - planar displacement `0.5469 m`
    - mean forward speed `0.4060 m/s`
    - mean yaw rate `1.2716 rad/s`

Interpretation:
- faster than the original passing walk
- still too spin-heavy
- not good enough as the final fast-run solution

#### 2. Anti-spin fast-run resume

Run:
- `2026-04-02_16-56-24_staged_fix_fast_forward_spin_control`

Command:

```bash
/home/wishai/vscode/IsaacLab/isaaclab.sh -p -m algorithms.urdf_learn_wasd_walk.scripts.train \
  --robot landau --stage fwd_only --headless \
  --resume True \
  --load_run 2026-04-02_16-18-45_staged_fix_fast_forward \
  --checkpoint model_1750.pt \
  --max_iterations 1000 \
  --run_name staged_fix_fast_forward_spin_control
```

Result:
- this pushed speed materially higher
- but it exposed a new failure mode at high speed: the right side stopped switching contacts

Representative checkpoints at semantic command `(1.0, 0.0, 0.0)`:
- `model_2200.pt`
  - planar displacement `0.8096 m`
  - mean forward speed `0.6074 m/s`
  - mean yaw rate `1.3031 rad/s`
  - failed validator: `right_leg contact switches = 0`
- `model_2500.pt`
  - planar displacement `0.7538 m`
  - mean forward speed `0.5995 m/s`
  - mean yaw rate `1.2900 rad/s`
  - failed validator: `right_leg contact switches = 0`
- `model_2749.pt`
  - planar displacement `0.8822 m`
  - mean forward speed `0.6693 m/s`
  - mean yaw rate `1.2676 rad/s`
  - failed validator: `right_leg contact switches = 0`

Important diagnosis:
- the policy got faster and cleaner in yaw than the earlier fast-forward run
- but it did so by drifting into a one-sided gait
- this is why PPO reward alone was misleading here

#### 3. Contact-recovery resume with grouped side-landing reward

Run:
- `2026-04-02_17-20-34_staged_fix_fast_forward_contact_recover`

Command:

```bash
/home/wishai/vscode/IsaacLab/isaaclab.sh -p -m algorithms.urdf_learn_wasd_walk.scripts.train \
  --robot landau --stage fwd_only --headless \
  --resume True \
  --load_run 2026-04-02_16-56-24_staged_fix_fast_forward_spin_control \
  --checkpoint model_2749.pt \
  --max_iterations 500 \
  --run_name staged_fix_fast_forward_contact_recover
```

This run kept the fast forward speed but restored actual contact switching.

Best balanced checkpoint:
- `model_3050.pt`
  - at semantic command `(1.0, 0.0, 0.0)`
    - planar displacement `0.9028 m`
    - contact switches `[2, 2]`
    - mean forward speed `0.5951 m/s`
    - mean yaw rate `1.2702 rad/s`
    - validator pass
  - at semantic command `(0.5, 0.0, 0.0)`
    - planar displacement `0.8005 m`
    - contact switches `[4, 10]`
    - mean forward speed `0.3005 m/s`
    - mean yaw rate `0.6442 rad/s`
    - validator pass

Fastest passing checkpoint:
- `model_3248.pt`
  - at semantic command `(1.0, 0.0, 0.0)`
    - planar displacement `0.8609 m`
    - contact switches `[2, 2]`
    - mean forward speed `0.6788 m/s`
    - mean yaw rate `1.4363 rad/s`
    - validator pass
  - at semantic command `(0.5, 0.0, 0.0)`
    - planar displacement `0.7114 m`
    - contact switches `[4, 54]`
    - mean forward speed `0.3289 m/s`
    - mean yaw rate `0.6893 rad/s`
    - validator pass

### Recommended checkpoints now

For general use:
- `2026-04-02_17-20-34_staged_fix_fast_forward_contact_recover/model_3050.pt`
  - better balanced fast-run checkpoint
  - still substantially faster than the pre-optimization Stage A walk

For maximum straight-line speed:
- `2026-04-02_17-20-34_staged_fix_fast_forward_contact_recover/model_3248.pt`
  - highest validated mean forward speed so far
  - slightly more yaw and more asymmetric low-speed switching than `model_3050.pt`

### Updated method for future agent

If the goal is specifically "make Landau run faster" rather than "just pass Stage A":

1. Start from the first passing walk:
   - `2026-04-02_15-59-09_staged_fix_fullbody_stage_a/model_500.pt`
2. Do the fast-forward resume:
   - `2026-04-02_16-18-45_staged_fix_fast_forward`
3. Do the anti-spin resume:
   - `2026-04-02_16-56-24_staged_fix_fast_forward_spin_control`
4. If high-speed validation fails contact switches, use the grouped side-landing reward and recover from the late fast checkpoint:
   - `2026-04-02_17-20-34_staged_fix_fast_forward_contact_recover`
5. Validate both:
   - normal forward command `(0.5, 0.0, 0.0)`
   - fast forward command `(1.0, 0.0, 0.0)`

### Current bottom line

- The original Stage A pass is preserved.
- Fast-running support now works with validated checkpoints.
- PPO reward alone was not sufficient to judge success; the high-speed gait needed contact-switch validation to rule out a one-sided hop.
