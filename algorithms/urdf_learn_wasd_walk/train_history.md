## Landau Training History

Last updated: 2026-04-05

### Executive Summary

This file is a handoff, not just a log. The detailed chronology below is still useful, but the opening sections should be treated as the current truth.

Landau command semantics:

- semantic forward is `(+vx, 0, 0)`
- Landau body forward axis is body `+Y`
- the environment remaps semantic `vx` into env `lin_vel_y`
- GUI `teleop` keyboard `W` currently sends semantic forward `0.8`

Current state:

- there is still no Landau Stage A checkpoint that cleanly passes the latest strict proper-walk validator
- the original user complaint about seated crawling was real, and that failure mode is now much better controlled
- the current best phase-clock checkpoints are clearly better than the old crawl / seated-slide baselines
- the remaining blocker is now mostly one metric: strict `mean support width`

Plain-language diagnosis:

- the policy can now learn a strong forward alternating gait
- it is no longer mainly moving by sitting, dragging, or using forbidden non-support contacts
- however, it still prefers a support base that is too wide for the strict validator
- the remaining motion looks more like a wide, stabilizing walk than a natural narrow-stance walk

### What To Load First

If another agent or human only reads one section, read this one.

Best current Stage A reference:

- `2026-04-05_18-31-42_phase_clock_v6_posture_narrowing/model_400.pt`
  - forward displacement `4.6926 m`
  - single-support ratio `0.8480`
  - double-support ratio `0.1460`
  - mean primary-foot force share `0.7303`
  - `done_count=0`
  - `non_support_contact_step_sum=0`
  - mean support width `0.3591`
  - strict validator result: failed only on mean support width

Latest "last scalar retune" test:

- `2026-04-05_18-38-24_phase_clock_v7_twist_clamp/model_449.pt`
  - forward displacement `4.9389 m`
  - single-support ratio `0.8560`
  - double-support ratio `0.1380`
  - mean primary-foot force share `0.7393`
  - `done_count=0`
  - `non_support_contact_step_sum=0`
  - mean support width `0.3640`
  - strict validator result: failed only on mean support width

Legacy comparison checkpoint:

- `2026-04-02_19-17-48_staged_fix_fast_forward_guard_trim_resume/model_3348.pt`
  - important as the best old anti-crawl / GUI comparison point
  - no longer the main recommended Stage A playback target
  - useful only as a reference for how much the phase-clock family improved forward alternating gait

### Old Failure Summary

Do not forget these older failure modes:

- `model_500.pt`, `model_3050.pt`, and `model_3248.pt` were later exposed as false positives by stricter anti-crawl validation
- the old seated / crawling cheat looked like "moving while sitting on the ground"
- some later runs improved foot loading and anti-crawl cleanliness without actually producing real forward alternating gait
- the project has repeatedly seen "partial wins" that improved one metric while making another worse

The most important old lessons:

- do not trust PPO reward alone
- do not trust raw forward displacement alone
- always re-run strict `validate_walk.py`
- always check `done_count`, `non_support_contact_step_sum`, `single_support_ratio`, `double_support_ratio`, and `mean support width`

### What Not To Retry Blindly

These paths were informative, but the current evidence says they should not be retried as isolated scalar tweaks:

- stronger width penalties by themselves
- stronger twist-clamp / foot-posture penalties by themselves
- claiming success from PPO reward curves or GUI appearance without strict validation
- treating `model_3348.pt` as the target behavior instead of a legacy anti-crawl baseline

### Current Code Snapshot

These notes describe the current code state in `LandauFwdOnlyEnvCfg`, not the early history.

Policy / interface:

- action dim: `29`
- observation dim: `111`
- observation additions relative to the old staged setup:
  - gait clock `sin/cos`
  - foot positions in the control-root frame
  - foot contact state
  - foot mode time

Current Stage A command distribution:

- `lin_vel_x = (0.0, 0.0)`
- `lin_vel_y = (0.4, 0.7)`
- `ang_vel_z = (0.0, 0.0)`

Current Stage A action scales:

- `leg_scale = 0.4`
- `foot_scale = 0.24`
- `toe_scale = 0.10`
- `torso_scale = 0.26`
- `arm_scale = 0.18`
- `hand_scale = 0.12`

Current Stage A reward / termination ideas:

- forward / command tracking:
  - `track_lin_vel_xy_exp.weight = 5.0`
  - `track_ang_vel_z_exp.weight = 2.0`
- phase-driven gait shaping:
  - `phase_clock_gait.weight = 5.0`
  - `primary_single_support.weight = 0.5`
- anti-crawl shaping:
  - `non_support_contacts.weight = -1.5`
  - `non_support_contact_force.weight = -0.02`
  - gait-guard illegal-contact termination
- posture / geometry shaping:
  - `control_root_height_floor.weight = -15.0`
  - `stance_width_deviation.weight = -0.25`
  - `stance_width_excess.weight = -3.0`
  - `touchdown_step_length_deficit.weight = -1.0`
  - `touchdown_support_width_excess.weight = -2.0`
  - `landing_step_ahead.weight = 2.5`
  - `joint_deviation_leg_twist.weight = -0.12`
  - `joint_deviation_feet.weight = -0.06`

Current PPO facts that matter:

- `num_steps_per_env = 48`
- `save_interval = 50`
- actor / critic hidden dims: `[512, 256, 128]`

Important note:

- this latest code state includes the `v7` twist-clamp experiment
- that experiment did not produce a strict-pass checkpoint
- so this is the latest experimental code snapshot, not a proven final setting

### Current Conclusion

What the latest evidence now says:

- support-geometry alignment was correct, but insufficient
- lower-body posture priors were reasonable, but insufficient
- stronger twist-clamp / width penalties were also insufficient
- simple scalar reward tuning has probably saturated on the remaining failure mode

Best current interpretation:

- the phase-clock redesign was absolutely worth doing
- it produced the strongest forward alternating gait in this repo so far
- but it still does not satisfy the strict narrow-stance criterion

Best next step from here:

- stop treating this as only a scalar reward-weight problem
- likely next credible directions:
  - a more explicit lateral foot-placement / stance-control objective
  - training specialized around the strict semantic `0.5` command instead of the broader Stage A band
  - correcting the unrealistic Landau URDF mass / dynamics and retraining under more physical support costs
  - imitation / motion-prior guidance if natural gait quality is the real target

### Workflow Reminders

Fresh Stage A run:

```bash
/home/wishai/vscode/IsaacLab/isaaclab.sh -p -m algorithms.urdf_learn_wasd_walk.scripts.train \
  --robot landau --stage fwd_only --headless \
  --run_name <RUN_NAME>
```

Resume run:

```bash
/home/wishai/vscode/IsaacLab/isaaclab.sh -p -m algorithms.urdf_learn_wasd_walk.scripts.train \
  --robot landau --stage fwd_only --headless \
  --resume True \
  --experiment_name geo_landau_fwd_only \
  --load_run <RUN_NAME> \
  --checkpoint model_<N>.pt \
  --max_iterations <K> \
  --run_name <NEW_RUN_NAME>
```

Resume gotcha:

- `--max_iterations` on resume means "train this many more iterations", not "train until absolute iteration K"

Strict Stage A validation:

```bash
/home/wishai/vscode/IsaacLab/isaaclab.sh -p -m algorithms.urdf_learn_wasd_walk.scripts.validate_walk \
  --robot landau --stage fwd_only --headless \
  --experiment_name geo_landau_fwd_only \
  --load_run <RUN_NAME> \
  --checkpoint model_<N>.pt
```

Polling reminder:

- 10-20 minute polling is fine
- never trust PPO reward alone
- always rerun strict validation before claiming success

### Detailed Chronology Archive

Everything below is the longer historical log. Some earlier sections record what was believed at the time and were later superseded by stricter validation, but they are still useful for understanding what was already tried and what failed.

### Important checkpoints to know

These are the checkpoints another agent should remember first.

1. `2026-04-02_15-59-09_staged_fix_fullbody_stage_a/model_500.pt`
   - historical importance:
     - first checkpoint that passed the old Stage A validator
     - first proof that the 29-action / 99-observation full-body policy was much better than the earlier lower-body-only version
   - later problem:
     - strict anti-crawl validation showed it was still a false positive
     - `non_support_contact_step_sum=1235`

2. `2026-04-02_17-20-34_staged_fix_fast_forward_contact_recover/model_3050.pt`
   - historical importance:
     - first "good" fast-run checkpoint we used in GUI
   - later problem:
     - user correctly noticed it was moving while seated / dragging
     - strict anti-crawl validation:
       - `non_support_contact_step_sum=1171`
       - `min_control_root_height=0.0947`

3. `2026-04-02_19-12-13_staged_fix_fast_forward_crawl_guard_force_resume/model_3249.pt`
   - importance:
     - first checkpoint in the recovery line that was clearly no longer dominated by crawling
   - performance at semantic `(0.5, 0.0, 0.0)`:
     - planar displacement `0.5550 m`
     - mean forward speed `0.1353 m/s`
     - `done_count=6`
     - `non_support_contact_step_sum=20`
   - problem:
     - still too unstable for the strict Stage A validator

4. `2026-04-02_19-17-48_staged_fix_fast_forward_guard_trim_resume/model_3300.pt`
   - importance:
     - good "speed side" checkpoint in the latest recovery family
   - performance at semantic `(0.5, 0.0, 0.0)`:
     - planar displacement `2.2823 m`
     - mean forward speed `0.3903 m/s`
     - `done_count=1`
     - `non_support_contact_step_sum=4`
   - problem:
     - orthogonal drift still too high:
       - `0.132 > 0.1`

5. `2026-04-02_19-17-48_staged_fix_fast_forward_guard_trim_resume/model_3348.pt`
   - importance:
     - current cleanest anti-crawl checkpoint
     - best GUI / teleop replacement for the old `model_3050.pt`
   - performance:
     - semantic `(0.8, 0.0, 0.0)`:
       - planar displacement `1.1738 m`
       - mean forward speed `0.1624 m/s`
       - `done_count=0`
       - `non_support_contact_step_sum=0`
       - strict validator pass
     - semantic `(1.0, 0.0, 0.0)`:
       - planar displacement `1.0851 m`
       - mean forward speed `0.1574 m/s`
       - `done_count=0`
       - `non_support_contact_step_sum=0`
       - strict validator pass
     - semantic `(0.5, 0.0, 0.0)`:
       - planar displacement `0.6986 m`
       - mean forward speed `0.1208 m/s`
       - `done_count=0`
       - `non_support_contact_step_sum=0`
       - fails only because forward speed is slightly below the `0.15 m/s` strict Stage A threshold
   - current qualitative problem:
     - cleaner than the old crawl checkpoints
     - still looks like a stiff wide-leg stabilizing gait rather than natural alternating walking

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

### Historical diagnosis at that time

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

### Historical bottom line at that time

- The original Stage A pass is preserved.
- Fast-running support now works with validated checkpoints.
- PPO reward alone was not sufficient to judge success; the high-speed gait needed contact-switch validation to rule out a one-sided hop.

## Anti-crawl recovery after GUI visualizer report

### Problem that was discovered later

The GUI report about `model_3050.pt` "moving while sitting on the ground" was correct.

The older Stage A and fast-run checkpoints were false positives under the earlier validator:

- `2026-04-02_15-59-09_staged_fix_fullbody_stage_a/model_500.pt`
  - `non_support_contact_step_sum=1235`
  - `min_control_root_height=0.1030`
- `2026-04-02_17-20-34_staged_fix_fast_forward_contact_recover/model_3050.pt`
  - `non_support_contact_step_sum=1171`
  - `min_control_root_height=0.0947`

The dominant contact bodies were `leg_stretch_*` plus fingertip links such as `middle3_r` and `index3_r`.
That means the policy had learned a seated or crawling locomotion mode that still satisfied the earlier foot-contact and displacement checks.

### Code changes that fixed the diagnosis

The recovery used four changes:

1. `validate_walk.py` gained default Stage A/B gait-guard contact reporting and limits.
2. `landau_env_cfg.py` gained:
   - non-support contact count penalty
   - dense non-support contact force penalty
   - gait-guard illegal-contact termination
   - stronger control-root height floor reward
3. `robot_specs.py` now defines `gait_guard_link_names`.
4. Finger phalanges were explicitly excluded from the gait guard because Stage A does not control finger joints, so fingertip contacts were creating optimizer noise without giving the policy a clean way to fix them.

Important lesson:

- Do not gate Landau on "all non-support links" blindly.
- Exclude uncontrolled fingertip chains from the gait guard.
- Keep shins, thighs, arms, forearms, and hands in the gait guard.

### Recovery runs

#### 1. Crawl-guard probe from `model_3050.pt`

Run:
- `2026-04-02_19-06-49_staged_fix_fast_forward_crawl_guard_probe`

Method:
- resumed from `model_3050.pt`
- added gait-guard penalties and termination

Representative result:
- `model_3100.pt` at semantic `(0.5, 0.0, 0.0)`
  - planar displacement `0.8989 m`
  - contact switches `[12, 24]`
  - mean forward speed `0.5302 m/s`
  - `non_support_contact_step_sum=1056`
  - `min_control_root_height=0.1027`
  - still failed badly on crawling

Interpretation:
- resuming from `3050` was still the right direction
- but count-only gait-guard penalties were too weak as dense signals

#### 2. Dense-force crawl-guard resume from `model_3100.pt`

Run:
- `2026-04-02_19-12-13_staged_fix_fast_forward_crawl_guard_force_resume`

Method:
- resumed from `model_3100.pt`
- added `mdp.contact_forces` penalty on gait-guard links
- lowered gait-guard termination threshold

Representative checkpoints:
- `model_3200.pt` at semantic `(0.5, 0.0, 0.0)`
  - planar displacement `0.3885 m`
  - mean forward speed `0.1364 m/s`
  - `done_count=7`
  - `non_support_contact_step_sum=112`
- `model_3249.pt` at semantic `(0.5, 0.0, 0.0)`
  - planar displacement `0.5550 m`
  - mean forward speed `0.1353 m/s`
  - `done_count=6`
  - `non_support_contact_step_sum=20`

Interpretation:
- this run removed the seated sliding failure
- the remaining problem was no longer crawling
- the new failure mode was a stability / low-speed tracking tradeoff near the validator's `0.5` command

#### 3. Guard-trim resume from `model_3249.pt`

Run:
- `2026-04-02_19-17-48_staged_fix_fast_forward_guard_trim_resume`

Method:
- resumed from `model_3249.pt`
- trimmed gait guard to exclude fingertip phalanges
- lowered Stage A command floor from `(0.55, 1.0)` to `(0.45, 1.0)` so semantic `0.5` stayed on-manifold while keeping fast-forward support

Key checkpoints:
- `model_3300.pt` at semantic `(0.5, 0.0, 0.0)`
  - planar displacement `2.2823 m`
  - mean forward speed `0.3903 m/s`
  - `done_count=1`
  - `non_support_contact_step_sum=4`
  - failed only on orthogonal drift: `0.132 > 0.1`
- `model_3348.pt` at semantic `(0.5, 0.0, 0.0)`
  - planar displacement `0.6986 m`
  - mean forward speed `0.1208 m/s`
  - `done_count=0`
  - `non_support_contact_step_sum=0`
  - failed only on forward speed threshold `0.1208 < 0.15`
- `model_3348.pt` at semantic `(0.8, 0.0, 0.0)`
  - planar displacement `1.1738 m`
  - mean forward speed `0.1624 m/s`
  - `done_count=0`
  - `non_support_contact_step_sum=0`
  - validator pass
- `model_3348.pt` at semantic `(1.0, 0.0, 0.0)`
  - planar displacement `1.0851 m`
  - mean forward speed `0.1574 m/s`
  - `done_count=0`
  - `non_support_contact_step_sum=0`
  - validator pass

### Current recommended starting points

For GUI `play` / `teleop` and the original "it is sitting on the ground" user complaint:

- `2026-04-02_19-17-48_staged_fix_fast_forward_guard_trim_resume/model_3348.pt`
  - cleanest anti-crawl checkpoint
  - validated at semantic `0.8` and `1.0`
  - `done_count=0`
  - `non_support_contact_step_sum=0`
  - but this is still not final-quality human-like walking

For future tuning if the goal is specifically "pass the strict Stage A validator at semantic `0.5`":

- continue from either:
  - `2026-04-02_19-17-48_staged_fix_fast_forward_guard_trim_resume/model_3300.pt`
    - enough forward speed, but too much orthogonal drift
  - `2026-04-02_19-17-48_staged_fix_fast_forward_guard_trim_resume/model_3348.pt`
    - very clean gait, but slightly under the `0.15 m/s` forward-speed bar at semantic `0.5`

### Reproduction method for future agent

If a future agent sees Landau "moving while seated" again:

1. Validate with gait-guard contacts, not just foot contacts and displacement.
2. Start the recovery from `model_3050.pt`, not from scratch.
3. Add both:
   - count-based gait-guard penalty
   - dense contact-force penalty
4. Add a gait-guard illegal-contact termination.
5. Exclude fingertip phalanges from the gait guard.
6. Lower the Stage A minimum commanded forward speed from `0.55` to `0.45` when recovering the semantic `0.5` validation case.
7. Treat `model_3348.pt` as the clean non-crawling baseline for GUI usage, then continue tuning low-speed drift / speed from there if needed.

### Current qualitative diagnosis

This is the most important plain-language summary for a future coding agent:

- the latest checkpoint is **not** doing the old seated crawl anymore
- it is also better than the previous fast-run checkpoints because the upper body stays up and the forbidden crawl contacts are gone
- however, it still does not look like ordinary walking or running
- the current motion is closer to:
  - opening the legs wide
  - using that wide stance to keep the torso from falling
  - moving forward with a conservative stiff shuffle
- that is why the project should be considered:
  - improved
  - anti-crawl
  - but still not solved as natural gait

If a future agent continues from here, the target is no longer "remove sitting crawl".
The target becomes:

- convert the current anti-crawl wide-stance shuffle into a real alternating walk / run
- keep the upper body stable
- keep non-support contact near zero
- keep the new strict validator behavior

### 2026-04-04: control-path fix plus primary-foot load / single-support redesign

#### 1. Control-path bug fix for `play`, `teleop`, and `validate_walk`

Problem discovered:
- Stage A `fwd_only` commands use a positive-only forward band.
- `clamp_base_velocity_command()` used to snap semantic zero to the minimum forward speed in that band.
- `play.py` and `validate_walk.py` also forced commands and then inferred actions from stale observations.

Fixes:
- `isaac_workflow.py`
  - zero or backward user intent in a positive-only band now stays at env command `0.0` instead of snapping to the minimum forward command
- `teleop.py`
  - already refreshed observations after forcing commands; that behavior was kept
  - default idle behavior remains `default_pose`, so zero-command teleop visibly stops instead of letting policy drift dominate
- `play.py`
  - now refreshes observations after forcing a fixed command
  - if any command axis is provided, omitted axes default to `0.0` instead of disabling fixed-command mode
- `validate_walk.py`
  - now refreshes observations after forcing the validation command
  - now reports `mean_primary_force_share`

Zero-command verification:
- `2026-04-04_12-12-38_primary_foot_motion_recovery_v11/model_3549.pt`
  - semantic command `(0.0, 0.0, 0.0)` now maps to env command `(0.0, 0.0, 0.0)`
  - planar displacement `0.1135 m`
  - single-support ratio `0.0240`
  - double-support ratio `0.9680`
  - mean primary-foot force share `0.7778`

Interpretation:
- the "auto-moves forward even at zero command" control bug is fixed
- remaining zero-command drift is policy behavior, not command plumbing

#### 2. `2026-04-04_12-45-29_primary_foot_load_phase_v14`

Run:
- resumed from `2026-04-04_12-12-38_primary_foot_motion_recovery_v11/model_3549.pt`

Method:
- reduced Stage A toe action scale from `0.20` to `0.14`
- raised `rel_standing_envs` from `0.05` to `0.10`
- moved prolonged double-support penalty back to primary feet only
- strengthened primary-foot flat-contact penalty
- strengthened auxiliary-support penalty
- added auxiliary-contact force-share penalty so toe / edge loading is penalized even when the primary foot still has a light touch
- validator now measures `mean_primary_force_share`

Best representative checkpoint:
- `model_3698.pt` at semantic `(0.5, 0.0, 0.0)`
  - planar displacement `0.1087 m`
  - forward displacement `0.1087 m`
  - single-support ratio `0.0240`
  - double-support ratio `0.9680`
  - mean support width `0.3586`
  - mean primary-foot force share `0.9951`
  - min control-root height `0.2432`
  - `done_count=0`
  - `non_support_contact_step_sum=0`

Interpretation:
- this run dramatically improved load path cleanliness
- the robot now mostly loads the primary foot instead of the toe / inner-edge cheat
- but it did **not** improve the actual gait phase
- the failure mode became: cleaner feet, same near-static double-support shuffle

#### 3. `2026-04-04_12-52-12_single_support_reward_v15`

Run:
- resumed from `2026-04-04_12-45-29_primary_foot_load_phase_v14/model_3698.pt`

Method:
- added a new direct primary-foot single-support reward with brief double-support grace
- this followed the "single foot contact" direction from recent standing-and-walking reward-design literature

Best representative checkpoint:
- `model_3797.pt` at semantic `(0.5, 0.0, 0.0)`
  - planar displacement `0.0939 m`
  - forward displacement `0.0892 m`
  - single-support ratio `0.0240`
  - double-support ratio `0.9680`
  - mean support width `0.3721`
  - mean primary-foot force share `0.9973`
  - min control-root height `0.2263`
  - `done_count=0`
  - `non_support_contact_step_sum=0`

Interpretation:
- the new reward became large during PPO training
- but it still did not convert the behavior into real forward alternating gait under strict validation
- this run is not promotable

### Current recommendation after v14 / v15

- keep `2026-04-02_19-17-48_staged_fix_fast_forward_guard_trim_resume/model_3348.pt` as the best motion baseline for GUI playback and debugging
- treat `v14` / `v15` as useful diagnosis runs, not deployment checkpoints
- current evidence says:
  - control responsiveness is better now
  - primary-foot loading is cleaner now
  - but Stage A still does not produce a dramatic real-walk improvement

### Best next step from here

Small scalar reward tuning is no longer the highest-value path.
The next credible options are:

- explicit phase / clock structure with commanded walk-to-stand transitions
- imitation / motion prior guidance
- stronger foothold / capture-point style touchdown objectives tied to root progression

### 2026-04-05: phase-clock width refinement follow-up (`v5` / `v6` / `v7`)

After the earlier phase-clock rework, the remaining blocker was no longer "can it alternate feet" or "is it still crawling". The remaining blocker was:

- strict Stage A `mean support width`

The best recent checkpoints were already strong on:

- forward displacement
- single-support ratio
- low double-support ratio
- zero non-support contact steps
- zero done count

but they still failed the strict width threshold.

#### 1. `2026-04-05_18-26-53_phase_clock_v5_support_width_align`

Run:

- resumed from `2026-04-05_18-20-27_phase_clock_v4_width_refine/model_300.pt`

Method:

- aligned the width-related rewards with the same support geometry the validator uses
- specifically, `stance_width_*` and `touchdown_support_width_excess` were switched from primary-foot-only geometry to `foot + toes` support geometry

Why this mattered:

- `validate_walk.py` measures support width using support links (`foot_*` plus `toes_01_*`)
- the prior training signal only used primary foot links
- that mismatch meant the reward was optimizing a looser proxy than the actual failing validator metric

Smoke:

- `phase_clock_v5_support_width_align_smoke` passed

Validation:

- `model_350.pt`
  - forward displacement `4.6351 m`
  - single-support ratio `0.8160`
  - double-support ratio `0.1780`
  - mean primary-foot force share `0.7257`
  - mean support width `0.3611`
  - strict validator: failed only on mean support width

Interpretation:

- the reward/validator geometry mismatch was real and worth fixing
- but fixing that mismatch alone did **not** narrow the gait into the strict support-width range

#### 2. `2026-04-05_18-31-42_phase_clock_v6_posture_narrowing`

Run:

- resumed from `2026-04-05_18-26-53_phase_clock_v5_support_width_align/model_350.pt`

Method:

- reduced Stage A foot action scale from `0.28` to `0.24`
- reduced Stage A toe action scale from `0.14` to `0.10`
- added `joint_deviation_leg_twist`
- added `joint_deviation_feet`
- enabled them for Stage A with:
  - `joint_deviation_leg_twist.weight = -0.04`
  - `joint_deviation_feet.weight = -0.04`

Reasoning:

- the remaining width failure looked more like a lower-body posture / leg-splay problem than a missing contact penalty
- this run tried to keep the distal foot posture and leg twist closer to the default standing posture without removing the learned phase-clock gait

Smoke:

- `phase_clock_v6_posture_narrowing_smoke` passed

Validation:

- `model_400.pt`
  - forward displacement `4.6926 m`
  - single-support ratio `0.8480`
  - double-support ratio `0.1460`
  - mean primary-foot force share `0.7303`
  - mean support width `0.3591`
  - strict validator: failed only on mean support width
- `model_450.pt`
  - forward displacement `4.8664 m`
  - single-support ratio `0.8280`
  - double-support ratio `0.1660`
  - mean primary-foot force share `0.7455`
  - mean support width `0.3634`
  - strict validator: failed only on mean support width

Interpretation:

- this did produce the best width among the latest refinement passes at `model_400.pt`
- but the improvement was small
- continuing farther to `model_450.pt` made the width worse again
- that strongly suggests the run was saturating around the same failure mode rather than converging to the strict width target

#### 3. `2026-04-05_18-38-24_phase_clock_v7_twist_clamp`

Run:

- resumed from `2026-04-05_18-31-42_phase_clock_v6_posture_narrowing/model_400.pt`
- short 50-iteration experiment, intentionally limited

Method:

- strengthened the remaining width / splay penalties:
  - `stance_width_excess.weight = -3.0`
  - `touchdown_support_width_excess.weight = -2.0`
  - `joint_deviation_leg_twist.weight = -0.12`
  - `joint_deviation_feet.weight = -0.06`

Reasoning:

- this was the "last scalar retune" test
- if strong twist clamp and stronger width penalties still could not bring the support width under the validator threshold, then this reward family was very likely saturated

Smoke:

- `phase_clock_v7_twist_clamp_smoke` passed

Validation:

- `model_449.pt`
  - forward displacement `4.9389 m`
  - single-support ratio `0.8560`
  - double-support ratio `0.1380`
  - mean primary-foot force share `0.7393`
  - mean support width `0.3640`
  - strict validator: failed only on mean support width

Interpretation:

- the stronger twist clamp did **not** solve the problem
- it improved neither the strict width metric nor the validator outcome
- the model kept learning a very strong forward alternating gait, but still preferred a support base that was too wide for the strict criterion

### Current conclusion after `v5` / `v6` / `v7`

- the phase-clock family is a real improvement over the old anti-crawl shuffle
- the latest runs produce much better forward alternating gait than the old `single_support_reward_v15` / `primary_foot_load_phase_v14` era
- however, there is still **no** Landau Stage A checkpoint that cleanly passes the latest strict proper-walk validator
- the remaining blocker is still support width

The new evidence is important:

- simple reward-weight increases are **not** enough
- support-geometry alignment was correct, but insufficient
- lower-body posture priors were directionally reasonable, but still insufficient
- a stronger twist clamp was also insufficient

This now supports the higher-level diagnosis in `problems/landau_walk_fix_plan.md`:

- reward-only local tuning is no longer the highest-value path for the remaining failure mode

Best current non-pass reference from these latest runs:

- `2026-04-05_18-31-42_phase_clock_v6_posture_narrowing/model_400.pt`
  - strongest strict-validator-like gait among the latest width retunes
  - still failed on mean support width `0.3591`

Best next step from here:

- stop treating this as a scalar reward-weight problem
- either:
  - move to a more structural lateral-stance / foot-placement objective
  - specialize training around the strict semantic `0.5` command instead of the wider Stage A command band
  - or finally correct the unrealistic Landau URDF mass / dynamics and retrain under physically less "cheap" wide-stance behavior
