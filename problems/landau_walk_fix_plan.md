# Landau Stage A Natural Walk: Fix & Improvement Plan

Last updated: 2026-04-05

## Executive Summary

After 15+ training runs and extensive reward tuning, Landau Stage A still produces a **wide-leg stabilizing shuffle** instead of natural alternating bipedal walking. The best checkpoint (`model_3348.pt`) is clean on anti-crawl, clean on non-support contacts, but has:
- single-support ratio as low as 0.024 (v14/v15 runs)
- double-support ratio as high as 0.968
- forward speed of 0.12 m/s at semantic 0.5 (below 0.15 threshold)
- mean support width 0.34+ (too wide)

**Root cause**: scalar reward tuning alone cannot break the robot out of a local minimum where "standing still with wide legs" scores higher than "risking a step". The history shows diminishing returns from v5 through v15 -- each run improved one metric while regressing another.

---

## Problem Analysis

### Problem 1: No Gait Phase Structure (Critical)

**What**: The policy has no notion of "when" to lift or place each foot. All gait rewards are reactive (rewarding states after they happen) rather than prescriptive (telling the policy where in the gait cycle it should be).

**Evidence**: 
- v7 async gait timing reward did not solve the shuffle
- v15 single-support reward got large in PPO training but still did not convert to real stepping
- double-support ratio remains at 0.96+ in the latest checkpoints

**Why it matters**: Without a phase clock, the policy gets no gradient signal for *initiating* a step. It only gets credit *after* lifting a foot, but the risk of falling during a step attempt is penalized by height/termination rewards. This creates a stable equilibrium at "don't step".

### Problem 2: URDF Mass is Unrealistic (Critical)

**What**: Total URDF mass is ~1.63 kg. A real humanoid of this scale would weigh 20-60x more. The actuator limits (120 N-m effort for legs) are vastly overpowered for a 1.63 kg robot.

**Evidence**: Noted in the training history under "Official/upstream comparison" section.

**Impact**:
- Gravity contributes almost nothing to the dynamics
- The robot can "hover" in wide-stance configurations with minimal effort
- Foot contacts generate unrealistically low forces, making contact-based rewards noisy
- Actuator effort ratios are meaningless -- every joint can trivially overcome gravity
- This partly explains why the policy prefers standing wide: the energy cost is near-zero

### Problem 3: Reward Weight Interactions Create Opposing Gradients (High)

**What**: The reward stack has grown to 25+ terms with several that create directly opposing incentives.

**Key conflicts**:
- `track_lin_vel_xy_exp` (weight=5.5) says "move forward"
- `control_root_height_floor` (weight=-20.0) says "don't let root drop"
- `flight_time_penalty` (weight=-2.0) says "don't go airborne"
- `double_support_time_penalty` (weight=-5.0) says "don't stay in double support"
- The last two together say "always be in single support but never airborne" -- but single support is risky for a lightweight robot

**Result**: The optimizer finds a narrow valley where it shuffles with minimal height change, satisfying the floor constraint while minimally violating everything else.

### Problem 4: Touchdown Rewards Cannot Fire Without Steps (Medium)

**What**: Several high-value rewards (`landing_step_ahead=2.5`, `touchdown_root_straddle=0.75`, `feet_step_contact=4.0`) are gated on `first_contact` events that require a prior swing phase with `min_air_time >= 0.06s`. If the policy never initiates a swing, these rewards produce zero gradient.

**Evidence**: v14/v15 showed single-support ratio 0.024, meaning swing phases almost never occur, so these rewards never activate.

### Problem 5: Observation Space Lacks Gait-Relevant Information (Medium)

**What**: The 99-dim observation includes joint positions/velocities and base state, but does NOT include:
- Current contact state per foot
- Time since last contact switch
- Any gait phase indicator
- Foot positions relative to root

**Impact**: The policy has no way to plan foot placement relative to its current stance configuration. It must learn a purely reactive mapping from joint state to action, which is much harder for alternating gait than for shuffling.

### Problem 6: PPO Hyperparameter Interaction with Short Horizons (Low-Medium)

**What**: `num_steps_per_env=24` means the policy only sees ~0.48s of experience per rollout (at 50 Hz). A single walking step cycle takes ~0.6-1.0s for a humanoid.

**Impact**: The policy rarely sees a complete stance-swing-stance cycle in one rollout, making it harder to learn temporally extended behaviors like walking.

---

## Proposed Fix Plan

### Phase 1: URDF Physics Correction (Prerequisite)

**Goal**: Make the simulation dynamics realistic enough that standing in a wide stance actually costs energy and has stability consequences.

**Tasks**:

1. **Scale URDF link masses to a realistic total** (target: ~30-50 kg for a small humanoid, or match the intended robot scale)
   - File: URDF XML in [inputs/landau_v10/](algorithms/urdf_learn_wasd_walk/inputs/landau_v10/)
   - Scale all `<mass>` values proportionally
   - Keep CoM positions unchanged
   - Recompute `<inertia>` tensors to match new masses

2. **Re-tune actuator limits to match the new mass scale**
   - File: [landau_env_cfg.py](algorithms/urdf_learn_wasd_walk/landau_env_cfg.py) (lines 144-180)
   - Scale `effort_limit_sim` and `stiffness`/`damping` so joint authority is realistic
   - Rule of thumb: leg joints should be able to support ~1.5x body weight but not 100x

3. **Verify the robot can stand in its default pose under the new dynamics**
   - Run a headless zero-command test and verify root height stays near nominal
   - If it collapses, increase leg stiffness/damping

4. **Update `init_root_height` and nominal values** in [robot_specs.py](algorithms/urdf_learn_wasd_walk/robot_specs.py) if the default standing pose changes

**Risk**: This invalidates ALL existing checkpoints. A fresh training run is required. Accept this cost -- training on incorrect dynamics produces policies that don't transfer.

### Phase 2: Gait Phase Clock (Core Fix)

**Goal**: Give the policy an explicit periodic signal that drives alternating stance-swing cycles.

**Tasks**:

1. **Add a gait clock observation** to the observation space
   - Create a new observation term that provides `sin(phase)` and `cos(phase)` for a configurable gait period
   - Phase should be per-environment and wrap around at 2*pi
   - Each side (left/right) gets a 180-degree offset
   - Add to policy observation group in [landau_env_cfg.py](algorithms/urdf_learn_wasd_walk/landau_env_cfg.py)

2. **Add a phase-conditioned gait reward** in [custom_rewards.py](algorithms/urdf_learn_wasd_walk/custom_rewards.py)
   - When `phase_left < pi` (first half), reward left foot in contact AND right foot airborne
   - When `phase_left >= pi` (second half), reward right foot in contact AND left foot airborne
   - Use a soft Gaussian kernel around the phase boundaries, not a hard switch
   - This gives the optimizer a clear gradient for *when* to step, not just *whether* it stepped

3. **Make gait period a configurable parameter**
   - Start with period = 0.5s (typical walk frequency ~2 Hz)
   - Scale period inversely with commanded speed:
     - Slow walk (0.3 m/s): period ~0.6s
     - Fast walk (0.8 m/s): period ~0.4s
   - The policy should learn to adapt to the commanded rhythm

4. **Reduce or remove the `double_support_time_penalty`** since the phase clock will handle timing
   - The penalty currently fights against the policy's natural equilibrium without providing directional guidance
   - The clock reward is strictly more informative

**Implementation sketch** for the clock observation:
```python
# In a new observation term
def gait_phase_observation(env, gait_period: float) -> torch.Tensor:
    phase = (env.episode_length_buf * env.step_dt / gait_period) % 1.0
    phase_rad = phase * 2.0 * torch.pi
    return torch.stack([torch.sin(phase_rad), torch.cos(phase_rad)], dim=1)
```

### Phase 3: Foot-State Observations (Enables Stepping)

**Goal**: Let the policy know where its feet are and what contact state they're in.

**Tasks**:

1. **Add foot position observations relative to root**
   - Left foot (x, y, z) in root frame
   - Right foot (x, y, z) in root frame
   - 6 additional observation dims

2. **Add binary contact state per foot**
   - Left foot contact: 0.0 or 1.0
   - Right foot contact: 0.0 or 1.0
   - 2 additional observation dims

3. **Add time-since-contact-switch per foot**
   - Normalized to [0, 1] over a 1-second window
   - 2 additional observation dims

4. **Update observation dim**: 99 -> 111 (approximately)
   - File: [landau_env_cfg.py](algorithms/urdf_learn_wasd_walk/landau_env_cfg.py), observation policy group

**These observations are standard in state-of-the-art bipedal locomotion** (H1, Digit, Cassie papers all use foot state observations).

### Phase 4: Reward Stack Simplification (Reduce Conflicts)

**Goal**: Fewer, stronger, clearer reward signals that don't cancel each other out.

**Proposed simplified Stage A reward stack**:

| Reward | Weight | Purpose |
|--------|--------|---------|
| `track_lin_vel_xy_exp` | 4.0 | Forward tracking |
| `track_ang_vel_z_exp` | 1.5 | Yaw tracking |
| `phase_gait_reward` | **6.0** | **NEW**: alternating contacts matching clock |
| `control_root_height_floor` | -15.0 | Don't collapse |
| `feet_slide` | -0.08 | Don't slide feet |
| `non_support_contacts` | -1.5 | Anti-crawl |
| `non_support_contact_force` | -0.02 | Anti-crawl dense |
| `landing_step_ahead` | 2.0 | Forward stepping |
| `action_rate_l2` | -0.005 | Smooth actions |
| `flat_orientation_l2` | -1.0 | Stay upright |
| `termination_penalty` | -200.0 | Don't die |

**Remove or disable** for the initial training:
- `stance_width_excess` -- let the clock and physics naturally narrow the stance
- `touchdown_step_length_deficit` -- redundant with clock-driven stepping
- `touchdown_support_width_excess` -- same
- `touchdown_root_straddle` -- secondary shaping, add back later
- `swing_foot_ahead_of_stance` -- conflicts with `landing_step_ahead`
- `single_support_root_straddle` -- secondary
- `primary_foot_flat_contact` -- secondary, add back in a refinement phase
- `aux_support_*` -- secondary, add back later

**Rationale**: The current 25+ reward terms create a complex optimization landscape with many saddle points. Start with ~10 terms that have clear non-conflicting gradients, then add refinement terms once walking is established.

### Phase 5: Training Protocol (Execute)

**Goal**: Structured training that doesn't repeat the diminishing-returns pattern.

**Step 1: Fresh run with Phase 1-4 changes**
```bash
/home/wishai/vscode/IsaacLab/isaaclab.sh -p -m algorithms.urdf_learn_wasd_walk.scripts.train \
  --robot landau --stage fwd_only --headless \
  --max_iterations 3000 \
  --run_name phase_clock_v1
```

**Step 2: Validate at regular intervals**
- Check every 200 iterations (not just PPO reward)
- Primary success metric: `single_support_ratio > 0.3` AND `forward_displacement > 0.3m`
- If single-support ratio stays below 0.1 by iteration 500, something is still wrong

**Step 3: If Phase clock works but gait quality is rough, add refinement rewards**
- Resume from the best clock-trained checkpoint
- Re-enable: `stance_width_excess`, `primary_foot_flat_contact`, `swing_height_difference_floor`
- Use smaller weights than before (these are refinement, not primary signals)

**Step 4: Increase PPO horizon**
- Change `num_steps_per_env` from 24 to 48
- This gives the policy a full gait cycle in each rollout
- Watch for memory -- may need to reduce `num_envs` from 1024 to 512

---

## Alternative / Additional Approaches

These are options if the phase clock approach alone is insufficient:

### A. Motion Prior / Imitation Learning
- Capture reference walk trajectories from MoCap or keyframe animation
- Add an imitation reward: penalize deviation from reference joint angles at each phase
- Highest probability of producing natural-looking gait
- Requires obtaining or creating reference motions for the Landau skeleton

### B. Capture Point / ZMP Reward
- Compute the instantaneous capture point from CoM velocity and height
- Reward foot placements that contain the capture point within the support polygon
- This is the "viability / capture-point style touchdown shaping" mentioned in the training history
- More robust than geometric step-length rewards because it accounts for dynamics

### C. Curriculum from Simplicity
- Start with a 2D sagittal-plane walking task (lock lateral motion)
- Transfer the sagittal walk policy to 3D
- This reduces the initial search space significantly

### D. Network Architecture
- The current `[512, 256, 128]` MLP may be too small to represent both gait timing and posture control
- Consider: `[512, 256, 256, 128]` or adding a small recurrent layer (GRU with 64 units) for temporal pattern memory
- The recurrent layer would give the policy implicit phase tracking even without an explicit clock

---

## Implementation Priority

| Priority | Task | Effort | Impact |
|----------|------|--------|--------|
| 1 | Phase 2: Gait phase clock | Medium | Critical - breaks the no-step equilibrium |
| 2 | Phase 1: URDF mass fix | Medium | Critical - makes physics realistic |
| 3 | Phase 4: Reward simplification | Low | High - removes conflicting gradients |
| 4 | Phase 3: Foot-state observations | Low | High - enables informed stepping |
| 5 | Phase 5: Training protocol | Low | Medium - proper execution |
| 6 | Alt D: Network size or RNN | Low | Medium - may help temporal learning |
| 7 | Alt A: Motion prior | High | Very high - but needs reference data |
| 8 | Alt B: Capture point reward | Medium | High - physics-grounded foot placement |

---

## Files That Need Changes

| File | Changes |
|------|---------|
| [custom_rewards.py](algorithms/urdf_learn_wasd_walk/custom_rewards.py) | Add `gait_phase_reward`, `gait_phase_observation` |
| [landau_env_cfg.py](algorithms/urdf_learn_wasd_walk/landau_env_cfg.py) | Rewire reward weights, add clock obs, add foot-state obs, tune actuators |
| [robot_specs.py](algorithms/urdf_learn_wasd_walk/robot_specs.py) | Update nominal values after mass fix |
| URDF file in `inputs/landau_v10/` | Scale link masses and inertias |
| [validate_walk.py](algorithms/urdf_learn_wasd_walk/scripts/validate_walk.py) | No changes needed initially |
| [isaac_workflow.py](algorithms/urdf_learn_wasd_walk/isaac_workflow.py) | Minor: support `num_steps_per_env` override if needed |

---

## Success Criteria

A checkpoint passes if `validate_walk.py --robot landau --stage fwd_only --headless` reports:

- forward displacement >= 0.4 m over 500 steps
- single-support ratio >= 0.3 (currently 0.024 -- needs 12x improvement)
- double-support ratio <= 0.5 (currently 0.968)
- flight ratio <= 0.05
- mean support width <= nominal * 1.3
- min control-root height >= 0.17
- done_count <= 2
- non_support_contact_step_sum <= 10
- Qualitatively: visible alternating left-right stepping, not a shuffle

## Key Lessons from Training History

1. **Don't resume from false-positive checkpoints** -- model_500.pt and model_3050.pt were both invalidated
2. **PPO reward alone is not a reliable signal** -- always run `validate_walk.py`
3. **Scalar reward tuning has diminishing returns** -- v5 through v15 each added incremental improvements but none broke through
4. **The 1.63 kg mass has been known since day one but never fixed** -- this should be addressed
5. **`--max_iterations` during resume means "train for N more", not "train until iteration N"**
6. **Fingertip phalanges must be excluded from gait guard** (no control authority over finger joints)
