# URDF Learn WASD Walk — Clean Room

This sandbox is being rebuilt from the clean restart contract one milestone at a time.

Current milestone truth (2026-09-22): **6/12 milestones certified; training paused
at the user's request.** No GPU worker is running. Resume only when requested,
starting with M7: the four-direction 10 m gates. M7 has no implementation or run yet.

The custom `balanced_hands_v1` Landau checkpoint passed the complete cumulative set:

| Gate | Exact M6 checkpoint result |
| --- | --- |
| M1 passive standing | 30 s, 3.484 mm drift |
| M2 policy standing | 30 s, 4.037 mm drift |
| M3 forward 5 m | 5.835 m in 45 s |
| M4 forward 10 m | 10.516 m in 80 s |
| M5 turn and hold | 91.378°, 1.378° hold error, 18.45 mm drift |
| M6 forward/turn commands | 60 s, left +27.0°, right −21.4°, successful stop/restart |

All six runs have zero falls/resets/done events and reviewed proof videos. M6 is a
scripted joystick replay, not a live human teleoperation claim. Its final hold drift
is 18.46 mm. The 10 m run's peak force is 2.967 body weights, close to the unchanged
3 BW limit; these results certify the recorded nominal runs, not broad robustness.

Resume checkpoint (relative to this algorithm folder):
`outputs/tk2-backend-20260918/source/algorithms/urdf_learn_wasd_walk/outputs/mujoco/training/rsl_transfer_20260922_teleop_preserve_left/model_1.pt`

SHA-256: `0c07a031b1921995f5c17e4a14f9a7dfee3d6854d376da4f2b0c205f2c63fb13`.
Its certificate is under the same runtime's `outputs/mujoco/` in
`rsl_transfer_20260922_teleop_preserve1_m6/milestone_validation.json`.
All cumulative folders share the prefix `rsl_transfer_20260922_teleop_preserve1_`
with suffixes `m1`, `m2`, `gate5m45`, `gate10m80`, `m5`, and `m6`.
Generated checkpoints, videos, and traces remain in ignored local outputs; this
Git commit records code, status and evidence hashes, not those binary artifacts.

The successful correction freezes the seven certified left-turn parameters and
learns separate right-turn/hold and restart parameters. An earlier M6 candidate
passed the command test but failed M5 while settling; it remains rejected history.
Mass, geometry, joint gains, physical guards and cumulative requirements stayed
fixed. Preserve the saved controller sources with the checkpoint when resuming.
G1 verification is complete and is not the next training target.

Historical implementation notes below describe earlier states; `milestones.json`
and its hash-bound validation artifacts determine current status.

Kept:

- `TRAINING_RULES.md`: human-readable cumulative acceptance ladder
- `milestones.json`: machine-readable copy of the same twelve milestones
- `inputs/landau_v10/`: the robot input needed to rebuild the environment
- `gui/manifest.json`: milestone status and browser URDF viewer contract

Still intentionally absent:

- all removed pre-restart environment, policy, reward, teleop, play, validation, and curriculum logic
- old tests and agent configurations tied to that implementation
- run history, checkpoint lineage, restart notes, and problem investigations

Current clean implementation:

- `model_spec.py` audits the exact retained URDF, inertia, collision package, root transform, limits, and zero-pose joint axes.
- The 68 STL files under `inputs/landau_v10/mesh_collision_stl/` are versioned source assets because the URDF uses them for both visual and collision geometry. The audit fails closed unless their tree hash matches the repository `usd_parallel_urdf` package (8,864 triangles, including the rabbit-ear head silhouette).
- `robot_spec.json` records the 17 action joints, every explicitly locked joint, nominal pose, PD gains, and Landau's body-`+Y` semantic command mapping.
- `passive_stand.py` implements only milestone 1 as two independent Isaac components: camera-free passive dynamics and a viewport-rendered proof replay.
- `passive_pipeline.py` runs those components sequentially and creates final milestone evidence only when both pass.
- The repository 68-mesh visual/collision package is being re-certified from milestone 1.
  Earlier stand checkpoints and evidence live only under invalidated mesh-tree branches and
  cannot be resumed or promoted.
- `policy_stand_env.py` is the milestone-2 flat manager-based environment, with the audited 17-joint residual action and a 60-value proprioceptive actor observation.
- `policy_stand.py` trains RSL-RL PPO or evaluates one checkpoint, and `policy_stand_pipeline.py` keeps camera-free dynamics separate from viewport proof.

Commands:

- `./geo walk milestones`
- `./geo walk inspect`
- `./geo walk test -v`
- `./geo walk validate-passive-dynamics --steps 32 --smoke --reuse-usd-cache --headless`
- `./geo walk validate-passive-dynamics --steps 1500 --smoke --reuse-usd-cache --headless` (bounded 3 s stability diagnostic)
- `./geo walk validate-passive-dynamics --headless`
- `./geo walk render-passive-proof --headless`
- `./geo walk finalize-passive`
- `./geo walk validate-passive --headless`
- `./geo walk train-policy-stand --headless --num-envs 512 --iterations 200`
- `./geo walk validate-policy-stand-dynamics --steps 500 --smoke --reuse-usd-cache --headless`
- `./geo walk validate-policy-stand --headless`
- `./geo walk finalize-policy-stand`
- `./geo walk train-forward-walk --headless --num-envs 512 --iterations 600`
- `./geo walk train-forward-walk --resume-gate-checkpoint --headless --num-envs 512 --iterations 300` resumes a failed 5 m candidate into the bounded phase-gait/L2 method; the prior canonical checkpoint remains intact until completion.
- `./geo walk validate-forward-walk-stand --steps 32 --smoke --reuse-usd-cache --headless`
- `./geo walk validate-forward-walk-dynamics --steps 32 --smoke --reuse-usd-cache --headless`
- `./geo walk validate-forward-walk --headless`
- `./geo walk finalize-forward-walk`

Each exact all-in-one validator runs its independent Isaac components sequentially; it never overlaps Isaac processes. Camera-free results remain available when proof rendering fails. A final `validation.json` appears only after every exact component passes. Policy training writes a durable `checkpoint.pt`, its originating run checkpoint, and `training.json` below the active milestone output; training completion alone cannot promote a milestone. Short smokes are never promotable.

The bounded `phase_gait_l2_v2` experiment belongs to the invalidated stale-mesh branch. Its
checkpoint and diagnostics are retained for audit visibility only and are not gate candidates.

No walking-policy hypothesis is active while rabbit-ear-mesh passive and policy standing remain
unresolved. The first current-lineage task is the exact 30-second passive zero-signal gate.

Use the Geo Web GUI to launch TK2 commands, inspect the always-visible checkpoint lineage, and preview declared JSON, MP4, and contact-sheet artifacts. Policy transfer and the 5 m gate remain blocked until the exact rabbit-ear asset passes both standing gates again.


## Continuation and evidence review (2026-09-22)

The current MuJoCo development runtime and artifacts are preserved under
`outputs/tk2-backend-20260918/source/algorithms/urdf_learn_wasd_walk/`.
Its `outputs/backend_progress.json` is separate from the canonical Isaac milestone
ledger. The September 18 student demonstrated only about 0.223 m in 120 s with assistance
off; it did not pass a 5 m gate. On September 22 its exact zero-command check still moved
0.063 m in 30 s, so it cannot be treated as a standing-capable walking policy.

`continuation.py` is the reviewable adapter for bounded experiments in that
preserved runtime. Copy the adapter into that algorithm folder and submit it
through its existing isolated GPU worker; dependencies live there. It supports
student diagnostics, passive standing, PPO standing, checkpoint evaluation and
sequential proof rendering. Use a fresh name for every run. It changes no
canonical milestone, input URDF, mesh, joint limit or gain.

The `balanced_hands_v1` model variant moves 25% of each of the 38 finger-link masses
(0.190 kg total) into `root_x` (40%) and the three spine links (20% each), maintaining
total mass 1.829753 kg and bilateral symmetry. Every affected inertia tensor is
scaled by the same positive mass factor, with its origin and rotation retained.
This is a sensitivity experiment, not a claim of anatomically measured mass:
the source assigns 48.1% of total mass to hands, and all inertial origins are zero.
The changed model requires fresh standing and walking evidence. Its first passive
30 s diagnostic had 3.5 mm drift, no falls/resets, and positive 27.97 mm support margin.
Fresh 100-iteration PPO and a 200-iteration lower-learning-rate comparison both
failed exact standing evaluation; their checkpoints remain unpromoted and their
failure videos are retained.

The main approaches, in priority order, are: bounded mass-distribution comparisons;
faster gait-period/step-length curriculum; contact-based balance feedback; residual
PPO around a stable gait; retargeted motion imitation; phase/contact PPO from
scratch; whole-body/MPC teacher; trajectory optimization; physical randomization
after nominal success; and separately measured GPU throughput improvements.
More imitation of the same 22.5-second gait cycle cannot establish fast walking.
Motion-imitation reference: https://xbpeng.github.io/projects/DeepMimic/index.html

`g1_fresh_control.py` runs the pinned official Unitree G1 PPO recipe from fresh
weights and separately evaluates an exact checkpoint with a fixed 0.5 m/s command,
retained observation normalization/action clipping, explicit nominal-evaluation
overrides, and immediate failure on any done/reset. The old successful G1 video
uses bundled pretrained ONNX weights. Earlier fresh 2/10-iteration runs established
throughput only. These controls must remain distinct in every report.
Official recipe: https://github.com/unitreerobotics/unitree_rl_mjlab/tree/1425b15f73bd4095f0df53709d7c389c3eb9e790

The fresh September 22 run completed 1,000 iterations × 1,024 environments ×
24 rollout steps (24.576 million transitions) in 482.78 s on an RTX 4080 SUPER.
Steady throughput including PPO was 52,700 transitions/s; sampled total GPU
memory peaked at 844 MiB, average GPU utilization was 72.3%, and process-tree
RSS peaked at 2.65 GiB. Only compiled simulation kernels were reused. A preceding
cold-cache attempt crashed during compilation before any training iterations.
The earliest saved checkpoint (`model_0`, after one update) fell at 1.38 s.
`model_999` stayed upright for 30 s without reset but advanced only 0.167 m under
a 0.5 m/s command; one foot never lifted. This first 1,000-iteration checkpoint had not learned walking; later continuation results follow below.
Both evaluations retain same-rollout videos, trajectories, hashes and metrics.
See the runtime's `outputs/backend/g1_fresh_20260922_warm1000/control_summary.json`.

The official README recommends 4,096 environments, and its recipe defaults to
10,001 iterations, 24 rollout steps, 50 Hz control, 200 Hz physics, a 0.6 s gait
period, normalized observations, and a 512/256/128 ELU actor/critic. PPO uses five
epochs, four minibatches, adaptive learning rate starting at 0.001, clip 0.2,
gamma 0.99, lambda 0.95 and entropy coefficient 0.01, with command curriculum,
physical randomization and pushes. Our bounded run used only 2.5% of that
recommended sample budget. No verified original training wall time or hardware
provenance for the bundled checkpoint was found; our timings are our measurements.
This tests the official G1 implementation, not the custom Landau trainer.
The five new checkpoints are backed up under Nextcloud
`Projects/geo_lib/checkpoints/urdf_learn_wasd_walk/g1_fresh_20260922_warm1000`
and declared in root `large_files.json`. Remote SHA-256 hashes were verified;
local runtime copies are retained.

Continuing the same fresh lineage for another 2,000 iterations at 1,024 environments,
then 500 at 4,096 environments, produced `model_3497`. Its exact 30-second
nominal evaluation with the official **heading-hold command generator** passed:
12.077 m forward, 1.690 m lateral, no done/reset, 49/50 completed swings with
at least 15 mm clearance and 60 ms air time, and about 0.022 m/s contact slip RMS.
The fixed zero-yaw evaluation of the same checkpoint still turned excessively
(6.116 m forward, 8.328 m lateral), so these command modes are not interchangeable.
The heading-hold reference is recorded in the runtime's
`outputs/backend/g1_fresh_20260922_reference.json`, with exact checkpoint and
same-rollout video/trajectory hashes. Sampled visual review covered the full
30 seconds at 1 fps and a three-second gait segment at 10 fps.

Cumulative training to this reference took 32.26 minutes and 122.88 million
transitions. The 4,096-environment stage achieved 97,118 transitions/s including
PPO and peaked at 2,544 MiB sampled GPU memory. This establishes fresh G1
learning in our environment; it does not certify disturbance robustness.

A further 500 iterations at 4,096 environments changed only the angular-velocity
reward width from sqrt(0.5) to 0.2. `model_3996` then passed the **fixed zero-yaw**
30-second evaluation: 11.848 m forward, 1.270 m lateral, 49/50 completed swings,
no done/reset and 0.104 m/s body-forward velocity RMSE. Independent native CPU
MuJoCo deployment playback also passed: 11.392 m forward, -1.943 m lateral,
50/49 completed swings and no fall. Total training to this checkpoint was
41.04 minutes and 172.032 million transitions; evaluation time is additional.
Both `model_3497` and `model_3996` have SHA-256-verified Nextcloud backups declared
in `large_files.json`. The earlier heading-hold checkpoint additionally passed
startup-randomized seeds 43 and 44. The new fixed-yaw checkpoint stayed upright
but failed lateral-drift acceptance on those seeds (5.020 m and -3.402 m).
Heading feedback is therefore retained for direction control; these checks are
not a robustness certification.

`landau_rsl_control.py` transfers the PPO library, saved observation normalization,
reward scaling by control dt, and explicit local drift/remaining-time observations
to the bounded Landau standing task. It retains the `balanced_hands_v1` geometry,
gains and action bounds. Its saved initial checkpoint is an **untrained interface
baseline**, not learned standing. Every trained checkpoint requires the existing
independent 30-second physical validator and matching video before claiming progress.

The first transfer run used 256 environments for 500 iterations (3.072 million
transitions) in 299.28 seconds. Its final policy remained upright under independent
30-second evaluation with no reset, no fall, no non-foot ground contact, and
32.12 mm maximum drift. This passes the existing dynamics checks, but misses
the training objective's stricter 30 mm drift target. The untrained baseline and
two-update smoke policy drifted 3.43 and 3.45 mm respectively: longer training
has not demonstrated an improvement over the underlying passive pose.
Training action standard deviation increased from 0.10 to about 0.30; reducing
exploration/entropy is a candidate comparison, not an established fix. The earlier
`model_100` checkpoint passed the same independent 30-second check with 4.04 mm
drift, no fall/reset, and no auxiliary forces. Retain this checkpoint rather than
the last one for the next standing comparison. This demonstrates preserved
nominal standing, not learned disturbance recovery or walking; canonical Isaac
milestones remain unchanged. Recorded yaw changed only 0.10 degrees for
`model_100`; the final checkpoint turned 32.09 degrees without a command. Its
video review rejects it as a standing reference despite passing the existing
dynamics checks. Absolute heading error is not yet part of the transferred
standing observation/reward, so heading retention needs an explicit comparison.

Refresh the review data with `./geo walk evolution`, then open `./geo gui`.
The walking dashboard puts the latest observed result and inline video first;
current ancestry is visible by default, while old experiments, Isaac launch tools,
raw evidence, model metadata and the full ladder are expandable. The tree imports
current runtime results and identity-checked proof media. Missing videos are
explicitly marked; no video is fabricated or borrowed from another asset lineage.
Run-specific previews are restricted to JSON/image/video files under this
algorithm's outputs, and checkpoints remain metadata-only.

### Landau continuation, 2026-09-22

The current MuJoCo lineage has passed both standing gates and the 5 m gate.
The exact checkpoint `rsl_transfer_20260922_matched_distance1024/model_9.pt`
reached 6.494 m in 45 seconds in `rsl_transfer_20260922_matched9_gate45`,
with 47 left/48 right qualifying swings and no fall/reset/done. Maximum joint
speed was 3.299 rad/s and peak support 2.858 body weights. Fresh passive and
zero-command policy standing rechecks passed. The certificate binds full traces,
video, model, checkpoint and source hashes. The 10 m gate is now in progress. Its first evaluation fell after 6.369 m; training now uses 80-second episodes.
Consult `milestones.json` for current status; older stepping references remain history.

This moving controller uses CEM-trained periodic gait and proprioceptive feedback
parameters, with the trained standing MLP frozen at zero command. Earlier positive
stride controllers mostly flexed the stance knee. Signed negative stride and a
different lateral phase corrected the swing-knee coordination; independent contact
traces verify the change. No further mass, geometry, joint-gain or action-limit
change was used. Increasing lateral feedback alone worsened this new gait.

Full-duration refinement retains four starting states per candidate and checks
all 69 joint speeds, actual ground forces, and non-foot contacts every 2 ms.
The trainer now matches the validator's post-step physics refresh cadence.
Fixed-state CPU/GPU observations agree within 3e-7, but numerical rollout
differences remain, so batched success never replaces independent evaluation.
The current limiting mechanism is recurrent touchdown impact, including body
descent and rocking. Force timing is retained at 2 ms in subsequent evaluations.

The 4.046 m reference's exact 2 ms trace shows the tightest force margin at early
right touchdown (2.858 body weights at 1.962 s); the maximum after 3 s is 2.689.
Its 31 swings per foot clear 37.5–41.8 mm. An old training-only rule discarded an
entire swing after any opposite-foot contact gap. Applied to the same recorded
motion, that rule counted only 14/24 swings at 50 Hz, introducing phase aliasing.
The corrected trainer uses the validator's swing definition and explicitly
measures continuous flight and total flight fraction every 2 ms. The existing
0.12 s / 5% flight limits remain unchanged. A full-duration GPU smoke test of the
same checkpoint now counts 31/31 swings across all four starting states.

Matched batch benchmarks measured 25% more candidate trials per second at 1,024
worlds than 512, and 39% more at 2,048. Current searches use 1,024 worlds to retain
shorter generation latency. Once all walking constraints hold, candidate fitness
prioritizes worst-start forward distance; extra clearance only breaks near ties.
The optional 5.15 m training stop triggers independent validation, not promotion.

The preserved 5 m walking contract has no deadline; the 30 s requirement applies to the two standing gates. On 2026-09-22, the evaluator-added 30 s walking rejection was removed. Walking certificates now verify every physics sample, control step, and video frame against the declared complete duration (currently 30–120 s). Physical, gait, visual, and cumulative checkpoint checks remain required. Earlier failed runs are retained unchanged; a fresh evaluation is required.
