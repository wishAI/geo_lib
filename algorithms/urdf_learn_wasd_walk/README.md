# URDF Learn WASD Walk — Clean Room

This sandbox is being rebuilt from the clean restart contract one milestone at a time.

Current milestone truth (2026-10-03): **6/12 milestones certified; work resumed
on M7.** Four directions mean turning and walking toward world-direction 10 m
gates from the same nominal pose, as confirmed by the user. The M7 diagnostic
emits only semantic forward/yaw commands; it preserves the saved M6 controller.
Each direction run keeps one yaw sign (right for the right gate, left for the
left/backward gates) to avoid repeatedly switching balance modes near alignment.
Its bounded horizon is 240 s because the existing trained turn rate needs 88 s
for 180 degrees. Standing, contact, speed, flight and no-reset limits are unchanged.
Speed improvements require measured faster completion and the same quality checks.

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

M7 continuation uses `landau_forward_control --mode evaluate --direction right
--target-distance 10 --seconds 180 --forward .2` (or `forward`, `left`, `backward`).
Run through the preserved runtime's pinned GPU Python, with an explicit checkpoint
and fresh `--name`. Copy the current evaluator, controller adapter and
`landau_direction_contract.py` into that runtime before launching; never change
runtime sources during an active evaluation. A directional episode ends on the
first valid gate crossing at a 20 ms control boundary; the original horizon and
actual crossing time remain in its evidence. The validator reconstructs commands
and crossing from the full saved trajectory, rejects a rotated start pose, and
keeps the 2 ms contact/force checks and proof-video requirement.

The bounded M7 trainer adds `--direction-train --teleop --right-only` to
`landau_turn_control`; it freezes all existing parameters except right yaw scale,
right heading feedback, and two new right-specific lateral sway parameters.
The original 12-parameter controller retains identical inference. Training
qualification never promotes a milestone. `./geo walk mujoco-milestone
certify-directions` requires four distinct `--direction-directory` arguments,
`--checkpoint`, all six prior component directories (including
`--teleop-directory`), and a new output `--directory` for the certificate.

The first M7 candidate (`m7_20261003_right_roll_cem/model_3.pt`) cleared the
forward gate in 76.20 s and right gate in 97.20 s, and repeated M6 with 10.24 mm
final hold drift. It is **not promoted**: its left run stayed upright for 180 s
but curved away from the gate after the yaw command stopped. The follow-up
`--direction-train --teleop --left-cruise-only` freezes those 14 parameters and
searches three additional heading/sway parameters. They blend in only after
0.2 s of uninterrupted full-forward, zero-yaw cruising following a left turn.
The certified M5/M6 command schedules never activate this branch; regression
tests check that their actions retain the original parameters.
The one-parameter heading-only grid failed the force limit and is rejected.
Fresh four-direction and cumulative evidence is required for any new candidate.

The fixed-turn cruising candidate qualified over 90 s in four training starts,
but failed the independently steered left gate at 79.91 s / 7.07 m. Pursuit used
117.87° of commanded rotation and intermittent tiny yaw inputs; fixed training
used 90° followed by uninterrupted straight walking. The next refinement bounds
cruise-blend changes in both directions and accepts `--direction-command-trace`
for training. It snapshots only the recorded semantic commands with hashes,
checks their clock/ranges, and extends final straight walking by at least 20 s.
It never replays poses or actions. Candidate 0 retains the unchanged parameters
under the new transition behavior for comparison; actual gate evaluation still
recomputes commands from its own poses and remains required before promotion.

The pulsed candidate then survived 180 s but missed the gate (10 m crossing at
107.86 s, 0.871 m lateral error) and exceeded 3 BW before cruising started.
Repeating the unchanged 14-parameter controller also reached 3.042 BW. A CPU
replay audit found bit-identical pre-cruise actions on identical observations;
the saved physical traces already differ at their first 2 ms step. The specific
numerical operation remains unidentified; upstream tracks deterministic execution
in [MuJoCo Warp issue 562](https://github.com/google-deepmind/mujoco_warp/issues/562).
The ignored `outputs/m7_20261003_repeatability_audit/` contains the repeatable audit.
`--left-turn-refine` therefore also searches the existing left heading/sway
parameters, while freezing walking/standing weights and the other 11 parameters.
It prefers a 2.85 BW post-onset training margin, retaining the full-run 3 BW hard
limit. Any resulting checkpoint must re-pass M5 as well as every other component.
The final `m7_20261003_left_margin_cem/model_2.pt` training candidate completed
110 s in all four starts, with peak 2.821 BW and heading errors below 10.5°.
Its exact M5 recheck remained upright with peak 2.889 BW and 19.73 mm hold
drift, but held only 72.16° and failed heading. `--left-yaw-only` calibrates an
independent left yaw scale on the exact M5 command sequence while freezing all
17 balance parameters and walking/standing weights. It remains unpromoted;
passing training still requires fresh independent and cumulative evaluations.
The selected `m7_20261003_left_yaw_grid/model_0.pt` preserved those 17 parameters
bit-for-bit and independently re-passed M5 at 93.96° with 13.73 mm hold drift and
a reviewed full proof. Its left gate stayed upright for 180 s with peak 2.885 BW,
but crossed the 10 m plane at 108.62 s with −0.927 m cross-track error, outside ±0.75 m.
Thus M7 remains unresolved despite the improved physical stability. A same-checkpoint
trial reducing proportional yaw gain from 0.3 to 0.1 also missed the gate and
reached 3.040 BW; that command change is rejected and reverted. The force peak
at 43.474 s precedes the first changed command at 45.92 s, so it is not evidence
that the gentler taper caused the force spike. The next training
problem is prolonged left-cruise heading control with force margin, using the
actual steering traces. Long proof renders now scale both subprocess timeouts
with recorded duration.

A gain-only diagnostic (`--left-heading-grid`, 8 candidates × 4 starts, 180 s)
preserves all 18 parameters except left-cruise heading feedback. The incumbent
combined coefficient is +0.00653 versus −0.00338 while turning; its sign change
is a hypothesis, not proof of closed-loop instability. Lower gains reduced
observed heading drift. The selected delta 0.02 survived all four starts with
peak 2.989 BW, but final errors of 17.8–27.3° missed the 15° training target.
The independent closed-loop left gate then passed at 104.005 s, with −0.113 m
cross-track error and peak 2.891 BW. Other directions and fresh cumulative
checks remain required; it is not promoted. The backward trial fell at 144.058 s.
Its command trace exposed a near-180° ambiguity: initial negative body heading
made the shortest arc rightward, which the left-only adapter clamped to zero.
Yaw started only at 17.4 s and pulsed before sustained turning. Backward steering
now chooses the intended left arc for large negative angle errors while leaving
small overshoot at zero. Fresh four-direction evidence must use this protocol hash. The corrected
backward trial reached its gate at 174.675 s (−0.203 m cross-track) without a
fall, but one 2 ms sample at 48.834 s reached 3.031 BW and fails the hard limit.
`--left-balance-only` restricts force-margin refinement to the two active-turn
sway parameters, preserving yaw calibration and cruising feedback. Training
response blocks accumulate local heading increments so turns beyond 180°
retain their direction. The bounded two-generation, 256-world refinement
finished with zero qualifiers: the final selected candidate survived four
180 s starts with peak 2.975 BW, but missed the 2.85 BW training margin and had
30.39–36.47° final heading errors. It is rejected for promotion. The certified
M6 checkpoint remains the default. The earlier successful left proof predates the backward
protocol fix, so a fresh left proof is also required before certification.

The next refinement uses `--closed-loop-direction backward`: each training
world generates the same joystick commands as the serial gate evaluator from
its own position and heading. The objective is a valid gate crossing, force
margin and completion time, rather than matching a replay's final heading.
Completed and failed worlds freeze their metrics before explicit training
resets; neither can re-enter the search episode. `landau_direction_training.py`
keeps this batch implementation separate from the acceptance protocol.
Tests compare its commands, memory and actions with independent serial
instances, including asynchronous cruise transitions. The 32-world smoke
completed, and the bounded search retains the same two sway parameters and
unchanged walking speed. Training results remain candidates requiring fresh
serial M1–M7 proofs. Per-environment command generation also follows the
separation described in the [Isaac Lab command manager documentation](https://isaac-sim.github.io/IsaacLab/develop/source/api/lab/isaaclab.managers.html).

The two-generation closed-loop run found one first-generation candidate with
four backward crossings in 176.56–182.02 s and a 2.988 BW worst peak. It did not
reach the extra 2.85 BW training margin. The identical candidate repeated in
generation two crossed only two starts; two perturbed starts exceeded 3 BW.
This is not evidence of robust stability. Its saved `model_0.pt` independently
passed fresh M5 and M6 proofs, including complete visual reviews. M5 heading
error was 2.227°, drift 17.98 mm and peak force 2.9615 BW; M6 peak force was
2.96498 BW. Its independent backward run reached the gate at 177.553 s with
−0.711 m cross-track but failed with a 3.13417 BW peak at 18.460 s and another
3.10066 BW event at 27.396 s. It remains unpromoted; no speed increase is claimed.

The next comparison uses `--common-starts --left-balance-grid`: each candidate
gets the same nominal start and three bounded joint perturbations, and each
generation repeats the same sway grid. Candidates 0 and 1 are unchanged
controls. This applies [common-random-number comparison](https://pubsonline.informs.org/doi/10.1287/mnsc.45.11.1570)
without claiming deterministic GPU physics. Each generation's candidate table
is saved and hash-bound; explicit row selection uses the checkpoint's matching
generation. New walking foot traces also record per-foot force, contact
velocity and position, root vertical velocity, and hip-roll target/position
at each 2 ms physics step to diagnose load transfer. These are read-only
measurements; controller actions and acceptance limits are unchanged.

The repeated sway grid produced no qualifiers in either generation. Only
candidate 55 survived all eight starts across both repetitions, with a worst
2.9941 BW peak, and none of those starts reached the gate. A separate unchanged
60 s instrumented probe peaked at 2.8253 BW: about 48.6 N came from the landing
left foot and 2.1 N from the other foot. It was a load-transfer diagnostic,
not a successful 10 m run.

The next optional extension adds two bounded torso roll/rate feedback gains
while freezing the previous 18 parameters. Its correction is limited to
0.03 rad at each hip and fades in/out at a command-envelope rate of 1/s.
`--left-feedback-only --left-damping-grid --common-starts` first compares six
rate-gain offsets with two unchanged controls; proportional correction stays
zero in that grid. The [Digit feedback study](https://arxiv.org/abs/2103.15309)
and [residual control study](https://arxiv.org/abs/1812.03201) motivate this
bounded feedback experiment, not its numerical gains. It can alter the
pre-landing state at 50 Hz; it cannot cancel an impact developing over 4–8 ms.
Zero-extension action equivalence and scalar/batched envelope behavior are
tested. No acceptance bound or policy speed is relaxed.

The repeated damping-only grid had no qualifying backward crossings. The
−0.01 rate offset kept all eight starts within hard physical limits for over
200 s, but none reached the gate. An exact M5 screening smoke then showed
underrotation despite lower forces. `--turn-hold-training --feedback-yaw-refine`
therefore searches the existing left-yaw scale together with the two feedback
gains while preserving the first 17 parameters.

The initial screen exposed a conflicting objective: rewarding a 90° change
from turn onset favors a final heading near 70° because pre-turn and stopping
drift total about 20°. The corrected M5 screen uses absolute final/settled
heading error and hold drift, retaining force penalties and every eligibility
condition. It saves maximum hold error for every candidate. Screening remains
approximate and cannot replace serial M5 force, drift, speed, slip and video
validation. The corrected search starts from a hash-bound near-target row of
the rejected screen rather than silently treating its winner as a pass.

The corrected four-generation screen selected `model_3.pt` with a worst-start
force peak of 2.821 BW and settled heading error of 4.425°. Independent M5
evaluation passed dynamics, full-video review and the separate validator:
2.851 BW, 1.953° hold error and 17.4 mm drift. M6 dynamics also passed at
2.986 BW. These are candidate component checks; M7 remains unresolved and
the certified checkpoint remains unchanged. Earlier training winners varied
on repeat, so the screening result is not a robustness claim.

That candidate reached the backward gate at 173.032 s with −0.293 m
cross-track error and no falls/resets, but failed one 2 ms force sample at
132.802 s: 3.690 BW, versus 2.878 BW for the next highest sample. The peak
occurred during landing just after a hip-roll target update. The next
`--left-cruise-balance-only` search changes only indices 15/16 (post-left-turn
sway amplitude/phase), preserving every other seed parameter bit-exactly.
It uses actual per-world backward-gate steering and shared starts, retains
the 2.85 BW training target, and does not change acceptance or walking speed.

The two-generation cruise search selected a changed candidate with four
training crossings and a 2.958 BW peak, but no candidate met the extra
training margin. Its independent backward run reached the gate at 175.580 s
and failed two late force samples (3.029 and 3.011 BW). A bounded optional
21st parameter now offsets hip-roll angular-rate feedback during left cruise
only. `--left-cruise-rate-grid` compares six offsets with two unchanged
controls while retaining the first 20 parameters exactly. Positive offsets
reduce the existing damping magnitude; the correction is capped at 0.05 rad.
This is an unvalidated diagnostic, not a force-reduction or milestone claim.

The repeated cruise-rate grid did not improve reliability: no nonzero offset
cleared all four starts in either repeat, and neither generation met the
extra force margin. The retained 20-parameter cruise candidate is next
tested with forward command 0.18 instead of 0.20, retaining yaw, physics and
all acceptance bounds. Command magnitude is not achieved walking speed;
gate time and walking quality must be measured independently.

The 0.18 run fell at 94.662 s. It was not an isolated speed reduction:
the saved controller enables cruise corrections only at exactly 0.20 and
mixes standing actions at lower commands. Return to 0.20 for diagnosis.
The newer failed force peaks occurred before a target update; unlike the
earlier isolated spike, they do not support a general rate-kick explanation.
Their landing speeds were also below the late-run median. The next repeated
grid therefore broadens existing cruise sway to ±0.018 rad amplitude and
±0.15 rad phase (about ±21.3 ms), using direct force and gate results.
Rows 0/1 are exact controls; row 2 tests the active-turn sway pair in cruise.
This investigation distinguishes impact-specific feedback effects from
general balance, as discussed in [Impact-Invariant Control](https://arxiv.org/abs/2303.00817).

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
- `passive_stand.py` implements only milestone 1 as two independent Isaac components: camera-free passive dynamics and a viewport-rendered proof replay. The 5 s `gravity_static_pose_release_v1` candidate is promoted as the shared dynamics/proof control configuration; exact validation changes only the free-root duration to 30 s.
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
## Isolated TK2 MuJoCo experiment (2026-09-18)

`mujoco_backend.py` converts the exact audited rabbit-ear input to MJCF and
checks all link frames, joint axes, masses and inertia tensors against the URDF.
It retains 69 movable joints (17 policy actions, 52 compliant PD holds), all 68
collision meshes, gravity and a free base. Like the Isaac configuration, it uses
per-mesh convex hull collision and disables self-collision. Effort and position
limits are enforced; velocity-limit violations fail evaluation rather than
clamping the physical state. No canonical milestone or input file is changed.

Create a task-local environment inside ignored outputs, installing the CPU Torch
wheel first. `backend_requirements.txt` pins the tested environment; use the CPU
wheel index as an extra index when installing its `torch==2.7.1+cpu` entry:

```sh
python3 -m venv algorithms/urdf_learn_wasd_walk/outputs/backend/venv
algorithms/urdf_learn_wasd_walk/outputs/backend/venv/bin/pip install \
  --extra-index-url https://download.pytorch.org/whl/cpu \
  -r algorithms/urdf_learn_wasd_walk/backend_requirements.txt
MUJOCO_GL=egl algorithms/urdf_learn_wasd_walk/outputs/backend/venv/bin/python \
  -m algorithms.urdf_learn_wasd_walk.mujoco_backend \
  --name passive_fresh --seconds 30 --noslip-iterations 20 --render
algorithms/urdf_learn_wasd_walk/outputs/backend/venv/bin/python \
  -m algorithms.urdf_learn_wasd_walk.mujoco_policy \
  --name stand_fresh --num-envs 16 --iterations 400 --budget-s 180
```

Run names must be new: existing evidence directories are never overwritten.
Use `--checkpoint <path>` with the diagnostic command to evaluate policy
standing; retain `--noslip-iterations 20` to match the trained physics.
`mujoco_evidence.finalize` requires two passing dynamics runs, a full 30-second
video tied to the recorded state trajectory, an explicit visual review, matching
asset/model/checkpoint identities and the passive predecessor for policy stand.
It writes only a separate MuJoCo evidence artifact. Rendering replays the exact
recorded states after camera-free dynamics; it cannot improve the dynamics.

The optional `--assistance` training coefficient multiplies bounded vertical
pelvis support (9 N maximum before scaling) and orientation PD (1 Nm maximum).
There is no horizontal pulling force. Three consecutive windows of at least 20
successful 30-second episodes reduce the coefficient by 0.1; success below 60%
rolls it back by 0.1, bounded to [0, 1]. Success requires no fall and under 3 cm
root drift. Policy evaluation rejects nonzero assistance. Assistance is distinct
from the normal joint motors, and training reward is never milestone evidence.

`outputs/backend_progress.json` is the restart record. Full failed and successful
runs, exact commands, CPU timing, dependency freezes, pinned upstream revisions
and proof artifacts live under `outputs/`. The official Unitree G1 environment
is separate (`outputs/backend_env_mjlab`) because its pinned mjlab 1.2.0 requires
MuJoCo/Warp 3.5.0. Reference repositories remain under ignored `helper_repos/`.
The Codex sandbox exposes no NVIDIA devices. On TK2, GPU execution is available
through the user-authorized task worker described in
`outputs/backend_gpu/worker_usage.txt`. Its CUDA environment is separate from
both CPU environments. CPU/GPU results still cannot establish an Isaac speed
comparison without an Isaac measurement.

The active experiment is now `ragdoll_walk_first_20260918`, following the
user's assisted-walking-first direction. `mujoco_ragdoll_teacher.py` generates
alternating foot placements with scratch-data inverse kinematics and applies
ordinary joint PD targets within the original URDF limits. The physical free
root is never prescribed. Explicit bounded lateral/vertical support and
orientation torques assist balance; the external forward force is exactly zero.
Coefficient 1 enables the teacher and support, while coefficient 0 disables
both. A fresh student must replace teacher actions before a zero-coefficient
walking result is possible. No old custom walking checkpoint is a resume source.

Motor weight transfer includes waist and hip-roll targets; an optional sustained
waist profile and phase lead change transfer timing without changing physical
limits. Contact-gated hip feedback is a diagnostic motor target correction,
filtered over 40 ms and bounded before the existing target slew and URDF limits.
Its requested offsets and actual motor torques are logged separately. A tilted
orientation reference acts only through bounded external torque. Diagnostic
`--teacher-blend` can separate motor guidance from physical support, but any
unequal coefficients are rejected by curriculum acceptance. Actual contributions
and wrench components are logged, not inferred from the nominal coefficient.

For bounded transfer diagnostics, `--motor-pose-sequence` reads a model-hashed
reference containing only the 17 motor positions and times. Scratch root poses
are ignored; the 52 held joints retain their normal targets. Static IK support
margins do not establish physical balance. Optional teacher tracking integral
is bounded to 0.08 rad with anti-windup at the original actuator target limits.
It changes motor targets only, is logged, and is disabled when teacher blending
is zero. Failed transfer probes remain diagnostic evidence, not walking passes.
A repeating reference may specify `loop_start_s` after a one-time startup
transfer; its last motor pose must equal the pose at that loop boundary. Set
`--period` to the loop duration and `--phase-delay` to its start time for the
student's phase observation. The startup trajectory is never replayed as part
of each gait cycle, and scratch root coordinates remain unused.

Launch a fresh named bounded teacher job through the GPU worker with module
`algorithms.urdf_learn_wasd_walk.mujoco_ragdoll_teacher` and arguments
`["--name", "fresh_trial", "--seconds", "12", "--coefficient", "1"]`.
Inspect dynamics, actual wrench traces, motor tracking and a full-duration video
before accepting assisted gait. Teacher evidence cannot promote a milestone.
Reduce assistance only after sustained stepping and forward progress; hold or
roll back on regression. Final acceptance disables teacher blending, reference
forcing and all external assistance for the entire cumulative evaluation.

Authorized cleanup removed obsolete failed checkpoint files and executable
experiment requests. `outputs/backend/ragdoll_cleanup_manifest.json` records
every removed path/hash, retired request/source text and negative results.
Two validated CPU/GPU standing checkpoints, their complete proofs, the G1
control, environments and failed dynamics evidence remain available.

`mujoco_ragdoll_student.py` initializes a new 63-observation, 17-action network
and distills only reviewed teacher demonstrations. `--aggregate <run-directory>`
adds expert motor labels on states visited by the blended student, following
[dataset aggregation](https://proceedings.mlr.press/v15/ross11a.html). It does not
label a failed rollout as a gait pass. Each trajectory contributes its first
80% to training and last 20% to held-out checks, so newly collected failure
states cannot disappear into an entirely held-out trajectory.
`--balanced-trajectories` samples each trajectory equally so a short startup
failure is not overwhelmed by long steady-gait recordings. Optional
`--interleaved-split` holds out every fifth sample, including late failure
states in training. Its held-out error measures interpolation between nearby
times, not independent generalization; physical evaluation remains decisive.
`--observation-noise-scale` optionally applies small uniform perturbations to
proprioception and previous actions during training, preserving commands and
phase. This tests measured response sensitivity; unchanged expert labels at
perturbed observations are a local approximation. Evaluation uses clean
observations, and improved imitation error never substitutes for physical
walking validation.
`--observation-noise-mode previous_action` restricts that augmentation to action
history for a controlled motor-tracking precision comparison.
`--startup-clock-s` adds one bounded startup-progress input when testing whether
the periodic clock hides the one-time startup sequence. Its duration is stored
in the fresh student checkpoint; inference computes it from elapsed time and
does not query teacher motion. Legacy 63-input checkpoints remain supported.
`--residual-step-rad` optionally learns a bounded correction to the previous
applied motor target, using the existing action-history input. This tests whether
an explicit previous-command skip improves tracking precision. Its bound and
architecture are saved with the checkpoint; inference needs no teacher state.
`--phase-clock-only` tests a learned gait generator that masks proprioception
and action history, retaining command, phase and startup progress. It requires
the startup clock and absolute motor targets. Runtime uses learned weights and
normal motor PD, with no motion-file lookup. This diagnostic architecture does
not establish command generalization, balance recovery or cumulative gate passes.
Optional `--temporal-harmonics` adds bounded-count Fourier features of phase and
startup time to improve temporal fitting precision without reading motion data.
`--teacher-rescue-time` is a separate diagnostic that smoothly hands control
back to the full motor teacher before a developing fall, with external support
kept off. Its time-varying blend is logged, and the resulting rollout is rejected
as a curriculum pass or a distillation demonstration. This tests whether the
teacher can recover visited states before collecting more imitation labels.
`--diagnostic-oracle-student` instead fills the student branch with the current
teacher target sampled and held at 50 Hz. This isolates the actor's command
cadence from prediction error while the other branch retains 500 Hz updates.
Both branches are teacher guidance: the total guidance coefficient is 1,
and these runs are excluded from curriculum acceptance and student datasets.
`--diagnostic-observation-replay` supplies saved teacher observations to a fixed
student, separating feedback on visited states from feedforward approximation
error. It is explicit reference forcing, hashed and excluded from both acceptance
and datasets; no simulated pose is overwritten.
Ground loads are additionally audited at all 500 physics steps per second,
including unsupported intervals and body-weight support ratios.
`mujoco_ragdoll_zero_evidence.py` checks three complete 120-second development
runs with teacher/reference assistance off, actual landed foot placements,
full-rate ground loads, matching identities, and reviewed full-duration video.
Its result explicitly cannot promote standing or the 5 m milestone: those need
the exact checkpoint's separate cumulative evaluation and required distance.
`--action-scale`
sets student output normalization; motor targets still obey the original URDF
limits. A scale of 0.75 covers the demonstrated knee and waist targets that
exceed the earlier 0.5 output range. `mujoco_ragdoll_curriculum.py` runs stages
1 → 0.8 → 0.6 → 0.4 → 0.2 → 0.1 → 0.05 → 0, requiring three complete 30-second trials per
reduced stage with at least 1 m travel, ten sustained lifts per foot, bounded
slip, source velocity/effort limits, and no fall/reset/done/non-foot contact.
The first regression stops the cycle and records rollback. Those are assisted
curriculum checks; the canonical 5 m and cumulative standing gates remain
independent. At coefficient zero, teacher targets are not evaluated and every
external wrench component must be exactly zero. The learned student still uses
normal robot motor PD and a phase-clock observation.
Teacher diagnostics count a flight only once, requiring two consecutive loaded
samples before another lift can be counted; a clearance dip while airborne
cannot manufacture another step. Historical raw counts remain in saved evidence,
with separate recount records where needed.

For the official pretrained G1 positive control, clone
`https://github.com/unitreerobotics/unitree_rl_mjlab` into
`helper_repos/unitree_rl_mjlab` and check out revision
`1425b15f73bd4095f0df53709d7c389c3eb9e790`. Install
`backend_g1_requirements.txt` in a separate task-local Python 3.12 environment,
with the CPU Torch wheel index. Then run `mujoco_g1_control.py --duration 30
--forward 0.5 --seed 42 --video --output <new-algorithm-output-directory>`.
It uses only MuJoCo and ONNX Runtime, never hardware SDK or deployment binaries.
The G1 control is pretrained-policy playback; the separate official PPO smoke
only verifies the training pipeline, not training convergence.

Validated TK2 results and failures are intentionally kept in ignored outputs.
The authoritative standing evidence for this backend is
`outputs/mujoco/passive_hull_validation.json` and
`outputs/mujoco/policy_hull_validation.json`; earlier preliminary validation
files are superseded. Both include the retained positive COM support-hull test,
full-duration repeat runs, video hashes and visual review. Do not infer a 5 m
pass from these standing artifacts or from an assisted rollout.

`mujoco_warp_benchmark` probes the same audited Landau geometry on a selected
Warp device. The pinned Warp 3.5 backend cannot transfer `noslip_iterations=20`;
the default probe records that incompatibility and stops. Explicit
`--noslip-iterations 0` creates exploratory physics requiring new standing
validation. Warp's step function does not implement automatic state resets, so
the adapter omits its unsupported `mjDSBL_AUTORESET` flag on transfer only,
retains it for native CPU comparison, and checks finite states and monotonic
simulation time. A benchmark is never milestone evidence.

```sh
WARP_CACHE_PATH="$PWD/algorithms/urdf_learn_wasd_walk/outputs/backend/warp_fresh_cache" \
  timeout 300 <task-local-gpu-python> \
  -m algorithms.urdf_learn_wasd_walk.mujoco_warp_benchmark \
  --name warp_fresh --device cuda:0 --worlds 64 --blocks 50 \
  --noslip-iterations 0
```

Use separate processes and fresh cache directories for cold compilation runs.
The probe separately records asset/model setup, device transfer, first-step
compilation/execution, graph capture, steady physics transitions, and CPU/Warp
state divergence. The worker probe samples GPU utilization and VRAM each second;
keep both existing CPU environments intact when provisioning CUDA dependencies.

`mujoco_g1_benchmark` is the algorithm-local worker entry point for the pinned
official G1 PPO timing benchmark (4 environments / 2 iterations, then 1,024 / 10).
It verifies CUDA tensor access, times synchronized rollout/learning boundaries,
and retains failed child jobs as failures. NVIDIA's compiler requires a short
temporary directory: the authorized worker's private `/tmp` is used, while
compiled caches, checkpoints and metrics remain under algorithm outputs.

For complete Landau GPU evaluation, use `mujoco_backend --backend
mujoco_warp_cuda --noslip-iterations 0`. This explicitly chooses different
physics from the CPU standing proof. Every step retrieves actual Warp contact,
constraint and actuator forces; it never substitutes CPU dynamics for GPU
validation. This expensive evaluation path is separate from batched throughput
measurement. Float32 GPU clock increments are checked every step; video times
count the fixed integration steps in float64 without modifying GPU state.
`--contact-timeconst` changes only the documented contact solver time constant;
it changes the model hash and requires a fresh standing proof. Meshes, joint
limits, inertias, motor gains and gravity are preserved.

The current GPU proofs are `outputs/mujoco/gpu_passive_health_validation.json`
and `outputs/mujoco/gpu_policy_health_validation.json`. Earlier GPU standing
artifacts predate the strengthened numerical/collision-capacity checks and are
superseded. The validator checks broadphase pairs, contacts and constraints
both before and after post-step forward dynamics, along with finite positions,
velocities, accelerations, forces, transforms and COM. It binds the runtime
snapshot and Warp force-transfer source hashes to evidence.

`mujoco_policy --backend mujoco_warp_cuda --contact-timeconst .004` uses CUDA
physics and CUDA policy tensors, with the same observation/action definitions
as the native adaptation. `mujoco_warp_batch_probe` checks observations, contact
classification, foot slip, exact-mesh clearance and physical-assistance math
against values extracted from actual GPU states. Training uses explicit resets;
evaluation permits none. Every training physics substep records maximum
collision/constraint occupancy so intermediate overflow cannot be hidden by
the final step. The GPU training wrench updates at 50 Hz and is held for ten
physics steps; actual applied vectors and coefficients are logged. The CPU
training wrench updates every physics step, so assistance timing is an explicit
backend difference.

Earlier forward PPO reward/exploration variants remain only as reproducibility
helpers required by saved metrics and standing-policy loaders. Their failed
checkpoints and launch requests were retired; they are not the active training
plan or default resume path. New student training starts only after the physical
walking teacher is demonstrated and visually reviewed.
