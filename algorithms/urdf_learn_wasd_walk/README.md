# URDF Learn WASD Walk — Clean Room

This sandbox is being rebuilt from the clean restart contract one milestone at a time.

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
