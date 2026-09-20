# Kimodo → Landau animation sandbox

Watch `outputs/preview.mp4`: a native, clean Landau animation. `outputs/proof.mp4` shows the actual SOMA source and target with fixed camera directions, forward arrows and named feet. `outputs/facing_foot_comparison/` contains the same-source before/after video, contact sheet and geometric measurements. The video-first GUI keeps diagnostic details collapsed.

This task is animation generation and retargeting. Physics, balance, torque, walking gates and actuator speed limits are not acceptance requirements. No hardware is actuated. Original source motion, old target arrays, unsuccessful alternatives and original videos remain inspectable. `kinematic_pass` in older reports is a **legacy screening result**, not animation usability. Current reports distinguish generated/rendered clips from visual quality notes.

## Current result

Real pinned Kimodo unconditional inference ran on CPU and RTX 4080 SUPER. CPU seed-42/43/44 six-second samples and the two-second CUDA smoke have complete videos. The corrected `animation_feet_facing_6s42` reuses the exact six-second seed-42 source bytes; it performs no additional diffusion. It improves facing and foot pose, while some foot sliding and arm-shape differences remain. Source and target proportions differ markedly.

The original coordinate swap had determinant −1 and aligned the target pelvis backwards. The corrected proper rotation is `C=[[-1,0,0],[0,0,1],[0,1,0]]`. SOMA neutral +Y is up, +Z is anatomical forward, and +X is named left. The mounted canonical Landau mesh faces native −Y, with named left at +X. Application +Y is forward after explicit frame transport. The actual `root_x` +90° mounting rotation is retained: `Rbase = C Rsource_hips Rmount.T`. This directly aligns root anatomy without a guessed video flip. Rotation operators change basis as `C R C.T`; reflections cannot be converted directly to quaternions. Signed-axis/conjugation tests cover both proper and reflected bases.

The feet have approximately ±16.3° rest toe-out. Ankle-to-toe pivots descend to the toe, so that vector is not a sole-plane normal. Foot/toe local forward and up axes are calibrated from unchanged canonical rest geometry. Orientation residuals track `C Rsource_foot eZ/eY`, with head/chest orientation also tracked. Shin twists are now enabled for animation: the foot-pose Jacobian gains the missing sixth degree of freedom. Fingers remain locked. Canonical input hashes remain unchanged. `outputs/frame_rest_audit.json` records source-rest reconstruction, mesh sole fits and axis evidence.

Old XZ debug projection and depth ordering also disagreed. Before/after videos now use identical fixed world XZ-from−Y and YZ-from+X cameras; they are not mislabeled anatomical views. Clean previews use a fixed three-quarter camera. No video rotation masks frame errors.

## Reproduce

Run from the dedicated repository root; use fresh run IDs. Task-local downloads, environments, models and generated artifacts are ignored.

```sh
python3 -m venv algorithms/motion_anim_generate/outputs/venv
PIP_CACHE_DIR=algorithms/motion_anim_generate/outputs/pip_cache algorithms/motion_anim_generate/outputs/venv/bin/pip install -r algorithms/motion_anim_generate/requirements-runtime.txt --extra-index-url https://download.pytorch.org/whl/cu128
git clone https://github.com/nv-tlabs/kimodo algorithms/motion_anim_generate/outputs/vendor/kimodo
git -C algorithms/motion_anim_generate/outputs/vendor/kimodo checkout 1aece8c124d73d255ceff5086d983b844c9f4e94
algorithms/motion_anim_generate/outputs/venv/bin/python algorithms/motion_anim_generate/prepare.py assets
HF_HUB_DISABLE_XET=1 HF_HOME=algorithms/motion_anim_generate/outputs/hf_cache algorithms/motion_anim_generate/outputs/venv/bin/python algorithms/motion_anim_generate/prepare.py model
# Existing exact source, animation-only orientation retarget:
OPENBLAS_NUM_THREADS=2 algorithms/motion_anim_generate/outputs/venv/bin/python algorithms/motion_anim_generate/refine.py --parent-run unconditional_cpu_6s_seed42 --run-id animation_feet_facing_repeat --anchor-rigid --temporal-contact --animation --foot-orientation
OPENBLAS_NUM_THREADS=2 algorithms/motion_anim_generate/outputs/venv/bin/python algorithms/motion_anim_generate/compare.py --before unconditional_cpu_6s_seed42 --after animation_feet_facing_repeat
algorithms/motion_anim_generate/outputs/venv/bin/python -m pytest algorithms/motion_anim_generate/tests -q
algorithms/motion_anim_generate/outputs/venv/bin/python algorithms/motion_anim_generate/report.py
```

`prepare.py assets` copies the canonical URDF and 68 meshes as real files and verifies both hashes in `provenance.json`. There are 71 links, 69 revolutes and 8,864 triangles. No walking code is imported. `asset_audit.json` records units, axes, limits and the active/locked partition. Each run records source/model pins, seeds, commands, hashes, targets, diagnostics and full-duration evidence. Generated → retargeted → validated evolution branches retain parentage.

Temporal processing uses soft joint smoothing and bounded root corrections for contact appearance; `--animation` removes hard actuator speed/acceleration constraints. Raw IK targets and both original/adjusted landmark references remain saved. Rigid hip/spine attachment correction changes the target reference, so lower RMSE is not a pure solver improvement. New runs record all options. `compare.py` preserves old debug files before re-rendering the same target from the updated cameras.

## Diagnostics

Finite arrays, valid shapes, positive uniform timestamps (1e−5 s), quaternion norms (1e−4), and consistent base matrices protect data integrity. Animation notes flag pose-limit excursions, >0.35 rad/frame joint/root orientation jumps, >0.12 rad third joint differences, >0.08 m root steps, floor penetration >5 mm, heuristic contact sliding >0.08 m/s, and capsule overlaps >3 mm. These are review prompts, not physical safety gates. Foot contact is estimated within 12 mm of the floor and can flag rolling feet. Capsule checks omit fingers/ears and are not exact mesh collision tests.

Speed, acceleration and landmark-error values and historical thresholds remain recorded as raw diagnostics. They do not reject animation. `directions.json` measures pelvis/chest/head forward dot products, source-relative foot yaw/sole errors, stance tilt and heel/toe heights per frame. Visual review is still needed for fidelity. Idle/walk/turn/wave checks are explicit heuristics; unconditional and composite clips make no requested-action success claim.

## Inference and access

Upstream is pinned to `1aece8c124d73d255ceff5086d983b844c9f4e94` (Apache-2.0). The public SOMA-RP-v1.1 checkpoint is pinned to `6c9233af1180b8151e3c4703477104af5dce9dd5` (NVIDIA Open Model License); checkpoint SHA-256 is `ef0a0ca45a6089ab4532dde609785771ae3f38755b4ae6cf314b0213e07cd4a3`. Native SOMA/G1/SMPL-X support does not establish arbitrary URDF support: SOMA is the source, copied Landau is the target.

CUDA job `cudasmoke20260920a` succeeded on clean source `9703f1b74e56f7f581b77ef2b5d07b5c2df918c9`: 0.722854 s load/inference, 8.485102 s total, 1,196,900,864 B peak allocated. Comparable CPU load/inference was 6.874261 s (9.51× longer). CPU/CUDA RNG streams differ despite identical seed42; this is not a controlled trajectory-quality comparison. No encoder was loaded, so this does not validate full text-pipeline VRAM usage.

The isolated parent GPU service and Torch CUDA probe work. Agent D-Bus dispatch is intentionally unavailable; do not retry it. Future jobs are unique JSON requests in `outputs/backend_gpu/jobs/`, dispatched by the parent after host occupancy checks. The service is network-isolated; download authorized assets beforehand. Never overlap GPU/walking workers or modify the service.

The requested text-conditioned idle/walk/turn/wave suite still needs legitimate access to pinned Meta-Llama-3-8B-Instruct revision `8afb486c1db24fe5011ec46dfbe5b5dccdb575c2` plus the pinned official LLM2Vec adapters. The latest authenticated exact-revision config request returned HTTP403 GatedRepoError. No official public prompt-embedding bundle was found; no gated access is retried or bypassed. Alternatively provide authorized exact-prompt embeddings: NPZ Unicode `prompts` and float `embeddings` `[N,1,4096]`, plus matching JSON provenance/pins/hashes checked by `gpu_worker.py`. Exact prompts are `gpu_worker.PROMPTS`. Official `TEXT_ENCODER_DEVICE=cpu` is implemented in the pinned wrapper; no substitute encoder is used.

The executed unconditional path uses upstream CFG weight zero (`out_uncond`), so placeholder text features do not condition motion. These are actual model samples, not fixtures or a completed semantic suite.

Mac supervisor owns Git integration and artifact sync every20 minutes. Native `preview.mp4` aliases must be preserved rather than overwritten by debug crops. Files over5 MiB follow the root Nextcloud contract; `outputs/large_files.pending.json` supplies registration metadata. Environment/cache/vendor directories stay out of ordinary artifact sync. Run `./geo storage audit` before handoff. Do not push or edit parent checkouts/helpers.

References: [official source](https://github.com/nv-tlabs/kimodo), [official docs](https://research.nvidia.com/labs/sil/projects/kimodo/docs/), [selected public model](https://huggingface.co/nvidia/Kimodo-SOMA-RP-v1.1).

## Multi-frame improvement and official encoder preparation

`quality.py RUN_ID...` checks every frame, with mean/p50/p95/p99/max, worst frame/time and contiguous flagged intervals. It records facing, source-relative sole/foot yaw, heel/toe heights, stance drift/sliding, swing-height changes, limb directions, landmark error and temporal discontinuities. Raw source and target sliding are retained alongside added drift: source translation and toe-off are not automatically errors. Thresholds select frames for visual review, not physical acceptance.

`contact_refine.py --parent-run BASELINE --run-id NEW_ID` tests a bounded source-aware stance correction on retained motion. It modifies leg joints only, follows actual source displacement during each stance, retains foot orientation/height and smooths only the correction. Whole-clip stance does not receive artificial touchdown/liftoff ramps. Original target, exact source and numerical failures remain saved. `review.py RUN_ID...` renders full-duration foot closeups, selects at least24 evenly spaced plus worst/contact/turn frames, and provides all decoded frames as sequential sheets. Review metadata explicitly distinguishes dense frame inspection from real-time player playback.

The parent confirmed normal SDK credentials exist, but the exact pinned Llama config returned **HTTP403 GatedRepoError**. User authorization does not grant Hugging Face account access. Do not retry until the parent reports account approval. No alternate encoder or masking algorithm is substituted.

An isolated encoder environment is necessary: Transformers5.1.0 bypasses the pinned official bidirectional-mask override. The [official LLM2Vec dependency range](https://github.com/McGill-NLP/llm2vec/blob/6bbd52528bee4936786ff0e9eb8a569698b1c731/setup.py) supports4.44.2. `requirements-encoder.txt` pins that compatible runtime; its task-local environment reads shared task Torch/utilities without changing the diffusion environment. `encoder_check.py` uses tiny random CPU fixtures to verify future-token influence, padding exclusion, MNTP/supervised loading order and exact official prompt framing. These fixtures are dependency tests, never generated-animation or pretrained-embedding evidence.

From the sandbox directory, **after account approval**, the preparation flow is:

```sh
outputs/venv/bin/python encoder_prepare.py setup
outputs/venv/bin/python encoder_prepare.py download
# Bounded CPU execution; no GPU overlap, no network during encoding:
timeout 1800 outputs/encoder_venv/bin/python encoder_prepare.py encode
```

Downloads use standard SDK credentials in memory, immutable exact revisions, task-local paths and hash inventories. A derived MNTP directory rewrites only its local base path while retaining the canonical model name required for official prompt framing. Original snapshots stay intact. CPU BF16 runs the unmodified official wrapper, first MNTP merge then supervised adapter, internal batch_size1, exporting verified float32 `[4,1,4096]` embeddings with pins/hashes/dtypes/RSS/timing. The existing GPU worker consumes that bundle. The parent dispatches unique job JSON only after embeddings and host occupancy are verified. `encoder_large_files.pending.json` supplies the parent's Nextcloud registration handoff; do not duplicate the existing motion-checkpoint transfer.

Constraint-only smoke (no training or text encoder): the pinned `model/cfg.py`
regular-CFG weight0 branch clears both text and the motion mask. The official
separated branch with weights `[0,2]` retains only constraint guidance. Worker
`--pose-anchor --seconds 2 --steps 30 --seed 42 --run-id UNIQUE` constructs an
official `FullBodyConstraintSet` at frame30 from retained seed43 frame30, maps
77 source joints to the30 internal SOMA names and subtracts horizontal pelvis
translation in SOMA XZ coordinates. Global rotations are supplied to the API
but upstream does not constrain them; their error is reported separately.

One bounded job generates a matched-null source with separated weights `[0,0]`
and the same CUDA seed/constraint input/heading, then the guided source. It saves
both, reports anchor RMS/max/root/heading errors and adjacent-frame continuity,
and renders the guided Landau target. Expected diagnostic values are RMS50mm,
max100mm and50% improvement; these test constraint following, not animation
acceptance. No output is clamped. Constraint evidence does not automatically
replace the reviewed default preview. `constraint_preflight/preflight.json`
only verifies CPU constructor/metric plumbing; it is not inference evidence.

Actual CUDA job `poseanchor20260920a` completed in11.70s. Anchor RMS decreased
from0.1759645m for the matched null to0.0110718m guided (93.71% reduction), with
guided maximum0.0241503m. `outputs/constraint_feasibility.json` separates source
constraint following from remaining Landau pose concerns. This demonstrates
one learned pose anchor without text; it does not demonstrate a prompted action
suite or arbitrary root paths. The next specified experiment is a two-second,
seed43,30-step matched-null test of five official root waypoints spanning0.4m;
it has not been requested or executed.

## Hand orientation follow-up

`outputs/hand_comparison.json` compares three retained six-second sources.
Position-only landmarks leave wrist pitch and collinear forearm roll unobserved.
`hand_refine.py` calibrates proper anatomical bases from neutral middle-finger
and thumb-side rays, then adjusts only four forearm/wrist joints. Source motion,
root, legs, upper-arm pose and locked finger joints are preserved. Smoothing only
the correction over1.5 frames removes added jitter spikes seen in the retained
0.6-frame alternative. Every frame has orientation and continuity diagnostics;
clean, paired-hand and whole-body comparison videos retain original timing.

From this sandbox, reproduce with a fresh ID:
```sh
outputs/venv/bin/python hand_refine.py --parent-run animation_contact_v3_seed42 --run-id UNIQUE --smoothing-sigma 1.5
outputs/venv/bin/python review.py UNIQUE
```

The published candidates are `animation_hands_v3_seed42/43/44`; the GUI default
remains reviewed `animation_contact_v3_seed42`. All540 candidate clean frames,
at least24 evenly spaced frames per clip and worst hand frames were inspected
in chronological sheets. New hand-variant real-time playback remains pending.
Residual hand-direction mismatch and high/compressed arm placement remain.
Canonical right upper-arm roll is perpendicular to its bone whereas left roll
aligns; the canonical URDF is preserved. Exact hand/torso mesh intersections and
individual finger fidelity are not established.

Character proportions and movement style are distinct. Uniform scaling or
reusing rotations does not create childlike gait. The pinned fixed-skeleton
Kimodo source is retargeted to Landau proportions; arbitrary child proportions
are not a native conditioning input, and childlike behavior has not been
demonstrated. No model training, generator change or GLB migration is involved.
