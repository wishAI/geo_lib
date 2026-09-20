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

The requested text-conditioned idle/walk/turn/wave suite still needs legitimate access to pinned Meta-Llama-3-8B-Instruct revision `8afb486c1db24fe5011ec46dfbe5b5dccdb575c2` plus the pinned official LLM2Vec adapters. The observed config request returned HTTP401. No official public prompt-embedding bundle was found; no gated access is retried or bypassed. Alternatively provide authorized exact-prompt embeddings: NPZ Unicode `prompts` and float `embeddings` `[N,1,4096]`, plus matching JSON provenance/pins/hashes checked by `gpu_worker.py`. Exact prompts are `gpu_worker.PROMPTS`. Official `TEXT_ENCODER_DEVICE=cpu` is implemented in the pinned wrapper; no substitute encoder is used.

The executed unconditional path uses upstream CFG weight zero (`out_uncond`), so placeholder text features do not condition motion. These are actual model samples, not fixtures or a completed semantic suite.

Mac supervisor owns Git integration and artifact sync every20 minutes. Native `preview.mp4` aliases must be preserved rather than overwritten by debug crops. Files over5 MiB follow the root Nextcloud contract; `outputs/large_files.pending.json` supplies registration metadata. Environment/cache/vendor directories stay out of ordinary artifact sync. Run `./geo storage audit` before handoff. Do not push or edit parent checkouts/helpers.

References: [official source](https://github.com/nv-tlabs/kimodo), [official docs](https://research.nvidia.com/labs/sil/projects/kimodo/docs/), [selected public model](https://huggingface.co/nvidia/Kimodo-SOMA-RP-v1.1).
