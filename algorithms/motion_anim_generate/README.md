# Kimodo → Landau v10 feasibility sandbox

This sandbox is independent of walking training. It never actuates hardware or updates walking gates. Source changes belong only here; the Mac supervisor fetches this branch and syncs declared evidence every 20 minutes. `backend.mjs` is the parent's persistent Astra High / Standard (`serviceTier: default`) launcher.

## Current evidence and limitations

Two actual bundled Kimodo SOMA walk/turn animations have been retargeted onto the copied rabbit-ear robot (same source clip; two IK configurations). They are **upstream-example retargeting evidence, not locally generated requested clips**. The bundled clip does not record its checkpoint revision and is not attributed to the selected v1.1 model.

The first fit has 0.03723 m landmark RMSE, 0.02183 m penetration, 34.08 rad/s maximum joint speed and 85 sliding frames. Hard velocity bounds reduce speed to 4 rad/s but leave 0.03738 m RMSE, 0.02189 m penetration, 240 rad/s² acceleration and 94 sliding frames. Both fail. No exact mesh self-collision or dynamic feasibility claim is made.

Requested text-conditioned idle, forward walk, turn and wave (seed 42 smoke, then 42/43/44 representative runs) require the official gated Llama encoder or authorized exact-prompt embeddings. The public SOMA checkpoint is ungated. Official source contains only the LLM2Vec encoder preset, no public precomputed prompt embeddings. Do not use a different encoder or label unconditional samples as requested actions.

## Reproducible setup

Run from this dedicated repository root. Everything downloaded stays in ignored task-local `outputs/`.

```sh
python3 -m venv algorithms/motion_anim_generate/outputs/venv
algorithms/motion_anim_generate/outputs/venv/bin/pip install -r algorithms/motion_anim_generate/requirements-cpu.txt
PIP_CACHE_DIR=algorithms/motion_anim_generate/outputs/pip_cache algorithms/motion_anim_generate/outputs/venv/bin/pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
# Runtime versions actually exercised are recorded in requirements-runtime.txt once inference runs.
git clone https://github.com/nv-tlabs/kimodo algorithms/motion_anim_generate/outputs/vendor/kimodo
git -C algorithms/motion_anim_generate/outputs/vendor/kimodo checkout 1aece8c124d73d255ceff5086d983b844c9f4e94
algorithms/motion_anim_generate/outputs/venv/bin/python algorithms/motion_anim_generate/prepare.py assets
HF_HUB_DISABLE_XET=1 HF_HOME=algorithms/motion_anim_generate/outputs/hf_cache algorithms/motion_anim_generate/outputs/venv/bin/python algorithms/motion_anim_generate/prepare.py model
```

`prepare.py assets` copies URDF and all 68 meshes as files, verifies the two canonical hashes, and writes `outputs/asset_audit.json`. It imports no walking code. The small URDF is versioned; meshes follow the existing STL ignore rule and must be copied/hydrated on a new machine. Never copy the environment into Git. The standard Nextcloud destination for generated files is `~/Nextcloud/Projects/geo_lib/remote_outputs/<repository-relative-path>`. `outputs/model_inventory.json` and `outputs/large_files.pending.json` supply hashes/sizes for parent registration in root `large_files.json`; this agent cannot edit the root manifest or external cloud folder. Exclude environments, caches and vendor clones from ordinary evidence sync.

## CPU evidence and tests

```sh
OPENBLAS_NUM_THREADS=2 algorithms/motion_anim_generate/outputs/venv/bin/python -m pytest algorithms/motion_anim_generate/tests -q
OPENBLAS_NUM_THREADS=2 algorithms/motion_anim_generate/outputs/venv/bin/python algorithms/motion_anim_generate/experiment.py --upstream-example 02_multi_text_prompt --run-id upstream_walk_turn_v1 --action composite
OPENBLAS_NUM_THREADS=2 algorithms/motion_anim_generate/outputs/venv/bin/python algorithms/motion_anim_generate/experiment.py --upstream-example 02_multi_text_prompt --run-id upstream_walk_turn_speed_bounded --action composite --speed-bounded
```

Run IDs are immutable: use a fresh ID to repeat. Each run retains source NPZ and metadata, target joint/base trajectories, mapping/error report, validation with per-frame violations, complete source/target front/side video, and six-frame contact sheet. Root `outputs/proof.mp4`, `validation.json`, `contact_sheet.png` alias the latest run; `latest.json` identifies its provenance. Evolution nodes use the root GUI's schema version 1 and `parentIds` for generated → retargeted → validated branches. Source-example nodes explicitly label their origin. Inspect the stable manifest's named per-run artifacts to compare failures.

## Isolated CUDA worker

The ordinary Codex namespace hides NVIDIA devices and host processes. The parent verified idle RTX 4080 SUPER (39 MiB, 0%) and no walking job, then provisioned `motion-anim-gpu@JOB.service`. Only this sandbox's outputs are writable in that worker; source/home are read-only and network is isolated. Never modify the service. Dispatch **one job at a time** after the parent/host GPU occupancy check; a successful device probe is not an occupancy audit.

Write `outputs/backend_gpu/jobs/JOB.json`, using a unique alphanumeric job ID, for example:

```json
{"kind":"module","module":"algorithms.motion_anim_generate.gpu_worker","args":["--probe"],"timeout_s":120}
```

```sh
XDG_RUNTIME_DIR=/run/user/1000 DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus systemctl --user start motion-anim-gpu@JOB.service
```

Results/logs are `outputs/backend_gpu/results/JOB.json` and `JOB.log`. For the bounded **unconditional** smoke replace args with `["--unconditional", "--seconds", "2", "--steps", "30", "--seed", "42", "--run-id", "unconditional_smoke_seed42"]`, timeout 1800. This uses the actual trained model's regular CFG with weight zero: upstream `cfg.py` returns `out_uncond` exactly. Placeholder text features never supply semantic conditioning. It is real diffusion, not a synthetic motion fixture, but cannot satisfy any requested action.

For authorized exact-prompt embeddings, provide an NPZ containing Unicode `prompts` and float `embeddings` `[N,1,4096]`, plus a same-stem `.json` containing encoder pins from `provenance.json`, hashes, dtype and provenance. The four exact prompts are in `gpu_worker.PROMPTS`. Worker arguments use `--embeddings /absolute/task/outputs/path.npz --action walk --seconds 2 --steps 30 --seed 42 --run-id walk_smoke_seed42`. Repeat seeds 42/43/44 at 6 seconds and 100 steps only after the smoke works. No credentials should appear in files or chat. All snapshots must be downloaded outside the network-isolated worker.

`TEXT_ENCODER_DEVICE=cpu` is implemented by upstream `llm2vec_wrapper.py`; `Kimodo.text_encoder` is not an nn.Module, so moving the motion model does not override that device. NVIDIA documents <3 GB VRAM with CPU encoding; measure it for the actual worker run. An unconditional run omits the encoder and cannot validate full text-encoder memory use. Postprocessing is explicitly disabled for initial raw-diffusion tests; compiled upstream correction is not silently assumed available.

## Skeleton and frame contract

Landau is 71 links, 69 revolutes and one fixed root mount; 68 meshes and 8,864 triangles. Finger joints and distal shin twists are held at zero; all other URDF revolutes are animation variables, including elbows and wrists needed for waving. This differs deliberately from the walking policy's 17-action contract, which is untouched. Full partition, local axes, limits, units and zero-pose world transforms are in `asset_audit.json`.

FK preserves `root_x_base_fixed` and its +90° roll. World +Z is up, and Landau body +Y is forward. SOMA is +Y-up and +Z-forward. Semantic conversion is `C=[[1,0,0],[0,0,1],[0,1,0]]`: it preserves named left/right (positive X) and exchanges forward/up. **C has determinant −1**, so it is a semantic reflection rather than a rigid rotation; source rotations are never directly assigned to robot joints. A rotation conversion, if added, must be `C R C.T`. Base heading comes from the transformed left/right hip vector; base translation transports the preserved root mount. Tests explicitly cover these conventions.

Bounded least squares fits named joint landmarks with an analytic Jacobian, per-segment target proportions, previous-pose regularization 0.012 m/rad and neutral regularization 0.003 m/rad. Root displacement scales with leg length. Only one floor offset, computed at frame zero, is applied; later penetrations remain visible. The optional speed-bound variant intersects each frame's limits with the previous pose ± URDF velocity / fps. Orientation/twist matching, global trajectory optimization, acceleration bounds and contact-aware IK are not implemented. Source and per-frame fitted/desired landmark errors remain inspectable.

## Validation contract

All thresholds are serialized in every report. `kinematic_pass` means the implemented screening checks and a requested single-action heuristic passed. It never means physically executable or collision-free under a full mesh test.

| Check | Tolerance / method |
|---|---|
| Finite data and schema | Reject NaN/Inf, malformed arrays, wrong joint order, fewer than 3 frames |
| Time | Positive uniform dt; 1e-5 s tolerance; declared 30 Hz source contract |
| Quaternions/base | xyzw norm 1±1e-4; consistent signs; ≤0.35 rad/frame; base matrix agrees within 1e-4 |
| Joints | URDF bounds +1e-5 rad; locked joints ±1e-6 rad; URDF 4 rad/s; acceleration ≤80 rad/s² (screening threshold, not actuator specification) |
| Root continuity | ≤0.08 m/frame |
| Floor | All transformed mesh vertices above −0.005 m; exact for planar triangle minima |
| Foot contact/sliding | Foot/toe mesh min z ≤0.012 m estimates contact; consecutive contact frames permit ≤0.08 m/s horizontal mesh-centroid motion; rolling feet can false-positive |
| Self-intersection | Nonadjacent limb capsules, radii 0.014–0.035 m; overlap >0.003 m flags. Exact mesh intersections unavailable; fingers/ears not covered by capsule proxy |
| Retargeting | Landmark Euclidean RMSE ≤0.03 m; per-frame peak ≤0.08 m |
| Idle | ≤0.04 m root drift; yaw change <0.25 rad |
| Walk | ≥0.15 m displacement along initial +Y; ≥0.015 m relative foot excursion; yaw change <0.6 rad |
| Turn | ≥45° net yaw; ≤0.35 m root drift |
| Wave | Hand >0.04 m above chest for >20% of frames; ≥0.04 m lateral excursion and ≥2 direction changes while raised; ≤0.15 m drift |

Semantics/contact/capsule checks are heuristics. Composite and unconditional clips report single-action semantics unavailable and cannot receive an overall kinematic pass. Dynamics, balance, falls under gravity, torque saturation and control tracking are not tested. Meaningful static/invalid fixtures exercise validators only; they are not generation evidence.

## Official references

- [Pinned implementation](https://github.com/nv-tlabs/kimodo/tree/1aece8c124d73d255ceff5086d983b844c9f4e94), Apache-2.0.
- [SOMA-RP-v1.1 model](https://huggingface.co/nvidia/Kimodo-SOMA-RP-v1.1), NVIDIA Open Model License; all revision pins in `provenance.json`.
- [Installation/access requirements](https://research.nvidia.com/labs/sil/projects/kimodo/docs/getting_started/installation.html).
- [Native skeleton support](https://research.nvidia.com/labs/sil/projects/kimodo/docs/key_concepts/skeleton.html). Native SOMA/G1/SMPL-X do not establish Landau/arbitrary URDF support.

## Executed CPU diffusion and runtime lock

The first **locally inferred**, unconditional seed-42 smoke generated 60 frames in 6.813 s on CPU, using the actual pinned 1,133,185,036-byte checkpoint (SHA-256 `ef0a0ca45a6089ab4532dde609785771ae3f38755b4ae6cf314b0213e07cd4a3`). Its target fails with 0.03504 m RMSE, 0.01308 m penetration, 23.57 rad/s peak speed, 15 sliding frames and 19 capsule-proxy overlap frames. See `outputs/runs/unconditional_cpu_seed42/`. This proves inference and the file-based retarget/evidence pipeline execute; it does not prove prompt control or a valid robot animation.

The installed environment is pinned in `requirements-runtime.txt` (Torch 2.7.1+cu128, Transformers 5.1.0). Rebuild with task-local pip cache and `pip install -r requirements-runtime.txt --extra-index-url https://download.pytorch.org/whl/cu128`. Upstream is imported from the separately pinned ignored vendor directory; no unpinned editable install is required. Pip dependency checks pass.

```sh
TEXT_ENCODER_DEVICE=cpu HF_HOME=algorithms/motion_anim_generate/outputs/hf_cache XDG_CACHE_HOME=algorithms/motion_anim_generate/outputs/cache OPENBLAS_NUM_THREADS=2 timeout 300 algorithms/motion_anim_generate/outputs/venv/bin/python -m algorithms.motion_anim_generate.gpu_worker --device cpu --unconditional --seconds 2 --steps 30 --seed 42 --run-id unconditional_cpu_seed42
# Sequential representative unconditional baseline: 6 s, 100 steps, seeds 42/43/44, bounded joint speeds.
algorithms/motion_anim_generate/outputs/venv/bin/python algorithms/motion_anim_generate/cpu_suite.py
```

The supplied service-start command initially failed from Codex with `Failed to connect to bus: No data available` despite the socket being visible. The parent was notified; CPU inference continued. `outputs/blockers.json` records this separately from the encoder gate. The GUI's worker buttons require permitted host D-Bus access; they do not bypass sandbox restrictions. `dispatch.py` refuses a second active motion service and the GPU worker checks NVIDIA compute-process occupancy before creating its CUDA context.
