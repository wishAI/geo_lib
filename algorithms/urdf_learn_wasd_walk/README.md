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
