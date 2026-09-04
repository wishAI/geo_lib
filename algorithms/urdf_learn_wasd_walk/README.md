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
- `robot_spec.json` records the 17 action joints, every explicitly locked joint, nominal pose, PD gains, and Landau's body-`+Y` semantic command mapping.
- `passive_stand.py` implements only milestone 1 as two independent Isaac components: camera-free passive dynamics and a viewport-rendered proof replay.
- `passive_pipeline.py` runs those components sequentially and creates final milestone evidence only when both pass.
- Milestones 1 and 2 passed from the canonical derived pose and PPO checkpoint;
  `milestones.json` pins every evidence hash. The 5 m flat forward gate is active.
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

Use the Geo Web GUI to launch TK2 commands and preview declared JSON, MP4, and contact-sheet artifacts. The active implementation transfers the proven stand actor into the 5 m flat forward gate while requiring the candidate checkpoint to re-pass standing.
