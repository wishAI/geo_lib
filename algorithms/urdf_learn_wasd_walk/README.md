# URDF Learn WASD Walk — Clean Room

This sandbox is being rebuilt from the clean restart contract one milestone at a time.

Kept:

- `TRAINING_RULES.md`: human-readable cumulative acceptance ladder
- `milestones.json`: machine-readable copy of the same twelve milestones
- `inputs/landau_v10/`: the robot input needed to rebuild the environment
- `gui/manifest.json`: milestone status and browser URDF viewer contract

Still intentionally absent:

- prior environment, policy, reward, teleop, play, validation, and curriculum logic
- old tests and agent configurations tied to that implementation
- run history, checkpoint lineage, restart notes, and problem investigations

Current clean implementation:

- `model_spec.py` audits the exact retained URDF, inertia, collision package, root transform, limits, and zero-pose joint axes.
- `robot_spec.json` records the 17 action joints, every explicitly locked joint, nominal pose, PD gains, and Landau's body-`+Y` semantic command mapping.
- `passive_stand.py` implements only milestone 1 as two independent Isaac components: camera-free passive dynamics and a viewport-rendered proof replay.
- `passive_pipeline.py` runs those components sequentially and creates final milestone evidence only when both pass.

Commands:

- `./geo walk milestones`
- `./geo walk inspect`
- `./geo walk test -v`
- `./geo walk validate-passive-dynamics --steps 32 --smoke --reuse-usd-cache --headless`
- `./geo walk validate-passive-dynamics --headless`
- `./geo walk render-passive-proof --headless`
- `./geo walk finalize-passive`
- `./geo walk validate-passive --headless`

The exact all-in-one command runs the same two Isaac components sequentially; it never overlaps Isaac processes. Camera-free results are retained as `dynamics_validation.json` if proof rendering fails. The final `validation.json` is fail-closed and appears only after both exact components pass. All evidence is ignored below `outputs/stand_zero_signal_30s_no_reset/`. The milestone remains `not_started` until that exact 30-second run and its proof video pass; a short smoke never changes milestone state.

Use the Geo Web GUI to launch the TK2 validator and preview its declared JSON, MP4, and contact-sheet artifacts. Later policy, walking, teleop, and terrain code must not be added while milestone 1 is unresolved.
