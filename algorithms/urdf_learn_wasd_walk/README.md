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
- `passive_stand.py` implements only milestone 1: one passive PD-controlled Isaac environment, exact validation traces, contact forces, and full-duration visual evidence.

Commands:

- `./geo walk milestones`
- `./geo walk inspect`
- `./geo walk test -v`
- `./geo walk validate-passive --steps 32 --smoke --no-record-video --headless`
- `./geo walk validate-passive --headless`

The exact validator writes ignored evidence below `outputs/stand_zero_signal_30s_no_reset/`. The milestone remains `not_started` until that exact 30-second run and its proof video pass; a short smoke never changes milestone state.

Use the Geo Web GUI to launch the TK2 validator and preview its declared JSON, MP4, and contact-sheet artifacts. Later policy, walking, teleop, and terrain code must not be added while milestone 1 is unresolved.
