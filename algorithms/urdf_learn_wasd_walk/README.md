# URDF Learn WASD Walk — Clean Room

This sandbox has intentionally been reset for a future coding agent.

Kept:

- `TRAINING_RULES.md`: human-readable cumulative acceptance ladder
- `milestones.json`: machine-readable copy of the same twelve milestones
- `inputs/landau_v10/`: the robot input needed to rebuild the environment
- `gui/manifest.json`: milestone status and browser URDF viewer contract

Removed:

- prior environment, policy, reward, teleop, play, validation, and curriculum logic
- old tests and agent configurations tied to that implementation
- run history, checkpoint lineage, restart notes, and problem investigations

There are no supported train, play, or validation commands yet. That is deliberate. A future agent must rebuild the smallest implementation required for the earliest unresolved milestone, prove it, and only then add the next milestone.

Use `./geo walk milestones` to inspect the clean ladder. Use the Geo Web GUI to inspect and articulate the retained URDF.
