# Motion Anim Generate

Implement the user-requested NVIDIA Kimodo feasibility experiment for Landau v10. This separate animation sandbox is not a walking milestone and may progress independently of the walking ladder.

- Research and pin the official https://github.com/nv-tlabs/kimodo implementation and weights. The user called it Komodo; Kimodo is the verified NVIDIA motion generator. Do not silently substitute another generator.
- Copy the canonical rabbit-ear Landau v10 URDF/meshes into this sandbox inputs before use. File-based handoff only; no imports from other algorithm folders. Record source hashes, units, axes, rest pose and joint mapping. Never alter canonical inputs.
- Generate and retarget basic idle, walk, turn and wave clips with fixed seeds and reproducible commands. Native Kimodo support for an arbitrary URDF must be tested, not assumed. Explain and measure retargeting if required.
- Record source and retargeted debug videos, full-body front/side views, contact overlays, joint-limit warnings and a contact sheet. Keep original failed clips and failure reasons. Videos must show actual generated motion on the copied Landau model.
- Validate finite samples, time continuity, joint limits, joint speed/acceleration, foot penetration/sliding, self-intersection and expected action semantics. Store thresholds, per-frame violations and aggregate pass/fail in JSON. Synthetic fixtures only test validators; they are never generation evidence.
- Separate kinematic validity from dynamic feasibility. Kinematic playback never proves balance or deployable robot control. If physical replay is attempted, report contacts, falls, actuator saturation and tracking error separately. No real robot actuation.
- All environments, models, downloads and generated results belong in ignored task-local outputs/inputs; files above 5 MiB stay out of Git and follow the Nextcloud contract. Do not change global environments.
- Publish outputs/backend_progress.json and outputs/evolution.json at meaningful transitions, with source commit, model revision, seeds, asset/checkpoint/config hashes, actual results, blockers and next step. GUI manifest must expose actual videos and reports. Never label pending work successful.
- Run bounded experiments without overlapping existing heavy GPU/Isaac jobs. Inspect live jobs first. Preserve the stopped walking task.
- Commit only this sandbox's source files on the dedicated task branch. The Mac supervisor pulls source via Git and evidence via explicit artifact sync every 20 minutes. Do not mutate the parent developer checkout, GUI cache or unrelated source.

Latest user GUI override: do not expose an evolution tree or training UI in this animation sandbox. Keep videos, comparisons and collapsed quality checks. Preserve reproducibility internally; the walking sandbox remains separate.
