# Landau `W` / `S` Teleop Handoff

## Scope

- Project: `algorithms/urdf_learn_wasd_walk`
- Robot: `landau`
- Branch at investigation time: `task/fix_urdf_learn`
- Reported symptom:
  - `Q` / `E` visibly rotate the body
  - `W` / `S` and arrow up/down appear to do nothing

## What Was Checked

### 1. Keyboard input path

- The keyboard events do reach the teleop layer.
- `problems/log.txt` showed:
  - repeated `KEY_REPEAT` / `CHAR` for `W`
  - a valid `KEY_PRESS` for `S`
  - `env_command` changing when keys were pressed
- A bug in [teleop_input.py](/home/wishai/vscode/geo_lib/algorithms/urdf_learn_wasd_walk/teleop_input.py) was fixed so `KEY_REPEAT` behaves like a held key even when Isaac does not emit an initial `KEY_PRESS`.
- That fix is real, but it did **not** solve the visible “no walking” symptom by itself.

### 2. USD visual model vs hidden URDF

- The synced `.usdc` visual model is **not** offset from the hidden URDF articulation.
- A headless probe compared the expected visual root transform against the live `LandauVisual` transform.
- Result:
  - multiple sampled frames had `max_abs_diff = 0.0`
- Conclusion:
  - the visible model is following the URDF exactly
  - this is **not** a display desync bug

### 3. Current policy quality

- The current Landau checkpoints barely translate under forward commands.
- Latest tested run:
  - `logs/rsl_rl/geo_landau_flat/2026-03-31_22-40-53_cmdfix_long_resume2/model_650.pt`
- Headless validation with forced forward command:
  - semantic command: `(0.5, 0.0, 0.0)`
  - current runtime env command: `(0.0, 0.5, 0.0)`
  - max planar displacement after `500` steps: about `0.0956 m`
  - mean base-frame linear velocity: about `[0.0002, 0.0410, 0.0018]`
- Headless validation with forced yaw command:
  - semantic command: `(0.0, 0.0, 0.5)`
  - env command: `(0.0, 0.0, 0.5)`
  - mean yaw rate in world frame: about `0.456 rad/s`
- Practical conclusion:
  - yaw is learned well enough to be visible
  - forward translation is weak enough to look like “not walking”

### 4. Other recent checkpoints

- Several March 31, 2026 Landau checkpoints were tested.
- None showed strong forward tracking.
- Best observed forward displacement over `256` steps was still only about `0.095 m`.
- This looks systemic, not a single bad checkpoint.

## Important Code Observations

### 1. Landau runtime command remap is questionable

- [command_frame.py](/home/wishai/vscode/geo_lib/algorithms/urdf_learn_wasd_walk/command_frame.py) hard-codes:
  - for `landau`, semantic `(forward, strafe, yaw)` becomes env `(strafe, forward, yaw)`
- That means Landau “forward” is assumed to be body `y`, not body `x`.
- This assumption is not encoded as explicit robot metadata. It is hidden behind `robot_key == "landau"`.

### 2. Root-link setup is suspicious

- [robot_specs.py](/home/wishai/vscode/geo_lib/algorithms/urdf_learn_wasd_walk/robot_specs.py) currently picks:
  - `root_link_name = model.root_links[0]`
- For this URDF that resolves to `base_link`.
- But in the URDF:
  - `base_link` is a tiny dummy root
  - `root_x` is attached under a fixed joint:
    - joint name: `root_x_base_fixed`
    - origin: `xyz="0.000000 -0.012911 0.299642" rpy="1.570796 -0.000000 -0.000003"`
- Live probe result:
  - articulation root link name: `base_link`
  - real skeleton root body: `root_x`
  - `root_x` is mounted under a fixed `+90 deg` roll relative to the articulation root
- This is a strong candidate for broken control / reward semantics during training.

### 3. Validation metric is partly circular

- [validate_walk.py](/home/wishai/vscode/geo_lib/algorithms/urdf_learn_wasd_walk/scripts/validate_walk.py) uses the same Landau-specific semantic remap to define “forward”.
- That means the forward metric depends on the same assumption that is already under suspicion.
- The validator should report raw displacement along root-frame `x` and `y` regardless of semantic interpretation.

## What Is Most Likely Wrong

1. The current Landau policy is too weak on linear velocity tracking.
2. The Landau frame conventions are under-specified and likely wrong in runtime tools.
3. The task may be training against the wrong root/control frame because `base_link` is used as the root even though `root_x` is the real body.

## What Is Probably Not Wrong

1. Keyboard events reaching teleop
   - they do reach teleop
2. The `.usdc` model being elsewhere in the scene
   - the visual root matches the URDF exactly

## Files Worth Inspecting First

- [command_frame.py](/home/wishai/vscode/geo_lib/algorithms/urdf_learn_wasd_walk/command_frame.py)
- [robot_specs.py](/home/wishai/vscode/geo_lib/algorithms/urdf_learn_wasd_walk/robot_specs.py)
- [landau_env_cfg.py](/home/wishai/vscode/geo_lib/algorithms/urdf_learn_wasd_walk/landau_env_cfg.py)
- [scripts/teleop.py](/home/wishai/vscode/geo_lib/algorithms/urdf_learn_wasd_walk/scripts/teleop.py)
- [scripts/play.py](/home/wishai/vscode/geo_lib/algorithms/urdf_learn_wasd_walk/scripts/play.py)
- [scripts/validate_walk.py](/home/wishai/vscode/geo_lib/algorithms/urdf_learn_wasd_walk/scripts/validate_walk.py)
- [usd_visualizer.py](/home/wishai/vscode/geo_lib/algorithms/urdf_learn_wasd_walk/usd_visualizer.py)

## Existing Notes

- A broader redesign note was also written here:
  - [redesign_plan.md](/home/wishai/vscode/geo_lib/algorithms/urdf_learn_wasd_walk/redesign_plan.md)

## Suggested Next Fix Path

1. Make the Landau control frame explicit in robot metadata instead of hard-coding a `landau` swap in `command_frame.py`.
2. Update validation to print:
   - env command
   - actual root linear velocity in env frame
   - displacement along root-frame `x`
   - displacement along root-frame `y`
3. Revisit `root_link_name` for Landau and confirm whether training should use `root_x` instead of `base_link`.
4. Run a short forward-only Landau retrain before reintroducing full `x/y/yaw` command space.
