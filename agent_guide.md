# Agent Guide

This is the primary reference for AI agents working on the `geo_lib` repository. Read this before starting any task.

---

## 1. Sandbox Rules

### Algorithm isolation

Every algorithm lives under `algorithms/<name>/` and is a self-contained unit.

- **No cross-folder Python imports.** Never `import` or call Python code from one algorithm folder inside another.
- **File-based handoff only.** If data must flow between algorithms, copy output files into the consuming algorithm's `inputs/` directory.
- **Input/output directories.** Each algorithm keeps its inputs under `inputs/` and its outputs under `outputs/`.

### Outputs folders

- **Ignore `outputs/` directories.** Every algorithm's `outputs/` folder contains generated run artifacts, not source code. Do not read, modify, or commit files in `outputs/` folders unless the user explicitly asks.
- Similarly, `logs/` is gitignored and contains training checkpoints, tensorboard logs, etc. Reference specific checkpoints by path when needed, but do not browse `logs/` aimlessly.

### File discipline

- Do not create files unless necessary. Prefer editing existing files.
- Do not create README or documentation files unless asked.
- Do not modify `.gitignore`, CI configs, or shared infrastructure without explicit permission.
- Large binaries (`.usdc`, `.blend`, `.stl`, `.glb`) are gitignored. Do not commit them.

### Git discipline

- Never force-push, reset --hard, or amend published commits without explicit user instruction.
- Commit only when asked. Stage specific files, not `git add -A`.
- Never commit `.env`, credentials, or log files.

---

## 2. Python Environments

This machine uses `pyenv` with Python 3.10.13 as the base for most virtual environments.

### ptenv (general purpose)

```bash
pyenv activate ptenv
```

- Python 3.10.13
- Used for: `svg_scene_builder`, `simple_auto_slam_mapping`, `path_order_orientation_ortools`, `ikfast_urdf_solver`, general scripts, unit tests
- MuJoCo runs headless: `MUJOCO_GL=egl`
- This environment is currently available on TK2 but not on the Mac. The Geo Web GUI therefore defaults dependency-heavy examples to TK2 and allows local execution only as an explicit choice.

### Isaac Sim / Isaac Lab environment

```bash
/home/wishai/vscode/IsaacLab/isaaclab.sh -p <script_or_module>
```

- Isaac Lab has its **own Python environment** managed by `isaaclab.sh -p`. Do NOT use `ptenv` or `python3` directly for Isaac tasks.
- Any future `urdf_learn_wasd_walk` Isaac scripts must be launched through `isaaclab.sh -p`; none exist in the clean sandbox today.
- Isaac Lab location: `/home/wishai/vscode/IsaacLab/`
- The `-p` flag runs a Python script or module inside Isaac Lab's managed conda/pip environment.

### When to use which

| Task | Environment |
|------|-------------|
| Pure Python tests (`pytest`) | `pyenv activate ptenv && python3 -m pytest ...` |
| MuJoCo scene builder / SLAM mapping | `pyenv activate ptenv` |
| Future walk Isaac work | rebuild the milestone-specific launcher first, then use `isaaclab.sh -p` on TK2 |
| ROS2 CLI tools | source `/opt/ros/humble/setup.bash` first |
| USD mesh workbench on the Mac | `blender --background --python algorithms/usd_parallel_urdf/local_mesh_workbench_blender.py -- ...` |

### Mac + TK2 update workflow

The Mac checkout is the canonical editable source. For every code or GUI change that must also run on TK2:

1. Make and test source edits in the Mac checkout.
2. Run `./geo storage sync-code-tk2` to refresh the isolated TK2 workspace at `/home/wishai/.cache/geo-lib-webgui/current`.
3. Run TK2 validation from that isolated workspace, preferably through `./geo gui`. Every GUI-started TK2 job performs the same source refresh automatically before launch.
4. Pull generated results through the GUI or the declared storage/output commands; never edit the Mac checkout from TK2 or copy generated `outputs/` back into source control.

Do not use `/home/wishai/vscode/geo_lib` as the synchronization target for agent changes. It is a developer checkout and the GUI must not rewrite it. Large declared assets still require `./geo storage push-tk2` in addition to the code refresh because they follow the Nextcloud contract.

The USD mesh workbench defaults to the Mac Blender backend. It copies extracted per-link source STL files into the ignored local output cache, solidifies open skin patches, preserves loose shells, generates manifold voxel/hull/box STL assets, validates edge topology and source-bounds retention, and touches the local URDF so the WebGUI resolves previews locally. The TK2 target remains available for Isaac-based source extraction and uses a stable closed-hull fallback. Never restore triangle sampling or a "keep largest component" cleanup: the former creates open collision noise and the latter silently deletes valid body geometry. A link is acceptable only when it is watertight and retains its source extents; use a closed bounds-box guard if a pathological link cannot satisfy both.

---

## 3. Long-Running Tasks: Strategy and Sub-Agents

### The problem

Training runs, large refactors, and multi-step debugging can exceed a single conversation's context window or hit diminishing returns as context fills up.

### When to start fresh or spawn a sub-agent

- **Context pollution.** If you have read 15+ files and the conversation is getting long, consider spawning a sub-agent with a clean, focused prompt for the next phase.
- **Stuck in a loop.** If you have tried 3+ approaches for the same problem without progress, stop. Write down what you know, what you tried, and why it failed. Then either ask the user or spawn a fresh sub-agent with that summary.
- **Parallel independent work.** Use background agents for tasks that do not depend on each other (e.g., running tests while researching a separate issue).
- **Research vs implementation.** Spawn an `Explore` agent for codebase research. Do implementation yourself in the main conversation so the user can review diffs.

### How to hand off to a sub-agent

Write a prompt that includes:
1. What the goal is and why.
2. What has already been tried and what failed (with specifics: file paths, error messages).
3. The exact files worth reading first.
4. What output you expect (a summary, a code change, a diagnosis).

Bad: "Rebuild walking." Good: "The clean walk sandbox has no implementation. Read `TRAINING_RULES.md` and `milestones.json`, then design only the passive 30-second stand environment and validator for milestone 1. Return the exact files and proof contract required; do not add policy training yet."

### When NOT to use sub-agents

- Simple file reads or grep searches. Use Glob/Grep/Read directly.
- Tasks with 1-3 steps that you already understand.
- When the user didn't ask for sub-agents and the task is straightforward.

---

## 4. Walk Training: Clean-Room Protocol

The old `urdf_learn_wasd_walk` implementation and all experimental history were intentionally removed on 2026-08-22. Do not restore them from Git history or old branches unless the user explicitly reverses that decision.

1. Read `README.md`, `TRAINING_RULES.md`, and `milestones.json` in the sandbox.
2. Identify the first `not_started` milestone.
3. Rebuild only the minimal environment, launcher, validator, and tests required for that gate.
4. Keep Isaac single-process and run it on TK2.
5. Before a gate is marked passed, require machine-readable validation and a short proof video that visibly covers the full behavior.
6. Record new evidence in a new clean lineage. Never import claims or checkpoint status from the removed implementation.
7. Keep checkpoints and generated runs out of Git; use the root Nextcloud large-file contract for durable artifacts above 5 MiB.

`./geo walk milestones` is currently the only supported walk command. Train, play, teleop, and validation commands do not exist until a future agent rebuilds them milestone by milestone.

---

## 5. Key Repository Structure

```
geo_lib/
  CONTEXT.md                  -- file handoff rules, environment notes
  agent_guide.md              -- this file
  algorithms/
    svg_scene_builder/        -- SVG to MuJoCo scene
    simple_auto_slam_mapping/ -- route planning, lidar, occupancy mapping
    urdf_learn_wasd_walk/     -- intentionally empty locomotion clean room
      README.md               -- reset scope and future-agent instructions
      TRAINING_RULES.md       -- cumulative acceptance contract
      milestones.json         -- all twelve gates, currently not started
      inputs/                 -- retained Landau URDF input
      gui/manifest.json       -- milestone and URDF-viewer surface
    ikfast_urdf_solver/       -- IKFast URDF solver wrapper
    usd_parallel_urdf/        -- USD/URDF conversion (Kit-based)
    path_order_orientation_ortools/ -- path optimization
    fake_cloud/               -- cloud simulation stub
    avp_remote/               -- Apple Vision Pro remote
  assets/                     -- shared robot assets (gitignored large files)
  problems/                   -- investigation notes and fix plans
  helper_repos/               -- third-party cloned repos (gitignored)
```

### helper_repos/ (gitignored)

Third-party repositories cloned locally for build-time or reference use. These are NOT part of geo_lib source and are gitignored. Do not commit changes into them.

| Directory | Source | Purpose |
|-----------|--------|---------|
| `ikfastpy/` | [github.com/andyzeng/ikfastpy](https://github.com/andyzeng/ikfastpy) | IKFast Python bindings for analytical inverse kinematics |
| `jagua-rs/` | [github.com/JeroenGar/jagua-rs](https://github.com/JeroenGar/jagua-rs) | Fast 2D irregular cutting/packing collision detection engine (Rust) |
| `pytracik/` | [github.com/chenhaox/pytracik](https://github.com/chenhaox/pytracik) | Python wrapper for TracIK numerical inverse kinematics solver |
| `tracikpy/` | [github.com/mjd3/tracikpy](https://github.com/mjd3/tracikpy) | Alternative TracIK Python bindings |
| `debs/` | local | Pre-downloaded `.deb` packages: `ros-humble-ur-description`, `ros-humble-xacro` |
| `ur_description_pkg/` | extracted from deb | UR robot URDF description files |
| `xacro_pkg/` | extracted from deb | ROS2 xacro macro processor |

If you need functionality from a helper repo, import or call it as an external dependency. Do not copy its source into an algorithm folder.

---

## 6. Common Pitfalls

### Isaac Lab

- Never run Isaac scripts with `python3` directly. Always use `isaaclab.sh -p`.
- Isaac Lab headless mode: always pass `--headless`.
- Teleop requires GUI: do NOT pass `--headless` for teleop.
- The walk sandbox has no runnable Isaac task yet. A future implementation must add and test its own launcher before any training run.

### Landau-specific

- Landau body forward axis is body `+Y`, not `+X`.
- The semantic command contract remains `forward`, `strafe`, `yaw`; a future implementation must define and test any frame mapping explicitly.
- Re-check mass and inertia directly from the retained URDF rather than carrying forward old prose claims.
- The `base_link` is a tiny dummy root; the real skeleton root is `root_x` mounted with a +90 deg roll.

### Testing

- Pure Python tests: `pyenv activate ptenv && python3 -m pytest algorithms/<name>/tests -q`
- Isaac smoke tests: use `isaaclab.sh -p` with `--steps 32 --headless`
- MuJoCo tests: `MUJOCO_GL=egl` must be set for headless rendering

---

## 7. Working with Existing Documentation

Read these in order of priority for any task:

1. **This file (`agent_guide.md`)** -- environment, rules, workflow
2. **`CONTEXT.md`** -- algorithm handoff rules, file contracts
3. **Algorithm-specific `README.md`** -- commands, layout, validation
4. **`TRAINING_RULES.md` + `milestones.json`** (for walk work) -- clean acceptance truth
5. **`problems/` directory** -- current investigation notes and fix plans

---

## 8. Behavioral Guidelines for Agents

### Do

- Read before writing. Understand existing code before modifying it.
- Run tests after making changes.
- For future walk work, update only the new clean milestone evidence created by that implementation.
- Name training runs descriptively.
- Report validation metrics, not just PPO reward.
- Keep changes minimal and focused.
- Ask the user when genuinely stuck after investigation.

### Do not

- Guess at file contents or code behavior without reading them.
- Trust old notes that claim something "passes" without re-validating.
- Pile on more reward terms when existing ones have conflicting gradients. Simplify first.
- Run destructive git operations without user confirmation.
- Create unnecessary files or abstractions.
- Retry the same failing approach without diagnosing the root cause.
- Modify code outside the algorithm folder you are working on (sandbox rule).

### When stuck

1. Write down what you know, what you tried, and why it failed.
2. Check the current algorithm README, milestone contract, or `problems/` entry.
3. If the context window is polluted, consider spawning a fresh sub-agent with a focused prompt.
4. If genuinely blocked, ask the user with a specific question (not a vague "I'm stuck").

---

## 9. Internet Research During Projects

Agents should actively use web search (WebSearch tool) to find relevant papers, repos, and techniques **during project work**, not just rely on what is already in the codebase. This is especially important for:

### When to search

- **Before designing a new reward function or training approach.** Search for recent papers and open-source implementations for the specific problem (e.g., "bipedal gait phase clock reward reinforcement learning").
- **When hitting a training plateau.** Search for how others solved similar issues (e.g., "PPO humanoid locomotion local minima double support shuffle").
- **When implementing a new algorithm or technique.** Find reference implementations on GitHub to avoid reinventing poorly.
- **When debugging unfamiliar framework behavior.** Search for Isaac Lab / Isaac Sim / RSL-RL specific issues and solutions.
- **When a clean-room design requires an unfamiliar technique** (e.g., capture point, ZMP, or gait phase clocks). Research it before adding it to the new implementation.

### How to search effectively

- Use specific technical terms, not vague queries. Good: `"Isaac Lab bipedal locomotion gait phase clock observation PPO 2025"`. Bad: `"how to make robot walk"`.
- Include year filters (`2024`, `2025`, `2026`) to get recent results.
- Search GitHub directly for reference implementations: `"humanoid locomotion reward github isaac"`.
- Search arXiv for papers: `"bipedal gait reinforcement learning arXiv 2025"`.

### What to do with findings

- **Store useful references** in a new, focused research note only when implementation work actually needs it. Include:
  - The URL
  - A 1-2 sentence summary of what is useful
  - How it applies to the current project
- **Do not blindly copy code** from external repos. Understand the approach, then implement it within the existing codebase patterns.
- **Cite the source** in the new implementation's evidence or commit message when a technique was inspired by external research.

### Key domains to search for in this repo

| Domain | Example search terms |
|--------|---------------------|
| Bipedal RL locomotion | `humanoid walking PPO reward shaping gait phase`, `sim-to-real bipedal`, `periodic reward composition` |
| Isaac Lab / Isaac Sim | `Isaac Lab custom reward`, `Isaac Lab URDF articulation`, `omni.isaac.lab locomotion` |
| URDF / robot modeling | `URDF mass inertia calculation`, `URDF actuator tuning simulation` |
| Reward design | `reward shaping bipedal locomotion 2025`, `contact-based gait reward`, `imitation learning humanoid` |
| 2D packing / nesting | `irregular cutting packing optimization`, `jagua-rs nesting`, `no-fit polygon` |
| Inverse kinematics | `IKFast analytical IK`, `TracIK numerical IK URDF` |
| SLAM / mapping | `occupancy grid mapping lidar simulation`, `ROS2 map server pgm yaml` |
