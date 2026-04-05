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

### Isaac Sim / Isaac Lab environment

```bash
/home/wishai/vscode/IsaacLab/isaaclab.sh -p <script_or_module>
```

- Isaac Lab has its **own Python environment** managed by `isaaclab.sh -p`. Do NOT use `ptenv` or `python3` directly for Isaac tasks.
- All `urdf_learn_wasd_walk` scripts (train, play, teleop, validate, smoke_test) must be launched through `isaaclab.sh -p`.
- Isaac Lab location: `/home/wishai/vscode/IsaacLab/`
- The `-p` flag runs a Python script or module inside Isaac Lab's managed conda/pip environment.

### When to use which

| Task | Environment |
|------|-------------|
| Pure Python tests (`pytest`) | `pyenv activate ptenv && python3 -m pytest ...` |
| MuJoCo scene builder / SLAM mapping | `pyenv activate ptenv` |
| Isaac Lab training / play / teleop / validation | `isaaclab.sh -p -m algorithms.urdf_learn_wasd_walk.scripts.<cmd>` |
| Isaac Lab smoke tests | `isaaclab.sh -p -m algorithms.urdf_learn_wasd_walk.scripts.smoke_test` |
| ROS2 CLI tools | source `/opt/ros/humble/setup.bash` first |

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

Bad: "Fix the walking bug." Good: "The Landau Stage A training produces wide shuffles instead of alternating steps. Training history is in `algorithms/urdf_learn_wasd_walk/train_history.md`. The latest checkpoint `model_3348.pt` passes anti-crawl but fails single-support ratio (0.024). Read `custom_rewards.py` and `landau_env_cfg.py` to propose specific reward weight changes that could improve single-support ratio."

### When NOT to use sub-agents

- Simple file reads or grep searches. Use Glob/Grep/Read directly.
- Tasks with 1-3 steps that you already understand.
- When the user didn't ask for sub-agents and the task is straightforward.

---

## 4. Training Tasks: Protocol

Training RL policies (especially `urdf_learn_wasd_walk`) is the most common long-running task in this repo.

### Before training

1. **Read `train_history.md`** for the algorithm you are working on. It is the source of truth for what has been tried, what works, and what is broken.
2. **Run pure Python tests first.** They are fast and catch config/import errors before wasting GPU time.
   ```bash
   pyenv activate ptenv && python3 -m pytest algorithms/urdf_learn_wasd_walk/tests -q
   ```
3. **Run a 2-iteration smoke test** to verify the pipeline before a real run.
   ```bash
   /home/wishai/vscode/IsaacLab/isaaclab.sh -p -m algorithms.urdf_learn_wasd_walk.scripts.train \
     --robot landau --stage fwd_only --headless --max_iterations 2
   ```
4. **Name your run.** Always pass `--run_name <descriptive_name>` so future agents can identify it.

### During training

- Training runs are hours-long. Launch them in the background.
- Do NOT trust PPO reward alone. Always validate with `validate_walk.py`.
- If you need to poll, 10-20 minute intervals are fine.

### After training

1. **Run strict validation** on any promising checkpoint:
   ```bash
   /home/wishai/vscode/IsaacLab/isaaclab.sh -p -m algorithms.urdf_learn_wasd_walk.scripts.validate_walk \
     --robot landau --stage fwd_only --headless \
     --experiment_name geo_landau_fwd_only \
     --load_run <RUN_NAME> \
     --checkpoint model_<N>.pt
   ```
2. **Update `train_history.md`** with:
   - The run command used
   - Key validation metrics
   - What improved and what regressed
   - Whether the checkpoint is promotable or just a reference
3. **Never claim a checkpoint passes unless `validate_walk.py` agrees.**

### Resume gotcha

`--max_iterations` during resume means "train for N more iterations from the loaded checkpoint", NOT "train until iteration N". A resume from iteration 3050 with `--max_iterations 200` trains up to ~3250.

---

## 5. Key Repository Structure

```
geo_lib/
  CONTEXT.md                  -- file handoff rules, environment notes
  agent_guide.md              -- this file
  algorithms/
    svg_scene_builder/        -- SVG to MuJoCo scene
    simple_auto_slam_mapping/ -- route planning, lidar, occupancy mapping
    urdf_learn_wasd_walk/     -- RL locomotion (Isaac Lab)
      train_history.md        -- THE source of truth for training
      README.md               -- commands and usage
      README.txt              -- operator checkpoint summary
      custom_rewards.py       -- reward functions
      landau_env_cfg.py       -- Landau environment config
      robot_specs.py          -- robot metadata
      isaac_workflow.py       -- task/env registration
      scripts/                -- train, play, teleop, validate_walk, smoke_test
      tests/                  -- fast pure-Python tests
      inputs/                 -- copied URDF assets
      agents/                 -- PPO configs
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
- Checkpoint compatibility: Landau staged setup uses action dim 29 / observation dim 99. Older checkpoints have different shapes and will crash with `size mismatch for ActorCritic`.
- Always pass `--stage fwd_only` (or the appropriate stage) for Landau.

### Landau-specific

- Landau body forward axis is body `+Y`, not `+X`.
- Semantic forward `(vx, 0, 0)` is remapped to env `(0, vy, 0)` by `command_frame.py`.
- Total URDF mass is ~1.63 kg (unrealistically low). This is a known issue documented in `train_history.md`.
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
4. **`train_history.md`** (for training tasks) -- what was tried, current truth
5. **`problems/` directory** -- investigation notes, fix plans
6. **`internet_walk_reward_notes.md`** (for reward design) -- external references

---

## 8. Behavioral Guidelines for Agents

### Do

- Read before writing. Understand existing code before modifying it.
- Run tests after making changes.
- Update `train_history.md` after any training run.
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
2. Check if there is a relevant entry in `train_history.md` or `problems/`.
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
- **When the existing `internet_walk_reward_notes.md` or `train_history.md` mentions a technique you are unfamiliar with** (e.g., "capture point reward", "ZMP", "gait phase clock"). Search to understand it before implementing.

### How to search effectively

- Use specific technical terms, not vague queries. Good: `"Isaac Lab bipedal locomotion gait phase clock observation PPO 2025"`. Bad: `"how to make robot walk"`.
- Include year filters (`2024`, `2025`, `2026`) to get recent results.
- Search GitHub directly for reference implementations: `"humanoid locomotion reward github isaac"`.
- Search arXiv for papers: `"bipedal gait reinforcement learning arXiv 2025"`.

### What to do with findings

- **Store useful references** in the algorithm's research notes file (e.g., `internet_walk_reward_notes.md` for `urdf_learn_wasd_walk`). Include:
  - The URL
  - A 1-2 sentence summary of what is useful
  - How it applies to the current project
- **Do not blindly copy code** from external repos. Understand the approach, then implement it within the existing codebase patterns.
- **Cite the source** in commit messages or `train_history.md` when a technique was inspired by external research.

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
