# IKFast URDF Solver

This algorithm folder wraps pre-generated IKFast C++ solvers behind a small Python API that behaves like `pytracik`, but returns an explicit bounded result:

- `ok` with joint values, position error, rotation error, and requested bounds
- `no` when no valid solution survives IK, joint-limit filtering, and error checks

The folder is self-contained. It does not import code from other `algorithms/` folders.

## What It Supports

- any URDF arm chain, as long as you provide:
  - a URDF path
  - base and tip link names
  - a generated IKFast C++ file or a previously built shared library
- optional free-joint handling for IKFast solvers with free parameters
- headless MuJoCo validation on a sanitized kinematics-only URDF
- speed and accuracy comparison against `pytracik`
- MuJoCo screenshot generation for side-by-side pose comparisons

## Install

From the repo root on another machine:

```bash
python -m pip install -e algorithms/ikfast_urdf_solver
```

If you also want the benchmark and screenshot tools:

```bash
python -m pip install -e "algorithms/ikfast_urdf_solver[benchmark,viz]"
```

The installed package keeps the import path:

```python
from algorithms.ikfast_urdf_solver import IkFastSolver, get_builtin_config
```

For the `pytracik` comparison path, also install the helper repo checkout if you keep the same repo layout:

```bash
python -m pip install -e helper_repos/pytracik
```

For headless MuJoCo rendering on Linux, use `MUJOCO_GL=egl`. The screenshot CLI defaults to `egl` automatically when no `DISPLAY` is present.

If you want a one-command bootstrap from the repo root:

```bash
algorithms/ikfast_urdf_solver/scripts/install_local_env.sh --with-tools
```

## Built-In Sample

The built-in sample config is `ur5_helper_sample`.

It expects:

- `algorithms/ikfast_urdf_solver/inputs/ur5/ur5_kinematics.urdf`
- `helper_repos/ikfastpy/ikfast61.cpp`

Refresh the UR5 sample URDF from the official ROS `ur_description` package:

```bash
pyenv activate ptenv
python algorithms/ikfast_urdf_solver/scripts/prepare_ur5_sample.py
```

## Python Usage

```python
import numpy as np
from algorithms.ikfast_urdf_solver import IkFastSolver, get_builtin_config

solver = IkFastSolver(get_builtin_config("ur5_helper_sample"))
target_position = np.array([0.3, -0.2, 0.5], dtype=np.float64)
target_rotation = np.eye(3, dtype=np.float64)
seed = np.zeros(solver.library.num_joints, dtype=np.float64)

result = solver.ik(target_position, target_rotation, seed_joint_values=seed)
print(result.to_dict())
```

If a solution is not found, `result.status` is `"no"`.

If you want the pytracik-style bare joint vector, call `solver.ik_values(...)`.

The structured result looks like:

```python
{
    "status": "ok",
    "joint_values": [...],
    "position_error": 2.1e-08,
    "rotation_error": 5.6e-08,
    "position_tolerance": 1e-04,
    "rotation_tolerance": 1e-04,
}
```

When no candidate stays inside the configured limits and error bounds, the API returns:

```python
{"status": "no"}
```

The installed CLI entry points are:

- `ikfast-urdf-prepare-ur5-sample`
- `ikfast-urdf-benchmark`
- `ikfast-urdf-render-pose-cases`

## Benchmark

```bash
pyenv activate ptenv
python -m algorithms.ikfast_urdf_solver.benchmark --config ur5_helper_sample --samples 100
```

The benchmark:

- strips the URDF for MuJoCo loading
- samples joint configurations from URDF limits
- uses MuJoCo FK as the target pose source
- solves with IKFast and optionally `pytracik`
- writes a JSON summary under `algorithms/ikfast_urdf_solver/outputs/`

## MuJoCo Screenshots

Generate 5 MuJoCo screenshots comparing the same target pose solved by IKFast and `pytracik`:

```bash
pyenv activate ptenv
python -m algorithms.ikfast_urdf_solver.render_pose_cases \
  --config ur5_helper_sample \
  --cases 5 \
  --seed-mode zero
```

This writes:

- `outputs/pose_renders/case_01.png` ... `case_05.png`
- `outputs/pose_renders/contact_sheet.png`
- `outputs/pose_renders/pose_cases_summary.json`

Each image is a MuJoCo camera screenshot with:

- blue arm: IKFast
- orange arm: `pytracik`
- green marker: target end-effector position

For a fully headless shell, the explicit form is:

```bash
pyenv activate ptenv
MUJOCO_GL=egl python -m algorithms.ikfast_urdf_solver.render_pose_cases \
  --config ur5_helper_sample \
  --cases 5 \
  --seed-mode zero
```

## Config

Edit `algorithms/ikfast_urdf_solver/config.py` or provide a JSON config with:

- `name`
- `urdf_path`
- `base_link`
- `tip_link`
- `ikfast_cpp_path` or `ikfast_library_path`
- optional `free_joint_names`
- optional tolerances and build settings
