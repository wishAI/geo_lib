#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

python -m pip install -e "${repo_root}/algorithms/ikfast_urdf_solver"

if [[ "${1:-}" == "--with-tools" ]]; then
  python -m pip install -e "${repo_root}/algorithms/ikfast_urdf_solver[benchmark,viz]"

  if [[ -d "${repo_root}/helper_repos/pytracik" ]]; then
    python -m pip install -e "${repo_root}/helper_repos/pytracik"
  fi
fi

cat <<EOF
Installed geo-lib-ikfast-urdf-solver from:
  ${repo_root}/algorithms/ikfast_urdf_solver

Optional tool mode:
  algorithms/ikfast_urdf_solver/scripts/install_local_env.sh --with-tools

Headless MuJoCo render hint:
  export MUJOCO_GL=egl
EOF
