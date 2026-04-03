from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
import sys

import mujoco

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from algorithms.ikfast_urdf_solver.config import ALGORITHM_ROOT, REPO_ROOT as PACKAGE_REPO_ROOT
from algorithms.ikfast_urdf_solver.urdf_utils import sanitize_urdf_for_mujoco


def main() -> None:
    ur_description_prefix = PACKAGE_REPO_ROOT / "helper_repos" / "ur_description_pkg" / "opt" / "ros" / "humble"
    xacro_dist_path = PACKAGE_REPO_ROOT / "helper_repos" / "xacro_pkg" / "opt" / "ros" / "humble" / "local" / "lib" / "python3.10" / "dist-packages"
    xacro_file = ur_description_prefix / "share" / "ur_description" / "urdf" / "ur.urdf.xacro"
    if not xacro_file.exists():
        raise FileNotFoundError(
            "Missing unpacked ur_description package. "
            "Expected helper_repos/ur_description_pkg/... after local apt download and dpkg-deb extraction."
        )

    output_path = ALGORITHM_ROOT / "inputs" / "ur5" / "ur5_kinematics.urdf"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="ur5_xacro_") as temp_dir:
        temp_dir_path = Path(temp_dir)
        raw_urdf_path = temp_dir_path / "ur5_raw.urdf"
        command = f"""
source /opt/ros/humble/setup.zsh >/dev/null 2>&1
export AMENT_PREFIX_PATH="{ur_description_prefix}:$AMENT_PREFIX_PATH"
export CMAKE_PREFIX_PATH="$AMENT_PREFIX_PATH"
export PYTHONPATH="{xacro_dist_path}:$PYTHONPATH"
xacro "{xacro_file}" ur_type:=ur5 name:=ur5_helper_sample > "{raw_urdf_path}"
"""
        completed = subprocess.run(
            ["zsh", "-lc", command],
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "Failed to generate the UR5 URDF from the official ROS description.\n"
                f"stdout:\n{completed.stdout}\n"
                f"stderr:\n{completed.stderr}"
            )
        sanitize_urdf_for_mujoco(raw_urdf_path, output_path)

    mujoco.MjModel.from_xml_path(str(output_path))
    print(output_path)


if __name__ == "__main__":
    main()
