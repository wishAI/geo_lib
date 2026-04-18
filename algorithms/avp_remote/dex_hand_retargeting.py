from __future__ import annotations

import atexit
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from avp_config import AVP_TRACKING_ROTATE_XYZ, AVP_TRACKING_SCALE_XYZ, AVP_TRACKING_TRANSLATE_XYZ
from avp_tracking_schema import HAND_JOINT_NAMES
from avp_transform_utils import TransformOptions, build_xyz_transform, to_usd_world


TRACKING_OPTIONS = TransformOptions(
    column_major=False,
    pretransform=build_xyz_transform(
        AVP_TRACKING_ROTATE_XYZ,
        AVP_TRACKING_TRANSLATE_XYZ,
        scale_xyz=AVP_TRACKING_SCALE_XYZ,
    ).T,
    posttransform=None,
)
HAND_JOINT_INDEX = {name: index for index, name in enumerate(HAND_JOINT_NAMES)}
_SANITIZED_ENV_KEYS = (
    "PYTHONHOME",
    "PYTHONPATH",
    "PYTHONSTARTUP",
    "PYTHONUSERBASE",
    "PYTHONEXECUTABLE",
    "__PYVENV_LAUNCHER__",
    "VIRTUAL_ENV",
    "CONDA_PREFIX",
    "CONDA_DEFAULT_ENV",
    "CONDA_PROMPT_MODIFIER",
)


def _build_worker_env(helper_python: Path, *, base_env: dict[str, str] | None = None) -> dict[str, str]:
    env = dict(os.environ if base_env is None else base_env)
    for key in _SANITIZED_ENV_KEYS:
        env.pop(key, None)

    helper_root = helper_python.parent.parent
    helper_bin = str(helper_python.parent)
    existing_path = env.get("PATH", "")
    env["PATH"] = helper_bin if not existing_path else f"{helper_bin}{os.pathsep}{existing_path}"
    env["VIRTUAL_ENV"] = str(helper_root)
    env["PYTHONNOUSERSITE"] = "1"
    return env


class DexHandRetargetingClient:
    def __init__(
        self,
        *,
        helper_python: Path,
        landau_urdf_path: Path,
        snapshot_path: Path,
        baseline_urdf_path: Path | None = None,
    ) -> None:
        self.helper_python = Path(helper_python).expanduser()
        if not self.helper_python.is_absolute():
            self.helper_python = self.helper_python.absolute()
        if not self.helper_python.exists():
            raise FileNotFoundError(f"Dex hand helper Python was not found: {self.helper_python}")

        self.worker_script = Path(__file__).resolve().with_name("dex_hand_retarget_worker.py")
        command = [
            str(self.helper_python),
            "-I",
            str(self.worker_script),
            "--landau-urdf",
            str(Path(landau_urdf_path).expanduser().resolve()),
            "--snapshot-path",
            str(Path(snapshot_path).expanduser().resolve()),
        ]
        if baseline_urdf_path is not None and Path(baseline_urdf_path).exists():
            command.extend(["--baseline-urdf", str(Path(baseline_urdf_path).expanduser().resolve())])

        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=sys.stderr,
            text=True,
            bufsize=1,
            env=_build_worker_env(self.helper_python),
        )
        self._closed = False
        atexit.register(self.close)
        response = self._read_response()
        if response.get("status") != "ready":
            self.close()
            raise RuntimeError(f"Dex hand retargeting worker failed to start: {response}")

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._send({"command": "close"})
        except Exception:
            pass
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=2.0)
            except Exception:
                self.process.kill()

    def reset(self) -> None:
        self._send({"command": "reset"})
        response = self._read_response()
        if response.get("status") != "ok":
            raise RuntimeError(f"Dex hand retargeting reset failed: {response}")

    def retarget_frame(self, frame) -> dict[str, dict[str, float]]:
        request = {"command": "retarget"}
        for side in ("left", "right"):
            local_positions = _frame_hand_local_positions(frame, side)
            if local_positions is not None:
                request[f"{side}_local_positions"] = local_positions.tolist()
        self._send(request)
        response = self._read_response()
        if response.get("status") != "ok":
            raise RuntimeError(f"Dex hand retargeting failed: {response}")
        result = response.get("result", {})
        return {
            "landau": {name: float(value) for name, value in result.get("landau", {}).items()},
            "h1_2": {name: float(value) for name, value in result.get("h1_2", {}).items()},
        }

    def _send(self, payload: dict[str, object]) -> None:
        if self.process.stdin is None:
            raise RuntimeError("Dex hand retargeting worker stdin is not available")
        self.process.stdin.write(json.dumps(payload) + "\n")
        self.process.stdin.flush()

    def _read_response(self) -> dict[str, object]:
        if self.process.stdout is None:
            raise RuntimeError("Dex hand retargeting worker stdout is not available")
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError("Dex hand retargeting worker exited unexpectedly")
        return json.loads(line)


def _tracking_stack_world(stack) -> np.ndarray | None:
    if stack is None:
        return None
    return np.stack(
        [np.asarray(to_usd_world(mat, options=TRACKING_OPTIONS), dtype=float) for mat in stack],
        axis=0,
    )


def _frame_hand_local_positions(frame, side: str) -> np.ndarray | None:
    stack = _tracking_stack_world(frame.get(f"{side}_arm"))
    if stack is None or len(stack) != len(HAND_JOINT_NAMES):
        return None
    wrist_inv = np.linalg.inv(stack[HAND_JOINT_INDEX["wrist"]])
    return np.stack([(wrist_inv @ stack[index])[:3, 3] for index in range(len(HAND_JOINT_NAMES))], axis=0)
