from __future__ import annotations

import ctypes
import subprocess
from dataclasses import dataclass
from pathlib import Path
from string import Template

import numpy as np

from .config import BuildConfig, RobotArmConfig


WRAPPER_TEMPLATE = Template(
    r"""
#include <algorithm>
#include <vector>

#define IKFAST_HAS_LIBRARY
#define IKFAST_NO_MAIN
#include "$ikfast_cpp_path"

extern "C" {

int ikfast_wrapper_get_num_joints() {
    return GetNumJoints();
}

int ikfast_wrapper_get_num_free_parameters() {
    return GetNumFreeParameters();
}

int ikfast_wrapper_get_free_parameters(int* out_indices, int max_count) {
    int count = GetNumFreeParameters();
    int* free_indices = GetFreeParameters();
    if (free_indices == nullptr || out_indices == nullptr || max_count <= 0) {
        return count;
    }
    int copy_count = std::min(count, max_count);
    for (int i = 0; i < copy_count; ++i) {
        out_indices[i] = free_indices[i];
    }
    return count;
}

void ikfast_wrapper_compute_fk(const double* joints, double* out_translation, double* out_rotation) {
    ComputeFk(joints, out_translation, out_rotation);
}

int ikfast_wrapper_compute_ik(
    const double* in_translation,
    const double* in_rotation,
    const double* in_free_parameters,
    double* out_solutions,
    int max_solutions
) {
    ikfast::IkSolutionList<IkReal> solutions;
    bool success = ComputeIk(in_translation, in_rotation, in_free_parameters, solutions);
    if (!success) {
        return 0;
    }

    const int total_solutions = static_cast<int>(solutions.GetNumSolutions());
    const int joint_count = GetNumJoints();
    if (out_solutions == nullptr || max_solutions <= 0) {
        return total_solutions;
    }

    const int copy_count = std::min(total_solutions, max_solutions);
    std::vector<IkReal> joint_values(joint_count);
    for (int solution_index = 0; solution_index < copy_count; ++solution_index) {
        const ikfast::IkSolutionBase<IkReal>& solution = solutions.GetSolution(solution_index);
        const std::vector<int>& free_indices = solution.GetFree();
        std::vector<IkReal> free_values(free_indices.size(), IkReal(0));
        if (in_free_parameters != nullptr) {
            for (std::size_t free_index = 0; free_index < free_indices.size(); ++free_index) {
                free_values[free_index] = in_free_parameters[free_indices[free_index]];
            }
        }
        solution.GetSolution(&joint_values[0], free_values.empty() ? nullptr : &free_values[0]);
        for (int joint_index = 0; joint_index < joint_count; ++joint_index) {
            out_solutions[(solution_index * joint_count) + joint_index] = joint_values[joint_index];
        }
    }
    return total_solutions;
}

}
"""
)


DoublePtr = ctypes.POINTER(ctypes.c_double)
IntPtr = ctypes.POINTER(ctypes.c_int)


@dataclass(slots=True)
class IkFastLibrary:
    library_path: Path
    _lib: ctypes.CDLL
    num_joints: int
    num_free_parameters: int
    free_parameter_indices: tuple[int, ...]

    @classmethod
    def from_config(cls, config: RobotArmConfig) -> "IkFastLibrary":
        resolved = config.resolved()
        if resolved.ikfast_library_path is not None and resolved.ikfast_library_path.exists():
            library_path = resolved.ikfast_library_path
        else:
            if resolved.ikfast_cpp_path is None:
                raise ValueError(f"Config {resolved.name!r} must provide either ikfast_cpp_path or ikfast_library_path.")
            library_path = build_ikfast_library(resolved.ikfast_cpp_path, resolved.build, library_name=resolved.name)
        return load_ikfast_library(library_path)

    def compute_fk(self, joint_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        values = np.asarray(joint_values, dtype=np.float64)
        if values.shape != (self.num_joints,):
            raise ValueError(f"Expected fk input shape {(self.num_joints,)}, received {values.shape}.")
        translation = np.empty(3, dtype=np.float64)
        rotation = np.empty(9, dtype=np.float64)
        self._lib.ikfast_wrapper_compute_fk(
            values.ctypes.data_as(DoublePtr),
            translation.ctypes.data_as(DoublePtr),
            rotation.ctypes.data_as(DoublePtr),
        )
        return translation, rotation.reshape(3, 3)

    def compute_ik(
        self,
        translation: np.ndarray,
        rotation: np.ndarray,
        *,
        free_parameters: np.ndarray | None = None,
        max_solutions: int = 128,
    ) -> np.ndarray:
        translation_arr = np.asarray(translation, dtype=np.float64)
        rotation_arr = np.asarray(rotation, dtype=np.float64)
        if translation_arr.shape != (3,):
            raise ValueError(f"Expected translation shape (3,), received {translation_arr.shape}.")
        if rotation_arr.shape != (3, 3):
            raise ValueError(f"Expected rotation shape (3, 3), received {rotation_arr.shape}.")
        free_arr = np.asarray(free_parameters if free_parameters is not None else (), dtype=np.float64)
        if free_arr.shape != (self.num_free_parameters,):
            raise ValueError(
                f"Expected {self.num_free_parameters} free parameters, received shape {free_arr.shape}."
            )
        capacity = max(1, int(max_solutions))
        out = np.empty(capacity * self.num_joints, dtype=np.float64)
        count = int(
            self._lib.ikfast_wrapper_compute_ik(
                translation_arr.ctypes.data_as(DoublePtr),
                rotation_arr.reshape(-1).ctypes.data_as(DoublePtr),
                None if self.num_free_parameters == 0 else free_arr.ctypes.data_as(DoublePtr),
                out.ctypes.data_as(DoublePtr),
                capacity,
            )
        )
        if count == 0:
            return np.empty((0, self.num_joints), dtype=np.float64)
        if count > capacity:
            out = np.empty(count * self.num_joints, dtype=np.float64)
            count = int(
                self._lib.ikfast_wrapper_compute_ik(
                    translation_arr.ctypes.data_as(DoublePtr),
                    rotation_arr.reshape(-1).ctypes.data_as(DoublePtr),
                    None if self.num_free_parameters == 0 else free_arr.ctypes.data_as(DoublePtr),
                    out.ctypes.data_as(DoublePtr),
                    count,
                )
            )
        return out[: count * self.num_joints].reshape(count, self.num_joints)


def build_ikfast_library(ikfast_cpp_path: Path, build: BuildConfig, *, library_name: str) -> Path:
    ikfast_cpp_path = ikfast_cpp_path.resolve()
    output_dir = build.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    wrapper_cpp_path = output_dir / f"{library_name}_ikfast_wrapper.cpp"
    library_path = output_dir / f"lib{library_name}_ikfast.so"
    command_path = output_dir / f"{library_name}_ikfast_build_command.txt"
    source = WRAPPER_TEMPLATE.substitute(ikfast_cpp_path=str(ikfast_cpp_path))
    if not wrapper_cpp_path.exists() or wrapper_cpp_path.read_text() != source:
        wrapper_cpp_path.write_text(source)
    cmd = [build.compiler, *build.compile_args, "-o", str(library_path), str(wrapper_cpp_path), *build.link_args]
    needs_rebuild = build.force_rebuild or not library_path.exists()
    if not needs_rebuild:
        needs_rebuild = wrapper_cpp_path.stat().st_mtime > library_path.stat().st_mtime
    if not needs_rebuild:
        needs_rebuild = ikfast_cpp_path.stat().st_mtime > library_path.stat().st_mtime
    if not needs_rebuild:
        needs_rebuild = not command_path.exists() or command_path.read_text() != " ".join(cmd)
    if needs_rebuild:
        completed = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if completed.returncode != 0:
            raise RuntimeError(
                "Failed to build IKFast wrapper library.\n"
                f"Command: {' '.join(cmd)}\n"
                f"stdout:\n{completed.stdout}\n"
                f"stderr:\n{completed.stderr}"
            )
        command_path.write_text(" ".join(cmd))
    return library_path


def load_ikfast_library(library_path: str | Path) -> IkFastLibrary:
    path = Path(library_path).resolve()
    lib = ctypes.CDLL(str(path))
    lib.ikfast_wrapper_get_num_joints.restype = ctypes.c_int
    lib.ikfast_wrapper_get_num_free_parameters.restype = ctypes.c_int
    lib.ikfast_wrapper_get_free_parameters.argtypes = [IntPtr, ctypes.c_int]
    lib.ikfast_wrapper_get_free_parameters.restype = ctypes.c_int
    lib.ikfast_wrapper_compute_fk.argtypes = [DoublePtr, DoublePtr, DoublePtr]
    lib.ikfast_wrapper_compute_fk.restype = None
    lib.ikfast_wrapper_compute_ik.argtypes = [DoublePtr, DoublePtr, DoublePtr, DoublePtr, ctypes.c_int]
    lib.ikfast_wrapper_compute_ik.restype = ctypes.c_int

    num_joints = int(lib.ikfast_wrapper_get_num_joints())
    num_free_parameters = int(lib.ikfast_wrapper_get_num_free_parameters())
    if num_free_parameters > 0:
        indices = np.empty(num_free_parameters, dtype=np.int32)
        lib.ikfast_wrapper_get_free_parameters(indices.ctypes.data_as(IntPtr), num_free_parameters)
        free_parameter_indices = tuple(int(value) for value in indices.tolist())
    else:
        free_parameter_indices = ()
    return IkFastLibrary(
        library_path=path,
        _lib=lib,
        num_joints=num_joints,
        num_free_parameters=num_free_parameters,
        free_parameter_indices=free_parameter_indices,
    )
