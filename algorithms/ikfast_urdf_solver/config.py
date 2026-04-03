from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any


ALGORITHM_ROOT = Path(__file__).resolve().parent
REPO_ROOT = ALGORITHM_ROOT.parent.parent
IDENTITY_MATRIX4 = (
    (1.0, 0.0, 0.0, 0.0),
    (0.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, 0.0),
    (0.0, 0.0, 0.0, 1.0),
)


@dataclass(frozen=True, slots=True)
class BuildConfig:
    output_dir: Path
    compiler: str = "g++"
    compile_args: tuple[str, ...] = ("-O3", "-std=c++17", "-shared", "-fPIC")
    link_args: tuple[str, ...] = ("-llapack", "-lblas")
    force_rebuild: bool = False

    def resolved(self, base_dir: Path) -> "BuildConfig":
        output_dir = self.output_dir if self.output_dir.is_absolute() else (base_dir / self.output_dir)
        return replace(self, output_dir=output_dir.resolve())


@dataclass(frozen=True, slots=True)
class RobotArmConfig:
    name: str
    urdf_path: Path
    base_link: str
    tip_link: str
    ikfast_cpp_path: Path | None = None
    ikfast_library_path: Path | None = None
    free_joint_names: tuple[str, ...] = ()
    urdf_base_to_ikfast: tuple[tuple[float, float, float, float], ...] = IDENTITY_MATRIX4
    urdf_tip_to_ikfast: tuple[tuple[float, float, float, float], ...] = IDENTITY_MATRIX4
    continuous_joint_limits: tuple[float, float] = (-2.0 * math.pi, 2.0 * math.pi)
    position_tolerance: float = 1e-4
    rotation_tolerance: float = 1e-4
    max_solutions: int = 128
    build: BuildConfig | None = None

    def resolved(self, base_dir: Path | None = None) -> "RobotArmConfig":
        root = (base_dir or REPO_ROOT).resolve()
        urdf_path = self.urdf_path if self.urdf_path.is_absolute() else (root / self.urdf_path)
        ikfast_cpp_path = None
        if self.ikfast_cpp_path is not None:
            ikfast_cpp_path = self.ikfast_cpp_path if self.ikfast_cpp_path.is_absolute() else (root / self.ikfast_cpp_path)
            ikfast_cpp_path = ikfast_cpp_path.resolve()
        ikfast_library_path = None
        if self.ikfast_library_path is not None:
            ikfast_library_path = self.ikfast_library_path if self.ikfast_library_path.is_absolute() else (root / self.ikfast_library_path)
            ikfast_library_path = ikfast_library_path.resolve()
        build = self.build
        if build is None:
            build = BuildConfig(output_dir=ALGORITHM_ROOT / "outputs" / "build" / self.name)
        build = build.resolved(root)
        return replace(
            self,
            urdf_path=urdf_path.resolve(),
            ikfast_cpp_path=ikfast_cpp_path,
            ikfast_library_path=ikfast_library_path,
            build=build,
        )


def _coerce_path(base_dir: Path, value: str | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def load_config_from_json(path: str | Path) -> RobotArmConfig:
    config_path = Path(path).resolve()
    data = json.loads(config_path.read_text())
    build_data = data.get("build", {})
    build = BuildConfig(
        output_dir=Path(build_data.get("output_dir", ALGORITHM_ROOT / "outputs" / "build" / data["name"])),
        compiler=build_data.get("compiler", "g++"),
        compile_args=tuple(build_data.get("compile_args", ("-O3", "-std=c++17", "-shared", "-fPIC"))),
        link_args=tuple(build_data.get("link_args", ("-llapack", "-lblas"))),
        force_rebuild=bool(build_data.get("force_rebuild", False)),
    )
    return RobotArmConfig(
        name=data["name"],
        urdf_path=Path(data["urdf_path"]),
        base_link=data["base_link"],
        tip_link=data["tip_link"],
        ikfast_cpp_path=Path(data["ikfast_cpp_path"]) if data.get("ikfast_cpp_path") else None,
        ikfast_library_path=Path(data["ikfast_library_path"]) if data.get("ikfast_library_path") else None,
        free_joint_names=tuple(data.get("free_joint_names", ())),
        urdf_base_to_ikfast=tuple(tuple(float(value) for value in row) for row in data.get("urdf_base_to_ikfast", IDENTITY_MATRIX4)),
        urdf_tip_to_ikfast=tuple(tuple(float(value) for value in row) for row in data.get("urdf_tip_to_ikfast", IDENTITY_MATRIX4)),
        continuous_joint_limits=tuple(data.get("continuous_joint_limits", (-2.0 * math.pi, 2.0 * math.pi))),
        position_tolerance=float(data.get("position_tolerance", 1e-4)),
        rotation_tolerance=float(data.get("rotation_tolerance", 1e-4)),
        max_solutions=int(data.get("max_solutions", 128)),
        build=build,
    ).resolved(config_path.parent)


def get_builtin_config(name: str) -> RobotArmConfig:
    configs = _builtin_configs()
    try:
        return configs[name].resolved(REPO_ROOT)
    except KeyError as exc:
        raise KeyError(f"Unknown IKFast config {name!r}. Available: {sorted(configs)}") from exc


def list_builtin_configs() -> list[str]:
    return sorted(_builtin_configs())


def _builtin_configs() -> dict[str, RobotArmConfig]:
    return {
        "ur5_helper_sample": RobotArmConfig(
            name="ur5_helper_sample",
            urdf_path=ALGORITHM_ROOT / "inputs" / "ur5" / "ur5_kinematics.urdf",
            base_link="base_link",
            tip_link="tool0",
            ikfast_cpp_path=REPO_ROOT / "helper_repos" / "ikfastpy" / "ikfast61.cpp",
            free_joint_names=(),
            urdf_base_to_ikfast=(
                (-1.0, 0.0, 0.0, 0.0),
                (0.0, -1.0, 0.0, 0.0),
                (0.0, 0.0, 1.0, 0.0),
                (0.0, 0.0, 0.0, 1.0),
            ),
            continuous_joint_limits=(-2.0 * math.pi, 2.0 * math.pi),
            position_tolerance=1e-4,
            rotation_tolerance=1e-4,
            max_solutions=128,
            build=BuildConfig(output_dir=ALGORITHM_ROOT / "outputs" / "build" / "ur5_helper_sample"),
        ),
    }


def to_jsonable(config: RobotArmConfig) -> dict[str, Any]:
    resolved = config.resolved(REPO_ROOT)
    return {
        "name": resolved.name,
        "urdf_path": str(resolved.urdf_path),
        "base_link": resolved.base_link,
        "tip_link": resolved.tip_link,
        "ikfast_cpp_path": None if resolved.ikfast_cpp_path is None else str(resolved.ikfast_cpp_path),
        "ikfast_library_path": None if resolved.ikfast_library_path is None else str(resolved.ikfast_library_path),
        "free_joint_names": list(resolved.free_joint_names),
        "urdf_base_to_ikfast": [list(row) for row in resolved.urdf_base_to_ikfast],
        "urdf_tip_to_ikfast": [list(row) for row in resolved.urdf_tip_to_ikfast],
        "continuous_joint_limits": list(resolved.continuous_joint_limits),
        "position_tolerance": resolved.position_tolerance,
        "rotation_tolerance": resolved.rotation_tolerance,
        "max_solutions": resolved.max_solutions,
        "build": {
            "output_dir": str(resolved.build.output_dir if resolved.build is not None else ""),
            "compiler": resolved.build.compiler if resolved.build is not None else "g++",
            "compile_args": list(resolved.build.compile_args if resolved.build is not None else ()),
            "link_args": list(resolved.build.link_args if resolved.build is not None else ()),
            "force_rebuild": bool(resolved.build.force_rebuild if resolved.build is not None else False),
        },
    }
