from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import omni.client
import omni.kit.commands
from omni.client import Result
from pxr import Usd


ISAAC_ROOT = Path("/home/wishai/vscode/IsaacLab/_isaac_sim")
_REGISTERED = False


def _log(message: str) -> None:
    print(f"[AVP-URDF] {message}", flush=True)


def _find_urdf_binary() -> Path:
    extscache_root = ISAAC_ROOT / "extscache"
    for extension_root in sorted(extscache_root.glob("isaacsim.asset.importer.urdf-*"), reverse=True):
        matches = sorted((extension_root / "isaacsim" / "asset" / "importer" / "urdf").glob("_urdf*.so"))
        if matches:
            return matches[0]
    raise FileNotFoundError("Could not locate Isaac Sim's native URDF importer module")


def _load_native_urdf_module():
    module = sys.modules.get("_urdf")
    if module is not None:
        _log("Reusing loaded native _urdf module")
        return module

    module_path = _find_urdf_binary()
    _log(f"Loading native _urdf module from {module_path}")
    spec = importlib.util.spec_from_file_location("_urdf", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load URDF importer module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["_urdf"] = module
    spec.loader.exec_module(module)
    _log("Loaded native _urdf module")
    return module


_urdf = _load_native_urdf_module()


class URDFCreateImportConfig(omni.kit.commands.Command):
    def do(self) -> _urdf.ImportConfig:
        return _urdf.ImportConfig()

    def undo(self) -> None:
        pass


class URDFParseText(omni.kit.commands.Command):
    def __init__(self, urdf_string: str = "", import_config: _urdf.ImportConfig = _urdf.ImportConfig()) -> None:
        self._import_config = import_config
        self._urdf_string = urdf_string
        self._urdf_interface = _urdf.acquire_urdf_interface()

    def do(self) -> _urdf.UrdfRobot:
        return self._urdf_interface.parse_string_urdf(self._urdf_string, self._import_config)

    def undo(self) -> None:
        pass


class URDFParseFile(omni.kit.commands.Command):
    def __init__(self, urdf_path: str = "", import_config: _urdf.ImportConfig = _urdf.ImportConfig()) -> None:
        self._root_path, self._filename = os.path.split(os.path.abspath(urdf_path))
        self._import_config = import_config
        self._urdf_interface = _urdf.acquire_urdf_interface()

    def do(self) -> _urdf.UrdfRobot:
        return self._urdf_interface.parse_urdf(self._root_path, self._filename, self._import_config)

    def undo(self) -> None:
        pass


class URDFImportRobot(omni.kit.commands.Command):
    def __init__(
        self,
        urdf_path: str = "",
        urdf_robot: _urdf.UrdfRobot | None = None,
        import_config: _urdf.ImportConfig = _urdf.ImportConfig(),
        dest_path: str = "",
        get_articulation_root: bool = False,
    ) -> None:
        self._urdf_path = urdf_path
        self._root_path, self._filename = os.path.split(os.path.abspath(urdf_path))
        self._urdf_robot = urdf_robot
        self._dest_path = dest_path
        self._import_config = import_config
        self._urdf_interface = _urdf.acquire_urdf_interface()
        self._get_articulation_root = get_articulation_root

    def do(self) -> str:
        if self._dest_path:
            self._dest_path = self._dest_path.replace("\\", "/")
            result = omni.client.read_file(self._dest_path)
            if result[0] != Result.OK:
                stage = Usd.Stage.CreateNew(self._dest_path)
                stage.Save()
        return self._urdf_interface.import_robot(
            self._root_path,
            self._filename,
            self._urdf_robot,
            self._import_config,
            self._dest_path,
            self._get_articulation_root,
        )

    def undo(self) -> None:
        pass


class URDFParseAndImportFile(omni.kit.commands.Command):
    def __init__(
        self,
        urdf_path: str = "",
        import_config: _urdf.ImportConfig = _urdf.ImportConfig(),
        dest_path: str = "",
        get_articulation_root: bool = False,
    ) -> None:
        self.dest_path = dest_path
        self._urdf_path = urdf_path
        self._root_path, self._filename = os.path.split(os.path.abspath(urdf_path))
        self._import_config = import_config
        self._urdf_interface = _urdf.acquire_urdf_interface()
        self._get_articulation_root = get_articulation_root

    def do(self) -> str:
        status, imported_robot = omni.kit.commands.execute(
            "URDFParseFile",
            urdf_path=self._urdf_path,
            import_config=self._import_config,
        )
        if not status:
            raise RuntimeError(f"Failed to parse URDF: {self._urdf_path}")
        if self.dest_path:
            self.dest_path = self.dest_path.replace("\\", "/")
            result = omni.client.read_file(self.dest_path)
            if result[0] != Result.OK:
                stage = Usd.Stage.CreateNew(self.dest_path)
                stage.Save()
        return self._urdf_interface.import_robot(
            self._root_path,
            self._filename,
            imported_robot,
            self._import_config,
            self.dest_path,
            self._get_articulation_root,
        )

    def undo(self) -> None:
        pass


def register_urdf_commands() -> None:
    global _REGISTERED
    if _REGISTERED:
        _log("Headless URDF commands already registered")
        return
    _log("Registering headless URDF command classes")
    omni.kit.commands.register_all_commands_in_module(__name__)
    _REGISTERED = True
    _log("Registered headless URDF command classes")
