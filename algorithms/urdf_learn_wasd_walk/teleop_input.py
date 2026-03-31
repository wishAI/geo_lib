from __future__ import annotations

import numpy as np
import weakref
from collections.abc import Callable

import carb
import omni


class WasdSe2Keyboard:
    """Keyboard controller for ``(v_x, v_y, yaw_rate)`` commands using WASD plus Q/E."""

    def __init__(
        self,
        v_x_sensitivity: float = 0.8,
        v_y_sensitivity: float = 0.5,
        omega_z_sensitivity: float = 1.0,
        debug_print: bool = True,
    ):
        self.v_x_sensitivity = v_x_sensitivity
        self.v_y_sensitivity = v_y_sensitivity
        self.omega_z_sensitivity = omega_z_sensitivity
        self.debug_print = debug_print
        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()
        self._keyboard_sub = self._input.subscribe_to_keyboard_events(
            self._keyboard,
            lambda event, *args, obj=weakref.proxy(self): obj._on_keyboard_event(event, *args),
        )
        self._base_command = np.zeros(3, dtype=np.float32)
        self._additional_callbacks: dict[str, Callable[[], None]] = {}
        self._create_key_bindings()

    def __del__(self):
        keyboard_sub = getattr(self, "_keyboard_sub", None)
        if keyboard_sub is None:
            return
        self._input.unsubscribe_from_keyboard_events(self._keyboard, keyboard_sub)
        self._keyboard_sub = None

    def __str__(self) -> str:
        return (
            "WASD Keyboard Controller\n"
            "\tW / S or Up / Down or Numpad 8 / 2: forward / backward\n"
            "\tA / D or Left / Right or Numpad 4 / 6: left / right strafe\n"
            "\tQ / E or Z / X or Numpad 7 / 9: yaw left / yaw right\n"
            "\tL: zero the command"
        )

    def reset(self) -> None:
        self._base_command.fill(0.0)

    def add_callback(self, key: str, func: Callable[[], None]) -> None:
        self._additional_callbacks[key] = func

    def advance(self) -> np.ndarray:
        return self._base_command

    @staticmethod
    def _normalize_event_name(value: object) -> str:
        raw_value = getattr(value, "name", value)
        return str(raw_value).split(".")[-1].upper()

    def _on_keyboard_event(self, event, *args, **kwargs):
        key_name = self._normalize_event_name(getattr(event, "input", ""))
        event_name = self._normalize_event_name(getattr(event, "type", ""))
        if self.debug_print and key_name in self._DEBUG_KEYS:
            print(f"[TELEOP] key_event type={event_name} key={key_name}", flush=True)
        if event_name == "KEY_PRESS":
            if key_name == "L":
                self.reset()
                if self.debug_print:
                    print(f"[TELEOP] command reset -> {self._base_command.tolist()}", flush=True)
            elif key_name in self._INPUT_KEY_MAPPING:
                self._base_command += self._INPUT_KEY_MAPPING[key_name]
                if self.debug_print:
                    print(f"[TELEOP] command -> {self._base_command.tolist()}", flush=True)
            if key_name in self._additional_callbacks:
                self._additional_callbacks[key_name]()
        if event_name == "KEY_RELEASE":
            if key_name in self._INPUT_KEY_MAPPING:
                self._base_command -= self._INPUT_KEY_MAPPING[key_name]
                if self.debug_print:
                    print(f"[TELEOP] command -> {self._base_command.tolist()}", flush=True)
        return True

    def _create_key_bindings(self) -> None:
        self._INPUT_KEY_MAPPING = {
            "W": np.asarray([1.0, 0.0, 0.0]) * self.v_x_sensitivity,
            "S": np.asarray([-1.0, 0.0, 0.0]) * self.v_x_sensitivity,
            "A": np.asarray([0.0, 1.0, 0.0]) * self.v_y_sensitivity,
            "D": np.asarray([0.0, -1.0, 0.0]) * self.v_y_sensitivity,
            "UP": np.asarray([1.0, 0.0, 0.0]) * self.v_x_sensitivity,
            "DOWN": np.asarray([-1.0, 0.0, 0.0]) * self.v_x_sensitivity,
            "LEFT": np.asarray([0.0, 1.0, 0.0]) * self.v_y_sensitivity,
            "RIGHT": np.asarray([0.0, -1.0, 0.0]) * self.v_y_sensitivity,
            "NUMPAD_8": np.asarray([1.0, 0.0, 0.0]) * self.v_x_sensitivity,
            "NUMPAD_2": np.asarray([-1.0, 0.0, 0.0]) * self.v_x_sensitivity,
            "NUMPAD_4": np.asarray([0.0, 1.0, 0.0]) * self.v_y_sensitivity,
            "NUMPAD_6": np.asarray([0.0, -1.0, 0.0]) * self.v_y_sensitivity,
            "Q": np.asarray([0.0, 0.0, 1.0]) * self.omega_z_sensitivity,
            "E": np.asarray([0.0, 0.0, -1.0]) * self.omega_z_sensitivity,
            "Z": np.asarray([0.0, 0.0, 1.0]) * self.omega_z_sensitivity,
            "X": np.asarray([0.0, 0.0, -1.0]) * self.omega_z_sensitivity,
            "NUMPAD_7": np.asarray([0.0, 0.0, 1.0]) * self.omega_z_sensitivity,
            "NUMPAD_9": np.asarray([0.0, 0.0, -1.0]) * self.omega_z_sensitivity,
        }
        self._DEBUG_KEYS = set(self._INPUT_KEY_MAPPING) | {"L"}
