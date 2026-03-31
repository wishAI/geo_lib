from __future__ import annotations

import numpy as np
import weakref
from collections.abc import Callable

import carb
import omni


class WasdSe2Keyboard:
    """Keyboard controller for ``(v_x, v_y, yaw_rate)`` commands using WASD plus Q/E."""

    def __init__(self, v_x_sensitivity: float = 0.8, v_y_sensitivity: float = 0.5, omega_z_sensitivity: float = 1.0):
        self.v_x_sensitivity = v_x_sensitivity
        self.v_y_sensitivity = v_y_sensitivity
        self.omega_z_sensitivity = omega_z_sensitivity
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
        self._input.unsubscribe_from_keyboard_events(self._keyboard, self._keyboard_sub)
        self._keyboard_sub = None

    def __str__(self) -> str:
        return (
            "WASD Keyboard Controller\n"
            "\tW / S: forward / backward\n"
            "\tA / D: left / right strafe\n"
            "\tQ / E: yaw left / yaw right\n"
            "\tL: zero the command"
        )

    def reset(self) -> None:
        self._base_command.fill(0.0)

    def add_callback(self, key: str, func: Callable[[], None]) -> None:
        self._additional_callbacks[key] = func

    def advance(self) -> np.ndarray:
        return self._base_command

    def _on_keyboard_event(self, event, *args, **kwargs):
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input.name == "L":
                self.reset()
            elif event.input.name in self._INPUT_KEY_MAPPING:
                self._base_command += self._INPUT_KEY_MAPPING[event.input.name]
            if event.input.name in self._additional_callbacks:
                self._additional_callbacks[event.input.name]()
        if event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            if event.input.name in self._INPUT_KEY_MAPPING:
                self._base_command -= self._INPUT_KEY_MAPPING[event.input.name]
        return True

    def _create_key_bindings(self) -> None:
        self._INPUT_KEY_MAPPING = {
            "W": np.asarray([1.0, 0.0, 0.0]) * self.v_x_sensitivity,
            "S": np.asarray([-1.0, 0.0, 0.0]) * self.v_x_sensitivity,
            "A": np.asarray([0.0, 1.0, 0.0]) * self.v_y_sensitivity,
            "D": np.asarray([0.0, -1.0, 0.0]) * self.v_y_sensitivity,
            "Q": np.asarray([0.0, 0.0, 1.0]) * self.omega_z_sensitivity,
            "E": np.asarray([0.0, 0.0, -1.0]) * self.omega_z_sensitivity,
        }

