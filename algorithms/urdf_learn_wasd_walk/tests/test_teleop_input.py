from __future__ import annotations

import importlib
from types import SimpleNamespace
import sys

import numpy as np


def _load_teleop_module(monkeypatch):
    class FakeInputInterface:
        def __init__(self) -> None:
            self.unsubscribe_calls = 0

        def subscribe_to_keyboard_events(self, keyboard, callback):
            self.keyboard = keyboard
            self.callback = callback
            return object()

        def unsubscribe_from_keyboard_events(self, keyboard, subscription):
            self.unsubscribe_calls += 1

    fake_input = FakeInputInterface()
    fake_carb = SimpleNamespace(
        input=SimpleNamespace(
            acquire_input_interface=lambda: fake_input,
            KeyboardEventType=SimpleNamespace(
                KEY_PRESS="KEY_PRESS",
                KEY_RELEASE="KEY_RELEASE",
                KEY_REPEAT="KEY_REPEAT",
            ),
        )
    )
    fake_omni = SimpleNamespace(
        appwindow=SimpleNamespace(
            get_default_app_window=lambda: SimpleNamespace(get_keyboard=lambda: object()),
        )
    )
    monkeypatch.setitem(sys.modules, "carb", fake_carb)
    monkeypatch.setitem(sys.modules, "omni", fake_omni)
    sys.modules.pop("algorithms.urdf_learn_wasd_walk.teleop_input", None)
    module = importlib.import_module("algorithms.urdf_learn_wasd_walk.teleop_input")
    return module, fake_input


def test_keyboard_events_accept_string_inputs(monkeypatch) -> None:
    module, fake_input = _load_teleop_module(monkeypatch)
    device = module.WasdSe2Keyboard(debug_print=False)

    try:
        device._on_keyboard_event(SimpleNamespace(input="A", type="KEY_PRESS"))
        np.testing.assert_allclose(
            device.advance(),
            np.asarray([0.0, device.v_y_sensitivity, 0.0], dtype=np.float32),
        )

        device._on_keyboard_event(SimpleNamespace(input="A", type="KEY_REPEAT"))
        np.testing.assert_allclose(
            device.advance(),
            np.asarray([0.0, device.v_y_sensitivity, 0.0], dtype=np.float32),
        )

        device._on_keyboard_event(SimpleNamespace(input="A", type="KEY_RELEASE"))
        np.testing.assert_allclose(device.advance(), np.zeros(3, dtype=np.float32))
    finally:
        device.__del__()

    assert fake_input.unsubscribe_calls == 1


def test_keyboard_events_accept_named_inputs(monkeypatch) -> None:
    module, _ = _load_teleop_module(monkeypatch)
    device = module.WasdSe2Keyboard(debug_print=False)

    try:
        device._on_keyboard_event(SimpleNamespace(input=SimpleNamespace(name="D"), type=SimpleNamespace(name="KEY_PRESS")))
        np.testing.assert_allclose(
            device.advance(),
            np.asarray([0.0, -device.v_y_sensitivity, 0.0], dtype=np.float32),
        )
    finally:
        device.__del__()
