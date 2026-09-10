#!/usr/bin/env python3
"""Small dependency-free XTest helper for the isolated TK2 validation desktop."""

from __future__ import annotations

import argparse
import ctypes
import os
import subprocess
import time


X11 = ctypes.CDLL("libX11.so.6")
XTST = ctypes.CDLL("libXtst.so.6")
X11.XOpenDisplay.argtypes = [ctypes.c_char_p]
X11.XOpenDisplay.restype = ctypes.c_void_p
X11.XStringToKeysym.argtypes = [ctypes.c_char_p]
X11.XStringToKeysym.restype = ctypes.c_ulong
X11.XKeysymToKeycode.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
X11.XKeysymToKeycode.restype = ctypes.c_uint
X11.XFlush.argtypes = [ctypes.c_void_p]
XTST.XTestFakeKeyEvent.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_int, ctypes.c_ulong]
XTST.XTestFakeButtonEvent.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_int, ctypes.c_ulong]
XTST.XTestFakeMotionEvent.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_ulong]


def open_display() -> ctypes.c_void_p:
    display = X11.XOpenDisplay(os.environ.get("DISPLAY", ":1").encode())
    if not display:
        raise SystemExit("Could not open X display")
    return display


def keycode(display: ctypes.c_void_p, name: str) -> int:
    keysym = X11.XStringToKeysym(name.encode())
    code = X11.XKeysymToKeycode(display, keysym)
    if not code:
        raise SystemExit(f"Unknown key: {name}")
    return code


def key(display: ctypes.c_void_p, name: str, pressed: bool) -> None:
    XTST.XTestFakeKeyEvent(display, keycode(display, name), int(pressed), 0)


def hotkey(display: ctypes.c_void_p, names: list[str]) -> None:
    for name in names:
        key(display, name, True)
    for name in reversed(names):
        key(display, name, False)
    X11.XFlush(display)


def type_text(display: ctypes.c_void_p, text: str) -> None:
    """Type the small ASCII subset used by Stellaris debug commands."""
    shifted = {
        "_": "minus",
        ":": "semicolon",
        "+": "equal",
    }
    named = {
        " ": "space",
        ".": "period",
        "-": "minus",
        "/": "slash",
    }
    for char in text:
        if char in shifted:
            key(display, "Shift_L", True)
            key(display, shifted[char], True)
            key(display, shifted[char], False)
            key(display, "Shift_L", False)
        elif char.isupper():
            key(display, "Shift_L", True)
            key(display, char.lower(), True)
            key(display, char.lower(), False)
            key(display, "Shift_L", False)
        else:
            name = named.get(char, char)
            key(display, name, True)
            key(display, name, False)
        X11.XFlush(display)
        time.sleep(0.015)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    keys = sub.add_parser("hotkey")
    keys.add_argument("keys", nargs="+")
    paste = sub.add_parser("paste")
    paste.add_argument("text")
    typed = sub.add_parser("type")
    typed.add_argument("text")
    click = sub.add_parser("click")
    click.add_argument("x", type=int)
    click.add_argument("y", type=int)
    click.add_argument("--button", type=int, default=1)
    shot = sub.add_parser("screenshot")
    shot.add_argument("path")
    args = parser.parse_args()

    env = dict(os.environ)
    env.setdefault("DISPLAY", ":1")
    env.setdefault("XAUTHORITY", "/home/wishai/.Xauthority")
    if args.command == "screenshot":
        subprocess.run(["xfce4-screenshooter", "-f", "-s", args.path], env=env, check=True)
        return

    display = open_display()
    if args.command == "hotkey":
        hotkey(display, args.keys)
    elif args.command == "paste":
        subprocess.run(["xclip", "-selection", "clipboard"], input=args.text.encode(), env=env, check=True)
        hotkey(display, ["Control_L", "v"])
    elif args.command == "type":
        type_text(display, args.text)
    elif args.command == "click":
        XTST.XTestFakeMotionEvent(display, -1, args.x, args.y, 0)
        XTST.XTestFakeButtonEvent(display, args.button, 1, 0)
        XTST.XTestFakeButtonEvent(display, args.button, 0, 0)
        X11.XFlush(display)
    time.sleep(0.1)


if __name__ == "__main__":
    main()
