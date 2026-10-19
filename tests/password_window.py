# Copyright (C) 2026 The ignite-scribe contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""A native test window for the UI Automation tests: a label, a plain Edit and a password Edit.

Run as a child process: prints its HWND on stdout, then pumps messages until it is killed. The texts come from
argv so the test owns the strings it searches for: ``password_window.py TITLE LABEL PLAIN SECRET``.
"""
import ctypes
import sys
from ctypes import wintypes

user32 = ctypes.WinDLL("user32", use_last_error=True)
user32.CreateWindowExW.restype = wintypes.HWND
user32.CreateWindowExW.argtypes = [wintypes.DWORD, wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.DWORD,
                                   ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, wintypes.HWND,
                                   wintypes.HMENU, wintypes.HINSTANCE, wintypes.LPVOID]

WS_OVERLAPPEDWINDOW, WS_VISIBLE, WS_CHILD, WS_BORDER = 0x00CF0000, 0x10000000, 0x40000000, 0x00800000
ES_PASSWORD, ES_AUTOHSCROLL = 0x20, 0x80


def main() -> None:
    title, label, plain, secret = sys.argv[1:5]
    user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
    top = user32.CreateWindowExW(0, "STATIC", title, WS_OVERLAPPEDWINDOW | WS_VISIBLE, 120, 120, 520, 260,
                                 None, None, None, None)
    child = WS_CHILD | WS_VISIBLE
    user32.CreateWindowExW(0, "STATIC", label, child, 20, 20, 460, 24, top, None, None, None)
    user32.CreateWindowExW(0, "EDIT", plain, child | WS_BORDER | ES_AUTOHSCROLL, 20, 60, 460, 28, top, None, None,
                           None)
    user32.CreateWindowExW(0, "EDIT", secret, child | WS_BORDER | ES_AUTOHSCROLL | ES_PASSWORD, 20, 100, 460, 28,
                           top, None, None, None)
    user32.SetForegroundWindow(top)
    print(int(top), flush=True)
    msg = wintypes.MSG()
    while user32.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:
        user32.TranslateMessage(ctypes.byref(msg))
        user32.DispatchMessageW(ctypes.byref(msg))


if __name__ == "__main__":
    main()
