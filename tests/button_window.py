# Copyright (C) 2026 The ignite-scribe contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""A native test window for the element map: an enabled push button, a disabled one and a check box.

Run as a child process: prints its HWND on stdout, then pumps messages until it is killed. The labels come from
argv so the test owns the strings it searches for: ``button_window.py TITLE ENABLED DISABLED CHECKBOX``.
"""
import ctypes
import sys
from ctypes import wintypes

user32 = ctypes.WinDLL("user32", use_last_error=True)
user32.CreateWindowExW.restype = wintypes.HWND
user32.CreateWindowExW.argtypes = [wintypes.DWORD, wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.DWORD,
                                   ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, wintypes.HWND,
                                   wintypes.HMENU, wintypes.HINSTANCE, wintypes.LPVOID]

WS_OVERLAPPEDWINDOW, WS_VISIBLE, WS_CHILD, WS_DISABLED = 0x00CF0000, 0x10000000, 0x40000000, 0x08000000
BS_PUSHBUTTON, BS_AUTOCHECKBOX, WS_EX_TOPMOST = 0x0, 0x3, 0x8   # topmost: a background process cannot take focus


def main() -> None:
    title, enabled, disabled, check = sys.argv[1:5]
    user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
    top = user32.CreateWindowExW(WS_EX_TOPMOST, "STATIC", title, WS_OVERLAPPEDWINDOW | WS_VISIBLE, 160, 160, 480,
                                 260, None, None, None, None)
    child = WS_CHILD | WS_VISIBLE
    user32.CreateWindowExW(0, "BUTTON", enabled, child | BS_PUSHBUTTON, 20, 20, 180, 36, top, None, None, None)
    user32.CreateWindowExW(0, "BUTTON", disabled, child | BS_PUSHBUTTON | WS_DISABLED, 220, 20, 180, 36, top, None,
                           None, None)
    user32.CreateWindowExW(0, "BUTTON", check, child | BS_AUTOCHECKBOX, 20, 80, 220, 30, top, None, None, None)
    user32.SetForegroundWindow(top)
    print(int(top), flush=True)
    msg = wintypes.MSG()
    while user32.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:
        user32.TranslateMessage(ctypes.byref(msg))
        user32.DispatchMessageW(ctypes.byref(msg))


if __name__ == "__main__":
    main()
