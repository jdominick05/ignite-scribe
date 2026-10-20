# Copyright (C) 2026 The ignite-scribe contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""A native test window for get_selection: an Edit holding known text, part of it selected and focused.

Run as a child process: prints "HWND PID EDIT_HWND" on stdout, then pumps messages until it is killed. The
texts come from argv so the test owns the strings it checks: ``selection_window.py TITLE TEXT START END`` selects
TEXT[START:END].
"""
import ctypes
import os
import sys
from ctypes import wintypes

user32 = ctypes.WinDLL("user32", use_last_error=True)
user32.CreateWindowExW.restype = wintypes.HWND
user32.CreateWindowExW.argtypes = [wintypes.DWORD, wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.DWORD,
                                   ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, wintypes.HWND,
                                   wintypes.HMENU, wintypes.HINSTANCE, wintypes.LPVOID]
user32.SendMessageW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
user32.SetFocus.argtypes = [wintypes.HWND]

WS_OVERLAPPEDWINDOW, WS_VISIBLE, WS_CHILD, WS_BORDER = 0x00CF0000, 0x10000000, 0x40000000, 0x00800000
ES_MULTILINE, ES_NOHIDESEL, WS_EX_TOPMOST, EM_SETSEL = 0x4, 0x100, 0x8, 0xB1


def main() -> None:
    title, text, start, end = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
    top = user32.CreateWindowExW(WS_EX_TOPMOST, "STATIC", title, WS_OVERLAPPEDWINDOW | WS_VISIBLE, 200, 200, 480,
                                 220, None, None, None, None)
    edit = user32.CreateWindowExW(0, "EDIT", text, WS_CHILD | WS_VISIBLE | WS_BORDER | ES_MULTILINE | ES_NOHIDESEL,
                                  20, 20, 420, 120, top, None, None, None)
    user32.SetForegroundWindow(top)
    user32.SetFocus(edit)
    user32.SendMessageW(edit, EM_SETSEL, start, end)
    print(int(top), os.getpid(), int(edit), flush=True)
    msg = wintypes.MSG()
    while user32.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:
        user32.TranslateMessage(ctypes.byref(msg))
        user32.DispatchMessageW(ctypes.byref(msg))


if __name__ == "__main__":
    main()
