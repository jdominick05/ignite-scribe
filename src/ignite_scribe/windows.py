# Copyright (C) 2026 The ignite-scribe contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The first source: top-level desktop windows from Win32, no dependencies (ctypes only).

Lists visible, uncloaked, non-tool top-level windows in z-order (topmost first): title, owning process, bounds in
physical pixels (the process is made per-monitor DPI aware; a minimized window reports its restored
bounds), focus and minimized state; plus the monitors. This is
what UI Automation and OCR will later fill with content.
"""
from __future__ import annotations

import ctypes
import os
import sys
from ctypes import wintypes

from .snapshot import Rect, Snapshot, Window

if sys.platform != "win32":
    raise ImportError("ignite_scribe.windows needs Windows")

user32 = ctypes.WinDLL("user32", use_last_error=True)
kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
dwmapi = ctypes.WinDLL("dwmapi")

GW_OWNER, GWL_EXSTYLE = 4, -20
WS_EX_TOOLWINDOW, WS_EX_APPWINDOW = 0x80, 0x40000
DWMWA_CLOAKED = 14
PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
EnumWindowsProc = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
MonitorEnumProc = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HMONITOR, wintypes.HDC, ctypes.POINTER(wintypes.RECT),
                                     wintypes.LPARAM)
user32.GetWindowLongPtrW.restype = ctypes.c_ssize_t
user32.GetWindowLongPtrW.argtypes = [wintypes.HWND, ctypes.c_int]
user32.GetWindow.restype = wintypes.HWND
user32.GetWindow.argtypes = [wintypes.HWND, wintypes.UINT]
user32.GetForegroundWindow.restype = wintypes.HWND
user32.WindowFromPoint.restype = wintypes.HWND
user32.WindowFromPoint.argtypes = [wintypes.POINT]
user32.GetAncestor.restype = wintypes.HWND
user32.GetAncestor.argtypes = [wintypes.HWND, wintypes.UINT]
GA_ROOT = 2



class WINDOWPLACEMENT(ctypes.Structure):
    _fields_ = [("length", wintypes.UINT), ("flags", wintypes.UINT), ("showCmd", wintypes.UINT),
                ("ptMinPosition", wintypes.POINT), ("ptMaxPosition", wintypes.POINT),
                ("rcNormalPosition", wintypes.RECT)]


_dpi_done = False


def dpi_aware():
    """Per-monitor v2 DPI awareness, so window rectangles are physical pixels (what a capture sees)."""
    global _dpi_done
    if not _dpi_done:
        try:
            user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
        except (AttributeError, OSError):
            pass
        _dpi_done = True


def _title(hwnd) -> str:
    n = user32.GetWindowTextLengthW(hwnd)
    buf = ctypes.create_unicode_buffer(n + 1)
    user32.GetWindowTextW(hwnd, buf, n + 1)
    return buf.value


def _process(hwnd) -> tuple[int, str]:
    pid = wintypes.DWORD()
    user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    h = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid.value)
    name = ""
    if h:
        try:
            size = wintypes.DWORD(1024)
            buf = ctypes.create_unicode_buffer(size.value)
            if kernel32.QueryFullProcessImageNameW(h, 0, buf, ctypes.byref(size)):
                name = os.path.basename(buf.value)
        finally:
            kernel32.CloseHandle(h)
    return pid.value, name


def _cloaked(hwnd) -> bool:
    v = wintypes.DWORD()
    r = dwmapi.DwmGetWindowAttribute(hwnd, DWMWA_CLOAKED, ctypes.byref(v), ctypes.sizeof(v))
    return r == 0 and v.value != 0


def _listed(hwnd) -> bool:
    """A window the taskbar would show: visible, uncloaked, not a tool window, unowned or an app window."""
    if not user32.IsWindowVisible(hwnd) or _cloaked(hwnd):
        return False
    ex = user32.GetWindowLongPtrW(hwnd, GWL_EXSTYLE)
    if ex & WS_EX_TOOLWINDOW:
        return False
    return not user32.GetWindow(hwnd, GW_OWNER) or bool(ex & WS_EX_APPWINDOW)


def monitors() -> list[Rect]:
    dpi_aware()
    out = []

    def cb(hmon, hdc, rc, lp):
        r = rc.contents
        out.append(Rect(r.left, r.top, r.right, r.bottom))
        return True
    user32.EnumDisplayMonitors(None, None, MonitorEnumProc(cb), 0)
    return out


def windows() -> list[Window]:
    dpi_aware()
    found = []

    def cb(hwnd, lp):
        if _listed(hwnd):
            found.append(hwnd)
        return True
    user32.EnumWindows(EnumWindowsProc(cb), 0)   # top-level windows in z-order, topmost first
    fg = user32.GetForegroundWindow()
    out = []
    for z, hwnd in enumerate(found):
        r = wintypes.RECT()
        if user32.IsIconic(hwnd):   # minimized: its restored bounds, not the (-32000, -32000) parking spot
            wp = WINDOWPLACEMENT()
            wp.length = ctypes.sizeof(wp)
            user32.GetWindowPlacement(hwnd, ctypes.byref(wp))
            r = wp.rcNormalPosition
        else:
            user32.GetWindowRect(hwnd, ctypes.byref(r))
        pid, proc = _process(hwnd)
        out.append(Window(hwnd=int(hwnd), title=_title(hwnd), process=proc, pid=pid,
                          rect=Rect(r.left, r.top, r.right, r.bottom), z=z, focused=hwnd == fg,
                          minimized=bool(user32.IsIconic(hwnd))))
    return out


def root_at(x: int, y: int) -> int | None:
    """The top-level window a click at screen pixel (x, y) would reach: whatever is really on top there, menus,
    tooltips, the taskbar and unlisted tool windows included."""
    dpi_aware()
    h = user32.WindowFromPoint(wintypes.POINT(x, y))
    root = user32.GetAncestor(h, GA_ROOT) if h else None
    return int(root) if root else None


def snapshot() -> Snapshot:
    return Snapshot.now(windows(), monitors(), ["windows"])
