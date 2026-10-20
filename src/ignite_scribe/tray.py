# Copyright (C) 2026 The ignite-scribe contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""A notification-area icon that shows when ignite-scribe is reading the screen. Win32 through ctypes only.

Every MCP server writes a small status beacon (``%LOCALAPPDATA%\\ignite-scribe\\status\\<pid>.json``) every two
seconds: whether it is watching, reading pixels (vision or OCR), recording history, paused. A beacon holds flags and
counts only, never screen content, and is deleted when the server exits. One server per machine, the holder of
``tray.lock``, shows the icon and summarizes every live beacon. When it exits, another server takes over within
``takeover_s``. ``IGNITE_SCRIBE_TRAY=0`` turns it off.

The icon is a dot: grey when idle (no server watching), blue when watching (window list and UI Automation text),
amber while pixels are being read (``find_ui_elements``, ``ocr_windows``, always-on OCR), white when paused.
The tooltip says the same in words. The menu pauses or resumes always-on OCR (the same pause file as
``python -m ignite_scribe pause``) and hides the icon until that server exits.
"""
from __future__ import annotations

import ctypes
import json
import os
import threading
import time
from ctypes import wintypes
from pathlib import Path

from .always_on import PAUSE_FILE, SCRIBE_DIR, pid_alive, take_lock

STATUS_DIR = SCRIBE_DIR / "status"
LOCK = SCRIBE_DIR / "tray.lock"
COLORS = {"idle": (150, 150, 150), "watching": (40, 120, 230), "reading": (240, 160, 20), "paused": (250, 250, 250)}


def enabled() -> bool:
    return os.environ.get("IGNITE_SCRIBE_TRAY", "1") != "0"


# ---- beacons --------------------------------------------------------------------------------------------------
def beacon_of(server) -> dict:
    """What one MCP server is doing, without any screen content."""
    w = server.watcher
    vision = server.vision
    always = server.always
    a = always.status() if always is not None else None
    h = server.history
    return dict(pid=os.getpid(), t=time.time(),
                watching=bool(w is not None and getattr(w, "running", False)),
                vision_running=bool(vision is not None and vision.running),
                always_on=None if a is None else dict(running=a["running"], paused=a["paused"],
                                                      reading=bool(a["npu_held"] and not a["paused"]),
                                                      reads=a["reads"]),
                recording=bool(h is not None and h.recorder))


class Beacon:
    def __init__(self, server, every_s: float = 2.0, directory: Path | None = None):
        self.server, self.every_s = server, every_s
        self.dir = directory or STATUS_DIR
        self.path = self.dir / f"{os.getpid()}.json"
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="ignite-scribe-beacon", daemon=True)

    def start(self) -> "Beacon":
        self._thread.start()
        return self

    def write(self):
        self.dir.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_bytes(json.dumps(beacon_of(self.server)).encode("utf-8"))
        os.replace(tmp, self.path)

    def _run(self):
        while not self._stop.is_set():
            try:
                self.write()
            except Exception:  # noqa: BLE001 - a beacon must never take the server down
                pass
            self._stop.wait(self.every_s)

    def stop(self):
        self._stop.set()
        self._thread.join(3)
        try:
            self.path.unlink(missing_ok=True)
        except OSError:
            pass


def read_beacons(directory: Path | None = None, max_age_s: float = 10.0, alive=pid_alive) -> list[dict]:
    """Live beacons: fresh, and from a process that still exists. Stale files of dead servers are removed."""
    out = []
    d = directory or STATUS_DIR
    for p in d.glob("*.json") if d.is_dir() else []:
        try:
            b = json.loads(p.read_bytes())
        except (OSError, ValueError):
            continue
        if not alive(int(b.get("pid", 0))):
            try:
                p.unlink()
            except OSError:
                pass
            continue
        if time.time() - float(b.get("t", 0)) <= max_age_s:
            out.append(b)
    return out


def summarize(beacons: list[dict], paused: bool) -> tuple[str, str]:
    """-> (state, tooltip). state: idle, watching, reading or paused."""
    watching = sum(1 for b in beacons if b.get("watching"))
    reading = [b for b in beacons if b.get("vision_running") or (b.get("always_on") or {}).get("reading")]
    always = any((b.get("always_on") or {}).get("running") for b in beacons)
    recording = any(b.get("recording") for b in beacons)
    if reading and not paused:
        state = "reading"
    elif paused and always:
        state = "paused"
    elif watching:
        state = "watching"
    else:
        state = "idle"
    parts = [f"{len(beacons)} session{'s' if len(beacons) != 1 else ''}"]
    if watching:
        parts.append(f"watching windows ({watching})")
    if reading and not paused:
        parts.append("reading pixels (OCR / vision)")
    if always:
        parts.append("always-on OCR paused" if paused else "always-on OCR on")
    if recording:
        parts.append("recording history")
    if not watching and not reading:
        parts.append("not reading the screen")
    return state, ("ignite-scribe: " + ", ".join(parts))[:127]


# ---- the icon (Win32) -----------------------------------------------------------------------------------------
user32 = ctypes.WinDLL("user32", use_last_error=True)
gdi32 = ctypes.WinDLL("gdi32", use_last_error=True)
shell32 = ctypes.WinDLL("shell32", use_last_error=True)
kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

LRESULT = ctypes.c_ssize_t
WNDPROC = ctypes.WINFUNCTYPE(LRESULT, wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM)
WM_DESTROY, WM_CLOSE, WM_TIMER, WM_COMMAND, WM_APP = 0x0002, 0x0010, 0x0113, 0x0111, 0x8000
WM_LBUTTONUP, WM_RBUTTONUP = 0x0202, 0x0205
NIM_ADD, NIM_MODIFY, NIM_DELETE = 0, 1, 2
NIF_MESSAGE, NIF_ICON, NIF_TIP = 1, 2, 4
MF_STRING, MF_GRAYED, MF_SEPARATOR = 0, 1, 0x800
TPM_RIGHTBUTTON, TPM_RETURNCMD, TPM_NONOTIFY = 0x2, 0x100, 0x80
CMD_PAUSE, CMD_HIDE = 1, 2
CALLBACK = WM_APP + 1


class WNDCLASSW(ctypes.Structure):
    _fields_ = [("style", wintypes.UINT), ("lpfnWndProc", WNDPROC), ("cbClsExtra", ctypes.c_int),
                ("cbWndExtra", ctypes.c_int), ("hInstance", wintypes.HINSTANCE), ("hIcon", wintypes.HICON),
                ("hCursor", wintypes.HANDLE), ("hbrBackground", wintypes.HBRUSH), ("lpszMenuName", wintypes.LPCWSTR),
                ("lpszClassName", wintypes.LPCWSTR)]


class NOTIFYICONDATAW(ctypes.Structure):
    _fields_ = [("cbSize", wintypes.DWORD), ("hWnd", wintypes.HWND), ("uID", wintypes.UINT),
                ("uFlags", wintypes.UINT), ("uCallbackMessage", wintypes.UINT), ("hIcon", wintypes.HICON),
                ("szTip", ctypes.c_wchar * 128), ("dwState", wintypes.DWORD), ("dwStateMask", wintypes.DWORD),
                ("szInfo", ctypes.c_wchar * 256), ("uVersion", wintypes.UINT), ("szInfoTitle", ctypes.c_wchar * 64),
                ("dwInfoFlags", wintypes.DWORD), ("guidItem", ctypes.c_byte * 16), ("hBalloonIcon", wintypes.HICON)]


class ICONINFO(ctypes.Structure):
    _fields_ = [("fIcon", wintypes.BOOL), ("xHotspot", wintypes.DWORD), ("yHotspot", wintypes.DWORD),
                ("hbmMask", wintypes.HBITMAP), ("hbmColor", wintypes.HBITMAP)]


class BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [("biSize", wintypes.DWORD), ("biWidth", wintypes.LONG), ("biHeight", wintypes.LONG),
                ("biPlanes", wintypes.WORD), ("biBitCount", wintypes.WORD), ("biCompression", wintypes.DWORD),
                ("biSizeImage", wintypes.DWORD), ("biXPelsPerMeter", wintypes.LONG),
                ("biYPelsPerMeter", wintypes.LONG), ("biClrUsed", wintypes.DWORD), ("biClrImportant", wintypes.DWORD)]


user32.DefWindowProcW.restype = LRESULT
user32.DefWindowProcW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
user32.CreateWindowExW.restype = wintypes.HWND
user32.CreateWindowExW.argtypes = [wintypes.DWORD, wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.DWORD,
                                   ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, wintypes.HWND,
                                   wintypes.HMENU, wintypes.HINSTANCE, wintypes.LPVOID]
user32.RegisterClassW.argtypes = [ctypes.POINTER(WNDCLASSW)]
user32.RegisterClassW.restype = wintypes.ATOM
user32.UnregisterClassW.argtypes = [wintypes.LPCWSTR, wintypes.HINSTANCE]
user32.RegisterWindowMessageW.argtypes = [wintypes.LPCWSTR]
user32.RegisterWindowMessageW.restype = wintypes.UINT
user32.GetMessageW.argtypes = [ctypes.POINTER(wintypes.MSG), wintypes.HWND, wintypes.UINT, wintypes.UINT]
user32.TranslateMessage.argtypes = [ctypes.POINTER(wintypes.MSG)]
user32.DispatchMessageW.argtypes = [ctypes.POINTER(wintypes.MSG)]
user32.PostMessageW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
user32.SetTimer.argtypes = [wintypes.HWND, ctypes.c_size_t, wintypes.UINT, ctypes.c_void_p]
user32.SetTimer.restype = ctypes.c_size_t
user32.DestroyWindow.argtypes = [wintypes.HWND]
user32.CreatePopupMenu.restype = wintypes.HMENU
user32.AppendMenuW.argtypes = [wintypes.HMENU, wintypes.UINT, ctypes.c_size_t, wintypes.LPCWSTR]
user32.TrackPopupMenu.argtypes = [wintypes.HMENU, wintypes.UINT, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                  wintypes.HWND, ctypes.c_void_p]
user32.DestroyMenu.argtypes = [wintypes.HMENU]
user32.SetForegroundWindow.argtypes = [wintypes.HWND]
user32.GetCursorPos.argtypes = [ctypes.POINTER(wintypes.POINT)]
user32.CreateIconIndirect.argtypes = [ctypes.POINTER(ICONINFO)]
user32.CreateIconIndirect.restype = wintypes.HICON
user32.DestroyIcon.argtypes = [wintypes.HICON]
gdi32.CreateDIBSection.argtypes = [wintypes.HDC, ctypes.c_void_p, wintypes.UINT, ctypes.POINTER(ctypes.c_void_p),
                                   wintypes.HANDLE, wintypes.DWORD]
gdi32.CreateDIBSection.restype = wintypes.HBITMAP
gdi32.CreateBitmap.argtypes = [ctypes.c_int, ctypes.c_int, wintypes.UINT, wintypes.UINT, ctypes.c_void_p]
gdi32.CreateBitmap.restype = wintypes.HBITMAP
gdi32.DeleteObject.argtypes = [wintypes.HGDIOBJ]
shell32.Shell_NotifyIconW.argtypes = [wintypes.DWORD, ctypes.POINTER(NOTIFYICONDATAW)]
shell32.Shell_NotifyIconW.restype = wintypes.BOOL
kernel32.GetModuleHandleW.restype = wintypes.HMODULE
kernel32.GetModuleHandleW.argtypes = [wintypes.LPCWSTR]


def make_icon(rgb: tuple[int, int, int], size: int = 16) -> int:
    """A filled dot with a dark rim, as an HICON (32-bit with alpha)."""
    bmi = BITMAPINFOHEADER(ctypes.sizeof(BITMAPINFOHEADER), size, -size, 1, 32, 0, 0, 0, 0, 0, 0)
    bits = ctypes.c_void_p()
    color = gdi32.CreateDIBSection(None, ctypes.byref(bmi), 0, ctypes.byref(bits), None, 0)
    px = (ctypes.c_uint32 * (size * size)).from_address(bits.value)
    c, r = (size - 1) / 2, size / 2 - 0.5
    for y in range(size):
        for x in range(size):
            d = ((x - c) ** 2 + (y - c) ** 2) ** 0.5
            a = max(0.0, min(1.0, r - d + 0.5))                     # anti-aliased edge
            rim = d > r - 1.6
            red, green, blue = (60, 60, 60) if rim else rgb
            al = int(a * 255)
            px[y * size + x] = (al << 24) | ((red * al // 255) << 16) | ((green * al // 255) << 8) | (blue * al // 255)
    mask = gdi32.CreateBitmap(size, size, 1, 1, None)
    info = ICONINFO(True, 0, 0, mask, color)
    icon = user32.CreateIconIndirect(ctypes.byref(info))
    gdi32.DeleteObject(mask)
    gdi32.DeleteObject(color)
    return icon


class Tray:
    """Shows the icon while this process holds ``tray.lock``; waits for it otherwise."""

    def __init__(self, takeover_s: float = 15.0, refresh_ms: int = 2000, status_dir: Path | None = None,
                 lock: Path | None = None, pause_file: Path | None = None):
        self.takeover_s, self.refresh_ms = takeover_s, refresh_ms
        self.status_dir, self.lock_path = status_dir or STATUS_DIR, lock or LOCK
        self.pause_file = pause_file or PAUSE_FILE
        self.state, self.tip, self.shown, self.hidden = None, "", False, False
        self.hwnd = None
        self._lock = None
        self._icons: dict[str, int] = {}
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="ignite-scribe-tray", daemon=True)

    def start(self) -> "Tray":
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        if self.hwnd:
            user32.PostMessageW(self.hwnd, WM_CLOSE, 0, 0)
        self._thread.join(5)

    @property
    def owner(self) -> bool:
        return self._lock is not None

    def _run(self):
        while not self._stop.is_set():
            self._lock = take_lock(self.lock_path)
            if self._lock is not None:
                try:
                    self._loop()
                finally:
                    self._lock.close()
                    self._lock = None
                return
            self._stop.wait(self.takeover_s)

    def _nid(self) -> NOTIFYICONDATAW:
        n = NOTIFYICONDATAW()
        n.cbSize, n.hWnd, n.uID = ctypes.sizeof(NOTIFYICONDATAW), self.hwnd, 1
        n.uFlags, n.uCallbackMessage = NIF_MESSAGE | NIF_ICON | NIF_TIP, CALLBACK
        n.hIcon = self._icons[self.state]
        n.szTip = self.tip
        return n

    def refresh(self):
        state, tip = summarize(read_beacons(self.status_dir), self.pause_file.exists())
        if self.hidden or (state, tip) == (self.state, self.tip) and self.shown:
            return
        self.state, self.tip = state, tip
        n = self._nid()
        if self.shown:
            shell32.Shell_NotifyIconW(NIM_MODIFY, ctypes.byref(n))
        else:
            self.shown = bool(shell32.Shell_NotifyIconW(NIM_ADD, ctypes.byref(n)))

    def _menu(self):
        paused = self.pause_file.exists()
        m = user32.CreatePopupMenu()
        user32.AppendMenuW(m, MF_STRING | MF_GRAYED, 0, self.tip[:60])
        user32.AppendMenuW(m, MF_SEPARATOR, 0, None)
        user32.AppendMenuW(m, MF_STRING, CMD_PAUSE, "Resume always-on OCR" if paused else "Pause always-on OCR")
        user32.AppendMenuW(m, MF_STRING, CMD_HIDE, "Hide this icon")
        pt = wintypes.POINT()
        user32.GetCursorPos(ctypes.byref(pt))
        user32.SetForegroundWindow(self.hwnd)
        cmd = user32.TrackPopupMenu(m, TPM_RIGHTBUTTON | TPM_RETURNCMD | TPM_NONOTIFY, pt.x, pt.y, 0, self.hwnd, None)
        user32.DestroyMenu(m)
        if cmd == CMD_PAUSE:
            if paused:
                self.pause_file.unlink(missing_ok=True)
            else:
                self.pause_file.parent.mkdir(parents=True, exist_ok=True)
                self.pause_file.write_bytes(b"")
            self.refresh()
        elif cmd == CMD_HIDE:
            self._remove()
            self.hidden = True

    def _remove(self):
        if self.shown:
            shell32.Shell_NotifyIconW(NIM_DELETE, ctypes.byref(self._nid()))
            self.shown = False

    def _loop(self):
        hinst = kernel32.GetModuleHandleW(None)
        taskbar_created = user32.RegisterWindowMessageW("TaskbarCreated")
        cls = f"ignite-scribe-tray-{os.getpid()}"

        def proc(hwnd, msg, wp, lp):
            try:
                if msg == WM_TIMER:
                    self.refresh()
                elif msg == CALLBACK and lp in (WM_RBUTTONUP, WM_LBUTTONUP):
                    self._menu()
                elif msg == taskbar_created:                 # Explorer restarted: add the icon again
                    self.shown = False
                    self.refresh()
                elif msg == WM_CLOSE:
                    self._remove()
                    user32.DestroyWindow(hwnd)
                    return 0
                elif msg == WM_DESTROY:
                    user32.PostQuitMessage(0)
                    return 0
            except Exception:  # noqa: BLE001 - never let a callback exception reach Win32
                pass
            return user32.DefWindowProcW(hwnd, msg, wp, lp)

        self._proc = WNDPROC(proc)
        wc = WNDCLASSW(0, self._proc, 0, 0, hinst, None, None, None, None, cls)
        user32.RegisterClassW(ctypes.byref(wc))
        self.hwnd = user32.CreateWindowExW(0, cls, "ignite-scribe", 0, 0, 0, 0, 0, None, None, hinst, None)
        if not self.hwnd:
            return
        self._icons = {k: make_icon(v) for k, v in COLORS.items()}
        try:
            if self._stop.is_set():
                return
            self.refresh()
            user32.SetTimer(self.hwnd, 1, self.refresh_ms, None)
            msg = wintypes.MSG()
            while user32.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:
                user32.TranslateMessage(ctypes.byref(msg))
                user32.DispatchMessageW(ctypes.byref(msg))
        finally:
            self._remove()
            for h in self._icons.values():
                user32.DestroyIcon(h)
            user32.UnregisterClassW(cls, hinst)
            self.hwnd = None
