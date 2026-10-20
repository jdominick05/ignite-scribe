# Copyright (C) 2026 The ignite-scribe contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""A notification-area icon that shows when ignite-scribe is reading the screen. Win32 through ctypes only.

Every MCP server writes a small status beacon (``%LOCALAPPDATA%\\ignite-scribe\\status\\<pid>.json``) every two
seconds: whether it is watching, reading pixels (vision or OCR), recording history, paused. A beacon holds flags and
counts only, never screen content, and is deleted when the server exits. One process per machine, the holder of
``tray.lock``, shows the icon and summarizes every live beacon. The background service owns it whenever it runs:
its beacon says ``service``, and an agent's server gives the lock up (and does not take it) while such a beacon is
live. Without a service, the first agent's server shows it, and when that exits another takes over within
``takeover_s``. ``IGNITE_SCRIBE_TRAY=0`` turns it off.

The icon is a dot with a shape in it, so it reads without a colour key: a slashed grey ring when idle (no server
watching), a blue eye when watching (window list and UI Automation text), amber lines of text while pixels are
being read (``find_ui_elements``, ``ocr_windows``, always-on OCR), white pause bars when always-on OCR is paused.
A red dot at its lower right means history is recording. The tooltip and the menu say the same in words, and the
menu has a key to the icons. The menu pauses or resumes always-on OCR (the same pause file as
``python -m ignite_scribe pause``; offered only while always-on OCR runs or is paused) and hides the icon until that
server exits.
"""
from __future__ import annotations

import ctypes
import json
import math
import os
import sys
import threading
import time
from ctypes import wintypes
from pathlib import Path

from .always_on import PAUSE_FILE, SCRIBE_DIR, pid_alive, take_lock
from .redact import DEFAULT_EXCLUDE_PROCESSES

STATUS_DIR = SCRIBE_DIR / "status"
LOCK = SCRIBE_DIR / "tray.lock"
COLORS = {"idle": (150, 150, 150), "watching": (40, 120, 230), "reading": (240, 160, 20), "paused": (250, 250, 250)}


def enabled() -> bool:
    return os.environ.get("IGNITE_SCRIBE_TRAY", "1") != "0"


# ---- beacons --------------------------------------------------------------------------------------------------
def beacon_of(server, service: bool = False) -> dict:
    """What one MCP server (or the background service, ``service``) is doing, without any screen content."""
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
                recording=bool(h is not None and h.recorder), service=service,
                tray=bool(getattr(server, "tray", None) is not None and server.tray.owner))


class Beacon:
    """Writes this process's beacon every ``every_s``, and applies policy-file changes to the running server
    (``settings.apply_live``). The beacon's ``policy`` is the file's modification time it last applied."""

    def __init__(self, server, every_s: float = 2.0, directory: Path | None = None, service: bool = False,
                 policy_file: Path | None = None, live: bool = True):
        self.server, self.every_s, self.service = server, every_s, service
        from . import settings
        self.policy_file = policy_file or settings.path()
        self.live = live and self.policy_file is not None
        self.applied = settings.mtime(self.policy_file)      # the server loaded this version at its start
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
        b = beacon_of(self.server, self.service)
        b["policy"] = self.applied
        tmp.write_bytes(json.dumps(b).encode("utf-8"))
        os.replace(tmp, self.path)

    def check_policy(self):
        """Apply the policy file when it changed since this process last applied it."""
        from . import settings
        from .redact import Policy
        m = settings.mtime(self.policy_file)
        if not self.live or m == self.applied:
            return
        new = Policy.load(self.policy_file)
        settings.apply_live(self.server, new, is_service=self.service)
        self.applied = m

    def _run(self):
        while not self._stop.is_set():
            try:
                self.check_policy()
            except Exception as e:  # noqa: BLE001 - a beacon must never take the server down
                print(f"settings not applied: {type(e).__name__}: {e}", file=sys.stderr)
            try:
                self.write()
            except Exception:  # noqa: BLE001
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


HEADLINES = {"idle": "not reading the screen", "watching": "watching windows",
             "reading": "reading screen pixels", "paused": "always-on OCR paused"}
# What each icon means, in the words the menu's key uses. (state, recording badge) -> text.
LEGEND = [(("watching", False), "Eye: watching window titles and UI text, no pixels"),
          (("reading", False), "Text lines: reading pixels (OCR or UI-element vision)"),
          (("paused", False), "Pause bars: always-on OCR is paused"),
          (("idle", False), "Slashed ring: nothing is reading the screen"),
          (("watching", True), "Red dot: history is being recorded (encrypted)")]


def describe(beacons: list[dict], paused: bool, policy_mtime: float | None = None) -> dict:
    """What every live session is doing, in words: state (idle, watching, reading or paused), a headline, detail
    lines, the tooltip (at most 127 characters, the shell's limit) and the flags the icon and menu need.
    ``stale`` names the processes that have not applied the policy file as of ``policy_mtime`` (older code, which
    cannot reload it, or a change made in the last few seconds)."""
    watching = sum(1 for b in beacons if b.get("watching"))
    always = any((b.get("always_on") or {}).get("running") for b in beacons)
    # The pause file pauses always-on OCR only; a find_ui_elements / ocr_windows call still reads pixels.
    reading = any(b.get("vision_running") or ((b.get("always_on") or {}).get("reading") and not paused)
                  for b in beacons)
    recording = any(b.get("recording") for b in beacons)
    if reading:
        state = "reading"
    elif paused and always:
        state = "paused"
    elif watching:
        state = "watching"
    else:
        state = "idle"
    headline = HEADLINES[state] + (f" ({watching})" if state == "watching" and watching > 1 else "")
    if reading:
        what = "Reading window pixels (OCR / vision)"
    elif watching:
        what = "Titles and UI text only, no pixels"
    else:
        what = "Nothing is being read"
    lines = [what, "History: recording" if recording else "History: off",
             "Always-on OCR: " + ("paused" if always and paused else "on" if always else "off")]
    tip = "ignite-scribe: " + headline
    for line in lines:                               # whole lines only, within the shell's 127 characters
        if len(tip) + 1 + len(line) > 127:
            break
        tip += "\n" + line
    n = len(beacons)
    sessions = f"{n} ignite-scribe process{'es' if n != 1 else ''} running (agents and the background service)"
    old = [b for b in beacons if "policy" not in b or (policy_mtime is not None and b["policy"] != policy_mtime
                                                       and time.time() - policy_mtime > 6)]
    stale = (f"{len(old)} process{'es' if len(old) != 1 else ''} (pid {', '.join(str(b.get('pid')) for b in old)})"
             " cannot apply settings changes: restart {0}".format("them" if len(old) != 1 else "it")) if old else ""
    return dict(state=state, headline=headline, lines=lines, sessions=sessions, tip=tip[:127], always=always,
                recording=recording, paused=paused, stale=stale)


def summarize(beacons: list[dict], paused: bool) -> tuple[str, str]:
    """-> (state, tooltip). state: idle, watching, reading or paused."""
    d = describe(beacons, paused)
    return d["state"], d["tip"]


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
MF_STRING, MF_GRAYED, MF_CHECKED, MF_POPUP, MF_SEPARATOR = 0, 1, 0x8, 0x10, 0x800
MIIM_BITMAP = 0x80
SM_CXSMICON = 49
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


class MENUITEMINFOW(ctypes.Structure):
    _fields_ = [("cbSize", wintypes.UINT), ("fMask", wintypes.UINT), ("fType", wintypes.UINT),
                ("fState", wintypes.UINT), ("wID", wintypes.UINT), ("hSubMenu", wintypes.HMENU),
                ("hbmpChecked", wintypes.HBITMAP), ("hbmpUnchecked", wintypes.HBITMAP),
                ("dwItemData", ctypes.c_size_t), ("dwTypeData", wintypes.LPWSTR), ("cch", wintypes.UINT),
                ("hbmpItem", wintypes.HBITMAP)]


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
user32.GetMenuItemCount.argtypes = [wintypes.HMENU]
user32.SetMenuItemInfoW.argtypes = [wintypes.HMENU, wintypes.UINT, wintypes.BOOL, ctypes.POINTER(MENUITEMINFOW)]
user32.GetSystemMetrics.argtypes = [ctypes.c_int]
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


RIM, WHITE, CLEAR = (60, 60, 60, 255), (255, 255, 255, 255), (0, 0, 0, 0)


def glyph(state: str, recording: bool, x: float, y: float) -> tuple[int, int, int, int]:
    """The colour (r, g, b, a) of the icon at (x, y), both -1..1 with y down. Each state has its own shape as well
    as its own colour, so the icon reads without knowing the colours: an eye (watching), lines of text (reading
    pixels), pause bars (paused), a slashed ring (idle). A red dot at the lower right means history is recording."""
    if recording:
        b = math.hypot(x - 0.6, y - 0.6)
        if b < 0.4:
            return (220, 40, 40, 255) if b < 0.27 else WHITE
    d = math.hypot(x, y)
    if d > 1:
        return CLEAR
    fill = (*COLORS[state], 255)
    if state == "idle":                                  # hollow: nothing inside is being read
        return fill if d > 0.78 or abs(x - y) < 0.2 else CLEAR
    if d > 0.84:
        return RIM
    if state == "watching":                              # an eye: white almond, pupil in the fill colour
        if abs(x) < 0.72 and abs(y) < 0.46 * (1 - (x / 0.72) ** 2):
            return fill if math.hypot(x, y) < 0.2 else WHITE
    elif state == "reading":                             # three lines of text
        for cy, end in ((-0.36, 0.42), (0.0, 0.42), (0.36, 0.12)):
            if abs(y - cy) < 0.11 and -0.42 < x < end:
                return WHITE
    elif state == "paused":                              # two bars
        if abs(y) < 0.42 and (abs(x - 0.22) < 0.11 or abs(x + 0.22) < 0.11):
            return RIM
    return fill


def make_bitmap(state: str, recording: bool, size: int) -> int:
    """The icon as a 32-bit premultiplied-alpha DIB section (an HBITMAP), 4x4 supersampled."""
    bmi = BITMAPINFOHEADER(ctypes.sizeof(BITMAPINFOHEADER), size, -size, 1, 32, 0, 0, 0, 0, 0, 0)
    bits = ctypes.c_void_p()
    bmp = gdi32.CreateDIBSection(None, ctypes.byref(bmi), 0, ctypes.byref(bits), None, 0)
    px = (ctypes.c_uint32 * (size * size)).from_address(bits.value)
    n = 4
    for py in range(size):
        for qx in range(size):
            acc = [0.0, 0.0, 0.0, 0.0]                   # premultiplied r, g, b and alpha
            for sy in range(n):
                for sx in range(n):
                    x = ((qx + (sx + 0.5) / n) / size) * 2 - 1
                    y = ((py + (sy + 0.5) / n) / size) * 2 - 1
                    r, g, b, a = glyph(state, recording, x, y)
                    acc[0] += r * a
                    acc[1] += g * a
                    acc[2] += b * a
                    acc[3] += a
            r, g, b = (int(v / (n * n * 255)) for v in acc[:3])
            a = int(acc[3] / (n * n))
            px[py * size + qx] = (a << 24) | (r << 16) | (g << 8) | b
    return bmp


def make_icon(state: str, recording: bool, size: int) -> int:
    """The icon as an HICON."""
    color = make_bitmap(state, recording, size)
    mask = gdi32.CreateBitmap(size, size, 1, 1, None)
    info = ICONINFO(True, 0, 0, mask, color)
    icon = user32.CreateIconIndirect(ctypes.byref(info))
    gdi32.DeleteObject(mask)
    gdi32.DeleteObject(color)
    return icon


def service_running(beacons: list[dict]) -> bool:
    """A live beacon from the background service, other than this process: it shows the icon, so agents' servers
    leave it to it. Only a service that starts its tray writes a beacon."""
    return any(b.get("service") and b.get("pid") != os.getpid() for b in beacons)


class Tray:
    """Shows the icon while this process holds ``tray.lock``; waits for it otherwise. With ``yield_to_service`` (an
    agent's server) it neither takes the lock nor keeps it while the background service runs, so the service owns
    the icon whenever it is up and an agent's server shows it only when there is no service."""

    def __init__(self, takeover_s: float = 15.0, refresh_ms: int = 2000, status_dir: Path | None = None,
                 lock: Path | None = None, pause_file: Path | None = None, yield_to_service: bool = False):
        self.takeover_s, self.refresh_ms, self.yield_to_service = takeover_s, refresh_ms, yield_to_service
        self.status_dir, self.lock_path = status_dir or STATUS_DIR, lock or LOCK
        self.pause_file = pause_file or PAUSE_FILE
        self.state, self.tip, self.shown, self.hidden = None, "", False, False
        self.recording = False
        self.hwnd = None
        self._lock = None
        self._icons: dict[tuple[str, bool], int] = {}       # (state, recording) -> HICON, made on first use
        self._bitmaps: dict[tuple[str, bool], int] = {}     # the same pictures for the menu, as HBITMAPs
        self._size = 16
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

    def _yielding(self, beacons: list[dict] | None = None) -> bool:
        return self.yield_to_service and service_running(read_beacons(self.status_dir) if beacons is None
                                                         else beacons)

    def _run(self):
        while not self._stop.is_set():
            if not self._yielding():
                self._lock = take_lock(self.lock_path)
                if self._lock is not None:
                    try:
                        self._loop()                     # until stop, or until the service comes up
                    finally:
                        self._lock.close()
                        self._lock = None
                    continue
            self._stop.wait(self.takeover_s)

    def _nid(self) -> NOTIFYICONDATAW:
        n = NOTIFYICONDATAW()
        n.cbSize, n.hWnd, n.uID = ctypes.sizeof(NOTIFYICONDATAW), self.hwnd, 1
        n.uFlags, n.uCallbackMessage = NIF_MESSAGE | NIF_ICON | NIF_TIP, CALLBACK
        key = (self.state, self.recording)
        if key not in self._icons:
            self._icons[key] = make_icon(*key, self._size)
        n.hIcon = self._icons[key]
        n.szTip = self.tip
        return n

    def refresh(self):
        beacons = read_beacons(self.status_dir)
        if self._yielding(beacons) and self.hwnd:
            user32.PostMessageW(self.hwnd, WM_CLOSE, 0, 0)    # hand the icon to the service; _run waits again
            return
        d = describe(beacons, self.pause_file.exists())
        now = (d["state"], d["recording"], d["tip"])
        if self.hidden or now == (self.state, self.recording, self.tip) and self.shown:
            return
        self.state, self.recording, self.tip = now
        n = self._nid()
        if self.shown:
            shell32.Shell_NotifyIconW(NIM_MODIFY, ctypes.byref(n))
        else:
            self.shown = bool(shell32.Shell_NotifyIconW(NIM_ADD, ctypes.byref(n)))

    def _append(self, m, flags: int, cmd: int, text: str, picture: tuple[str, bool] | None = None):
        """One menu item, with the icon for ``picture`` beside it."""
        user32.AppendMenuW(m, flags, cmd, text)
        if picture is not None:
            if picture not in self._bitmaps:
                self._bitmaps[picture] = make_bitmap(*picture, self._size)
            mii = MENUITEMINFOW(cbSize=ctypes.sizeof(MENUITEMINFOW), fMask=MIIM_BITMAP,
                                hbmpItem=self._bitmaps[picture])
            user32.SetMenuItemInfoW(m, user32.GetMenuItemCount(m) - 1, True, ctypes.byref(mii))

    def _settings_menu(self, actions: dict) -> int:
        """The Settings submenu. Each item's command id maps to its action in ``actions``."""
        from . import always_on, secure, settings
        from . import service as svc

        def act(fn) -> int:
            cid = 100 + len(actions)
            actions[cid] = fn
            return cid

        def check(m, on: bool, text: str, fn, why: str | None = None):
            # Not ready yet (its model is not installed): it can still be chosen, as in setup, and starts once it is.
            label = f"{text}  (model not installed yet: starts once it is)" if why else text
            user32.AppendMenuW(m, MF_STRING | (MF_CHECKED if on else 0), act(fn), label)

        s = user32.CreatePopupMenu()
        p = settings.path()
        if p is None:
            user32.AppendMenuW(s, MF_STRING | MF_GRAYED, 0, "Fixed for this process by IGNITE_SCRIBE_POLICY")
            return s
        pol = settings.current(p)
        flag = lambda k: bool(pol.get(k, False))                                      # noqa: E731
        toggle = lambda k: (lambda: settings.set_flag(k, not flag(k), p))             # noqa: E731
        check(s, flag("history"), "Record screen history (encrypted, kept 7 days)", toggle("history"))
        ready, why = always_on.npu_recognizer_ready()
        check(s, flag("always_on_ocr"), "Always-on OCR (on the NPU)", toggle("always_on_ocr"), None if ready else why)
        ok, why = secure.available()
        check(s, flag("secure_mode"), "Secure mode: check text for prompt injection (iGPU)", toggle("secure_mode"),
              None if ok else why)
        check(s, flag("capture_elevated"), "Let vision read administrator windows", toggle("capture_elevated"))
        user32.AppendMenuW(s, MF_SEPARATOR, 0, None)
        logon = svc.installed() is not None
        check(s, logon, "Run in the background from logon (tray, history)",
              lambda: settings.service("uninstall" if logon else "install"))

        ex = user32.CreatePopupMenu()
        mine = settings.user_exclusions(p)
        for name in mine:
            user32.AppendMenuW(ex, MF_STRING | MF_CHECKED, act(lambda n=name: settings.remove_exclusion(n, p)),
                               f"{name}  (click to read it again)")
        if mine:
            user32.AppendMenuW(ex, MF_SEPARATOR, 0, None)
        add = user32.CreatePopupMenu()
        try:
            apps = [a for a in settings.running_apps() if a not in mine and a not in DEFAULT_EXCLUDE_PROCESSES]
        except Exception:  # noqa: BLE001
            apps = []
        for name in apps:
            user32.AppendMenuW(add, MF_STRING, act(lambda n=name: settings.add_exclusion(n, p)), name)
        if not apps:
            user32.AppendMenuW(add, MF_STRING | MF_GRAYED, 0, "(no other apps with open windows)")
        user32.AppendMenuW(ex, MF_POPUP, add, "Never read an open app")
        user32.AppendMenuW(ex, MF_SEPARATOR, 0, None)
        user32.AppendMenuW(ex, MF_STRING | MF_GRAYED, 0, "Always excluded (built in):")
        for name in DEFAULT_EXCLUDE_PROCESSES:
            user32.AppendMenuW(ex, MF_STRING | MF_GRAYED | MF_CHECKED, 0, name)
        user32.AppendMenuW(s, MF_SEPARATOR, 0, None)
        user32.AppendMenuW(s, MF_POPUP, ex,
                           f"Apps never read ({len(mine)} of yours, {len(DEFAULT_EXCLUDE_PROCESSES)} built in)")
        user32.AppendMenuW(s, MF_STRING, act(lambda: settings.open_file(p)),
                           "Edit the policy file (window-title patterns and more)...")
        return s

    def _menu(self):
        from . import settings
        paused = self.pause_file.exists()
        d = describe(read_beacons(self.status_dir), paused, settings.mtime(settings.path()))
        actions: dict = {}
        m = user32.CreatePopupMenu()
        self._append(m, MF_STRING, 0, "ignite-scribe: " + d["headline"], (d["state"], d["recording"]))
        for line in d["lines"] + [d["sessions"]] + ([d["stale"]] if d["stale"] else []):
            user32.AppendMenuW(m, MF_STRING | MF_GRAYED, 0, line)
        user32.AppendMenuW(m, MF_SEPARATOR, 0, None)
        user32.AppendMenuW(m, MF_POPUP, self._settings_menu(actions), "Settings")
        key = user32.CreatePopupMenu()                       # destroyed with its parent
        for picture, text in LEGEND:
            self._append(key, MF_STRING, 0, text, picture)
        user32.AppendMenuW(m, MF_POPUP, key, "What the icon means")
        user32.AppendMenuW(m, MF_SEPARATOR, 0, None)
        if d["always"] or paused:
            user32.AppendMenuW(m, MF_STRING, CMD_PAUSE, "Resume always-on OCR" if paused else "Pause always-on OCR")
        elif settings.path() is not None:                    # not running: on but waiting for its model, or off
            p = settings.path()
            on = bool(settings.current(p).get("always_on_ocr", False))
            cid = 100 + len(actions)
            actions[cid] = lambda: settings.set_flag("always_on_ocr", not on, p)
            user32.AppendMenuW(m, MF_STRING, cid, "Turn off always-on OCR (waiting for its model)" if on
                               else "Turn on always-on OCR")
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
        elif cmd in actions:
            try:
                actions[cmd]()
            except Exception as e:  # noqa: BLE001 - never let a menu action reach Win32
                print(f"tray action failed: {type(e).__name__}: {e}", file=sys.stderr)

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
        self._size = user32.GetSystemMetrics(SM_CXSMICON) or 16      # 16 at 100 % scaling, larger above it
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
            for h in self._bitmaps.values():
                gdi32.DeleteObject(h)
            self._icons, self._bitmaps = {}, {}
            user32.UnregisterClassW(cls, hinst)
            self.hwnd = None
