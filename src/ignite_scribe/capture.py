# Copyright (C) 2026 The ignite-scribe contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Pixels, for the vision stage only: one window's on-screen rectangle, masked before anything sees it.

This is the only module that reads pixels, and it keeps them in memory: nothing here writes an image anywhere.
``grab`` copies a screen rectangle with GDI (``BitBlt`` from the screen DC, ``CAPTUREBLT``), so it sees what is
on screen: the window plus whatever overlaps it. ``masked_window`` therefore never copies these areas at all
(only the rest of the rectangle is blitted, onto a canvas of the detector's pad grey, 114):
- every window above the target in z-order (their pixels are not this window's, and may be excluded ones);
- every window the policy excludes, wherever it is;
- every password field UI Automation reported (``[password field]`` blocks, in any window).

``elevated(pid)`` says whether a process runs elevated. UI Automation cannot read those (UIPI), but a screen
copy can, so vision skips them unless the policy sets ``capture_elevated``. A process whose token cannot be
queried counts as elevated.
"""
from __future__ import annotations

import ctypes
import sys
from ctypes import wintypes

import numpy as np

from .snapshot import Rect, Snapshot, Window

if sys.platform != "win32":
    raise ImportError("ignite_scribe.capture needs Windows")

PAD = 114
user32 = ctypes.WinDLL("user32", use_last_error=True)
gdi32 = ctypes.WinDLL("gdi32", use_last_error=True)
kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)
user32.GetDC.restype = wintypes.HDC
user32.GetDC.argtypes = [wintypes.HWND]
user32.ReleaseDC.argtypes = [wintypes.HWND, wintypes.HDC]
user32.GetDpiForWindow.restype = wintypes.UINT
user32.GetDpiForWindow.argtypes = [wintypes.HWND]
gdi32.CreateCompatibleDC.restype = wintypes.HDC
gdi32.CreateCompatibleDC.argtypes = [wintypes.HDC]
gdi32.CreateDIBSection.restype = wintypes.HBITMAP
gdi32.CreateDIBSection.argtypes = [wintypes.HDC, ctypes.c_void_p, wintypes.UINT, ctypes.POINTER(ctypes.c_void_p),
                                   wintypes.HANDLE, wintypes.DWORD]
gdi32.SelectObject.restype = wintypes.HGDIOBJ
gdi32.SelectObject.argtypes = [wintypes.HDC, wintypes.HGDIOBJ]
gdi32.BitBlt.argtypes = [wintypes.HDC, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, wintypes.HDC,
                         ctypes.c_int, ctypes.c_int, wintypes.DWORD]
gdi32.DeleteObject.argtypes = [wintypes.HGDIOBJ]
gdi32.DeleteDC.argtypes = [wintypes.HDC]
kernel32.OpenProcess.restype = wintypes.HANDLE
advapi32.OpenProcessToken.argtypes = [wintypes.HANDLE, wintypes.DWORD, ctypes.POINTER(wintypes.HANDLE)]
advapi32.GetTokenInformation.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD,
                                         ctypes.POINTER(wintypes.DWORD)]
SRCCOPY, CAPTUREBLT = 0x00CC0020, 0x40000000
PROCESS_QUERY_LIMITED_INFORMATION, TOKEN_QUERY, TOKEN_ELEVATION = 0x1000, 0x0008, 20


class BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [("biSize", wintypes.DWORD), ("biWidth", wintypes.LONG), ("biHeight", wintypes.LONG),
                ("biPlanes", wintypes.WORD), ("biBitCount", wintypes.WORD), ("biCompression", wintypes.DWORD),
                ("biSizeImage", wintypes.DWORD), ("biXPelsPerMeter", wintypes.LONG),
                ("biYPelsPerMeter", wintypes.LONG), ("biClrUsed", wintypes.DWORD),
                ("biClrImportant", wintypes.DWORD)]


def grab(r: Rect) -> np.ndarray:
    """The screen pixels of ``r`` (physical pixels) -> HxWx3 uint8 RGB, in memory."""
    w, h = r.width, r.height
    if w <= 0 or h <= 0:
        return np.zeros((0, 0, 3), np.uint8)
    screen = user32.GetDC(None)
    mem = gdi32.CreateCompatibleDC(screen)
    bits = ctypes.c_void_p()
    bmi = BITMAPINFOHEADER(ctypes.sizeof(BITMAPINFOHEADER), w, -h, 1, 32, 0, 0, 0, 0, 0, 0)   # top-down BGRA
    dib = gdi32.CreateDIBSection(screen, ctypes.byref(bmi), 0, ctypes.byref(bits), None, 0)
    old = gdi32.SelectObject(mem, dib)
    try:
        if not gdi32.BitBlt(mem, 0, 0, w, h, screen, r.left, r.top, SRCCOPY | CAPTUREBLT):
            raise OSError(ctypes.get_last_error(), "BitBlt failed")
        buf = (ctypes.c_ubyte * (w * h * 4)).from_address(bits.value)
        bgra = np.frombuffer(buf, np.uint8).reshape(h, w, 4)
        return bgra[:, :, 2::-1].copy()                          # BGRA -> RGB, owned copy
    finally:
        gdi32.SelectObject(mem, old)
        gdi32.DeleteObject(dib)
        gdi32.DeleteDC(mem)
        user32.ReleaseDC(None, screen)


def dpi(hwnd: int) -> int:
    return int(user32.GetDpiForWindow(hwnd)) or 96


def elevated(pid: int) -> bool:
    h = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not h:
        return True
    tok = wintypes.HANDLE()
    try:
        if not advapi32.OpenProcessToken(h, TOKEN_QUERY, ctypes.byref(tok)):
            return True
        try:
            v, n = wintypes.DWORD(), wintypes.DWORD()
            if not advapi32.GetTokenInformation(tok, TOKEN_ELEVATION, ctypes.byref(v), 4, ctypes.byref(n)):
                return True
            return bool(v.value)
        finally:
            kernel32.CloseHandle(tok)
    finally:
        kernel32.CloseHandle(h)


def clip(r: Rect, bounds: list[Rect]) -> Rect | None:
    """``r`` clipped to the union's bounding box of the monitors (off-screen parts cannot be copied)."""
    if not bounds:
        return r
    b = Rect(min(m.left for m in bounds), min(m.top for m in bounds), max(m.right for m in bounds),
             max(m.bottom for m in bounds))
    c = Rect(max(r.left, b.left), max(r.top, b.top), min(r.right, b.right), min(r.bottom, b.bottom))
    return c if c.width > 0 and c.height > 0 else None


def mask_rects(snap: Snapshot, target: Window, excluded) -> list[Rect]:
    """Screen rectangles to blank in ``target``'s capture (see the module docstring)."""
    out = []
    for w in snap.windows:
        if w.hwnd == target.hwnd or w.minimized:
            continue
        if w.z < target.z or excluded(w):
            out.append(w.rect)
    for w in snap.windows:
        out += [b.rect for b in w.text if b.rect is not None and b.text == "[password field]"]
    return out


def free_rects(r: Rect, masks: list[Rect]) -> list[Rect]:
    """``r`` minus the union of ``masks``, as disjoint rectangles (coordinate compression, rows merged)."""
    ms = [Rect(max(m.left, r.left), max(m.top, r.top), min(m.right, r.right), min(m.bottom, r.bottom))
          for m in masks]
    ms = [m for m in ms if m.width > 0 and m.height > 0]
    if not ms:
        return [r]
    xs = sorted({r.left, r.right, *[m.left for m in ms], *[m.right for m in ms]})
    ys = sorted({r.top, r.bottom, *[m.top for m in ms], *[m.bottom for m in ms]})
    out = []
    for y0, y1 in zip(ys, ys[1:]):
        run = None
        for x0, x1 in zip(xs, xs[1:]):
            covered = any(m.left <= x0 and x1 <= m.right and m.top <= y0 and y1 <= m.bottom for m in ms)
            if not covered:
                run = (run[0], x1) if run else (x0, x1)
            elif run:
                out.append(Rect(run[0], y0, run[1], y1))
                run = None
        if run:
            out.append(Rect(run[0], y0, run[1], y1))
    return out


def masked_window(snap: Snapshot, target: Window, excluded,
                  region: Rect | None = None) -> tuple[np.ndarray, Rect, float] | None:
    """-> (RGB pixels of the target's visible rectangle with every mask area PAD grey, that rectangle, masked
    fraction) or None. Masked areas are never copied from the screen: only the free rectangles are blitted onto
    a grey canvas, so an excluded window's pixels never enter this process. ``region`` (screen pixels) limits
    the copy to that part of the window (always-on OCR reads only what changed)."""
    r = clip(target.rect, snap.monitors)
    if r is not None and region is not None:
        r = Rect(max(r.left, region.left), max(r.top, region.top), min(r.right, region.right),
                 min(r.bottom, region.bottom))
        if r.width <= 0 or r.height <= 0:
            r = None
    if r is None:
        return None
    img = np.full((r.height, r.width, 3), PAD, np.uint8)
    free = free_rects(r, mask_rects(snap, target, excluded))
    copied = 0
    for f in free:
        img[f.top - r.top:f.bottom - r.top, f.left - r.left:f.right - r.left] = grab(f)
        copied += f.width * f.height
    return img, r, 1.0 - copied / (r.width * r.height)
