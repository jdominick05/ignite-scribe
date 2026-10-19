# Copyright (C) 2026 The ignite-scribe contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""A synthetic "canvas" window for always-on OCR measurements: text painted with GDI, nothing for UI Automation.

It stands in for an image viewer, a video or a canvas app: a plain top-level window whose client area is painted
with ``DrawTextW`` and has no child controls, so UI Automation finds no text and always-on OCR picks it. Every
``--every`` seconds it appends ``--lines`` new lines of random words (from a fixed vocabulary, seeded) to a
scrolling log, like a chat or a log viewer. Prints its HWND, then runs until killed or ``--seconds`` pass.
"""
import argparse
import ctypes
import random
import time
from ctypes import wintypes

user32 = ctypes.WinDLL("user32", use_last_error=True)
gdi32 = ctypes.WinDLL("gdi32", use_last_error=True)
user32.CreateWindowExW.restype = wintypes.HWND
user32.CreateWindowExW.argtypes = [wintypes.DWORD, wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.DWORD,
                                   ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, wintypes.HWND,
                                   wintypes.HMENU, wintypes.HINSTANCE, wintypes.LPVOID]
user32.GetDC.restype = wintypes.HDC
user32.GetDC.argtypes = [wintypes.HWND]
user32.ReleaseDC.argtypes = [wintypes.HWND, wintypes.HDC]
user32.FillRect.argtypes = [wintypes.HDC, ctypes.POINTER(wintypes.RECT), wintypes.HBRUSH]
user32.DrawTextW.argtypes = [wintypes.HDC, wintypes.LPCWSTR, ctypes.c_int, ctypes.POINTER(wintypes.RECT),
                             wintypes.UINT]
user32.GetClientRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
gdi32.GetStockObject.restype = wintypes.HGDIOBJ
gdi32.SelectObject.argtypes = [wintypes.HDC, wintypes.HGDIOBJ]
gdi32.SetBkMode.argtypes = [wintypes.HDC, ctypes.c_int]
gdi32.CreateFontW.restype = wintypes.HFONT
gdi32.CreateFontW.argtypes = [ctypes.c_int] * 5 + [wintypes.DWORD] * 8 + [wintypes.LPCWSTR]

WS_OVERLAPPEDWINDOW, WS_VISIBLE, PM_REMOVE, WHITE_BRUSH, TRANSPARENT = 0x00CF0000, 0x10000000, 1, 0, 1
WORDS = ("build test deploy queue latency cache window kernel token frame shader buffer render socket thread "
         "pixel batch tensor driver engine packet stream index commit branch merge review").split()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--every", type=float, default=1.0)
    ap.add_argument("--lines", type=int, default=2)
    ap.add_argument("--seconds", type=float, default=600)
    a = ap.parse_args()
    rng = random.Random(11)
    user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
    top = user32.CreateWindowExW(0, "STATIC", "", WS_OVERLAPPEDWINDOW | WS_VISIBLE, 160, 140, 1000, 700,
                                 None, None, None, None)
    user32.SetWindowTextW(top, "Canvas log - synthetic")
    font = gdi32.CreateFontW(-18, 0, 0, 0, 400, 0, 0, 0, 0, 0, 0, 5, 0, "Segoe UI")   # 5 = CLEARTYPE_QUALITY
    print(int(top), flush=True)
    log, msg, end, nxt = [], wintypes.MSG(), time.monotonic() + a.seconds, 0.0
    while time.monotonic() < end:
        while user32.PeekMessageW(ctypes.byref(msg), None, 0, 0, PM_REMOVE):
            user32.TranslateMessage(ctypes.byref(msg))
            user32.DispatchMessageW(ctypes.byref(msg))
        if time.monotonic() >= nxt:
            nxt = time.monotonic() + a.every
            log += [" ".join(rng.choice(WORDS) for _ in range(rng.randint(4, 10))) for _ in range(a.lines)]
            log = log[-28:]
            dc = user32.GetDC(top)
            r = wintypes.RECT()
            user32.GetClientRect(top, ctypes.byref(r))
            user32.FillRect(dc, ctypes.byref(r), gdi32.GetStockObject(WHITE_BRUSH))
            gdi32.SelectObject(dc, font)
            gdi32.SetBkMode(dc, TRANSPARENT)
            r.left, r.top = 16, 12
            user32.DrawTextW(dc, "\n".join(log), -1, ctypes.byref(r), 0)
            user32.ReleaseDC(top, dc)
        time.sleep(0.02)


if __name__ == "__main__":
    main()
