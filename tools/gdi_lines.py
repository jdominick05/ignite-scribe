# Copyright (C) 2026 The ignite-scribe contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Text lines drawn the way Windows draws UI text: GDI ``DrawTextW`` with ClearType, into an off-screen bitmap.

Nothing here reads the screen. Each line is rendered into its own memory DC in a UI font (Segoe UI, Tahoma, Arial,
Calibri, Consolas, Verdana) at UI pixel sizes, on light or dark backgrounds, so the eval set has the subpixel
colour fringes and hinting of real screen text with exact transcriptions. It is still synthetic, and numbers
measured on it are labelled synthetic. ``--out DIR`` writes ``NNNN.png`` and ``labels.tsv``.
"""
import argparse
import ctypes
import random
import string
from ctypes import wintypes
from pathlib import Path

import numpy as np
from PIL import Image

gdi32 = ctypes.WinDLL("gdi32")
user32 = ctypes.WinDLL("user32")
gdi32.CreateCompatibleDC.restype = wintypes.HDC
gdi32.CreateCompatibleDC.argtypes = [wintypes.HDC]
gdi32.CreateDIBSection.restype = wintypes.HBITMAP
gdi32.CreateDIBSection.argtypes = [wintypes.HDC, ctypes.c_void_p, wintypes.UINT, ctypes.POINTER(ctypes.c_void_p),
                                   wintypes.HANDLE, wintypes.DWORD]
gdi32.SelectObject.restype = wintypes.HGDIOBJ
gdi32.SelectObject.argtypes = [wintypes.HDC, wintypes.HGDIOBJ]
gdi32.CreateFontW.restype = wintypes.HFONT
gdi32.CreateFontW.argtypes = [ctypes.c_int] * 5 + [wintypes.DWORD] * 8 + [wintypes.LPCWSTR]
gdi32.SetTextColor.argtypes = [wintypes.HDC, wintypes.DWORD]
gdi32.SetBkColor.argtypes = [wintypes.HDC, wintypes.DWORD]
gdi32.DeleteObject.argtypes = [wintypes.HGDIOBJ]
gdi32.DeleteDC.argtypes = [wintypes.HDC]
user32.DrawTextW.argtypes = [wintypes.HDC, wintypes.LPCWSTR, ctypes.c_int, ctypes.POINTER(wintypes.RECT), wintypes.UINT]
DT_CALCRECT, DT_SINGLELINE, DT_NOPREFIX, CLEARTYPE_QUALITY, OPAQUE = 0x400, 0x20, 0x800, 5, 2

FACES = ["Segoe UI", "Tahoma", "Arial", "Calibri", "Consolas", "Verdana"]
WORDS = ("Preferences Explorer Bookmarks History Extensions Repository Commit Branch Merge Pull Request Issues "
         "Notifications Calendar Contacts Attachment Reply Forward Archive Spam Trash Filter Sort Ascending "
         "Descending Columns Rows Chart Pivot Table Formula Sheet Slide Presentation Zoom Fit Width Page Ruler "
         "Grid Snap Align Center Justify Bold Italic Underline Font Size Color Theme Light Dark System Battery "
         "Wi-Fi Bluetooth Volume Brightness Keyboard Mouse Printer Scanner Storage Memory Processor Uptime "
         "Running Stopped Paused Queued Completed Failed Retry Ignore Abort Continue Skip Overwrite Replace "
         "a an as be it no or so up we you your this that with without into over under after before").split()


def token(rng: random.Random) -> str:
    r = rng.random()
    if r < 0.6:
        return rng.choice(WORDS)
    if r < 0.75:
        return str(rng.choice([rng.randint(0, 99), rng.randint(100, 99999), f"{rng.uniform(0, 100):.1f}"]))
    if r < 0.85:
        return rng.choice(["%s.%s" % ("".join(rng.choices(string.ascii_lowercase, k=rng.randint(3, 8))),
                                       rng.choice(["txt", "py", "json", "png", "md", "exe"])),
                           "%02d:%02d" % (rng.randint(0, 23), rng.randint(0, 59)),
                           "%d/%d/%d" % (rng.randint(1, 12), rng.randint(1, 28), rng.randint(2020, 2026))])
    return "".join(rng.choices(string.ascii_letters + string.digits, k=rng.randint(2, 7)))


def sample(rng: random.Random) -> str:
    return " ".join(token(rng) for _ in range(rng.randint(1, 6)))


def render(text: str, face: str, px: int, fg, bg, pad: int = 3) -> np.ndarray:
    dc = gdi32.CreateCompatibleDC(None)
    font = gdi32.CreateFontW(-px, 0, 0, 0, 400, 0, 0, 0, 1, 0, 0, CLEARTYPE_QUALITY, 0, face)
    old_font = gdi32.SelectObject(dc, font)
    r = wintypes.RECT(0, 0, 0, 0)
    user32.DrawTextW(dc, text, -1, ctypes.byref(r), DT_CALCRECT | DT_SINGLELINE | DT_NOPREFIX)
    w, h = r.right + 2 * pad, r.bottom + 2 * pad

    class BMI(ctypes.Structure):
        _fields_ = [("biSize", wintypes.DWORD), ("biWidth", wintypes.LONG), ("biHeight", wintypes.LONG),
                    ("biPlanes", wintypes.WORD), ("biBitCount", wintypes.WORD), ("rest", wintypes.DWORD * 6)]

    bmi = BMI(ctypes.sizeof(BMI), w, -h, 1, 32)
    bits = ctypes.c_void_p()
    dib = gdi32.CreateDIBSection(dc, ctypes.byref(bmi), 0, ctypes.byref(bits), None, 0)
    old_bmp = gdi32.SelectObject(dc, dib)
    buf = np.frombuffer((ctypes.c_ubyte * (w * h * 4)).from_address(bits.value), np.uint8).reshape(h, w, 4)
    buf[:, :, :3] = bg[::-1]                                    # BGR fill
    gdi32.SetTextColor(dc, fg[0] | fg[1] << 8 | fg[2] << 16)
    gdi32.SetBkColor(dc, bg[0] | bg[1] << 8 | bg[2] << 16)
    rr = wintypes.RECT(pad, pad, w - pad, h - pad)
    user32.DrawTextW(dc, text, -1, ctypes.byref(rr), DT_SINGLELINE | DT_NOPREFIX)
    out = buf[:, :, 2::-1].copy()
    gdi32.SelectObject(dc, old_bmp)
    gdi32.SelectObject(dc, old_font)
    gdi32.DeleteObject(dib)
    gdi32.DeleteObject(font)
    gdi32.DeleteDC(dc)
    return out


def lines(n: int, seed: int):
    rng = random.Random(seed)
    for _ in range(n):
        t = sample(rng)
        dark = rng.random() < 0.3
        bg = tuple(rng.randint(10, 45) for _ in range(3)) if dark else tuple(rng.randint(225, 255) for _ in range(3))
        fg = tuple(rng.randint(200, 250) for _ in range(3)) if dark else tuple(rng.randint(0, 60) for _ in range(3))
        yield t, render(t, rng.choice(FACES), rng.choice([12, 13, 14, 15, 16, 18, 20, 24]), fg, bg)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "labels.tsv", "w", encoding="utf-8", newline="\n") as f:
        for i, (t, img) in enumerate(lines(a.n, a.seed)):
            Image.fromarray(img).save(out / f"{i:04d}.png")
            f.write(f"{i:04d}.png\t{t}\n")
    print("wrote", a.n, "lines to", out)


if __name__ == "__main__":
    main()
