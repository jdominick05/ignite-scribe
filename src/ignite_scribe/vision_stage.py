# Copyright (C) 2026 The ignite-scribe contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The vision stage: UI elements in the windows UI Automation could not describe, found on the NPU.

Opt-in per call (the MCP tool ``find_ui_elements``); nothing here runs on its own. For each chosen window:
1. **choose**: not minimized, not excluded (by process or title, or already redacted), and not elevated unless
   the policy sets ``capture_elevated``. Without ``match``, only windows UI Automation left thin are chosen:
   no text, or a "no readable content" note.
2. **capture**: ``capture.masked_window``. Windows above it, excluded windows and password fields are never
   copied (grey 114 instead), and the frame stays in memory.
3. **tile**: at scale 96 / the window's DPI (``docs/VISION.md``: native at 100%), 640 tiles overlapping by 96.
4. **detect**: ``vision_client.VisionWorker`` (the NPU, else the CPU on the identical reference model).
5. **merge**: map to screen pixels, NMS across tiles, drop boxes mostly inside masked areas, and drop boxes
   UI Automation already covers (an element block whose rect overlaps it).
The result is ``Window.elements``: boxes and scores, no text and no pixels.
"""
from __future__ import annotations

import time

import numpy as np

from . import capture
from . import vision as V
from .snapshot import Element, Rect, Snapshot, Window

THIN = "no readable content"


def resize_box(img: np.ndarray, wh: tuple[int, int]) -> np.ndarray:
    """Area-average downscale (or nearest upscale) in numpy: the client process has no OpenCV."""
    w, h = wh
    H, W = img.shape[:2]
    if w <= W and h <= H:
        ys = (np.arange(h + 1) * H / h).astype(int)
        xs = (np.arange(w + 1) * W / w).astype(int)
        c = np.cumsum(np.cumsum(np.pad(img.astype(np.float64), ((1, 0), (1, 0), (0, 0))), 0), 1)
        s = c[ys[1:]][:, xs[1:]] - c[ys[:-1]][:, xs[1:]] - c[ys[1:]][:, xs[:-1]] + c[ys[:-1]][:, xs[:-1]]
        area = np.outer(np.diff(ys), np.diff(xs))[:, :, None]
        return np.clip(np.round(s / area), 0, 255).astype(np.uint8)
    return img[(np.arange(h) * H // h)][:, (np.arange(w) * W // w)]


def excluded_fn(policy):
    return lambda w: w.redacted or policy.excluded(w)


def choose(snap: Snapshot, policy, match: str | None = None, max_windows: int = 3,
           is_elevated=None) -> tuple[list[Window], dict[int, str]]:
    """-> (windows to look at, focused first then z; {hwnd: why skipped})."""
    is_elevated = is_elevated or capture.elevated
    excl = excluded_fn(policy)
    m = (match or "").lower()
    picked, skipped = [], {}
    for w in sorted(snap.windows, key=lambda w: (not w.focused, w.z)):
        if m and m not in w.title.lower() and m not in w.process.lower():
            continue
        if w.minimized:
            skipped[w.hwnd] = "minimized"
        elif excl(w):
            skipped[w.hwnd] = "excluded by policy"
        elif not policy.capture_elevated and is_elevated(w.pid):
            skipped[w.hwnd] = "elevated (set capture_elevated in the policy to allow)"
        elif not m and w.text and not any(THIN in n for n in w.notes):
            skipped[w.hwnd] = "UI Automation already describes it"
        elif len(picked) < max_windows:
            picked.append(w)
    return picked, skipped


def _overlap(a: np.ndarray, rects: list[Rect]) -> np.ndarray:
    """Fraction of each box (N x 4, screen xyxy) covered by ``rects`` (summed, capped at 1)."""
    if not len(a) or not rects:
        return np.zeros(len(a))
    r = np.array([[x.left, x.top, x.right, x.bottom] for x in rects], np.float32)
    ix = np.clip(np.minimum(a[:, None, 2], r[None, :, 2]) - np.maximum(a[:, None, 0], r[None, :, 0]), 0, None)
    iy = np.clip(np.minimum(a[:, None, 3], r[None, :, 3]) - np.maximum(a[:, None, 1], r[None, :, 1]), 0, None)
    area = np.maximum((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), 1e-6)
    return np.minimum((ix * iy).sum(1) / area, 1.0)


def run(snap: Snapshot, policy, worker, match: str | None = None, max_windows: int = 3, max_tiles: int = 48,
        conf: float = 0.05, iou: float = 0.45) -> dict:
    """Fill ``Window.elements`` of the chosen windows in ``snap`` (a redacted snapshot) -> a report."""
    picked, skipped = choose(snap, policy, match, max_windows)
    excl = excluded_fn(policy)
    report = dict(windows=[], skipped=[dict(hwnd=h, why=s) for h, s in skipped.items()], tiles=0)
    budget = max_tiles
    for w in picked:
        t0 = time.perf_counter()
        cap = capture.masked_window(snap, w, excl)
        if cap is None:
            report["windows"].append(dict(hwnd=w.hwnd, title=w.title, note="not on screen"))
            continue
        img, origin, masked = cap
        scale = min(1.0, 96 / capture.dpi(w.hwnd))
        tiles = list(V.tiles(img, 640, scale, 96, resize_box))
        if len(tiles) > budget:
            report["windows"].append(dict(hwnd=w.hwnd, title=w.title, note=f"needs {len(tiles)} tiles, "
                                                                            f"{budget} left in this call's budget"))
            continue
        budget -= len(tiles)
        reply = worker.detect([t for t, *_ in tiles], conf=conf, iou=iou)
        bs, ss = [], []
        for (t, x0, y0, s), res in zip(tiles, reply["tiles"]):
            if res["boxes"]:
                b = V.to_screen(np.array(res["boxes"], np.float32), x0, y0, s)
                b[:, [0, 2]] += origin.left
                b[:, [1, 3]] += origin.top
                bs.append(b)
                ss.append(np.array(res["scores"], np.float32))
        found = kept_n = 0
        if bs:
            b, s = np.concatenate(bs), np.concatenate(ss)
            k = V.nms(b, s, iou)
            b, s = b[k], s[k]
            found = len(b)
            masks = capture.mask_rects(snap, w, excl)
            uia = [x.rect for x in w.text if x.rect is not None]
            keep = (_overlap(b, masks) < 0.5) & (_overlap(b, uia) < 0.5)
            b, s = b[keep], s[keep]
            kept_n = len(b)
            w.elements = [Element(Rect(int(x0), int(y0), int(x1), int(y1)), float(sc), "icon", reply["backend"])
                          for (x0, y0, x1, y1), sc in zip(b.round().tolist(), s.tolist())]
        report["tiles"] += len(tiles)
        report["windows"].append(dict(
            hwnd=w.hwnd, title=w.title, process=w.process, backend=reply["backend"], tiles=len(tiles),
            scale=scale, masked_fraction=round(masked, 3), detected=found, kept=kept_n,
            detector_ms=round(sum(reply["ms"]), 2), total_ms=round((time.perf_counter() - t0) * 1000, 1)))
    return report


def ocr_run(snap: Snapshot, policy, worker, match: str | None = None, max_windows: int = 3,
            max_pixels: int = 12_000_000) -> tuple[Snapshot, dict]:
    """Read the text in the chosen windows' pixels -> (a redacted copy of ``snap`` with OCR blocks added, a report).

    Same choice and masked capture as ``run``. Each window's lines become ``TextBlock(source="ocr")`` in screen
    pixels, and lines lying mostly in masked areas are dropped. The result goes through ``policy.apply`` again,
    so OCR text is scrubbed exactly like UI Automation text.
    """
    from .snapshot import TextBlock
    picked, skipped = choose(snap, policy, match, max_windows)
    excl = excluded_fn(policy)
    report = dict(windows=[], skipped=[dict(hwnd=h, why=s) for h, s in skipped.items()])
    budget = max_pixels
    for w in picked:
        t0 = time.perf_counter()
        cap = capture.masked_window(snap, w, excl)
        if cap is None:
            report["windows"].append(dict(hwnd=w.hwnd, title=w.title, note="not on screen"))
            continue
        img, origin, masked = cap
        if img.shape[0] * img.shape[1] > budget:
            report["windows"].append(dict(hwnd=w.hwnd, title=w.title, note="over this call's pixel budget"))
            continue
        budget -= img.shape[0] * img.shape[1]
        reply = worker.ocr([img])
        res = reply["images"][0]
        lines = res["lines"]
        b = np.array([l["box"] for l in lines], np.float32).reshape(-1, 4)
        b[:, [0, 2]] += origin.left
        b[:, [1, 3]] += origin.top
        keep = _overlap(b, capture.mask_rects(snap, w, excl)) < 0.5
        added = [TextBlock(l["text"], Rect(*(int(v) for v in bb)), "ocr")
                 for l, bb, k in zip(lines, b.tolist(), keep) if k]
        w.text = list(w.text) + added
        report["windows"].append(dict(
            hwnd=w.hwnd, title=w.title, process=w.process, ocr_backend=reply.get("ocr_backend"),
            masked_fraction=round(masked, 3), lines=len(added), dropped_masked=int((~keep).sum()),
            det_ms=res.get("det_ms"), rec_ms=res.get("rec_ms"),
            total_ms=round((time.perf_counter() - t0) * 1000, 1)))
    red, counts = policy.apply(snap)
    report["secrets_withheld"] = counts["secrets_withheld"]
    return red, report
