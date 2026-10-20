# Copyright (C) 2026 The ignite-scribe contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Rectangle arithmetic in screen pixels, stdlib only: shared by the masked capture (``capture``) and the element
map (``elements``), which must not need numpy."""
from __future__ import annotations

from .snapshot import Rect


def clip(r: Rect, bounds: list[Rect]) -> Rect | None:
    """``r`` clipped to the union's bounding box of the monitors (off-screen parts cannot be copied)."""
    if not bounds:
        return r
    b = Rect(min(m.left for m in bounds), min(m.top for m in bounds), max(m.right for m in bounds),
             max(m.bottom for m in bounds))
    c = Rect(max(r.left, b.left), max(r.top, b.top), min(r.right, b.right), min(r.bottom, b.bottom))
    return c if c.width > 0 and c.height > 0 else None


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
