# Copyright (C) 2026 The ignite-scribe contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The element map: where to click. ``find(snap, ...)`` searches a *redacted* snapshot walked with detail
(``collect(detail=True)``) for controls by name, role and window, and says where each one can be clicked.

- **Coordinates** are physical screen pixels: this process is per-monitor DPI aware (``windows``). A click tool
  that is not DPI aware sees logical pixels and misses by the display scale (125% here), so the result says so.
- **The click point is a visible one.** The control's rectangle is clipped to its window and the monitors, then
  every window above its own (lower z) is subtracted. ``occluded`` is the covered share; ``click`` is the centre
  when that is visible, else the centre of the largest visible piece, else None (clicking would hit another
  window).
- **Ranking:** an exact name first, then a name starting with the query, then a word of the name starting with
  it, then a substring of the name, then of its AutomationId, then of its value. Ties go to the focused window,
  then by z, then top to bottom.
- **Untrusted:** a name is screen text, and a page can label anything "Save". Every hit carries its window (hwnd,
  title, process, z) and whether that window's text is flagged as addressed to an agent.

Nothing is read here: the walk (and the policy before it, so an excluded window is never walked) happened in
``collect``, and ``Policy.apply`` scrubbed the names. A redacted window yields no hits. A password field is a hit
(an agent may need to click into one) with its rectangle and role only.
"""
from __future__ import annotations

from .rects import clip, free_rects
from .snapshot import Rect, Snapshot, Window
from .uia import CLICKABLE

COORDS = ("physical screen pixels (per-monitor DPI aware); a click tool that is not DPI aware must divide by the "
          "display scale")


def _area(r: Rect) -> int:
    return max(0, r.width) * max(0, r.height)


def _inside(x: int, y: int, r: Rect) -> bool:
    return r.left <= x < r.right and r.top <= y < r.bottom


def visible(snap: Snapshot, w: Window, r: Rect) -> tuple[list[Rect], float]:
    """-> (the visible pieces of ``r``, the covered share of it). ``r`` is first clipped to its window and the
    monitors; windows above ``w`` cover it."""
    full = _area(r)
    c = Rect(max(r.left, w.rect.left), max(r.top, w.rect.top), min(r.right, w.rect.right),
             min(r.bottom, w.rect.bottom))
    c = clip(c, snap.monitors) if c.width > 0 and c.height > 0 else None
    if c is None or not full:
        return [], 1.0
    above = [o.rect for o in snap.windows if o.z < w.z and not o.minimized and o.hwnd != w.hwnd]
    pieces = free_rects(c, above)
    return pieces, round(1 - sum(_area(p) for p in pieces) / full, 3)


def click_point(r: Rect, pieces: list[Rect]) -> tuple[int, int] | None:
    x, y = (r.left + r.right) // 2, (r.top + r.bottom) // 2
    if any(_inside(x, y, p) for p in pieces):
        return x, y
    if not pieces:
        return None
    p = max(pieces, key=_area)
    return (p.left + p.right) // 2, (p.top + p.bottom) // 2


def _score(q: str, name: str, aid: str, text: str) -> float:
    if not q:
        return 1.0
    n = name.lower()
    if n == q:
        return 6.0
    if n.startswith(q):
        return 5.0
    if any(word.startswith(q) for word in n.replace("&", "").split()):
        return 4.0
    if q in n:
        return 3.0
    if q in aid.lower():
        return 2.0
    if q in text.lower():
        return 1.0
    return 0.0


def find(snap: Snapshot, name: str | None = None, role: str | None = None, match: str | None = None,
         enabled_only: bool = False, clickable: bool = True, limit: int = 10) -> list[dict]:
    """Hits, best first (see the module docstring). ``role``: a UIA control type (case-insensitive); without it,
    only clickable roles count unless ``clickable`` is False."""
    q = (name or "").strip().lower()
    want = (role or "").strip().lower()
    m = (match or "").lower()
    hits = []
    for w in snap.windows:
        if w.redacted or w.minimized:
            continue
        if m and m not in w.title.lower() and m not in w.process.lower():
            continue
        for b in w.text:
            if b.source != "uia" or b.rect is None or b.role is None:
                continue
            if want and b.role.lower() != want:
                continue
            if not want and clickable and b.role not in CLICKABLE:
                continue
            d = b.detail or {}
            password = b.text == "[password field]"
            label = "" if password else (d.get("name") or (b.text if "\n" not in b.text else ""))
            score = _score(q, label, d.get("automation_id") or "", "" if password else b.text)
            if not score or (enabled_only and d.get("enabled") is False):
                continue
            pieces, occluded = visible(snap, w, b.rect)
            hit = dict(name=label or None, role=b.role, rect=[b.rect.left, b.rect.top, b.rect.right, b.rect.bottom],
                       click=click_point(b.rect, pieces), occluded=occluded,
                       window=dict(hwnd=w.hwnd, title=w.title, process=w.process, z=w.z, focused=w.focused,
                                   flagged=bool(w.flags)))
            if password:
                hit["password_field"] = True
            else:
                for k in ("enabled", "keys", "automation_id", "focused", "actions"):
                    if d.get(k) not in (None, [], ""):
                        hit[k] = d[k]
                value = b.text if label and b.text.startswith(label + ": ") else None
                if value:
                    hit["value"] = value[len(label) + 2:][:200]
            hits.append((-score, not w.focused, w.z, b.rect.top, b.rect.left, hit))
    hits.sort(key=lambda h: h[:5])
    return [h[5] for h in hits[:max(1, limit)]]
