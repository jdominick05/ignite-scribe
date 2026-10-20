# Copyright (C) 2026 The ignite-scribe contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The element map: where to click. ``find(snap, ...)`` searches a *redacted* snapshot walked with detail
(``collect(detail=True)``) for controls by name, role and window, and says where each one can be clicked.

- **Coordinates** are physical screen pixels: this process is per-monitor DPI aware (``windows``). A click tool
  that is not DPI aware sees logical pixels and misses by the display scale (125% here), so the result says so.
- **The click point reaches the control's window.** The snapshot lists only taskbar windows, so menus, tooltips,
  the taskbar and topmost tool windows are not in it. The OS decides instead (``owner_at``, by default
  ``windows.root_at``: ``WindowFromPoint``, then its root): the centre is tried first, then the centre of each
  visible piece, largest first, and the first point whose root is the hit's window is ``click``; None when no
  point reaches it. ``click_checked`` says the OS confirmed it. Without ``owner_at`` (offline), the centre or the
  largest visible piece is used unchecked.
- ``visible_share`` is the rectangle arithmetic only, an estimate: the control clipped to its window and the
  monitors, minus every listed window above its own. Off-screen parts count as not visible.
- **Keys:** ``accelerator`` (e.g. Ctrl+S) works while the app has focus; ``access_key`` (Alt+s from a ``&Save``
  label) only while that dialog or menu is active.
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

from collections.abc import Callable

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
    """-> (the visible pieces of ``r``, the visible share of it). ``r`` is first clipped to its window and the
    monitors; listed windows above ``w`` cover it."""
    full = _area(r)
    c = Rect(max(r.left, w.rect.left), max(r.top, w.rect.top), min(r.right, w.rect.right),
             min(r.bottom, w.rect.bottom))
    c = clip(c, snap.monitors) if c.width > 0 and c.height > 0 else None
    if c is None or not full:
        return [], 0.0
    above = [o.rect for o in snap.windows if o.z < w.z and not o.minimized and o.hwnd != w.hwnd]
    pieces = free_rects(c, above)
    return pieces, round(sum(_area(p) for p in pieces) / full, 3)


def click_point(r: Rect, pieces: list[Rect], hwnd: int,
                owner_at: Callable[[int, int], int | None] | None = None) -> tuple[tuple[int, int] | None, bool]:
    """-> (a point that clicks the control, whether the OS confirmed it reaches ``hwnd``)."""
    centre = ((r.left + r.right) // 2, (r.top + r.bottom) // 2)
    pts = [((p.left + p.right) // 2, (p.top + p.bottom) // 2) for p in sorted(pieces, key=_area, reverse=True)]
    if owner_at is None:
        if any(_inside(*centre, p) for p in pieces):
            return centre, False
        return (pts[0] if pts else None), False
    for pt in [centre] + [p for p in pts if p != centre]:
        if owner_at(*pt) == hwnd:
            return pt, True
    return None, True


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
         enabled_only: bool = False, clickable: bool = True, limit: int = 10,
         owner_at: Callable[[int, int], int | None] | None = None) -> list[dict]:
    """Hits, best first (see the module docstring). ``role``: a UIA control type (case-insensitive); without it,
    only clickable roles count unless ``clickable`` is False. The OS check runs only for the hits returned."""
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
            pieces, share = visible(snap, w, b.rect)
            hit = dict(name=label or None, role=b.role, rect=[b.rect.left, b.rect.top, b.rect.right, b.rect.bottom],
                       visible_share=share, _pieces=pieces,
                       window=dict(hwnd=w.hwnd, title=w.title, process=w.process, z=w.z, focused=w.focused,
                                   flagged=bool(w.flags)))
            if password:
                hit["password_field"] = True
            else:
                for k in ("enabled", "accelerator", "access_key", "automation_id", "focused", "actions"):
                    if d.get(k) not in (None, [], ""):
                        hit[k] = d[k]
                value = b.text if label and b.text.startswith(label + ": ") else None
                if value:
                    hit["value"] = value[len(label) + 2:][:200]
            hits.append((-score, not w.focused, w.z, b.rect.top, b.rect.left, hit))
    hits.sort(key=lambda h: h[:5])
    out = [h[5] for h in hits[:max(1, limit)]]
    for h in out:
        r = Rect(*h["rect"])
        h["click"], checked = click_point(r, h.pop("_pieces"), h["window"]["hwnd"], owner_at)
        if checked:
            h["click_checked"] = True
    return out
