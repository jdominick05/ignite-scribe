# Copyright (C) 2026 The ignite-scribe contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""``get_selection``: the text the user highlighted in the focused window ("explain this", "fix this").

The policy runs first, as everywhere: the focused window comes from the Win32 list, and when the policy excludes
it, UI Automation is never asked; the result says only that its selection is withheld. Otherwise ``read(pid)``
(``uia.selection``) reads the focused element's Text-pattern selection, refusing password fields and focus in
another process. The text then goes through ``Policy.apply`` as a one-window snapshot, so it is scrubbed and
flagged like any other screen text: the selection may be a web page written to instruct an agent.
"""
from __future__ import annotations

from collections.abc import Callable

from .redact import Policy, scrub
from .snapshot import UNTRUSTED, Rect, Snapshot, TextBlock


def get(policy: Policy, snap: Snapshot, read: Callable[[int], dict]) -> dict:
    """``snap``: the raw Win32 window list (no content). -> the tool's result."""
    w = snap.focused()
    if w is None:
        return dict(note=UNTRUSTED, selection=None, why="no focused window")
    if policy.excluded(w):
        red, _ = policy.apply(Snapshot(snap.taken_at, [w], snap.monitors, list(snap.sources)))
        return dict(note=UNTRUSTED, selection=None, why="the focused window is withheld by the redaction policy",
                    window=_window(red.windows[0]))
    raw = read(w.pid)
    if "text" not in raw:
        red, _ = policy.apply(Snapshot(snap.taken_at, [w], snap.monitors, list(snap.sources)))
        why = ("a password field has the focus: never read" if raw.get("password") else
               raw.get("none") or raw.get("error") or "no selection")
        return dict(note=UNTRUSTED, selection=None, why=why, window=_window(red.windows[0]))
    rects = [Rect(*r) for r in raw.get("rects", [])]
    box = (Rect(min(r.left for r in rects), min(r.top for r in rects), max(r.right for r in rects),
                max(r.bottom for r in rects)) if rects else None)
    w.text = [TextBlock(raw["text"], box, "uia", raw.get("role"))]
    red, counts = policy.apply(Snapshot(snap.taken_at, [w], snap.monitors, list(snap.sources)))
    rw = red.windows[0]
    text = rw.text[0].text if rw.text else ""
    control, k = scrub(raw.get("name") or "")
    out = dict(note=UNTRUSTED, selection=text or None, why=None if text else "nothing is selected (a caret only)",
               truncated=bool(raw.get("truncated")), role=raw.get("role"), control=control or None,
               rects=raw.get("rects", []), window=_window(rw), secrets_withheld=counts["secrets_withheld"] + k)
    if rw.flags:
        out["flags"] = rw.flags
    return out


def _window(w) -> dict:
    return dict(hwnd=w.hwnd, title=w.title, process=w.process, redacted=w.redacted)
