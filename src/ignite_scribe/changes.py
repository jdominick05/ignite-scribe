# Copyright (C) 2026 The ignite-scribe contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""What changed between two redacted snapshots, as a list of events.

Both inputs must already be redacted (``Policy.apply``), so an event can never carry a withheld title or a
secret: the diff only ever sees what a snapshot would have shown.

Event kinds: ``opened``, ``closed``, ``focus``, ``title``, ``moved``, ``minimized``, ``restored``, ``content``.
A ``content`` event lists the lines that appeared (in screen order, capped) and counts the lines that went. It is
emitted only when both snapshots read the window's content, so a window dropping out of the read set (or a
walk that hit its cap) is not reported as text disappearing.
"""
from __future__ import annotations

from .snapshot import Snapshot, Window

MAX_LINES = 40          # lines listed per content event
MAX_CHARS = 4000        # characters listed per content event
MERGED = ("title", "moved", "content")


def compact(events: list[dict]) -> list[dict]:
    """Merge repeated ``title`` / ``moved`` / ``content`` events of one window into one, at its first position.

    A spinner in a title or a terminal printing line by line otherwise fills the stream. A merged title keeps the
    first ``old`` and the last ``title``; a merged move keeps the last rect; merged content keeps the lines that
    appeared (deduplicated, capped) and sums the counts. ``merged`` says how many events it stands for, and
    ``seq`` is the last one's, so ``since`` keeps working.
    """
    out, at = [], {}
    for e in events:
        key = (e.get("hwnd"), e["kind"])
        if e["kind"] not in MERGED or key not in at:
            if e["kind"] in MERGED:
                at[key] = len(out)
            out.append(dict(e))
            continue
        m = out[at[key]]
        m["merged"] = m.get("merged", 1) + 1
        m["seq"], m["t"], m["title"] = e["seq"], e["t"], e["title"]
        if e["kind"] == "moved":
            m["rect"] = e["rect"]
        elif e["kind"] == "content":
            seen = set(m["added"])
            size = sum(len(x) for x in m["added"])
            for ln in e["added"]:
                if ln not in seen and len(m["added"]) < MAX_LINES and size + len(ln) <= MAX_CHARS:
                    m["added"].append(ln)
                    seen.add(ln)
                    size += len(ln)
            m["added_total"] += e["added_total"]
            m["removed"] += e["removed"]
    return out


def _lines(w: Window) -> list[str]:
    return [ln for b in w.text for ln in b.text.split("\n") if ln.strip()]


def _who(w: Window) -> dict:
    return dict(hwnd=w.hwnd, title=w.title, process=w.process)


def diff(old: Snapshot | None, new: Snapshot) -> list[dict]:
    if old is None:
        return []
    before = {w.hwnd: w for w in old.windows}
    after = {w.hwnd: w for w in new.windows}
    events = []
    for h, w in after.items():
        if h not in before:
            events.append(dict(kind="opened", **_who(w)))
    for h, w in before.items():
        if h not in after:
            events.append(dict(kind="closed", **_who(w)))
    of, nf = old.focused(), new.focused()
    if nf is not None and (of is None or of.hwnd != nf.hwnd):
        events.append(dict(kind="focus", **_who(nf)))
    for h, w in after.items():
        p = before.get(h)
        if p is None:
            continue
        if p.title != w.title:
            events.append(dict(kind="title", old=p.title, **_who(w)))
        if p.minimized != w.minimized:
            events.append(dict(kind="minimized" if w.minimized else "restored", **_who(w)))
        elif p.rect != w.rect:
            r = w.rect
            events.append(dict(kind="moved", rect=[r.left, r.top, r.right, r.bottom], **_who(w)))
        if p.text and w.text:
            old_lines, new_lines = _lines(p), _lines(w)
            seen = set(old_lines)
            added = [ln for ln in new_lines if ln not in seen]
            now = set(new_lines)
            removed = sum(1 for ln in old_lines if ln not in now)
            if added or removed:
                shown, size = [], 0
                for ln in added[:MAX_LINES]:
                    if size + len(ln) > MAX_CHARS:
                        break
                    shown.append(ln)
                    size += len(ln)
                events.append(dict(kind="content", added=shown, added_total=len(added), removed=removed,
                                   **_who(w)))
    return events
