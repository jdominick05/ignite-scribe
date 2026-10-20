# Copyright (C) 2026 The ignite-scribe contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Wait for something on the screen instead of polling it: the ``wait_for`` tool.

An agent that would otherwise take screenshot after screenshot asks once and is woken by the watcher's events
(``Watcher.wait``). Conditions, on the redacted state and its change events only:
- ``text``: appears, in a window title or a text line. It counts when it is already on screen, so a build that
  finished before the call is not missed, and when it appeared in an event after ``since``, so text that came and
  went between two calls is not missed either;
- ``gone``: is no longer on screen (a dialog closed, a spinner stopped);
- neither: any change.
``match`` (title or process substring) and ``kinds`` (event kinds) narrow which windows and events count.
``quiet_s`` then waits until the counted windows have been still that long. This is the action check: note
``seq``, act (click, type), then ``wait_for(since=seq, quiet_s=0.5)`` returns what the action changed, or
``timeout`` if it changed nothing.
Text is matched case-insensitively. Only windows the watcher reads (focused first, up to its window count)
have text lines; titles cover every window.
"""
from __future__ import annotations

import time

from .changes import compact
from .injection import flag_lines

MAX_EVENTS = 60


def _fits(title: str, process: str, match: str | None) -> bool:
    return not match or match.lower() in (title or "").lower() or match.lower() in (process or "").lower()


def _event_hit(e: dict, needle: str) -> str | None:
    for s in [e.get("title") or ""] + list(e.get("added") or []):
        if needle in s.lower():
            return s
    return None


def _state_hit(snap, needle: str, match: str | None):
    """-> (window, matching line) of the first counted window that shows ``needle``, else None."""
    if snap is None:
        return None
    for w in snap.windows:
        if w.redacted or not _fits(w.title, w.process, match):
            continue
        for s in [w.title or ""] + [ln for b in w.text for ln in b.text.split("\n")]:
            if needle in s.lower():
                return w, s
    return None


def wait_for(watcher, text: str | None = None, gone: str | None = None, match: str | None = None,
             kinds: list[str] | None = None, quiet_s: float = 0.0, since: int | None = None,
             timeout_s: float = 30.0, events: bool | None = None, clock=time.monotonic) -> dict:
    """``events``: list the counted events in the reply. Default: only when waiting for a change (no
    ``text`` / ``gone``), since a text wait needs the hit, not every line printed before it."""
    t0 = clock()
    deadline = t0 + max(0.0, timeout_s)
    base = watcher.changes(10 ** 12)["seq"] if since is None else int(since)
    cursor, seen, matched, hit, last_change = base, [], None, None, t0
    needle = (text or gone or "").lower() or None
    while True:
        now = clock()
        got = watcher.changes(cursor, limit=1000, merge=False)
        cursor = got["next_since"]
        mine = [e for e in got["events"] if _fits(e.get("title"), e.get("process"), match)
                and (not kinds or e["kind"] in kinds)]
        if mine:
            seen += mine
            last_change = now
        if matched is None:
            snap = watcher.snapshot()[0]
            if text:
                ev = next(((e, h) for e in mine for h in [_event_hit(e, needle)] if h), None)
                st = _state_hit(snap, needle, match)
                if ev or st:
                    matched = now
                    hit = (dict(window=dict(hwnd=ev[0].get("hwnd"), title=ev[0].get("title"),
                                            process=ev[0].get("process")), line=ev[1], via=ev[0]["kind"])
                           if ev else dict(window=dict(hwnd=st[0].hwnd, title=st[0].title, process=st[0].process),
                                           line=st[1], via="on screen"))
            elif gone:
                if _state_hit(snap, needle, match) is None:
                    matched = now
            elif seen:
                matched = now
        if matched is not None and (quiet_s <= 0 or now - max(matched, last_change) >= quiet_s):
            result = "matched"
            break
        if now >= deadline:
            result = "timeout"
            break
        step = deadline - now
        if matched is not None:
            step = min(step, quiet_s - (now - max(matched, last_change)))
        watcher.wait(cursor, min(max(step, 0.01), 0.5))   # wake on an event, and re-check at least every 0.5 s
    if hit is not None:
        fl = flag_lines([hit["line"]], hit["window"].get("process") or "")
        if fl:
            hit["flags"] = fl
    evs = compact(seen)
    out = dict(result=result, waited_s=round(clock() - t0, 2), since=base, next_since=cursor, hit=hit,
               events_total=len(evs))
    if events if events is not None else not (text or gone):
        out["events"] = evs[-MAX_EVENTS:]
    return out
