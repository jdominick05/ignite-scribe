# Copyright (C) 2026 The ignite-scribe contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""One pass over the desktop: windows, then UI Automation content for the windows worth reading.

The policy is applied in two places on purpose:
- **before the walk**: an excluded window (process or title) is never opened with UI Automation, so its content
  never enters this process;
- **after**: ``Policy.apply`` withholds the excluded windows' titles and scrubs secret-shaped text from the rest.
``collect`` does the first; callers (the CLI, the MCP server) always do the second.
"""
from __future__ import annotations

import time

from . import uia, windows
from .redact import Policy
from .snapshot import Snapshot


def collect(policy: Policy, text: bool = True, max_windows: int = 6, node_cap: int = 600,
            seconds: float = 1.0, budget: float = 3.0) -> Snapshot:
    """The raw (not yet redacted) snapshot. ``text``: walk UI Automation for the focused window first, then the
    topmost ones, up to ``max_windows`` and ``budget`` seconds in all."""
    snap = windows.snapshot()
    if not text:
        return snap
    if not uia.available():
        snap.sources.append("uia unavailable (install the uia extra)")
        return snap
    snap.sources.append("uia")
    order = sorted(snap.windows, key=lambda w: (not w.focused, w.z))
    end = time.monotonic() + budget
    walked = 0
    for w in order:
        if walked >= max_windows or time.monotonic() > end:
            break
        if w.minimized or policy.excluded(w):
            continue
        w.text, w.notes = uia.walk(w.hwnd, node_cap, min(seconds, max(0.05, end - time.monotonic())))
        if not w.text and not w.notes:   # e.g. an elevated app: UIPI hides its tree from a normal process
            w.notes.append("no readable content (an elevated app, or one that exposes none)")
        walked += 1
    return snap
