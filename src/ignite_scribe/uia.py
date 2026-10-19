# Copyright (C) 2026 The ignite-scribe contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The second source: what is inside a window, from Windows UI Automation (the ``uia`` extra, comtypes).

``walk(hwnd)`` returns the visible text of one window's control tree: element names, the values of value-bearing
controls (edits, documents), with their control type and bounds. It is bounded so a big or hung app cannot stall
the caller:
- one cached cross-process call per node (``FindAllBuildCache`` over the control view), with a node cap and a
  wall-clock deadline per window;
- UI Automation's own connection / transaction timeouts, so a hung provider fails instead of blocking;
- scroll bars, thumbs, title bars and separators are not descended into, and off-screen elements are skipped.

**Password fields are never read.** ``IsPassword`` is in the cached properties; a password element is reported as
``[password field]`` and its value is never requested. Values are fetched only for elements that are not
passwords and expose the Value pattern.

The caller decides which windows to walk; ``collect`` never walks a window the redaction policy excludes.
"""
from __future__ import annotations

import contextlib
import sys
import time

from .snapshot import Rect, TextBlock

_auto = None
_cache = None
_UIA = None

# property ids (UIAutomationClient.h)
NAME, CONTROL_TYPE, BOUNDS, IS_OFFSCREEN, IS_PASSWORD = 30005, 30003, 30001, 30022, 30019
IS_VALUE_AVAILABLE, VALUE = 30043, 30045
TREESCOPE_CHILDREN = 2

CONTROL_TYPES = {50000 + i: n for i, n in enumerate(
    "Button Calendar CheckBox ComboBox Edit Hyperlink Image ListItem List Menu MenuBar MenuItem ProgressBar "
    "RadioButton ScrollBar Slider Spinner StatusBar Tab TabItem Text ToolBar ToolTip Tree TreeItem Custom Group "
    "Thumb DataGrid DataItem Document SplitButton Window Pane Header HeaderItem Table TitleBar Separator "
    "SemanticZoom AppBar".split())}
NO_DESCEND = {"ScrollBar", "Thumb", "TitleBar", "Separator"}
NO_EMIT = {"TitleBar"}     # its name is the window title, already in the snapshot
MAX_VALUE = 4000          # characters kept from one element's value


def available() -> bool:
    try:
        import comtypes.client  # noqa: F401
        return True
    except ImportError:
        return False


def _init():
    global _auto, _cache, _UIA
    if _auto is not None:
        return
    import comtypes
    import comtypes.client
    with contextlib.redirect_stdout(sys.stderr):   # the stdio MCP channel must stay clean
        _UIA = comtypes.client.GetModule("UIAutomationCore.dll")
    comtypes.CoInitialize()
    auto = comtypes.client.CreateObject(_UIA.CUIAutomation, interface=_UIA.IUIAutomation)
    try:
        a2 = auto.QueryInterface(_UIA.IUIAutomation2)
        a2.ConnectionTimeout = 2000
        a2.TransactionTimeout = 1000
        auto = a2
    except Exception:  # noqa: BLE001 - older Windows: default timeouts
        pass
    cache = auto.CreateCacheRequest()
    for pid in (NAME, CONTROL_TYPE, BOUNDS, IS_OFFSCREEN, IS_PASSWORD, IS_VALUE_AVAILABLE):
        cache.AddProperty(pid)
    _auto, _cache = auto, cache


def _rect(el) -> Rect | None:
    r = el.CachedBoundingRectangle
    if r.right <= r.left or r.bottom <= r.top:
        return None
    return Rect(int(r.left), int(r.top), int(r.right), int(r.bottom))


def _block(el) -> tuple[TextBlock | None, str]:
    """-> (text block or None, role)."""
    role = CONTROL_TYPES.get(el.CachedControlType, "Unknown")
    if el.CachedIsPassword:
        return TextBlock("[password field]", _rect(el), "uia", role), role
    parts = []
    name = (el.CachedName or "").strip()
    if name:
        parts.append(name)
    if el.GetCachedPropertyValue(IS_VALUE_AVAILABLE):
        try:
            v = el.GetCurrentPropertyValue(VALUE)
        except Exception:  # noqa: BLE001 - provider refused or timed out: keep the name
            v = None
        if isinstance(v, str) and v.strip() and v.strip() != name:
            parts.append(v.strip()[:MAX_VALUE])
    if not parts:
        return None, role
    return TextBlock(": ".join(parts), _rect(el), "uia", role), role


def walk(hwnd: int, node_cap: int = 600, seconds: float = 1.0) -> tuple[list[TextBlock], list[str]]:
    """-> (text blocks in tree order, notes such as 'node cap' or 'timeout'). Never raises for app errors."""
    _init()
    deadline = time.monotonic() + seconds
    notes: list[str] = []
    try:
        root = _auto.ElementFromHandleBuildCache(hwnd, _cache)
    except Exception as e:  # noqa: BLE001
        return [], [f"uia unavailable: {type(e).__name__}"]
    blocks, stack, seen = [], [root], 0
    cond = _auto.ControlViewCondition
    while stack:
        if seen >= node_cap:
            notes.append(f"stopped at node cap {node_cap}")
            break
        if time.monotonic() > deadline:
            notes.append(f"stopped at {seconds:g} s")
            break
        el = stack.pop()
        seen += 1
        try:
            if el.CachedIsOffscreen and seen > 1:
                continue
            b, role = _block(el)
            if b is not None and seen > 1 and role not in NO_EMIT:   # the window itself is its title
                blocks.append(b)
            if role in NO_DESCEND or el.CachedIsPassword:
                continue
            kids = el.FindAllBuildCache(TREESCOPE_CHILDREN, cond, _cache)
            stack.extend(kids.GetElement(i) for i in reversed(range(kids.Length)))   # keep tree order
        except Exception as e:  # noqa: BLE001 - element vanished, provider hung or refused
            if f"error: {type(e).__name__}" not in notes:
                notes.append(f"error: {type(e).__name__}")
    return _dedupe(blocks), notes


def _dedupe(blocks: list[TextBlock]) -> list[TextBlock]:
    """Drop exact repeats of (text, role): lists and toolbars often echo the same label several times."""
    out, seen = [], set()
    for b in blocks:
        k = (b.text, b.role)
        if k not in seen:
            seen.add(k)
            out.append(b)
    return out
