# Copyright (C) 2026 The ignite-scribe contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The second source: what is inside a window, from Windows UI Automation (the ``uia`` extra, comtypes).

``walk(hwnd)`` returns the visible text of one window's control tree: element names, the values of value-bearing
controls (edits), and the on-screen text of controls with the Text pattern (terminals, documents, rich edits),
with their control type and bounds. It is bounded so a big or hung app cannot stall
the caller:
- one cached cross-process call per node (``FindAllBuildCache`` over the control view), with a node cap and a
  wall-clock deadline per window;
- UI Automation's own connection / transaction timeouts, so a hung provider fails instead of blocking;
- scroll bars, thumbs, title bars and separators are not descended into, and off-screen elements are skipped.

**Password fields are never read.** ``IsPassword`` is in the cached properties; a password element is reported as
``[password field]``, and neither its Value nor its Text pattern is ever requested. Value and text are fetched
only for elements that are not passwords and expose the pattern.

**Text pattern:** only the *visible* ranges are read (what is on screen, not a whole scrollback or document).
Lines are right-stripped and blank runs collapsed, then capped at ``MAX_TEXT``. Text that only repeats the
element's name or value is dropped. An element whose Text pattern gave multi-line text is not descended into:
its children (a document's links, a terminal's parts) would repeat that text.

**Detail** (``walk(detail=True)``, for ``find_element`` only): a second cache request adds whether each element is
enabled, its accelerator and access keys, AutomationId, keyboard focus and which actions it supports (invoke, toggle, select,
expand), all in the same cached call; a clickable control with no name is kept too. A password element gets none
of it: its early return comes first. The watcher and snapshots never ask for detail, so their content and events
are unchanged.

The caller decides which windows to walk; ``collect`` never walks a window the redaction policy excludes.
"""
from __future__ import annotations

import contextlib
import sys
import threading
import time

from .snapshot import Rect, TextBlock

_UIA = None                 # the generated type library module, shared
_tls = threading.local()    # the automation object and cache request: one per thread (COM apartment)

# property ids (UIAutomationClient.h)
NAME, CONTROL_TYPE, BOUNDS, IS_OFFSCREEN, IS_PASSWORD = 30005, 30003, 30001, 30022, 30019
IS_VALUE_AVAILABLE, VALUE, IS_TEXT_AVAILABLE = 30043, 30045, 30040
TEXT_PATTERN = 10014
TREESCOPE_CHILDREN = 2
# detail only (UIA_*PropertyId in UIAutomationCore's type library)
IS_ENABLED, ACCELERATOR_KEY, ACCESS_KEY, AUTOMATION_ID, HAS_FOCUS = 30010, 30006, 30007, 30011, 30008
ACTIONS = {"invoke": 30031, "toggle": 30041, "select": 30036, "expand": 30028}
DETAIL_PROPS = (IS_ENABLED, ACCELERATOR_KEY, ACCESS_KEY, AUTOMATION_ID, HAS_FOCUS, *ACTIONS.values())

CONTROL_TYPES = {50000 + i: n for i, n in enumerate(
    "Button Calendar CheckBox ComboBox Edit Hyperlink Image ListItem List Menu MenuBar MenuItem ProgressBar "
    "RadioButton ScrollBar Slider Spinner StatusBar Tab TabItem Text ToolBar ToolTip Tree TreeItem Custom Group "
    "Thumb DataGrid DataItem Document SplitButton Window Pane Header HeaderItem Table TitleBar Separator "
    "SemanticZoom AppBar".split())}
NO_DESCEND = {"ScrollBar", "Thumb", "TitleBar", "Separator"}
NO_EMIT = {"TitleBar"}     # its name is the window title, already in the snapshot
CLICKABLE = {"Button", "CheckBox", "ComboBox", "Edit", "Hyperlink", "ListItem", "MenuItem", "RadioButton",
             "Slider", "Spinner", "SplitButton", "TabItem", "TreeItem", "DataItem", "HeaderItem"}
MAX_VALUE = 4000          # characters kept from one element's value
MAX_TEXT = 8000           # characters kept from one element's visible text (after stripping)
READ_TEXT = 60000         # characters requested from the visible ranges (terminals pad every line with spaces)


def available() -> bool:
    try:
        import comtypes.client  # noqa: F401
        return True
    except ImportError:
        return False


def _init():
    """-> (automation, cache request) for the calling thread, creating them (and its COM apartment) once."""
    global _UIA
    if getattr(_tls, "auto", None) is not None:
        return _tls.auto, _tls.cache
    import comtypes
    import comtypes.client
    if _UIA is None:
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
    base = (NAME, CONTROL_TYPE, BOUNDS, IS_OFFSCREEN, IS_PASSWORD, IS_VALUE_AVAILABLE, IS_TEXT_AVAILABLE)
    cache, detail = auto.CreateCacheRequest(), auto.CreateCacheRequest()
    for pid in base:
        cache.AddProperty(pid)
    for pid in base + DETAIL_PROPS:
        detail.AddProperty(pid)
    _tls.auto, _tls.cache, _tls.detail = auto, cache, detail
    return auto, cache


def _rect(el) -> Rect | None:
    r = el.CachedBoundingRectangle
    if r.right <= r.left or r.bottom <= r.top:
        return None
    return Rect(int(r.left), int(r.top), int(r.right), int(r.bottom))


def clean(text: str) -> str:
    """Right-strip each line, collapse runs of blank lines, trim, cap at MAX_TEXT."""
    out, blank = [], False
    for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        line = line.rstrip()
        if not line:
            if not blank and out:
                out.append("")
            blank = True
            continue
        out.append(line)
        blank = False
    text = "\n".join(out).strip()
    return text if len(text) <= MAX_TEXT else text[:MAX_TEXT] + "\n[... truncated]"


def _visible_text(el) -> str:
    try:
        tp = el.GetCurrentPattern(TEXT_PATTERN).QueryInterface(_UIA.IUIAutomationTextPattern)
        ranges = tp.GetVisibleRanges()
        return clean("\n".join(ranges.GetElement(i).GetText(READ_TEXT) for i in range(ranges.Length)))
    except Exception:  # noqa: BLE001 - provider refused or timed out
        return ""


def _detail(el, name: str) -> dict:
    """The cached detail properties of a non-password element (see the module docstring)."""
    def s(pid):
        v = el.GetCachedPropertyValue(pid)
        return v.strip() if isinstance(v, str) and v.strip() else None
    return dict(name=name or None, enabled=bool(el.GetCachedPropertyValue(IS_ENABLED)),
                accelerator=s(ACCELERATOR_KEY), access_key=s(ACCESS_KEY), automation_id=s(AUTOMATION_ID),
                focused=bool(el.GetCachedPropertyValue(HAS_FOCUS)),
                actions=[a for a, pid in ACTIONS.items() if el.GetCachedPropertyValue(pid)])


def _block(el, detail: bool = False) -> tuple[TextBlock | None, str, bool]:
    """-> (text block or None, role, whether its Text pattern already covered its subtree)."""
    role = CONTROL_TYPES.get(el.CachedControlType, "Unknown")
    if el.CachedIsPassword:                  # first, before any detail: nothing more is asked of a password field
        return TextBlock("[password field]", _rect(el), "uia", role), role, False
    parts = []
    name = (el.CachedName or "").strip()
    if name:
        parts.append(name)
    if el.GetCachedPropertyValue(IS_VALUE_AVAILABLE):
        try:
            v = el.GetCurrentPropertyValue(VALUE)
        except Exception:  # noqa: BLE001 - provider refused or timed out: keep the name
            v = None
        v = clean(v)[:MAX_VALUE] if isinstance(v, str) else ""   # multiline edits return \r\n and padding
        if v and v != name:
            parts.append(v)
    covered = False
    if el.GetCachedPropertyValue(IS_TEXT_AVAILABLE):
        t = _visible_text(el)
        if t and not any(t == p or t in p for p in parts):
            multiline = "\n" in t
            if multiline:                    # a value the text already contains is not repeated
                parts = [p for p in parts if p not in t]
            parts.append(t)
            covered = multiline
    d = _detail(el, name) if detail else None
    if not parts and not (d and role in CLICKABLE and d["automation_id"]):   # an icon button: its id names it
        return None, role, False
    sep = "\n" if any("\n" in p for p in parts) else ": "
    return TextBlock(sep.join(parts), _rect(el), "uia", role, d), role, covered


def walk(hwnd: int, node_cap: int = 600, seconds: float = 1.0,
         detail: bool = False) -> tuple[list[TextBlock], list[str]]:
    """-> (text blocks in tree order, notes such as 'node cap' or 'timeout'). Never raises for app errors.
    ``detail``: blocks carry ``TextBlock.detail`` (``find_element``), and they are not deduplicated, since two
    buttons with one label are two places to click."""
    auto, cache = _init()
    if detail:
        cache = _tls.detail
    deadline = time.monotonic() + seconds
    notes: list[str] = []
    try:
        root = auto.ElementFromHandleBuildCache(hwnd, cache)
    except Exception as e:  # noqa: BLE001
        return [], [f"uia unavailable: {type(e).__name__}"]
    blocks, stack, seen = [], [root], 0
    cond = auto.ControlViewCondition
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
            b, role, covered = _block(el, detail)
            if b is not None and seen > 1 and role not in NO_EMIT:   # the window itself is its title
                blocks.append(b)
            if role in NO_DESCEND or el.CachedIsPassword or covered:
                continue
            kids = el.FindAllBuildCache(TREESCOPE_CHILDREN, cond, cache)
            stack.extend(kids.GetElement(i) for i in reversed(range(kids.Length)))   # keep tree order
        except Exception as e:  # noqa: BLE001 - element vanished, provider hung or refused
            if f"error: {type(e).__name__}" not in notes:
                notes.append(f"error: {type(e).__name__}")
    return (blocks if detail else _dedupe(blocks)), notes


PROCESS_ID, NATIVE_HWND = 30002, 30020
CLIMB = 8                 # ancestors searched for a Text pattern above the focused element
EM_GETSEL, ES_PASSWORD, GWL_STYLE, SMTO_ABORTIFHUNG = 0xB0, 0x20, -16, 0x2


def selection(pid: int) -> dict:
    """The text selected in the focused element, which must belong to process ``pid`` (the focused window's: the
    caller checked the policy for that window first). -> dict with ``text`` (may be empty: a caret only), ``role``,
    ``name``, ``rects`` (screen pixels, one per selected line); or ``password`` / ``none`` / ``error``. A password
    element, or an ancestor that is one, is never read. Never raises for app errors.

    The focused element's Text pattern is used, else that of the nearest ancestor that has one. A classic Win32
    Edit or RichEdit has none: its selection offsets come from ``EM_GETSEL`` (with a timeout) and its text from the
    UIA Value, which is read for any non-password element anyway; such a selection has no line rectangles."""
    try:
        auto, _ = _init()
        if getattr(_tls, "sel", None) is None:
            c = auto.CreateCacheRequest()
            for p in (NAME, CONTROL_TYPE, IS_PASSWORD, IS_TEXT_AVAILABLE, IS_VALUE_AVAILABLE, PROCESS_ID,
                      NATIVE_HWND):
                c.AddProperty(p)
            _tls.sel = c
        cache = _tls.sel
        el = auto.GetFocusedElementBuildCache(cache)
        if el.GetCachedPropertyValue(PROCESS_ID) != pid:
            return dict(error="the keyboard focus is not in the focused window's process")
        walker = auto.ControlViewWalker
        for _ in range(CLIMB + 1):
            if el is None:
                break
            if el.CachedIsPassword:
                return dict(password=True)
            if el.GetCachedPropertyValue(IS_TEXT_AVAILABLE):
                break
            if _classic_edit(el):
                return _edit_selected(el)
            el = walker.GetParentElementBuildCache(el, cache)
            if el is not None and el.GetCachedPropertyValue(PROCESS_ID) != pid:   # climbed out of the app
                el = None
        else:
            el = None
        if el is None or not el.GetCachedPropertyValue(IS_TEXT_AVAILABLE):
            return dict(none="the focused control exposes no text selection")
        return _selected(el)
    except Exception as e:  # noqa: BLE001 - provider refused, timed out or vanished
        return dict(error=f"UI Automation: {type(e).__name__}")


def _classic_edit(el) -> int | None:
    """The window handle of a classic Win32 Edit / RichEdit element (not a password one), else None."""
    import ctypes
    h = el.GetCachedPropertyValue(NATIVE_HWND)
    if not h:
        return None
    buf = ctypes.create_unicode_buffer(64)
    ctypes.windll.user32.GetClassNameW(ctypes.c_void_p(h), buf, 64)
    cls = buf.value.lower()
    if cls != "edit" and not cls.startswith("richedit"):
        return None
    if ctypes.windll.user32.GetWindowLongW(ctypes.c_void_p(h), GWL_STYLE) & ES_PASSWORD:
        return None                          # IsPassword already said no; the style is checked as well
    return h


def _edit_selected(el) -> dict:
    """A classic Edit's selection: offsets from EM_GETSEL, text from its UIA Value (UTF-16 offsets)."""
    import ctypes
    from ctypes import wintypes
    h = _classic_edit(el)
    if h is None or el.CachedIsPassword or not el.GetCachedPropertyValue(IS_VALUE_AVAILABLE):
        return dict(none="the focused control exposes no text selection")
    start, end, res = wintypes.DWORD(), wintypes.DWORD(), ctypes.c_size_t()
    ok = ctypes.windll.user32.SendMessageTimeoutW(ctypes.c_void_p(h), EM_GETSEL, ctypes.byref(start),
                                                  ctypes.byref(end), SMTO_ABORTIFHUNG, 500, ctypes.byref(res))
    if not ok:
        return dict(error="the edit control did not answer")
    v = el.GetCurrentPropertyValue(VALUE)
    v = v if isinstance(v, str) else ""
    u = v.encode("utf-16-le")
    a, b = sorted((start.value, end.value))
    text = u[2 * a:2 * b].decode("utf-16-le", "replace").replace("\r\n", "\n").replace("\r", "\n")
    return dict(text=text[:MAX_TEXT], truncated=len(text) > MAX_TEXT, rects=[],
                role=CONTROL_TYPES.get(el.CachedControlType, "Unknown"), name=(el.CachedName or "").strip())


def _selected(el) -> dict:
    """The Text-pattern selection of a non-password element (the caller checked)."""
    tp = el.GetCurrentPattern(TEXT_PATTERN).QueryInterface(_UIA.IUIAutomationTextPattern)
    ranges = tp.GetSelection()
    texts, rects = [], []
    for i in range(ranges.Length if ranges else 0):
        r = ranges.GetElement(i)
        texts.append(r.GetText(MAX_TEXT + 1))
        b = list(r.GetBoundingRectangles() or [])
        rects += [[int(b[k]), int(b[k + 1]), int(b[k] + b[k + 2]), int(b[k + 1] + b[k + 3])]
                  for k in range(0, len(b) - 3, 4)]
    text = "\n".join(texts).replace("\r\n", "\n").replace("\r", "\n")
    return dict(text=text[:MAX_TEXT], truncated=len(text) > MAX_TEXT, rects=rects[:200],
                role=CONTROL_TYPES.get(el.CachedControlType, "Unknown"), name=(el.CachedName or "").strip())


def _dedupe(blocks: list[TextBlock]) -> list[TextBlock]:
    """Drop exact repeats of (text, role): lists and toolbars often echo the same label several times."""
    out, seen = [], set()
    for b in blocks:
        k = (b.text, b.role)
        if k not in seen:
            seen.add(k)
            out.append(b)
    return out
