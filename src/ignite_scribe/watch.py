# Copyright (C) 2026 The ignite-scribe contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Change-driven capture: keep the redacted screen state current, re-reading only the windows that changed.

One background thread owns everything (Win32 hooks need their thread's message loop, and UI Automation objects
belong to their thread's COM apartment). Two change signals mark windows dirty:
- **WinEvents** (``SetWinEventHook``, out of context): focus, windows created / destroyed / shown / hidden /
  moved / renamed / minimized, and name or value changes inside a window;
- **Desktop Duplication** (``dirty.DirtyRects``): the screen rectangles that changed, mapped to the topmost
  windows under them. This catches what raises no WinEvent (a terminal printing). Only the rectangles are read,
  never the pixels.

At most every ``interval`` seconds, if anything is dirty, the window list is re-read (cheap) and only the dirty
windows among the ones worth reading (focused first, then by z, up to ``max_windows``, never an excluded one) are
walked again; the rest keep their last content. The result is redacted, compared with the previous state
(``changes.diff``) and the events get increasing sequence numbers in a bounded in-memory ring. Nothing is
written to disk, unless a ``recorder`` (``history.History``, opt-in) is given: it then receives each refresh's
redacted events and redacted snapshot, after redaction, and writes them on its own thread.

Without Desktop Duplication (secure desktop, a full-screen exclusive app, too many duplications), the focused
window is re-walked every ``fallback_poll`` seconds instead, and ``status()`` says so.
"""
from __future__ import annotations

import collections
import ctypes
import threading
import time
from ctypes import wintypes

from . import uia, windows
from .changes import compact, diff
from .redact import Policy
from .snapshot import Snapshot

user32 = ctypes.WinDLL("user32", use_last_error=True)
WINEVENTPROC = ctypes.WINFUNCTYPE(None, wintypes.HANDLE, wintypes.DWORD, wintypes.HWND, wintypes.LONG,
                                  wintypes.LONG, wintypes.DWORD, wintypes.DWORD)
user32.SetWinEventHook.restype = wintypes.HANDLE
user32.SetWinEventHook.argtypes = [wintypes.DWORD, wintypes.DWORD, wintypes.HMODULE, WINEVENTPROC, wintypes.DWORD,
                                   wintypes.DWORD, wintypes.DWORD]
user32.UnhookWinEvent.argtypes = [wintypes.HANDLE]
user32.GetAncestor.restype = wintypes.HWND
user32.GetAncestor.argtypes = [wintypes.HWND, wintypes.UINT]
user32.MsgWaitForMultipleObjects.argtypes = [wintypes.DWORD, ctypes.c_void_p, wintypes.BOOL, wintypes.DWORD,
                                             wintypes.DWORD]

WINEVENT_OUTOFCONTEXT, WINEVENT_SKIPOWNPROCESS = 0x0, 0x2
EVENT_SYSTEM_FOREGROUND, EVENT_SYSTEM_MINIMIZESTART, EVENT_SYSTEM_MINIMIZEEND = 0x0003, 0x0016, 0x0017
EVENT_OBJECT_CREATE, EVENT_OBJECT_DESTROY, EVENT_OBJECT_SHOW, EVENT_OBJECT_HIDE = 0x8000, 0x8001, 0x8002, 0x8003
EVENT_OBJECT_LOCATIONCHANGE, EVENT_OBJECT_NAMECHANGE, EVENT_OBJECT_VALUECHANGE = 0x800B, 0x800C, 0x800E
OBJID_WINDOW, GA_ROOT, QS_ALLINPUT, PM_REMOVE = 0, 2, 0x04FF, 0x0001
STRUCTURE = {EVENT_SYSTEM_FOREGROUND, EVENT_SYSTEM_MINIMIZESTART, EVENT_SYSTEM_MINIMIZEEND, EVENT_OBJECT_CREATE,
             EVENT_OBJECT_DESTROY, EVENT_OBJECT_SHOW, EVENT_OBJECT_HIDE, EVENT_OBJECT_LOCATIONCHANGE}


class Watcher:
    def __init__(self, policy: Policy, max_windows: int = 6, interval: float = 0.25, history: int = 2000,
                 budget: float = 1.5, fallback_poll: float = 2.0, duplication: bool = True, recorder=None):
        self.policy, self.max_windows, self.interval, self.budget = policy, max_windows, interval, budget
        self.recorder = recorder
        self.pixel_dirty: dict[int, tuple] = {}    # hwnd -> (left, top, right, bottom) changed since taken (always_on)
        self.extra_text: dict[int, list] = {}      # hwnd -> TextBlocks from always-on OCR, merged before redaction
        self._extra_dirty = False
        self.fallback_poll, self.duplication = fallback_poll, duplication
        self.events: collections.deque = collections.deque(maxlen=history)
        self.seq = 0
        self.state: Snapshot | None = None
        self.counts: dict = {}
        self.as_of = 0.0
        self.stats = dict(refreshes=0, walks=0, refresh_ms_total=0.0, winevents=0, dirty_rects=0)
        self.dirty_source = "starting"
        self._lock = threading.Lock()
        self._changed = threading.Condition(self._lock)   # notified when events are appended (wait_for)
        self._ready = threading.Event()
        self._stop = threading.Event()
        self._dirty: set[int] = set()
        self._structure = True
        self._content: dict[int, tuple] = {}
        self._thread = threading.Thread(target=self._run, name="ignite-scribe-watch", daemon=True)
        self._cpu0 = self._wall0 = 0.0

    # ---- public, any thread -------------------------------------------------------------------------------
    def start(self, wait: float = 10.0) -> "Watcher":
        self._thread.start()
        self._ready.wait(wait)
        return self

    def stop(self, wait: float = 5.0):
        self._stop.set()
        self._thread.join(wait)

    @property
    def running(self) -> bool:
        return self._thread.is_alive()

    def snapshot(self) -> tuple[Snapshot | None, dict, float]:
        """-> (current redacted snapshot, redaction counts, its time)."""
        with self._lock:
            return self.state, dict(self.counts), self.as_of

    def changes(self, since: int = 0, limit: int = 200, merge: bool = True) -> dict:
        """Up to ``limit`` events after ``since`` (merged with ``compact`` unless ``merge`` is False).
        ``truncated``: the ring no longer holds some of them. Ask again with ``since`` = ``next_since``."""
        with self._lock:
            evs = [dict(e) for e in self.events if e["seq"] > since]
            oldest = self.events[0]["seq"] if self.events else self.seq + 1
            seq = self.seq
        truncated = since + 1 < oldest and since < seq
        more, evs = len(evs) > limit, evs[:limit]           # limit the raw events, so next_since skips nothing
        next_since = evs[-1]["seq"] if evs else max(since, seq)
        return dict(seq=seq, since=since, next_since=next_since, truncated=truncated, more=more,
                    events=compact(evs) if merge else evs)

    def wait(self, since: int, timeout: float) -> int:
        """Block until an event after ``since`` exists, or ``timeout`` seconds pass. -> the current seq."""
        with self._changed:
            self._changed.wait_for(lambda: self.seq > since, max(0.0, timeout))
            return self.seq

    def _append(self, new: list[dict], t: float):
        """Number and store events, then wake waiters. The caller holds the lock."""
        for e in new:
            self.seq += 1
            e["seq"], e["t"] = self.seq, t
            self.events.append(e)
        if new:
            self._changed.notify_all()

    def take_pixel_dirty(self) -> dict[int, tuple]:
        """The changed rectangle per window since the last call (from Desktop Duplication; rectangles only)."""
        with self._lock:
            d, self.pixel_dirty = self.pixel_dirty, {}
        return d

    def reapply(self):
        """Re-apply the policy soon, reusing every window's content (secure mode's scores arrived)."""
        self._extra_dirty = True

    def set_extra_text(self, hwnd: int, blocks: list | None):
        """Always-on OCR's text for a window (None removes it); the window is re-read on the next refresh."""
        with self._lock:
            if blocks:
                self.extra_text[hwnd] = list(blocks)
            else:
                self.extra_text.pop(hwnd, None)
        self._extra_dirty = True                   # refresh soon; the window's UIA content is reused, not re-walked

    def status(self) -> dict:
        with self._lock:
            s = dict(self.stats)
            n = s["refreshes"]
            s["refresh_ms_mean"] = round(s.pop("refresh_ms_total") / n, 1) if n else None
            wall = time.perf_counter() - self._wall0
            return dict(running=self.running, dirty_source=self.dirty_source, seq=self.seq,
                        as_of=self.as_of, interval_s=self.interval, max_windows=self.max_windows, **s,
                        thread_cpu_percent=round(100 * self._thread_cpu() / wall, 2) if wall > 0 else None)

    # ---- the watcher thread ---------------------------------------------------------------------------------
    def _thread_cpu(self) -> float:
        return getattr(self, "_cpu_now", 0.0) - self._cpu0

    def _on_event(self, hook, event, hwnd, id_object, id_child, thread, ms):
        self.stats["winevents"] += 1
        if not hwnd:
            return
        if event in STRUCTURE:
            if id_object == OBJID_WINDOW and (event != EVENT_OBJECT_LOCATIONCHANGE or
                                              user32.GetAncestor(hwnd, GA_ROOT) == hwnd):
                self._structure = True
            return
        root = user32.GetAncestor(hwnd, GA_ROOT)
        if root:
            self._dirty.add(int(root))
            if root == hwnd and event == EVENT_OBJECT_NAMECHANGE:
                self._structure = True          # a top-level title changed

    def _hooks(self):
        self._proc = WINEVENTPROC(self._on_event)
        flags = WINEVENT_OUTOFCONTEXT | WINEVENT_SKIPOWNPROCESS
        ranges = [(EVENT_SYSTEM_FOREGROUND, EVENT_SYSTEM_FOREGROUND),
                  (EVENT_SYSTEM_MINIMIZESTART, EVENT_SYSTEM_MINIMIZEEND),
                  (EVENT_OBJECT_CREATE, EVENT_OBJECT_HIDE),
                  (EVENT_OBJECT_LOCATIONCHANGE, EVENT_OBJECT_NAMECHANGE),
                  (EVENT_OBJECT_VALUECHANGE, EVENT_OBJECT_VALUECHANGE)]
        return [h for h in (user32.SetWinEventHook(a, b, None, self._proc, 0, 0, flags) for a, b in ranges) if h]

    def _map_rects(self, rects, snap: Snapshot):
        """Mark the windows a changed rectangle shows: topmost first, down to one that contains it."""
        self.stats["dirty_rects"] += len(rects)
        shown = [w for w in snap.windows if not w.minimized]
        for r in rects:
            for w in shown:
                a = w.rect
                if a.left < r.right and r.left < a.right and a.top < r.bottom and r.top < a.bottom:
                    self._dirty.add(w.hwnd)
                    rr = (max(a.left, r.left), max(a.top, r.top), min(a.right, r.right), min(a.bottom, r.bottom))
                    with self._lock:
                        old = self.pixel_dirty.get(w.hwnd)
                        self.pixel_dirty[w.hwnd] = rr if old is None else (
                            min(old[0], rr[0]), min(old[1], rr[1]), max(old[2], rr[2]), max(old[3], rr[3]))
                    if a.left <= r.left and a.top <= r.top and a.right >= r.right and a.bottom >= r.bottom:
                        break

    def _refresh(self, full: bool = False):
        t0 = time.perf_counter()
        self._extra_dirty = False
        snap = windows.snapshot()
        dirty, self._dirty = self._dirty, set()
        self._structure = False
        if uia.available():
            snap.sources.append("uia")
            end = time.monotonic() + self.budget
            order = [w for w in sorted(snap.windows, key=lambda w: (not w.focused, w.z))
                     if not w.minimized and not self.policy.excluded(w)][:self.max_windows]
            keep = {}
            for w in order:
                cached = self._content.get(w.hwnd)
                if full or cached is None or w.hwnd in dirty:
                    if time.monotonic() > end:     # out of budget: keep the stale copy, read it next time
                        self._dirty.add(w.hwnd)
                        if cached is None:
                            continue
                        w.text, w.notes = cached[0], cached[1] + ["stale: refresh budget spent"]
                    else:
                        text, notes = uia.walk(w.hwnd, 600, min(1.0, max(0.05, end - time.monotonic())))
                        if not text and not notes:
                            notes = ["no readable content (an elevated app, or one that exposes none)"]
                        self.stats["walks"] += 1
                        cached = (text, notes)
                        w.text, w.notes = list(text), list(notes)
                else:
                    w.text, w.notes = list(cached[0]), list(cached[1])
                keep[w.hwnd] = cached
            self._content = keep
        snap.sources.append(f"watch ({self.dirty_source})")
        with self._lock:
            extra = {h: list(b) for h, b in self.extra_text.items()}
        if extra:                                  # always-on OCR text: redacted below like UIA text
            snap.sources.append("ocr (always-on)")
            for w in snap.windows:
                if w.hwnd in extra and not self.policy.excluded(w):
                    w.text = list(w.text) + extra[w.hwnd]
        red, counts = self.policy.apply(snap)
        with self._lock:
            new = diff(self.state, red)
            self._append(new, red.taken_at)
            self.state, self.counts, self.as_of = red, counts, red.taken_at
            self.stats["refreshes"] += 1
            self.stats["refresh_ms_total"] += (time.perf_counter() - t0) * 1000
        if self.recorder is not None:              # redacted only; the recorder copies nothing raw
            self.recorder.record([dict(e) for e in new], red)

    def _run(self):
        self._wall0 = time.perf_counter()
        self._cpu0 = self._cpu_now = time.thread_time()
        hooks = self._hooks()
        dup = None
        if self.duplication:
            try:
                from .dirty import DirtyRects
                dup = DirtyRects()
            except Exception as e:  # noqa: BLE001
                self.dirty_source = f"winevents only (duplication: {type(e).__name__})"
        try:
            self._refresh(full=True)
            self._ready.set()
            last, last_poll = time.monotonic(), time.monotonic()
            msg = wintypes.MSG()
            while not self._stop.is_set():
                user32.MsgWaitForMultipleObjects(0, None, False, 50, QS_ALLINPUT)
                while user32.PeekMessageW(ctypes.byref(msg), None, 0, 0, PM_REMOVE):
                    user32.TranslateMessage(ctypes.byref(msg))
                    user32.DispatchMessageW(ctypes.byref(msg))
                if dup is not None:
                    rects = dup.poll()
                    if dup.available:
                        self.dirty_source = "duplication + winevents"
                        if rects and self.state is not None:
                            self._map_rects(rects, self.state)
                    else:
                        self.dirty_source = f"winevents + {self.fallback_poll:g} s poll ({dup.error})"
                elif self.dirty_source == "starting" or not self.duplication:
                    self.dirty_source = f"winevents + {self.fallback_poll:g} s poll"
                now = time.monotonic()
                if (dup is None or not dup.available) and now - last_poll >= self.fallback_poll:
                    f = self.state.focused() if self.state else None
                    if f is not None:
                        self._dirty.add(f.hwnd)
                    last_poll = now
                if (self._dirty or self._structure or self._extra_dirty) and now - last >= self.interval:
                    self._refresh()
                    last = now
                self._cpu_now = time.thread_time()
        finally:
            self._ready.set()
            for h in hooks:
                user32.UnhookWinEvent(h)
            if dup is not None:
                dup.close()
