# Copyright (C) 2026 The ignite-scribe contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Always-on OCR: read the text of windows UI Automation cannot describe, as their pixels change, on the NPU.

Opt-in (``IGNITE_SCRIBE_ALWAYS_ON=1`` or the policy's ``"always_on_ocr": true``), separate from history.
- **One owner per machine:** an exclusive lock on ``%LOCALAPPDATA%\\ignite-scribe\\always-on.lock``; other server
  processes do not run it, so the device sees one context, not one per session.
- **What is read:** the watcher reports the screen rectangles that changed (Desktop Duplication: rectangles, never
  pixels). For each window UI Automation left empty (or thin), not excluded, not minimized and not elevated
  (unless ``capture_elevated``), the changed part is captured with the same masking as ``ocr_windows``: windows
  above it, excluded windows and password fields are never copied. Pixels stay in memory.
- **Where it goes:** the lines become ``TextBlock(source="ocr")`` in the watcher's state, so ``Policy.apply``
  scrubs them like UIA text, and they reach change events and, when history is on, the history.
- **How much:** one read per window per ``min_gap_s`` (2 s), lines from a re-read area replace the old ones there,
  and an NPU time budget (``budget_ms_per_min``) caps the duty cycle.
- **NPU only:** if the recognizer does not run on the NPU, or the NPU is reserved or failing, it pauses; it never
  falls back to the CPU around the clock. ``IGNITE_SCRIBE_ALWAYS_ON_CPU=1`` allows the CPU for measurement only.
- **Pauses** (``pauses()``): an ignite-xdna NPU quiet request (``%LOCALAPPDATA%\\ignite-xdna\\npu-quiet*.request``
  whose pid is alive), running on battery, or the user's pause (``python -m ignite_scribe pause``). While paused
  the vision worker is closed, so no NPU context is held.
"""
from __future__ import annotations

import ctypes
import json
import msvcrt
import os
import threading
import time
from ctypes import wintypes
from pathlib import Path

from .snapshot import Rect, TextBlock

LOCAL = Path(os.environ.get("LOCALAPPDATA", str(Path.home())))
SCRIBE_DIR = LOCAL / "ignite-scribe"
QUIET_DIR = LOCAL / "ignite-xdna"
PAUSE_FILE = SCRIBE_DIR / "always-on.paused"

kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)


class PowerStatus(ctypes.Structure):
    _fields_ = [("ACLineStatus", ctypes.c_ubyte), ("BatteryFlag", ctypes.c_ubyte),
                ("BatteryLifePercent", ctypes.c_ubyte), ("SystemStatusFlag", ctypes.c_ubyte),
                ("BatteryLifeTime", wintypes.DWORD), ("BatteryFullLifeTime", wintypes.DWORD)]


kernel32.GetSystemPowerStatus.argtypes = [ctypes.POINTER(PowerStatus)]
kernel32.OpenProcess.restype = wintypes.HANDLE
kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
kernel32.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
PROCESS_QUERY_LIMITED_INFORMATION, STILL_ACTIVE = 0x1000, 259


def enabled(policy=None) -> bool:
    v = os.environ.get("IGNITE_SCRIBE_ALWAYS_ON")
    if v is not None:
        return v == "1"
    return bool(getattr(policy, "always_on_ocr", False))


def cpu_allowed() -> bool:
    return os.environ.get("IGNITE_SCRIBE_ALWAYS_ON_CPU") == "1"


def pid_alive(pid: int) -> bool:
    h = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid))
    if not h:
        return False
    try:
        code = wintypes.DWORD()
        return bool(kernel32.GetExitCodeProcess(h, ctypes.byref(code))) and code.value == STILL_ACTIVE
    finally:
        kernel32.CloseHandle(h)


def quiet_requests(d: Path | None = None) -> list[dict]:
    """Live NPU quiet requests from ignite-xdna's runs (a request whose pid has exited is stale and ignored)."""
    out = []
    for f in (d or QUIET_DIR).glob("npu-quiet*.request"):
        try:
            r = json.loads(f.read_text(encoding="utf-8"))
            if pid_alive(int(r["pid"])):
                out.append(r)
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return out


def on_battery() -> bool:
    s = PowerStatus()
    return bool(kernel32.GetSystemPowerStatus(ctypes.byref(s))) and s.ACLineStatus == 0


def npu_recognizer_ready() -> tuple[bool, str]:
    """Whether an NPU build of the recognizer exists, checked from files, so nothing opens the NPU to find out."""
    if cpu_allowed():
        return True, ""
    from . import vision_client
    p = Path(vision_client.config().get("ocr_rec_container", ""))
    if not p.is_file():
        return False, f"the OCR recognizer is not on the NPU yet (no {p.name}; docs/OCR.md)"
    return True, ""


def pauses(quiet_dir: Path | None = None, pause_file: Path | None = None) -> list[str]:
    why = []
    q = quiet_requests(quiet_dir)
    if q:
        why.append("NPU reserved for a measurement (" + ", ".join(str(r.get("reason", r["pid"])) for r in q) + ")")
    if on_battery():
        why.append("on battery")
    if (pause_file or PAUSE_FILE).exists():
        why.append("paused by the user (python -m ignite_scribe resume)")
    return why


def take_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    f = open(path, "a+b")
    try:
        f.seek(0)
        msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError:
        f.close()
        return None
    return f


def _inter(a: Rect, b: Rect) -> Rect | None:
    r = Rect(max(a.left, b.left), max(a.top, b.top), min(a.right, b.right), min(a.bottom, b.bottom))
    return r if r.width > 0 and r.height > 0 else None


class AlwaysOn:
    def __init__(self, watcher, policy, worker_factory, min_gap_s: float = 2.0, budget_ms_per_min: float = 6000.0,
                 pad: int = 24, max_lines: int = 400, tick_s: float = 0.5, is_elevated=None,
                 quiet_dir: Path | None = None, pause_file: Path | None = None, ready=None):
        """``worker_factory()`` -> an object with ``ocr(images)`` (the vision worker) and ``close()``."""
        self.watcher, self.policy, self.worker_factory = watcher, policy, worker_factory
        self.min_gap_s, self.budget_ms, self.pad, self.max_lines, self.tick_s = (
            min_gap_s, budget_ms_per_min, pad, max_lines, tick_s)
        self.is_elevated, self.quiet_dir, self.pause_file = is_elevated, quiet_dir, pause_file
        self.ready = ready or npu_recognizer_ready
        self.worker = None
        self.lines: dict[int, list[TextBlock]] = {}
        self._last: dict[int, float] = {}
        self._held: dict[int, Rect] = {}             # changed areas waiting for their window's next turn
        self._spent: list[tuple[float, float]] = []  # (time, ms) of OCR calls in the last minute
        self.paused: list[str] = []
        self.stats = dict(reads=0, lines=0, cache_hits=0, ms_total=0.0, det_ms=0.0, rec_ms=0.0, pixels=0,
                          deferred_budget=0,
                          errors=0, pause_changes=0)
        self.last_error = None
        self.backend = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="ignite-scribe-always-on", daemon=True)

    def start(self) -> "AlwaysOn":
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        self._thread.join(15)
        self._release()

    def _release(self):
        w, self.worker = self.worker, None
        if w is not None:
            w.close()                                # the worker exits; its NPU context goes with it

    # ---- one pass ----------------------------------------------------------------------------------------------
    def candidates(self, snap) -> dict[int, object]:
        from . import capture
        from .vision_stage import THIN, excluded_fn
        excl = excluded_fn(self.policy)
        elevated = self.is_elevated or capture.elevated
        out = {}
        for w in snap.windows:
            uia = [b for b in w.text if b.source != "ocr"]
            if w.minimized or w.redacted or excl(w):
                continue
            if uia and not any(THIN in n for n in w.notes):
                continue                             # UI Automation describes it
            if not self.policy.capture_elevated and elevated(w.pid):
                continue
            out[w.hwnd] = w
        return out

    def step(self, now: float | None = None) -> int:
        """One pass over the changed windows. -> windows read."""
        from . import capture
        from .vision_stage import _overlap, excluded_fn
        import numpy as np
        now = time.time() if now is None else now
        snap, _, _ = self.watcher.snapshot()
        if snap is None:
            return 0
        for h, r in self.watcher.take_pixel_dirty().items():
            r = Rect(*r)
            old = self._held.get(h)
            self._held[h] = r if old is None else Rect(min(old.left, r.left), min(old.top, r.top),
                                                       max(old.right, r.right), max(old.bottom, r.bottom))
        cands = self.candidates(snap)
        for h in [h for h in self.lines if h not in cands]:
            self.lines.pop(h)                         # closed, excluded, or now described by UI Automation
            self.watcher.set_extra_text(h, None)
        for h in [h for h in self._held if h not in cands]:
            self._held.pop(h)
        self._spent = [(t, ms) for t, ms in self._spent if now - t < 60]
        read = 0
        excl = excluded_fn(self.policy)
        for h in sorted(self._held, key=lambda h: (not cands[h].focused, cands[h].z)):
            if now - self._last.get(h, 0) < self.min_gap_s:
                continue
            if sum(ms for _, ms in self._spent) >= self.budget_ms:
                self.stats["deferred_budget"] += 1
                break
            w = cands[h]
            r = self._held.pop(h)
            region = _inter(Rect(r.left - self.pad, r.top - self.pad, r.right + self.pad, r.bottom + self.pad), w.rect)
            if region is None:
                continue
            cap = capture.masked_window(snap, w, excl, region=region)
            if cap is None:
                continue
            img, origin, _ = cap
            if self.worker is None:
                self.worker = self.worker_factory()
            t0 = time.perf_counter()
            reply = self.worker.ocr([img])
            ms = (time.perf_counter() - t0) * 1000
            self.backend = reply.get("ocr_backend")
            if self.backend != "npu" and not cpu_allowed():
                self._release()
                raise RuntimeError("the OCR recognizer is not on the NPU yet (docs/OCR.md); always-on waits for it")
            self._spent.append((now, ms))
            self._last[h] = now
            res = reply["images"][0]
            b = np.array([ln["box"] for ln in res["lines"]], np.float32).reshape(-1, 4)
            b[:, [0, 2]] += origin.left
            b[:, [1, 3]] += origin.top
            keep = _overlap(b, capture.mask_rects(snap, w, excl)) < 0.5
            new = [TextBlock(ln["text"], Rect(*(int(v) for v in bb)), "ocr")
                   for ln, bb, k in zip(res["lines"], b.tolist(), keep) if k]
            kept = [x for x in self.lines.get(h, [])
                    if not (origin.left <= (x.rect.left + x.rect.right) / 2 <= origin.right and
                            origin.top <= (x.rect.top + x.rect.bottom) / 2 <= origin.bottom)]
            merged = sorted(kept + new, key=lambda x: (x.rect.top, x.rect.left))[-self.max_lines:]
            self.lines[h] = merged
            self.watcher.set_extra_text(h, merged)
            self.stats["reads"] += 1
            self.stats["lines"] += len(new)
            self.stats["ms_total"] += ms
            self.stats["det_ms"] += res.get("det_ms") or 0
            self.stats["rec_ms"] += res.get("rec_ms") or 0
            self.stats["cache_hits"] += res.get("cache_hits") or 0
            self.stats["pixels"] += img.shape[0] * img.shape[1]
            read += 1
        return read

    def _run(self):
        while not self._stop.wait(self.tick_s):
            why = pauses(self.quiet_dir, self.pause_file)
            ok, not_ready = self.ready()
            if not ok:
                why = why + [not_ready]
            if why != self.paused:
                self.stats["pause_changes"] += 1
                self.paused = why
            if why:
                self._release()                       # hold no NPU context while paused
                self.watcher.take_pixel_dirty()      # what changed meanwhile is read on the next change
                continue
            try:
                self.step()
            except Exception as e:  # noqa: BLE001 - never take the server down; back off
                self.stats["errors"] += 1
                self.last_error = f"{type(e).__name__}: {e}"
                self._release()
                self._stop.wait(30)

    def status(self) -> dict:
        s = dict(self.stats)
        n = s["reads"]
        for k in ("ms_total", "det_ms", "rec_ms"):
            s[k] = round(s[k], 1)
        spent = sum(ms for t, ms in self._spent if time.time() - t < 60)
        return dict(running=self._thread.is_alive(), paused=self.paused or None, backend=self.backend,
                    windows=len(self.lines), ms_per_read=round(s["ms_total"] / n, 1) if n else None,
                    ms_last_minute=round(spent, 1), budget_ms_per_min=self.budget_ms, min_gap_s=self.min_gap_s,
                    npu_held=self.worker is not None, cpu_allowed=cpu_allowed(), **s, last_error=self.last_error)
