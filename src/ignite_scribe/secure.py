# Copyright (C) 2026 The ignite-scribe contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Secure mode: the injection classifier (docs/INJECTION.md, tier 2) scores window text on the iGPU.

Opt-in (``"secure_mode": true`` in the policy, or ``IGNITE_SCRIBE_SECURE=1``). The model runs in the embedding
worker (``embed_worker``, DirectML, below-normal priority, exits when idle), never in this process and never on the
CPU: when DirectML is unavailable, secure mode stays off and ``scribe_status`` says why.

Scoring is asynchronous and incremental. Each window's (already scrubbed) text is cut into pieces of about one model
window (``chunks``), and ``Policy.apply`` asks ``lookup`` for each piece: a known score is
returned at once, an unknown text is queued and scored in the background, and ``on_scored`` asks the watcher to
re-apply the policy, so the flag appears a moment later. Scores are kept by a digest of the text, not the text.
A flag: ``{"kind": "classifier", "score": p, "source": "window", "app": ...}`` when ``p`` reaches the model's
threshold (chosen on validation data by its trainer).
"""
from __future__ import annotations

import hashlib
import os
import threading
import time
from collections import OrderedDict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODEL = "scribe-guard"


def model_dir() -> Path:
    """``IGNITE_SCRIBE_SECURE_MODEL``, else the user's install (``%LOCALAPPDATA%\\ignite-scribe\\models``), else a
    checkout's ``models/``."""
    v = os.environ.get("IGNITE_SCRIBE_SECURE_MODEL")
    if v:
        return Path(v)
    user = Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "ignite-scribe" / "models" / "injection" / MODEL
    return user if (user / "config.json").is_file() else ROOT / "models" / "injection" / MODEL


def enabled(policy=None) -> bool:
    v = os.environ.get("IGNITE_SCRIBE_SECURE")
    if v is not None:
        return v == "1"
    return bool(getattr(policy, "secure_mode", False))


def available() -> tuple[bool, str]:
    """From files and imports only: nothing is started."""
    import importlib.util
    d = model_dir()
    missing = [f for f in ("tokenizer.json", "config.json") if not (d / f).is_file()]
    if not (d / "model_fp16.onnx").is_file() and not (d / "model.onnx").is_file():   # DirectML loads the fp16 one
        missing.append("model_fp16.onnx")
    if missing:
        return False, f"the secure-mode model is not installed (missing {', '.join(missing)} in {d})"
    for mod in ("numpy", "onnxruntime", "tokenizers"):
        if importlib.util.find_spec(mod) is None:
            return False, f"{mod} is not installed (pip install ignite-scribe[embed])"
    return True, ""


def window_text(w) -> str:
    return "\n".join([w.title or ""] + [b.text for b in w.text])


CHUNK_CHARS = 1500      # about one 512-token window


def chunks(text: str, target: int = CHUNK_CHARS) -> list[str]:
    """Cut text into pieces of about ``target`` characters at content-defined line boundaries: a line whose digest
    is 0 mod 8 ends a piece, as does reaching ``target``. Boundaries depend on the lines themselves, not their
    position, so a terminal that adds or scrolls lines re-scores only the pieces that changed. Each piece starts with
    the previous piece's last line, so an instruction split across a boundary is still seen whole."""
    lines = [l for l in text.split("\n") if l.strip()]
    out, cur, size = [], [], 0
    for line in lines:
        line = line[:target]                       # one very long line still makes a bounded piece
        cur.append(line)
        size += len(line) + 1
        if size >= target or digest(line)[0] % 8 == 0:
            out.append("\n".join(cur))
            cur, size = [line], len(line) + 1         # overlap: the boundary line opens the next piece
    if len(cur) > 1 or not out:
        out.append("\n".join(cur))
    return [c for c in out if c.strip()]


def digest(text: str) -> bytes:
    return hashlib.blake2b(text.encode("utf-8", "surrogatepass"), digest_size=16).digest()


class Scanner:
    """Scores texts in the background through ``worker.request(dict(op="classify", ...))``."""

    def __init__(self, worker, model: Path | None = None, cache: int = 4096, batch: int = 8, max_queue: int = 256,
                 on_scored=None):
        self.worker, self.model = worker, str(model or model_dir())
        self.batch, self.max_queue = batch, max_queue
        self.on_scored = on_scored
        self.scores: OrderedDict[bytes, float] = OrderedDict()
        self.cache = cache
        self.queue: OrderedDict[bytes, str] = OrderedDict()
        self.threshold = None
        self.backend = None
        self.error = None
        self.stats = dict(scored=0, flagged=0, dropped=0, ms_total=0.0, batches=0)
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        self._thread = threading.Thread(target=self._run, name="ignite-scribe-secure", daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    @property
    def active(self) -> bool:
        return self.error is None and not self._stop.is_set()

    def lookup(self, text: str) -> float | None:
        """The text's score when known; else it is queued (when active) and None is returned."""
        if not text.strip() or not self.active:
            return None
        k = digest(text)
        with self._lock:
            if k in self.scores:
                self.scores.move_to_end(k)
                return self.scores[k]
            if k not in self.queue:
                if len(self.queue) >= self.max_queue:
                    self.queue.popitem(last=False)            # the oldest unscored text is likely gone already
                    self.stats["dropped"] += 1
                self.queue[k] = text
        self._wake.set()
        return None

    def flag(self, w, category: str) -> dict | None:
        """The classifier flag for a window, when its text is scored at or above the threshold."""
        if getattr(w, "redacted", False):
            return None
        known = [p for p in (self.lookup(c) for c in chunks(window_text(w))) if p is not None]
        p = max(known, default=None)                   # the window scores as its most suspicious piece
        if p is None or self.threshold is None or p < self.threshold:
            return None
        return dict(kind="classifier", score=round(p, 3), source="window", role=None, app=category)

    def _run(self):
        while not self._stop.is_set():
            self._wake.wait(1.0)
            self._wake.clear()
            while not self._stop.is_set():
                with self._lock:
                    items = [self.queue.popitem(last=False) for _ in range(min(self.batch, len(self.queue)))]
                if not items:
                    break
                try:
                    h, _ = self.worker.request(dict(op="classify", model_dir=self.model, texts=[t for _, t in items]))
                except Exception as e:  # noqa: BLE001 - secure mode turns itself off, the patterns keep working
                    self.error = f"{type(e).__name__}: {e}"
                    return
                if h.get("backend") != "dml":                     # never a CPU load
                    self.error = f"the classifier ran on {h.get('backend')!r}, not the iGPU; secure mode is off"
                    return
                self.backend, self.threshold = h["backend"], float(h["threshold"])
                with self._lock:
                    for (k, _), p in zip(items, h["scores"]):
                        self.scores[k] = float(p)
                        self.stats["flagged"] += p >= self.threshold
                    while len(self.scores) > self.cache:
                        self.scores.popitem(last=False)
                    self.stats["scored"] += len(items)
                    self.stats["batches"] += 1
                    self.stats["ms_total"] += float(h.get("ms", 0.0))
                if self.on_scored is not None:
                    try:
                        self.on_scored()
                    except Exception:  # noqa: BLE001
                        pass

    def status(self) -> dict:
        with self._lock:
            s = dict(self.stats)
            pending = len(self.queue)
        b = max(s["batches"], 1)
        return dict(on=self.active, backend=self.backend, threshold=self.threshold, error=self.error,
                    pending=pending, scored=s["scored"], flagged=s["flagged"], dropped=s["dropped"],
                    ms_per_batch=round(s["ms_total"] / b, 1), model=Path(self.model).name)


def wait_idle(scanner: Scanner, timeout: float = 10.0) -> bool:
    """Tests: wait until the queue is empty."""
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if not scanner.queue:
            return True
        time.sleep(0.02)
    return False
