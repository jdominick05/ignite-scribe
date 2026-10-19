# Copyright (C) 2026 The ignite-scribe contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Semantic screen history: embed the history's window passages in the background, search them by meaning.

Stdlib only on the server side; the model runs in ``embed_worker`` (a subprocess on the iGPU, below-normal
priority, started on demand, stopped after ``idle`` seconds). Only the recording process indexes.
- **Indexer:** every ``every_s`` it takes passage chunks not yet embedded (``History.unembedded``), embeds up to
  ``max_per_round`` passages, and stores their vectors as a sealed ``vectors`` chunk (``History.add_vectors``).
  Vectors are encrypted like the text, since text can be partly recovered from an embedding. When the worker
  cannot start, passages wait, and ``status`` says why.
- **Search:** the server decrypts the vectors in range, the worker embeds the query and ranks them, and the best
  passages come back with their window, time span and score. Reads stay inside the retention window.
"""
from __future__ import annotations

import json
import os
import struct
import subprocess
import sys
import threading
import time

from . import history as H

from pathlib import Path

MODEL = "bge-small-en-v1.5"
DIM = 384
ROOT = Path(__file__).resolve().parents[2]


def model_dir() -> Path:
    return Path(os.environ.get("IGNITE_SCRIBE_EMBED_DIR", str(ROOT / "models" / "embed" / MODEL)))


def enabled() -> bool:
    return os.environ.get("IGNITE_SCRIBE_EMBED", "1") != "0"


def available() -> tuple[bool, str]:
    """Checked in a subprocess-free way: the model files and the worker's imports, in this interpreter."""
    import importlib.util
    d = model_dir()
    missing = [f for f in ("model.onnx", "tokenizer.json") if not (d / f).is_file()]
    if missing:
        return False, f"missing {', '.join(missing)} in {d} (README: Getting the models)"
    for mod in ("numpy", "onnxruntime", "tokenizers"):
        if importlib.util.find_spec(mod) is None:
            return False, f"{mod} is not installed (pip install -e .[embed])"
    return True, ""


class EmbedWorker:
    def __init__(self, idle: float = 60.0, python: str | None = None):
        self.idle, self.python = idle, python or sys.executable
        self.proc = None
        self.backend = self.reason = self.precision = None
        self.last_used = 0.0
        self.starts = self.calls = 0
        self._lock = threading.Lock()
        self._timer = None

    def _start(self):
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        self.proc = subprocess.Popen([self.python, "-m", "ignite_scribe.embed_worker"], stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, creationflags=flags,
                                     env=dict(os.environ, PYTHONIOENCODING="utf-8"))
        self.starts += 1
        h, _ = self._call(dict(op="ping"))
        if not h.get("ok"):
            self._stop()
            raise RuntimeError(f"embed worker did not start: {h}")
        self.backend, self.reason, self.precision = h.get("backend"), h.get("reason"), h.get("precision")
        if self._timer is None or not self._timer.is_alive():
            self._timer = threading.Thread(target=self._idle_loop, name="ignite-scribe-embed-idle", daemon=True)
            self._timer.start()

    def _stop(self):
        p, self.proc = self.proc, None
        if p is None:
            return
        try:
            p.stdin.close()                         # EOF: the worker exits and releases the GPU
            p.wait(timeout=10)
        except Exception:  # noqa: BLE001
            p.kill()
            p.wait(timeout=10)

    def _idle_loop(self):
        while True:
            time.sleep(1.0)
            with self._lock:
                if self.proc is None:
                    return
                if time.monotonic() - self.last_used > self.idle:
                    self._stop()
                    return

    def close(self):
        with self._lock:
            self._stop()

    @property
    def running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def _call(self, header: dict, raw: bytes = b"") -> tuple[dict, bytes]:
        body = json.dumps(header).encode("utf-8") + b"\n" + raw
        self.proc.stdin.write(struct.pack("<I", len(body)) + body)
        self.proc.stdin.flush()
        n = self.proc.stdout.read(4)
        if len(n) < 4:
            raise RuntimeError("embed worker exited")
        reply = self.proc.stdout.read(struct.unpack("<I", n)[0])
        i = reply.index(b"\n")
        return json.loads(reply[:i].decode("utf-8")), reply[i + 1:]

    def request(self, header: dict, raw: bytes = b"") -> tuple[dict, bytes]:
        with self._lock:
            if not self.running:
                self._start()
            self.last_used = time.monotonic()
            try:
                h, out = self._call(header, raw)
            except Exception:
                self._stop()
                raise
            self.last_used = time.monotonic()
            self.calls += 1
        if not h.get("ok"):
            raise RuntimeError(f"embed worker: {h.get('error')}")
        return h, out

    def embed(self, texts: list[str]) -> bytes:
        return self.request(dict(op="embed", texts=texts))[1]

    def rank(self, query: str, vecs: bytes, k: int) -> tuple[list[int], list[float]]:
        h, _ = self.request(dict(op="search", query=query, n=len(vecs) // (2 * DIM), k=k), vecs)
        return h["idx"], h["score"]

    def status(self) -> dict:
        return dict(running=self.running, backend=self.backend, precision=self.precision, reason=self.reason,
                    idle_stop_s=self.idle, starts=self.starts, calls=self.calls)


class Indexer:
    def __init__(self, hist: H.History, worker, every_s: float = 60.0, max_per_round: int = 256, batch: int = 32):
        self.hist, self.worker, self.every_s = hist, worker, every_s
        self.max_per_round, self.batch = max_per_round, batch
        self.stats = dict(rounds=0, embedded=0, embed_s=0.0, errors=0)
        self.last_error = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="ignite-scribe-index", daemon=True)

    def start(self) -> "Indexer":
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        self._thread.join(15)

    def run_once(self) -> int:
        """Embed one round of the backlog. -> passages embedded."""
        from .passages import embed_text
        if not self.hist.recorder:
            return 0
        done = 0
        for cid, t0, t1, ps in self.hist.unembedded(MODEL):
            if done >= self.max_per_round or self._stop.is_set():
                break
            vecs = b""
            t = time.perf_counter()
            for s in range(0, len(ps), self.batch):
                vecs += self.worker.embed([embed_text(p) for p in ps[s:s + self.batch]])
            self.stats["embed_s"] += time.perf_counter() - t
            meta = [dict(i=i, t0=p["t0"], t1=p["t1"], title=p.get("title"), process=p.get("process"))
                    for i, p in enumerate(ps)]
            self.hist.add_vectors(MODEL, cid, t0, t1, meta, vecs)
            done += len(ps)
        self.stats["embedded"] += done
        self.stats["rounds"] += 1
        return done

    def _run(self):
        while not self._stop.wait(self.every_s):
            try:
                self.run_once()
            except Exception as e:  # noqa: BLE001 - indexing must never take the server down
                self.stats["errors"] += 1
                self.last_error = f"{type(e).__name__}: {e}"

    def status(self) -> dict:
        s = dict(self.stats)
        s["embed_s"] = round(s["embed_s"], 2)
        return dict(**self.hist.index_status(MODEL), every_s=self.every_s, **s, last_error=self.last_error)


def search(hist: H.History, worker, query: str, since: float | None = None, until: float | None = None,
           match: str | None = None, k: int = 10) -> dict:
    """The ``k`` passages closest in meaning to ``query``, best first."""
    now = time.time()
    since = hist.horizon(now) if since is None else max(since, hist.horizon(now))
    until = now if until is None else until
    m = (match or "").lower()
    bad: list = []
    rows, vecs = [], []
    for _, _, c in hist.vector_chunks(MODEL, since, until, bad):
        raw = bytes.fromhex(c["vecs"])
        for j, meta in enumerate(c["meta"]):
            if meta["t1"] < since or meta["t0"] > until:
                continue
            if m and m not in (meta.get("title") or "").lower() and m not in (meta.get("process") or "").lower():
                continue
            rows.append((c["src"], meta))
            vecs.append(raw[j * 2 * DIM:(j + 1) * 2 * DIM])
    res = dict(query=query, since=H.iso(since), until=H.iso(until), searched=len(rows), hits=[],
               index=hist.index_status(MODEL))
    if bad:
        res["unreadable_chunks"] = len(bad)
    if not rows:
        return res
    idx, score = worker.rank(query, b"".join(vecs), k)
    texts: dict[int, list] = {}
    for i, sc in zip(idx, score):
        src, meta = rows[i]
        if src not in texts:
            texts[src] = hist.passages_of(src)
        ps = texts[src]
        text = ps[meta["i"]]["text"] if meta["i"] < len(ps) else None
        res["hits"].append(dict(score=sc, first_seen=H.iso(meta["t0"]), last_seen=H.iso(meta["t1"]),
                                title=meta.get("title"), process=meta.get("process"), text=text))
    return res
