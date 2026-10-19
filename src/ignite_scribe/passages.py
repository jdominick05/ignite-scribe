# Copyright (C) 2026 The ignite-scribe contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Window passages for semantic history: what each window showed in a span, as embeddable text. Stdlib only.

The recorder feeds ``Passages`` the same redacted events and snapshots it stores. Per window (hwnd, title,
process) and per ``span_s`` bucket (5 minutes), it collects the distinct text lines seen, in order; when a bucket
closes, they become passages of at most ``max_chars`` (about 200 tokens), split at line boundaries. A passage that
repeats one the same window already produced (a static document re-read every snapshot) is skipped. Windows with
no text (an excluded window has none) yield nothing.
"""
from __future__ import annotations

import hashlib
from collections import OrderedDict


class Passages:
    def __init__(self, span_s: float = 300.0, max_chars: int = 900, remember: int = 4096):
        self.span_s, self.max_chars, self.remember = span_s, max_chars, remember
        self._open: dict[tuple, dict] = {}            # (bucket, hwnd, title, process) -> {"t0", "t1", "lines"}
        self._seen: OrderedDict = OrderedDict()       # digests of recent passages, per window

    def _add(self, t, hwnd, title, process, lines):
        lines = [ln.rstrip() for ln in lines if ln and ln.strip()]
        if not lines:
            return
        key = (int(t // self.span_s), hwnd, title or "", process or "")
        b = self._open.get(key)
        if b is None:
            b = self._open[key] = dict(t0=t, t1=t, lines=OrderedDict())
        b["t0"], b["t1"] = min(b["t0"], t), max(b["t1"], t)
        for ln in lines:
            b["lines"].setdefault(ln, None)

    def feed(self, events: list[dict], snapshot: dict | None = None):
        """``snapshot`` is a redacted ``Snapshot.to_dict()``."""
        for e in events:
            if e.get("kind") == "content":
                self._add(e["t"], e.get("hwnd"), e.get("title"), e.get("process"), e.get("added", []))
        if snapshot is not None:
            t = snapshot["taken_at"]
            for w in snapshot.get("windows", []):
                if w.get("redacted"):
                    continue
                lines = [ln for b in w.get("text", []) for ln in b.get("text", "").split("\n")]
                self._add(t, w.get("hwnd"), w.get("title"), w.get("process"), lines)

    def close(self, now: float | None = None) -> list[dict]:
        """Passages of the buckets that ended before ``now`` (all of them when None)."""
        out = []
        for key in sorted(k for k in self._open if now is None or (k[0] + 1) * self.span_s <= now):
            b = self._open.pop(key)
            _, hwnd, title, process = key
            for text in self._split(list(b["lines"])):
                digest = hashlib.sha256(f"{hwnd}|{title}|{process}|{text}".encode()).digest()
                if digest in self._seen:
                    self._seen.move_to_end(digest)
                    continue
                self._seen[digest] = None
                if len(self._seen) > self.remember:
                    self._seen.popitem(last=False)
                out.append(dict(t0=b["t0"], t1=b["t1"], hwnd=hwnd, title=title, process=process, text=text))
        return out

    def _split(self, lines: list[str]) -> list[str]:
        out, cur, size = [], [], 0
        for ln in lines:
            ln = ln[:self.max_chars]
            if cur and size + len(ln) + 1 > self.max_chars:
                out.append("\n".join(cur))
                cur, size = [], 0
            cur.append(ln)
            size += len(ln) + 1
        if cur:
            out.append("\n".join(cur))
        return out


def embed_text(p: dict) -> str:
    """What is embedded: the window's title and process give the lines their context."""
    head = " - ".join(x for x in (p.get("title"), p.get("process")) if x)
    return f"{head}\n{p['text']}" if head else p["text"]
