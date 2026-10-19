# Copyright (C) 2026 The ignite-scribe contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Semantic history, offline: passages, the indexer, encrypted vectors, ranked search, the MCP mode. A hashed
bag-of-words stands in for the model; the real worker is checked by tools/semantic_live.py."""
import hashlib
import json
import time

import numpy as np
import pytest

from ignite_scribe import history as H
from ignite_scribe import semantic as S
from ignite_scribe.mcp_server import Server
from ignite_scribe.passages import Passages, embed_text
from ignite_scribe.redact import Policy
from ignite_scribe.snapshot import Rect, Snapshot, TextBlock, Window


class FakeWorker:
    """Hashed bag of words, unit length: texts sharing words score high."""
    def __init__(self):
        self.calls = 0

    @staticmethod
    def vec(text):
        v = np.zeros(S.DIM, np.float32)
        for w in text.lower().split():
            v[int.from_bytes(hashlib.sha256(w.encode()).digest()[:4], "little") % S.DIM] += 1
        return v / max(np.linalg.norm(v), 1e-9)

    def embed(self, texts):
        self.calls += 1
        return np.stack([self.vec(t) for t in texts]).astype(np.float16).tobytes()

    def rank(self, query, vecs, k):
        m = np.frombuffer(vecs, np.float16).reshape(-1, S.DIM).astype(np.float32)
        s = m @ self.vec(query)
        top = np.argsort(-s)[:k]
        return top.tolist(), [float(s[i]) for i in top]

    def close(self):
        pass

    def status(self):
        return dict(running=False, backend="fake")


def content(t, hwnd, title, process, lines, seq=1):
    return dict(kind="content", hwnd=hwnd, title=title, process=process, added=lines, added_total=len(lines),
                removed=0, seq=seq, t=t)


def test_passages_bucket_dedupe_split_and_skip_redacted():
    p = Passages(span_s=300, max_chars=40)
    t = 3000.0
    p.feed([content(t, 1, "Notes", "notepad.exe", ["alpha beta", "gamma delta"]),
            content(t + 10, 1, "Notes", "notepad.exe", ["alpha beta", "epsilon zeta eta theta iota"])])
    snap = dict(taken_at=t + 20, windows=[dict(hwnd=2, title="Withheld", process="keepass.exe", redacted=True,
                                               text=[dict(text="never")])])
    p.feed([], snap)
    assert p.close(now=t + 100) == []                    # the bucket is still open
    out = p.close(now=t + 400)
    assert [x["text"] for x in out] == ["alpha beta\ngamma delta", "epsilon zeta eta theta iota"]
    assert all(x["process"] == "notepad.exe" for x in out)
    p.feed([content(t + 600, 1, "Notes", "notepad.exe", ["alpha beta", "gamma delta"])])
    assert [x["text"] for x in p.close(now=t + 1000)] == []   # the same window said the same thing again
    assert embed_text(out[0]).startswith("Notes - notepad.exe\n")


@pytest.fixture
def hist(tmp_path):
    h = H.History(tmp_path / "s.db", flush_s=0.05)
    h.passages = Passages(span_s=1)                       # buckets close after a second here
    yield h
    h.close()


def wait(cond, s=5):
    end = time.time() + s
    while time.time() < end:
        if cond():
            return
        time.sleep(0.05)
    raise AssertionError("timed out")


def fill(h):
    now = time.time()
    h.record([content(now - 3, 1, "GPU prices - Browser", "firefox.exe",
                      ["graphics card prices fell this quarter", "the radeon and geforce lines"], 1),
              content(now - 3, 2, "Budget.xlsx", "excel.exe", ["rent groceries utilities total"], 2),
              content(now - 3, 3, "Terminal", "wt.exe", ["git push origin main", "compile the kernel"], 3)])
    wait(lambda: h.stats.get("passages_written", 0) >= 3)
    return now


def test_indexer_embeds_the_backlog_and_vectors_are_encrypted(hist, tmp_path):
    fill(hist)
    w = FakeWorker()
    ix = S.Indexer(hist, w)
    assert hist.index_status(S.MODEL)["backlog_passages"] == 3
    assert ix.run_once() == 3 and w.calls >= 1
    st = hist.index_status(S.MODEL)
    assert st["vectors"] == 3 and st["backlog_passages"] == 0
    assert ix.run_once() == 0                             # nothing left
    hist.close()
    raw = (tmp_path / "s.db").read_bytes() + b"".join(f.read_bytes() for f in tmp_path.glob("s.db-*"))
    for s in (b"radeon", b"firefox", b"GPU prices", b"groceries"):
        assert s not in raw


def test_semantic_search_ranks_by_meaning_and_filters(hist):
    fill(hist)
    w = FakeWorker()
    S.Indexer(hist, w).run_once()
    out = S.search(hist, w, "radeon graphics card prices")
    assert out["hits"][0]["process"] == "firefox.exe" and "radeon" in out["hits"][0]["text"]
    assert out["searched"] == 3
    only = S.search(hist, w, "radeon graphics card prices", match="excel")
    assert [x["process"] for x in only["hits"]] == ["excel.exe"]
    assert S.search(hist, w, "radeon", since=time.time() + 60)["hits"] == []


def test_a_reader_never_indexes(tmp_path):
    a = H.History(tmp_path / "r.db")
    b = H.History(tmp_path / "r.db")
    try:
        assert a.recorder and not b.recorder
        assert S.Indexer(b, FakeWorker()).run_once() == 0
    finally:
        b.close()
        a.close()


def test_mcp_semantic_mode_and_status(tmp_path):
    h = H.History(tmp_path / "m.db", flush_s=0.05)
    h.passages = Passages(span_s=1)

    class FakeWatcher:
        max_windows = 6

        def start(self):
            return self

        def stop(self):
            pass

        def status(self):
            return dict(running=True)

    s = Server(source=lambda **k: Snapshot(time.time(), [Window(1, "x", "x.exe", 1, Rect(0, 0, 1, 1), 0)]),
               policy=Policy(), watcher=FakeWatcher(), history=h)
    s.embedder = FakeWorker()
    try:
        fill(h)
        S.Indexer(h, s.embedder).run_once()
        out = json.loads(s.call("search_screen_history", {"query": "graphics card prices", "mode": "semantic"}))
        assert out["note"] and out["hits"][0]["process"] == "firefox.exe"
        with pytest.raises(ValueError):
            s.call("search_screen_history", {"mode": "semantic"})
        sem = s.call("scribe_status", {})["history"]["semantic"]
        assert sem["index"]["vectors"] == 3 and sem["worker"]["backend"] == "fake"
    finally:
        s.close()
