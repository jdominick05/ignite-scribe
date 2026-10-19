# Copyright (C) 2026 The ignite-scribe contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Always-on OCR, offline: pauses (quiet requests, user pause), window choice, region reads and their merge, the
budget, the NPU-only gate, the watcher's redaction of OCR text, and the per-call refusal during a quiet request."""
import json
import os
import subprocess
import sys
import time

import numpy as np
import pytest

from ignite_scribe import always_on as A
from ignite_scribe import capture
from ignite_scribe.redact import Policy
from ignite_scribe.snapshot import Rect, Snapshot, TextBlock, Window


def win(hwnd, title, process, text=(), z=0, focused=False, rect=Rect(0, 0, 800, 600), notes=()):
    return Window(hwnd, title, process, 100 + hwnd, rect, z, focused,
                  text=[TextBlock(t) for t in text], notes=list(notes))


class FakeWatcher:
    def __init__(self, snap):
        self.snap, self.dirty, self.extra = snap, {}, {}

    def snapshot(self):
        return self.snap, {}, time.time()

    def take_pixel_dirty(self):
        d, self.dirty = self.dirty, {}
        return d

    def set_extra_text(self, hwnd, blocks):
        if blocks:
            self.extra[hwnd] = list(blocks)
        else:
            self.extra.pop(hwnd, None)


class FakeWorker:
    def __init__(self, backend="npu", lines=None):
        self.backend, self.calls, self.closed = backend, [], 0
        self.lines = lines or [dict(box=[10, 10, 200, 30], text="hello from the canvas", score=0.9)]

    def ocr(self, images):
        self.calls.append(images[0].shape)
        return dict(ok=True, ocr_backend=self.backend, images=[dict(lines=self.lines, det_ms=5.0, rec_ms=2.0)])

    def close(self):
        self.closed += 1


@pytest.fixture(autouse=True)
def no_screen(monkeypatch):
    """No real pixels in tests: the capture returns grey of the requested region."""
    def fake(snap, target, excluded, region=None):
        r = region or target.rect
        return np.full((r.height, r.width, 3), 114, np.uint8), r, 0.0
    monkeypatch.setattr(capture, "masked_window", fake)
    monkeypatch.setattr(capture, "mask_rects", lambda snap, target, excluded: [])


def engine(snap, worker, **kw):
    w = FakeWatcher(snap)
    e = A.AlwaysOn(w, Policy(), lambda: worker, is_elevated=lambda pid: False, ready=lambda: (True, ""), **kw)
    return w, e


def test_quiet_requests_live_stale_and_half_written(tmp_path):
    (tmp_path / f"npu-quiet.{os.getpid()}.request").write_text(json.dumps(dict(pid=os.getpid(), reason="bench")))
    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    dead.wait()
    (tmp_path / f"npu-quiet.{dead.pid}.request").write_text(json.dumps(dict(pid=dead.pid, reason="old")))
    (tmp_path / "npu-quiet.999.request").write_text('{"pid": 12')          # being written
    assert [r["reason"] for r in A.quiet_requests(tmp_path)] == ["bench"]
    why = A.pauses(tmp_path, tmp_path / "nopause")
    assert any("measurement" in w and "bench" in w for w in why)
    (tmp_path / "pause").write_bytes(b"")
    assert any("user" in w for w in A.pauses(tmp_path / "empty", tmp_path / "pause"))


def test_candidates_skip_excluded_minimized_and_uia_described():
    snap = Snapshot(time.time(), [
        win(1, "Canvas", "figma.exe"),
        win(2, "Notes", "notepad.exe", ["described by UIA"]),
        win(3, "Thin", "app.exe", ["x"], notes=["thin: few UIA nodes"]),
        win(4, "Vault", "keepass.exe"),
        Window(5, "Min", "a.exe", 105, Rect(0, 0, 9, 9), 4, minimized=True),
        win(6, "Image", "photos.exe", ()),
    ])
    snap.windows[5].text = [TextBlock("old ocr", Rect(0, 0, 1, 1), "ocr")]   # its own OCR text is not UIA's
    _, e = engine(snap, FakeWorker())
    from ignite_scribe.vision_stage import THIN
    snap.windows[2].notes = [THIN]
    assert set(e.candidates(snap)) == {1, 3, 6}


def test_step_reads_only_the_changed_region_and_merges():
    snap = Snapshot(time.time(), [win(1, "Canvas", "figma.exe", focused=True, rect=Rect(100, 100, 1100, 900))])
    wk = FakeWorker()
    w, e = engine(snap, wk, pad=0)
    w.dirty = {1: (200, 200, 400, 260)}
    assert e.step(now=1000.0) == 1
    assert wk.calls == [(60, 200, 3)]                                  # only the changed area was captured
    first = w.extra[1]
    assert [b.text for b in first] == ["hello from the canvas"] and first[0].source == "ocr"
    assert (first[0].rect.left, first[0].rect.top) == (210, 210)        # image pixels -> screen pixels
    wk.lines = [dict(box=[5, 5, 100, 20], text="new line", score=0.9)]
    w.dirty = {1: (600, 600, 700, 650)}                                 # elsewhere: the old line stays
    assert e.step(now=1001.0) == 0                                      # too soon (min_gap_s)
    assert e.step(now=1003.0) == 1
    assert [b.text for b in w.extra[1]] == ["hello from the canvas", "new line"]
    wk.lines = [dict(box=[0, 0, 150, 20], text="replaced", score=0.9)]
    w.dirty = {1: (200, 200, 400, 260)}                                 # the first area again: replaced
    e.step(now=1006.0)
    assert [b.text for b in w.extra[1]] == ["replaced", "new line"]


def test_window_leaving_the_candidates_drops_its_text():
    snap = Snapshot(time.time(), [win(1, "Canvas", "figma.exe")])
    w, e = engine(snap, FakeWorker(), pad=0)
    w.dirty = {1: (0, 0, 100, 100)}
    e.step(now=10.0)
    assert 1 in w.extra
    snap.windows[0].text = [TextBlock("now UIA describes it")]
    e.step(now=20.0)
    assert 1 not in w.extra


def test_budget_defers_reads():
    snap = Snapshot(time.time(), [win(i, f"C{i}", "c.exe", z=i, rect=Rect(0, 0, 500, 500)) for i in range(1, 4)])
    w, e = engine(snap, FakeWorker(), budget_ms_per_min=0.0)
    w.dirty = {i: (0, 0, 50, 50) for i in range(1, 4)}
    assert e.step(now=50.0) == 0 and e.stats["deferred_budget"] == 1
    assert set(e._held) == {1, 2, 3}                                    # kept for later, not lost


def test_cpu_recognizer_is_refused_unless_measuring(monkeypatch):
    snap = Snapshot(time.time(), [win(1, "Canvas", "figma.exe")])
    wk = FakeWorker(backend="cpu")
    w, e = engine(snap, wk)
    w.dirty = {1: (0, 0, 100, 100)}
    monkeypatch.delenv("IGNITE_SCRIBE_ALWAYS_ON_CPU", raising=False)
    with pytest.raises(RuntimeError, match="not on the NPU"):
        e.step(now=10.0)
    assert wk.closed == 1 and e.worker is None
    monkeypatch.setenv("IGNITE_SCRIBE_ALWAYS_ON_CPU", "1")
    w.dirty = {1: (0, 0, 100, 100)}
    assert e.step(now=20.0) == 1


def test_not_ready_pauses_without_starting_the_worker(tmp_path):
    snap = Snapshot(time.time(), [win(1, "Canvas", "figma.exe")])
    started = []
    w = FakeWatcher(snap)
    e = A.AlwaysOn(w, Policy(), lambda: started.append(1), ready=lambda: (False, "no npu recognizer"),
                   quiet_dir=tmp_path, pause_file=tmp_path / "p", tick_s=0.05).start()
    try:
        w.dirty = {1: (0, 0, 10, 10)}
        time.sleep(0.4)
        assert e.paused == ["no npu recognizer"] and started == []
    finally:
        e.stop()


def test_watcher_redacts_always_on_text(monkeypatch):
    from ignite_scribe import uia, watch, windows
    raw = Snapshot(time.time(), [win(1, "Canvas", "figma.exe"), win(2, "Vault", "keepass.exe", z=1)])
    monkeypatch.setattr(windows, "snapshot", lambda: Snapshot(raw.taken_at, [
        win(1, "Canvas", "figma.exe"), win(2, "Vault", "keepass.exe", z=1)]))
    monkeypatch.setattr(uia, "available", lambda: False)
    wt = watch.Watcher(Policy())
    secret = "sk-abcdefghijklmnopqrstuvwxyz0123"
    wt.set_extra_text(1, [TextBlock(f"token {secret}", Rect(0, 0, 9, 9), "ocr")])
    wt.set_extra_text(2, [TextBlock("never shown", Rect(0, 0, 9, 9), "ocr")])
    wt._refresh()
    state, counts, _ = wt.snapshot()
    text = json.dumps(state.to_dict())
    assert "token" in text and secret not in text and "never shown" not in text
    assert counts["secrets_withheld"] >= 1


def test_per_call_vision_refuses_during_a_quiet_request(monkeypatch, tmp_path):
    from ignite_scribe.mcp_server import Server
    monkeypatch.setattr(A, "quiet_requests", lambda d=None: [dict(pid=1, reason="gemv sitting")])
    s = Server(source=lambda **k: Snapshot(time.time(), [win(1, "x", "x.exe")]), policy=Policy())
    with pytest.raises(RuntimeError, match="reserved for a measurement"):
        s._vision_snapshot(lambda: (True, ""))


def test_ready_needs_the_model_and_every_bucket_on_the_npu(tmp_path, monkeypatch):
    monkeypatch.delenv("IGNITE_SCRIBE_ALWAYS_ON_CPU", raising=False)
    model, build = tmp_path / "model", tmp_path / "build"
    model.mkdir()
    build.mkdir()
    monkeypatch.setenv("IGNITE_SCRIBE_OCR_REC_MODEL", str(model))
    monkeypatch.setenv("IGNITE_SCRIBE_OCR_REC_CONTAINERS", str(build))
    assert A.npu_recognizer_ready()[0] is False                  # no model
    (model / "tail.onnx").write_bytes(b"")
    for w in (320, 640, 960):
        (build / f"scriberec_{w}.ignite").write_bytes(b"")
    ok, why = A.npu_recognizer_ready()
    assert ok is False and "scriberec_1280" in why               # one bucket would fall back to the CPU
    (build / "scriberec_1280.ignite").write_bytes(b"")
    assert A.npu_recognizer_ready() == (True, "")
