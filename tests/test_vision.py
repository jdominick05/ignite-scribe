# Copyright (C) 2026 The ignite-scribe contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The vision stage, offline: decode, tiling, masking geometry, window choice, the merge, and the MCP tool over a
fake worker and a fake capture. The NPU itself is checked by tools/worker_parity.py, not here."""
import json

import numpy as np
import pytest

from ignite_scribe import vision as V
from ignite_scribe.redact import Policy
from ignite_scribe.snapshot import Rect, Snapshot, TextBlock, Window


def test_tile_origins_cover_with_overlap():
    for length in (300, 640, 641, 1280, 2560):
        xs = V.tile_origins(length, 640, 96)
        assert xs[0] == 0 and xs[-1] + 640 >= length
        assert all(b - a <= 640 - 96 for a, b in zip(xs, xs[1:]))


def test_decode_heads_places_a_box_where_the_logits_say():
    boxes = [np.zeros((64, 80, 80), np.float32), np.zeros((64, 40, 40), np.float32), np.zeros((64, 20, 20), np.float32)]
    classes = [np.full((1, 80, 80), -10, np.float32), np.full((1, 40, 40), -10, np.float32),
               np.full((1, 20, 20), -10, np.float32)]
    classes[0][0, 10, 20] = 5.0                       # stride 8, cell (x=20, y=10)
    boxes[0][:, 10, 20] = -10.0
    for side in range(4):
        boxes[0][side * 16 + 2, 10, 20] = 10.0         # every side 2 cells away
    b, s = V.decode_heads(boxes, classes, conf=0.5)
    assert len(b) == 1 and s[0] > 0.99
    np.testing.assert_allclose(b[0], [(20.5 - 2) * 8, (10.5 - 2) * 8, (20.5 + 2) * 8, (10.5 + 2) * 8], atol=1e-3)


def test_nms_and_to_screen():
    b = np.array([[0, 0, 10, 10], [1, 1, 10, 10], [50, 50, 60, 60]], np.float32)
    keep = V.nms(b, np.array([0.9, 0.8, 0.7], np.float32), 0.5)
    assert keep.tolist() == [0, 2]
    np.testing.assert_allclose(V.to_screen(b[:1], 100, 200, 0.5), [[200, 400, 220, 420]])


def test_free_rects_never_include_a_masked_pixel():
    pytest.importorskip("ctypes")
    from ignite_scribe import capture
    r = Rect(0, 0, 100, 80)
    masks = [Rect(10, 10, 40, 30), Rect(30, 20, 120, 50), Rect(-5, 70, 20, 90)]
    free = capture.free_rects(r, masks)
    grid = np.zeros((80, 100), int)
    for f in free:
        grid[f.top:f.bottom, f.left:f.right] += 1
    masked = np.zeros((80, 100), bool)
    for m in masks:
        masked[max(m.top, 0):min(m.bottom, 80), max(m.left, 0):min(m.right, 100)] = True
    assert (grid[masked] == 0).all() and (grid[~masked] == 1).all()   # exact cover of the free area, no overlap


def win(hwnd, z, title="App", text=(), notes=(), **kw):
    return Window(hwnd, title, kw.pop("process", "app.exe"), 1000 + hwnd, kw.pop("rect", Rect(0, 0, 800, 600)), z,
                  text=list(text), notes=list(notes), **kw)


def test_choose_skips_excluded_elevated_minimized_and_described():
    from ignite_scribe import vision_stage as S
    snap = Snapshot.now([
        win(1, 0, "Canvas", notes=["no readable content (an elevated app, or one that exposes none)"], focused=True),
        win(2, 1, "Vault", process="KeePassXC.exe"),
        win(3, 2, "Admin", notes=[S.THIN]),
        win(4, 3, "Docs", text=[TextBlock("hello", Rect(0, 0, 10, 10))]),
        win(5, 4, "Min", minimized=True),
        win(6, 5, "Empty")])
    picked, skipped = S.choose(snap, Policy(), is_elevated=lambda pid: pid == 1003)
    assert [w.hwnd for w in picked] == [1, 6]
    assert skipped[2].startswith("excluded") and skipped[3].startswith("elevated") and skipped[5] == "minimized"
    assert skipped[4].startswith("UI Automation")
    picked, _ = S.choose(snap, Policy(), match="docs", is_elevated=lambda pid: False)
    assert [w.hwnd for w in picked] == [4]                     # an explicit match looks even at a described one
    p = Policy()
    p.capture_elevated = True
    assert 3 in [w.hwnd for w in S.choose(snap, p, is_elevated=lambda pid: pid == 1003)[0]]


def test_resize_box_averages():
    from ignite_scribe.vision_stage import resize_box
    img = np.zeros((4, 4, 3), np.uint8)
    img[:2, :2] = 200
    out = resize_box(img, (2, 2))
    assert out.shape == (2, 2, 3) and out[0, 0, 0] == 200 and out[1, 1, 0] == 0
    assert resize_box(img, (8, 8)).shape == (8, 8, 3)


class FakeWorker:
    def __init__(self):
        self.calls = []

    def detect(self, tiles, conf=0.05, iou=0.45):
        self.calls.append(len(tiles))
        res = [dict(boxes=[[100, 100, 140, 130], [300, 300, 340, 330], [10, 10, 40, 40]], scores=[0.9, 0.8, 0.7])]
        res += [dict(boxes=[], scores=[]) for _ in tiles[1:]]
        return dict(ok=True, backend="npu", tiles=res, ms=[7.0] * len(tiles))

    def status(self):
        return dict(running=False)


def test_run_merges_and_drops_masked_and_uia_covered(monkeypatch):
    from ignite_scribe import capture
    from ignite_scribe import vision_stage as S
    target = win(1, 1, "Canvas", rect=Rect(1000, 500, 1640, 1140), notes=[S.THIN],
                 text=[TextBlock("Ok", Rect(1300, 800, 1340, 830), "uia", "Button")])
    above = win(2, 0, "Above", rect=Rect(990, 490, 1050, 550),
                text=[TextBlock("described", Rect(990, 490, 1050, 550))])   # UIA covers it: not chosen itself
    snap = Snapshot.now([above, target], [Rect(0, 0, 2560, 1440)])
    monkeypatch.setattr(capture, "dpi", lambda hwnd: 96)
    monkeypatch.setattr(capture, "masked_window",
                        lambda s, w, ex: (np.zeros((640, 640, 3), np.uint8), w.rect, 0.01))
    worker = FakeWorker()
    monkeypatch.setattr(capture, "elevated", lambda pid: False)
    rep = S.run(snap, Policy(), worker)
    t = next(w for w in snap.windows if w.hwnd == 1)
    # tile box (100,100)-(140,130) -> screen (1100,600)-(1140,630): kept; (300,300) -> (1300,800): covered by the
    # UIA button, dropped; (10,10)-(40,40) -> (1010,510)-(1040,540): under the window above, dropped.
    assert [(e.rect.left, e.rect.top) for e in t.elements] == [(1100, 600)]
    assert rep["windows"][0]["detected"] == 3 and rep["windows"][0]["kept"] == 1 and worker.calls == [1]


def test_mcp_find_ui_elements_over_fakes(monkeypatch):
    from ignite_scribe import capture, vision_client
    from ignite_scribe import vision_stage as S
    from ignite_scribe.mcp_server import Server
    snap = Snapshot.now([win(1, 0, "Canvas", rect=Rect(0, 0, 640, 640), notes=[S.THIN], focused=True)],
                        [Rect(0, 0, 2560, 1440)])
    monkeypatch.setattr(vision_client, "available", lambda cfg=None: (True, ""))
    monkeypatch.setattr(capture, "dpi", lambda hwnd: 96)
    monkeypatch.setattr(capture, "elevated", lambda pid: False)
    monkeypatch.setattr(capture, "masked_window", lambda s, w, ex: (np.zeros((640, 640, 3), np.uint8), w.rect, 0.0))
    s = Server(lambda text=True, max_windows=6: snap, Policy())
    s.vision = FakeWorker()
    md = s.call("find_ui_elements", {})
    assert "UI elements found by vision (3" in md and "Vision:" in md
    d = json.loads(s.call("find_ui_elements", {"format": "json"}))
    assert d["vision"]["windows"][0]["kept"] == 3 and d["windows"][0]["elements"][0]["source"] == "npu"
    assert not snap.windows[0].elements                       # the source snapshot was not written into
