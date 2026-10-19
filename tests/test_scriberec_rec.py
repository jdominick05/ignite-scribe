# Copyright (C) 2026 The ignite-scribe contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""ScribeRec's inference path offline: batching is exact, the NPU open is all-or-nothing, and the OCR pipeline
hands its cache misses to the recognizer in one call. Tiny stand-in ONNX models are built here; no NPU."""
import sys
import types

import numpy as np
import pytest

onnx = pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")
pytest.importorskip("PIL")
from onnx import TensorProto, helper, numpy_helper  # noqa: E402

from ignite_scribe import scriberec_rec as R  # noqa: E402
from ignite_scribe.ocr_pipeline import Line, OcrPipeline  # noqa: E402


def _save(graph, path):
    m = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    m.ir_version = 8
    onnx.save(m, str(path))


def tiny_model(d):
    """backbone_<W>: 1x3x80xW -> 1x3x20x(W/4) (4x4 average pool); tail: nx3x20xt -> nxtx438 log-probabilities."""
    d.mkdir()
    for w in R.BUCKETS:
        _save(helper.make_graph(
            [helper.make_node("AveragePool", ["x"], ["f"], kernel_shape=[4, 4], strides=[4, 4])], "bb",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 80, w])],
            [helper.make_tensor_value_info("f", TensorProto.FLOAT, [1, 3, 20, w // 4])]), d / f"backbone_{w}.onnx")
    wts = np.random.default_rng(3).normal(0, 40, (3, 438)).astype(np.float32)
    _save(helper.make_graph(
        [helper.make_node("ReduceMean", ["f"], ["m"], axes=[2], keepdims=0),
         helper.make_node("Transpose", ["m"], ["t"], perm=[0, 2, 1]),
         helper.make_node("MatMul", ["t", "w"], ["z"]),
         helper.make_node("LogSoftmax", ["z"], ["logprobs"], axis=2)], "tail",
        [helper.make_tensor_value_info("f", TensorProto.FLOAT, ["n", 3, 20, "t"])],
        [helper.make_tensor_value_info("logprobs", TensorProto.FLOAT, ["n", "t", 438])],
        [numpy_helper.from_array(wts, "w")]), d / "tail.onnx")
    (d / "dict.txt").write_bytes("\n".join(chr(33 + i) for i in range(436)).encode() + b"\n")
    return d


def crops(n, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        h = int(rng.integers(14, 40))
        w = int(rng.integers(20, 1100)) if rng.random() < 0.8 else int(rng.integers(700, 1300))  # some split
        out.append(rng.integers(0, 256, (h, w, 3), dtype=np.uint8))
    return out


def test_read_many_equals_one_line_at_a_time(tmp_path):
    d = tiny_model(tmp_path / "m")
    rec = R.ScribeRec(d, d / "dict.txt")
    cs = crops(24)
    assert any(c.shape[1] * 80 / c.shape[0] > 1280 for c in cs)          # the split is exercised
    one = [rec.read_many([c], batch=1)[0] for c in cs]
    many = rec.read_many(cs, batch=8)
    assert [t for t, _ in many] == [t for t, _ in one]
    assert np.allclose([s for _, s in many], [s for _, s in one], atol=1e-5)
    assert rec.backend == "cpu" and rec(cs[0])[0] == one[0][0]


def test_prepare_and_padding_follow_the_training_contract():
    x, nw = R.prepare(np.zeros((40, 101, 3), np.uint8))
    assert x.shape == (80, 202, 3) and nw == 202 and x.dtype == np.float32
    p = R.pad_to(np.arange(2 * 3 * 3, dtype=np.float32).reshape(2, 3, 3), 5)
    assert p.shape == (2, 5, 3) and (p[:, 3] == p[:, 2]).all() and (p[:, 4] == p[:, 2]).all()


class _FakeSession:
    opened, closed, fail_at = [], [], 3

    def __init__(self, path):
        if len(_FakeSession.opened) + 1 == _FakeSession.fail_at:
            raise RuntimeError("device busy")
        _FakeSession.opened.append(self)

    def close(self):
        _FakeSession.closed.append(self)


@pytest.fixture
def fake_npu(monkeypatch):
    _FakeSession.opened, _FakeSession.closed = [], []
    for name in ("ignite_xdna", "ignite_xdna.runtime"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    mod = types.ModuleType("ignite_xdna.runtime.dense_session")
    mod.DenseTensorSession = _FakeSession
    monkeypatch.setitem(sys.modules, "ignite_xdna.runtime.dense_session", mod)
    return _FakeSession


def test_a_failed_npu_open_closes_the_others_and_falls_back(tmp_path, fake_npu):
    d = tiny_model(tmp_path / "m")
    cont = {}
    for w in R.BUCKETS:
        (tmp_path / f"scriberec_{w}.ignite").write_bytes(b"")
        cont[w] = str(tmp_path / f"scriberec_{w}.ignite")
    rec = R.ScribeRec(d, d / "dict.txt", cont)
    assert len(fake_npu.opened) == 2 and fake_npu.closed == fake_npu.opened   # nothing left holding the NPU
    assert rec.backend == "cpu" and "device busy" in rec.npu_error and rec._npu == []
    fake_npu.opened, fake_npu.closed = [], []
    with pytest.raises(RuntimeError):
        R.ScribeRec(d, d / "dict.txt", cont, require_npu=True)
    assert len(fake_npu.opened) == 2 and fake_npu.closed == fake_npu.opened


def test_all_npu_buckets_open_reports_npu(tmp_path, fake_npu):
    fake_npu.fail_at = 99
    try:
        d = tiny_model(tmp_path / "m")
        cont = {}
        for w in R.BUCKETS:
            (tmp_path / f"scriberec_{w}.ignite").write_bytes(b"")
            cont[w] = str(tmp_path / f"scriberec_{w}.ignite")
        rec = R.ScribeRec(d, d / "dict.txt", cont)
        assert rec.backend == "npu" and len(fake_npu.opened) == 4
        rec.close()
        assert len(fake_npu.closed) == 4
    finally:
        fake_npu.fail_at = 3


class _FakeRec:
    backend = "npu"

    def __init__(self):
        self.calls = []

    def read_many(self, cs):
        self.calls.append([c.shape for c in cs])
        return [(f"line{c.shape[1]}", 0.99) for c in cs]

    def close(self):
        pass


def test_pipeline_batches_cache_misses_into_one_recognizer_call():
    p = OcrPipeline.__new__(OcrPipeline)
    from collections import OrderedDict
    p.recognizer, p.rec, p.min_score = _FakeRec(), None, 0.5
    p.cache, p.cache_size, p.cache_min_score = OrderedDict(), 64, 0.9
    img = np.full((100, 300, 3), 250, np.uint8)
    boxes = [(10, 10, 110, 30), (10, 40, 150, 60), (10, 70, 190, 90)]
    for i, b in enumerate(boxes):
        img[b[1] + 4:b[3] - 4, b[0] + 4:b[2] - 4 - 10 * i] = 20 + i     # distinct ink per line
    p.boxes = lambda bgr: boxes
    lines = p(img)
    assert p.recognizer.calls == [[(20, 100, 3), (20, 140, 3), (20, 180, 3)]]   # one call, in reading order
    assert [l.text for l in lines] == ["line100", "line140", "line180"] and all(isinstance(l, Line) for l in lines)
    assert p.rec_backend == "npu" and p.last["cache_hits"] == 0
    p(img)
    assert len(p.recognizer.calls) == 1 and p.last["cache_hits"] == 3          # all reused, recognizer not called


def test_pipeline_close_closes_the_recognizer():
    p = OcrPipeline.__new__(OcrPipeline)
    closed = []
    p.recognizer = types.SimpleNamespace(close=lambda: closed.append(1))
    p.close()
    assert closed == [1]


def test_greek_twins_of_latin_capitals_are_mapped_back():
    assert "Χ-ray ΟΚ Δ".translate(R.LATIN_TWINS) == "X-ray OK Δ"   # Delta has no twin


def test_split_cuts_in_the_widest_blank_run_and_spaces_only_word_gaps():
    x = np.ones((80, 1500, 3), np.float32)
    x[20:60] = 0.0                                             # ink everywhere ...
    x[:, 1100:1104] = 1.0                                      # ... but a letter gap (4 columns)
    x[:, 1200:1220] = 1.0                                      # ... and a word gap (20 columns)
    (s0, c, gap), (s1, e, last) = R.split_points(x)
    assert (s0, s1, e) == (0, c, 1500) and 1200 <= c < 1220 and gap is True and last is False
    x[:, 1200:1220] = 0.0                                      # only the letter gap is left
    (_, c, gap), _ = R.split_points(x)
    assert 1100 <= c < 1104 and gap is False                    # cut there, but join without a space
