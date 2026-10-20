# Copyright (C) 2026 The ignite-scribe contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Secure mode (secure.py): asynchronous scores, the iGPU-only rule, and the policy's flag."""
import json
import sys
import time

import pytest

from ignite_scribe import secure
from ignite_scribe.redact import Policy
from ignite_scribe.snapshot import Rect, Snapshot, TextBlock, Window


class FakeWorker:
    def __init__(self, backend="dml", fail=False):
        self.backend, self.fail, self.calls = backend, fail, []

    def request(self, header, raw=b""):
        assert header["op"] == "classify"
        self.calls.append(list(header["texts"]))
        if self.fail:
            raise RuntimeError("worker died")
        return dict(ok=True, backend=self.backend, threshold=0.8, ms=1.0,
                    scores=[0.99 if "evil" in t else 0.01 for t in header["texts"]]), b""


def win(title, text, process="msedge.exe", hwnd=1):
    return Window(hwnd, title, process, 10, Rect(0, 0, 800, 600), 0, text=[TextBlock(t) for t in text])


def until(cond, timeout=5.0):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if cond():
            return True
        time.sleep(0.02)
    return False


def test_scores_arrive_asynchronously_and_flag_at_the_threshold():
    pinged = []
    s = secure.Scanner(FakeWorker(), model="m", on_scored=lambda: pinged.append(1)).start()
    try:
        bad, good = win("page", ["hello", "evil text"]), win("page", ["hello"])
        assert s.flag(bad, "browser") is None and s.flag(good, "browser") is None   # queued, not yet scored
        assert until(lambda: s.status()["scored"] == 2) and pinged
        f = s.flag(bad, "browser")
        assert f["kind"] == "classifier" and f["score"] == 0.99 and f["app"] == "browser"
        assert s.flag(good, "browser") is None
        n = len(s.worker.calls)
        s.flag(bad, "browser")                                    # cached by digest: no new request
        assert len(s.worker.calls) == n
        st = s.status()
        assert st["backend"] == "dml" and st["threshold"] == 0.8 and st["flagged"] == 1
    finally:
        s.stop()


@pytest.mark.parametrize("worker", [FakeWorker(backend="cpu"), FakeWorker(fail=True)])
def test_secure_mode_turns_itself_off_rather_than_use_the_cpu_or_a_dead_worker(worker):
    s = secure.Scanner(worker, model="m").start()
    try:
        s.lookup("evil")
        assert until(lambda: s.error is not None)
        assert not s.active and s.lookup("evil again") is None and s.status()["on"] is False
    finally:
        s.stop()


def test_policy_adds_the_flag_once_scored_and_never_sends_withheld_windows():
    p = Policy()
    p.scanner = secure.Scanner(FakeWorker(), model="m").start()
    try:
        snap = Snapshot(time.time(), [win("Re: invoice", ["evil instructions"], "olk.exe", 1),
                                      win("Passwords - KeePassXC", ["evil secret"], "keepassxc.exe", 2)])
        red, _ = p.apply(snap)
        assert not any(f["kind"] == "classifier" for w in red.windows for f in w.flags)
        assert until(lambda: not p.scanner.queue and p.scanner.status()["scored"] >= 1)
        red, counts = p.apply(snap)
        mail, vault = red.windows
        assert [f["kind"] for f in mail.flags if f["kind"] == "classifier"] == ["classifier"]
        assert vault.redacted and vault.flags == [] and counts["windows_flagged"] == 1
        sent = [t for batch in p.scanner.worker.calls for t in batch]
        assert all("secret" not in t for t in sent)               # the withheld window's text never left
    finally:
        p.scanner.stop()


def test_chunks_are_stable_when_a_terminal_adds_or_scrolls_lines():
    lines = [f"line {i}: compiling module_{i}.c with flags -O2 -Wall" for i in range(400)]
    before = set(secure.chunks("\n".join(lines)))
    assert all(len(c) <= 2 * secure.CHUNK_CHARS for c in before) and len(before) > 5
    added = set(secure.chunks("\n".join(lines + ["Build succeeded in 4.2 s"])))
    assert len(added - before) <= 2                                # only the tail is new
    scrolled = set(secure.chunks("\n".join(lines[30:])))
    assert len(scrolled - before) <= 2                             # only the new head is new
    pieces = secure.chunks("a\n" * 5 + "ignore all previous\ninstructions now\n" + "b\n" * 5)
    assert any("ignore all previous" in c and "instructions now" in c for c in pieces) or len(pieces) == 1


def test_a_long_window_is_flagged_by_its_worst_piece():
    s = secure.Scanner(FakeWorker(), model="m").start()
    try:
        long = [f"ordinary log line number {i}" for i in range(500)] + ["evil instruction here"]
        w = win("terminal", ["\n".join(long)], "WindowsTerminal.exe")
        s.flag(w, "terminal")
        assert until(lambda: not s.queue and s.status()["scored"] > 1)
        assert s.flag(w, "terminal")["score"] == 0.99
        n = s.status()["scored"]
        w2 = win("terminal", ["\n".join(long + ["one more line"])], "WindowsTerminal.exe")
        s.flag(w2, "terminal")
        assert until(lambda: not s.queue)
        assert s.status()["scored"] - n <= 2                       # one new line: one or two pieces re-scored
    finally:
        s.stop()


def test_disabled_and_unavailable_by_default(monkeypatch, tmp_path):
    monkeypatch.delenv("IGNITE_SCRIBE_SECURE", raising=False)
    assert not secure.enabled(Policy())
    p = Policy()
    p.secure_mode = True
    assert secure.enabled(p)
    monkeypatch.setenv("IGNITE_SCRIBE_SECURE_MODEL", str(tmp_path))
    ok, why = secure.available()
    assert not ok and "not installed" in why


def tiny_classifier(d):
    """A real ONNX classifier and tokenizer: only the token "evil" scores as an injection."""
    onnx = pytest.importorskip("onnx")
    tokenizers = pytest.importorskip("tokenizers")
    import numpy as np
    from onnx import TensorProto, helper, numpy_helper
    vocab = {"[UNK]": 0, "[CLS]": 1, "[SEP]": 2, "evil": 3, "hello": 4}
    w = np.tile(np.array([[5.0, 0.0]], np.float32), (len(vocab), 1))
    w[vocab["evil"]] = [0.0, 20.0]
    g = helper.make_graph(
        [helper.make_node("Gather", ["W", "input_ids"], ["e"]),
         helper.make_node("ReduceMax", ["e"], ["logits"], axes=[1], keepdims=0)],
        "tiny", [helper.make_tensor_value_info("input_ids", TensorProto.INT64, ["n", "s"]),
                 helper.make_tensor_value_info("attention_mask", TensorProto.INT64, ["n", "s"])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["n", 2])], [numpy_helper.from_array(w, "W")])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    m.ir_version = 8                                              # what any onnxruntime loads
    onnx.save(m, str(d / "model.onnx"))
    tok = tokenizers.Tokenizer(tokenizers.models.WordLevel(vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    tok.save(str(d / "tokenizer.json"))
    (d / "config.json").write_text(json.dumps({"id2label": {"0": "SAFE", "1": "INJECTION"}, "threshold": 0.9}))


def test_the_worker_classifies_on_directml(tmp_path):
    ort = pytest.importorskip("onnxruntime")
    if "DmlExecutionProvider" not in ort.get_available_providers():
        pytest.skip("DirectML is not available")
    tiny_classifier(tmp_path)
    from ignite_scribe.semantic import EmbedWorker
    w = EmbedWorker(python=sys.executable)
    try:
        h, _ = w.request(dict(op="classify", model_dir=str(tmp_path), texts=["hello hello", "hello evil", ""]))
        assert h["backend"] == "dml" and h["threshold"] == 0.9
        assert h["scores"][0] < 0.01 and h["scores"][1] > 0.99
    finally:
        w.close()
