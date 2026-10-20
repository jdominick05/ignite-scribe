# Copyright (C) 2026 The ignite-scribe contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The injection flag's classifier tier: a sequence classifier (SAFE / INJECTION) on ONNX Runtime.

Needs numpy, ``tokenizers`` and ``onnxruntime``, so it runs in a worker, never in the MCP server's Python (the
core stays dependency-free). The model directory holds ``model.onnx``, ``tokenizer.json`` and ``config.json``
with ``id2label``. Text longer than the model's window is scored in overlapping windows, and the highest
INJECTION probability counts, since an injection planted at the end of an email must not be truncated away.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np


class InjectionClassifier:
    def __init__(self, model_dir: str | Path, threads: int = 4, max_len: int = 512, stride: int = 256,
                 providers=("CPUExecutionProvider",)):
        import onnxruntime as ort
        from tokenizers import Tokenizer
        d = Path(model_dir)
        cfg = json.loads((d / "config.json").read_text(encoding="utf-8"))
        labels = {int(k): v.upper() for k, v in cfg["id2label"].items()}
        self.pos = next(k for k, v in labels.items() if "INJECT" in v)
        self.tok = Tokenizer.from_file(str(d / "tokenizer.json"))
        self.tok.no_padding()
        self.tok.no_truncation()
        o = ort.SessionOptions()
        o.intra_op_num_threads = threads
        o.log_severity_level = 3
        self.sess = ort.InferenceSession(str(d / "model.onnx"), o, providers=list(providers))
        self.inputs = {i.name for i in self.sess.get_inputs()}
        self.max_len, self.stride = max_len, stride
        self.cls = self.tok.token_to_id("[CLS]")
        self.sep = self.tok.token_to_id("[SEP]")
        self.ms_total, self.windows_total = 0.0, 0

    def _windows(self, text: str) -> list[list[int]]:
        ids = self.tok.encode(text, add_special_tokens=False).ids
        body = self.max_len - 2
        if len(ids) <= body:
            return [[self.cls] + ids + [self.sep]]
        out, s = [], 0
        while True:
            out.append([self.cls] + ids[s:s + body] + [self.sep])
            if s + body >= len(ids):
                return out
            s += self.stride

    def score(self, texts: list[str], batch: int = 8) -> list[float]:
        """-> the INJECTION probability of each text (its highest window)."""
        wins, owner = [], []
        for i, t in enumerate(texts):
            for w in self._windows(t):
                wins.append(w)
                owner.append(i)
        best = [0.0] * len(texts)
        t0 = time.perf_counter()
        order = sorted(range(len(wins)), key=lambda j: len(wins[j]))      # similar lengths pad less
        for k in range(0, len(order), batch):
            grp = order[k:k + batch]
            n = max(len(wins[j]) for j in grp)
            ids = np.zeros((len(grp), n), np.int64)
            mask = np.zeros((len(grp), n), np.int64)
            for r, j in enumerate(grp):
                ids[r, :len(wins[j])] = wins[j]
                mask[r, :len(wins[j])] = 1
            feed = {"input_ids": ids, "attention_mask": mask}
            if "token_type_ids" in self.inputs:
                feed["token_type_ids"] = np.zeros_like(ids)
            logits = self.sess.run(None, {k: v for k, v in feed.items() if k in self.inputs})[0]
            z = logits - logits.max(axis=1, keepdims=True)
            p = np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)
            for r, j in enumerate(grp):
                best[owner[j]] = max(best[owner[j]], float(p[r, self.pos]))
        self.ms_total += (time.perf_counter() - t0) * 1000
        self.windows_total += len(wins)
        return best
