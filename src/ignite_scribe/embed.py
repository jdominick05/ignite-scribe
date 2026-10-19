# Copyright (C) 2026 The ignite-scribe contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Text embeddings for the history: bge-small-en-v1.5 (MIT, 384-d) through ONNX Runtime, on the iGPU.

Runs only inside ``embed_worker`` (a subprocess), never in the MCP server: it needs numpy, ``tokenizers`` and
``onnxruntime-directml`` (the ``embed`` extra), and a GPU driver stall must not block the server.
- **Device:** DirectML (the Radeon 780M on Desktop 2), falling back to the CPU when DirectML cannot start;
  ``backend`` says which, and ``reason`` why.
- **Model:** CLS pooling of ``last_hidden_state``, L2-normalized, as the model card specifies. Queries get the
  model's retrieval instruction; passages do not.
- **Precision:** passages run in fp16 on DirectML (``model_fp16.onnx``, made by ``tools/make_embed_fp16.py``),
  queries in fp32 on the CPU. Measured on the 780M at batch 32: 38 passages/s in fp16 against 19 in fp32 (8 CPU
  threads: 39), cosine 0.99999 to the CPU's vectors; a single query takes 5.6 ms on the CPU and 48 ms on the
  780M. Larger fp16 batches were slower (11-12/s at 64-128).
- **Files:** ``models/embed/bge-small-en-v1.5/`` (``model.onnx``, ``tokenizer.json``; git-ignored, see the
  README). ``IGNITE_SCRIBE_EMBED_DIR`` overrides it.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .semantic import DIM, MODEL, model_dir  # noqa: F401 - one place for the model's name and files

QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
MAX_TOKENS = 256


class Embedder:
    def __init__(self, d: Path | None = None, device: str = "dml", threads: int = 2, model_file: str = "model.onnx"):
        import onnxruntime as ort
        from tokenizers import Tokenizer
        d = d or model_dir()
        self.tok = Tokenizer.from_file(str(d / "tokenizer.json"))
        self.tok.enable_truncation(MAX_TOKENS)
        self.tok.no_padding()
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.log_severity_level = 3
        self.backend, self.reason = "cpu", None
        self.sess = None
        if device == "dml" and "DmlExecutionProvider" in ort.get_available_providers():
            try:
                dml = ort.SessionOptions()
                dml.log_severity_level = 3
                dml.enable_mem_pattern = False             # DirectML requires both of these
                dml.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
                self.sess = ort.InferenceSession(str(d / model_file), dml, providers=["DmlExecutionProvider"])
                self.backend = "dml"
            except Exception as e:  # noqa: BLE001 - fall back to the CPU
                self.reason = f"DirectML failed: {type(e).__name__}: {e}"
        elif device == "dml":
            self.reason = "DirectML provider not available"
        if self.sess is None:
            self.sess = ort.InferenceSession(str(d / model_file), so, providers=["CPUExecutionProvider"])

    def encode(self, texts: list[str], query: bool = False, batch: int = 32) -> np.ndarray:
        """-> float32 [len(texts), 384], unit length. Batches are length-sorted, so padding stays small."""
        if query:
            texts = [QUERY_PREFIX + t for t in texts]
        enc = self.tok.encode_batch(texts)
        order = sorted(range(len(enc)), key=lambda i: len(enc[i].ids))
        out = np.zeros((len(texts), DIM), np.float32)
        for s in range(0, len(order), batch):
            idx = order[s:s + batch]
            n = max(len(enc[i].ids) for i in idx)
            ids = np.zeros((len(idx), n), np.int64)
            mask = np.zeros((len(idx), n), np.int64)
            for r, i in enumerate(idx):
                e = enc[i].ids
                ids[r, :len(e)] = e
                mask[r, :len(e)] = 1
            h = self.sess.run(None, {"input_ids": ids, "attention_mask": mask,
                                     "token_type_ids": np.zeros_like(ids)})[0]
            cls = h[:, 0, :]
            out[idx] = cls / np.maximum(np.linalg.norm(cls, axis=1, keepdims=True), 1e-12)
        return out
