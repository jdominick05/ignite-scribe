# Copyright (C) 2026 The ignite-scribe contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The embedding worker: one process that owns the iGPU session, fed over stdin, at below-normal priority.

``semantic.EmbedWorker`` starts it on demand (the same interpreter as the server, with the ``embed`` extra) and
stops it when idle. Frames are those of ``npu_worker``: a 4-byte little-endian length, a JSON line, raw bytes.
- ``{"op": "ping"}`` -> backend (``dml`` or ``cpu``; none until the first embed opens the GPU session) and why;
- ``{"op": "embed", "texts": [...]}`` -> raw float16 [n, 384] passage vectors;
- ``{"op": "search", "query": str, "n": N, "k": K}`` + raw float16 [N, 384] -> the K best rows and scores;
- EOF on stdin (the parent closed it or died) -> the process exits, releasing the GPU.
Passages are embedded on DirectML in fp16; queries on the CPU in fp32 (faster for one short text).
"""
from __future__ import annotations

import ctypes
import sys
import traceback

import numpy as np

from .embed import DIM, Embedder, model_dir
from .npu_worker import read_frame, write_frame

BELOW_NORMAL_PRIORITY_CLASS = 0x4000


def main(argv=None) -> int:
    try:
        k32 = ctypes.WinDLL("kernel32")
        k32.SetPriorityClass(k32.GetCurrentProcess(), BELOW_NORMAL_PRIORITY_CLASS)   # indexing yields to the user
    except Exception:  # noqa: BLE001
        pass
    d = model_dir()
    f16 = (d / "model_fp16.onnx").is_file()
    passages = queries = None              # each opened on first use: a session that only searches never opens the GPU
    stdin, stdout = sys.stdin.buffer, sys.stdout.buffer

    def device():
        if passages is None:
            return dict(backend=None, reason="the GPU session opens on the first embed", precision=None)
        return dict(backend=passages.backend, reason=passages.reason,
                    precision="fp16" if f16 and passages.backend == "dml" else "fp32")

    while True:
        msg = read_frame(stdin)
        if msg is None:
            return 0
        h, raw = msg
        try:
            op = h.get("op")
            if op == "ping":
                write_frame(stdout, dict(ok=True, **device()))
            elif op == "embed":
                if passages is None:
                    passages = Embedder(d, device="dml", model_file="model_fp16.onnx" if f16 else "model.onnx")
                v = passages.encode(h["texts"]).astype(np.float16)
                write_frame(stdout, dict(ok=True, n=len(v), dim=DIM, **device()), v.tobytes())
            elif op == "search":
                if queries is None:
                    queries = Embedder(d, device="cpu", threads=4)
                q = queries.encode([h["query"]], query=True)[0]
                m = np.frombuffer(raw, np.float16).reshape(int(h["n"]), DIM).astype(np.float32)
                s = m @ q
                k = min(int(h.get("k", 10)), len(s))
                top = np.argpartition(-s, k - 1)[:k] if k else np.array([], int)
                top = top[np.argsort(-s[top])]
                write_frame(stdout, dict(ok=True, idx=top.tolist(), score=[round(float(x), 4) for x in s[top]]))
            else:
                write_frame(stdout, dict(ok=False, error=f"unknown op {op}"))
        except Exception as e:  # noqa: BLE001
            traceback.print_exc(file=sys.stderr)
            write_frame(stdout, dict(ok=False, error=f"{type(e).__name__}: {e}"))


if __name__ == "__main__":
    sys.exit(main())
