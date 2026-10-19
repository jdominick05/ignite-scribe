# Copyright (C) 2026 The ignite-scribe contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The detector worker: one process that owns the NPU hardware context, fed tiles over stdin.

Runs under the Python ignite-xdna's runtime needs (3.13: PyXRT is built for it), with ignite-xdna's ``src`` and
this repo's ``src`` on ``PYTHONPATH``. ``vision_client.VisionWorker`` starts it on demand and stops it when idle.

Protocol, both directions: a 4-byte little-endian length, then a payload of one JSON header line (UTF-8,
ending in a newline) and optional raw bytes.
- request ``{"op": "detect", "n": k, "conf": c, "iou": i}`` + k x 640 x 640 x 3 uint8 RGB tiles;
  reply ``{"ok": true, "backend": ..., "tiles": [{"boxes": [[x0, y0, x1, y1], ...], "scores": [...]}], "ms": [...]}``
  with boxes in tile pixels, NMS applied per tile;
- request ``{"op": "ocr", "shapes": [[h, w], ...]}`` + the images' h x w x 3 uint8 RGB bytes, concatenated;
  reply ``{"ok": true, "ocr_backend": "cpu", "images": [{"lines": [{"box": [x0, y0, x1, y1], "text": ..., "score":
  ...}], "det_ms": ..., "rec_ms": ...}]}``, boxes in image pixels (``ocr_pipeline``). The OCR models load on the
  first such request. Both models run on the CPU: the recognizer needs the engine's bf16 datapath (docs/OCR.md);
- ``{"op": "ping"}`` -> ``{"ok": true, "backend": ..., "reason": ...}``;
- ``{"op": "close"}``, or EOF on stdin -> the session is closed (the NPU context released) and the process exits.

Backends: ``npu`` (the ``.ignite`` container through ``ignite_xdna.InferenceSession``) or, when that cannot
open, ``cpu`` (ONNX Runtime on the same quantized ONNX); ``--backend`` forces one. Heads come back as int8 and
are dequantized with the container's own scales; ``tests`` check the NPU heads against ONNX Runtime.
stdout carries only protocol frames; diagnostics go to stderr.
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
import time

import numpy as np

from .vision import decode_heads, nms

SIZE = 640


def read_frame(f) -> tuple[dict, bytes] | None:
    n = f.read(4)
    if len(n) < 4:
        return None
    (size,) = struct.unpack("<I", n)
    body = f.read(size)
    i = body.index(b"\n")
    return json.loads(body[:i].decode("utf-8")), body[i + 1:]


def write_frame(f, header: dict, raw: bytes = b"") -> None:
    body = json.dumps(header).encode("utf-8") + b"\n" + raw
    f.write(struct.pack("<I", len(body)) + body)
    f.flush()


class NpuBackend:
    name = "npu"

    def __init__(self, container: str):
        from ignite_xdna import InferenceSession
        self.s = InferenceSession.from_file(container)
        names = list(self.s.head_names)
        self.box = [n for n in sorted(names) if n.endswith("_box")]
        self.cls = [n for n in sorted(names) if n.endswith("_cls")]
        assert len(self.box) == 3 and len(self.cls) == 3, names

    def heads(self, rgb: np.ndarray):
        x = (rgb.astype(np.int16) - 128).astype(np.int8).transpose(2, 0, 1)[None]
        out = self.s.run_yolo_monolithic(x, unswizzle=True)
        if not out.get("heads_present"):
            raise RuntimeError(f"no heads: {out.get('head_status')}")
        sc = out["scales"]

        def deq(n):
            s, zp = sc[n]
            return (out[n].astype(np.float32) - zp) * s
        return [deq(n) for n in self.box], [deq(n) for n in self.cls], self.s.last_dispatch_ms

    def close(self):
        self.s.close()


class CpuBackend:
    name = "cpu"

    def __init__(self, onnx_path: str):
        import onnxruntime as ort
        o = ort.SessionOptions()
        o.intra_op_num_threads = 4
        self.s = ort.InferenceSession(onnx_path, o, providers=["CPUExecutionProvider"])
        self.inp = self.s.get_inputs()[0].name
        outs = [x.name for x in self.s.get_outputs()]
        self.box = [o for o in outs if "/cv2." in o]
        self.cls = [o for o in outs if "/cv3." in o]

    def heads(self, rgb: np.ndarray):
        t = time.perf_counter()
        r = dict(zip([x.name for x in self.s.get_outputs()],
                     self.s.run(None, {self.inp: (rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]})))
        return [r[n][0] for n in self.box], [r[n][0] for n in self.cls], (time.perf_counter() - t) * 1000

    def close(self):
        pass


def open_backend(kind: str, container: str, onnx_path: str):
    if kind in ("auto", "npu"):
        try:
            return NpuBackend(container), None
        except Exception as e:  # noqa: BLE001 - reported in ping / every reply
            if kind == "npu":
                raise
            reason = f"npu unavailable: {type(e).__name__}: {e}"
            print(reason, file=sys.stderr, flush=True)
            return CpuBackend(onnx_path), reason
    return CpuBackend(onnx_path), "cpu requested"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--container", required=True)
    ap.add_argument("--onnx", required=True, help="the quantized ONNX the container was built from (CPU fallback)")
    ap.add_argument("--backend", choices=["auto", "npu", "cpu"], default="auto")
    ap.add_argument("--ocr-det", help="text detector ONNX (ppocrv5_mobile_det)")
    ap.add_argument("--ocr-rec", help="text-line recognizer ONNX (en_ppocrv5_rec, dynamic width)")
    ap.add_argument("--ocr-dict", help="the recognizer's character list")
    a = ap.parse_args(argv)
    stdin, stdout = sys.stdin.buffer, sys.stdout.buffer
    sys.stdout = sys.stderr                      # nothing but frames may reach the real stdout
    backend, reason = open_backend(a.backend, a.container, a.onnx)
    ocr_pipe = None
    try:
        while True:
            msg = read_frame(stdin)
            if msg is None:
                break
            h, raw = msg
            op = h.get("op")
            if op == "close":
                break
            if op == "ping":
                write_frame(stdout, dict(ok=True, backend=backend.name, reason=reason))
                continue
            if op == "ocr":
                try:
                    if ocr_pipe is None:
                        if not (a.ocr_det and a.ocr_rec and a.ocr_dict):
                            raise RuntimeError("the worker was started without OCR models")
                        from .ocr_pipeline import OcrPipeline
                        ocr_pipe = OcrPipeline(a.ocr_det, a.ocr_rec, a.ocr_dict)
                    out, off = [], 0
                    for hh, ww in h["shapes"]:
                        n = int(hh) * int(ww) * 3
                        img = np.frombuffer(raw, np.uint8, n, off).reshape(int(hh), int(ww), 3)
                        off += n
                        lines = ocr_pipe(img)
                        out.append(dict(lines=[dict(box=list(map(int, l.box)), text=l.text, score=round(l.score, 4))
                                               for l in lines], **ocr_pipe.last))
                    write_frame(stdout, dict(ok=True, ocr_backend="cpu", images=out))
                except Exception as e:  # noqa: BLE001
                    write_frame(stdout, dict(ok=False, error=f"{type(e).__name__}: {e}"))
                continue
            if op != "detect":
                write_frame(stdout, dict(ok=False, error=f"unknown op {op}"))
                continue
            try:
                k = int(h["n"])
                tiles = np.frombuffer(raw, np.uint8).reshape(k, SIZE, SIZE, 3)
                res, ms = [], []
                for t in tiles:
                    boxes, classes, dt = backend.heads(t)
                    b, s = decode_heads(boxes, classes, conf=float(h.get("conf", 0.05)))
                    keep = nms(b, s, float(h.get("iou", 0.45)))
                    res.append(dict(boxes=b[keep].round(1).tolist(), scores=s[keep].round(4).tolist()))
                    ms.append(round(float(dt), 3))
                write_frame(stdout, dict(ok=True, backend=backend.name, tiles=res, ms=ms))
            except Exception as e:  # noqa: BLE001 - one bad request must not kill the worker
                write_frame(stdout, dict(ok=False, error=f"{type(e).__name__}: {e}"))
    finally:
        backend.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
