# Copyright (C) 2026 The ignite-scribe contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Run a segment-task .ignite container on the NPU and compare its output map with the engine's integer reference.

ignite-xdna's tools/verify_engine_container.py does not open segment containers, and a backbone cut (one raw
map out) compiles only as one. This checks the final map, not every layer:
- **reference:** ``graph_reference.run_direct`` on the lowered IR of the same QDQ model (skipped for a container
  with host regions, which run_direct does not execute);
- **ONNX Runtime:** the same QDQ model on the CPU, for context. It is not the exactness bar, since the engine's
  integer epilogues may legitimately differ from ORT's float-between-QDQ evaluation.
Differences are in output quantization steps. The input is an image resized to the model input, RGB / 255, as
verify_engine_container feeds a non-SR container. Run with ignite-xdna's research-iron.sh (PyXRT, Python 3.13).
"""
import argparse
import json
import sys
import time

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--container", required=True)
    ap.add_argument("--model", required=True, help="the QDQ ONNX the container was compiled from")
    ap.add_argument("--image", required=True)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--json")
    a = ap.parse_args()
    import cv2
    import onnxruntime as ort
    from ignite_xdna.compiler import graph_ir
    from ignite_xdna.compiler import graph_reference as gr
    from ignite_xdna.compiler.serializer import IgniteModelReader
    from ignite_xdna.runtime.dense_session import DenseTensorSession

    with IgniteModelReader(a.container) as r:
        man = r.manifest
    ge = man["graph_engine"]
    hosts = [s["name"] for s in ge.get("segments", []) if s["kind"] == "host"]
    n, c, h, w = man["input_shape"]
    img = cv2.resize(cv2.imread(a.image), (w, h), interpolation=cv2.INTER_LINEAR)
    x = np.transpose(cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0, (2, 0, 1))[None]
    out = man["dense_output"]
    res = dict(container=a.container.replace("\\", "/").split("/")[-1], input=[n, c, h, w], host_segments=hosts)

    s = ort.InferenceSession(a.model, providers=["CPUExecutionProvider"])
    y_ort = s.run(None, {s.get_inputs()[0].name: x})[0][0]

    ref = None
    if not hosts:
        ir = graph_ir.lower_yolov8n(a.model)
        t_in = ir.tensors[ir.input]
        q = gr.quantize_input(x[0], t_in.scale, t_in.zero_point)
        tensors = gr.run_direct(ir, q)
        tq = tensors[out["tensor"]].astype(np.int32)
        ref = (tq[:out["channels"]] - out["zero_point"]) * out["scale"]

    with DenseTensorSession(a.container) as sess:
        y, st = sess.run(x)
        ms = []
        for _ in range(a.iters):
            t0 = time.perf_counter()
            sess.run(x)
            ms.append(sess.last_dispatch_ms)
    y = np.asarray(y)[0] if np.asarray(y).ndim == 4 else np.asarray(y)
    step = out.get("scale") or 1.0
    if ref is not None:
        d = np.abs(y - ref) / step
        res.update(ref_worst_steps=float(d.max()), ref_exact_fraction=float((d < 0.5).mean()))
    d = np.abs(y - y_ort) / step
    res.update(ort_worst_steps=float(d.max()), ort_mean_steps=float(d.mean()),
               npu_ms_median=float(np.median(ms)), npu_ms_min=float(min(ms)), shape=list(y.shape))
    res["verdict"] = ("EXACT" if ref is not None and res["ref_worst_steps"] == 0 else
                      "MISMATCH" if ref is not None else "no reference (host segments)")
    print(json.dumps(res))
    if a.json:
        open(a.json, "w").write(json.dumps(res, indent=1))
    return 0 if res["verdict"] != "MISMATCH" else 1


if __name__ == "__main__":
    sys.exit(main())
