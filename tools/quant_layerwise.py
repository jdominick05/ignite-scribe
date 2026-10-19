# Copyright (C) 2026 The ignite-scribe contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Where does a QDQ model part from its float source? Cosine and range of every shared node output, in order.

Exposes every node output whose name exists in both graphs (Quark keeps node and tensor names) as a graph output,
runs both on the same inputs (``--image`` line crops through ``ocr.line_input``) and prints, per Conv and
activation, the float range and the cosine between float and QDQ.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ignite_scribe import ocr  # noqa: E402


def expose(path, names):
    m = onnx.load(path)
    have = {o.name for o in m.graph.output}
    for n in names:
        if n not in have:
            m.graph.output.append(onnx.helper.make_empty_tensor_value_info(n))
    o = ort.SessionOptions()
    o.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(m.SerializeToString(), o, providers=["CPUExecutionProvider"])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--float", dest="fl", required=True)
    ap.add_argument("--qdq", required=True)
    ap.add_argument("--image", nargs="+", required=True)
    ap.add_argument("--height", type=int, default=48)
    ap.add_argument("--width", type=int, default=320)
    ap.add_argument("--ops", default="Conv,HardSwish,Relu,HardSigmoid,Mul,Add,GlobalAveragePool")
    a = ap.parse_args()
    fm, qm = onnx.load(a.fl), onnx.load(a.qdq)
    ops = set(a.ops.split(","))
    qouts = {o for n in qm.graph.node for o in n.output}
    order = [(n.op_type, n.output[0]) for n in fm.graph.node if n.op_type in ops and n.output[0] in qouts]
    names = [t for _, t in order]
    fs, qs = expose(a.fl, names), expose(a.qdq, names)
    acc = {t: [[], []] for t in names}
    for f in a.image:
        x, _ = ocr.line_input(np.asarray(Image.open(f).convert("RGB")), a.height, a.width)
        fo = dict(zip([o.name for o in fs.get_outputs()], fs.run(None, {fs.get_inputs()[0].name: x})))
        qo = dict(zip([o.name for o in qs.get_outputs()], qs.run(None, {qs.get_inputs()[0].name: x})))
        for t in names:
            acc[t][0].append(fo[t].ravel())
            acc[t][1].append(qo[t].ravel())
    for op, t in order:
        u, v = np.concatenate(acc[t][0]), np.concatenate(acc[t][1])
        cos = float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12))
        print(f"{op:18s} {t[-28:]:28s} float [{u.min():9.3f}, {u.max():9.3f}]  |p99.9| {np.percentile(np.abs(u), 99.9):8.3f}  cos {cos:.4f}")


if __name__ == "__main__":
    main()
