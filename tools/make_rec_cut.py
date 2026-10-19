# Copyright (C) 2026 The ignite-scribe contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Fix a PaddleOCR text-line recognizer to a static shape and cut it where the engine's vocabulary must stop.

Writes ``<out>_static.onnx`` (the whole model, input fixed to 1x3xHxW, shapes inferred) and ``<out>_cut.onnx``,
which ends at ``--cut-before``'s input: the convolutional backbone, without the pool, the SVTR neck (attention)
and the CTC head, which stay on the CPU. Run in an env with ``onnx`` (mlir-aie-iron or resnet_env).
"""
import argparse

import onnx
from onnx import shape_inference, utils


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", required=True, help="output prefix")
    ap.add_argument("--height", type=int, default=48)
    ap.add_argument("--width", type=int, default=320)
    ap.add_argument("--cut-before", default="AveragePool.0", help="node whose input becomes the cut's output")
    a = ap.parse_args()
    m = onnx.load(a.src)
    inp = m.graph.input[0]
    for d, v in zip(inp.type.tensor_type.shape.dim, (1, 3, a.height, a.width)):
        d.Clear()
        d.dim_value = v
    m = shape_inference.infer_shapes(m)
    static = f"{a.out}_static.onnx"
    onnx.save(m, static)
    node = next(n for n in m.graph.node if n.name == a.cut_before)
    cut = f"{a.out}_cut.onnx"
    utils.extract_model(static, cut, [inp.name], [node.input[0]])
    c = onnx.load(cut)
    print("static", static, "| cut", cut, "->", node.input[0],
          [d.dim_value for d in c.graph.output[0].type.tensor_type.shape.dim], len(c.graph.node), "nodes")


if __name__ == "__main__":
    main()
