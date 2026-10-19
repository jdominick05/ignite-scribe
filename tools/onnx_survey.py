# Copyright (C) 2026 The ignite-scribe contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""What an ONNX graph is made of, for judging it against the NPU engine's op vocabulary.

Prints the inputs and outputs, an op histogram, and every Conv's kernel / stride / group / pads with its input
and output shapes (from shape inference at ``--shape``, which fixes the dynamic input dimensions).
"""
import argparse
import collections

import onnx
from onnx import helper, shape_inference


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("onnx")
    ap.add_argument("--shape", help="fix the first input's shape, e.g. 1,3,48,320")
    ap.add_argument("--convs", action="store_true", help="list every Conv")
    a = ap.parse_args()
    m = onnx.load(a.onnx)
    g = m.graph
    if a.shape:
        dims = [int(x) for x in a.shape.split(",")]
        inp = g.input[0]
        for d, v in zip(inp.type.tensor_type.shape.dim, dims):
            d.Clear()
            d.dim_value = v
        m = shape_inference.infer_shapes(m)
        g = m.graph
    shapes = {}
    for vi in list(g.value_info) + list(g.input) + list(g.output):
        shapes[vi.name] = [d.dim_value if d.HasField("dim_value") else (d.dim_param or "?")
                           for d in vi.type.tensor_type.shape.dim]
    inits = {i.name for i in g.initializer}
    print("opset", [(o.domain or "ai.onnx", o.version) for o in m.opset_import])
    for i in g.input:
        if i.name not in inits:
            print("input ", i.name, shapes.get(i.name))
    for o in g.output:
        print("output", o.name, shapes.get(o.name))
    print("ops", dict(collections.Counter(n.op_type for n in g.node).most_common()))
    if a.convs:
        for n in g.node:
            if n.op_type != "Conv":
                continue
            at = {x.name: helper.get_attribute_value(x) for x in n.attribute}
            print(f"{n.name:40s} k{at.get('kernel_shape')} s{at.get('strides')} g{at.get('group', 1)} "
                  f"p{at.get('pads')} {shapes.get(n.input[0])} -> {shapes.get(n.output[0])}")
    for n in g.node:
        if n.op_type in ("MatMul", "Softmax", "LSTM", "GRU", "Transpose", "Reshape", "AveragePool",
                         "GlobalAveragePool", "Resize", "ReduceMean"):
            print(f"{n.op_type:18s} {n.name:40s} {[shapes.get(x) for x in n.input[:2]]} -> {shapes.get(n.output[0])}")


if __name__ == "__main__":
    main()
