# Copyright (C) 2026 The ignite-scribe contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Print a stretch of an ONNX graph in topological order: op, name, inputs (initializers shown as their shape or
scalar value) and consumers. For reading a block's pattern before deciding how to rewrite or cut it."""
import argparse

import onnx
from onnx import numpy_helper


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("onnx")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--count", type=int, default=40)
    ap.add_argument("--skip", default="Constant,QuantizeLinear,DequantizeLinear", help="ops to hide")
    a = ap.parse_args()
    g = onnx.load(a.onnx).graph
    init = {i.name: numpy_helper.to_array(i) for i in g.initializer}
    for n in g.node:
        if n.op_type == "Constant":
            init[n.output[0]] = numpy_helper.to_array(n.attribute[0].t)
    cons = {}
    for n in g.node:
        for x in n.input:
            cons.setdefault(x, []).append(n.op_type)
    skip = set(a.skip.split(","))
    nodes = [n for n in g.node if n.op_type not in skip]
    for n in nodes[a.start:a.start + a.count]:
        ins = []
        for x in n.input:
            if x in init:
                v = init[x]
                ins.append(f"{v.item():.4g}" if v.size == 1 else f"W{list(v.shape)}")
            else:
                ins.append(x[-18:])
        print(f"{n.op_type:12s} {n.name[-34:]:34s} in={ins} -> {cons.get(n.output[0], ['OUT'])}")


if __name__ == "__main__":
    main()
