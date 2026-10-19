# Copyright (C) 2026 The ignite-scribe contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Simulate a W8A16 datapath on a float ONNX: every node output rounded to bfloat16, weights optionally int8.

Inserts Cast(bfloat16) -> Cast(float) after every node output (round-to-nearest-even, as the casts do), and with
``--weights int8-tensor|int8-channel`` rounds each Conv weight to int8 at a power-of-two scale first (see
fakequant_weights.py). It approximates the engine's bf16 activations; it is not its exact accumulate model
(engine_bf16_emulator), so it says whether the route is worth asking for, not what it will measure.
"""
import argparse

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from fakequant_weights import pow2_scale, quant


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--weights", choices=["float", "bf16", "int8-tensor", "int8-channel"], default="bf16")
    a = ap.parse_args()
    m = onnx.load(a.src)
    g = m.graph
    init = {i.name: i for i in g.initializer}
    for n in g.node:
        if n.op_type == "Conv" and n.input[1] in init:
            w = numpy_helper.to_array(init[n.input[1]]).astype(np.float64)
            if a.weights == "bf16":
                w = (np.asarray(w, np.float32).view(np.uint32) + 0x7FFF + ((np.asarray(w, np.float32).view(np.uint32) >> 16) & 1)) & 0xFFFF0000
                w = w.astype(np.uint32).view(np.float32)
            elif a.weights.startswith("int8"):
                ch = np.abs(w).reshape(w.shape[0], -1).max(1)
                s = pow2_scale(ch)[:, None, None, None] if a.weights == "int8-channel" else pow2_scale(ch.max())
                w = quant(w, s)
            init[n.input[1]].CopyFrom(numpy_helper.from_array(np.asarray(w, np.float32), n.input[1]))
    nodes = []
    for n in g.node:
        nodes.append(n)
        if n.op_type in ("Constant", "Shape"):
            continue
        for i, o in enumerate(n.output):
            r = o + "__f32in"
            n.output[i] = r
            nodes.append(helper.make_node("Cast", [r], [o + "__bf"], to=TensorProto.BFLOAT16))
            nodes.append(helper.make_node("Cast", [o + "__bf"], [o], to=TensorProto.FLOAT))
    del g.node[:]
    g.node.extend(nodes)
    del g.value_info[:]
    onnx.save(m, a.out)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
