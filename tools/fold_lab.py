# Copyright (C) 2026 The ignite-scribe contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Fold PP-LCNetV3's per-tensor affine steps into its convolutions, so the graph is plain Conv -> activation chains.

PP-LCNetV3 wraps every convolution as Conv -> bias Add -> Mul(s1) -> Add(b1) -> HardSwish -> Mul(s2) -> Add(b2)
(its "learnable affine blocks", scalar s and b), and the stem as Conv -> BatchNormalization. Folds:
- **into the producing conv (exact):** BatchNormalization; the bias Add; Mul(s1) -> Add(b1).
- **into every consuming 1x1 conv** (``--inexact``: k x k ones too): the post-activation Mul(s2) -> Add(b2). The weights become w * s2 and the bias
  gains b2 * sum(w). That is exact for a 1x1 convolution, and exact for a k x k one except at the zero-padded
  border. There the original reads 0 where the folded graph reads -b2 / s2, so the border output differs by
  b2 times the padded taps' weights. ``--check`` measures what that does to the model's outputs.
Constant subgraphs (the Reshape of a bias) are evaluated first. Run in an env with onnx + onnxruntime.
"""
import argparse

import numpy as np
import onnx
from onnx import helper, numpy_helper


def _set(field, items):
    items = list(items)
    del field[:]
    field.extend(items)


def consts(g):
    c = {i.name: numpy_helper.to_array(i) for i in g.initializer}
    for n in g.node:
        if n.op_type == "Constant":
            c[n.output[0]] = numpy_helper.to_array(n.attribute[0].t)
    changed = True
    while changed:
        changed = False
        for n in g.node:
            if n.op_type == "Reshape" and n.output[0] not in c and all(x in c for x in n.input):
                c[n.output[0]] = c[n.input[0]].reshape(c[n.input[1]].astype(np.int64))
                changed = True
    return c


def fold(m, inexact=False):
    g = m.graph
    c = consts(g)
    prod = {o: n for n in g.node for o in n.output}
    cons = {}
    for n in g.node:
        for x in n.input:
            cons.setdefault(x, []).append(n)
    W = {}       # conv name -> [weight array, bias array]
    for n in g.node:
        if n.op_type == "Conv":
            w = c[n.input[1]].astype(np.float64)
            b = c[n.input[2]].astype(np.float64) if len(n.input) > 2 and n.input[2] else np.zeros(w.shape[0])
            W[n.name] = [w, b]
    dead, rename = set(), {}

    def scalar(n, other):
        x = [i for i in n.input if i != other]
        return float(c[x[0]].reshape(-1)[0]) if len(x) == 1 and x[0] in c and c[x[0]].size == 1 else None

    def only(x):
        cs = cons.get(x, [])
        return cs[0] if len(cs) == 1 else None

    folded_out = 0
    for n in list(g.node):
        if n.op_type != "Conv":
            continue
        w, b = W[n.name]
        out = n.output[0]
        while True:
            nx = only(out)
            if nx is None:
                break
            if nx.op_type == "BatchNormalization":
                sc, bb, mu, var = (c[nx.input[i]].astype(np.float64) for i in range(1, 5))
                eps = next((a.f for a in nx.attribute if a.name == "epsilon"), 1e-5)
                k = sc / np.sqrt(var + eps)
                w *= k[:, None, None, None]
                b[:] = (b - mu) * k + bb
            elif nx.op_type == "Add" and any(x in c and c[x].size == w.shape[0] for x in nx.input) \
                    and scalar(nx, out) is None:
                bias = next(c[x] for x in nx.input if x in c and c[x].size == w.shape[0])
                b += bias.reshape(-1)
            elif nx.op_type == "Mul" and scalar(nx, out) is not None:
                s = scalar(nx, out)
                w *= s
                b *= s
            elif nx.op_type == "Add" and scalar(nx, out) is not None:
                b += scalar(nx, out)
            else:
                break
            dead.add(nx.name)
            rename[nx.output[0]] = n.output[0]
            out = nx.output[0]
            folded_out += 1
        if out != n.output[0]:
            pass
    # rewire: consumers of folded outputs read the conv's own output
    for n in g.node:
        for i, x in enumerate(n.input):
            while x in rename:
                x = rename[x]
            n.input[i] = x
    for o in g.output:
        while o.name in rename:
            o.name = rename[o.name]
    _set(g.node, [n for n in g.node if n.name not in dead])
    # post-activation affine (act -> Mul(s2) -> Add(b2) -> convs only) folded into the consumers
    cons = {}
    for n in g.node:
        for x in n.input:
            cons.setdefault(x, []).append(n)
    dead, post, border = set(), 0, 0
    for n in list(g.node):
        if n.op_type not in ("HardSwish", "Relu", "HardSigmoid"):
            continue
        mul = cons.get(n.output[0], [])
        if len(mul) != 1 or mul[0].op_type != "Mul" or scalar(mul[0], n.output[0]) is None:
            continue
        add = cons.get(mul[0].output[0], [])
        if len(add) != 1 or add[0].op_type != "Add" or scalar(add[0], mul[0].output[0]) is None:
            continue
        users = cons.get(add[0].output[0], [])
        if not users or any(u.op_type != "Conv" for u in users):
            continue
        if not inexact and any(W[u.name][0].shape[2] * W[u.name][0].shape[3] > 1 for u in users):
            continue          # a k x k consumer pads with zeros: the folded offset would be wrong at the border
        s2, b2 = scalar(mul[0], n.output[0]), scalar(add[0], mul[0].output[0])
        for u in users:
            w, b = W[u.name]
            b += b2 * w.sum(axis=(1, 2, 3))
            w *= s2
            border += w.shape[2] * w.shape[3] > 1 and abs(b2) > 0
            u.input[0] = n.output[0]
        dead |= {mul[0].name, add[0].name}
        post += 1
    _set(g.node, [n for n in g.node if n.name not in dead])
    # write the folded weights and biases
    keep = {i.name for i in g.initializer}
    for n in g.node:
        if n.op_type == "Conv":
            w, b = W[n.name]
            wn, bn = n.name + "_W", n.name + "_B"
            g.initializer.append(numpy_helper.from_array(w.astype(np.float32), wn))
            g.initializer.append(numpy_helper.from_array(b.astype(np.float32), bn))
            n.input[1] = wn
            if len(n.input) > 2:
                n.input[2] = bn
            else:
                n.input.append(bn)
    used = {x for n in g.node for x in n.input}
    _set(g.initializer, [i for i in g.initializer if i.name in used])
    live = {o for n in g.node for o in n.output} | {i.name for i in g.input} | {i.name for i in g.initializer}
    _set(g.node, [n for n in g.node if n.op_type != "Constant" or n.output[0] in used])
    del keep, live
    return folded_out, post, border


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--check", type=int, default=16, help="compare outputs on this many seeded inputs (0: skip)")
    ap.add_argument("--inexact", action="store_true", help="also fold post-activation offsets into k x k convs")
    a = ap.parse_args()
    m = onnx.load(a.src)
    pre, post, border = fold(m, a.inexact)
    m = onnx.shape_inference.infer_shapes(m)
    onnx.checker.check_model(m)
    onnx.save(m, a.out)
    ops = {}
    for n in m.graph.node:
        ops[n.op_type] = ops.get(n.op_type, 0) + 1
    print(f"folded {pre} producer-side steps, {post} post-activation affines ({border} into k>1 convs, inexact at "
          f"the border); ops now {ops}")
    if a.check:
        import onnxruntime as ort
        s0 = ort.InferenceSession(a.src, providers=["CPUExecutionProvider"])
        s1 = ort.InferenceSession(a.out, providers=["CPUExecutionProvider"])
        i = s0.get_inputs()[0]
        rng = np.random.default_rng(0)
        worst = 0.0
        for _ in range(a.check):
            x = rng.uniform(-1, 1, i.shape).astype(np.float32)
            y0, y1 = s0.run(None, {i.name: x})[0], s1.run(None, {i.name: x})[0]
            worst = max(worst, float(np.abs(y0 - y1).max() / (np.abs(y0).max() + 1e-12)))
        print(f"worst |diff| / max|out| over {a.check} seeded inputs: {worst:.3e}")


if __name__ == "__main__":
    main()
