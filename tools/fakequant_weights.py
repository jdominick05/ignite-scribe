# Copyright (C) 2026 The ignite-scribe contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Weight-only fake quantization of a float ONNX: which weights survive XINT8's per-tensor power-of-two grid?

Every Conv weight is rounded to int8 in [-127, 127] at a power-of-two scale, per tensor (the engine's format) or
per output channel (``--per-channel``, a diagnostic), and written back as float. Activations stay float, so a
CER change measured on the result is the weights' doing alone. ``--report`` lists, per conv, the spread of its
per-channel max |w| (max / median) and the share of channels whose weights round to all zeros at the
per-tensor scale. Run with onnx + numpy.
"""
import argparse

import numpy as np
import onnx
from onnx import numpy_helper


def pow2_scale(m):
    m = np.maximum(m, 1e-12)
    return 2.0 ** np.ceil(np.log2(m / 127.0))


def quant(w, s):
    return np.clip(np.round(w / s), -127, 127) * s


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out")
    ap.add_argument("--per-channel", action="store_true")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    m = onnx.load(a.src)
    init = {i.name: i for i in m.graph.initializer}
    for n in m.graph.node:
        if n.op_type != "Conv" or n.input[1] not in init:
            continue
        w = numpy_helper.to_array(init[n.input[1]]).astype(np.float64)
        ch = np.abs(w).reshape(w.shape[0], -1).max(1)
        st = pow2_scale(ch.max())
        if a.report:
            dead = float((np.abs(np.round(w / st)).reshape(w.shape[0], -1).max(1) == 0).mean())
            g = next((x.i for x in n.attribute if x.name == "group"), 1)
            print(f"{n.name:10s} {list(w.shape)} g{g:<4d} chan max|w| spread {ch.max() / np.median(ch):8.1f}x  "
                  f"channels all-zero at per-tensor scale {dead:6.1%}")
        s = pow2_scale(ch)[:, None, None, None] if a.per_channel else st
        init[n.input[1]].CopyFrom(numpy_helper.from_array(quant(w, s).astype(np.float32), n.input[1]))
    if a.out:
        onnx.save(m, a.out)


if __name__ == "__main__":
    main()
