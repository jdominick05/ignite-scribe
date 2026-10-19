# Copyright (C) 2026 The ignite-scribe contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tiny one-feature ONNX models that ask the NPU engine one question each: does it take what a text-line
recognizer (PP-OCRv5 / PP-LCNetV3) needs?

Each model is a 3x3 stem (3 -> 32 channels, stride 1) on a 1x3xHxW input, then the feature under test, then a 1x1
convolution to 32 channels. Weights are seeded random; the models are for the compiler's refusals and for exactness
on the device, not for accuracy. ``--build`` writes ``<out>/<name>.onnx`` and the XINT8 ``<name>_xint8.onnx``
(Quark, calibrated on seeded noise in [-1, 1]); run it in resnet_env.

Probes (name: what it isolates):
  ctl_40x320     control: a line shape the rules allow (height >= 20, width >= 20, 3x3 s1 and s2)
  ctl_20x160     control: a 20-row map, one quad tall
  h24            a 24-row map (heights between quads: 24 = 20 + 4)
  h12            a 12-row map (between one tile and one quad)
  h6 / h3        6 and 3 rows (under one 5-row tile)
  s21            3x3 stride (2, 1): halves rows, keeps columns
  s12            3x3 stride (1, 2): keeps rows, halves columns
  k5s21          5x5 stride (2, 1)
  dw3            depthwise 3x3 (group = channels)
  se             squeeze-excite: GlobalAveragePool, 1x1, ReLU, 1x1, HardSigmoid, Mul with the map
  hswish         HardSwish activation after a convolution
  lab            HardSwish, then a scalar Mul and scalar Add (PP-LCNetV3's affine block), then a depthwise 3x3
  noact          a convolution with no activation feeding another convolution (PP-LCNetV3's stem)
"""
import argparse
import copy
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

PROBES = {
    "ctl_40x320": dict(h=40, w=320, body=["c3s2", "c3"]),
    "ctl_20x160": dict(h=20, w=160, body=["c3"]),
    "h24": dict(h=24, w=160, body=["c3"]),
    "h12": dict(h=12, w=160, body=["c3"]),
    "h6": dict(h=6, w=160, body=["c3"]),
    "h3": dict(h=3, w=160, body=["c3"]),
    "s21": dict(h=40, w=160, body=["c3s21"]),
    "s12": dict(h=40, w=160, body=["c3s12"]),
    "k5s21": dict(h=40, w=160, body=["c5s21"]),
    "dw3": dict(h=40, w=160, body=["dw3"]),
    "se": dict(h=40, w=160, body=["se"]),
    "hswish": dict(h=40, w=160, body=["hswish"]),
    "lab": dict(h=40, w=160, body=["hswish", "affine", "dw3"]),
    "noact": dict(h=40, w=160, body=["c3noact", "dw3"]),
}


class G:
    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)
        self.nodes, self.inits, self.n = [], [], 0

    def name(self, p):
        self.n += 1
        return f"{p}{self.n}"

    def w(self, shape, scale=None):
        n = self.name("w")
        fan = int(np.prod(shape[1:]))
        a = self.rng.normal(0, scale or (1.0 / np.sqrt(fan)), shape).astype(np.float32)
        self.inits.append(numpy_helper.from_array(a, n))
        return n

    def conv(self, x, cin, cout, k, s=(1, 1), group=1, act="relu"):
        y = self.name("conv")
        b = self.name("b")
        self.inits.append(numpy_helper.from_array(self.rng.normal(0, 0.05, cout).astype(np.float32), b))
        self.nodes.append(helper.make_node("Conv", [x, self.w((cout, cin // group, k, k)), b], [y], name=y,
                                           kernel_shape=[k, k], strides=list(s), pads=[k // 2] * 4, group=group))
        if act:
            z = self.name(act)
            self.nodes.append(helper.make_node({"relu": "Relu", "hswish": "HardSwish"}[act], [y], [z], name=z))
            return z
        return y


def build(name, seed=0):
    p = PROBES[name]
    g = G(seed)
    x = g.conv("x", 3, 32, 3)
    for op in p["body"]:
        if op == "c3":
            x = g.conv(x, 32, 32, 3)
        elif op == "c3s2":
            x = g.conv(x, 32, 32, 3, (2, 2))
        elif op == "c3s21":
            x = g.conv(x, 32, 32, 3, (2, 1))
        elif op == "c3s12":
            x = g.conv(x, 32, 32, 3, (1, 2))
        elif op == "c5s21":
            x = g.conv(x, 32, 32, 5, (2, 1))
        elif op == "dw3":
            x = g.conv(x, 32, 32, 3, group=32)
        elif op == "hswish":
            x = g.conv(x, 32, 32, 3, act="hswish")
        elif op == "c3noact":
            x = g.conv(x, 32, 32, 3, act=None)
        elif op == "affine":          # PP-LCNetV3's learnable affine block: scalar Mul then scalar Add
            for kind, v in (("Mul", 0.15), ("Add", -0.25)):
                cn, y = g.name("k"), g.name(kind.lower())
                g.inits.append(numpy_helper.from_array(np.array(v, np.float32), cn))
                g.nodes.append(helper.make_node(kind, [x, cn], [y], name=y))
                x = y
        elif op == "se":
            pooled = g.name("gap")
            g.nodes.append(helper.make_node("GlobalAveragePool", [x], [pooled], name=pooled))
            r = g.conv(pooled, 32, 8, 1)
            e = g.conv(r, 8, 32, 1, act=None)
            hs = g.name("hsig")
            g.nodes.append(helper.make_node("HardSigmoid", [e], [hs], name=hs, alpha=0.2, beta=0.5))
            m = g.name("se_mul")
            g.nodes.append(helper.make_node("Mul", [x, hs], [m], name=m))
            x = m
    out = g.conv(x, 32, 32, 1, act=None)
    graph = helper.make_graph(g.nodes, name, [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, p["h"], p["w"]])],
                              [helper.make_tensor_value_info(out, TensorProto.FLOAT, None)], g.inits)
    m = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    m.ir_version = 8
    return onnx.shape_inference.infer_shapes(m)


def quantize(src, dst, h, w, n=16):
    from onnxruntime.quantization import CalibrationDataReader
    from quark.onnx import ModelQuantizer
    from quark.onnx.quantization.config import Config, get_default_config

    class Noise(CalibrationDataReader):
        def __init__(self):
            rng = np.random.default_rng(1)
            self.data = iter([{"x": rng.uniform(-1, 1, (1, 3, h, w)).astype(np.float32)} for _ in range(n)])

        def get_next(self):
            return next(self.data, None)

    ModelQuantizer(Config(global_quant_config=copy.deepcopy(get_default_config("XINT8")))).quantize_model(
        str(src), str(dst), Noise())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", help="comma-separated probe names (default: all)")
    ap.add_argument("--build", action="store_true", help="also quantize to XINT8 (needs Quark: resnet_env)")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for name in (a.only.split(",") if a.only else PROBES):
        f = out / f"{name}.onnx"
        onnx.save(build(name), f)
        if a.build:
            quantize(f, out / f"{name}_xint8.onnx", PROBES[name]["h"], PROBES[name]["w"])
        print("wrote", f)


if __name__ == "__main__":
    main()
