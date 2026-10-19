# Copyright (C) 2026 The ignite-scribe contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Quantize a recognizer ONNX (whole or cut) to AMD Quark XINT8, calibrated on line crops through ``ocr.line_input``.

Run in resnet_env (Quark). ``--calib`` is a directory of line images (e.g. from tools/rec_lines.py --out); the
transform is the one inference uses, so calibration and inference cannot drift apart.
"""
import argparse
import copy
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from onnxruntime.quantization import CalibrationDataReader  # noqa: E402
from quark.onnx import ModelQuantizer  # noqa: E402
from quark.onnx.quantization.config import Config, get_default_config  # noqa: E402

from ignite_scribe import ocr  # noqa: E402


class Lines(CalibrationDataReader):
    def __init__(self, folder, limit, height, width, name):
        self.files = sorted(Path(folder).glob("*.png"))[:limit]
        self.h, self.w, self.name = height, width, name
        self.rewind()

    def get_next(self):
        if self.i >= len(self.files):
            return None
        x, _ = ocr.line_input(np.asarray(Image.open(self.files[self.i]).convert("RGB")), self.h, self.w)
        self.i += 1
        return {self.name: x}

    def rewind(self):
        self.i = 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--limit", type=int, default=128)
    ap.add_argument("--height", type=int, default=48)
    ap.add_argument("--width", type=int, default=320)
    ap.add_argument("--no-cle", action="store_true", help="turn off the XINT8 preset's cross-layer equalization")
    ap.add_argument("--adaround", type=int, default=0, metavar="ITERS", help="AdaRound iterations (0: off)")
    ap.add_argument("--device", default="cpu", help="AdaRound device (cpu, or a torch device)")
    ap.add_argument("--per-channel", action="store_true",
                    help="per-channel weight scales (diagnostic only: the NPU engine takes per-tensor weights)")
    a = ap.parse_args()
    import onnx
    name = onnx.load(a.src).graph.input[0].name
    qc = copy.deepcopy(get_default_config("XINT8"))
    if not hasattr(qc, "include_cle"):      # a misspelled attribute on Quark's dataclass assigns silently
        raise SystemExit(f"unexpected Quark config type {type(qc)}")
    qc.include_cle = not a.no_cle
    if a.per_channel:                       # a diagnostic: the engine's weights are per tensor
        if not hasattr(qc, "per_channel"):
            raise SystemExit("this Quark config has no per_channel")
        qc.per_channel = True
    if a.adaround:
        qc.include_fast_ft = True               # without this FastFinetune is ignored
        qc.extra_options["FastFinetune"] = {
            "DataSize": a.limit, "FixedSeed": 1705472343, "BatchSize": 2, "NumIterations": a.adaround,
            "LearningRate": 0.1, "OptimAlgorithm": "adaround", "OptimDevice": a.device, "InferDevice": a.device,
            "EarlyStop": True}
    dr = Lines(a.calib, a.limit, a.height, a.width, name)
    print(f"quantizing {a.src} with XINT8 (CLE {'off' if a.no_cle else 'on'}, AdaRound "
          f"{a.adaround or 'off'}) on {len(dr.files)} lines -> {a.out}")
    ModelQuantizer(Config(global_quant_config=qc)).quantize_model(a.src, a.out, dr)


if __name__ == "__main__":
    main()
