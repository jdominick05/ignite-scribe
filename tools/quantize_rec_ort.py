# Copyright (C) 2026 The ignite-scribe contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""A diagnostic, not a deployable model: ONNX Runtime's own static QDQ quantizer on a recognizer backbone.

uint8 per-tensor activations (as the NPU engine has), int8 weights per channel or per tensor (``--per-channel``),
scales NOT restricted to powers of two. It answers "would per-channel weight scales alone recover the accuracy"
without Quark, whose XINT8 preset refuses per-channel. Calibrates on line crops through ``ocr.line_input``.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ignite_scribe import ocr  # noqa: E402


class Lines(CalibrationDataReader):
    def __init__(self, folder, limit, h, w, name):
        self.it = iter([{name: ocr.line_input(np.asarray(Image.open(f).convert("RGB")), h, w)[0]}
                        for f in sorted(Path(folder).glob("*.png"))[:limit]])

    def get_next(self):
        return next(self.it, None)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--limit", type=int, default=128)
    ap.add_argument("--per-channel", action="store_true")
    a = ap.parse_args()
    import onnx
    name = onnx.load(a.src).graph.input[0].name
    quantize_static(a.src, a.out, Lines(a.calib, a.limit, 48, 320, name), quant_format=QuantFormat.QDQ,
                    per_channel=a.per_channel, activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
