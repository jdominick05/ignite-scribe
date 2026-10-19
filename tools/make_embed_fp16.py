# Copyright (C) 2026 The ignite-scribe contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Make ``model_fp16.onnx`` next to the embedding model: fp16 weights and math, fp32 inputs and outputs.

Cast nodes stay fp32 (onnxconverter-common otherwise mislabels a Cast output and the model fails to load).
Needs ``pip install onnx onnxconverter-common`` (tools only; the worker does not). Prints the worst cosine to
the fp32 model's vectors on the CPU, over random passages.
"""
import argparse
import random
import string
import sys
from pathlib import Path

import onnx
from onnxconverter_common import float16

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ignite_scribe.embed import Embedder, model_dir  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=str(model_dir()))
    a = ap.parse_args()
    d = Path(a.dir)
    m = float16.convert_float_to_float16(onnx.load(str(d / "model.onnx")), keep_io_types=True,
                                         op_block_list=list(float16.DEFAULT_OP_BLOCK_LIST) + ["Cast"])
    onnx.save(m, str(d / "model_fp16.onnx"))
    random.seed(3)
    vocab = ["".join(random.choice(string.ascii_lowercase) for _ in range(random.randint(2, 9))) for _ in range(3000)]
    texts = [" ".join(random.choice(vocab) for _ in range(random.randint(5, 140))) for _ in range(64)]
    ref = Embedder(d, device="cpu", threads=4).encode(texts)
    e16 = Embedder(d, device="dml", model_file="model_fp16.onnx")
    got = e16.encode(texts)
    print(f"wrote {d / 'model_fp16.onnx'}; backend {e16.backend}; worst cosine to fp32 CPU "
          f"{float((got * ref).sum(1).min()):.6f}")


if __name__ == "__main__":
    main()
