# Copyright (C) 2026 The ignite-scribe contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Write the ONNX that computes what a ``--silu-sigmoid`` container computes, for ONNX Runtime on the CPU.

``ignite-compile --silu-sigmoid`` replaces every SiLU with the core's integer sigmoid epilogue, so the container
no longer equals the QDQ model it was built from. ignite-xdna's ``silu_sigmoid.reference_model`` rewrites the QDQ
model so ONNX Runtime evaluates that same epilogue (a lookup table per SiLU). The NPU worker's CPU fallback runs
this file, and the parity check compares the NPU against it.

    python tools/make_sigmoid_reference.py --in models/.../x_xint8.onnx --out models/.../x_xint8_sigref.onnx
Needs ignite-xdna's src on the path (onnx + numpy).
"""
import argparse

import onnx


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from ignite_xdna.compiler import silu_sigmoid
    m = silu_sigmoid.reference_model(onnx.load(a.src))
    onnx.save(m, a.out)
    print(f"{len(silu_sigmoid.silu_sites(onnx.load(a.src)))} SiLU sites rewritten -> {a.out}")


if __name__ == "__main__":
    main()
