# Copyright (C) 2026 The ignite-scribe contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""ScribeRec on the NPU against the same model on ONNX Runtime: features, text, time per line. Uses the NPU.

Lines are synthetic (``gdi_lines.render``, nothing from the screen), chosen so every width bucket (320-1280) and
the long-line split are used. Each line goes through ``ScribeRec`` twice: NPU containers for every bucket, and
the XINT8 ONNX backbones on the CPU. Reports, per bucket, the worst feature difference in output quantization
steps, whether the decoded text is identical, and ms per line both ways. With ``--truth`` (a trained model) it
also reports the CER of each path. Run from a session that has agreed an NPU sitting; run in mlir-aie-iron.
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from gdi_lines import FACES, render, sample  # noqa: E402
from ocr_eval import cer  # noqa: E402
from ignite_scribe import scriberec_rec as R  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--containers", required=True, help="directory with scriberec_<W>.ignite (or *_<W>.ignite)")
    ap.add_argument("--lines", type=int, default=12)
    ap.add_argument("--truth", action="store_true", help="the model is trained: report CER too")
    a = ap.parse_args()
    cont = {}
    for w in R.BUCKETS:
        got = sorted(Path(a.containers).glob(f"*_{w}.ignite"))
        if got:
            cont[w] = str(got[0])
    from ignite_xdna.compiler.serializer import IgniteModelReader
    steps = {}
    for w, c in cont.items():
        with IgniteModelReader(c) as r:
            steps[w] = float(r.manifest["dense_output"].get("scale") or 1.0)
    d = Path(a.model)
    npu = R.ScribeRec(d, d / "dict.txt", cont)
    cpu = R.ScribeRec(d, d / "dict.txt", {})
    rng = random.Random(41)
    res = dict(npu_buckets=npu.where, cpu_buckets=cpu.where, buckets={})
    targets = [200, 500, 800, 1150, 1900]               # resized widths: each bucket, then a split
    edits = dict(npu=0, cpu=0, chars=0)
    for tw in targets:
        rows = []
        for _ in range(a.lines):
            t = sample(rng)
            while True:
                img = render(t, rng.choice(FACES), rng.choice([13, 15, 18]), (20, 20, 20), (245, 245, 245), pad=4)
                if img.shape[1] * 80 / img.shape[0] >= tw * 0.8 or len(t) > 400:
                    break
                t = t + " " + sample(rng)
            rgb = img                                    # render() returns RGB
            x, nw = R.prepare(rgb)
            t0 = time.perf_counter()
            tn = npu(rgb)
            tn_ms = (time.perf_counter() - t0) * 1000
            t0 = time.perf_counter()
            tc = cpu(rgb)
            tc_ms = (time.perf_counter() - t0) * 1000
            w32 = max(32, -(-min(nw, 1280) // 32) * 32)
            b = npu._bucket(w32)
            inp = np.ascontiguousarray(R.pad_to(x, b).transpose(2, 0, 1)[None], np.float32)
            fn, fc = np.asarray(npu.backbones[b](inp)), np.asarray(cpu.backbones[b](inp))
            step = steps[b]
            rows.append(dict(worst_steps=float(np.abs(fn - fc).max() / step), same=tn[0] == tc[0], npu_ms=tn_ms,
                             cpu_ms=tc_ms, split=nw > 1280))
            if a.truth:
                edits["npu"] += cer(t, tn[0])
                edits["cpu"] += cer(t, tc[0])
                edits["chars"] += len(t)
        res["buckets"][str(tw)] = dict(
            worst_steps=max(r["worst_steps"] for r in rows), same_text=f"{sum(r['same'] for r in rows)}/{len(rows)}",
            npu_ms_median=round(float(np.median([r["npu_ms"] for r in rows])), 2),
            cpu_ms_median=round(float(np.median([r["cpu_ms"] for r in rows])), 2), split=any(r["split"] for r in rows))
    if a.truth:
        res["cer_npu"] = round(edits["npu"] / max(edits["chars"], 1), 5)
        res["cer_cpu_int8"] = round(edits["cpu"] / max(edits["chars"], 1), 5)
    for name, r in (("npu", npu), ("cpu", cpu)):
        n = max(r.ms.pop("lines"), 1)
        res[f"{name}_ms_per_line"] = {k: round(v / n, 2) for k, v in r.ms.items()}
        if r.npu_ms:
            res["npu_session_ms_per_line"] = {k: round(v / n, 2) for k, v in r.npu_ms.items()}
    npu.close()
    print(json.dumps(res))


if __name__ == "__main__":
    main()
