# Copyright (C) 2026 The ignite-scribe contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""CER of a ScribeRec model directory through the shipped inference path (``scriberec_rec.ScribeRec``).

Reads a line directory with ``labels.tsv`` (``tools/gdi_lines.py`` output, e.g. ``calib/gdi_eval``: seed 100,
synthetic). ``--containers DIR`` runs the backbones on the NPU (``scriberec_<W>.ignite``; run through
ignite-xdna's ``research-lowlevel.sh --npu``); without it they run on ONNX Runtime, XINT8 unless ``--float``.
Prints one JSON line: CER, exact-line share, ms per line, the backend per bucket. Labels and reads stay in
memory; nothing is printed but numbers.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from ocr_eval import cer  # noqa: E402
from ignite_scribe import scriberec_rec as R  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default=str(ROOT / "calib" / "gdi_eval"))
    ap.add_argument("--containers")
    ap.add_argument("--float", action="store_true", help="the float backbones, not XINT8 (CPU only)")
    ap.add_argument("--batch", type=int, default=8)
    a = ap.parse_args()
    d, data = Path(a.model), Path(a.data)
    cont = {}
    if a.containers:
        cont = {w: str(p) for w in R.BUCKETS for p in [Path(a.containers) / f"scriberec_{w}.ignite"] if p.is_file()}
    rec = R.ScribeRec(d, d / "dict.txt", cont, prefer_int8=not a.float, require_npu=bool(cont))
    rows = [l.split("\t", 1) for l in (data / "labels.tsv").read_text(encoding="utf-8").splitlines() if l]
    imgs = [np.asarray(Image.open(data / f).convert("RGB")) for f, _ in rows]
    t = time.perf_counter()
    out = []
    for i in range(0, len(imgs), a.batch):
        out += rec.read_many(imgs[i:i + a.batch], batch=a.batch)
    ms = (time.perf_counter() - t) * 1000 / len(imgs)
    edits = sum(cer(ref, hyp) for (_, ref), (hyp, _) in zip(rows, out))
    chars = sum(len(ref) for _, ref in rows)
    exact = sum(ref == hyp for (_, ref), (hyp, _) in zip(rows, out))
    rec.close()
    print(json.dumps(dict(model=d.name, data=data.name, lines=len(rows), backend=rec.backend, buckets=rec.where,
                          cer=round(edits / chars, 5), exact=round(exact / len(rows), 3), ms_per_line=round(ms, 2))))


if __name__ == "__main__":
    main()
