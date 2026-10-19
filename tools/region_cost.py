# Copyright (C) 2026 The ignite-scribe contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""What one always-on read costs, by the size of the changed region: detector and recognizer ms on the CPU.

Synthetic pages (``ocr_page_eval.page``: GDI ClearType lines, nothing from the screen) are cropped to the sizes a
changed region takes (a line or two of a log, a chat message, a panel, a whole window), and the OCR pipeline reads
each crop. Medians over ``--reps`` crops per size. Run in mlir-aie-iron (OpenCV, ONNX Runtime) on a quiet host.
"""
import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from ocr_page_eval import page  # noqa: E402
from ignite_scribe.ocr_pipeline import OcrPipeline  # noqa: E402

SIZES = [(1000, 60), (1000, 120), (1000, 250), (1280, 400), (1000, 700), (1920, 1040)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reps", type=int, default=15)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    m = ROOT / "models" / "ocr"
    pipe = OcrPipeline(str(m / "ppocrv5_mobile_det.onnx"), str(m / "en_ppocrv5_rec.onnx"), str(m / "dict.txt"),
                       threads=a.threads)
    rng = random.Random(77)
    full, _ = page(rng, 1920, 1080)
    pipe(full)                                           # warm-up
    out = []
    for w, h in SIZES:
        det, rec, lines, tot = [], [], [], []
        for _ in range(a.reps):
            img, _ = page(rng, 1920, 1080)
            y = rng.randint(0, 1080 - h)
            x = rng.randint(0, 1920 - w)
            crop = img[y:y + h, x:x + w].copy()
            t = time.perf_counter()
            got = pipe(crop)
            tot.append((time.perf_counter() - t) * 1000)
            det.append(pipe.last["det_ms"])
            rec.append(pipe.last["rec_ms"])
            lines.append(len(got))
        out.append(dict(region=f"{w}x{h}", det_ms=round(statistics.median(det), 1),
                        rec_ms=round(statistics.median(rec), 1), total_ms=round(statistics.median(tot), 1),
                        lines=statistics.median(lines)))
    print(json.dumps(dict(threads=a.threads, reps=a.reps, sizes=out)))


if __name__ == "__main__":
    main()
