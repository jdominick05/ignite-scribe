# Copyright (C) 2026 The ignite-scribe contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""End-to-end OCR on synthetic window-sized pages: GDI ClearType lines (tools/gdi_lines.py) placed at known boxes.

Each page is ``--size`` pixels, one background per page (light or dark), with lines in rows at random x offsets.
The pipeline (``ignite_scribe.ocr_pipeline``) reads it, and the score is:
- **line recall:** a truth line counts as found when a read line covers it at IoU >= 0.5;
- **CER over all truth characters:** a missed line costs all its characters;
- **spurious lines:** read lines that match no truth line.
Nothing reads the screen; the pages are synthetic and the numbers are labelled so. Run in mlir-aie-iron
(OpenCV, ONNX Runtime) or through the worker's interpreter.
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


def page(rng: random.Random, w: int, h: int):
    dark = rng.random() < 0.3
    bg = tuple(rng.randint(10, 45) for _ in range(3)) if dark else tuple(rng.randint(225, 255) for _ in range(3))
    img = np.empty((h, w, 3), np.uint8)
    img[:] = bg
    truth, y = [], rng.randint(4, 20)
    while True:
        t = sample(rng)
        fg = tuple(rng.randint(200, 250) for _ in range(3)) if dark else tuple(rng.randint(0, 60) for _ in range(3))
        line = render(t, rng.choice(FACES), rng.choice([12, 13, 14, 15, 16, 18, 20]), fg, bg, pad=0)
        lh, lw = line.shape[:2]
        if y + lh >= h - 4:
            break
        if lw < w - 8:
            x = rng.randint(4, w - lw - 4)
            img[y:y + lh, x:x + lw] = line
            truth.append(((x, y, x + lw, y + lh), t))
        y += lh + rng.randint(6, 30)
    return img, truth


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
    return ix * iy / u if u else 0.0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pages", type=int, default=20)
    ap.add_argument("--seed", type=int, default=500)
    ap.add_argument("--size", default="1280x800")
    ap.add_argument("--det", default=str(ROOT / "models/ocr/ppocrv5_mobile_det.onnx"))
    ap.add_argument("--rec", default=str(ROOT / "models/ocr/en_ppocrv5_rec.onnx"))
    ap.add_argument("--dict", default=str(ROOT / "models/ocr/dict.txt"))
    ap.add_argument("--max-side", type=int, default=1920)
    ap.add_argument("--merge-gap", type=float, default=1.0)
    ap.add_argument("--out")
    a = ap.parse_args()
    from ignite_scribe.ocr_pipeline import OcrPipeline
    w, h = (int(v) for v in a.size.split("x"))
    pipe = OcrPipeline(a.det, a.rec, a.dict, max_side=a.max_side, merge_gap=a.merge_gap)
    rng = random.Random(a.seed)
    found = total = errs = chars = spurious = 0
    det_ms, rec_ms, tot_ms, misses = [], [], [], []
    pipe(page(random.Random(0), w, h)[0])                              # warm-up
    for _ in range(a.pages):
        img, truth = page(rng, w, h)
        t0 = time.perf_counter()
        lines = pipe(img)
        tot_ms.append((time.perf_counter() - t0) * 1000)
        det_ms.append(pipe.last["det_ms"])
        rec_ms.append(pipe.last["rec_ms"])
        used = set()
        for box, t in truth:
            total += 1
            chars += len(t)
            best = max(range(len(lines)), key=lambda i: iou(box, lines[i].box), default=None)
            if best is not None and iou(box, lines[best].box) >= 0.5 and best not in used:
                used.add(best)
                found += 1
                errs += cer(t, lines[best].text)
            else:
                errs += len(t)
                if len(misses) < 8:
                    misses.append(dict(truth=t, box=box, best=lines[best].text if best is not None else None))
        spurious += len(lines) - len(used)
    res = dict(pages=a.pages, size=a.size, truth_lines=total, line_recall=round(found / total, 4),
               cer_all=round(errs / chars, 4), spurious_lines=spurious,
               det_ms_median=round(float(np.median(det_ms)), 1), rec_ms_median=round(float(np.median(rec_ms)), 1),
               total_ms_median=round(float(np.median(tot_ms)), 1),
               lines_per_page=round(total / a.pages, 1), synthetic=True)
    print(json.dumps(res))
    for m in misses:
        print("miss:", m)
    if a.out:
        Path(a.out).write_text(json.dumps(dict(summary=res, misses=misses), indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
