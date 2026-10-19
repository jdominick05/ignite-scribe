# Copyright (C) 2026 The ignite-scribe contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The line cache on scrolling content: hit rate, recognize time, and whether cached text equals a fresh read.

A tall synthetic page (``ocr_page_eval.page``: GDI lines, nothing from the screen) is viewed through a 1000x700
window that scrolls by ``--step`` pixels a frame, the way a log or a chat moves. Each frame goes through the OCR
pipeline twice: with the line cache, and without it (a fresh pipeline per frame). Reports the cache's hit rate,
recognize ms per frame with and without it, and the share of frames whose lines (text and boxes) come out
identical either way. Run in mlir-aie-iron (OpenCV, ONNX Runtime) on a quiet host.
"""
import argparse
import json
import random
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from ocr_page_eval import page  # noqa: E402
from ocr_eval import cer  # noqa: E402
from ignite_scribe.ocr_pipeline import OcrPipeline  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", type=int, default=40)
    ap.add_argument("--step", type=int, default=37)
    ap.add_argument("--seed", type=int, default=91)
    a = ap.parse_args()
    m = ROOT / "models" / "ocr"
    args = (str(m / "ppocrv5_mobile_det.onnx"), str(m / "en_ppocrv5_rec.onnx"), str(m / "dict.txt"))
    cached, fresh = OcrPipeline(*args), OcrPipeline(*args, cache_size=0)
    tall, truth = page(random.Random(a.seed), 1000, 700 + a.step * a.frames)
    diff_lines, err_c, err_f = 0, [], []
    tot = dict(cached=0, fresh=0, chars=0)
    hits = boxes = same = 0
    rc, rf = [], []
    for f in range(a.frames):
        frame = tall[f * a.step:f * a.step + 700].copy()
        x = cached(frame)
        rc.append(cached.last["rec_ms"])
        hits += cached.last["cache_hits"]
        boxes += cached.last["boxes"]
        y = fresh(frame)
        rf.append(fresh.last["rec_ms"])
        same += [(l.box, l.text) for l in x] == [(l.box, l.text) for l in y]
        top = f * a.step
        for name, ls in (("cached", x), ("fresh", y)):        # every read line against the page's own text
            for ln in ls:
                if ln.box[1] <= 0 or ln.box[3] >= 700:
                    continue                               # cut off by the frame: no whole truth to compare
                cy = (ln.box[1] + ln.box[3]) / 2 + top
                t = min(truth, key=lambda tb: abs((tb[0][1] + tb[0][3]) / 2 - cy))[1]
                tot[name] += cer(t, ln.text)
                if name == "cached":
                    tot["chars"] += len(t)
        for lc, lf in zip(x, y):
            if lc.text != lf.text:                       # score both against the page's own text
                diff_lines += 1
                cy = (lc.box[1] + lc.box[3]) / 2 + top
                t = min(truth, key=lambda tb: abs((tb[0][1] + tb[0][3]) / 2 - cy))[1]
                err_c.append(cer(lc.text, t))
                err_f.append(cer(lf.text, t))
    later = slice(1, None)                               # frame 0 has nothing to reuse
    print(json.dumps(dict(frames=a.frames, step_px=a.step, boxes_per_frame=round(boxes / a.frames, 1),
                          hit_rate=round(hits / max(boxes, 1), 3),
                          rec_ms_cached_median=round(statistics.median(rc[later]), 1),
                          rec_ms_fresh_median=round(statistics.median(rf[later]), 1),
                          identical_frames=f"{same}/{a.frames}", differing_lines=diff_lines,
                          edits_per_differing_line_cached=round(statistics.mean(err_c), 4) if err_c else None,
                          edits_per_differing_line_fresh=round(statistics.mean(err_f), 4) if err_f else None,
                          cer_all_cached=round(tot["cached"] / max(tot["chars"], 1), 5),
                          cer_all_fresh=round(tot["fresh"] / max(tot["chars"], 1), 5))))


if __name__ == "__main__":
    main()
