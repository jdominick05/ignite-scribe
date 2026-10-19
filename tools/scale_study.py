# Copyright (C) 2026 The ignite-scribe contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Which input scale suits the icon detector on desktop screenshots? CPU (ONNX Runtime), float weights.

Modes, each run over the ScreenSpot-v2 desktop screenshots (Apache-2.0, public, never the user's screen):
    down       the whole screenshot letterboxed to 640 (one inference)
    tiles@S    the screenshot resized by S, cut into overlapping 640 tiles, detections merged
Metric: of the labelled targets (ScreenSpot's one element per sample, typed icon or text), the share covered by
a detection: ``iou50`` (a detection with IoU >= 0.5) and ``centre`` (the target's centre inside a detection).
Also detections per image, inferences per image and CPU milliseconds per image.

First it checks ``vision.decode_heads`` on the cut model against Ultralytics' in-graph decode in the full model.

    python tools/scale_study.py --data calib/screenspot_v2 --modes down,tiles@1.0,tiles@0.75,tiles@0.5
Run with numpy, opencv and onnxruntime (ignite-xdna's mlir-aie-iron env).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ignite_scribe import vision as V  # noqa: E402


def resize(img, wh):
    return cv2.resize(img, wh, interpolation=cv2.INTER_AREA if wh[0] < img.shape[1] else cv2.INTER_LINEAR)


def blob(rgb):
    return (rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]


class Detector:
    def __init__(self, path):
        o = ort.SessionOptions()
        o.intra_op_num_threads = 4
        self.s = ort.InferenceSession(path, o, providers=["CPUExecutionProvider"])
        self.inp = self.s.get_inputs()[0].name
        self.outs = [x.name for x in self.s.get_outputs()]
        self.n = 0

    def heads(self, rgb640):
        self.n += 1
        r = self.s.run(None, {self.inp: blob(rgb640)})
        by = dict(zip(self.outs, r))
        boxes = [by[f"/model.22/cv2.{i}/cv2.{i}.2/Conv_output_0"][0] for i in range(3)]
        classes = [by[f"/model.22/cv3.{i}/cv3.{i}.2/Conv_output_0"][0] for i in range(3)]
        return boxes, classes


def check_decode(cut: Detector, full_path: str, rgb640: np.ndarray):
    full = ort.InferenceSession(full_path, providers=["CPUExecutionProvider"])
    ref = full.run(None, {full.get_inputs()[0].name: blob(rgb640)})[0][0]          # 5 x 8400: cx cy w h score
    keep = ref[4] > 0.05
    cx, cy, w, h = ref[0][keep], ref[1][keep], ref[2][keep], ref[3][keep]
    ref_b = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)
    b, s = V.decode_heads(*cut.heads(rgb640), conf=0.05)
    order_r, order_m = np.argsort(-ref[4][keep]), np.argsort(-s)
    n = min(len(order_r), len(order_m))
    err = float(np.abs(ref_b[order_r][:n] - b[order_m][:n]).max()) if n else 0.0
    serr = float(np.abs(ref[4][keep][order_r][:n] - s[order_m][:n]).max()) if n else 0.0
    return dict(ref_count=int(keep.sum()), ours=len(s), max_box_err_px=round(err, 4), max_score_err=round(serr, 6))


def detect(det: Detector, rgb, mode, conf, iou):
    if mode == "down":
        lb, s, px, py = V.letterbox(rgb, 640, resize)
        b, sc = V.decode_heads(*det.heads(lb), conf=conf)
        b = V.to_screen(b, 0, 0, s, px, py)
    else:
        scale = float(mode.split("@")[1])
        bs, ss = [], []
        for t, x0, y0, s in V.tiles(rgb, 640, scale, 96, resize):
            b, sc = V.decode_heads(*det.heads(t), conf=conf)
            bs.append(V.to_screen(b, x0, y0, s))
            ss.append(sc)
        b, sc = np.concatenate(bs), np.concatenate(ss)
    k = V.nms(b, sc, iou)
    return b[k], sc[k]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(ROOT / "calib/screenspot_v2"))
    ap.add_argument("--cut", default=str(ROOT / "models/omniparser_v1/icon_detect_v1_640_cut.onnx"))
    ap.add_argument("--full", default=str(ROOT / "models/omniparser_v1/icon_detect_v1_640.onnx"))
    ap.add_argument("--modes", default="down,tiles@1.0,tiles@0.75,tiles@0.5")
    ap.add_argument("--confs", default="0.05,0.25")
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--limit", type=int, default=0, help="images (0: all)")
    ap.add_argument("--split", choices=["all", "calib", "eval"], default="all",
                    help="make_calib_tiles' split: even sorted positions calibrate, odd ones are held out")
    ap.add_argument("--skip-decode-check", action="store_true", help="for a cut model without a full twin")
    ap.add_argument("--container", help="run this .ignite container on the NPU instead of --cut on the CPU "
                                        "(Python 3.13 with ignite-xdna's src on the path)")
    ap.add_argument("--out", help="write the results as JSON")
    a = ap.parse_args()
    data = Path(a.data)
    samples = json.loads((data / "screenspot_desktop_v2.json").read_text(encoding="utf-8"))
    by_img = {}
    for x in samples:
        by_img.setdefault(x["img_filename"], []).append(x)
    names = sorted(by_img)
    if a.split != "all":
        names = [n for i, n in enumerate(names) if (i % 2 == 0) == (a.split == "calib")]
    names = names[:a.limit or None]
    det = Detector(a.cut) if not a.container else type("NpuDetector", (), {"n": 0})()
    chk = None
    if a.container:
        from ignite_scribe.npu_worker import NpuBackend
        npu = NpuBackend(a.container)

        def heads(rgb640):
            det.n += 1
            b, c, _ = npu.heads(rgb640)
            return b, c
        det.heads = heads
        a.skip_decode_check = True
    if not a.skip_decode_check:
        first = cv2.cvtColor(cv2.imread(str(data / "desktop" / names[0])), cv2.COLOR_BGR2RGB)
        lb, *_ = V.letterbox(first, 640, resize)
        chk = check_decode(det, a.full, lb)
        print("decode check vs in-graph decode:", chk, flush=True)
    images = {n: cv2.cvtColor(cv2.imread(str(data / "desktop" / n)), cv2.COLOR_BGR2RGB) for n in names}
    sizes = {}
    for img in images.values():
        sizes[f"{img.shape[1]}x{img.shape[0]}"] = sizes.get(f"{img.shape[1]}x{img.shape[0]}", 0) + 1
    print("images", len(images), "sizes", dict(sorted(sizes.items(), key=lambda kv: -kv[1])[:6]), flush=True)
    results = dict(decode_check=chk, images=len(images), rows=[])
    for conf in [float(c) for c in a.confs.split(",")]:
        for mode in a.modes.split(","):
            hit = {"icon": [0, 0, 0], "text": [0, 0, 0]}      # n, iou50, centre
            ndet, det.n, t0 = 0, 0, time.perf_counter()
            for n in names:
                b, s = detect(det, images[n], mode, conf, a.iou)
                ndet += len(b)
                ih, iw = images[n].shape[:2]
                f = V.letterbox_params(iw, ih, 640)[0] if mode == "down" else float(mode.split("@")[1])
                for x in by_img[n]:
                    bx, by_, bw, bh = x["bbox"]
                    tgt = np.array([[bx, by_, bx + bw, by_ + bh]], np.float32)
                    h = hit[x["data_type"]]
                    h[0] += 1
                    i50 = c = 0
                    if len(b):
                        i50 = int(V.iou_matrix(tgt, b).max() >= 0.5)
                        cx, cy = bx + bw / 2, by_ + bh / 2
                        c = int(((b[:, 0] <= cx) & (cx <= b[:, 2]) & (b[:, 1] <= cy) & (cy <= b[:, 3])).any())
                    h[1] += i50
                    h[2] += c
                    results.setdefault("targets", []).append(dict(
                        conf=conf, mode=mode, type=x["data_type"], img=f"{iw}x{ih}",
                        input_px=round(float(np.sqrt(bw * bh)) * f, 1), iou50=i50, centre=c))
            ms = (time.perf_counter() - t0) * 1000 / len(names)
            row = dict(conf=conf, mode=mode, dets_per_img=round(ndet / len(names), 1),
                       infer_per_img=round(det.n / len(names), 2), cpu_ms_per_img=round(ms, 1),
                       **{f"{k}_{m}": round(v[i] / v[0], 3) for k, v in hit.items() for i, m in ((1, "iou50"),
                                                                                              (2, "centre"))},
                       n_icon=hit["icon"][0], n_text=hit["text"][0])
            results["rows"].append(row)
            print(json.dumps(row), flush=True)
    # recall by the target's size at the detector input (geometric-mean side, pixels), all modes pooled
    edges = [0, 8, 12, 16, 24, 32, 48, 64, 96, 10 ** 6]
    for conf in sorted({t["conf"] for t in results["targets"]}):
        print(f"recall by target size at the detector input, conf {conf} (all modes pooled):")
        for typ in ("icon", "text"):
            cells = []
            for lo, hi in zip(edges, edges[1:]):
                ts = [t for t in results["targets"] if t["conf"] == conf and t["type"] == typ
                      and lo <= t["input_px"] < hi]
                if ts:
                    cells.append(f"{lo}-{hi if hi < 10 ** 6 else 'inf'}px: {sum(t['iou50'] for t in ts) / len(ts):.2f}"
                                 f"/{sum(t['centre'] for t in ts) / len(ts):.2f} (n={len(ts)})")
            print(f"  {typ}: " + "; ".join(cells))
    if a.out:
        Path(a.out).write_text(json.dumps(results, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
