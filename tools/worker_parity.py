# Copyright (C) 2026 The ignite-scribe contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""NPU heads against ONNX Runtime on the same quantized ONNX, on held-out public screenshot tiles.

For each tile: both backends of ``npu_worker`` produce the six dequantized heads. Reports the largest head
difference (in units of that head's quantization step), and whether the decoded, NMS'd boxes are identical. Also
reports NPU dispatch milliseconds. Uses the held-out (odd) half of ScreenSpot-v2 desktop, tiled like the runtime.

    python tools/worker_parity.py --tiles 32     (Python 3.13, ignite-xdna src + this repo's src on the path)
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R / "src"))
sys.path.insert(0, str(R / "tools"))
from ignite_scribe import vision as V  # noqa: E402
from ignite_scribe.npu_worker import CpuBackend, NpuBackend  # noqa: E402
from make_calib_tiles import resize, split  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--container", default=str(R / "build/icon_detect_v1.ignite"))
    ap.add_argument("--onnx", default=str(R / "models/omniparser_v1/icon_detect_v1_640_cut_xint8.onnx"))
    ap.add_argument("--data", default=str(R / "calib/screenspot_v2"))
    ap.add_argument("--tiles", type=int, default=32)
    ap.add_argument("--json")
    a = ap.parse_args()
    names = split([p.name for p in (Path(a.data) / "desktop").glob("*.png")], "eval")
    tiles = []
    for n in names:
        rgb = cv2.cvtColor(cv2.imread(str(Path(a.data) / "desktop" / n)), cv2.COLOR_BGR2RGB)
        for t, *_ in V.tiles(rgb, 640, 1.0, 96, resize):
            tiles.append(t)
            break
        if len(tiles) >= a.tiles:
            break
    npu, cpu = NpuBackend(a.container), CpuBackend(a.onnx)
    sc = None
    rows, worst, same, ms = [], 0.0, 0, []
    try:
        for t in tiles:
            nb, nc, dt = npu.heads(t)
            cb, cc, _ = cpu.heads(t)
            ms.append(dt)
            if sc is None:
                sc = npu.s.run_yolo_monolithic((t.astype(np.int16) - 128).astype(np.int8).transpose(2, 0, 1)[None])["scales"]
            steps = []
            for name, x, y in zip(npu.box + npu.cls, nb + nc, cb + cc):
                steps.append(float(np.abs(x - y.reshape(x.shape)).max() / sc[name][0]))
            worst = max(worst, max(steps))
            (b1, s1), (b2, s2) = V.decode_heads(nb, nc), V.decode_heads(cb, cc)
            k1, k2 = V.nms(b1, s1), V.nms(b2, s2)
            eq = len(k1) == len(k2) and np.allclose(b1[k1], b2[k2], atol=1e-3) and np.allclose(s1[k1], s2[k2], atol=1e-6)
            same += int(eq)
            rows.append(dict(max_step_diff=max(steps), dets_npu=len(k1), dets_cpu=len(k2), same=bool(eq)))
    finally:
        npu.close()
    ms_sorted = sorted(ms[1:]) or ms
    out = dict(tiles=len(tiles), worst_head_diff_in_steps=worst, identical_detections=f"{same}/{len(tiles)}",
               npu_ms_first=round(ms[0], 3), npu_ms_median=round(ms_sorted[len(ms_sorted) // 2], 3),
               npu_ms_min=round(ms_sorted[0], 3), npu_ms_max=round(ms_sorted[-1], 3),
               dets_per_tile=round(sum(r["dets_npu"] for r in rows) / len(rows), 1))
    print(json.dumps(out))
    if a.json:
        Path(a.json).write_text(json.dumps(dict(summary=out, rows=rows), indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
