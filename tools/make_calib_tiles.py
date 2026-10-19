# Copyright (C) 2026 The ignite-scribe contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Calibration tiles for quantizing the detector: public screenshots cut by the runtime's own tiler.

Takes the ScreenSpot-v2 desktop screenshots (Apache-2.0, public, never the user's screen). The **calibration**
split is images at even positions in sorted order; odd positions are held out for evaluating the int8 model.
Each image is tiled with ``vision.tiles`` at scales 1.0 and 0.5 (the range the runtime uses), and ``--count``
tiles are sampled with a fixed seed and written as 640x640 JPEGs. ignite-xdna's quantizer reads only ``*.jpg``,
and at 640 its letterbox is a no-op, so calibration sees exactly the runtime's input.

    python tools/make_calib_tiles.py --data calib/screenspot_v2 --out calib/tiles640 --count 64
"""
import argparse
import random
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ignite_scribe import vision as V  # noqa: E402


def resize(img, wh):
    return cv2.resize(img, wh, interpolation=cv2.INTER_AREA if wh[0] < img.shape[1] else cv2.INTER_LINEAR)


def split(names: list[str], part: str) -> list[str]:
    return [n for i, n in enumerate(sorted(names)) if (i % 2 == 0) == (part == "calib")]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="calib/screenspot_v2")
    ap.add_argument("--out", default="calib/tiles640")
    ap.add_argument("--count", type=int, default=64)
    ap.add_argument("--scales", default="1.0,0.5")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    names = split([p.name for p in (Path(a.data) / "desktop").glob("*.png")], "calib")
    pool = []
    for n in names:
        for s in (float(x) for x in a.scales.split(",")):
            pool.append((n, s))
    rng = random.Random(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    written = 0
    rng.shuffle(pool)
    for n, s in pool:
        if written >= a.count:
            break
        bgr = cv2.imread(str(Path(a.data) / "desktop" / n))
        ts = list(V.tiles(bgr, 640, s, 96, resize))
        tile, x0, y0, _ = ts[rng.randrange(len(ts))]
        cv2.imwrite(str(out / f"{Path(n).stem}_s{s:g}_{x0}_{y0}.jpg"), tile, [cv2.IMWRITE_JPEG_QUALITY, 95])
        written += 1
    print(f"{written} tiles from {len(names)} calibration images -> {out}")


if __name__ == "__main__":
    main()
