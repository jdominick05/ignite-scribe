# Copyright (C) 2026 The ignite-scribe contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Synthetic text lines, rendered with PIL in UI-like fonts, sizes and colours, with their transcriptions.

For calibration and for a sanity check of the float recognizer. It is NOT an accuracy benchmark: synthetic lines
are cleaner than real screens, and any number measured on them is labelled synthetic. ``--out DIR`` writes
``NNNN.png`` and ``labels.tsv``; ``--check ONNX`` runs a recognizer on them (float input, line_input transform)
and reports exact-line matches and character error rate.
"""
import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

FONTS = ["segoeui.ttf", "arial.ttf", "calibri.ttf", "tahoma.ttf", "verdana.ttf", "consola.ttf"]
WORDS = ("File Edit View Help Settings Open Save Cancel Apply Close Search Home Insert Layout Review Tools "
         "Window Format Properties Network Display Sound Update Download Upload Share Delete Rename Copy Paste "
         "New Folder Document Image Project Terminal Output Problems Debug Console Account Sign in out Next Back "
         "Finish Install Uninstall Version Status Ready Loading Error Warning Info Message Inbox Draft Sent "
         "the of and to in for on with at by from is are was 2026 10:42 AM PM 3.5 MB 128 GB 100% #1 v2.1 "
         "localhost:8080 C:/Users/me/Desktop report_final.docx Q3 Revenue $1,234.56 (optional) [beta] e-mail").split()


def sample(rng: random.Random) -> str:
    return " ".join(rng.choice(WORDS) for _ in range(rng.randint(1, 5)))


def render(text: str, rng: random.Random) -> np.ndarray:
    font = ImageFont.truetype(str(Path(r"C:\Windows\Fonts") / rng.choice(FONTS)), rng.choice([11, 12, 13, 14, 16, 18, 20]))
    dark = rng.random() < 0.3
    bg = tuple(rng.randint(0, 50) for _ in range(3)) if dark else tuple(rng.randint(205, 255) for _ in range(3))
    fg = tuple(rng.randint(200, 255) for _ in range(3)) if dark else tuple(rng.randint(0, 70) for _ in range(3))
    l, t, r, b = font.getbbox(text)
    pad = rng.randint(1, 4)
    img = Image.new("RGB", (r - l + 2 * pad, b - t + 2 * pad), bg)
    ImageDraw.Draw(img).text((pad - l, pad - t), text, font=font, fill=fg)
    return np.asarray(img)


def cer(ref: str, hyp: str) -> float:
    d = list(range(len(hyp) + 1))
    for i, rc in enumerate(ref, 1):
        prev, d[0] = d[0], i
        for j, hc in enumerate(hyp, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (rc != hc))
    return d[-1] / max(len(ref), 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", help="write the lines here")
    ap.add_argument("--check", help="recognizer ONNX (whole model, float or QDQ) to run on them")
    ap.add_argument("--dict", default=str(ROOT / "models/ocr/dict.txt"))
    ap.add_argument("--height", type=int, default=48)
    ap.add_argument("--width", type=int, default=320)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    lines = [(t, render(t, rng)) for t in (sample(rng) for _ in range(a.n))]
    if a.out:
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "labels.tsv", "w", encoding="utf-8", newline="\n") as f:
            for i, (t, img) in enumerate(lines):
                Image.fromarray(img).save(out / f"{i:04d}.png")
                f.write(f"{i:04d}.png\t{t}\n")
        print("wrote", len(lines), "lines to", out)
    if a.check:
        import onnxruntime as ort
        from ignite_scribe import ocr
        chars = ocr.load_dict(a.dict)
        s = ort.InferenceSession(a.check, providers=["CPUExecutionProvider"])
        name = s.get_inputs()[0].name
        exact, errs, ms = 0, [], []
        for t, img in lines:
            x, _ = ocr.line_input(img, a.height, a.width)
            t0 = time.perf_counter()
            probs = s.run(None, {name: x})[0][0]
            ms.append((time.perf_counter() - t0) * 1000)
            hyp, _ = ocr.ctc_decode(probs, chars)
            exact += hyp == t
            errs.append(cer(t, hyp))
        print(f"synthetic lines {len(lines)}: exact {exact}/{len(lines)}, CER {np.mean(errs):.4f}, "
              f"CPU ms/line median {sorted(ms)[len(ms) // 2]:.2f}")
        for t, img in lines[:5]:
            x, _ = ocr.line_input(img, a.height, a.width)
            print(repr(t), "->", repr(ocr.ctc_decode(s.run(None, {name: x})[0][0], chars)[0]))


if __name__ == "__main__":
    main()
