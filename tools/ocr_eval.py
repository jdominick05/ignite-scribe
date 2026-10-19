# Copyright (C) 2026 The ignite-scribe contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Character error rate and latency of a line recognizer on a directory of line images with ``labels.tsv``.

Backends:
  winocr   Windows' in-box OCR (Windows.Media.Ocr, CPU), through the winrt projection (repo .venv). ``--upscale``
           enlarges each line first; the in-box engine is weak on small text.
  onnx     a whole recognizer ONNX on ONNX Runtime CPU (mlir-aie-iron or resnet_env).
  split    ``--backbone`` (e.g. the XINT8 QDQ cut) then ``--tail`` (the float pool + neck + CTC head) on ONNX
           Runtime: the arithmetic the NPU path computes, the backbone being bit-exact on the device.
The data are synthetic unless the directory says otherwise; label results accordingly. Prints one JSON line.
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


def cer(ref: str, hyp: str) -> int:
    d = list(range(len(hyp) + 1))
    for i, rc in enumerate(ref, 1):
        prev, d[0] = d[0], i
        for j, hc in enumerate(hyp, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (rc != hc))
    return d[-1]


def winocr_reader(upscale: float):
    import asyncio
    from winrt.windows.graphics.imaging import BitmapPixelFormat, SoftwareBitmap
    from winrt.windows.media.ocr import OcrEngine
    from winrt.windows.storage.streams import DataWriter
    eng = OcrEngine.try_create_from_user_profile_languages()

    async def rec(bgra, w, h):
        dw = DataWriter()
        dw.write_bytes(bgra)
        sb = SoftwareBitmap.create_copy_from_buffer(dw.detach_buffer(), BitmapPixelFormat.BGRA8, w, h)
        r = await eng.recognize_async(sb)
        return " ".join(line.text for line in r.lines)

    def run(rgb):
        img = Image.fromarray(rgb)
        if upscale != 1:
            img = img.resize((round(img.width * upscale), round(img.height * upscale)), Image.BICUBIC)
        a = np.asarray(img.convert("RGBA"))[:, :, [2, 1, 0, 3]].copy()
        return asyncio.run(rec(a.tobytes(), img.width, img.height))
    return run


def onnx_reader(model, backbone, tail, chars, height, width):
    import onnxruntime as ort
    from ignite_scribe import ocr
    o = ort.SessionOptions()
    o.intra_op_num_threads = 4
    if model:
        s = ort.InferenceSession(model, o, providers=["CPUExecutionProvider"])
        name = s.get_inputs()[0].name

        def run(rgb):
            w = width
            if not w:                           # dynamic: the line's own aspect width, padded up to 32
                w = -(-int(np.ceil(rgb.shape[1] * height / rgb.shape[0])) // 32) * 32
            x, _ = ocr.line_input(rgb, height, w)
            return ocr.ctc_decode(s.run(None, {name: x})[0][0], chars)[0]
        return run
    b = ort.InferenceSession(backbone, o, providers=["CPUExecutionProvider"])
    t = ort.InferenceSession(tail, o, providers=["CPUExecutionProvider"])
    bn, tn = b.get_inputs()[0].name, t.get_inputs()[0].name

    def run(rgb):
        x, _ = ocr.line_input(rgb, height, width)
        return ocr.ctc_decode(t.run(None, {tn: b.run(None, {bn: x})[0]})[0][0], chars)[0]
    return run


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--backend", choices=["winocr", "onnx", "split"], required=True)
    ap.add_argument("--model")
    ap.add_argument("--backbone")
    ap.add_argument("--tail")
    ap.add_argument("--dict", default=str(ROOT / "models/ocr/dict.txt"))
    ap.add_argument("--height", type=int, default=48)
    ap.add_argument("--width", type=int, default=320, help="0: each line at its own width (a dynamic-width model)")
    ap.add_argument("--upscale", type=float, default=1.0)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--bgr", action="store_true", help="feed channels in BGR order, as PaddleOCR (cv2) does")
    ap.add_argument("--out", help="also write per-line results (JSON)")
    a = ap.parse_args()
    rows = [l.split("\t", 1) for l in (Path(a.data) / "labels.tsv").read_text(encoding="utf-8").splitlines()]
    rows = rows[:a.limit] if a.limit else rows
    if a.backend == "winocr":
        run = winocr_reader(a.upscale)
    else:
        from ignite_scribe import ocr
        run = onnx_reader(a.model, a.backbone, a.tail, ocr.load_dict(a.dict), a.height, a.width)
    errs = chars = exact = 0
    ms, per = [], []
    run(np.asarray(Image.open(Path(a.data) / rows[0][0]).convert("RGB")))       # warm-up
    for f, ref in rows:
        rgb = np.asarray(Image.open(Path(a.data) / f).convert("RGB"))
        if a.bgr:
            rgb = rgb[:, :, ::-1].copy()
        t0 = time.perf_counter()
        hyp = run(rgb)
        ms.append((time.perf_counter() - t0) * 1000)
        e = cer(ref, hyp)
        errs, chars, exact = errs + e, chars + len(ref), exact + (hyp == ref)
        per.append(dict(file=f, ref=ref, hyp=hyp, errors=e))
    res = dict(data=Path(a.data).name, backend=a.backend, model=Path(a.model or a.backbone or "winocr").name,
               upscale=a.upscale, lines=len(rows), cer=round(errs / max(chars, 1), 4),
               exact=round(exact / len(rows), 4), ms_median=round(float(np.median(ms)), 2),
               ms_p90=round(float(np.percentile(ms, 90)), 2))
    print(json.dumps(res))
    if a.out:
        Path(a.out).write_text(json.dumps(dict(summary=res, lines=per), indent=1, ensure_ascii=False),
                               encoding="utf-8")


if __name__ == "__main__":
    main()
