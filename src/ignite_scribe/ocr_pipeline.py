# Copyright (C) 2026 The ignite-scribe contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Text lines out of a masked window image: PP-OCRv5 mobile detection, then recognition. Worker side (OpenCV).

Runs inside the vision worker (ignite-xdna's Python, which has OpenCV and ONNX Runtime), never in the MCP server.
- **detect:** the DBNet detector (``ppocrv5_mobile_det``, CPU) gives a text-probability map at input resolution.
  The image is BGR, ImageNet-normalized, and resized so each side is a multiple of 32 and the longer is at most
  ``max_side``. Pixels above ``thresh`` form regions. Each region becomes an axis-aligned box (screen text is
  upright), kept when its mean probability reaches ``box_thresh``, then grown by DB's unclip distance
  (area * ``unclip`` / perimeter). Boxes on one row closer than ``merge_gap`` x their height are joined.
- **recognize:** each box, in reading order, is cropped from the full-resolution image and read by the
  recognizer. The crop is BGR, at height 48, padded to a width ``bucket``: a multiple of 32 on the CPU's
  dynamic-width model, or one of the fixed widths an NPU build is compiled for.
Lines whose recognizer confidence is under ``min_score`` are dropped.
"""
from __future__ import annotations

import hashlib
import time
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np

from . import ocr

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


@dataclass
class Line:
    box: tuple[int, int, int, int]    # x0, y0, x1, y1 in the image's pixels
    text: str
    score: float


def bucket_dynamic(w: int) -> int:
    return max(32, -(-w // 32) * 32)


def merge_rows(boxes, gap: float) -> list[tuple[int, int, int, int]]:
    """Join boxes on one text row whose horizontal gap is under ``gap`` x the row height -> reading order.

    The detector splits a row at every wider space. Each piece costs a recognizer call and loses its neighbours'
    context. Two boxes share a row when each one's vertical centre lies inside the other's span. ``gap`` 0 keeps
    the pieces as detected."""
    rows = []
    for b in sorted(boxes, key=lambda b: (b[1] + b[3]) / 2):
        cy = (b[1] + b[3]) / 2
        for r in rows:
            last = r[-1]
            if last[1] <= cy <= last[3] and b[1] <= (last[1] + last[3]) / 2 <= b[3]:
                r.append(b)
                break
        else:
            rows.append([b])
    out = []
    for r in rows:
        r.sort(key=lambda b: b[0])
        cur = list(r[0])
        for b in r[1:]:
            h = max(cur[3] - cur[1], b[3] - b[1])
            if gap and b[0] - cur[2] <= gap * h:
                cur = [cur[0], min(cur[1], b[1]), max(cur[2], b[2]), max(cur[3], b[3])]
            else:
                out.append(tuple(cur))
                cur = list(b)
        out.append(tuple(cur))
    out.sort(key=lambda b: (round(b[1] / 8), b[0]))
    return out


class OcrPipeline:
    def __init__(self, det_path, rec_path, dict_path, max_side: int = 1920, thresh: float = 0.3,
                 box_thresh: float = 0.6, unclip: float = 1.5, min_score: float = 0.5, bucket=bucket_dynamic,
                 threads: int = 4, merge_gap: float = 1.0, cache_size: int = 2048, recognizer=None):
        """``recognizer``: an object with ``read_many(rgb_crops) -> [(text, score)]``, ``backend`` and ``close()``
        (``scriberec_rec.ScribeRec``), used instead of the PP-OCRv5 recognizer at ``rec_path``."""
        self.merge_gap = merge_gap
        import onnxruntime as ort
        o = ort.SessionOptions()
        o.intra_op_num_threads = threads
        self.det = ort.InferenceSession(str(det_path), o, providers=["CPUExecutionProvider"])
        self.recognizer = recognizer
        self.rec = None if recognizer is not None else ort.InferenceSession(
            str(rec_path), o, providers=["CPUExecutionProvider"])
        self.chars = ocr.load_dict(dict_path)
        self.max_side, self.thresh, self.box_thresh = max_side, thresh, box_thresh
        self.unclip, self.min_score, self.bucket = unclip, min_score, bucket
        self.last = {}
        self.cache: OrderedDict = OrderedDict()      # line digest -> (text, score); see line_key
        self.cache_size, self.cache_min_score = cache_size, 0.9

    def boxes(self, bgr: np.ndarray) -> list[tuple[int, int, int, int]]:
        import cv2
        h, w = bgr.shape[:2]
        s = min(1.0, self.max_side / max(h, w))
        nh, nw = max(32, int(round(h * s / 32)) * 32), max(32, int(round(w * s / 32)) * 32)
        x = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
        x = ((x - MEAN) / STD).transpose(2, 0, 1)[None]
        prob = self.det.run(None, {self.det.get_inputs()[0].name: x})[0][0, 0]
        mask = (prob > self.thresh).astype(np.uint8)
        n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        fx, fy = w / nw, h / nh
        out = []
        for i in range(1, n):
            x0, y0, bw, bh, area = stats[i]
            if bw < 3 or bh < 3:
                continue
            if float(prob[lab == i].mean()) < self.box_thresh:
                continue
            d = bw * bh * self.unclip / (2 * (bw + bh))
            box = (max(0, int((x0 - d) * fx)), max(0, int((y0 - d) * fy)),
                   min(w, int(np.ceil((x0 + bw + d) * fx))), min(h, int(np.ceil((y0 + bh + d) * fy))))
            out.append(box)
        return merge_rows(out, self.merge_gap)

    def read(self, bgr_crop: np.ndarray) -> tuple[str, float]:
        h, w = bgr_crop.shape[:2]
        width = self.bucket(int(np.ceil(w * 48 / max(h, 1))))
        x, _ = ocr.line_input(bgr_crop, 48, width, resize=_cv_resize)
        return ocr.ctc_decode(self.rec.run(None, {self.rec.get_inputs()[0].name: x})[0][0], self.chars)

    @property
    def rec_backend(self) -> str:
        return self.recognizer.backend if self.recognizer is not None else "cpu"

    def close(self):
        if self.recognizer is not None:
            self.recognizer.close()

    def __call__(self, rgb: np.ndarray) -> list[Line]:
        bgr = np.ascontiguousarray(rgb[:, :, ::-1])
        t0 = time.perf_counter()
        boxes = self.boxes(bgr)
        t1 = time.perf_counter()
        got, keys, misses, hits = [None] * len(boxes), [None] * len(boxes), [], 0
        for i, b in enumerate(boxes):
            edge = b[0] <= 0 or b[1] <= 0 or b[2] >= bgr.shape[1] or b[3] >= bgr.shape[0]
            key = line_key(bgr[b[1]:b[3], b[0]:b[2]]) if self.cache_size and not edge else None  # cut-off: never
            keys[i] = key
            hit = self.cache.get(key) if key is not None else None
            if hit is not None:
                self.cache.move_to_end(key)
                got[i] = hit
                hits += 1
            else:
                misses.append(i)
        if not misses:
            reads = []
        elif self.recognizer is not None:
            reads = self.recognizer.read_many([rgb[boxes[i][1]:boxes[i][3], boxes[i][0]:boxes[i][2]] for i in misses])
        else:
            reads = [self.read(bgr[boxes[i][1]:boxes[i][3], boxes[i][0]:boxes[i][2]]) for i in misses]
        for i, r in zip(misses, reads):
            got[i] = r
            if keys[i] is not None and r[1] >= self.cache_min_score:   # a doubtful read is read again next time
                self.cache[keys[i]] = r
                if len(self.cache) > self.cache_size:
                    self.cache.popitem(last=False)
        lines = [Line(b, t, s) for b, (t, s) in zip(boxes, got) if t.strip() and s >= self.min_score]
        t2 = time.perf_counter()
        self.last = dict(det_ms=round((t1 - t0) * 1000, 2), rec_ms=round((t2 - t1) * 1000, 2), boxes=len(boxes),
                         kept=len(lines), cache_hits=hits)
        return lines


def line_key(crop: np.ndarray, tol: int = 24) -> bytes | None:
    """A digest of a line's ink, the same however loosely the detector's box sits around it.

    The crop is trimmed to the rows and columns that differ from its corner colour (the background) by more than
    ``tol``, and the trimmed pixels are hashed with their shape. Identical text drawn identically gives the same
    key wherever it moved (a scrolling log), and any changed pixel inside the ink gives another. None when the
    crop is all background. Only digests are kept, never pixels."""
    if crop.size == 0:
        return None
    bg = crop[0, 0].astype(np.int16)
    ink = np.abs(crop.astype(np.int16) - bg).max(axis=2) > tol
    rows, cols = np.flatnonzero(ink.any(axis=1)), np.flatnonzero(ink.any(axis=0))
    if not len(rows):
        return None
    t = np.ascontiguousarray(crop[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1])
    return hashlib.blake2b(t.tobytes() + repr(t.shape).encode(), digest_size=16).digest()


def _cv_resize(img, wh):
    import cv2
    return cv2.resize(img, wh, interpolation=cv2.INTER_LINEAR)
