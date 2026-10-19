# Copyright (C) 2026 The ignite-scribe contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""UI element detection: the pure-numpy half, shared by the NPU worker and the CPU fallback.

The detector is OmniParser V1's ``icon_detect`` (a 1-class YOLOv8n, AGPL-3.0), head-cut the way ignite-xdna does:
the graph ends at the six raw head convolutions (per stride 8 / 16 / 32: a 64-channel box head, 4 sides x 16 DFL
bins, and a 1-channel class logit). ``decode_heads`` reproduces the tail that was cut off. ``tests`` check it
against Ultralytics' own in-graph decode.

Input preparation: ``letterbox`` (square, pad 114, RGB) and ``tiles`` (fixed-size crops at a chosen scale,
overlapping, so a large screen is seen at a resolution the detector was trained for). Detections from all tiles
are mapped back to screen coordinates and merged with ``nms``.

Needs numpy only; reading or resizing images is the caller's business (``resize`` takes a function).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

STRIDES = (8, 16, 32)
REG_MAX = 16
PAD = 114


@dataclass
class Detection:
    x0: float
    y0: float
    x1: float
    y1: float
    score: float


def letterbox_params(w: int, h: int, size: int) -> tuple[float, int, int]:
    """-> (scale, pad_x, pad_y): resize by ``scale`` (never up) and place top-left at (pad_x, pad_y), centered."""
    s = min(size / w, size / h, 1.0)
    nw, nh = round(w * s), round(h * s)
    return s, (size - nw) // 2, (size - nh) // 2


def letterbox(rgb: np.ndarray, size: int, resize) -> tuple[np.ndarray, float, int, int]:
    """HxWx3 uint8 RGB -> (size x size x 3 uint8, scale, pad_x, pad_y). ``resize(img, (w, h))`` does the resampling."""
    h, w = rgb.shape[:2]
    s, px, py = letterbox_params(w, h, size)
    img = rgb if s == 1.0 else resize(rgb, (round(w * s), round(h * s)))
    out = np.full((size, size, 3), PAD, np.uint8)
    out[py:py + img.shape[0], px:px + img.shape[1]] = img
    return out, s, px, py


def tile_origins(length: int, size: int, overlap: int) -> list[int]:
    """Start offsets covering [0, length) with windows of ``size`` that overlap by at least ``overlap``."""
    if length <= size:
        return [0]
    n = int(np.ceil((length - overlap) / (size - overlap)))
    step = (length - size) / (n - 1)
    return [round(i * step) for i in range(n)]


def tiles(rgb: np.ndarray, size: int = 640, scale: float = 1.0, overlap: int = 96, resize=None):
    """Yield (tile size x size x 3 uint8, x_origin, y_origin, scale): the image resized by ``scale``, cut into
    overlapping square tiles; edge tiles smaller than ``size`` are padded with 114 at the right / bottom."""
    img = rgb
    if scale != 1.0:
        h, w = rgb.shape[:2]
        img = resize(rgb, (max(1, round(w * scale)), max(1, round(h * scale))))
    h, w = img.shape[:2]
    for y in tile_origins(h, size, overlap):
        for x in tile_origins(w, size, overlap):
            crop = img[y:y + size, x:x + size]
            if crop.shape[0] != size or crop.shape[1] != size:
                t = np.full((size, size, 3), PAD, np.uint8)
                t[:crop.shape[0], :crop.shape[1]] = crop
                crop = t
            yield crop, x, y, scale


def _softmax(x, axis):
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def decode_heads(boxes: list[np.ndarray], classes: list[np.ndarray], conf: float = 0.05,
                 strides=STRIDES) -> tuple[np.ndarray, np.ndarray]:
    """Raw heads -> (N x 4 xyxy in input pixels, N scores) above ``conf``.

    ``boxes[i]``: (64, H, W) or (1, 64, H, W) float, the DFL logits for stride ``strides[i]``;
    ``classes[i]``: (1, H, W) or (1, 1, H, W) float logits. Matches Ultralytics' Detect tail: softmax over 16 bins,
    expected distance per side, anchors at cell centres, sigmoid class score.
    """
    out_b, out_s = [], []
    for b, c, st in zip(boxes, classes, strides):
        b = np.asarray(b, np.float32).reshape(4, REG_MAX, *b.shape[-2:])
        c = np.asarray(c, np.float32).reshape(-1, *c.shape[-2:])
        score = 1.0 / (1.0 + np.exp(-c.max(axis=0)))
        ys, xs = np.nonzero(score > conf)
        if not len(ys):
            continue
        logits = b[:, :, ys, xs]                                # 4 x 16 x n
        dist = (_softmax(logits, 1) * np.arange(REG_MAX, dtype=np.float32)[None, :, None]).sum(1)   # 4 x n
        ax, ay = xs + 0.5, ys + 0.5
        out_b.append(np.stack([ax - dist[0], ay - dist[1], ax + dist[2], ay + dist[3]], 1) * st)
        out_s.append(score[ys, xs])
    if not out_b:
        return np.zeros((0, 4), np.float32), np.zeros(0, np.float32)
    return np.concatenate(out_b).astype(np.float32), np.concatenate(out_s).astype(np.float32)


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ix0, iy0 = np.maximum(a[:, None, 0], b[None, :, 0]), np.maximum(a[:, None, 1], b[None, :, 1])
    ix1, iy1 = np.minimum(a[:, None, 2], b[None, :, 2]), np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix1 - ix0, 0, None) * np.clip(iy1 - iy0, 0, None)
    area = lambda r: (r[:, 2] - r[:, 0]) * (r[:, 3] - r[:, 1])   # noqa: E731
    return inter / np.maximum(area(a)[:, None] + area(b)[None, :] - inter, 1e-9)


def nms(boxes: np.ndarray, scores: np.ndarray, iou: float = 0.45, max_det: int = 1000) -> np.ndarray:
    """Greedy NMS -> kept indices, best score first."""
    order = np.argsort(-scores)
    keep = []
    while len(order) and len(keep) < max_det:
        i = order[0]
        keep.append(i)
        if len(order) == 1:
            break
        rest = order[1:]
        order = rest[iou_matrix(boxes[i:i + 1], boxes[rest])[0] <= iou]
    return np.array(keep, np.int64)


def to_screen(boxes: np.ndarray, x0: int, y0: int, scale: float, pad_x: int = 0, pad_y: int = 0) -> np.ndarray:
    """Input-pixel boxes of one tile / letterboxed image -> screen pixels."""
    out = boxes.copy()
    out[:, [0, 2]] = (out[:, [0, 2]] - pad_x + x0) / scale
    out[:, [1, 3]] = (out[:, [1, 3]] - pad_y + y0) / scale
    return out
