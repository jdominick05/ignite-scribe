# Copyright (C) 2026 The ignite-scribe contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Text-line recognition helpers, numpy only: the PaddleOCR line transform and greedy CTC decoding.

The recognizer (PP-OCRv5 mobile, English, Apache-2.0) reads one text line at a fixed height. ``line_input``
resizes a crop to that height keeping its aspect, normalizes it to [-1, 1] and pads the right with zeros (the
normalized value of mid grey), as PaddleOCR's own ``resize_norm_img`` does. ``ctc_decode`` takes the per-column
class probabilities: class 0 is the CTC blank, classes 1..N the dictionary, N+1 a space.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def load_dict(path: str | Path) -> list[str]:
    chars = Path(path).read_bytes().decode("utf-8").split("\n")
    if chars and chars[-1] == "":
        chars.pop()
    return ["<blank>"] + chars + [" "]


def resize_line(rgb: np.ndarray, height: int, max_width: int, resize) -> tuple[np.ndarray, int]:
    """-> (H x w x 3 uint8 resized crop, w), w = round(width * height / crop height) capped at ``max_width``."""
    h, w = rgb.shape[:2]
    nw = max(1, min(max_width, int(np.ceil(w * height / max(h, 1)))))
    return resize(rgb, (nw, height)), nw


def line_input(rgb: np.ndarray, height: int = 48, width: int = 320, resize=None) -> tuple[np.ndarray, int]:
    """One crop -> (1 x 3 x height x width float32 in [-1, 1], used width)."""
    if resize is None:
        from .vision_stage import resize_box as resize
    img, nw = resize_line(rgb, height, width, resize)
    x = np.zeros((3, height, width), np.float32)
    x[:, :, :nw] = img.transpose(2, 0, 1).astype(np.float32) / 127.5 - 1.0
    return x[None], nw


def ctc_decode(probs: np.ndarray, chars: list[str]) -> tuple[str, float]:
    """T x C probabilities -> (text, mean probability of the kept characters; 0.0 when empty)."""
    idx = probs.argmax(1)
    p = probs.max(1)
    keep = (idx != 0) & np.concatenate([[True], idx[1:] != idx[:-1]])
    text = "".join(chars[i] if i < len(chars) else "" for i in idx[keep])
    return text, float(p[keep].mean()) if keep.any() else 0.0
