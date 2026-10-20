# Copyright (C) 2026 The ignite-scribe contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The ScribeRec line recognizer at inference: the backbone on the NPU (or ONNX Runtime), the tail on the CPU.

Runs inside the vision worker (Python 3.13). The input contract is the training code's (Model Work, Desktop 1:
``scribe/train/data.py`` ``prepare``), matched exactly:
- RGB uint8 line crop -> PIL ``resize((nw, 80), BILINEAR)``, ``nw = max(1, round(w * 80 / h))``, then / 255;
- right-padded by repeating the last column to ``W32 = max(32, ceil(nw / 32) * 32)``, then to the bucket (320,
  640, 960 or 1280; the backbones are static, one per bucket);
- the backbone gives C x 20 x W/4 features; ``tail.onnx`` turns them into log-probabilities over 438 classes
  (blank 0, PP-OCRv5's English dict as 1..436, space 437); greedy CTC over the first ``W32 / 4`` columns only
  (bucket padding was never part of a training target).
- Lines wider than 1280 after the resize were never trained: they are split at the lowest-ink column near each
  1280 boundary, read in pieces, and joined (with a space when the cut fell in a gap between words).

Backbones: ``backbone_<W>.onnx`` / ``backbone_<W>_xint8.onnx`` in the model directory, and on the NPU
``scriberec_<W>.ignite`` containers (ignite-xdna compiles them from its recipe). ``backend`` is ``npu`` only
when every bucket in use runs on the NPU.
"""
from __future__ import annotations

import re
import time
import unicodedata
from pathlib import Path

import numpy as np

from . import ocr

BUCKETS = (320, 640, 960, 1280)
H = 80
# release-01/02 sometimes read a Latin letter or ASCII punctuation as a dictionary symbol drawn like it: the
# symbol generator sampled those twins (Model Work, train/data.py HOMOGLYPHS; fixed for r02b). Screen text here
# is English, so they are mapped back: Greek capitals, Roman numerals (NFKC: Ⅻ -> XII), letterlike symbols and
# punctuation twins. A cedilla between two words stood for a space (tools/ocr_diag.py); after a space it is dropped.
LATIN_TWINS = str.maketrans({**dict(zip("ΑΒΕΖΗΙΚΜΝΟΡΤΥΧ", "ABEZHIKMNOPTYX")),
                             **{chr(c): unicodedata.normalize("NFKC", chr(c)) for c in range(0x2160, 0x2180)
                                if unicodedata.normalize("NFKC", chr(c)).isascii()},
                             "∪": "U", "∟": "L", "℉": "F", "℮": "e", "‐": "-", "‑": "-", "‒": "-", "−": "-",
                             "‚": ",", "‛": "'", "′": "'", "´": "'", "″": '"', "․": ".", "∶": ":", "∕": "/",
                             "∖": "\\", "∗": "*", "∼": "~", "﹥": ">", "‹": "<", "›": ">"})
CEDILLA_AFTER_SPACE, CEDILLA_WORDS = re.compile(r"(?<=\s)¸\s?"), re.compile(r"(?<=\w)¸(?=\w)")


def prepare(rgb: np.ndarray) -> tuple[np.ndarray, int]:
    """-> (H x nw x 3 float32 in [0, 1], nw), the training code's resize."""
    from PIL import Image
    h, w = rgb.shape[:2]
    nw = max(1, round(w * H / max(h, 1)))
    img = np.asarray(Image.fromarray(np.ascontiguousarray(rgb)).resize((nw, H), Image.BILINEAR), np.float32)
    return img / 255.0, nw


def pad_to(x: np.ndarray, width: int) -> np.ndarray:
    """H x w x 3 -> H x width x 3, repeating the last column."""
    if x.shape[1] >= width:
        return x[:, :width]
    return np.concatenate([x, np.repeat(x[:, -1:], width - x.shape[1], axis=1)], axis=1)


def split_points(x: np.ndarray, limit: int = BUCKETS[-1], window: int = 320,
                 word_gap: float = 0.1) -> list[tuple[int, int, bool]]:
    """Pieces of at most ``limit`` columns -> [(start, end, cut_at_a_word_gap)].

    Each cut falls in the ``window`` before the boundary, in the middle of its widest run of blank columns (ink
    is the distance from the median, background, colour). A run at least ``word_gap`` x the line height wide is a
    word gap, and the pieces are joined with a space. A narrower run is a gap between letters, joined without one.
    With no blank column, the lowest-ink column is cut."""
    bg = np.median(x.reshape(-1, 3), axis=0)
    ink = np.abs(x - bg).max(axis=2).sum(axis=0)
    min_run = max(2, int(round(word_gap * x.shape[0])))
    out, s = [], 0
    while x.shape[1] - s > limit:
        lo, hi = max(s + 1, s + limit - window), s + limit
        blank = ink[lo:hi] < 1e-3
        best, run, start = (0, 0), 0, 0
        for i, v in enumerate(blank):
            if v:
                if run == 0:
                    start = i
                run += 1
                if run > best[0]:
                    best = (run, start)
            else:
                run = 0
        if best[0]:
            c, gap = lo + best[1] + best[0] // 2, best[0] >= min_run
        else:
            c, gap = lo + int(np.argmin(ink[lo:hi])), False
        out.append((s, c, gap))
        s = c
    out.append((s, x.shape[1], False))
    return out


class ScribeRec:
    def __init__(self, model_dir: str | Path, dict_path: str | Path, containers: dict[int, str] | None = None,
                 threads: int = 4, prefer_int8: bool = True, require_npu: bool = False,
                 latin_twins: bool = True, charset: str | None = None):
        """``containers``: {bucket width: .ignite path} for the NPU; missing buckets run on ONNX Runtime (the XINT8
        backbone when present, as it matches the NPU's arithmetic, else the float one). If any NPU bucket fails to
        open, the ones already open are closed and every bucket runs on the CPU (``npu_error`` says why), or
        the error is raised with ``require_npu``. ``charset``: when given, only these characters (and the blank and
        space) can be decoded; every other class is masked before the greedy argmax."""
        import onnxruntime as ort
        d = Path(model_dir)
        o = ort.SessionOptions()
        o.intra_op_num_threads = threads
        o.log_severity_level = 3
        self.tail = ort.InferenceSession(str(d / "tail.onnx"), o, providers=["CPUExecutionProvider"])
        self.chars = ocr.load_dict(dict_path)
        self.backbones: dict[int, object] = {}
        self.where: dict[int, str] = {}
        self._npu = []
        self.npu_error = None
        self.latin_twins = latin_twins
        self.masked = None if charset is None else np.array(
            [i for i, c in enumerate(self.chars) if i != 0 and c != " " and c not in charset], np.int64)
        npu = {w: c for w, c in (containers or {}).items() if w in BUCKETS and Path(c).is_file()}
        try:                                          # all NPU buckets open, or none stays open
            for w, c in sorted(npu.items()):
                from ignite_xdna.runtime.dense_session import DenseTensorSession
                s = DenseTensorSession(c)
                self._npu.append(s)
                self.backbones[w], self.where[w] = (lambda x, s=s: self._npu_run(s, x)), "npu"
        except Exception as e:  # noqa: BLE001 - e.g. the device is held by another session
            self.close()
            self.backbones.clear()
            self.where.clear()
            if require_npu:
                raise
            self.npu_error = f"{type(e).__name__}: {e}"   # every bucket on the CPU; backend reports cpu
        for w in BUCKETS:
            if w in self.backbones:
                continue
            names = ([f"backbone_{w}_xint8.onnx"] if prefer_int8 else []) + [f"backbone_{w}.onnx"]
            if w == 320:
                names.append("backbone.onnx")
            f = next((d / n for n in names if (d / n).is_file()), None)
            if f is None:
                continue
            s = ort.InferenceSession(str(f), o, providers=["CPUExecutionProvider"])
            name = s.get_inputs()[0].name
            self.backbones[w] = lambda x, s=s, name=name: s.run(None, {name: x})[0]
            self.where[w] = "cpu-int8" if "xint8" in f.name else "cpu"
        if not self.backbones:
            raise FileNotFoundError(f"no ScribeRec backbone in {d}")
        self.ms = dict(prepare=0.0, backbone=0.0, tail=0.0, decode=0.0, lines=0)   # where the time goes
        self.npu_ms: dict[str, float] = {}        # DenseTensorSession's own split of the backbone call

    def _mask(self, lp: np.ndarray) -> np.ndarray:
        if self.masked is None or not len(self.masked):
            return lp
        lp = lp.copy()
        lp[:, self.masked] = -np.inf
        return lp

    def _post(self, text: str) -> str:
        if not self.latin_twins:
            return text
        text = CEDILLA_WORDS.sub(" ", CEDILLA_AFTER_SPACE.sub("", text))
        return text.translate(LATIN_TWINS)

    def _npu_run(self, s, x):
        y, st = s.run(x)
        for k in ("stage_ms", "npu_ms", "readback_ms"):
            self.npu_ms[k] = self.npu_ms.get(k, 0.0) + float(st.get(k) or 0)
        return np.asarray(y)

    @property
    def backend(self) -> str:
        return "npu" if self.where and all(v == "npu" for v in self.where.values()) else (
            "mixed" if "npu" in self.where.values() else "cpu")

    def _bucket(self, w32: int) -> int:
        for b in BUCKETS:
            if b >= w32 and b in self.backbones:
                return b
        return max(self.backbones)

    def _piece(self, x: np.ndarray) -> tuple[str, float]:
        w32 = max(32, -(-x.shape[1] // 32) * 32)
        b = self._bucket(w32)
        inp = np.ascontiguousarray(pad_to(x, b).transpose(2, 0, 1)[None], np.float32)
        t0 = time.perf_counter()
        f = self.backbones[b](inp)
        f = f if f.ndim == 4 else f[None]
        t1 = time.perf_counter()
        lp = self.tail.run(None, {self.tail.get_inputs()[0].name: f.astype(np.float32)})[0][0]
        t2 = time.perf_counter()
        out = ocr.ctc_decode(np.exp(self._mask(lp[:min(w32, b) // 4])), self.chars)
        self.ms["backbone"] += (t1 - t0) * 1000
        self.ms["tail"] += (t2 - t1) * 1000
        self.ms["decode"] += (time.perf_counter() - t2) * 1000
        return out

    def __call__(self, rgb: np.ndarray) -> tuple[str, float]:
        """One RGB uint8 line crop -> (text, mean probability of the kept characters)."""
        return self.read_many([rgb])[0]

    def read_many(self, crops: list[np.ndarray], batch: int = 8) -> list[tuple[str, float]]:
        """RGB uint8 line crops -> [(text, score)] in order. The backbone runs once per piece (it is static at
        batch 1); the tail runs once per ``batch`` pieces of one bucket, which is exact (they share its T)."""
        pieces = []                                   # (line index, piece x, gap after it)
        for i, rgb in enumerate(crops):
            t = time.perf_counter()
            x, _ = prepare(rgb)
            self.ms["prepare"] += (time.perf_counter() - t) * 1000
            self.ms["lines"] += 1
            for s, e, gap in split_points(x):
                pieces.append((i, x[:, s:e], gap))
        by_bucket: dict[int, list[int]] = {}
        w32s = []
        for j, (_, x, _) in enumerate(pieces):
            w32 = max(32, -(-x.shape[1] // 32) * 32)
            w32s.append(w32)
            by_bucket.setdefault(self._bucket(w32), []).append(j)
        decoded: list[tuple[str, float] | None] = [None] * len(pieces)
        for b, js in by_bucket.items():
            for k in range(0, len(js), batch):
                group = js[k:k + batch]
                t0 = time.perf_counter()
                feats = []
                for j in group:
                    inp = np.ascontiguousarray(pad_to(pieces[j][1], b).transpose(2, 0, 1)[None], np.float32)
                    f = self.backbones[b](inp)
                    feats.append(f if f.ndim == 4 else f[None])
                t1 = time.perf_counter()
                lp = self.tail.run(None, {self.tail.get_inputs()[0].name:
                                          np.concatenate(feats).astype(np.float32)})[0]
                t2 = time.perf_counter()
                for r, j in enumerate(group):
                    decoded[j] = ocr.ctc_decode(np.exp(self._mask(lp[r][:min(w32s[j], b) // 4])), self.chars)
                self.ms["backbone"] += (t1 - t0) * 1000
                self.ms["tail"] += (t2 - t1) * 1000
                self.ms["decode"] += (time.perf_counter() - t2) * 1000
        texts: list[list[str]] = [[] for _ in crops]
        scores: list[list[float]] = [[] for _ in crops]
        for (i, _, gap), (t, p) in zip(pieces, decoded):
            texts[i].append(t + (" " if gap and t and not t.endswith(" ") else ""))
            scores[i].append(p)
        return [(self._post("".join(t).strip()), float(np.mean(s))) for t, s in zip(texts, scores)]

    def close(self):
        for s in self._npu:
            try:
                s.close()
            except Exception:  # noqa: BLE001
                pass
        self._npu = []
