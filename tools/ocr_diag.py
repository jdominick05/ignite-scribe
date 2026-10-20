# Copyright (C) 2026 The ignite-scribe contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Why OCR misreads a real window: aggregate diagnostics against the window's own UI Automation text.

For each window whose process matches ``--match`` (a terminal: its Text pattern is ground truth for its pixels):
UI Automation is read, the window is captured (masked, in memory), and UI Automation is read again. Only lines
present in both reads are ground truth, so a line that changed mid-measurement (a spinner, a clock) is not truth.
An OCR line whose best match still has more than half its characters wrong has no counterpart in that truth: it
is counted as unmatched, not scored. Lines are
detected once (PP-OCRv5's detector) and read by three recognizers on the same crops:
- ``scriberec``: ScribeRec as shipped;
- ``scriberec_ascii``: ScribeRec with every class outside printable ASCII masked before the argmax;
- ``ppocr``: PP-OCRv5's recognizer.

Printed, aggregated over all samples (no text, no crops):
- CER per recognizer (edit distance to the closest substring of a UIA line, whitespace runs collapsed);
- CER by crop height (pixels, before the resize to 80);
- ink padding above and below each crop;
- the top substitution pairs (reference char -> read char, with counts; '' is an insertion or deletion);
- the terminal's renderer settings (font, size, antialiasing, graphics API) and the window's DPI.

Run with the vision worker's Python (3.13, OpenCV and ONNX Runtime), on the CPU:
``python tools/ocr_diag.py --match WindowsTerminal`` with ``src`` on PYTHONPATH. UI Automation comes from the
repo ``.venv`` (``--venv``), through the MCP server, in memory.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import re
import string
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def norm(s: str) -> str:
    return " ".join(s.split())


def align(hyp: str, truth: str) -> tuple[int, list[tuple[str, str]]]:
    """Edit distance from ``hyp`` to the closest substring of ``truth``, and its edits as (ref, read) pairs."""
    n, m = len(hyp), len(truth)
    d = np.zeros((n + 1, m + 1), np.int32)
    d[:, 0] = np.arange(n + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + (hyp[i - 1] != truth[j - 1]))
    j = int(np.argmin(d[n])) if m else 0
    i, edits = n, []
    while i > 0 and j > 0:
        if d[i, j] == d[i - 1, j - 1] + (hyp[i - 1] != truth[j - 1]):
            if hyp[i - 1] != truth[j - 1]:
                edits.append((truth[j - 1], hyp[i - 1]))
            i, j = i - 1, j - 1
        elif d[i, j] == d[i - 1, j] + 1:
            edits.append(("", hyp[i - 1]))
            i -= 1
        elif d[i, j] == d[i, j - 1] + 1:
            if d[i, 0] == d[i, j]:                    # the free start: stop taking deletions before the match
                break
            edits.append((truth[j - 1], ""))
            j -= 1
        else:
            break
    edits += [("", hyp[k]) for k in range(i)]
    return int(d[n, int(np.argmin(d[n]))]) if m else n, edits


def padding(rgb: np.ndarray) -> tuple[int, int]:
    """Rows of background above and below the ink in a crop."""
    g = rgb.astype(np.float32).mean(2)
    border = np.concatenate([g[0], g[-1], g[:, 0], g[:, -1]])
    ink = np.abs(g - np.median(border)) > 40
    rows = np.where(ink.any(1))[0]
    if not len(rows):
        return -1, -1
    return int(rows[0]), int(g.shape[0] - 1 - rows[-1])


def terminal_settings() -> dict:
    """Only the rendering keys of Windows Terminal's settings: font, size, antialiasing, graphics API."""
    base = Path(os.environ.get("LOCALAPPDATA", "")) / "Packages"
    found = sorted(base.glob("Microsoft.WindowsTerminal*/LocalState/settings.json"))
    if not found:
        return dict(settings="not found (defaults: Cascadia Mono 12 pt, grayscale antialiasing, AtlasEngine)")
    raw = found[0].read_text(encoding="utf-8", errors="replace")
    raw = re.sub(r"^\s*//.*$", "", raw, flags=re.M)
    try:
        d = json.loads(raw)
    except ValueError:
        return dict(settings="unparsed")
    keys = ("font", "fontFace", "fontSize", "antialiasingMode", "useAtlasEngine")
    prof = d.get("profiles", {})
    defaults = prof.get("defaults", {}) if isinstance(prof, dict) else {}
    out = {f"defaults.{k}": defaults.get(k) for k in keys if k in defaults}
    plist = prof.get("list", []) if isinstance(prof, dict) else prof
    for i, p in enumerate(plist):
        for k in keys:
            if k in p:
                out[f"profile{i}.{k}"] = p[k]
    for k in ("rendering.graphicsAPI", "rendering.disablePartialInvalidation", "rendering.software",
              "experimental.rendering.forceFullRepaint", "experimental.rendering.software"):
        if k in d:
            out[k] = d[k]
    return out or dict(settings="defaults (Cascadia Mono 12 pt, grayscale antialiasing)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--match", default="WindowsTerminal")
    ap.add_argument("--samples", type=int, default=3, help="stable samples per window")
    ap.add_argument("--venv", default=str(ROOT / ".venv" / "Scripts" / "python.exe"))
    a = ap.parse_args()
    from ignite_scribe import capture, windows
    from ignite_scribe.ocr_pipeline import OcrPipeline
    from ignite_scribe.redact import Policy
    from ignite_scribe.scriberec_rec import ScribeRec
    from ignite_scribe.vision_client import config
    from ignite_scribe.vision_stage import excluded_fn
    cfg = config()
    pp = OcrPipeline(cfg["ocr_det"], cfg["ocr_rec"], cfg["ocr_dict"], cache_size=0)
    sr = ScribeRec(cfg["ocr_rec_model"], cfg["ocr_dict"])
    sr_ascii = ScribeRec(cfg["ocr_rec_model"], cfg["ocr_dict"], charset=string.printable)
    policy = Policy()

    env = dict(os.environ, IGNITE_SCRIBE_WATCH="0", IGNITE_SCRIBE_VISION="0", IGNITE_SCRIBE_TRAY="0",
               IGNITE_SCRIBE_ALWAYS_ON="0", IGNITE_SCRIBE_SECURE="0", PYTHONPATH="")
    srv = subprocess.Popen([a.venv, "-m", "ignite_scribe", "mcp"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                           stderr=subprocess.DEVNULL, text=True, encoding="utf-8", env=env)
    srv.stdin.write(json.dumps({"jsonrpc": "2.0", "id": 0, "method": "initialize", "params": {}}) + "\n")
    srv.stdin.flush()
    srv.stdout.readline()
    ids = iter(range(1, 10 ** 6))

    def uia_lines() -> dict[int, list[str]]:
        srv.stdin.write(json.dumps({"jsonrpc": "2.0", "id": next(ids), "method": "tools/call", "params": {
            "name": "get_screen_state", "arguments": {"format": "json", "max_windows": 12, "refresh": True}}}) + "\n")
        srv.stdin.flush()
        d = json.loads(json.loads(srv.stdout.readline())["result"]["content"][0]["text"])
        return {w["hwnd"]: [norm(l) for b in w["text"] if b["source"] == "uia" for l in b["text"].split("\n")
                            if l.strip()] for w in d["windows"]}

    stats = {k: [0, 0] for k in ("scriberec", "scriberec_ascii", "ppocr")}
    by_h = collections.defaultdict(lambda: [0, 0])
    subs = collections.Counter()
    pads, n_lines, skipped, used, dpis = [], 0, 0, 0, set()
    unmatched = collections.Counter()
    snap0 = windows.snapshot()
    targets = [w for w in snap0.windows if a.match.lower() in w.process.lower() and not w.minimized]
    for target in targets:
        got = 0
        for _ in range(a.samples * 4):
            if got >= a.samples:
                break
            before = uia_lines().get(target.hwnd)
            snap = windows.snapshot()
            tw = next((w for w in snap.windows if w.hwnd == target.hwnd), None)
            res = capture.masked_window(snap, tw, excluded_fn(policy)) if tw else None
            after = uia_lines().get(target.hwnd)
            stable = [l for l in before or [] if l in set(after or [])]
            if not stable or res is None or len(stable) < 0.5 * len(before):
                skipped += 1
                time.sleep(0.5)
                continue
            got += 1
            used += 1
            dpis.add(capture.dpi(target.hwnd))
            rgb = res[0]
            lines = pp(rgb)
            crops = [rgb[l.box[1]:l.box[3], l.box[0]:l.box[2]] for l in lines]
            reads = dict(ppocr=[l.text for l in lines], scriberec=[t for t, _ in sr.read_many(crops)],
                         scriberec_ascii=[t for t, _ in sr_ascii.read_many(crops)])
            for k, texts in reads.items():
                for crop, t in zip(crops, texts):
                    t = norm(t)
                    if not t:
                        continue
                    best = min(((align(t, ref)) for ref in stable), key=lambda x: x[0])
                    if best[0] > 0.5 * len(t):
                        unmatched[k] += 1
                        continue
                    stats[k][0] += best[0]
                    stats[k][1] += len(t)
                    if k == "scriberec":
                        h = crop.shape[0]
                        bin_ = f"{h // 4 * 4}-{h // 4 * 4 + 3}px"
                        by_h[bin_][0] += best[0]
                        by_h[bin_][1] += len(t)
                        subs.update(best[1])
            for crop in crops:
                pads.append(padding(crop))
            n_lines += len(crops)
    srv.stdin.close()
    srv.wait(timeout=30)
    pads = [p for p in pads if p[0] >= 0]
    top = sorted(p[0] for p in pads)
    bot = sorted(p[1] for p in pads)
    print(json.dumps(dict(
        windows=len(targets), stable_samples=used, skipped_unstable=skipped, lines=n_lines,
        unmatched_lines=dict(unmatched),
        cer={k: round(e / max(c, 1), 4) for k, (e, c) in stats.items()},
        scriberec_cer_by_crop_height={k: dict(cer=round(e / max(c, 1), 4), chars=c) for k, (e, c) in sorted(by_h.items())},
        ink_padding_px=dict(top_median=top[len(top) // 2] if top else None, bottom_median=bot[len(bot) // 2] if bot else None,
                            top_zero_share=round(sum(1 for x in top if x == 0) / max(len(top), 1), 3),
                            bottom_zero_share=round(sum(1 for x in bot if x == 0) / max(len(bot), 1), 3)),
        scriberec_top_substitutions=[dict(ref=r, read=h, n=n) for (r, h), n in subs.most_common(20)],
        dpi=sorted(dpis), terminal=terminal_settings(),
        crop_source="PP-OCRv5 DBNet boxes from ink (unclip 1.5), rows merged; not UIA ranges"), ensure_ascii=False))
    sr.close()
    sr_ascii.close()


if __name__ == "__main__":
    main()
