# Copyright (C) 2026 The ignite-scribe contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Why OCR misreads a real window: aggregate diagnostics against the window's own UI Automation text.

For each window whose process or title contains ``--match`` (a terminal: its Text pattern is ground truth for
its pixels):
UI Automation is read, the window is captured (masked, in memory), and UI Automation is read again. Only lines
present in both reads are ground truth, so a line that changed mid-measurement (a spinner, a clock) is not truth.
An OCR line whose best match still has more than half its characters wrong has no counterpart in that truth: it
is counted as unmatched, not scored. Lines are
detected once (PP-OCRv5's detector) and read by three recognizers on the same crops:
- ``scriberec``: ScribeRec as shipped;
- ``scriberec_ascii``: ScribeRec with every class outside printable ASCII masked before the argmax;
- ``scriberec_nomap``: ScribeRec without its lookalike map (``latin_twins=False``);
- with ``--candidate DIR``: ``candidate`` and ``candidate_nomap``, another ScribeRec model on the same crops (the
  acceptance gate for a new release: same captures, so the comparison is paired);
- ``ppocr``: PP-OCRv5's recognizer.

Printed, aggregated over all samples (no text, no crops):
- CER per recognizer (edit distance to the closest substring of a UIA line, whitespace runs collapsed);
- CER by crop height (pixels, before the resize to 80);
- ink padding above and below each crop;
- the top substitution pairs (reference char -> read char, with counts; '' is an insertion or deletion);
- whether each ScribeRec space error sits in a run of 2+ spaces in the raw UIA line, or at a single space;
- edge ink: the share of crops with ink in the outer 3 px rows (top, bottom) and columns (left, right), for all
  lines and for the lines ScribeRec (and the candidate) got wrong; the background's noise;
- where insertions sit in the read line (first 2, last 2 characters, or the middle);
- the row pitch (median gap between line boxes' tops: the terminal's cell height) and crop height over it;
- the terminal's renderer settings (font, size, antialiasing, graphics API) and the window's DPI.

**The static gate** (``--static FILE``, the baseline for new ScribeRec releases): a new Windows Terminal window
shows a tracked file (``type``, then it waits), so its text cannot change while it is measured, and it holds
only repo text, never the user's. Its handle is found by comparing the window list before and after the
launch; it is closed with WM_CLOSE to that handle only (the user's terminals share its process). Baseline,
``--static docs/DESIGN.md`` on Desktop 2 (2026-10-20, 3 samples, 92 lines, 0% masked): PP-OCRv5 0.0045,
ScribeRec release-01 0.0157, r02b 0.0198.

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


def norm_runs(s: str) -> tuple[str, list[int]]:
    """``norm`` plus, for each character kept, the length of the whitespace run it stands for (1 for others)."""
    out, runs, i = [], [], 0
    s = s.strip()
    while i < len(s):
        if s[i].isspace():
            j = i
            while j < len(s) and s[j].isspace():
                j += 1
            out.append(" ")
            runs.append(j - i)
            i = j
        else:
            out.append(s[i])
            runs.append(1)
            i += 1
    return "".join(out), runs


def align(hyp: str, truth: str) -> tuple[int, list[tuple[str, str, int]]]:
    """Edit distance from ``hyp`` to the closest substring of ``truth``, and its edits as (ref, read, truth index,
    hypothesis index) (truth index -1 for an insertion)."""
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
                edits.append((truth[j - 1], hyp[i - 1], j - 1, i - 1))
            i, j = i - 1, j - 1
        elif d[i, j] == d[i - 1, j] + 1:
            edits.append(("", hyp[i - 1], -1, i - 1))
            i -= 1
        elif d[i, j] == d[i, j - 1] + 1:
            if d[i, 0] == d[i, j]:                    # the free start: stop taking deletions before the match
                break
            edits.append((truth[j - 1], "", j - 1, i))
            j -= 1
        else:
            break
    edits += [("", hyp[k], -1, k) for k in range(i)]
    return int(d[n, int(np.argmin(d[n]))]) if m else n, edits


def edges(rgb: np.ndarray, band: int = 3, thr: int = 24) -> dict:
    """Ink in the outer ``band`` rows / columns of a crop: pixels differing from the crop's modal colour by more
    than ``thr`` in any channel. Also the background's noise (std of the pixels within ``thr`` of the mode)."""
    px = rgb.reshape(-1, 3).astype(np.int32)
    q = px // 8
    keys, counts = np.unique(q[:, 0] * 1024 + q[:, 1] * 32 + q[:, 2], return_counts=True)
    k = keys[counts.argmax()]
    mode = np.array([k // 1024, (k // 32) % 32, k % 32]) * 8 + 4
    ink = (np.abs(rgb.astype(np.int32) - mode) > thr).any(2)
    bg = px[~ink.reshape(-1)]
    b = min(band, max(1, rgb.shape[0] // 4))
    return dict(top=bool(ink[:b].any()), bottom=bool(ink[-b:].any()), left=bool(ink[:, :band].any()),
                right=bool(ink[:, -band:].any()), bg_std=float(bg.std()) if len(bg) else 0.0)


def _shares(d: dict) -> dict:
    out = {k: round(sum(v) / len(v), 3) for k, v in d.items() if k != "bg_std" and v}
    if d.get("bg_std"):
        out["bg_std_median"] = round(float(np.median(d["bg_std"])), 2)
    out["n"] = len(next(iter(d.values()), []))
    return out


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


def static_window(path: str) -> int:
    """Open a new Windows Terminal window showing ``path`` (it waits 3 minutes, then exits); -> its handle."""
    from ignite_scribe import windows
    f = Path(path).resolve()
    if not f.is_file():
        raise SystemExit(f"--static: no file {f}")
    before = {w.hwnd for w in windows.windows()}
    script = Path(os.environ.get("TEMP", ".")) / "ignite_scribe_ocr_gate.cmd"
    script.write_bytes(f"@echo off\r\nchcp 65001 >nul\r\ntype \"{f}\"\r\nping -n 180 127.0.0.1 >nul\r\n".encode())
    subprocess.Popen(["wt.exe", "-w", "new", "-d", str(f.parent), "cmd", "/c", str(script)])
    for _ in range(40):
        new = [w for w in windows.windows() if w.hwnd not in before and "windowsterminal" in w.process.lower()]
        if new:
            time.sleep(2.0)                        # let it finish printing
            return new[0].hwnd
        time.sleep(0.5)
    raise SystemExit("--static: the terminal window did not appear")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--match", default="WindowsTerminal")
    ap.add_argument("--samples", type=int, default=3, help="stable samples per window")
    ap.add_argument("--venv", default=str(ROOT / ".venv" / "Scripts" / "python.exe"))
    ap.add_argument("--candidate", help="another ScribeRec model directory, read on the same crops")
    ap.add_argument("--hwnd", type=int, help="measure only this window (overrides --match)")
    ap.add_argument("--static", metavar="FILE", help="open a new terminal showing this tracked file and measure it")
    ap.add_argument("--subs-of", default="scriberec",
                    help="whose substitutions, crop heights and space errors to report")
    a = ap.parse_args()
    from ignite_scribe import capture, windows
    from ignite_scribe.ocr_pipeline import OcrPipeline
    from ignite_scribe.redact import Policy
    from ignite_scribe.scriberec_rec import ScribeRec
    from ignite_scribe.vision_client import config
    from ignite_scribe.vision_stage import excluded_fn
    cfg = config()
    gate = static_window(a.static) if a.static else None
    if gate:
        a.hwnd = gate
    pp = OcrPipeline(cfg["ocr_det"], cfg["ocr_rec"], cfg["ocr_dict"], cache_size=0)
    sr = ScribeRec(cfg["ocr_rec_model"], cfg["ocr_dict"])
    sr_ascii = ScribeRec(cfg["ocr_rec_model"], cfg["ocr_dict"], charset=string.printable)
    extra = dict(scriberec_nomap=ScribeRec(cfg["ocr_rec_model"], cfg["ocr_dict"], latin_twins=False))
    if a.candidate:
        extra["candidate"] = ScribeRec(a.candidate, cfg["ocr_dict"])
        extra["candidate_nomap"] = ScribeRec(a.candidate, cfg["ocr_dict"], latin_twins=False)
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
        return {w["hwnd"]: [norm_runs(l) for b in w["text"] if b["source"] == "uia" for l in b["text"].split("\n")
                            if l.strip()] for w in d["windows"]}

    stats = {k: [0, 0] for k in ("scriberec", "scriberec_ascii", "ppocr", *extra)}
    by_h = collections.defaultdict(lambda: [0, 0])
    subs = collections.Counter()
    pads, n_lines, skipped, used, dpis = [], 0, 0, 0, set()
    unmatched = collections.Counter()
    masked = []
    space_err = collections.Counter()
    edge_all, edge_err = collections.defaultdict(list), {k: collections.defaultdict(list) for k in
                                                         ("scriberec", "candidate")}
    ins_pos, pitches, rel_h = collections.Counter(), [], []
    snap0 = windows.snapshot()
    m = a.match.lower()
    targets = [w for w in snap0.windows if (m in w.process.lower() or m in w.title.lower()) and not w.minimized]
    if a.hwnd:
        targets = [w for w in snap0.windows if w.hwnd == a.hwnd]
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
            after_set = {t for t, _ in after or []}
            stable = [l for l in before or [] if l[0] in after_set]
            if not stable or res is None or len(stable) < 0.5 * len(before):
                skipped += 1
                time.sleep(0.5)
                continue
            got += 1
            used += 1
            dpis.add(capture.dpi(target.hwnd))
            rgb = res[0]
            masked.append(round(res[2], 3))
            lines = pp(rgb)
            crops = [rgb[l.box[1]:l.box[3], l.box[0]:l.box[2]] for l in lines]
            tops = sorted(l.box[1] for l in lines)
            gaps = [b - a for a, b in zip(tops, tops[1:]) if 8 <= b - a <= 80]
            pitch = float(np.median(gaps)) if gaps else None
            if pitch:
                pitches.append(pitch)
                rel_h += [round(c.shape[0] / pitch, 1) for c in crops]
            crop_edges = [edges(c) for c in crops]
            for e in crop_edges:
                for key, v in e.items():
                    edge_all[key].append(v)
            reads = dict(ppocr=[l.text for l in lines], scriberec=[t for t, _ in sr.read_many(crops)],
                         scriberec_ascii=[t for t, _ in sr_ascii.read_many(crops)],
                         **{k: [t for t, _ in r.read_many(crops)] for k, r in extra.items()})
            for k, texts in reads.items():
                for ci, (crop, t) in enumerate(zip(crops, texts)):
                    t = norm(t)
                    if not t:
                        continue
                    best, runs = min(((align(t, ref), runs) for ref, runs in stable), key=lambda x: x[0][0])
                    if best[0] > 0.5 * len(t):
                        unmatched[k] += 1
                        continue
                    stats[k][0] += best[0]
                    stats[k][1] += len(t)
                    if k in edge_err and best[0] > 0:
                        for key, v in crop_edges[ci].items():
                            edge_err[k][key].append(v)
                    if k == a.subs_of:
                        h = crop.shape[0]
                        bin_ = f"{h // 4 * 4}-{h // 4 * 4 + 3}px"
                        by_h[bin_][0] += best[0]
                        by_h[bin_][1] += len(t)
                        subs.update((r, h) for r, h, _, _ in best[1])
                        for r, h, _, i in best[1]:
                            if r == "":
                                ins_pos["first_2" if i < 2 else "last_2" if i >= len(t) - 2 else "middle"] += 1
                        for r, h, j, _ in best[1]:
                            if r == " ":
                                space_err["in_run_of_2+" if runs[j] >= 2 else "single_space"] += 1
            for crop in crops:
                pads.append(padding(crop))
            n_lines += len(crops)
    srv.stdin.close()
    srv.wait(timeout=30)
    if gate:
        import ctypes
        ctypes.windll.user32.PostMessageW(ctypes.c_void_p(gate), 0x0010, 0, 0)   # WM_CLOSE, that window only
    pads = [p for p in pads if p[0] >= 0]
    top = sorted(p[0] for p in pads)
    bot = sorted(p[1] for p in pads)
    print(json.dumps(dict(
        windows=len(targets), stable_samples=used, skipped_unstable=skipped, lines=n_lines,
        unmatched_lines=dict(unmatched), masked_fraction=masked,
        cer={k: round(e / max(c, 1), 4) for k, (e, c) in stats.items()},
        scriberec_cer_by_crop_height={k: dict(cer=round(e / max(c, 1), 4), chars=c)
                                      for k, (e, c) in sorted(by_h.items())},
        ink_padding_px=dict(top_median=top[len(top) // 2] if top else None,
                            bottom_median=bot[len(bot) // 2] if bot else None,
                            top_zero_share=round(sum(1 for x in top if x == 0) / max(len(top), 1), 3),
                            bottom_zero_share=round(sum(1 for x in bot if x == 0) / max(len(bot), 1), 3)),
        edge_ink_share=dict(all_lines=_shares(edge_all),
                            **{f"{k}_error_lines": _shares(v) for k, v in edge_err.items() if v}),
        insertion_position=dict(ins_pos), row_pitch_px=sorted(round(x, 1) for x in pitches),
        crop_height_over_row_pitch=dict(collections.Counter(rel_h).most_common(8)),
        scriberec_space_errors=dict(space_err), scorer="whitespace runs collapsed to one space on both sides",
        scriberec_top_substitutions=[dict(ref=r, read=h, n=n) for (r, h), n in subs.most_common(20)],
        dpi=sorted(dpis), terminal=terminal_settings(),
        crop_source="PP-OCRv5 DBNet boxes from ink (unclip 1.5), rows merged; not UIA ranges"), ensure_ascii=False))
    for r in (sr, sr_ascii, *extra.values()):
        r.close()


if __name__ == "__main__":
    main()
