# Copyright (C) 2026 The ignite-scribe contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""OCR accuracy on a real window, against the UI Automation text of the same window: numbers only.

For a window UI Automation reads completely (a terminal's Text pattern, a document), its text is ground truth for
what the pixels show. This reads the window both ways through the MCP server, back to back, and prints the
character error rate of the OCR text against it: per OCR line, the edit distance to the closest substring of
any UIA line, over the OCR line's length. Whitespace runs are collapsed on both sides. That measures whether what OCR read is right, not whether it read
everything. It prints no text, saves no pixels and writes nothing. The window may change between the two reads, which only makes the number
worse. Run from the repo .venv: ``python tools/ocr_vs_uia.py --match WindowsTerminal``.

``--recognizer ppocr`` reads lines with PP-OCRv5's recognizer, and ``scriberec`` (the default) with ScribeRec.
``--backend npu`` runs ScribeRec's NPU buckets, so it is an NPU sitting: run it through ignite-xdna's
``research-lowlevel.sh --npu``. The output names the recognizer and the backend the worker reports, and an NPU run
that did not use the NPU is refused rather than reported.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
def sub_cer(hyp: str, truth: str) -> int:
    """Edit distance from ``hyp`` to the closest substring of ``truth`` (free start and end in ``truth``).
    OCR cuts a row into fragments at wide gaps, so a correct fragment must score 0 against its whole row."""
    prev = [0] * (len(truth) + 1)
    for i, hc in enumerate(hyp, 1):
        cur = [i] + [0] * len(truth)
        for j, tc in enumerate(truth, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (hc != tc))
        prev = cur
    return min(prev)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--match", required=True)
    ap.add_argument("--backend", default="cpu", help="IGNITE_SCRIBE_VISION_BACKEND for the worker (default cpu)")
    ap.add_argument("--recognizer", choices=["scriberec", "ppocr"], default="scriberec")
    a = ap.parse_args()
    env = dict(os.environ, IGNITE_SCRIBE_VISION_BACKEND=a.backend, IGNITE_SCRIBE_WATCH="0",
               IGNITE_SCRIBE_ALWAYS_ON="0", IGNITE_SCRIBE_TRAY="0", IGNITE_SCRIBE_SECURE="0")
    if a.recognizer == "ppocr":                          # no ScribeRec model: the worker uses PP-OCRv5's recognizer
        env["IGNITE_SCRIBE_OCR_REC_MODEL"] = str(ROOT / "models" / "no-scriberec")
    p = subprocess.Popen([sys.executable, "-m", "ignite_scribe", "mcp"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.DEVNULL, text=True, encoding="utf-8", env=env)

    def call(i, name, args):
        p.stdin.write(json.dumps({"jsonrpc": "2.0", "id": i, "method": "tools/call",
                                  "params": {"name": name, "arguments": args}}) + "\n")
        p.stdin.flush()
        r = json.loads(p.stdout.readline())["result"]
        if r["isError"]:
            raise SystemExit(r["content"][0]["text"][:300])
        return json.loads(r["content"][0]["text"])

    p.stdin.write(json.dumps({"jsonrpc": "2.0", "id": 0, "method": "initialize", "params": {}}) + "\n")
    p.stdin.flush()
    p.stdout.readline()
    call(1, "ocr_windows", {"format": "json", "match": a.match})                      # warm the worker
    uia = call(2, "get_screen_state", {"format": "json", "max_windows": 12, "refresh": True})
    ocr = call(3, "ocr_windows", {"format": "json", "match": a.match})
    p.stdin.close()
    p.wait(timeout=30)
    w_ocr = ocr["windows"][0]
    rep = ocr["ocr"]["windows"][0]
    backend = rep.get("ocr_backend")
    if a.backend == "npu" and backend != "npu":
        raise SystemExit(f"refused: asked for the NPU, but the recognizer ran on {backend!r}")
    w_uia = next(w for w in uia["windows"] if w["hwnd"] == w_ocr["hwnd"])
    def norm(s):
        return " ".join(s.split())                  # OCR cannot know how many spaces a gap held

    truth = [norm(l) for b in w_uia["text"] if b["source"] == "uia" for l in b["text"].split("\n") if l.strip()]
    read = [norm(b["text"]) for b in w_ocr["text"] if b["source"] == "ocr"]
    errs = chars = exact = 0
    diag = dict(ascii=[0, 0], non_ascii=[0, 0], no_spaces=[0, 0])   # [errors, characters], numbers only
    for r in read:
        e = min((sub_cer(r, t) for t in truth), default=len(r))
        errs, chars, exact = errs + e, chars + max(len(r), 1), exact + (e == 0)
        k = "ascii" if r.isascii() else "non_ascii"
        diag[k][0] += e
        diag[k][1] += max(len(r), 1)
        r0 = r.replace(" ", "")                      # how much of the error is only spacing
        e0 = min((sub_cer(r0, t.replace(" ", "")) for t in truth), default=len(r0))
        diag["no_spaces"][0] += e0
        diag["no_spaces"][1] += max(len(r0), 1)
    n_non_ascii = sum(1 for r in read if not r.isascii())
    print(json.dumps(dict(window=w_ocr["process"], recognizer=a.recognizer, rec_backend=backend,
                          uia_lines=len(truth), ocr_lines=len(read),
                          cer_vs_uia=round(errs / max(chars, 1), 4), lines_exact=round(exact / max(len(read), 1), 3),
                          ocr_ms=rep["total_ms"], det_ms=rep.get("det_ms"), rec_ms=rep.get("rec_ms"),
                          cer_ascii_lines=round(diag["ascii"][0] / max(diag["ascii"][1], 1), 4),
                          cer_non_ascii_lines=round(diag["non_ascii"][0] / max(diag["non_ascii"][1], 1), 4),
                          non_ascii_lines=n_non_ascii,
                          cer_ignoring_spaces=round(diag["no_spaces"][0] / max(diag["no_spaces"][1], 1), 4))))


if __name__ == "__main__":
    main()
