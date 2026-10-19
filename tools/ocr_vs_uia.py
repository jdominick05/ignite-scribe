# Copyright (C) 2026 The ignite-scribe contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""OCR accuracy on a real window, against the UI Automation text of the same window: numbers only.

For a window UI Automation reads completely (a terminal's Text pattern, a document), its text is ground truth for
what the pixels show. This reads the window both ways through the MCP server, back to back, and prints the
character error rate of the OCR text against it: per OCR line, the edit distance to the closest substring of
any UIA line, over the OCR line's length. Whitespace runs are collapsed on both sides. That measures whether what OCR read is right, not whether it read
everything. It prints no text, saves no pixels and writes nothing. The window may change between the two reads, which only makes the number
worse. Run from the repo .venv: ``python tools/ocr_vs_uia.py --match WindowsTerminal``.
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
    a = ap.parse_args()
    env = dict(os.environ, IGNITE_SCRIBE_VISION_BACKEND=a.backend, IGNITE_SCRIBE_WATCH="0")
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
    w_uia = next(w for w in uia["windows"] if w["hwnd"] == w_ocr["hwnd"])
    def norm(s):
        return " ".join(s.split())                  # OCR cannot know how many spaces a gap held

    truth = [norm(l) for b in w_uia["text"] if b["source"] == "uia" for l in b["text"].split("\n") if l.strip()]
    read = [norm(b["text"]) for b in w_ocr["text"] if b["source"] == "ocr"]
    errs = chars = exact = 0
    for r in read:
        e = min((sub_cer(r, t) for t in truth), default=len(r))
        errs, chars, exact = errs + e, chars + max(len(r), 1), exact + (e == 0)
    print(json.dumps(dict(window=w_ocr["process"], uia_lines=len(truth), ocr_lines=len(read),
                          cer_vs_uia=round(errs / max(chars, 1), 4), lines_exact=round(exact / max(len(read), 1), 3),
                          ocr_ms=ocr["ocr"]["windows"][0]["total_ms"])))


if __name__ == "__main__":
    main()
