# Copyright (C) 2026 The ignite-scribe contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Precision and recall of the injection flag (``injection.scan``) on public sets, one row per set. Numbers only.

Sets, fetched once into git-ignored ``calib/injection_eval/`` (licences checked on their cards):
- ``deepset``: deepset/prompt-injections, test split (Apache-2.0). Direct injections and ordinary prompts, in
  English and German, labelled. Precision and recall.
- ``gandalf``: Lakera/gandalf_ignore_instructions, test split (MIT). Direct attacks only: recall.
- ``bipia-email``: microsoft/BIPIA (MIT), the indirect case, closest to the screen. Its benign emails measure
  false positives. Each of its 75 text and code attack instructions is then inserted in an email (middle and
  end), which measures recall.
- ``seed100``: this repo's 500 synthetic UI lines (``calib/gdi_eval``), if built. False positives only.
Precision is not reported on sets with only attacks or only benign text.
"""
import argparse
import json
import random
import sys
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
CACHE = ROOT / "calib" / "injection_eval"
BIPIA = "https://raw.githubusercontent.com/microsoft/BIPIA/main/benchmark/"

from ignite_scribe.injection import scan  # noqa: E402


def fetch(name: str, url: str) -> bytes:
    p = CACHE / name
    if not p.is_file():
        CACHE.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url, timeout=60) as r:
            p.write_bytes(r.read())
    return p.read_bytes()


def hf_rows(dataset: str, split: str) -> list[dict]:
    rows, off = [], 0
    while True:
        q = urllib.parse.urlencode(dict(dataset=dataset, config="default", split=split, offset=off, length=100))
        page = json.loads(fetch(f"{dataset.replace('/', '__')}_{split}_{off}.json",
                                f"https://datasets-server.huggingface.co/rows?{q}"))
        got = [r["row"] for r in page.get("rows", [])]
        rows += got
        off += len(got)
        if not got or off >= page.get("num_rows_total", 0):
            return rows


def score(pos: list[str], neg: list[str]) -> dict:
    tp = sum(1 for t in pos if scan(t))
    fp = sum(1 for t in neg if scan(t))
    out = dict(attacks=len(pos), benign=len(neg))
    if pos:
        out["recall"] = round(tp / len(pos), 3)
    if neg:
        out["false_positive_rate"] = round(fp / len(neg), 4)
    if pos and neg:
        out["precision"] = round(tp / max(tp + fp, 1), 3)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.parse_args()
    res = {}
    d = hf_rows("deepset/prompt-injections", "test")
    res["deepset"] = score([r["text"] for r in d if r["label"] == 1], [r["text"] for r in d if r["label"] == 0])
    g = hf_rows("Lakera/gandalf_ignore_instructions", "test")
    res["gandalf"] = score([r.get("text") or next(v for v in r.values() if isinstance(v, str)) for r in g], [])
    emails = [json.loads(l)["context"] for l in fetch("bipia_email_test.jsonl", BIPIA + "email/test.jsonl")
              .decode("utf-8").splitlines() if l.strip()]
    attacks = []
    for f in ("text_attack_test.json", "code_attack_test.json"):
        for v in json.loads(fetch("bipia_" + f, BIPIA + f)).values():
            attacks += v
    rng = random.Random(7)
    planted = []
    for a in attacks:
        e = rng.choice(emails)
        mid = len(e) // 2
        planted += [e[:mid] + "\n" + a + "\n" + e[mid:], e + "\n\n" + a]
    res["bipia-email"] = score(planted, emails)
    res["bipia-email"]["attack_instructions"] = len(attacks)
    labels = ROOT / "calib" / "gdi_eval" / "labels.tsv"
    if labels.is_file():
        res["seed100"] = score([], [l.split("\t", 1)[1] for l in labels.read_text(encoding="utf-8").splitlines()
                                    if "\t" in l])
    print(json.dumps(res))


if __name__ == "__main__":
    main()
