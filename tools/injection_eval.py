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

``--model DIR`` adds the classifier tier (``injection_model``; e.g. protectai/deberta-v3-base-prompt-injection-v2,
Apache-2.0, in git-ignored ``models/injection/``): each set is scored by the patterns, by the classifier at
``--threshold``, and by either. Also its ms per text on the CPU.
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
sys.path.insert(0, str(ROOT / "tools"))
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


def score(pos: list[str], neg: list[str], flagged=None) -> dict:
    flagged = flagged or (lambda texts: [bool(scan(t)) for t in texts])
    tp, fp = sum(flagged(pos)), sum(flagged(neg))
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
    ap.add_argument("--model", help="a classifier directory (model.onnx, tokenizer.json, config.json)")
    ap.add_argument("--threshold", type=float, help="default: the model's own (config.json), else 0.5")
    ap.add_argument("--device", choices=["dml", "cpu"], default="dml", help="dml: the iGPU, as secure mode runs")
    a = ap.parse_args()
    clf = None
    if a.model:
        from ignite_scribe.injection_model import InjectionClassifier
        clf = InjectionClassifier(a.model, device=a.device)
    sets = {}
    res = {}
    d = hf_rows("deepset/prompt-injections", "test")
    sets["deepset"] = ([r["text"] for r in d if r["label"] == 1], [r["text"] for r in d if r["label"] == 0])
    from injection_data import is_german                     # pre-registered: English and German reported apart
    for lang, keep in (("en", lambda t: not is_german(t)), ("de", is_german)):
        sets[f"deepset-{lang}"] = ([r["text"] for r in d if r["label"] == 1 and keep(r["text"])],
                                   [r["text"] for r in d if r["label"] == 0 and keep(r["text"])])
    g = hf_rows("Lakera/gandalf_ignore_instructions", "test")
    sets["gandalf"] = ([r.get("text") or next(v for v in r.values() if isinstance(v, str)) for r in g], [])
    emails = [json.loads(l)["context"] for l in fetch("bipia_email_test.jsonl", BIPIA + "email/test.jsonl")
              .decode("utf-8").splitlines() if l.strip()]
    attacks = []
    for f in ("text_attack_test.json", "code_attack_test.json"):
        for v in json.loads(fetch("bipia_" + f, BIPIA + f)).values():
            attacks += v
    rng = random.Random(7)
    planted = []
    for atk in attacks:
        e = rng.choice(emails)
        mid = len(e) // 2
        planted += [e[:mid] + "\n" + atk + "\n" + e[mid:], e + "\n\n" + atk]
    sets["bipia-email"] = (planted, emails)
    labels = ROOT / "calib" / "gdi_eval" / "labels.tsv"
    if labels.is_file():
        sets["seed100"] = ([], [l.split("\t", 1)[1] for l in labels.read_text(encoding="utf-8").splitlines()
                                if "\t" in l])
    held = ROOT / "calib" / "injection_train"                 # tools/injection_data.py's held-out tests, if built
    for name in ("test_generated", "test_hardneg", "test_transfer"):
        f = held / f"{name}.jsonl"
        if f.is_file():
            rows = [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
            sets[name] = ([e["text"] for e in rows if e["label"] == 1], [e["text"] for e in rows if e["label"] == 0])
    for name, (pos, neg) in sets.items():
        if clf is None:
            res[name] = score(pos, neg)
            continue
        probs = dict(zip(pos + neg, clf.score(pos + neg)))
        thr = a.threshold if a.threshold is not None else clf.threshold
        model = lambda texts: [probs[t] >= thr for t in texts]          # noqa: E731
        either = lambda texts: [probs[t] >= thr or bool(scan(t)) for t in texts]   # noqa: E731
        res[name] = dict(patterns=score(pos, neg), classifier=score(pos, neg, model), either=score(pos, neg, either))
    if clf is not None:
        res["classifier_ms_per_window"] = round(clf.ms_total / max(clf.windows_total, 1), 1)
        res["threshold"], res["device"] = (a.threshold if a.threshold is not None else clf.threshold), clf.backend
    print(json.dumps(res))


if __name__ == "__main__":
    main()
