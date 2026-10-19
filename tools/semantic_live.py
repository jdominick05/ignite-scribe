# Copyright (C) 2026 The ignite-scribe contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Semantic history with the real embed worker, on a temporary history file. Prints numbers only.

``--synthetic``: 12 windows on distinct topics (text written here), then paraphrased queries that share few
words with their window. Reports top-1 / top-3 and the mean reciprocal rank for semantic search, and how often
keyword search (all query words) finds the window at all; plus indexing and query speed and the worker's backend.
``--live SECONDS``: the real watcher records the desktop into a temporary file, then everything is indexed.
Reports passages produced per hour and indexing time, never any text. The temporary file is deleted.
"""
import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ignite_scribe import history as H, semantic as S  # noqa: E402
from ignite_scribe.passages import Passages  # noqa: E402

WINDOWS = [
    ("Weeknight lentil soup - Recipes", "msedge.exe", ["Rinse one cup of red lentils", "Saute onion, carrot and garlic in olive oil",
     "Add cumin, smoked paprika and vegetable stock", "Simmer 25 minutes, then blend half of it"]),
    ("Pull request #482: fix race in cache eviction", "chrome.exe", ["The LRU eviction thread could free an entry still being read",
     "Take the shard lock before decrementing the refcount", "Added a stress test with 64 reader threads", "CI: all checks passed"]),
    ("Flight confirmation - Mail", "outlook.exe", ["Your booking is confirmed: Lisbon to Toronto", "Departs Thursday 14:05, gate assigned at check-in",
     "One checked bag included", "Seat 23A, window"]),
    ("RX 9070 review - Hardware site", "firefox.exe", ["The card trades blows with its rival at 1440p", "Power draw peaked at 304 watts in our test",
     "Ray tracing performance improved a lot over last generation", "Street price has dropped below launch MSRP"]),
    ("Q3 budget.xlsx", "excel.exe", ["Rent 1450", "Groceries 520", "Electricity and heating 180", "Savings transfer 400", "Total monthly outgoings 2550"]),
    ("Terminal", "WindowsTerminal.exe", ["Traceback (most recent call last):", "File train.py, line 88, in step",
     "RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB", "reduce the batch size or enable gradient checkpointing"]),
    ("Dentist - Calendar", "olk.exe", ["Dental checkup and cleaning", "Tuesday 9:30, Dr. Okafor, Riverside clinic", "Bring insurance card"]),
    ("Tenancy agreement.pdf", "Acrobat.exe", ["The tenant shall give two months written notice", "The deposit is held in a protection scheme",
     "Pets are permitted with prior written consent of the landlord"]),
    ("Chess - Lichess", "chrome.exe", ["Sicilian Defence, Najdorf variation", "Black to move: knight sacrifice on e4 wins material", "Rapid 10+0, rating 1840"]),
    ("Running log", "notepad.exe", ["Monday easy 6 km at 5:40 per km", "Wednesday intervals 6 x 800 m", "Saturday long run 18 km, legs heavy after 14"]),
    ("Kubernetes docs - Pods", "msedge.exe", ["A Pod is the smallest deployable unit", "Liveness probes restart a container that hangs",
     "Readiness probes take a Pod out of the Service until it is ready"]),
    ("Team standup notes", "Teams.exe", ["Blocked on the database migration review", "Release candidate goes out Friday", "Ana will pair on the flaky login test"]),
]
QUERIES = [  # (query, index of the window it should find)
    ("how do I make that red bean dinner", 0), ("bug where memory was freed while another thread used it", 1),
    ("when is my trip to Canada", 2), ("graphics card benchmark and wattage", 3), ("how much do I spend each month", 4),
    ("GPU ran out of VRAM during training", 5), ("teeth appointment", 6), ("can I keep a dog in the flat", 7),
    ("board game opening analysis", 8), ("my jogging distances this week", 9), ("container health checks", 10),
    ("what is holding up the release", 11),
]


def make(path, span=1):
    h = H.History(path, flush_s=0.05)
    h.passages = Passages(span_s=span)
    return h


def synthetic(d):
    h = make(Path(d) / "syn.db")
    now = time.time()
    h.record([dict(kind="content", hwnd=i + 1, title=t, process=p, added=lines, added_total=len(lines), removed=0,
                   seq=i + 1, t=now - 5) for i, (t, p, lines) in enumerate(WINDOWS)])
    end = time.time() + 10
    while h.stats.get("passages_written", 0) < len(WINDOWS) and time.time() < end:
        time.sleep(0.05)
    w = S.EmbedWorker()
    t = time.perf_counter()
    n = S.Indexer(h, w).run_once()
    index_s = time.perf_counter() - t
    ranks, kw_found, q_ms = [], 0, []
    for q, want in QUERIES:
        t = time.perf_counter()
        out = S.search(h, w, q, k=len(WINDOWS))
        q_ms.append((time.perf_counter() - t) * 1000)
        titles = [x["title"] for x in out["hits"]]
        ranks.append(titles.index(WINDOWS[want][0]) + 1 if WINDOWS[want][0] in titles else 99)
        kw = h.search(q, since=now - 60)["hits"]
        kw_found += any(x["title"] == WINDOWS[want][0] for x in kw)
    st = w.status()
    w.close()
    h.close()
    print(json.dumps(dict(windows=len(WINDOWS), passages_indexed=n, queries=len(QUERIES),
                          top1=sum(r == 1 for r in ranks), top3=sum(r <= 3 for r in ranks),
                          mrr=round(sum(1 / r for r in ranks) / len(ranks), 3), ranks=ranks,
                          keyword_found=kw_found, index_s_incl_worker_start=round(index_s, 2),
                          query_ms_first=round(q_ms[0], 1), query_ms_median=round(sorted(q_ms)[len(q_ms) // 2], 1),
                          backend=st["backend"], precision=st["precision"], reason=st["reason"])))


def live(d, seconds):
    from ignite_scribe.redact import Policy
    from ignite_scribe.watch import Watcher
    h = make(Path(d) / "live.db", span=300)
    wt = Watcher(Policy(), recorder=h).start()
    time.sleep(seconds)
    wt.stop()
    h.close()                                            # closes the open buckets into passages
    h = H.History(Path(d) / "live.db", flush_s=0.05)
    w = S.EmbedWorker()
    ix = S.Indexer(h, w, max_per_round=10 ** 6)
    t = time.perf_counter()
    n = ix.run_once()
    dt = time.perf_counter() - t
    st = w.status()
    t = time.perf_counter()
    S.search(h, w, "an error message")
    q = (time.perf_counter() - t) * 1000
    w.close()
    size = h.status()["size_mb"]
    h.close()
    print(json.dumps(dict(seconds=seconds, passages=n, passages_per_hour=round(n * 3600 / seconds),
                          index_s=round(dt, 2), backend=st["backend"], precision=st["precision"],
                          query_ms=round(q, 1), db_mb=size)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--live", type=float, default=0)
    a = ap.parse_args()
    with tempfile.TemporaryDirectory() as d:
        if a.synthetic:
            synthetic(d)
        if a.live:
            live(d, a.live)


if __name__ == "__main__":
    main()
