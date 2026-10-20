# Copyright (C) 2026 The ignite-scribe contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The injection flag on the real desktop: how often it fires, and what it costs a watcher refresh. Counts only.

Runs the watcher for ``--seconds`` and, every second, counts the flagged windows in its redacted state by app
category and kind. It also times ``Policy.apply`` with and without the flag on that state. No text, title or
excerpt is printed or kept.
"""
import argparse
import collections
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seconds", type=float, default=60)
    a = ap.parse_args()
    from ignite_scribe import injection, redact
    from ignite_scribe.redact import Policy
    from ignite_scribe.watch import Watcher
    pol = Policy()
    w = Watcher(pol).start()
    samples, flagged_samples, windows_seen = 0, 0, 0
    by = collections.Counter()
    t_end = time.time() + a.seconds
    snap = None
    while time.time() < t_end:
        time.sleep(1.0)
        snap = w.snapshot()[0]
        if snap is None:
            continue
        samples += 1
        windows_seen += len(snap.windows)
        hit = [x for x in snap.windows if x.flags]
        flagged_samples += bool(hit)
        for x in hit:
            for f in x.flags:
                by[f"{f['app']}/{f['kind']}"] += 1
    st = w.status()
    w.stop()
    ms_with = ms_without = None
    if snap is not None:
        t = time.perf_counter()
        for _ in range(20):
            pol.apply(snap)
        ms_with = (time.perf_counter() - t) * 50
        keep = injection.flag_window
        redact.injection.flag_window = lambda win: []
        try:
            t = time.perf_counter()
            for _ in range(20):
                pol.apply(snap)
            ms_without = (time.perf_counter() - t) * 50
        finally:
            redact.injection.flag_window = keep
    print(json.dumps(dict(seconds=a.seconds, samples=samples, windows_per_sample=round(windows_seen / max(samples, 1), 1),
                          samples_with_a_flag=flagged_samples, flags_by_app_and_kind=dict(by),
                          apply_ms_with_flag=round(ms_with, 2) if ms_with else None,
                          apply_ms_without=round(ms_without, 2) if ms_without else None,
                          refresh_ms_avg=round(st.get("refresh_ms_total", 0) / max(st.get("refreshes", 1), 1), 1)
                          if "refresh_ms_total" in st else st.get("refresh_ms_avg"))))


if __name__ == "__main__":
    main()
