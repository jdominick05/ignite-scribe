# Copyright (C) 2026 The ignite-scribe contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""A stand-in build for tools/waitfor_bench.py: prints numbered compile lines, then a success line with a nonce.

Runs in its own console window titled ``scribe-bench build``. Writes the wall-clock time of the success line to
``--stamp`` so the benchmark can measure how late ``wait_for`` returned. The text is synthetic.
"""
import argparse
import ctypes
import sys
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=20)
    ap.add_argument("--every", type=float, default=0.5)
    ap.add_argument("--nonce", required=True)
    ap.add_argument("--stamp", required=True)
    ap.add_argument("--hold", type=float, default=15, help="keep the window open this long afterwards")
    a = ap.parse_args()
    ctypes.windll.kernel32.SetConsoleTitleW("scribe-bench build")
    n, t_end = 0, time.time() + a.seconds
    while time.time() < t_end:
        n += 1
        print(f"[{n:3d}] compiling unit_{n:03d}.c", flush=True)
        time.sleep(a.every)
    print(f"BENCH BUILD SUCCEEDED {a.nonce}", flush=True)
    with open(a.stamp, "w") as f:
        f.write(repr(time.time()))
    time.sleep(a.hold)
    return 0


if __name__ == "__main__":
    sys.exit(main())
