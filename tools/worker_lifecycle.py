# Copyright (C) 2026 The ignite-scribe contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Does the vision worker release the NPU on every stop path? Run from this repo's own Python (the venv).

1. start (first detect), report backend and per-tile NPU ms over ``--tiles`` tiles (one grey, the rest seeded
   random noise: timing and release only, not accuracy);
2. ``close()``: the NPU must show no hardware context while this process is still alive;
3. idle stop: a worker with ``idle=3`` must exit by itself and release the NPU;
4. parent death: a child process starts a worker and is killed; the worker must exit and release the NPU.
Reads ``xrt-smi examine -r aie-partitions``. Prints one JSON line per step and exits non-zero on a failure.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
XRT_SMI = r"C:\Windows\System32\AMD\xrt-smi.exe"


def contexts() -> bool:
    out = subprocess.run([XRT_SMI, "examine", "-r", "aie-partitions"], capture_output=True, text=True).stdout
    return "No hardware contexts running" not in out


def wait_idle(timeout=15.0) -> float | None:
    t = time.monotonic()
    while time.monotonic() - t < timeout:
        if not contexts():
            return round(time.monotonic() - t, 2)
        time.sleep(0.25)
    return None


def child():
    from ignite_scribe.vision_client import VisionWorker
    import numpy as np
    w = VisionWorker(idle=600)
    w.detect([np.full((640, 640, 3), 114, np.uint8)])
    print(w.proc.pid, flush=True)
    time.sleep(600)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tiles", type=int, default=16)
    ap.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.child:
        return child()
    import numpy as np
    from ignite_scribe.vision_client import VisionWorker, available
    ok, why = available()
    assert ok, why
    rng = np.random.default_rng(0)
    batch = [np.full((640, 640, 3), 114, np.uint8)]
    batch += [rng.integers(0, 255, (640, 640, 3), dtype=np.uint8) for _ in range(a.tiles - 1)]
    fails = 0
    assert not contexts(), "NPU busy before the test"

    w = VisionWorker(idle=600)
    t = time.perf_counter()
    r = w.detect(batch[:1])
    first = (time.perf_counter() - t) * 1000
    t = time.perf_counter()
    r = w.detect(batch)
    call = (time.perf_counter() - t) * 1000
    ms = sorted(r["ms"])
    print(json.dumps(dict(step="start", backend=r["backend"], reason=w.reason, first_call_ms_incl_start=round(first),
                          tiles=len(batch), call_ms=round(call, 1), per_tile_call_ms=round(call / len(batch), 2),
                          npu_ms_median=ms[len(ms) // 2], npu_ms_min=ms[0], npu_ms_max=ms[-1],
                          context_while_running=contexts())), flush=True)
    fails += r["backend"] != "npu"
    w.close()
    t_idle = wait_idle()
    print(json.dumps(dict(step="close", worker_exited=w.proc is None, npu_idle_after_s=t_idle)), flush=True)
    fails += t_idle is None

    w2 = VisionWorker(idle=3)
    w2.detect(batch[:1])
    t0 = time.monotonic()
    while w2.running and time.monotonic() - t0 < 15:
        time.sleep(0.25)
    t_idle = wait_idle()
    print(json.dumps(dict(step="idle-stop", stopped_after_s=round(time.monotonic() - t0, 1), running=w2.running,
                          npu_idle_after_s=t_idle)), flush=True)
    fails += w2.running or t_idle is None

    c = subprocess.Popen([sys.executable, __file__, "--child"], stdout=subprocess.PIPE, text=True)
    wpid = int(c.stdout.readline())
    held = contexts()
    c.kill()
    c.wait()
    t0 = time.monotonic()
    gone = None
    while time.monotonic() - t0 < 15:
        alive = subprocess.run(["tasklist", "/FI", f"PID eq {wpid}", "/NH"], capture_output=True, text=True).stdout
        if str(wpid) not in alive:
            gone = round(time.monotonic() - t0, 2)
            break
        time.sleep(0.25)
    t_idle = wait_idle()
    print(json.dumps(dict(step="parent-killed", context_before_kill=held, worker_exited_after_s=gone,
                          npu_idle_after_s=t_idle)), flush=True)
    fails += gone is None or t_idle is None
    print(json.dumps(dict(result="PASS" if not fails else f"FAIL ({fails})")))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
