# Copyright (C) 2026 The ignite-scribe contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The always-on OCR workload on the real desktop, measured. Prints numbers only, never text.

Runs the real watcher and ``AlwaysOn`` for ``--seconds``. By default it uses the CPU measurement mode
(``IGNITE_SCRIBE_ALWAYS_ON_CPU=1``, set here) and the vision worker with ``--backend cpu``, so no NPU context is
opened. ``--npu`` runs it as shipped instead: the NPU detector and the ScribeRec buckets, with always-on's own
NPU-only gate. Run that through ignite-xdna's ``research-lowlevel.sh --npu``. The worker process's CPU time is
reported too (``worker_cpu_pct``), because that is where OCR's host work runs. Reports reads per minute, pixels per read, detector and recognizer ms per read, the
share of each minute spent, lines, and the engine process's CPU time. Run it only on a quiet host.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def worker_cpu_s(worker) -> float | None:
    """CPU seconds (user + kernel) used so far by the vision worker's interpreter and its children."""
    proc = getattr(worker, "proc", None)
    if proc is None:
        return None
    import ctypes
    from ctypes import wintypes
    k32 = ctypes.windll.kernel32
    snap = k32.CreateToolhelp32Snapshot(2, 0)

    class PE(ctypes.Structure):
        _fields_ = [("dwSize", wintypes.DWORD), ("cntUsage", wintypes.DWORD), ("th32ProcessID", wintypes.DWORD),
                    ("th32DefaultHeapID", ctypes.c_size_t), ("th32ModuleID", wintypes.DWORD),
                    ("cntThreads", wintypes.DWORD), ("th32ParentProcessID", wintypes.DWORD),
                    ("pcPriClassBase", ctypes.c_long), ("dwFlags", wintypes.DWORD), ("szExeFile", ctypes.c_char * 260)]

    e = PE()
    e.dwSize = ctypes.sizeof(PE)
    kids, ok = {}, k32.Process32First(snap, ctypes.byref(e))
    while ok:
        kids.setdefault(e.th32ParentProcessID, []).append(e.th32ProcessID)
        ok = k32.Process32Next(snap, ctypes.byref(e))
    k32.CloseHandle(snap)
    todo, total = [proc.pid], 0.0
    while todo:
        pid = todo.pop()
        todo += kids.get(pid, [])
        h = k32.OpenProcess(0x1000, False, pid)              # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            continue
        t = [wintypes.FILETIME() for _ in range(4)]
        if k32.GetProcessTimes(h, *[ctypes.byref(x) for x in t]):
            total += sum((x.dwHighDateTime << 32 | x.dwLowDateTime) / 1e7 for x in t[2:])
        k32.CloseHandle(h)
    return total


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seconds", type=float, default=300)
    ap.add_argument("--min-gap", type=float, default=2.0)
    ap.add_argument("--npu", action="store_true", help="as shipped: NPU detector and recognizer")
    ap.add_argument("--canvas", action="store_true",
                    help="also open tools/canvas_window.py (synthetic GDI text, nothing for UI Automation)")
    a = ap.parse_args()
    canvas = None
    if a.canvas:
        import subprocess
        canvas = subprocess.Popen([sys.executable, str(Path(__file__).with_name("canvas_window.py")),
                                   "--seconds", str(a.seconds + 30), "--topmost"], stdout=subprocess.PIPE, text=True)
        canvas.stdout.readline()
        time.sleep(1.0)
    if not a.npu:
        os.environ["IGNITE_SCRIBE_ALWAYS_ON_CPU"] = "1"
        os.environ["IGNITE_SCRIBE_VISION_BACKEND"] = "cpu"
    from ignite_scribe import vision_client
    from ignite_scribe.always_on import AlwaysOn
    from ignite_scribe.redact import Policy
    from ignite_scribe.watch import Watcher
    policy = Policy()
    w = Watcher(policy).start()
    worker = vision_client.VisionWorker(idle=30)
    eng = AlwaysOn(w, policy, lambda: worker, min_gap_s=a.min_gap, budget_ms_per_min=60_000).start()
    c0, t0 = time.process_time(), time.time()
    time.sleep(a.seconds)
    st = eng.status()
    wall = time.time() - t0
    cpu = time.process_time() - c0
    wcpu = worker_cpu_s(worker)
    eng.stop()
    worker.close()
    w.stop()
    if canvas is not None:
        canvas.kill()
        canvas.wait()
    n = st["reads"]
    print(json.dumps(dict(
        seconds=round(wall), reads=n, reads_per_min=round(n * 60 / wall, 1), windows_with_text=st["windows"],
        lines_read=st["lines"], kpixels_per_read=round(st["pixels"] / n / 1000, 1) if n else None,
        det_ms_per_read=round(st["det_ms"] / n, 1) if n else None,
        rec_ms_per_read=round(st["rec_ms"] / n, 1) if n else None,
        call_ms_per_read=st["ms_per_read"], busy_pct=round(100 * st["ms_total"] / 1000 / wall, 2),
        server_process_cpu_pct=round(100 * cpu / wall, 2),
        worker_cpu_pct=round(100 * wcpu / wall, 2) if wcpu is not None else None,
        cache_hits=st.get("cache_hits"), backend=st["backend"], errors=st["errors"],
        last_error=(st["last_error"] or "")[:80] or None, paused=st["paused"])))


if __name__ == "__main__":
    main()
