# Copyright (C) 2026 The ignite-scribe contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""How fast always-on OCR gives the NPU back when an ignite-xdna quiet request appears. Uses the NPU.

The engine runs with a real vision worker whose detector holds an NPU context (one grey tile is detected so the
context is open). Then a quiet request naming this process is written the way ignite-xdna writes it (``.tmp``,
then rename), and ``xrt-smi`` is polled until no hardware context remains. Prints the seconds from the request
to an empty device (ignite-xdna's requester waits at least 5 s), then removes the request and checks that the
engine resumes. Run it from a session that has agreed an NPU sitting.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
SMI = r"C:\Windows\System32\AMD\xrt-smi.exe"


def contexts() -> bool:
    out = subprocess.run([SMI, "examine", "-r", "aie-partitions"], capture_output=True, text=True).stdout
    return "No hardware contexts running" not in out


def main():
    import numpy as np
    from ignite_scribe import always_on as A, vision_client
    from ignite_scribe.redact import Policy
    from ignite_scribe.snapshot import Snapshot

    class StillWatcher:                             # nothing changes on this fake screen
        def snapshot(self):
            return Snapshot(time.time(), []), {}, time.time()

        def take_pixel_dirty(self):
            return {}

        def set_extra_text(self, h, b):
            pass

    assert not contexts(), "the NPU already has a context: not a clean sitting"
    worker = vision_client.VisionWorker(idle=600)
    eng = A.AlwaysOn(StillWatcher(), Policy(), lambda: worker, ready=lambda: (True, ""), tick_s=0.5)
    eng.worker = worker
    worker.detect([np.full((640, 640, 3), 114, np.uint8)])
    held = contexts()
    eng.start()
    qdir = A.QUIET_DIR
    qdir.mkdir(parents=True, exist_ok=True)
    req = qdir / f"npu-quiet.{os.getpid()}.request"
    tmp = req.with_suffix(".tmp")
    tmp.write_text(json.dumps(dict(pid=os.getpid(), since=time.time(), reason="quiet_release.py")))
    t0 = time.monotonic()
    os.replace(tmp, req)
    while contexts() and time.monotonic() - t0 < 30:
        time.sleep(0.1)
    released_s = time.monotonic() - t0
    paused = list(eng.paused)
    req.unlink()
    time.sleep(1.5)
    resumed = not eng.paused
    eng.stop()
    print(json.dumps(dict(backend=worker.backend, context_held_before=held, released_s=round(released_s, 2),
                          released=not contexts(), paused_reason=paused[:1], resumed_after_request=resumed)))


if __name__ == "__main__":
    main()
